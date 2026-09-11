use super::{
    ABI_VERSION, BufferAbi, JitError, KernelAbi, KernelPointerAbi, NativeOutputInitialization,
    RenderedC, emit_with_substitution, linear_store_iteration, native_cache_key,
    native_scalar_operation_can_signal, scalar_kernel_prologue,
};
use crate::{DType, IndexValue, Operation, Shape, UOp};
use std::collections::{BTreeMap, BTreeSet};

fn authenticates_external_broadcast(
    input_shape: &Shape,
    output_shape: &Shape,
    elements: usize,
    output_domain: &Shape,
) -> bool {
    if elements == 0
        || input_shape.rank() > output_domain.rank()
        || output_shape != output_domain
        || input_shape.numel().ok() != Some(elements)
    {
        return false;
    }
    input_shape
        .broadcast_with(output_domain)
        .is_ok_and(|shape| shape == *output_domain)
}

pub(crate) fn render_native_store_group(
    root: &UOp,
) -> Result<(RenderedC, NativeOutputInitialization), JitError> {
    root.validate()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    let nodes = root
        .topological()
        .map_err(|error| JitError::Unsupported(error.to_string()))?;
    if nodes.iter().any(|node| {
        native_scalar_operation_can_signal(node.operation(), node.ty().map(|ty| ty.scalar))
    }) {
        return Err(JitError::Unsupported(
            "native store group contains a failure-capable operation".into(),
        ));
    }
    let stores = root
        .sources()
        .iter()
        .filter(|node| matches!(node.operation(), Operation::Store))
        .collect::<Vec<_>>();
    if stores.len() < 2 {
        return Err(JitError::Unsupported(
            "native store group requires multiple stores".into(),
        ));
    }
    let mut outputs = Vec::with_capacity(stores.len());
    let mut output_ids = BTreeSet::new();
    let mut extent = None;
    let mut output_domain = None;
    for store in &stores {
        let [index, _] = store.sources() else {
            return Err(JitError::Unsupported(
                "native store group store is malformed".into(),
            ));
        };
        let Operation::Index(IndexValue::Buffer {
            buffer,
            elements,
            input_shape,
            output_shape,
            addressing: crate::IndexAddressing::Broadcast,
        }) = index.operation()
        else {
            return Err(JitError::Unsupported(
                "native store group output is not dense".into(),
            ));
        };
        if index.ty().map(|ty| ty.scalar) != Some(DType::F32)
            || *elements == 0
            || input_shape != output_shape
            || extent.is_some_and(|expected| expected != *elements)
            || output_domain
                .as_ref()
                .is_some_and(|expected| expected != output_shape)
            || !output_ids.insert(*buffer)
        {
            return Err(JitError::Unsupported(
                "native store group output descriptor differs".into(),
            ));
        }
        let iteration = linear_store_iteration(index)?;
        extent = Some(*elements);
        output_domain = Some(output_shape.clone());
        outputs.push((*buffer, *elements, iteration));
    }
    let output_domain = output_domain.expect("native store group outputs have a domain");

    let output_positions = outputs
        .iter()
        .enumerate()
        .map(|(position, (buffer, _, _))| (*buffer, position))
        .collect::<BTreeMap<_, _>>();
    let mut buffers = Vec::<BufferAbi>::new();
    let mut seen = BTreeMap::<u64, usize>::new();
    for (consumer_position, (store, (_, _, consumer_iteration))) in
        stores.iter().zip(&outputs).enumerate()
    {
        let value = store.sources().get(1).ok_or_else(|| {
            JitError::Unsupported("native store group Store missing value".into())
        })?;
        for node in value
            .topological()
            .map_err(|error| JitError::Unsupported(error.to_string()))?
        {
            if !matches!(node.operation(), Operation::Load) {
                continue;
            }
            let Some(index) = node.sources().first() else {
                return Err(JitError::Unsupported(
                    "native store group load has no index".into(),
                ));
            };
            let Operation::Index(IndexValue::Buffer {
                buffer,
                elements,
                input_shape,
                output_shape,
                addressing: crate::IndexAddressing::Broadcast,
            }) = index.operation()
            else {
                return Err(JitError::Unsupported(
                    "native store group load is not dense".into(),
                ));
            };
            let dtype = node
                .ty()
                .ok_or_else(|| JitError::Unsupported("untyped native store group load".into()))?
                .scalar;
            if let Some(&producer_position) = output_positions.get(buffer) {
                let (_, output_elements, _) = outputs[producer_position];
                let load_iteration = linear_store_iteration(index)?;
                if producer_position >= consumer_position
                    || dtype != DType::F32
                    || *elements != output_elements
                    || input_shape != &output_domain
                    || output_shape != &output_domain
                    || !load_iteration.shares_node_with(consumer_iteration)
                {
                    return Err(JitError::Unsupported(
                        "native store group has an unordered output dependency".into(),
                    ));
                }
                continue;
            }
            if !authenticates_external_broadcast(
                input_shape,
                output_shape,
                *elements,
                &output_domain,
            ) {
                return Err(JitError::Unsupported(
                    "native store group input broadcast descriptor differs".into(),
                ));
            }
            let candidate = BufferAbi {
                id: *buffer,
                dtype,
                elements: *elements,
                mutable: false,
            };
            if let Some(&position) = seen.get(buffer) {
                if buffers[position] != candidate {
                    return Err(JitError::Unsupported(
                        "native store group input descriptor differs".into(),
                    ));
                }
                continue;
            }
            seen.insert(*buffer, buffers.len());
            buffers.push(candidate);
        }
    }
    for (buffer, elements, _) in &outputs {
        seen.insert(*buffer, buffers.len());
        buffers.push(BufferAbi {
            id: *buffer,
            dtype: DType::F32,
            elements: *elements,
            mutable: true,
        });
    }
    let abi = KernelAbi {
        version: ABI_VERSION,
        pointer_order: (0..buffers.len()).map(KernelPointerAbi::Dense).collect(),
        buffers,
        quantized_buffers: Vec::new(),
        symbol_count: 0,
    };
    let ids = abi
        .buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| (buffer.id, index))
        .collect::<BTreeMap<_, _>>();
    let extent = extent.expect("native store group outputs have an extent");
    let mut lines = scalar_kernel_prologue(
        "/* private native store group v1 */".into(),
        false,
        false,
        false,
        "int rustgrad_kernel(void **buffers, const int64_t *symbols, uint64_t *failure) { (void)symbols; failure[0]=UINT64_MAX; failure[1]=0;".into(),
    );
    lines.push(format!(
        "  for (size_t rg_i = 0; rg_i < {extent}u; ++rg_i) {{"
    ));
    let mut source_map = BTreeMap::new();
    for (store, (output, _, iteration)) in stores.iter().zip(&outputs) {
        let value = emit_with_substitution(
            store.sources().get(1).ok_or_else(|| {
                JitError::Unsupported("native store group Store missing value".into())
            })?,
            &ids,
            &mut source_map,
            &mut lines,
            Some((iteration, "((int64_t)rg_i)")),
            None,
        )?;
        lines.push(format!(
            "    ((float*)buffers[{}])[rg_i] = (float)({value});",
            ids[output]
        ));
    }
    lines.push("  }".into());
    lines.push("  return 0;".into());
    lines.push("}".into());
    let source = lines.join("\n") + "\n";
    let cache_key = native_cache_key("native-store-group-v1", &source);
    Ok((
        RenderedC {
            source,
            source_map,
            abi,
            cache_key,
        },
        NativeOutputInitialization::FullyOverwritten,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Slice};
    use std::collections::HashMap;

    fn adamw_shaped_kernels(cross_lane: bool) -> (Vec<UOp>, [u64; 4]) {
        let mut graph = Graph::new();
        let parameter = graph.input("parameter", [3]);
        let first = graph.input("first", [3]);
        let second = graph.input("second", [3]);
        let accumulator = graph.input("accumulator", [3]);
        let gradient = graph.input("gradient", [3]);
        let next_first = graph.add(first, gradient).unwrap();
        let gradient_squared = graph.square(gradient).unwrap();
        let next_second = graph.add(second, gradient_squared).unwrap();
        let parameter_first = if cross_lane {
            graph
                .stride(
                    next_first,
                    [Slice {
                        start: None,
                        stop: None,
                        step: -1,
                    }],
                )
                .unwrap()
        } else {
            next_first
        };
        let normalized = graph.add(parameter_first, next_second).unwrap();
        let next_parameter = graph.sub(parameter, normalized).unwrap();
        let next_accumulator = graph.sub(accumulator, accumulator).unwrap();
        let outputs = [next_first, next_second, next_parameter, next_accumulator];
        let schedule = crate::schedule_many(&graph, &outputs).unwrap();
        let kernels = outputs
            .iter()
            .map(|output| {
                schedule
                    .items
                    .iter()
                    .find(|item| item.primary_output().id == output.index() as u64)
                    .unwrap()
                    .kernel
                    .clone()
            })
            .collect();
        (kernels, outputs.map(|output| output.index() as u64))
    }

    fn retarget_store_output(kernel: &UOp, output_buffer: u64) -> UOp {
        let [store, end_range] = kernel.sources() else {
            panic!("test kernel is not a scalar Sink");
        };
        let [index, value] = store.sources() else {
            panic!("test kernel Store is malformed");
        };
        let Operation::Index(mut output) = index.operation().clone() else {
            panic!("test kernel output is not indexed");
        };
        let IndexValue::Buffer { buffer, .. } = &mut output else {
            panic!("test kernel output is not dense");
        };
        *buffer = output_buffer;
        let index = UOp::from_operation(
            Operation::Index(output),
            index.ty(),
            index.sources().to_vec(),
        )
        .retag(index.tag().cloned());
        let store = UOp::from_operation(Operation::Store, store.ty(), vec![index, value.clone()])
            .retag(store.tag().cloned());
        UOp::from_operation(
            kernel.operation().clone(),
            kernel.ty(),
            vec![store, end_range.clone()],
        )
        .retag(kernel.tag().cloned())
    }

    fn broadcast_kernels() -> (Vec<UOp>, [u64; 3]) {
        let mut graph = Graph::new();
        let dense = graph.input("dense", [2, 3]);
        let scalar = graph.input("scalar", []);
        let row = graph.input("row", [3]);
        let scalar_output = graph.add(dense, scalar).unwrap();
        let row_output = graph.add(dense, row).unwrap();
        let outputs = [scalar_output, row_output];
        let schedule = crate::schedule_many(&graph, &outputs).unwrap();
        let kernels = outputs
            .iter()
            .map(|output| {
                schedule
                    .items
                    .iter()
                    .find(|item| item.primary_output().id == output.index() as u64)
                    .unwrap()
                    .kernel
                    .clone()
            })
            .collect();
        (
            kernels,
            [
                dense.index() as u64,
                scalar.index() as u64,
                row.index() as u64,
            ],
        )
    }

    fn rewrite_buffer_input(root: &UOp, target: u64, input_shape: Shape, elements: usize) -> UOp {
        fn rewrite(
            node: &UOp,
            target: u64,
            input_shape: &Shape,
            elements: usize,
            memo: &mut HashMap<UOp, UOp>,
        ) -> UOp {
            if let Some(rewritten) = memo.get(node) {
                return rewritten.clone();
            }
            let sources = node
                .sources()
                .iter()
                .map(|source| rewrite(source, target, input_shape, elements, memo))
                .collect();
            let operation = match node.operation().clone() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    output_shape,
                    addressing,
                    ..
                }) if buffer == target => Operation::Index(IndexValue::Buffer {
                    buffer,
                    elements,
                    input_shape: input_shape.clone(),
                    output_shape,
                    addressing,
                }),
                operation => operation,
            };
            let rewritten =
                UOp::from_operation(operation, node.ty(), sources).retag(node.tag().cloned());
            memo.insert(node.clone(), rewritten.clone());
            rewritten
        }
        rewrite(root, target, &input_shape, elements, &mut HashMap::new())
    }

    #[test]
    fn ordered_same_lane_output_dependencies_are_mutable_once() {
        let (kernels, outputs) = adamw_shaped_kernels(false);
        let members = kernels.iter().collect::<Vec<_>>();
        let fused = crate::kernel::fuse_native_store_group(&members).unwrap();
        let (rendered, initialization) = render_native_store_group(&fused).unwrap();
        assert_eq!(initialization, NativeOutputInitialization::FullyOverwritten);
        for &output in &outputs {
            let matching = rendered
                .abi
                .buffers
                .iter()
                .filter(|buffer| buffer.id == output)
                .collect::<Vec<_>>();
            assert_eq!(matching.len(), 1);
            assert!(matching[0].mutable);
        }
        let output_slot = |output| {
            rendered
                .abi
                .buffers
                .iter()
                .position(|buffer| buffer.id == output)
                .unwrap()
        };
        let first_store = rendered
            .source
            .find(&format!(
                "((float*)buffers[{}])[rg_i] =",
                output_slot(outputs[0])
            ))
            .unwrap();
        let parameter_store = rendered
            .source
            .find(&format!(
                "((float*)buffers[{}])[rg_i] =",
                output_slot(outputs[2])
            ))
            .unwrap();
        assert!(first_store < parameter_store);
    }

    #[test]
    fn immutable_inputs_admit_scalar_and_right_aligned_broadcasts() {
        let (kernels, [dense, scalar, row]) = broadcast_kernels();
        let members = kernels.iter().collect::<Vec<_>>();
        let fused = crate::kernel::fuse_native_store_group(&members).unwrap();
        let (rendered, _) = render_native_store_group(&fused).unwrap();
        let input = |id| {
            rendered
                .abi
                .buffers
                .iter()
                .find(|buffer| buffer.id == id)
                .map(|buffer| (buffer.elements, buffer.mutable))
                .unwrap()
        };
        assert_eq!(input(dense), (6, false));
        assert_eq!(input(scalar), (1, false));
        assert_eq!(input(row), (3, false));
    }

    #[test]
    fn incompatible_external_broadcast_descriptors_reject() {
        let (kernels, [_, _, row]) = broadcast_kernels();
        let members = kernels.iter().collect::<Vec<_>>();
        let fused = crate::kernel::fuse_native_store_group(&members).unwrap();
        for (shape, elements) in [
            (Shape::from([2, 2]), 4),
            (Shape::from([1, 2, 3]), 6),
            (Shape::from([0, 1]), 0),
            (Shape::from([3]), 2),
        ] {
            let malformed = rewrite_buffer_input(&fused, row, shape, elements);
            assert!(render_native_store_group(&malformed).is_err());
        }
    }

    #[test]
    fn self_forward_and_cross_lane_output_dependencies_reject() {
        let (kernels, _) = adamw_shaped_kernels(false);
        let first_input = kernels[0]
            .topological()
            .unwrap()
            .into_iter()
            .find_map(|node| {
                if !matches!(node.operation(), Operation::Load) {
                    return None;
                }
                let index = node.sources().first()?;
                let Operation::Index(IndexValue::Buffer { buffer, .. }) = index.operation() else {
                    return None;
                };
                Some(*buffer)
            })
            .unwrap();
        let self_dependent = retarget_store_output(&kernels[0], first_input);
        let members = [&self_dependent, &kernels[1], &kernels[2], &kernels[3]];
        let fused = crate::kernel::fuse_native_store_group(&members).unwrap();
        let error = render_native_store_group(&fused).err().unwrap();
        assert!(error.to_string().contains("unordered output dependency"));

        let forward = [&kernels[2], &kernels[0], &kernels[1], &kernels[3]];
        let fused = crate::kernel::fuse_native_store_group(&forward).unwrap();
        let error = render_native_store_group(&fused).err().unwrap();
        assert!(error.to_string().contains("unordered output dependency"));

        let (kernels, _) = adamw_shaped_kernels(true);
        let members = kernels.iter().collect::<Vec<_>>();
        let fused = crate::kernel::fuse_native_store_group(&members).unwrap();
        assert!(render_native_store_group(&fused).is_err());
    }
}
