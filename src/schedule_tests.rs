use crate::{
    Backend, BufferDesc, CpuBackend, DType, Graph, ScheduleItem, ScheduledOutputs, Shape,
    TensorData, UOp, plan_temporary_reuse, schedule, schedule_many,
    schedule_with_external_materializations,
};
use std::collections::HashMap;

fn buffer(id: u64, bytes: usize, alignment: usize) -> BufferDesc {
    BufferDesc {
        id,
        shape: Shape::from([bytes]),
        dtype: DType::U8,
        bytes,
        alignment,
        read_only: false,
        view: None,
    }
}

#[test]
fn static_position_movement_owns_exact_schedule_capture_and_native_abi() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let output = graph
        .scatter_positions(input, Shape::from([5]), vec![4], vec![-2])
        .unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert_eq!(item.node, output);
    assert_ne!(item.inputs[0].id, item.primary_output().id);
    assert!(item.inputs[0].read_only);
    assert!(!item.primary_output().read_only);
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input]
    );
    assert!(item.dependencies.is_empty());
    assert!(matches!(
        item.kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::ScatterPositions { .. })
    ));
    let memory = crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();
    assert!(
        memory
            .temporaries
            .iter()
            .all(|allocation| allocation.buffer_id != item.inputs[0].id)
    );

    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let bindings = std::collections::BTreeMap::from([(
        "input".into(),
        TensorData::new([2], vec![3.0, -0.0]).unwrap(),
    )]);
    let interpreted = decoded.replay(&bindings).unwrap();
    assert_eq!(
        interpreted[0]
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0, 0, (-0.0f32).to_bits(), 0, 3.0f32.to_bits()]
    );
    let native = decoded
        .replay_with_options(
            &bindings,
            &crate::CapturedReplayExecutor::default(),
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(
        native.outputs[0]
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        interpreted[0]
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let mut computed = Graph::new();
    let input = computed.input_dtype("input", [2], DType::F32);
    let producer = computed.square(input).unwrap();
    let output = computed
        .scatter_positions(producer, Shape::from([5]), vec![0], vec![2])
        .unwrap();
    let scheduled = schedule(&computed, output).unwrap();
    scheduled.validate().unwrap();
    let producer_item = scheduled
        .items
        .iter()
        .find(|item| item.node == producer)
        .unwrap();
    let output_item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(output_item.dependencies, vec![producer_item.id]);
    assert_ne!(output_item.inputs[0].id, output_item.primary_output().id);
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![producer]
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();

    let external =
        schedule_with_external_materializations(&computed, &[output], &[producer]).unwrap();
    assert_eq!(external.items.len(), 1);
    assert_eq!(external.items[0].external_materializations, vec![producer]);
    assert!(external.items[0].dependencies.is_empty());
    assert_ne!(
        external.items[0].inputs[0].id,
        external.items[0].primary_output().id
    );

    let cotangent = computed.input_dtype("cotangent", [5], DType::F32);
    let vjp = computed
        .scatter_positions_vjp(cotangent, Shape::from([2]), vec![4], vec![-2])
        .unwrap();
    let vjp_schedule = schedule(&computed, vjp).unwrap();
    assert_eq!(vjp_schedule.items.len(), 1);
    assert!(matches!(
        vjp_schedule.items[0].kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
    ));
}

#[test]
fn static_position_capture_preserves_all_storage_bits_in_interpreter_and_native() {
    for dtype in DType::ALL {
        let width = dtype.itemsize();
        let input_bytes = match dtype {
            DType::Bool => vec![1, 0],
            DType::F16 => [0x7e01u16, 0x8000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            DType::BF16 => [0x7fc1u16, 0x8000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            DType::F32 => [0x7fc0_0001u32, 0x8000_0000]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            DType::F64 => [0x7ff8_0000_0000_0001u64, 0x8000_0000_0000_0000]
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect(),
            DType::F8E4M3 | DType::F8E4M3FNUZ | DType::F8E5M2 | DType::F8E5M2FNUZ => {
                vec![0x81, 0x7f]
            }
            _ => (0..2 * width)
                .map(|index| 0x81u8.wrapping_add(index as u8))
                .collect::<Vec<_>>(),
        };
        let input_value = TensorData::from_le_bytes([2], dtype, &input_bytes).unwrap();
        let mut expected = vec![0; 5 * width];
        expected[4 * width..5 * width].copy_from_slice(&input_bytes[..width]);
        expected[2 * width..3 * width].copy_from_slice(&input_bytes[width..]);

        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let output = graph
            .scatter_positions(input, Shape::from([5]), vec![4], vec![-2])
            .unwrap();
        let scheduled = schedule(&graph, output).unwrap();
        scheduled.validate().unwrap();
        let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        let capture = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(capture.to_bytes().unwrap(), bytes, "{dtype:?}");
        let bindings = std::collections::BTreeMap::from([("input".into(), input_value)]);
        let interpreted = capture.replay(&bindings).unwrap();
        assert_eq!(interpreted[0].to_le_bytes().unwrap(), expected, "{dtype:?}");
        let native = capture
            .replay_with_options(
                &bindings,
                &crate::CapturedReplayExecutor::default(),
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(
            native.outputs[0].to_le_bytes().unwrap(),
            expected,
            "{dtype:?}"
        );
    }
}

fn item(id: u64, inputs: Vec<BufferDesc>, output: BufferDesc) -> ScheduleItem {
    ScheduleItem {
        id,
        node: crate::NodeId::from_index(0),
        dependencies: vec![],
        consumers: vec![],
        inputs,
        input_bindings: vec![],
        quantized_input_bindings: vec![],
        external_materializations: vec![],
        outputs: crate::ScheduledOutputs::single(output),
        kernel: UOp::sink(vec![]),
        boundary: None,
        cache_key: 0,
    }
}

#[test]
fn requested_source_values_are_passthroughs_not_schedule_producers() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::F32);
    let constant = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::F32,
            [crate::Scalar::F(-0.0), crate::Scalar::F(f64::NAN)],
        )
        .unwrap(),
    );
    let computed = graph.add(input, constant).unwrap();

    for source in [input, constant] {
        let scheduled = schedule(&graph, source).unwrap();
        assert!(scheduled.items.is_empty());
        scheduled.validate().unwrap();
    }

    let scheduled = schedule_many(&graph, &[input, constant, computed]).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert_eq!(scheduled.items[0].node, computed);
    assert_eq!(
        scheduled.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input, constant]
    );
}

#[test]
fn requested_source_affine_aliases_are_zero_kernel_passthroughs() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2, 3]), DType::F32);
    let transposed = graph.permute(input, [1, 0]).unwrap();
    let computed = graph.neg(transposed).unwrap();
    let scalar = graph.input_dtype("scalar", Shape::new([]), DType::I32);
    let scalar_identity = graph.permute(scalar, []).unwrap();
    let identity = graph.permute(input, [0, 1]).unwrap();
    assert_eq!(scalar_identity, scalar);
    assert_eq!(identity, input);

    let alias_only = schedule(&graph, transposed).unwrap();
    alias_only.validate().unwrap();
    assert!(alias_only.items.is_empty());
    let [passthrough] = alias_only.requested_passthroughs.as_slice() else {
        panic!("one requested affine passthrough")
    };
    assert_eq!(passthrough.requested, transposed);
    assert_eq!(passthrough.source, input);
    assert_eq!(passthrough.desc.id, input.index() as u64);
    assert!(passthrough.desc.read_only);
    assert_eq!(
        passthrough.desc.view.as_ref().unwrap().logical_shape,
        Shape::from([3, 2])
    );

    let mixed = schedule_many(&graph, &[transposed, computed, scalar_identity]).unwrap();
    mixed.validate().unwrap();
    assert_eq!(mixed.requested_passthroughs, vec![passthrough.clone()]);
    assert_eq!(mixed.items.len(), 1);
    assert_eq!(mixed.items[0].node, computed);
    assert_eq!(
        mixed.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.desc.id)
            .collect::<Vec<_>>(),
        vec![input.index() as u64]
    );

    let mut malformed = alias_only.clone();
    malformed.requested_passthroughs[0].desc.id = transposed.index() as u64;
    assert!(matches!(
        malformed.validate(),
        Err(crate::ScheduleError::Binding(_))
    ));
}

#[test]
fn sole_use_contiguous_redirects_ordinary_producer_into_owned_output() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [3], DType::F32);
    let producer = graph.square(input).unwrap();
    let output = graph.contiguous(producer).unwrap();

    let first = schedule(&graph, output).unwrap();
    let second = schedule(&graph, output).unwrap();
    first.validate().unwrap();
    assert_eq!(first.items.len(), 1);
    assert_eq!(first.items[0].node, output);
    assert_eq!(first.items[0].primary_output().id, output.index() as u64);
    assert_eq!(first.items[0].dependencies, Vec::<u64>::new());
    assert_eq!(
        first.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input]
    );
    assert!(matches!(
        first.items[0].kernel.operation(),
        crate::Operation::Sink
    ));
    let nodes = first.items[0].kernel.topological().unwrap();
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::GraphBinary(crate::BinaryOp::Mul)
    )));
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            if *buffer == output.index() as u64
    )));
    assert!(
        !nodes
            .iter()
            .any(|node| matches!(node.operation(), crate::Operation::Movement(_)))
    );
    assert_eq!(
        first
            .items
            .iter()
            .map(|item| (item.node, item.cache_key))
            .collect::<Vec<_>>(),
        second
            .items
            .iter()
            .map(|item| (item.node, item.cache_key))
            .collect::<Vec<_>>()
    );
    assert!(first.internal_temporaries(&[output]).is_empty());
    assert!(
        crate::MemoryPlan::from_schedule(&first, &[output], true)
            .unwrap()
            .temporaries
            .is_empty()
    );

    let actual = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::new([3], vec![-2.0, 0.0, 3.0]).unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(actual.storage(), &crate::Storage::F32(vec![4.0, 0.0, 9.0]));

    let mut tampered = first.clone();
    let store = &tampered.items[0].kernel.sources()[0];
    let index = &store.sources()[0];
    let value = store.sources()[1].clone();
    let crate::Operation::Index(crate::IndexValue::Buffer {
        buffer: _,
        elements,
        input_shape,
        output_shape,
    }) = index.operation()
    else {
        panic!("redirected Store must use a dense output index")
    };
    let wrong_buffer = producer.index() as u64;
    let wrong_address = UOp::from_operation(
        crate::Operation::DefineGlobal(crate::AddressValue {
            space: crate::AddressSpace::Global,
            name: format!("b{wrong_buffer}"),
            element: crate::UType::scalar(DType::F32),
        }),
        Some(crate::UType::scalar(DType::F32)),
        vec![],
    );
    let wrong_index = UOp::from_operation(
        crate::Operation::Index(crate::IndexValue::Buffer {
            buffer: wrong_buffer,
            elements: *elements,
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
        }),
        index.ty(),
        vec![wrong_address, index.sources()[1].clone()],
    );
    let wrong_store = UOp::from_operation(crate::Operation::Store, None, vec![wrong_index, value]);
    tampered.items[0].kernel = UOp::sink(vec![
        wrong_store,
        tampered.items[0].kernel.sources()[1].clone(),
    ]);
    assert!(tampered.validate().is_err());
    assert!(crate::CapturedSchedule::capture(&graph, &tampered, &[output]).is_err());
    let mut encoded = crate::CapturedSchedule::capture(&graph, &first, &[output]).unwrap();
    encoded.items[0].kernel = tampered.items[0].kernel.clone();
    encoded.items[0].cache_key = crate::schedule::item_cache_key(&encoded.items[0]).unwrap();
    encoded.identity = crate::schedule::artifact::identity(&encoded).unwrap();
    assert!(encoded.to_bytes().is_err());

    let mut missing_store = first.clone();
    missing_store.items[0].kernel = UOp::sink(vec![first.items[0].kernel.sources()[1].clone()]);
    assert!(missing_store.validate().is_err());
    assert!(crate::CapturedSchedule::capture(&graph, &missing_store, &[output]).is_err());

    let mut empty_sink = first.clone();
    empty_sink.items[0].kernel = UOp::sink(vec![]);
    assert!(empty_sink.validate().is_err());
    assert!(crate::CapturedSchedule::capture(&graph, &empty_sink, &[output]).is_err());
    let mut current = crate::CapturedSchedule::capture(&graph, &first, &[output]).unwrap();
    current.items[0].kernel = UOp::sink(vec![]);
    current.items[0].cache_key = crate::schedule::item_cache_key(&current.items[0]).unwrap();
    current.identity = crate::schedule::artifact::identity(&current).unwrap();
    assert!(current.to_bytes().is_err());

    let mut add_graph = Graph::new();
    let lhs = add_graph.input_dtype("lhs", [3], DType::F32);
    let rhs = add_graph.input_dtype("rhs", [3], DType::F32);
    let sum = add_graph.binary(crate::BinaryOp::Add, lhs, rhs).unwrap();
    let copied = add_graph.contiguous(sum).unwrap();
    let added = schedule(&add_graph, copied).unwrap();
    added.validate().unwrap();
    assert_eq!(added.items.len(), 1);
    assert_eq!(added.items[0].node, copied);
    assert_eq!(
        added.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![lhs, rhs]
    );
}

#[test]
fn contiguous_redirection_preserves_portable_and_raw_copy_dtype_routes() {
    for dtype in DType::ALL {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let producer = graph.detach(input).unwrap();
        let output = graph.contiguous(producer).unwrap();
        let scheduled = schedule(&graph, output).unwrap();
        scheduled.validate().unwrap();
        let portable = matches!(dtype, DType::Bool | DType::I32 | DType::U32 | DType::F32);
        assert_eq!(
            scheduled.items.len(),
            if portable { 1 } else { 2 },
            "{dtype:?}"
        );
        let output_item = scheduled
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap();
        assert_eq!(output_item.node, output, "{dtype:?}");
        assert_eq!(output_item.primary_output().dtype, dtype, "{dtype:?}");
        assert_eq!(
            matches!(output_item.kernel.operation(), crate::Operation::Sink),
            portable,
            "{dtype:?}"
        );
    }

    for shape in [Shape::from([]), Shape::from([0])] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", shape.clone(), DType::F32);
        let producer = graph.detach(input).unwrap();
        let output = graph.contiguous(producer).unwrap();
        let scheduled = schedule(&graph, output).unwrap();
        scheduled.validate().unwrap();
        assert_eq!(scheduled.items.len(), 1, "{shape:?}");
        assert_eq!(scheduled.items[0].primary_output().shape, shape);
    }
}

#[test]
fn contiguous_redirection_preserves_requested_shared_specialized_and_external_producers() {
    let mut requested_graph = Graph::new();
    let input = requested_graph.input_dtype("input", [2], DType::F32);
    let producer = requested_graph.square(input).unwrap();
    let copied = requested_graph.contiguous(producer).unwrap();
    let requested = schedule_many(&requested_graph, &[producer, copied]).unwrap();
    assert_eq!(requested.items.len(), 2);
    assert!(matches!(
        requested
            .items
            .iter()
            .find(|item| item.node == copied)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::Contiguous { input } if input.node == producer)
    ));

    let sibling = requested_graph.neg(producer).unwrap();
    let shared = schedule_many(&requested_graph, &[copied, sibling]).unwrap();
    assert_eq!(shared.items.len(), 3);
    assert!(shared.items.iter().any(|item| item.node == producer));
    assert!(matches!(
        shared
            .items
            .iter()
            .find(|item| item.node == copied)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::Contiguous { .. })
    ));

    let mut specialized_graph = Graph::new();
    let input = specialized_graph.input_dtype("input", [2, 2], DType::F32);
    let reduced = specialized_graph.sum_all(input).unwrap();
    let copied = specialized_graph.contiguous(reduced).unwrap();
    let specialized = schedule(&specialized_graph, copied).unwrap();
    assert_eq!(specialized.items.len(), 3);
    assert!(matches!(
        specialized
            .items
            .iter()
            .find(|item| item.node == copied)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::Contiguous { input } if input.node == reduced)
    ));

    let mut external_graph = Graph::new();
    let input = external_graph.input_dtype("input", [2], DType::F32);
    let producer = external_graph.square(input).unwrap();
    let copied = external_graph.contiguous(producer).unwrap();
    let external =
        schedule_with_external_materializations(&external_graph, &[copied], &[producer]).unwrap();
    assert_eq!(external.items.len(), 1);
    assert_eq!(external.items[0].external_materializations, vec![producer]);
    assert!(matches!(
        external.items[0].kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::Contiguous { input } if input.node == producer)
    ));

    let mut faulting_graph = Graph::new();
    let lhs = faulting_graph.input_dtype("lhs", [2], DType::I32);
    let rhs = faulting_graph.input_dtype("rhs", [2], DType::I32);
    let quotient = faulting_graph
        .binary(crate::BinaryOp::Div, lhs, rhs)
        .unwrap();
    let copied = faulting_graph.contiguous(quotient).unwrap();
    let faulting = schedule(&faulting_graph, copied).unwrap();
    assert_eq!(faulting.items.len(), 2);
    assert!(matches!(
        faulting
            .items
            .iter()
            .find(|item| item.node == copied)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::Contiguous { .. })
    ));

    let mut backward_graph = Graph::new();
    let input = backward_graph.input_dtype("input", [2], DType::F32);
    let backward = backward_graph.contiguous_backward(input).unwrap();
    let copied = backward_graph.contiguous(backward).unwrap();
    let backward_schedule = schedule(&backward_graph, copied).unwrap();
    assert_eq!(backward_schedule.items.len(), 2);
    assert!(
        backward_schedule
            .items
            .iter()
            .any(|item| item.node == backward)
    );
}

#[test]
fn unrequested_contiguous_redirection_remains_a_downstream_materialization() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let producer = graph.square(input).unwrap();
    let contiguous = graph.contiguous(producer).unwrap();
    let output = graph.neg(contiguous).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 2);
    let redirected = scheduled
        .items
        .iter()
        .find(|item| item.node == contiguous)
        .unwrap();
    let consumer = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert!(matches!(
        redirected.kernel.operation(),
        crate::Operation::Sink
    ));
    assert_eq!(consumer.dependencies, vec![redirected.id]);
    assert_eq!(consumer.ordered_inputs()[0].input_node, contiguous);
    assert_eq!(
        scheduled
            .internal_temporaries(&[output])
            .into_iter()
            .map(|temporary| temporary.id)
            .collect::<Vec<_>>(),
        vec![contiguous.index() as u64]
    );
}

#[test]
fn contiguous_redirection_does_not_change_graph_vjp_or_uncomposable_affine_boundaries() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [3], DType::F32);
    let producer = graph.square(input).unwrap();
    let contiguous = graph.contiguous(producer).unwrap();
    let loss = graph.sum_all(contiguous).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([3]));
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::new([3], vec![-2.0, 0.0, 3.0]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-4.0, 0.0, 6.0]
    );

    // This broadcasted leaf needs producer-coordinate decomposition after the
    // reshape, which the existing affine load descriptor cannot represent.
    // Preserve the producer + AffineCopy fallback rather than misindexing it.
    let mut fallback = Graph::new();
    let input = fallback.input_dtype("input", [2, 3], DType::F32);
    let bias = fallback.input_dtype("bias", [3], DType::F32);
    let producer = fallback.add(input, bias).unwrap();
    let reshaped = fallback.reshape(producer, [3, 2]).unwrap();
    let materialized = fallback.contiguous(reshaped).unwrap();
    let scheduled = schedule(&fallback, materialized).unwrap();
    assert_eq!(scheduled.items.len(), 2);
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == materialized)
        .unwrap();
    assert!(matches!(
        item.kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
    ));
}

#[test]
fn affine_contiguous_redirects_exact_shape_producer_loads_into_owned_output() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let producer = graph.square(input).unwrap();
    let permuted = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.contiguous(permuted).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert_eq!(item.node, output);
    assert_eq!(item.primary_output().id, output.index() as u64);
    assert_eq!(item.dependencies, Vec::<u64>::new());
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input]
    );
    assert!(scheduled.internal_temporaries(&[output]).is_empty());
    assert!(
        crate::MemoryPlan::from_schedule(&scheduled, &[output], true)
            .unwrap()
            .temporaries
            .is_empty()
    );
    let nodes = item.kernel.topological().unwrap();
    assert!(nodes.iter().all(|node| !matches!(
        node.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(_))
    )));
    let source_shape = Shape::from([2, 3]);
    let logical_shape = Shape::from([3, 2]);
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
            if *buffer == input.index() as u64
                && view.source_shape == source_shape
                && view.logical_shape == logical_shape
                && view.strides.as_slice() == [1, 3]
    )));
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
            if *buffer == output.index() as u64
    )));

    crate::PtxRenderer::new(80)
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    crate::runtime::opencl::OpenClRenderer::default()
        .render(&item.kernel)
        .unwrap();
    crate::runtime::metal::MetalRenderer::new(
        1,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: usize::MAX,
            unified_memory: false,
            family: "affine-redirection-test".into(),
        },
    )
    .unwrap()
    .render(&item.kernel)
    .unwrap();
    crate::runtime::webgpu::WgslRenderer::new(
        1,
        crate::runtime::webgpu::WebGpuCapabilities {
            max_buffer_size: usize::MAX,
            max_storage_buffers_per_shader_stage: u32::MAX,
            max_compute_workgroup_size_x: 1,
            max_compute_workgroups_per_dimension: u32::MAX,
            timestamp_query: false,
            shader_f16: false,
        },
    )
    .unwrap()
    .render(&item.kernel)
    .unwrap();

    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let provided = std::collections::BTreeMap::from([(
        "input".into(),
        TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )]);
    let replayed = decoded.replay(&provided).unwrap();
    assert_eq!(
        replayed[0].storage(),
        &crate::Storage::F32(vec![1.0, 16.0, 4.0, 25.0, 9.0, 36.0])
    );
    let native = decoded
        .replay_with_options(
            &provided,
            &crate::CapturedReplayExecutor::default(),
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(native.outputs[0].storage(), replayed[0].storage());
    assert_eq!(native.trace.items[0].backend, crate::ItemBackend::NativeJit);
}

#[test]
fn affine_contiguous_redirects_expand_reverse_scalar_and_empty_geometry() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 1], DType::F32);
    let scalar = graph.constant(TensorData::scalar(1.0));
    let producer = graph.add(input, scalar).unwrap();
    let expanded = graph.expand(producer, [2, 3]).unwrap();
    let reversed = graph
        .stride(
            expanded,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let output = graph.contiguous(reversed).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert_eq!(
        scheduled.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input]
    );
    assert!(
        scheduled.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::Index(crate::IndexValue::View { view, .. })
                if view.offset == 1 && view.strides.as_slice() == [-1, 0]
            ))
    );

    let actual = crate::CapturedSchedule::capture(&graph, &scheduled, &[output])
        .unwrap()
        .replay(&std::collections::BTreeMap::from([(
            "input".into(),
            TensorData::new([2, 1], vec![2.0, 4.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(
        actual[0].storage(),
        &crate::Storage::F32(vec![5.0, 5.0, 5.0, 3.0, 3.0, 3.0])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::F32);
    let producer = empty.square(input).unwrap();
    let permuted = empty.permute(producer, [1, 0]).unwrap();
    let output = empty.contiguous(permuted).unwrap();
    let scheduled = schedule(&empty, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert_eq!(
        scheduled.items[0].primary_output().shape,
        Shape::from([2, 0])
    );
}

#[test]
fn affine_contiguous_binds_two_exact_shape_producer_leaves() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 3], DType::F32);
    let rhs = graph.input_dtype("rhs", [2, 3], DType::F32);
    let producer = graph.add(lhs, rhs).unwrap();
    let permuted = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.contiguous(permuted).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert_eq!(
        scheduled.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![lhs, rhs]
    );
    assert_eq!(
        scheduled.items[0]
            .inputs
            .iter()
            .map(|desc| desc.id)
            .collect::<Vec<_>>(),
        vec![lhs.index() as u64, rhs.index() as u64]
    );
}

#[test]
fn affine_contiguous_composes_right_aligned_broadcast_leaf_reads() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let row = graph.input_dtype("row", [3], DType::F32);
    let column = graph.input_dtype("column", [2, 1], DType::F32);
    let producer = graph.add(input, row).unwrap();
    let producer = graph.add(producer, column).unwrap();
    let permuted = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.contiguous(permuted).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input, row, column]
    );
    let nodes = item.kernel.topological().unwrap();
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
            if *buffer == row.index() as u64
                && view.source_shape == Shape::from([3])
                && view.logical_shape == Shape::from([3, 2])
                && view.strides.as_slice() == [1, 0]
    )));
    assert!(nodes.iter().any(|node| matches!(
        node.operation(),
        crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
            if *buffer == column.index() as u64
                && view.source_shape == Shape::from([2, 1])
                && view.logical_shape == Shape::from([3, 2])
                && view.strides.as_slice() == [0, 1]
    )));
    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let bindings = std::collections::BTreeMap::from([
        (
            "input".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        (
            "row".into(),
            TensorData::new([3], vec![10.0, 20.0, 30.0]).unwrap(),
        ),
        (
            "column".into(),
            TensorData::new([2, 1], vec![100.0, 200.0]).unwrap(),
        ),
    ]);
    let replayed = decoded.replay(&bindings).unwrap();
    assert_eq!(
        replayed[0].storage(),
        &crate::Storage::F32(vec![111.0, 214.0, 122.0, 225.0, 133.0, 236.0])
    );
    let native = decoded
        .replay_with_options(
            &bindings,
            &crate::CapturedReplayExecutor::default(),
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(native.outputs[0].storage(), replayed[0].storage());
}

#[test]
fn affine_contiguous_broadcasted_materialized_leaf_retains_exact_dependency() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let lhs = graph.input_dtype("lhs", [1, 2], DType::F32);
    let rhs = graph.input_dtype("rhs", [2, 3], DType::F32);
    let matrix = graph.matmul(lhs, rhs).unwrap();
    let producer = graph.add(input, matrix).unwrap();
    let permuted = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.contiguous(permuted).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 2);
    assert!(!scheduled.items.iter().any(|item| item.node == producer));
    let matrix_item = scheduled
        .items
        .iter()
        .find(|item| item.node == matrix)
        .unwrap();
    let output_item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input, matrix]
    );
    assert_eq!(output_item.dependencies, vec![matrix_item.id]);
    assert!(
        output_item
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
                    if *buffer == matrix.index() as u64 && view.strides.as_slice() == [1, 0]
            ))
    );
    let memory = crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();
    assert_eq!(
        memory
            .temporaries
            .iter()
            .map(|temporary| temporary.buffer_id)
            .collect::<Vec<_>>(),
        vec![matrix.index() as u64]
    );
    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let bindings = std::collections::BTreeMap::from([
        (
            "input".into(),
            TensorData::new([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        ),
        (
            "lhs".into(),
            TensorData::new([1, 2], vec![1.0, 2.0]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
    ]);
    let replayed = decoded.replay(&bindings).unwrap();
    assert_eq!(
        replayed[0].storage(),
        &crate::Storage::F32(vec![19.0, 49.0, 32.0, 62.0, 45.0, 75.0])
    );
}

#[test]
fn scalar_consumer_fuses_independent_computed_affine_branches() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::F32);
    let y = graph.input_dtype("y", [2, 3], DType::F32);
    let x_producer = graph.square(x).unwrap();
    let x_view = graph.permute(x_producer, [1, 0]).unwrap();
    let y_producer = graph.neg(y).unwrap();
    let y_view = graph.permute(y_producer, [1, 0]).unwrap();
    let output = graph.add(x_view, y_view).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![x, y]
    );
    for source in [x, y] {
        assert!(
            item.kernel
                .topological()
                .unwrap()
                .iter()
                .any(|node| matches!(
                    node.operation(),
                    crate::Operation::Index(crate::IndexValue::View { buffer, view, .. })
                        if *buffer == source.index() as u64
                            && view.source_shape == Shape::from([2, 3])
                            && view.logical_shape == Shape::from([3, 2])
                            && view.strides.as_slice() == [1, 3]
                ))
        );
    }
    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let bindings = std::collections::BTreeMap::from([
        (
            "x".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ),
        (
            "y".into(),
            TensorData::new([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        ),
    ]);
    let replayed = decoded.replay(&bindings).unwrap();
    assert_eq!(
        replayed[0].storage(),
        &crate::Storage::F32(vec![-9.0, -24.0, -16.0, -25.0, -21.0, -24.0])
    );
    let native = decoded
        .replay_with_options(
            &bindings,
            &crate::CapturedReplayExecutor::default(),
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(native.outputs[0].storage(), replayed[0].storage());
}

#[test]
fn scalar_affine_fusion_shares_one_map_and_rejects_two_maps() {
    let mut shared_graph = Graph::new();
    let x = shared_graph.input_dtype("x", [2, 3], DType::F32);
    let producer = shared_graph.square(x).unwrap();
    let view = shared_graph.permute(producer, [1, 0]).unwrap();
    let doubled = shared_graph.add(view, view).unwrap();
    let output = shared_graph.add(doubled, view).unwrap();
    let shared = schedule(&shared_graph, output).unwrap();
    shared.validate().unwrap();
    assert_eq!(shared.items.len(), 1);
    assert_eq!(
        shared.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .filter(|node| matches!(
                node.operation(),
                crate::Operation::GraphBinary(crate::BinaryOp::Mul)
            ))
            .count(),
        1
    );

    let mut distinct_graph = Graph::new();
    let x = distinct_graph.input_dtype("x", [2, 3], DType::F32);
    let producer = distinct_graph.square(x).unwrap();
    let first = distinct_graph.permute(producer, [1, 0]).unwrap();
    let second = distinct_graph
        .stride(
            first,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let output = distinct_graph.add(first, second).unwrap();
    let distinct = schedule(&distinct_graph, output).unwrap();
    distinct.validate().unwrap();
    assert!(distinct.items.iter().any(|item| item.node == producer));
    crate::MemoryPlan::from_schedule(&distinct, &[output], true).unwrap();
}

#[test]
fn scalar_affine_fusion_owns_shared_intermediate_equivalent_view_paths() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let producer = graph.square(input).unwrap();
    let shared = graph.permute(producer, [1, 0]).unwrap();
    let first = graph.reshape(shared, [1, 3, 2]).unwrap();
    let second = graph.reshape(shared, [3, 2, 1]).unwrap();
    let second = graph.permute(second, [2, 0, 1]).unwrap();
    let output = graph.add(first, second).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert!(
        [producer, shared, first, second]
            .into_iter()
            .all(|removed| scheduled.items.iter().all(|item| item.node != removed))
    );
    assert_eq!(
        scheduled.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input]
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();

    let capture = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let bytes = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    let replayed = decoded
        .replay(&std::collections::BTreeMap::from([(
            "input".into(),
            TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(
        replayed[0].storage(),
        &crate::Storage::F32(vec![2.0, 32.0, 8.0, 50.0, 18.0, 72.0])
    );
}

#[test]
fn scalar_affine_fusion_uses_normalized_loads_and_exact_materialized_dependencies() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let lhs = graph.input_dtype("lhs", [1, 2], DType::F32);
    let rhs = graph.input_dtype("rhs", [2, 3], DType::F32);
    let matrix = graph.matmul(lhs, rhs).unwrap();
    let producer = graph.add(input, matrix).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.relu(view).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 2);
    assert!(!scheduled.items.iter().any(|item| item.node == producer));
    let matrix_item = scheduled
        .items
        .iter()
        .find(|item| item.node == matrix)
        .unwrap();
    let output_item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(output_item.dependencies, vec![matrix_item.id]);
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![input, matrix]
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();

    let mut pruned = Graph::new();
    let selected_input = pruned.input_dtype("selected", [2, 2], DType::F32);
    let lhs = pruned.input_dtype("lhs", [2, 2], DType::F32);
    let rhs = pruned.input_dtype("rhs", [2, 2], DType::F32);
    let stale = pruned.matmul(lhs, rhs).unwrap();
    let condition = pruned.constant(TensorData::scalar_with_dtype(
        crate::Scalar::Bool(true),
        DType::Bool,
    ));
    let selected = pruned.square(selected_input).unwrap();
    let mapped = pruned.permute(selected, [1, 0]).unwrap();
    let output = pruned.select(condition, mapped, stale).unwrap();
    let scheduled = schedule(&pruned, output).unwrap();
    scheduled.validate().unwrap();
    let output_item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![selected_input]
    );
    assert!(output_item.dependencies.is_empty());
    assert!(scheduled.items.iter().all(|item| item.node != selected));
    let stale_item = scheduled
        .items
        .iter()
        .find(|item| item.node == stale)
        .expect("the specialized false branch remains independently materialized");
    assert!(!output_item.dependencies.contains(&stale_item.id));
    assert!(
        output_item
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::GraphBinary(crate::BinaryOp::Mul)
            ))
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();
}

#[test]
fn scalar_affine_fusion_preserves_observable_faulting_and_specialized_roots() {
    fn mapped_output(graph: &mut Graph) -> (crate::NodeId, crate::NodeId) {
        let input = graph.input_dtype("input", [2, 3], DType::F32);
        let producer = graph.square(input).unwrap();
        let view = graph.permute(producer, [1, 0]).unwrap();
        (producer, graph.relu(view).unwrap())
    }

    let mut requested_graph = Graph::new();
    let (producer, output) = mapped_output(&mut requested_graph);
    let requested = schedule_many(&requested_graph, &[producer, output]).unwrap();
    assert!(requested.items.iter().any(|item| item.node == producer));

    let sibling = requested_graph.neg(producer).unwrap();
    let shared = schedule_many(&requested_graph, &[output, sibling]).unwrap();
    assert!(shared.items.iter().any(|item| item.node == producer));

    let symbolic = crate::schedule::schedule_many_for_symbolic_capture(
        &requested_graph,
        &[output],
        &std::collections::BTreeSet::new(),
    )
    .unwrap();
    assert!(symbolic.items.iter().any(|item| item.node == producer));

    let mut external_graph = Graph::new();
    let (producer, output) = mapped_output(&mut external_graph);
    let external =
        schedule_with_external_materializations(&external_graph, &[output], &[producer]).unwrap();
    assert_eq!(external.items.len(), 1);
    let output_item = external
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![producer]
    );
    assert!(
        output_item
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| {
                matches!(
                    node.operation(),
                    crate::Operation::Index(crate::IndexValue::View { buffer, .. })
                        if *buffer == producer.index() as u64
                )
            })
    );
    assert_eq!(output_item.external_materializations, vec![producer]);
    assert!(output_item.dependencies.is_empty());

    let mut specialized_external_graph = Graph::new();
    let input = specialized_external_graph.input_dtype("input", [2, 3], DType::F32);
    let producer = specialized_external_graph.square(input).unwrap();
    let view = specialized_external_graph
        .permute(producer, [1, 0])
        .unwrap();
    let rhs = specialized_external_graph.input_dtype("rhs", [2, 4], DType::F32);
    let output = specialized_external_graph.matmul(view, rhs).unwrap();
    let specialized_external = schedule_with_external_materializations(
        &specialized_external_graph,
        &[output],
        &[producer],
    )
    .unwrap();
    specialized_external.validate().unwrap();
    assert_eq!(specialized_external.items.len(), 2);
    let view_item = specialized_external
        .items
        .iter()
        .find(|item| item.node == view)
        .unwrap();
    let output_item = specialized_external
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(view_item.external_materializations, vec![producer]);
    assert!(view_item.dependencies.is_empty());
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![view, rhs]
    );
    assert_eq!(output_item.dependencies, vec![view_item.id]);
    crate::MemoryPlan::from_schedule(&specialized_external, &[output], true).unwrap();

    let mut faulting_graph = Graph::new();
    let lhs = faulting_graph.input_dtype("lhs", [2, 3], DType::I32);
    let rhs = faulting_graph.input_dtype("rhs", [2, 3], DType::I32);
    let producer = faulting_graph
        .binary(crate::BinaryOp::Div, lhs, rhs)
        .unwrap();
    let view = faulting_graph.permute(producer, [1, 0]).unwrap();
    let output = faulting_graph.neg(view).unwrap();
    let faulting = schedule(&faulting_graph, output).unwrap();
    assert!(faulting.items.iter().any(|item| item.node == producer));

    let mut specialized_graph = Graph::new();
    let lhs = specialized_graph.input_dtype("lhs", [2, 2], DType::F32);
    let rhs = specialized_graph.input_dtype("rhs", [2, 2], DType::F32);
    let producer = specialized_graph.matmul(lhs, rhs).unwrap();
    let view = specialized_graph.permute(producer, [1, 0]).unwrap();
    let output = specialized_graph.relu(view).unwrap();
    let specialized = schedule(&specialized_graph, output).unwrap();
    assert!(specialized.items.iter().any(|item| item.node == producer));
}

#[test]
fn affine_contiguous_inventory_uses_normalized_select_loads() {
    let mut graph = Graph::new();
    let selected = graph.input_dtype("selected", [2, 2], DType::F32);
    let lhs = graph.input_dtype("lhs", [2, 2], DType::F32);
    let rhs = graph.input_dtype("rhs", [2, 2], DType::F32);
    let unselected = graph.matmul(lhs, rhs).unwrap();
    let condition = graph.constant(TensorData::scalar_with_dtype(
        crate::Scalar::Bool(true),
        DType::Bool,
    ));
    let producer = graph.select(condition, selected, unselected).unwrap();
    let permuted = graph.permute(producer, [1, 0]).unwrap();
    let output = graph.contiguous(permuted).unwrap();

    let scheduled = schedule(&graph, output).unwrap();
    scheduled.validate().unwrap();
    let unselected_item = scheduled
        .items
        .iter()
        .find(|item| item.node == unselected)
        .expect("the specialized Matmul branch remains independently materialized");
    let output_item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(
        output_item
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![selected]
    );
    assert_eq!(
        output_item
            .inputs
            .iter()
            .map(|desc| desc.id)
            .collect::<Vec<_>>(),
        vec![selected.index() as u64]
    );
    assert!(output_item.dependencies.is_empty());
    assert!(!output_item.dependencies.contains(&unselected_item.id));
    assert!(
        output_item
            .kernel
            .topological()
            .unwrap()
            .iter()
            .all(|node| {
                !matches!(
                    node.operation(),
                    crate::Operation::Index(crate::IndexValue::Buffer { buffer, .. })
                        | crate::Operation::Index(crate::IndexValue::View { buffer, .. })
                        if *buffer == unselected.index() as u64
                )
            })
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[output], true).unwrap();
}

#[test]
fn affine_contiguous_redirects_reshape_and_shrink_geometry() {
    let mut reshape_graph = Graph::new();
    let input = reshape_graph.input_dtype("input", [2, 3], DType::F32);
    let producer = reshape_graph.square(input).unwrap();
    let reshaped = reshape_graph.reshape(producer, [3, 2]).unwrap();
    let output = reshape_graph.contiguous(reshaped).unwrap();
    let scheduled = schedule(&reshape_graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let source_shape = Shape::from([2, 3]);
    let logical_shape = Shape::from([3, 2]);
    assert!(
        scheduled.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::Index(crate::IndexValue::View { view, .. })
                    if view.source_shape == source_shape
                        && view.logical_shape == logical_shape
                        && view.strides.as_slice() == [2, 1]
            ))
    );

    let mut shrink_graph = Graph::new();
    let input = shrink_graph.input_dtype("input", [3, 4], DType::F32);
    let producer = shrink_graph.square(input).unwrap();
    let shrunk = shrink_graph.shrink(producer, [(1, 3), (1, 4)]).unwrap();
    let output = shrink_graph.contiguous(shrunk).unwrap();
    let scheduled = schedule(&shrink_graph, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert!(
        scheduled.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::Index(crate::IndexValue::View { view, .. })
                    if view.offset == 5 && view.strides.as_slice() == [4, 1]
            ))
    );
}

#[test]
fn affine_contiguous_preserves_requested_shared_and_nested_view_fallbacks() {
    let mut requested_graph = Graph::new();
    let input = requested_graph.input_dtype("input", [2, 3], DType::F32);
    let producer = requested_graph.square(input).unwrap();
    let permuted = requested_graph.permute(producer, [1, 0]).unwrap();
    let output = requested_graph.contiguous(permuted).unwrap();
    let requested = schedule_many(&requested_graph, &[permuted, output]).unwrap();
    requested.validate().unwrap();
    assert_eq!(requested.items.len(), 3);
    assert!(requested.items.iter().any(|item| item.node == producer));
    assert!(matches!(
        requested
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
    ));
    let requested_copy = requested
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert_eq!(
        requested_copy
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![producer]
    );
    let producer_item = requested
        .items
        .iter()
        .find(|item| item.node == producer)
        .unwrap();
    assert_eq!(requested_copy.dependencies, vec![producer_item.id]);
    crate::MemoryPlan::from_schedule(&requested, &[permuted, output], true).unwrap();
    crate::CapturedSchedule::capture(&requested_graph, &requested, &[permuted, output]).unwrap();

    let sibling = requested_graph.neg(permuted).unwrap();
    let shared = schedule_many(&requested_graph, &[output, sibling]).unwrap();
    shared.validate().unwrap();
    assert!(shared.items.iter().any(|item| item.node == producer));
    assert!(matches!(
        shared
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
    ));

    let mut nested = Graph::new();
    let input = nested.input_dtype("input", [2, 3], DType::F32);
    let viewed = nested.permute(input, [1, 0]).unwrap();
    let rhs = nested.input_dtype("rhs", [3, 2], DType::F32);
    let producer = nested.add(viewed, rhs).unwrap();
    let reversed = nested
        .stride(
            producer,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let output = nested.contiguous(reversed).unwrap();
    let scheduled = schedule(&nested, output).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 2);
    assert!(matches!(
        scheduled
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap()
            .kernel
            .operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(&plan.kind, crate::MovementKernelKind::AffineCopy { .. })
    ));
}

#[test]
fn computed_affine_read_outputs_share_one_materialized_producer() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 1], DType::F32);
    let producer = graph.square(input).unwrap();
    let expanded = graph.expand(producer, [2, 3]).unwrap();
    let scheduled = schedule_many(&graph, &[producer, expanded]).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.items.len(), 2);
    assert_eq!(scheduled.items[0].node, producer);
    assert_eq!(scheduled.items[1].node, expanded);
    assert_eq!(scheduled.items[1].dependencies, vec![scheduled.items[0].id]);
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
        scheduled.items[1].kernel.operation()
    else {
        panic!("computed view must materialize through one movement plan")
    };
    let crate::MovementKernelKind::AffineCopy {
        input: operand,
        view,
    } = &plan.kind
    else {
        panic!("computed view must use an affine read copy")
    };
    assert_eq!(operand.node, producer);
    assert_eq!(view.strides, vec![1, 0]);
    assert_eq!(scheduled.items[1].outputs.primary().view, None);
}

#[test]
fn diagonal_schedules_pad_then_affine_read_copy_and_keeps_zero_exact() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [3, 3], DType::F32);
    let diagonal = graph.diagonal_default(input).unwrap();
    let scheduled = schedule(&graph, diagonal).unwrap();
    scheduled.validate().unwrap();
    let pad = scheduled
        .items
        .iter()
        .find(|item| {
            matches!(
                item.kernel.operation(),
                crate::Operation::Movement(crate::MovementValue::Plan(plan))
                    if matches!(&plan.kind, crate::MovementKernelKind::Pad { .. })
            )
        })
        .unwrap();
    let copied = scheduled
        .items
        .iter()
        .find(|item| item.node == diagonal)
        .unwrap();
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = copied.kernel.operation()
    else {
        panic!("diagonal output must be an affine copy")
    };
    let crate::MovementKernelKind::AffineCopy {
        input: operand,
        view,
    } = &plan.kind
    else {
        panic!("diagonal output must be an affine copy")
    };
    assert_eq!(operand.node, pad.node);
    assert_eq!(view.logical_shape, Shape::from([3]));
    assert_eq!(view.strides, vec![4]);
    assert!(copied.dependencies.contains(&pad.id));

    let mut zero = Graph::new();
    let input = zero.input_dtype("input", [0, 3], DType::F32);
    let diagonal = zero.diagonal_default(input).unwrap();
    let scheduled = schedule(&zero, diagonal).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(zero.shape(diagonal).unwrap(), &Shape::from([0]));
    assert!(scheduled.items.iter().all(|item| !matches!(
        item.kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(
                &plan.kind,
                crate::MovementKernelKind::Pad { .. }
                    | crate::MovementKernelKind::AffineCopy { .. }
            )
    )));
}

#[test]
fn computed_reverse_view_retains_signed_affine_read_metadata() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F32);
    let producer = graph.square(input).unwrap();
    let reversed = graph
        .stride(
            producer,
            [crate::Slice {
                start: None,
                stop: None,
                step: -1,
            }],
        )
        .unwrap();
    let scheduled = schedule(&graph, reversed).unwrap();
    scheduled.validate().unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == reversed)
        .unwrap();
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) = item.kernel.operation()
    else {
        panic!("computed reverse must be a movement plan")
    };
    let crate::MovementKernelKind::AffineCopy { view, .. } = &plan.kind else {
        panic!("computed reverse must be an affine copy")
    };
    assert_eq!(view.offset, 3);
    assert_eq!(view.strides, vec![-1]);
    assert_eq!(item.outputs.primary().view, None);
}

#[test]
fn computed_affine_materialization_preserves_graph_vjp_routing() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 1], DType::F32);
    let producer = graph.square(input).unwrap();
    let expanded = graph.expand(producer, [2, 3]).unwrap();
    let reversed = graph
        .stride(
            expanded,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    let loss = graph.sum_all(reversed).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([2, 1]));
    schedule(&graph, gradient).unwrap().validate().unwrap();
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::new([2, 1], vec![2.0, -3.0]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![12.0, -18.0]
    );
}

#[test]
fn typed_unbroadcast_vjp_is_an_owned_scheduleable_reduction_chain() {
    let mut graph = Graph::new();
    let target = graph.input_dtype("target", [1, 2], DType::BF16);
    let other = graph.input_dtype("other", [3, 2], DType::BF16);
    let output = graph.add(target, other).unwrap();
    let seed = graph.input_dtype_requires_grad("seed", [3, 2], DType::BF16, false);
    let gradient = graph.grad_with(output, target, Some(seed), true).unwrap();

    assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([1, 2]));
    assert_eq!(graph.dtype(gradient).unwrap(), DType::BF16);
    assert!(
        (0..graph.node_count())
            .map(crate::NodeId::from_index)
            .all(|node| !matches!(graph.op(node).unwrap(), crate::Op::SumTo { .. }))
    );

    let scheduled = schedule(&graph, gradient).unwrap();
    scheduled.validate().unwrap();
    assert!(scheduled.items.iter().all(|item| item.boundary.is_none()));
    assert!(scheduled.items.iter().any(|item| {
        item.kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(node.operation(), crate::Operation::ReduceFinalize))
    }));
    assert!(
        scheduled
            .items
            .iter()
            .flat_map(|item| item.ordered_inputs())
            .any(|binding| binding.input_node == seed)
    );
    crate::MemoryPlan::from_schedule(&scheduled, &[gradient], true).unwrap();
}

#[test]
fn generalized_raw_matmul_vjp_has_owned_bindings_dependencies_and_memory() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 1, 2, 3]);
    let rhs = graph.input("rhs", [1, 2, 3, 2]);
    let output = graph.matmul(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradients = graph.gradient_default(loss, &[lhs, rhs]).unwrap();
    assert!(
        (0..graph.node_count())
            .map(crate::NodeId::from_index)
            .all(|node| !matches!(
                graph.op(node).unwrap(),
                crate::Op::MatmulGrad { .. } | crate::Op::MatmulGradVjp { .. }
            ))
    );

    let scheduled = schedule_many(&graph, &gradients).unwrap();
    scheduled.validate().unwrap();
    assert!(scheduled.items.iter().all(|item| item.boundary.is_none()));
    assert!(scheduled.items.iter().any(|item| {
        item.kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(node.operation(), crate::Operation::ReduceFinalize))
    }));
    let bound = scheduled
        .items
        .iter()
        .flat_map(|item| item.ordered_inputs())
        .map(|binding| binding.input_node)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(bound.contains(&lhs));
    assert!(bound.contains(&rhs));
    crate::MemoryPlan::from_schedule(&scheduled, &gradients, true).unwrap();
}

#[test]
fn scheduled_outputs_are_nonempty_ordered_and_define_cache_identity() {
    let output = buffer(7, 4, 1);
    assert!(ScheduledOutputs::new(vec![]).is_err());
    assert!(ScheduledOutputs::new(vec![output.clone(), output.clone()]).is_err());

    let single = item(0, vec![], output.clone());
    let canonical_identity = crate::schedule::item_cache_key(&single).unwrap();
    let mut rebuilt = single.clone();
    rebuilt.outputs = ScheduledOutputs::new(vec![output.clone()]).unwrap();
    assert_eq!(
        crate::schedule::item_cache_key(&rebuilt).unwrap(),
        canonical_identity
    );

    let mut second = output.clone();
    second.id = 8;
    let mut paired = single.clone();
    paired.outputs = ScheduledOutputs::new(vec![output.clone(), second]).unwrap();
    assert_ne!(
        crate::schedule::item_cache_key(&paired).unwrap(),
        canonical_identity
    );

    let live = crate::Schedule {
        items: vec![single],
        requested_passthroughs: vec![],
        value_bindings: vec![],
        state_bindings: vec![],
    };
    assert!(matches!(
        live.validate(),
        Err(crate::ScheduleError::Binding(reason))
            if reason == "scheduled single-output Sink has no Store"
    ));
}

#[test]
fn external_concat_materialization_makes_local_add_renderable() {
    let mut graph = Graph::new();
    let left = graph.input("left", Shape::from([1, 2]));
    let right = graph.input("right", Shape::from([1, 2]));
    let addend = graph.input("addend", Shape::from([1, 4]));
    let joined = graph.concat([left, right], 1).unwrap();
    let out = graph.add(joined, addend).unwrap();
    let direct = schedule(&graph, out).unwrap();
    assert!(direct.items.iter().all(|item| item.boundary.is_none()));
    assert!(
        direct
            .items
            .iter()
            .any(|item| matches!(item.kernel.operation(), crate::Operation::Movement(_)))
    );
    let scheduled = schedule_with_external_materializations(&graph, &[out], &[joined]).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == out)
        .unwrap();
    assert!(item.boundary.is_none());
    assert_eq!(item.external_materializations, vec![joined]);
    assert_eq!(item.ordered_inputs()[0].input_node, joined);
    assert!(
        item.kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(node.operation(), crate::Operation::Load))
    );
    assert!(schedule_with_external_materializations(&graph, &[out], &[out]).is_err());
    assert!(schedule_with_external_materializations(&graph, &[out], &[left]).is_err());
}

#[test]
fn ordered_input_bindings_follow_lowered_operand_not_node_id() {
    let mut graph = Graph::new();
    let right = graph.input("right", Shape::from([4]));
    let left = graph.input("left", Shape::from([4]));
    let out = graph.sub(left, right).unwrap();
    let scheduled = schedule(&graph, out).unwrap();
    let item = &scheduled.items[0];
    assert_eq!(
        item.inputs.iter().map(|x| x.id).collect::<Vec<_>>(),
        vec![right.index() as u64, left.index() as u64]
    );
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|x| x.input_node)
            .collect::<Vec<_>>(),
        vec![left, right]
    );
    assert_eq!(
        item.ordered_inputs()
            .iter()
            .map(|x| x.abi_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    item.validate_input_bindings().unwrap();
    let mut missing = item.clone();
    missing.input_bindings.pop();
    assert!(missing.validate_input_bindings().is_err());
    let mut malformed = item.clone();
    malformed.input_bindings[1].abi_index = 0;
    assert!(malformed.validate_input_bindings().is_err());
}

#[test]
fn scalar_elementwise_schedule_is_deterministic_and_lowered() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::new([]));
    let one = graph.constant(TensorData::scalar(1.0));
    let y = graph.add(x, one).unwrap();
    let first = schedule(&graph, y).unwrap();
    let second = schedule(&graph, y).unwrap();
    assert_eq!(first.items.len(), 1);
    assert_eq!(first.items[0].cache_key, second.items[0].cache_key);
    assert!(first.items[0].boundary.is_none());
    assert_eq!(first.items[0].inputs.len(), 2);
    assert_eq!(first.items[0].ordered_inputs().len(), 1);
    assert_eq!(first.items[0].ordered_inputs()[0].input_node, x);
    assert!(
        first.items[0]
            .inputs
            .iter()
            .any(|input| input.id == one.index() as u64)
    );
    first.items[0].kernel.validate().unwrap();
}

#[test]
fn producer_aware_dag_is_topological_and_deterministic() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2]));
    let shared = graph.square(x).unwrap();
    let one = graph.constant(TensorData::scalar(1.0));
    let left = graph.add(shared, one).unwrap();
    let right = graph.mul(shared, one).unwrap();
    let first = schedule_many(&graph, &[left, right]).unwrap();
    let second = schedule_many(&graph, &[left, right]).unwrap();
    assert_eq!(first.items.len(), 3);
    assert_eq!(
        first
            .items
            .iter()
            .map(|item| (item.id, item.dependencies.clone(), item.cache_key))
            .collect::<Vec<_>>(),
        second
            .items
            .iter()
            .map(|item| (item.id, item.dependencies.clone(), item.cache_key))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first.items[0].consumers,
        vec![first.items[1].id, first.items[2].id]
    );
    assert!(
        first.items[1]
            .dependencies
            .iter()
            .all(|dependency| *dependency < first.items[1].id)
    );
}

#[test]
fn schedule_validation_requires_canonical_ordered_reverse_edges() {
    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([2]));
    let shared = graph.square(input).unwrap();
    let one = graph.constant(TensorData::scalar(1.0));
    let left = graph.add(shared, one).unwrap();
    let right = graph.mul(shared, one).unwrap();
    let schedule = schedule_many(&graph, &[left, right]).unwrap();
    schedule.validate().unwrap();
    let cache_keys = schedule
        .items
        .iter()
        .map(|item| item.cache_key)
        .collect::<Vec<_>>();
    let repeated = schedule_many(&graph, &[left, right]).unwrap();
    repeated.validate().unwrap();
    assert_eq!(
        repeated
            .items
            .iter()
            .map(|item| item.cache_key)
            .collect::<Vec<_>>(),
        cache_keys,
        "canonical validation must preserve fixed schedule cache identities"
    );

    let mut duplicate_dependency = schedule.clone();
    let dependent = duplicate_dependency
        .items
        .iter()
        .position(|item| item.dependencies.contains(&0))
        .unwrap();
    let original_dependencies = duplicate_dependency.items[dependent].dependencies.clone();
    duplicate_dependency.items[dependent].dependencies.push(0);
    assert!(duplicate_dependency.validate().is_err());
    assert_eq!(
        duplicate_dependency.items[dependent].dependencies,
        [original_dependencies, vec![0]].concat(),
        "validation must not normalize a malformed schedule in place"
    );

    let mut stale_consumer = schedule.clone();
    let original_consumers = stale_consumer.items[0].consumers.clone();
    stale_consumer.items[0].consumers.push(99);
    assert!(stale_consumer.validate().is_err());
    assert_eq!(
        stale_consumer.items[0].consumers,
        [original_consumers, vec![99]].concat()
    );
    assert!(matches!(
        crate::MemoryPlan::from_schedule(&stale_consumer, &[left, right], true),
        Err(crate::MemoryPlanError::InvalidSchedule(_))
    ));
    assert!(crate::CapturedSchedule::capture(&graph, &stale_consumer, &[left, right]).is_err());

    let mut forward_dependency = schedule.clone();
    forward_dependency.items[0].dependencies.push(1);
    forward_dependency.items[1].consumers.push(0);
    assert!(forward_dependency.validate().is_err());
    assert_eq!(
        forward_dependency
            .items
            .iter()
            .map(|item| item.cache_key)
            .collect::<Vec<_>>(),
        cache_keys,
        "rejection must not mutate cache identities"
    );

    let mut independent = Graph::new();
    let lhs = independent.input("lhs", Shape::from([2]));
    let rhs = independent.input("rhs", Shape::from([2]));
    let left = independent.neg(lhs).unwrap();
    let right = independent.neg(rhs).unwrap();
    let mut reordered = schedule_many(&independent, &[left, right]).unwrap();
    reordered.validate().unwrap();
    reordered.items.swap(0, 1);
    assert!(reordered.validate().is_err());
}

#[test]
fn schedule_descriptor_validation_rejects_before_memory_or_capture_work() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::F32);
    let output = graph.neg(input).unwrap();
    let schedule = schedule(&graph, output).unwrap();
    let cache_keys = schedule
        .items
        .iter()
        .map(|item| item.cache_key)
        .collect::<Vec<_>>();

    let mut wrong_bytes = schedule.clone();
    let mut output_desc = wrong_bytes.items[0].primary_output().clone();
    output_desc.bytes += 1;
    wrong_bytes.items[0].outputs = ScheduledOutputs::single(output_desc);
    assert!(matches!(
        wrong_bytes.validate(),
        Err(crate::ScheduleError::Binding(message))
            if message == "buffer descriptor byte size mismatch"
    ));
    assert_eq!(
        wrong_bytes
            .items
            .iter()
            .map(|item| item.cache_key)
            .collect::<Vec<_>>(),
        cache_keys,
        "descriptor rejection must not rewrite cache identities"
    );
    assert!(crate::MemoryPlan::from_schedule(&wrong_bytes, &[output], true).is_err());
    assert!(crate::CapturedSchedule::capture(&graph, &wrong_bytes, &[output]).is_err());

    let mut wrong_alignment = schedule.clone();
    wrong_alignment.items[0].inputs[0].alignment = 3;
    assert!(matches!(
        wrong_alignment.validate(),
        Err(crate::ScheduleError::Binding(message))
            if message == "buffer descriptor alignment is invalid"
    ));

    let mut wrong_view = schedule;
    wrong_view.items[0].inputs[0].view = Some(crate::AffineView::identity(Shape::from([3])));
    assert!(matches!(
        wrong_view.validate(),
        Err(crate::ScheduleError::Binding(message))
            if message == "buffer descriptor view source shape mismatch"
    ));
}
#[test]
fn nonscalar_is_lowered_and_unsupported_nodes_are_visible_boundaries() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2]));
    let y = graph.neg(x).unwrap();
    let item = &schedule(&graph, y).unwrap().items[0];
    assert_eq!(item.boundary, None);
    item.kernel.validate().unwrap();
    for kind in [
        crate::ReduceKind::Sum,
        crate::ReduceKind::Mean,
        crate::ReduceKind::Product,
        crate::ReduceKind::Min,
        crate::ReduceKind::Max,
    ] {
        let reduced = graph.reduce(y, kind, Some(vec![0]), false).unwrap();
        let item = &schedule(&graph, reduced).unwrap().items[0];
        assert!(item.boundary.is_none(), "{kind:?}");
        item.kernel.validate().unwrap();
    }
}

#[test]
fn matmul_materializes_computed_operands_and_participates_in_lifetimes() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F32);
    let rhs = graph.input_dtype("rhs", [3, 2], DType::F32);
    let bias = graph.input_dtype("bias", [2, 2], DType::F32);
    let lhs = graph.square(input).unwrap();
    let product = graph.matmul(lhs, rhs).unwrap();
    let output = graph.add(product, bias).unwrap();
    let first = schedule(&graph, output).unwrap();
    let second = schedule(&graph, output).unwrap();
    assert_eq!(first.items.len(), 3);
    assert_eq!(
        first
            .items
            .iter()
            .map(|item| (item.node, item.dependencies.clone(), item.cache_key))
            .collect::<Vec<_>>(),
        second
            .items
            .iter()
            .map(|item| (item.node, item.dependencies.clone(), item.cache_key))
            .collect::<Vec<_>>()
    );
    let matmul = first
        .items
        .iter()
        .find(|item| item.node == product)
        .unwrap();
    assert!(matches!(
        matmul.kernel.operation(),
        crate::Operation::Matmul(_)
    ));
    assert_eq!(matmul.dependencies.len(), 1);
    assert_eq!(
        matmul
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![lhs, rhs]
    );
    let temporaries = first.internal_temporaries(&[output]);
    assert_eq!(
        temporaries
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        vec![lhs.index() as u64, product.index() as u64]
    );
    let memory = plan_temporary_reuse(&first.items, &temporaries).unwrap();
    assert_eq!(memory.temporaries.len(), 2);
}

#[test]
fn sum_and_mean_schedule_to_accumulator_uops() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2, 3]));
    let mean = graph
        .reduce(x, crate::ReduceKind::Mean, Some(vec![-1]), false)
        .unwrap();
    let item = &schedule(&graph, mean).unwrap().items[0];
    assert!(item.boundary.is_none());
    item.kernel.validate().unwrap();
    assert!(format!("{}", item.kernel).contains("ReduceFinalize"));
}

#[test]
fn single_reduction_epilogue_is_one_item_without_intermediate_storage() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2, 3]));
    let bias = graph.input("bias", Shape::from([2]));
    let reduced = graph.sum(x, 1).unwrap();
    let output = graph.add(reduced, bias).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert!(item.boundary.is_none());
    assert!(item.dependencies.is_empty());
    assert_eq!(item.node, output);
    assert!(
        item.inputs
            .iter()
            .all(|input| input.id != reduced.index() as u64)
    );
    let store = item
        .kernel
        .sources()
        .iter()
        .find(|node| matches!(node.operation(), crate::Operation::Store))
        .unwrap();
    let kernel = crate::reduction_native::NativeReductionKernel::from_store(store)
        .unwrap()
        .unwrap();
    assert!(kernel.has_epilogue());
    assert_eq!(kernel.output_dtype, DType::F32);

    let values = HashMap::from([
        (
            "x".into(),
            TensorData::new(Shape::from([2, 3]), vec![1.0, 2.0, 3.0, -4.0, 1.0, 2.0]).unwrap(),
        ),
        (
            "bias".into(),
            TensorData::new(Shape::from([2]), vec![0.5, 2.0]).unwrap(),
        ),
    ]);
    let result = CpuBackend.execute(&graph, output, &values).unwrap();
    assert_eq!(result.storage(), &crate::Storage::F32(vec![6.5, 1.0]));
}

#[test]
fn reduction_epilogue_fusion_respects_requested_shared_and_shape_boundaries() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2, 3]));
    let reduced = graph.sum(x, 1).unwrap();
    let one = graph.constant(TensorData::new(Shape::new([]), vec![1.0]).unwrap());
    let epilogue = graph.add(reduced, one).unwrap();

    let requested = schedule_many(&graph, &[reduced, epilogue]).unwrap();
    assert_eq!(requested.items.len(), 2);

    let other = graph.mul(reduced, one).unwrap();
    let shared = schedule_many(&graph, &[epilogue, other]).unwrap();
    assert!(shared.items.iter().any(|item| item.node == reduced));

    let two = graph.constant(TensorData::new(Shape::new([]), vec![2.0]).unwrap());
    let outer = graph.mul(epilogue, two).unwrap();
    let nearest = schedule_many(&graph, &[epilogue, outer]).unwrap();
    assert_eq!(nearest.items.len(), 2);
    let inner_item = nearest
        .items
        .iter()
        .find(|item| item.node == epilogue)
        .unwrap();
    let outer_item = nearest
        .items
        .iter()
        .find(|item| item.node == outer)
        .unwrap();
    assert!(
        crate::reduction_native::NativeReductionKernel::from_store(
            inner_item
                .kernel
                .sources()
                .iter()
                .find(|node| matches!(node.operation(), crate::Operation::Store))
                .unwrap(),
        )
        .unwrap()
        .is_some()
    );
    assert_eq!(outer_item.dependencies, vec![inner_item.id]);
    assert!(
        outer_item
            .inputs
            .iter()
            .any(|input| input.id == epilogue.index() as u64)
    );
    let inputs = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )]);
    let realized = crate::realize(
        &graph,
        &nearest,
        &[epilogue, outer],
        &inputs,
        crate::RealizationPolicy::Interpreter,
    )
    .unwrap();
    assert_eq!(
        realized.outputs[0].storage(),
        &crate::Storage::F32(vec![7.0, 16.0])
    );
    assert_eq!(
        realized.outputs[1].storage(),
        &crate::Storage::F32(vec![14.0, 32.0])
    );

    let moved = graph.reshape(reduced, Shape::from([1, 2])).unwrap();
    let mixed_use = schedule_many(&graph, &[epilogue, moved]).unwrap();
    assert!(mixed_use.items.iter().any(|item| item.node == reduced));

    let expanded = graph.expand(reduced, Shape::from([2, 2])).unwrap();
    let shape_changing = schedule(&graph, expanded).unwrap();
    assert!(shape_changing.items.iter().any(|item| item.node == reduced));

    let second_reduction = graph.sum(epilogue, 0).unwrap();
    let two_reductions = schedule(&graph, second_reduction).unwrap();
    assert!(two_reductions.items.iter().any(|item| item.node == reduced));

    let external =
        schedule_with_external_materializations(&graph, &[epilogue], &[reduced]).unwrap();
    assert!(
        external.items.iter().any(|item| {
            item.node == epilogue && item.external_materializations == vec![reduced]
        })
    );

    let mut integer_graph = Graph::new();
    let integers = integer_graph.input_dtype("integers", Shape::from([2, 3]), DType::I32);
    let divisor = integer_graph.input_dtype("divisor", Shape::from([2]), DType::I32);
    let integer_sum = integer_graph
        .reduce_with_dtypes(
            integers,
            crate::ReduceKind::Sum,
            Some(vec![1]),
            false,
            crate::ReductionDType::new(DType::I32, DType::I32),
        )
        .unwrap();
    let faulting = integer_graph
        .binary(crate::BinaryOp::Div, integer_sum, divisor)
        .unwrap();
    let faulting_schedule = schedule(&integer_graph, faulting).unwrap();
    assert!(
        faulting_schedule
            .items
            .iter()
            .any(|item| item.node == integer_sum)
    );

    let mut narrow_graph = Graph::new();
    let narrow = narrow_graph.input_dtype("narrow", [2, 3], DType::F16);
    let narrow_sum = narrow_graph
        .reduce_with_dtypes(
            narrow,
            crate::ReduceKind::Sum,
            Some(vec![1]),
            false,
            crate::ReductionDType::new(DType::F16, DType::F16),
        )
        .unwrap();
    let one = narrow_graph.constant(
        TensorData::from_storage(Shape::new([]), crate::Storage::F16(vec![0x3c00])).unwrap(),
    );
    let sibling_input = narrow_graph.input_dtype("sibling", [2], DType::F16);
    let sibling_sum = narrow_graph.add(sibling_input, sibling_input).unwrap();
    let sibling = narrow_graph.mul(sibling_sum, one).unwrap();
    let second = narrow_graph.add(narrow_sum, sibling).unwrap();
    let narrow_schedule = schedule(&narrow_graph, second).unwrap();
    assert!(
        narrow_schedule
            .items
            .iter()
            .any(|item| item.node == narrow_sum)
    );
    let narrow_inputs = HashMap::from([
        (
            "narrow".into(),
            TensorData::from_storage(
                [2, 3],
                crate::Storage::F16(vec![0x6800, 0x3c00, 0xe800, 0x8000, 0, 0]),
            )
            .unwrap(),
        ),
        (
            "sibling".into(),
            TensorData::from_storage([2], crate::Storage::F16(vec![0x3c00, 0x8000])).unwrap(),
        ),
    ]);
    let expected = CpuBackend
        .execute(&narrow_graph, second, &narrow_inputs)
        .unwrap();
    let actual = crate::realize(
        &narrow_graph,
        &narrow_schedule,
        &[second],
        &narrow_inputs,
        crate::RealizationPolicy::Interpreter,
    )
    .unwrap();
    assert_eq!(actual.outputs[0].storage(), expected.storage());
}

#[test]
fn shared_epilogue_owns_reduction_before_two_requested_siblings() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2, 3]));
    let reduced = graph.sum(x, 1).unwrap();
    let one = graph.constant(TensorData::new(Shape::new([]), vec![1.0]).unwrap());
    let shared = graph.add(reduced, one).unwrap();
    let two = graph.constant(TensorData::new(Shape::new([]), vec![2.0]).unwrap());
    let three = graph.constant(TensorData::new(Shape::new([]), vec![3.0]).unwrap());
    let left = graph.mul(shared, two).unwrap();
    let right = graph.mul(shared, three).unwrap();

    let scheduled = schedule_many(&graph, &[left, right]).unwrap();
    assert_eq!(scheduled.items.len(), 3);
    assert!(!scheduled.items.iter().any(|item| item.node == reduced));
    let shared_item = scheduled
        .items
        .iter()
        .find(|item| item.node == shared)
        .unwrap();
    assert!(
        crate::reduction_native::NativeReductionKernel::from_store(
            shared_item
                .kernel
                .sources()
                .iter()
                .find(|node| matches!(node.operation(), crate::Operation::Store))
                .unwrap(),
        )
        .unwrap()
        .is_some()
    );
    for output in [left, right] {
        let item = scheduled
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap();
        assert_eq!(item.dependencies, vec![shared_item.id]);
        assert!(
            item.inputs
                .iter()
                .any(|input| input.id == shared.index() as u64)
        );
        assert!(
            item.inputs
                .iter()
                .all(|input| input.id != reduced.index() as u64)
        );
    }

    let inputs = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )]);
    let realized = crate::realize(
        &graph,
        &scheduled,
        &[left, right],
        &inputs,
        crate::RealizationPolicy::Interpreter,
    )
    .unwrap();
    assert_eq!(
        realized.outputs[0].storage(),
        &crate::Storage::F32(vec![14.0, 32.0])
    );
    assert_eq!(
        realized.outputs[1].storage(),
        &crate::Storage::F32(vec![21.0, 48.0])
    );
}

#[test]
fn public_mean_and_min_compositions_keep_one_checked_reduction_kernel() {
    let mut mean_graph = Graph::new();
    let input = mean_graph.input("x", Shape::from([2, 3]));
    let mean = mean_graph
        .mean_with_axes(input, Some(vec![1]), false)
        .unwrap();
    let mean_schedule = schedule(&mean_graph, mean).unwrap();
    assert_eq!(mean_schedule.items.len(), 1);
    assert_eq!(
        mean_schedule.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .filter(|node| matches!(node.operation(), crate::Operation::ReduceFinalize))
            .count(),
        1
    );
    let values = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&mean_graph, mean, &values)
            .unwrap()
            .storage(),
        &crate::Storage::F32(vec![2.0, 5.0])
    );
    let default_mean = mean_graph.mean_default(input).unwrap();
    assert_eq!(schedule(&mean_graph, default_mean).unwrap().items.len(), 1);
    let gradient = mean_graph.grad(default_mean, input).unwrap();
    assert_eq!(
        CpuBackend
            .execute(&mean_graph, gradient, &values)
            .unwrap()
            .storage(),
        &crate::Storage::F32(vec![1.0 / 6.0; 6])
    );

    let mut min_graph = Graph::new();
    let input = min_graph.input("x", Shape::from([2, 3]));
    let minimum = min_graph
        .min_with_axes(input, Some(vec![1]), false)
        .unwrap();
    let min_schedule = schedule(&min_graph, minimum).unwrap();
    assert_eq!(min_schedule.items.len(), 1);
    min_schedule.items[0].kernel.validate().unwrap();
}

#[test]
fn temporary_reuse_is_deterministic_and_never_overlaps_or_mismatches() {
    let a = buffer(10, 16, 4);
    let b = buffer(11, 16, 4);
    let c = buffer(12, 16, 8);
    let items = vec![
        item(0, vec![], a.clone()),
        item(1, vec![], b.clone()),
        item(2, vec![], c.clone()),
    ];
    let first = plan_temporary_reuse(&items, &[a.clone(), b.clone(), c.clone()]).unwrap();
    let second = plan_temporary_reuse(&items, &[a, b, c]).unwrap();
    assert_eq!(first, second);
    let ids = first
        .temporaries
        .iter()
        .map(|entry| (entry.buffer_id, entry.allocation_id))
        .collect::<Vec<_>>();
    assert_eq!(ids[0].1, ids[1].1, "separated compatible temporaries reuse");
    assert_ne!(
        ids[1].1, ids[2].1,
        "alignment-incompatible temporary cannot reuse"
    );

    let malformed = buffer(13, 16, 3);
    assert!(matches!(
        plan_temporary_reuse(&[item(0, vec![], malformed.clone())], &[malformed]),
        Err(crate::MemoryPlanError::InvalidAlignment {
            buffer: 13,
            alignment: 3
        })
    ));
}

#[test]
fn sharded_graph_local_shrink_feeds_a_fused_binary_kernel() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([4, 2]));
    let rhs = graph.input("rhs", Shape::from([4, 2]));
    let group = crate::collective::DeviceGroup::new([
        crate::collective::DeviceId::new("cuda:0").unwrap(),
        crate::collective::DeviceId::new("cuda:1").unwrap(),
    ])
    .unwrap();
    let left = graph.shard_node(x, group.clone(), Some(0)).unwrap();
    let right = graph.shard_node(rhs, group, Some(0)).unwrap();
    let output = graph
        .sharded_binary(&left, &right, crate::BinaryOp::Add)
        .unwrap()
        .nodes()[0];
    let item = &schedule(&graph, output).unwrap().items[0];
    assert!(item.boundary.is_none());
    assert_eq!(
        item.inputs
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        vec![x.index() as u64, rhs.index() as u64]
    );
    assert!(
        item.inputs
            .iter()
            .all(|buffer| buffer.shape == Shape::from([4, 2]))
    );
    assert!(item.inputs.iter().all(|buffer| buffer.view.is_some()));
    assert!(
        item.kernel
            .topological()
            .unwrap()
            .iter()
            .any(|node| matches!(
                node.operation(),
                crate::Operation::Index(crate::IndexValue::View { .. })
            ))
    );
}
