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

    let schedule = crate::Schedule {
        items: vec![single],
        value_bindings: vec![],
        state_bindings: vec![],
    };
    schedule.validate().unwrap();
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
