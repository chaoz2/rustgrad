use crate::{
    BufferDesc, DType, Graph, ScheduleItem, Shape, TensorData, UOp, plan_temporary_reuse, schedule,
    schedule_many, schedule_with_external_materializations,
};

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
        output,
        kernel: UOp::sink(vec![]),
        boundary: None,
        cache_key: 0,
    }
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
            .any(|item| matches!(item.kernel.kind(), crate::UOpKind::Movement))
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
            .any(|node| matches!(node.kind(), crate::UOpKind::Load))
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
    assert!(matches!(matmul.kernel.kind(), crate::UOpKind::Matmul));
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
            .any(|node| matches!(node.arg(), crate::UArg::ViewBufferIndex { .. }))
    );
}
