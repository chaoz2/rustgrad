use crate::{
    BufferDesc, DType, Graph, ScheduleBoundary, ScheduleItem, Shape, TensorData, UOp,
    plan_temporary_reuse, schedule,
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
fn item(inputs: Vec<BufferDesc>, output: BufferDesc) -> ScheduleItem {
    ScheduleItem {
        inputs,
        output,
        kernel: UOp::sink(vec![]),
        boundary: None,
        cache_key: 0,
    }
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
fn nonscalar_is_lowered_and_unsupported_nodes_are_visible_boundaries() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([2]));
    let y = graph.neg(x).unwrap();
    let item = &schedule(&graph, y).unwrap().items[0];
    assert_eq!(item.boundary, None);
    item.kernel.validate().unwrap();
    let reduced = graph
        .reduce(y, crate::ReduceKind::Product, Some(vec![0]), false)
        .unwrap();
    let item = &schedule(&graph, reduced).unwrap().items[0];
    assert!(matches!(
        item.boundary,
        Some(ScheduleBoundary::Unsupported(_))
    ));
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
    let b = buffer(11, 8, 4);
    let c = buffer(12, 16, 8);
    let items = vec![
        item(vec![], a.clone()),
        item(vec![], b.clone()),
        item(vec![], c.clone()),
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
