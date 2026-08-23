use crate::{Graph, ScheduleBoundary, Shape, TensorData, schedule};

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
    let reduced = graph.sum(y, 0).unwrap();
    let item = &schedule(&graph, reduced).unwrap().items[0];
    assert!(matches!(
        item.boundary,
        Some(ScheduleBoundary::Unsupported(_))
    ));
}
