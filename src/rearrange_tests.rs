use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape, TensorData};
use std::collections::{BTreeMap, HashMap};

fn data(shape: impl Into<Shape>, values: &[i64]) -> TensorData {
    TensorData::from_scalars(shape, DType::I32, values.iter().copied().map(Scalar::I)).unwrap()
}

fn run_rearrange(
    shape: impl Into<Shape>,
    values: &[i64],
    pattern: &str,
    sizes: &[(&str, usize)],
) -> TensorData {
    let mut graph = Graph::new();
    let input = graph.constant(data(shape, values));
    let sizes = sizes
        .iter()
        .map(|(name, size)| ((*name).into(), *size))
        .collect::<BTreeMap<_, _>>();
    let output = graph.rearrange(input, pattern, &sizes).unwrap();
    CpuBackend.execute(&graph, output, &HashMap::new()).unwrap()
}

#[test]
fn rearrange_parses_and_lowers_static_einops_patterns() {
    assert_eq!(
        run_rearrange(
            [2, 3, 4],
            &(0..24).collect::<Vec<_>>(),
            "a b c -> c a b",
            &[]
        )
        .shape(),
        &Shape::from([4, 2, 3])
    );
    assert_eq!(
        run_rearrange(
            [2, 6],
            &(0..12).collect::<Vec<_>>(),
            "a (b c) -> c a b",
            &[("c", 3)]
        )
        .shape(),
        &Shape::from([3, 2, 2])
    );
    assert_eq!(
        run_rearrange(
            [2, 3, 4],
            &(0..24).collect::<Vec<_>>(),
            "a b c -> a (b c)",
            &[]
        )
        .to_vec_f64(),
        (0..24).map(f64::from).collect::<Vec<_>>()
    );
    assert_eq!(
        run_rearrange([2, 3], &(0..6).collect::<Vec<_>>(), "a b -> b a ()", &[]).shape(),
        &Shape::from([3, 2, 1])
    );
    assert_eq!(
        run_rearrange(
            [2, 3, 4],
            &(0..24).collect::<Vec<_>>(),
            "a ... -> ... a",
            &[]
        )
        .shape(),
        &Shape::from([3, 4, 2])
    );
    assert_eq!(
        run_rearrange([2, 3], &(0..6).collect::<Vec<_>>(), "... -> (...)", &[]).shape(),
        &Shape::from([6])
    );
    assert_eq!(run_rearrange([], &[7], " -> ", &[]).to_vec_f64(), vec![7.]);
    assert_eq!(
        run_rearrange([0, 3], &[], "a b -> b a", &[]).shape(),
        &Shape::from([3, 0])
    );
}

#[test]
fn rearrange_rejects_invalid_static_equations() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 6]);
    let empty = BTreeMap::new();
    for pattern in [
        "a b -> a",
        "a (b c) -> c a b",
        "a ... ... -> a ...",
        "a (b ...)-> a b",
        "a b -> a b -> b a",
        "a b -> a a",
    ] {
        assert!(graph.rearrange(x, pattern, &empty).is_err(), "{pattern}");
    }
    let sizes = BTreeMap::from([("b".into(), 4)]);
    assert!(graph.rearrange(x, "a b -> a b", &sizes).is_err());
}

#[test]
fn repeat_tile_and_interleave_are_traceable_and_exact() {
    let mut graph = Graph::new();
    let x = graph.constant(data([2], &[1, 2]));
    let repeated = graph.repeat(x, &[3, 2]).unwrap();
    let tiled = graph.tile(x, &[2]).unwrap();
    let flat = graph.repeat_interleave(x, 3, None).unwrap();
    let axis = graph.repeat_interleave(x, 0, Some(0)).unwrap();
    let cpu = CpuBackend;
    assert_eq!(
        cpu.execute(&graph, repeated, &HashMap::new())
            .unwrap()
            .shape(),
        &Shape::from([3, 4])
    );
    assert_eq!(
        cpu.execute(&graph, repeated, &HashMap::new())
            .unwrap()
            .to_vec_f64(),
        vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.]
    );
    assert_eq!(
        cpu.execute(&graph, tiled, &HashMap::new())
            .unwrap()
            .to_vec_f64(),
        vec![1., 2., 1., 2.]
    );
    assert_eq!(
        cpu.execute(&graph, flat, &HashMap::new())
            .unwrap()
            .to_vec_f64(),
        vec![1., 1., 1., 2., 2., 2.]
    );
    assert_eq!(
        cpu.execute(&graph, axis, &HashMap::new()).unwrap().shape(),
        &Shape::from([0])
    );
    assert!(graph.trace(flat).unwrap().to_string().contains("expand"));
    assert!(graph.repeat(x, &[-1]).is_err());
    assert!(graph.repeat_interleave(x, -1, Some(0)).is_err());
}

#[test]
fn movement_lowering_backpropagates_repeat_and_rearrange() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 2]);
    let repeated = graph.repeat_interleave(x, 2, Some(1)).unwrap();
    let sizes = BTreeMap::new();
    let reordered = graph.rearrange(repeated, "a b -> b a", &sizes).unwrap();
    let loss = graph
        .reduce(reordered, crate::ReduceKind::Sum, None, false)
        .unwrap();
    let dx = graph.grad(loss, x).unwrap();
    let values = HashMap::from([(
        "x".into(),
        TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&graph, dx, &values)
            .unwrap()
            .to_vec_f64(),
        vec![2., 2., 2., 2.]
    );
}
