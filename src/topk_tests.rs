use crate::{Backend, CpuBackend, DType, Error, Graph, Scalar, Shape, TensorData};
use std::collections::HashMap;

fn cpu(graph: &Graph, output: crate::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("x".into(), input)]))
        .unwrap()
}

#[test]
fn topk_is_a_stable_sort_pair_followed_by_checked_slices() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 4], DType::I32);
    let (values, indices) = graph.topk(x, 3, -1, true, true).unwrap();
    let input =
        TensorData::from_scalars([2, 4], DType::I32, [1, 1, 0, 1, 1, 3, 3, 2].map(Scalar::I))
            .unwrap();
    assert_eq!(
        cpu(&graph, values, input.clone()).to_vec_f64(),
        vec![1., 1., 1., 3., 3., 2.]
    );
    assert_eq!(
        cpu(&graph, indices, input).to_vec_f64(),
        vec![0., 1., 3., 1., 2., 3.]
    );
    assert_eq!(graph.dtype(values).unwrap(), DType::I32);
    assert_eq!(graph.dtype(indices).unwrap(), DType::I32);
    let trace = graph.trace(indices).unwrap().to_string();
    assert!(trace.contains("sort(%"));
    assert!(trace.contains("argsort(%"));
    assert!(trace.contains("shrink(%"));

    let schedule = crate::schedule_many(&graph, &[values, indices]).unwrap();
    assert_eq!(
        schedule
            .items
            .iter()
            .filter(|item| matches!(item.kernel.kind(), crate::UOpKind::Sort))
            .count(),
        1
    );
    assert_eq!(
        schedule
            .items
            .iter()
            .find(|item| matches!(item.kernel.kind(), crate::UOpKind::Sort))
            .unwrap()
            .outputs
            .len(),
        2
    );
}

#[test]
fn topk_smallest_signed_axis_zero_and_zero_extent_preserve_static_contracts() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::I16);
    let (values, indices) = graph.topk(x, 2, -2, false, true).unwrap();
    let input =
        TensorData::from_scalars([2, 3], DType::I16, [3, 1, 2, 0, 4, -1].map(Scalar::I)).unwrap();
    assert_eq!(
        cpu(&graph, values, input.clone()).to_vec_f64(),
        vec![0., 1., -1., 3., 4., 2.]
    );
    assert_eq!(
        cpu(&graph, indices, input).to_vec_f64(),
        vec![1., 0., 1., 0., 1., 0.]
    );

    let mut zero = Graph::new();
    let x = zero.input_dtype("x", [2, 3], DType::Bool);
    let (values, indices) = zero.topk(x, 0, 1, true, true).unwrap();
    let input = TensorData::from_scalars(
        [2, 3],
        DType::Bool,
        [true, false, true, false, true, false].map(Scalar::Bool),
    )
    .unwrap();
    assert_eq!(
        cpu(&zero, values, input.clone()).shape(),
        &Shape::new([2, 0])
    );
    assert_eq!(cpu(&zero, indices, input).shape(), &Shape::new([2, 0]));

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [2, 0, 3], DType::F32);
    let (values, indices) = empty.topk(x, 0, -2, true, true).unwrap();
    let input = TensorData::from_scalars([2, 0, 3], DType::F32, []).unwrap();
    assert_eq!(
        cpu(&empty, values, input.clone()).shape(),
        &Shape::new([2, 0, 3])
    );
    assert_eq!(cpu(&empty, indices, input).dtype(), DType::I32);
}

#[test]
fn topk_preflights_axis_rank_and_k_without_graph_mutation() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::F32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.topk(x, 4, 1, true, true),
        Err(Error::InvalidBounds {
            axis: 1,
            end: 4,
            dim: 3,
            ..
        })
    ));
    assert_eq!(graph.trace(x).unwrap(), before);
    assert!(matches!(
        graph.topk(x, 1, 2, true, true),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);

    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let scalar_before = graph.trace(scalar).unwrap();
    assert!(matches!(
        graph.topk(scalar, 1, -1, true, true),
        Err(Error::InvalidAxis {
            node,
            axis: 0,
            rank: 0,
        }) if node == scalar
    ));
    assert_eq!(graph.trace(scalar).unwrap(), scalar_before);
}
