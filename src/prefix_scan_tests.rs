use crate::{Backend, CpuBackend, DType, Error, Graph, Shape, TensorData};
use std::collections::HashMap;

fn execute(graph: &Graph, output: crate::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("x".into(), input)]))
        .unwrap()
}

#[test]
fn cumsum_matches_tinygrad_values_for_signed_axes_and_empty_extents() {
    let cases = [
        (
            Shape::new([2, 3]),
            1,
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 3, 6, 4, 9, 15],
        ),
        (
            Shape::new([2, 3]),
            -2,
            vec![1, 2, 3, 4, 5, 6],
            vec![1, 2, 3, 5, 7, 9],
        ),
    ];
    for (shape, axis, input, expected) in cases {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", shape.clone(), DType::I16);
        let output = graph.cumsum(x, axis).unwrap();
        let actual = execute(
            &graph,
            output,
            TensorData::from_scalars(shape, DType::I16, input.into_iter().map(crate::Scalar::I))
                .unwrap(),
        );
        assert_eq!(actual.dtype(), DType::I32);
        assert_eq!(actual.to_vec_f64(), expected.into_iter().map(f64::from).collect::<Vec<_>>());
    }

    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 0, 3], DType::I8);
    let output = graph.cumsum(x, -2).unwrap();
    let actual = execute(
        &graph,
        output,
        TensorData::from_scalars([2, 0, 3], DType::I8, []).unwrap(),
    );
    assert_eq!(actual.shape(), &Shape::new([2, 0, 3]));
    assert_eq!(actual.dtype(), DType::I32);
}

#[test]
fn cumsum_dtype_scalar_trace_and_artifact_are_canonical() {
    let dtype_cases = [
        (DType::Bool, DType::I32),
        (DType::I8, DType::I32),
        (DType::U8, DType::U32),
        (DType::F16, DType::F16),
        (DType::F32, DType::F32),
    ];
    for (input_dtype, output_dtype) in dtype_cases {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", [2], input_dtype);
        let output = graph.cumsum(x, 0).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), output_dtype);
    }

    let mut graph = Graph::new();
    let scalar = graph.input_dtype("x", [], DType::I8);
    let output = graph.cumsum(scalar, -1).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    assert_eq!(
        execute(
            &graph,
            output,
            TensorData::from_scalars([], DType::I8, [crate::Scalar::I(7)]).unwrap(),
        )
        .to_vec_f64(),
        vec![7.0]
    );
    let trace = graph.trace(output).unwrap().to_string();
    assert!(trace.contains("cumsum(%"));
    assert!(trace.contains("axis=0"));
    assert!(trace.contains("[] I32"));
    let lowered = crate::lower_graph_prefix_scan(&graph, output).unwrap();
    lowered.validate().unwrap();
    let bytes = crate::uop::artifact::encode(&lowered).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), lowered);
    assert_eq!(crate::uop::artifact::encode(&lowered).unwrap(), bytes);
}

#[test]
fn cumsum_rejects_invalid_axes_without_graph_mutation() {
    let mut graph = Graph::new();
    let x = graph.input("x", [2, 3]);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cumsum(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);

    let scalar = graph.input("scalar", []);
    let before = graph.trace(scalar).unwrap();
    assert!(matches!(
        graph.cumsum(scalar, 1),
        Err(Error::InvalidReductionAxes { node, rank: 0, .. }) if node == scalar
    ));
    assert_eq!(graph.trace(scalar).unwrap(), before);
}

#[test]
fn prefix_scan_artifact_rejects_malformed_static_geometry() {
    let malformed = crate::UOp::new(
        crate::UOpKind::PrefixScan,
        Some(crate::UType::scalar(DType::I32)),
        vec![],
        crate::UArg::PrefixScan {
            input: crate::NodeId::from_index(0),
            input_shape: Shape::new([2]),
            output_shape: Shape::new([3]),
            axis: 0,
            dtype: DType::I32,
        },
    );
    assert!(crate::uop::artifact::encode(&malformed).is_err());
}
