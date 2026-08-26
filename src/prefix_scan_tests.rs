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
        assert_eq!(
            actual.to_vec_f64(),
            expected.into_iter().map(f64::from).collect::<Vec<_>>()
        );
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
            kind: crate::PrefixScanKind::Sum,
            output: crate::PrefixScanOutput::Values,
            dtype: DType::I32,
        },
    );
    assert!(crate::uop::artifact::encode(&malformed).is_err());
}

#[test]
fn cumprod_matches_tinygrad_values_dtypes_and_empty_scalar_contracts() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::I16);
    let output = graph.cumprod(x, -1).unwrap();
    let actual = execute(
        &graph,
        output,
        TensorData::from_scalars(
            [2, 3],
            DType::I16,
            [2, 3, 4, -1, 2, 3].into_iter().map(crate::Scalar::I),
        )
        .unwrap(),
    );
    assert_eq!(actual.dtype(), DType::I16);
    assert_eq!(actual.to_vec_f64(), vec![2., 6., 24., -1., -2., -6.]);

    let mut boolean_graph = Graph::new();
    let boolean = boolean_graph.input_dtype("x", [3], DType::Bool);
    let boolean_output = boolean_graph.cumprod(boolean, 0).unwrap();
    assert_eq!(boolean_graph.dtype(boolean_output).unwrap(), DType::Bool);
    assert_eq!(
        execute(
            &boolean_graph,
            boolean_output,
            TensorData::from_scalars(
                [3],
                DType::Bool,
                [true, false, true].into_iter().map(crate::Scalar::Bool),
            )
            .unwrap(),
        )
        .to_vec_f64(),
        vec![1., 0., 0.]
    );

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input_dtype("x", [2, 0], DType::U8);
    let empty_output = empty_graph.cumprod(empty, 1).unwrap();
    let empty_value = execute(
        &empty_graph,
        empty_output,
        TensorData::from_scalars([2, 0], DType::U8, []).unwrap(),
    );
    assert_eq!(empty_value.shape(), &Shape::new([2, 0]));
    assert_eq!(empty_value.dtype(), DType::U8);

    let mut scalar_graph = Graph::new();
    let scalar = scalar_graph.input_dtype("x", [], DType::I8);
    let scalar_output = scalar_graph.cumprod(scalar, -1).unwrap();
    assert_eq!(scalar_graph.shape(scalar_output).unwrap(), &Shape::new([]));
    assert_eq!(scalar_graph.dtype(scalar_output).unwrap(), DType::I8);
    assert_eq!(
        execute(
            &scalar_graph,
            scalar_output,
            TensorData::from_scalars([], DType::I8, [crate::Scalar::I(-3)]).unwrap(),
        )
        .to_vec_f64(),
        vec![-3.]
    );
    let trace = scalar_graph.trace(scalar_output).unwrap().to_string();
    assert!(trace.contains("cumprod(%"));
}

#[test]
fn cumprod_artifact_round_trip_and_invalid_axis_leave_graph_unchanged() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 3], DType::I32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cumprod(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);

    let output = graph.cumprod(x, 0).unwrap();
    let lowered = crate::lower_graph_prefix_scan(&graph, output).unwrap();
    lowered.validate().unwrap();
    let bytes = crate::uop::artifact::encode(&lowered).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), lowered);
}

#[test]
fn cumulative_extrema_match_tinygrad_last_tie_indices_and_static_edges() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 4], DType::I32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.cummax(x, 2),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);
    let (maximum, max_indices) = graph.cummax(x, -1).unwrap();
    let (minimum, min_indices) = graph.cummin(x, 1).unwrap();
    let input = TensorData::from_scalars(
        [2, 4], DType::I32,
        [1, 3, 3, 2, 4, 2, 2, 5].into_iter().map(crate::Scalar::I),
    ).unwrap();
    assert_eq!(execute(&graph, maximum, input.clone()).to_vec_f64(), vec![1., 3., 3., 3., 4., 4., 4., 5.]);
    assert_eq!(execute(&graph, max_indices, input.clone()).to_vec_f64(), vec![0., 1., 2., 2., 0., 0., 0., 3.]);
    assert_eq!(execute(&graph, minimum, input.clone()).to_vec_f64(), vec![1., 1., 1., 1., 4., 2., 2., 2.]);
    assert_eq!(execute(&graph, min_indices, input).to_vec_f64(), vec![0., 0., 0., 0., 0., 1., 2., 2.]);
    assert_eq!(graph.dtype(max_indices).unwrap(), DType::I32);
    let trace = graph.trace(max_indices).unwrap().to_string();
    assert!(trace.contains("cummax_indices(%"));
    assert!(trace.contains("axis=1"));
    let kernel = crate::lower_graph_prefix_scan(&graph, max_indices).unwrap();
    let bytes = crate::uop::artifact::encode(&kernel).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), kernel);

    let mut empty_graph = Graph::new();
    let empty = empty_graph.input_dtype("x", [0], DType::F32);
    let (values, indices) = empty_graph.cummax(empty, 0).unwrap();
    assert_eq!(empty_graph.shape(values).unwrap(), &Shape::new([0]));
    assert_eq!(empty_graph.dtype(indices).unwrap(), DType::I32);

    let mut scalar_graph = Graph::new();
    let scalar = scalar_graph.input_dtype("x", [], DType::I16);
    let (value, index) = scalar_graph.cummin(scalar, -1).unwrap();
    let input = TensorData::from_scalars([], DType::I16, [crate::Scalar::I(-7)]).unwrap();
    assert_eq!(execute(&scalar_graph, value, input.clone()).to_vec_f64(), vec![-7.]);
    assert_eq!(execute(&scalar_graph, index, input).to_vec_f64(), vec![0.]);

    // tinygrad's Ops.MAX uses left-biased `max`: NaNs do not replace a finite
    // prefix and an equal positive zero does not replace an earlier negative zero.
    let mut float_graph = Graph::new();
    let float = float_graph.input_dtype("x", [4], DType::F32);
    let (values, indices) = float_graph.cummax(float, 0).unwrap();
    let input = TensorData::from_scalars(
        [4],
        DType::F32,
        [
            crate::Scalar::F(-0.0),
            crate::Scalar::F(0.0),
            crate::Scalar::F(f64::NAN),
            crate::Scalar::F(-1.0),
        ],
    )
    .unwrap();
    let actual = execute(&float_graph, values, input.clone());
    let crate::Storage::F32(actual) = actual.storage() else {
        panic!("expected F32 cumulative maximum")
    };
    assert_eq!(
        actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        vec![(-0.0f32).to_bits(); 4]
    );
    assert_eq!(
        execute(&float_graph, indices, input).to_vec_f64(),
        vec![0., 1., 1., 1.]
    );
}
