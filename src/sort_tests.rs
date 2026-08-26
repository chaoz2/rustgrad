use crate::{
    Backend, CpuBackend, DType, Error, Graph, RealizationPolicy, Scalar, Shape, TensorData,
};
use std::collections::HashMap;

fn cpu(graph: &Graph, output: crate::NodeId, input: TensorData) -> TensorData {
    CpuBackend
        .execute(graph, output, &HashMap::from([("x".into(), input)]))
        .unwrap()
}

#[test]
fn sort_is_stable_and_pairs_values_with_i32_indices_across_axes() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 4], DType::I32);
    let (values, indices) = graph.sort(x, -1, false).unwrap();
    let input =
        TensorData::from_scalars([2, 4], DType::I32, [2, 1, 1, 3, 4, 2, 2, 1].map(Scalar::I))
            .unwrap();
    assert_eq!(
        cpu(&graph, values, input.clone()).to_vec_f64(),
        vec![1., 1., 2., 3., 1., 2., 2., 4.]
    );
    assert_eq!(
        cpu(&graph, indices, input).to_vec_f64(),
        vec![1., 2., 0., 3., 3., 1., 2., 0.]
    );
    assert_eq!(graph.dtype(values).unwrap(), DType::I32);
    assert_eq!(graph.dtype(indices).unwrap(), DType::I32);

    let mut descending = Graph::new();
    let x = descending.input_dtype("x", [2, 2], DType::I16);
    let (values, indices) = descending.sort(x, 0, true).unwrap();
    let input = TensorData::from_scalars([2, 2], DType::I16, [1, 4, 3, 2].map(Scalar::I)).unwrap();
    assert_eq!(
        cpu(&descending, values, input.clone()).to_vec_f64(),
        vec![3., 4., 1., 2.]
    );
    assert_eq!(
        cpu(&descending, indices, input).to_vec_f64(),
        vec![1., 0., 0., 1.]
    );
}

#[test]
fn sort_preserves_bool_float_special_values_and_static_edges() {
    let mut bool_graph = Graph::new();
    let x = bool_graph.input_dtype("x", [3], DType::Bool);
    let (values, indices) = bool_graph.sort(x, 0, false).unwrap();
    let input =
        TensorData::from_scalars([3], DType::Bool, [true, false, true].map(Scalar::Bool)).unwrap();
    assert_eq!(
        cpu(&bool_graph, values, input.clone()).to_vec_f64(),
        vec![0., 1., 1.]
    );
    assert_eq!(
        cpu(&bool_graph, indices, input).to_vec_f64(),
        vec![1., 0., 2.]
    );

    // tinygrad's left-biased min/max network treats equal signed zero and
    // unordered NaN comparisons as stable ties, retaining their raw lanes.
    let mut float_graph = Graph::new();
    let x = float_graph.input_dtype("x", [3], DType::F32);
    let (values, indices) = float_graph.sort(x, 0, false).unwrap();
    let input = TensorData::from_scalars(
        [3],
        DType::F32,
        [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(f64::NAN)],
    )
    .unwrap();
    let actual = cpu(&float_graph, values, input.clone());
    let crate::Storage::F32(raw) = actual.storage() else {
        panic!("expected F32 sort output")
    };
    assert_eq!(raw[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(raw[1].to_bits(), 0.0f32.to_bits());
    assert!(raw[2].is_nan());
    assert_eq!(
        cpu(&float_graph, indices, input).to_vec_f64(),
        vec![0., 1., 2.]
    );

    let mut empty = Graph::new();
    let x = empty.input_dtype("x", [2, 0, 3], DType::U8);
    let (values, indices) = empty.sort(x, -2, false).unwrap();
    let input = TensorData::from_scalars([2, 0, 3], DType::U8, []).unwrap();
    assert_eq!(
        cpu(&empty, values, input.clone()).shape(),
        &Shape::new([2, 0, 3])
    );
    assert_eq!(cpu(&empty, indices, input).dtype(), DType::I32);

    let mut scalar = Graph::new();
    let x = scalar.input_dtype("x", [], DType::I8);
    let (values, indices) = scalar.sort(x, -1, false).unwrap();
    let input = TensorData::from_scalars([], DType::I8, [Scalar::I(-7)]).unwrap();
    assert_eq!(cpu(&scalar, values, input.clone()).to_vec_f64(), vec![-7.]);
    assert_eq!(cpu(&scalar, indices, input).to_vec_f64(), vec![0.]);
}

#[test]
fn sort_schedule_coalesces_one_typed_producer_and_cpu_materializes_both() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [3], DType::F32);
    let (values, indices) = graph.sort(x, 0, false).unwrap();
    let schedule = crate::schedule_many(&graph, &[values, indices]).unwrap();
    assert_eq!(schedule.items.len(), 1);
    assert_eq!(schedule.items[0].outputs.len(), 2);
    assert_eq!(
        schedule.items[0].outputs.primary().id,
        values.index() as u64
    );
    assert_eq!(
        schedule.items[0].outputs.iter().nth(1).unwrap().id,
        indices.index() as u64
    );
    assert!(matches!(
        schedule.items[0].kernel.kind(),
        crate::UOpKind::Sort
    ));

    let realized = crate::realize_graph(
        &graph,
        &[values, indices],
        &HashMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [Scalar::F(2.), Scalar::F(1.), Scalar::F(1.)],
            )
            .unwrap(),
        )]),
        RealizationPolicy::Interpreter,
    )
    .unwrap();
    assert_eq!(realized.outputs[0].to_vec_f64(), vec![1., 1., 2.]);
    assert_eq!(realized.outputs[1].to_vec_f64(), vec![1., 2., 0.]);
    assert_eq!(realized.trace.items.len(), 1);
    assert!(matches!(
        crate::realize_graph(
            &graph,
            &[values, indices],
            &HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [3],
                    DType::F32,
                    [Scalar::F(2.), Scalar::F(1.), Scalar::F(1.)],
                )
                .unwrap(),
            )]),
            RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
        ),
        Err(crate::RealizationError::Unsupported(_))
    ));
}

#[test]
fn sort_artifact_trace_rejection_and_invalid_axis_are_explicit() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", [2, 2], DType::F32);
    let before = graph.trace(x).unwrap();
    assert!(matches!(
        graph.sort(x, 2, false),
        Err(Error::InvalidReductionAxes { node, rank: 2, .. }) if node == x
    ));
    assert_eq!(graph.trace(x).unwrap(), before);
    let (values, indices) = graph.sort(x, -1, false).unwrap();
    let trace = graph.trace(indices).unwrap().to_string();
    assert!(trace.contains("argsort(%"));
    assert!(trace.contains("axis=1"));
    let uop = crate::kernel::lower_graph_sort_pair(&graph, values, indices).unwrap();
    let bytes = crate::uop::artifact::encode(&uop).unwrap();
    assert_eq!(crate::uop::artifact::decode(&bytes).unwrap(), uop);
    assert!(
        crate::uop::artifact::encode(&crate::UOp::new(
            crate::UOpKind::Sort,
            Some(crate::UType::scalar(DType::F32)),
            vec![],
            crate::UArg::Sort {
                input: x,
                input_shape: Shape::new([2, 2]),
                axis: 2,
                descending: false,
                values,
                indices,
                dtype: DType::F32,
            },
        ))
        .is_err()
    );
    let loss = graph.sum_all(values).unwrap();
    assert!(matches!(
        graph.grad(loss, x),
        Err(Error::NonDifferentiableIndexing(_))
    ));
    let capture = crate::CapturedSchedule::capture(
        &graph,
        &crate::schedule_many(&graph, &[values, indices]).unwrap(),
        &[values, indices],
    )
    .unwrap();
    assert!(capture.to_bytes().is_err());
}
