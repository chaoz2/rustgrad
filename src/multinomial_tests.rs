use crate::{
    Backend, CpuBackend, CpuSession, DType, Error, Graph, RandomStream, Scalar, TensorData,
};
use std::collections::HashMap;

fn explicit_stream() -> RandomStream {
    RandomStream {
        device: 0,
        key: [0x14B8_1119, 0],
        counter: [0, 0],
    }
}

#[test]
fn multinomial_replacement_is_explicit_stream_replayable_and_typed() {
    let mut graph = Graph::new();
    let weights = graph.input_dtype("weights", [2, 3], DType::F32);
    let sampled = graph
        .multinomial_with_stream(weights, 4, -1, true, explicit_stream())
        .unwrap();
    let values = TensorData::from_scalars(
        [2, 3],
        DType::F32,
        [
            Scalar::F(1.0),
            Scalar::F(0.0),
            Scalar::F(2.0),
            Scalar::F(0.0),
            Scalar::F(3.0),
            Scalar::F(1.0),
        ],
    )
    .unwrap();
    let bindings = HashMap::from([("weights".into(), values)]);
    let first = CpuBackend.execute(&graph, sampled, &bindings).unwrap();
    let second = CpuBackend.execute(&graph, sampled, &bindings).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.dtype(), DType::I32);
    assert_eq!(first.shape().dims(), &[2, 4]);
    assert!(
        first
            .to_vec_f64()
            .iter()
            .all(|index| (0.0..3.0).contains(index))
    );
    assert!(
        graph
            .trace(sampled)
            .unwrap()
            .to_string()
            .contains("tensor_guard")
    );
}

#[test]
fn multinomial_without_replacement_uses_checked_stable_topk_indices() {
    let mut graph = Graph::new();
    let weights = graph.input_dtype("weights", [3], DType::F32);
    let sampled = graph
        .multinomial_with_stream(weights, 2, 0, false, explicit_stream())
        .unwrap();
    let values = TensorData::from_scalars(
        [3],
        DType::F32,
        [Scalar::F(1.0), Scalar::F(2.0), Scalar::F(3.0)],
    )
    .unwrap();
    let output = CpuBackend
        .execute(
            &graph,
            sampled,
            &HashMap::from([("weights".into(), values)]),
        )
        .unwrap();
    assert_eq!(output.dtype(), DType::I32);
    assert_eq!(output.shape().dims(), &[2]);
    assert_ne!(output.to_vec_f64()[0], output.to_vec_f64()[1]);
    let trace = graph.trace(sampled).unwrap().to_string();
    assert!(trace.contains("sort("));
    assert!(trace.contains("argsort("));
}

#[test]
fn multinomial_preflights_shape_count_and_dtype_without_graph_mutation() {
    let mut graph = Graph::new();
    let integer = graph.input_dtype("integer", [3], DType::I32);
    let before = graph.node_count();
    assert!(
        graph
            .multinomial_with_stream(integer, 1, 0, true, explicit_stream())
            .is_err()
    );
    assert_eq!(graph.node_count(), before);
    let weights = graph.input_dtype("weights", [2, 3], DType::F32);
    let before = graph.node_count();
    assert!(matches!(
        graph.multinomial_with_stream(weights, 4, 1, false, explicit_stream()),
        Err(Error::InvalidBounds { .. })
    ));
    assert_eq!(graph.node_count(), before);
    assert!(
        graph
            .multinomial_with_stream(weights, 1, 2, true, explicit_stream())
            .is_err()
    );
    assert_eq!(graph.node_count(), before);
}

#[test]
fn session_multinomial_implicit_rolls_back_invalid_weights_before_stream_reservation() {
    let _stream_guard = Graph::lock_implicit_random_tests();
    Graph::manual_seed(91);
    let mut session = CpuSession::new();
    let invalid = session.tensor([2], [f32::NAN, 1.0]).unwrap();
    let before = session.graph().node_count();
    assert!(matches!(
        session.multinomial_implicit(&invalid, 2, 0, true),
        Err(Error::TensorGuard { .. })
    ));
    assert_eq!(session.graph().node_count(), before);
    let after_failure = session.rand_implicit([2], DType::F32).unwrap();
    let observed = session.realize(&after_failure).unwrap();

    Graph::manual_seed(91);
    let mut baseline = CpuSession::new();
    let first = baseline.rand_implicit([2], DType::F32).unwrap();
    assert_eq!(observed, baseline.realize(&first).unwrap());

    let valid = session.tensor([2], [1.0, 3.0]).unwrap();
    let sampled = session.multinomial_implicit(&valid, 3, -1, true).unwrap();
    assert_eq!(sampled.dtype(), DType::I32);
    assert_eq!(sampled.shape().dims(), &[3]);
    assert!(
        session
            .trace(&sampled)
            .unwrap()
            .to_string()
            .contains("random_Uniform")
    );
}
