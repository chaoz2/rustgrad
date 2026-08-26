use crate::{Backend, CpuBackend, CpuSession, DType, Error, Graph, Scalar, TensorData};
use std::collections::HashMap;

fn execute(graph: &Graph, output: crate::NodeId, value: TensorData) -> crate::Result<TensorData> {
    CpuBackend.execute(graph, output, &HashMap::from([("x".into(), value)]))
}

#[test]
fn tensor_guard_preserves_valid_distribution_storage() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F32);
    let guard = graph.tensor_guard_distribution(input, -1).unwrap();
    let value = TensorData::from_scalars(
        [2, 2],
        DType::F32,
        [
            Scalar::F(0.0),
            Scalar::F(1.0),
            Scalar::F(2.0),
            Scalar::F(3.0),
        ],
    )
    .unwrap();
    assert_eq!(
        execute(&graph, guard, value.clone()).unwrap().storage(),
        value.storage()
    );
    assert!(
        graph
            .trace(guard)
            .unwrap()
            .to_string()
            .contains("tensor_guard")
    );
}

#[test]
fn tensor_guard_reports_first_invalid_lane_deterministically() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F32);
    let guard = graph.tensor_guard_distribution(input, 1).unwrap();
    let nan = TensorData::from_scalars(
        [2, 2],
        DType::F32,
        [
            Scalar::F(1.0),
            Scalar::F(f64::NAN),
            Scalar::F(-1.0),
            Scalar::F(2.0),
        ],
    )
    .unwrap();
    assert!(matches!(
        execute(&graph, guard, nan),
        Err(Error::TensorGuard {
            reason: "value is not finite",
            row: 0,
            index: 1
        })
    ));
    let zero = TensorData::from_scalars(
        [2, 2],
        DType::F32,
        [
            Scalar::F(0.0),
            Scalar::F(0.0),
            Scalar::F(1.0),
            Scalar::F(0.0),
        ],
    )
    .unwrap();
    assert!(matches!(
        execute(&graph, guard, zero),
        Err(Error::TensorGuard {
            reason: "row has nonpositive total",
            row: 0,
            index: 0
        })
    ));
}

#[test]
fn tensor_guard_preflight_is_atomic() {
    let mut graph = Graph::new();
    let integer = graph.input_dtype("x", [2], DType::I32);
    let before = graph.node_count();
    assert!(graph.tensor_guard_distribution(integer, 0).is_err());
    assert_eq!(graph.node_count(), before);
    let float = graph.input_dtype("f", [2], DType::F32);
    let before = graph.node_count();
    assert!(graph.tensor_guard_distribution(float, 2).is_err());
    assert_eq!(graph.node_count(), before);
}

#[test]
fn session_tensor_guard_enforces_ownership_and_preserves_trace() {
    let mut session = CpuSession::new();
    let value = session.tensor([2], [1.0, 2.0]).unwrap();
    let guard = session.tensor_guard_distribution(&value, -1).unwrap();
    assert_eq!(
        session.realize(&guard).unwrap().to_vec_f64(),
        vec![1.0, 2.0]
    );
    let before = session.graph().node_count();
    assert!(session.tensor_guard_distribution(&value, 2).is_err());
    assert_eq!(session.graph().node_count(), before);
    let mut other = CpuSession::new();
    assert!(matches!(
        other.tensor_guard_distribution(&value, 0),
        Err(Error::SessionHandleMismatch { .. })
    ));
}

#[test]
fn session_implicit_random_uses_graph_stream_and_preflights_dtype() {
    let mut session = CpuSession::new();
    let first = session.rand_implicit([2], DType::F32).unwrap();
    let second = session.rand_implicit([2], DType::F32).unwrap();
    assert_ne!(
        session.realize(&first).unwrap().storage(),
        session.realize(&second).unwrap().storage()
    );
    let before = session.graph().node_count();
    assert!(session.rand_implicit([2], DType::I32).is_err());
    assert_eq!(session.graph().node_count(), before);
}

#[test]
fn pending_random_reservation_is_stale_retryable_and_exactly_once() {
    let mut session = CpuSession::new();
    let weights = session.tensor([2], [1.0, 1.0]).unwrap();
    let guard = session.tensor_guard_distribution(&weights, 0).unwrap();
    let mut pending = session
        .pending_uniform_after_guard(&guard, [2], DType::F32)
        .unwrap();
    let before = session.graph().node_count();
    let _ordinary = session.rand_implicit([2], DType::F32).unwrap();
    assert!(
        session
            .commit_pending_uniform(&guard, &mut pending)
            .is_err()
    );
    assert_eq!(session.graph().node_count(), before + 1);

    let mut retry = session
        .pending_uniform_after_guard(&guard, [2], DType::F32)
        .unwrap();
    let random = session.commit_pending_uniform(&guard, &mut retry).unwrap();
    assert_eq!(random.dtype(), DType::F32);
    assert!(session.commit_pending_uniform(&guard, &mut retry).is_err());
}

#[test]
fn pending_random_guard_failure_rolls_back_and_a_new_guard_retries() {
    Graph::manual_seed(41);
    let mut session = CpuSession::new();
    let invalid = session.tensor([2], [f32::NAN, 1.0]).unwrap();
    let invalid_guard = session.tensor_guard_distribution(&invalid, 0).unwrap();
    let mut pending = session
        .pending_uniform_after_guard(&invalid_guard, [2], DType::F32)
        .unwrap();
    let before = session.graph().node_count();
    assert!(matches!(
        session.commit_pending_uniform(&invalid_guard, &mut pending),
        Err(Error::TensorGuard { .. })
    ));
    assert_eq!(session.graph().node_count(), before);

    let valid = session.tensor([2], [1.0, 1.0]).unwrap();
    let valid_guard = session.tensor_guard_distribution(&valid, 0).unwrap();
    let mut retry = session
        .pending_uniform_after_guard(&valid_guard, [2], DType::F32)
        .unwrap();
    let random = session
        .commit_pending_uniform(&valid_guard, &mut retry)
        .unwrap();
    assert!(
        session
            .trace(&random)
            .unwrap()
            .to_string()
            .contains("random_Uniform")
    );
}

#[test]
fn pending_random_reservation_rejects_wrong_nodes_and_graphs_without_mutation() {
    let mut session = CpuSession::new();
    let weights = session.tensor([2], [1.0, 1.0]).unwrap();
    let guard = session.tensor_guard_distribution(&weights, 0).unwrap();
    let before = session.graph().node_count();
    assert!(
        session
            .pending_uniform_after_guard(&weights, [2], DType::F32)
            .is_err()
    );
    assert_eq!(session.graph().node_count(), before);

    let mut pending = session
        .pending_uniform_after_guard(&guard, [2], DType::F32)
        .unwrap();
    assert!(
        session
            .commit_pending_uniform(&weights, &mut pending)
            .is_err()
    );
    assert_eq!(session.graph().node_count(), before);

    let mut other = CpuSession::new();
    let other_weights = other.tensor([2], [1.0, 1.0]).unwrap();
    let other_guard = other.tensor_guard_distribution(&other_weights, 0).unwrap();
    let other_before = other.graph().node_count();
    assert!(
        other
            .commit_pending_uniform(&other_guard, &mut pending)
            .is_err()
    );
    assert_eq!(other.graph().node_count(), other_before);
    assert!(matches!(
        other.pending_uniform_after_guard(&guard, [2], DType::F32),
        Err(Error::SessionHandleMismatch { .. })
    ));
}

#[test]
fn pending_random_zero_words_preserves_the_next_implicit_stream_draw() {
    Graph::manual_seed(73);
    let mut guarded = CpuSession::new();
    let weights = guarded.tensor([2], [1.0, 1.0]).unwrap();
    let guard = guarded.tensor_guard_distribution(&weights, 0).unwrap();
    let mut pending = guarded
        .pending_uniform_after_guard(&guard, [0], DType::F32)
        .unwrap();
    let empty = guarded
        .commit_pending_uniform(&guard, &mut pending)
        .unwrap();
    assert_eq!(guarded.realize(&empty).unwrap().shape().dims(), &[0]);
    let guarded_next = guarded.rand_implicit([2], DType::F32).unwrap();
    let guarded_values = guarded.realize(&guarded_next).unwrap();

    Graph::manual_seed(73);
    let mut ordinary = CpuSession::new();
    let ordinary_next = ordinary.rand_implicit([2], DType::F32).unwrap();
    assert_eq!(guarded_values, ordinary.realize(&ordinary_next).unwrap());
}

#[test]
fn pending_random_shape_overflow_rejects_before_graph_mutation() {
    let mut session = CpuSession::new();
    let weights = session.tensor([2], [1.0, 1.0]).unwrap();
    let guard = session.tensor_guard_distribution(&weights, 0).unwrap();
    let before = session.graph().node_count();
    assert!(
        session
            .pending_uniform_after_guard(&guard, [usize::MAX], DType::F32)
            .is_err()
    );
    assert_eq!(session.graph().node_count(), before);
    // Counter-overflow itself is intentionally unconstructible through the
    // public API: its counter snapshot is private and never mutable by users.
    // The checked arithmetic is covered at its defining creation seam.
}
