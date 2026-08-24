use super::{
    LlamaNativeCache, LlamaNativeExecutor, LlamaNativeStageKind,
    model_tests::{VOCAB, assert_close, make_model, reference_logits},
};
use crate::{ItemBackend, TensorData};

#[test]
fn strict_native_artifacts_match_graph_and_independent_dense_reference() {
    let (model, _, state) = make_model(8);
    let tokens = [3, 4, 5];
    let oracle = model.forward(&tokens).unwrap();
    let independent = reference_logits(&tokens, &state);
    let plan = model.plan_native(&tokens).unwrap();
    assert!(plan.artifacts().count() > 10);
    for bytes in plan.artifacts() {
        let decoded = crate::CapturedSchedule::from_bytes(bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
    }
    assert!(plan.artifacts().any(|bytes| {
        crate::CapturedSchedule::from_bytes(bytes)
            .unwrap()
            .items
            .iter()
            .any(|item| matches!(item.kernel.kind(), crate::UOpKind::Movement))
    }));
    let executor = LlamaNativeExecutor::new();
    let actual = plan.execute(&executor).unwrap();
    assert_close(actual.logits(), &oracle, 4e-5);
    assert_close(actual.logits(), &independent, 4e-5);
    assert!(
        actual
            .trace()
            .iter()
            .any(|stage| matches!(stage.kind, LlamaNativeStageKind::Movement(_)))
    );
    assert!(actual.trace().iter().all(|stage| match stage.kind {
        LlamaNativeStageKind::NativeSchedule => true,
        LlamaNativeStageKind::Movement(kind) => {
            matches!(kind, "reshape" | "permute" | "expand")
        }
    }));
    assert!(
        actual
            .trace()
            .iter()
            .filter(|stage| stage.kind == LlamaNativeStageKind::NativeSchedule)
            .flat_map(|stage| &stage.items)
            .all(|item| item.backend == ItemBackend::NativeJit)
    );
}

#[test]
fn strict_native_cache_matches_full_token_and_chunk_execution() {
    let (model, _, _) = make_model(8);
    let tokens = [3, 4, 5];
    let full = model.forward_native(&tokens).unwrap().logits().clone();
    let mut cache = LlamaNativeCache::new(model.config().clone());
    let mut values = Vec::new();
    for token in tokens {
        values.extend_from_slice(cache.forward(&model, &[token]).unwrap().logits().values());
    }
    assert_eq!(cache.len(), 3);
    assert_close(&TensorData::new([3, VOCAB], values).unwrap(), &full, 4e-5);
    assert!(cache.compile_cache_len() > 0);

    cache.clear();
    cache.forward(&model, &[3, 4]).unwrap();
    let suffix = cache.forward(&model, &[5]).unwrap();
    assert_close(
        suffix.logits(),
        &TensorData::new([1, VOCAB], full.values()[2 * VOCAB..].to_vec()).unwrap(),
        4e-5,
    );
    assert!(
        suffix
            .trace()
            .iter()
            .flat_map(|stage| &stage.items)
            .any(|item| item.cache_hit)
    );
}

#[test]
fn strict_native_fixed_batch_preserves_row_isolation() {
    let (model, _, _) = make_model(8);
    let rows = vec![vec![3, 4, 5], vec![6, 3]];
    let expected = model.forward_batch(&rows).unwrap();
    let plan = model.plan_batch_native(&rows).unwrap();
    assert!(plan.artifacts().count() > 10);
    let actual = plan.execute(&LlamaNativeExecutor::new()).unwrap();
    for (actual, expected) in actual.rows().iter().zip(&expected) {
        assert_close(actual, expected, 5e-5);
    }
    assert!(
        actual
            .trace()
            .iter()
            .flat_map(|stage| &stage.items)
            .all(|item| item.backend == ItemBackend::NativeJit)
    );
}

#[test]
fn native_artifact_bindings_and_cache_failures_are_typed_and_transactional() {
    let (model, _, _) = make_model(4);
    let plan = model.plan_native(&[3]).unwrap();
    let artifact = plan
        .artifacts()
        .find_map(|bytes| {
            let artifact = crate::CapturedSchedule::from_bytes(bytes).unwrap();
            (!artifact.inputs.is_empty()).then_some(artifact)
        })
        .unwrap();
    assert!(matches!(
        artifact.replay_with_options(
            &std::collections::BTreeMap::new(),
            &crate::CapturedReplayExecutor::default(),
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false }
            },
        ),
        Err(crate::ReplayError::Missing(_))
    ));

    let mut cache = LlamaNativeCache::new(model.config().clone());
    let mut control = LlamaNativeCache::new(model.config().clone());
    cache.forward(&model, &[3, 4]).unwrap();
    control.forward(&model, &[3, 4]).unwrap();
    let before = cache.len();
    assert!(matches!(
        cache.forward(&model, &[5, 6, 3]),
        Err(super::LlamaNativeError::Model(
            super::LlamaModelError::ContextLength {
                requested: 5,
                maximum: 4,
            }
        ))
    ));
    assert_eq!(cache.len(), before);
    let actual = cache.forward(&model, &[5]).unwrap();
    let expected = control.forward(&model, &[5]).unwrap();
    assert_close(actual.logits(), expected.logits(), 0.0);
}
