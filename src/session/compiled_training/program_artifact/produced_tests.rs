//! Producer ownership is compared with independent public byte admission.

use super::*;
use std::sync::Arc;

fn assert_produced_matches_import<M: Module>(
    owner: &CompiledModuleAdamWPlan<M>,
) -> CompiledAdamWProgramArtifact {
    let before = program_artifact::portable_resume_decode_counts();
    let produced = owner.program_artifact().unwrap();
    let after = program_artifact::portable_resume_decode_counts();
    let metal = produced.info().format_version() == 2;
    let mut expected_counts = before;
    expected_counts.recurrent_execution_plans +=
        usize::from(metal) * (1 + usize::from(owner.plan.partial_flush.is_some()));
    expected_counts.evaluation_execution_plans +=
        usize::from(metal && owner.plan.evaluation.is_some());
    assert_eq!(
        after, expected_counts,
        "only Metal wrapper reconstruction may add work"
    );

    let expected = std::iter::once(owner.plan.inner.recurrent_frontier.clone())
        .chain(
            [
                owner
                    .plan
                    .inner
                    .accumulation
                    .as_ref()
                    .map(|phase| phase.phase().recurrent_frontier.clone()),
                owner
                    .plan
                    .partial_flush
                    .as_ref()
                    .map(|phase| phase.phase().recurrent_frontier.clone()),
                owner
                    .plan
                    .zero_grad
                    .as_ref()
                    .map(|phase| phase.phase().recurrent_frontier.clone()),
            ]
            .into_iter()
            .flatten(),
        )
        .collect::<Vec<_>>();
    let actual = produced.recurrent_capture_owners();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(Arc::ptr_eq(actual, expected));
        assert!(Arc::ptr_eq(&actual.capture_arc(), &expected.capture_arc()));
    }
    let evaluation = owner
        .plan
        .evaluation
        .as_ref()
        .map(|phase| phase.inference.capture_arc());
    let actual_evaluation = produced.evaluation_capture_owner();
    assert_eq!(actual_evaluation.is_some(), evaluation.is_some());
    if let (Some(actual), Some(expected)) = (&actual_evaluation, &evaluation) {
        assert!(Arc::ptr_eq(actual, expected));
    }
    assert!(
        produced
            .retained_capture_extents()
            .iter()
            .all(|extent| *extent == (0, 0))
    );
    assert!(
        produced
            .retained_metal_recipe_extents()
            .iter()
            .all(|extent| *extent == (0, 0))
    );

    let imported = CompiledAdamWProgramArtifact::from_bytes(produced.as_bytes().to_vec()).unwrap();
    let imported_counts = program_artifact::portable_resume_decode_counts();
    expected_counts = after;
    expected_counts.program_wire += 1;
    expected_counts.mixed_captures += expected.len();
    expected_counts.evaluation_captures += usize::from(evaluation.is_some());
    expected_counts.recurrent_execution_plans +=
        usize::from(metal) * (1 + usize::from(owner.plan.partial_flush.is_some()));
    expected_counts.evaluation_execution_plans +=
        usize::from(metal && owner.plan.evaluation.is_some());
    assert_eq!(
        imported_counts, expected_counts,
        "import must independently decode once"
    );
    assert_eq!(imported, produced);
    assert_eq!(imported.info(), produced.info());
    assert_eq!(imported.as_bytes(), produced.as_bytes());
    for (imported, source) in imported.recurrent_capture_owners().iter().zip(&actual) {
        assert!(
            !Arc::ptr_eq(imported, source),
            "import must independently allocate and admit"
        );
        assert_eq!(imported.capture_identity(), source.capture_identity());
        assert_eq!(
            imported.capture().to_bytes().unwrap(),
            source.capture().to_bytes().unwrap()
        );
    }
    if let Some(imported) = imported.evaluation_capture_owner() {
        assert!(!Arc::ptr_eq(&imported, actual_evaluation.as_ref().unwrap()));
        assert_eq!(
            imported.to_bytes().unwrap(),
            actual_evaluation.unwrap().to_bytes().unwrap()
        );
    }
    produced
}

#[test]
fn produced_artifacts_preserve_cpu_and_metal_phase_inventory() {
    // Scheduled single-step CPU-only v1, then Metal-capable v2 with all four
    // recurrent phases plus evaluation, tied parameters and frozen constants.
    for accumulating in [false, true] {
        let config = if accumulating {
            module_config()
                .with_gradient_accumulation(3)
                .unwrap()
                .with_max_gradient_norm(1.0)
                .unwrap()
        } else {
            module_config()
                .with_captured_multi_step_lr(CompiledMultiStepLr::new(0.01, 0.5, [2, 4]).unwrap())
        };
        let owner = CompiledModuleAdamWPlan::compile_graph_with_dropout(
            config,
            tied_token_mean_dropout(),
            TiedFrozenModule::new([1.0, -1.0]),
            |module, graph, inputs, dropout| {
                let (loss, outputs) = build_tied_dropout(module, graph, inputs, dropout)?;
                Ok(CompiledAdamWGraph::scalar(loss, outputs))
            },
        )
        .unwrap();
        let owner = if accumulating {
            owner
                .with_evaluation_graph(|module, graph, inputs| {
                    let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
                    Ok(CompiledAdamWGraph::scalar(loss, outputs))
                })
                .unwrap()
        } else {
            owner
        };
        let artifact = assert_produced_matches_import(&owner);
        assert_eq!(
            artifact.info().format_version(),
            if accumulating { 2 } else { 1 }
        );
        assert_eq!(
            artifact.recurrent_capture_owners().len(),
            if accumulating { 4 } else { 1 }
        );
    }
}

#[test]
fn produced_token_dropout_artifacts_retain_no_module_and_release_capture_owners() {
    struct ObservedModule {
        module: TiedFrozenModule,
        dropped: Rc<Cell<bool>>,
    }
    impl Module for ObservedModule {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            self.module.visit(prefix, visitor);
        }
    }
    impl Drop for ObservedModule {
        fn drop(&mut self) {
            self.dropped.set(true);
        }
    }
    let dropped = Rc::new(Cell::new(false));
    let owner = CompiledModuleAdamWPlan::compile_graph_with_dropout(
        tied_token_mean_config()
            .with_zero_valid_token_microbatches()
            .unwrap()
            .with_window_loss_report(),
        tied_token_mean_dropout(),
        ObservedModule {
            module: TiedFrozenModule::new([1.0, -1.0]),
            dropped: dropped.clone(),
        },
        |module, graph, inputs, dropout| {
            build_tied_frozen_token_mean(&module.module, graph, inputs, dropout)
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let shared = module.module.shared.bind(graph)?;
        let scaled = graph.mul(shared, inputs["x"])?;
        let tied = graph.add(scaled, shared)?;
        let frozen = module.module.frozen.bind(graph)?;
        let token_output = graph.add(tied, frozen)?;
        let losses = graph.square(token_output)?;
        Ok(CompiledAdamWGraph::token_mean(
            losses,
            BTreeMap::from([("output".into(), token_output)]),
        ))
    })
    .unwrap();
    let artifact = assert_produced_matches_import(&owner);
    let weak = artifact
        .recurrent_capture_owners()
        .iter()
        .map(Arc::downgrade)
        .collect::<Vec<_>>();
    let evaluation = Arc::downgrade(&artifact.evaluation_capture_owner().unwrap());
    let cloned = artifact.clone();
    assert!(artifact.shares_admission_with(&cloned));
    drop(owner);
    assert!(
        dropped.get(),
        "artifact must not retain the module or its seal/stateful wrappers"
    );
    assert!(weak.iter().all(|capture| capture.upgrade().is_some()));
    assert!(evaluation.upgrade().is_some());
    drop(artifact);
    assert!(weak.iter().all(|capture| capture.upgrade().is_some()));
    drop(cloned);
    assert!(weak.iter().all(|capture| capture.upgrade().is_none()));
    assert!(evaluation.upgrade().is_none());
}

#[test]
fn produced_host_index_recipes_preserve_training_and_evaluation_policies() {
    let owner = CompiledModuleAdamWPlan::compile_graph(
        CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.0)
            .unwrap()
            .with_host_token_input("tokens", [1, 2])
            .unwrap(),
        crate::nn::Embedding::new_static(3, 2, None, 17).unwrap(),
        |module, graph, inputs| {
            let output = module.forward(graph, inputs["tokens"])?;
            let squared = graph.square(output)?;
            Ok(CompiledAdamWGraph::scalar(
                graph.sum_all(squared)?,
                BTreeMap::new(),
            ))
        },
    )
    .unwrap()
    .with_evaluation_graph(|module, graph, inputs| {
        let output = module.forward(graph, inputs["tokens"])?;
        let squared = graph.square(output)?;
        Ok(CompiledAdamWGraph::scalar(
            graph.sum_all(squared)?,
            BTreeMap::new(),
        ))
    })
    .unwrap();
    let artifact = assert_produced_matches_import(&owner);
    assert_eq!(artifact.info().format_version(), 2);
    let (bytes, _) = program_artifact::rewrite_json_for_test(&artifact, |json| {
        json["host_token_inputs"] = serde_json::json!({});
    });
    assert_eq!(
        CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err(),
        training("compiled program artifact Metal host policy differs")
    );
}

#[test]
fn independent_import_still_rejects_corrupt_produced_phase_and_policy() {
    let owner = CompiledModuleAdamWPlan::compile_graph(
        module_config().with_gradient_accumulation(2).unwrap(),
        TiedFrozenModule::new([1.0, -1.0]),
        |module, graph, inputs| {
            let (loss, outputs) = build_tied_frozen(module, graph, inputs)?;
            Ok(CompiledAdamWGraph::scalar(loss, outputs))
        },
    )
    .unwrap();
    let artifact = assert_produced_matches_import(&owner);
    for phase in ["main", "accumulation", "partial_flush", "zero_grad"] {
        let (bytes, _) = program_artifact::rewrite_json_for_test(&artifact, |json| {
            let phase = if phase == "main" {
                &mut json["main"]["phase"]
            } else {
                &mut json[phase]
            };
            phase["capture"][0] = serde_json::json!(0);
        });
        let error = CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err();
        assert!(
            matches!(error, Error::SessionTraining { reason } if reason.starts_with("compiled replay:"))
        );
    }
    let (bytes, _) = program_artifact::rewrite_json_for_test(&artifact, |json| {
        json["allow_zero_valid_token_microbatches"] = serde_json::json!(true);
    });
    assert_eq!(
        CompiledAdamWProgramArtifact::from_bytes(bytes).unwrap_err(),
        training("compiled program artifact policy is inconsistent")
    );
}
