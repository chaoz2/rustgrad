use super::*;

struct TemporaryCheckpointFile {
    path: PathBuf,
}

impl TemporaryCheckpointFile {
    fn new() -> std::result::Result<Self, Box<dyn Error>> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "rustgrad-compiled-module-resume-{}-{nonce}.rgab",
            std::process::id()
        ));
        Ok(Self { path })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TemporaryCheckpointFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

const CROSS_PROCESS_EVIDENCE_VERSION: u64 = 1;
const CROSS_PROCESS_EVIDENCE_MAX_BYTES: u64 = 64 << 10;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum CrossProcessBackend {
    Cpu,
    NativeCpu,
}

impl CrossProcessBackend {
    pub(super) fn parse(value: &str) -> std::result::Result<Self, Box<dyn Error>> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "native-cpu" => Ok(Self::NativeCpu),
            other => Err(format!(
                "unknown cross-process backend {other:?}; expected `cpu` or `native-cpu`"
            )
            .into()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::NativeCpu => "native CPU",
        }
    }
}

struct CrossProcessResumePaths {
    pending_bundle: PathBuf,
    terminal_checkpoint: PathBuf,
    evidence: PathBuf,
}

impl CrossProcessResumePaths {
    fn new(directory: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let metadata = fs::symlink_metadata(directory)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("cross-process resume directory must be a real directory".into());
        }
        Ok(Self {
            pending_bundle: directory.join("pending.rgab"),
            terminal_checkpoint: directory.join("expected-terminal.safetensors"),
            evidence: directory.join("expected.json"),
        })
    }

    fn require_absent(&self) -> std::result::Result<(), Box<dyn Error>> {
        for path in [
            &self.pending_bundle,
            &self.terminal_checkpoint,
            &self.evidence,
        ] {
            match fs::symlink_metadata(path) {
                Ok(_) => {
                    return Err(format!(
                        "cross-process resume producer refuses existing output {}",
                        path.display()
                    )
                    .into());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CrossProcessCheckpointEvidence {
    capture_identity: u64,
    replay_step: u64,
    optimizer_step: u64,
    gradient_accumulation_steps: u64,
    accumulation_index: u64,
    discarded_microbatches: u64,
    flushed_window_count: u64,
    flushed_microbatch_count: u64,
    flush_capture_identity: Option<u64>,
    dropout_block_counter: Option<u64>,
    accumulated_token_count: Option<u64>,
    accumulated_loss_numerator_bits: Option<u32>,
    window_loss_report_enabled: bool,
    reset_transition_count: u64,
    reset_capture_identity: Option<u64>,
    accumulation_capture_identity: Option<u64>,
}

impl CrossProcessCheckpointEvidence {
    fn new(checkpoint: &CompiledAdamWCheckpoint) -> Self {
        let info = checkpoint.info();
        Self {
            capture_identity: info.capture_identity(),
            replay_step: info.replay_step(),
            optimizer_step: info.optimizer_step(),
            gradient_accumulation_steps: info.gradient_accumulation_steps(),
            accumulation_index: info.accumulation_index(),
            discarded_microbatches: info.discarded_microbatches(),
            flushed_window_count: info.flushed_window_count(),
            flushed_microbatch_count: info.flushed_microbatch_count(),
            flush_capture_identity: info.flush_capture_identity(),
            dropout_block_counter: info.dropout_block_counter(),
            accumulated_token_count: info.accumulated_token_count(),
            accumulated_loss_numerator_bits: info.accumulated_loss_numerator().map(f32::to_bits),
            window_loss_report_enabled: info.window_loss_report_enabled(),
            reset_transition_count: info.reset_transition_count(),
            reset_capture_identity: info.reset_capture_identity(),
            accumulation_capture_identity: info.accumulation_capture_identity(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CrossProcessResumeEvidence {
    format_version: u64,
    backend: CrossProcessBackend,
    pending_bundle_bytes: u64,
    pending_bundle_checksum: u64,
    terminal_checkpoint_bytes: u64,
    terminal_checkpoint_checksum: u64,
    evaluation_capture_identity: u64,
    initial_evaluation_loss_bits: u64,
    terminal_evaluation_loss_bits: u64,
    pending: CrossProcessCheckpointEvidence,
    terminal: CrossProcessCheckpointEvidence,
}

impl CrossProcessResumeEvidence {
    fn canonical_bytes(&self) -> std::result::Result<Vec<u8>, Box<dyn Error>> {
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn save(&self, path: &Path) -> std::result::Result<(), Box<dyn Error>> {
        let bytes = self.canonical_bytes()?;
        if u64::try_from(bytes.len())? > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load(path: &Path) -> std::result::Result<Self, Box<dyn Error>> {
        let length = fs::metadata(path)?.len();
        if length > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        let bytes = fs::read(path)?;
        Self::from_canonical_bytes(&bytes)
    }

    fn from_canonical_bytes(bytes: &[u8]) -> std::result::Result<Self, Box<dyn Error>> {
        if u64::try_from(bytes.len())? > CROSS_PROCESS_EVIDENCE_MAX_BYTES {
            return Err("cross-process resume evidence exceeds its byte bound".into());
        }
        let evidence: Self = serde_json::from_slice(bytes)?;
        if evidence.format_version != CROSS_PROCESS_EVIDENCE_VERSION {
            return Err(format!(
                "unsupported cross-process resume evidence version {}",
                evidence.format_version
            )
            .into());
        }
        if evidence.canonical_bytes()? != bytes {
            return Err("cross-process resume evidence is not canonical".into());
        }
        Ok(evidence)
    }

    fn validate_backend(
        &self,
        expected: CrossProcessBackend,
    ) -> std::result::Result<(), Box<dyn Error>> {
        if self.backend != expected {
            return Err("cross-process resume evidence backend differs".into());
        }
        Ok(())
    }
}

fn cross_process_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn validate_cross_process_file_binding(
    label: &str,
    bytes: &[u8],
    expected_bytes: u64,
    expected_checksum: u64,
) -> std::result::Result<(), Box<dyn Error>> {
    let actual_bytes = u64::try_from(bytes.len())?;
    if actual_bytes != expected_bytes || cross_process_checksum(bytes) != expected_checksum {
        return Err(format!("cross-process {label} differs from its evidence binding").into());
    }
    Ok(())
}

fn replay_prepared_transformer<R>(runtime: &mut R, replay: u64) -> Result<R::Step>
where
    R: CompiledTrainingWindowRuntime,
{
    runtime.step_batch(batch(replay)?, 0.05)
}

fn replay_prepared_policy_batch<R, B>(runtime: &mut R, batch: B) -> Result<R::Step>
where
    R: CompiledTrainingRatePolicyRuntime,
    B: CompiledInputBatch,
{
    runtime.step_batch_with_rate_policy(batch)
}

fn assert_file_resume_steps_match<S>(actual: &S, expected: &S)
where
    S: CompiledAdamWStep,
{
    assert_eq!(actual.loss(), expected.loss());
    assert_eq!(actual.outputs(), expected.outputs());
    assert_eq!(
        actual
            .output("logits")
            .expect("the file-resume capture exposes logits")
            .shape(),
        &Shape::new([BATCH, TIME, VOCAB])
    );
    assert_eq!(actual.step(), expected.step());
    assert_eq!(actual.optimizer_step(), expected.optimizer_step());
    assert_eq!(actual.accumulation_index(), expected.accumulation_index());
    assert_eq!(actual.loss_weight(), expected.loss_weight());
    assert_eq!(actual.did_update(), expected.did_update());
    assert_eq!(actual.clip_report(), expected.clip_report());
    assert_eq!(actual.window_loss_report(), expected.window_loss_report());
}

pub(super) fn run_exact_resume<R, P>(target_name: &str, mut prepare: P) -> Result<()>
where
    R: CompiledAdamWRuntime + CompiledTrainingWindowCommitRuntime + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<TinyCausalTransformer>,
    ) -> Result<CompiledModuleAdamWSession<TinyCausalTransformer, R>>,
{
    let model = TinyCausalTransformer::new(7)?;
    let initial_mean_sparse_loss = evaluate_mean_sparse_loss(&model)?;
    let plan = compile(model)?;
    let capture_identity = plan.capture_identity();
    let flush_capture_identity = plan
        .flush_capture_identity()
        .expect("three-step accumulation exposes a partial-flush capture");
    let mut uninterrupted = prepare(plan)?.into_training_session();
    let evaluation_identity = uninterrupted
        .evaluation_capture_identity()
        .expect("evaluation was attached before preparation");
    let parameters = uninterrupted.parameter_snapshots()?;
    assert!(parameters.contains_key("tokens.weight"));
    assert!(
        !parameters.contains_key("lm_head.weight"),
        "the tied output head must share the embedding's recurrent state"
    );
    assert_eq!(uninterrupted.gradient_window_size(), ACCUMULATION_STEPS);
    assert_eq!(
        uninterrupted.runtime().max_gradient_norm(),
        Some(MAX_GRADIENT_NORM)
    );
    let initial_parameters = uninterrupted.parameter_snapshots()?;
    let initial_first_moments = uninterrupted.runtime().first_moment_snapshots()?;
    let initial_second_moments = uninterrupted.runtime().second_moment_snapshots()?;
    let empty_accumulators = uninterrupted.runtime().gradient_accumulator_snapshots()?;

    for replay in 1..=2 {
        let step = replay_prepared_transformer(&mut uninterrupted, replay)?;
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay);
        assert!(!step.did_update());
        assert_eq!(step.pending_microbatch_count(), replay);
        assert!(!step.did_close_gradient_window());
    }
    assert_ne!(
        uninterrupted.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    assert_eq!(uninterrupted.parameter_snapshots()?, initial_parameters);
    assert_eq!(
        uninterrupted.runtime().first_moment_snapshots()?,
        initial_first_moments
    );
    assert_eq!(
        uninterrupted.runtime().second_moment_snapshots()?,
        initial_second_moments
    );
    assert_eq!(
        uninterrupted
            .reset_gradient_window()?
            .discarded_microbatches(),
        2
    );
    assert_eq!(uninterrupted.step_count(), 2);
    assert_eq!(uninterrupted.runtime().optimizer_step()?, 0);
    assert_eq!(uninterrupted.runtime().accumulation_index()?, 0);
    assert_eq!(
        uninterrupted.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    let after_reset = uninterrupted.checkpoint()?;
    assert_eq!(
        uninterrupted
            .reset_gradient_window()?
            .discarded_microbatches(),
        0
    );
    assert_eq!(uninterrupted.checkpoint()?, after_reset);

    for replay in 3..=INITIAL_STEPS as u64 {
        let step = replay_prepared_transformer(&mut uninterrupted, replay)?;
        assert_eq!(step.optimizer_step(), 0);
        assert_eq!(step.accumulation_index(), replay - 2);
        assert!(!step.did_update());
    }

    let saved = uninterrupted.checkpoint()?;
    let checkpoint = CompiledAdamWCheckpoint::from_bytes(saved.into_bytes())?;
    let checkpoint_info = checkpoint.info();
    assert_eq!(checkpoint_info.capture_identity(), capture_identity);
    assert_eq!(checkpoint_info.replay_step(), INITIAL_STEPS as u64);
    assert_eq!(checkpoint_info.optimizer_step(), 0);
    assert_eq!(
        checkpoint_info.gradient_accumulation_steps(),
        ACCUMULATION_STEPS
    );
    assert_eq!(checkpoint_info.accumulation_index(), 2);
    assert_eq!(checkpoint_info.discarded_microbatches(), 2);
    assert_eq!(checkpoint_info.flushed_window_count(), 0);
    assert_eq!(checkpoint_info.flushed_microbatch_count(), 0);
    assert_eq!(
        checkpoint_info.flush_capture_identity(),
        Some(flush_capture_identity)
    );
    assert_eq!(
        uninterrupted.partial_window_commit_capture_identity(),
        Some(flush_capture_identity)
    );
    assert_eq!(checkpoint_info.dropout_block_counter(), Some(48));
    let resumed_first_replay = checkpoint_info.replay_step() + 1;
    let resumed_last_replay = checkpoint_info.replay_step() + RESUMED_STEPS as u64;
    let restored_model = TinyCausalTransformer::new(7)?;
    let tied = restored_model.tokens.weight.clone();
    let frozen = restored_model.frozen_scale.clone();
    let restored_plan = CompiledModuleAdamWPlan::compile_with_dropout_from_checkpoint(
        config()?,
        dropout_config(),
        restored_model,
        &checkpoint,
        build,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation(build_evaluation)
    .map_err(|error| error.into_parts().1)?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(restored_plan.step_count(), INITIAL_STEPS as u64);
    let mut resumed = prepare(restored_plan)?.into_training_session();
    assert_eq!(resumed.runtime().optimizer_step()?, 0);
    assert_eq!(resumed.runtime().accumulation_index()?, 2);
    assert_eq!(resumed.checkpoint()?, checkpoint);

    for replay in resumed_first_replay..=resumed_last_replay {
        let expected = replay_prepared_transformer(&mut uninterrupted, replay)?;
        let actual = replay_prepared_transformer(&mut resumed, replay)?;
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        assert_eq!(
            actual.pending_microbatch_count(),
            expected.pending_microbatch_count()
        );
        assert_eq!(
            actual.did_close_gradient_window(),
            expected.did_close_gradient_window()
        );
        let (optimizer_step, accumulation_index, did_update) = match replay {
            5 => (1, 0, true),
            6 => (1, 1, false),
            7 => (1, 2, false),
            _ => unreachable!(),
        };
        assert_eq!(actual.optimizer_step(), optimizer_step);
        assert_eq!(actual.accumulation_index(), accumulation_index);
        assert_eq!(actual.did_update(), did_update);
    }
    let expected_flush = uninterrupted.commit_partial_window(TensorData::scalar(0.05))?;
    let actual_flush = resumed.commit_partial_window(TensorData::scalar(0.05))?;
    assert_eq!(actual_flush.committed_microbatches(), 2);
    assert_eq!(
        actual_flush.committed_microbatches(),
        expected_flush.committed_microbatches()
    );
    assert_eq!(
        actual_flush.did_commit_window(),
        expected_flush.did_commit_window()
    );
    assert!(actual_flush.did_commit_window());
    let after_flush = resumed.checkpoint()?;
    let empty_flush = resumed.commit_partial_window(TensorData::scalar(0.05))?;
    assert_eq!(empty_flush.committed_microbatches(), 0);
    assert!(!empty_flush.did_commit_window());
    assert_eq!(resumed.checkpoint()?, after_flush);
    assert_eq!(resumed.runtime().optimizer_step()?, 2);
    assert_eq!(resumed.runtime().accumulation_index()?, 0);

    assert_eq!(
        resumed.parameter_snapshots()?,
        uninterrupted.parameter_snapshots()?
    );
    assert_eq!(
        resumed.runtime().first_moment_snapshots()?,
        uninterrupted.runtime().first_moment_snapshots()?
    );
    assert_eq!(
        resumed.runtime().second_moment_snapshots()?,
        uninterrupted.runtime().second_moment_snapshots()?
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        uninterrupted.runtime().gradient_accumulator_snapshots()?
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        empty_accumulators
    );
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let before_evaluation_checkpoint = resumed.checkpoint()?;
    let mut final_mean_sparse_loss = 0.0;
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluated = resumed.evaluate_batch(batch(replay)?)?;
        assert_eq!(evaluated.capture_identity(), evaluation_identity);
        assert_eq!(
            evaluated.output("logits").unwrap().shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        final_mean_sparse_loss += evaluated.loss().scalar_at(0).as_f64();
    }
    final_mean_sparse_loss /= ACCUMULATION_STEPS as f64;
    assert_eq!(resumed.checkpoint()?, before_evaluation_checkpoint);
    let tied_version = tied.version()?;
    let frozen_before = frozen.snapshot()?;
    let runtime_step = resumed.step_count();
    let (uninterrupted_model, uninterrupted_checkpoint) = uninterrupted
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    let (restored_model, finished_checkpoint) = resumed
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(finished_checkpoint, before_evaluation_checkpoint);
    assert_eq!(finished_checkpoint, uninterrupted_checkpoint);
    let live = restored_model.state_dict()?;
    assert_eq!(live, uninterrupted_model.state_dict()?);
    assert_eq!(
        restored_model.tokens.weight.value()?,
        live.tensors()["tokens.weight"]
    );
    assert!(!live.tensors().contains_key("lm_head.weight"));
    assert_eq!(restored_model.tokens.weight.version()?, tied_version + 1);
    let frozen_after = restored_model.frozen_scale.snapshot()?;
    assert_eq!(frozen_after.data, frozen_before.data);
    assert_eq!(frozen_after.version, frozen_before.version);
    let versions_before_eval = restored_model
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| Ok((name, parameter.version()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let first_eval = evaluate(&restored_model)?;
    let second_eval = evaluate(&restored_model)?;
    let published_mean_sparse_loss = evaluate_mean_sparse_loss(&restored_model)?;
    assert_eq!(first_eval.shape(), &Shape::new([BATCH, TIME, VOCAB]));
    assert_eq!(first_eval, second_eval);
    assert!(
        (0..first_eval.shape().numel()?)
            .all(|index| first_eval.scalar_at(index).as_f64().is_finite())
    );
    for (name, parameter) in restored_model.trainable_parameters()? {
        assert_eq!(parameter.version()?, versions_before_eval[&name]);
    }
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "compiled causal Transformer eval loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    assert!((final_mean_sparse_loss - published_mean_sparse_loss).abs() < 1e-5);
    println!(
        "{target_name}: capture={capture_identity:016x}, steps={}, eval_mean_sparse_loss={:.6} -> {:.6}, exact_resume=true, published=true",
        runtime_step, initial_mean_sparse_loss, final_mean_sparse_loss
    );
    Ok(())
}

pub(super) fn run_cpu_reuse() -> Result<()> {
    let source = BufferedTinyCausalTransformer::new(7)?;
    let initial_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&source.transformer)?;
    assert_eq!(
        masked_batch(3)?.loss_mask.to_vec_f64(),
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "the third replay is a right-padded final partial row"
    );
    let source_policy_frozen = source
        .trainable_parameters()?
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .expect("the maintained Transformer exposes the policy-frozen parameter")
        .value()?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let plan = CompiledAdamWPlan::compile_module_graph_with_dropout(
        reuse_config(schedule.clone())?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            let (losses, outputs) = build_buffered(model, graph, inputs, dropout)?;
            Ok(CompiledAdamWGraph::token_mean(losses, outputs))
        },
    )?;
    assert_eq!(builds.get(), 1, "the training graph must compile once");
    let capture_identity = plan.capture_identity();
    assert_eq!(
        plan.token_weighted_gradient_accumulation_mask(),
        Some(LOSS_MASK)
    );
    let target =
        CpuSessionTarget::new().with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let mut uninterrupted = plan.prepare(&target)?;
    assert_eq!(
        uninterrupted.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    assert_eq!(uninterrupted.captured_multi_step_lr(), Some(&schedule));
    assert!(
        !uninterrupted
            .parameter_snapshots()?
            .contains_key(POLICY_FROZEN)
    );

    for replay in 1..=4 {
        let batch = masked_batch(replay)?;
        let loss_weight = loss_mask_weight(&batch.loss_mask);
        let step = uninterrupted.step_batch_scheduled(batch)?;
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }
    assert_eq!(uninterrupted.optimizer_step()?, 1);
    assert_eq!(uninterrupted.accumulation_index()?, 1);
    let checkpoint = uninterrupted.checkpoint()?;
    assert_eq!(checkpoint.info().replay_step(), 4);
    assert_eq!(checkpoint.info().optimizer_step(), 1);
    assert_eq!(checkpoint.info().accumulation_index(), 1);
    assert_eq!(checkpoint.info().accumulated_token_count(), Some(5));
    assert_eq!(uninterrupted.dropout_block_counter()?, Some(48));

    let restored_plan = plan.restore_checkpoint(&checkpoint)?;
    assert_eq!(
        builds.get(),
        1,
        "checkpoint restore must not rebuild the graph"
    );
    assert_eq!(
        plan.step_count(),
        0,
        "the borrowed source plan stays reusable"
    );
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(restored_plan.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(restored_plan.step_count(), 4);
    let mut resumed = restored_plan.prepare(&target)?;
    assert_eq!(resumed.captured_multi_step_lr(), Some(&schedule));
    assert_eq!(
        resumed.non_finite_policy(),
        CpuNonFinitePolicy::RejectTransition
    );
    assert_eq!(resumed.checkpoint()?, checkpoint);

    let before_wrong_entrypoint = resumed.checkpoint()?;
    assert!(resumed.step_batch(masked_batch(5)?, 0.05).is_err());
    assert_eq!(resumed.checkpoint()?, before_wrong_entrypoint);
    for replay in 5..=6 {
        let loss_weight = loss_mask_weight(&masked_batch(replay)?.loss_mask);
        let expected = uninterrupted.step_batch_scheduled(masked_batch(replay)?)?;
        let actual = resumed.step_batch_scheduled(masked_batch(replay)?)?;
        assert_eq!(actual.loss(), expected.loss());
        assert_eq!(actual.loss_weight(), expected.loss_weight());
        assert_eq!(actual.loss_weight(), loss_weight);
        assert_eq!(
            actual
                .output("logits")
                .expect("the reuse capture exposes logits")
                .shape(),
            &Shape::new([BATCH, TIME, VOCAB])
        );
        assert_eq!(actual.outputs(), expected.outputs());
        assert_eq!(actual.step(), expected.step());
        assert_eq!(actual.optimizer_step(), expected.optimizer_step());
        assert_eq!(actual.accumulation_index(), expected.accumulation_index());
        assert_eq!(actual.did_update(), expected.did_update());
        assert_eq!(
            resumed.parameter_snapshots()?,
            uninterrupted.parameter_snapshots()?
        );
        assert_eq!(
            resumed.first_moment_snapshots()?,
            uninterrupted.first_moment_snapshots()?
        );
        assert_eq!(
            resumed.second_moment_snapshots()?,
            uninterrupted.second_moment_snapshots()?
        );
        assert_eq!(
            resumed.gradient_accumulator_snapshots()?,
            uninterrupted.gradient_accumulator_snapshots()?
        );
        assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    }
    assert_eq!(resumed.optimizer_step()?, 2);
    assert_eq!(resumed.accumulation_index()?, 0);
    assert_eq!(resumed.dropout_block_counter()?, Some(72));

    let final_parameters = resumed.parameter_snapshots()?;
    let destination = BufferedTinyCausalTransformer::new(0xdecafbad)?;
    let tied_before = destination.transformer.tokens.weight.snapshot()?;
    let tied_identity = tied_before.identity;
    assert_ne!(
        destination.transformer.tokens.weight.value()?,
        final_parameters["tokens.weight"]
    );
    let policy_frozen = destination
        .trainable_parameters()?
        .into_iter()
        .find_map(|(name, parameter)| (name == POLICY_FROZEN).then_some(parameter))
        .expect("the destination exposes the policy-frozen parameter");
    assert_ne!(policy_frozen.value()?, source_policy_frozen);
    policy_frozen.replace(source_policy_frozen)?;
    let policy_frozen_before = policy_frozen.snapshot()?;
    let inherent_frozen_before = destination.transformer.frozen_scale.snapshot()?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let running_marker_before = destination.running_marker.snapshot()?;
    let effective_trainables_before = destination
        .trainable_parameters()?
        .into_iter()
        .filter(|(name, _)| name != POLICY_FROZEN)
        .map(|(name, parameter)| Ok((name, parameter.snapshot()?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let final_checkpoint = resumed.checkpoint()?;
    assert!(resumed.publish_parameters(&destination)?.is_clean());
    assert_eq!(resumed.checkpoint()?, final_checkpoint);

    let mut tied_alias = None;
    destination.visit("", &mut |name, parameter, _| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output alias");
    assert_eq!(tied_alias.id(), tied_identity);
    assert_eq!(destination.transformer.tokens.weight.id(), tied_identity);
    assert!(
        !destination
            .state_dict()?
            .tensors()
            .contains_key("lm_head.weight")
    );
    for (name, parameter) in destination.trainable_parameters()? {
        if name != POLICY_FROZEN {
            let before = &effective_trainables_before[&name];
            let after = parameter.snapshot()?;
            assert_eq!(after.data, final_parameters[&name]);
            assert_eq!(after.identity, before.identity);
            assert_eq!(after.trainable, before.trainable);
            assert_eq!(
                after.version,
                before
                    .version
                    .checked_add(1)
                    .expect("successful publication cannot overflow a version")
            );
        }
    }
    let tied_after = destination.transformer.tokens.weight.snapshot()?;
    assert_eq!(tied_after.data, final_parameters["tokens.weight"]);
    assert_eq!(tied_after.identity, tied_before.identity);
    assert_eq!(tied_after.trainable, tied_before.trainable);
    assert_eq!(
        tied_after.version,
        tied_before
            .version
            .checked_add(1)
            .expect("successful publication cannot overflow the tied weight version")
    );
    let tied_alias_after = tied_alias.snapshot()?;
    assert_eq!(tied_alias_after.data, tied_after.data);
    assert_eq!(tied_alias_after.identity, tied_after.identity);
    assert_eq!(tied_alias_after.version, tied_after.version);
    assert_eq!(tied_alias_after.trainable, tied_after.trainable);
    let policy_frozen_after = policy_frozen.snapshot()?;
    assert_eq!(policy_frozen_after.data, policy_frozen_before.data);
    assert_eq!(policy_frozen_after.version, policy_frozen_before.version);
    assert_eq!(policy_frozen_after.identity, policy_frozen_before.identity);
    assert_eq!(
        policy_frozen_after.trainable,
        policy_frozen_before.trainable
    );
    let inherent_frozen_after = destination.transformer.frozen_scale.snapshot()?;
    assert_eq!(inherent_frozen_after.data, inherent_frozen_before.data);
    assert_eq!(
        inherent_frozen_after.version,
        inherent_frozen_before.version
    );
    assert_eq!(
        inherent_frozen_after.identity,
        inherent_frozen_before.identity
    );
    assert_eq!(
        inherent_frozen_after.trainable,
        inherent_frozen_before.trainable
    );
    let running_marker_after = destination.running_marker.snapshot()?;
    assert_eq!(running_marker_after.data, running_marker_before.data);
    assert_eq!(running_marker_after.version, running_marker_before.version);
    assert_eq!(
        running_marker_after.identity,
        running_marker_before.identity
    );
    assert_eq!(
        running_marker_after.trainable,
        running_marker_before.trainable
    );

    let final_mean_sparse_loss = evaluate_mean_masked_sparse_loss(&destination.transformer)?;
    assert!(
        final_mean_sparse_loss < initial_mean_sparse_loss,
        "compile-once causal Transformer loss did not decrease: {initial_mean_sparse_loss} -> {final_mean_sparse_loss}"
    );
    println!(
        "CPU reuse: capture={capture_identity:016x}, builds={}, checkpoint=(replay=4, optimizer=1, accumulation=1), optimizer_steps={}, eval_mean_sparse_loss={:.6} -> {:.6}, exact_resume=true, published=true",
        builds.get(),
        resumed.optimizer_step()?,
        initial_mean_sparse_loss,
        final_mean_sparse_loss
    );
    Ok(())
}

pub(super) fn run_cpu_file_resume() -> std::result::Result<(), Box<dyn Error>> {
    let target =
        CpuSessionTarget::new().with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    run_file_resume::<CpuCompiledAdamW, _, _, _, _>(
        "CPU file resume",
        |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
        |_| {},
        |_| {},
        |_| {},
    )
}

fn assert_cross_process_pending_checkpoint(checkpoint: &CompiledAdamWCheckpoint) {
    let info = checkpoint.info();
    assert_eq!(info.replay_step(), 4);
    assert_eq!(info.optimizer_step(), 1);
    assert_eq!(info.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(info.accumulation_index(), 1);
    assert_eq!(info.accumulated_token_count(), Some(5));
    assert!(
        info.dropout_block_counter()
            .is_some_and(|counter| counter > 0)
    );
    assert!(
        info.accumulated_loss_numerator()
            .is_some_and(|value| value.is_finite() && value != 0.0)
    );
    assert!(info.window_loss_report_enabled());
}

fn assert_cross_process_terminal_checkpoint(checkpoint: &CompiledAdamWCheckpoint) {
    let info = checkpoint.info();
    assert_eq!(info.replay_step(), 11);
    assert_eq!(info.optimizer_step(), 4);
    assert_eq!(info.gradient_accumulation_steps(), ACCUMULATION_STEPS);
    assert_eq!(info.accumulation_index(), 0);
    assert_eq!(info.discarded_microbatches(), 1);
    assert_eq!(info.flushed_window_count(), 1);
    assert_eq!(info.flushed_microbatch_count(), 1);
    assert_eq!(info.accumulated_token_count(), Some(0));
    assert_eq!(info.accumulated_loss_numerator(), Some(0.0));
    assert_eq!(info.reset_transition_count(), 1);
    assert!(info.window_loss_report_enabled());
}

fn assert_cross_process_zero_accumulators<R>(
    session: &CompiledModuleAdamWSession<FileResumeTransformer, R>,
) -> Result<()>
where
    R: CompiledScheduledAdamWRuntime,
{
    let accumulators = session.gradient_accumulator_snapshots()?;
    assert!(!accumulators.is_empty());
    for (name, accumulator) in accumulators {
        for value in accumulator.values() {
            assert_eq!(
                value.to_bits(),
                0,
                "terminal accumulator {name} must contain positive zero"
            );
        }
    }
    Ok(())
}

fn run_cross_process_file_resume_suffix<R, S, E>(
    session: &mut CompiledModuleAdamWSession<FileResumeTransformer, R>,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<f64, Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    const WINDOW_LOSS_WEIGHT: u64 = 11;
    for replay in 5..=9 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(session, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), matches!(replay, 6 | 9));
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(report) = step.window_loss_report() {
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
    }

    let partial = replay_prepared_policy_batch(session, file_resume_batch(10)?)?;
    validate_step(&partial);
    assert!(!partial.did_update());
    assert_eq!(partial.accumulation_index(), 1);
    let flush = session.commit_partial_window_with_rate_policy()?;
    assert!(flush.did_commit_window());
    assert_eq!(flush.committed_microbatches(), 1);

    let discard = replay_prepared_policy_batch(session, file_resume_batch(11)?)?;
    validate_step(&discard);
    assert!(!discard.did_update());
    assert_eq!(discard.accumulation_index(), 1);
    let reset = session.reset_gradient_window()?;
    assert_eq!(reset.discarded_microbatches(), 1);
    assert_eq!(session.optimizer_step()?, 4);
    assert_eq!(session.accumulation_index()?, 0);
    assert_cross_process_zero_accumulators(session)?;

    let before_evaluation = session.checkpoint()?;
    let before_counter = before_evaluation.info().dropout_block_counter();
    let before_accumulators = session.gradient_accumulator_snapshots()?;
    let terminal_loss = evaluate_mean_file_resume_loss(session, &mut validate_evaluation)?;
    assert_eq!(session.checkpoint()?, before_evaluation);
    assert_eq!(
        session.checkpoint()?.info().dropout_block_counter(),
        before_counter
    );
    assert_eq!(
        session.gradient_accumulator_snapshots()?,
        before_accumulators
    );
    Ok(terminal_loss)
}

fn produce_cross_process_file_resume<R, P, V, S, E>(
    backend: CrossProcessBackend,
    paths: &CrossProcessResumePaths,
    mut prepare: P,
    mut validate_preparation: V,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    paths.require_absent()?;
    let schedule = CompiledMultiStepLr::new(1e-3, 0.5, [1])?;
    let config = file_resume_config(schedule)?;
    let source = FileResumeTransformer::new(0x5678)?;
    let builds = Cell::new(0_u64);
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        config,
        dropout_config(),
        source,
        |model, graph, inputs, ignore_index, dropout| {
            builds.set(builds.get() + 1);
            build_file_resume(model, graph, inputs, ignore_index, dropout)
        },
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_file_resume_evaluation)
    .map_err(|error| error.into_parts().1)?;
    assert_eq!(builds.get(), 1, "the producer training graph compiles once");
    let evaluation_capture_identity = source_plan
        .evaluation_capture_identity()
        .expect("the cross-process evaluator is attached");
    let program_artifact = source_plan.program_artifact()?;
    let mut source = prepare(source_plan)?;
    validate_preparation(&source);

    let before_initial_evaluation = source.checkpoint()?;
    let initial_loss = evaluate_mean_file_resume_loss(&mut source, &mut validate_evaluation)?;
    assert_eq!(source.checkpoint()?, before_initial_evaluation);
    for replay in 1..=4 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(&mut source, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
    }

    let pending_checkpoint = source.module_checkpoint()?;
    assert_cross_process_pending_checkpoint(pending_checkpoint.optimizer_checkpoint());
    let pending_bundle =
        CompiledAdamWResumeBundle::new(program_artifact, pending_checkpoint.clone())?;
    pending_bundle.save_file(&paths.pending_bundle)?;

    let terminal_loss = run_cross_process_file_resume_suffix(
        &mut source,
        &mut validate_step,
        &mut validate_evaluation,
    )?;
    assert!(
        terminal_loss < initial_loss,
        "cross-process producer loss did not decrease: {initial_loss} -> {terminal_loss}"
    );
    let terminal_checkpoint = source.module_checkpoint()?;
    assert_cross_process_terminal_checkpoint(terminal_checkpoint.optimizer_checkpoint());
    terminal_checkpoint.save_file(&paths.terminal_checkpoint)?;

    let evidence = CrossProcessResumeEvidence {
        format_version: CROSS_PROCESS_EVIDENCE_VERSION,
        backend,
        pending_bundle_bytes: u64::try_from(pending_bundle.as_bytes().len())?,
        pending_bundle_checksum: cross_process_checksum(pending_bundle.as_bytes()),
        terminal_checkpoint_bytes: u64::try_from(terminal_checkpoint.as_bytes().len())?,
        terminal_checkpoint_checksum: cross_process_checksum(terminal_checkpoint.as_bytes()),
        evaluation_capture_identity,
        initial_evaluation_loss_bits: initial_loss.to_bits(),
        terminal_evaluation_loss_bits: terminal_loss.to_bits(),
        pending: CrossProcessCheckpointEvidence::new(pending_checkpoint.optimizer_checkpoint()),
        terminal: CrossProcessCheckpointEvidence::new(terminal_checkpoint.optimizer_checkpoint()),
    };
    evidence.save(&paths.evidence)?;
    assert_eq!(CrossProcessResumeEvidence::load(&paths.evidence)?, evidence);
    println!(
        "{} cross-process producer: replay=4, optimizer=1, accumulation=1, terminal_replay=11, terminal_optimizer=4, loss={initial_loss:.6}->{terminal_loss:.6}",
        backend.label()
    );
    Ok(())
}

fn consume_cross_process_file_resume<R, P, V, S, E>(
    backend: CrossProcessBackend,
    paths: &CrossProcessResumePaths,
    mut prepare: P,
    mut validate_preparation: V,
    validate_step: S,
    validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    let evidence = CrossProcessResumeEvidence::load(&paths.evidence)?;
    evidence.validate_backend(backend)?;
    let pending_bundle = CompiledAdamWResumeBundle::load_file(&paths.pending_bundle)?;
    let terminal_checkpoint = CompiledModuleAdamWCheckpoint::load_file(&paths.terminal_checkpoint)?;
    validate_cross_process_file_binding(
        "pending bundle",
        pending_bundle.as_bytes(),
        evidence.pending_bundle_bytes,
        evidence.pending_bundle_checksum,
    )?;
    validate_cross_process_file_binding(
        "terminal checkpoint",
        terminal_checkpoint.as_bytes(),
        evidence.terminal_checkpoint_bytes,
        evidence.terminal_checkpoint_checksum,
    )?;
    assert_eq!(
        CrossProcessCheckpointEvidence::new(pending_bundle.checkpoint().optimizer_checkpoint()),
        evidence.pending
    );
    assert_eq!(
        CrossProcessCheckpointEvidence::new(terminal_checkpoint.optimizer_checkpoint()),
        evidence.terminal
    );
    assert_eq!(
        terminal_checkpoint.evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    assert_eq!(
        pending_bundle.checkpoint().evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    assert_cross_process_pending_checkpoint(pending_bundle.checkpoint().optimizer_checkpoint());
    assert_cross_process_terminal_checkpoint(terminal_checkpoint.optimizer_checkpoint());

    let initial_loss = f64::from_bits(evidence.initial_evaluation_loss_bits);
    let expected_terminal_loss = f64::from_bits(evidence.terminal_evaluation_loss_bits);
    assert!(initial_loss.is_finite());
    assert!(expected_terminal_loss.is_finite());
    assert!(expected_terminal_loss < initial_loss);

    let destination = FileResumeTransformer::new(0x9abc)?;
    destination.frozen_scale.replace(TensorData::scalar(7.0))?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let destination_tied_identity = destination.tokens.weight.id();
    let mut tied_alias = None;
    let mut destination_states = Vec::new();
    destination.visit("", &mut |name, parameter, kind| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
        destination_states.push((
            name,
            parameter.clone(),
            kind,
            parameter
                .snapshot()
                .expect("the cross-process destination remains readable"),
        ));
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output head");
    assert_eq!(tied_alias.id(), destination_tied_identity);

    let restored_plan =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &pending_bundle)
            .map_err(|error| error.into_parts().1)?;
    assert_eq!(
        restored_plan.capture_identity(),
        evidence.pending.capture_identity
    );
    assert_eq!(
        restored_plan.evaluation_capture_identity(),
        Some(evidence.evaluation_capture_identity)
    );
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let mut resumed = prepare(restored_plan)?;
    validate_preparation(&resumed);
    assert_eq!(&resumed.module_checkpoint()?, pending_bundle.checkpoint());
    let terminal_loss =
        run_cross_process_file_resume_suffix(&mut resumed, validate_step, validate_evaluation)?;
    assert_eq!(
        terminal_loss.to_bits(),
        evidence.terminal_evaluation_loss_bits
    );
    assert_eq!(resumed.module_checkpoint()?, terminal_checkpoint);

    let (resumed_model, resumed_checkpoint) = resumed
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(resumed_checkpoint, terminal_checkpoint);
    let mut resumed_states = Vec::new();
    resumed_model.visit("", &mut |name, parameter, kind| {
        resumed_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the cross-process result remains readable"),
        ));
    });
    assert_eq!(resumed_states.len(), destination_states.len());
    for ((name, _, kind, before), (actual_name, actual_kind, actual)) in
        destination_states.iter().zip(&resumed_states)
    {
        assert_eq!(actual_name, name);
        assert_eq!(actual_kind, kind);
        assert_eq!(actual.identity, before.identity);
        assert_eq!(actual.trainable, before.trainable);
        assert_eq!(
            actual.version,
            before
                .version
                .checked_add(1)
                .expect("successful publication cannot overflow a version")
        );
    }
    assert_eq!(resumed_model.tokens.weight.id(), destination_tied_identity);
    assert_eq!(tied_alias.id(), destination_tied_identity);
    assert_eq!(tied_alias.value()?, resumed_model.tokens.weight.value()?);
    assert!(resumed_model.positions.weight.is_trainable());
    assert!(!resumed_model.frozen_scale.is_trainable());
    assert!(!resumed_model.running_marker.is_trainable());
    assert_ne!(resumed_model.frozen_scale.value()?, TensorData::scalar(7.0));
    assert_ne!(
        resumed_model.running_marker.value()?,
        TensorData::scalar(29.0)
    );
    println!(
        "{} cross-process consumer: exact_terminal=true, different_init=true, published=true, loss={initial_loss:.6}->{terminal_loss:.6}",
        backend.label()
    );
    Ok(())
}

fn assert_native_file_resume_preparation(
    session: &CompiledModuleAdamWSession<FileResumeTransformer, NativeCpuCompiledAdamW<'_>>,
) {
    let preparation = session.native_cpu_preparation_report();
    for program in [
        preparation.main(),
        preparation
            .accumulation()
            .expect("three-step accumulation has a native sibling"),
        preparation
            .partial_flush()
            .expect("partial commit has a native sibling"),
        preparation
            .zero_grad()
            .expect("window reset has a native sibling"),
        preparation
            .evaluation()
            .expect("file resume has a native evaluator"),
    ] {
        assert!(program.native_item_count() > 0);
        assert_eq!(program.fallback_count(), 0);
    }
}

fn assert_native_file_resume_step(step: &NativeCpuCompiledAdamWStepResult) {
    assert!(step.report().native_item_count() > 0);
    assert!(step.report().executed_native_item_count() > 0);
    assert!(step.report().module_dispatch_count() > 0);
    assert_eq!(
        step.report().module_dispatched_native_item_count(),
        step.report().executed_native_item_count()
    );
    assert_eq!(step.report().fallback_count(), 0);
}

fn assert_native_file_resume_evaluation(evaluation: &NativeCpuCompiledEvaluationResult) {
    assert!(evaluation.report().native_item_count() > 0);
    assert!(evaluation.report().executed_native_item_count() > 0);
    assert!(evaluation.report().module_dispatch_count() > 0);
    assert_eq!(
        evaluation.report().module_dispatched_native_item_count(),
        evaluation.report().executed_native_item_count()
    );
    assert_eq!(evaluation.report().fallback_count(), 0);
}

pub(super) fn run_native_cpu_file_resume() -> std::result::Result<(), Box<dyn Error>> {
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .vectorized(true)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let preparations = Cell::new(0_u64);
    run_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
        "native CPU file resume",
        |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
        |session| {
            assert_native_file_resume_preparation(session);
            let invocation = preparations.get();
            if invocation > 0 {
                assert_eq!(
                    session
                        .native_cpu_preparation_report()
                        .compiler_process_count(),
                    0,
                    "artifact-restored preparation must reuse durable native modules"
                );
            }
            preparations.set(invocation + 1);
        },
        assert_native_file_resume_step,
        assert_native_file_resume_evaluation,
    )
}

pub(super) fn run_cross_process_producer(
    backend: CrossProcessBackend,
    directory: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let paths = CrossProcessResumePaths::new(directory)?;
    match backend {
        CrossProcessBackend::Cpu => {
            let target = CpuSessionTarget::new()
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            produce_cross_process_file_resume::<CpuCompiledAdamW, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                |_| {},
                |_| {},
                |_| {},
            )
        }
        CrossProcessBackend::NativeCpu => {
            let executor = CapturedReplayExecutor::default();
            let target = NativeCpuSessionTarget::new(&executor)
                .vectorized(true)
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            produce_cross_process_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                assert_native_file_resume_preparation,
                assert_native_file_resume_step,
                assert_native_file_resume_evaluation,
            )
        }
    }
}

pub(super) fn run_cross_process_consumer(
    backend: CrossProcessBackend,
    directory: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let paths = CrossProcessResumePaths::new(directory)?;
    match backend {
        CrossProcessBackend::Cpu => {
            let target = CpuSessionTarget::new()
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            consume_cross_process_file_resume::<CpuCompiledAdamW, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                |_| {},
                |_| {},
                |_| {},
            )
        }
        CrossProcessBackend::NativeCpu => {
            let executor = CapturedReplayExecutor::default();
            let target = NativeCpuSessionTarget::new(&executor)
                .vectorized(true)
                .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
            consume_cross_process_file_resume::<NativeCpuCompiledAdamW<'_>, _, _, _, _>(
                backend,
                &paths,
                |plan| plan.prepare(&target).map_err(|error| error.into_parts().1),
                assert_native_file_resume_preparation,
                assert_native_file_resume_step,
                assert_native_file_resume_evaluation,
            )
        }
    }
}

fn run_file_resume<R, P, V, S, E>(
    target_name: &str,
    mut prepare: P,
    mut validate_preparation: V,
    mut validate_step: S,
    mut validate_evaluation: E,
) -> std::result::Result<(), Box<dyn Error>>
where
    R: CompiledScheduledAdamWRuntime
        + CompiledCheckpointRestoreRuntime<Checkpoint = CompiledAdamWCheckpoint>
        + CompiledTrainingRatePolicyWindowCommitRuntime
        + CompiledEvaluationRuntime,
    P: FnMut(
        CompiledModuleAdamWPlan<FileResumeTransformer>,
    ) -> Result<CompiledModuleAdamWSession<FileResumeTransformer, R>>,
    V: FnMut(&CompiledModuleAdamWSession<FileResumeTransformer, R>),
    S: FnMut(&R::Step),
    E: FnMut(&R::Evaluation),
{
    const LAST_REPLAY: u64 = 9;
    const WINDOW_LOSS_WEIGHT: u64 = 11;

    let schedule = CompiledMultiStepLr::new(1e-3, 0.5, [1])?;
    let config = file_resume_config(schedule.clone())?;
    assert!(config.clip_report_enabled());
    assert!(config.window_loss_report_enabled());
    assert_eq!(
        config.token_weighted_ignore_index(),
        Some((MaskedTransformerBatch::TARGETS, FILE_RESUME_IGNORE_INDEX))
    );
    assert_eq!(
        config.inputs().map(|(name, _, _)| name).collect::<Vec<_>>(),
        [
            MaskedTransformerBatch::TARGETS,
            MaskedTransformerBatch::TOKENS
        ],
        "sentinel targets are the only attention-validity source"
    );
    assert!(
        file_resume_batch(3)?.has_fully_masked_sample(),
        "the zero-length row must exercise fully masked attention"
    );
    let source = FileResumeTransformer::new(0x5678)?;
    let source_initial = source.state_dict()?;
    let source_plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        config.clone(),
        dropout_config(),
        source,
        build_file_resume,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_file_resume_evaluation)
    .map_err(|error| error.into_parts().1)?;
    let capture_identity = source_plan.capture_identity();
    let evaluation_identity = source_plan
        .evaluation_capture_identity()
        .expect("the compiler-owned token-mean evaluator is attached");
    let program_artifact = source_plan.program_artifact()?;
    assert_eq!(source_plan.captured_multi_step_lr(), Some(&schedule));
    let mut uninterrupted = prepare(source_plan)?;
    validate_preparation(&uninterrupted);
    let before_initial_evaluation = uninterrupted.checkpoint()?;
    let initial_loss =
        evaluate_mean_file_resume_loss(&mut uninterrupted, &mut validate_evaluation)?;
    assert_eq!(uninterrupted.checkpoint()?, before_initial_evaluation);
    for replay in 1..=4 {
        let batch = file_resume_batch(replay)?;
        let loss_weight = loss_mask_weight(batch.loss_mask());
        let step = replay_prepared_policy_batch(&mut uninterrupted, batch)?;
        validate_step(&step);
        assert_eq!(step.loss_weight(), loss_weight);
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(report) = step.clip_report() {
            assert!(report.is_finite());
            assert!(report.did_clip().is_some());
        }
        if let Some(report) = step.window_loss_report() {
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
    }
    assert_eq!(uninterrupted.step_count(), 4);
    assert_eq!(uninterrupted.optimizer_step()?, 1);
    assert_eq!(uninterrupted.accumulation_index()?, 1);
    let saved_optimizer_checkpoint = uninterrupted.checkpoint()?;
    let saved_dropout_cursor = saved_optimizer_checkpoint
        .info()
        .dropout_block_counter()
        .expect("attention and residual dropout are captured");
    assert!(saved_dropout_cursor > 0);

    let checkpoint = uninterrupted.module_checkpoint()?;
    assert_eq!(
        checkpoint.optimizer_checkpoint(),
        &saved_optimizer_checkpoint
    );
    assert_eq!(checkpoint.optimizer_checkpoint().info().replay_step(), 4);
    assert_eq!(
        checkpoint
            .optimizer_checkpoint()
            .info()
            .accumulation_index(),
        1
    );
    assert_eq!(
        checkpoint
            .optimizer_checkpoint()
            .info()
            .accumulated_token_count(),
        Some(5)
    );
    let resume_bundle = CompiledAdamWResumeBundle::new(program_artifact, checkpoint.clone())?;
    let resume_file = TemporaryCheckpointFile::new()?;
    resume_bundle.save_file(resume_file.path())?;
    let resume_bundle = CompiledAdamWResumeBundle::load_file(resume_file.path())?;
    assert_eq!(resume_bundle.checkpoint(), &checkpoint);
    let decoded = resume_bundle.checkpoint().clone();
    assert_eq!(decoded, checkpoint);
    assert!(
        decoded
            .optimizer_checkpoint()
            .info()
            .window_loss_report_enabled()
    );
    let pending_loss_numerator = decoded
        .optimizer_checkpoint()
        .info()
        .accumulated_loss_numerator()
        .expect("window-loss reporting checkpoints its pending F32 numerator");
    assert!(pending_loss_numerator.is_finite());
    assert_ne!(pending_loss_numerator, 0.0);

    let destination = FileResumeTransformer::new(0x9abc)?;
    destination.frozen_scale.replace(TensorData::scalar(7.0))?;
    destination
        .running_marker
        .replace(TensorData::scalar(29.0))?;
    let destination_initial = destination.state_dict()?;
    assert_ne!(destination_initial.tensors(), source_initial.tensors());
    assert_ne!(
        destination.positions.weight.value()?,
        source_initial.tensors()[FILE_RESUME_POLICY_FROZEN]
    );
    let destination_tied_identity = destination.tokens.weight.id();
    let mut tied_alias = None;
    let mut destination_states = Vec::new();
    destination.visit("", &mut |name, parameter, kind| {
        if name == "lm_head.weight" {
            tied_alias = Some(parameter.clone());
        }
        destination_states.push((
            name,
            parameter.clone(),
            kind,
            parameter
                .snapshot()
                .expect("the destination state remains readable"),
        ));
    });
    let tied_alias = tied_alias.expect("the destination exposes the tied output head");
    assert_eq!(tied_alias.id(), destination_tied_identity);

    let restored_plan =
        CompiledModuleAdamWPlan::restore_from_resume_bundle(destination, &resume_bundle)
            .map_err(|error| error.into_parts().1)?;
    assert_eq!(restored_plan.capture_identity(), capture_identity);
    assert_eq!(
        restored_plan.evaluation_capture_identity(),
        Some(evaluation_identity)
    );
    assert_eq!(restored_plan.captured_multi_step_lr(), Some(&schedule));
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let resumed = prepare(restored_plan)?;
    validate_preparation(&resumed);
    assert_eq!(
        resumed.checkpoint()?,
        decoded.optimizer_checkpoint().clone()
    );
    assert_eq!(
        resumed.checkpoint()?.info().dropout_block_counter(),
        Some(saved_dropout_cursor)
    );
    let mut resumed = resumed.into_training_session();
    let restored_checkpoint = resumed.checkpoint()?;
    assert_eq!(uninterrupted.checkpoint()?, restored_checkpoint);
    let expected_restore_probe =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(5)?)?;
    let actual_restore_probe = replay_prepared_policy_batch(&mut resumed, file_resume_batch(5)?)?;
    validate_step(&expected_restore_probe);
    validate_step(&actual_restore_probe);
    assert_file_resume_steps_match(&actual_restore_probe, &expected_restore_probe);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    uninterrupted.restore_checkpoint_in_place(&restored_checkpoint)?;
    resumed.restore_checkpoint_in_place(&restored_checkpoint)?;
    assert_eq!(resumed.checkpoint()?, restored_checkpoint);
    assert_eq!(uninterrupted.checkpoint()?, restored_checkpoint);
    for replay in 5..=LAST_REPLAY {
        let expected =
            replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(replay)?)?;
        let actual = replay_prepared_policy_batch(&mut resumed, file_resume_batch(replay)?)?;
        validate_step(&expected);
        validate_step(&actual);
        assert_file_resume_steps_match(&actual, &expected);
        assert_eq!(
            actual.clip_report().is_some(),
            actual.accumulation_index() == 0
        );
        assert_eq!(actual.window_loss_report().is_some(), actual.did_update());
        if let Some(report) = actual.window_loss_report() {
            assert!(matches!(replay, 6 | 9));
            assert!(report.is_finite());
            assert_eq!(report.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(report.loss_weight(), WINDOW_LOSS_WEIGHT);
        }
        let actual_checkpoint = resumed.checkpoint()?;
        let expected_checkpoint = uninterrupted.checkpoint()?;
        assert_eq!(
            actual_checkpoint.info().dropout_block_counter(),
            expected_checkpoint.info().dropout_block_counter()
        );
        assert_eq!(actual_checkpoint, expected_checkpoint);
    }
    let expected_partial =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(10)?)?;
    let actual_partial = replay_prepared_policy_batch(&mut resumed, file_resume_batch(10)?)?;
    validate_step(&expected_partial);
    validate_step(&actual_partial);
    assert_file_resume_steps_match(&actual_partial, &expected_partial);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_flush = uninterrupted.commit_partial_window_with_rate_policy()?;
    let actual_flush = resumed.commit_partial_window_with_rate_policy()?;
    assert_eq!(
        actual_flush.did_commit_window(),
        expected_flush.did_commit_window()
    );
    assert_eq!(actual_flush.committed_microbatches(), 1);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_discard =
        replay_prepared_policy_batch(&mut uninterrupted, file_resume_batch(11)?)?;
    let actual_discard = replay_prepared_policy_batch(&mut resumed, file_resume_batch(11)?)?;
    validate_step(&expected_discard);
    validate_step(&actual_discard);
    assert_file_resume_steps_match(&actual_discard, &expected_discard);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    let expected_reset = uninterrupted.reset_gradient_window()?;
    let actual_reset = resumed.reset_gradient_window()?;
    assert_eq!(actual_reset, expected_reset);
    assert_eq!(actual_reset.discarded_microbatches(), 1);
    assert_eq!(resumed.checkpoint()?, uninterrupted.checkpoint()?);
    assert_eq!(resumed.runtime().optimizer_step()?, 4);
    assert_eq!(resumed.runtime().accumulation_index()?, 0);
    for (_, parameter, _, before) in &destination_states {
        let after = parameter.snapshot()?;
        assert_eq!(after.data, before.data);
        assert_eq!(after.version, before.version);
        assert_eq!(after.identity, before.identity);
        assert_eq!(after.trainable, before.trainable);
    }

    let before_evaluation_checkpoint = resumed.checkpoint()?;
    let before_evaluation_counter = before_evaluation_checkpoint.info().dropout_block_counter();
    let before_evaluation_accumulators = resumed.runtime().gradient_accumulator_snapshots()?;
    let uninterrupted_final_loss =
        evaluate_mean_file_resume_loss(&mut uninterrupted, &mut validate_evaluation)?;
    let final_loss = evaluate_mean_file_resume_loss(&mut resumed, &mut validate_evaluation)?;
    assert_eq!(final_loss.to_bits(), uninterrupted_final_loss.to_bits());
    assert_eq!(resumed.checkpoint()?, before_evaluation_checkpoint);
    assert_eq!(
        resumed.checkpoint()?.info().dropout_block_counter(),
        before_evaluation_counter
    );
    assert_eq!(
        resumed.runtime().gradient_accumulator_snapshots()?,
        before_evaluation_accumulators
    );

    let (uninterrupted_model, uninterrupted_checkpoint) = uninterrupted
        .finish_with_module_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    let (resumed_model, resumed_checkpoint) = resumed
        .finish_with_checkpoint()
        .map_err(|error| error.into_parts().1)?;
    assert_eq!(
        &resumed_checkpoint,
        uninterrupted_checkpoint.optimizer_checkpoint()
    );
    assert_eq!(
        resumed_checkpoint.as_bytes(),
        uninterrupted_checkpoint.optimizer_checkpoint().as_bytes()
    );
    assert_eq!(
        resumed_model.state_dict()?,
        uninterrupted_model.state_dict()?
    );
    let mut expected_states = Vec::new();
    uninterrupted_model.visit("", &mut |name, parameter, kind| {
        expected_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the finished reference state remains readable"),
        ));
    });
    let mut resumed_states = Vec::new();
    resumed_model.visit("", &mut |name, parameter, kind| {
        resumed_states.push((
            name,
            kind,
            parameter
                .snapshot()
                .expect("the finished destination state remains readable"),
        ));
    });
    assert_eq!(resumed_states.len(), destination_states.len());
    assert_eq!(resumed_states.len(), expected_states.len());
    for (
        ((name, _, kind, before), (expected_name, expected_kind, expected)),
        (actual_name, actual_kind, actual),
    ) in destination_states
        .iter()
        .zip(&expected_states)
        .zip(&resumed_states)
    {
        assert_eq!(actual_name, name);
        assert_eq!(expected_name, name);
        assert_eq!(actual_kind, kind);
        assert_eq!(expected_kind, kind);
        assert_eq!(actual.data, expected.data);
        assert_eq!(actual.identity, before.identity);
        assert_eq!(actual.trainable, before.trainable);
        assert_eq!(
            actual.version,
            before
                .version
                .checked_add(1)
                .expect("successful publication cannot overflow a version")
        );
    }
    assert_eq!(resumed_model.tokens.weight.id(), destination_tied_identity);
    assert_eq!(tied_alias.id(), destination_tied_identity);
    assert_eq!(tied_alias.value()?, resumed_model.tokens.weight.value()?);
    assert!(resumed_model.positions.weight.is_trainable());
    assert!(!resumed_model.frozen_scale.is_trainable());
    assert!(!resumed_model.running_marker.is_trainable());
    assert_eq!(
        resumed_model.frozen_scale.value()?,
        uninterrupted_model.frozen_scale.value()?
    );
    assert_eq!(
        resumed_model.running_marker.value()?,
        uninterrupted_model.running_marker.value()?
    );

    assert!(
        final_loss < initial_loss,
        "file-resumed two-block causal Transformer loss did not decrease: {initial_loss} -> {final_loss}"
    );
    println!(
        "{target_name}: capture={capture_identity:016x}, checkpoint=(replay=4, optimizer=1, accumulation=1), optimizer_steps=4, eval_mean_sparse_loss={initial_loss:.6} -> {final_loss:.6}, exact_resume=true, different_init=true, published=true"
    );
    Ok(())
}

#[cfg(test)]
mod cross_process_evidence_tests {
    use super::*;

    fn checkpoint(replay_step: u64) -> CrossProcessCheckpointEvidence {
        CrossProcessCheckpointEvidence {
            capture_identity: 11,
            replay_step,
            optimizer_step: 1,
            gradient_accumulation_steps: 3,
            accumulation_index: 1,
            discarded_microbatches: 0,
            flushed_window_count: 0,
            flushed_microbatch_count: 0,
            flush_capture_identity: Some(12),
            dropout_block_counter: Some(48),
            accumulated_token_count: Some(5),
            accumulated_loss_numerator_bits: Some(1.25_f32.to_bits()),
            window_loss_report_enabled: true,
            reset_transition_count: 0,
            reset_capture_identity: Some(13),
            accumulation_capture_identity: Some(14),
        }
    }

    fn evidence() -> CrossProcessResumeEvidence {
        CrossProcessResumeEvidence {
            format_version: CROSS_PROCESS_EVIDENCE_VERSION,
            backend: CrossProcessBackend::Cpu,
            pending_bundle_bytes: 3,
            pending_bundle_checksum: cross_process_checksum(b"one"),
            terminal_checkpoint_bytes: 3,
            terminal_checkpoint_checksum: cross_process_checksum(b"two"),
            evaluation_capture_identity: 15,
            initial_evaluation_loss_bits: 2.0_f64.to_bits(),
            terminal_evaluation_loss_bits: 1.0_f64.to_bits(),
            pending: checkpoint(4),
            terminal: checkpoint(11),
        }
    }

    #[test]
    fn cross_process_evidence_round_trips_canonically() {
        let expected = evidence();
        let bytes = expected.canonical_bytes().unwrap();
        assert_eq!(
            CrossProcessResumeEvidence::from_canonical_bytes(&bytes).unwrap(),
            expected
        );
    }

    #[test]
    fn cross_process_evidence_rejects_unknown_version_and_fields() {
        let mut unsupported = evidence();
        unsupported.format_version += 1;
        let bytes = unsupported.canonical_bytes().unwrap();
        assert!(CrossProcessResumeEvidence::from_canonical_bytes(&bytes).is_err());

        let mut value = serde_json::to_value(evidence()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::Value::Bool(true));
        let mut bytes = serde_json::to_vec_pretty(&value).unwrap();
        bytes.push(b'\n');
        assert!(CrossProcessResumeEvidence::from_canonical_bytes(&bytes).is_err());
    }

    #[test]
    fn cross_process_evidence_rejects_backend_mismatch() {
        assert!(
            evidence()
                .validate_backend(CrossProcessBackend::NativeCpu)
                .is_err()
        );
    }

    #[test]
    fn cross_process_evidence_rejects_checksum_mismatch() {
        let bytes = b"one";
        assert!(
            validate_cross_process_file_binding(
                "pending bundle",
                bytes,
                u64::try_from(bytes.len()).unwrap(),
                cross_process_checksum(bytes) ^ 1,
            )
            .is_err()
        );
    }
}
