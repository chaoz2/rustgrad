use super::*;
use rustgrad::{BenchmarkDuration, NativeCpuReplayTraffic, NativeCpuRunReport};
use std::{
    fs::{self, OpenOptions},
    io::Write,
};

const FORMAT_VERSION: u32 = 1;
const EVIDENCE_KIND: &str = "native_cpu_compiled_transformer_steady_replay";
const MAX_EVIDENCE_BYTES: usize = 1 << 20;
const WARMUP_WINDOWS: u64 = 1;
const MEASURED_WINDOWS: u64 = 32;
const WARMUP_REPLAYS: u64 = WARMUP_WINDOWS * ACCUMULATION_STEPS;
const MEASURED_REPLAYS: u64 = MEASURED_WINDOWS * ACCUMULATION_STEPS;
const VALID_TOKEN_COUNTS: [u64; ACCUMULATION_STEPS as usize] = [5, 3, 3];

pub(super) struct SteadyMeasurementRequest {
    path: PathBuf,
    git_sha: String,
    cargo_profile: String,
}

impl SteadyMeasurementRequest {
    pub(super) fn from_arguments(
        arguments: &mut impl Iterator<Item = String>,
    ) -> std::result::Result<Option<Self>, Box<dyn Error>> {
        let Some(flag) = arguments.next() else {
            return Ok(None);
        };
        if flag != "--steady-evidence" {
            return Err(format!(
                "unexpected native CPU scoreboard argument {flag:?}; expected `--steady-evidence <path> <git-sha> <dev|release>`"
            )
            .into());
        }
        let path = arguments
            .next()
            .ok_or("--steady-evidence requires an output path")?;
        let git_sha = arguments
            .next()
            .ok_or("--steady-evidence requires a Git SHA")?;
        let cargo_profile = arguments
            .next()
            .ok_or("--steady-evidence requires a Cargo profile")?;
        if let Some(extra) = arguments.next() {
            return Err(format!("unexpected native CPU scoreboard argument {extra:?}").into());
        }
        if git_sha.len() != 40
            || !git_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("steady evidence requires a lowercase full Git SHA".into());
        }
        validate_build_profile(&cargo_profile)?;
        Ok(Some(Self {
            path: PathBuf::from(path),
            git_sha,
            cargo_profile,
        }))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ReleaseProvenance {
    git_sha: String,
    cargo_profile: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointProgressEvidence {
    capture_identity: u64,
    accumulation_capture_identity: Option<u64>,
    replay_step: u64,
    optimizer_step: u64,
    accumulation_index: u64,
    dropout_block_counter: Option<u64>,
}

impl CheckpointProgressEvidence {
    fn from_checkpoint(checkpoint: &CompiledAdamWCheckpoint) -> Self {
        let info = checkpoint.info();
        Self {
            capture_identity: info.capture_identity(),
            accumulation_capture_identity: info.accumulation_capture_identity(),
            replay_step: info.replay_step(),
            optimizer_step: info.optimizer_step(),
            accumulation_index: info.accumulation_index(),
            dropout_block_counter: info.dropout_block_counter(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CapturedScheduleEvidence {
    base_bits: u32,
    gamma_bits: u32,
    milestones: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasurementPreparationEvidence {
    cache_scope: String,
    wall_time: BenchmarkDuration,
    render_capsule_hit_count: u64,
    render_capsule_miss_count: u64,
    local_render_job_count: u64,
    max_parallel_render_job_count: u64,
    compiler_process_count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayTrafficEvidence {
    external_input_import_count: u64,
    external_input_import_bytes: u64,
    borrowed_recurrent_input_bytes: u64,
    borrowed_recurrent_output_bytes: u64,
    retained_recurrent_state_count: u64,
    retained_recurrent_state_bytes: u64,
    replaced_recurrent_state_count: u64,
    replaced_recurrent_state_bytes: u64,
    materialized_egress_count: u64,
    materialized_egress_bytes: u64,
}

impl From<&NativeCpuReplayTraffic> for ReplayTrafficEvidence {
    fn from(traffic: &NativeCpuReplayTraffic) -> Self {
        Self {
            external_input_import_count: traffic.external_input_import_count(),
            external_input_import_bytes: traffic.external_input_import_bytes(),
            borrowed_recurrent_input_bytes: traffic.borrowed_recurrent_input_bytes(),
            borrowed_recurrent_output_bytes: traffic.borrowed_recurrent_output_bytes(),
            retained_recurrent_state_count: traffic.retained_recurrent_state_count(),
            retained_recurrent_state_bytes: traffic.retained_recurrent_state_bytes(),
            replaced_recurrent_state_count: traffic.replaced_recurrent_state_count(),
            replaced_recurrent_state_bytes: traffic.replaced_recurrent_state_bytes(),
            materialized_egress_count: traffic.materialized_egress_count(),
            materialized_egress_bytes: traffic.materialized_egress_bytes(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ProgramReplayEvidence {
    capture_identity: u64,
    native_identity: u64,
    vectorized: bool,
    native_item_count: u64,
    executed_native_item_count: u64,
    module_dispatch_count: u64,
    module_dispatched_native_item_count: u64,
    schedule_cache_keys: Vec<u64>,
    traffic: ReplayTrafficEvidence,
}

impl ProgramReplayEvidence {
    fn from_report(report: &NativeCpuRunReport) -> std::result::Result<Self, Box<dyn Error>> {
        Ok(Self {
            capture_identity: report.capture_identity(),
            native_identity: report.native_identity(),
            vectorized: report.is_vectorized(),
            native_item_count: u64::try_from(report.native_item_count())?,
            executed_native_item_count: u64::try_from(report.executed_native_item_count())?,
            module_dispatch_count: u64::try_from(report.module_dispatch_count())?,
            module_dispatched_native_item_count: u64::try_from(
                report.module_dispatched_native_item_count(),
            )?,
            schedule_cache_keys: report.schedule_cache_keys().to_vec(),
            traffic: report.traffic().into(),
        })
    }

    fn validate(&self, expected_capture_identity: u64) -> std::result::Result<(), Box<dyn Error>> {
        if self.capture_identity != expected_capture_identity
            || !self.vectorized
            || self.native_item_count == 0
            || self.executed_native_item_count == 0
            || self.executed_native_item_count > self.native_item_count
            || self.module_dispatch_count == 0
            || self.module_dispatched_native_item_count != self.executed_native_item_count
            || u64::try_from(self.schedule_cache_keys.len())? != self.native_item_count
            || self.traffic.external_input_import_count != 0
            || self.traffic.external_input_import_bytes != 0
        {
            return Err("steady replay program evidence differs".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayTimingSample {
    measurement_ordinal: u64,
    window_ordinal: u64,
    microbatch_ordinal: u64,
    replay_step: u64,
    successful_invocation: u64,
    valid_token_count: u64,
    total_wall_time: BenchmarkDuration,
    executor_wall_time: BenchmarkDuration,
    native_dispatcher_wall_time: BenchmarkDuration,
    executor_host_wall_time: BenchmarkDuration,
    recurrent_overhead_wall_time: BenchmarkDuration,
}

impl ReplayTimingSample {
    fn from_report(
        report: &NativeCpuRunReport,
        measurement_ordinal: u64,
        window_ordinal: u64,
        microbatch_ordinal: u64,
        replay_step: u64,
        valid_token_count: u64,
    ) -> std::result::Result<Self, Box<dyn Error>> {
        let executor = report.executor_wall_time();
        let dispatcher = report.native_dispatcher_wall_time();
        let executor_host = report.executor_host_wall_time();
        let overhead = report.replay_overhead_wall_time();
        let total = report.wall_time();
        if dispatcher.checked_add(executor_host) != Some(executor)
            || executor.checked_add(overhead) != Some(total)
        {
            return Err("steady replay timing partitions do not match".into());
        }
        Ok(Self {
            measurement_ordinal,
            window_ordinal,
            microbatch_ordinal,
            replay_step,
            successful_invocation: report.successful_invocation(),
            valid_token_count,
            total_wall_time: BenchmarkDuration::from_duration(total),
            executor_wall_time: BenchmarkDuration::from_duration(executor),
            native_dispatcher_wall_time: BenchmarkDuration::from_duration(dispatcher),
            executor_host_wall_time: BenchmarkDuration::from_duration(executor_host),
            recurrent_overhead_wall_time: BenchmarkDuration::from_duration(overhead),
        })
    }

    fn validate(
        &self,
        expected_measurement_ordinal: u64,
        expected_replay_step: u64,
        expected_successful_invocation: u64,
        expected_window_ordinal: u64,
        expected_microbatch_ordinal: u64,
        expected_valid_token_count: u64,
    ) -> std::result::Result<(), Box<dyn Error>> {
        if self.measurement_ordinal != expected_measurement_ordinal
            || self.replay_step != expected_replay_step
            || self.successful_invocation != expected_successful_invocation
            || self.window_ordinal != expected_window_ordinal
            || self.microbatch_ordinal != expected_microbatch_ordinal
            || self.valid_token_count != expected_valid_token_count
        {
            return Err("steady replay sample order differs".into());
        }
        let total = self.total_wall_time.to_duration()?;
        let executor = self.executor_wall_time.to_duration()?;
        let dispatcher = self.native_dispatcher_wall_time.to_duration()?;
        let executor_host = self.executor_host_wall_time.to_duration()?;
        let overhead = self.recurrent_overhead_wall_time.to_duration()?;
        if dispatcher.checked_add(executor_host) != Some(executor)
            || executor.checked_add(overhead) != Some(total)
        {
            return Err("steady replay sample timing partitions do not match".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeCpuSteadyTrainingEvidence {
    format_version: u32,
    evidence_kind: String,
    provenance: ReleaseProvenance,
    warmup_window_count: u64,
    warmup_replay_count: u64,
    measured_window_count: u64,
    measured_replay_count: u64,
    validated_replay_count: u64,
    validated_valid_token_count: u64,
    measured_valid_token_count: u64,
    compiled_graph_build_count: u64,
    dropout_blocks_per_replay: u64,
    final_checkpoint_byte_count: u64,
    preparation: MeasurementPreparationEvidence,
    captured_schedule: CapturedScheduleEvidence,
    starting_checkpoint: CheckpointProgressEvidence,
    warmup_checkpoint: CheckpointProgressEvidence,
    final_checkpoint: CheckpointProgressEvidence,
    accumulation_only_program: ProgramReplayEvidence,
    optimizer_commit_program: ProgramReplayEvidence,
    accumulation_only_samples: Vec<ReplayTimingSample>,
    optimizer_commit_samples: Vec<ReplayTimingSample>,
}

impl NativeCpuSteadyTrainingEvidence {
    fn validate(&self) -> std::result::Result<(), Box<dyn Error>> {
        if self.format_version != FORMAT_VERSION || self.evidence_kind != EVIDENCE_KIND {
            return Err("unsupported steady training evidence format".into());
        }
        if self.provenance.git_sha.len() != 40
            || !self
                .provenance
                .git_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("invalid steady training evidence provenance".into());
        }
        validate_build_profile(&self.provenance.cargo_profile)?;
        let start_dropout = self
            .starting_checkpoint
            .dropout_block_counter
            .ok_or("steady starting dropout counter is absent")?;
        let warmup_dropout = self
            .warmup_checkpoint
            .dropout_block_counter
            .ok_or("steady warmup dropout counter is absent")?;
        let final_dropout = self
            .final_checkpoint
            .dropout_block_counter
            .ok_or("steady final dropout counter is absent")?;
        let warmup_dropout_delta = warmup_dropout
            .checked_sub(start_dropout)
            .ok_or("steady warmup dropout counter regressed")?;
        if self.dropout_blocks_per_replay == 0
            || warmup_dropout_delta
                != self
                    .dropout_blocks_per_replay
                    .checked_mul(WARMUP_REPLAYS)
                    .ok_or("steady warmup dropout progression overflows")?
        {
            return Err("steady warmup dropout progression differs".into());
        }
        let expected_final_dropout = warmup_dropout
            .checked_add(
                self.dropout_blocks_per_replay
                    .checked_mul(MEASURED_REPLAYS)
                    .ok_or("steady measured dropout progression overflows")?,
            )
            .ok_or("steady final dropout counter overflows")?;
        if self.warmup_window_count != WARMUP_WINDOWS
            || self.warmup_replay_count != WARMUP_REPLAYS
            || self.measured_window_count != MEASURED_WINDOWS
            || self.measured_replay_count != MEASURED_REPLAYS
            || self.validated_replay_count != WARMUP_REPLAYS + MEASURED_REPLAYS
            || self.validated_valid_token_count
                != (WARMUP_WINDOWS + MEASURED_WINDOWS)
                    * VALID_TOKEN_COUNTS.iter().copied().sum::<u64>()
            || self.measured_valid_token_count
                != MEASURED_WINDOWS * VALID_TOKEN_COUNTS.iter().copied().sum::<u64>()
            || self.compiled_graph_build_count != 1
            || self.final_checkpoint_byte_count == 0
            || self.starting_checkpoint.replay_step != ACCUMULATION_STEPS
            || self.starting_checkpoint.optimizer_step != 1
            || self.starting_checkpoint.accumulation_index != 0
            || self.warmup_checkpoint.capture_identity != self.starting_checkpoint.capture_identity
            || self.warmup_checkpoint.accumulation_capture_identity
                != self.starting_checkpoint.accumulation_capture_identity
            || self.warmup_checkpoint.replay_step
                != self.starting_checkpoint.replay_step + WARMUP_REPLAYS
            || self.warmup_checkpoint.optimizer_step
                != self.starting_checkpoint.optimizer_step + WARMUP_WINDOWS
            || self.warmup_checkpoint.accumulation_index != 0
            || self.final_checkpoint.capture_identity != self.starting_checkpoint.capture_identity
            || self.final_checkpoint.accumulation_capture_identity
                != self.starting_checkpoint.accumulation_capture_identity
            || self.final_checkpoint.replay_step
                != self
                    .starting_checkpoint
                    .replay_step
                    .checked_add(self.validated_replay_count)
                    .ok_or("steady replay step overflows")?
            || self.final_checkpoint.optimizer_step
                != self
                    .starting_checkpoint
                    .optimizer_step
                    .checked_add(WARMUP_WINDOWS + MEASURED_WINDOWS)
                    .ok_or("steady optimizer step overflows")?
            || self.final_checkpoint.accumulation_index != 0
            || final_dropout != expected_final_dropout
        {
            return Err("steady training progress evidence differs".into());
        }
        if self.preparation.cache_scope != "same_process_warm_cache"
            || self.preparation.render_capsule_hit_count != 4
            || self.preparation.render_capsule_miss_count != 0
            || self.preparation.local_render_job_count != 0
            || self.preparation.max_parallel_render_job_count != 0
            || self.preparation.compiler_process_count != 0
            || self.captured_schedule.base_bits != 0.05f32.to_bits()
            || self.captured_schedule.gamma_bits != 0.5f32.to_bits()
            || self.captured_schedule.milestones != [1]
        {
            return Err("steady preparation or schedule evidence differs".into());
        }
        self.preparation.wall_time.to_duration()?;
        self.optimizer_commit_program
            .validate(self.starting_checkpoint.capture_identity)?;
        self.accumulation_only_program.validate(
            self.starting_checkpoint
                .accumulation_capture_identity
                .ok_or("steady accumulation capture identity is absent")?,
        )?;
        if self.accumulation_only_program.capture_identity
            == self.optimizer_commit_program.capture_identity
            || self.accumulation_only_program.native_identity
                == self.optimizer_commit_program.native_identity
        {
            return Err("steady replay program evidence differs".into());
        }

        let mut accumulation = self.accumulation_only_samples.iter();
        let mut commits = self.optimizer_commit_samples.iter();
        for measurement_ordinal in 1..=MEASURED_REPLAYS {
            let replay_step = self
                .starting_checkpoint
                .replay_step
                .checked_add(WARMUP_REPLAYS)
                .and_then(|step| step.checked_add(measurement_ordinal))
                .ok_or("steady measured replay step overflows")?;
            let successful_invocation = WARMUP_REPLAYS
                .checked_add(measurement_ordinal)
                .ok_or("steady invocation count overflows")?;
            let window_ordinal = (measurement_ordinal - 1) / ACCUMULATION_STEPS + 1;
            let microbatch_ordinal = (measurement_ordinal - 1) % ACCUMULATION_STEPS + 1;
            let valid_token_count = VALID_TOKEN_COUNTS[(microbatch_ordinal - 1) as usize];
            let sample = if microbatch_ordinal == ACCUMULATION_STEPS {
                commits.next()
            } else {
                accumulation.next()
            }
            .ok_or("steady replay sample is absent")?;
            sample.validate(
                measurement_ordinal,
                replay_step,
                successful_invocation,
                window_ordinal,
                microbatch_ordinal,
                valid_token_count,
            )?;
        }
        if accumulation.next().is_some() || commits.next().is_some() {
            return Err("steady replay sample inventory differs".into());
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> std::result::Result<Vec<u8>, Box<dyn Error>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        if bytes.len() > MAX_EVIDENCE_BYTES {
            return Err("steady training evidence exceeds its byte bound".into());
        }
        Ok(bytes)
    }
}

pub(super) fn record(
    plan: &CompiledAdamWPlan,
    checkpoint: &CompiledAdamWCheckpoint,
    target: &NativeCpuSessionTarget<'_>,
    compile_wall_time: Duration,
    compiled_graph_build_count: &Cell<usize>,
    request: SteadyMeasurementRequest,
) -> std::result::Result<(), Box<dyn Error>> {
    validate_new_destination(&request.path)?;
    let starting_checkpoint = CheckpointProgressEvidence::from_checkpoint(checkpoint);
    let restored = plan.restore_checkpoint(checkpoint)?;
    let inspection = restored.inspection()?;
    if inspection.initial_replay_step() != starting_checkpoint.replay_step {
        return Err("steady restored inspection starts at the wrong replay".into());
    }
    let prepare_started = Instant::now();
    let mut session = restored.prepare(target)?;
    let prepare_wall_time = prepare_started.elapsed();
    let mut validator = NativeTrainingScoreboard::new(
        inspection,
        session.preparation_report(),
        compile_wall_time,
        prepare_wall_time,
    )?;
    let preparation_report = session.preparation_report();
    let preparation = MeasurementPreparationEvidence {
        cache_scope: "same_process_warm_cache".into(),
        wall_time: BenchmarkDuration::from_duration(prepare_wall_time),
        render_capsule_hit_count: u64::try_from(preparation_report.render_capsule_hit_count())?,
        render_capsule_miss_count: u64::try_from(preparation_report.render_capsule_miss_count())?,
        local_render_job_count: u64::try_from(preparation_report.local_render_job_count())?,
        max_parallel_render_job_count: u64::try_from(
            preparation_report.max_parallel_render_job_count(),
        )?,
        compiler_process_count: u64::try_from(preparation_report.compiler_process_count())?,
    };
    let dropout_blocks_per_replay = plan
        .dropout_blocks_per_replay()
        .ok_or("steady measurement requires captured dropout")?;
    let schedule = session
        .captured_multi_step_lr()
        .ok_or("steady measurement requires captured MultiStep learning rate")?;
    let captured_schedule = CapturedScheduleEvidence {
        base_bits: schedule.base().to_bits(),
        gamma_bits: schedule.gamma().to_bits(),
        milestones: schedule.milestones().to_vec(),
    };

    let mut accumulation_only_program = None;
    let mut optimizer_commit_program = None;
    let mut accumulation_only_samples = Vec::with_capacity((MEASURED_WINDOWS * 2) as usize);
    let mut optimizer_commit_samples = Vec::with_capacity(MEASURED_WINDOWS as usize);
    let mut warmup_checkpoint = None;
    for invocation in 1..=WARMUP_REPLAYS + MEASURED_REPLAYS {
        let replay_step = starting_checkpoint
            .replay_step
            .checked_add(invocation)
            .ok_or("steady replay step overflows")?;
        let microbatch_ordinal = (invocation - 1) % ACCUMULATION_STEPS + 1;
        let valid_token_count = VALID_TOKEN_COUNTS[(microbatch_ordinal - 1) as usize];
        let batch = masked_batch(replay_step)?;
        if loss_mask_weight(&batch.loss_mask) != valid_token_count {
            return Err("steady replay valid-token count differs".into());
        }
        let step = session.step_batch_commit_only_scheduled(batch)?;
        let did_update = microbatch_ordinal == ACCUMULATION_STEPS;
        if step.step() != replay_step
            || step.optimizer_step()
                != starting_checkpoint.optimizer_step + invocation / ACCUMULATION_STEPS
            || step.accumulation_index() != invocation % ACCUMULATION_STEPS
            || step.loss_weight() != valid_token_count
            || step.did_update() != did_update
            || !step.outputs().is_empty()
            || !step.loss().values()[0].is_finite()
            || step.clip_report().is_some() != did_update
            || step.window_loss_report().is_some() != did_update
            || step.report().successful_invocation() != invocation
            || step.report().fallback_count() != 0
        {
            return Err("steady replay result differs".into());
        }
        if let Some(window) = step.window_loss_report()
            && (!window.is_finite()
                || window.microbatch_count() != ACCUMULATION_STEPS
                || window.loss_weight() != VALID_TOKEN_COUNTS.iter().copied().sum::<u64>())
        {
            return Err("steady replay window report differs".into());
        }
        validator.record_step(&step)?;
        let program = ProgramReplayEvidence::from_report(step.report())?;
        let stable_program = if did_update {
            &mut optimizer_commit_program
        } else {
            &mut accumulation_only_program
        };
        if let Some(expected) = stable_program.as_ref() {
            if expected != &program {
                return Err("steady replay program identity or traffic changed".into());
            }
        } else {
            *stable_program = Some(program);
        }

        if invocation == WARMUP_REPLAYS {
            warmup_checkpoint = Some(CheckpointProgressEvidence::from_checkpoint(
                &session.checkpoint()?,
            ));
        }

        if invocation > WARMUP_REPLAYS {
            let measurement_ordinal = invocation - WARMUP_REPLAYS;
            let window_ordinal = (measurement_ordinal - 1) / ACCUMULATION_STEPS + 1;
            let sample = ReplayTimingSample::from_report(
                step.report(),
                measurement_ordinal,
                window_ordinal,
                microbatch_ordinal,
                replay_step,
                valid_token_count,
            )?;
            if did_update {
                optimizer_commit_samples.push(sample);
            } else {
                accumulation_only_samples.push(sample);
            }
        }
    }

    let checkpoint_started = Instant::now();
    let final_checkpoint = session.checkpoint()?;
    let checkpoint_wall_time = checkpoint_started.elapsed();
    validator.observe_checkpoint(&final_checkpoint, checkpoint_wall_time)?;
    let validation_report = validator.report()?;
    if validation_report.successful_replay_count() != WARMUP_REPLAYS + MEASURED_REPLAYS {
        return Err("steady validator replay count differs".into());
    }
    let phases = validation_report
        .step_phases()
        .ok_or("steady validator phase report is absent")?;
    if phases.first().phase() != rustgrad::NativeTrainingStepPhase::AccumulationOnly
        || phases
            .warm_accumulation_only()
            .map(|phase| phase.sample_count())
            != Some((WARMUP_WINDOWS + MEASURED_WINDOWS) * (ACCUMULATION_STEPS - 1) - 1)
        || phases
            .warm_optimizer_commit()
            .map(|phase| phase.sample_count())
            != Some(WARMUP_WINDOWS + MEASURED_WINDOWS)
    {
        return Err("steady validator phase counts differ".into());
    }

    let evidence = NativeCpuSteadyTrainingEvidence {
        format_version: FORMAT_VERSION,
        evidence_kind: EVIDENCE_KIND.into(),
        provenance: ReleaseProvenance {
            git_sha: request.git_sha,
            cargo_profile: request.cargo_profile,
        },
        warmup_window_count: WARMUP_WINDOWS,
        warmup_replay_count: WARMUP_REPLAYS,
        measured_window_count: MEASURED_WINDOWS,
        measured_replay_count: MEASURED_REPLAYS,
        validated_replay_count: WARMUP_REPLAYS + MEASURED_REPLAYS,
        validated_valid_token_count: (WARMUP_WINDOWS + MEASURED_WINDOWS)
            * VALID_TOKEN_COUNTS.iter().copied().sum::<u64>(),
        measured_valid_token_count: MEASURED_WINDOWS
            * VALID_TOKEN_COUNTS.iter().copied().sum::<u64>(),
        compiled_graph_build_count: u64::try_from(compiled_graph_build_count.get())?,
        dropout_blocks_per_replay,
        final_checkpoint_byte_count: validation_report
            .checkpoint_byte_count()
            .ok_or("steady validator checkpoint evidence is absent")?,
        preparation,
        captured_schedule,
        starting_checkpoint,
        warmup_checkpoint: warmup_checkpoint.ok_or("steady warmup checkpoint is absent")?,
        final_checkpoint: CheckpointProgressEvidence::from_checkpoint(&final_checkpoint),
        accumulation_only_program: accumulation_only_program
            .ok_or("steady accumulation program evidence is absent")?,
        optimizer_commit_program: optimizer_commit_program
            .ok_or("steady commit program evidence is absent")?,
        accumulation_only_samples,
        optimizer_commit_samples,
    };
    let bytes = evidence.canonical_bytes()?;
    let decoded: NativeCpuSteadyTrainingEvidence = serde_json::from_slice(&bytes)?;
    decoded.validate()?;
    if decoded.canonical_bytes()? != bytes {
        return Err("steady training evidence serialization is not deterministic".into());
    }
    write_new_evidence(&request.path, &bytes)?;
    Ok(())
}

fn validate_build_profile(profile: &str) -> std::result::Result<(), Box<dyn Error>> {
    // Protected evidence invokes only Cargo's default `dev` profile or
    // `--release`; bind those fixed invocations to their binary assertions.
    if !matches!(profile, "dev" | "release") || (profile == "dev") != cfg!(debug_assertions) {
        return Err("steady evidence Cargo profile does not match the binary".into());
    }
    Ok(())
}

fn validate_new_destination(path: &Path) -> std::result::Result<(), Box<dyn Error>> {
    match fs::symlink_metadata(path) {
        Ok(_) => return Err("steady evidence output already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err("steady evidence output requires an existing parent directory".into());
    }
    Ok(())
}

fn write_new_evidence(path: &Path, bytes: &[u8]) -> std::result::Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let filename = path
        .file_name()
        .ok_or("steady evidence output requires a filename")?;
    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let process = std::process::id();
    let mut stage_path = None;
    let mut stage_file = None;
    for attempt in 0..16u8 {
        let candidate = parent.join(format!(
            ".{}.rustgrad-stage-{process}-{unique}-{attempt}",
            filename.to_string_lossy()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                stage_path = Some(candidate);
                stage_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    let stage_path = stage_path.ok_or("unable to create unique steady evidence staging file")?;
    let mut stage_file = stage_file.expect("stage path and file are created together");
    let publish = (|| -> std::io::Result<()> {
        stage_file.write_all(bytes)?;
        stage_file.sync_all()?;
        drop(stage_file);
        fs::hard_link(&stage_path, path)?;
        fs::remove_file(&stage_path)
    })();
    if publish.is_err() {
        let _ = fs::remove_file(&stage_path);
    }
    publish?;
    Ok(())
}

#[cfg(test)]
mod tests;
