use super::*;

fn zero_duration() -> BenchmarkDuration {
    BenchmarkDuration::from_duration(Duration::ZERO)
}

fn cargo_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    }
}

fn sample(measurement_ordinal: u64) -> ReplayTimingSample {
    let replay_step = ACCUMULATION_STEPS + WARMUP_REPLAYS + measurement_ordinal;
    let microbatch_ordinal = (measurement_ordinal - 1) % ACCUMULATION_STEPS + 1;
    ReplayTimingSample {
        measurement_ordinal,
        window_ordinal: (measurement_ordinal - 1) / ACCUMULATION_STEPS + 1,
        microbatch_ordinal,
        replay_step,
        successful_invocation: WARMUP_REPLAYS + measurement_ordinal,
        valid_token_count: VALID_TOKEN_COUNTS[(microbatch_ordinal - 1) as usize],
        total_wall_time: zero_duration(),
        executor_wall_time: zero_duration(),
        native_dispatcher_wall_time: zero_duration(),
        executor_host_wall_time: zero_duration(),
        recurrent_overhead_wall_time: zero_duration(),
    }
}

fn program(capture_identity: u64) -> ProgramReplayEvidence {
    ProgramReplayEvidence {
        capture_identity,
        native_identity: capture_identity + 10,
        vectorized: true,
        native_item_count: 2,
        executed_native_item_count: 1,
        module_dispatch_count: 1,
        module_dispatched_native_item_count: 1,
        schedule_cache_keys: vec![capture_identity, capture_identity + 1],
        traffic: ReplayTrafficEvidence {
            external_input_import_count: 0,
            external_input_import_bytes: 0,
            borrowed_recurrent_input_bytes: 4,
            borrowed_recurrent_output_bytes: 4,
            retained_recurrent_state_count: 0,
            retained_recurrent_state_bytes: 0,
            replaced_recurrent_state_count: 1,
            replaced_recurrent_state_bytes: 4,
            materialized_egress_count: 1,
            materialized_egress_bytes: 4,
        },
    }
}

fn evidence() -> NativeCpuSteadyTrainingEvidence {
    let samples = (1..=MEASURED_REPLAYS).map(sample).collect::<Vec<_>>();
    NativeCpuSteadyTrainingEvidence {
        format_version: FORMAT_VERSION,
        evidence_kind: EVIDENCE_KIND.into(),
        provenance: ReleaseProvenance {
            git_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            cargo_profile: cargo_profile().into(),
        },
        warmup_window_count: WARMUP_WINDOWS,
        warmup_replay_count: WARMUP_REPLAYS,
        measured_window_count: MEASURED_WINDOWS,
        measured_replay_count: MEASURED_REPLAYS,
        validated_replay_count: WARMUP_REPLAYS + MEASURED_REPLAYS,
        validated_valid_token_count: 363,
        measured_valid_token_count: 352,
        compiled_graph_build_count: 1,
        dropout_blocks_per_replay: 8,
        final_checkpoint_byte_count: 1024,
        starting_checkpoint: checkpoint(3, 1, 10),
        warmup_checkpoint: checkpoint(6, 2, 34),
        final_checkpoint: checkpoint(102, 34, 802),
        preparation: MeasurementPreparationEvidence {
            cache_scope: "same_process_warm_cache".into(),
            wall_time: zero_duration(),
            render_capsule_hit_count: 4,
            render_capsule_miss_count: 0,
            local_render_job_count: 0,
            max_parallel_render_job_count: 0,
            compiler_process_count: 0,
        },
        captured_schedule: CapturedScheduleEvidence {
            base_bits: 0.05f32.to_bits(),
            gamma_bits: 0.5f32.to_bits(),
            milestones: vec![1],
        },
        accumulation_only_program: program(2),
        optimizer_commit_program: program(1),
        accumulation_only_samples: samples
            .iter()
            .copied()
            .filter(|sample| sample.microbatch_ordinal != ACCUMULATION_STEPS)
            .collect(),
        optimizer_commit_samples: samples
            .into_iter()
            .filter(|sample| sample.microbatch_ordinal == ACCUMULATION_STEPS)
            .collect(),
    }
}

fn checkpoint(
    replay_step: u64,
    optimizer_step: u64,
    dropout_block_counter: u64,
) -> CheckpointProgressEvidence {
    CheckpointProgressEvidence {
        capture_identity: 1,
        accumulation_capture_identity: Some(2),
        replay_step,
        optimizer_step,
        accumulation_index: 0,
        dropout_block_counter: Some(dropout_block_counter),
    }
}

#[test]
fn bounded_measurement_inventory_is_canonical_and_phase_separated() {
    let evidence = evidence();
    evidence.validate().unwrap();
    assert_eq!(evidence.accumulation_only_samples.len(), 64);
    assert_eq!(evidence.optimizer_commit_samples.len(), 32);
    assert_eq!(evidence.accumulation_only_samples[0].replay_step, 7);
    assert_eq!(evidence.optimizer_commit_samples[0].replay_step, 9);
    assert_eq!(evidence.final_checkpoint.replay_step, 102);
    assert_eq!(evidence.final_checkpoint.optimizer_step, 34);
    let bytes = evidence.canonical_bytes().unwrap();
    let decoded = serde_json::from_slice::<NativeCpuSteadyTrainingEvidence>(&bytes).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.canonical_bytes().unwrap(), bytes);
    assert_eq!(decoded, evidence);
}

#[test]
fn bounded_measurement_rejects_sample_or_progress_tampering() {
    let mut reordered = evidence();
    reordered.accumulation_only_samples.swap(0, 1);
    assert!(reordered.validate().is_err());

    let mut wrong_phase_count = evidence();
    wrong_phase_count.optimizer_commit_samples.pop();
    assert!(wrong_phase_count.validate().is_err());

    let mut wrong_frontier = evidence();
    wrong_frontier.final_checkpoint.accumulation_index = 1;
    assert!(wrong_frontier.validate().is_err());

    let mut wrong_program_role = evidence();
    wrong_program_role
        .accumulation_only_program
        .capture_identity = 3;
    assert!(wrong_program_role.validate().is_err());

    let mut wrong_program_inventory = evidence();
    wrong_program_inventory
        .accumulation_only_program
        .schedule_cache_keys
        .pop();
    assert!(wrong_program_inventory.validate().is_err());

    let mut wrong_dropout_stride = evidence();
    wrong_dropout_stride.dropout_blocks_per_replay += 1;
    assert!(wrong_dropout_stride.validate().is_err());

    let mut wrong_build_count = evidence();
    wrong_build_count.compiled_graph_build_count = 2;
    assert!(wrong_build_count.validate().is_err());

    let mut wrong_capsule_hits = evidence();
    wrong_capsule_hits.preparation.render_capsule_hit_count = 3;
    assert!(wrong_capsule_hits.validate().is_err());
}

#[test]
fn optional_arguments_require_explicit_build_provenance() {
    let mut absent = Vec::<String>::new().into_iter();
    assert!(
        SteadyMeasurementRequest::from_arguments(&mut absent)
            .unwrap()
            .is_none()
    );

    let mut valid = [
        "--steady-evidence",
        "steady.json",
        "0123456789abcdef0123456789abcdef01234567",
        cargo_profile(),
    ]
    .map(str::to_owned)
    .into_iter();
    let request = SteadyMeasurementRequest::from_arguments(&mut valid)
        .unwrap()
        .unwrap();
    assert_eq!(request.path, PathBuf::from("steady.json"));
    assert_eq!(request.cargo_profile, cargo_profile());

    let invalid_profile = if cfg!(debug_assertions) {
        "release"
    } else {
        "dev"
    };
    let mut invalid = [
        "--steady-evidence",
        "steady.json",
        "0123456789abcdef0123456789abcdef01234567",
        invalid_profile,
    ]
    .map(str::to_owned)
    .into_iter();
    assert!(SteadyMeasurementRequest::from_arguments(&mut invalid).is_err());
}

#[test]
fn evidence_publication_is_atomic_and_never_overwrites() {
    let root = env::temp_dir().join(format!(
        "rustgrad-steady-evidence-test-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let path = root.join("evidence.json");
    validate_new_destination(&path).unwrap();
    write_new_evidence(&path, b"first\n").unwrap();
    assert!(write_new_evidence(&path, b"replacement\n").is_err());
    assert_eq!(fs::read(&path).unwrap(), b"first\n");
    assert!(validate_new_destination(&path).is_err());
    fs::remove_file(path).unwrap();
    fs::remove_dir(root).unwrap();
}
