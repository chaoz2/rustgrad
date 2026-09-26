use super::compile_phase::{CompilePhaseValidationContext, CompileProgramInventory};
use super::*;

fn zero_duration() -> BenchmarkDuration {
    BenchmarkDuration::from_duration(Duration::ZERO)
}

fn zero_latency_summary() -> BenchmarkLatencySummary {
    BenchmarkLatencySummary {
        sample_count: 1,
        min: zero_duration(),
        nearest_rank_p50: zero_duration(),
        nearest_rank_p95: zero_duration(),
        max: zero_duration(),
    }
}

fn zero_replay_timing() -> NativeTrainingReplayTiming {
    NativeTrainingReplayTiming {
        first: zero_duration(),
        steady_total: zero_duration(),
        steady: zero_latency_summary(),
    }
}

fn zero_compile_phase(
    graph_node_count: Option<u64>,
    logical_schedule_item_count: Option<u64>,
) -> NativeTrainingCompilePhase {
    NativeTrainingCompilePhase {
        wall_time: zero_duration(),
        graph_node_count,
        logical_schedule_item_count,
        recurrent_capture: None,
    }
}

fn zero_recurrent_capture(
    preview_schedule_count: u64,
    cursor_projection: bool,
    recurrent_state_count: u64,
) -> NativeTrainingRecurrentCaptureReport {
    NativeTrainingRecurrentCaptureReport {
        alias_planning_wall_time: zero_duration(),
        preview_schedule_count,
        final_schedule_wall_time: zero_duration(),
        pure_capture_binding_wall_time: zero_duration(),
        effect_assembly_sealing_wall_time: zero_duration(),
        recurrent_authentication_wall_time: zero_duration(),
        cursor_projection_wall_time: cursor_projection.then_some(zero_duration()),
        residual_wall_time: zero_duration(),
        recurrent_state_count,
    }
}

fn zero_compile_phases(main_schedule_item_count: u64) -> NativeTrainingCompilePhaseReport {
    let mut main_capture = zero_compile_phase(None, Some(main_schedule_item_count));
    main_capture.recurrent_capture = Some(zero_recurrent_capture(2, false, 4));
    NativeTrainingCompilePhaseReport {
        compile_count: 1,
        objective_forward: zero_compile_phase(Some(1), None),
        autograd: zero_compile_phase(Some(2), None),
        optimizer_lowering: zero_compile_phase(Some(3), None),
        main_capture,
        accumulation_capture: None,
        partial_flush: None,
        zero_grad: None,
        evaluation: None,
        residual_wall_time: zero_duration(),
    }
}

fn empty_execution_plan_summary() -> ExecutionPlanSummary {
    ExecutionPlanSummary {
        requested_outputs: Vec::new(),
        items: Vec::new(),
        schedule_item_count: 0,
        temporary_allocation_count: 0,
        peak_logical_allocations: 0,
        peak_logical_bytes: 0,
        reuse_enabled: false,
        reuse_count: 0,
        zero_domain_item_count: 0,
        zero_byte_sentinel_count: 0,
        identity: 13,
    }
}

fn observed_compile(duration: Duration) -> CompiledTrainingCompileObservation {
    CompiledTrainingCompileObservation::new(
        CompiledTrainingCompilePhaseObservation::graph(duration, 1),
        CompiledTrainingCompilePhaseObservation::graph(duration, 2),
        CompiledTrainingCompilePhaseObservation::graph(duration, 3),
        CompiledTrainingCompilePhaseObservation::schedule(duration, 0),
        None,
    )
}

#[test]
fn inspection_equality_is_logical_not_compile_observational() {
    let inspection = |compile_phases| {
        CompiledAdamWInspection::new(
            0,
            (7, empty_execution_plan_summary(), 4),
            None,
            None,
            None,
            None,
            (4, 16),
        )
        .with_compile_phases(compile_phases)
    };
    let first = inspection(Some(observed_compile(Duration::from_nanos(1))));
    let later = inspection(Some(observed_compile(Duration::from_nanos(2))));
    let unavailable = inspection(None);
    assert_eq!(first, later);
    assert_eq!(first, unavailable);
    assert_ne!(
        first.compile_phases(),
        later.compile_phases(),
        "the excluded observations remain independently inspectable"
    );
}

fn zero_preparation_timing() -> NativeTrainingPreparationTiming {
    NativeTrainingPreparationTiming {
        total: zero_duration(),
        layout: zero_duration(),
        render: zero_duration(),
        compiler_process: zero_duration(),
        compiler_process_total: None,
        linker_process: None,
        module_load: zero_duration(),
        residual: zero_duration(),
    }
}

fn set_single_steady_replay_duration(
    report: &mut NativeTrainingReport,
    elapsed: BenchmarkDuration,
) {
    let summary = BenchmarkLatencySummary {
        sample_count: 1,
        min: elapsed,
        nearest_rank_p50: elapsed,
        nearest_rank_p95: elapsed,
        max: elapsed,
    };
    report.steady_replay_total_wall_time = elapsed;
    report.steady_replay_wall_time = summary.clone();
    let executor = report
        .main_replay_executor_wall_time
        .as_mut()
        .expect("current test report has executor timing");
    executor.steady_total = elapsed;
    executor.steady = summary;
    report.steady_microbatches_per_second = rate_from_total(1, elapsed).unwrap();
    let phase = report
        .step_phases
        .as_mut()
        .unwrap()
        .warm_optimizer_commit
        .as_mut()
        .unwrap();
    phase.total_wall_time = elapsed;
    phase.wall_time = report.steady_replay_wall_time.clone();
    phase.steps_per_second = report.steady_microbatches_per_second;
    phase.executor_total_wall_time = elapsed;
    phase.executor_wall_time = report.steady_replay_wall_time.clone();
}

fn remove_module_preparation(json: &mut serde_json::Value) {
    for program in ["main", "partial_flush", "zero_grad", "evaluation"] {
        let Some(program) = json[program].as_object_mut() else {
            continue;
        };
        for field in [
            "rendered_entry_count",
            "loaded_module_count",
            "durable_artifact_cache_hit_count",
            "durable_artifact_cache_miss_count",
            "compiler_invocation_count",
        ] {
            program.remove(field);
        }
    }
}

fn remove_replay_phase_timing(json: &mut serde_json::Value) {
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executor_wall_time");
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_recurrent_overhead_wall_time");
}

fn remove_dispatcher_timing(json: &mut serde_json::Value) {
    remove_compiler_critical_path(json);
    for field in [
        "main_replay_native_dispatcher_wall_time",
        "main_replay_executor_host_wall_time",
    ] {
        json.as_object_mut().unwrap().remove(field);
    }
    let Some(phases) = json
        .get_mut("step_phases")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    if let Some(first) = phases
        .get_mut("first")
        .and_then(serde_json::Value::as_object_mut)
    {
        first.remove("native_dispatcher_wall_time");
        first.remove("executor_host_wall_time");
    }
    for phase in ["warm_accumulation_only", "warm_optimizer_commit"] {
        let Some(warm) = phases
            .get_mut(phase)
            .and_then(serde_json::Value::as_object_mut)
        else {
            continue;
        };
        for field in [
            "native_dispatcher_total_wall_time",
            "native_dispatcher_wall_time",
            "executor_host_total_wall_time",
            "executor_host_wall_time",
        ] {
            warm.remove(field);
        }
    }
}

fn remove_compiler_critical_path(json: &mut serde_json::Value) {
    let report = json.as_object_mut().unwrap();
    report.remove("prepare_compiler_process_timings");
    report.remove("prepare_compiler_critical_tail");
}

fn remove_module_overlap_evidence(json: &mut serde_json::Value) {
    let report = json.as_object_mut().unwrap();
    report.remove("prepare_module_overlaps");
    report.remove("prepare_program_pair_overlaps");
    report.remove("prepare_translation_units");
    if let Some(timings) = report
        .get_mut("prepare_compiler_process_timings")
        .and_then(serde_json::Value::as_array_mut)
    {
        for timing in timings {
            timing
                .as_object_mut()
                .unwrap()
                .remove("rendered_source_bytes");
        }
    }
    for program in [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        if let Some(program) = report
            .get_mut(program)
            .and_then(serde_json::Value::as_object_mut)
        {
            program.remove("rendered_source_bytes");
            program.remove("shared_prefix_source_bytes");
        }
    }
}

fn remove_v21_module_evidence(json: &mut serde_json::Value) {
    let report = json.as_object_mut().unwrap();
    report.remove("prepare_program_pair_overlaps");
    report.remove("prepare_translation_units");
}

fn remove_step_phases(json: &mut serde_json::Value) {
    json.as_object_mut().unwrap().remove("step_phases");
}

fn remove_preparation_phase_timing(json: &mut serde_json::Value) {
    json.as_object_mut()
        .unwrap()
        .remove("prepare_runtime_overhead_wall_time");
    json.as_object_mut()
        .unwrap()
        .remove("prepare_parallel_module_overlap_wall_time");
    for field in [
        "prepare_compiler_process_overlap_wall_time",
        "prepare_compiler_process_count",
        "prepare_max_parallel_compiler_process_count",
    ] {
        json.as_object_mut().unwrap().remove(field);
    }
    for program in ["main", "partial_flush", "zero_grad", "evaluation"] {
        if let Some(program) = json[program].as_object_mut() {
            program.remove("preparation_timing");
        }
    }
}

fn remove_prefix_module_evidence(json: &mut serde_json::Value) {
    for program in [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        let Some(program) = json[program].as_object_mut() else {
            continue;
        };
        for field in [
            "referenced_module_count",
            "unique_rendered_entry_count",
            "shared_prefix_entry_count",
            "shared_prefix_source_program_index",
            "shared_prefix_source_native_identity",
        ] {
            program.remove(field);
        }
    }
}

fn remove_chunk_compiler_evidence(json: &mut serde_json::Value) {
    for program in [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        let Some(program) = json[program].as_object_mut() else {
            continue;
        };
        for field in [
            "combined_compile_link_count",
            "object_compile_count",
            "linker_invocation_count",
        ] {
            program.remove(field);
        }
        if let Some(timing) = program["preparation_timing"].as_object_mut() {
            timing.remove("compiler_process_total");
            timing.remove("linker_process");
        }
    }
}

fn remove_parallel_render_evidence(json: &mut serde_json::Value) {
    json.as_object_mut()
        .unwrap()
        .remove("prepare_parallel_render_overlap_wall_time");
    json.as_object_mut()
        .unwrap()
        .remove("prepare_max_parallel_render_job_count");
    remove_render_capsule_evidence(json);
}

fn remove_render_capsule_evidence(json: &mut serde_json::Value) {
    for field in [
        "prepare_render_capsule_hit_count",
        "prepare_render_capsule_miss_count",
        "prepare_local_render_job_count",
    ] {
        json.as_object_mut().unwrap().remove(field);
    }
}

fn remove_compile_phase_evidence(json: &mut serde_json::Value) {
    json.as_object_mut().unwrap().remove("compile_phases");
    remove_program_recurrent_state_evidence(json);
}

fn remove_recurrent_capture_evidence(json: &mut serde_json::Value) {
    let phases = json["compile_phases"].as_object_mut().unwrap();
    for phase in [
        "main_capture",
        "accumulation_capture",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        if let Some(phase) = phases
            .get_mut(phase)
            .and_then(serde_json::Value::as_object_mut)
        {
            phase.remove("recurrent_capture");
        }
    }
    remove_program_recurrent_state_evidence(json);
}

fn remove_program_recurrent_state_evidence(json: &mut serde_json::Value) {
    for program in [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        if let Some(program) = json[program].as_object_mut() {
            program.remove("recurrent_state_count");
        }
    }
}

fn remove_dispatch_segmentation_evidence(json: &mut serde_json::Value) {
    for program in [
        "main",
        "accumulation",
        "partial_flush",
        "zero_grad",
        "evaluation",
    ] {
        if let Some(program) = json[program].as_object_mut() {
            program.remove("dispatch_segmentation");
        }
    }
}

fn zero_report() -> NativeTrainingReport {
    NativeTrainingReport {
        format_version: NATIVE_TRAINING_REPORT_FORMAT_V10,
        compile_wall_time: zero_duration(),
        compile_phases: None,
        prepare_wall_time: zero_duration(),
        prepare_runtime_overhead_wall_time: Some(zero_duration()),
        prepare_parallel_module_overlap_wall_time: Some(zero_duration()),
        prepare_parallel_render_overlap_wall_time: None,
        prepare_max_parallel_render_job_count: None,
        prepare_render_capsule_hit_count: None,
        prepare_render_capsule_miss_count: None,
        prepare_local_render_job_count: None,
        prepare_compiler_process_overlap_wall_time: Some(zero_duration()),
        prepare_compiler_process_count: Some(1),
        prepare_max_parallel_compiler_process_count: Some(1),
        prepare_compiler_process_timings: None,
        prepare_compiler_critical_tail: None,
        prepare_module_overlaps: None,
        prepare_program_pair_overlaps: None,
        prepare_translation_units: None,
        initial_replay_step: 0,
        successful_replay_count: 2,
        main: NativeTrainingProgramReport {
            capture_identity: 7,
            native_identity: 11,
            vectorized: true,
            execution_plan_identity: 13,
            logical_schedule_item_count: 2,
            recurrent_state_count: None,
            peak_logical_temporary_allocation_count: 1,
            peak_logical_temporary_bytes: 4,
            native_item_count: 2,
            cache_hit_count: 0,
            cache_miss_count: 2,
            rendered_entry_count: 2,
            rendered_source_bytes: None,
            loaded_module_count: 1,
            referenced_module_count: None,
            unique_rendered_entry_count: None,
            shared_prefix_entry_count: None,
            shared_prefix_source_bytes: None,
            shared_prefix_source_program_index: None,
            shared_prefix_source_native_identity: None,
            durable_artifact_cache_hit_count: 0,
            durable_artifact_cache_miss_count: 1,
            compiler_invocation_count: 1,
            combined_compile_link_count: None,
            object_compile_count: None,
            linker_invocation_count: None,
            preparation_timing: Some(zero_preparation_timing()),
            dispatch_segmentation: None,
        },
        accumulation: None,
        partial_flush: None,
        zero_grad: None,
        evaluation: None,
        recurrent_logical_state_count: 4,
        recurrent_logical_state_bytes: 16,
        main_replay_traffic: Some(NativeCpuReplayTraffic::new(2, 12, 16, 16)),
        main_replay_executed_native_item_count: Some(1),
        accumulation_replay_traffic: None,
        accumulation_replay_executed_native_item_count: None,
        main_replay_executor_wall_time: Some(zero_replay_timing()),
        main_replay_native_dispatcher_wall_time: None,
        main_replay_executor_host_wall_time: None,
        main_replay_recurrent_overhead_wall_time: Some(zero_replay_timing()),
        step_phases: Some(NativeTrainingStepPhaseReport {
            first: NativeTrainingFirstStepReport {
                phase: NativeTrainingStepPhase::AccumulationOnly,
                total_wall_time: zero_duration(),
                executor_wall_time: zero_duration(),
                native_dispatcher_wall_time: None,
                executor_host_wall_time: None,
                recurrent_overhead_wall_time: zero_duration(),
            },
            warm_accumulation_only: None,
            warm_optimizer_commit: Some(NativeTrainingWarmStepReport {
                total_wall_time: zero_duration(),
                wall_time: BenchmarkLatencySummary {
                    sample_count: 1,
                    min: zero_duration(),
                    nearest_rank_p50: zero_duration(),
                    nearest_rank_p95: zero_duration(),
                    max: zero_duration(),
                },
                steps_per_second: None,
                executor_total_wall_time: zero_duration(),
                executor_wall_time: BenchmarkLatencySummary {
                    sample_count: 1,
                    min: zero_duration(),
                    nearest_rank_p50: zero_duration(),
                    nearest_rank_p95: zero_duration(),
                    max: zero_duration(),
                },
                native_dispatcher_total_wall_time: None,
                native_dispatcher_wall_time: None,
                executor_host_total_wall_time: None,
                executor_host_wall_time: None,
                recurrent_overhead_total_wall_time: zero_duration(),
                recurrent_overhead_wall_time: BenchmarkLatencySummary {
                    sample_count: 1,
                    min: zero_duration(),
                    nearest_rank_p50: zero_duration(),
                    nearest_rank_p95: zero_duration(),
                    max: zero_duration(),
                },
            }),
        }),
        first_replay_wall_time: zero_duration(),
        steady_replay_wall_time: BenchmarkLatencySummary {
            sample_count: 1,
            min: zero_duration(),
            nearest_rank_p50: zero_duration(),
            nearest_rank_p95: zero_duration(),
            max: zero_duration(),
        },
        steady_replay_total_wall_time: zero_duration(),
        steady_microbatches_per_second: None,
        schedule_cache_keys: vec![17, 19],
        accumulation_schedule_cache_keys: Vec::new(),
        checkpoint: Some(CheckpointReport {
            capture_identity: 7,
            replay_step: 2,
            byte_count: 32,
            wall_time: zero_duration(),
        }),
        fallback_count: 0,
        kernel_launch_count: None,
        host_to_device: None,
        device_to_host: None,
        measured_peak_host_memory_bytes: None,
    }
}

fn phase_specialized_report() -> NativeTrainingReport {
    let mut report = zero_report();
    report.format_version = NATIVE_TRAINING_REPORT_FORMAT_VERSION;
    report.compile_phases = Some(zero_compile_phases(report.main.logical_schedule_item_count));
    report.main.recurrent_state_count = Some(4);
    report.prepare_parallel_render_overlap_wall_time = Some(zero_duration());
    report.prepare_max_parallel_render_job_count = Some(0);
    report.prepare_render_capsule_hit_count = Some(2);
    report.prepare_render_capsule_miss_count = Some(0);
    report.prepare_local_render_job_count = Some(0);
    report.main_replay_native_dispatcher_wall_time = Some(zero_replay_timing());
    report.main_replay_executor_host_wall_time = Some(zero_replay_timing());
    let phases = report.step_phases.as_mut().unwrap();
    phases.first.native_dispatcher_wall_time = Some(zero_duration());
    phases.first.executor_host_wall_time = Some(zero_duration());
    let warm = phases.warm_optimizer_commit.as_mut().unwrap();
    warm.native_dispatcher_total_wall_time = Some(zero_duration());
    warm.native_dispatcher_wall_time = Some(zero_latency_summary());
    warm.executor_host_total_wall_time = Some(zero_duration());
    warm.executor_host_wall_time = Some(zero_latency_summary());
    report.main_replay_traffic = report
        .main_replay_traffic
        .map(|traffic| traffic.with_materialized_egress(5, 24));
    report.main.referenced_module_count = Some(1);
    report.main.rendered_source_bytes = Some(20);
    report.main.unique_rendered_entry_count = Some(2);
    report.main.shared_prefix_entry_count = Some(0);
    report.main.shared_prefix_source_bytes = Some(0);
    report.main.combined_compile_link_count = Some(1);
    report.main.object_compile_count = Some(0);
    report.main.linker_invocation_count = Some(0);
    report.main.dispatch_segmentation = Some(NativeCpuDispatchSegmentation {
        segment_count: 1,
        dispatch_reached_module_count: 1,
        terminal_segment_count: 1,
        non_dispatch_boundary_count: 0,
        module_change_count: 0,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    });
    let timing = report.main.preparation_timing.as_mut().unwrap();
    timing.compiler_process_total = Some(timing.compiler_process);
    timing.linker_process = Some(zero_duration());
    let mut accumulation = report.main.clone();
    accumulation.capture_identity = 8;
    accumulation.native_identity = 12;
    accumulation.execution_plan_identity = 14;
    accumulation.cache_hit_count = 1;
    accumulation.cache_miss_count = 1;
    accumulation.referenced_module_count = Some(2);
    accumulation.unique_rendered_entry_count = Some(1);
    accumulation.shared_prefix_entry_count = Some(1);
    accumulation.shared_prefix_source_bytes = Some(10);
    accumulation.shared_prefix_source_program_index = Some(0);
    accumulation.shared_prefix_source_native_identity = Some(report.main.native_identity);
    accumulation.dispatch_segmentation = Some(NativeCpuDispatchSegmentation {
        segment_count: 2,
        dispatch_reached_module_count: 2,
        terminal_segment_count: 1,
        non_dispatch_boundary_count: 0,
        module_change_count: 1,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    });
    report.accumulation = Some(accumulation);
    report.compile_phases.as_mut().unwrap().accumulation_capture = Some(zero_compile_phase(
        None,
        Some(
            report
                .accumulation
                .as_ref()
                .unwrap()
                .logical_schedule_item_count,
        ),
    ));
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .accumulation_capture
        .as_mut()
        .unwrap()
        .recurrent_capture = Some(zero_recurrent_capture(3, true, 4));
    report.accumulation_replay_traffic = Some(
        NativeCpuReplayTraffic::new(2, 12, 16, 8)
            .with_recurrent_inventory(2, 8, 2, 8)
            .with_materialized_egress(1, 4),
    );
    report.accumulation_replay_executed_native_item_count =
        report.main_replay_executed_native_item_count;
    report.accumulation_schedule_cache_keys = vec![23, 29];
    report.prepare_compiler_process_count = Some(2);
    report.prepare_compiler_process_timings = Some(vec![
        NativeTrainingCompilerProcessTiming {
            program_index: 0,
            native_identity: report.main.native_identity,
            process: NativeTrainingCompilerProcessKind::Combined,
            rendered_source_bytes: Some(20),
            permit_request_offset: zero_duration(),
            permit_wait: zero_duration(),
            process_wall_time: zero_duration(),
        },
        NativeTrainingCompilerProcessTiming {
            program_index: 1,
            native_identity: report.accumulation.as_ref().unwrap().native_identity,
            process: NativeTrainingCompilerProcessKind::Combined,
            rendered_source_bytes: Some(10),
            permit_request_offset: zero_duration(),
            permit_wait: zero_duration(),
            process_wall_time: zero_duration(),
        },
    ]);
    report.prepare_compiler_critical_tail = Some(NativeTrainingCompilerCriticalTail {
        program_index: 0,
        native_identity: report.main.native_identity,
        process: NativeTrainingCompilerProcessKind::Combined,
        finish_offset: zero_duration(),
        post_main_tail: zero_duration(),
    });
    report.prepare_module_overlaps = Some(vec![
        NativeTrainingModuleOverlap {
            program_index: 1,
            native_identity: report.accumulation.as_ref().unwrap().native_identity,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity(report.main.native_identity),
    ]);
    report.prepare_program_pair_overlaps = Some(vec![
        NativeTrainingProgramPairOverlap {
            source_program_index: 0,
            source_native_identity: report.main.native_identity,
            target_program_index: 1,
            target_native_identity: report.accumulation.as_ref().unwrap().native_identity,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity(),
    ]);
    report.prepare_translation_units = Some(vec![
        NativeTrainingTranslationUnit {
            program_index: 0,
            native_identity: report.main.native_identity,
            ordinal: 0,
            translation_unit_identity: 101,
            evidence_identity: 0,
            entry_count: 2,
            rendered_source_bytes: 20,
        }
        .with_evidence_identity(),
        NativeTrainingTranslationUnit {
            program_index: 1,
            native_identity: report.accumulation.as_ref().unwrap().native_identity,
            ordinal: 0,
            translation_unit_identity: 102,
            evidence_identity: 0,
            entry_count: 1,
            rendered_source_bytes: 10,
        }
        .with_evidence_identity(),
    ]);
    report
}

fn set_main_unique_rendered_entry_count(report: &mut NativeTrainingReport, count: u64) {
    let count_usize = usize::try_from(count).unwrap();
    let rendered_source_bytes = count.checked_mul(10).unwrap();
    report.main.logical_schedule_item_count = count;
    report.main.native_item_count = count;
    report.main.cache_hit_count = 0;
    report.main.cache_miss_count = count;
    report.main.rendered_entry_count = count;
    report.main.rendered_source_bytes = Some(rendered_source_bytes);
    report.main.unique_rendered_entry_count = Some(count);
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .logical_schedule_item_count = Some(count);
    report.schedule_cache_keys = (0..count_usize)
        .map(|index| u64::try_from(index).unwrap() + 101)
        .collect();
    let timings = report.prepare_compiler_process_timings.as_mut().unwrap();
    let source_process_count = timings
        .iter()
        .filter(|timing| {
            timing.program_index == 0
                && !matches!(timing.process, NativeTrainingCompilerProcessKind::Link)
        })
        .count();
    assert_ne!(source_process_count, 0);
    let source_process_count = u64::try_from(source_process_count).unwrap();
    let bytes_per_process = rendered_source_bytes / source_process_count;
    let remainder = rendered_source_bytes % source_process_count;
    let mut source_process_ordinal = 0u64;
    for timing in timings
        .iter_mut()
        .filter(|timing| timing.program_index == 0)
    {
        timing.rendered_source_bytes = Some(
            if matches!(timing.process, NativeTrainingCompilerProcessKind::Link) {
                0
            } else {
                let bytes = bytes_per_process + u64::from(source_process_ordinal < remainder);
                source_process_ordinal += 1;
                bytes
            },
        );
    }
    let units = report.prepare_translation_units.as_mut().unwrap();
    units.retain(|unit| unit.program_index != 0);
    let unit_count = if count <= 512 { 1 } else { 2 };
    let entries_per_unit = count / unit_count;
    let entry_remainder = count % unit_count;
    let bytes_per_unit = rendered_source_bytes / unit_count;
    let byte_remainder = rendered_source_bytes % unit_count;
    let mut main_units = (0..unit_count)
        .map(|ordinal| {
            NativeTrainingTranslationUnit {
                program_index: 0,
                native_identity: report.main.native_identity,
                ordinal,
                translation_unit_identity: 10_000 + ordinal,
                evidence_identity: 0,
                entry_count: entries_per_unit + u64::from(ordinal < entry_remainder),
                rendered_source_bytes: bytes_per_unit + u64::from(ordinal < byte_remainder),
            }
            .with_evidence_identity()
        })
        .collect::<Vec<_>>();
    main_units.append(units);
    *units = main_units;
}

#[test]
fn phase_specialized_report_round_trips_and_authenticates_both_programs() {
    let report = phase_specialized_report();
    let bytes = report.to_json_bytes().unwrap();
    let decoded = NativeTrainingReport::from_json_bytes(&bytes).unwrap();
    assert_eq!(decoded, report);
    assert_eq!(
        decoded.format_version,
        NATIVE_TRAINING_REPORT_FORMAT_VERSION
    );
    assert_eq!(decoded.accumulation().unwrap().capture_identity(), 8);
    assert_eq!(decoded.accumulation_schedule_cache_keys(), [23, 29]);
    assert_eq!(
        decoded.prepare_parallel_render_overlap_wall_time(),
        Some(zero_duration())
    );
    assert_eq!(decoded.prepare_max_parallel_render_job_count(), Some(0));
    assert!(decoded.main_replay_native_dispatcher_wall_time().is_some());
    assert!(decoded.main_replay_executor_host_wall_time().is_some());
    assert_eq!(
        decoded
            .prepare_compiler_process_timings
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        decoded.prepare_module_overlaps.as_ref().unwrap()[0],
        NativeTrainingModuleOverlap {
            program_index: 1,
            native_identity: 12,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity(11)
    );
    assert_eq!(
        decoded.prepare_program_pair_overlaps.as_ref().unwrap()[0],
        NativeTrainingProgramPairOverlap {
            source_program_index: 0,
            source_native_identity: 11,
            target_program_index: 1,
            target_native_identity: 12,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity()
    );
    assert_eq!(decoded.prepare_translation_units.as_ref().unwrap().len(), 2);
    assert_eq!(
        decoded
            .prepare_compiler_critical_tail
            .as_ref()
            .unwrap()
            .program_index,
        0
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_process_timings"][1]["rendered_source_bytes"] = serde_json::json!(9);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "translation-unit source bytes must cover the exact unique suffix"
    );

    for pointer in [
        "/main/rendered_source_bytes",
        "/main/shared_prefix_source_bytes",
        "/prepare_compiler_process_timings/0/rendered_source_bytes",
    ] {
        let mut json = serde_json::to_value(&report).unwrap();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        json.pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "current rendered-source evidence must be complete: {pointer}"
        );
    }

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_module_overlaps"][0]["additional_scattered_entry_count"] = serde_json::json!(1);
    json["prepare_module_overlaps"][0]["additional_scattered_source_bytes"] = serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "the overlap identity rejects an otherwise in-range count-and-byte partition tamper"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_program_pair_overlaps"][0]["additional_scattered_entry_count"] =
        serde_json::json!(1);
    json["prepare_program_pair_overlaps"][0]["additional_scattered_source_bytes"] =
        serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "the pair identity rejects an otherwise in-range count-and-byte tamper"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_translation_units"][0]["entry_count"] = serde_json::json!(1);
    json["prepare_translation_units"][0]["rendered_source_bytes"] = serde_json::json!(10);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "the translation-unit identity rejects a paired partition tamper"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_module_overlaps"][0]["additional_scattered_entry_count"] = serde_json::json!(2);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "overlap entry partitions cannot exceed the target inventory"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_module_overlaps"][0]["program_index"] = serde_json::json!(0);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "module overlaps remain in attached-program order"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("prepare_module_overlaps");
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "current overlap evidence must be complete"
    );

    for field in ["prepare_program_pair_overlaps", "prepare_translation_units"] {
        let mut json = serde_json::to_value(&report).unwrap();
        json.as_object_mut().unwrap().remove(field);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "current v21 module evidence must be complete: {field}"
        );
    }

    let mut json = serde_json::to_value(&report).unwrap();
    json["accumulation_schedule_cache_keys"] = serde_json::json!([23]);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_process_timings"][1]["native_identity"] = serde_json::json!(99);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "compiler process timing must remain bound to its ordered program"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_process_timings"][0]["process"] =
        serde_json::json!({ "kind": "object", "ordinal": 0 });
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "compiler process roles must match the authenticated program build mode"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_process_timings"][0]["permit_wait"]["nanos"] =
        serde_json::json!(1_000_000_000u64);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "malformed compiler interval durations must fail closed"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_process_timings"][0]["permit_request_offset"]["nanos"] =
        serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "compiler process intervals must remain inside caller-observed preparation"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["prepare_compiler_critical_tail"]["program_index"] = serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "critical-tail ownership is derived from authenticated intervals"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["accumulation_replay_executed_native_item_count"] = serde_json::json!(3);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&report).unwrap();
    let main_identity = json["main"]["capture_identity"].clone();
    json["accumulation"]["capture_identity"] = main_identity;
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&report).unwrap();
    json["accumulation_replay_traffic"]["borrowed_recurrent_output_bytes"] = serde_json::json!(15);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut json = serde_json::to_value(&report).unwrap();
    json["accumulation_replay_traffic"]["retained_recurrent_state_count"] = serde_json::json!(1);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    for field in ["materialized_egress_count", "materialized_egress_bytes"] {
        let mut json = serde_json::to_value(&report).unwrap();
        json["main_replay_traffic"][field] = serde_json::json!(0);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );

        let mut json = serde_json::to_value(&report).unwrap();
        json["main_replay_traffic"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err()
        );
    }

    let mut json = serde_json::to_value(&report).unwrap();
    let traffic = json["main_replay_traffic"].as_object_mut().unwrap();
    traffic.remove("materialized_egress_count");
    traffic.remove("materialized_egress_bytes");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn program_pair_overlap_inventory_is_target_major_triangular() {
    let report = phase_specialized_report();
    let main = report.main.clone();
    let accumulation = report.accumulation.clone().unwrap();
    let mut later = accumulation.clone();
    later.capture_identity = 9;
    later.native_identity = 15;
    later.execution_plan_identity = 16;
    later.shared_prefix_source_native_identity = Some(main.native_identity);
    let programs = vec![&main, &accumulation, &later];
    let main_overlaps = vec![
        NativeTrainingModuleOverlap {
            program_index: 1,
            native_identity: accumulation.native_identity,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity(main.native_identity),
        NativeTrainingModuleOverlap {
            program_index: 2,
            native_identity: later.native_identity,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity(main.native_identity),
    ];
    let pair = |source: usize, target: usize| {
        NativeTrainingProgramPairOverlap {
            source_program_index: u64::try_from(source).unwrap(),
            source_native_identity: programs[source].native_identity,
            target_program_index: u64::try_from(target).unwrap(),
            target_native_identity: programs[target].native_identity,
            evidence_identity: 0,
            contiguous_prefix_entry_count: 1,
            contiguous_prefix_source_bytes: 10,
            additional_scattered_entry_count: 0,
            additional_scattered_source_bytes: 0,
        }
        .with_evidence_identity()
    };
    let overlaps = vec![pair(0, 1), pair(0, 2), pair(1, 2)];
    assert!(validate_program_pair_overlaps(&overlaps, &programs, &main_overlaps).is_ok());
    let mut reordered = overlaps.clone();
    reordered.swap(1, 2);
    assert!(validate_program_pair_overlaps(&reordered, &programs, &main_overlaps).is_err());
}

#[test]
fn phase_specialized_v11_report_decodes_without_recurrent_retention_inventory() {
    let report = phase_specialized_report();
    let mut json = serde_json::to_value(report).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V11);
    remove_compile_phase_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    remove_dispatch_segmentation_evidence(&mut json);
    remove_parallel_render_evidence(&mut json);
    remove_prefix_module_evidence(&mut json);
    remove_chunk_compiler_evidence(&mut json);
    for phase in ["main_replay_traffic", "accumulation_replay_traffic"] {
        let traffic = json[phase].as_object_mut().unwrap();
        traffic.remove("retained_recurrent_state_count");
        traffic.remove("retained_recurrent_state_bytes");
        traffic.remove("replaced_recurrent_state_count");
        traffic.remove("replaced_recurrent_state_bytes");
        traffic.remove("materialized_egress_count");
        traffic.remove("materialized_egress_bytes");
    }
    json["accumulation_replay_traffic"]["borrowed_recurrent_output_bytes"] = serde_json::json!(16);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V11);
    assert_eq!(
        decoded
            .accumulation_replay_traffic()
            .unwrap()
            .retained_recurrent_state_count(),
        0
    );
}

#[test]
fn phase_specialized_v12_report_decodes_without_egress_evidence() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V12);
    remove_compile_phase_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    remove_dispatch_segmentation_evidence(&mut json);
    remove_parallel_render_evidence(&mut json);
    remove_prefix_module_evidence(&mut json);
    remove_chunk_compiler_evidence(&mut json);
    for phase in ["main_replay_traffic", "accumulation_replay_traffic"] {
        let traffic = json[phase].as_object_mut().unwrap();
        traffic.remove("materialized_egress_count");
        traffic.remove("materialized_egress_bytes");
    }
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V12);
    assert_eq!(
        decoded
            .main_replay_traffic()
            .unwrap()
            .materialized_egress_count(),
        0
    );
    assert_eq!(
        decoded
            .accumulation_replay_traffic()
            .unwrap()
            .materialized_egress_bytes(),
        0
    );
}

#[test]
fn version_thirteen_report_decodes_without_prefix_module_evidence() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V13);
    remove_compile_phase_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    remove_dispatch_segmentation_evidence(&mut json);
    remove_parallel_render_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v13 cannot claim v14 prefix-module evidence"
    );
    for program in ["main", "accumulation"] {
        let program = json[program].as_object_mut().unwrap();
        for field in [
            "referenced_module_count",
            "unique_rendered_entry_count",
            "shared_prefix_entry_count",
        ] {
            program.insert(field.into(), serde_json::json!(0));
        }
        program.remove("shared_prefix_source_program_index");
        program.remove("shared_prefix_source_native_identity");
    }
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v13 rejects even zero-valued v14 prefix-module fields"
    );
    remove_prefix_module_evidence(&mut json);
    remove_chunk_compiler_evidence(&mut json);
    for program in ["main", "accumulation"] {
        let program = json[program].as_object_mut().unwrap();
        program.insert("loaded_module_count".into(), serde_json::json!(1));
    }
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V13);
    assert_eq!(decoded.main().referenced_module_count(), 0);
    assert_eq!(
        decoded.accumulation().unwrap().shared_prefix_entry_count(),
        0
    );
}

#[test]
fn version_fourteen_preserves_prefix_evidence_and_rejects_v15_compiler_fields() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V14);
    remove_compile_phase_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    remove_dispatch_segmentation_evidence(&mut json);
    remove_parallel_render_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v14 cannot claim v15 chunk compiler evidence"
    );
    for program in ["main", "accumulation"] {
        json[program]["combined_compile_link_count"] = serde_json::json!(0);
        json[program]["object_compile_count"] = serde_json::json!(0);
        json[program]["linker_invocation_count"] = serde_json::json!(0);
    }
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v14 rejects even zero-valued v15 chunk compiler fields"
    );
    remove_chunk_compiler_evidence(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V14);
    assert_eq!(decoded.main().referenced_module_count(), 1);
    assert_eq!(
        decoded.accumulation().unwrap().shared_prefix_entry_count(),
        1
    );
    assert_eq!(decoded.main().object_compile_count(), 0);
}

#[test]
fn current_report_authenticates_chunked_compile_and_link_inventory() {
    let mut report = phase_specialized_report();
    set_main_unique_rendered_entry_count(&mut report, 682);
    report.main.combined_compile_link_count = Some(0);
    report.main.object_compile_count = Some(2);
    report.main.linker_invocation_count = Some(1);
    report.main.compiler_invocation_count = 3;
    report.prepare_compiler_process_count = Some(4);
    report.prepare_max_parallel_compiler_process_count = Some(2);
    report.prepare_compiler_process_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    let timing = report.main.preparation_timing.as_mut().unwrap();
    timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    timing.compiler_process = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    timing.compiler_process_total = Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
    timing.linker_process = Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    report.prepare_compiler_process_timings = Some(vec![
        NativeTrainingCompilerProcessTiming {
            program_index: 0,
            native_identity: report.main.native_identity,
            process: NativeTrainingCompilerProcessKind::Object { ordinal: 0 },
            rendered_source_bytes: Some(3_410),
            permit_request_offset: zero_duration(),
            permit_wait: zero_duration(),
            process_wall_time: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
        },
        NativeTrainingCompilerProcessTiming {
            program_index: 0,
            native_identity: report.main.native_identity,
            process: NativeTrainingCompilerProcessKind::Object { ordinal: 1 },
            rendered_source_bytes: Some(3_410),
            permit_request_offset: zero_duration(),
            permit_wait: zero_duration(),
            process_wall_time: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
        },
        NativeTrainingCompilerProcessTiming {
            program_index: 0,
            native_identity: report.main.native_identity,
            process: NativeTrainingCompilerProcessKind::Link,
            rendered_source_bytes: Some(0),
            permit_request_offset: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
            permit_wait: zero_duration(),
            process_wall_time: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
        },
        NativeTrainingCompilerProcessTiming {
            program_index: 1,
            native_identity: report.accumulation.as_ref().unwrap().native_identity,
            process: NativeTrainingCompilerProcessKind::Combined,
            rendered_source_bytes: Some(10),
            permit_request_offset: zero_duration(),
            permit_wait: zero_duration(),
            process_wall_time: zero_duration(),
        },
    ]);
    report.prepare_compiler_critical_tail = Some(NativeTrainingCompilerCriticalTail {
        program_index: 0,
        native_identity: report.main.native_identity,
        process: NativeTrainingCompilerProcessKind::Link,
        finish_offset: BenchmarkDuration::from_duration(Duration::from_nanos(2)),
        post_main_tail: zero_duration(),
    });
    assert!(report.validate().is_ok());

    let valid = report.clone();

    let mut direct = phase_specialized_report();
    set_main_unique_rendered_entry_count(&mut direct, 512);
    assert!(direct.validate().is_ok());
    set_main_unique_rendered_entry_count(&mut direct, 682);
    assert!(
        direct.validate().is_err(),
        "an oversized unique suffix cannot claim combined compilation"
    );

    let mut small_chunked = valid.clone();
    set_main_unique_rendered_entry_count(&mut small_chunked, 512);
    assert!(
        small_chunked.validate().is_err(),
        "a bounded unique suffix cannot claim chunked compilation"
    );

    report.main.combined_compile_link_count = Some(1);
    assert!(report.validate().is_err(), "compiler modes cannot be mixed");

    let mut report = valid.clone();
    report.main.linker_invocation_count = Some(0);
    assert!(
        report.validate().is_err(),
        "chunked compilation must link once"
    );

    let mut report = valid.clone();
    report
        .accumulation
        .as_mut()
        .unwrap()
        .preparation_timing
        .as_mut()
        .unwrap()
        .compiler_process_total = Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    assert!(
        report.validate().is_err(),
        "one compiler process has no internal overlap"
    );

    let mut json = serde_json::to_value(valid).unwrap();
    json["main"]
        .as_object_mut()
        .unwrap()
        .remove("object_compile_count");
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v15 requires complete chunk compiler evidence"
    );
}

#[test]
fn version_fifteen_decodes_without_parallel_render_evidence() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V15);
    remove_compile_phase_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    remove_dispatch_segmentation_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v15 cannot claim v16 parallel render evidence"
    );
    remove_parallel_render_evidence(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V15);
    assert!(
        decoded
            .prepare_parallel_render_overlap_wall_time()
            .is_none()
    );
    assert!(decoded.prepare_max_parallel_render_job_count().is_none());
}

#[test]
fn version_sixteen_rejects_dispatch_segmentation_even_when_zero() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V16);
    remove_compile_phase_evidence(&mut json);
    remove_render_capsule_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    remove_dispatcher_timing(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v16 cannot claim v17 dispatch segmentation evidence"
    );
    for program in ["main", "accumulation"] {
        json[program]["dispatch_segmentation"] = serde_json::json!({
            "segment_count": 0,
            "dispatch_reached_module_count": 0,
            "terminal_segment_count": 0,
            "non_dispatch_boundary_count": 0,
            "module_change_count": 0,
            "output_slot_alias_count": 0,
            "derived_slot_dependency_count": 0,
        });
    }
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v16 rejects explicit zero-valued v17 evidence"
    );
    remove_dispatch_segmentation_evidence(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V16);
    assert!(decoded.main().dispatch_segmentation().is_none());
}

#[test]
fn version_seventeen_decodes_without_dispatcher_timing() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V17);
    remove_compile_phase_evidence(&mut json);
    remove_render_capsule_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v17 cannot claim v18 dispatcher timing"
    );
    remove_dispatcher_timing(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V17);
    assert!(decoded.main_replay_native_dispatcher_wall_time().is_none());
    assert!(decoded.main_replay_executor_host_wall_time().is_none());
    assert!(
        decoded
            .step_phases()
            .unwrap()
            .first()
            .native_dispatcher_wall_time()
            .is_none()
    );
}

#[test]
fn version_eighteen_decodes_without_compiler_critical_path() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V18);
    remove_compile_phase_evidence(&mut json);
    remove_render_capsule_evidence(&mut json);
    remove_module_overlap_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v18 cannot claim v19 compiler critical-path evidence"
    );
    remove_compiler_critical_path(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V18);
    assert!(decoded.main_replay_native_dispatcher_wall_time().is_some());
}

#[test]
fn version_nineteen_decodes_without_module_overlap_evidence() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V19);
    remove_compile_phase_evidence(&mut json);
    remove_render_capsule_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v19 cannot claim v20 module-overlap evidence"
    );
    remove_module_overlap_evidence(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V19);
    assert!(decoded.prepare_compiler_critical_tail.is_some());
}

#[test]
fn version_twenty_decodes_without_pair_and_translation_unit_evidence() {
    let mut json = serde_json::to_value(phase_specialized_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V20);
    remove_compile_phase_evidence(&mut json);
    remove_render_capsule_evidence(&mut json);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v20 cannot claim v21 pair or translation-unit evidence"
    );
    remove_v21_module_evidence(&mut json);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V20);
    assert!(decoded.prepare_module_overlaps.is_some());
    assert!(decoded.prepare_program_pair_overlaps.is_none());
    assert!(decoded.prepare_translation_units.is_none());

    let mut with_capsule_evidence = serde_json::to_value(decoded).unwrap();
    with_capsule_evidence["prepare_render_capsule_hit_count"] = serde_json::json!(2);
    with_capsule_evidence["prepare_render_capsule_miss_count"] = serde_json::json!(0);
    with_capsule_evidence["prepare_local_render_job_count"] = serde_json::json!(0);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&with_capsule_evidence).unwrap())
            .is_err(),
        "v20 rejects explicit v22 render capsule evidence"
    );
}

#[test]
fn version_twenty_one_rejects_capsule_evidence_and_current_requires_it() {
    let report = phase_specialized_report();
    let mut legacy = serde_json::to_value(&report).unwrap();
    legacy["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V21);
    remove_compile_phase_evidence(&mut legacy);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err(),
        "v21 rejects v22 render capsule evidence"
    );
    remove_render_capsule_evidence(&mut legacy);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V21);
    assert!(decoded.prepare_render_capsule_hit_count().is_none());

    let mut current = serde_json::to_value(report).unwrap();
    remove_render_capsule_evidence(&mut current);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&current).unwrap()).is_err(),
        "current report requires complete render capsule evidence"
    );
}

#[test]
fn version_twenty_two_decodes_without_compile_phases_and_current_requires_them() {
    let report = phase_specialized_report();
    let mut legacy = serde_json::to_value(&report).unwrap();
    legacy["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V22);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err(),
        "v22 rejects v23 compile-phase evidence"
    );
    remove_compile_phase_evidence(&mut legacy);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V22);
    assert!(decoded.compile_phases().is_none());
    assert!(decoded.prepare_render_capsule_hit_count().is_some());

    let mut current = serde_json::to_value(report).unwrap();
    remove_compile_phase_evidence(&mut current);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&current).unwrap()).is_err(),
        "current report requires compile-phase evidence"
    );
}

#[test]
fn version_twenty_three_decodes_without_recurrent_capture_breakdown() {
    let report = phase_specialized_report();
    let mut legacy = serde_json::to_value(&report).unwrap();
    legacy["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V23);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err(),
        "v23 rejects v24 recurrent capture evidence"
    );
    remove_recurrent_capture_evidence(&mut legacy);
    let decoded =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V23);
    assert!(decoded.main().recurrent_state_count().is_none());
    assert!(
        decoded
            .compile_phases()
            .unwrap()
            .main_capture()
            .recurrent_capture()
            .is_none()
    );

    let mut current = serde_json::to_value(report).unwrap();
    remove_recurrent_capture_evidence(&mut current);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&current).unwrap()).is_err(),
        "v24 requires role-aligned recurrent capture evidence"
    );
}

#[test]
fn current_report_authenticates_compile_phase_partition_and_inventory() {
    let mut report = phase_specialized_report();
    report.compile_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(10));
    let phases = report.compile_phases.as_mut().unwrap();
    phases.objective_forward.wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    phases.autograd.wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases.optimizer_lowering.wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases.main_capture.wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .residual_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases.accumulation_capture.as_mut().unwrap().wall_time =
        BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases
        .accumulation_capture
        .as_mut()
        .unwrap()
        .recurrent_capture
        .as_mut()
        .unwrap()
        .residual_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    phases.residual_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(4));
    assert!(report.validate().is_ok());

    let valid = report.clone();
    report.compile_phases.as_mut().unwrap().compile_count = 2;
    assert!(report.validate().is_err(), "compile count is exact");

    let mut report = valid.clone();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .logical_schedule_item_count = Some(3);
    assert!(report.validate().is_err(), "main inventory is exact");

    let mut report = valid.clone();
    report.compile_phases.as_mut().unwrap().accumulation_capture = None;
    assert!(report.validate().is_err(), "phase presence is exact");

    let mut report = valid;
    report.compile_phases.as_mut().unwrap().residual_wall_time =
        BenchmarkDuration::from_duration(Duration::from_nanos(3));
    assert!(report.validate().is_err(), "compile partition is exact");

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .preview_schedule_count = 3;
    assert!(
        report.validate().is_err(),
        "main preview inventory is exact"
    );

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .cursor_projection_wall_time = Some(zero_duration());
    assert!(report.validate().is_err(), "main has no cursor projection");

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .accumulation_capture
        .as_mut()
        .unwrap()
        .recurrent_capture
        .as_mut()
        .unwrap()
        .cursor_projection_wall_time = None;
    assert!(
        report.validate().is_err(),
        "accumulation cursor projection is required"
    );

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .recurrent_state_count = 3;
    assert!(report.validate().is_err(), "main state inventory is exact");

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .alias_planning_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    assert!(
        report.validate().is_err(),
        "recurrent capture stage partition is exact"
    );

    let mut report = phase_specialized_report();
    report
        .compile_phases
        .as_mut()
        .unwrap()
        .objective_forward
        .wall_time = BenchmarkDuration {
        secs: u64::MAX,
        nanos: 999_999_999,
    };
    assert!(report.validate().is_err(), "compile phase sums are checked");
}

#[test]
fn version_twenty_four_authenticates_exact_auxiliary_recurrent_state_counts() {
    let report = phase_specialized_report();
    let main = report.main.clone();
    let accumulation = report.accumulation.clone().unwrap();
    let mut partial_flush = accumulation.clone();
    partial_flush.recurrent_state_count = Some(3);
    let mut zero_grad = accumulation.clone();
    zero_grad.recurrent_state_count = Some(2);
    let mut phases = report.compile_phases.unwrap();
    let mut partial_flush_phase = phases.accumulation_capture.unwrap();
    partial_flush_phase
        .recurrent_capture
        .as_mut()
        .unwrap()
        .recurrent_state_count = 3;
    phases.partial_flush = Some(partial_flush_phase);
    let mut zero_grad_phase = phases.accumulation_capture.unwrap();
    let zero_grad_capture = zero_grad_phase.recurrent_capture.as_mut().unwrap();
    zero_grad_capture.preview_schedule_count = 1;
    zero_grad_capture.recurrent_state_count = 2;
    phases.zero_grad = Some(zero_grad_phase);
    let validation_context = || CompilePhaseValidationContext {
        require_recurrent_capture: true,
        compile_wall_time: zero_duration(),
        recurrent_state_count: 4,
        programs: CompileProgramInventory {
            main: &main,
            accumulation: Some(&accumulation),
            partial_flush: Some(&partial_flush),
            zero_grad: Some(&zero_grad),
            evaluation: None,
        },
    };
    assert!(phases.validate(validation_context()).is_ok());

    let mut smaller_partial = phases.clone();
    smaller_partial
        .partial_flush
        .as_mut()
        .unwrap()
        .recurrent_capture
        .as_mut()
        .unwrap()
        .recurrent_state_count = 2;
    assert!(
        smaller_partial.validate(validation_context()).is_err(),
        "an in-range partial-flush count cannot replace its retained inventory"
    );

    let mut smaller_zero_grad = phases;
    smaller_zero_grad
        .zero_grad
        .as_mut()
        .unwrap()
        .recurrent_capture
        .as_mut()
        .unwrap()
        .recurrent_state_count = 1;
    assert!(
        smaller_zero_grad.validate(validation_context()).is_err(),
        "an in-range zero-grad count cannot replace its retained inventory"
    );
}

#[test]
fn version_twenty_four_rejects_invalid_and_overflowing_nested_durations() {
    let mut invalid = phase_specialized_report();
    invalid
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap()
        .alias_planning_wall_time = BenchmarkDuration {
        secs: 0,
        nanos: 1_000_000_000,
    };
    assert!(
        invalid.validate().is_err(),
        "a malformed nested duration must reject"
    );

    let mut overflowing = phase_specialized_report();
    let capture = overflowing
        .compile_phases
        .as_mut()
        .unwrap()
        .main_capture
        .recurrent_capture
        .as_mut()
        .unwrap();
    capture.alias_planning_wall_time = BenchmarkDuration {
        secs: u64::MAX,
        nanos: 999_999_999,
    };
    capture.final_schedule_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    assert!(
        overflowing.validate().is_err(),
        "nested duration summation must reject overflow"
    );
}

#[test]
fn current_report_attributes_an_auxiliary_post_main_compiler_tail() {
    let mut report = phase_specialized_report();
    report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(3));
    report.prepare_runtime_overhead_wall_time = Some(zero_duration());
    let main_timing = report.main.preparation_timing.as_mut().unwrap();
    main_timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    main_timing.compiler_process = main_timing.total;
    main_timing.compiler_process_total = Some(main_timing.total);
    let accumulation_timing = report
        .accumulation
        .as_mut()
        .unwrap()
        .preparation_timing
        .as_mut()
        .unwrap();
    accumulation_timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    accumulation_timing.compiler_process = accumulation_timing.total;
    accumulation_timing.compiler_process_total = Some(accumulation_timing.total);
    let timings = report.prepare_compiler_process_timings.as_mut().unwrap();
    timings[0].process_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    timings[1].permit_request_offset = zero_duration();
    timings[1].permit_wait = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    timings[1].process_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    report.prepare_compiler_critical_tail = Some(NativeTrainingCompilerCriticalTail {
        program_index: 1,
        native_identity: report.accumulation.as_ref().unwrap().native_identity,
        process: NativeTrainingCompilerProcessKind::Combined,
        finish_offset: BenchmarkDuration::from_duration(Duration::from_nanos(3)),
        post_main_tail: BenchmarkDuration::from_duration(Duration::from_nanos(2)),
    });
    assert!(report.validate().is_ok());
}

#[test]
fn current_report_accepts_an_empty_warm_cache_compiler_critical_path() {
    let mut report = phase_specialized_report();
    for program in [&mut report.main, report.accumulation.as_mut().unwrap()] {
        program.durable_artifact_cache_hit_count = 1;
        program.durable_artifact_cache_miss_count = 0;
        program.compiler_invocation_count = 0;
        program.combined_compile_link_count = Some(0);
    }
    report.prepare_compiler_process_count = Some(0);
    report.prepare_max_parallel_compiler_process_count = Some(0);
    report.prepare_compiler_process_timings = Some(Vec::new());
    report.prepare_compiler_critical_tail = None;
    assert!(report.validate().is_ok());

    report.prepare_compiler_process_timings = None;
    assert!(
        report.validate().is_err(),
        "v21 preserves authenticated overlap with an empty warm-cache timing inventory"
    );
}

#[test]
fn current_report_authenticates_dispatcher_executor_partition() {
    let report = phase_specialized_report();
    assert!(report.validate().is_ok());

    let mut json = serde_json::to_value(&report).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executor_host_wall_time");
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "v18 requires both sides of the executor partition"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["main_replay_native_dispatcher_wall_time"]["first"]["nanos"] = serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "dispatcher and executor-host time must partition executor time"
    );

    let mut json = serde_json::to_value(&report).unwrap();
    json["step_phases"]["first"]["executor_host_wall_time"]["nanos"] = serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "classified phase timing must match the global partition"
    );

    let mut json = serde_json::to_value(report).unwrap();
    json["main_replay_native_dispatcher_wall_time"]["steady_total"]["secs"] =
        serde_json::json!(u64::MAX);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "out-of-range dispatcher partitions fail closed"
    );
    assert!(
        NativeTrainingReplayTiming::from_durations(
            Duration::ZERO,
            &[Duration::MAX, Duration::from_nanos(1)],
        )
        .is_err(),
        "steady dispatcher duration aggregation rejects overflow"
    );
}

#[test]
fn current_report_authenticates_dispatch_segmentation_partition() {
    let report = phase_specialized_report();
    let segmentation = report.main().dispatch_segmentation().unwrap();
    assert_eq!(segmentation.segment_count(), 1);
    assert_eq!(segmentation.dispatch_reached_module_count(), 1);
    assert_eq!(segmentation.terminal_segment_count(), 1);
    assert_eq!(segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(segmentation.module_change_count(), 0);
    assert_eq!(segmentation.output_slot_alias_count(), 0);
    assert_eq!(segmentation.derived_slot_dependency_count(), 0);
    let accumulation = report
        .accumulation()
        .unwrap()
        .dispatch_segmentation()
        .unwrap();
    assert_eq!(accumulation.segment_count(), 2);
    assert_eq!(accumulation.dispatch_reached_module_count(), 2);
    assert_eq!(accumulation.terminal_segment_count(), 1);
    assert_eq!(accumulation.non_dispatch_boundary_count(), 0);
    assert_eq!(accumulation.module_change_count(), 1);
    assert_eq!(report.accumulation().unwrap().referenced_module_count(), 2);
    assert!(report.validate().is_ok());

    let all_elided = NativeCpuDispatchSegmentation {
        segment_count: 0,
        dispatch_reached_module_count: 0,
        terminal_segment_count: 0,
        non_dispatch_boundary_count: 0,
        module_change_count: 0,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    };
    assert!(
        all_elided.authenticates(1, 1),
        "structural validation permits rendered work omitted by authenticated elision"
    );

    let elided_only_suffix = NativeCpuDispatchSegmentation {
        segment_count: 1,
        dispatch_reached_module_count: 1,
        terminal_segment_count: 1,
        non_dispatch_boundary_count: 0,
        module_change_count: 0,
        output_slot_alias_count: 0,
        derived_slot_dependency_count: 0,
    };
    assert!(
        elided_only_suffix.authenticates(2, 2),
        "a referenced suffix module may contain only elided entries"
    );

    let mut missing_terminal = report.clone();
    let segmentation = missing_terminal
        .main
        .dispatch_segmentation
        .as_mut()
        .unwrap();
    segmentation.terminal_segment_count = 0;
    segmentation.output_slot_alias_count = 1;
    assert!(
        missing_terminal.validate().is_err(),
        "partition-preserving evidence must retain one terminal segment"
    );

    let mut fallback_boundary = report.clone();
    let segmentation = fallback_boundary
        .accumulation
        .as_mut()
        .unwrap()
        .dispatch_segmentation
        .as_mut()
        .unwrap();
    segmentation.non_dispatch_boundary_count = 1;
    segmentation.module_change_count = 0;
    assert!(
        fallback_boundary.validate().is_err(),
        "strict-native evidence cannot replace a module change with a fallback boundary"
    );

    let mut missing_module_change = report.clone();
    let segmentation = missing_module_change
        .accumulation
        .as_mut()
        .unwrap()
        .dispatch_segmentation
        .as_mut()
        .unwrap();
    segmentation.module_change_count = 0;
    segmentation.derived_slot_dependency_count = 1;
    assert!(
        missing_module_change.validate().is_err(),
        "two referenced modules require one module-change segment while preserving the partition"
    );

    let mut one_module_two_segments = report.clone();
    let segmentation = one_module_two_segments
        .main
        .dispatch_segmentation
        .as_mut()
        .unwrap();
    segmentation.segment_count = 2;
    segmentation.output_slot_alias_count = 1;
    assert!(one_module_two_segments.validate().is_ok());
    let segmentation = one_module_two_segments
        .main
        .dispatch_segmentation
        .as_mut()
        .unwrap();
    segmentation.dispatch_reached_module_count = 2;
    segmentation.module_change_count = 1;
    segmentation.output_slot_alias_count = 0;
    assert!(
        one_module_two_segments.validate().is_err(),
        "one referenced module cannot claim a partition-preserving module change"
    );

    let mut empty_tape_with_reached_module = report.clone();
    empty_tape_with_reached_module.main.dispatch_segmentation =
        Some(NativeCpuDispatchSegmentation {
            dispatch_reached_module_count: 1,
            ..all_elided
        });
    assert!(
        empty_tape_with_reached_module.validate().is_err(),
        "an empty dispatch tape cannot claim a reached module"
    );

    let mut missing = serde_json::to_value(&report).unwrap();
    missing["main"]
        .as_object_mut()
        .unwrap()
        .remove("dispatch_segmentation");
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&missing).unwrap()).is_err(),
        "v17 requires dispatch segmentation evidence"
    );

    let mut missing_reached_modules = serde_json::to_value(&report).unwrap();
    missing_reached_modules["main"]["dispatch_segmentation"]
        .as_object_mut()
        .unwrap()
        .remove("dispatch_reached_module_count");
    assert!(
        NativeTrainingReport::from_json_bytes(
            &serde_json::to_vec(&missing_reached_modules).unwrap()
        )
        .is_err(),
        "v17 requires the dispatch-reached module inventory"
    );
}

#[test]
fn current_report_authenticates_parallel_render_overlap_and_bound() {
    let mut report = phase_specialized_report();
    {
        let timing = report.main.preparation_timing.as_mut().unwrap();
        timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    }
    {
        let timing = report
            .accumulation
            .as_mut()
            .unwrap()
            .preparation_timing
            .as_mut()
            .unwrap();
        timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(2));
        timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    }
    report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(3));
    report.prepare_parallel_render_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    report.prepare_max_parallel_render_job_count = Some(2);
    report.prepare_render_capsule_hit_count = Some(0);
    report.prepare_render_capsule_miss_count = Some(2);
    report.prepare_local_render_job_count = Some(2);
    assert!(report.validate().is_ok());

    let valid = report.clone();
    report.prepare_max_parallel_render_job_count = Some(3);
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    report.prepare_parallel_render_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    report.prepare_parallel_render_overlap_wall_time = None;
    assert!(report.validate().is_err());

    let mut all_capsule_hits = valid;
    for program in std::iter::once(&mut all_capsule_hits.main)
        .chain(all_capsule_hits.accumulation.iter_mut())
        .chain(all_capsule_hits.partial_flush.iter_mut())
        .chain(&mut all_capsule_hits.zero_grad)
        .chain(&mut all_capsule_hits.evaluation)
    {
        let timing = program.preparation_timing.as_mut().unwrap();
        let render = timing.render.to_duration().unwrap();
        let total = timing.total.to_duration().unwrap();
        timing.render = zero_duration();
        timing.total = BenchmarkDuration::from_duration(total.checked_sub(render).unwrap());
    }
    all_capsule_hits.prepare_parallel_render_overlap_wall_time = Some(zero_duration());
    all_capsule_hits.prepare_max_parallel_render_job_count = Some(0);
    all_capsule_hits.prepare_render_capsule_hit_count = Some(2);
    all_capsule_hits.prepare_render_capsule_miss_count = Some(0);
    all_capsule_hits.prepare_local_render_job_count = Some(0);
    let program_total = std::iter::once(&all_capsule_hits.main)
        .chain(all_capsule_hits.accumulation.iter())
        .chain(all_capsule_hits.partial_flush.iter())
        .chain(&all_capsule_hits.zero_grad)
        .chain(&all_capsule_hits.evaluation)
        .try_fold(Duration::ZERO, |total, program| {
            total.checked_add(
                program
                    .preparation_timing
                    .as_ref()
                    .unwrap()
                    .total
                    .to_duration()
                    .unwrap(),
            )
        })
        .unwrap();
    let runtime_overhead = all_capsule_hits
        .prepare_runtime_overhead_wall_time
        .unwrap()
        .to_duration()
        .unwrap();
    let module_overlap = all_capsule_hits
        .prepare_parallel_module_overlap_wall_time
        .unwrap()
        .to_duration()
        .unwrap();
    all_capsule_hits.prepare_wall_time = BenchmarkDuration::from_duration(
        runtime_overhead
            .checked_add(program_total)
            .and_then(|total| total.checked_sub(module_overlap))
            .unwrap(),
    );
    assert!(all_capsule_hits.validate().is_ok());
    let exact_all_hits = all_capsule_hits.clone();
    let mut zero_duration_misses = exact_all_hits.clone();
    zero_duration_misses.prepare_render_capsule_hit_count = Some(0);
    zero_duration_misses.prepare_render_capsule_miss_count = Some(2);
    zero_duration_misses.prepare_local_render_job_count = Some(2);
    assert!(
        zero_duration_misses.validate().is_ok(),
        "explicit local-render counts do not depend on timer resolution"
    );
    let mut wrong_hits = exact_all_hits.clone();
    wrong_hits.prepare_render_capsule_hit_count = Some(1);
    assert!(wrong_hits.validate().is_err());
    let mut wrong_local = exact_all_hits.clone();
    wrong_local.prepare_local_render_job_count = Some(1);
    assert!(wrong_local.validate().is_err());
    let mut wrong_parallel = exact_all_hits;
    wrong_parallel.prepare_max_parallel_render_job_count = Some(1);
    assert!(wrong_parallel.validate().is_err());
}

#[test]
fn current_report_authenticates_exact_prefix_source_and_partition() {
    let report = phase_specialized_report();
    assert_eq!(report.main().unique_rendered_entry_count(), 2);
    assert_eq!(report.main().shared_prefix_entry_count(), 0);
    let accumulation = report.accumulation().unwrap();
    assert_eq!(accumulation.unique_rendered_entry_count(), 1);
    assert_eq!(accumulation.shared_prefix_entry_count(), 1);
    assert_eq!(accumulation.referenced_module_count(), 2);
    assert_eq!(accumulation.shared_prefix_source_program_index(), Some(0));
    assert_eq!(
        accumulation.shared_prefix_source_native_identity(),
        Some(report.main().native_identity())
    );
    assert!(report.validate().is_ok());

    let mut equal_identity = report.clone();
    let main_native_identity = equal_identity.main.native_identity;
    equal_identity
        .accumulation
        .as_mut()
        .unwrap()
        .native_identity = main_native_identity;
    equal_identity
        .prepare_compiler_process_timings
        .as_mut()
        .unwrap()[1]
        .native_identity = main_native_identity;
    let overlap = &mut equal_identity.prepare_module_overlaps.as_mut().unwrap()[0];
    overlap.native_identity = main_native_identity;
    overlap.evidence_identity = overlap.expected_evidence_identity(main_native_identity);
    let overlap = &mut equal_identity
        .prepare_program_pair_overlaps
        .as_mut()
        .unwrap()[0];
    overlap.target_native_identity = main_native_identity;
    overlap.evidence_identity = overlap.expected_evidence_identity();
    let unit = &mut equal_identity.prepare_translation_units.as_mut().unwrap()[1];
    unit.native_identity = main_native_identity;
    unit.evidence_identity = unit.expected_evidence_identity();
    assert!(
        equal_identity.validate().is_ok(),
        "the earlier program index disambiguates equal native identities"
    );

    let mut full_prefix = report.clone();
    let source_segmentation = full_prefix.main.dispatch_segmentation.unwrap();
    let accumulation_native_identity = {
        let accumulation = full_prefix.accumulation.as_mut().unwrap();
        accumulation.cache_hit_count = 2;
        accumulation.cache_miss_count = 0;
        accumulation.unique_rendered_entry_count = Some(0);
        accumulation.shared_prefix_entry_count = Some(2);
        accumulation.shared_prefix_source_bytes = Some(20);
        accumulation.loaded_module_count = 0;
        accumulation.referenced_module_count = Some(1);
        accumulation.durable_artifact_cache_miss_count = 0;
        accumulation.compiler_invocation_count = 0;
        accumulation.combined_compile_link_count = Some(0);
        accumulation.dispatch_segmentation = Some(source_segmentation);
        accumulation.native_identity()
    };
    full_prefix.prepare_compiler_process_count = Some(1);
    full_prefix
        .prepare_compiler_process_timings
        .as_mut()
        .unwrap()
        .truncate(1);
    let overlap = &mut full_prefix.prepare_module_overlaps.as_mut().unwrap()[0];
    overlap.contiguous_prefix_entry_count = 2;
    overlap.contiguous_prefix_source_bytes = 20;
    overlap.evidence_identity = overlap.expected_evidence_identity(main_native_identity);
    let overlap = &mut full_prefix.prepare_program_pair_overlaps.as_mut().unwrap()[0];
    overlap.contiguous_prefix_entry_count = 2;
    overlap.contiguous_prefix_source_bytes = 20;
    overlap.evidence_identity = overlap.expected_evidence_identity();
    full_prefix
        .prepare_translation_units
        .as_mut()
        .unwrap()
        .retain(|unit| unit.program_index == 0);
    assert!(
        full_prefix.validate().is_ok(),
        "a full exact prefix has no suffix compilation or module load"
    );
    assert_eq!(
        full_prefix
            .accumulation
            .as_ref()
            .unwrap()
            .dispatch_segmentation,
        full_prefix.main.dispatch_segmentation,
        "a full prefix reuses the source program's sealed segmentation"
    );

    for (field, value) in [
        ("unique_rendered_entry_count", serde_json::json!(0)),
        ("shared_prefix_entry_count", serde_json::json!(2)),
        ("referenced_module_count", serde_json::json!(0)),
        ("shared_prefix_source_program_index", serde_json::json!(1)),
        (
            "shared_prefix_source_native_identity",
            serde_json::json!(accumulation_native_identity),
        ),
    ] {
        let mut json = serde_json::to_value(&report).unwrap();
        json["accumulation"][field] = value;
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "tampered {field} must reject"
        );
    }

    let mut impossible = report;
    impossible.main.rendered_entry_count = 1;
    impossible.main.unique_rendered_entry_count = Some(1);
    let source_segmentation = impossible.main.dispatch_segmentation.unwrap();
    let accumulation = impossible.accumulation.as_mut().unwrap();
    accumulation.cache_hit_count = 2;
    accumulation.cache_miss_count = 0;
    accumulation.unique_rendered_entry_count = Some(0);
    accumulation.shared_prefix_entry_count = Some(2);
    accumulation.loaded_module_count = 0;
    accumulation.referenced_module_count = Some(1);
    accumulation.durable_artifact_cache_miss_count = 0;
    accumulation.compiler_invocation_count = 0;
    accumulation.combined_compile_link_count = Some(0);
    accumulation.dispatch_segmentation = Some(source_segmentation);
    impossible.prepare_compiler_process_count = Some(1);
    impossible
        .prepare_compiler_process_timings
        .as_mut()
        .unwrap()
        .truncate(1);
    assert!(
        impossible
            .accumulation
            .as_ref()
            .unwrap()
            .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
            .is_ok(),
        "the per-program partition is otherwise coherent"
    );
    assert!(
        impossible.validate().is_err(),
        "a shared prefix cannot exceed the named source program"
    );
}

#[test]
fn current_report_prefix_source_indices_compact_absent_optional_programs() {
    let mut report = phase_specialized_report();
    let evaluation = report.accumulation.take();
    report.evaluation = evaluation;
    report.evaluation.as_mut().unwrap().recurrent_state_count = None;
    let compile_phases = report.compile_phases.as_mut().unwrap();
    compile_phases.evaluation = compile_phases.accumulation_capture.take();
    compile_phases
        .evaluation
        .as_mut()
        .unwrap()
        .recurrent_capture = None;
    report.accumulation_replay_traffic = None;
    report.accumulation_replay_executed_native_item_count = None;
    report.accumulation_schedule_cache_keys.clear();
    report.step_phases = None;
    assert!(report.validate().is_ok());

    report
        .evaluation
        .as_mut()
        .unwrap()
        .shared_prefix_source_program_index = Some(1);
    assert!(
        report.validate().is_err(),
        "absent optional programs cannot leave a hole in the source ordinal"
    );
}

#[test]
fn zero_durations_and_unavailable_cpu_measurements_round_trip() {
    let report = zero_report();
    let bytes = report.to_json_bytes().unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["format_version"], NATIVE_TRAINING_REPORT_FORMAT_V10);
    for field in [
        "kernel_launch_count",
        "host_to_device",
        "device_to_host",
        "measured_peak_host_memory_bytes",
        "steady_microbatches_per_second",
    ] {
        assert!(json[field].is_null());
    }
    assert_eq!(
        json["main_replay_traffic"]["borrowed_recurrent_input_bytes"],
        16
    );
    assert_eq!(
        json["main_replay_traffic"]["borrowed_recurrent_output_bytes"],
        16
    );
    assert_eq!(json["main_replay_executed_native_item_count"], 1);
    assert_eq!(json["main"]["preparation_timing"]["layout"]["secs"], 0);
    assert_eq!(json["prepare_runtime_overhead_wall_time"]["nanos"], 0);
    assert_eq!(
        json["prepare_parallel_module_overlap_wall_time"]["nanos"],
        0
    );
    assert_eq!(json["prepare_compiler_process_count"], 1);
    assert_eq!(json["prepare_max_parallel_compiler_process_count"], 1);
    assert_eq!(json["main_replay_executor_wall_time"]["first"]["secs"], 0);
    assert_eq!(
        json["main_replay_recurrent_overhead_wall_time"]["steady"]["sample_count"],
        1
    );
    assert_eq!(json["step_phases"]["first"]["phase"], "accumulation_only");
    assert!(json["step_phases"]["warm_accumulation_only"].is_null());
    assert_eq!(
        json["step_phases"]["warm_optimizer_commit"]["wall_time"]["sample_count"],
        1
    );
    assert_eq!(
        NativeTrainingReport::from_json_bytes(&bytes).unwrap(),
        report
    );
}

#[test]
fn legacy_report_without_zero_grad_evidence_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(1);
    remove_step_phases(&mut json);
    json.as_object_mut().unwrap().remove("zero_grad");
    json.as_object_mut().unwrap().remove("main_replay_traffic");
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executed_native_item_count");
    remove_module_preparation(&mut json);
    remove_replay_phase_timing(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.zero_grad().is_none());
}

#[test]
fn version_two_report_without_replay_traffic_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V2);
    remove_step_phases(&mut json);
    json.as_object_mut().unwrap().remove("main_replay_traffic");
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executed_native_item_count");
    remove_module_preparation(&mut json);
    remove_replay_phase_timing(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.main_replay_traffic().is_none());
}

#[test]
fn version_three_report_without_execution_count_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V3);
    remove_step_phases(&mut json);
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executed_native_item_count");
    remove_module_preparation(&mut json);
    remove_replay_phase_timing(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.main_replay_executed_native_item_count().is_none());
    assert!(report.main_replay_traffic().is_some());
}

#[test]
fn version_four_report_without_module_preparation_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V4);
    remove_step_phases(&mut json);
    remove_module_preparation(&mut json);
    remove_replay_phase_timing(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(report.main().rendered_entry_count(), 0);
}

#[test]
fn version_five_report_without_replay_phase_timing_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V5);
    remove_step_phases(&mut json);
    remove_replay_phase_timing(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.main_replay_executor_wall_time().is_none());
    assert!(report.main_replay_recurrent_overhead_wall_time().is_none());
}

#[test]
fn version_six_report_without_preparation_phase_timing_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V6);
    remove_step_phases(&mut json);
    remove_preparation_phase_timing(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.main().preparation_timing().is_none());
    assert!(report.prepare_runtime_overhead_wall_time().is_none());
    assert!(report.main_replay_executor_wall_time().is_some());
}

#[test]
fn version_seven_report_without_parallel_module_overlap_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V7);
    remove_step_phases(&mut json);
    json.as_object_mut()
        .unwrap()
        .remove("prepare_parallel_module_overlap_wall_time");
    for field in [
        "prepare_compiler_process_overlap_wall_time",
        "prepare_compiler_process_count",
        "prepare_max_parallel_compiler_process_count",
    ] {
        json.as_object_mut().unwrap().remove(field);
    }
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.prepare_parallel_module_overlap_wall_time().is_none());
    assert!(report.prepare_runtime_overhead_wall_time().is_some());
}

#[test]
fn version_eight_report_with_exact_native_inventory_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V8);
    remove_step_phases(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert_eq!(report.main().native_item_count(), 2);
    assert_eq!(report.main().rendered_entry_count(), 2);
    assert_eq!(report.main_replay_executed_native_item_count(), Some(1));
}

#[test]
fn version_nine_report_without_step_phases_still_decodes() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V9);
    remove_step_phases(&mut json);
    let report =
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(report.step_phases().is_none());
    assert_eq!(report.main().rendered_entry_count(), 2);
}

#[test]
fn legacy_reports_reject_reduced_rendered_and_executed_inventories() {
    for version in [
        NATIVE_TRAINING_REPORT_FORMAT_V5,
        NATIVE_TRAINING_REPORT_FORMAT_V6,
        NATIVE_TRAINING_REPORT_FORMAT_V7,
        NATIVE_TRAINING_REPORT_FORMAT_V8,
    ] {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["format_version"] = serde_json::json!(version);
        remove_step_phases(&mut json);
        if version == NATIVE_TRAINING_REPORT_FORMAT_V5 {
            remove_replay_phase_timing(&mut json);
        }
        if version <= NATIVE_TRAINING_REPORT_FORMAT_V6 {
            remove_preparation_phase_timing(&mut json);
        } else if version == NATIVE_TRAINING_REPORT_FORMAT_V7 {
            json.as_object_mut()
                .unwrap()
                .remove("prepare_parallel_module_overlap_wall_time");
            for field in [
                "prepare_compiler_process_overlap_wall_time",
                "prepare_compiler_process_count",
                "prepare_max_parallel_compiler_process_count",
            ] {
                json.as_object_mut().unwrap().remove(field);
            }
        }
        json["main_replay_executed_native_item_count"] = serde_json::json!(2);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_ok(),
            "legacy v{version} exact native inventory did not decode"
        );
        json["main"]["rendered_entry_count"] = serde_json::json!(1);
        json["main_replay_executed_native_item_count"] = serde_json::json!(1);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "legacy v{version} accepted reduced rendered/executed inventories"
        );
    }
}

#[test]
fn single_program_report_accepts_grouped_physical_native_inventory() {
    let mut report = zero_report();
    report.main.logical_schedule_item_count = 4;
    report.main.native_item_count = 4;
    report.main.cache_miss_count = 4;
    report.main.rendered_entry_count = 1;
    report.schedule_cache_keys.extend([23, 29]);
    let bytes = report.to_json_bytes().unwrap();
    let decoded = NativeTrainingReport::from_json_bytes(&bytes).unwrap();
    assert_eq!(decoded.format_version, NATIVE_TRAINING_REPORT_FORMAT_V10);
    assert_eq!(decoded.main().native_item_count(), 4);
    assert_eq!(decoded.main().rendered_entry_count(), 1);
    assert_eq!(decoded.main_replay_executed_native_item_count(), Some(1));
}

#[test]
fn current_report_authenticates_preparation_phase_partition() {
    let mut report = zero_report();
    report.prepare_wall_time = BenchmarkDuration::from_duration(Duration::from_nanos(10));
    report.prepare_runtime_overhead_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(3)));
    report.prepare_parallel_module_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(2)));
    report.prepare_compiler_process_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    report.prepare_compiler_process_count = Some(2);
    report.prepare_max_parallel_compiler_process_count = Some(2);
    let timing = report.main.preparation_timing.as_mut().unwrap();
    timing.total = BenchmarkDuration::from_duration(Duration::from_nanos(7));
    timing.render = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    timing.compiler_process = BenchmarkDuration::from_duration(Duration::from_nanos(3));
    timing.module_load = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    timing.residual = BenchmarkDuration::from_duration(Duration::from_nanos(1));
    let mut evaluation = report.main.clone();
    evaluation.preparation_timing = Some(NativeTrainingPreparationTiming {
        total: BenchmarkDuration::from_duration(Duration::from_nanos(2)),
        layout: zero_duration(),
        render: zero_duration(),
        compiler_process: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
        compiler_process_total: None,
        linker_process: None,
        module_load: zero_duration(),
        residual: BenchmarkDuration::from_duration(Duration::from_nanos(1)),
    });
    report.evaluation = Some(evaluation);
    assert!(report.validate().is_ok());
    let valid = report.clone();

    report.main.preparation_timing.as_mut().unwrap().residual =
        BenchmarkDuration::from_duration(Duration::from_nanos(2));
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    report.prepare_runtime_overhead_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(6)));
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    report.prepare_parallel_module_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(8)));
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    report.prepare_compiler_process_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(2)));
    assert!(report.validate().is_err());

    let mut report = valid.clone();
    let main = report.main.preparation_timing.as_mut().unwrap();
    main.compiler_process = zero_duration();
    main.residual = BenchmarkDuration::from_duration(Duration::from_nanos(4));
    let evaluation = report
        .evaluation
        .as_mut()
        .unwrap()
        .preparation_timing
        .as_mut()
        .unwrap();
    evaluation.compiler_process = zero_duration();
    evaluation.residual = BenchmarkDuration::from_duration(Duration::from_nanos(2));
    assert!(report.validate().is_err());

    let mut report = valid;
    report.prepare_compiler_process_overlap_wall_time = Some(zero_duration());
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report.prepare_parallel_module_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    report.main.preparation_timing.as_mut().unwrap().total =
        BenchmarkDuration::from_duration(Duration::from_nanos(1));
    report.main.preparation_timing.as_mut().unwrap().residual =
        BenchmarkDuration::from_duration(Duration::from_nanos(1));
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report.prepare_compiler_process_count = Some(0);
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report.prepare_max_parallel_compiler_process_count = Some(3);
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report.prepare_compiler_process_overlap_wall_time =
        Some(BenchmarkDuration::from_duration(Duration::from_nanos(1)));
    assert!(report.validate().is_err());

    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["main"]
        .as_object_mut()
        .unwrap()
        .remove("preparation_timing");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn current_report_authenticates_replay_phase_partition() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executor_wall_time");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut report = zero_report();
    let total = BenchmarkDuration::from_duration(Duration::from_nanos(11));
    let executor = BenchmarkDuration::from_duration(Duration::from_nanos(7));
    let overhead = BenchmarkDuration::from_duration(Duration::from_nanos(4));
    report.first_replay_wall_time = total;
    report
        .main_replay_executor_wall_time
        .as_mut()
        .unwrap()
        .first = executor;
    report
        .main_replay_recurrent_overhead_wall_time
        .as_mut()
        .unwrap()
        .first = overhead;
    let classified = &mut report.step_phases.as_mut().unwrap().first;
    classified.total_wall_time = total;
    classified.executor_wall_time = executor;
    classified.recurrent_overhead_wall_time = overhead;
    assert!(report.validate().is_ok());

    report
        .main_replay_recurrent_overhead_wall_time
        .as_mut()
        .unwrap()
        .first = BenchmarkDuration::from_duration(Duration::from_nanos(5));
    assert!(report.validate().is_err());
}

#[test]
fn current_report_requires_complete_classified_step_phases() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json.as_object_mut().unwrap().remove("step_phases");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut report = zero_report();
    report
        .step_phases
        .as_mut()
        .unwrap()
        .warm_optimizer_commit
        .as_mut()
        .unwrap()
        .executor_wall_time
        .sample_count = 2;
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report
        .step_phases
        .as_mut()
        .unwrap()
        .warm_optimizer_commit
        .as_mut()
        .unwrap()
        .recurrent_overhead_total_wall_time =
        BenchmarkDuration::from_duration(Duration::from_nanos(1));
    assert!(report.validate().is_err());

    let mut legacy = serde_json::to_value(zero_report()).unwrap();
    legacy["format_version"] = serde_json::json!(NATIVE_TRAINING_REPORT_FORMAT_V9);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&legacy).unwrap()).is_err());
}

#[test]
fn current_report_authenticates_module_preparation_evidence() {
    let report = zero_report();
    assert_eq!(report.main().rendered_entry_count(), 2);
    assert_eq!(report.main().loaded_module_count(), 1);
    assert_eq!(report.main().durable_artifact_cache_miss_count(), 1);
    assert_eq!(report.main().compiler_invocation_count(), 1);

    for field in [
        "rendered_entry_count",
        "loaded_module_count",
        "durable_artifact_cache_miss_count",
        "compiler_invocation_count",
    ] {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["main"][field] = serde_json::json!(0);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
            "tampered {field} must reject"
        );
    }
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json["main"]["durable_artifact_cache_hit_count"] = serde_json::json!(1);
    assert!(
        NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err(),
        "one loaded artifact cannot be both a durable hit and miss"
    );

    for (hit_count, reason) in [(1, "durable hit"), (0, "in-memory reuse")] {
        let mut json = serde_json::to_value(zero_report()).unwrap();
        json["main"]["durable_artifact_cache_hit_count"] = serde_json::json!(hit_count);
        json["main"]["durable_artifact_cache_miss_count"] = serde_json::json!(0);
        json["main"]["compiler_invocation_count"] = serde_json::json!(0);
        json["prepare_compiler_process_count"] = serde_json::json!(0);
        json["prepare_max_parallel_compiler_process_count"] = serde_json::json!(0);
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_ok(),
            "{reason} module preparation must remain representable"
        );
    }
}

#[test]
fn current_program_distinguishes_zero_domain_items_from_missing_rendered_work() {
    let mut program = phase_specialized_report().main;
    program.rendered_entry_count = 1;
    program.unique_rendered_entry_count = Some(1);
    assert!(
        program
            .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
            .is_ok(),
        "one rendered entry plus one zero-domain item is valid"
    );

    program.rendered_entry_count = 0;
    program.loaded_module_count = 0;
    program.referenced_module_count = Some(0);
    program.unique_rendered_entry_count = Some(0);
    program.durable_artifact_cache_miss_count = 0;
    program.compiler_invocation_count = 0;
    assert!(
        program
            .validate(NATIVE_TRAINING_REPORT_FORMAT_VERSION)
            .is_err(),
        "nonempty logical native inventory cannot claim no rendered work"
    );
}

#[test]
fn current_report_requires_exact_recurrent_traffic() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json.as_object_mut().unwrap().remove("main_replay_traffic");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut report = zero_report();
    report.main_replay_traffic = Some(NativeCpuReplayTraffic::new(2, 12, 16, 15));
    assert!(report.validate().is_err());
}

#[test]
fn current_report_requires_bounded_execution_count() {
    let mut json = serde_json::to_value(zero_report()).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("main_replay_executed_native_item_count");
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());

    let mut report = zero_report();
    report.main_replay_executed_native_item_count = Some(3);
    assert!(report.validate().is_err());

    let mut report = zero_report();
    report.main.native_item_count = 3;
    report.main.cache_miss_count = 3;
    report.schedule_cache_keys.push(23);
    report.main_replay_executed_native_item_count = Some(3);
    assert!(report.validate().is_err());
}

#[test]
fn current_report_rejects_incomplete_auxiliary_inventory() {
    let mut report = zero_report();
    report.zero_grad = Some(report.main.clone());
    assert!(report.validate().is_err());
}

#[test]
fn report_rejects_invented_cpu_measurements_and_throughput() {
    let mut report = zero_report();
    report.kernel_launch_count = Some(0);
    assert!(report.validate().is_err());

    let mut report = zero_report();
    let elapsed = BenchmarkDuration::from_duration(Duration::from_nanos(10));
    set_single_steady_replay_duration(&mut report, elapsed);
    assert!(report.validate().is_ok());
    let mut json = serde_json::to_value(report).unwrap();
    json["steady_microbatches_per_second"] = serde_json::json!(1.0);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn positive_derived_rate_round_trips_exactly_and_rejects_tampering() {
    let mut report = zero_report();
    let elapsed = BenchmarkDuration::from_duration(Duration::from_nanos(63));
    set_single_steady_replay_duration(&mut report, elapsed);

    let bytes = report.to_json_bytes().unwrap();
    assert_eq!(
        NativeTrainingReport::from_json_bytes(&bytes).unwrap(),
        report
    );
    let mut json = serde_json::to_value(&report).unwrap();
    json["steady_microbatches_per_second"] = serde_json::json!(1.0);
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&json).unwrap()).is_err());
}

#[test]
fn report_rejects_infeasible_steady_totals() {
    let mut report = zero_report();
    let positive = BenchmarkDuration::from_duration(Duration::from_nanos(10));
    report.steady_replay_total_wall_time = positive;
    report.steady_microbatches_per_second = rate_from_total(1, positive).unwrap();
    assert!(NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&report).unwrap()).is_err());

    let min = BenchmarkDuration::from_duration(Duration::from_nanos(10));
    let median = BenchmarkDuration::from_duration(Duration::from_nanos(20));
    let max = BenchmarkDuration::from_duration(Duration::from_nanos(30));
    report.successful_replay_count = 4;
    report.checkpoint = None;
    report.steady_replay_wall_time = BenchmarkLatencySummary {
        sample_count: 3,
        min,
        nearest_rank_p50: median,
        nearest_rank_p95: max,
        max,
    };
    for nanos in [40, 80] {
        let total = BenchmarkDuration::from_duration(Duration::from_nanos(nanos));
        report.steady_replay_total_wall_time = total;
        report.steady_microbatches_per_second = rate_from_total(3, total).unwrap();
        assert!(
            NativeTrainingReport::from_json_bytes(&serde_json::to_vec(&report).unwrap()).is_err()
        );
    }
}
