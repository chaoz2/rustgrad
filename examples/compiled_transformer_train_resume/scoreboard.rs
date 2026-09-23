use super::*;

pub(super) fn run_native_cpu_scoreboard() -> std::result::Result<(), Box<dyn Error>> {
    const SAMPLES: u64 = 3;
    const EXPECTED_LOSS_WEIGHTS: [u64; SAMPLES as usize] = [5, 3, 3];
    const EXPECTED_MAIN_MODULE_DISPATCHES: usize = 1;
    const EXPECTED_ACCUMULATION_MODULE_DISPATCHES: usize = 2;
    // The module exposes 37 parameter traversal entries. The tied LM head
    // deduplicates with tokens.weight, and positions.weight is policy-frozen.
    const EXPECTED_ADAMW_UPDATE_GROUPS: usize = 35;
    // Parameter/m1/m2 plus optimizer step are definitionally unchanged.
    // Prepared schedule simplification may prove additional accumulator
    // pass-throughs; index/token/loss/dropout must still receive new banks.
    const GUARANTEED_RETAINED_RECURRENT_STATES: u64 = (EXPECTED_ADAMW_UPDATE_GROUPS * 3 + 1) as u64;
    const REMAINING_RECURRENT_STATES: u64 = (EXPECTED_ADAMW_UPDATE_GROUPS + 4) as u64;
    const MANDATORY_REPLACED_RECURRENT_STATES: u64 = 4;
    const EXPECTED_MAIN_RENDERED_ENTRIES: usize = 787 - EXPECTED_ADAMW_UPDATE_GROUPS * 3;
    const EXPECTED_ACCUMULATION_RENDERED_ENTRIES: usize = 465;
    const EXPECTED_MAIN_EXECUTED_ENTRIES: u64 = 681;
    const EXPECTED_ACCUMULATION_EXECUTED_ENTRIES: u64 = 358;
    const EXPECTED_RECURRENT_STATES: usize = 145;
    const EXPECTED_RECURRENT_BYTES: u64 = 1_924;
    const EXPECTED_SHARED_ACCUMULATION_PREFIX: usize = 319;
    const EXPECTED_COLD_COMPILER_PROCESSES: usize = 6;
    // One accumulator per update group, plus loss numerator, index, and token count.
    const EXPECTED_ZERO_GRAD_ENTRIES: usize = EXPECTED_ADAMW_UPDATE_GROUPS + 3;

    let source = FileResumeTransformer::new(7)?;
    let optimized_parameters = source
        .trainable_parameters()?
        .into_iter()
        .filter(|(name, _)| name != FILE_RESUME_POLICY_FROZEN)
        .collect::<Vec<_>>();
    assert_eq!(optimized_parameters.len(), EXPECTED_ADAMW_UPDATE_GROUPS);
    let optimized_parameter_bytes = optimized_parameters
        .iter()
        .map(|(_, parameter)| {
            parameter
                .value()
                .map(|value| value.len() * value.dtype().itemsize())
        })
        .sum::<Result<usize>>()?;
    let guaranteed_retained_recurrent_bytes =
        u64::try_from(optimized_parameter_bytes * 3 + DType::U64.itemsize())?;
    let remaining_recurrent_bytes = u64::try_from(
        optimized_parameter_bytes + DType::U64.itemsize() * 3 + DType::F32.itemsize(),
    )?;
    let mandatory_replaced_recurrent_bytes =
        u64::try_from(DType::U64.itemsize() * 3 + DType::F32.itemsize())?;
    let schedule = CompiledMultiStepLr::new(0.05, 0.5, [1])?;
    let builds = Cell::new(0);
    let compile_started = Instant::now();
    let plan = CompiledAdamWPlan::compile_module_graph_with_dropout(
        scoreboard_config(schedule)?,
        dropout_config(),
        &source,
        |model, graph, inputs, dropout| {
            builds.set(builds.get() + 1);
            build_scoreboard(model, graph, inputs, dropout)
        },
    )?;
    let compile_wall_time = compile_started.elapsed();
    assert_eq!(builds.get(), 1, "the training graph must compile once");
    let inspection = plan.inspection()?;
    let compile_phases = inspection
        .compile_phases()
        .expect("fresh compilation retains phase evidence");
    assert_eq!(compile_phases.compile_count(), 1);
    assert_eq!(
        compile_phases.main_capture().logical_schedule_item_count(),
        Some(inspection.main().1.schedule_item_count)
    );
    assert!(compile_phases.accumulation_capture().is_some());
    assert!(compile_phases.partial_flush().is_some());
    assert!(compile_phases.zero_grad().is_some());
    assert!(compile_phases.evaluation().is_none());
    assert!(compile_phases.measured_wall_time().unwrap() <= compile_wall_time);
    assert_eq!(
        u64::try_from(inspection.recurrent_state_count())?,
        GUARANTEED_RETAINED_RECURRENT_STATES + REMAINING_RECURRENT_STATES
    );
    assert_eq!(
        inspection.recurrent_state_count(),
        EXPECTED_RECURRENT_STATES
    );
    assert_eq!(
        u64::try_from(inspection.recurrent_state_bytes())?,
        EXPECTED_RECURRENT_BYTES
    );

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor)
        .vectorized(true)
        .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition);
    let prepare_started = Instant::now();
    let mut session = plan.prepare(&target)?;
    let prepare_wall_time = prepare_started.elapsed();
    let main_preparation = session.preparation_report().main();
    assert_eq!(main_preparation.native_item_count(), 787);
    assert_eq!(
        main_preparation.work().rendered_entry_count(),
        EXPECTED_MAIN_RENDERED_ENTRIES
    );
    assert_eq!(main_preparation.work().loaded_module_count(), 1);
    assert_eq!(main_preparation.work().referenced_module_count(), 1);
    assert_eq!(
        main_preparation.work().unique_rendered_entry_count(),
        EXPECTED_MAIN_RENDERED_ENTRIES
    );
    assert_eq!(main_preparation.work().shared_prefix_entry_count(), 0);
    assert!(main_preparation.work().rendered_source_bytes() > 0);
    assert_eq!(main_preparation.work().shared_prefix_source_bytes(), 0);
    assert_eq!(
        main_preparation.work().unique_rendered_source_bytes(),
        main_preparation.work().rendered_source_bytes()
    );
    assert!(main_preparation.work().compiler_invocation_count() <= 3);
    if main_preparation.work().compiler_invocation_count() != 0 {
        assert_eq!(main_preparation.work().combined_compile_link_count(), 0);
        assert_eq!(main_preparation.work().object_compile_count(), 2);
        assert_eq!(main_preparation.work().linker_invocation_count(), 1);
    }
    let accumulation_preparation = session
        .preparation_report()
        .accumulation()
        .expect("scoreboard configuration captures accumulation-only replay");
    assert_ne!(
        accumulation_preparation.capture_identity(),
        main_preparation.capture_identity()
    );
    assert!(accumulation_preparation.native_item_count() < main_preparation.native_item_count());
    assert!(
        accumulation_preparation.work().rendered_entry_count()
            < main_preparation.work().rendered_entry_count()
    );
    assert_eq!(
        accumulation_preparation.work().rendered_entry_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES
    );
    assert_eq!(accumulation_preparation.work().loaded_module_count(), 1);
    assert_eq!(accumulation_preparation.work().referenced_module_count(), 2);
    assert_eq!(
        accumulation_preparation
            .work()
            .unique_rendered_entry_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert_eq!(
        accumulation_preparation.work().shared_prefix_entry_count(),
        EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert!(accumulation_preparation.work().shared_prefix_source_bytes() > 0);
    assert_eq!(
        accumulation_preparation.work().shared_prefix_source_bytes()
            + accumulation_preparation
                .work()
                .unique_rendered_source_bytes(),
        accumulation_preparation.work().rendered_source_bytes()
    );
    assert_eq!(
        accumulation_preparation.cache_hit_count(),
        EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert_eq!(
        accumulation_preparation.cache_miss_count(),
        EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
    );
    assert!(accumulation_preparation.work().compiler_invocation_count() <= 1);
    let partial_preparation = session
        .preparation_report()
        .partial_flush()
        .expect("scoreboard configuration captures partial flush");
    assert_eq!(partial_preparation.native_item_count(), 357);
    assert_eq!(
        partial_preparation.work().rendered_entry_count(),
        357 - EXPECTED_ADAMW_UPDATE_GROUPS * 3
    );
    assert_eq!(partial_preparation.work().shared_prefix_entry_count(), 0);
    assert_eq!(
        partial_preparation.work().unique_rendered_entry_count(),
        partial_preparation.work().rendered_entry_count()
    );
    assert_eq!(partial_preparation.work().referenced_module_count(), 1);
    assert!(partial_preparation.work().compiler_invocation_count() <= 1);
    assert_eq!(
        session
            .preparation_report()
            .zero_grad()
            .expect("scoreboard configuration captures zero grad")
            .native_item_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    let zero_grad_preparation = session
        .preparation_report()
        .zero_grad()
        .expect("scoreboard configuration captures zero grad");
    assert_eq!(
        zero_grad_preparation.work().rendered_entry_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert_eq!(
        zero_grad_preparation.work().unique_rendered_entry_count(),
        EXPECTED_ZERO_GRAD_ENTRIES
    );
    assert_eq!(zero_grad_preparation.work().referenced_module_count(), 1);
    assert!(zero_grad_preparation.work().compiler_invocation_count() <= 1);
    assert_eq!(
        [
            main_preparation,
            accumulation_preparation,
            partial_preparation,
            zero_grad_preparation,
        ]
        .into_iter()
        .map(|program| program.work().unique_rendered_entry_count())
        .sum::<usize>(),
        1_118
    );
    assert!(
        session.preparation_report().compiler_process_count() <= EXPECTED_COLD_COMPILER_PROCESSES
    );
    assert!(
        session
            .preparation_report()
            .max_parallel_compiler_process_count()
            <= 2
    );
    assert!((1..=2).contains(&session.preparation_report().max_parallel_render_job_count()));
    if env::var_os("RUSTGRAD_REQUIRE_COLD_NATIVE_SCOREBOARD").is_some() {
        assert_eq!(
            session.preparation_report().compiler_process_count(),
            EXPECTED_COLD_COMPILER_PROCESSES
        );
        assert_eq!(
            session
                .preparation_report()
                .max_parallel_compiler_process_count(),
            2
        );
        assert!(
            session
                .preparation_report()
                .compiler_process_overlap_wall_time()
                > Duration::ZERO
        );
        assert!(
            session
                .preparation_report()
                .parallel_module_overlap_wall_time()
                > Duration::ZERO
        );
        assert_eq!(
            session.preparation_report().max_parallel_render_job_count(),
            2
        );
        assert!(
            session
                .preparation_report()
                .parallel_render_overlap_wall_time()
                > Duration::ZERO
        );
        assert_eq!(main_preparation.work().object_compile_count(), 2);
        assert_eq!(main_preparation.work().linker_invocation_count(), 1);
        for preparation in [
            accumulation_preparation,
            partial_preparation,
            zero_grad_preparation,
        ] {
            assert_eq!(preparation.work().combined_compile_link_count(), 1);
            assert_eq!(preparation.work().object_compile_count(), 0);
            assert_eq!(preparation.work().linker_invocation_count(), 0);
        }
    }
    let main_segmentation = *session.preparation_report().main().dispatch_segmentation();
    let expected_module_dispatches = u64::try_from(EXPECTED_MAIN_MODULE_DISPATCHES)?;
    assert_eq!(
        main_segmentation.segment_count(),
        expected_module_dispatches
    );
    assert_eq!(main_segmentation.dispatch_reached_module_count(), 1);
    assert_eq!(main_segmentation.terminal_segment_count(), 1);
    assert_eq!(main_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(main_segmentation.module_change_count(), 0);
    assert_eq!(main_segmentation.output_slot_alias_count(), 0);
    assert_eq!(main_segmentation.derived_slot_dependency_count(), 0);
    let accumulation_segmentation = *accumulation_preparation.dispatch_segmentation();
    assert_eq!(accumulation_segmentation.terminal_segment_count(), 1);
    assert_eq!(accumulation_segmentation.dispatch_reached_module_count(), 2);
    assert_eq!(accumulation_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(
        accumulation_segmentation.segment_count(),
        u64::try_from(EXPECTED_ACCUMULATION_MODULE_DISPATCHES)?
    );
    assert_eq!(
        accumulation_segmentation.module_change_count(),
        u64::try_from(accumulation_preparation.work().referenced_module_count() - 1)?,
        "the accumulation preparation crosses once from its shared prefix to its suffix module"
    );
    assert_eq!(accumulation_segmentation.output_slot_alias_count(), 0);
    assert_eq!(accumulation_segmentation.derived_slot_dependency_count(), 0);
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        session.preparation_report(),
        compile_wall_time,
        prepare_wall_time,
    )?;
    let recurrent_state_bytes = u64::try_from(inspection.recurrent_state_bytes())?;
    assert_eq!(
        recurrent_state_bytes,
        guaranteed_retained_recurrent_bytes + remaining_recurrent_bytes
    );
    let mut stable_accumulation_executed_native_items = None;
    let mut committed_executed_native_items = None;
    for replay in 1..=SAMPLES {
        let batch = masked_batch(replay)?;
        assert_eq!(
            loss_mask_weight(&batch.loss_mask),
            EXPECTED_LOSS_WEIGHTS[(replay - 1) as usize]
        );
        let step = session.step_batch_commit_only_scheduled(batch)?;
        assert_eq!(step.did_update(), replay == ACCUMULATION_STEPS);
        assert!(step.outputs().is_empty());
        assert!(step.loss().values()[0].is_finite());
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        if let Some(clip) = step.clip_report() {
            assert!(clip.is_finite());
            assert!(clip.did_clip().is_some());
        }
        if let Some(window) = step.window_loss_report() {
            assert!(window.is_finite());
            assert_eq!(window.microbatch_count(), ACCUMULATION_STEPS);
            assert_eq!(
                window.loss_weight(),
                EXPECTED_LOSS_WEIGHTS.iter().copied().sum::<u64>()
            );
        }
        let report = step.report();
        assert_eq!(
            report.traffic().materialized_egress_count(),
            if step.did_update() { 5 } else { 1 }
        );
        assert_eq!(
            report.traffic().materialized_egress_bytes(),
            if step.did_update() { 24 } else { 4 }
        );
        let executed = report.executed_native_item_count();
        assert!(executed > 0);
        assert!(executed <= report.native_item_count());
        if step.did_update() {
            assert_eq!(
                u64::try_from(report.module_dispatch_count())?,
                main_segmentation.segment_count(),
                "the commit program must retain its authenticated safe-segment partition"
            );
        }
        assert_eq!(report.module_dispatched_native_item_count(), executed);
        assert!(report.module_dispatch_count() < executed);
        if step.did_update() {
            assert!(committed_executed_native_items.replace(executed).is_none());
        } else if let Some(expected) = stable_accumulation_executed_native_items {
            assert_eq!(executed, expected);
        } else {
            stable_accumulation_executed_native_items = Some(executed);
        }
        scoreboard.record_step(&step)?;
    }

    let checkpoint_started = Instant::now();
    let checkpoint = session.checkpoint()?;
    let checkpoint_wall_time = checkpoint_started.elapsed();
    scoreboard.observe_checkpoint(&checkpoint, checkpoint_wall_time)?;
    let report = scoreboard.report()?;
    let reported_compile = report.compile_phases().expect("v23 reports compile phases");
    assert_eq!(reported_compile.compile_count(), 1);
    assert!(reported_compile.evaluation().is_none());
    let executed_native_items = report
        .main_replay_executed_native_item_count()
        .expect("current native CPU scoreboard reports executed JIT items");
    let expected_main_rendered_entries = u64::try_from(EXPECTED_MAIN_RENDERED_ENTRIES)?;
    let expected_executed_native_items = expected_main_rendered_entries
        .checked_sub(1)
        .expect("the main program has one nonexecuted native entry");
    assert_eq!(executed_native_items, expected_executed_native_items);
    assert_eq!(executed_native_items, EXPECTED_MAIN_EXECUTED_ENTRIES);
    assert_eq!(
        report.main().rendered_entry_count(),
        expected_main_rendered_entries
    );
    assert_eq!(report.main().loaded_module_count(), 1);
    assert_eq!(report.main().referenced_module_count(), 1);
    assert_eq!(
        report.main().unique_rendered_entry_count(),
        expected_main_rendered_entries
    );
    assert_eq!(report.main().shared_prefix_entry_count(), 0);
    let recorded_segmentation = report
        .main()
        .dispatch_segmentation()
        .expect("current scoreboard reports dispatch segmentation");
    assert_eq!(
        recorded_segmentation.segment_count(),
        u64::try_from(EXPECTED_MAIN_MODULE_DISPATCHES)?
    );
    assert_eq!(recorded_segmentation.dispatch_reached_module_count(), 1);
    assert_eq!(recorded_segmentation.terminal_segment_count(), 1);
    assert_eq!(recorded_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(recorded_segmentation.module_change_count(), 0);
    assert_eq!(recorded_segmentation.output_slot_alias_count(), 0);
    assert_eq!(recorded_segmentation.derived_slot_dependency_count(), 0);
    assert!(report.main().compiler_invocation_count() <= 3);
    if report.main().compiler_invocation_count() != 0 {
        assert_eq!(report.main().combined_compile_link_count(), 0);
        assert_eq!(report.main().object_compile_count(), 2);
        assert_eq!(report.main().linker_invocation_count(), 1);
    }
    let accumulation_program = report
        .accumulation()
        .expect("current scoreboard reports accumulation preparation");
    assert_eq!(
        accumulation_program.rendered_entry_count(),
        u64::try_from(EXPECTED_ACCUMULATION_RENDERED_ENTRIES)?
    );
    assert_eq!(accumulation_program.loaded_module_count(), 1);
    assert_eq!(accumulation_program.referenced_module_count(), 2);
    assert_eq!(
        accumulation_program.unique_rendered_entry_count(),
        u64::try_from(
            EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
        )?
    );
    assert_eq!(
        accumulation_program.shared_prefix_entry_count(),
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert_eq!(
        accumulation_program.cache_hit_count(),
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert_eq!(
        accumulation_program.cache_miss_count(),
        u64::try_from(
            EXPECTED_ACCUMULATION_RENDERED_ENTRIES - EXPECTED_SHARED_ACCUMULATION_PREFIX
        )?
    );
    assert_eq!(
        accumulation_program.shared_prefix_source_program_index(),
        Some(0)
    );
    assert_eq!(
        accumulation_program.shared_prefix_source_native_identity(),
        Some(report.main().native_identity())
    );
    let accumulation_segmentation = accumulation_program
        .dispatch_segmentation()
        .expect("current scoreboard reports accumulation dispatch segmentation");
    assert_eq!(accumulation_segmentation.terminal_segment_count(), 1);
    assert_eq!(accumulation_segmentation.dispatch_reached_module_count(), 2);
    assert_eq!(accumulation_segmentation.non_dispatch_boundary_count(), 0);
    assert_eq!(
        accumulation_segmentation.segment_count(),
        u64::try_from(EXPECTED_ACCUMULATION_MODULE_DISPATCHES)?
    );
    assert_eq!(
        accumulation_segmentation.module_change_count(),
        accumulation_program.referenced_module_count() - 1,
        "the accumulation program crosses exactly once from its shared prefix to its suffix module"
    );
    assert_eq!(accumulation_segmentation.output_slot_alias_count(), 0);
    assert_eq!(accumulation_segmentation.derived_slot_dependency_count(), 0);
    let program_prepare_wall_time = [
        Some(report.main()),
        report.accumulation(),
        report.partial_flush(),
        report.zero_grad(),
        report.evaluation(),
    ]
    .into_iter()
    .flatten()
    .fold(Duration::ZERO, |total, program| {
        let timing = program
            .preparation_timing()
            .expect("current native CPU scoreboard reports every program preparation phase");
        total
            .checked_add(
                timing
                    .total()
                    .to_duration()
                    .expect("program preparation total is representable"),
            )
            .expect("program preparation totals fit")
    });
    assert_eq!(
        program_prepare_wall_time
            .checked_sub(
                report
                    .prepare_parallel_module_overlap_wall_time()
                    .expect("current native CPU scoreboard reports parallel module overlap")
                    .to_duration()?,
            )
            .expect("parallel module overlap fits program preparation")
            .checked_sub(
                report
                    .prepare_parallel_render_overlap_wall_time()
                    .expect("current native CPU scoreboard reports parallel render overlap")
                    .to_duration()?,
            )
            .expect("parallel render overlap fits program preparation")
            .checked_add(
                report
                    .prepare_runtime_overhead_wall_time()
                    .expect("current native CPU scoreboard reports whole-prepare overhead")
                    .to_duration()?,
            )
            .expect("whole preparation phases fit"),
        report.prepare_wall_time().to_duration()?
    );
    assert_eq!(report.successful_replay_count(), SAMPLES);
    let step_phases = report
        .step_phases()
        .expect("phase-aware scoreboard reports successful main-step classes");
    assert_eq!(
        step_phases.first().phase(),
        rustgrad::NativeTrainingStepPhase::AccumulationOnly
    );
    assert_eq!(
        step_phases
            .first()
            .native_dispatcher_wall_time()
            .expect("first phase reports native dispatcher time")
            .to_duration()?
            .checked_add(
                step_phases
                    .first()
                    .executor_host_wall_time()
                    .expect("first phase reports executor host time")
                    .to_duration()?
            )
            .expect("first classified executor durations fit"),
        step_phases.first().executor_wall_time().to_duration()?
    );
    assert_eq!(
        step_phases
            .warm_accumulation_only()
            .expect("the second replay remains accumulation-only")
            .sample_count(),
        1
    );
    assert_eq!(
        step_phases
            .warm_optimizer_commit()
            .expect("the third replay commits the accumulated window")
            .sample_count(),
        1
    );
    for phase in [
        step_phases.warm_accumulation_only(),
        step_phases.warm_optimizer_commit(),
    ]
    .into_iter()
    .flatten()
    {
        assert_eq!(
            phase
                .native_dispatcher_total_wall_time()
                .expect("current phase reports native dispatcher time")
                .to_duration()?
                .checked_add(
                    phase
                        .executor_host_total_wall_time()
                        .expect("current phase reports executor host time")
                        .to_duration()?
                )
                .expect("classified executor phase durations fit"),
            phase.executor_total_wall_time().to_duration()?
        );
        assert_eq!(
            phase
                .executor_total_wall_time()
                .to_duration()?
                .checked_add(phase.recurrent_overhead_total_wall_time().to_duration()?)
                .expect("classified warm replay phase durations fit"),
            phase.total_wall_time().to_duration()?
        );
    }
    let executor_timing = report
        .main_replay_executor_wall_time()
        .expect("current native CPU scoreboard reports sealed executor timing");
    let dispatcher_timing = report
        .main_replay_native_dispatcher_wall_time()
        .expect("current native CPU scoreboard reports native dispatcher timing");
    let executor_host_timing = report
        .main_replay_executor_host_wall_time()
        .expect("current native CPU scoreboard reports executor host timing");
    let overhead_timing = report
        .main_replay_recurrent_overhead_wall_time()
        .expect("current native CPU scoreboard reports recurrent overhead timing");
    assert_eq!(
        dispatcher_timing
            .first()
            .to_duration()?
            .checked_add(executor_host_timing.first().to_duration()?)
            .expect("first executor phase durations fit"),
        executor_timing.first().to_duration()?
    );
    assert_eq!(
        dispatcher_timing
            .steady_total()
            .to_duration()?
            .checked_add(executor_host_timing.steady_total().to_duration()?)
            .expect("steady executor phase durations fit"),
        executor_timing.steady_total().to_duration()?
    );
    assert_eq!(
        executor_timing
            .first()
            .to_duration()?
            .checked_add(overhead_timing.first().to_duration()?)
            .expect("first replay phase durations fit"),
        report.first_replay_wall_time().to_duration()?
    );
    assert_eq!(
        executor_timing
            .steady_total()
            .to_duration()?
            .checked_add(overhead_timing.steady_total().to_duration()?)
            .expect("steady replay phase durations fit"),
        report.steady_replay_total_wall_time().to_duration()?
    );
    assert_eq!(
        Some(usize::try_from(executed_native_items)?),
        committed_executed_native_items
    );
    assert!(executed_native_items <= report.main().native_item_count());
    let replay_traffic = report
        .main_replay_traffic()
        .expect("current native CPU scoreboard reports replay traffic");
    assert_eq!(replay_traffic.external_input_import_count(), 0);
    assert_eq!(replay_traffic.external_input_import_bytes(), 0);
    assert_eq!(replay_traffic.materialized_egress_count(), 5);
    assert_eq!(replay_traffic.materialized_egress_bytes(), 24);
    assert_eq!(
        replay_traffic.borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        replay_traffic.borrowed_recurrent_output_bytes(),
        recurrent_state_bytes
    );
    let accumulation = report
        .accumulation()
        .expect("phase-specialized scoreboard reports the accumulation program");
    let accumulation_executed = report
        .accumulation_replay_executed_native_item_count()
        .expect("phase-specialized scoreboard reports accumulation execution");
    assert_eq!(
        accumulation_executed,
        EXPECTED_ACCUMULATION_EXECUTED_ENTRIES
    );
    assert_eq!(
        Some(usize::try_from(accumulation_executed)?),
        stable_accumulation_executed_native_items
    );
    assert!(accumulation_executed < executed_native_items);
    assert_eq!(
        report.accumulation_schedule_cache_keys().len(),
        usize::try_from(accumulation.native_item_count())?
    );
    let accumulation_traffic = report
        .accumulation_replay_traffic()
        .expect("phase-specialized scoreboard reports accumulation traffic");
    assert_eq!(accumulation_traffic.external_input_import_count(), 0);
    assert_eq!(accumulation_traffic.external_input_import_bytes(), 0);
    assert_eq!(accumulation_traffic.materialized_egress_count(), 1);
    assert_eq!(accumulation_traffic.materialized_egress_bytes(), 4);
    assert_eq!(
        accumulation_traffic.borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        accumulation_traffic.borrowed_recurrent_output_bytes()
            + accumulation_traffic.retained_recurrent_state_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        accumulation_traffic.replaced_recurrent_state_bytes(),
        accumulation_traffic.borrowed_recurrent_output_bytes()
    );
    assert_eq!(
        accumulation_traffic.retained_recurrent_state_count()
            + accumulation_traffic.replaced_recurrent_state_count(),
        u64::try_from(inspection.recurrent_state_count())?
    );
    assert!(
        accumulation_traffic.retained_recurrent_state_count()
            >= GUARANTEED_RETAINED_RECURRENT_STATES
    );
    assert!(accumulation_traffic.replaced_recurrent_state_count() <= REMAINING_RECURRENT_STATES);
    assert!(
        accumulation_traffic.replaced_recurrent_state_count()
            >= MANDATORY_REPLACED_RECURRENT_STATES
    );
    assert!(
        accumulation_traffic.retained_recurrent_state_bytes()
            >= guaranteed_retained_recurrent_bytes
    );
    assert!(accumulation_traffic.replaced_recurrent_state_bytes() <= remaining_recurrent_bytes);
    assert!(
        accumulation_traffic.replaced_recurrent_state_bytes() >= mandatory_replaced_recurrent_bytes
    );

    let prepared_retained_recurrent_states = accumulation_traffic.retained_recurrent_state_count();
    let prepared_replaced_recurrent_states = accumulation_traffic.replaced_recurrent_state_count();
    let prepared_retained_recurrent_bytes = accumulation_traffic.retained_recurrent_state_bytes();
    let prepared_replaced_recurrent_bytes = accumulation_traffic.replaced_recurrent_state_bytes();

    let restored = plan.restore_checkpoint(&checkpoint)?;
    let restored_inspection = restored.inspection()?;
    assert_eq!(restored_inspection.main(), inspection.main());
    assert_eq!(
        restored_inspection.accumulation(),
        inspection.accumulation()
    );
    assert_eq!(
        restored_inspection.partial_flush(),
        inspection.partial_flush()
    );
    assert_eq!(
        restored_inspection.recurrent_state_count(),
        inspection.recurrent_state_count()
    );
    assert_eq!(
        restored_inspection.recurrent_state_bytes(),
        inspection.recurrent_state_bytes()
    );
    assert_eq!(builds.get(), 1, "checkpoint restore must not rebuild");
    let mut restored_session = restored.prepare(&target)?;
    let restored_preparation = restored_session.preparation_report();
    assert_eq!(restored_preparation.compiler_process_count(), 0);
    assert_eq!(
        restored_preparation.max_parallel_compiler_process_count(),
        0
    );
    assert_eq!(
        restored_preparation.parallel_module_overlap_wall_time(),
        Duration::ZERO
    );
    let capsule_diagnostics = restored_preparation.render_capsule_diagnostics();
    assert_eq!(
        capsule_diagnostics
            .iter()
            .map(|diagnostic| diagnostic.role())
            .collect::<Vec<_>>(),
        [
            NativeCpuRenderCapsuleProgramRole::Main,
            NativeCpuRenderCapsuleProgramRole::Accumulation,
            NativeCpuRenderCapsuleProgramRole::PartialFlush,
            NativeCpuRenderCapsuleProgramRole::ZeroGrad,
        ]
    );
    assert_eq!(
        restored_preparation.max_parallel_render_job_count(),
        0,
        "warm render capsule diagnostics: {capsule_diagnostics:#?}"
    );
    assert_eq!(restored_preparation.render_capsule_hit_count(), 4);
    assert_eq!(restored_preparation.render_capsule_miss_count(), 0);
    assert_eq!(restored_preparation.local_render_job_count(), 0);
    assert_eq!(
        restored_preparation.parallel_render_overlap_wall_time(),
        Duration::ZERO
    );
    let restored_programs = [
        Some(restored_preparation.main()),
        restored_preparation.accumulation(),
        restored_preparation.partial_flush(),
        restored_preparation.zero_grad(),
        restored_preparation.evaluation(),
    ];
    assert_eq!(restored_programs.iter().flatten().count(), 4);
    assert!(restored_preparation.evaluation().is_none());
    for program in restored_programs.into_iter().flatten() {
        assert_eq!(program.cache_miss_count(), 0);
        assert_eq!(program.cache_hit_count(), program.native_item_count());
        assert!(program.work().rendered_entry_count() <= program.native_item_count());
        assert_eq!(
            program.work().loaded_module_count(),
            usize::from(program.work().unique_rendered_entry_count() != 0)
        );
        assert_eq!(program.work().durable_artifact_cache_hit_count(), 0);
        assert_eq!(program.work().durable_artifact_cache_miss_count(), 0);
        assert_eq!(program.work().compiler_invocation_count(), 0);
        assert_eq!(
            program.phases().compiler_process_wall_time(),
            Duration::ZERO
        );
    }
    let restored_step = restored_session.step_batch_scheduled(masked_batch(SAMPLES + 1)?)?;
    let restored_report = restored_step.report();
    assert_eq!(restored_report.fallback_count(), 0);
    assert_eq!(
        u64::try_from(restored_report.native_item_count())?,
        accumulation.native_item_count()
    );
    assert_eq!(restored_report.traffic().external_input_import_count(), 0);
    assert_eq!(restored_report.traffic().external_input_import_bytes(), 0);
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_input_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.traffic().borrowed_recurrent_output_bytes()
            + restored_report.traffic().retained_recurrent_state_bytes(),
        recurrent_state_bytes
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_bytes(),
        restored_report.traffic().borrowed_recurrent_output_bytes()
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_count()
            + restored_report.traffic().replaced_recurrent_state_count(),
        u64::try_from(restored_inspection.recurrent_state_count())?
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_count(),
        prepared_retained_recurrent_states
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_count(),
        prepared_replaced_recurrent_states
    );
    assert_eq!(
        restored_report.traffic().retained_recurrent_state_bytes(),
        prepared_retained_recurrent_bytes
    );
    assert_eq!(
        restored_report.traffic().replaced_recurrent_state_bytes(),
        prepared_replaced_recurrent_bytes
    );
    assert_eq!(
        restored_report.executed_native_item_count(),
        stable_accumulation_executed_native_items
            .expect("the scoreboard recorded successful accumulation replays")
    );

    let report_bytes = report.to_json_bytes()?;
    let report_json: serde_json::Value = serde_json::from_slice(&report_bytes)?;
    let compiler_process_count = report_json["prepare_compiler_process_count"]
        .as_u64()
        .expect("current scoreboard reports compiler process count");
    assert_eq!(
        report_json["prepare_compiler_process_timings"]
            .as_array()
            .expect("current scoreboard reports bounded compiler process timings")
            .len(),
        usize::try_from(compiler_process_count)?
    );
    for timing in report_json["prepare_compiler_process_timings"]
        .as_array()
        .expect("current scoreboard reports bounded compiler process timings")
    {
        let source_bytes = timing["rendered_source_bytes"]
            .as_u64()
            .expect("current scoreboard reports translation-unit source bytes");
        assert!((source_bytes == 0) == (timing["process"]["kind"] == "link"));
    }
    let overlaps = report_json["prepare_module_overlaps"]
        .as_array()
        .expect("current scoreboard reports ordered main-program overlaps");
    assert!(
        report_json["main"]["rendered_source_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
    assert_eq!(report_json["main"]["shared_prefix_source_bytes"], 0);
    assert_eq!(overlaps.len(), 3);
    for (index, overlap) in overlaps.iter().enumerate() {
        assert_eq!(
            overlap["program_index"],
            u64::try_from(
                index
                    .checked_add(1)
                    .expect("program ordinal remains bounded")
            )?
        );
        for field in [
            "evidence_identity",
            "contiguous_prefix_entry_count",
            "contiguous_prefix_source_bytes",
            "additional_scattered_entry_count",
            "additional_scattered_source_bytes",
        ] {
            assert!(overlap[field].is_u64());
        }
    }
    assert_eq!(
        overlaps[0]["contiguous_prefix_entry_count"],
        u64::try_from(EXPECTED_SHARED_ACCUMULATION_PREFIX)?
    );
    assert!(
        overlaps[0]["contiguous_prefix_source_bytes"]
            .as_u64()
            .is_some_and(|bytes| bytes > 0)
    );
    let pair_overlaps = report_json["prepare_program_pair_overlaps"]
        .as_array()
        .expect("current scoreboard reports every ordered earlier/later program pair");
    assert_eq!(pair_overlaps.len(), 6);
    assert_eq!(
        pair_overlaps
            .iter()
            .map(|overlap| (
                overlap["source_program_index"].as_u64().unwrap(),
                overlap["target_program_index"].as_u64().unwrap(),
            ))
            .collect::<Vec<_>>(),
        [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
    );
    for (overlap, main_overlap) in pair_overlaps
        .iter()
        .filter(|overlap| overlap["source_program_index"] == 0)
        .zip(overlaps)
    {
        assert_eq!(
            overlap["contiguous_prefix_entry_count"],
            main_overlap["contiguous_prefix_entry_count"]
        );
        assert_eq!(
            overlap["additional_scattered_entry_count"],
            main_overlap["additional_scattered_entry_count"]
        );
    }
    let translation_units = report_json["prepare_translation_units"]
        .as_array()
        .expect("current scoreboard reports the exact translation-unit build plan");
    assert_eq!(translation_units.len(), 5);
    for unit in translation_units {
        assert!(unit["translation_unit_identity"].is_u64());
        assert!(unit["evidence_identity"].is_u64());
        assert!(unit["entry_count"].as_u64().is_some_and(|count| count > 0));
        assert!(
            unit["rendered_source_bytes"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0)
        );
    }
    assert_eq!(
        report_json["prepare_compiler_critical_tail"].is_object(),
        compiler_process_count != 0
    );
    print!("{}", String::from_utf8(report_bytes)?);
    Ok(())
}
