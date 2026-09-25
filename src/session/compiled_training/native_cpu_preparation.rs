//! Strict-native CPU AdamW preparation and authenticated plan finalization.

use super::*;

impl<'a> NativeCpuCompiledAdamW<'a> {
    pub(super) fn prepare(
        inner: CpuCompiledAdamW,
        executor: &'a CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<Self> {
        let external_learning_rate = matches!(
            &inner.contract.learning_rate,
            CompiledLearningRatePolicy::External
        );
        let program_count = 1
            + usize::from(inner.inner.accumulation.is_some())
            + usize::from(inner.partial_flush.is_some())
            + usize::from(inner.zero_grad.is_some())
            + usize::from(inner.evaluation.is_some());
        let mut drafts = NativeCpuTrainingProgramDrafts::new();
        let (mut main_preparation, main_residual) = inner
            .inner
            .preflight_native(vectorized, external_learning_rate)?;
        {
            let (pure, inputs) = main_preparation.pure_and_inputs();
            let draft = executor
                .preflight_native_items_with_store_groups(
                    pure,
                    inputs,
                    &inner.inner.recurrent_store_groups,
                )
                .map_err(replay_error)?;
            drafts.insert(NativeCpuTrainingProgramRole::Main, draft)?;
        }
        main_preparation.release_input_witnesses();
        let accumulation_preparation = match inner.inner.accumulation.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_accumulation(transition, vectorized)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_recurrent_retention(
                            pure,
                            inputs,
                            preparation.retained_recurrent_states(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::Accumulation, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let partial_flush_preparation = match inner.partial_flush.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) =
                    inner.inner.preflight_native_auxiliary_transition(
                        transition.phase(),
                        vectorized,
                        external_learning_rate,
                    )?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items_with_store_groups(
                            pure,
                            inputs,
                            transition.phase().store_groups(),
                        )
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::PartialFlush, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let zero_grad_preparation = match inner.zero_grad.as_ref() {
            Some(transition) => {
                let (mut preparation, residual) = inner
                    .inner
                    .preflight_native_auxiliary_transition(transition.phase(), vectorized, false)?;
                {
                    let (pure, inputs) = preparation.pure_and_inputs();
                    let draft = executor
                        .preflight_native_items(pure, inputs)
                        .map_err(replay_error)?;
                    drafts.insert(NativeCpuTrainingProgramRole::ZeroGrad, draft)?;
                }
                preparation.release_input_witnesses();
                Some((preparation, residual))
            }
            None => None,
        };
        let evaluation_preparation = match inner.evaluation.as_ref() {
            Some(evaluation) => {
                let mut preparation = evaluation.plan.preflight_native(
                    inner.parameter_snapshots()?,
                    &inner.inner.parameter_buffers,
                )?;
                let draft = executor
                    .preflight_native_items(
                        evaluation.plan.inference.capture(),
                        preparation
                            .inputs
                            .as_ref()
                            .expect("native evaluation input witnesses are present"),
                    )
                    .map_err(replay_error)?;
                drafts.insert(NativeCpuTrainingProgramRole::Evaluation, draft)?;
                preparation.inputs = None;
                Some(preparation)
            }
            None => None,
        };
        let mut programs = NativeCpuTrainingProgramBatch::with_capacity(program_count);
        programs.push(
            NativeCpuTrainingProgramRole::Main,
            main_preparation.pure(),
            drafts.take(NativeCpuTrainingProgramRole::Main)?,
        )?;
        if let Some((preparation, _)) = &accumulation_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::Accumulation,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::Accumulation)?,
            )?;
        }
        if let Some((preparation, _)) = &partial_flush_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::PartialFlush,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::PartialFlush)?,
            )?;
        }
        if let Some((preparation, _)) = &zero_grad_preparation {
            programs.push(
                NativeCpuTrainingProgramRole::ZeroGrad,
                preparation.pure(),
                drafts.take(NativeCpuTrainingProgramRole::ZeroGrad)?,
            )?;
        }
        if let Some(evaluation) = inner.evaluation.as_ref() {
            programs.push(
                NativeCpuTrainingProgramRole::Evaluation,
                evaluation.plan.inference.capture(),
                drafts.take(NativeCpuTrainingProgramRole::Evaluation)?,
            )?;
        }
        if !drafts.is_empty() {
            return Err(training(
                "compiled native CPU planning draft inventory is excessive",
            ));
        }
        let (roles, programs) = programs.into_planning_inputs();
        let (plans, compilation) = executor
            .plan_native_item_drafts(programs, vectorized)
            .map_err(replay_error)?;
        let render_capsule_diagnostics = NativeCpuRenderCapsuleDiagnostic::from_ordered_native(
            &roles,
            compilation.render_capsule_diagnostics,
        )?;
        let NativeCpuTrainingPrograms {
            main: main_plan,
            accumulation: accumulation_plan,
            partial_flush: partial_flush_plan,
            zero_grad: zero_grad_plan,
            evaluation: evaluation_plan,
        } = NativeCpuTrainingPrograms::from_ordered(roles, plans)?;
        let main = inner
            .inner
            .finish_native(main_preparation, main_plan, main_residual)?;
        let accumulation = match (
            inner.inner.accumulation.as_ref(),
            accumulation_preparation,
            accumulation_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => Some(
                inner
                    .inner
                    .finish_native_accumulation(transition, preparation, plan, residual)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU accumulation preparation differs",
                ));
            }
        };
        let partial_flush = match (
            inner.partial_flush.as_ref(),
            partial_flush_preparation,
            partial_flush_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU partial-flush preparation differs",
                ));
            }
        };
        let zero_grad = match (
            inner.zero_grad.as_ref(),
            zero_grad_preparation,
            zero_grad_plan,
        ) {
            (Some(transition), Some((preparation, residual)), Some(plan)) => {
                Some(inner.inner.finish_native_auxiliary_transition(
                    transition.phase(),
                    preparation,
                    plan,
                    residual,
                )?)
            }
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU zero-grad preparation differs",
                ));
            }
        };
        let evaluation = match (
            inner.evaluation.as_ref(),
            evaluation_preparation,
            evaluation_plan,
        ) {
            (Some(evaluation), Some(preparation), Some(plan)) => Some(
                evaluation
                    .plan
                    .finish_native(preparation, &inner.inner.parameter_buffers, plan)?,
            ),
            (None, None, None) => None,
            _ => {
                return Err(training(
                    "compiled native CPU evaluation preparation differs",
                ));
            }
        };
        let (recurrent_state_count, recurrent_state_bytes) = checked_recurrent_state_extent(
            inner
                .inner
                .cursor
                .frontier()
                .iter()
                .map(|state| state.bytes),
        )?;
        let PreparedNativeCpuProgram {
            report: main_report,
            replay: main_replay,
        } = main;
        let (accumulation_report, accumulation_replay) = accumulation
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (partial_flush_report, partial_flush_replay) = partial_flush
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let (zero_grad_report, zero_grad_replay) = zero_grad
            .map(|prepared| (prepared.report, prepared.replay))
            .unzip();
        let evaluation_report = evaluation.as_ref().map(|prepared| prepared.report.clone());
        Ok(Self {
            inner,
            executor,
            main_replay,
            accumulation_replay,
            partial_flush_replay,
            zero_grad_replay,
            evaluation_replay: evaluation,
            preparation: NativeCpuCompiledAdamWPreparationReport {
                main: main_report,
                accumulation: accumulation_report,
                partial_flush: partial_flush_report,
                zero_grad: zero_grad_report,
                evaluation: evaluation_report,
                recurrent_state_count,
                recurrent_state_bytes,
                render_capsule_hit_count: compilation.render_capsule_hit_count,
                render_capsule_miss_count: compilation.render_capsule_miss_count,
                local_render_job_count: compilation.local_render_job_count,
                parallel_render_overlap_wall_time: compilation.parallel_render_overlap_wall_time,
                max_parallel_render_job_count: compilation.max_parallel_render_job_count,
                parallel_module_overlap_wall_time: compilation.parallel_work_overlap_wall_time,
                compiler_process_overlap_wall_time: compilation.compiler_process_overlap_wall_time,
                compiler_process_count: compilation.compiler_process_count,
                max_parallel_compiler_process_count: compilation
                    .max_parallel_compiler_process_count,
                compiler_process_timings: compilation
                    .compiler_process_timings
                    .into_iter()
                    .map(NativeCpuCompilerProcessTiming::from_native)
                    .collect(),
                module_overlaps: compilation
                    .module_overlaps
                    .into_iter()
                    .map(NativeCpuModuleOverlap::from_native)
                    .collect(),
                program_pair_overlaps: compilation
                    .program_pair_overlaps
                    .into_iter()
                    .map(NativeCpuProgramPairOverlap::from_native)
                    .collect(),
                translation_units: compilation
                    .translation_units
                    .into_iter()
                    .map(NativeCpuTranslationUnitEvidence::from_native)
                    .collect(),
                render_capsule_diagnostics,
            },
            successful_steps: 0,
            successful_flushes: 0,
            successful_zero_grads: 0,
            successful_evaluations: 0,
        })
    }
}
