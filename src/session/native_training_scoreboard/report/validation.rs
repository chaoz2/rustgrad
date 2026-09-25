use super::super::compiler_evidence::{
    CompilerProcessValidationContext, validate_compiler_process_evidence, validate_module_overlaps,
    validate_program_pair_overlaps, validate_translation_units,
};
use super::super::program_report::{
    NativeCompilerAggregate, NativeRenderAggregate, NativeTrainingProgramReport,
};
use super::super::step_phases::ReplayTimingPartition;
use super::super::{
    MAX_REPLAY_SAMPLES, NATIVE_TRAINING_REPORT_FORMAT_V2, NATIVE_TRAINING_REPORT_FORMAT_V3,
    NATIVE_TRAINING_REPORT_FORMAT_V4, NATIVE_TRAINING_REPORT_FORMAT_V5,
    NATIVE_TRAINING_REPORT_FORMAT_V6, NATIVE_TRAINING_REPORT_FORMAT_V7,
    NATIVE_TRAINING_REPORT_FORMAT_V8, NATIVE_TRAINING_REPORT_FORMAT_V9,
    NATIVE_TRAINING_REPORT_FORMAT_V10, NATIVE_TRAINING_REPORT_FORMAT_V11,
    NATIVE_TRAINING_REPORT_FORMAT_V12, NATIVE_TRAINING_REPORT_FORMAT_V13,
    NATIVE_TRAINING_REPORT_FORMAT_V14, NATIVE_TRAINING_REPORT_FORMAT_V15,
    NATIVE_TRAINING_REPORT_FORMAT_V16, NATIVE_TRAINING_REPORT_FORMAT_V17,
    NATIVE_TRAINING_REPORT_FORMAT_V18, NATIVE_TRAINING_REPORT_FORMAT_V19,
    NATIVE_TRAINING_REPORT_FORMAT_V20, NATIVE_TRAINING_REPORT_FORMAT_V21,
    NATIVE_TRAINING_REPORT_FORMAT_V22, NATIVE_TRAINING_REPORT_FORMAT_VERSION, count, invalid,
    rate_from_total, validate_phase_partition, validate_total_duration,
};
use super::NativeTrainingReport;
use crate::Result;

pub(super) fn validate(report: &NativeTrainingReport) -> Result<()> {
    validate_header(report)?;
    let programs = validate_program_inventory(report)?;
    validate_preparation(report, &programs)?;
    validate_replay(report)?;
    validate_availability_and_checkpoint(report)
}

fn validate_header(report: &NativeTrainingReport) -> Result<()> {
    if !matches!(
        report.format_version,
        1 | NATIVE_TRAINING_REPORT_FORMAT_V2
            | NATIVE_TRAINING_REPORT_FORMAT_V3
            | NATIVE_TRAINING_REPORT_FORMAT_V4
            | NATIVE_TRAINING_REPORT_FORMAT_V5
            | NATIVE_TRAINING_REPORT_FORMAT_V6
            | NATIVE_TRAINING_REPORT_FORMAT_V7
            | NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11
            | NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION
    ) {
        return Err(invalid("unsupported native training report version"));
    }
    if report.format_version == 1 && report.zero_grad.is_some() {
        return Err(invalid(
            "legacy native training report has zero-grad evidence",
        ));
    }
    if report.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V2
        && report.partial_flush.is_some() != report.zero_grad.is_some()
    {
        return Err(invalid(
            "native training auxiliary program inventory differs",
        ));
    }
    for duration in [
        report.compile_wall_time,
        report.prepare_wall_time,
        report.first_replay_wall_time,
        report.steady_replay_total_wall_time,
        report.steady_replay_wall_time.min,
        report.steady_replay_wall_time.nearest_rank_p50,
        report.steady_replay_wall_time.nearest_rank_p95,
        report.steady_replay_wall_time.max,
    ] {
        duration
            .to_duration()
            .map_err(|_| invalid("invalid native training duration"))?;
    }
    match (report.format_version, &report.compile_phases) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, None) => {}
        (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(phases)) => phases.validate(
            report.compile_wall_time,
            &report.main,
            report.accumulation.as_ref(),
            report.partial_flush.as_ref(),
            report.zero_grad.as_ref(),
            report.evaluation.as_ref(),
        )?,
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, Some(_)) => {
            return Err(invalid("legacy native training report has compile phases"));
        }
        _ => return Err(invalid("native training compile phases are absent")),
    }
    report.main.validate(report.format_version)?;
    if report.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V14
        && (report.main.shared_prefix_entry_count() != 0
            || report.main.shared_prefix_source_program_index.is_some()
            || report.main.shared_prefix_source_native_identity.is_some())
    {
        return Err(invalid(
            "native main program cannot reference an earlier prefix module",
        ));
    }
    Ok(())
}

fn validate_program_inventory(
    report: &NativeTrainingReport,
) -> Result<Vec<&NativeTrainingProgramReport>> {
    validate_accumulation_inventory(report)?;
    validate_main_replay_traffic(report)?;
    validate_materialized_egress(report)?;
    validate_main_execution_count(report)?;
    collect_program_inventory(report)
}

fn validate_accumulation_inventory(report: &NativeTrainingReport) -> Result<()> {
    match (
        report.format_version,
        &report.accumulation,
        &report.accumulation_replay_traffic,
        report.accumulation_replay_executed_native_item_count,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V10, None, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            None,
            None,
            None,
        ) => {}
        (NATIVE_TRAINING_REPORT_FORMAT_V11, Some(program), Some(traffic), Some(executed))
            if traffic.borrowed_recurrent_input_bytes() == report.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == report.recurrent_logical_state_bytes
                && traffic.retained_recurrent_state_count() == 0
                && traffic.retained_recurrent_state_bytes() == 0
                && traffic.replaced_recurrent_state_count() == 0
                && traffic.replaced_recurrent_state_bytes() == 0
                && executed <= program.rendered_entry_count
                && program.capture_identity != report.main.capture_identity =>
        {
            program.validate(report.format_version)?;
            if program.vectorized != report.main.vectorized {
                return Err(invalid("native program vectorization policy differs"));
            }
        }
        (
            NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(program),
            Some(traffic),
            Some(executed),
        ) if traffic.borrowed_recurrent_input_bytes() == report.recurrent_logical_state_bytes
            && traffic
                .borrowed_recurrent_output_bytes()
                .checked_add(traffic.retained_recurrent_state_bytes())
                == Some(report.recurrent_logical_state_bytes)
            && traffic.retained_recurrent_state_count() <= report.recurrent_logical_state_count
            && ((traffic.retained_recurrent_state_count() == 0
                && traffic.retained_recurrent_state_bytes() == 0
                && traffic.replaced_recurrent_state_count() == 0
                && traffic.replaced_recurrent_state_bytes() == 0)
                || (traffic
                    .retained_recurrent_state_count()
                    .checked_add(traffic.replaced_recurrent_state_count())
                    == Some(report.recurrent_logical_state_count)
                    && traffic.replaced_recurrent_state_bytes()
                        == traffic.borrowed_recurrent_output_bytes()))
            && executed <= program.rendered_entry_count
            && program.capture_identity != report.main.capture_identity =>
        {
            program.validate(report.format_version)?;
            if program.vectorized != report.main.vectorized {
                return Err(invalid("native program vectorization policy differs"));
            }
        }
        _ => return Err(invalid("native accumulation replay inventory differs")),
    }
    Ok(())
}

fn validate_main_replay_traffic(report: &NativeTrainingReport) -> Result<()> {
    match (report.format_version, &report.main_replay_traffic) {
        (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V3
            | NATIVE_TRAINING_REPORT_FORMAT_V4
            | NATIVE_TRAINING_REPORT_FORMAT_V5
            | NATIVE_TRAINING_REPORT_FORMAT_V6
            | NATIVE_TRAINING_REPORT_FORMAT_V7
            | NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11,
            Some(traffic),
        ) if traffic.borrowed_recurrent_input_bytes() == report.recurrent_logical_state_bytes
            && traffic.borrowed_recurrent_output_bytes()
                == report.recurrent_logical_state_bytes
            && traffic.retained_recurrent_state_count() == 0
            && traffic.retained_recurrent_state_bytes() == 0
            && traffic.replaced_recurrent_state_count() == 0
            && traffic.replaced_recurrent_state_bytes() == 0 => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(traffic),
        ) if traffic.borrowed_recurrent_input_bytes() == report.recurrent_logical_state_bytes
            && traffic.borrowed_recurrent_output_bytes()
                == report.recurrent_logical_state_bytes
            && traffic.retained_recurrent_state_count() == 0
            && traffic.retained_recurrent_state_bytes() == 0
            && traffic.replaced_recurrent_state_count() == 0
            && traffic.replaced_recurrent_state_bytes() == 0 => {}
        (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, Some(_)) => {
            return Err(invalid("legacy native training report has replay traffic"));
        }
        _ => return Err(invalid("native training replay traffic differs")),
    }
    Ok(())
}

fn validate_materialized_egress(report: &NativeTrainingReport) -> Result<()> {
    for traffic in report
        .main_replay_traffic
        .iter()
        .chain(&report.accumulation_replay_traffic)
    {
        match report.format_version {
            1..=NATIVE_TRAINING_REPORT_FORMAT_V12
                if traffic.materialized_egress_count() == 0
                    && traffic.materialized_egress_bytes() == 0 => {}
            NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION
                if traffic.materialized_egress_count() != 0
                    && traffic.materialized_egress_bytes() != 0 => {}
            1..=NATIVE_TRAINING_REPORT_FORMAT_V12 => {
                return Err(invalid("legacy native report has CPU egress evidence"));
            }
            NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION => {
                return Err(invalid("native CPU egress evidence is absent"));
            }
            _ => unreachable!("format version was validated"),
        }
    }
    Ok(())
}

fn validate_main_execution_count(report: &NativeTrainingReport) -> Result<()> {
    match (
        report.format_version,
        report.main_replay_executed_native_item_count,
    ) {
        (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, None) => {}
        (NATIVE_TRAINING_REPORT_FORMAT_V4, Some(executed))
            if executed <= report.main.native_item_count => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V5
            | NATIVE_TRAINING_REPORT_FORMAT_V6
            | NATIVE_TRAINING_REPORT_FORMAT_V7
            | NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11,
            Some(executed),
        ) if executed <= report.main.rendered_entry_count => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(executed),
        ) if executed <= report.main.rendered_entry_count => {}
        (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, Some(_)) => {
            return Err(invalid("legacy native training report has execution count"));
        }
        _ => return Err(invalid("native training execution count differs")),
    }
    Ok(())
}

fn collect_program_inventory(
    report: &NativeTrainingReport,
) -> Result<Vec<&NativeTrainingProgramReport>> {
    let mut prior_programs = vec![&report.main];
    for program in report
        .accumulation
        .iter()
        .chain(&report.partial_flush)
        .chain(&report.zero_grad)
        .chain(&report.evaluation)
    {
        program.validate(report.format_version)?;
        if program.vectorized != report.main.vectorized {
            return Err(invalid("native program vectorization policy differs"));
        }
        if let Some(source_index) = program.shared_prefix_source_program_index {
            let source_index = usize::try_from(source_index)
                .map_err(|_| invalid("native shared-prefix source index overflows"))?;
            let source = prior_programs.get(source_index).copied().ok_or_else(|| {
                invalid("native shared-prefix source index is not an earlier program")
            })?;
            if program.shared_prefix_source_native_identity != Some(source.native_identity) {
                return Err(invalid(
                    "native shared-prefix source identity differs from its program index",
                ));
            }
            if source.rendered_entry_count == 0
                || source.shared_prefix_entry_count() != 0
                || program.shared_prefix_entry_count() > source.rendered_entry_count
            {
                return Err(invalid(
                    "native shared prefix exceeds its source program inventory",
                ));
            }
        }
        prior_programs.push(program);
    }
    Ok(prior_programs)
}

#[derive(Clone, Copy, Debug)]
struct CompilerParallelism {
    overlap: u128,
    module_job_count: u64,
    internal_overlap: u128,
}

fn validate_preparation(
    report: &NativeTrainingReport,
    prior_programs: &[&NativeTrainingProgramReport],
) -> Result<()> {
    validate_module_evidence(report, prior_programs)?;
    let compiler_parallelism = validate_compiler_parallelism(report)?;
    validate_compiler_critical_tail(report, prior_programs)?;
    validate_render_capsules(report)?;
    let render_overlap = validate_render_parallelism(report)?;
    validate_preparation_partition(report, compiler_parallelism, render_overlap)
}

fn validate_module_evidence(
    report: &NativeTrainingReport,
    prior_programs: &[&NativeTrainingProgramReport],
) -> Result<()> {
    match (report.format_version, &report.prepare_module_overlaps) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(overlaps),
        ) => {
            validate_module_overlaps(overlaps, prior_programs)?;
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, Some(_)) => {
            return Err(invalid("legacy native report has module overlap evidence"));
        }
        _ => return Err(invalid("native module overlap evidence is absent")),
    }
    match (
        report.format_version,
        &report.prepare_program_pair_overlaps,
        &report.prepare_translation_units,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V20, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(overlaps),
            Some(translation_units),
        ) => {
            validate_program_pair_overlaps(
                overlaps,
                prior_programs,
                report
                    .prepare_module_overlaps
                    .as_deref()
                    .ok_or_else(|| invalid("native main overlap evidence is absent"))?,
            )?;
            validate_translation_units(
                translation_units,
                prior_programs,
                report
                    .prepare_compiler_process_timings
                    .as_deref()
                    .ok_or_else(|| invalid("native compiler process evidence is absent"))?,
            )?;
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V20, _, _) => {
            return Err(invalid("legacy native report has v21 module evidence"));
        }
        _ => return Err(invalid("native v21 module evidence is absent")),
    }
    Ok(())
}

fn validate_compiler_parallelism(
    report: &NativeTrainingReport,
) -> Result<Option<CompilerParallelism>> {
    let parallel_evidence = match report.format_version {
        1..=NATIVE_TRAINING_REPORT_FORMAT_V7 => {
            if report.prepare_compiler_process_overlap_wall_time.is_some()
                || report.prepare_compiler_process_count.is_some()
                || report.prepare_max_parallel_compiler_process_count.is_some()
            {
                return Err(invalid(
                    "legacy native report has parallel compiler evidence",
                ));
            }
            None
        }
        NATIVE_TRAINING_REPORT_FORMAT_V8
        | NATIVE_TRAINING_REPORT_FORMAT_V9
        | NATIVE_TRAINING_REPORT_FORMAT_V10
        | NATIVE_TRAINING_REPORT_FORMAT_V11
        | NATIVE_TRAINING_REPORT_FORMAT_V12
        | NATIVE_TRAINING_REPORT_FORMAT_V13
        | NATIVE_TRAINING_REPORT_FORMAT_V14
        | NATIVE_TRAINING_REPORT_FORMAT_V15
        | NATIVE_TRAINING_REPORT_FORMAT_V16
        | NATIVE_TRAINING_REPORT_FORMAT_V17
        | NATIVE_TRAINING_REPORT_FORMAT_V18
        | NATIVE_TRAINING_REPORT_FORMAT_V19
        | NATIVE_TRAINING_REPORT_FORMAT_V20
        | NATIVE_TRAINING_REPORT_FORMAT_V21
        | NATIVE_TRAINING_REPORT_FORMAT_V22
        | NATIVE_TRAINING_REPORT_FORMAT_VERSION => {
            let compiler_overlap = report
                .prepare_compiler_process_overlap_wall_time
                .ok_or_else(|| invalid("native compiler overlap timing is absent"))?
                .as_nanos()
                .map_err(|_| invalid("invalid native compiler overlap duration"))?;
            let compiler_count = report
                .prepare_compiler_process_count
                .ok_or_else(|| invalid("native compiler process count is absent"))?;
            let max_parallel = report
                .prepare_max_parallel_compiler_process_count
                .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
            let aggregate = NativeCompilerAggregate::from_programs(
                std::iter::once(&report.main)
                    .chain(report.accumulation.iter())
                    .chain(report.partial_flush.iter())
                    .chain(&report.zero_grad)
                    .chain(&report.evaluation),
                report.format_version,
            )?;
            let internal_compiler_overlap = aggregate.internal_overlap()?;
            let overlap_is_valid = if report.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V15 {
                let active_union = aggregate
                    .cumulative_wall_sum
                    .checked_sub(compiler_overlap)
                    .ok_or_else(|| invalid("native compiler overlap exceeds cumulative work"))?;
                compiler_overlap >= internal_compiler_overlap
                    && aggregate.effective_wall_max <= active_union
                    && active_union <= aggregate.effective_wall_sum
                    && compiler_overlap <= active_union
            } else {
                let maximum_compiler_overlap = aggregate
                    .effective_wall_sum
                    .checked_sub(aggregate.effective_wall_max)
                    .ok_or_else(|| invalid("native compiler process overlap underflows"))?;
                compiler_overlap <= maximum_compiler_overlap
            };
            if compiler_count != aggregate.process_count
                || max_parallel > 2
                || max_parallel > compiler_count
                || (compiler_count == 0) != (max_parallel == 0)
                || (compiler_overlap == 0) != (max_parallel <= 1)
                || !overlap_is_valid
            {
                return Err(invalid("native parallel compiler evidence differs"));
            }
            Some(CompilerParallelism {
                overlap: compiler_overlap,
                module_job_count: aggregate.module_job_count,
                internal_overlap: internal_compiler_overlap,
            })
        }
        _ => unreachable!("format version was validated"),
    };
    Ok(parallel_evidence)
}

fn validate_compiler_critical_tail(
    report: &NativeTrainingReport,
    prior_programs: &[&NativeTrainingProgramReport],
) -> Result<()> {
    match (
        report.format_version,
        &report.prepare_compiler_process_timings,
        &report.prepare_compiler_critical_tail,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V18, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(timings),
            claimed_tail,
        ) => {
            let process_count = report
                .prepare_compiler_process_count
                .ok_or_else(|| invalid("native compiler process count is absent"))?;
            let max_parallel = report
                .prepare_max_parallel_compiler_process_count
                .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
            let overlap = report
                .prepare_compiler_process_overlap_wall_time
                .ok_or_else(|| invalid("native compiler overlap timing is absent"))?;
            let derived_tail = validate_compiler_process_evidence(
                timings,
                claimed_tail.as_ref(),
                prior_programs,
                CompilerProcessValidationContext {
                    format_version: report.format_version,
                    process_count,
                    max_parallel,
                    claimed_overlap: overlap,
                    prepare_wall_time: report.prepare_wall_time,
                },
            )?;
            if derived_tail.is_some() != claimed_tail.is_some() {
                return Err(invalid("native compiler critical-tail presence differs"));
            }
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V18, _, _) => {
            return Err(invalid(
                "legacy native report has compiler critical-path evidence",
            ));
        }
        _ => return Err(invalid("native compiler critical-path evidence is absent")),
    }
    Ok(())
}

fn validate_render_capsules(report: &NativeTrainingReport) -> Result<()> {
    match (
        report.format_version,
        report.prepare_render_capsule_hit_count,
        report.prepare_render_capsule_miss_count,
        report.prepare_local_render_job_count,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V21, None, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V22 | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(_),
            Some(_),
            Some(_),
        ) => {}
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V21, _, _, _) => {
            return Err(invalid("legacy native report has render capsule evidence"));
        }
        _ => return Err(invalid("native render capsule evidence is absent")),
    }
    Ok(())
}

fn validate_render_parallelism(report: &NativeTrainingReport) -> Result<u128> {
    let render_overlap = match (
        report.format_version,
        report.prepare_parallel_render_overlap_wall_time,
        report.prepare_max_parallel_render_job_count,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V15, None, None) => 0,
        (
            NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22,
            Some(overlap),
            Some(max_parallel),
        ) => {
            let overlap = overlap
                .as_nanos()
                .map_err(|_| invalid("invalid native render overlap duration"))?;
            let aggregate = NativeRenderAggregate::from_programs(
                std::iter::once(&report.main)
                    .chain(report.accumulation.iter())
                    .chain(report.partial_flush.iter())
                    .chain(&report.zero_grad)
                    .chain(&report.evaluation),
            )?;
            let maximum_overlap = aggregate
                .wall_sum
                .checked_sub(aggregate.wall_max)
                .ok_or_else(|| invalid("native render overlap underflows"))?;
            let no_render_jobs = aggregate.wall_sum == 0;
            if (no_render_jobs && (max_parallel != 0 || overlap != 0))
                || (!no_render_jobs && max_parallel == 0)
                || max_parallel > 2
                || max_parallel > aggregate.job_count
                || overlap > maximum_overlap
                || (!no_render_jobs && (overlap == 0) != (max_parallel == 1))
            {
                return Err(invalid("native parallel render evidence differs"));
            }
            overlap
        }
        (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(overlap), Some(max_parallel)) => {
            let overlap = overlap
                .as_nanos()
                .map_err(|_| invalid("invalid native render overlap duration"))?;
            let programs = std::iter::once(&report.main)
                .chain(report.accumulation.iter())
                .chain(report.partial_flush.iter())
                .chain(&report.zero_grad)
                .chain(&report.evaluation)
                .collect::<Vec<_>>();
            let program_count = count(programs.len(), "native render program")?;
            let aggregate = NativeRenderAggregate::from_programs(programs)?;
            let maximum_overlap = aggregate
                .wall_sum
                .checked_sub(aggregate.wall_max)
                .ok_or_else(|| invalid("native render overlap underflows"))?;
            let hit_count = report
                .prepare_render_capsule_hit_count
                .ok_or_else(|| invalid("native render capsule hit count is absent"))?;
            let miss_count = report
                .prepare_render_capsule_miss_count
                .ok_or_else(|| invalid("native render capsule miss count is absent"))?;
            let local_render_count = report
                .prepare_local_render_job_count
                .ok_or_else(|| invalid("native local render job count is absent"))?;
            let zero_timing = max_parallel == 0 && overlap == 0 && aggregate.wall_sum == 0;
            let timing_partition_matches = if local_render_count == 0 || aggregate.wall_sum == 0 {
                zero_timing
            } else {
                max_parallel != 0 && (overlap == 0) == (max_parallel == 1)
            };
            if hit_count.checked_add(miss_count) != Some(program_count)
                || miss_count != local_render_count
                || !timing_partition_matches
                || max_parallel > 2
                || max_parallel > local_render_count
                || overlap > maximum_overlap
            {
                return Err(invalid("native render capsule evidence differs"));
            }
            overlap
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V15, _, _) => {
            return Err(invalid("legacy native report has parallel render evidence"));
        }
        _ => return Err(invalid("native parallel render evidence is absent")),
    };
    Ok(render_overlap)
}

fn validate_preparation_partition(
    report: &NativeTrainingReport,
    compiler_parallelism: Option<CompilerParallelism>,
    render_overlap: u128,
) -> Result<()> {
    match (
        report.format_version,
        report.prepare_runtime_overhead_wall_time,
        report.prepare_parallel_module_overlap_wall_time,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V7
            | NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11
            | NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(overhead),
            overlap,
        ) => {
            let overlap = match (report.format_version, overlap) {
                (NATIVE_TRAINING_REPORT_FORMAT_V7, None) => 0,
                (
                    NATIVE_TRAINING_REPORT_FORMAT_V8
                    | NATIVE_TRAINING_REPORT_FORMAT_V9
                    | NATIVE_TRAINING_REPORT_FORMAT_V10
                    | NATIVE_TRAINING_REPORT_FORMAT_V11
                    | NATIVE_TRAINING_REPORT_FORMAT_V12
                    | NATIVE_TRAINING_REPORT_FORMAT_V13
                    | NATIVE_TRAINING_REPORT_FORMAT_V14
                    | NATIVE_TRAINING_REPORT_FORMAT_V15
                    | NATIVE_TRAINING_REPORT_FORMAT_V16
                    | NATIVE_TRAINING_REPORT_FORMAT_V17
                    | NATIVE_TRAINING_REPORT_FORMAT_V18
                    | NATIVE_TRAINING_REPORT_FORMAT_V19
                    | NATIVE_TRAINING_REPORT_FORMAT_V20
                    | NATIVE_TRAINING_REPORT_FORMAT_V21
                    | NATIVE_TRAINING_REPORT_FORMAT_V22
                    | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                    Some(overlap),
                ) => overlap
                    .as_nanos()
                    .map_err(|_| invalid("invalid native prepare overlap duration"))?,
                _ => return Err(invalid("native prepare overlap timing differs")),
            };
            if let Some(parallelism) = compiler_parallelism {
                let CompilerParallelism {
                    overlap: compiler_overlap,
                    module_job_count,
                    internal_overlap,
                } = parallelism;
                let cross_program_compiler_overlap = compiler_overlap
                    .checked_sub(internal_overlap)
                    .ok_or_else(|| invalid("native internal compiler overlap exceeds total"))?;
                if cross_program_compiler_overlap > overlap {
                    return Err(invalid(
                        "native compiler overlap exceeds parallel module overlap",
                    ));
                }
                if overlap > 0 && module_job_count < 2 {
                    return Err(invalid(
                        "native parallel module overlap lacks two module jobs",
                    ));
                }
            }
            let prepare_total = report
                .prepare_wall_time
                .as_nanos()
                .map_err(|_| invalid("invalid native prepare duration"))?;
            let mut programs = std::iter::once(&report.main)
                .chain(report.accumulation.iter())
                .chain(report.partial_flush.iter())
                .chain(&report.zero_grad)
                .chain(&report.evaluation);
            let program_total = programs.try_fold(0u128, |total, program| {
                let timing = program
                    .preparation_timing
                    .as_ref()
                    .ok_or_else(|| invalid("native program preparation timing is absent"))?;
                timing
                    .total
                    .as_nanos()
                    .map_err(|_| invalid("invalid native program preparation duration"))?
                    .checked_add(total)
                    .ok_or_else(|| invalid("native program preparation duration overflows"))
            })?;
            let effective_program_total = program_total
                .checked_sub(overlap)
                .and_then(|total| total.checked_sub(render_overlap))
                .ok_or_else(|| invalid("native prepare overlap exceeds program duration"))?;
            let partitioned = overhead
                .as_nanos()
                .map_err(|_| invalid("invalid native prepare overhead duration"))?
                .checked_add(effective_program_total)
                .ok_or_else(|| invalid("native prepare duration overflows"))?;
            if partitioned != prepare_total {
                return Err(invalid("native prepare phases do not partition total"));
            }
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, _, _) => {
            return Err(invalid("legacy native report has preparation timing"));
        }
        _ => return Err(invalid("native prepare timing differs")),
    }
    Ok(())
}

fn validate_replay(report: &NativeTrainingReport) -> Result<()> {
    validate_replay_summary(report)?;
    validate_replay_phase_partition(report)?;
    validate_dispatcher_phase_partition(report)?;
    validate_step_phases(report)?;
    validate_replay_inventory(report)
}

fn validate_replay_summary(report: &NativeTrainingReport) -> Result<()> {
    if report.successful_replay_count < 2
        || report.successful_replay_count > MAX_REPLAY_SAMPLES as u64
        || report.steady_replay_wall_time.sample_count != report.successful_replay_count - 1
        || report.steady_replay_wall_time.min > report.steady_replay_wall_time.nearest_rank_p50
        || report.steady_replay_wall_time.nearest_rank_p50
            > report.steady_replay_wall_time.nearest_rank_p95
        || report.steady_replay_wall_time.nearest_rank_p95 > report.steady_replay_wall_time.max
        || report.steady_replay_total_wall_time < report.steady_replay_wall_time.max
    {
        return Err(invalid("invalid native training replay summary"));
    }
    validate_total_duration(
        &report.steady_replay_wall_time,
        report.steady_replay_total_wall_time,
    )?;
    Ok(())
}

fn validate_replay_phase_partition(report: &NativeTrainingReport) -> Result<()> {
    match (
        report.format_version,
        &report.main_replay_executor_wall_time,
        &report.main_replay_recurrent_overhead_wall_time,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V5, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V6
            | NATIVE_TRAINING_REPORT_FORMAT_V7
            | NATIVE_TRAINING_REPORT_FORMAT_V8
            | NATIVE_TRAINING_REPORT_FORMAT_V9
            | NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11
            | NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(executor),
            Some(overhead),
        ) => {
            let steady_count = report.successful_replay_count - 1;
            executor.validate(steady_count)?;
            overhead.validate(steady_count)?;
            validate_phase_partition(
                report.first_replay_wall_time,
                executor.first,
                overhead.first,
                "first replay",
            )?;
            validate_phase_partition(
                report.steady_replay_total_wall_time,
                executor.steady_total,
                overhead.steady_total,
                "steady replay",
            )?;
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V5, _, _) => {
            return Err(invalid("legacy native training report has replay phases"));
        }
        _ => return Err(invalid("native training replay phases differ")),
    }
    Ok(())
}

fn validate_dispatcher_phase_partition(report: &NativeTrainingReport) -> Result<()> {
    match (
        report.format_version,
        &report.main_replay_native_dispatcher_wall_time,
        &report.main_replay_executor_host_wall_time,
    ) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V17, None, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(native_dispatcher),
            Some(executor_host),
        ) => {
            let steady_count = report.successful_replay_count - 1;
            native_dispatcher.validate(steady_count)?;
            executor_host.validate(steady_count)?;
            let executor = report
                .main_replay_executor_wall_time
                .as_ref()
                .ok_or_else(|| invalid("native replay executor timing is absent"))?;
            validate_phase_partition(
                executor.first,
                native_dispatcher.first,
                executor_host.first,
                "first replay executor",
            )?;
            validate_phase_partition(
                executor.steady_total,
                native_dispatcher.steady_total,
                executor_host.steady_total,
                "steady replay executor",
            )?;
        }
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V17, _, _) => {
            return Err(invalid(
                "legacy native training report has dispatcher timing",
            ));
        }
        _ => return Err(invalid("native training dispatcher timing differs")),
    }
    Ok(())
}

fn validate_step_phases(report: &NativeTrainingReport) -> Result<()> {
    match (report.format_version, &report.step_phases) {
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, None) => {}
        (
            NATIVE_TRAINING_REPORT_FORMAT_V10
            | NATIVE_TRAINING_REPORT_FORMAT_V11
            | NATIVE_TRAINING_REPORT_FORMAT_V12
            | NATIVE_TRAINING_REPORT_FORMAT_V13
            | NATIVE_TRAINING_REPORT_FORMAT_V14
            | NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17,
            Some(phases),
        ) => phases.validate(
            report.successful_replay_count,
            report.first_replay_wall_time,
            report.steady_replay_total_wall_time,
            ReplayTimingPartition {
                executor: report
                    .main_replay_executor_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                native_dispatcher: None,
                executor_host: None,
                overhead: report
                    .main_replay_recurrent_overhead_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
            },
        )?,
        (1..=NATIVE_TRAINING_REPORT_FORMAT_V9, Some(_)) => {
            return Err(invalid("legacy native training report has step phases"));
        }
        (
            NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            Some(phases),
        ) => phases.validate(
            report.successful_replay_count,
            report.first_replay_wall_time,
            report.steady_replay_total_wall_time,
            ReplayTimingPartition {
                executor: report
                    .main_replay_executor_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                native_dispatcher: report.main_replay_native_dispatcher_wall_time.as_ref(),
                executor_host: report.main_replay_executor_host_wall_time.as_ref(),
                overhead: report
                    .main_replay_recurrent_overhead_wall_time
                    .as_ref()
                    .ok_or_else(|| invalid("classified replay overhead timing is absent"))?,
            },
        )?,
        (
            NATIVE_TRAINING_REPORT_FORMAT_V15
            | NATIVE_TRAINING_REPORT_FORMAT_V16
            | NATIVE_TRAINING_REPORT_FORMAT_V17
            | NATIVE_TRAINING_REPORT_FORMAT_V18
            | NATIVE_TRAINING_REPORT_FORMAT_V19
            | NATIVE_TRAINING_REPORT_FORMAT_V20
            | NATIVE_TRAINING_REPORT_FORMAT_V21
            | NATIVE_TRAINING_REPORT_FORMAT_V22
            | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            None,
        ) if report.accumulation.is_none() => {}
        _ => return Err(invalid("native training step phases differ")),
    }
    Ok(())
}

fn validate_replay_inventory(report: &NativeTrainingReport) -> Result<()> {
    let expected_rate = rate_from_total(
        report.steady_replay_wall_time.sample_count,
        report.steady_replay_total_wall_time,
    )?;
    if report.steady_microbatches_per_second.map(f64::to_bits) != expected_rate.map(f64::to_bits)
        || report.schedule_cache_keys.len()
            != usize::try_from(report.main.native_item_count)
                .map_err(|_| invalid("native item count overflows usize"))?
        || match &report.accumulation {
            Some(program) => {
                report.accumulation_schedule_cache_keys.len()
                    != usize::try_from(program.native_item_count)
                        .map_err(|_| invalid("native accumulation item count overflows usize"))?
            }
            None => !report.accumulation_schedule_cache_keys.is_empty(),
        }
    {
        return Err(invalid("invalid native training replay inventory"));
    }
    Ok(())
}

fn validate_availability_and_checkpoint(report: &NativeTrainingReport) -> Result<()> {
    if report.fallback_count != 0
        || report.kernel_launch_count.is_some()
        || report.host_to_device.is_some()
        || report.device_to_host.is_some()
        || report.measured_peak_host_memory_bytes.is_some()
    {
        return Err(invalid("native CPU availability fields are inconsistent"));
    }
    if let Some(checkpoint) = &report.checkpoint {
        checkpoint
            .wall_time
            .to_duration()
            .map_err(|_| invalid("invalid checkpoint duration"))?;
        let expected_step = report
            .initial_replay_step
            .checked_add(report.successful_replay_count)
            .ok_or_else(|| invalid("replay step overflows"))?;
        if checkpoint.byte_count == 0
            || checkpoint.capture_identity != report.main.capture_identity
            || checkpoint.replay_step != expected_step
        {
            return Err(invalid("checkpoint does not match recorded replays"));
        }
    }
    Ok(())
}
