//! Validated observational evidence for strict-native CPU compiled training.
//!
//! The runtime remains the source of preparation and replay facts. This module
//! only checks and aggregates detached reports; it does not time calls, execute
//! programs, or infer unavailable device and allocator measurements.

mod step_phases;

use step_phases::ReplayTimingPartition;
pub use step_phases::{
    NativeTrainingFirstStepReport, NativeTrainingStepPhase, NativeTrainingStepPhaseReport,
    NativeTrainingWarmStepReport,
};

use super::compiled_training::{
    NativeCpuCompilerProcessTiming, NativeCpuModuleOverlap, NativeCpuProgramPairOverlap,
    NativeCpuTranslationUnitEvidence,
};
use super::{
    CompiledAdamWCheckpoint, NativeCpuCompiledAdamWPreparationReport,
    NativeCpuCompiledAdamWStepResult, NativeCpuDispatchSegmentation,
    NativeCpuProgramPreparationReport, NativeCpuReplayTraffic, NativeCpuRunReport,
};
use crate::cpu_jit::NativeCompilerProcessKind;
use crate::{
    BenchmarkDuration, BenchmarkLatencySummary, BenchmarkTransfer, Error, ExecutionPlanSummary,
    Result,
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const NATIVE_TRAINING_REPORT_FORMAT_V2: u32 = 2;
const NATIVE_TRAINING_REPORT_FORMAT_V3: u32 = 3;
const NATIVE_TRAINING_REPORT_FORMAT_V4: u32 = 4;
const NATIVE_TRAINING_REPORT_FORMAT_V5: u32 = 5;
const NATIVE_TRAINING_REPORT_FORMAT_V6: u32 = 6;
const NATIVE_TRAINING_REPORT_FORMAT_V7: u32 = 7;
const NATIVE_TRAINING_REPORT_FORMAT_V8: u32 = 8;
const NATIVE_TRAINING_REPORT_FORMAT_V9: u32 = 9;
const NATIVE_TRAINING_REPORT_FORMAT_V10: u32 = 10;
const NATIVE_TRAINING_REPORT_FORMAT_V11: u32 = 11;
const NATIVE_TRAINING_REPORT_FORMAT_V12: u32 = 12;
const NATIVE_TRAINING_REPORT_FORMAT_V13: u32 = 13;
const NATIVE_TRAINING_REPORT_FORMAT_V14: u32 = 14;
const NATIVE_TRAINING_REPORT_FORMAT_V15: u32 = 15;
const NATIVE_TRAINING_REPORT_FORMAT_V16: u32 = 16;
const NATIVE_TRAINING_REPORT_FORMAT_V17: u32 = 17;
const NATIVE_TRAINING_REPORT_FORMAT_V18: u32 = 18;
const NATIVE_TRAINING_REPORT_FORMAT_V19: u32 = 19;
const NATIVE_TRAINING_REPORT_FORMAT_V20: u32 = 20;
const NATIVE_TRAINING_REPORT_FORMAT_V21: u32 = 21;
const NATIVE_TRAINING_REPORT_FORMAT_V22: u32 = 22;
pub const NATIVE_TRAINING_REPORT_FORMAT_VERSION: u32 = 23;
const MAX_REPLAY_SAMPLES: usize = 10_000;

/// Compiler subprocess role in one cold native preparation batch.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
enum NativeTrainingCompilerProcessKind {
    Combined,
    Object { ordinal: u64 },
    Link,
}

/// Portable compiler subprocess timing normalized to one preparation-batch
/// origin. These observations never participate in executable identities.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeTrainingCompilerProcessTiming {
    program_index: u64,
    native_identity: u64,
    process: NativeTrainingCompilerProcessKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rendered_source_bytes: Option<u64>,
    permit_request_offset: BenchmarkDuration,
    permit_wait: BenchmarkDuration,
    process_wall_time: BenchmarkDuration,
}

impl NativeTrainingCompilerProcessTiming {
    fn from_preparation(
        timing: &NativeCpuCompilerProcessTiming,
        programs: &[&NativeTrainingProgramReport],
    ) -> Result<Self> {
        let program = programs
            .get(timing.program_index())
            .ok_or_else(|| invalid("native compiler process program is absent"))?;
        let process = match timing.kind() {
            NativeCompilerProcessKind::Combined => NativeTrainingCompilerProcessKind::Combined,
            NativeCompilerProcessKind::Object(ordinal) => {
                NativeTrainingCompilerProcessKind::Object {
                    ordinal: count(ordinal, "native compiler object ordinal")?,
                }
            }
            NativeCompilerProcessKind::Link => NativeTrainingCompilerProcessKind::Link,
        };
        Ok(Self {
            program_index: count(timing.program_index(), "native compiler program")?,
            native_identity: program.native_identity(),
            process,
            rendered_source_bytes: Some(count(
                timing.rendered_source_bytes(),
                "native compiler rendered source byte",
            )?),
            permit_request_offset: BenchmarkDuration::from_duration(timing.permit_request_offset()),
            permit_wait: BenchmarkDuration::from_duration(timing.permit_wait_time()),
            process_wall_time: BenchmarkDuration::from_duration(timing.process_wall_time()),
        })
    }

    fn interval(&self) -> Result<(u128, u128, u128)> {
        let requested = self
            .permit_request_offset
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler permit request offset"))?;
        let wait = self
            .permit_wait
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler permit wait duration"))?;
        let process = self
            .process_wall_time
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler process duration"))?;
        let started = requested
            .checked_add(wait)
            .ok_or_else(|| invalid("native compiler process start overflows"))?;
        let finished = started
            .checked_add(process)
            .ok_or_else(|| invalid("native compiler process finish overflows"))?;
        Ok((requested, started, finished))
    }
}

/// Exact content overlap between main and one attached program. Counts are
/// target-entry counts; source bytes are the corresponding `RenderedC` bytes.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeTrainingModuleOverlap {
    program_index: u64,
    native_identity: u64,
    evidence_identity: u64,
    contiguous_prefix_entry_count: u64,
    contiguous_prefix_source_bytes: u64,
    additional_scattered_entry_count: u64,
    additional_scattered_source_bytes: u64,
}

impl NativeTrainingModuleOverlap {
    fn from_preparation(
        overlap: &NativeCpuModuleOverlap,
        programs: &[&NativeTrainingProgramReport],
    ) -> Result<Self> {
        let program = programs
            .get(overlap.program_index())
            .ok_or_else(|| invalid("native module overlap program is absent"))?;
        let overlap = Self {
            program_index: count(overlap.program_index(), "native module overlap program")?,
            native_identity: program.native_identity(),
            evidence_identity: 0,
            contiguous_prefix_entry_count: count(
                overlap.contiguous_prefix_entry_count(),
                "native contiguous overlap entry",
            )?,
            contiguous_prefix_source_bytes: count(
                overlap.contiguous_prefix_source_bytes(),
                "native contiguous overlap source byte",
            )?,
            additional_scattered_entry_count: count(
                overlap.additional_scattered_entry_count(),
                "native scattered overlap entry",
            )?,
            additional_scattered_source_bytes: count(
                overlap.additional_scattered_source_bytes(),
                "native scattered overlap source byte",
            )?,
        };
        let main_native_identity = programs
            .first()
            .ok_or_else(|| invalid("native main program is absent"))?
            .native_identity();
        Ok(overlap.with_evidence_identity(main_native_identity))
    }

    fn with_evidence_identity(mut self, main_native_identity: u64) -> Self {
        self.evidence_identity = self.expected_evidence_identity(main_native_identity);
        self
    }

    fn expected_evidence_identity(&self, main_native_identity: u64) -> u64 {
        let mut identity = 0xcbf29ce484222325u64;
        for byte in b"rustgrad-native-training-module-overlap-v20" {
            identity = (identity ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
        for value in [
            self.program_index,
            main_native_identity,
            self.native_identity,
            self.contiguous_prefix_entry_count,
            self.contiguous_prefix_source_bytes,
            self.additional_scattered_entry_count,
            self.additional_scattered_source_bytes,
        ] {
            for byte in value.to_le_bytes() {
                identity = (identity ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        identity
    }
}

fn validate_module_overlaps(
    overlaps: &[NativeTrainingModuleOverlap],
    programs: &[&NativeTrainingProgramReport],
) -> Result<()> {
    let main = programs
        .first()
        .ok_or_else(|| invalid("native main program is absent"))?;
    if overlaps.len().checked_add(1) != Some(programs.len()) {
        return Err(invalid("native module overlap inventory differs"));
    }
    let main_source_bytes = main
        .rendered_source_bytes
        .ok_or_else(|| invalid("native main rendered source bytes are absent"))?;
    for (program_index, overlap) in overlaps.iter().enumerate() {
        let expected_index = program_index
            .checked_add(1)
            .ok_or_else(|| invalid("native module overlap index overflows"))?;
        let program = programs[expected_index];
        let target_source_bytes = program
            .rendered_source_bytes
            .ok_or_else(|| invalid("native rendered source bytes are absent"))?;
        let overlap_entries = overlap
            .contiguous_prefix_entry_count
            .checked_add(overlap.additional_scattered_entry_count)
            .ok_or_else(|| invalid("native module overlap entry count overflows"))?;
        let overlap_source_bytes = overlap
            .contiguous_prefix_source_bytes
            .checked_add(overlap.additional_scattered_source_bytes)
            .ok_or_else(|| invalid("native module overlap source bytes overflow"))?;
        if overlap.program_index != count(expected_index, "native module overlap program")?
            || overlap.native_identity != program.native_identity
            || overlap.evidence_identity != overlap.expected_evidence_identity(main.native_identity)
            || overlap.contiguous_prefix_entry_count > main.rendered_entry_count
            || overlap.contiguous_prefix_source_bytes > main_source_bytes
            || overlap.contiguous_prefix_source_bytes < overlap.contiguous_prefix_entry_count
            || overlap_entries > program.rendered_entry_count
            || overlap_source_bytes > target_source_bytes
            || overlap.additional_scattered_source_bytes < overlap.additional_scattered_entry_count
            || (overlap.contiguous_prefix_entry_count == 0)
                != (overlap.contiguous_prefix_source_bytes == 0)
            || (overlap.additional_scattered_entry_count == 0)
                != (overlap.additional_scattered_source_bytes == 0)
        {
            return Err(invalid("native module overlap evidence differs"));
        }
        if program.shared_prefix_source_program_index == Some(0)
            && (overlap.contiguous_prefix_entry_count != program.shared_prefix_entry_count()
                || Some(overlap.contiguous_prefix_source_bytes)
                    != program.shared_prefix_source_bytes)
        {
            return Err(invalid("native main-prefix overlap evidence differs"));
        }
    }
    Ok(())
}

/// Exact target-entry overlap for one ordered earlier-program/later-program
/// pair. The evidence does not imply that either program reuses the other's
/// compiled objects.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeTrainingProgramPairOverlap {
    source_program_index: u64,
    source_native_identity: u64,
    target_program_index: u64,
    target_native_identity: u64,
    evidence_identity: u64,
    contiguous_prefix_entry_count: u64,
    contiguous_prefix_source_bytes: u64,
    additional_scattered_entry_count: u64,
    additional_scattered_source_bytes: u64,
}

impl NativeTrainingProgramPairOverlap {
    fn from_preparation(
        overlap: &NativeCpuProgramPairOverlap,
        programs: &[&NativeTrainingProgramReport],
    ) -> Result<Self> {
        let source = programs
            .get(overlap.source_program_index())
            .ok_or_else(|| invalid("native pair-overlap source program is absent"))?;
        let target = programs
            .get(overlap.target_program_index())
            .ok_or_else(|| invalid("native pair-overlap target program is absent"))?;
        Ok(Self {
            source_program_index: count(
                overlap.source_program_index(),
                "native pair-overlap source program",
            )?,
            source_native_identity: source.native_identity(),
            target_program_index: count(
                overlap.target_program_index(),
                "native pair-overlap target program",
            )?,
            target_native_identity: target.native_identity(),
            evidence_identity: 0,
            contiguous_prefix_entry_count: count(
                overlap.contiguous_prefix_entry_count(),
                "native pair-overlap prefix entry",
            )?,
            contiguous_prefix_source_bytes: count(
                overlap.contiguous_prefix_source_bytes(),
                "native pair-overlap prefix source byte",
            )?,
            additional_scattered_entry_count: count(
                overlap.additional_scattered_entry_count(),
                "native pair-overlap scattered entry",
            )?,
            additional_scattered_source_bytes: count(
                overlap.additional_scattered_source_bytes(),
                "native pair-overlap scattered source byte",
            )?,
        }
        .with_evidence_identity())
    }

    fn with_evidence_identity(mut self) -> Self {
        self.evidence_identity = self.expected_evidence_identity();
        self
    }

    fn expected_evidence_identity(&self) -> u64 {
        evidence_identity(
            b"rustgrad-native-training-program-pair-overlap-v21",
            &[
                self.source_program_index,
                self.source_native_identity,
                self.target_program_index,
                self.target_native_identity,
                self.contiguous_prefix_entry_count,
                self.contiguous_prefix_source_bytes,
                self.additional_scattered_entry_count,
                self.additional_scattered_source_bytes,
            ],
        )
    }
}

/// One exact source unit in the immutable build plan for a program's unique
/// suffix. Equal identities denote the same generated C and toolchain context;
/// this report does not claim that the object was reused.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeTrainingTranslationUnit {
    program_index: u64,
    native_identity: u64,
    ordinal: u64,
    translation_unit_identity: u64,
    evidence_identity: u64,
    entry_count: u64,
    rendered_source_bytes: u64,
}

impl NativeTrainingTranslationUnit {
    fn from_preparation(
        unit: &NativeCpuTranslationUnitEvidence,
        programs: &[&NativeTrainingProgramReport],
    ) -> Result<Self> {
        let program = programs
            .get(unit.program_index())
            .ok_or_else(|| invalid("native translation-unit program is absent"))?;
        Ok(Self {
            program_index: count(unit.program_index(), "native translation-unit program")?,
            native_identity: program.native_identity(),
            ordinal: count(unit.ordinal(), "native translation-unit ordinal")?,
            translation_unit_identity: unit.identity(),
            evidence_identity: 0,
            entry_count: count(unit.entry_count(), "native translation-unit entry")?,
            rendered_source_bytes: count(
                unit.rendered_source_bytes(),
                "native translation-unit rendered source byte",
            )?,
        }
        .with_evidence_identity())
    }

    fn with_evidence_identity(mut self) -> Self {
        self.evidence_identity = self.expected_evidence_identity();
        self
    }

    fn expected_evidence_identity(&self) -> u64 {
        evidence_identity(
            b"rustgrad-native-training-translation-unit-v21",
            &[
                self.program_index,
                self.native_identity,
                self.ordinal,
                self.translation_unit_identity,
                self.entry_count,
                self.rendered_source_bytes,
            ],
        )
    }
}

fn evidence_identity(domain: &[u8], values: &[u64]) -> u64 {
    let mut identity = 0xcbf29ce484222325u64;
    for byte in domain {
        identity = (identity ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    for value in values {
        for byte in value.to_le_bytes() {
            identity = (identity ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    identity
}

fn validate_program_pair_overlaps(
    overlaps: &[NativeTrainingProgramPairOverlap],
    programs: &[&NativeTrainingProgramReport],
    main_overlaps: &[NativeTrainingModuleOverlap],
) -> Result<()> {
    let expected_count = programs
        .len()
        .checked_mul(programs.len().saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or_else(|| invalid("native pair-overlap inventory overflows"))?;
    if overlaps.len() != expected_count {
        return Err(invalid("native pair-overlap inventory differs"));
    }
    let mut ordinal = 0usize;
    for target_index in 1..programs.len() {
        let target = programs[target_index];
        let target_source_bytes = target
            .rendered_source_bytes
            .ok_or_else(|| invalid("native target rendered source bytes are absent"))?;
        for (source_index, source) in programs[..target_index].iter().enumerate() {
            let overlap = &overlaps[ordinal];
            ordinal = ordinal
                .checked_add(1)
                .ok_or_else(|| invalid("native pair-overlap ordinal overflows"))?;
            let source_source_bytes = source
                .rendered_source_bytes
                .ok_or_else(|| invalid("native source rendered source bytes are absent"))?;
            let overlap_entries = overlap
                .contiguous_prefix_entry_count
                .checked_add(overlap.additional_scattered_entry_count)
                .ok_or_else(|| invalid("native pair-overlap entry count overflows"))?;
            let overlap_bytes = overlap
                .contiguous_prefix_source_bytes
                .checked_add(overlap.additional_scattered_source_bytes)
                .ok_or_else(|| invalid("native pair-overlap source bytes overflow"))?;
            if overlap.source_program_index
                != count(source_index, "native pair-overlap source program")?
                || overlap.source_native_identity != source.native_identity
                || overlap.target_program_index
                    != count(target_index, "native pair-overlap target program")?
                || overlap.target_native_identity != target.native_identity
                || overlap.evidence_identity != overlap.expected_evidence_identity()
                || overlap.contiguous_prefix_entry_count > source.rendered_entry_count
                || overlap.contiguous_prefix_entry_count > target.rendered_entry_count
                || overlap.contiguous_prefix_source_bytes > source_source_bytes
                || overlap.contiguous_prefix_source_bytes > target_source_bytes
                || overlap.contiguous_prefix_source_bytes < overlap.contiguous_prefix_entry_count
                || overlap_entries > target.rendered_entry_count
                || overlap_bytes > target_source_bytes
                || overlap.additional_scattered_source_bytes
                    < overlap.additional_scattered_entry_count
                || (overlap.contiguous_prefix_entry_count == 0)
                    != (overlap.contiguous_prefix_source_bytes == 0)
                || (overlap.additional_scattered_entry_count == 0)
                    != (overlap.additional_scattered_source_bytes == 0)
            {
                return Err(invalid("native pair-overlap evidence differs"));
            }
            if target.shared_prefix_source_program_index == Some(overlap.source_program_index)
                && (overlap.contiguous_prefix_entry_count != target.shared_prefix_entry_count()
                    || Some(overlap.contiguous_prefix_source_bytes)
                        != target.shared_prefix_source_bytes)
            {
                return Err(invalid("native pair-prefix overlap evidence differs"));
            }
            if source_index == 0 {
                let main = &main_overlaps[target_index - 1];
                if overlap.target_program_index != main.program_index
                    || overlap.target_native_identity != main.native_identity
                    || overlap.contiguous_prefix_entry_count != main.contiguous_prefix_entry_count
                    || overlap.contiguous_prefix_source_bytes != main.contiguous_prefix_source_bytes
                    || overlap.additional_scattered_entry_count
                        != main.additional_scattered_entry_count
                    || overlap.additional_scattered_source_bytes
                        != main.additional_scattered_source_bytes
                {
                    return Err(invalid("native main overlap projections differ"));
                }
            }
        }
    }
    Ok(())
}

fn validate_translation_units(
    units: &[NativeTrainingTranslationUnit],
    programs: &[&NativeTrainingProgramReport],
    compiler_processes: &[NativeTrainingCompilerProcessTiming],
) -> Result<()> {
    let mut unit_index = 0usize;
    for (program_index, program) in programs.iter().enumerate() {
        let program_index_wire = count(program_index, "native translation-unit program")?;
        let unique_entries = program
            .unique_rendered_entry_count
            .ok_or_else(|| invalid("native unique rendered entry count is absent"))?;
        let mode = crate::cpu_jit::NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(
            unique_entries,
        );
        let expected_units = mode.map_or(0, |mode| {
            let (combined, objects, _) = mode.process_inventory();
            combined + objects
        });
        let expected_units = usize::try_from(expected_units)
            .map_err(|_| invalid("native translation-unit count overflows"))?;
        let end = unit_index
            .checked_add(expected_units)
            .ok_or_else(|| invalid("native translation-unit inventory overflows"))?;
        let program_units = units
            .get(unit_index..end)
            .ok_or_else(|| invalid("native translation-unit inventory differs"))?;
        let mut entry_count = 0u64;
        let mut source_bytes = 0u64;
        for (ordinal, unit) in program_units.iter().enumerate() {
            if unit.program_index != program_index_wire
                || unit.native_identity != program.native_identity
                || unit.ordinal != count(ordinal, "native translation-unit ordinal")?
                || unit.evidence_identity != unit.expected_evidence_identity()
                || unit.entry_count == 0
                || unit.rendered_source_bytes < unit.entry_count
            {
                return Err(invalid("native translation-unit evidence differs"));
            }
            entry_count = entry_count
                .checked_add(unit.entry_count)
                .ok_or_else(|| invalid("native translation-unit entry count overflows"))?;
            source_bytes = source_bytes
                .checked_add(unit.rendered_source_bytes)
                .ok_or_else(|| invalid("native translation-unit source bytes overflow"))?;
        }
        let unique_source_bytes = program
            .rendered_source_bytes
            .and_then(|rendered| {
                program
                    .shared_prefix_source_bytes
                    .and_then(|shared| rendered.checked_sub(shared))
            })
            .ok_or_else(|| invalid("native unique rendered source bytes are invalid"))?;
        if entry_count != unique_entries || source_bytes != unique_source_bytes {
            return Err(invalid("native translation-unit partition differs"));
        }
        if program.durable_artifact_cache_miss_count != 0 {
            let process_units = compiler_processes
                .iter()
                .filter(|timing| {
                    timing.program_index == program_index_wire
                        && !matches!(timing.process, NativeTrainingCompilerProcessKind::Link)
                })
                .collect::<Vec<_>>();
            if process_units.len() != program_units.len()
                || process_units
                    .iter()
                    .zip(program_units)
                    .any(|(process, unit)| {
                        process.rendered_source_bytes != Some(unit.rendered_source_bytes)
                    })
            {
                return Err(invalid(
                    "native translation-unit compiler process partition differs",
                ));
            }
        }
        unit_index = end;
    }
    if unit_index != units.len() {
        return Err(invalid("native translation-unit inventory differs"));
    }
    for (index, unit) in units.iter().enumerate() {
        for prior in &units[..index] {
            if unit.translation_unit_identity == prior.translation_unit_identity
                && (unit.entry_count != prior.entry_count
                    || unit.rendered_source_bytes != prior.rendered_source_bytes)
            {
                return Err(invalid("native translation-unit identity payload differs"));
            }
        }
    }
    Ok(())
}

/// Derived owner of the compiler subprocess that finishes last in the batch.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeTrainingCompilerCriticalTail {
    program_index: u64,
    native_identity: u64,
    process: NativeTrainingCompilerProcessKind,
    finish_offset: BenchmarkDuration,
    post_main_tail: BenchmarkDuration,
}

struct CompilerProcessValidationContext {
    format_version: u32,
    process_count: u64,
    max_parallel: u64,
    claimed_overlap: BenchmarkDuration,
    prepare_wall_time: BenchmarkDuration,
}

fn compiler_process_evidence(
    preparation: &NativeCpuCompiledAdamWPreparationReport,
    programs: &[&NativeTrainingProgramReport],
    prepare_wall_time: Duration,
) -> Result<(
    Vec<NativeTrainingCompilerProcessTiming>,
    Option<NativeTrainingCompilerCriticalTail>,
)> {
    let timings = preparation
        .compiler_process_timings()
        .iter()
        .map(|timing| NativeTrainingCompilerProcessTiming::from_preparation(timing, programs))
        .collect::<Result<Vec<_>>>()?;
    validate_compiler_process_evidence(
        &timings,
        None,
        programs,
        CompilerProcessValidationContext {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            process_count: count(
                preparation.compiler_process_count(),
                "native compiler process",
            )?,
            max_parallel: count(
                preparation.max_parallel_compiler_process_count(),
                "parallel native compiler process",
            )?,
            claimed_overlap: BenchmarkDuration::from_duration(
                preparation.compiler_process_overlap_wall_time(),
            ),
            prepare_wall_time: BenchmarkDuration::from_duration(prepare_wall_time),
        },
    )
    .map(|tail| (timings, tail))
}

fn validate_compiler_process_evidence(
    timings: &[NativeTrainingCompilerProcessTiming],
    claimed_tail: Option<&NativeTrainingCompilerCriticalTail>,
    programs: &[&NativeTrainingProgramReport],
    context: CompilerProcessValidationContext,
) -> Result<Option<NativeTrainingCompilerCriticalTail>> {
    let CompilerProcessValidationContext {
        format_version,
        process_count,
        max_parallel,
        claimed_overlap,
        prepare_wall_time,
    } = context;
    if count(timings.len(), "native compiler process")? != process_count {
        return Err(invalid("native compiler process timing inventory differs"));
    }
    let mut expected_program = 0usize;
    let mut program_total = vec![0u128; programs.len()];
    let mut program_source_bytes = vec![0u64; programs.len()];
    let mut intervals = Vec::with_capacity(timings.len());
    let mut latest: Option<(u128, &NativeTrainingCompilerProcessTiming)> = None;
    let mut main_finish = 0u128;
    let prepare_wall_time = prepare_wall_time
        .as_nanos()
        .map_err(|_| invalid("invalid native preparation wall duration"))?;
    for timing in timings {
        let program_index = usize::try_from(timing.program_index)
            .map_err(|_| invalid("native compiler process program index overflows"))?;
        if program_index < expected_program || program_index >= programs.len() {
            return Err(invalid("native compiler process program order differs"));
        }
        expected_program = program_index;
        let program = programs[program_index];
        if timing.native_identity != program.native_identity {
            return Err(invalid("native compiler process program identity differs"));
        }
        match (format_version, timing.rendered_source_bytes) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(bytes),
            ) => {
                let expected_zero =
                    matches!(timing.process, NativeTrainingCompilerProcessKind::Link);
                if (bytes == 0) != expected_zero {
                    return Err(invalid("native compiler rendered source bytes differ"));
                }
                program_source_bytes[program_index] = program_source_bytes[program_index]
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("native compiler rendered source bytes overflow"))?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, Some(_)) => {
                return Err(invalid("legacy compiler process has rendered source bytes"));
            }
            _ => return Err(invalid("native compiler rendered source bytes are absent")),
        }
        let (requested, started, finished) = timing.interval()?;
        if requested > prepare_wall_time
            || started > prepare_wall_time
            || finished > prepare_wall_time
        {
            return Err(invalid(
                "native compiler process exceeds preparation wall time",
            ));
        }
        let duration = finished
            .checked_sub(started)
            .ok_or_else(|| invalid("native compiler process interval is reversed"))?;
        program_total[program_index] = program_total[program_index]
            .checked_add(duration)
            .ok_or_else(|| invalid("native compiler program duration overflows"))?;
        intervals.push((started, finished));
        if program_index == 0 {
            main_finish = main_finish.max(finished);
        }
        if latest.is_none_or(|(latest_finish, _)| finished > latest_finish) {
            latest = Some((finished, timing));
        }
    }
    for (program_index, program) in programs.iter().enumerate() {
        let program_index_wire = count(program_index, "native compiler program")?;
        let expected = program
            .preparation_timing
            .as_ref()
            .and_then(|timing| timing.compiler_process_total)
            .ok_or_else(|| invalid("native cumulative compiler timing is absent"))?
            .as_nanos()
            .map_err(|_| invalid("invalid native cumulative compiler duration"))?;
        if program_total[program_index] != expected {
            return Err(invalid("native compiler process program duration differs"));
        }
        let actual = timings
            .iter()
            .filter(|timing| timing.program_index == program_index_wire)
            .map(|timing| timing.process)
            .collect::<Vec<_>>();
        let mut expected_kinds = Vec::new();
        if program.combined_compile_link_count() == 1 {
            expected_kinds.push(NativeTrainingCompilerProcessKind::Combined);
        }
        for ordinal in 0..program.object_compile_count() {
            expected_kinds.push(NativeTrainingCompilerProcessKind::Object { ordinal });
        }
        if program.linker_invocation_count() == 1 {
            expected_kinds.push(NativeTrainingCompilerProcessKind::Link);
        }
        if actual != expected_kinds {
            return Err(invalid("native compiler process kind inventory differs"));
        }
        if matches!(
            format_version,
            NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION
        ) {
            let rendered = program
                .rendered_source_bytes
                .ok_or_else(|| invalid("native rendered source bytes are absent"))?;
            let shared = program
                .shared_prefix_source_bytes
                .ok_or_else(|| invalid("native shared-prefix source bytes are absent"))?;
            let unique = rendered
                .checked_sub(shared)
                .ok_or_else(|| invalid("native shared-prefix source bytes exceed program"))?;
            let expected = if program.durable_artifact_cache_miss_count == 0 {
                0
            } else {
                unique
            };
            if program_source_bytes[program_index] != expected {
                return Err(invalid("native compiler source-byte inventory differs"));
            }
        }
    }
    intervals.sort_unstable();
    let total = intervals.iter().try_fold(0u128, |total, (start, finish)| {
        finish
            .checked_sub(*start)
            .and_then(|duration| total.checked_add(duration))
            .ok_or_else(|| invalid("native compiler process duration overflows"))
    })?;
    let mut union = 0u128;
    let mut current: Option<(u128, u128)> = None;
    let mut events = Vec::with_capacity(intervals.len() * 2);
    for (start, finish) in intervals {
        if start != finish {
            events.push((start, true));
            events.push((finish, false));
        }
        match current {
            Some((range_start, range_finish)) if start <= range_finish => {
                current = Some((range_start, range_finish.max(finish)));
            }
            Some((range_start, range_finish)) => {
                union = union
                    .checked_add(range_finish - range_start)
                    .ok_or_else(|| invalid("native compiler process union overflows"))?;
                current = Some((start, finish));
            }
            None => current = Some((start, finish)),
        }
    }
    if let Some((start, finish)) = current {
        union = union
            .checked_add(finish - start)
            .ok_or_else(|| invalid("native compiler process union overflows"))?;
    }
    events.sort_unstable_by_key(|(at, starts)| (*at, *starts));
    let mut active = 0u64;
    let mut observed_max_parallel = u64::from(!timings.is_empty());
    for (_, starts) in events {
        if starts {
            active = active
                .checked_add(1)
                .ok_or_else(|| invalid("native compiler concurrency overflows"))?;
            observed_max_parallel = observed_max_parallel.max(active);
        } else {
            active = active
                .checked_sub(1)
                .ok_or_else(|| invalid("native compiler concurrency underflows"))?;
        }
    }
    let overlap = total
        .checked_sub(union)
        .ok_or_else(|| invalid("native compiler overlap underflows"))?;
    if overlap
        != claimed_overlap
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler overlap duration"))?
        || observed_max_parallel != max_parallel
    {
        return Err(invalid("native compiler process interval evidence differs"));
    }
    let tail = match latest {
        Some((finish, timing)) => Some(NativeTrainingCompilerCriticalTail {
            program_index: timing.program_index,
            native_identity: timing.native_identity,
            process: timing.process,
            finish_offset: benchmark_duration_from_nanos(finish)?,
            post_main_tail: benchmark_duration_from_nanos(finish.saturating_sub(main_finish))?,
        }),
        None => None,
    };
    if claimed_tail.is_some() && claimed_tail != tail.as_ref() {
        return Err(invalid("native compiler critical-tail attribution differs"));
    }
    Ok(tail)
}

fn benchmark_duration_from_nanos(nanos: u128) -> Result<BenchmarkDuration> {
    let seconds = nanos / 1_000_000_000;
    let subsecond = nanos % 1_000_000_000;
    Ok(BenchmarkDuration {
        secs: u64::try_from(seconds)
            .map_err(|_| invalid("native compiler timing seconds overflow"))?,
        nanos: u32::try_from(subsecond)
            .map_err(|_| invalid("native compiler timing nanoseconds overflow"))?,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProgramInspection {
    capture_identity: u64,
    execution_plan: ExecutionPlanSummary,
}

/// Exact host wall-time partition for preparing one strict-native CPU program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingPreparationTiming {
    total: BenchmarkDuration,
    layout: BenchmarkDuration,
    render: BenchmarkDuration,
    compiler_process: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compiler_process_total: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    linker_process: Option<BenchmarkDuration>,
    module_load: BenchmarkDuration,
    residual: BenchmarkDuration,
}

impl NativeTrainingPreparationTiming {
    fn from_preparation(preparation: &NativeCpuProgramPreparationReport) -> Self {
        let phases = preparation.phases();
        Self {
            total: BenchmarkDuration::from_duration(preparation.wall_time()),
            layout: BenchmarkDuration::from_duration(phases.layout_wall_time()),
            render: BenchmarkDuration::from_duration(phases.render_wall_time()),
            compiler_process: BenchmarkDuration::from_duration(phases.compiler_process_wall_time()),
            compiler_process_total: Some(BenchmarkDuration::from_duration(
                phases.compiler_process_total_wall_time(),
            )),
            linker_process: Some(BenchmarkDuration::from_duration(
                phases.linker_process_wall_time(),
            )),
            module_load: BenchmarkDuration::from_duration(phases.module_load_wall_time()),
            residual: BenchmarkDuration::from_duration(phases.residual_wall_time()),
        }
    }

    fn validate(&self, work: &NativeTrainingProgramReport, format_version: u32) -> Result<()> {
        let total = self
            .total
            .as_nanos()
            .map_err(|_| invalid("invalid native preparation total duration"))?;
        let partitioned = [
            self.layout,
            self.render,
            self.compiler_process,
            self.module_load,
            self.residual,
        ]
        .into_iter()
        .try_fold(0u128, |total, duration| {
            duration
                .as_nanos()
                .map_err(|_| invalid("invalid native preparation phase duration"))?
                .checked_add(total)
                .ok_or_else(|| invalid("native preparation phase duration overflows"))
        })?;
        let extended_timing_is_present =
            self.compiler_process_total.is_some() || self.linker_process.is_some();
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V14 && extended_timing_is_present {
            return Err(invalid(
                "legacy native preparation has chunk compiler timing",
            ));
        }
        let compiler_process = self
            .compiler_process
            .as_nanos()
            .map_err(|_| invalid("invalid native compiler process duration"))?;
        let compiler_process_total = match (
            format_version,
            self.compiler_process_total,
            self.linker_process,
        ) {
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
                Some(total),
                Some(linker),
            ) => {
                let total = total
                    .as_nanos()
                    .map_err(|_| invalid("invalid native cumulative compiler duration"))?;
                let linker = linker
                    .as_nanos()
                    .map_err(|_| invalid("invalid native linker duration"))?;
                if total < compiler_process
                    || (work.compiler_invocation_count() <= 1 && total != compiler_process)
                    || linker > compiler_process
                    || (work.linker_invocation_count() == 0 && linker != 0)
                {
                    return Err(invalid("native chunk compiler timing differs"));
                }
                total
            }
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
                _,
                _,
            ) => {
                return Err(invalid("native chunk compiler timing is absent"));
            }
            (_, None, None) => compiler_process,
            _ => unreachable!("legacy extended compiler timing was rejected"),
        };
        if partitioned != total
            || (work.compiler_invocation_count == 0
                && self.compiler_process != BenchmarkDuration::from_duration(Duration::ZERO))
            || (work.compiler_invocation_count == 0 && compiler_process_total != 0)
            || (work.loaded_module_count == 0
                && self.module_load != BenchmarkDuration::from_duration(Duration::ZERO))
        {
            return Err(invalid("native preparation phases do not match work"));
        }
        Ok(())
    }

    pub const fn total(&self) -> BenchmarkDuration {
        self.total
    }

    pub const fn layout(&self) -> BenchmarkDuration {
        self.layout
    }

    pub const fn render(&self) -> BenchmarkDuration {
        self.render
    }

    pub const fn compiler_process(&self) -> BenchmarkDuration {
        self.compiler_process
    }

    pub const fn compiler_process_total(&self) -> Option<BenchmarkDuration> {
        self.compiler_process_total
    }

    pub const fn linker_process(&self) -> Option<BenchmarkDuration> {
        self.linker_process
    }

    pub const fn module_load(&self) -> BenchmarkDuration {
        self.module_load
    }

    pub const fn residual(&self) -> BenchmarkDuration {
        self.residual
    }
}

/// Immutable logical work and recurrent-state facts for one compiled AdamW
/// plan. Inspection prepares no target and exposes no capture.
///
/// Equality compares only that logical topology. Optional compile-phase
/// observations describe one construction and remain independently
/// inspectable, but do not make equivalent plans unequal.
#[derive(Clone, Debug)]
pub struct CompiledAdamWInspection {
    initial_replay_step: u64,
    main: ProgramInspection,
    accumulation: Option<ProgramInspection>,
    partial_flush: Option<ProgramInspection>,
    zero_grad: Option<ProgramInspection>,
    evaluation: Option<ProgramInspection>,
    recurrent_state_count: usize,
    recurrent_state_bytes: usize,
    compile_phases: Option<CompiledTrainingCompileObservation>,
}

impl PartialEq for CompiledAdamWInspection {
    fn eq(&self, other: &Self) -> bool {
        self.initial_replay_step == other.initial_replay_step
            && self.main == other.main
            && self.accumulation == other.accumulation
            && self.partial_flush == other.partial_flush
            && self.zero_grad == other.zero_grad
            && self.evaluation == other.evaluation
            && self.recurrent_state_count == other.recurrent_state_count
            && self.recurrent_state_bytes == other.recurrent_state_bytes
    }
}

impl Eq for CompiledAdamWInspection {}

/// One backend-neutral compiled-training construction phase observed before
/// target preparation. Counts describe the immutable graph or schedule at the
/// end of the phase; neither durations nor counts enter program identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledTrainingCompilePhaseObservation {
    wall_time: Duration,
    graph_node_count: Option<usize>,
    logical_schedule_item_count: Option<usize>,
}

impl CompiledTrainingCompilePhaseObservation {
    pub(crate) const fn graph(wall_time: Duration, graph_node_count: usize) -> Self {
        Self {
            wall_time,
            graph_node_count: Some(graph_node_count),
            logical_schedule_item_count: None,
        }
    }

    pub(crate) const fn schedule(wall_time: Duration, logical_schedule_item_count: usize) -> Self {
        Self {
            wall_time,
            graph_node_count: None,
            logical_schedule_item_count: Some(logical_schedule_item_count),
        }
    }

    pub fn wall_time(&self) -> Duration {
        self.wall_time
    }

    pub const fn graph_node_count(&self) -> Option<usize> {
        self.graph_node_count
    }

    pub const fn logical_schedule_item_count(&self) -> Option<usize> {
        self.logical_schedule_item_count
    }
}

/// Typed observation of one successful backend-neutral training compilation.
/// The caller-observed wrapper residual is derived later by the scoreboard so
/// configuration, module traversal, and optional evaluator work remain visible.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledTrainingCompileObservation {
    objective_forward: CompiledTrainingCompilePhaseObservation,
    autograd: CompiledTrainingCompilePhaseObservation,
    optimizer_lowering: CompiledTrainingCompilePhaseObservation,
    main_capture: CompiledTrainingCompilePhaseObservation,
    accumulation_capture: Option<CompiledTrainingCompilePhaseObservation>,
    partial_flush: Option<CompiledTrainingCompilePhaseObservation>,
    zero_grad: Option<CompiledTrainingCompilePhaseObservation>,
    evaluation: Option<CompiledTrainingCompilePhaseObservation>,
}

impl CompiledTrainingCompileObservation {
    pub(crate) const fn new(
        objective_forward: CompiledTrainingCompilePhaseObservation,
        autograd: CompiledTrainingCompilePhaseObservation,
        optimizer_lowering: CompiledTrainingCompilePhaseObservation,
        main_capture: CompiledTrainingCompilePhaseObservation,
        accumulation_capture: Option<CompiledTrainingCompilePhaseObservation>,
    ) -> Self {
        Self {
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
            partial_flush: None,
            zero_grad: None,
            evaluation: None,
        }
    }

    pub(crate) fn set_auxiliary(
        &mut self,
        partial_flush: Option<CompiledTrainingCompilePhaseObservation>,
        zero_grad: Option<CompiledTrainingCompilePhaseObservation>,
    ) {
        self.partial_flush = partial_flush;
        self.zero_grad = zero_grad;
    }

    pub(crate) fn set_evaluation(&mut self, evaluation: CompiledTrainingCompilePhaseObservation) {
        self.evaluation = Some(evaluation);
    }

    pub const fn compile_count(&self) -> u64 {
        1
    }

    pub const fn objective_forward(&self) -> CompiledTrainingCompilePhaseObservation {
        self.objective_forward
    }

    pub const fn autograd(&self) -> CompiledTrainingCompilePhaseObservation {
        self.autograd
    }

    pub const fn optimizer_lowering(&self) -> CompiledTrainingCompilePhaseObservation {
        self.optimizer_lowering
    }

    pub const fn main_capture(&self) -> CompiledTrainingCompilePhaseObservation {
        self.main_capture
    }

    pub const fn accumulation_capture(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.accumulation_capture
    }

    pub const fn partial_flush(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.partial_flush
    }

    pub const fn zero_grad(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.zero_grad
    }

    pub const fn evaluation(&self) -> Option<CompiledTrainingCompilePhaseObservation> {
        self.evaluation
    }

    /// Checked sum of the instrumented compile phases. The enclosing caller
    /// wall time may be larger; the scoreboard records that remainder.
    pub fn measured_wall_time(&self) -> Option<Duration> {
        [
            Some(self.objective_forward),
            Some(self.autograd),
            Some(self.optimizer_lowering),
            Some(self.main_capture),
            self.accumulation_capture,
            self.partial_flush,
            self.zero_grad,
            self.evaluation,
        ]
        .into_iter()
        .flatten()
        .try_fold(Duration::ZERO, |total, phase| {
            total.checked_add(phase.wall_time())
        })
    }
}

impl CompiledAdamWInspection {
    pub(crate) fn new(
        initial_replay_step: u64,
        main: (u64, ExecutionPlanSummary),
        accumulation: Option<(u64, ExecutionPlanSummary)>,
        partial_flush: Option<(u64, ExecutionPlanSummary)>,
        zero_grad: Option<(u64, ExecutionPlanSummary)>,
        evaluation: Option<(u64, ExecutionPlanSummary)>,
        recurrent_state: (usize, usize),
    ) -> Self {
        let program = |(capture_identity, execution_plan)| ProgramInspection {
            capture_identity,
            execution_plan,
        };
        Self {
            initial_replay_step,
            main: program(main),
            accumulation: accumulation.map(program),
            partial_flush: partial_flush.map(program),
            zero_grad: zero_grad.map(program),
            evaluation: evaluation.map(program),
            recurrent_state_count: recurrent_state.0,
            recurrent_state_bytes: recurrent_state.1,
            compile_phases: None,
        }
    }

    pub(crate) fn with_compile_phases(
        mut self,
        compile_phases: Option<CompiledTrainingCompileObservation>,
    ) -> Self {
        self.compile_phases = compile_phases;
        self
    }

    pub const fn initial_replay_step(&self) -> u64 {
        self.initial_replay_step
    }

    pub const fn main(&self) -> (u64, &ExecutionPlanSummary) {
        (self.main.capture_identity, &self.main.execution_plan)
    }

    pub fn accumulation(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.accumulation
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub fn partial_flush(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.partial_flush
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub fn evaluation(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.evaluation
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub fn zero_grad(&self) -> Option<(u64, &ExecutionPlanSummary)> {
        self.zero_grad
            .as_ref()
            .map(|program| (program.capture_identity, &program.execution_plan))
    }

    pub const fn recurrent_state_count(&self) -> usize {
        self.recurrent_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> usize {
        self.recurrent_state_bytes
    }

    pub fn compile_phases(&self) -> Option<&CompiledTrainingCompileObservation> {
        self.compile_phases.as_ref()
    }
}

/// Static logical work, identity, and cache facts for one prepared pure
/// training program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingProgramReport {
    capture_identity: u64,
    native_identity: u64,
    vectorized: bool,
    execution_plan_identity: u64,
    logical_schedule_item_count: u64,
    peak_logical_temporary_allocation_count: u64,
    peak_logical_temporary_bytes: u64,
    native_item_count: u64,
    cache_hit_count: u64,
    cache_miss_count: u64,
    #[serde(default)]
    rendered_entry_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rendered_source_bytes: Option<u64>,
    #[serde(default)]
    loaded_module_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    referenced_module_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unique_rendered_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_source_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_source_program_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix_source_native_identity: Option<u64>,
    #[serde(default)]
    durable_artifact_cache_hit_count: u64,
    #[serde(default)]
    durable_artifact_cache_miss_count: u64,
    #[serde(default)]
    compiler_invocation_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    combined_compile_link_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    object_compile_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    linker_invocation_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preparation_timing: Option<NativeTrainingPreparationTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dispatch_segmentation: Option<NativeCpuDispatchSegmentation>,
}

impl NativeTrainingProgramReport {
    fn new(
        inspection: &ProgramInspection,
        preparation: &NativeCpuProgramPreparationReport,
        prior_native_identities: &[u64],
    ) -> Result<Self> {
        if inspection.capture_identity != preparation.capture_identity()
            || &inspection.execution_plan != preparation.execution_plan()
        {
            return Err(invalid("plan and native preparation differ"));
        }
        let plan = &inspection.execution_plan;
        let shared_prefix_source_program_index = preparation
            .work()
            .shared_prefix_source_program()
            .map(|source| count(source, "native shared-prefix source program"))
            .transpose()?;
        let shared_prefix_source_native_identity = preparation
            .work()
            .shared_prefix_source_program()
            .map(|source| {
                prior_native_identities
                    .get(source)
                    .copied()
                    .ok_or_else(|| invalid("native shared-prefix source program is not earlier"))
            })
            .transpose()?;
        Ok(Self {
            capture_identity: preparation.capture_identity(),
            native_identity: preparation.native_identity(),
            vectorized: preparation.is_vectorized(),
            execution_plan_identity: plan.identity,
            logical_schedule_item_count: count(plan.schedule_item_count, "schedule item")?,
            peak_logical_temporary_allocation_count: count(
                plan.peak_logical_allocations,
                "peak logical allocation",
            )?,
            peak_logical_temporary_bytes: count(
                plan.peak_logical_bytes,
                "peak logical temporary byte",
            )?,
            native_item_count: count(preparation.native_item_count(), "native item")?,
            cache_hit_count: count(preparation.cache_hit_count(), "cache hit")?,
            cache_miss_count: count(preparation.cache_miss_count(), "cache miss")?,
            rendered_entry_count: count(
                preparation.work().rendered_entry_count(),
                "rendered entry",
            )?,
            rendered_source_bytes: Some(count(
                preparation.work().rendered_source_bytes(),
                "rendered source byte",
            )?),
            loaded_module_count: count(preparation.work().loaded_module_count(), "loaded module")?,
            referenced_module_count: Some(count(
                preparation.work().referenced_module_count(),
                "referenced module",
            )?),
            unique_rendered_entry_count: Some(count(
                preparation.work().unique_rendered_entry_count(),
                "unique rendered entry",
            )?),
            shared_prefix_entry_count: Some(count(
                preparation.work().shared_prefix_entry_count(),
                "shared prefix entry",
            )?),
            shared_prefix_source_bytes: Some(count(
                preparation.work().shared_prefix_source_bytes(),
                "shared prefix source byte",
            )?),
            shared_prefix_source_program_index,
            shared_prefix_source_native_identity,
            durable_artifact_cache_hit_count: count(
                preparation.work().durable_artifact_cache_hit_count(),
                "durable artifact cache hit",
            )?,
            durable_artifact_cache_miss_count: count(
                preparation.work().durable_artifact_cache_miss_count(),
                "durable artifact cache miss",
            )?,
            compiler_invocation_count: count(
                preparation.work().compiler_invocation_count(),
                "compiler invocation",
            )?,
            combined_compile_link_count: Some(count(
                preparation.work().combined_compile_link_count(),
                "combined compiler invocation",
            )?),
            object_compile_count: Some(count(
                preparation.work().object_compile_count(),
                "object compiler invocation",
            )?),
            linker_invocation_count: Some(count(
                preparation.work().linker_invocation_count(),
                "linker invocation",
            )?),
            preparation_timing: Some(NativeTrainingPreparationTiming::from_preparation(
                preparation,
            )),
            dispatch_segmentation: Some(*preparation.dispatch_segmentation()),
        })
    }

    fn validate(&self, format_version: u32) -> Result<()> {
        if self
            .cache_hit_count
            .checked_add(self.cache_miss_count)
            .ok_or_else(|| invalid("native program cache count overflows"))?
            != self.native_item_count
        {
            return Err(invalid("native program cache inventory differs"));
        }
        let prefix_evidence_is_present = self.referenced_module_count.is_some()
            || self.unique_rendered_entry_count.is_some()
            || self.shared_prefix_entry_count.is_some()
            || self.shared_prefix_source_program_index.is_some()
            || self.shared_prefix_source_native_identity.is_some();
        let source_byte_evidence_is_present =
            self.rendered_source_bytes.is_some() || self.shared_prefix_source_bytes.is_some();
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V19 {
            if source_byte_evidence_is_present {
                return Err(invalid(
                    "legacy native program has rendered source evidence",
                ));
            }
        } else {
            let rendered_source_bytes = self
                .rendered_source_bytes
                .ok_or_else(|| invalid("native rendered source bytes are absent"))?;
            let shared_prefix_source_bytes = self
                .shared_prefix_source_bytes
                .ok_or_else(|| invalid("native shared-prefix source bytes are absent"))?;
            let unique_rendered_source_bytes = rendered_source_bytes
                .checked_sub(shared_prefix_source_bytes)
                .ok_or_else(|| invalid("native shared-prefix source bytes exceed program"))?;
            if (self.rendered_entry_count == 0) != (rendered_source_bytes == 0)
                || (self.shared_prefix_entry_count() == 0) != (shared_prefix_source_bytes == 0)
                || rendered_source_bytes < self.rendered_entry_count
                || shared_prefix_source_bytes < self.shared_prefix_entry_count()
                || unique_rendered_source_bytes < self.unique_rendered_entry_count()
            {
                return Err(invalid("native rendered source byte evidence differs"));
            }
        }
        let chunk_compiler_evidence_is_present = self.combined_compile_link_count.is_some()
            || self.object_compile_count.is_some()
            || self.linker_invocation_count.is_some();
        let referenced_module_count = self.referenced_module_count.unwrap_or(0);
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V16 {
            if self.dispatch_segmentation.is_some() {
                return Err(invalid(
                    "legacy native program has dispatch segmentation evidence",
                ));
            }
        } else {
            self.dispatch_segmentation
                .as_ref()
                .ok_or_else(|| invalid("native dispatch segmentation evidence is absent"))?
                .authenticates(self.rendered_entry_count, referenced_module_count)
                .then_some(())
                .ok_or_else(|| invalid("native dispatch segmentation evidence differs"))?;
        }
        if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V14 && chunk_compiler_evidence_is_present
        {
            return Err(invalid("legacy native program has chunk compiler evidence"));
        }
        let unique_rendered_entry_count = self.unique_rendered_entry_count.unwrap_or(0);
        let shared_prefix_entry_count = self.shared_prefix_entry_count.unwrap_or(0);
        let preparation = [
            self.rendered_entry_count,
            self.loaded_module_count,
            self.durable_artifact_cache_hit_count,
            self.durable_artifact_cache_miss_count,
            self.compiler_invocation_count,
        ];
        if format_version < NATIVE_TRAINING_REPORT_FORMAT_V5 {
            if preparation.into_iter().any(|value| value != 0) || prefix_evidence_is_present {
                return Err(invalid(
                    "legacy native program has module preparation evidence",
                ));
            }
        } else if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V13 {
            let durable_access_count = self
                .durable_artifact_cache_hit_count
                .checked_add(self.durable_artifact_cache_miss_count)
                .ok_or_else(|| invalid("durable artifact cache count overflows"))?;
            let expected_modules = u64::from(self.rendered_entry_count != 0);
            let rendered_inventory_is_valid = if format_version <= NATIVE_TRAINING_REPORT_FORMAT_V8
            {
                self.rendered_entry_count == self.native_item_count
            } else {
                self.rendered_entry_count <= self.native_item_count
                    && (self.rendered_entry_count == 0) == (self.native_item_count == 0)
            };
            if !rendered_inventory_is_valid
                || self.loaded_module_count != expected_modules
                || durable_access_count > self.loaded_module_count
                || self.compiler_invocation_count != self.durable_artifact_cache_miss_count
            {
                return Err(invalid(
                    "native program module preparation evidence differs",
                ));
            }
            if prefix_evidence_is_present {
                return Err(invalid("legacy native program has prefix-module evidence"));
            }
        } else {
            if self.referenced_module_count.is_none()
                || self.unique_rendered_entry_count.is_none()
                || self.shared_prefix_entry_count.is_none()
            {
                return Err(invalid("native program prefix-module evidence is absent"));
            }
            let durable_access_count = self
                .durable_artifact_cache_hit_count
                .checked_add(self.durable_artifact_cache_miss_count)
                .ok_or_else(|| invalid("durable artifact cache count overflows"))?;
            let rendered_inventory_is_valid = self.rendered_entry_count <= self.native_item_count
                && (self.rendered_entry_count == 0) == (self.native_item_count == 0);
            let prefix_partition = self
                .unique_rendered_entry_count()
                .checked_add(self.shared_prefix_entry_count())
                .ok_or_else(|| invalid("native prefix entry count overflows"))?;
            if !rendered_inventory_is_valid
                || prefix_partition != self.rendered_entry_count
                || self.cache_hit_count < shared_prefix_entry_count
                || self.loaded_module_count != u64::from(unique_rendered_entry_count != 0)
                || (self.rendered_entry_count == 0) != (referenced_module_count == 0)
                || (shared_prefix_entry_count == 0)
                    != self.shared_prefix_source_program_index.is_none()
                || (shared_prefix_entry_count == 0)
                    != self.shared_prefix_source_native_identity.is_none()
                || referenced_module_count
                    != self
                        .loaded_module_count
                        .checked_add(u64::from(shared_prefix_entry_count != 0))
                        .ok_or_else(|| invalid("native module reference count overflows"))?
                || durable_access_count > self.loaded_module_count
            {
                return Err(invalid(
                    "native program prefix-module preparation evidence differs",
                ));
            }
            if format_version == NATIVE_TRAINING_REPORT_FORMAT_V14 {
                if self.compiler_invocation_count != self.durable_artifact_cache_miss_count {
                    return Err(invalid(
                        "native program prefix-module preparation evidence differs",
                    ));
                }
            } else {
                let (Some(combined), Some(objects), Some(linker)) = (
                    self.combined_compile_link_count,
                    self.object_compile_count,
                    self.linker_invocation_count,
                ) else {
                    return Err(invalid("native program chunk compiler evidence is absent"));
                };
                let compiler_invocations = combined
                    .checked_add(objects)
                    .and_then(|count| count.checked_add(linker))
                    .ok_or_else(|| invalid("native compiler invocation count overflows"))?;
                let compiler_mode_is_valid = if self.durable_artifact_cache_miss_count == 0 {
                    compiler_invocations == 0
                } else {
                    crate::cpu_jit::NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(
                        unique_rendered_entry_count,
                    )
                    .is_some_and(|mode| mode.process_inventory() == (combined, objects, linker))
                };
                if compiler_invocations != self.compiler_invocation_count
                    || objects > unique_rendered_entry_count
                    || !compiler_mode_is_valid
                {
                    return Err(invalid("native program chunk compiler evidence differs"));
                }
            }
        }
        match (format_version, &self.preparation_timing) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, None) => {}
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
                Some(timing),
            ) => timing.validate(self, format_version)?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V6, Some(_)) => {
                return Err(invalid(
                    "legacy native program has preparation phase timing",
                ));
            }
            _ => return Err(invalid("native program preparation timing differs")),
        }
        Ok(())
    }

    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn native_identity(&self) -> u64 {
        self.native_identity
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    pub const fn execution_plan_identity(&self) -> u64 {
        self.execution_plan_identity
    }

    pub const fn logical_schedule_item_count(&self) -> u64 {
        self.logical_schedule_item_count
    }

    pub const fn peak_logical_temporary_allocation_count(&self) -> u64 {
        self.peak_logical_temporary_allocation_count
    }

    pub const fn peak_logical_temporary_bytes(&self) -> u64 {
        self.peak_logical_temporary_bytes
    }

    /// Logical schedule-item coverage used by the cache inventory.
    pub const fn native_item_count(&self) -> u64 {
        self.native_item_count
    }

    /// Logical schedule items covered by process-local cache hits.
    pub const fn cache_hit_count(&self) -> u64 {
        self.cache_hit_count
    }

    /// Logical schedule items covered by process-local cache misses.
    pub const fn cache_miss_count(&self) -> u64 {
        self.cache_miss_count
    }

    /// Physical compiled entries emitted for the logical schedule inventory.
    pub const fn rendered_entry_count(&self) -> u64 {
        self.rendered_entry_count
    }

    pub const fn loaded_module_count(&self) -> u64 {
        self.loaded_module_count
    }

    pub const fn referenced_module_count(&self) -> u64 {
        match self.referenced_module_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn unique_rendered_entry_count(&self) -> u64 {
        match self.unique_rendered_entry_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn shared_prefix_entry_count(&self) -> u64 {
        match self.shared_prefix_entry_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn shared_prefix_source_program_index(&self) -> Option<u64> {
        self.shared_prefix_source_program_index
    }

    pub const fn shared_prefix_source_native_identity(&self) -> Option<u64> {
        self.shared_prefix_source_native_identity
    }

    pub const fn durable_artifact_cache_hit_count(&self) -> u64 {
        self.durable_artifact_cache_hit_count
    }

    pub const fn durable_artifact_cache_miss_count(&self) -> u64 {
        self.durable_artifact_cache_miss_count
    }

    pub const fn compiler_invocation_count(&self) -> u64 {
        self.compiler_invocation_count
    }

    pub const fn combined_compile_link_count(&self) -> u64 {
        match self.combined_compile_link_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn object_compile_count(&self) -> u64 {
        match self.object_compile_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn linker_invocation_count(&self) -> u64 {
        match self.linker_invocation_count {
            Some(count) => count,
            None => 0,
        }
    }

    pub const fn preparation_timing(&self) -> Option<&NativeTrainingPreparationTiming> {
        self.preparation_timing.as_ref()
    }

    pub const fn dispatch_segmentation(&self) -> Option<&NativeCpuDispatchSegmentation> {
        self.dispatch_segmentation.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NativeCompilerAggregate {
    process_count: u64,
    module_job_count: u64,
    effective_wall_sum: u128,
    effective_wall_max: u128,
    cumulative_wall_sum: u128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct NativeRenderAggregate {
    job_count: u64,
    wall_sum: u128,
    wall_max: u128,
}

impl NativeRenderAggregate {
    fn from_programs<'a>(
        programs: impl IntoIterator<Item = &'a NativeTrainingProgramReport>,
    ) -> Result<Self> {
        programs
            .into_iter()
            .try_fold(Self::default(), |aggregate, program| {
                let render = program
                    .preparation_timing
                    .as_ref()
                    .ok_or_else(|| invalid("native program preparation timing is absent"))?
                    .render
                    .as_nanos()
                    .map_err(|_| invalid("invalid native render duration"))?;
                Ok(Self {
                    job_count: aggregate
                        .job_count
                        .checked_add(u64::from(render != 0))
                        .ok_or_else(|| invalid("native render job count overflows"))?,
                    wall_sum: aggregate
                        .wall_sum
                        .checked_add(render)
                        .ok_or_else(|| invalid("native render duration overflows"))?,
                    wall_max: aggregate.wall_max.max(render),
                })
            })
    }
}

impl NativeCompilerAggregate {
    fn from_programs<'a>(
        programs: impl IntoIterator<Item = &'a NativeTrainingProgramReport>,
        format_version: u32,
    ) -> Result<Self> {
        programs
            .into_iter()
            .try_fold(Self::default(), |sum, program| {
                let process_count = sum
                    .process_count
                    .checked_add(program.compiler_invocation_count)
                    .ok_or_else(|| invalid("native compiler process count overflows"))?;
                let jobs = program
                    .durable_artifact_cache_hit_count
                    .checked_add(program.durable_artifact_cache_miss_count)
                    .ok_or_else(|| invalid("native module job count overflows"))?;
                let module_job_count = sum
                    .module_job_count
                    .checked_add(jobs)
                    .ok_or_else(|| invalid("native module job count overflows"))?;
                let timing = program
                    .preparation_timing
                    .as_ref()
                    .ok_or_else(|| invalid("native program preparation timing is absent"))?;
                let effective = timing
                    .compiler_process
                    .as_nanos()
                    .map_err(|_| invalid("invalid native compiler process duration"))?;
                let cumulative = if format_version >= NATIVE_TRAINING_REPORT_FORMAT_V15 {
                    timing
                        .compiler_process_total
                        .ok_or_else(|| invalid("native cumulative compiler timing is absent"))?
                        .as_nanos()
                        .map_err(|_| invalid("invalid native cumulative compiler duration"))?
                } else {
                    effective
                };
                Ok(Self {
                    process_count,
                    module_job_count,
                    effective_wall_sum: sum
                        .effective_wall_sum
                        .checked_add(effective)
                        .ok_or_else(|| invalid("native compiler process duration overflows"))?,
                    effective_wall_max: sum.effective_wall_max.max(effective),
                    cumulative_wall_sum: sum
                        .cumulative_wall_sum
                        .checked_add(cumulative)
                        .ok_or_else(|| invalid("native cumulative compiler duration overflows"))?,
                })
            })
    }

    fn internal_overlap(self) -> Result<u128> {
        self.cumulative_wall_sum
            .checked_sub(self.effective_wall_sum)
            .ok_or_else(|| invalid("native internal compiler overlap underflows"))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointReport {
    capture_identity: u64,
    replay_step: u64,
    byte_count: u64,
    wall_time: BenchmarkDuration,
}

/// First-replay and bounded steady-replay wall time for one measured portion
/// of successful strict-native CPU training-step replay.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReplayTiming {
    first: BenchmarkDuration,
    steady_total: BenchmarkDuration,
    steady: BenchmarkLatencySummary,
}

impl NativeTrainingReplayTiming {
    pub const fn first(&self) -> BenchmarkDuration {
        self.first
    }

    pub const fn steady_total(&self) -> BenchmarkDuration {
        self.steady_total
    }

    pub const fn steady(&self) -> &BenchmarkLatencySummary {
        &self.steady
    }

    fn from_durations(first: Duration, steady: &[Duration]) -> Result<Self> {
        Ok(Self {
            first: BenchmarkDuration::from_duration(first),
            steady_total: sum_durations(steady)?,
            steady: latency_summary(steady)?,
        })
    }

    fn validate(&self, steady_sample_count: u64) -> Result<()> {
        self.first
            .to_duration()
            .map_err(|_| invalid("invalid native training phase duration"))?;
        if self.steady.sample_count != steady_sample_count
            || self.steady.min > self.steady.nearest_rank_p50
            || self.steady.nearest_rank_p50 > self.steady.nearest_rank_p95
            || self.steady.nearest_rank_p95 > self.steady.max
            || self.steady_total < self.steady.max
        {
            return Err(invalid("invalid native training phase summary"));
        }
        for duration in [
            self.steady_total,
            self.steady.min,
            self.steady.nearest_rank_p50,
            self.steady.nearest_rank_p95,
            self.steady.max,
        ] {
            duration
                .to_duration()
                .map_err(|_| invalid("invalid native training phase duration"))?;
        }
        validate_total_duration(&self.steady, self.steady_total)
    }
}

#[derive(Clone, Copy, Debug)]
struct ReplayTiming {
    total: Duration,
    executor: Duration,
    native_dispatcher: Duration,
    executor_host: Duration,
    overhead: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecordingMode {
    Unset,
    Raw,
    Phased,
}

/// One typed backend-neutral compilation phase in the portable training
/// scoreboard. Exactly one inventory kind is present: graph nodes for graph
/// construction phases or logical schedule items for captured programs.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhase {
    wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    graph_node_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    logical_schedule_item_count: Option<u64>,
}

impl NativeTrainingCompilePhase {
    fn from_observation(observation: CompiledTrainingCompilePhaseObservation) -> Result<Self> {
        Ok(Self {
            wall_time: BenchmarkDuration::from_duration(observation.wall_time()),
            graph_node_count: observation
                .graph_node_count()
                .map(|value| count(value, "compiled graph node"))
                .transpose()?,
            logical_schedule_item_count: observation
                .logical_schedule_item_count()
                .map(|value| count(value, "compiled logical schedule item"))
                .transpose()?,
        })
    }

    fn duration(self) -> Result<Duration> {
        self.wall_time
            .to_duration()
            .map_err(|_| invalid("invalid compiled training phase duration"))
    }

    fn validate_graph(self) -> Result<()> {
        self.duration()?;
        if self.graph_node_count.is_none() || self.logical_schedule_item_count.is_some() {
            return Err(invalid("compiled graph phase inventory differs"));
        }
        Ok(())
    }

    fn validate_schedule(self, expected_items: u64) -> Result<()> {
        self.duration()?;
        if self.graph_node_count.is_some()
            || self.logical_schedule_item_count != Some(expected_items)
        {
            return Err(invalid("compiled capture phase inventory differs"));
        }
        Ok(())
    }

    pub const fn wall_time(&self) -> BenchmarkDuration {
        self.wall_time
    }

    pub const fn graph_node_count(&self) -> Option<u64> {
        self.graph_node_count
    }

    pub const fn logical_schedule_item_count(&self) -> Option<u64> {
        self.logical_schedule_item_count
    }
}

/// Exact partition of one caller-observed backend-neutral training compile.
/// Residual time contains wrapper work outside the timed compiler phases; the
/// checked sum of all phase durations and residual equals `compile_wall_time`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingCompilePhaseReport {
    compile_count: u64,
    objective_forward: NativeTrainingCompilePhase,
    autograd: NativeTrainingCompilePhase,
    optimizer_lowering: NativeTrainingCompilePhase,
    main_capture: NativeTrainingCompilePhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_capture: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    partial_flush: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_grad: Option<NativeTrainingCompilePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    evaluation: Option<NativeTrainingCompilePhase>,
    residual_wall_time: BenchmarkDuration,
}

impl NativeTrainingCompilePhaseReport {
    fn from_observation(
        observation: &CompiledTrainingCompileObservation,
        compile_wall_time: Duration,
    ) -> Result<Self> {
        let objective_forward =
            NativeTrainingCompilePhase::from_observation(observation.objective_forward())?;
        let autograd = NativeTrainingCompilePhase::from_observation(observation.autograd())?;
        let optimizer_lowering =
            NativeTrainingCompilePhase::from_observation(observation.optimizer_lowering())?;
        let main_capture =
            NativeTrainingCompilePhase::from_observation(observation.main_capture())?;
        let accumulation_capture = observation
            .accumulation_capture()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let partial_flush = observation
            .partial_flush()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let zero_grad = observation
            .zero_grad()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let evaluation = observation
            .evaluation()
            .map(NativeTrainingCompilePhase::from_observation)
            .transpose()?;
        let measured = observation
            .measured_wall_time()
            .ok_or_else(|| invalid("compiled training phase duration overflows"))?;
        let residual = compile_wall_time
            .checked_sub(measured)
            .ok_or_else(|| invalid("compiled training phases exceed compile wall time"))?;
        Ok(Self {
            compile_count: observation.compile_count(),
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
            partial_flush,
            zero_grad,
            evaluation,
            residual_wall_time: BenchmarkDuration::from_duration(residual),
        })
    }

    fn validate(
        &self,
        compile_wall_time: BenchmarkDuration,
        main: &NativeTrainingProgramReport,
        accumulation: Option<&NativeTrainingProgramReport>,
        partial_flush: Option<&NativeTrainingProgramReport>,
        zero_grad: Option<&NativeTrainingProgramReport>,
        evaluation: Option<&NativeTrainingProgramReport>,
    ) -> Result<()> {
        if self.compile_count != 1 {
            return Err(invalid("compiled training compile count differs"));
        }
        self.objective_forward.validate_graph()?;
        self.autograd.validate_graph()?;
        self.optimizer_lowering.validate_graph()?;
        let objective_nodes = self.objective_forward.graph_node_count.unwrap_or(0);
        let autograd_nodes = self.autograd.graph_node_count.unwrap_or(0);
        let optimizer_nodes = self.optimizer_lowering.graph_node_count.unwrap_or(0);
        if objective_nodes == 0
            || objective_nodes > autograd_nodes
            || autograd_nodes > optimizer_nodes
        {
            return Err(invalid("compiled graph phase inventory order differs"));
        }
        self.main_capture
            .validate_schedule(main.logical_schedule_item_count)?;
        for (phase, program) in [
            (self.accumulation_capture, accumulation),
            (self.partial_flush, partial_flush),
            (self.zero_grad, zero_grad),
            (self.evaluation, evaluation),
        ] {
            match (phase, program) {
                (Some(phase), Some(program)) => {
                    phase.validate_schedule(program.logical_schedule_item_count)?
                }
                (None, None) => {}
                _ => return Err(invalid("compiled phase program inventory differs")),
            }
        }
        let total = [
            Some(self.objective_forward),
            Some(self.autograd),
            Some(self.optimizer_lowering),
            Some(self.main_capture),
            self.accumulation_capture,
            self.partial_flush,
            self.zero_grad,
            self.evaluation,
        ]
        .into_iter()
        .flatten()
        .try_fold(Duration::ZERO, |total, phase| {
            total
                .checked_add(phase.duration()?)
                .ok_or_else(|| invalid("compiled training phase duration overflows"))
        })?
        .checked_add(
            self.residual_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid compiled training residual duration"))?,
        )
        .ok_or_else(|| invalid("compiled training phase duration overflows"))?;
        if total
            != compile_wall_time
                .to_duration()
                .map_err(|_| invalid("invalid native training compile duration"))?
        {
            return Err(invalid("compiled training phase partition differs"));
        }
        Ok(())
    }

    pub const fn compile_count(&self) -> u64 {
        self.compile_count
    }

    pub const fn objective_forward(&self) -> NativeTrainingCompilePhase {
        self.objective_forward
    }

    pub const fn autograd(&self) -> NativeTrainingCompilePhase {
        self.autograd
    }

    pub const fn optimizer_lowering(&self) -> NativeTrainingCompilePhase {
        self.optimizer_lowering
    }

    pub const fn main_capture(&self) -> NativeTrainingCompilePhase {
        self.main_capture
    }

    pub const fn accumulation_capture(&self) -> Option<NativeTrainingCompilePhase> {
        self.accumulation_capture
    }

    pub const fn partial_flush(&self) -> Option<NativeTrainingCompilePhase> {
        self.partial_flush
    }

    pub const fn zero_grad(&self) -> Option<NativeTrainingCompilePhase> {
        self.zero_grad
    }

    pub const fn evaluation(&self) -> Option<NativeTrainingCompilePhase> {
        self.evaluation
    }

    pub const fn residual_wall_time(&self) -> BenchmarkDuration {
        self.residual_wall_time
    }
}

/// Versioned strict-native CPU compiled-training observation.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingReport {
    format_version: u32,
    compile_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compile_phases: Option<NativeTrainingCompilePhaseReport>,
    prepare_wall_time: BenchmarkDuration,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_runtime_overhead_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_parallel_module_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_parallel_render_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_render_capsule_hit_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_render_capsule_miss_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_local_render_job_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_overlap_wall_time: Option<BenchmarkDuration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_max_parallel_compiler_process_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_process_timings: Option<Vec<NativeTrainingCompilerProcessTiming>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_compiler_critical_tail: Option<NativeTrainingCompilerCriticalTail>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_module_overlaps: Option<Vec<NativeTrainingModuleOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_program_pair_overlaps: Option<Vec<NativeTrainingProgramPairOverlap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    prepare_translation_units: Option<Vec<NativeTrainingTranslationUnit>>,
    initial_replay_step: u64,
    successful_replay_count: u64,
    main: NativeTrainingProgramReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation: Option<NativeTrainingProgramReport>,
    partial_flush: Option<NativeTrainingProgramReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    recurrent_logical_state_count: u64,
    recurrent_logical_state_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_replay_traffic: Option<NativeCpuReplayTraffic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    accumulation_replay_executed_native_item_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executor_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_native_dispatcher_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_executor_host_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    main_replay_recurrent_overhead_wall_time: Option<NativeTrainingReplayTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    step_phases: Option<NativeTrainingStepPhaseReport>,
    first_replay_wall_time: BenchmarkDuration,
    steady_replay_total_wall_time: BenchmarkDuration,
    steady_replay_wall_time: BenchmarkLatencySummary,
    steady_microbatches_per_second: Option<f64>,
    schedule_cache_keys: Vec<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accumulation_schedule_cache_keys: Vec<u64>,
    checkpoint: Option<CheckpointReport>,
    fallback_count: u64,
    kernel_launch_count: Option<u64>,
    host_to_device: Option<BenchmarkTransfer>,
    device_to_host: Option<BenchmarkTransfer>,
    measured_peak_host_memory_bytes: Option<u64>,
}

impl NativeTrainingReport {
    pub const fn compile_wall_time(&self) -> BenchmarkDuration {
        self.compile_wall_time
    }

    pub fn compile_phases(&self) -> Option<&NativeTrainingCompilePhaseReport> {
        self.compile_phases.as_ref()
    }

    pub const fn prepare_wall_time(&self) -> BenchmarkDuration {
        self.prepare_wall_time
    }

    /// Whole-prepare time outside the attached native program preparations.
    pub const fn prepare_runtime_overhead_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_runtime_overhead_wall_time
    }

    /// Exact overlap among complete native module compiler/loader jobs.
    pub const fn prepare_parallel_module_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_parallel_module_overlap_wall_time
    }

    /// Exact overlap among immutable per-program native render jobs.
    pub const fn prepare_parallel_render_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_parallel_render_overlap_wall_time
    }

    /// Observed maximum concurrency of immutable per-program native renders.
    pub const fn prepare_max_parallel_render_job_count(&self) -> Option<u64> {
        self.prepare_max_parallel_render_job_count
    }

    pub const fn prepare_render_capsule_hit_count(&self) -> Option<u64> {
        self.prepare_render_capsule_hit_count
    }

    pub const fn prepare_render_capsule_miss_count(&self) -> Option<u64> {
        self.prepare_render_capsule_miss_count
    }

    pub const fn prepare_local_render_job_count(&self) -> Option<u64> {
        self.prepare_local_render_job_count
    }

    pub const fn prepare_compiler_process_overlap_wall_time(&self) -> Option<BenchmarkDuration> {
        self.prepare_compiler_process_overlap_wall_time
    }

    pub const fn prepare_compiler_process_count(&self) -> Option<u64> {
        self.prepare_compiler_process_count
    }

    pub const fn prepare_max_parallel_compiler_process_count(&self) -> Option<u64> {
        self.prepare_max_parallel_compiler_process_count
    }

    pub const fn main(&self) -> &NativeTrainingProgramReport {
        &self.main
    }

    pub const fn accumulation(&self) -> Option<&NativeTrainingProgramReport> {
        self.accumulation.as_ref()
    }

    pub const fn partial_flush(&self) -> Option<&NativeTrainingProgramReport> {
        self.partial_flush.as_ref()
    }

    pub const fn evaluation(&self) -> Option<&NativeTrainingProgramReport> {
        self.evaluation.as_ref()
    }

    pub const fn zero_grad(&self) -> Option<&NativeTrainingProgramReport> {
        self.zero_grad.as_ref()
    }

    pub const fn successful_replay_count(&self) -> u64 {
        self.successful_replay_count
    }

    pub const fn steady_replay_wall_time(&self) -> &BenchmarkLatencySummary {
        &self.steady_replay_wall_time
    }

    pub const fn steady_replay_total_wall_time(&self) -> BenchmarkDuration {
        self.steady_replay_total_wall_time
    }

    pub const fn first_replay_wall_time(&self) -> BenchmarkDuration {
        self.first_replay_wall_time
    }

    pub fn schedule_cache_keys(&self) -> &[u64] {
        &self.schedule_cache_keys
    }

    pub const fn steady_microbatches_per_second(&self) -> Option<f64> {
        self.steady_microbatches_per_second
    }

    pub const fn recurrent_state_count(&self) -> u64 {
        self.recurrent_logical_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> u64 {
        self.recurrent_logical_state_bytes
    }

    pub const fn main_replay_traffic(&self) -> Option<&NativeCpuReplayTraffic> {
        self.main_replay_traffic.as_ref()
    }

    /// Stable number of prepared CPU JIT items actually invoked by each
    /// successfully published optimizer-commit replay.
    pub const fn main_replay_executed_native_item_count(&self) -> Option<u64> {
        self.main_replay_executed_native_item_count
    }

    pub const fn accumulation_replay_traffic(&self) -> Option<&NativeCpuReplayTraffic> {
        self.accumulation_replay_traffic.as_ref()
    }

    pub const fn accumulation_replay_executed_native_item_count(&self) -> Option<u64> {
        self.accumulation_replay_executed_native_item_count
    }

    pub fn accumulation_schedule_cache_keys(&self) -> &[u64] {
        &self.accumulation_schedule_cache_keys
    }

    /// Wall time inside the sealed native executor for successful training
    /// steps across their authenticated phase-specific programs.
    pub const fn main_replay_executor_wall_time(&self) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_executor_wall_time.as_ref()
    }

    /// Time inside authenticated native schedule-module dispatcher calls.
    pub const fn main_replay_native_dispatcher_wall_time(
        &self,
    ) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_native_dispatcher_wall_time.as_ref()
    }

    /// Checked executor remainder outside native dispatcher calls, including
    /// workspace work and any conservative per-entry execution.
    pub const fn main_replay_executor_host_wall_time(&self) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_executor_host_wall_time.as_ref()
    }

    /// Checked end-to-end remainder outside the sealed native executor across
    /// the authenticated phase-specific programs. This covers recurrent
    /// staging, validation, and atomic commit/publication.
    pub const fn main_replay_recurrent_overhead_wall_time(
        &self,
    ) -> Option<&NativeTrainingReplayTiming> {
        self.main_replay_recurrent_overhead_wall_time.as_ref()
    }

    /// First replay and warm replay timings classified solely by whether the
    /// successful training step committed an optimizer update.
    pub const fn step_phases(&self) -> Option<&NativeTrainingStepPhaseReport> {
        self.step_phases.as_ref()
    }

    pub const fn checkpoint_byte_count(&self) -> Option<u64> {
        match &self.checkpoint {
            Some(checkpoint) => Some(checkpoint.byte_count),
            None => None,
        }
    }

    pub const fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| invalid(format!("JSON encoding failed: {error}")))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let report: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("JSON decoding failed: {error}")))?;
        report.validate()?;
        Ok(report)
    }

    fn validate(&self) -> Result<()> {
        if !matches!(
            self.format_version,
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
        if self.format_version == 1 && self.zero_grad.is_some() {
            return Err(invalid(
                "legacy native training report has zero-grad evidence",
            ));
        }
        if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V2
            && self.partial_flush.is_some() != self.zero_grad.is_some()
        {
            return Err(invalid(
                "native training auxiliary program inventory differs",
            ));
        }
        for duration in [
            self.compile_wall_time,
            self.prepare_wall_time,
            self.first_replay_wall_time,
            self.steady_replay_total_wall_time,
            self.steady_replay_wall_time.min,
            self.steady_replay_wall_time.nearest_rank_p50,
            self.steady_replay_wall_time.nearest_rank_p95,
            self.steady_replay_wall_time.max,
        ] {
            duration
                .to_duration()
                .map_err(|_| invalid("invalid native training duration"))?;
        }
        match (self.format_version, &self.compile_phases) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, None) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_VERSION, Some(phases)) => phases.validate(
                self.compile_wall_time,
                &self.main,
                self.accumulation.as_ref(),
                self.partial_flush.as_ref(),
                self.zero_grad.as_ref(),
                self.evaluation.as_ref(),
            )?,
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V22, Some(_)) => {
                return Err(invalid("legacy native training report has compile phases"));
            }
            _ => return Err(invalid("native training compile phases are absent")),
        }
        self.main.validate(self.format_version)?;
        if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V14
            && (self.main.shared_prefix_entry_count() != 0
                || self.main.shared_prefix_source_program_index.is_some()
                || self.main.shared_prefix_source_native_identity.is_some())
        {
            return Err(invalid(
                "native main program cannot reference an earlier prefix module",
            ));
        }
        match (
            self.format_version,
            &self.accumulation,
            &self.accumulation_replay_traffic,
            self.accumulation_replay_executed_native_item_count,
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
                if traffic.borrowed_recurrent_input_bytes()
                    == self.recurrent_logical_state_bytes
                    && traffic.borrowed_recurrent_output_bytes()
                        == self.recurrent_logical_state_bytes
                    && traffic.retained_recurrent_state_count() == 0
                    && traffic.retained_recurrent_state_bytes() == 0
                    && traffic.replaced_recurrent_state_count() == 0
                    && traffic.replaced_recurrent_state_bytes() == 0
                    && executed <= program.rendered_entry_count
                    && program.capture_identity != self.main.capture_identity =>
            {
                program.validate(self.format_version)?;
                if program.vectorized != self.main.vectorized {
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
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic
                    .borrowed_recurrent_output_bytes()
                    .checked_add(traffic.retained_recurrent_state_bytes())
                    == Some(self.recurrent_logical_state_bytes)
                && traffic.retained_recurrent_state_count()
                    <= self.recurrent_logical_state_count
                && ((traffic.retained_recurrent_state_count() == 0
                    && traffic.retained_recurrent_state_bytes() == 0
                    && traffic.replaced_recurrent_state_count() == 0
                    && traffic.replaced_recurrent_state_bytes() == 0)
                    || (traffic
                        .retained_recurrent_state_count()
                        .checked_add(traffic.replaced_recurrent_state_count())
                        == Some(self.recurrent_logical_state_count)
                        && traffic.replaced_recurrent_state_bytes()
                            == traffic.borrowed_recurrent_output_bytes()))
                && executed <= program.rendered_entry_count
                && program.capture_identity != self.main.capture_identity =>
            {
                program.validate(self.format_version)?;
                if program.vectorized != self.main.vectorized {
                    return Err(invalid("native program vectorization policy differs"));
                }
            }
            _ => return Err(invalid("native accumulation replay inventory differs")),
        }
        match (self.format_version, &self.main_replay_traffic) {
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
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == self.recurrent_logical_state_bytes
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
            ) if traffic.borrowed_recurrent_input_bytes() == self.recurrent_logical_state_bytes
                && traffic.borrowed_recurrent_output_bytes()
                    == self.recurrent_logical_state_bytes
                && traffic.retained_recurrent_state_count() == 0
                && traffic.retained_recurrent_state_bytes() == 0
                && traffic.replaced_recurrent_state_count() == 0
                && traffic.replaced_recurrent_state_bytes() == 0 => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2, Some(_)) => {
                return Err(invalid("legacy native training report has replay traffic"));
            }
            _ => return Err(invalid("native training replay traffic differs")),
        }
        for traffic in self
            .main_replay_traffic
            .iter()
            .chain(&self.accumulation_replay_traffic)
        {
            match self.format_version {
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
        match (
            self.format_version,
            self.main_replay_executed_native_item_count,
        ) {
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, None) => {}
            (NATIVE_TRAINING_REPORT_FORMAT_V4, Some(executed))
                if executed <= self.main.native_item_count => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V5
                | NATIVE_TRAINING_REPORT_FORMAT_V6
                | NATIVE_TRAINING_REPORT_FORMAT_V7
                | NATIVE_TRAINING_REPORT_FORMAT_V8
                | NATIVE_TRAINING_REPORT_FORMAT_V9
                | NATIVE_TRAINING_REPORT_FORMAT_V10
                | NATIVE_TRAINING_REPORT_FORMAT_V11,
                Some(executed),
            ) if executed <= self.main.rendered_entry_count => {}
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
            ) if executed <= self.main.rendered_entry_count => {}
            (1 | NATIVE_TRAINING_REPORT_FORMAT_V2 | NATIVE_TRAINING_REPORT_FORMAT_V3, Some(_)) => {
                return Err(invalid("legacy native training report has execution count"));
            }
            _ => return Err(invalid("native training execution count differs")),
        }
        let mut prior_programs = vec![&self.main];
        for program in self
            .accumulation
            .iter()
            .chain(&self.partial_flush)
            .chain(&self.zero_grad)
            .chain(&self.evaluation)
        {
            program.validate(self.format_version)?;
            if program.vectorized != self.main.vectorized {
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
        match (self.format_version, &self.prepare_module_overlaps) {
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, None) => {}
            (
                NATIVE_TRAINING_REPORT_FORMAT_V20
                | NATIVE_TRAINING_REPORT_FORMAT_V21
                | NATIVE_TRAINING_REPORT_FORMAT_V22
                | NATIVE_TRAINING_REPORT_FORMAT_VERSION,
                Some(overlaps),
            ) => {
                validate_module_overlaps(overlaps, &prior_programs)?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V19, Some(_)) => {
                return Err(invalid("legacy native report has module overlap evidence"));
            }
            _ => return Err(invalid("native module overlap evidence is absent")),
        }
        match (
            self.format_version,
            &self.prepare_program_pair_overlaps,
            &self.prepare_translation_units,
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
                    &prior_programs,
                    self.prepare_module_overlaps
                        .as_deref()
                        .ok_or_else(|| invalid("native main overlap evidence is absent"))?,
                )?;
                validate_translation_units(
                    translation_units,
                    &prior_programs,
                    self.prepare_compiler_process_timings
                        .as_deref()
                        .ok_or_else(|| invalid("native compiler process evidence is absent"))?,
                )?;
            }
            (1..=NATIVE_TRAINING_REPORT_FORMAT_V20, _, _) => {
                return Err(invalid("legacy native report has v21 module evidence"));
            }
            _ => return Err(invalid("native v21 module evidence is absent")),
        }
        let parallel_evidence = match self.format_version {
            1..=NATIVE_TRAINING_REPORT_FORMAT_V7 => {
                if self.prepare_compiler_process_overlap_wall_time.is_some()
                    || self.prepare_compiler_process_count.is_some()
                    || self.prepare_max_parallel_compiler_process_count.is_some()
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
                let compiler_overlap = self
                    .prepare_compiler_process_overlap_wall_time
                    .ok_or_else(|| invalid("native compiler overlap timing is absent"))?
                    .as_nanos()
                    .map_err(|_| invalid("invalid native compiler overlap duration"))?;
                let compiler_count = self
                    .prepare_compiler_process_count
                    .ok_or_else(|| invalid("native compiler process count is absent"))?;
                let max_parallel = self
                    .prepare_max_parallel_compiler_process_count
                    .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
                let aggregate = NativeCompilerAggregate::from_programs(
                    std::iter::once(&self.main)
                        .chain(self.accumulation.iter())
                        .chain(self.partial_flush.iter())
                        .chain(&self.zero_grad)
                        .chain(&self.evaluation),
                    self.format_version,
                )?;
                let internal_compiler_overlap = aggregate.internal_overlap()?;
                let overlap_is_valid = if self.format_version >= NATIVE_TRAINING_REPORT_FORMAT_V15 {
                    let active_union = aggregate
                        .cumulative_wall_sum
                        .checked_sub(compiler_overlap)
                        .ok_or_else(|| {
                            invalid("native compiler overlap exceeds cumulative work")
                        })?;
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
                Some((
                    compiler_overlap,
                    aggregate.module_job_count,
                    internal_compiler_overlap,
                ))
            }
            _ => unreachable!("format version was validated"),
        };
        match (
            self.format_version,
            &self.prepare_compiler_process_timings,
            &self.prepare_compiler_critical_tail,
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
                let process_count = self
                    .prepare_compiler_process_count
                    .ok_or_else(|| invalid("native compiler process count is absent"))?;
                let max_parallel = self
                    .prepare_max_parallel_compiler_process_count
                    .ok_or_else(|| invalid("native compiler concurrency is absent"))?;
                let overlap = self
                    .prepare_compiler_process_overlap_wall_time
                    .ok_or_else(|| invalid("native compiler overlap timing is absent"))?;
                let derived_tail = validate_compiler_process_evidence(
                    timings,
                    claimed_tail.as_ref(),
                    &prior_programs,
                    CompilerProcessValidationContext {
                        format_version: self.format_version,
                        process_count,
                        max_parallel,
                        claimed_overlap: overlap,
                        prepare_wall_time: self.prepare_wall_time,
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
        match (
            self.format_version,
            self.prepare_render_capsule_hit_count,
            self.prepare_render_capsule_miss_count,
            self.prepare_local_render_job_count,
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
        let render_overlap = match (
            self.format_version,
            self.prepare_parallel_render_overlap_wall_time,
            self.prepare_max_parallel_render_job_count,
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
                    std::iter::once(&self.main)
                        .chain(self.accumulation.iter())
                        .chain(self.partial_flush.iter())
                        .chain(&self.zero_grad)
                        .chain(&self.evaluation),
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
                let programs = std::iter::once(&self.main)
                    .chain(self.accumulation.iter())
                    .chain(self.partial_flush.iter())
                    .chain(&self.zero_grad)
                    .chain(&self.evaluation)
                    .collect::<Vec<_>>();
                let program_count = count(programs.len(), "native render program")?;
                let aggregate = NativeRenderAggregate::from_programs(programs)?;
                let maximum_overlap = aggregate
                    .wall_sum
                    .checked_sub(aggregate.wall_max)
                    .ok_or_else(|| invalid("native render overlap underflows"))?;
                let hit_count = self
                    .prepare_render_capsule_hit_count
                    .ok_or_else(|| invalid("native render capsule hit count is absent"))?;
                let miss_count = self
                    .prepare_render_capsule_miss_count
                    .ok_or_else(|| invalid("native render capsule miss count is absent"))?;
                let local_render_count = self
                    .prepare_local_render_job_count
                    .ok_or_else(|| invalid("native local render job count is absent"))?;
                let zero_timing = max_parallel == 0 && overlap == 0 && aggregate.wall_sum == 0;
                let timing_partition_matches = if local_render_count == 0 || aggregate.wall_sum == 0
                {
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
        match (
            self.format_version,
            self.prepare_runtime_overhead_wall_time,
            self.prepare_parallel_module_overlap_wall_time,
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
                let overlap = match (self.format_version, overlap) {
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
                if let Some((compiler_overlap, module_job_count, internal_overlap)) =
                    parallel_evidence
                {
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
                let prepare_total = self
                    .prepare_wall_time
                    .as_nanos()
                    .map_err(|_| invalid("invalid native prepare duration"))?;
                let mut programs = std::iter::once(&self.main)
                    .chain(self.accumulation.iter())
                    .chain(self.partial_flush.iter())
                    .chain(&self.zero_grad)
                    .chain(&self.evaluation);
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
        if self.successful_replay_count < 2
            || self.successful_replay_count > MAX_REPLAY_SAMPLES as u64
            || self.steady_replay_wall_time.sample_count != self.successful_replay_count - 1
            || self.steady_replay_wall_time.min > self.steady_replay_wall_time.nearest_rank_p50
            || self.steady_replay_wall_time.nearest_rank_p50
                > self.steady_replay_wall_time.nearest_rank_p95
            || self.steady_replay_wall_time.nearest_rank_p95 > self.steady_replay_wall_time.max
            || self.steady_replay_total_wall_time < self.steady_replay_wall_time.max
        {
            return Err(invalid("invalid native training replay summary"));
        }
        validate_total_duration(
            &self.steady_replay_wall_time,
            self.steady_replay_total_wall_time,
        )?;
        match (
            self.format_version,
            &self.main_replay_executor_wall_time,
            &self.main_replay_recurrent_overhead_wall_time,
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
                let steady_count = self.successful_replay_count - 1;
                executor.validate(steady_count)?;
                overhead.validate(steady_count)?;
                validate_phase_partition(
                    self.first_replay_wall_time,
                    executor.first,
                    overhead.first,
                    "first replay",
                )?;
                validate_phase_partition(
                    self.steady_replay_total_wall_time,
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
        match (
            self.format_version,
            &self.main_replay_native_dispatcher_wall_time,
            &self.main_replay_executor_host_wall_time,
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
                let steady_count = self.successful_replay_count - 1;
                native_dispatcher.validate(steady_count)?;
                executor_host.validate(steady_count)?;
                let executor = self
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
        match (self.format_version, &self.step_phases) {
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
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                ReplayTimingPartition {
                    executor: self
                        .main_replay_executor_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                    native_dispatcher: None,
                    executor_host: None,
                    overhead: self
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
                self.successful_replay_count,
                self.first_replay_wall_time,
                self.steady_replay_total_wall_time,
                ReplayTimingPartition {
                    executor: self
                        .main_replay_executor_wall_time
                        .as_ref()
                        .ok_or_else(|| invalid("classified replay executor timing is absent"))?,
                    native_dispatcher: self.main_replay_native_dispatcher_wall_time.as_ref(),
                    executor_host: self.main_replay_executor_host_wall_time.as_ref(),
                    overhead: self
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
            ) if self.accumulation.is_none() => {}
            _ => return Err(invalid("native training step phases differ")),
        }
        let expected_rate = rate_from_total(
            self.steady_replay_wall_time.sample_count,
            self.steady_replay_total_wall_time,
        )?;
        if self.steady_microbatches_per_second.map(f64::to_bits) != expected_rate.map(f64::to_bits)
            || self.schedule_cache_keys.len()
                != usize::try_from(self.main.native_item_count)
                    .map_err(|_| invalid("native item count overflows usize"))?
            || match &self.accumulation {
                Some(program) => {
                    self.accumulation_schedule_cache_keys.len()
                        != usize::try_from(program.native_item_count).map_err(|_| {
                            invalid("native accumulation item count overflows usize")
                        })?
                }
                None => !self.accumulation_schedule_cache_keys.is_empty(),
            }
        {
            return Err(invalid("invalid native training replay inventory"));
        }
        if self.fallback_count != 0
            || self.kernel_launch_count.is_some()
            || self.host_to_device.is_some()
            || self.device_to_host.is_some()
            || self.measured_peak_host_memory_bytes.is_some()
        {
            return Err(invalid("native CPU availability fields are inconsistent"));
        }
        if let Some(checkpoint) = &self.checkpoint {
            checkpoint
                .wall_time
                .to_duration()
                .map_err(|_| invalid("invalid checkpoint duration"))?;
            let expected_step = self
                .initial_replay_step
                .checked_add(self.successful_replay_count)
                .ok_or_else(|| invalid("replay step overflows"))?;
            if checkpoint.byte_count == 0
                || checkpoint.capture_identity != self.main.capture_identity
                || checkpoint.replay_step != expected_step
            {
                return Err(invalid("checkpoint does not match recorded replays"));
            }
        }
        Ok(())
    }
}

/// Bounded collector for successful strict-native CPU training-step replays.
pub struct NativeTrainingScoreboard {
    compile_wall_time: Duration,
    compile_phases: NativeTrainingCompilePhaseReport,
    prepare_wall_time: Duration,
    prepare_runtime_overhead_wall_time: Duration,
    prepare_parallel_module_overlap_wall_time: Duration,
    prepare_parallel_render_overlap_wall_time: Duration,
    prepare_max_parallel_render_job_count: u64,
    prepare_render_capsule_hit_count: u64,
    prepare_render_capsule_miss_count: u64,
    prepare_local_render_job_count: u64,
    prepare_compiler_process_overlap_wall_time: Duration,
    prepare_compiler_process_count: u64,
    prepare_max_parallel_compiler_process_count: u64,
    prepare_compiler_process_timings: Vec<NativeTrainingCompilerProcessTiming>,
    prepare_compiler_critical_tail: Option<NativeTrainingCompilerCriticalTail>,
    prepare_module_overlaps: Vec<NativeTrainingModuleOverlap>,
    prepare_program_pair_overlaps: Vec<NativeTrainingProgramPairOverlap>,
    prepare_translation_units: Vec<NativeTrainingTranslationUnit>,
    inspection: CompiledAdamWInspection,
    main: NativeTrainingProgramReport,
    accumulation: Option<NativeTrainingProgramReport>,
    partial_flush: Option<NativeTrainingProgramReport>,
    zero_grad: Option<NativeTrainingProgramReport>,
    evaluation: Option<NativeTrainingProgramReport>,
    recording_mode: ReplayRecordingMode,
    replay_timings: Vec<ReplayTiming>,
    replay_step_phases: Vec<NativeTrainingStepPhase>,
    schedule_cache_keys: Option<Vec<u64>>,
    main_replay_traffic: Option<NativeCpuReplayTraffic>,
    main_replay_executed_native_item_count: Option<u64>,
    accumulation_schedule_cache_keys: Option<Vec<u64>>,
    accumulation_replay_traffic: Option<NativeCpuReplayTraffic>,
    accumulation_replay_executed_native_item_count: Option<u64>,
    checkpoint: Option<CheckpointReport>,
}

impl NativeTrainingScoreboard {
    /// Starts a bounded observation from one complete strict-native
    /// preparation. `compile_wall_time` is caller-observed around the fresh
    /// plan construction and must contain its measured compile phases;
    /// `prepare_wall_time` similarly encloses every attached program's
    /// measured preparation time.
    pub fn new(
        inspection: CompiledAdamWInspection,
        preparation: &NativeCpuCompiledAdamWPreparationReport,
        compile_wall_time: Duration,
        prepare_wall_time: Duration,
    ) -> Result<Self> {
        let mut prior_native_identities = Vec::new();
        let main = NativeTrainingProgramReport::new(
            &inspection.main,
            preparation.main(),
            &prior_native_identities,
        )?;
        prior_native_identities.push(main.native_identity());
        let accumulation = matching_program(
            "accumulation",
            inspection.accumulation.as_ref(),
            preparation.accumulation(),
            &prior_native_identities,
        )?;
        if let Some(program) = &accumulation {
            prior_native_identities.push(program.native_identity());
        }
        let partial_flush = matching_program(
            "partial flush",
            inspection.partial_flush.as_ref(),
            preparation.partial_flush(),
            &prior_native_identities,
        )?;
        if let Some(program) = &partial_flush {
            prior_native_identities.push(program.native_identity());
        }
        let zero_grad = matching_program(
            "zero grad",
            inspection.zero_grad.as_ref(),
            preparation.zero_grad(),
            &prior_native_identities,
        )?;
        if let Some(program) = &zero_grad {
            prior_native_identities.push(program.native_identity());
        }
        let evaluation = matching_program(
            "evaluation",
            inspection.evaluation.as_ref(),
            preparation.evaluation(),
            &prior_native_identities,
        )?;
        let compile_phases = NativeTrainingCompilePhaseReport::from_observation(
            inspection
                .compile_phases()
                .ok_or_else(|| invalid("compiled training phase observation is absent"))?,
            compile_wall_time,
        )?;
        let programs = std::iter::once(&main)
            .chain(accumulation.iter())
            .chain(partial_flush.iter())
            .chain(zero_grad.iter())
            .chain(evaluation.iter())
            .collect::<Vec<_>>();
        let (prepare_compiler_process_timings, prepare_compiler_critical_tail) =
            compiler_process_evidence(preparation, &programs, prepare_wall_time)?;
        let prepare_module_overlaps = preparation
            .module_overlaps()
            .iter()
            .map(|overlap| NativeTrainingModuleOverlap::from_preparation(overlap, &programs))
            .collect::<Result<Vec<_>>>()?;
        validate_module_overlaps(&prepare_module_overlaps, &programs)?;
        let prepare_program_pair_overlaps = preparation
            .program_pair_overlaps()
            .iter()
            .map(|overlap| NativeTrainingProgramPairOverlap::from_preparation(overlap, &programs))
            .collect::<Result<Vec<_>>>()?;
        let prepare_translation_units = preparation
            .translation_units()
            .iter()
            .map(|unit| NativeTrainingTranslationUnit::from_preparation(unit, &programs))
            .collect::<Result<Vec<_>>>()?;
        validate_program_pair_overlaps(
            &prepare_program_pair_overlaps,
            &programs,
            &prepare_module_overlaps,
        )?;
        validate_translation_units(
            &prepare_translation_units,
            &programs,
            &prepare_compiler_process_timings,
        )?;
        if inspection.recurrent_state_count != preparation.recurrent_state_count()
            || inspection.recurrent_state_bytes != preparation.recurrent_state_bytes()
        {
            return Err(invalid("plan and prepared recurrent state differ"));
        }
        let program_prepare_wall_time = std::iter::once(preparation.main())
            .chain(preparation.accumulation())
            .chain(preparation.partial_flush())
            .chain(preparation.zero_grad())
            .chain(preparation.evaluation())
            .try_fold(Duration::ZERO, |total, program| {
                total
                    .checked_add(program.wall_time())
                    .ok_or_else(|| invalid("native program preparation duration overflows"))
            })?;
        let prepare_parallel_module_overlap_wall_time =
            preparation.parallel_module_overlap_wall_time();
        let prepare_parallel_render_overlap_wall_time =
            preparation.parallel_render_overlap_wall_time();
        let prepare_max_parallel_render_job_count = count(
            preparation.max_parallel_render_job_count(),
            "parallel native render job",
        )?;
        let prepare_render_capsule_hit_count = count(
            preparation.render_capsule_hit_count(),
            "native render capsule hit",
        )?;
        let prepare_render_capsule_miss_count = count(
            preparation.render_capsule_miss_count(),
            "native render capsule miss",
        )?;
        let prepare_local_render_job_count = count(
            preparation.local_render_job_count(),
            "native local render job",
        )?;
        let prepare_compiler_process_count = count(
            preparation.compiler_process_count(),
            "native compiler process",
        )?;
        let prepare_max_parallel_compiler_process_count = count(
            preparation.max_parallel_compiler_process_count(),
            "parallel native compiler process",
        )?;
        let effective_program_prepare_wall_time = program_prepare_wall_time
            .checked_sub(prepare_parallel_module_overlap_wall_time)
            .and_then(|time| time.checked_sub(prepare_parallel_render_overlap_wall_time))
            .ok_or_else(|| invalid("native parallel work overlap exceeds program time"))?;
        let prepare_runtime_overhead_wall_time = prepare_wall_time
            .checked_sub(effective_program_prepare_wall_time)
            .ok_or_else(|| invalid("native program preparation exceeds whole prepare time"))?;
        Ok(Self {
            compile_wall_time,
            compile_phases,
            prepare_wall_time,
            prepare_runtime_overhead_wall_time,
            prepare_parallel_module_overlap_wall_time,
            prepare_parallel_render_overlap_wall_time,
            prepare_max_parallel_render_job_count,
            prepare_render_capsule_hit_count,
            prepare_render_capsule_miss_count,
            prepare_local_render_job_count,
            prepare_compiler_process_overlap_wall_time: preparation
                .compiler_process_overlap_wall_time(),
            prepare_compiler_process_count,
            prepare_max_parallel_compiler_process_count,
            prepare_compiler_process_timings,
            prepare_compiler_critical_tail,
            prepare_module_overlaps,
            prepare_program_pair_overlaps,
            prepare_translation_units,
            inspection,
            main,
            accumulation,
            partial_flush,
            zero_grad,
            evaluation,
            recording_mode: ReplayRecordingMode::Unset,
            replay_timings: Vec::new(),
            replay_step_phases: Vec::new(),
            schedule_cache_keys: None,
            main_replay_traffic: None,
            main_replay_executed_native_item_count: None,
            accumulation_schedule_cache_keys: None,
            accumulation_replay_traffic: None,
            accumulation_replay_executed_native_item_count: None,
            checkpoint: None,
        })
    }

    /// Records one unclassified report returned by a committed native training
    /// step. A scoreboard cannot mix this raw mode with [`Self::record_step`].
    pub fn record(&mut self, report: &NativeCpuRunReport) -> Result<()> {
        if self.recording_mode == ReplayRecordingMode::Phased {
            return Err(invalid("cannot mix raw and classified replay samples"));
        }
        if self.accumulation.is_some() {
            return Err(invalid(
                "phase-specialized replay requires classified step recording",
            ));
        }
        let phase = NativeTrainingStepPhase::OptimizerCommit;
        let (timing, executed) = self.validate_replay(report, phase)?;
        self.commit_replay(report, timing, executed, phase);
        self.recording_mode = ReplayRecordingMode::Raw;
        Ok(())
    }

    /// Records and classifies one successful compiled AdamW training step using
    /// only its authenticated public `did_update` result. Partial flushes are
    /// deliberately outside this training-step scoreboard.
    pub fn record_step(&mut self, step: &NativeCpuCompiledAdamWStepResult) -> Result<()> {
        if self.recording_mode == ReplayRecordingMode::Raw {
            return Err(invalid("cannot mix raw and classified replay samples"));
        }
        let phase = if step.did_update() {
            NativeTrainingStepPhase::OptimizerCommit
        } else {
            NativeTrainingStepPhase::AccumulationOnly
        };
        let (timing, executed) = self.validate_replay(step.report(), phase)?;
        self.commit_replay(step.report(), timing, executed, phase);
        self.replay_step_phases.push(phase);
        self.recording_mode = ReplayRecordingMode::Phased;
        Ok(())
    }

    fn validate_replay(
        &self,
        report: &NativeCpuRunReport,
        phase: NativeTrainingStepPhase,
    ) -> Result<(ReplayTiming, u64)> {
        if self.replay_timings.len() >= MAX_REPLAY_SAMPLES {
            return Err(invalid("native training replay sample limit exceeded"));
        }
        let expected_invocation = self.replay_timings.len() as u64 + 1;
        let expected = match phase {
            NativeTrainingStepPhase::AccumulationOnly => self
                .accumulation
                .as_ref()
                .ok_or_else(|| invalid("accumulation replay program is absent"))?,
            NativeTrainingStepPhase::OptimizerCommit => &self.main,
        };
        let (expected_cache_keys, expected_traffic, expected_executed) = match phase {
            NativeTrainingStepPhase::AccumulationOnly => (
                &self.accumulation_schedule_cache_keys,
                self.accumulation_replay_traffic,
                self.accumulation_replay_executed_native_item_count,
            ),
            NativeTrainingStepPhase::OptimizerCommit => (
                &self.schedule_cache_keys,
                self.main_replay_traffic,
                self.main_replay_executed_native_item_count,
            ),
        };
        if report.capture_identity() != expected.capture_identity
            || report.native_identity() != expected.native_identity
            || report.is_vectorized() != expected.vectorized
            || count(report.native_item_count(), "native item")? != expected.native_item_count
            || report.executed_native_item_count()
                > usize::try_from(expected.rendered_entry_count)
                    .map_err(|_| invalid("native rendered entry count overflows usize"))?
            || count(report.module_dispatch_count(), "native module dispatch")?
                != expected
                    .dispatch_segmentation
                    .as_ref()
                    .ok_or_else(|| invalid("native dispatch segmentation evidence is absent"))?
                    .segment_count
            || report.fallback_count() != 0
            || report.successful_invocation() != expected_invocation
            || report.schedule_cache_keys().len() != report.native_item_count()
        {
            return Err(invalid(
                "native replay report does not match the scoreboard",
            ));
        }
        if let Some(expected) = expected_cache_keys
            && expected != report.schedule_cache_keys()
        {
            return Err(invalid("native replay cache keys changed"));
        }
        let recurrent_state_bytes = count(
            self.inspection.recurrent_state_bytes,
            "recurrent state byte",
        )?;
        let has_recurrent_inventory = report.traffic().retained_recurrent_state_count() != 0
            || report.traffic().retained_recurrent_state_bytes() != 0
            || report.traffic().replaced_recurrent_state_count() != 0
            || report.traffic().replaced_recurrent_state_bytes() != 0;
        if report.traffic().borrowed_recurrent_input_bytes() != recurrent_state_bytes
            || report
                .traffic()
                .borrowed_recurrent_output_bytes()
                .checked_add(report.traffic().retained_recurrent_state_bytes())
                != Some(recurrent_state_bytes)
            || report.traffic().retained_recurrent_state_count()
                > count(self.inspection.recurrent_state_count, "recurrent state")?
            || (has_recurrent_inventory
                && (report
                    .traffic()
                    .retained_recurrent_state_count()
                    .checked_add(report.traffic().replaced_recurrent_state_count())
                    != Some(count(
                        self.inspection.recurrent_state_count,
                        "recurrent state",
                    )?)
                    || report.traffic().replaced_recurrent_state_bytes()
                        != report.traffic().borrowed_recurrent_output_bytes()))
        {
            return Err(invalid(
                "native replay traffic does not match recurrent state",
            ));
        }
        if let Some(expected) = expected_traffic
            && expected != *report.traffic()
        {
            return Err(invalid("native replay traffic changed"));
        }
        let executed = count(report.executed_native_item_count(), "executed native item")?;
        if let Some(expected) = expected_executed
            && expected != executed
        {
            return Err(invalid("native replay execution count changed"));
        }
        let total = report.wall_time();
        let executor = report.executor_wall_time();
        let native_dispatcher = report.native_dispatcher_wall_time();
        let executor_host = executor
            .checked_sub(native_dispatcher)
            .ok_or_else(|| invalid("native dispatcher time exceeds executor time"))?;
        let overhead = total
            .checked_sub(executor)
            .ok_or_else(|| invalid("native executor time exceeds replay time"))?;
        Ok((
            ReplayTiming {
                total,
                executor,
                native_dispatcher,
                executor_host,
                overhead,
            },
            executed,
        ))
    }

    fn commit_replay(
        &mut self,
        report: &NativeCpuRunReport,
        timing: ReplayTiming,
        executed: u64,
        phase: NativeTrainingStepPhase,
    ) {
        let (cache_keys, traffic, executed_count) = match phase {
            NativeTrainingStepPhase::AccumulationOnly => (
                &mut self.accumulation_schedule_cache_keys,
                &mut self.accumulation_replay_traffic,
                &mut self.accumulation_replay_executed_native_item_count,
            ),
            NativeTrainingStepPhase::OptimizerCommit => (
                &mut self.schedule_cache_keys,
                &mut self.main_replay_traffic,
                &mut self.main_replay_executed_native_item_count,
            ),
        };
        if cache_keys.is_none() {
            *cache_keys = Some(report.schedule_cache_keys().to_vec());
        }
        if traffic.is_none() {
            *traffic = Some(*report.traffic());
        }
        if executed_count.is_none() {
            *executed_count = Some(executed);
        }
        self.replay_timings.push(timing);
    }

    pub fn observe_checkpoint(
        &mut self,
        checkpoint: &CompiledAdamWCheckpoint,
        wall_time: Duration,
    ) -> Result<()> {
        let info = checkpoint.info();
        let replay_count = self.replay_timings.len() as u64;
        let expected_step = self
            .inspection
            .initial_replay_step
            .checked_add(replay_count)
            .ok_or_else(|| invalid("checkpoint replay step overflows"))?;
        let expected_accumulation_identity = self
            .accumulation
            .as_ref()
            .map(|program| program.capture_identity);
        if info.capture_identity() != self.main.capture_identity
            || info.replay_step() != expected_step
            || info.accumulation_capture_identity() != expected_accumulation_identity
        {
            return Err(invalid("checkpoint does not match recorded replays"));
        }
        self.checkpoint = Some(CheckpointReport {
            capture_identity: info.capture_identity(),
            replay_step: info.replay_step(),
            byte_count: count(checkpoint.as_bytes().len(), "checkpoint byte")?,
            wall_time: BenchmarkDuration::from_duration(wall_time),
        });
        Ok(())
    }

    pub fn report(&self) -> Result<NativeTrainingReport> {
        let Some((first, steady)) = self.replay_timings.split_first() else {
            return Err(invalid("native training scoreboard has no replay"));
        };
        if steady.is_empty() {
            return Err(invalid("native training scoreboard has no steady replay"));
        }
        let totals = steady.iter().map(|timing| timing.total).collect::<Vec<_>>();
        let executors = steady
            .iter()
            .map(|timing| timing.executor)
            .collect::<Vec<_>>();
        let overheads = steady
            .iter()
            .map(|timing| timing.overhead)
            .collect::<Vec<_>>();
        let native_dispatchers = steady
            .iter()
            .map(|timing| timing.native_dispatcher)
            .collect::<Vec<_>>();
        let executor_hosts = steady
            .iter()
            .map(|timing| timing.executor_host)
            .collect::<Vec<_>>();
        let (steady_replay_total_wall_time, steady_microbatches_per_second) = rate(&totals)?;
        let step_phases = match self.recording_mode {
            ReplayRecordingMode::Raw => None,
            ReplayRecordingMode::Phased => Some(NativeTrainingStepPhaseReport::from_timings(
                &self.replay_timings,
                &self.replay_step_phases,
            )?),
            ReplayRecordingMode::Unset => {
                return Err(invalid("native training scoreboard has no replay mode"));
            }
        };
        let report = NativeTrainingReport {
            format_version: NATIVE_TRAINING_REPORT_FORMAT_VERSION,
            compile_wall_time: BenchmarkDuration::from_duration(self.compile_wall_time),
            compile_phases: Some(self.compile_phases.clone()),
            prepare_wall_time: BenchmarkDuration::from_duration(self.prepare_wall_time),
            prepare_runtime_overhead_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_runtime_overhead_wall_time,
            )),
            prepare_parallel_module_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_parallel_module_overlap_wall_time,
            )),
            prepare_parallel_render_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_parallel_render_overlap_wall_time,
            )),
            prepare_max_parallel_render_job_count: Some(self.prepare_max_parallel_render_job_count),
            prepare_render_capsule_hit_count: Some(self.prepare_render_capsule_hit_count),
            prepare_render_capsule_miss_count: Some(self.prepare_render_capsule_miss_count),
            prepare_local_render_job_count: Some(self.prepare_local_render_job_count),
            prepare_compiler_process_overlap_wall_time: Some(BenchmarkDuration::from_duration(
                self.prepare_compiler_process_overlap_wall_time,
            )),
            prepare_compiler_process_count: Some(self.prepare_compiler_process_count),
            prepare_max_parallel_compiler_process_count: Some(
                self.prepare_max_parallel_compiler_process_count,
            ),
            prepare_compiler_process_timings: Some(self.prepare_compiler_process_timings.clone()),
            prepare_compiler_critical_tail: self.prepare_compiler_critical_tail.clone(),
            prepare_module_overlaps: Some(self.prepare_module_overlaps.clone()),
            prepare_program_pair_overlaps: Some(self.prepare_program_pair_overlaps.clone()),
            prepare_translation_units: Some(self.prepare_translation_units.clone()),
            initial_replay_step: self.inspection.initial_replay_step,
            successful_replay_count: self.replay_timings.len() as u64,
            main: self.main.clone(),
            accumulation: self.accumulation.clone(),
            partial_flush: self.partial_flush.clone(),
            zero_grad: self.zero_grad.clone(),
            evaluation: self.evaluation.clone(),
            recurrent_logical_state_count: count(
                self.inspection.recurrent_state_count,
                "recurrent state",
            )?,
            recurrent_logical_state_bytes: count(
                self.inspection.recurrent_state_bytes,
                "recurrent state byte",
            )?,
            main_replay_traffic: self.main_replay_traffic,
            main_replay_executed_native_item_count: self.main_replay_executed_native_item_count,
            accumulation_replay_traffic: self.accumulation_replay_traffic,
            accumulation_replay_executed_native_item_count: self
                .accumulation_replay_executed_native_item_count,
            main_replay_executor_wall_time: Some(NativeTrainingReplayTiming::from_durations(
                first.executor,
                &executors,
            )?),
            main_replay_native_dispatcher_wall_time: Some(
                NativeTrainingReplayTiming::from_durations(
                    first.native_dispatcher,
                    &native_dispatchers,
                )?,
            ),
            main_replay_executor_host_wall_time: Some(NativeTrainingReplayTiming::from_durations(
                first.executor_host,
                &executor_hosts,
            )?),
            main_replay_recurrent_overhead_wall_time: Some(
                NativeTrainingReplayTiming::from_durations(first.overhead, &overheads)?,
            ),
            step_phases,
            first_replay_wall_time: BenchmarkDuration::from_duration(first.total),
            steady_replay_total_wall_time,
            steady_replay_wall_time: latency_summary(&totals)?,
            steady_microbatches_per_second,
            schedule_cache_keys: self.schedule_cache_keys.clone().unwrap_or_default(),
            accumulation_schedule_cache_keys: self
                .accumulation_schedule_cache_keys
                .clone()
                .unwrap_or_default(),
            checkpoint: self.checkpoint.clone(),
            fallback_count: 0,
            kernel_launch_count: None,
            host_to_device: None,
            device_to_host: None,
            measured_peak_host_memory_bytes: None,
        };
        report.validate()?;
        Ok(report)
    }
}

fn matching_program(
    label: &str,
    inspection: Option<&ProgramInspection>,
    preparation: Option<&NativeCpuProgramPreparationReport>,
    prior_native_identities: &[u64],
) -> Result<Option<NativeTrainingProgramReport>> {
    match (inspection, preparation) {
        (None, None) => Ok(None),
        (Some(inspection), Some(preparation)) => {
            NativeTrainingProgramReport::new(inspection, preparation, prior_native_identities)
                .map(Some)
        }
        _ => Err(invalid(format!(
            "plan and prepared {label} presence differ"
        ))),
    }
}

fn latency_summary(durations: &[Duration]) -> Result<BenchmarkLatencySummary> {
    let mut ordered = durations.to_vec();
    ordered.sort_unstable();
    let nearest_rank = |percentile: usize| {
        let rank = (ordered.len() * percentile).div_ceil(100);
        ordered[rank.saturating_sub(1)]
    };
    Ok(BenchmarkLatencySummary {
        sample_count: count(ordered.len(), "steady replay")?,
        min: BenchmarkDuration::from_duration(ordered[0]),
        nearest_rank_p50: BenchmarkDuration::from_duration(nearest_rank(50)),
        nearest_rank_p95: BenchmarkDuration::from_duration(nearest_rank(95)),
        max: BenchmarkDuration::from_duration(ordered[ordered.len() - 1]),
    })
}

fn rate(durations: &[Duration]) -> Result<(BenchmarkDuration, Option<f64>)> {
    let total = sum_durations(durations)?;
    Ok((total, rate_from_total(durations.len() as u64, total)?))
}

fn sum_durations(durations: &[Duration]) -> Result<BenchmarkDuration> {
    let total = durations
        .iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(*duration)
                .ok_or_else(|| invalid("replay duration overflows"))
        })?;
    Ok(BenchmarkDuration::from_duration(total))
}

fn rate_from_total(sample_count: u64, total: BenchmarkDuration) -> Result<Option<f64>> {
    let total = total
        .to_duration()
        .map_err(|_| invalid("invalid steady replay total duration"))?;
    Ok((!total.is_zero()).then(|| sample_count as f64 / total.as_secs_f64()))
}

fn validate_total_duration(
    summary: &BenchmarkLatencySummary,
    total: BenchmarkDuration,
) -> Result<()> {
    let remaining = u128::from(summary.sample_count - 1);
    let min = summary
        .min
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay minimum duration"))?;
    let max = summary
        .max
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay maximum duration"))?;
    let total = total
        .as_nanos()
        .map_err(|_| invalid("invalid steady replay total duration"))?;
    let lower = remaining
        .checked_mul(min)
        .and_then(|rest| max.checked_add(rest))
        .ok_or_else(|| invalid("steady replay duration lower bound overflows"))?;
    let upper = remaining
        .checked_mul(max)
        .and_then(|rest| min.checked_add(rest))
        .ok_or_else(|| invalid("steady replay duration upper bound overflows"))?;
    if !(lower..=upper).contains(&total) {
        return Err(invalid("steady replay total is inconsistent with summary"));
    }
    Ok(())
}

fn validate_phase_partition(
    total: BenchmarkDuration,
    executor: BenchmarkDuration,
    overhead: BenchmarkDuration,
    label: &str,
) -> Result<()> {
    let total = total
        .as_nanos()
        .map_err(|_| invalid(format!("invalid {label} total duration")))?;
    let partitioned = executor
        .as_nanos()
        .map_err(|_| invalid(format!("invalid {label} executor duration")))?
        .checked_add(
            overhead
                .as_nanos()
                .map_err(|_| invalid(format!("invalid {label} overhead duration")))?,
        )
        .ok_or_else(|| invalid(format!("{label} phase duration overflows")))?;
    if partitioned != total {
        return Err(invalid(format!("{label} phases do not partition total")));
    }
    Ok(())
}

fn count(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid(format!("{label} count overflows u64")))
}

fn invalid(reason: impl Into<String>) -> Error {
    Error::SessionTraining {
        reason: format!("native training scoreboard: {}", reason.into()),
    }
}

#[cfg(test)]
mod tests;
