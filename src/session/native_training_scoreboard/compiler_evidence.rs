//! Native compiler-process and shared-module preparation evidence.
//!
//! These value types normalize detached preparation observations and validate
//! their overlap, translation-unit, and critical-tail relationships. They do
//! not execute compiler processes or participate in executable identities.

use super::super::NativeCpuCompiledTrainingPreparationReport;
use super::super::compiled_training::{
    NativeCpuCompilerProcessTiming, NativeCpuModuleOverlap, NativeCpuProgramPairOverlap,
    NativeCpuTranslationUnitEvidence,
};
use super::{
    NATIVE_TRAINING_REPORT_FORMAT_V19, NATIVE_TRAINING_REPORT_FORMAT_V20,
    NATIVE_TRAINING_REPORT_FORMAT_V21, NATIVE_TRAINING_REPORT_FORMAT_V22,
    NATIVE_TRAINING_REPORT_FORMAT_VERSION, NativeTrainingProgramReport, count, invalid,
};
use crate::cpu_jit::NativeCompilerProcessKind;
use crate::{BenchmarkDuration, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Compiler subprocess role in one cold native preparation batch.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub(super) enum NativeTrainingCompilerProcessKind {
    Combined,
    Object { ordinal: u64 },
    Link,
}

/// Portable compiler subprocess timing normalized to one preparation-batch
/// origin. These observations never participate in executable identities.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeTrainingCompilerProcessTiming {
    pub(super) program_index: u64,
    pub(super) native_identity: u64,
    pub(super) process: NativeTrainingCompilerProcessKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) rendered_source_bytes: Option<u64>,
    pub(super) permit_request_offset: BenchmarkDuration,
    pub(super) permit_wait: BenchmarkDuration,
    pub(super) process_wall_time: BenchmarkDuration,
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
pub(super) struct NativeTrainingModuleOverlap {
    pub(super) program_index: u64,
    pub(super) native_identity: u64,
    pub(super) evidence_identity: u64,
    pub(super) contiguous_prefix_entry_count: u64,
    pub(super) contiguous_prefix_source_bytes: u64,
    pub(super) additional_scattered_entry_count: u64,
    pub(super) additional_scattered_source_bytes: u64,
}

impl NativeTrainingModuleOverlap {
    pub(super) fn from_preparation(
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

    pub(super) fn with_evidence_identity(mut self, main_native_identity: u64) -> Self {
        self.evidence_identity = self.expected_evidence_identity(main_native_identity);
        self
    }

    pub(super) fn expected_evidence_identity(&self, main_native_identity: u64) -> u64 {
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

pub(super) fn validate_module_overlaps(
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
pub(super) struct NativeTrainingProgramPairOverlap {
    pub(super) source_program_index: u64,
    pub(super) source_native_identity: u64,
    pub(super) target_program_index: u64,
    pub(super) target_native_identity: u64,
    pub(super) evidence_identity: u64,
    pub(super) contiguous_prefix_entry_count: u64,
    pub(super) contiguous_prefix_source_bytes: u64,
    pub(super) additional_scattered_entry_count: u64,
    pub(super) additional_scattered_source_bytes: u64,
}

impl NativeTrainingProgramPairOverlap {
    pub(super) fn from_preparation(
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

    pub(super) fn with_evidence_identity(mut self) -> Self {
        self.evidence_identity = self.expected_evidence_identity();
        self
    }

    pub(super) fn expected_evidence_identity(&self) -> u64 {
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
pub(super) struct NativeTrainingTranslationUnit {
    pub(super) program_index: u64,
    pub(super) native_identity: u64,
    pub(super) ordinal: u64,
    pub(super) translation_unit_identity: u64,
    pub(super) evidence_identity: u64,
    pub(super) entry_count: u64,
    pub(super) rendered_source_bytes: u64,
}

impl NativeTrainingTranslationUnit {
    pub(super) fn from_preparation(
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

    pub(super) fn with_evidence_identity(mut self) -> Self {
        self.evidence_identity = self.expected_evidence_identity();
        self
    }

    pub(super) fn expected_evidence_identity(&self) -> u64 {
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

pub(super) fn validate_program_pair_overlaps(
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

pub(super) fn validate_translation_units(
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
pub(super) struct NativeTrainingCompilerCriticalTail {
    pub(super) program_index: u64,
    pub(super) native_identity: u64,
    pub(super) process: NativeTrainingCompilerProcessKind,
    pub(super) finish_offset: BenchmarkDuration,
    pub(super) post_main_tail: BenchmarkDuration,
}

pub(super) struct CompilerProcessValidationContext {
    pub(super) format_version: u32,
    pub(super) process_count: u64,
    pub(super) max_parallel: u64,
    pub(super) claimed_overlap: BenchmarkDuration,
    pub(super) prepare_wall_time: BenchmarkDuration,
}

pub(super) fn compiler_process_evidence(
    preparation: &NativeCpuCompiledTrainingPreparationReport,
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

pub(super) fn validate_compiler_process_evidence(
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
