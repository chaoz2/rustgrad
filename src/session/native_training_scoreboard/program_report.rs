//! Native compiled-training program preparation reports and aggregates.
//!
//! This module owns the serialized per-program preparation record and the
//! exact compiler/render aggregates derived from those records. It does not
//! execute training programs or collect timings.

use super::super::{NativeCpuDispatchSegmentation, NativeCpuProgramPreparationReport};
use super::inspection::{NativeTrainingPreparationTiming, ProgramInspection};
use super::{
    NATIVE_TRAINING_REPORT_FORMAT_V5, NATIVE_TRAINING_REPORT_FORMAT_V6,
    NATIVE_TRAINING_REPORT_FORMAT_V7, NATIVE_TRAINING_REPORT_FORMAT_V8,
    NATIVE_TRAINING_REPORT_FORMAT_V9, NATIVE_TRAINING_REPORT_FORMAT_V10,
    NATIVE_TRAINING_REPORT_FORMAT_V11, NATIVE_TRAINING_REPORT_FORMAT_V12,
    NATIVE_TRAINING_REPORT_FORMAT_V13, NATIVE_TRAINING_REPORT_FORMAT_V14,
    NATIVE_TRAINING_REPORT_FORMAT_V15, NATIVE_TRAINING_REPORT_FORMAT_V16,
    NATIVE_TRAINING_REPORT_FORMAT_V17, NATIVE_TRAINING_REPORT_FORMAT_V18,
    NATIVE_TRAINING_REPORT_FORMAT_V19, NATIVE_TRAINING_REPORT_FORMAT_V20,
    NATIVE_TRAINING_REPORT_FORMAT_V21, NATIVE_TRAINING_REPORT_FORMAT_V22,
    NATIVE_TRAINING_REPORT_FORMAT_VERSION, count, invalid,
};
use crate::Result;
use serde::{Deserialize, Serialize};

/// Static logical work, identity, and cache facts for one prepared pure
/// training program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTrainingProgramReport {
    pub(super) capture_identity: u64,
    pub(super) native_identity: u64,
    pub(super) vectorized: bool,
    pub(super) execution_plan_identity: u64,
    pub(super) logical_schedule_item_count: u64,
    pub(super) peak_logical_temporary_allocation_count: u64,
    pub(super) peak_logical_temporary_bytes: u64,
    pub(super) native_item_count: u64,
    pub(super) cache_hit_count: u64,
    pub(super) cache_miss_count: u64,
    #[serde(default)]
    pub(super) rendered_entry_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) rendered_source_bytes: Option<u64>,
    #[serde(default)]
    pub(super) loaded_module_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) referenced_module_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) unique_rendered_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shared_prefix_entry_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shared_prefix_source_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shared_prefix_source_program_index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shared_prefix_source_native_identity: Option<u64>,
    #[serde(default)]
    pub(super) durable_artifact_cache_hit_count: u64,
    #[serde(default)]
    pub(super) durable_artifact_cache_miss_count: u64,
    #[serde(default)]
    pub(super) compiler_invocation_count: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) combined_compile_link_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) object_compile_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) linker_invocation_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) preparation_timing: Option<NativeTrainingPreparationTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) dispatch_segmentation: Option<NativeCpuDispatchSegmentation>,
}

impl NativeTrainingProgramReport {
    pub(super) fn new(
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

    pub(super) fn validate(&self, format_version: u32) -> Result<()> {
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
pub(super) struct NativeCompilerAggregate {
    pub(super) process_count: u64,
    pub(super) module_job_count: u64,
    pub(super) effective_wall_sum: u128,
    pub(super) effective_wall_max: u128,
    pub(super) cumulative_wall_sum: u128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct NativeRenderAggregate {
    pub(super) job_count: u64,
    pub(super) wall_sum: u128,
    pub(super) wall_max: u128,
}

impl NativeRenderAggregate {
    pub(super) fn from_programs<'a>(
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
    pub(super) fn from_programs<'a>(
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

    pub(super) fn internal_overlap(self) -> Result<u128> {
        self.cumulative_wall_sum
            .checked_sub(self.effective_wall_sum)
            .ok_or_else(|| invalid("native internal compiler overlap underflows"))
    }
}
