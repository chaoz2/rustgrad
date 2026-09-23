//! Authenticated observability values for strict-native CPU preparation and replay.
//!
//! These types report compile, cache, dispatch, traffic, and timing evidence.
//! Prepared programs, replay transactions, and backend execution remain in the
//! parent module.

use super::*;
use serde::{Deserialize, Serialize};

/// Deterministic explanation of a strict-native program's module-dispatch
/// segmentation and the referenced modules it actually reaches. Counts
/// describe the sealed tape, not replay timing. Current preparation keeps
/// same-module derived dependencies inside typed dispatcher actions; the
/// derived-dependency field remains for authenticated v17 wire compatibility.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCpuDispatchSegmentation {
    pub(in crate::session) segment_count: u64,
    pub(in crate::session) dispatch_reached_module_count: u64,
    pub(in crate::session) terminal_segment_count: u64,
    pub(in crate::session) non_dispatch_boundary_count: u64,
    pub(in crate::session) module_change_count: u64,
    pub(in crate::session) output_slot_alias_count: u64,
    pub(in crate::session) derived_slot_dependency_count: u64,
}

impl NativeCpuDispatchSegmentation {
    pub(super) fn from_native(
        segmentation: crate::engine::NativeDispatchSegmentation,
    ) -> Result<Self> {
        let count = |value, label| {
            u64::try_from(value)
                .map_err(|_| training(format!("native CPU {label} count overflows")))
        };
        Ok(Self {
            segment_count: count(segmentation.segment_count, "dispatch segment")?,
            dispatch_reached_module_count: count(
                segmentation.dispatch_reached_module_count,
                "dispatch-reached module",
            )?,
            terminal_segment_count: count(
                segmentation.terminal_segment_count,
                "terminal dispatch segment",
            )?,
            non_dispatch_boundary_count: count(
                segmentation.non_dispatch_boundary_count,
                "non-dispatch boundary",
            )?,
            module_change_count: count(segmentation.module_change_count, "module-change boundary")?,
            output_slot_alias_count: count(
                segmentation.output_slot_alias_count,
                "output-slot-alias boundary",
            )?,
            derived_slot_dependency_count: count(
                segmentation.derived_slot_dependency_count,
                "derived-slot-dependency boundary",
            )?,
        })
    }

    pub(in crate::session) fn authenticates(
        &self,
        rendered_entry_count: u64,
        referenced_module_count: u64,
    ) -> bool {
        let explained = self
            .terminal_segment_count
            .checked_add(self.non_dispatch_boundary_count)
            .and_then(|count| count.checked_add(self.module_change_count))
            .and_then(|count| count.checked_add(self.output_slot_alias_count))
            .and_then(|count| count.checked_add(self.derived_slot_dependency_count));
        let has_dispatch_segments = self.segment_count != 0;
        let expected_module_change_count = self.dispatch_reached_module_count.saturating_sub(1);
        explained == Some(self.segment_count)
            && self.segment_count <= rendered_entry_count
            && self.terminal_segment_count == u64::from(has_dispatch_segments)
            && self.non_dispatch_boundary_count == 0
            && (self.dispatch_reached_module_count != 0) == has_dispatch_segments
            && self.dispatch_reached_module_count <= self.segment_count
            && self.dispatch_reached_module_count <= referenced_module_count
            && self.module_change_count == expected_module_change_count
    }

    pub(super) fn validate(
        &self,
        rendered_entry_count: usize,
        referenced_module_count: usize,
    ) -> Result<()> {
        let rendered_entry_count = u64::try_from(rendered_entry_count)
            .map_err(|_| training("native CPU rendered entry count overflows"))?;
        let referenced_module_count = u64::try_from(referenced_module_count)
            .map_err(|_| training("native CPU referenced module count overflows"))?;
        if !self.authenticates(rendered_entry_count, referenced_module_count) {
            return Err(training(
                "native CPU dispatch segmentation evidence differs",
            ));
        }
        Ok(())
    }

    /// Native module calls encoded by the sealed dispatch tape. This is zero
    /// when every rendered entry is omitted by authenticated elision.
    pub const fn segment_count(&self) -> u64 {
        self.segment_count
    }

    /// Distinct referenced modules reached by at least one sealed dispatch
    /// segment. Rendered modules containing only elided entries are excluded.
    pub const fn dispatch_reached_module_count(&self) -> u64 {
        self.dispatch_reached_module_count
    }

    /// Segment ending at the terminal edge; one for a nonempty dispatch tape.
    pub const fn terminal_segment_count(&self) -> u64 {
        self.terminal_segment_count
    }

    /// Segments ended before an item without a shared dispatcher; strict
    /// preparation authenticates this as zero.
    pub const fn non_dispatch_boundary_count(&self) -> u64 {
        self.non_dispatch_boundary_count
    }

    /// Segments ended because the next item uses a different reached module.
    pub const fn module_change_count(&self) -> u64 {
        self.module_change_count
    }

    /// Segments ended to prevent an output from aliasing an input slot in the
    /// same module invocation.
    pub const fn output_slot_alias_count(&self) -> u64 {
        self.output_slot_alias_count
    }

    /// Legacy count of segments ended before a derived slot dependency. New
    /// private Copy/Affine actions keep admitted same-module dependencies
    /// inside one call, so current prepared programs report zero here.
    pub const fn derived_slot_dependency_count(&self) -> u64 {
        self.derived_slot_dependency_count
    }
}

/// Preparation evidence for one strict-native CPU pure program.
///
/// Stable identities and cache counts describe compilation only. Wall time is
/// deliberately observational and does not participate in either identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCpuPreparationWork {
    pub(super) rendered_entry_count: usize,
    pub(super) rendered_source_bytes: usize,
    pub(super) loaded_module_count: usize,
    pub(super) referenced_module_count: usize,
    pub(super) unique_rendered_entry_count: usize,
    pub(super) unique_rendered_source_bytes: usize,
    pub(super) shared_prefix_entry_count: usize,
    pub(super) shared_prefix_source_bytes: usize,
    pub(super) shared_prefix_source_program: Option<usize>,
    pub(super) durable_artifact_cache_hit_count: usize,
    pub(super) durable_artifact_cache_miss_count: usize,
    pub(super) combined_compile_link_count: usize,
    pub(super) object_compile_count: usize,
    pub(super) linker_invocation_count: usize,
    pub(super) compiler_invocation_count: usize,
}

impl NativeCpuPreparationWork {
    pub(super) fn from_module(module: crate::backend::NativeScheduleModulePreparation) -> Self {
        Self {
            rendered_entry_count: module.rendered_entry_count,
            rendered_source_bytes: module.rendered_source_bytes,
            loaded_module_count: module.loaded_module_count,
            referenced_module_count: module.referenced_module_count,
            unique_rendered_entry_count: module.unique_rendered_entry_count,
            unique_rendered_source_bytes: module.unique_rendered_source_bytes,
            shared_prefix_entry_count: module.shared_prefix_entry_count,
            shared_prefix_source_bytes: module.shared_prefix_source_bytes,
            shared_prefix_source_program: module.shared_prefix_source_program,
            durable_artifact_cache_hit_count: module.durable_artifact_cache_hit_count,
            durable_artifact_cache_miss_count: module.durable_artifact_cache_miss_count,
            combined_compile_link_count: module.combined_compile_link_count,
            object_compile_count: module.object_compile_count,
            linker_invocation_count: module.linker_invocation_count,
            compiler_invocation_count: module.compiler_invocation_count,
        }
    }

    pub const fn rendered_entry_count(&self) -> usize {
        self.rendered_entry_count
    }

    /// Exact bytes in the authenticated rendered kernel sources. Wrapper and
    /// compiler-generated bytes are intentionally excluded.
    pub const fn rendered_source_bytes(&self) -> usize {
        self.rendered_source_bytes
    }

    pub const fn loaded_module_count(&self) -> usize {
        self.loaded_module_count
    }

    /// Distinct native schedule modules referenced by this program's entries.
    pub const fn referenced_module_count(&self) -> usize {
        self.referenced_module_count
    }

    /// Rendered entries compiled into this program's own native module.
    pub const fn unique_rendered_entry_count(&self) -> usize {
        self.unique_rendered_entry_count
    }

    pub const fn unique_rendered_source_bytes(&self) -> usize {
        self.unique_rendered_source_bytes
    }

    /// Exact leading entries reused from an earlier program's native module.
    pub const fn shared_prefix_entry_count(&self) -> usize {
        self.shared_prefix_entry_count
    }

    pub const fn shared_prefix_source_bytes(&self) -> usize {
        self.shared_prefix_source_bytes
    }

    /// Earlier batch-program ordinal used to bind public scoreboard identity.
    pub(crate) const fn shared_prefix_source_program(&self) -> Option<usize> {
        self.shared_prefix_source_program
    }

    pub const fn durable_artifact_cache_hit_count(&self) -> usize {
        self.durable_artifact_cache_hit_count
    }

    pub const fn durable_artifact_cache_miss_count(&self) -> usize {
        self.durable_artifact_cache_miss_count
    }

    pub const fn compiler_invocation_count(&self) -> usize {
        self.compiler_invocation_count
    }

    /// Single-process translation-unit compilation plus shared-library link.
    pub const fn combined_compile_link_count(&self) -> usize {
        self.combined_compile_link_count
    }

    /// PIC object translation units compiled before a separate final link.
    pub const fn object_compile_count(&self) -> usize {
        self.object_compile_count
    }

    /// Separate shared-library linker processes.
    pub const fn linker_invocation_count(&self) -> usize {
        self.linker_invocation_count
    }

    pub(super) fn validate(&self, native_item_count: usize) -> Result<()> {
        let durable_access_count = self
            .durable_artifact_cache_hit_count
            .checked_add(self.durable_artifact_cache_miss_count)
            .ok_or_else(|| training("compiled native CPU durable cache count overflows"))?;
        let observed_compiler_invocation_count = self
            .combined_compile_link_count
            .checked_add(self.object_compile_count)
            .and_then(|count| count.checked_add(self.linker_invocation_count))
            .ok_or_else(|| training("compiled native CPU compiler count overflows"))?;
        let unique_rendered_entry_count = u64::try_from(self.unique_rendered_entry_count)
            .map_err(|_| training("compiled native CPU unique entry count overflows"))?;
        let compiler_process_inventory = (
            u64::try_from(self.combined_compile_link_count)
                .map_err(|_| training("compiled native CPU compiler count overflows"))?,
            u64::try_from(self.object_compile_count)
                .map_err(|_| training("compiled native CPU compiler count overflows"))?,
            u64::try_from(self.linker_invocation_count)
                .map_err(|_| training("compiled native CPU compiler count overflows"))?,
        );
        let compiler_mode_is_valid = if self.durable_artifact_cache_miss_count == 0 {
            observed_compiler_invocation_count == 0
        } else {
            crate::cpu_jit::NativeScheduleModuleBuildMode::for_unique_rendered_entry_count(
                unique_rendered_entry_count,
            )
            .is_some_and(|mode| mode.process_inventory() == compiler_process_inventory)
        };
        if self.rendered_entry_count > native_item_count
            || (self.rendered_entry_count == 0) != (native_item_count == 0)
            || self
                .unique_rendered_entry_count
                .checked_add(self.shared_prefix_entry_count)
                != Some(self.rendered_entry_count)
            || self
                .unique_rendered_source_bytes
                .checked_add(self.shared_prefix_source_bytes)
                != Some(self.rendered_source_bytes)
            || (self.rendered_entry_count == 0) != (self.rendered_source_bytes == 0)
            || (self.unique_rendered_entry_count == 0) != (self.unique_rendered_source_bytes == 0)
            || (self.shared_prefix_entry_count == 0) != (self.shared_prefix_source_bytes == 0)
            || self.rendered_source_bytes < self.rendered_entry_count
            || self.unique_rendered_source_bytes < self.unique_rendered_entry_count
            || self.shared_prefix_source_bytes < self.shared_prefix_entry_count
            || self.loaded_module_count != usize::from(self.unique_rendered_entry_count != 0)
            || (self.rendered_entry_count == 0) != (self.referenced_module_count == 0)
            || (self.shared_prefix_entry_count == 0) != self.shared_prefix_source_program.is_none()
            || self
                .loaded_module_count
                .checked_add(usize::from(self.shared_prefix_entry_count != 0))
                != Some(self.referenced_module_count)
            || durable_access_count > self.loaded_module_count
            || self.object_compile_count > self.unique_rendered_entry_count
            || self.compiler_invocation_count != observed_compiler_invocation_count
            || !compiler_mode_is_valid
        {
            return Err(training(
                "compiled native CPU module preparation evidence mismatch",
            ));
        }
        Ok(())
    }
}

/// Observed wall-time partition for one strict-native CPU program preparation.
///
/// These durations describe host preparation only. They are excluded from
/// every capture, native-program, cache, and checkpoint identity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeCpuPreparationPhases {
    layout_wall_time: Duration,
    render_wall_time: Duration,
    compiler_process_wall_time: Duration,
    compiler_process_total_wall_time: Duration,
    linker_process_wall_time: Duration,
    module_load_wall_time: Duration,
    residual_wall_time: Duration,
}

impl NativeCpuPreparationPhases {
    pub(super) fn from_module(
        module: crate::backend::NativeScheduleModulePreparation,
        total_wall_time: Duration,
    ) -> Result<Self> {
        let accounted = [
            module.layout_wall_time,
            module.render_wall_time,
            module.compiler_process_wall_time,
            module.module_load_wall_time,
        ]
        .into_iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(duration)
                .ok_or_else(|| training("compiled native CPU preparation wall time overflows"))
        })?;
        let residual_wall_time = total_wall_time.checked_sub(accounted).ok_or_else(|| {
            training("compiled native CPU preparation phases exceed total wall time")
        })?;
        Ok(Self {
            layout_wall_time: module.layout_wall_time,
            render_wall_time: module.render_wall_time,
            compiler_process_wall_time: module.compiler_process_wall_time,
            compiler_process_total_wall_time: module.compiler_process_total_wall_time,
            linker_process_wall_time: module.linker_process_wall_time,
            module_load_wall_time: module.module_load_wall_time,
            residual_wall_time,
        })
    }

    fn validate(&self, total_wall_time: Duration, work: &NativeCpuPreparationWork) -> Result<()> {
        let total = [
            self.layout_wall_time,
            self.render_wall_time,
            self.compiler_process_wall_time,
            self.module_load_wall_time,
            self.residual_wall_time,
        ]
        .into_iter()
        .try_fold(Duration::ZERO, |total, duration| {
            total
                .checked_add(duration)
                .ok_or_else(|| training("compiled native CPU preparation wall time overflows"))
        })?;
        if total != total_wall_time
            || (work.compiler_invocation_count == 0
                && self.compiler_process_wall_time != Duration::ZERO)
            || self.compiler_process_total_wall_time < self.compiler_process_wall_time
            || (work.compiler_invocation_count <= 1
                && self.compiler_process_total_wall_time != self.compiler_process_wall_time)
            || self.linker_process_wall_time > self.compiler_process_wall_time
            || (work.compiler_invocation_count == 0
                && self.compiler_process_total_wall_time != Duration::ZERO)
            || (work.linker_invocation_count == 0
                && self.linker_process_wall_time != Duration::ZERO)
            || (work.loaded_module_count == 0 && self.module_load_wall_time != Duration::ZERO)
        {
            return Err(training(
                "compiled native CPU preparation phase evidence mismatch",
            ));
        }
        Ok(())
    }

    pub const fn layout_wall_time(&self) -> Duration {
        self.layout_wall_time
    }

    pub const fn render_wall_time(&self) -> Duration {
        self.render_wall_time
    }

    pub const fn compiler_process_wall_time(&self) -> Duration {
        self.compiler_process_wall_time
    }

    /// Sum of all compiler subprocess durations, including overlapped work.
    pub const fn compiler_process_total_wall_time(&self) -> Duration {
        self.compiler_process_total_wall_time
    }

    /// Wall time of the separate final link, zero for combined compilation.
    pub const fn linker_process_wall_time(&self) -> Duration {
        self.linker_process_wall_time
    }

    pub const fn module_load_wall_time(&self) -> Duration {
        self.module_load_wall_time
    }

    pub const fn residual_wall_time(&self) -> Duration {
        self.residual_wall_time
    }
}

pub(super) fn native_preparation_wall_time(
    module: crate::backend::NativeScheduleModulePreparation,
    residual_wall_time: Duration,
) -> Result<Duration> {
    [
        module.layout_wall_time,
        module.render_wall_time,
        module.compiler_process_wall_time,
        module.module_load_wall_time,
        module.residual_wall_time,
        residual_wall_time,
    ]
    .into_iter()
    .try_fold(Duration::ZERO, |total, duration| {
        total
            .checked_add(duration)
            .ok_or_else(|| training("compiled native CPU preparation wall time overflows"))
    })
}

#[derive(Clone, Debug)]
pub struct NativeCpuProgramPreparationReport {
    pub(super) capture_identity: u64,
    pub(super) native_identity: u64,
    pub(super) vectorized: bool,
    pub(super) native_item_count: usize,
    pub(super) cache_hit_count: usize,
    pub(super) cache_miss_count: usize,
    pub(super) work: NativeCpuPreparationWork,
    pub(super) phases: NativeCpuPreparationPhases,
    pub(super) dispatch_segmentation: NativeCpuDispatchSegmentation,
    pub(super) execution_plan: ExecutionPlanSummary,
    pub(super) wall_time: Duration,
}

/// One compiler subprocess interval normalized to the common native
/// preparation-batch origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeCpuCompilerProcessTiming {
    program_index: usize,
    kind: crate::cpu_jit::NativeCompilerProcessKind,
    rendered_source_bytes: usize,
    permit_request_offset: Duration,
    permit_wait_time: Duration,
    process_wall_time: Duration,
}

impl NativeCpuCompilerProcessTiming {
    pub(super) fn from_native(timing: crate::backend::NativeScheduleCompilerProcessTiming) -> Self {
        Self {
            program_index: timing.program_index,
            kind: timing.kind,
            rendered_source_bytes: timing.rendered_source_bytes,
            permit_request_offset: timing.permit_request_offset,
            permit_wait_time: timing.permit_wait_time,
            process_wall_time: timing.process_wall_time,
        }
    }

    pub(crate) const fn program_index(&self) -> usize {
        self.program_index
    }

    pub(crate) const fn kind(&self) -> crate::cpu_jit::NativeCompilerProcessKind {
        self.kind
    }

    pub(crate) const fn rendered_source_bytes(&self) -> usize {
        self.rendered_source_bytes
    }

    pub(crate) const fn permit_request_offset(&self) -> Duration {
        self.permit_request_offset
    }

    pub(crate) const fn permit_wait_time(&self) -> Duration {
        self.permit_wait_time
    }

    pub(crate) const fn process_wall_time(&self) -> Duration {
        self.process_wall_time
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeCpuModuleOverlap {
    program_index: usize,
    contiguous_prefix_entry_count: usize,
    contiguous_prefix_source_bytes: usize,
    additional_scattered_entry_count: usize,
    additional_scattered_source_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeCpuProgramPairOverlap {
    source_program_index: usize,
    target_program_index: usize,
    contiguous_prefix_entry_count: usize,
    contiguous_prefix_source_bytes: usize,
    additional_scattered_entry_count: usize,
    additional_scattered_source_bytes: usize,
}

impl NativeCpuProgramPairOverlap {
    pub(super) fn from_native(overlap: crate::backend::NativeScheduleProgramPairOverlap) -> Self {
        Self {
            source_program_index: overlap.source_program_index,
            target_program_index: overlap.target_program_index,
            contiguous_prefix_entry_count: overlap.contiguous_prefix_entry_count,
            contiguous_prefix_source_bytes: overlap.contiguous_prefix_source_bytes,
            additional_scattered_entry_count: overlap.additional_scattered_entry_count,
            additional_scattered_source_bytes: overlap.additional_scattered_source_bytes,
        }
    }

    pub(crate) const fn source_program_index(&self) -> usize {
        self.source_program_index
    }

    pub(crate) const fn target_program_index(&self) -> usize {
        self.target_program_index
    }

    pub(crate) const fn contiguous_prefix_entry_count(&self) -> usize {
        self.contiguous_prefix_entry_count
    }

    pub(crate) const fn contiguous_prefix_source_bytes(&self) -> usize {
        self.contiguous_prefix_source_bytes
    }

    pub(crate) const fn additional_scattered_entry_count(&self) -> usize {
        self.additional_scattered_entry_count
    }

    pub(crate) const fn additional_scattered_source_bytes(&self) -> usize {
        self.additional_scattered_source_bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeCpuTranslationUnitEvidence {
    program_index: usize,
    ordinal: usize,
    identity: u64,
    entry_count: usize,
    rendered_source_bytes: usize,
}

impl NativeCpuTranslationUnitEvidence {
    pub(super) fn from_native(unit: crate::backend::NativeScheduleTranslationUnit) -> Self {
        Self {
            program_index: unit.program_index,
            ordinal: unit.ordinal,
            identity: unit.identity,
            entry_count: unit.entry_count,
            rendered_source_bytes: unit.rendered_source_bytes,
        }
    }

    pub(crate) const fn program_index(&self) -> usize {
        self.program_index
    }

    pub(crate) const fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub(crate) const fn identity(&self) -> u64 {
        self.identity
    }

    pub(crate) const fn entry_count(&self) -> usize {
        self.entry_count
    }

    pub(crate) const fn rendered_source_bytes(&self) -> usize {
        self.rendered_source_bytes
    }
}

impl NativeCpuModuleOverlap {
    pub(super) fn from_native(overlap: crate::backend::NativeScheduleModuleOverlap) -> Self {
        Self {
            program_index: overlap.program_index,
            contiguous_prefix_entry_count: overlap.contiguous_prefix_entry_count,
            contiguous_prefix_source_bytes: overlap.contiguous_prefix_source_bytes,
            additional_scattered_entry_count: overlap.additional_scattered_entry_count,
            additional_scattered_source_bytes: overlap.additional_scattered_source_bytes,
        }
    }

    pub(crate) const fn program_index(&self) -> usize {
        self.program_index
    }

    pub(crate) const fn contiguous_prefix_entry_count(&self) -> usize {
        self.contiguous_prefix_entry_count
    }

    pub(crate) const fn contiguous_prefix_source_bytes(&self) -> usize {
        self.contiguous_prefix_source_bytes
    }

    pub(crate) const fn additional_scattered_entry_count(&self) -> usize {
        self.additional_scattered_entry_count
    }

    pub(crate) const fn additional_scattered_source_bytes(&self) -> usize {
        self.additional_scattered_source_bytes
    }
}

impl NativeCpuProgramPreparationReport {
    pub(super) fn validate_work(&self) -> Result<()> {
        self.work.validate(self.native_item_count)?;
        self.phases.validate(self.wall_time, &self.work)?;
        self.dispatch_segmentation.validate(
            self.work.rendered_entry_count,
            self.work.referenced_module_count,
        )
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

    /// Authenticated logical schedule-item inventory. Cache hit and miss counts
    /// use this same logical coverage; [`NativeCpuPreparationWork::rendered_entry_count`]
    /// reports the physical compiled-entry inventory.
    pub const fn native_item_count(&self) -> usize {
        self.native_item_count
    }

    pub const fn cache_hit_count(&self) -> usize {
        self.cache_hit_count
    }

    pub const fn cache_miss_count(&self) -> usize {
        self.cache_miss_count
    }

    /// Concrete rendering, shared-module, durable-cache, and compiler work
    /// performed while attaching this pure program.
    pub const fn work(&self) -> &NativeCpuPreparationWork {
        &self.work
    }

    /// Observed host-time phases within this program's total preparation.
    pub const fn phases(&self) -> &NativeCpuPreparationPhases {
        &self.phases
    }

    /// Exact, mutually exclusive reasons for every sealed module-dispatch
    /// segment in this program.
    pub const fn dispatch_segmentation(&self) -> &NativeCpuDispatchSegmentation {
        &self.dispatch_segmentation
    }

    /// Strict preparation never admits an interpreter fallback item.
    pub const fn fallback_count(&self) -> usize {
        0
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }
}

/// Result of attempting to admit one authenticated render capsule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeCpuRenderCapsuleLoadStatus {
    Hit,
    RecipeUnavailable,
    FileUnavailable,
    DecodeRejected,
    AuthenticationRejected,
}

/// Result of persisting a locally rendered program as a capsule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeCpuRenderCapsuleStoreStatus {
    NotAttempted,
    RecipeUnavailable,
    Stored,
    EncodeRejected,
    FilesystemRejected,
}

/// Compiled training program owning one render-capsule outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeCpuRenderCapsuleProgramRole {
    Main,
    Accumulation,
    PartialFlush,
    ZeroGrad,
    Evaluation,
}

/// Per-program capsule admission and persistence evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCpuRenderCapsuleDiagnostic {
    role: NativeCpuRenderCapsuleProgramRole,
    load: NativeCpuRenderCapsuleLoadStatus,
    store: NativeCpuRenderCapsuleStoreStatus,
}

impl NativeCpuRenderCapsuleDiagnostic {
    pub const fn role(&self) -> NativeCpuRenderCapsuleProgramRole {
        self.role
    }

    pub const fn load(&self) -> NativeCpuRenderCapsuleLoadStatus {
        self.load
    }

    pub const fn store(&self) -> NativeCpuRenderCapsuleStoreStatus {
        self.store
    }

    fn from_native(
        role: NativeCpuTrainingProgramRole,
        value: crate::backend::NativeRenderCapsuleDiagnostic,
    ) -> Self {
        use crate::backend::{NativeRenderCapsuleLoadStatus, NativeRenderCapsuleStoreStatus};
        Self {
            role: role.render_capsule_role(),
            load: match value.load {
                NativeRenderCapsuleLoadStatus::Hit => NativeCpuRenderCapsuleLoadStatus::Hit,
                NativeRenderCapsuleLoadStatus::RecipeUnavailable => {
                    NativeCpuRenderCapsuleLoadStatus::RecipeUnavailable
                }
                NativeRenderCapsuleLoadStatus::FileUnavailable => {
                    NativeCpuRenderCapsuleLoadStatus::FileUnavailable
                }
                NativeRenderCapsuleLoadStatus::DecodeRejected => {
                    NativeCpuRenderCapsuleLoadStatus::DecodeRejected
                }
                NativeRenderCapsuleLoadStatus::AuthenticationRejected => {
                    NativeCpuRenderCapsuleLoadStatus::AuthenticationRejected
                }
            },
            store: match value.store {
                NativeRenderCapsuleStoreStatus::NotAttempted => {
                    NativeCpuRenderCapsuleStoreStatus::NotAttempted
                }
                NativeRenderCapsuleStoreStatus::RecipeUnavailable => {
                    NativeCpuRenderCapsuleStoreStatus::RecipeUnavailable
                }
                NativeRenderCapsuleStoreStatus::Stored => NativeCpuRenderCapsuleStoreStatus::Stored,
                NativeRenderCapsuleStoreStatus::EncodeRejected => {
                    NativeCpuRenderCapsuleStoreStatus::EncodeRejected
                }
                NativeRenderCapsuleStoreStatus::FilesystemRejected => {
                    NativeCpuRenderCapsuleStoreStatus::FilesystemRejected
                }
            },
        }
    }

    pub(super) fn from_ordered_native(
        roles: &[NativeCpuTrainingProgramRole],
        values: Vec<crate::backend::NativeRenderCapsuleDiagnostic>,
    ) -> Result<Vec<Self>> {
        if roles.len() != values.len()
            || values
                .iter()
                .enumerate()
                .any(|(program_index, value)| value.program_index != program_index)
        {
            return Err(training(
                "compiled native CPU render capsule diagnostic inventory differs",
            ));
        }
        Ok(roles
            .iter()
            .copied()
            .zip(values)
            .map(|(role, value)| Self::from_native(role, value))
            .collect())
    }
}

/// Complete preparation evidence for a native CPU AdamW session.
#[derive(Clone, Debug)]
pub struct NativeCpuCompiledAdamWPreparationReport {
    pub(super) main: NativeCpuProgramPreparationReport,
    pub(super) accumulation: Option<NativeCpuProgramPreparationReport>,
    pub(super) partial_flush: Option<NativeCpuProgramPreparationReport>,
    pub(super) zero_grad: Option<NativeCpuProgramPreparationReport>,
    pub(super) evaluation: Option<NativeCpuProgramPreparationReport>,
    pub(super) recurrent_state_count: usize,
    pub(super) recurrent_state_bytes: usize,
    pub(super) render_capsule_hit_count: usize,
    pub(super) render_capsule_miss_count: usize,
    pub(super) local_render_job_count: usize,
    pub(super) parallel_render_overlap_wall_time: Duration,
    pub(super) max_parallel_render_job_count: usize,
    pub(super) parallel_module_overlap_wall_time: Duration,
    pub(super) compiler_process_overlap_wall_time: Duration,
    pub(super) compiler_process_count: usize,
    pub(super) max_parallel_compiler_process_count: usize,
    pub(super) compiler_process_timings: Vec<NativeCpuCompilerProcessTiming>,
    pub(super) module_overlaps: Vec<NativeCpuModuleOverlap>,
    pub(super) program_pair_overlaps: Vec<NativeCpuProgramPairOverlap>,
    pub(super) translation_units: Vec<NativeCpuTranslationUnitEvidence>,
    pub(super) render_capsule_diagnostics: Vec<NativeCpuRenderCapsuleDiagnostic>,
}

impl NativeCpuCompiledAdamWPreparationReport {
    pub const fn main(&self) -> &NativeCpuProgramPreparationReport {
        &self.main
    }

    /// Preparation evidence for the accumulation-only sibling program.
    pub const fn accumulation(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.accumulation.as_ref()
    }

    pub const fn partial_flush(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.partial_flush.as_ref()
    }

    /// Preparation evidence for the captured state-only accumulation reset.
    pub const fn zero_grad(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.zero_grad.as_ref()
    }

    pub const fn evaluation(&self) -> Option<&NativeCpuProgramPreparationReport> {
        self.evaluation.as_ref()
    }

    pub const fn recurrent_state_count(&self) -> usize {
        self.recurrent_state_count
    }

    pub const fn recurrent_state_bytes(&self) -> usize {
        self.recurrent_state_bytes
    }

    pub const fn render_capsule_hit_count(&self) -> usize {
        self.render_capsule_hit_count
    }

    pub const fn render_capsule_miss_count(&self) -> usize {
        self.render_capsule_miss_count
    }

    pub const fn local_render_job_count(&self) -> usize {
        self.local_render_job_count
    }

    /// Per-program cache admission and cold-write outcome in canonical
    /// preparation order. This distinguishes an ordinary missing capsule from
    /// codec, authentication, and filesystem rejection without changing the
    /// fail-soft cache contract.
    pub fn render_capsule_diagnostics(&self) -> &[NativeCpuRenderCapsuleDiagnostic] {
        &self.render_capsule_diagnostics
    }

    /// Exact overlap among the immutable per-program native render jobs.
    pub const fn parallel_render_overlap_wall_time(&self) -> Duration {
        self.parallel_render_overlap_wall_time
    }

    /// Maximum number of immutable per-program native render jobs observed
    /// concurrently. The private worker pool is bounded to two.
    pub const fn max_parallel_render_job_count(&self) -> usize {
        self.max_parallel_render_job_count
    }

    /// Exact overlap among independently authenticated native module compiler
    /// processes. Program preparation times remain their actual durations.
    pub const fn compiler_process_overlap_wall_time(&self) -> Duration {
        self.compiler_process_overlap_wall_time
    }

    /// Exact overlap among complete compiler-and-loader module jobs. This is
    /// the overlap subtracted when partitioning caller-observed preparation.
    pub const fn parallel_module_overlap_wall_time(&self) -> Duration {
        self.parallel_module_overlap_wall_time
    }

    pub const fn compiler_process_count(&self) -> usize {
        self.compiler_process_count
    }

    pub const fn max_parallel_compiler_process_count(&self) -> usize {
        self.max_parallel_compiler_process_count
    }

    /// Ordered compiler subprocess evidence normalized to one preparation
    /// batch origin. Warm durable-cache and full-prefix programs contribute no
    /// process records.
    pub(crate) fn compiler_process_timings(&self) -> &[NativeCpuCompilerProcessTiming] {
        &self.compiler_process_timings
    }

    pub(crate) fn module_overlaps(&self) -> &[NativeCpuModuleOverlap] {
        &self.module_overlaps
    }

    pub(crate) fn program_pair_overlaps(&self) -> &[NativeCpuProgramPairOverlap] {
        &self.program_pair_overlaps
    }

    pub(crate) fn translation_units(&self) -> &[NativeCpuTranslationUnitEvidence] {
        &self.translation_units
    }
}

/// Logical host traffic completed by one successful strict-native CPU replay.
///
/// External imports count only fallback owned copies into retained workspace
/// storage; supported dense F32/I32 inputs bind caller storage read-only for
/// the invocation instead. Recurrent bytes are borrowed directly from the
/// authoritative host banks. Exact unchanged recurrent successors may retain
/// the active bank; all other successors borrow the inactive bank. Requested
/// egress counts describe detached CPU outputs. None of these fields are
/// host/device transfer measurements.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeCpuReplayTraffic {
    pub(super) external_input_import_count: u64,
    pub(super) external_input_import_bytes: u64,
    pub(super) borrowed_recurrent_input_bytes: u64,
    pub(super) borrowed_recurrent_output_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) retained_recurrent_state_count: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) retained_recurrent_state_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) replaced_recurrent_state_count: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) replaced_recurrent_state_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) materialized_egress_count: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub(super) materialized_egress_bytes: u64,
}

const fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

impl NativeCpuReplayTraffic {
    pub(crate) const fn new(
        external_input_import_count: u64,
        external_input_import_bytes: u64,
        borrowed_recurrent_input_bytes: u64,
        borrowed_recurrent_output_bytes: u64,
    ) -> Self {
        Self {
            external_input_import_count,
            external_input_import_bytes,
            borrowed_recurrent_input_bytes,
            borrowed_recurrent_output_bytes,
            retained_recurrent_state_count: 0,
            retained_recurrent_state_bytes: 0,
            replaced_recurrent_state_count: 0,
            replaced_recurrent_state_bytes: 0,
            materialized_egress_count: 0,
            materialized_egress_bytes: 0,
        }
    }

    pub(crate) const fn with_recurrent_inventory(
        mut self,
        retained_count: u64,
        retained_bytes: u64,
        replaced_count: u64,
        replaced_bytes: u64,
    ) -> Self {
        self.retained_recurrent_state_count = retained_count;
        self.retained_recurrent_state_bytes = retained_bytes;
        self.replaced_recurrent_state_count = replaced_count;
        self.replaced_recurrent_state_bytes = replaced_bytes;
        self
    }

    pub(crate) const fn with_materialized_egress(mut self, count: u64, bytes: u64) -> Self {
        self.materialized_egress_count = count;
        self.materialized_egress_bytes = bytes;
        self
    }

    pub const fn external_input_import_count(&self) -> u64 {
        self.external_input_import_count
    }

    pub const fn external_input_import_bytes(&self) -> u64 {
        self.external_input_import_bytes
    }

    pub const fn borrowed_recurrent_input_bytes(&self) -> u64 {
        self.borrowed_recurrent_input_bytes
    }

    pub const fn borrowed_recurrent_output_bytes(&self) -> u64 {
        self.borrowed_recurrent_output_bytes
    }

    /// Logical recurrent states whose authenticated successor retains the
    /// active bank instead of writing and flipping an identical inactive bank.
    pub const fn retained_recurrent_state_count(&self) -> u64 {
        self.retained_recurrent_state_count
    }

    /// Logical bytes served from authenticated active recurrent-state banks.
    pub const fn retained_recurrent_state_bytes(&self) -> u64 {
        self.retained_recurrent_state_bytes
    }

    /// Logical recurrent states physically written into inactive banks.
    pub const fn replaced_recurrent_state_count(&self) -> u64 {
        self.replaced_recurrent_state_count
    }

    /// Recurrent successor bytes physically written into inactive banks.
    pub const fn replaced_recurrent_state_bytes(&self) -> u64 {
        self.replaced_recurrent_state_bytes
    }

    /// Logical workspace-backed requested CPU tensors detached for the caller.
    pub const fn materialized_egress_count(&self) -> u64 {
        self.materialized_egress_count
    }

    /// Logical descriptor bytes detached from the replay workspace. This is
    /// host output materialization, not a device transfer measurement.
    pub const fn materialized_egress_bytes(&self) -> u64 {
        self.materialized_egress_bytes
    }
}

/// Truthful per-invocation evidence for strict-native CPU replay.
#[derive(Clone, Debug)]
pub struct NativeCpuRunReport {
    pub(super) capture_identity: u64,
    pub(super) native_identity: u64,
    pub(super) vectorized: bool,
    pub(super) successful_invocation: u64,
    /// Prepared native items include authenticated entries whose execution may
    /// be elided by the retained workspace.
    pub(super) native_item_count: usize,
    pub(super) executed_native_item_count: usize,
    pub(super) module_dispatch_count: usize,
    pub(super) module_dispatched_native_item_count: usize,
    pub(super) skipped_output_clear_count: usize,
    pub(super) schedule_cache_keys: Vec<u64>,
    pub(super) traffic: NativeCpuReplayTraffic,
    pub(super) native_dispatcher_wall_time: Duration,
    pub(super) executor_wall_time: Duration,
    pub(super) wall_time: Duration,
}

impl NativeCpuRunReport {
    pub const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub const fn native_identity(&self) -> u64 {
        self.native_identity
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    pub const fn successful_invocation(&self) -> u64 {
        self.successful_invocation
    }

    pub const fn first_successful_invocation(&self) -> bool {
        self.successful_invocation == 1
    }

    /// Authenticated logical schedule-item inventory for this replay. Retained
    /// or grouped logical items need not each invoke a physical JIT entry.
    pub const fn native_item_count(&self) -> usize {
        self.native_item_count
    }

    /// Physical native entries that actually invoked a prepared JIT function
    /// during this successfully published replay.
    pub const fn executed_native_item_count(&self) -> usize {
        self.executed_native_item_count
    }

    /// Contiguous authenticated native-module tape segments entered during
    /// this replay. Logical item accounting and failure ordinals are unchanged.
    pub const fn module_dispatch_count(&self) -> usize {
        self.module_dispatch_count
    }

    /// Physical native entries invoked through authenticated module tape
    /// metadata rather than the conservative per-entry binding path.
    pub const fn module_dispatched_native_item_count(&self) -> usize {
        self.module_dispatched_native_item_count
    }

    /// Output clears skipped because the prepared native kernel is
    /// authenticated as fully overwriting every output lane on success.
    pub const fn skipped_output_clear_count(&self) -> usize {
        self.skipped_output_clear_count
    }

    /// Strict replay never executes an interpreter fallback item.
    pub const fn fallback_count(&self) -> usize {
        0
    }

    pub fn schedule_cache_keys(&self) -> &[u64] {
        &self.schedule_cache_keys
    }

    pub const fn traffic(&self) -> &NativeCpuReplayTraffic {
        &self.traffic
    }

    /// Wall time spent inside the sealed native executor for this successful
    /// invocation. Validation failures and uncommitted calls expose no report.
    pub const fn executor_wall_time(&self) -> Duration {
        self.executor_wall_time
    }

    /// Wall time spent inside authenticated native schedule-module dispatcher
    /// calls. This excludes Rust-side workspace and pointer preparation.
    pub const fn native_dispatcher_wall_time(&self) -> Duration {
        self.native_dispatcher_wall_time
    }

    /// Checked executor remainder outside native dispatcher calls. This covers
    /// Rust-side workspace, binding, input/output work, and any conservative
    /// per-entry execution that did not enter a sealed dispatcher segment.
    pub fn executor_host_wall_time(&self) -> Duration {
        self.executor_wall_time
            .checked_sub(self.native_dispatcher_wall_time)
            .expect("native CPU run report validates dispatcher timing")
    }

    /// End-to-end replay time outside the sealed native executor. For
    /// recurrent programs this covers staging, validation, and atomic commit.
    pub fn replay_overhead_wall_time(&self) -> Duration {
        self.wall_time
            .checked_sub(self.executor_wall_time)
            .expect("native CPU run report validates nested executor timing")
    }

    /// End-to-end wall time for the successful invocation.
    pub const fn wall_time(&self) -> Duration {
        self.wall_time
    }
}
