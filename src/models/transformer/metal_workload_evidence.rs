//! Versioned, deterministic JSON evidence for one completed Metal Llama workload.

use crate::runtime::metal::{
    MetalDeviceInfo, MetalDevicePreparationReport, MetalDeviceRunReport, MetalDeviceSessionSummary,
};
use serde::Serialize;
use std::{
    error, fmt,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
    time::Duration,
};

/// Current deterministic Metal Llama workload-evidence JSON format.
pub const LLAMA_METAL_WORKLOAD_EVIDENCE_FORMAT_VERSION: u32 = 2;

const MAX_CONTEXT_FIELD_BYTES: usize = 1_024;

/// Host-observed performance evidence for one completed Llama workload.
///
/// Planned bytes are static Metal allocation facts. Transfer counts and bytes
/// are host API calls, not physical-bus measurements. Durations are host wall
/// time; no field claims GPU time, peak memory, or GPU throughput.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMetalWorkloadEvidence {
    /// Exact token-step capture plus resident/state payload identity.
    pub token_step_deployment_identity: u64,
    /// Exact fixed-span capture plus resident/state payload identity when enabled.
    pub fixed_prefill_deployment_identity: Option<u64>,
    /// Deterministic token-step allocation, kernel, and fallback facts.
    pub plan: MetalDeviceSessionSummary,
    /// Fixed-span prompt-program facts when that program is enabled.
    pub fixed_prefill_plan: Option<MetalDeviceSessionSummary>,
    /// One-time token-step planning, native preparation, and host API writes.
    pub token_step_preparation: MetalDevicePreparationReport,
    /// One-time fixed-span prefill preparation when that program is enabled.
    pub fixed_prefill_preparation: Option<MetalDevicePreparationReport>,
    /// The first successful device invocation, if the workload executed one.
    pub first_successful_run: Option<LlamaMetalWorkloadPhase>,
    /// All successful device work that ingested prompt tokens.
    pub prompt_prefill: LlamaMetalWorkloadPhase,
    /// Successful generated tokens fed back through token-step device work.
    pub steady_decode: LlamaMetalWorkloadPhase,
}

/// Aggregate host-observed evidence for one workload phase.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMetalWorkloadPhase {
    pub token_count: usize,
    pub successful_invocation_count: usize,
    pub host_run_wall_time: Duration,
    pub host_synchronous_transaction_wall_time: Duration,
    pub kernel_launch_count: usize,
    /// Metal compute command buffers committed in this phase; host API H2D/D2H
    /// copy calls are counted separately and excluded.
    pub command_submission_count: usize,
    /// Metal compute command buffers synchronously waited in this phase; host
    /// API H2D/D2H copy calls are counted separately and excluded.
    pub command_wait_count: usize,
    pub transient_h2d_calls: usize,
    pub transient_h2d_bytes: usize,
    pub runtime_control_h2d_calls: usize,
    pub runtime_control_h2d_bytes: usize,
    pub retained_d2h_calls: usize,
    pub retained_d2h_bytes: usize,
}

impl LlamaMetalWorkloadPhase {
    pub(crate) fn from_reports(token_count: usize, reports: &[MetalDeviceRunReport]) -> Self {
        let mut phase = Self {
            token_count,
            successful_invocation_count: reports.len(),
            host_run_wall_time: Duration::ZERO,
            host_synchronous_transaction_wall_time: Duration::ZERO,
            kernel_launch_count: 0,
            command_submission_count: 0,
            command_wait_count: 0,
            transient_h2d_calls: 0,
            transient_h2d_bytes: 0,
            runtime_control_h2d_calls: 0,
            runtime_control_h2d_bytes: 0,
            retained_d2h_calls: 0,
            retained_d2h_bytes: 0,
        };
        for report in reports {
            phase.host_run_wall_time = phase
                .host_run_wall_time
                .saturating_add(report.run_wall_time);
            phase.host_synchronous_transaction_wall_time = phase
                .host_synchronous_transaction_wall_time
                .saturating_add(report.synchronous_transaction_wall_time);
            phase.kernel_launch_count = phase
                .kernel_launch_count
                .saturating_add(report.kernel_launch_count);
            phase.command_submission_count = phase
                .command_submission_count
                .saturating_add(report.command_submission_count);
            phase.command_wait_count = phase
                .command_wait_count
                .saturating_add(report.command_wait_count);
            phase.transient_h2d_calls = phase
                .transient_h2d_calls
                .saturating_add(report.transient_h2d_calls);
            phase.transient_h2d_bytes = phase
                .transient_h2d_bytes
                .saturating_add(report.transient_h2d_bytes);
            phase.runtime_control_h2d_calls = phase
                .runtime_control_h2d_calls
                .saturating_add(report.runtime_control_h2d_calls);
            phase.runtime_control_h2d_bytes = phase
                .runtime_control_h2d_bytes
                .saturating_add(report.runtime_control_h2d_bytes);
            phase.retained_d2h_calls = phase
                .retained_d2h_calls
                .saturating_add(report.retained_d2h_calls);
            phase.retained_d2h_bytes = phase
                .retained_d2h_bytes
                .saturating_add(report.retained_d2h_bytes);
        }
        phase
    }

    /// Host-observed tokens per second, never a GPU-throughput claim.
    pub fn host_tokens_per_second(&self) -> Option<f64> {
        (!self.host_run_wall_time.is_zero() && self.token_count != 0)
            .then(|| self.token_count as f64 / self.host_run_wall_time.as_secs_f64())
    }
}

/// Bounded provenance attached to a workload artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMetalWorkloadEvidenceContext {
    workload: String,
    revision: String,
    evidence: String,
}

impl LlamaMetalWorkloadEvidenceContext {
    pub fn new(
        workload: impl Into<String>,
        revision: impl Into<String>,
        evidence: impl Into<String>,
    ) -> Result<Self, LlamaMetalWorkloadEvidenceError> {
        let context = Self {
            workload: workload.into(),
            revision: revision.into(),
            evidence: evidence.into(),
        };
        validate_context_field("workload", &context.workload)?;
        validate_context_field("revision", &context.revision)?;
        validate_context_field("evidence", &context.evidence)?;
        Ok(context)
    }

    pub fn workload(&self) -> &str {
        &self.workload
    }
    pub fn revision(&self) -> &str {
        &self.revision
    }
    pub fn evidence(&self) -> &str {
        &self.evidence
    }
}

/// A validated, handle-free snapshot suitable for durable publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMetalWorkloadEvidenceArtifact {
    context: LlamaMetalWorkloadEvidenceContext,
    device: MetalDeviceInfo,
    evidence: LlamaMetalWorkloadEvidence,
}

impl LlamaMetalWorkloadEvidenceArtifact {
    pub fn new(
        context: LlamaMetalWorkloadEvidenceContext,
        device: MetalDeviceInfo,
        evidence: LlamaMetalWorkloadEvidence,
    ) -> Result<Self, LlamaMetalWorkloadEvidenceError> {
        let has_fixed_prefill = evidence.fixed_prefill_plan.is_some();
        if has_fixed_prefill != evidence.fixed_prefill_preparation.is_some()
            || has_fixed_prefill != evidence.fixed_prefill_deployment_identity.is_some()
        {
            return Err(LlamaMetalWorkloadEvidenceError::Inconsistent(
                "fixed-prefill identity, plan, and preparation must appear together",
            ));
        }
        let fallback_count = evidence.plan.fallback_count.saturating_add(
            evidence
                .fixed_prefill_plan
                .as_ref()
                .map_or(0, |plan| plan.fallback_count),
        );
        if fallback_count != 0 {
            return Err(LlamaMetalWorkloadEvidenceError::FallbackCount(
                fallback_count,
            ));
        }
        Ok(Self {
            context,
            device,
            evidence,
        })
    }

    pub fn context(&self) -> &LlamaMetalWorkloadEvidenceContext {
        &self.context
    }
    pub fn device(&self) -> &MetalDeviceInfo {
        &self.device
    }
    pub fn evidence(&self) -> &LlamaMetalWorkloadEvidence {
        &self.evidence
    }
    pub fn fallback_count(&self) -> usize {
        0
    }

    /// Encodes stable pretty JSON followed by exactly one newline.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, LlamaMetalWorkloadEvidenceError> {
        let mut bytes = serde_json::to_vec_pretty(&ArtifactJson::from(self))
            .map_err(|error| LlamaMetalWorkloadEvidenceError::Json(error.to_string()))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Creates a new artifact without replacing any existing filesystem entry.
    pub fn write_json_create_new(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), LlamaMetalWorkloadEvidenceError> {
        let bytes = self.to_json_bytes()?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.as_ref())
            .map_err(|source| LlamaMetalWorkloadEvidenceError::Io {
                operation: "create workload evidence",
                kind: source.kind(),
            })?;
        file.write_all(&bytes)
            .map_err(|source| LlamaMetalWorkloadEvidenceError::Io {
                operation: "write workload evidence",
                kind: source.kind(),
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaMetalWorkloadEvidenceError {
    InvalidContext {
        field: &'static str,
        reason: &'static str,
    },
    Inconsistent(&'static str),
    FallbackCount(usize),
    Json(String),
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
}

impl fmt::Display for LlamaMetalWorkloadEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContext { field, reason } => {
                write!(f, "invalid {field} context: {reason}")
            }
            Self::Inconsistent(reason) => write!(f, "inconsistent workload evidence: {reason}"),
            Self::FallbackCount(count) => {
                write!(f, "workload evidence contains {count} fallback paths")
            }
            Self::Json(reason) => write!(f, "workload evidence JSON encoding failed: {reason}"),
            Self::Io { operation, kind } => write!(f, "{operation} failed: {kind:?}"),
        }
    }
}

impl error::Error for LlamaMetalWorkloadEvidenceError {}

fn validate_context_field(
    field: &'static str,
    value: &str,
) -> Result<(), LlamaMetalWorkloadEvidenceError> {
    let reason = if value.is_empty() {
        Some("must be nonempty")
    } else if value.len() > MAX_CONTEXT_FIELD_BYTES {
        Some("exceeds 1024 UTF-8 bytes")
    } else if value.chars().any(char::is_control) {
        Some("contains a control character")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(LlamaMetalWorkloadEvidenceError::InvalidContext { field, reason }),
        None => Ok(()),
    }
}

#[derive(Serialize)]
struct ArtifactJson<'a> {
    format_version: u32,
    context: ContextJson<'a>,
    device: DeviceJson<'a>,
    evidence: EvidenceJson<'a>,
}

impl<'a> From<&'a LlamaMetalWorkloadEvidenceArtifact> for ArtifactJson<'a> {
    fn from(artifact: &'a LlamaMetalWorkloadEvidenceArtifact) -> Self {
        Self {
            format_version: LLAMA_METAL_WORKLOAD_EVIDENCE_FORMAT_VERSION,
            context: ContextJson {
                workload: artifact.context.workload(),
                revision: artifact.context.revision(),
                evidence: artifact.context.evidence(),
            },
            device: DeviceJson {
                name: &artifact.device.name,
                registry_id: artifact.device.registry_id,
                max_buffer_length: artifact.device.capabilities.max_buffer_length,
                unified_memory: artifact.device.capabilities.unified_memory,
                family: &artifact.device.capabilities.family,
            },
            evidence: EvidenceJson::from(&artifact.evidence),
        }
    }
}

#[derive(Serialize)]
struct ContextJson<'a> {
    workload: &'a str,
    revision: &'a str,
    evidence: &'a str,
}

#[derive(Serialize)]
struct DeviceJson<'a> {
    name: &'a str,
    registry_id: u64,
    max_buffer_length: usize,
    unified_memory: bool,
    family: &'a str,
}

#[derive(Serialize)]
struct EvidenceJson<'a> {
    token_step_deployment_identity: u64,
    fixed_prefill_deployment_identity: Option<u64>,
    token_step_plan: SummaryJson<'a>,
    fixed_prefill_plan: Option<SummaryJson<'a>>,
    token_step_preparation: PreparationJson,
    fixed_prefill_preparation: Option<PreparationJson>,
    first_successful_run: Option<PhaseJson>,
    prompt_prefill: PhaseJson,
    steady_decode: PhaseJson,
    fallback_count: usize,
}

impl<'a> From<&'a LlamaMetalWorkloadEvidence> for EvidenceJson<'a> {
    fn from(evidence: &'a LlamaMetalWorkloadEvidence) -> Self {
        Self {
            token_step_deployment_identity: evidence.token_step_deployment_identity,
            fixed_prefill_deployment_identity: evidence.fixed_prefill_deployment_identity,
            token_step_plan: SummaryJson::from(&evidence.plan),
            fixed_prefill_plan: evidence.fixed_prefill_plan.as_ref().map(SummaryJson::from),
            token_step_preparation: PreparationJson::from(&evidence.token_step_preparation),
            fixed_prefill_preparation: evidence
                .fixed_prefill_preparation
                .as_ref()
                .map(PreparationJson::from),
            first_successful_run: evidence.first_successful_run.as_ref().map(PhaseJson::from),
            prompt_prefill: PhaseJson::from(&evidence.prompt_prefill),
            steady_decode: PhaseJson::from(&evidence.steady_decode),
            fallback_count: 0,
        }
    }
}

#[derive(Serialize)]
struct SummaryJson<'a> {
    capture_identity: u64,
    resident_input_names: &'a [String],
    transient_input_names: &'a [String],
    runtime_control_input_names: &'a [String],
    requested_output_count: usize,
    constant_count: usize,
    constant_bytes: usize,
    quantized_constant_count: usize,
    quantized_constant_bytes: usize,
    resident_input_bytes: usize,
    transient_input_bytes: usize,
    runtime_control_input_bytes: usize,
    planned_slot_count: usize,
    planned_device_bytes: usize,
    zero_byte_sentinel_count: usize,
    nonzero_item_count: usize,
    zero_item_count: usize,
    rendered_cache_keys: &'a [String],
    fallback_count: usize,
    state_pair_count: usize,
    logical_state_bytes: usize,
    state_bank_count: usize,
    state_device_bytes: usize,
    append_state_row_bytes: usize,
    append_state_work_items: usize,
}

impl<'a> From<&'a MetalDeviceSessionSummary> for SummaryJson<'a> {
    fn from(value: &'a MetalDeviceSessionSummary) -> Self {
        Self {
            capture_identity: value.capture_identity,
            resident_input_names: &value.resident_input_names,
            transient_input_names: &value.transient_input_names,
            runtime_control_input_names: &value.runtime_control_input_names,
            requested_output_count: value.requested_output_count,
            constant_count: value.constant_count,
            constant_bytes: value.constant_bytes,
            quantized_constant_count: value.quantized_constant_count,
            quantized_constant_bytes: value.quantized_constant_bytes,
            resident_input_bytes: value.resident_input_bytes,
            transient_input_bytes: value.transient_input_bytes,
            runtime_control_input_bytes: value.runtime_control_input_bytes,
            planned_slot_count: value.planned_slot_count,
            planned_device_bytes: value.planned_device_bytes,
            zero_byte_sentinel_count: value.zero_byte_sentinel_count,
            nonzero_item_count: value.nonzero_item_count,
            zero_item_count: value.zero_item_count,
            rendered_cache_keys: &value.rendered_cache_keys,
            fallback_count: value.fallback_count,
            state_pair_count: value.state_pair_count,
            logical_state_bytes: value.logical_state_bytes,
            state_bank_count: value.state_bank_count,
            state_device_bytes: value.state_device_bytes,
            append_state_row_bytes: value.append_state_row_bytes,
            append_state_work_items: value.append_state_work_items,
        }
    }
}

#[derive(Serialize)]
struct PreparationJson {
    planning_wall_time: DurationJson,
    native_prepare_wall_time: DurationJson,
    cache_miss_pipeline_build_wall_time: DurationJson,
    initialization_upload_wall_time: DurationJson,
    pipeline_cache_request_count: usize,
    pipeline_cache_hit_count: usize,
    pipeline_cache_miss_count: usize,
    resident_h2d_calls: usize,
    resident_h2d_bytes: usize,
    initial_state_h2d_calls: usize,
    initial_state_h2d_bytes: usize,
}

impl From<&MetalDevicePreparationReport> for PreparationJson {
    fn from(value: &MetalDevicePreparationReport) -> Self {
        Self {
            planning_wall_time: value.planning_wall_time.into(),
            native_prepare_wall_time: value.native_prepare_wall_time.into(),
            cache_miss_pipeline_build_wall_time: value.cache_miss_pipeline_build_wall_time.into(),
            initialization_upload_wall_time: value.initialization_upload_wall_time.into(),
            pipeline_cache_request_count: value.pipeline_cache_request_count,
            pipeline_cache_hit_count: value.pipeline_cache_hit_count,
            pipeline_cache_miss_count: value.pipeline_cache_miss_count,
            resident_h2d_calls: value.resident_h2d_calls,
            resident_h2d_bytes: value.resident_h2d_bytes,
            initial_state_h2d_calls: value.initial_state_h2d_calls,
            initial_state_h2d_bytes: value.initial_state_h2d_bytes,
        }
    }
}

#[derive(Serialize)]
struct PhaseJson {
    token_count: usize,
    successful_invocation_count: usize,
    host_run_wall_time: DurationJson,
    host_synchronous_transaction_wall_time: DurationJson,
    kernel_launch_count: usize,
    command_submission_count: usize,
    command_wait_count: usize,
    transient_h2d_calls: usize,
    transient_h2d_bytes: usize,
    runtime_control_h2d_calls: usize,
    runtime_control_h2d_bytes: usize,
    retained_d2h_calls: usize,
    retained_d2h_bytes: usize,
}

impl From<&LlamaMetalWorkloadPhase> for PhaseJson {
    fn from(value: &LlamaMetalWorkloadPhase) -> Self {
        Self {
            token_count: value.token_count,
            successful_invocation_count: value.successful_invocation_count,
            host_run_wall_time: value.host_run_wall_time.into(),
            host_synchronous_transaction_wall_time: value
                .host_synchronous_transaction_wall_time
                .into(),
            kernel_launch_count: value.kernel_launch_count,
            command_submission_count: value.command_submission_count,
            command_wait_count: value.command_wait_count,
            transient_h2d_calls: value.transient_h2d_calls,
            transient_h2d_bytes: value.transient_h2d_bytes,
            runtime_control_h2d_calls: value.runtime_control_h2d_calls,
            runtime_control_h2d_bytes: value.runtime_control_h2d_bytes,
            retained_d2h_calls: value.retained_d2h_calls,
            retained_d2h_bytes: value.retained_d2h_bytes,
        }
    }
}

#[derive(Serialize)]
struct DurationJson {
    seconds: u64,
    nanoseconds: u32,
}

impl From<Duration> for DurationJson {
    fn from(value: Duration) -> Self {
        Self {
            seconds: value.as_secs(),
            nanoseconds: value.subsec_nanos(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::metal::MetalCapabilities;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn summary() -> MetalDeviceSessionSummary {
        MetalDeviceSessionSummary {
            capture_identity: 11,
            resident_input_names: vec!["weights".into()],
            transient_input_names: vec!["token".into()],
            runtime_control_input_names: vec!["position".into()],
            requested_output_count: 1,
            constant_count: 1,
            constant_bytes: 4,
            quantized_constant_count: 0,
            quantized_constant_bytes: 0,
            resident_input_bytes: 8,
            transient_input_bytes: 4,
            runtime_control_input_bytes: 4,
            planned_slot_count: 2,
            planned_device_bytes: 16,
            zero_byte_sentinel_count: 0,
            nonzero_item_count: 1,
            zero_item_count: 0,
            rendered_cache_keys: vec!["kernel".into()],
            fallback_count: 0,
            state_pair_count: 1,
            logical_state_bytes: 8,
            state_bank_count: 1,
            state_device_bytes: 8,
            append_state_row_bytes: 4,
            append_state_work_items: 1,
        }
    }

    fn preparation() -> MetalDevicePreparationReport {
        MetalDevicePreparationReport {
            planning_wall_time: Duration::new(1, 2),
            native_prepare_wall_time: Duration::new(3, 4),
            cache_miss_pipeline_build_wall_time: Duration::new(5, 6),
            initialization_upload_wall_time: Duration::new(7, 8),
            pipeline_cache_request_count: 1,
            pipeline_cache_hit_count: 0,
            pipeline_cache_miss_count: 1,
            resident_h2d_calls: 1,
            resident_h2d_bytes: 8,
            initial_state_h2d_calls: 1,
            initial_state_h2d_bytes: 8,
        }
    }

    fn phase(tokens: usize) -> LlamaMetalWorkloadPhase {
        LlamaMetalWorkloadPhase {
            token_count: tokens,
            successful_invocation_count: if tokens == 0 { 0 } else { 1 },
            host_run_wall_time: Duration::new(9, 10),
            host_synchronous_transaction_wall_time: Duration::new(11, 12),
            kernel_launch_count: 1,
            command_submission_count: 1,
            command_wait_count: 1,
            transient_h2d_calls: 1,
            transient_h2d_bytes: tokens * 4,
            runtime_control_h2d_calls: 1,
            runtime_control_h2d_bytes: tokens * 4,
            retained_d2h_calls: 1,
            retained_d2h_bytes: 16,
        }
    }

    fn artifact() -> LlamaMetalWorkloadEvidenceArtifact {
        LlamaMetalWorkloadEvidenceArtifact::new(
            LlamaMetalWorkloadEvidenceContext::new("workload", "revision", "evidence").unwrap(),
            MetalDeviceInfo {
                name: "Mock Metal".into(),
                registry_id: 7,
                capabilities: MetalCapabilities {
                    max_buffer_length: 1024,
                    unified_memory: true,
                    family: "Apple9".into(),
                },
            },
            LlamaMetalWorkloadEvidence {
                token_step_deployment_identity: 13,
                fixed_prefill_deployment_identity: None,
                plan: summary(),
                fixed_prefill_plan: None,
                token_step_preparation: preparation(),
                fixed_prefill_preparation: None,
                first_successful_run: Some(phase(1)),
                prompt_prefill: phase(2),
                steady_decode: phase(1),
            },
        )
        .unwrap()
    }

    #[test]
    fn artifact_json_is_deterministic_integer_timed_and_newline_terminated() {
        let artifact = artifact();
        let first = artifact.to_json_bytes().unwrap();
        let second = artifact.to_json_bytes().unwrap();
        let json: serde_json::Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.last(), Some(&b'\n'));
        assert_eq!(json["format_version"], 2);
        assert_eq!(json["device"]["registry_id"], 7);
        assert_eq!(json["evidence"]["token_step_deployment_identity"], 13);
        assert_eq!(
            json["evidence"]["token_step_preparation"]["planning_wall_time"]["seconds"],
            1
        );
        assert_eq!(json["evidence"]["fallback_count"], 0);
        assert_eq!(
            json["evidence"]["prompt_prefill"]["command_submission_count"],
            1
        );
        assert_eq!(json["evidence"]["prompt_prefill"]["command_wait_count"], 1);
    }

    #[test]
    fn context_and_create_new_fail_closed() {
        assert!(LlamaMetalWorkloadEvidenceContext::new("", "revision", "evidence").is_err());
        assert!(
            LlamaMetalWorkloadEvidenceContext::new("workload", "revision\n", "evidence").is_err()
        );
        let path = std::env::temp_dir().join(format!(
            "rustgrad-llama-workload-evidence-{}-{}.json",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ));
        let artifact = artifact();
        artifact.write_json_create_new(&path).unwrap();
        assert!(artifact.write_json_create_new(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fixed_prefill_identity_plan_and_preparation_are_one_unit() {
        let base = artifact();
        let mut incomplete = base.evidence().clone();
        incomplete.fixed_prefill_deployment_identity = Some(17);
        assert!(matches!(
            LlamaMetalWorkloadEvidenceArtifact::new(
                base.context().clone(),
                base.device().clone(),
                incomplete,
            ),
            Err(LlamaMetalWorkloadEvidenceError::Inconsistent(_))
        ));

        let mut complete = base.evidence().clone();
        complete.fixed_prefill_deployment_identity = Some(17);
        complete.fixed_prefill_plan = Some(summary());
        complete.fixed_prefill_preparation = Some(preparation());
        let fixed = LlamaMetalWorkloadEvidenceArtifact::new(
            base.context().clone(),
            base.device().clone(),
            complete,
        )
        .unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&fixed.to_json_bytes().unwrap()).unwrap();
        assert_eq!(json["evidence"]["fixed_prefill_deployment_identity"], 17);
    }
}
