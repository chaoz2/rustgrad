//! Typed, deterministic evidence for comparing measured accelerator workloads.
//!
//! This module normalizes measurements; it does not run benchmarks or infer
//! unavailable values.  In particular, `None` means that a runtime did not
//! provide that measurement, while `Some(0)` is an observed zero.

use crate::{
    models::transformer::{
        LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION, LlamaMetalExecutionScoreboardReport,
        LlamaMetalScoreboardPhaseAggregate, LlamaMetalScoreboardProgram,
    },
    runtime::metal::{
        METAL_SESSION_SCOREBOARD_FORMAT_VERSION, MetalDeviceInfo, MetalHostWallTimeSummary,
        MetalSessionScoreboardReport,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt, time::Duration};

/// Current JSON schema emitted by [`BenchmarkObservation`] and
/// [`BenchmarkComparison`].
pub const BENCHMARK_FORMAT_VERSION: u32 = 1;

/// Exact workload label emitted by the maintained ResNet-18 Metal benchmark.
pub const RUSTGRAD_METAL_RESNET18_WORKLOAD: &str = "resnet18-eval-f32-1x3x224x224";
/// Exact workload label emitted by the maintained device-greedy GGUF Llama CLI.
pub const RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD: &str = "gguf-llama-metal-generate";

const MAX_LABEL_BYTES: usize = 1_024;
const MAX_COMMAND_BYTES: usize = 8_192;

/// Runtimes in the first supported accelerator comparison set.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub enum BenchmarkFramework {
    #[serde(rename = "rustgrad")]
    RustGrad,
    #[serde(rename = "tinygrad")]
    Tinygrad,
    #[serde(rename = "candle")]
    Candle,
    #[serde(rename = "llama.cpp")]
    LlamaCpp,
}

/// Exact implementation and invocation provenance for one observation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkImplementation {
    pub framework: BenchmarkFramework,
    pub version: String,
    pub revision: String,
    pub configuration: String,
    pub command: String,
}

/// Workload identity shared by every observation in one comparison.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BenchmarkWorkload {
    /// Fixed-shape ResNet-18 inference conformance workload.
    ResNet18 {
        model_identity: String,
        input_shape: [u64; 4],
        input_dtype: String,
        input_sha256: String,
        correctness_contract: String,
    },
    /// One exact GGUF Llama prompt-to-token contract.
    GgufLlama {
        model_sha256: String,
        prompt_sha256: String,
        prompt_token_count: u64,
        max_new_tokens: u64,
        expected_token_ids_sha256: String,
    },
}

/// Hardware identity that must match exactly across a comparison.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDevice {
    pub backend: String,
    pub name: String,
    pub hardware_identity: String,
    pub operating_system: String,
}

/// Integer duration representation used by the normalized wire format.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDuration {
    pub secs: u64,
    pub nanos: u32,
}

impl BenchmarkDuration {
    pub fn new(secs: u64, nanos: u32) -> Result<Self, BenchmarkError> {
        let value = Self { secs, nanos };
        value.validate()?;
        Ok(value)
    }

    pub fn from_duration(value: Duration) -> Self {
        Self {
            secs: value.as_secs(),
            nanos: value.subsec_nanos(),
        }
    }

    pub fn to_duration(self) -> Result<Duration, BenchmarkError> {
        self.validate()?;
        Ok(Duration::new(self.secs, self.nanos))
    }

    pub fn as_nanos(self) -> Result<u128, BenchmarkError> {
        Ok(self.to_duration()?.as_nanos())
    }

    fn validate(self) -> Result<(), BenchmarkError> {
        if self.nanos < 1_000_000_000 {
            Ok(())
        } else {
            Err(BenchmarkError::InvalidDuration)
        }
    }
}

/// Ordered integer summary for a measured steady-state sample set.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLatencySummary {
    pub sample_count: u64,
    pub min: BenchmarkDuration,
    pub nearest_rank_p50: BenchmarkDuration,
    pub nearest_rank_p95: BenchmarkDuration,
    pub max: BenchmarkDuration,
}

impl BenchmarkLatencySummary {
    fn validate(&self) -> Result<(), BenchmarkError> {
        self.min.validate()?;
        self.nearest_rank_p50.validate()?;
        self.nearest_rank_p95.validate()?;
        self.max.validate()?;
        if self.sample_count == 0
            || self.min > self.nearest_rank_p50
            || self.nearest_rank_p50 > self.nearest_rank_p95
            || self.nearest_rank_p95 > self.max
        {
            return Err(BenchmarkError::InvalidLatencySummary);
        }
        Ok(())
    }
}

/// Measured units and elapsed time for one workload phase.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPhase {
    pub unit_count: u64,
    pub host_wall_time: Option<BenchmarkDuration>,
    pub device_execution_time: Option<BenchmarkDuration>,
}

impl BenchmarkPhase {
    pub fn host_units_per_second(&self) -> Option<f64> {
        units_per_second(self.unit_count, self.host_wall_time?)
    }

    pub fn device_units_per_second(&self) -> Option<f64> {
        units_per_second(self.unit_count, self.device_execution_time?)
    }

    fn validate(&self) -> Result<(), BenchmarkError> {
        if self.unit_count == 0
            || (self.host_wall_time.is_none() && self.device_execution_time.is_none())
        {
            return Err(BenchmarkError::InvalidPhase);
        }
        if let Some(value) = self.host_wall_time {
            value.validate()?;
        }
        if let Some(value) = self.device_execution_time {
            value.validate()?;
        }
        Ok(())
    }
}

/// Measured host API transfer count and logical payload bytes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkTransfer {
    pub calls: u64,
    pub bytes: u64,
}

impl BenchmarkTransfer {
    fn validate(self) -> Result<(), BenchmarkError> {
        if self.calls == 0 && self.bytes != 0 {
            Err(BenchmarkError::InvalidTransfer)
        } else {
            Ok(())
        }
    }
}

/// Normalized metrics.  Optional fields are deliberately serialized as null.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkMetrics {
    pub planning_time: Option<BenchmarkDuration>,
    pub pipeline_compile_time: Option<BenchmarkDuration>,
    pub native_prepare_time: Option<BenchmarkDuration>,
    pub first_run_latency: Option<BenchmarkDuration>,
    pub steady_run_latency: Option<BenchmarkLatencySummary>,
    pub prompt_prefill: Option<BenchmarkPhase>,
    pub steady_decode: Option<BenchmarkPhase>,
    pub planned_device_memory_bytes: Option<u64>,
    pub measured_peak_device_memory_bytes: Option<u64>,
    pub planned_kernel_count: Option<u64>,
    pub executed_kernel_count: Option<u64>,
    pub host_to_device: Option<BenchmarkTransfer>,
    pub device_to_host: Option<BenchmarkTransfer>,
    pub fallback_count: Option<u64>,
}

impl BenchmarkMetrics {
    fn validate(&self) -> Result<(), BenchmarkError> {
        for value in [
            self.planning_time,
            self.pipeline_compile_time,
            self.native_prepare_time,
            self.first_run_latency,
        ]
        .into_iter()
        .flatten()
        {
            value.validate()?;
        }
        if let Some(value) = &self.steady_run_latency {
            value.validate()?;
        }
        if let Some(value) = &self.prompt_prefill {
            value.validate()?;
        }
        if let Some(value) = &self.steady_decode {
            value.validate()?;
        }
        if let Some(value) = self.host_to_device {
            value.validate()?;
        }
        if let Some(value) = self.device_to_host {
            value.validate()?;
        }
        if self.has_measurement() {
            Ok(())
        } else {
            Err(BenchmarkError::NoMeasurements)
        }
    }

    fn has_measurement(&self) -> bool {
        self.planning_time.is_some()
            || self.pipeline_compile_time.is_some()
            || self.native_prepare_time.is_some()
            || self.first_run_latency.is_some()
            || self.steady_run_latency.is_some()
            || self.prompt_prefill.is_some()
            || self.steady_decode.is_some()
            || self.planned_device_memory_bytes.is_some()
            || self.measured_peak_device_memory_bytes.is_some()
            || self.planned_kernel_count.is_some()
            || self.executed_kernel_count.is_some()
            || self.host_to_device.is_some()
            || self.device_to_host.is_some()
            || self.fallback_count.is_some()
    }
}

/// One measured implementation on one exact workload and device.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkObservation {
    pub format_version: u32,
    pub implementation: BenchmarkImplementation,
    pub workload: BenchmarkWorkload,
    pub device: BenchmarkDevice,
    pub metrics: BenchmarkMetrics,
}

impl BenchmarkObservation {
    pub fn new(
        implementation: BenchmarkImplementation,
        workload: BenchmarkWorkload,
        device: BenchmarkDevice,
        metrics: BenchmarkMetrics,
    ) -> Result<Self, BenchmarkError> {
        let value = Self {
            format_version: BENCHMARK_FORMAT_VERSION,
            implementation,
            workload,
            device,
            metrics,
        };
        value.validate()?;
        Ok(value)
    }

    /// Normalizes an in-memory single-session RustGrad Metal scoreboard.
    ///
    /// This adapter is intended for the ResNet-18 comparison workload. The
    /// caller retains responsibility for supplying the immutable workload,
    /// implementation, and operating-system provenance.
    pub fn from_metal_session_scoreboard(
        implementation: BenchmarkImplementation,
        workload: BenchmarkWorkload,
        operating_system: impl Into<String>,
        report: &MetalSessionScoreboardReport,
    ) -> Result<Self, BenchmarkError> {
        require_rustgrad(&implementation)?;
        if !matches!(&workload, BenchmarkWorkload::ResNet18 { .. }) {
            return Err(BenchmarkError::WorkloadMismatch);
        }
        validate_metal_source(report, &implementation, RUSTGRAD_METAL_RESNET18_WORKLOAD)?;
        let metrics = BenchmarkMetrics {
            planning_time: Some(BenchmarkDuration::from_duration(report.planning_wall_time)),
            pipeline_compile_time: Some(BenchmarkDuration::from_duration(
                report.cache_miss_pipeline_build_wall_time,
            )),
            native_prepare_time: Some(BenchmarkDuration::from_duration(
                report.native_prepare_wall_time,
            )),
            first_run_latency: report
                .first_run_host_wall_time
                .map(BenchmarkDuration::from_duration),
            steady_run_latency: metal_steady_latency(report)?,
            prompt_prefill: None,
            steady_decode: None,
            planned_device_memory_bytes: Some(count_to_u64(
                report.planned_physical_static_tensor_slot_bytes,
            )?),
            measured_peak_device_memory_bytes: None,
            planned_kernel_count: Some(count_to_u64(report.planned_kernel_count)?),
            executed_kernel_count: Some(count_to_u64(report.kernel_launch_count)?),
            host_to_device: Some(BenchmarkTransfer {
                calls: count_to_u64(report.host_api_h2d_calls)?,
                bytes: count_to_u64(report.host_api_h2d_bytes)?,
            }),
            device_to_host: Some(BenchmarkTransfer {
                calls: count_to_u64(report.host_api_d2h_calls)?,
                bytes: count_to_u64(report.host_api_d2h_bytes)?,
            }),
            fallback_count: Some(count_to_u64(report.fallback_count)?),
        };
        Self::new(
            implementation,
            workload,
            benchmark_metal_device(&report.device, operating_system.into()),
            metrics,
        )
    }

    /// Normalizes an in-memory ordered RustGrad Metal Llama scoreboard.
    ///
    /// Component preparation, transfer, and executed-launch counters are
    /// summed with overflow checks. Shared component ownership prevents the v2
    /// envelope from stating an exact whole-workload planned-memory or planned-
    /// kernel value, so those fields remain unavailable. The envelope also has
    /// no comparable global steady-run sample series.
    pub fn from_llama_metal_scoreboard(
        implementation: BenchmarkImplementation,
        workload: BenchmarkWorkload,
        operating_system: impl Into<String>,
        report: &LlamaMetalExecutionScoreboardReport,
    ) -> Result<Self, BenchmarkError> {
        require_rustgrad(&implementation)?;
        if !matches!(&workload, BenchmarkWorkload::GgufLlama { .. }) {
            return Err(BenchmarkError::WorkloadMismatch);
        }
        validate_llama_source(report, &implementation)?;
        let metrics = BenchmarkMetrics {
            planning_time: Some(BenchmarkDuration::from_duration(sum_component_duration(
                report,
                |component| component.planning_wall_time,
            )?)),
            pipeline_compile_time: Some(BenchmarkDuration::from_duration(sum_component_duration(
                report,
                |component| component.cache_miss_pipeline_build_wall_time,
            )?)),
            native_prepare_time: Some(BenchmarkDuration::from_duration(sum_component_duration(
                report,
                |component| component.native_prepare_wall_time,
            )?)),
            first_run_latency: llama_first_run_latency(report)?,
            steady_run_latency: None,
            prompt_prefill: phase_from_llama(&report.prompt_prefill)?,
            steady_decode: phase_from_llama(&report.steady_decode)?,
            planned_device_memory_bytes: None,
            measured_peak_device_memory_bytes: None,
            planned_kernel_count: None,
            executed_kernel_count: Some(sum_component_count(report, |component| {
                component.kernel_launch_count
            })?),
            host_to_device: Some(BenchmarkTransfer {
                calls: sum_component_count(report, |component| component.host_api_h2d_calls)?,
                bytes: sum_component_count(report, |component| component.host_api_h2d_bytes)?,
            }),
            device_to_host: Some(BenchmarkTransfer {
                calls: sum_component_count(report, |component| component.host_api_d2h_calls)?,
                bytes: sum_component_count(report, |component| component.host_api_d2h_bytes)?,
            }),
            fallback_count: Some(count_to_u64(report.fallback_count)?),
        };
        Self::new(
            implementation,
            workload,
            benchmark_metal_device(&report.token_step.device, operating_system.into()),
            metrics,
        )
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        let value = serde_json::from_slice::<Self>(bytes)
            .map_err(|error| BenchmarkError::Json(error.to_string()))?;
        value.validate()?;
        Ok(value)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        self.validate()?;
        encode_json(self)
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        validate_version(self.format_version)?;
        self.implementation.validate()?;
        self.workload.validate()?;
        self.device.validate()?;
        self.metrics.validate()
    }
}

/// Deterministic set of comparable observations and its explicit baseline.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkComparison {
    pub format_version: u32,
    pub baseline: BenchmarkFramework,
    pub workload: BenchmarkWorkload,
    pub device: BenchmarkDevice,
    pub observations: Vec<BenchmarkObservation>,
}

impl BenchmarkComparison {
    pub fn new(
        baseline: BenchmarkFramework,
        mut observations: Vec<BenchmarkObservation>,
    ) -> Result<Self, BenchmarkError> {
        observations.sort_by_key(|value| value.implementation.framework);
        let first = observations
            .first()
            .ok_or(BenchmarkError::TooFewObservations)?;
        let value = Self {
            format_version: BENCHMARK_FORMAT_VERSION,
            baseline,
            workload: first.workload.clone(),
            device: first.device.clone(),
            observations,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        let value = serde_json::from_slice::<Self>(bytes)
            .map_err(|error| BenchmarkError::Json(error.to_string()))?;
        value.validate()?;
        Ok(value)
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, BenchmarkError> {
        self.validate()?;
        encode_json(self)
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        validate_version(self.format_version)?;
        self.workload.validate()?;
        self.device.validate()?;
        if self.observations.len() < 2 {
            return Err(BenchmarkError::TooFewObservations);
        }
        let mut frameworks = BTreeSet::new();
        for observation in &self.observations {
            observation.validate()?;
            if observation.workload != self.workload {
                return Err(BenchmarkError::WorkloadMismatch);
            }
            if observation.device != self.device {
                return Err(BenchmarkError::DeviceMismatch);
            }
            if !frameworks.insert(observation.implementation.framework) {
                return Err(BenchmarkError::DuplicateFramework(
                    observation.implementation.framework,
                ));
            }
        }
        if !frameworks.contains(&self.baseline) {
            return Err(BenchmarkError::MissingBaseline(self.baseline));
        }
        if !self
            .observations
            .windows(2)
            .all(|pair| pair[0].implementation.framework < pair[1].implementation.framework)
        {
            return Err(BenchmarkError::NonCanonicalOrder);
        }
        Ok(())
    }
}

/// Validation or wire-format error for normalized benchmark evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BenchmarkError {
    UnsupportedFormatVersion(u32),
    InvalidMetadata(&'static str),
    InvalidSha256(&'static str),
    InvalidWorkload(&'static str),
    InvalidDuration,
    InvalidLatencySummary,
    InvalidPhase,
    InvalidTransfer,
    NoMeasurements,
    TooFewObservations,
    FrameworkMismatch,
    WorkloadMismatch,
    DeviceMismatch,
    DuplicateFramework(BenchmarkFramework),
    MissingBaseline(BenchmarkFramework),
    NonCanonicalOrder,
    InvalidSourceReport(&'static str),
    Overflow,
    Json(String),
}

impl fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFormatVersion(version) => {
                write!(f, "unsupported benchmark format version {version}")
            }
            Self::InvalidMetadata(field) => write!(f, "invalid benchmark metadata: {field}"),
            Self::InvalidSha256(field) => write!(f, "invalid lowercase SHA-256: {field}"),
            Self::InvalidWorkload(reason) => write!(f, "invalid benchmark workload: {reason}"),
            Self::InvalidDuration => f.write_str("invalid benchmark duration"),
            Self::InvalidLatencySummary => f.write_str("invalid benchmark latency summary"),
            Self::InvalidPhase => f.write_str("invalid benchmark phase"),
            Self::InvalidTransfer => f.write_str("invalid benchmark transfer counters"),
            Self::NoMeasurements => f.write_str("benchmark observation has no measurements"),
            Self::TooFewObservations => {
                f.write_str("benchmark comparison requires at least two observations")
            }
            Self::FrameworkMismatch => {
                f.write_str("RustGrad Metal adapters require the rustgrad framework")
            }
            Self::WorkloadMismatch => f.write_str("benchmark workloads do not match"),
            Self::DeviceMismatch => f.write_str("benchmark devices do not match"),
            Self::DuplicateFramework(framework) => {
                write!(f, "duplicate benchmark framework {framework:?}")
            }
            Self::MissingBaseline(framework) => {
                write!(f, "benchmark baseline {framework:?} is missing")
            }
            Self::NonCanonicalOrder => {
                f.write_str("benchmark observations are not canonically ordered")
            }
            Self::InvalidSourceReport(field) => {
                write!(f, "invalid benchmark source report: {field}")
            }
            Self::Overflow => f.write_str("benchmark source counter overflow"),
            Self::Json(error) => write!(f, "invalid benchmark JSON: {error}"),
        }
    }
}

impl std::error::Error for BenchmarkError {}

impl BenchmarkImplementation {
    fn validate(&self) -> Result<(), BenchmarkError> {
        validate_label("version", &self.version, MAX_LABEL_BYTES)?;
        validate_label("revision", &self.revision, MAX_LABEL_BYTES)?;
        validate_label("configuration", &self.configuration, MAX_LABEL_BYTES)?;
        validate_label("command", &self.command, MAX_COMMAND_BYTES)
    }
}

impl BenchmarkWorkload {
    fn validate(&self) -> Result<(), BenchmarkError> {
        match self {
            Self::ResNet18 {
                model_identity,
                input_shape,
                input_dtype,
                input_sha256,
                correctness_contract,
            } => {
                validate_label("model_identity", model_identity, MAX_LABEL_BYTES)?;
                validate_label("input_dtype", input_dtype, MAX_LABEL_BYTES)?;
                validate_sha256("input_sha256", input_sha256)?;
                validate_label(
                    "correctness_contract",
                    correctness_contract,
                    MAX_LABEL_BYTES,
                )?;
                if input_shape.contains(&0) {
                    return Err(BenchmarkError::InvalidWorkload(
                        "ResNet input extents must be nonzero",
                    ));
                }
            }
            Self::GgufLlama {
                model_sha256,
                prompt_sha256,
                prompt_token_count,
                max_new_tokens,
                expected_token_ids_sha256,
            } => {
                validate_sha256("model_sha256", model_sha256)?;
                validate_sha256("prompt_sha256", prompt_sha256)?;
                validate_sha256("expected_token_ids_sha256", expected_token_ids_sha256)?;
                if *prompt_token_count == 0 || *max_new_tokens == 0 {
                    return Err(BenchmarkError::InvalidWorkload(
                        "Llama prompt and token bound must be nonzero",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl BenchmarkDevice {
    fn validate(&self) -> Result<(), BenchmarkError> {
        validate_label("backend", &self.backend, MAX_LABEL_BYTES)?;
        validate_label("device name", &self.name, MAX_LABEL_BYTES)?;
        validate_label(
            "hardware_identity",
            &self.hardware_identity,
            MAX_LABEL_BYTES,
        )?;
        validate_label("operating_system", &self.operating_system, MAX_LABEL_BYTES)
    }
}

fn validate_version(version: u32) -> Result<(), BenchmarkError> {
    if version == BENCHMARK_FORMAT_VERSION {
        Ok(())
    } else {
        Err(BenchmarkError::UnsupportedFormatVersion(version))
    }
}

fn validate_label(
    field: &'static str,
    value: &str,
    max_bytes: usize,
) -> Result<(), BenchmarkError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        Err(BenchmarkError::InvalidMetadata(field))
    } else {
        Ok(())
    }
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), BenchmarkError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(BenchmarkError::InvalidSha256(field))
    }
}

fn units_per_second(count: u64, elapsed: BenchmarkDuration) -> Option<f64> {
    let nanos = elapsed.as_nanos().ok()?;
    (count != 0 && nanos != 0).then(|| count as f64 * 1_000_000_000.0 / nanos as f64)
}

fn require_rustgrad(implementation: &BenchmarkImplementation) -> Result<(), BenchmarkError> {
    if implementation.framework == BenchmarkFramework::RustGrad {
        Ok(())
    } else {
        Err(BenchmarkError::FrameworkMismatch)
    }
}

fn validate_metal_source(
    report: &MetalSessionScoreboardReport,
    implementation: &BenchmarkImplementation,
    expected_workload: &'static str,
) -> Result<(), BenchmarkError> {
    if report.format_version != METAL_SESSION_SCOREBOARD_FORMAT_VERSION {
        return Err(BenchmarkError::InvalidSourceReport(
            "Metal scoreboard format version",
        ));
    }
    if report.context.implementation_revision() != implementation.revision.as_str()
        || report.context.workload() != expected_workload
    {
        return Err(BenchmarkError::InvalidSourceReport(
            "workload or implementation revision",
        ));
    }
    let run_count =
        u64::try_from(report.successful_runs.len()).map_err(|_| BenchmarkError::Overflow)?;
    if report.successful_run_count != run_count
        || report.first_run_host_wall_time
            != report.successful_runs.first().map(|run| run.run_wall_time)
        || report.steady_run_host_wall_times
            != report
                .successful_runs
                .iter()
                .skip(1)
                .map(|run| run.run_wall_time)
                .collect::<Vec<_>>()
    {
        return Err(BenchmarkError::InvalidSourceReport(
            "successful run timing prefix",
        ));
    }
    Ok(())
}

fn validate_llama_source(
    report: &LlamaMetalExecutionScoreboardReport,
    implementation: &BenchmarkImplementation,
) -> Result<(), BenchmarkError> {
    if report.format_version != LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION {
        return Err(BenchmarkError::InvalidSourceReport(
            "Llama scoreboard format version",
        ));
    }
    validate_metal_source(
        &report.token_step,
        implementation,
        RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD,
    )?;
    if report.context.implementation_revision() != implementation.revision.as_str()
        || report.context != report.token_step.context
    {
        return Err(BenchmarkError::InvalidSourceReport("Llama context"));
    }
    if let Some(fixed_prefill) = &report.fixed_prefill {
        validate_metal_source(
            fixed_prefill,
            implementation,
            RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD,
        )?;
        if fixed_prefill.context != report.context
            || fixed_prefill.device != report.token_step.device
        {
            return Err(BenchmarkError::InvalidSourceReport(
                "Llama component identity",
            ));
        }
    }
    if report.successful_run_count
        != u64::try_from(report.successful_runs.len()).map_err(|_| BenchmarkError::Overflow)?
    {
        return Err(BenchmarkError::InvalidSourceReport(
            "Llama successful run count",
        ));
    }
    validate_empty_llama_phase(&report.standalone)?;
    Ok(())
}

fn validate_empty_llama_phase(
    phase: &LlamaMetalScoreboardPhaseAggregate,
) -> Result<(), BenchmarkError> {
    if phase.committed_token_count == 0
        && phase.successful_invocation_count == 0
        && phase.host_run_wall_time.is_zero()
        && phase.host_synchronous_transaction_wall_time.is_zero()
        && phase.gpu_command_execution_time.is_none()
        && phase.kernel_launch_count == 0
        && phase.command_submission_count == 0
        && phase.command_wait_count == 0
        && phase.transient_host_api_h2d_calls == 0
        && phase.transient_host_api_h2d_bytes == 0
        && phase.runtime_control_host_api_h2d_calls == 0
        && phase.runtime_control_host_api_h2d_bytes == 0
        && phase.retained_host_api_d2h_calls == 0
        && phase.retained_host_api_d2h_bytes == 0
    {
        Ok(())
    } else {
        Err(BenchmarkError::InvalidSourceReport(
            "Llama standalone phase is not empty",
        ))
    }
}

fn benchmark_metal_device(device: &MetalDeviceInfo, operating_system: String) -> BenchmarkDevice {
    BenchmarkDevice {
        backend: "metal".into(),
        name: device.name.clone(),
        hardware_identity: format!(
            "registry_id={};max_buffer_length={};unified_memory={};family={}",
            device.registry_id,
            device.capabilities.max_buffer_length,
            device.capabilities.unified_memory,
            device.capabilities.family,
        ),
        operating_system,
    }
}

fn metal_steady_latency(
    report: &MetalSessionScoreboardReport,
) -> Result<Option<BenchmarkLatencySummary>, BenchmarkError> {
    match (
        report.steady_run_host_wall_times.is_empty(),
        &report.steady_run_host_wall_time_summary,
    ) {
        (true, None) => Ok(None),
        (false, Some(summary)) => {
            let actual =
                latency_summary_from_metal(report.steady_run_host_wall_times.len(), summary)?;
            if Some(actual.clone()) != summarize_durations(&report.steady_run_host_wall_times) {
                return Err(BenchmarkError::InvalidSourceReport(
                    "steady run latency summary",
                ));
            }
            Ok(Some(actual))
        }
        _ => Err(BenchmarkError::InvalidSourceReport(
            "steady run latency summary",
        )),
    }
}

fn summarize_durations(samples: &[Duration]) -> Option<BenchmarkLatencySummary> {
    if samples.is_empty() {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let nearest_rank = |percentile: usize| {
        let rank =
            (ordered.len() / 100) * percentile + ((ordered.len() % 100) * percentile).div_ceil(100);
        ordered[rank.max(1) - 1]
    };
    Some(BenchmarkLatencySummary {
        sample_count: u64::try_from(samples.len()).ok()?,
        min: BenchmarkDuration::from_duration(ordered[0]),
        nearest_rank_p50: BenchmarkDuration::from_duration(nearest_rank(50)),
        nearest_rank_p95: BenchmarkDuration::from_duration(nearest_rank(95)),
        max: BenchmarkDuration::from_duration(ordered[ordered.len() - 1]),
    })
}

fn latency_summary_from_metal(
    sample_count: usize,
    summary: &MetalHostWallTimeSummary,
) -> Result<BenchmarkLatencySummary, BenchmarkError> {
    Ok(BenchmarkLatencySummary {
        sample_count: count_to_u64(sample_count)?,
        min: BenchmarkDuration::from_duration(summary.min),
        nearest_rank_p50: BenchmarkDuration::from_duration(summary.nearest_rank_p50),
        nearest_rank_p95: BenchmarkDuration::from_duration(summary.nearest_rank_p95),
        max: BenchmarkDuration::from_duration(summary.max),
    })
}

fn phase_from_llama(
    phase: &LlamaMetalScoreboardPhaseAggregate,
) -> Result<Option<BenchmarkPhase>, BenchmarkError> {
    if phase.committed_token_count == 0 {
        return Ok(None);
    }
    Ok(Some(BenchmarkPhase {
        unit_count: count_to_u64(phase.committed_token_count)?,
        host_wall_time: Some(BenchmarkDuration::from_duration(phase.host_run_wall_time)),
        device_execution_time: phase
            .gpu_command_execution_time
            .map(BenchmarkDuration::from_duration),
    }))
}

fn llama_first_run_latency(
    report: &LlamaMetalExecutionScoreboardReport,
) -> Result<Option<BenchmarkDuration>, BenchmarkError> {
    let Some(first) = report.successful_runs.first() else {
        return Ok(None);
    };
    if first.successful_invocation != 1 || !first.first_successful_run {
        return Err(BenchmarkError::InvalidSourceReport(
            "Llama first successful invocation",
        ));
    }
    let component = match first.program {
        LlamaMetalScoreboardProgram::TokenStep => &report.token_step,
        LlamaMetalScoreboardProgram::FixedPrefill => {
            report
                .fixed_prefill
                .as_ref()
                .ok_or(BenchmarkError::InvalidSourceReport(
                    "Llama first-run component",
                ))?
        }
    };
    let index = first
        .program_successful_invocation
        .checked_sub(1)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(BenchmarkError::InvalidSourceReport(
            "Llama first-run local ordinal",
        ))?;
    let run = component
        .successful_runs
        .get(index)
        .ok_or(BenchmarkError::InvalidSourceReport(
            "Llama first-run local record",
        ))?;
    if run.successful_invocation != first.program_successful_invocation {
        return Err(BenchmarkError::InvalidSourceReport(
            "Llama first-run local record",
        ));
    }
    Ok(Some(BenchmarkDuration::from_duration(run.run_wall_time)))
}

fn sum_component_duration(
    report: &LlamaMetalExecutionScoreboardReport,
    field: impl Fn(&MetalSessionScoreboardReport) -> Duration,
) -> Result<Duration, BenchmarkError> {
    match &report.fixed_prefill {
        Some(fixed_prefill) => field(&report.token_step)
            .checked_add(field(fixed_prefill))
            .ok_or(BenchmarkError::Overflow),
        None => Ok(field(&report.token_step)),
    }
}

fn sum_component_count(
    report: &LlamaMetalExecutionScoreboardReport,
    field: impl Fn(&MetalSessionScoreboardReport) -> usize,
) -> Result<u64, BenchmarkError> {
    match &report.fixed_prefill {
        Some(fixed_prefill) => count_to_u64(field(&report.token_step))?
            .checked_add(count_to_u64(field(fixed_prefill))?)
            .ok_or(BenchmarkError::Overflow),
        None => count_to_u64(field(&report.token_step)),
    }
}

fn count_to_u64(value: usize) -> Result<u64, BenchmarkError> {
    u64::try_from(value).map_err(|_| BenchmarkError::Overflow)
}

fn encode_json(value: &impl Serialize) -> Result<Vec<u8>, BenchmarkError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| BenchmarkError::Json(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        models::transformer::{
            LlamaMetalScoreboardInvocation, LlamaMetalScoreboardPhase,
            LlamaMetalScoreboardPhaseAggregate,
        },
        runtime::metal::{
            MetalCapabilities, MetalDeviceInfo, MetalScoreboardContext, MetalScoreboardRun,
            MetalScoreboardStatePolicy,
        },
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn implementation(framework: BenchmarkFramework) -> BenchmarkImplementation {
        BenchmarkImplementation {
            framework,
            version: "1.0".into(),
            revision: "revision".into(),
            configuration: "release; batch=1".into(),
            command: "benchmark --release".into(),
        }
    }

    fn device() -> BenchmarkDevice {
        BenchmarkDevice {
            backend: "metal".into(),
            name: "Apple GPU".into(),
            hardware_identity: "registry:42".into(),
            operating_system: "macOS fixture".into(),
        }
    }

    fn resnet_workload() -> BenchmarkWorkload {
        BenchmarkWorkload::ResNet18 {
            model_identity: "resnet18-seed-19".into(),
            input_shape: [1, 3, 224, 224],
            input_dtype: "f32".into(),
            input_sha256: sha('a'),
            correctness_contract: "full logits within declared tolerance".into(),
        }
    }

    fn llama_workload() -> BenchmarkWorkload {
        BenchmarkWorkload::GgufLlama {
            model_sha256: sha('b'),
            prompt_sha256: sha('c'),
            prompt_token_count: 4,
            max_new_tokens: 8,
            expected_token_ids_sha256: sha('d'),
        }
    }

    fn metrics() -> BenchmarkMetrics {
        BenchmarkMetrics {
            planning_time: Some(BenchmarkDuration::new(0, 10).unwrap()),
            pipeline_compile_time: None,
            native_prepare_time: Some(BenchmarkDuration::new(0, 20).unwrap()),
            first_run_latency: Some(BenchmarkDuration::new(0, 30).unwrap()),
            steady_run_latency: Some(BenchmarkLatencySummary {
                sample_count: 3,
                min: BenchmarkDuration::new(0, 4).unwrap(),
                nearest_rank_p50: BenchmarkDuration::new(0, 5).unwrap(),
                nearest_rank_p95: BenchmarkDuration::new(0, 6).unwrap(),
                max: BenchmarkDuration::new(0, 6).unwrap(),
            }),
            prompt_prefill: None,
            steady_decode: None,
            planned_device_memory_bytes: Some(1_024),
            measured_peak_device_memory_bytes: None,
            planned_kernel_count: Some(7),
            executed_kernel_count: Some(21),
            host_to_device: Some(BenchmarkTransfer {
                calls: 2,
                bytes: 16,
            }),
            device_to_host: Some(BenchmarkTransfer { calls: 1, bytes: 4 }),
            fallback_count: Some(0),
        }
    }

    fn observation(
        framework: BenchmarkFramework,
        workload: BenchmarkWorkload,
    ) -> BenchmarkObservation {
        BenchmarkObservation::new(implementation(framework), workload, device(), metrics()).unwrap()
    }

    fn metal_run(
        successful_invocation: u64,
        run_nanos: u32,
        gpu_nanos: Option<u32>,
    ) -> MetalScoreboardRun {
        MetalScoreboardRun {
            successful_invocation,
            first_successful_run: successful_invocation == 1,
            run_wall_time: Duration::new(0, run_nanos),
            synchronous_transaction_wall_time: Duration::new(0, run_nanos / 2),
            transient_host_api_h2d_calls: 1,
            transient_host_api_h2d_bytes: 4,
            runtime_control_host_api_h2d_calls: 0,
            runtime_control_host_api_h2d_bytes: 0,
            retained_host_api_d2h_calls: 1,
            retained_host_api_d2h_bytes: 4,
            kernel_launch_count: 2,
            command_submission_count: 1,
            command_wait_count: 1,
            gpu_command_execution_time: gpu_nanos.map(|nanos| Duration::new(0, nanos)),
            zero_item_count: 0,
            output_count: 1,
            committed_state_pair_count: 0,
            committed_state_bytes: 0,
            committed_state_work_items: 0,
            committed_state_position: None,
        }
    }

    fn metal_report(
        workload: &'static str,
        runs: Vec<MetalScoreboardRun>,
    ) -> MetalSessionScoreboardReport {
        let first_run_host_wall_time = runs.first().map(|run| run.run_wall_time);
        let steady_run_host_wall_times = runs
            .iter()
            .skip(1)
            .map(|run| run.run_wall_time)
            .collect::<Vec<_>>();
        let steady_run_host_wall_time_summary =
            (!steady_run_host_wall_times.is_empty()).then(|| MetalHostWallTimeSummary {
                min: *steady_run_host_wall_times.iter().min().unwrap(),
                nearest_rank_p50: steady_run_host_wall_times[0],
                nearest_rank_p95: *steady_run_host_wall_times.iter().max().unwrap(),
                max: *steady_run_host_wall_times.iter().max().unwrap(),
            });
        MetalSessionScoreboardReport {
            format_version: METAL_SESSION_SCOREBOARD_FORMAT_VERSION,
            context: MetalScoreboardContext::new(workload, "revision", "fixture evidence").unwrap(),
            deployment_identity: 1,
            capture_identity: 2,
            execution_plan_identity: 3,
            device: MetalDeviceInfo {
                name: "Apple Fixture GPU".into(),
                registry_id: 42,
                capabilities: MetalCapabilities {
                    max_buffer_length: 1 << 30,
                    unified_memory: true,
                    family: "apple9".into(),
                },
            },
            inputs: Vec::new(),
            state_policy: MetalScoreboardStatePolicy::Stateless,
            append_span_rows: 0,
            state_pair_count: 0,
            logical_state_bytes: 0,
            state_bank_count: 0,
            state_device_bytes: 0,
            append_state_row_bytes: 0,
            append_state_work_items: 0,
            rendered_cache_keys: vec!["fixture-kernel".into()],
            requested_output_count: 1,
            captured_constant_count: 1,
            captured_constant_bytes: 64,
            captured_quantized_constant_count: 0,
            captured_quantized_constant_bytes: 0,
            declared_resident_input_bytes: 64,
            declared_transient_input_bytes: 4,
            declared_runtime_control_input_bytes: 0,
            planning_wall_time: Duration::new(0, 11),
            native_prepare_wall_time: Duration::new(0, 13),
            cache_miss_pipeline_build_wall_time: Duration::new(0, 7),
            resident_upload_host_wall_time: Duration::new(0, 5),
            first_run_host_wall_time,
            steady_run_host_wall_times,
            steady_run_host_wall_time_summary,
            first_synchronous_transaction_host_wall_time: runs
                .first()
                .map(|run| run.synchronous_transaction_wall_time),
            steady_synchronous_transaction_host_wall_times: runs
                .iter()
                .skip(1)
                .map(|run| run.synchronous_transaction_wall_time)
                .collect(),
            steady_synchronous_transaction_host_wall_time_summary: None,
            logical_schedule_item_count: 2,
            peak_logical_temporary_allocation_count: 1,
            peak_logical_temporary_bytes: 512,
            planned_physical_static_tensor_slot_count: 2,
            planned_physical_static_tensor_slot_bytes: 1_024,
            planned_zero_byte_sentinel_count: 0,
            planned_kernel_count: 2,
            planned_zero_item_count: 0,
            pipeline_cache_request_count: 2,
            pipeline_cache_hit_count: 0,
            pipeline_cache_miss_count: 2,
            resident_host_api_h2d_calls: 1,
            resident_host_api_h2d_bytes: 64,
            initial_state_host_api_h2d_calls: 0,
            initial_state_host_api_h2d_bytes: 0,
            transient_host_api_h2d_calls: runs.len(),
            transient_host_api_h2d_bytes: runs.len() * 4,
            runtime_control_host_api_h2d_calls: 0,
            runtime_control_host_api_h2d_bytes: 0,
            retained_host_api_d2h_calls: runs.len(),
            retained_host_api_d2h_bytes: runs.len() * 4,
            host_api_h2d_calls: runs.len() + 1,
            host_api_h2d_bytes: runs.len() * 4 + 64,
            host_api_d2h_calls: runs.len(),
            host_api_d2h_bytes: runs.len() * 4,
            kernel_launch_count: runs.len() * 2,
            command_submission_count: runs.len(),
            command_wait_count: runs.len(),
            gpu_command_execution_time: runs
                .iter()
                .map(|run| run.gpu_command_execution_time)
                .try_fold(Duration::ZERO, |total, value| total.checked_add(value?)),
            zero_item_count: 0,
            committed_state_pair_count: 0,
            committed_state_bytes: 0,
            committed_state_work_items: 0,
            committed_state_position: None,
            successful_run_count: u64::try_from(runs.len()).unwrap(),
            successful_runs: runs,
            fallback_count: 0,
        }
    }

    fn llama_phase(
        token_count: usize,
        host_nanos: u32,
        gpu_nanos: Option<u32>,
    ) -> LlamaMetalScoreboardPhaseAggregate {
        LlamaMetalScoreboardPhaseAggregate {
            committed_token_count: token_count,
            successful_invocation_count: 1,
            host_run_wall_time: Duration::new(0, host_nanos),
            host_synchronous_transaction_wall_time: Duration::new(0, host_nanos / 2),
            gpu_command_execution_time: gpu_nanos.map(|nanos| Duration::new(0, nanos)),
            kernel_launch_count: 2,
            command_submission_count: 1,
            command_wait_count: 1,
            transient_host_api_h2d_calls: 1,
            transient_host_api_h2d_bytes: 4,
            runtime_control_host_api_h2d_calls: 0,
            runtime_control_host_api_h2d_bytes: 0,
            retained_host_api_d2h_calls: 1,
            retained_host_api_d2h_bytes: 4,
        }
    }

    fn empty_llama_phase() -> LlamaMetalScoreboardPhaseAggregate {
        LlamaMetalScoreboardPhaseAggregate {
            committed_token_count: 0,
            successful_invocation_count: 0,
            host_run_wall_time: Duration::ZERO,
            host_synchronous_transaction_wall_time: Duration::ZERO,
            gpu_command_execution_time: None,
            kernel_launch_count: 0,
            command_submission_count: 0,
            command_wait_count: 0,
            transient_host_api_h2d_calls: 0,
            transient_host_api_h2d_bytes: 0,
            runtime_control_host_api_h2d_calls: 0,
            runtime_control_host_api_h2d_bytes: 0,
            retained_host_api_d2h_calls: 0,
            retained_host_api_d2h_bytes: 0,
        }
    }

    #[test]
    fn resnet_metal_adapter_preserves_exact_session_measurements() {
        let report = metal_report(
            RUSTGRAD_METAL_RESNET18_WORKLOAD,
            vec![
                metal_run(1, 30, Some(12)),
                metal_run(2, 10, Some(4)),
                metal_run(3, 20, Some(8)),
            ],
        );
        let value = BenchmarkObservation::from_metal_session_scoreboard(
            implementation(BenchmarkFramework::RustGrad),
            resnet_workload(),
            "macOS fixture",
            &report,
        )
        .unwrap();
        assert_eq!(
            value.device.hardware_identity,
            "registry_id=42;max_buffer_length=1073741824;unified_memory=true;family=apple9"
        );
        assert_eq!(value.metrics.planning_time.unwrap().nanos, 11);
        assert_eq!(value.metrics.pipeline_compile_time.unwrap().nanos, 7);
        assert_eq!(value.metrics.native_prepare_time.unwrap().nanos, 13);
        assert_eq!(value.metrics.first_run_latency.unwrap().nanos, 30);
        assert_eq!(
            value.metrics.steady_run_latency.as_ref().unwrap(),
            &BenchmarkLatencySummary {
                sample_count: 2,
                min: BenchmarkDuration::new(0, 10).unwrap(),
                nearest_rank_p50: BenchmarkDuration::new(0, 10).unwrap(),
                nearest_rank_p95: BenchmarkDuration::new(0, 20).unwrap(),
                max: BenchmarkDuration::new(0, 20).unwrap(),
            }
        );
        assert_eq!(value.metrics.planned_device_memory_bytes, Some(1_024));
        assert_eq!(value.metrics.measured_peak_device_memory_bytes, None);
        assert_eq!(value.metrics.planned_kernel_count, Some(2));
        assert_eq!(value.metrics.executed_kernel_count, Some(6));
        assert_eq!(
            value.metrics.host_to_device,
            Some(BenchmarkTransfer {
                calls: 4,
                bytes: 76
            })
        );
        assert_eq!(value.metrics.fallback_count, Some(0));
    }

    #[test]
    fn llama_metal_adapter_sums_components_and_uses_global_first_run() {
        let mut token_step = metal_report(
            RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD,
            vec![metal_run(1, 70, Some(30))],
        );
        token_step.state_policy = MetalScoreboardStatePolicy::Append;
        token_step.append_span_rows = 1;
        token_step.planning_wall_time = Duration::new(0, 11);
        token_step.native_prepare_wall_time = Duration::new(0, 13);
        token_step.cache_miss_pipeline_build_wall_time = Duration::new(0, 7);
        token_step.kernel_launch_count = 5;
        token_step.host_api_h2d_calls = 3;
        token_step.host_api_h2d_bytes = 30;
        token_step.host_api_d2h_calls = 1;
        token_step.host_api_d2h_bytes = 4;

        let mut fixed_prefill = metal_report(
            RUSTGRAD_METAL_GGUF_LLAMA_WORKLOAD,
            vec![metal_run(1, 40, Some(20))],
        );
        fixed_prefill.state_policy = MetalScoreboardStatePolicy::Append;
        fixed_prefill.append_span_rows = 4;
        fixed_prefill.planning_wall_time = Duration::new(0, 17);
        fixed_prefill.native_prepare_wall_time = Duration::new(0, 19);
        fixed_prefill.cache_miss_pipeline_build_wall_time = Duration::new(0, 5);
        fixed_prefill.kernel_launch_count = 7;
        fixed_prefill.host_api_h2d_calls = 2;
        fixed_prefill.host_api_h2d_bytes = 20;
        fixed_prefill.host_api_d2h_calls = 0;
        fixed_prefill.host_api_d2h_bytes = 0;

        let report = LlamaMetalExecutionScoreboardReport {
            format_version: LLAMA_METAL_EXECUTION_SCOREBOARD_FORMAT_VERSION,
            context: token_step.context.clone(),
            successful_runs: vec![
                LlamaMetalScoreboardInvocation {
                    successful_invocation: 1,
                    first_successful_run: true,
                    program: LlamaMetalScoreboardProgram::FixedPrefill,
                    phase: LlamaMetalScoreboardPhase::PromptPrefill,
                    program_successful_invocation: 1,
                    append_span_rows: 4,
                    committed_state_pair_count: 0,
                    committed_state_bytes: 0,
                    committed_state_work_items: 0,
                    committed_state_position: 4,
                },
                LlamaMetalScoreboardInvocation {
                    successful_invocation: 2,
                    first_successful_run: false,
                    program: LlamaMetalScoreboardProgram::TokenStep,
                    phase: LlamaMetalScoreboardPhase::SteadyDecode,
                    program_successful_invocation: 1,
                    append_span_rows: 1,
                    committed_state_pair_count: 0,
                    committed_state_bytes: 0,
                    committed_state_work_items: 0,
                    committed_state_position: 5,
                },
            ],
            successful_run_count: 2,
            prompt_prefill: llama_phase(4, 40, Some(20)),
            steady_decode: llama_phase(1, 70, None),
            standalone: empty_llama_phase(),
            committed_state_position: 5,
            committed_state_pair_count: 0,
            committed_state_bytes: 0,
            committed_state_work_items: 0,
            fallback_count: 0,
            token_step,
            fixed_prefill: Some(fixed_prefill),
        };
        let value = BenchmarkObservation::from_llama_metal_scoreboard(
            implementation(BenchmarkFramework::RustGrad),
            llama_workload(),
            "macOS fixture",
            &report,
        )
        .unwrap();
        assert_eq!(value.metrics.planning_time.unwrap().nanos, 28);
        assert_eq!(value.metrics.pipeline_compile_time.unwrap().nanos, 12);
        assert_eq!(value.metrics.native_prepare_time.unwrap().nanos, 32);
        assert_eq!(value.metrics.first_run_latency.unwrap().nanos, 40);
        assert_eq!(value.metrics.steady_run_latency, None);
        assert_eq!(value.metrics.planned_device_memory_bytes, None);
        assert_eq!(value.metrics.measured_peak_device_memory_bytes, None);
        assert_eq!(value.metrics.planned_kernel_count, None);
        assert_eq!(value.metrics.executed_kernel_count, Some(12));
        assert_eq!(
            value.metrics.host_to_device,
            Some(BenchmarkTransfer {
                calls: 5,
                bytes: 50
            })
        );
        assert_eq!(
            value.metrics.device_to_host,
            Some(BenchmarkTransfer { calls: 1, bytes: 4 })
        );
        assert_eq!(
            value
                .metrics
                .prompt_prefill
                .as_ref()
                .unwrap()
                .device_execution_time
                .unwrap()
                .nanos,
            20
        );
        assert_eq!(
            value
                .metrics
                .steady_decode
                .as_ref()
                .unwrap()
                .device_execution_time,
            None
        );

        let mut nonempty_standalone = report.clone();
        nonempty_standalone.standalone = llama_phase(1, 1, None);
        assert_eq!(
            BenchmarkObservation::from_llama_metal_scoreboard(
                implementation(BenchmarkFramework::RustGrad),
                llama_workload(),
                "macOS fixture",
                &nonempty_standalone,
            ),
            Err(BenchmarkError::InvalidSourceReport(
                "Llama standalone phase is not empty"
            ))
        );
    }

    #[test]
    fn metal_adapters_reject_wrong_framework_and_workload() {
        let report = metal_report(
            RUSTGRAD_METAL_RESNET18_WORKLOAD,
            vec![metal_run(1, 30, Some(12))],
        );
        assert_eq!(
            BenchmarkObservation::from_metal_session_scoreboard(
                implementation(BenchmarkFramework::Candle),
                resnet_workload(),
                "macOS fixture",
                &report,
            ),
            Err(BenchmarkError::FrameworkMismatch)
        );
        assert_eq!(
            BenchmarkObservation::from_metal_session_scoreboard(
                implementation(BenchmarkFramework::RustGrad),
                llama_workload(),
                "macOS fixture",
                &report,
            ),
            Err(BenchmarkError::WorkloadMismatch)
        );

        let relabeled = metal_report("linear-1x4", vec![metal_run(1, 30, Some(12))]);
        assert_eq!(
            BenchmarkObservation::from_metal_session_scoreboard(
                implementation(BenchmarkFramework::RustGrad),
                resnet_workload(),
                "macOS fixture",
                &relabeled,
            ),
            Err(BenchmarkError::InvalidSourceReport(
                "workload or implementation revision"
            ))
        );
    }

    #[test]
    fn observation_preserves_unavailable_metrics_and_round_trips() {
        let value = observation(BenchmarkFramework::RustGrad, resnet_workload());
        let bytes = value.to_json_bytes().unwrap();
        assert_eq!(bytes, value.to_json_bytes().unwrap());
        assert_eq!(
            BenchmarkObservation::from_json_bytes(&bytes).unwrap(),
            value
        );
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["metrics"]["pipeline_compile_time"].is_null());
        assert!(json["metrics"]["measured_peak_device_memory_bytes"].is_null());
    }

    #[test]
    fn comparison_is_canonical_and_requires_exact_identity() {
        let rustgrad = observation(BenchmarkFramework::RustGrad, resnet_workload());
        let candle = observation(BenchmarkFramework::Candle, resnet_workload());
        let comparison = BenchmarkComparison::new(
            BenchmarkFramework::RustGrad,
            vec![candle.clone(), rustgrad.clone()],
        )
        .unwrap();
        assert_eq!(
            comparison
                .observations
                .iter()
                .map(|value| value.implementation.framework)
                .collect::<Vec<_>>(),
            [BenchmarkFramework::RustGrad, BenchmarkFramework::Candle]
        );
        let bytes = comparison.to_json_bytes().unwrap();
        assert_eq!(
            BenchmarkComparison::from_json_bytes(&bytes).unwrap(),
            comparison
        );

        let mut wrong_device = candle.clone();
        wrong_device.device.hardware_identity = "registry:7".into();
        assert_eq!(
            BenchmarkComparison::new(
                BenchmarkFramework::RustGrad,
                vec![rustgrad.clone(), wrong_device]
            ),
            Err(BenchmarkError::DeviceMismatch)
        );
        assert_eq!(
            BenchmarkComparison::new(
                BenchmarkFramework::RustGrad,
                vec![
                    rustgrad.clone(),
                    observation(BenchmarkFramework::Candle, llama_workload())
                ]
            ),
            Err(BenchmarkError::WorkloadMismatch)
        );
        assert_eq!(
            BenchmarkComparison::new(
                BenchmarkFramework::RustGrad,
                vec![rustgrad.clone(), rustgrad.clone()]
            ),
            Err(BenchmarkError::DuplicateFramework(
                BenchmarkFramework::RustGrad
            ))
        );
        assert_eq!(
            BenchmarkComparison::new(BenchmarkFramework::Tinygrad, vec![rustgrad, candle]),
            Err(BenchmarkError::MissingBaseline(
                BenchmarkFramework::Tinygrad
            ))
        );
    }

    #[test]
    fn validation_rejects_malformed_measurements_and_unknown_json() {
        assert_eq!(
            BenchmarkDuration::new(0, 1_000_000_000),
            Err(BenchmarkError::InvalidDuration)
        );
        let mut bad = metrics();
        bad.steady_run_latency.as_mut().unwrap().sample_count = 0;
        assert_eq!(
            BenchmarkObservation::new(
                implementation(BenchmarkFramework::RustGrad),
                resnet_workload(),
                device(),
                bad,
            ),
            Err(BenchmarkError::InvalidLatencySummary)
        );
        let mut bad = metrics();
        bad.host_to_device = Some(BenchmarkTransfer { calls: 0, bytes: 1 });
        assert_eq!(
            BenchmarkObservation::new(
                implementation(BenchmarkFramework::RustGrad),
                resnet_workload(),
                device(),
                bad,
            ),
            Err(BenchmarkError::InvalidTransfer)
        );

        let valid = observation(BenchmarkFramework::RustGrad, resnet_workload());
        let mut json = serde_json::to_value(valid).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("unknown".into(), serde_json::Value::Bool(true));
        assert!(matches!(
            BenchmarkObservation::from_json_bytes(&serde_json::to_vec(&json).unwrap()),
            Err(BenchmarkError::Json(_))
        ));
    }

    #[test]
    fn llama_phases_have_explicit_host_and_device_rates() {
        let mut measured = metrics();
        measured.prompt_prefill = Some(BenchmarkPhase {
            unit_count: 8,
            host_wall_time: Some(BenchmarkDuration::new(2, 0).unwrap()),
            device_execution_time: None,
        });
        measured.steady_decode = Some(BenchmarkPhase {
            unit_count: 4,
            host_wall_time: Some(BenchmarkDuration::new(1, 0).unwrap()),
            device_execution_time: Some(BenchmarkDuration::new(0, 500_000_000).unwrap()),
        });
        let value = BenchmarkObservation::new(
            implementation(BenchmarkFramework::LlamaCpp),
            llama_workload(),
            device(),
            measured,
        )
        .unwrap();
        assert_eq!(
            value
                .metrics
                .prompt_prefill
                .as_ref()
                .unwrap()
                .host_units_per_second(),
            Some(4.0)
        );
        assert_eq!(
            value
                .metrics
                .steady_decode
                .as_ref()
                .unwrap()
                .device_units_per_second(),
            Some(8.0)
        );
    }
}
