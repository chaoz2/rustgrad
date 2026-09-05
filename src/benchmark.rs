//! Typed, deterministic evidence for comparing measured accelerator workloads.
//!
//! This module normalizes measurements; it does not run benchmarks or infer
//! unavailable values.  In particular, `None` means that a runtime did not
//! provide that measurement, while `Some(0)` is an observed zero.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt, time::Duration};

/// Current JSON schema emitted by [`BenchmarkObservation`] and
/// [`BenchmarkComparison`].
pub const BENCHMARK_FORMAT_VERSION: u32 = 1;

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
        input_shape: [usize; 4],
        input_dtype: String,
        input_sha256: String,
        correctness_contract: String,
    },
    /// One exact GGUF Llama prompt-to-token contract.
    GgufLlama {
        model_sha256: String,
        prompt_sha256: String,
        prompt_token_count: usize,
        max_new_tokens: usize,
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
    pub sample_count: usize,
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
    pub unit_count: usize,
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
    pub calls: usize,
    pub bytes: usize,
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
    pub planned_device_memory_bytes: Option<usize>,
    pub measured_peak_device_memory_bytes: Option<usize>,
    pub planned_kernel_count: Option<usize>,
    pub executed_kernel_count: Option<usize>,
    pub host_to_device: Option<BenchmarkTransfer>,
    pub device_to_host: Option<BenchmarkTransfer>,
    pub fallback_count: Option<usize>,
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
    WorkloadMismatch,
    DeviceMismatch,
    DuplicateFramework(BenchmarkFramework),
    MissingBaseline(BenchmarkFramework),
    NonCanonicalOrder,
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

fn units_per_second(count: usize, elapsed: BenchmarkDuration) -> Option<f64> {
    let nanos = elapsed.as_nanos().ok()?;
    (count != 0 && nanos != 0).then(|| count as f64 * 1_000_000_000.0 / nanos as f64)
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
