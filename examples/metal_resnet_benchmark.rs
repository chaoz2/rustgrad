//! Live, exact-revision ResNet-18 evidence for one persistent Metal session.

use rustgrad::nn::{ResNet, ResNetConfig, ResNetMetalPlan};
use rustgrad::runtime::metal::{
    MetalDiscovery, MetalPlanOptions, MetalRuntime, MetalScoreboardContext, MetalSessionScoreboard,
};
use rustgrad::{
    Backend, BenchmarkFramework, BenchmarkImplementation, BenchmarkObservation, BenchmarkWorkload,
    CpuBackend, DType, MetalDeviceBufferMeasurement, Module, RUSTGRAD_METAL_RESNET18_WORKLOAD,
    Storage, TensorData,
};
use std::{
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

const WORKLOAD: &str = RUSTGRAD_METAL_RESNET18_WORKLOAD;
const EVIDENCE: &str = "live self-hosted Apple GPU benchmark";
const MODEL_IDENTITY: &str = "rustgrad-resnet18-default-seed-19";
const CORRECTNESS_CONTRACT: &str =
    "complete CPU-oracle logits under 5e-4 * max(abs(expected), 1) tolerance";
const COMMAND: &str = "cargo run --release --example metal_resnet_benchmark";
const DEFAULT_RUNS: usize = 10;
const MAX_RUNS: usize = 1_000;
const MAX_OS_LABEL_BYTES: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
struct EvidenceConfig {
    revision: String,
    scoreboard_path: PathBuf,
    observation_path: PathBuf,
    operating_system: String,
    runs: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let evidence = EvidenceConfig::from_env()?;
    validate_evidence_paths(&evidence)?;

    let runtime = MetalRuntime::load()?;
    let device = match runtime.discover()? {
        MetalDiscovery::Devices(mut devices) if !devices.is_empty() => devices.remove(0),
        MetalDiscovery::Devices(_) | MetalDiscovery::NoDevices => {
            return Err(
                io::Error::other("benchmark requires a process-visible Metal device").into(),
            );
        }
    };
    let device_buffer_measurement = MetalDeviceBufferMeasurement::begin(&device)?;
    let model = ResNet::new_static(ResNetConfig::default(), 19)?;
    let plan =
        ResNetMetalPlan::eval_f32(&model, &device, [1, 3, 224, 224], MetalPlanOptions::new(64))?;
    let image = benchmark_image()?;
    let mut oracle_bindings = model.input_bindings(plan.graph())?;
    oracle_bindings.insert("image".into(), image.clone());
    let expected = CpuBackend.execute(plan.graph(), plan.logits_node(), &oracle_bindings)?;
    ensure_finite_f32_logits(&expected)?;
    let logical_schedule_item_count = plan.execution_plan().schedule_item_count;
    let peak_logical_temporary_allocation_count = plan.execution_plan().peak_logical_allocations;
    let peak_logical_temporary_bytes = plan.execution_plan().peak_logical_bytes;
    let stable_summary = plan.summary().clone();
    let stable_resident_inputs = plan.resident_inputs().to_vec();
    let stable_cache_keys = stable_summary.rendered_cache_keys.clone();

    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new(WORKLOAD, &evidence.revision, EVIDENCE)?,
        plan.metal_plan(),
    );
    let cache = device.cache();
    if !cache.is_empty() {
        return Err(io::Error::other("new benchmark device cache is not empty").into());
    }
    let mut session = plan.prepare()?;
    scoreboard.bind(session.metal_session())?;
    for _ in 0..evidence.runs {
        let run = session.run(image.clone())?;
        compare_logits(run.logits(), &expected)?;
        if session.metal_session().summary() != &stable_summary
            || session.metal_session().resident_inputs() != stable_resident_inputs
            || session
                .metal_session()
                .compiled_kernels()
                .map(|kernel| kernel.cache_key.as_str())
                .ne(stable_cache_keys.iter().map(String::as_str))
        {
            return Err(io::Error::other("persistent Metal ownership changed during runs").into());
        }
        scoreboard.record(run.metal_run())?;
    }
    let report = scoreboard.report()?;
    if report.successful_run_count != u64::try_from(evidence.runs)?
        || report.fallback_count != 0
        || report.successful_runs.len() != evidence.runs
        || report.logical_schedule_item_count != logical_schedule_item_count
        || report.peak_logical_temporary_allocation_count != peak_logical_temporary_allocation_count
        || report.peak_logical_temporary_bytes != peak_logical_temporary_bytes
        || report.pipeline_cache_miss_count != cache.len()
        || report.cache_miss_pipeline_build_wall_time > report.native_prepare_wall_time
        || report.transient_host_api_h2d_calls
            != report
                .successful_runs
                .iter()
                .map(|run| run.transient_host_api_h2d_calls)
                .sum::<usize>()
        || report.retained_host_api_d2h_bytes
            != report
                .successful_runs
                .iter()
                .map(|run| run.retained_host_api_d2h_bytes)
                .sum::<usize>()
    {
        return Err(io::Error::other("benchmark report is incomplete").into());
    }
    let measured_device_buffer_peak = device_buffer_measurement.finish(&device)?;
    let observation = BenchmarkObservation::from_metal_session_scoreboard(
        benchmark_implementation(&evidence),
        benchmark_workload(),
        evidence.operating_system.clone(),
        &report,
    )?;
    let planned_device_memory_bytes = observation
        .metrics
        .planned_device_memory_bytes
        .ok_or_else(|| io::Error::other("normalized ResNet observation has no planned memory"))?;
    let observation = observation
        .with_rustgrad_device_buffer_peak(measured_device_buffer_peak)
        .map_err(|error| io::Error::other(format!("benchmark observation: {error}")))?;
    if observation.metrics.measured_peak_device_memory_bytes
        != Some(measured_device_buffer_peak.bytes())
        || observation.metrics.planned_device_memory_bytes != Some(planned_device_memory_bytes)
    {
        return Err(io::Error::other(
            "normalized ResNet observation lost measured or planned memory",
        )
        .into());
    }
    let scoreboard_json = report.to_json_bytes()?;
    let observation_json = deterministic_observation_json(&observation)?;
    write_new_evidence(&evidence.scoreboard_path, &scoreboard_json)?;
    write_new_evidence(&evidence.observation_path, &observation_json)?;
    Ok(())
}

impl EvidenceConfig {
    fn from_env() -> Result<Self, io::Error> {
        let runs = match env::var("RUSTGRAD_METAL_RESNET_RUNS") {
            Ok(value) => Some(value),
            Err(env::VarError::NotPresent) => None,
            Err(env::VarError::NotUnicode(_)) => {
                return Err(io::Error::other("RUSTGRAD_METAL_RESNET_RUNS must be UTF-8"));
            }
        };
        Self::new(
            required_env("RUSTGRAD_METAL_EXPECTED_SHA")?,
            PathBuf::from(required_env("RUSTGRAD_METAL_RESNET_SCOREBOARD_PATH")?),
            PathBuf::from(required_env("RUSTGRAD_METAL_RESNET_OBSERVATION_PATH")?),
            required_env("RUSTGRAD_METAL_RESNET_OPERATING_SYSTEM")?,
            runs.as_deref(),
        )
    }

    fn new(
        revision: String,
        scoreboard_path: PathBuf,
        observation_path: PathBuf,
        operating_system: String,
        runs: Option<&str>,
    ) -> Result<Self, io::Error> {
        validate_lower_hex("RUSTGRAD_METAL_EXPECTED_SHA", &revision, 40)?;
        if operating_system.is_empty()
            || operating_system.len() > MAX_OS_LABEL_BYTES
            || operating_system.chars().any(char::is_control)
        {
            return Err(io::Error::other(
                "RUSTGRAD_METAL_RESNET_OPERATING_SYSTEM must be a bounded single-line label",
            ));
        }
        Ok(Self {
            revision,
            scoreboard_path,
            observation_path,
            operating_system,
            runs: benchmark_runs(runs)?,
        })
    }
}

fn required_env(name: &'static str) -> Result<String, io::Error> {
    env::var(name).map_err(|_| io::Error::other(format!("missing {name}")))
}

fn validate_lower_hex(name: &'static str, value: &str, length: usize) -> Result<(), io::Error> {
    if value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{name} must be {length} lowercase hexadecimal characters"
        )))
    }
}

fn benchmark_runs(value: Option<&str>) -> Result<usize, io::Error> {
    let runs = match value {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| io::Error::other("RUSTGRAD_METAL_RESNET_RUNS must be an integer"))?,
        None => DEFAULT_RUNS,
    };
    if (3..=MAX_RUNS).contains(&runs) {
        Ok(runs)
    } else {
        Err(io::Error::other(
            "RUSTGRAD_METAL_RESNET_RUNS must be between 3 and 1000",
        ))
    }
}

fn benchmark_implementation(evidence: &EvidenceConfig) -> BenchmarkImplementation {
    BenchmarkImplementation {
        framework: BenchmarkFramework::RustGrad,
        version: env!("CARGO_PKG_VERSION").into(),
        revision: evidence.revision.clone(),
        configuration: format!(
            "release;eval_f32;batch=1;local_size=64;runs={}",
            evidence.runs
        ),
        command: COMMAND.into(),
    }
}

fn benchmark_workload() -> BenchmarkWorkload {
    BenchmarkWorkload::ResNet18 {
        model_identity: MODEL_IDENTITY.into(),
        input_shape: [1, 3, 224, 224],
        input_dtype: "f32".into(),
        input_sha256: BENCHMARK_IMAGE_RAW_LE_F32_SHA256.into(),
        correctness_contract: CORRECTNESS_CONTRACT.into(),
    }
}

fn deterministic_observation_json(
    observation: &BenchmarkObservation,
) -> Result<Vec<u8>, io::Error> {
    let json = observation
        .to_json_bytes()
        .map_err(|error| io::Error::other(format!("benchmark observation: {error}")))?;
    if observation
        .to_json_bytes()
        .map_err(|error| io::Error::other(format!("benchmark observation: {error}")))?
        != json
    {
        return Err(io::Error::other(
            "benchmark observation serialization is not deterministic",
        ));
    }
    Ok(json)
}

fn validate_evidence_paths(evidence: &EvidenceConfig) -> Result<(), io::Error> {
    let scoreboard = resolved_new_destination(&evidence.scoreboard_path, "scoreboard")?;
    let observation =
        resolved_new_destination(&evidence.observation_path, "benchmark observation")?;
    if scoreboard == observation {
        return Err(io::Error::other("evidence output paths must be distinct"));
    }
    Ok(())
}

fn resolved_new_destination(path: &Path, label: &str) -> Result<PathBuf, io::Error> {
    match fs::symlink_metadata(path) {
        Ok(_) => return Err(io::Error::other(format!("{label} path already exists"))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let filename = path
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| io::Error::other(format!("{label} path requires a UTF-8 filename")))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("{label} path requires an existing parent: {error}"),
        )
    })?;
    Ok(parent.join(filename))
}

fn write_new_evidence(path: &Path, bytes: &[u8]) -> Result<(), io::Error> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)
}

/// SHA-256 of the exact raw little-endian F32 payload produced by
/// [`benchmark_image`]: `3 * 224 * 224` lanes whose value is
/// `(((index * 17) % 257) - 128) / 1024`.
const BENCHMARK_IMAGE_RAW_LE_F32_SHA256: &str =
    "ec1e5c2285cbad8428cd135b829f9a216750d7ddc1b65de8b05eac114ed91b13";

fn benchmark_image() -> Result<TensorData, rustgrad::Error> {
    let values = (0usize..3 * 224 * 224)
        .map(|index| {
            let lane = index.wrapping_mul(17) % 257;
            (lane as f32 - 128.0) / 1024.0
        })
        .collect::<Vec<_>>();
    TensorData::new([1, 3, 224, 224], values)
}

fn ensure_finite_f32_logits(logits: &TensorData) -> Result<(), io::Error> {
    let Storage::F32(values) = logits.storage() else {
        return Err(io::Error::other("CPU oracle logits are not F32"));
    };
    if logits.shape().dims() != [1, 1_000] || values.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::other(
            "CPU oracle logits have an invalid shape or nonfinite lane",
        ));
    }
    Ok(())
}

fn compare_logits(actual: &TensorData, expected: &TensorData) -> Result<(), io::Error> {
    if actual.shape() != expected.shape() || actual.dtype() != DType::F32 {
        return Err(io::Error::other("Metal logits descriptor mismatch"));
    }
    let (Storage::F32(actual), Storage::F32(expected)) = (actual.storage(), expected.storage())
    else {
        return Err(io::Error::other("ResNet logits are not F32"));
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 5.0e-4 * expected.abs().max(1.0);
        if !actual.is_finite() || (actual - expected).abs() > tolerance {
            return Err(io::Error::other(format!(
                "logit {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustgrad::{BenchmarkDevice, BenchmarkMetrics, RustGradDeviceBufferPeak};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_root(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!(
            "rustgrad-metal-resnet-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    fn evidence_config(scoreboard_path: PathBuf, observation_path: PathBuf) -> EvidenceConfig {
        EvidenceConfig::new(
            "a".repeat(40),
            scoreboard_path,
            observation_path,
            "macOS 15 fixture; arm64".into(),
            Some("10"),
        )
        .unwrap()
    }

    #[test]
    fn protected_evidence_configuration_is_exact_and_bounded() {
        let config = evidence_config("scoreboard.json".into(), "observation.json".into());
        assert_eq!(config.runs, 10);
        assert_eq!(benchmark_runs(None).unwrap(), DEFAULT_RUNS);
        assert!(
            EvidenceConfig::new(
                "A".repeat(40),
                "scoreboard.json".into(),
                "observation.json".into(),
                "macOS fixture".into(),
                Some("10"),
            )
            .is_err()
        );
        assert!(
            EvidenceConfig::new(
                "a".repeat(40),
                "scoreboard.json".into(),
                "observation.json".into(),
                "macOS\nfixture".into(),
                Some("10"),
            )
            .is_err()
        );
        assert!(
            EvidenceConfig::new(
                "a".repeat(40),
                "scoreboard.json".into(),
                "observation.json".into(),
                "macOS fixture".into(),
                Some("2"),
            )
            .is_err()
        );
    }

    #[test]
    fn normalized_identity_and_json_are_deterministic() {
        let config = evidence_config("scoreboard.json".into(), "observation.json".into());
        let implementation = benchmark_implementation(&config);
        assert_eq!(implementation.framework, BenchmarkFramework::RustGrad);
        assert_eq!(implementation.revision, config.revision);
        assert_eq!(implementation.command, COMMAND);
        assert_eq!(WORKLOAD, RUSTGRAD_METAL_RESNET18_WORKLOAD);
        assert_eq!(
            benchmark_workload(),
            BenchmarkWorkload::ResNet18 {
                model_identity: MODEL_IDENTITY.into(),
                input_shape: [1, 3, 224, 224],
                input_dtype: "f32".into(),
                input_sha256: BENCHMARK_IMAGE_RAW_LE_F32_SHA256.into(),
                correctness_contract: CORRECTNESS_CONTRACT.into(),
            }
        );
        let observation = BenchmarkObservation::new(
            implementation,
            benchmark_workload(),
            BenchmarkDevice {
                backend: "metal".into(),
                name: "Apple fixture GPU".into(),
                hardware_identity: "registry_id=42;fixture".into(),
                operating_system: config.operating_system.clone(),
            },
            BenchmarkMetrics {
                planning_time: None,
                pipeline_compile_time: None,
                native_prepare_time: None,
                first_run_latency: None,
                steady_run_latency: None,
                prompt_prefill: None,
                steady_decode: None,
                planned_device_memory_bytes: Some(1_024),
                measured_peak_device_memory_bytes: None,
                planned_kernel_count: None,
                executed_kernel_count: None,
                host_to_device: None,
                device_to_host: None,
                fallback_count: Some(0),
            },
        )
        .unwrap()
        .with_rustgrad_device_buffer_peak(RustGradDeviceBufferPeak::new(2_048))
        .unwrap();
        assert_eq!(
            observation.metrics.measured_peak_device_memory_bytes,
            Some(2_048)
        );
        assert_eq!(observation.metrics.planned_device_memory_bytes, Some(1_024));
        let json = deterministic_observation_json(&observation).unwrap();
        assert_eq!(json, deterministic_observation_json(&observation).unwrap());
        assert_eq!(
            BenchmarkObservation::from_json_bytes(&json).unwrap(),
            observation
        );
    }

    #[test]
    fn evidence_paths_are_distinct_and_writes_are_create_new() {
        let root = temporary_root("create-new");
        fs::create_dir(&root).unwrap();
        let scoreboard = root.join("scoreboard.json");
        let observation = root.join("observation.json");
        let config = evidence_config(scoreboard.clone(), observation);
        validate_evidence_paths(&config).unwrap();
        write_new_evidence(&scoreboard, b"first").unwrap();
        assert!(write_new_evidence(&scoreboard, b"replacement").is_err());
        assert_eq!(fs::read(&scoreboard).unwrap(), b"first");
        assert!(validate_evidence_paths(&config).is_err());

        let same = evidence_config(root.join("same.json"), root.join("same.json"));
        assert!(validate_evidence_paths(&same).is_err());
        fs::remove_file(scoreboard).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn live_workflow_publishes_raw_and_normalized_resnet_evidence() {
        let source = include_str!("metal_resnet_benchmark.rs");
        let production = source.split_once("#[cfg(test)]").unwrap().0;
        let report = production.find("scoreboard.report()").unwrap();
        let peak = production
            .find("let measured_device_buffer_peak = device_buffer_measurement.finish(&device)?;")
            .unwrap();
        let normalize = production
            .find("BenchmarkObservation::from_metal_session_scoreboard(")
            .unwrap();
        let attach = production
            .find(".with_rustgrad_device_buffer_peak(measured_device_buffer_peak)")
            .unwrap();
        let require_measured = production
            .find("observation.metrics.measured_peak_device_memory_bytes")
            .unwrap();
        let write = production
            .find("write_new_evidence(&evidence.scoreboard_path")
            .unwrap();
        assert!(report < peak && peak < normalize && normalize < attach);
        assert!(attach < require_measured && require_measured < write);
        assert!(
            production
                .find("MetalDeviceBufferMeasurement::begin(&device)?;")
                .unwrap()
                < peak
        );
        assert_eq!(
            production
                .matches("BenchmarkObservation::from_metal_session_scoreboard(")
                .count(),
            1
        );
        assert_eq!(
            production
                .matches(".with_rustgrad_device_buffer_peak(measured_device_buffer_peak)")
                .count(),
            1
        );
        for required in [
            "MetalDeviceBufferMeasurement::begin(&device)?;",
            "device_buffer_measurement.finish(&device)?;",
            "observation.metrics.planned_device_memory_bytes",
        ] {
            assert!(production.contains(required), "production omits {required}");
        }

        let workflow = include_str!("../.github/workflows/metal-live.yml");
        let resnet_job = workflow
            .split_once("  live-metal:")
            .unwrap()
            .1
            .split_once("  live-metal-llama:")
            .unwrap()
            .0;
        for required in [
            "RUSTGRAD_METAL_RESNET_SCOREBOARD_PATH",
            "RUSTGRAD_METAL_RESNET_OBSERVATION_PATH",
            "RUSTGRAD_METAL_RESNET_OPERATING_SYSTEM",
            "sw_vers -productVersion",
            "cargo run --release --example metal_resnet_benchmark",
        ] {
            assert!(resnet_job.contains(required), "workflow omits {required}");
        }
        assert!(resnet_job.contains("-e \"$output_path\" || -L \"$output_path\""));
        assert_eq!(
            resnet_job
                .matches("cargo run --release --example metal_resnet_benchmark")
                .count(),
            1
        );
        assert!(resnet_job.contains("cat \"$RUSTGRAD_METAL_RESNET_OBSERVATION_PATH\""));
        assert!(resnet_job.contains("${{ env.RUSTGRAD_METAL_RESNET_OBSERVATION_PATH }}"));
        assert_eq!(
            BENCHMARK_IMAGE_RAW_LE_F32_SHA256,
            "ec1e5c2285cbad8428cd135b829f9a216750d7ddc1b65de8b05eac114ed91b13"
        );
        let image_documentation = production.find("raw little-endian F32 payload").unwrap();
        let image_identity = production
            .find("const BENCHMARK_IMAGE_RAW_LE_F32_SHA256")
            .unwrap();
        let image_generator = production.find("fn benchmark_image()").unwrap();
        assert!(image_documentation < image_identity && image_identity < image_generator);
    }
}
