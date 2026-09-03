//! Live, exact-revision ResNet-18 evidence for one persistent Metal session.

use rustgrad::nn::{ResNet, ResNetConfig, ResNetMetalPlan};
use rustgrad::runtime::metal::{
    MetalDiscovery, MetalPlanOptions, MetalRuntime, MetalScoreboardContext, MetalSessionScoreboard,
};
use rustgrad::{Backend, CpuBackend, DType, Module, Storage, TensorData};
use std::{env, error::Error, io, path::PathBuf};

const WORKLOAD: &str = "resnet18-eval-f32-1x3x224x224";
const EVIDENCE: &str = "live self-hosted Apple GPU benchmark";
const DEFAULT_RUNS: usize = 10;
const MAX_RUNS: usize = 1_000;

fn main() -> Result<(), Box<dyn Error>> {
    let revision = required_env("RUSTGRAD_METAL_EXPECTED_SHA")?;
    validate_revision(&revision)?;
    let output = PathBuf::from(required_env("RUSTGRAD_METAL_RESNET_SCOREBOARD_PATH")?);
    let runs = benchmark_runs()?;

    let runtime = MetalRuntime::load()?;
    let device = match runtime.discover()? {
        MetalDiscovery::Devices(mut devices) if !devices.is_empty() => devices.remove(0),
        MetalDiscovery::Devices(_) | MetalDiscovery::NoDevices => {
            return Err(
                io::Error::other("benchmark requires a process-visible Metal device").into(),
            );
        }
    };
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
        MetalScoreboardContext::new(WORKLOAD, revision, EVIDENCE)?,
        plan.metal_plan(),
    );
    let cache = device.cache();
    if !cache.is_empty() {
        return Err(io::Error::other("new benchmark device cache is not empty").into());
    }
    let mut session = plan.prepare()?;
    scoreboard.bind(session.metal_session())?;
    for _ in 0..runs {
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
    if report.successful_run_count != u64::try_from(runs)?
        || report.fallback_count != 0
        || report.successful_runs.len() != runs
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
    report.write_json(output)?;
    Ok(())
}

fn required_env(name: &'static str) -> Result<String, io::Error> {
    env::var(name).map_err(|_| io::Error::other(format!("missing {name}")))
}

fn validate_revision(revision: &str) -> Result<(), io::Error> {
    if revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(io::Error::other(
            "RUSTGRAD_METAL_EXPECTED_SHA must be one lowercase full Git SHA",
        ))
    }
}

fn benchmark_runs() -> Result<usize, io::Error> {
    let runs = match env::var("RUSTGRAD_METAL_RESNET_RUNS") {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| io::Error::other("RUSTGRAD_METAL_RESNET_RUNS must be an integer"))?,
        Err(env::VarError::NotPresent) => DEFAULT_RUNS,
        Err(env::VarError::NotUnicode(_)) => {
            return Err(io::Error::other("RUSTGRAD_METAL_RESNET_RUNS must be UTF-8"));
        }
    };
    if (3..=MAX_RUNS).contains(&runs) {
        Ok(runs)
    } else {
        Err(io::Error::other(
            "RUSTGRAD_METAL_RESNET_RUNS must be between 3 and 1000",
        ))
    }
}

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
