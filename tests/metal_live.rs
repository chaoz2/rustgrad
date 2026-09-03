#![cfg(target_os = "macos")]

use rustgrad::nn::{Linear, Mode, ResNet, ResNetConfig, ResNetMetalPlan};
use rustgrad::runtime::metal::*;
use rustgrad::{Backend, CapturedInference, CpuBackend, DType, Graph, Module, Storage, TensorData};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

const LIVE_EVIDENCE: &str = "live self-hosted Apple GPU";

fn input(values: [f32; 4]) -> TensorData {
    TensorData::new([2, 2], values.to_vec()).expect("fixed live input descriptor is valid")
}

fn resnet_image(phase: usize) -> TensorData {
    let values = (0usize..3 * 224 * 224)
        .map(|index| {
            let lane = (index.wrapping_mul(17).wrapping_add(phase * 29)) % 257;
            (lane as f32 - 128.0) / 1024.0
        })
        .collect::<Vec<_>>();
    TensorData::new([1, 3, 224, 224], values).expect("fixed live ResNet image descriptor is valid")
}

fn assert_resnet_logits_close(actual: &TensorData, expected: &TensorData) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let (Storage::F32(actual), Storage::F32(expected)) = (actual.storage(), expected.storage())
    else {
        panic!("fixed ResNet logits must use F32 storage")
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "logit {index}: expected NaN, got {actual}");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "logit {index}: infinity mismatch");
        } else {
            // The CPU oracle commits every F32 step, while a native MSL compiler
            // may fuse multiply-adds across this deep contraction graph. Keep the
            // tolerance local to finite logits and scale it by their magnitude.
            let tolerance = 5.0e-4 * expected.abs().max(1.0);
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "logit {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }
}

#[test]
#[ignore = "requires the protected self-hosted Apple-GPU lane"]
fn live_metal_linear_persistent_session_emits_scoreboard() {
    let expected_sha = env::var("RUSTGRAD_METAL_EXPECTED_SHA")
        .expect("the live lane must provide RUSTGRAD_METAL_EXPECTED_SHA");
    assert!(
        expected_sha.len() == 40
            && expected_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "the live evidence revision must be a lowercase full Git SHA"
    );
    let evidence_path = PathBuf::from(
        env::var_os("RUSTGRAD_METAL_SCOREBOARD_PATH")
            .expect("the live lane must provide RUSTGRAD_METAL_SCOREBOARD_PATH"),
    );

    let model = Linear::new_static(2, 2, true, 1_907).expect("fixed Linear is valid");
    model
        .weight
        .replace(input([2.0, -1.0, 0.5, 4.0]))
        .expect("fixed weight descriptor matches");
    model
        .bias
        .as_ref()
        .expect("Linear has bias")
        .replace(TensorData::new([2], vec![0.25, -0.5]).expect("fixed bias is valid"))
        .expect("fixed bias descriptor matches");

    let mut graph = Graph::new();
    let features = graph.input_dtype("features", [2, 2], DType::F32);
    let output = model
        .forward(&mut graph, features)
        .expect("fixed Linear graph is valid");
    let resident_oracle = model
        .input_bindings(&graph)
        .expect("module bindings match its graph leaves");
    let inference = CapturedInference::from_module_graph(&model, &graph, &[output])
        .expect("fixed Linear capture is valid");

    let runtime = MetalRuntime::load().expect("the live lane requires the native Metal runtime");
    let mut devices = match runtime
        .discover()
        .expect("native Metal discovery must complete")
    {
        MetalDiscovery::Devices(devices) => devices,
        MetalDiscovery::NoDevices => panic!("the live Metal lane requires a process-visible GPU"),
    };
    assert!(
        !devices.is_empty(),
        "typed discovery returned an empty device set"
    );
    let device = devices.remove(0);
    let device_info = device.info().clone();
    let renderer = device
        .renderer(64)
        .expect("selected device must produce its exact renderer identity");
    let plan = MetalInferencePlan::new(inference, renderer)
        .expect("the live Linear capture must be entirely Metal-admitted");
    assert_eq!(plan.resident_inputs().len(), 2);
    assert_eq!(plan.transient_inputs().len(), 1);
    assert_eq!(plan.transient_inputs()[0].name, "features");
    assert!(plan.summary().nonzero_item_count > 0);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(
        plan.summary().rendered_cache_keys.len(),
        plan.summary().nonzero_item_count
    );
    let stable_summary = plan.summary().clone();
    let stable_cache_keys = stable_summary.rendered_cache_keys.clone();
    let stable_resident_schema = plan.resident_inputs().to_vec();
    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new("linear-2x2", expected_sha.clone(), LIVE_EVIDENCE)
            .expect("fixed live evidence labels are valid"),
        &plan,
    );

    let cache = device.cache();
    assert!(
        cache.is_empty(),
        "freshly discovered device cache is not empty"
    );
    let mut session = plan
        .prepare(device.clone())
        .expect("live Metal preparation must compile, allocate, and upload residents");
    assert_eq!(session.device_info(), &device_info);
    assert_eq!(session.summary(), &stable_summary);
    assert_eq!(session.resident_inputs(), stable_resident_schema.as_slice());
    let stable_device_owner_id = session.device_owner_id();
    let preparation = session.preparation_report().clone();
    assert_eq!(preparation.resident_h2d_calls, 2);
    assert_eq!(
        preparation.resident_h2d_bytes,
        stable_resident_schema
            .iter()
            .map(|binding| binding.desc.bytes)
            .sum::<usize>()
    );
    assert_eq!(
        preparation.pipeline_cache_request_count,
        stable_summary.nonzero_item_count
    );
    assert_eq!(
        preparation.pipeline_cache_hit_count + preparation.pipeline_cache_miss_count,
        preparation.pipeline_cache_request_count
    );
    assert_eq!(cache.len(), preparation.pipeline_cache_miss_count);
    let stable_cache_len = cache.len();
    assert_eq!(
        session
            .compiled_kernels()
            .map(|kernel| kernel.cache_key.as_str())
            .collect::<Vec<_>>(),
        stable_cache_keys
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    scoreboard
        .bind(&session)
        .expect("scoreboard must bind the exact prepared deployment");

    let live_inputs = [input([1.0, 2.0, -1.0, 0.5]), input([-2.0, 0.25, 4.0, -0.5])];
    for (index, live_input) in live_inputs.into_iter().enumerate() {
        let mut oracle_bindings = resident_oracle.clone();
        oracle_bindings.insert("features".into(), live_input.clone());
        let expected = CpuBackend
            .execute(&graph, output, &oracle_bindings)
            .expect("CPU oracle must execute the fixed Linear graph");
        let run = session
            .run(&BTreeMap::from([("features".into(), live_input)]))
            .expect("live persistent Metal invocation must succeed");
        assert_eq!(run.outputs().len(), 1);
        assert_eq!(run.outputs()[0].shape(), expected.shape());
        assert_eq!(run.outputs()[0].dtype(), expected.dtype());
        assert_eq!(
            run.outputs()[0]
                .to_le_bytes()
                .expect("Metal output bytes are representable"),
            expected
                .to_le_bytes()
                .expect("CPU oracle bytes are representable")
        );
        assert_eq!(run.report().successful_invocation, index as u64 + 1);
        assert_eq!(run.report().first_successful_run, index == 0);
        assert_eq!(run.report().transient_h2d_calls, 1);
        assert_eq!(run.report().transient_h2d_bytes, 4 * DType::F32.itemsize());
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert_eq!(run.report().retained_d2h_bytes, 4 * DType::F32.itemsize());
        assert_eq!(
            run.report().kernel_launch_count,
            stable_summary.nonzero_item_count
        );
        scoreboard
            .record(&run)
            .expect("successful live runs must form one ordered prefix");
        assert_eq!(cache.len(), stable_cache_len);
        assert_eq!(session.summary(), &stable_summary);
        assert_eq!(session.resident_inputs(), stable_resident_schema.as_slice());
        assert_eq!(session.device_owner_id(), stable_device_owner_id);
        assert_eq!(session.preparation_report(), &preparation);
        assert_eq!(
            session
                .compiled_kernels()
                .map(|kernel| kernel.cache_key.as_str())
                .collect::<Vec<_>>(),
            stable_cache_keys
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
    }

    assert_eq!(session.successful_run_count(), 2);
    let report = scoreboard
        .report()
        .expect("two live runs must produce a scoreboard report");
    assert_eq!(
        report.context.implementation_revision(),
        expected_sha.as_str()
    );
    assert_eq!(report.context.evidence(), LIVE_EVIDENCE);
    assert_eq!(report.device, device_info);
    assert_eq!(report.rendered_cache_keys, stable_cache_keys);
    assert_eq!(report.successful_run_count, 2);
    assert_eq!(report.fallback_count, 0);
    assert_eq!(report.kernel_launch_count, 2 * report.planned_kernel_count);
    assert_eq!(
        report.resident_host_api_h2d_calls,
        preparation.resident_h2d_calls
    );
    assert_eq!(report.transient_host_api_h2d_calls, 2);
    assert_eq!(report.retained_host_api_d2h_calls, 2);
    assert_eq!(
        report.host_api_h2d_calls,
        preparation.resident_h2d_calls + 2
    );
    assert_eq!(
        report.host_api_h2d_bytes,
        preparation.resident_h2d_bytes + 2 * 4 * DType::F32.itemsize()
    );
    let encoded = report
        .to_json_bytes()
        .expect("live scoreboard JSON must serialize");
    assert_eq!(
        encoded,
        report
            .to_json_bytes()
            .expect("scoreboard serialization must be deterministic")
    );
    report
        .write_json(&evidence_path)
        .expect("live scoreboard evidence must be written");
    assert_eq!(
        fs::read(&evidence_path).expect("written live evidence must be readable"),
        encoded
    );
}

#[test]
#[ignore = "requires the protected self-hosted Apple-GPU lane"]
fn live_metal_resnet18_persistent_session_matches_full_cpu_oracle() {
    let expected_sha = env::var("RUSTGRAD_METAL_EXPECTED_SHA")
        .expect("the live lane must provide RUSTGRAD_METAL_EXPECTED_SHA");
    assert!(
        expected_sha.len() == 40
            && expected_sha
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "the live evidence revision must be a lowercase full Git SHA"
    );
    let evidence_path = PathBuf::from(
        env::var_os("RUSTGRAD_METAL_RESNET_SCOREBOARD_PATH")
            .expect("the live lane must provide RUSTGRAD_METAL_RESNET_SCOREBOARD_PATH"),
    );
    let runtime = MetalRuntime::load().expect("the live lane requires the native Metal runtime");
    let mut devices = match runtime
        .discover()
        .expect("native Metal discovery must complete")
    {
        MetalDiscovery::Devices(devices) => devices,
        MetalDiscovery::NoDevices => panic!("the live Metal lane requires a process-visible GPU"),
    };
    assert!(
        !devices.is_empty(),
        "typed discovery returned an empty device set"
    );
    let device = devices.remove(0);
    let device_info = device.info().clone();

    let model = ResNet::new_static(ResNetConfig::default(), 19)
        .expect("default ResNet-18 construction is valid");
    let mut graph = Graph::new();
    let image = graph.input_dtype("image", [1, 3, 224, 224], DType::F32);
    let forward = model
        .forward_mode(&mut graph, image, Mode::Eval)
        .expect("default Eval ResNet-18 graph is valid");
    assert!(forward.pending.is_empty());
    let logits = forward
        .output
        .logits()
        .expect("default ResNet-18 has classifier logits");
    assert_eq!(graph.shape(logits).unwrap().dims(), &[1, 1000]);
    let live_image = resnet_image(0);
    let mut oracle_bindings = model
        .input_bindings(&graph)
        .expect("module bindings match the full ResNet graph leaves");
    oracle_bindings.insert("image".into(), live_image.clone());
    let expected = CpuBackend
        .execute(&graph, logits, &oracle_bindings)
        .expect("the release-profile live lane must execute the full CPU ResNet oracle");
    assert_eq!(expected.shape().dims(), &[1, 1000]);
    assert_eq!(expected.dtype(), DType::F32);
    let Storage::F32(expected_logits) = expected.storage() else {
        panic!("full ResNet CPU oracle must use F32 storage")
    };
    assert!(
        expected_logits.iter().all(|value| value.is_finite()),
        "the deterministic initialized full-model oracle must produce finite logits"
    );
    let plan =
        ResNetMetalPlan::eval_f32(&model, &device, [1, 3, 224, 224], MetalPlanOptions::new(64))
            .expect("full default ResNet-18 capture must be entirely Metal-admitted");
    assert_eq!(plan.selected_device_info(), &device_info);
    assert_eq!(plan.selected_device_owner_id(), device.owner_id());
    assert!(!plan.execution_plan().items.is_empty());
    assert_eq!(
        plan.rendered_items().count(),
        plan.execution_plan().schedule_item_count
    );
    assert!(!plan.resident_inputs().is_empty());
    assert_eq!(
        plan.transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["image"]
    );
    assert!(plan.summary().nonzero_item_count > 0);
    assert_eq!(plan.summary().zero_item_count, 0);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(
        plan.summary().rendered_cache_keys.len(),
        plan.summary().nonzero_item_count
    );
    let stable_summary = plan.summary().clone();
    let stable_cache_keys = stable_summary.rendered_cache_keys.clone();
    let stable_resident_schema = plan.resident_inputs().to_vec();
    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new(
            "resnet18-eval-f32-1x3x224x224",
            expected_sha.clone(),
            LIVE_EVIDENCE,
        )
        .expect("fixed live evidence labels are valid"),
        plan.metal_plan(),
    );

    let cache = device.cache();
    assert!(
        cache.is_empty(),
        "freshly discovered device cache is not empty"
    );
    let mut session = plan
        .prepare()
        .expect("live ResNet-18 preparation must compile, allocate, and upload residents");
    assert_eq!(session.metal_session().device_info(), &device_info);
    assert_eq!(session.metal_session().summary(), &stable_summary);
    assert_eq!(
        session.metal_session().resident_inputs(),
        stable_resident_schema.as_slice()
    );
    let stable_device_owner_id = session.metal_session().device_owner_id();
    let preparation = session.metal_session().preparation_report().clone();
    assert!(preparation.resident_h2d_calls > 0);
    assert!(preparation.resident_h2d_bytes > 0);
    assert_eq!(
        preparation.pipeline_cache_request_count,
        stable_summary.nonzero_item_count
    );
    assert_eq!(
        preparation.pipeline_cache_hit_count + preparation.pipeline_cache_miss_count,
        preparation.pipeline_cache_request_count
    );
    assert_eq!(cache.len(), preparation.pipeline_cache_miss_count);
    let stable_cache_len = cache.len();
    assert_eq!(
        session
            .metal_session()
            .compiled_kernels()
            .map(|kernel| kernel.cache_key.as_str())
            .collect::<Vec<_>>(),
        stable_cache_keys
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    scoreboard
        .bind(session.metal_session())
        .expect("scoreboard must bind the exact prepared deployment");

    for index in 0..2 {
        let run = session
            .run(live_image.clone())
            .expect("live persistent ResNet-18 invocation must succeed");
        assert_resnet_logits_close(run.logits(), &expected);
        assert_eq!(run.report().successful_invocation, index as u64 + 1);
        assert_eq!(run.report().first_successful_run, index == 0);
        assert_eq!(run.report().transient_h2d_calls, 1);
        assert_eq!(
            run.report().transient_h2d_bytes,
            3 * 224 * 224 * DType::F32.itemsize()
        );
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert_eq!(
            run.report().retained_d2h_bytes,
            1000 * DType::F32.itemsize()
        );
        assert_eq!(
            run.report().kernel_launch_count,
            stable_summary.nonzero_item_count
        );
        scoreboard
            .record(run.metal_run())
            .expect("successful live runs must form one ordered prefix");
        assert_eq!(cache.len(), stable_cache_len);
        assert_eq!(session.metal_session().summary(), &stable_summary);
        assert_eq!(
            session.metal_session().resident_inputs(),
            stable_resident_schema.as_slice()
        );
        assert_eq!(
            session.metal_session().device_owner_id(),
            stable_device_owner_id
        );
        assert_eq!(session.metal_session().preparation_report(), &preparation);
        assert_eq!(
            session
                .metal_session()
                .compiled_kernels()
                .map(|kernel| kernel.cache_key.as_str())
                .collect::<Vec<_>>(),
            stable_cache_keys
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
    }

    assert_eq!(session.metal_session().successful_run_count(), 2);
    let report = scoreboard
        .report()
        .expect("two live runs must produce a scoreboard report");
    assert_eq!(
        report.context.implementation_revision(),
        expected_sha.as_str()
    );
    assert_eq!(report.context.evidence(), LIVE_EVIDENCE);
    assert_eq!(report.device, device_info);
    assert_eq!(report.rendered_cache_keys, stable_cache_keys);
    assert_eq!(report.successful_run_count, 2);
    assert_eq!(report.fallback_count, 0);
    assert_eq!(report.kernel_launch_count, 2 * report.planned_kernel_count);
    assert_eq!(
        report.resident_host_api_h2d_calls,
        preparation.resident_h2d_calls
    );
    assert_eq!(report.transient_host_api_h2d_calls, 2);
    assert_eq!(report.transient_host_api_h2d_bytes, 2 * 3 * 224 * 224 * 4);
    assert_eq!(report.retained_host_api_d2h_calls, 2);
    assert_eq!(report.retained_host_api_d2h_bytes, 2 * 1000 * 4);
    assert!(report.first_run_host_wall_time.is_some());
    assert_eq!(report.steady_run_host_wall_times.len(), 1);
    assert!(report.steady_run_host_wall_time_summary.is_some());
    let encoded = report
        .to_json_bytes()
        .expect("live scoreboard JSON must serialize");
    assert_eq!(
        encoded,
        report
            .to_json_bytes()
            .expect("scoreboard serialization must be deterministic")
    );
    report
        .write_json(&evidence_path)
        .expect("live scoreboard evidence must be written");
    assert_eq!(
        fs::read(&evidence_path).expect("written live evidence must be readable"),
        encoded
    );
}
