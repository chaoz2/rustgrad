//! A deterministic local safetensors → strict `Linear` → CPU inference route.

use rustgrad::nn::Linear;
use rustgrad::{
    CapturedReplayExecutor, Module, TensorData, infer_module_cpu, infer_module_native_cpu,
    save_safetensors_file,
};
use std::{collections::BTreeMap, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::temp_dir().join(format!(
        "rustgrad-strict-state-example-{}",
        std::process::id()
    ));
    fs::create_dir_all(&directory)?;
    let path = directory.join("linear.safetensors");
    let state = BTreeMap::from([
        ("weight".into(), TensorData::new([1, 2], vec![2.0f32, 3.0])?),
        ("bias".into(), TensorData::new([1], vec![1.0f32])?),
    ]);
    save_safetensors_file(&path, &state, &BTreeMap::new())?;

    let model = Linear::new_static(2, 1, true, 7)?;
    model.load_safetensors_file_strict(&path)?;
    let input = TensorData::new([2, 2], vec![1., 2., 3., 4.])?;
    let result = infer_module_cpu(&model, input.clone())?;
    let native_cache = CapturedReplayExecutor::default();
    let native = infer_module_native_cpu(&model, input.clone(), &native_cache, false)?;
    let repeated = infer_module_native_cpu(&model, input, &native_cache, false)?;
    assert_eq!(result.output().to_vec_f64(), [9., 19.]);
    assert_eq!(result.output(), native.output());
    assert_eq!(native.native_trace(), repeated.native_trace());
    println!(
        "native cache entries={}, trace={:016x}",
        native_cache.compile_cache_len(false),
        native.native_trace().identity
    );
    fs::remove_dir_all(directory)?;
    Ok(())
}
