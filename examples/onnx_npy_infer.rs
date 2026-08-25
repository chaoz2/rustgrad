//! Run one bounded local static ONNX model with explicit `name=path.npy` maps.
//! Pass `--native` to opt into strict native CPU replay for its narrow static
//! one-input/one-output F32 MatMul/Add/ReLU boundary.

use rustgrad::{
    CapturedReplayExecutor,
    onnx::{NamedPaths, OnnxWorkflowLimits, run_onnx_files, run_onnx_files_native},
};
use std::{env, path::PathBuf};

fn named(value: &str) -> Result<(String, PathBuf), String> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| format!("expected name=path.npy, got {value:?}"))?;
    if name.is_empty() || path.is_empty() {
        return Err(format!("expected nonempty name=path.npy, got {value:?}"));
    }
    Ok((name.into(), path.into()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let model = args.next().ok_or(
        "usage: onnx_npy_infer MODEL.onnx INPUT=INPUT.npy ... --output OUTPUT=OUTPUT.npy ...",
    )?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut output_mode = false;
    let mut native = false;
    for argument in args {
        if argument == "--native" {
            native = true;
        } else if argument == "--output" {
            output_mode = true;
        } else if output_mode {
            outputs.push(named(&argument)?);
        } else {
            inputs.push(named(&argument)?);
        }
    }
    let inputs = NamedPaths::new(inputs)?;
    let outputs = NamedPaths::new(outputs)?;
    if native {
        let executor = CapturedReplayExecutor::default();
        let result = run_onnx_files_native(
            model,
            &inputs,
            &outputs,
            OnnxWorkflowLimits::default(),
            &executor,
            false,
        )?;
        println!(
            "{}: {:?} {:?}; native cache keys={:?}",
            result.output_name(),
            result.output().dtype(),
            result.output().shape().dims(),
            result.native_trace().native_cache_keys
        );
    } else {
        let values = run_onnx_files(model, &inputs, &outputs, OnnxWorkflowLimits::default())?;
        for (name, value) in values {
            println!("{name}: {:?} {:?}", value.dtype(), value.shape().dims());
        }
    }
    Ok(())
}
