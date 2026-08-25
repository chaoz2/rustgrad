//! Strict native CPU replay for the narrow static ONNX inference boundary.

use super::{OnnxModel, OnnxValueInfo, bad};
use crate::{
    CapturedBackendPolicy, CapturedReplayExecutor, CapturedReplayOptions, CapturedReplayTrace,
    CapturedSchedule, Result, TensorData, schedule,
};
use std::collections::BTreeMap;

/// Deterministic, resource-free facts for one strict-native ONNX execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeOnnxInferenceTrace {
    /// Logical identity excluding input bytes, paths, pointers, and cache state.
    pub identity: u64,
    pub capture_identity: u64,
    pub inputs: Vec<OnnxValueInfo>,
    pub output: OnnxValueInfo,
    pub vectorized: bool,
    pub renderer_version: &'static str,
    pub native_cache_keys: Vec<Option<String>>,
}

/// Detached strict-native result for one static ONNX input/output model.
#[derive(Clone, Debug)]
pub struct NativeOnnxInferenceResult {
    output_name: String,
    output: TensorData,
    replay_trace: CapturedReplayTrace,
    native_trace: NativeOnnxInferenceTrace,
}

impl NativeOnnxInferenceResult {
    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn output(&self) -> &TensorData {
        &self.output
    }

    pub fn replay_trace(&self) -> &CapturedReplayTrace {
        &self.replay_trace
    }

    pub fn native_trace(&self) -> &NativeOnnxInferenceTrace {
        &self.native_trace
    }
}

impl OnnxModel {
    /// Executes the one-input/one-output static ONNX subset through strict
    /// native CPU replay. The caller owns the executor and its scalar/vector
    /// compilation caches; unsupported schedules fail before native execution
    /// and never fall back to CPU interpretation.
    pub fn run_native_static(
        &self,
        inputs: &BTreeMap<String, TensorData>,
        executor: &CapturedReplayExecutor,
        vectorized: bool,
    ) -> Result<NativeOnnxInferenceResult> {
        self.validate_named_inputs(inputs)?;
        let input_count = self.inputs.len();
        let output_count = self.outputs.len();
        if input_count != 1 || output_count != 1 {
            return Err(bad(format!(
                "strict native ONNX requires exactly one input and one output, got {input_count} input(s) and {output_count} output(s)"
            )));
        }
        let (output_name, output_node) = self
            .outputs
            .iter()
            .next()
            .map(|(name, node)| (name.clone(), *node))
            .ok_or_else(|| bad("strict native ONNX missing output"))?;
        let scheduled = schedule(&self.graph, output_node)
            .map_err(|error| bad(format!("strict native ONNX schedule: {error}")))?;
        let capture = CapturedSchedule::capture(&self.graph, &scheduled, &[output_node])
            .map_err(|error| bad(format!("strict native ONNX capture: {error}")))?;
        let replay = executor
            .replay(
                &capture,
                inputs,
                CapturedReplayOptions {
                    backend: CapturedBackendPolicy::NativeJit { vectorized },
                },
            )
            .map_err(|error| bad(format!("strict native ONNX replay: {error}")))?;
        let output = replay
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| bad("strict native ONNX replay returned no output"))?;
        let output_info = OnnxValueInfo {
            name: output_name.clone(),
            shape: self.graph.shape(output_node)?.clone(),
            dtype: self.graph.dtype(output_node)?,
        };
        let native_cache_keys = replay
            .trace
            .items
            .iter()
            .map(|item| item.native_cache_key.clone())
            .collect::<Vec<_>>();
        let identity = native_identity(
            capture.identity,
            self.input_info.values(),
            &output_info,
            vectorized,
            &native_cache_keys,
        );
        Ok(NativeOnnxInferenceResult {
            output_name,
            output,
            replay_trace: replay.trace,
            native_trace: NativeOnnxInferenceTrace {
                identity,
                capture_identity: capture.identity,
                inputs: self.input_info.values().cloned().collect(),
                output: output_info,
                vectorized,
                renderer_version: crate::cpu_jit::RENDERER_VERSION,
                native_cache_keys,
            },
        })
    }
}

fn native_identity<'a>(
    capture_identity: u64,
    inputs: impl Iterator<Item = &'a OnnxValueInfo>,
    output: &OnnxValueInfo,
    vectorized: bool,
    native_cache_keys: &[Option<String>],
) -> u64 {
    let mut bytes = format!(
        "{capture_identity}:{vectorized}:{}:{:?}:{:?}",
        output.name, output.shape, output.dtype
    )
    .into_bytes();
    for input in inputs {
        bytes.extend_from_slice(
            format!("{}:{:?}:{:?}", input.name, input.shape, input.dtype).as_bytes(),
        );
    }
    for key in native_cache_keys {
        bytes.extend_from_slice(key.as_deref().unwrap_or("").as_bytes());
    }
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
