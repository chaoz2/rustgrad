//! Fresh-graph static CPU module inference.
use crate::nn::ModuleForward;
use crate::{
    Backend, CapturedBackendPolicy, CapturedReplayExecutor, CapturedReplayTrace, CapturedSchedule,
    CompileTrace, CpuBackend, DType, Error, Graph, Result, TensorData, schedule,
};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct ModuleInferenceResult {
    output: TensorData,
    trace: CompileTrace,
    parameter_versions: BTreeMap<String, u64>,
}
#[derive(Clone, Debug)]
pub struct NativeModuleInferenceResult {
    output: TensorData,
    trace: CapturedReplayTrace,
    parameter_versions: BTreeMap<String, u64>,
    native_trace: NativeModuleInferenceTrace,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleInferenceTrace {
    pub identity: u64,
    pub capture_identity: u64,
    pub input_shape: crate::Shape,
    pub input_dtype: DType,
    pub parameter_versions: BTreeMap<String, u64>,
    pub vectorized: bool,
    pub renderer_version: &'static str,
    pub native_cache_keys: Vec<Option<String>>,
}
impl NativeModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CapturedReplayTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
    pub fn native_trace(&self) -> &NativeModuleInferenceTrace {
        &self.native_trace
    }
}
impl ModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CompileTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
}
/// Builds and discards one fresh CPU graph for a one-input static module.
pub fn infer_module_cpu(
    module: &impl ModuleForward,
    input: TensorData,
) -> Result<ModuleInferenceResult> {
    if input.dtype() != DType::F32 {
        return Err(Error::SessionTraining {
            reason: "module CPU inference input must have dtype F32".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    bindings.insert("module_input".into(), input);
    let value = CpuBackend.execute(&graph, output, &bindings)?;
    let parameter_versions: BTreeMap<String, u64> = parameters
        .into_iter()
        .map(|(n, p)| Ok((n, p.version()?)))
        .collect::<Result<_>>()?;
    Ok(ModuleInferenceResult {
        output: value,
        trace: graph.trace(output)?,
        parameter_versions,
    })
}

/// Fresh-graph strict native CPU inference. The caller owns the executor and
/// therefore its deterministic compilation cache; unsupported graphs fail
/// before a native item is executed and never fall back to interpretation.
pub fn infer_module_native_cpu(
    module: &impl ModuleForward,
    input: TensorData,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<NativeModuleInferenceResult> {
    if input.dtype() != DType::F32 {
        return Err(Error::SessionTraining {
            reason: "module native CPU inference input must have dtype F32".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    let input_shape = input.shape().clone();
    bindings.insert("module_input".into(), input);
    let bindings = bindings.into_iter().collect::<BTreeMap<_, _>>();
    let scheduled = schedule(&graph, output).map_err(|e| Error::SessionTraining {
        reason: e.to_string(),
    })?;
    let capture = CapturedSchedule::capture(&graph, &scheduled, &[output]).map_err(|e| {
        Error::SessionTraining {
            reason: e.to_string(),
        }
    })?;
    let replay = executor
        .replay(
            &capture,
            &bindings,
            crate::CapturedReplayOptions {
                backend: CapturedBackendPolicy::NativeJit { vectorized },
            },
        )
        .map_err(|e| Error::SessionTraining {
            reason: e.to_string(),
        })?;
    let parameter_versions: BTreeMap<String, u64> = parameters
        .into_iter()
        .map(|(n, p)| Ok((n, p.version()?)))
        .collect::<Result<_>>()?;
    let native_cache_keys = replay
        .trace
        .items
        .iter()
        .map(|item| item.native_cache_key.clone())
        .collect::<Vec<_>>();
    let mut bytes = format!(
        "{}:{:?}:{}:{:?}",
        capture.identity, input_shape, vectorized, parameter_versions
    )
    .into_bytes();
    for key in &native_cache_keys {
        bytes.extend_from_slice(key.as_deref().unwrap_or("").as_bytes());
    }
    let identity = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    Ok(NativeModuleInferenceResult {
        output: replay
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::SessionTraining {
                reason: "native inference missing output".into(),
            })?,
        trace: replay.trace,
        parameter_versions: parameter_versions.clone(),
        native_trace: NativeModuleInferenceTrace {
            identity,
            capture_identity: capture.identity,
            input_shape,
            input_dtype: DType::F32,
            parameter_versions: parameter_versions.clone(),
            vectorized,
            renderer_version: crate::cpu_jit::RENDERER_VERSION,
            native_cache_keys,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module, ModuleForward, Parameter, Sequential, StateKind};
    use crate::{NodeId, Scalar};

    struct DuplicateTraversal {
        first: Parameter,
        second: Parameter,
    }

    impl Module for DuplicateTraversal {
        fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor("weight".into(), &self.first, StateKind::Parameter);
            visitor("weight".into(), &self.second, StateKind::Parameter);
        }
    }

    impl ModuleForward for DuplicateTraversal {
        fn forward(&self, _: &mut Graph, input: NodeId) -> Result<NodeId> {
            Ok(input)
        }
    }

    #[test]
    fn inference_is_fresh_deterministic_and_nonmutating() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let before = model.state_dict().unwrap();
        let first = infer_module_cpu(&model, input.clone()).unwrap();
        let second = infer_module_cpu(&model, input.clone()).unwrap();
        assert_eq!(first.output().to_vec_f64(), vec![9., 19.]);
        assert_eq!(first.output(), second.output());
        assert_eq!(first.trace(), second.trace());
        assert_eq!(before, model.state_dict().unwrap());
        assert!(infer_module_cpu(&model, TensorData::new([1, 3], vec![0.; 3]).unwrap()).is_err());
        assert!(
            infer_module_cpu(
                &model,
                TensorData::from_scalars([1, 2], DType::F64, [crate::Scalar::F(0.); 2]).unwrap()
            )
            .is_err()
        );
        let empty =
            infer_module_cpu(&model, TensorData::new([0, 2], Vec::<f32>::new()).unwrap()).unwrap();
        assert_eq!(empty.output().shape().dims(), &[0, 1]);
    }

    #[test]
    fn strict_native_linear_matches_cpu_and_reuses_caller_cache() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let first = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), first.output());
        assert_eq!(first.output(), second.output());
        assert_eq!(first.native_trace(), second.native_trace());
        assert!(
            first
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        assert_eq!(cached, executor.compile_cache_len(false));
        model
            .weight
            .replace(TensorData::new([1, 2], vec![4., 3.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            first.native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(first.output(), changed.output());
        let vector = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            true,
        )
        .unwrap();
        assert_eq!(
            vector.output(),
            infer_module_cpu(
                &model,
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
            )
            .unwrap()
            .output()
        );
        assert!(vector.native_trace().vectorized);
        assert!(executor.compile_cache_len(true) > 0);
        let wider = infer_module_native_cpu(
            &model,
            TensorData::new([3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            changed.native_trace().identity,
            wider.native_trace().identity
        );
        assert_eq!(wider.output().shape().dims(), &[3, 1]);
    }

    #[test]
    fn strict_native_sequential_matches_cpu() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 2, true, 1).unwrap());
        model.push(Linear::new_static(2, 1, true, 2).unwrap());
        let input = TensorData::new([1, 2], vec![1., -2.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let native = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), native.output());
        assert!(
            native
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
    }

    #[test]
    fn inference_rejects_poisoned_or_duplicate_modules_before_execution() {
        let poisoned = Linear::new_static(2, 1, true, 1).unwrap();
        let before = poisoned.bias.as_ref().unwrap().snapshot().unwrap();
        poisoned.weight.poison_for_test();
        assert!(matches!(
            infer_module_cpu(&poisoned, TensorData::new([1, 2], vec![0., 1.]).unwrap()),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert_eq!(
            poisoned.bias.as_ref().unwrap().snapshot().unwrap().data,
            before.data
        );

        let duplicate = DuplicateTraversal {
            first: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(1.)]).unwrap(),
                true,
            ),
            second: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(2.)]).unwrap(),
                true,
            ),
        };
        let before = (
            duplicate.first.snapshot().unwrap(),
            duplicate.second.snapshot().unwrap(),
        );
        assert!(matches!(
            infer_module_cpu(&duplicate, TensorData::new([1, 1], vec![1.]).unwrap()),
            Err(Error::Serialization { .. })
        ));
        assert_eq!(duplicate.first.snapshot().unwrap().data, before.0.data);
        assert_eq!(duplicate.second.snapshot().unwrap().data, before.1.data);
    }
}
