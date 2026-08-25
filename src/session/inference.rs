//! Fresh-graph static CPU module inference.
use crate::nn::ModuleForward;
use crate::{Backend, CompileTrace, CpuBackend, DType, Error, Graph, Result, TensorData};
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
pub struct ModuleInferenceResult {
    output: TensorData,
    trace: CompileTrace,
    parameter_versions: BTreeMap<String, u64>,
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
    let parameter_versions = parameters
        .into_iter()
        .map(|(n, p)| Ok((n, p.version()?)))
        .collect::<Result<_>>()?;
    Ok(ModuleInferenceResult {
        output: value,
        trace: graph.trace(output)?,
        parameter_versions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module};

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
}
