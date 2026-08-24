//! Internal graph-to-native CPU JIT execution boundary.
//!
//! Schedule integration should pass its realized `ScheduleItem` buffers through
//! this same validated ABI boundary later; this module intentionally does not
//! change scheduling or lazily realize graphs.
use super::{Backend, CpuBackend};
use crate::{CpuJit, Graph, JitBuffer, JitError, JitKernel, NodeId, Op, TensorData};
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JitFallback {
    Error,
    CpuOracle,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JitBackendError {
    Unsupported(String),
    Binding(String),
    Native(String),
}
impl fmt::Display for JitBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "CPU JIT unsupported: {s}"),
            Self::Binding(s) => write!(f, "CPU JIT binding: {s}"),
            Self::Native(s) => write!(f, "CPU JIT native call: {s}"),
        }
    }
}
impl std::error::Error for JitBackendError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JitExecution {
    pub cache_key: String,
    pub native: bool,
}
pub struct CpuJitBackend {
    fallback: JitFallback,
    vectorized: bool,
    cache: Mutex<HashMap<String, Arc<JitKernel>>>,
}
impl CpuJitBackend {
    pub fn new(fallback: JitFallback) -> Self {
        Self {
            fallback,
            vectorized: false,
            cache: Mutex::new(HashMap::new()),
        }
    }
    pub fn vectorized(mut self, enabled: bool) -> Self {
        self.vectorized = enabled;
        self
    }
    pub fn cache_len(&self) -> usize {
        self.cache.lock().expect("JIT cache lock").len()
    }
    pub fn execute_native(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        let kernel = if matches!(
            graph
                .op(output)
                .map_err(|e| JitBackendError::Binding(e.to_string()))?,
            Op::Reduce { .. }
        ) {
            crate::lower_graph_reduction(graph, output)
        } else {
            crate::lower_graph_elementwise(graph, output)
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        let rendered = if self.vectorized {
            CpuJit::render_vectorized(&kernel)
        } else {
            CpuJit::render(&kernel)
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        let compiled = {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| JitBackendError::Native("cache lock poisoned".into()))?;
            if let Some(k) = cache.get(&rendered.cache_key) {
                k.clone()
            } else {
                let k = Arc::new(
                    if self.vectorized {
                        CpuJit::compile_vectorized(&kernel)
                    } else {
                        CpuJit::compile(&kernel)
                    }
                    .map_err(jit_error)?,
                );
                cache.insert(rendered.cache_key.clone(), k.clone());
                k
            }
        };
        let mut buffers = Vec::with_capacity(compiled.abi().buffers.len());
        for desc in &compiled.abi().buffers {
            let id = NodeId::from_index(desc.id as usize);
            let node = graph
                .nodes
                .get(id.index())
                .ok_or_else(|| JitBackendError::Binding(format!("unknown buffer {}", desc.id)))?;
            let value = match &node.op {
                Op::Input { name } => inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| JitBackendError::Binding(format!("missing input {name}")))?,
                Op::Constant(v) => v.clone(),
                _ if id == output => TensorData::from_scalars(
                    node.shape.clone(),
                    node.dtype,
                    (0..node
                        .shape
                        .numel()
                        .map_err(|e| JitBackendError::Binding(e.to_string()))?)
                        .map(|_| crate::Scalar::I(0)),
                )
                .map_err(|e| JitBackendError::Binding(e.to_string()))?,
                _ => {
                    return Err(JitBackendError::Binding(format!(
                        "buffer {} is not an input, constant, or output",
                        desc.id
                    )));
                }
            };
            buffers.push(JitBuffer::from_tensor(&value, desc.mutable));
        }
        compiled.call(&mut buffers, &[]).map_err(jit_error)?;
        let out_index = compiled
            .abi()
            .buffers
            .iter()
            .position(|b| b.id == output.index() as u64)
            .ok_or_else(|| JitBackendError::Binding("output missing from ABI".into()))?;
        let shape = graph
            .shape(output)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?
            .clone();
        let result = buffers
            .swap_remove(out_index)
            .into_tensor(shape)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        Ok((
            result,
            JitExecution {
                cache_key: rendered.cache_key,
                native: true,
            },
        ))
    }
    pub fn execute(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        match self.execute_native(graph, output, inputs) {
            Ok(x) => Ok(x),
            Err(_e) if self.fallback == JitFallback::CpuOracle => Ok((
                CpuBackend
                    .execute(graph, output, inputs)
                    .map_err(|x| JitBackendError::Native(x.to_string()))?,
                JitExecution {
                    cache_key: "cpu-fallback".into(),
                    native: false,
                },
            )),
            Err(e) => Err(e),
        }
    }
}
fn jit_error(e: JitError) -> JitBackendError {
    match e {
        JitError::Unsupported(s) | JitError::Symbolic(s) => JitBackendError::Unsupported(s),
        other => JitBackendError::Native(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Scalar, Shape};
    #[test]
    fn native_graph_boundary_matches_oracle_and_caches() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", Shape::from([5]), DType::F32);
        let y = g.square(x).unwrap();
        let input =
            TensorData::from_scalars([5], DType::F32, (0..5).map(|v| Scalar::F(v as f64 - 2.)))
                .unwrap();
        let inputs = HashMap::from([("x".into(), input)]);
        let b = CpuJitBackend::new(JitFallback::Error).vectorized(true);
        let (a, first) = b.execute(&g, y, &inputs).unwrap();
        let (c, second) = b.execute(&g, y, &inputs).unwrap();
        let expected = CpuBackend.execute(&g, y, &inputs).unwrap();
        assert!(first.native && second.native);
        assert_eq!(a.storage(), expected.storage());
        assert_eq!(c.storage(), expected.storage());
        assert_eq!(b.cache_len(), 1);
    }
    #[test]
    fn unsupported_can_be_precise_or_fallback() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([2]));
        let y = g.exp(x).unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2], vec![1., 2.]).unwrap())]);
        assert!(matches!(
            CpuJitBackend::new(JitFallback::Error).execute(&g, y, &inputs),
            Err(JitBackendError::Unsupported(_))
        ));
        let (result, trace) = CpuJitBackend::new(JitFallback::CpuOracle)
            .execute(&g, y, &inputs)
            .unwrap();
        assert!(!trace.native);
        assert_eq!(result.to_vec_f64().len(), 2);
    }

    #[test]
    fn native_reduction_and_narrow_select_match_oracle() {
        let mut reduce = Graph::new();
        let x = reduce.input_dtype("x", Shape::from([2, 2]), DType::BF16);
        let out = reduce
            .reduce(x, crate::ReduceKind::Mean, Some(vec![1]), true)
            .unwrap();
        let input = TensorData::from_scalars(
            [2, 2],
            DType::BF16,
            [Scalar::F(1.), Scalar::F(2.), Scalar::F(-3.), Scalar::F(4.)],
        )
        .unwrap();
        let inputs = HashMap::from([("x".into(), input)]);
        let backend = CpuJitBackend::new(JitFallback::Error);
        let (actual, trace) = backend.execute(&reduce, out, &inputs).unwrap();
        assert!(trace.native);
        assert_eq!(
            actual.storage(),
            CpuBackend.execute(&reduce, out, &inputs).unwrap().storage()
        );

        let mut select = Graph::new();
        let a = select.input_dtype("a", Shape::from([2]), DType::I32);
        let b = select.input_dtype("b", Shape::from([2]), DType::I32);
        let condition = select.gt(a, b).unwrap();
        let out = select.select(condition, a, b).unwrap();
        let inputs = HashMap::from([
            (
                "a".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(-4)]).unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_scalars([2], DType::I32, [Scalar::I(2), Scalar::I(-5)]).unwrap(),
            ),
        ]);
        let (actual, _) = CpuJitBackend::new(JitFallback::Error)
            .execute(&select, out, &inputs)
            .unwrap();
        assert_eq!(
            actual.storage(),
            CpuBackend.execute(&select, out, &inputs).unwrap().storage()
        );
    }
}
