//! Internal graph-to-native CPU JIT execution boundary.
//!
//! Schedule integration should pass its realized `ScheduleItem` buffers through
//! this same validated ABI boundary later; this module intentionally does not
//! change scheduling or lazily realize graphs.
use super::{Backend, CpuBackend};
use crate::{
    CpuJit, Graph, JitBuffer, JitError, JitKernel, NodeId, Op, ScheduleItem, TensorData, VectorPlan,
};
use std::{
    collections::{BTreeMap, HashMap},
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
    pub vector: VectorPlan,
    pub vector_main: usize,
    pub vector_tail: usize,
}
pub struct CpuJitBackend {
    fallback: JitFallback,
    vectorized: bool,
    cache: Mutex<HashMap<String, Arc<JitKernel>>>,
}
pub(crate) struct PreparedScheduleItem {
    kernel: Arc<JitKernel>,
    pub(crate) native_cache_key: String,
    pub(crate) cache_hit: bool,
    pub(crate) vector: VectorPlan,
    schedule_cache_key: u64,
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
    fn render_kernel(
        &self,
        kernel: &crate::UOp,
    ) -> Result<(VectorPlan, crate::cpu_jit::RenderedC), JitBackendError> {
        let vector = if self.vectorized {
            CpuJit::vector_plan(kernel).map_err(|e| JitBackendError::Unsupported(e.to_string()))?
        } else {
            VectorPlan {
                lanes: 1,
                enabled: false,
                reason: "scalar policy disabled vector lanes".into(),
            }
        };
        let rendered = if self.vectorized {
            CpuJit::render_vectorized(kernel)
        } else {
            CpuJit::render(kernel)
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        Ok((vector, rendered))
    }

    fn compile_cached(
        &self,
        kernel: &crate::UOp,
        cache_key: &str,
    ) -> Result<(Arc<JitKernel>, bool), JitBackendError> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| JitBackendError::Native("cache lock poisoned".into()))?;
        if let Some(compiled) = cache.get(cache_key) {
            return Ok((compiled.clone(), true));
        }
        let compiled = Arc::new(
            if self.vectorized {
                CpuJit::compile_vectorized(kernel)
            } else {
                CpuJit::compile(kernel)
            }
            .map_err(jit_error)?,
        );
        cache.insert(cache_key.to_owned(), compiled.clone());
        Ok((compiled, false))
    }

    pub(crate) fn validate_schedule_item(
        &self,
        item: &ScheduleItem,
    ) -> Result<(), JitBackendError> {
        item.validate_input_bindings()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        let (_, rendered) = self.render_kernel(&item.kernel)?;
        for (index, binding) in item.ordered_inputs().iter().enumerate() {
            if binding.abi_index != index {
                return Err(JitBackendError::Binding(
                    "non-contiguous schedule ABI index".into(),
                ));
            }
            let native = rendered
                .abi
                .buffers
                .iter()
                .find(|x| x.id == binding.desc.id)
                .ok_or_else(|| {
                    JitBackendError::Binding(format!(
                        "schedule buffer {} absent from native ABI",
                        binding.desc.id
                    ))
                })?;
            if native.dtype != binding.desc.dtype
                || native.elements.checked_mul(native.dtype.itemsize()) != Some(binding.desc.bytes)
                || native.mutable
            {
                return Err(JitBackendError::Binding(format!(
                    "schedule binding {index} mismatches native resource"
                )));
            }
        }
        if rendered.abi.symbol_count != 0 {
            return Err(JitBackendError::Unsupported(
                "captured symbolic native ABI is not specialized".into(),
            ));
        }
        if rendered.abi.buffers.len() != item.ordered_inputs().len() + 1 {
            return Err(JitBackendError::Binding(
                "native ABI has unexpected resources".into(),
            ));
        }
        let output = rendered
            .abi
            .buffers
            .last()
            .ok_or_else(|| JitBackendError::Binding("native output missing".into()))?;
        if self.vectorized
            && !matches!(item.kernel.kind(), crate::UOpKind::Matmul)
            && output.elements > 1
            && rendered.abi.buffers[..rendered.abi.buffers.len() - 1]
                .iter()
                .any(|x| x.elements == 1)
        {
            return Err(JitBackendError::Unsupported(
                "captured vector scalar broadcast is not proven lane-safe".into(),
            ));
        }
        let elements = item
            .output
            .shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        if output.id != item.output.id
            || output.dtype != item.output.dtype
            || output.elements != elements
            || !output.mutable
        {
            return Err(JitBackendError::Binding(
                "native output descriptor mismatch".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn prepare_schedule_item(
        &self,
        item: &ScheduleItem,
    ) -> Result<PreparedScheduleItem, JitBackendError> {
        self.validate_schedule_item(item)?;
        let (vector, rendered) = self.render_kernel(&item.kernel)?;
        let native_cache_key = format!("{}-schedule-{:016x}", rendered.cache_key, item.cache_key);
        let (kernel, cache_hit) = self.compile_cached(&item.kernel, &native_cache_key)?;
        Ok(PreparedScheduleItem {
            kernel,
            native_cache_key,
            cache_hit,
            vector,
            schedule_cache_key: item.cache_key,
        })
    }

    pub(crate) fn execute_prepared_schedule_item(
        &self,
        item: &ScheduleItem,
        values: &BTreeMap<u64, TensorData>,
        prepared: &PreparedScheduleItem,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        if prepared.schedule_cache_key != item.cache_key {
            return Err(JitBackendError::Binding(
                "prepared schedule identity mismatch".into(),
            ));
        }
        let mut buffers = Vec::with_capacity(prepared.kernel.abi().buffers.len());
        for desc in &prepared.kernel.abi().buffers {
            if desc.id == item.output.id {
                buffers.push(JitBuffer::zeroed(desc.dtype, desc.elements, true));
            } else {
                let value = values.get(&desc.id).ok_or_else(|| {
                    JitBackendError::Binding(format!("missing captured buffer {}", desc.id))
                })?;
                buffers.push(JitBuffer::from_tensor(value, false));
            }
        }
        prepared.kernel.call(&mut buffers, &[]).map_err(jit_error)?;
        let output_index = prepared
            .kernel
            .abi()
            .buffers
            .iter()
            .position(|x| x.id == item.output.id)
            .ok_or_else(|| JitBackendError::Binding("native output missing".into()))?;
        let output_elements = item
            .output
            .shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        let output = buffers
            .swap_remove(output_index)
            .into_tensor(item.output.shape.clone())
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        Ok((
            output,
            JitExecution {
                cache_key: prepared.native_cache_key.clone(),
                native: true,
                vector_main: if prepared.vector.enabled {
                    output_elements / prepared.vector.lanes * prepared.vector.lanes
                } else {
                    0
                },
                vector_tail: if prepared.vector.enabled {
                    output_elements % prepared.vector.lanes
                } else {
                    output_elements
                },
                vector: prepared.vector.clone(),
            },
        ))
    }
    pub fn execute_native(
        &self,
        graph: &Graph,
        output: NodeId,
        inputs: &HashMap<String, TensorData>,
    ) -> Result<(TensorData, JitExecution), JitBackendError> {
        let kernel = match graph
            .op(output)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?
        {
            Op::Reduce { .. } => crate::lower_graph_reduction(graph, output),
            Op::Matmul { .. } => crate::lower_graph_matmul(graph, output),
            _ => crate::lower_graph_elementwise(graph, output),
        }
        .map_err(|e| JitBackendError::Unsupported(e.to_string()))?;
        let (vector, rendered) = self.render_kernel(&kernel)?;
        let (compiled, _) = self.compile_cached(&kernel, &rendered.cache_key)?;
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
        let output_elements = shape
            .numel()
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        let result = buffers
            .swap_remove(out_index)
            .into_tensor(shape)
            .map_err(|e| JitBackendError::Binding(e.to_string()))?;
        Ok((
            result,
            JitExecution {
                cache_key: rendered.cache_key,
                native: true,
                vector_main: if vector.enabled {
                    output_elements / vector.lanes * vector.lanes
                } else {
                    0
                },
                vector_tail: if vector.enabled {
                    output_elements % vector.lanes
                } else {
                    output_elements
                },
                vector,
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
                    vector: VectorPlan {
                        lanes: 1,
                        enabled: false,
                        reason: "CPU oracle fallback".into(),
                    },
                    vector_main: 0,
                    vector_tail: 0,
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
    fn native_llama_unaries_match_oracle() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", Shape::from([3]), DType::F32);
        let exp = graph.exp(input).unwrap();
        let reciprocal = graph.reciprocal(exp).unwrap();
        let output = graph.rsqrt(reciprocal).unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::new([3], vec![-1.0, 0.0, 1.0]).unwrap(),
        )]);
        let actual = CpuJitBackend::new(JitFallback::Error)
            .execute_native(&graph, output, &inputs)
            .unwrap()
            .0;
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        for (actual, expected) in actual.values().iter().zip(expected.values()) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn native_float_extrema_reductions_match_oracle() {
        for (dtype, maximum) in [DType::F32, DType::F64]
            .into_iter()
            .flat_map(|dtype| [false, true].map(move |maximum| (dtype, maximum)))
        {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", Shape::from([3, 3]), dtype);
            let output = graph
                .reduce(
                    input,
                    if maximum {
                        crate::ReduceKind::Max
                    } else {
                        crate::ReduceKind::Min
                    },
                    Some(vec![1]),
                    true,
                )
                .unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [3, 3],
                    dtype,
                    [
                        f64::NAN,
                        2.0,
                        -1.0,
                        f64::NAN,
                        f64::NAN,
                        f64::NAN,
                        -0.0,
                        0.0,
                        0.0,
                    ]
                    .map(Scalar::F),
                )
                .unwrap(),
            )]);
            let actual = CpuJitBackend::new(JitFallback::Error)
                .execute_native(&graph, output, &inputs)
                .unwrap()
                .0;
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            assert_eq!(actual.storage(), expected.storage());
        }
    }

    #[test]
    fn vector_trace_covers_main_tail_and_scalar_fallbacks() {
        for len in [0usize, 1, 3, 4, 5, 8, 17] {
            let mut graph = Graph::new();
            let x = graph.input_dtype("x", Shape::from([len]), DType::F32);
            let y = graph.square(x).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [len],
                    DType::F32,
                    (0..len).map(|value| Scalar::F(value as f64 - 3.0)),
                )
                .unwrap(),
            )]);
            let backend = CpuJitBackend::new(JitFallback::Error).vectorized(true);
            let (actual, trace) = backend.execute_native(&graph, y, &inputs).unwrap();
            assert_eq!(
                actual.storage(),
                CpuBackend.execute(&graph, y, &inputs).unwrap().storage()
            );
            assert!(trace.vector.enabled);
            assert_eq!(trace.vector.lanes, 4);
            assert_eq!(trace.vector_main, len / 4 * 4);
            assert_eq!(trace.vector_tail, len % 4);
        }

        let mut scalar = Graph::new();
        let x = scalar.input_dtype("x", Shape::from([5]), DType::F64);
        let y = scalar.square(x).unwrap();
        let (_, trace) = CpuJitBackend::new(JitFallback::Error)
            .vectorized(true)
            .execute_native(
                &scalar,
                y,
                &HashMap::from([(
                    "x".into(),
                    TensorData::from_scalars([5], DType::F64, (0..5).map(|_| Scalar::F(1.0)))
                        .unwrap(),
                )]),
            )
            .unwrap();
        assert!(trace.vector.enabled);
        assert_eq!(trace.vector.lanes, 2);

        let mut broadcast = Graph::new();
        let lhs = broadcast.input_dtype("lhs", Shape::from([2, 3]), DType::F32);
        let rhs = broadcast.input_dtype("rhs", Shape::from([1, 3]), DType::F32);
        let out = broadcast.add(lhs, rhs).unwrap();
        let (_, trace) = CpuJitBackend::new(JitFallback::Error)
            .vectorized(true)
            .execute_native(
                &broadcast,
                out,
                &HashMap::from([
                    ("lhs".into(), TensorData::new([2, 3], vec![1.; 6]).unwrap()),
                    ("rhs".into(), TensorData::new([1, 3], vec![2.; 3]).unwrap()),
                ]),
            )
            .unwrap();
        assert!(!trace.vector.enabled);
        assert!(trace.vector.reason.contains("varying broadcast"));
    }
    #[test]
    fn unsupported_can_be_precise_or_fallback() {
        let mut g = Graph::new();
        let x = g.input("x", Shape::from([2]));
        let y = g.sin(x).unwrap();
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
