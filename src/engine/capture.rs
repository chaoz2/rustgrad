//! Immutable schedule capture and backend-neutral interpreter replay.
use crate::{Graph, NodeId, Op, Schedule, ScheduleItem, TensorData};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReplayInput {
    pub name: String,
    pub node: NodeId,
    pub desc: crate::BufferDesc,
}
#[derive(Clone, Debug)]
pub struct CapturedSchedule {
    pub items: Vec<ScheduleItem>,
    pub inputs: Vec<ReplayInput>,
    pub constants: BTreeMap<u64, TensorData>,
    pub quantized_constants: BTreeMap<u64, crate::QuantizedTensorData>,
    pub requested: Vec<u64>,
    pub identity: u64,
    pub(crate) symbolic: Option<super::symbolic::SymbolicSchema>,
    pub(crate) specialized_from: Option<super::symbolic::SpecializedFrom>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    Missing(String),
    Extra(String),
    Descriptor(String),
    Corrupt(String),
    Execute(String),
    Unsupported(String),
    Backend(String),
    Symbolic(String),
    Batch { invocation: usize, reason: String },
}
impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "replay error: {self:?}")
    }
}
impl std::error::Error for ReplayError {}
impl CapturedSchedule {
    pub fn capture(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
    ) -> Result<Self, ReplayError> {
        schedule
            .validate()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        if schedule.items.iter().any(ScheduleItem::is_effect) {
            return Err(ReplayError::Unsupported(
                "effect schedule capture is unsupported".into(),
            ));
        }
        if schedule
            .items
            .iter()
            .any(|item| matches!(item.kernel.kind(), crate::UOpKind::TensorGuard))
        {
            return Err(ReplayError::Unsupported(
                "tensor guard capture is unsupported".into(),
            ));
        }
        let mut inputs = BTreeMap::new();
        let mut constants = BTreeMap::new();
        let mut produced = BTreeSet::new();
        for item in &schedule.items {
            item.validate_input_bindings()
                .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
            for b in item.ordered_inputs() {
                match graph
                    .op(b.input_node)
                    .map_err(|e| ReplayError::Corrupt(e.to_string()))?
                {
                    Op::Input { name } => {
                        inputs.entry(name.clone()).or_insert(ReplayInput {
                            name: name.clone(),
                            node: b.input_node,
                            desc: b.desc.clone(),
                        });
                    }
                    Op::Constant(v) => {
                        constants.insert(b.desc.id, v.clone());
                    }
                    _ if produced.contains(&b.desc.id) => {}
                    _ if item.external_materializations.contains(&b.input_node) => {
                        let name = format!("@materialized/{}", b.desc.id);
                        inputs.entry(name.clone()).or_insert(ReplayInput {
                            name,
                            node: b.input_node,
                            desc: b.desc.clone(),
                        });
                    }
                    _ => {
                        return Err(ReplayError::Corrupt(format!(
                            "unproduced binding {}",
                            b.desc.id
                        )));
                    }
                }
            }
            produced.extend(item.outputs.iter().map(|output| output.id));
        }
        let inputs = inputs.into_values().collect::<Vec<_>>();
        let mut capture = Self {
            items: schedule.items.clone(),
            inputs,
            constants,
            quantized_constants: BTreeMap::new(),
            requested: requested.iter().map(|n| n.index() as u64).collect(),
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))?;
        Ok(capture)
    }

    /// Builds one graph-independent Llama-orientation quantized linear
    /// artifact. Packed weights remain an immutable typed constant and are
    /// never materialized as a dense graph tensor.
    pub fn capture_quantized_matmul(
        activation_name: impl Into<String>,
        activation: NodeId,
        weight_node: NodeId,
        output: NodeId,
        activation_shape: crate::Shape,
        weight: crate::QuantizedTensorData,
    ) -> Result<Self, ReplayError> {
        let activation_name = activation_name.into();
        if activation_name.is_empty() {
            return Err(ReplayError::Descriptor(
                "quantized activation name is empty".into(),
            ));
        }
        weight
            .validate()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let plan = crate::QuantizedMatmulPlan::new(
            activation,
            weight_node,
            output,
            activation_shape.clone(),
            weight.descriptor().clone(),
        )
        .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let kernel = crate::UOp::new(
            crate::UOpKind::Matmul,
            Some(crate::UType::scalar(crate::DType::F32)),
            Vec::new(),
            crate::UArg::QuantizedMatmul(Box::new(plan.clone())),
        );
        kernel
            .validate()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let activation_elements = activation_shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let output_elements = plan
            .output_shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let activation_desc = crate::BufferDesc {
            id: activation.index() as u64,
            shape: activation_shape,
            dtype: crate::DType::F32,
            bytes: activation_elements
                .checked_mul(crate::DType::F32.itemsize())
                .ok_or_else(|| ReplayError::Descriptor("activation byte overflow".into()))?,
            alignment: crate::DType::F32.itemsize(),
            read_only: true,
            view: None,
        };
        let output_desc = crate::BufferDesc {
            id: output.index() as u64,
            shape: plan.output_shape.clone(),
            dtype: crate::DType::F32,
            bytes: output_elements
                .checked_mul(crate::DType::F32.itemsize())
                .ok_or_else(|| ReplayError::Descriptor("output byte overflow".into()))?,
            alignment: crate::DType::F32.itemsize(),
            read_only: false,
            view: None,
        };
        let mut item = crate::ScheduleItem {
            id: 0,
            node: output,
            dependencies: Vec::new(),
            consumers: Vec::new(),
            inputs: vec![activation_desc.clone()],
            input_bindings: vec![crate::ScheduleInputBinding {
                input_node: activation,
                desc: activation_desc.clone(),
                abi_index: 0,
            }],
            quantized_input_bindings: vec![crate::QuantizedScheduleInputBinding {
                input_node: weight_node,
                desc: weight.descriptor().clone(),
                abi_index: 1,
            }],
            external_materializations: Vec::new(),
            outputs: crate::ScheduledOutputs::single(output_desc.clone()),
            output: output_desc,
            kernel,
            boundary: None,
            cache_key: 0,
        };
        item.cache_key = crate::schedule::item_cache_key(&item);
        item.validate_input_bindings()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let mut capture = Self {
            items: vec![item],
            inputs: vec![ReplayInput {
                name: activation_name,
                node: activation,
                desc: activation_desc,
            }],
            constants: BTreeMap::new(),
            quantized_constants: BTreeMap::from([(weight_node.index() as u64, weight)]),
            requested: vec![output.index() as u64],
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        Ok(capture)
    }

    /// Builds one graph-independent packed row-gather artifact. The ordered
    /// ABI is indices, packed weight, then output; only selected rows are
    /// decoded by either replay backend.
    pub fn capture_quantized_row_gather(
        indices_name: impl Into<String>,
        indices: NodeId,
        weight_node: NodeId,
        output: NodeId,
        indices_shape: crate::Shape,
        indices_dtype: crate::DType,
        weight: crate::QuantizedTensorData,
    ) -> Result<Self, ReplayError> {
        let indices_name = indices_name.into();
        if indices_name.is_empty() {
            return Err(ReplayError::Descriptor(
                "quantized gather indices name is empty".into(),
            ));
        }
        weight
            .validate()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let plan = crate::QuantizedRowGatherPlan::new(
            indices,
            weight_node,
            output,
            indices_shape.clone(),
            indices_dtype,
            &weight,
        )
        .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let kernel = crate::UOp::new(
            crate::UOpKind::Movement,
            Some(crate::UType::scalar(crate::DType::F32)),
            Vec::new(),
            crate::UArg::QuantizedRowGather(Box::new(plan.clone())),
        );
        kernel
            .validate()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let indices_elements = indices_shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let output_elements = plan
            .output_shape
            .numel()
            .map_err(|error| ReplayError::Descriptor(error.to_string()))?;
        let indices_desc = crate::BufferDesc {
            id: indices.index() as u64,
            shape: indices_shape,
            dtype: indices_dtype,
            bytes: indices_elements
                .checked_mul(indices_dtype.itemsize())
                .ok_or_else(|| ReplayError::Descriptor("indices byte overflow".into()))?,
            alignment: indices_dtype.itemsize().max(1),
            read_only: true,
            view: None,
        };
        let output_desc = crate::BufferDesc {
            id: output.index() as u64,
            shape: plan.output_shape.clone(),
            dtype: crate::DType::F32,
            bytes: output_elements
                .checked_mul(crate::DType::F32.itemsize())
                .ok_or_else(|| ReplayError::Descriptor("output byte overflow".into()))?,
            alignment: crate::DType::F32.itemsize(),
            read_only: false,
            view: None,
        };
        let mut item = crate::ScheduleItem {
            id: 0,
            node: output,
            dependencies: Vec::new(),
            consumers: Vec::new(),
            inputs: vec![indices_desc.clone()],
            input_bindings: vec![crate::ScheduleInputBinding {
                input_node: indices,
                desc: indices_desc.clone(),
                abi_index: 0,
            }],
            quantized_input_bindings: vec![crate::QuantizedScheduleInputBinding {
                input_node: weight_node,
                desc: weight.descriptor().clone(),
                abi_index: 1,
            }],
            external_materializations: Vec::new(),
            outputs: crate::ScheduledOutputs::single(output_desc.clone()),
            output: output_desc,
            kernel,
            boundary: None,
            cache_key: 0,
        };
        item.cache_key = crate::schedule::item_cache_key(&item);
        item.validate_input_bindings()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let mut capture = Self {
            items: vec![item],
            inputs: vec![ReplayInput {
                name: indices_name,
                node: indices,
                desc: indices_desc,
            }],
            constants: BTreeMap::new(),
            quantized_constants: BTreeMap::from([(weight_node.index() as u64, weight)]),
            requested: vec![output.index() as u64],
            identity: 0,
            symbolic: None,
            specialized_from: None,
        };
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        Ok(capture)
    }
    /// Captures a symbolic shape family from one validated concrete template.
    /// The original graph is used only to derive expressions and is never
    /// retained or reconstructed by replay.
    pub fn capture_symbolic(
        graph: &Graph,
        schedule: &Schedule,
        requested: &[NodeId],
        spec: &crate::SymbolicCaptureSpec,
        template_bindings: &BTreeMap<String, i64>,
    ) -> Result<Self, ReplayError> {
        let mut capture = Self::capture(graph, schedule, requested)?;
        capture.symbolic = Some(super::symbolic::build_schema(
            graph,
            schedule,
            &capture,
            spec,
            template_bindings,
        )?);
        capture.identity = crate::schedule::artifact::identity(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        crate::schedule::artifact::validate_capture(&capture)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        Ok(capture)
    }
    pub fn is_symbolic(&self) -> bool {
        self.symbolic.is_some()
    }
    pub fn symbolic_parameters(&self) -> &[crate::SymbolicParameter] {
        self.symbolic
            .as_ref()
            .map_or(&[], super::symbolic::SymbolicSchema::parameters)
    }
    pub fn symbolic_guards(&self) -> &[crate::SymbolicGuard] {
        self.symbolic
            .as_ref()
            .map_or(&[], super::symbolic::SymbolicSchema::guards)
    }
    /// Serializes this graph-independent executable schedule with bounded,
    /// checksummed typed descriptors and exact constant storage.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        if self
            .items
            .iter()
            .any(|item| matches!(item.kernel.kind(), crate::UOpKind::Sort))
        {
            return Err(ReplayError::Unsupported(
                "static sort capture serialization is unsupported".into(),
            ));
        }
        crate::schedule::artifact::encode(self).map_err(|e| ReplayError::Corrupt(e.to_string()))
    }

    /// Validates and reconstructs a graph-independent executable schedule.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        crate::schedule::artifact::decode(bytes).map_err(|e| ReplayError::Corrupt(e.to_string()))
    }

    pub fn replay(
        &self,
        provided: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<TensorData>, ReplayError> {
        Ok(crate::CapturedReplayExecutor::default()
            .replay(self, provided, crate::CapturedReplayOptions::default())?
            .outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, DType, Graph, Scalar, Shape};
    use std::collections::HashMap;

    #[test]
    fn tensor_guard_capture_rejects_before_capture_identity_is_created() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("weights", [2], DType::F32);
        let guard = graph.tensor_guard_distribution(input, 0).unwrap();
        let schedule = crate::schedule(&graph, guard).unwrap();
        assert!(matches!(
            CapturedSchedule::capture(&graph, &schedule, &[guard]),
            Err(ReplayError::Unsupported(reason)) if reason.contains("tensor guard")
        ));
    }

    #[test]
    fn capture_replays_without_graph_traversal() {
        let mut g = Graph::new();
        let x = g.input_dtype("x", Shape::from([3]), DType::F32);
        let y = g.square(x).unwrap();
        let s = crate::schedule(&g, y).unwrap();
        let c = CapturedSchedule::capture(&g, &s, &[y]).unwrap();
        let a = BTreeMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [Scalar::F(1.), Scalar::F(2.), Scalar::F(3.)],
            )
            .unwrap(),
        )]);
        let out = c.replay(&a).unwrap();
        let oracle = CpuBackend
            .execute(
                &g,
                y,
                &a.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            )
            .unwrap();
        assert_eq!(out[0].storage(), oracle.storage());
        assert!(matches!(
            c.replay(&BTreeMap::new()),
            Err(ReplayError::Missing(_))
        ));
        let mut extra = a;
        extra.insert("unexpected".into(), TensorData::scalar(0.0));
        assert!(matches!(c.replay(&extra), Err(ReplayError::Extra(_))));
    }

    fn replay_bytes_against_cpu(
        graph: &Graph,
        output: NodeId,
        provided: BTreeMap<String, TensorData>,
    ) {
        let schedule = crate::schedule(graph, output).unwrap();
        let capture = CapturedSchedule::capture(graph, &schedule, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        assert_eq!(bytes, capture.to_bytes().unwrap());
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, decoded.to_bytes().unwrap());
        let first = decoded.replay(&provided).unwrap();
        let second = decoded.replay(&provided).unwrap();
        let oracle = CpuBackend
            .execute(
                graph,
                output,
                &provided.clone().into_iter().collect::<HashMap<_, _>>(),
            )
            .unwrap();
        assert_eq!(first[0].storage(), oracle.storage());
        assert_eq!(second[0].storage(), oracle.storage());
    }

    #[test]
    fn serialized_view_and_reduction_replay_match_cpu() {
        let mut view_graph = Graph::new();
        let x = view_graph.input_dtype("x", Shape::from([4]), DType::I32);
        let view = view_graph.shrink(x, [(1, 4)]).unwrap();
        let doubled = view_graph.add(view, view).unwrap();
        replay_bytes_against_cpu(
            &view_graph,
            doubled,
            BTreeMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [4],
                    DType::I32,
                    [Scalar::I(2), Scalar::I(3), Scalar::I(5), Scalar::I(7)],
                )
                .unwrap(),
            )]),
        );

        let mut reduction_graph = Graph::new();
        let x = reduction_graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
        let reduced = reduction_graph
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        replay_bytes_against_cpu(
            &reduction_graph,
            reduced,
            BTreeMap::from([(
                "x".into(),
                TensorData::from_scalars(
                    [2, 3],
                    DType::F32,
                    [1., 2., 3., 4., 5., 6.].map(Scalar::F),
                )
                .unwrap(),
            )]),
        );
    }

    #[test]
    fn malformed_artifacts_fail_before_execution_and_matmul_replays() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2]));
        let y = graph.square(x).unwrap();
        let schedule = crate::schedule(&graph, y).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[y]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        for n in [0, 1, bytes.len() - 1] {
            assert!(matches!(
                CapturedSchedule::from_bytes(&bytes[..n]),
                Err(ReplayError::Corrupt(_))
            ));
        }
        let mut corrupt = bytes;
        corrupt[8] ^= 1;
        assert!(matches!(
            CapturedSchedule::from_bytes(&corrupt),
            Err(ReplayError::Corrupt(_))
        ));
        let mut stale = capture.clone();
        stale.items[0].dependencies.push(999);
        assert!(matches!(
            stale.replay(&BTreeMap::new()),
            Err(ReplayError::Corrupt(_))
        ));

        let mut matmul_graph = Graph::new();
        let a = matmul_graph.input("a", Shape::from([1, 2]));
        let b = matmul_graph.input("b", Shape::from([2, 1]));
        let product = matmul_graph.matmul(a, b).unwrap();
        let schedule = crate::schedule(&matmul_graph, product).unwrap();
        let capture = CapturedSchedule::capture(&matmul_graph, &schedule, &[product]).unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let provided = BTreeMap::from([
            ("a".into(), TensorData::new([1, 2], vec![2., 3.]).unwrap()),
            ("b".into(), TensorData::new([2, 1], vec![4., 5.]).unwrap()),
        ]);
        assert_eq!(decoded.replay(&provided).unwrap()[0].values(), &[23.]);
    }

    #[test]
    fn external_materialization_is_an_explicit_replay_input() {
        let mut graph = Graph::new();
        let left = graph.input("left", Shape::from([1, 2]));
        let right = graph.input("right", Shape::from([1, 2]));
        let addend = graph.input("addend", Shape::from([1, 4]));
        let joined = graph.concat([left, right], 1).unwrap();
        let output = graph.add(joined, addend).unwrap();
        let schedule =
            crate::schedule_with_external_materializations(&graph, &[output], &[joined]).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let external_name = format!("@materialized/{}", joined.index());
        assert!(decoded.inputs.iter().any(|x| x.name == external_name));
        let values = BTreeMap::from([
            (
                "addend".into(),
                TensorData::new([1, 4], vec![10., 20., 30., 40.]).unwrap(),
            ),
            (
                external_name,
                TensorData::new([1, 4], vec![1., 2., 3., 4.]).unwrap(),
            ),
        ]);
        assert_eq!(
            decoded.replay(&values).unwrap()[0].values(),
            &[11., 22., 33., 44.]
        );
    }
}
