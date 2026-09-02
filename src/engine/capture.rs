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
    pub requested_passthroughs: Vec<crate::RequestedPassthrough>,
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
        // Generic captured schedules have no durable representation for a
        // persistent-state version or view. Validate the complete source
        // schedule first, then keep that stateful ABI on the mixed capture
        // route instead of silently reducing it to an ordinary replay input.
        schedule
            .validate()
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        if schedule.items.iter().any(ScheduleItem::is_effect) {
            return Err(ReplayError::Unsupported(
                "effect schedule capture is unsupported".into(),
            ));
        }
        if !schedule.state_bindings.is_empty() {
            return Err(ReplayError::Unsupported(
                "state-bound schedule capture requires the mixed capture route".into(),
            ));
        }
        if schedule
            .items
            .iter()
            .any(|item| matches!(item.kernel.operation(), crate::Operation::TensorGuard(_)))
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
        let requested_ids = requested
            .iter()
            .map(|node| node.index() as u64)
            .collect::<BTreeSet<_>>();
        for passthrough in &schedule.requested_passthroughs {
            passthrough
                .validate_against_graph(graph)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
            let requested_id = passthrough.requested.index() as u64;
            if !requested_ids.contains(&requested_id) || produced.contains(&requested_id) {
                return Err(ReplayError::Corrupt(
                    "requested passthrough does not belong to this capture".into(),
                ));
            }
            let mut source_desc = passthrough.desc.clone();
            source_desc.view = None;
            match graph
                .op(passthrough.source)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?
            {
                Op::Input { name } => {
                    inputs.entry(name.clone()).or_insert(ReplayInput {
                        name: name.clone(),
                        node: passthrough.source,
                        desc: source_desc,
                    });
                }
                Op::Constant(value) => {
                    constants.insert(source_desc.id, value.clone());
                }
                _ => {
                    return Err(ReplayError::Corrupt(
                        "requested passthrough source is not immutable storage".into(),
                    ));
                }
            }
        }
        // A source-owned requested value has no scheduled producer. Preserve
        // it in the existing replay input/constant ownership tables so replay
        // can return the exact caller value without fabricating an aliasing
        // kernel item. Any computed requested value still requires one unique
        // scheduled producer and fails closed below.
        for node in requested {
            let id = node.index() as u64;
            if produced.contains(&id) {
                continue;
            }
            if schedule
                .requested_passthroughs
                .iter()
                .any(|passthrough| passthrough.requested == *node)
            {
                continue;
            }
            let shape = graph
                .shape(*node)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?
                .clone();
            let dtype = graph
                .dtype(*node)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
            let bytes = shape
                .numel()
                .map_err(|error| ReplayError::Descriptor(error.to_string()))?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| ReplayError::Descriptor("requested value byte overflow".into()))?;
            let desc = crate::BufferDesc {
                id,
                shape,
                dtype,
                bytes,
                alignment: dtype.itemsize().max(1),
                read_only: true,
                view: None,
            };
            match graph
                .op(*node)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?
            {
                Op::Input { name } => {
                    inputs.entry(name.clone()).or_insert(ReplayInput {
                        name: name.clone(),
                        node: *node,
                        desc,
                    });
                }
                Op::Constant(value) => {
                    constants.insert(id, value.clone());
                }
                _ => {
                    return Err(ReplayError::Corrupt(format!(
                        "requested value {id} has no scheduled producer"
                    )));
                }
            }
        }
        let inputs = inputs.into_values().collect::<Vec<_>>();
        let mut capture = Self {
            items: schedule.items.clone(),
            inputs,
            constants,
            quantized_constants: BTreeMap::new(),
            requested_passthroughs: schedule.requested_passthroughs.clone(),
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
        let kernel = crate::UOp::from_operation(
            crate::Operation::Matmul(crate::MatmulValue::Quantized(Box::new(plan.clone()))),
            Some(crate::UType::scalar(crate::DType::F32)),
            Vec::new(),
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
            outputs: crate::ScheduledOutputs::single(output_desc),
            kernel,
            boundary: None,
            cache_key: 0,
        };
        item.cache_key = crate::schedule::item_cache_key(&item)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
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
            requested_passthroughs: vec![],
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
        let kernel = crate::UOp::from_operation(
            crate::Operation::Movement(crate::MovementValue::QuantizedRowGather(Box::new(
                plan.clone(),
            ))),
            Some(crate::UType::scalar(crate::DType::F32)),
            Vec::new(),
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
            outputs: crate::ScheduledOutputs::single(output_desc),
            kernel,
            boundary: None,
            cache_key: 0,
        };
        item.cache_key = crate::schedule::item_cache_key(&item)
            .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
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
            requested_passthroughs: vec![],
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
        // Validate the caller's concrete schedule first, but authenticate the
        // symbolic family against an explicit movement boundary. Redirection
        // is a concrete ownership optimization; retaining Contiguous here
        // keeps specialization shapes, buffer schemas, and historical replay
        // structure independent of that optimization.
        let _validated = Self::capture(graph, schedule, requested)?;
        let external = schedule
            .items
            .iter()
            .flat_map(|item| item.external_materializations.iter())
            .map(|node| node.index())
            .collect::<BTreeSet<_>>();
        let symbolic_schedule =
            crate::schedule::schedule_many_for_symbolic_capture(graph, requested, &external)
                .map_err(|error| ReplayError::Corrupt(error.to_string()))?;
        let mut capture = Self::capture(graph, &symbolic_schedule, requested)?;
        capture.symbolic = Some(super::symbolic::build_schema(
            graph,
            &symbolic_schedule,
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
            .any(|item| matches!(item.kernel.operation(), crate::Operation::Sort(_)))
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

    /// Serializes the distinct inspection-only scheduled-output envelope.
    /// Unlike [`Self::to_bytes`], it can preserve a canonical ordered output
    /// collection, but such a capture remains unavailable to replay until a
    /// coupled producer ABI exists.
    pub fn to_scheduled_outputs_bytes(&self) -> Result<Vec<u8>, ReplayError> {
        crate::schedule::artifact::encode_scheduled_outputs(self)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))
    }

    /// Decodes an inspection-only scheduled-output envelope. The result is
    /// intentionally still rejected by normal replay validation when it
    /// contains more than one output for an item.
    pub fn from_scheduled_outputs_bytes(bytes: &[u8]) -> Result<Self, ReplayError> {
        crate::schedule::artifact::decode_scheduled_outputs(bytes)
            .map_err(|e| ReplayError::Corrupt(e.to_string()))
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
    use crate::{
        Backend, BufferState, CpuBackend, DType, Graph, Scalar, ScheduleStateBinding, Shape,
        TensorData, bind_schedule_states,
    };
    use std::collections::HashMap;

    #[test]
    fn tensor_guard_capture_rejects_before_capture_identity_is_created() {
        let mut graph = Graph::new();
        let input = graph.constant(
            TensorData::from_scalars([2], DType::F32, [Scalar::F(1.0), Scalar::F(1.0)]).unwrap(),
        );
        let guard = graph.tensor_guard_distribution(input, 0).unwrap();
        let schedule = crate::schedule(&graph, guard).unwrap();
        assert!(matches!(
            CapturedSchedule::capture(&graph, &schedule, &[guard]),
            Err(ReplayError::Unsupported(reason)) if reason.contains("tensor guard")
        ));
    }

    #[test]
    fn source_passthrough_capture_roundtrips_and_replays_without_items() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([3]), DType::F32);
        let constant_value = TensorData::from_scalars(
            [3],
            DType::F32,
            [
                Scalar::F(-0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap();
        let constant = graph.constant(constant_value.clone());
        let provided_value = TensorData::from_scalars(
            [3],
            DType::F32,
            [Scalar::F(1.0), Scalar::F(-2.0), Scalar::F(3.0)],
        )
        .unwrap();

        let requested = [input, constant];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert!(schedule.items.is_empty());
        let capture = CapturedSchedule::capture(&graph, &schedule, &[input, constant]).unwrap();
        assert!(capture.items.is_empty());
        assert_eq!(capture.inputs.len(), 1);
        assert_eq!(capture.constants.len(), 1);

        let bytes = capture.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::capture(&graph, &schedule, &requested)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let provided = BTreeMap::from([("input".into(), provided_value.clone())]);
        let replay = crate::CapturedReplayExecutor::default()
            .replay(
                &decoded,
                &provided,
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: true },
                },
            )
            .unwrap();
        assert!(replay.trace.items.is_empty());
        assert_eq!(replay.outputs[0].storage(), provided_value.storage());
        let raw_f32 = |value: &TensorData| -> Vec<u32> {
            match value.storage() {
                crate::Storage::F32(values) => values.iter().map(|value| value.to_bits()).collect(),
                _ => unreachable!("fixture is F32"),
            }
        };
        assert_eq!(raw_f32(&replay.outputs[1]), raw_f32(&constant_value));
    }

    #[test]
    fn affine_source_passthrough_roundtrips_with_mixed_computed_outputs() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([2, 3]), DType::F32);
        let input_view = graph.permute(input, [1, 0]).unwrap();
        let constant_value =
            TensorData::from_storage([2, 2], crate::Storage::U16(vec![0, u16::MAX, 0x8000, 7]))
                .unwrap();
        let constant = graph.constant(constant_value.clone());
        let constant_view = graph.permute(constant, [1, 0]).unwrap();
        let computed = graph.neg(input_view).unwrap();
        let alias_schedule = crate::schedule(&graph, input_view).unwrap();
        assert!(alias_schedule.items.is_empty());
        let requested = [input_view, constant_view, computed];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert_eq!(schedule.items.len(), 1);
        assert_eq!(schedule.requested_passthroughs.len(), 2);
        let capture = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
        assert_eq!(capture.requested_passthroughs.len(), 2);
        assert_eq!(capture.inputs.len(), 1);
        assert!(capture.constants.contains_key(&(constant.index() as u64)));

        let bytes = capture.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::capture(&graph, &schedule, &requested)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let input_value = TensorData::from_storage(
            [2, 3],
            crate::Storage::F32(vec![
                -0.0,
                f32::from_bits(0x7fc0_1234),
                2.0,
                f32::INFINITY,
                -3.0,
                f32::NEG_INFINITY,
            ]),
        )
        .unwrap();
        let alias_capture =
            CapturedSchedule::capture(&graph, &alias_schedule, &[input_view]).unwrap();
        let alias_replay = crate::CapturedReplayExecutor::default()
            .replay(
                &alias_capture,
                &BTreeMap::from([("input".into(), input_value.clone())]),
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert!(alias_replay.trace.items.is_empty());
        let replay = crate::CapturedReplayExecutor::default()
            .replay(
                &decoded,
                &BTreeMap::from([("input".into(), input_value)]),
                crate::CapturedReplayOptions {
                    backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
                },
            )
            .unwrap();
        assert_eq!(replay.trace.items.len(), 1);
        let crate::Storage::F32(input_lanes) = replay.outputs[0].storage() else {
            panic!("F32 passthrough")
        };
        assert_eq!(
            input_lanes
                .iter()
                .map(|lane| lane.to_bits())
                .collect::<Vec<_>>(),
            vec![
                (-0.0f32).to_bits(),
                f32::INFINITY.to_bits(),
                0x7fc0_1234,
                (-3.0f32).to_bits(),
                2.0f32.to_bits(),
                f32::NEG_INFINITY.to_bits(),
            ]
        );
        assert_eq!(
            replay.outputs[1].storage(),
            &crate::Storage::U16(vec![0, 0x8000, u16::MAX, 7])
        );

        let mut malformed = decoded;
        malformed.requested_passthroughs[0].source = input_view;
        assert!(matches!(malformed.to_bytes(), Err(ReplayError::Corrupt(_))));
    }

    #[test]
    fn mixed_source_and_computed_requests_keep_order_and_unique_ownership() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([2]), DType::I32);
        let constant_value =
            TensorData::from_scalars([2], DType::I32, [Scalar::I(4), Scalar::I(-1)]).unwrap();
        let constant = graph.constant(constant_value.clone());
        let computed = graph.add(input, constant).unwrap();
        let requested = [constant, computed, input];
        let schedule = crate::schedule_many(&graph, &requested).unwrap();
        assert_eq!(schedule.items.len(), 1);
        let capture = CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
        let decoded = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
        let input_value =
            TensorData::from_scalars([2], DType::I32, [Scalar::I(2), Scalar::I(3)]).unwrap();
        let outputs = decoded
            .replay(&BTreeMap::from([("input".into(), input_value.clone())]))
            .unwrap();

        assert_eq!(outputs[0].storage(), constant_value.storage());
        assert_eq!(outputs[1].storage(), &crate::Storage::I32(vec![6, 2]));
        assert_eq!(outputs[2].storage(), input_value.storage());

        let mut malformed = decoded;
        malformed.requested.push(999);
        assert!(matches!(
            malformed.replay(&BTreeMap::from([("input".into(), input_value)])),
            Err(ReplayError::Corrupt(_))
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

    #[test]
    fn redirected_contiguous_capture_roundtrips_one_owned_producer() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", Shape::from([3]), DType::F32);
        let producer = graph.square(input).unwrap();
        let output = graph.contiguous(producer).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        assert_eq!(schedule.items.len(), 1);
        assert_eq!(schedule.items[0].node, output);
        assert!(matches!(
            schedule.items[0].kernel.operation(),
            crate::Operation::Sink
        ));
        assert!(
            schedule.items[0]
                .kernel
                .topological()
                .unwrap()
                .iter()
                .all(|node| !matches!(node.operation(), crate::Operation::Movement(_)))
        );

        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let bytes = capture.to_bytes().unwrap();
        let decoded = CapturedSchedule::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert_eq!(decoded.items.len(), 1);
        assert_eq!(decoded.items[0].node, output);
        assert_eq!(decoded.items[0].primary_output().id, output.index() as u64);
        let value = TensorData::new([3], vec![-2.0, 0.0, 3.0]).unwrap();
        let replayed = decoded
            .replay(&BTreeMap::from([("input".into(), value)]))
            .unwrap();
        assert_eq!(replayed.len(), 1);
        assert_eq!(
            replayed[0].storage(),
            &crate::Storage::F32(vec![4.0, 0.0, 9.0])
        );
    }

    #[test]
    fn contiguous_capture_preserves_raw_bytes_across_redirect_and_copy_routes() {
        use crate::{Float8Format, Float8Storage, Storage};

        let values = vec![
            Storage::Float8(Float8Storage::from_raw(
                Float8Format::E4M3,
                vec![0x80, 0x7f],
            )),
            Storage::Float8(Float8Storage::from_raw(
                Float8Format::E5M2,
                vec![0x80, 0x7d],
            )),
            Storage::Float8(Float8Storage::from_raw(
                Float8Format::E4M3FNUZ,
                vec![0x80, 0xff],
            )),
            Storage::Float8(Float8Storage::from_raw(
                Float8Format::E5M2FNUZ,
                vec![0x80, 0xff],
            )),
            Storage::F16(vec![0x8000, 0x7e01]),
            Storage::BF16(vec![0x8000, 0x7fc1]),
            Storage::F32(vec![
                f32::from_bits(0x8000_0000),
                f32::from_bits(0x7fc0_0001),
            ]),
            Storage::F64(vec![
                f64::from_bits(0x8000_0000_0000_0000),
                f64::from_bits(0x7ff8_0000_0000_0001),
            ]),
            Storage::U8(vec![0, u8::MAX]),
            Storage::U16(vec![0, u16::MAX]),
            Storage::U32(vec![0, u32::MAX]),
            Storage::U64(vec![0, u64::MAX]),
            Storage::I8(vec![i8::MIN, i8::MAX]),
            Storage::I16(vec![i16::MIN, i16::MAX]),
            Storage::I32(vec![i32::MIN, i32::MAX]),
            Storage::I64(vec![i64::MIN, i64::MAX]),
            Storage::Bool(vec![false, true]),
        ];
        for storage in values {
            let dtype = storage.dtype();
            let value = TensorData::from_storage([2], storage).unwrap();
            let expected = value.to_le_bytes().unwrap();
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let producer = graph.detach(input).unwrap();
            let output = graph.contiguous(producer).unwrap();
            let schedule = crate::schedule(&graph, output).unwrap();
            let portable = matches!(dtype, DType::Bool | DType::I32 | DType::U32 | DType::F32);
            assert_eq!(
                schedule.items.len(),
                if portable { 1 } else { 2 },
                "{dtype:?}"
            );
            let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
            let replayed = capture
                .replay(&BTreeMap::from([("input".into(), value)]))
                .unwrap();
            assert_eq!(replayed[0].to_le_bytes().unwrap(), expected, "{dtype:?}");
        }
    }

    #[test]
    fn generic_capture_rejects_versioned_state_bindings_without_artifact_creation() {
        let mut graph = Graph::new();
        let state_input = graph.input_dtype("state", Shape::from([2]), DType::F32);
        let output = graph.square(state_input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let input = schedule.items[0]
            .input_bindings
            .iter()
            .find(|binding| binding.input_node == state_input)
            .unwrap()
            .clone();
        let binding = ScheduleStateBinding {
            state: BufferState {
                buffer: 41,
                version: 7,
                shape: Shape::from([2]),
                dtype: DType::F32,
                bytes: 8,
            },
            view: None,
            consumer_item: 0,
            consumer_node: output,
            input_node: state_input,
            desc: input.desc,
            abi_index: input.abi_index,
        };
        let state_bound = bind_schedule_states(schedule.clone(), vec![binding.clone()]).unwrap();
        let repeated = bind_schedule_states(schedule.clone(), vec![binding.clone()]).unwrap();
        let canonical = [(91, 4)];
        let specialized_base =
            crate::schedule::specialized_item_cache_key(&schedule.items[0], 77, &canonical)
                .unwrap();
        let specialized_state =
            crate::schedule::state_bound_item_cache_key(specialized_base, &[&binding]).unwrap();
        let mut specialized_items = schedule.items.clone();
        crate::schedule::rekey_schedule_items(
            &mut specialized_items,
            std::slice::from_ref(&binding),
            Some((77, &canonical)),
        )
        .unwrap();
        assert_eq!(specialized_items[0].cache_key, specialized_state);
        assert_ne!(specialized_items[0].cache_key, specialized_base);
        let mut next_version = binding;
        next_version.state.version += 1;
        let changed = bind_schedule_states(schedule, vec![next_version]).unwrap();
        let item_keys = state_bound
            .items
            .iter()
            .map(|item| item.cache_key)
            .collect::<Vec<_>>();
        assert_eq!(state_bound.items[0].cache_key, repeated.items[0].cache_key);
        assert_ne!(state_bound.items[0].cache_key, changed.items[0].cache_key);

        assert!(matches!(
            CapturedSchedule::capture(&graph, &state_bound, &[output]),
            Err(ReplayError::Unsupported(message))
                if message == "state-bound schedule capture requires the mixed capture route"
        ));
        assert_eq!(
            state_bound
                .items
                .iter()
                .map(|item| item.cache_key)
                .collect::<Vec<_>>(),
            item_keys
        );
        assert_eq!(state_bound.state_bindings.len(), 1);
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
