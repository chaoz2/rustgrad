use crate::runtime::metal::{
    MetalCapabilities, MetalDevice, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix,
};
use crate::{
    Backend, CompileTrace, CpuBackend, DType, Error, ExecutionPlanSummary, Graph, NodeId, Op,
    Result, Scalar, Shape, Slice, TensorData, schedule,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct MetalSessionResult {
    output: TensorData,
    /// Cache identities actually compiled or loaded by this call.
    pub cache_keys: Vec<String>,
    /// Immutable logical execution evidence without resource identities.
    pub trace: MetalSessionTrace,
}

/// Deterministic handle-free trace for strict static Metal realization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalSessionTrace {
    pub logical_identity: u64,
    pub planned_item_ids: Vec<u64>,
    pub cache_keys: Vec<String>,
    pub capabilities: MetalCapabilities,
    pub zero_domain_skipped: bool,
}

impl MetalSessionTrace {
    fn new(
        node: NodeId,
        tensor: &Tensor,
        planned_item_ids: Vec<u64>,
        cache_keys: Vec<String>,
        capabilities: MetalCapabilities,
        zero_domain_skipped: bool,
    ) -> Self {
        let mut logical_identity = 0xcbf2_9ce4_8422_2325_u64;
        for byte in format!(
            "metal-session-v1:{node:?}:{:?}:{:?}:{planned_item_ids:?}:{cache_keys:?}:{capabilities:?}:{zero_domain_skipped}",
            tensor.shape, tensor.dtype,
        )
        .bytes()
        {
            logical_identity = (logical_identity ^ u64::from(byte)).wrapping_mul(0x1000_0000_01b3);
        }
        Self {
            logical_identity,
            planned_item_ids,
            cache_keys,
            capabilities,
            zero_domain_skipped,
        }
    }
}
impl MetalSessionResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
}

/// Explicit device selection for [`CpuSession`].
///
/// Only [`Self::Cpu`] is currently implemented. Other choices are rejected at
/// construction; the session never falls back to the CPU silently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionDevice {
    Cpu,
    Cuda,
    OpenCl,
    Metal,
    WebGpu,
}

impl SessionDevice {
    const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::OpenCl => "opencl",
            Self::Metal => "metal",
            Self::WebGpu => "webgpu",
        }
    }
}

/// A graph-owned tensor value created by a [`CpuSession`].
#[derive(Clone, Debug)]
pub struct Tensor {
    session: u64,
    node: NodeId,
    shape: Shape,
    dtype: DType,
}

impl Tensor {
    /// Concrete, static shape propagated by the underlying graph.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Storage dtype propagated by the underlying graph.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// An explicit CPU realization session for ordinary Rust tensor workflows.
/// Constants need no binding. [`Self::variable`] creates a graph input and
/// retains its owned binding; [`Self::set`] only accepts the same shape/dtype.
#[derive(Debug)]
pub struct CpuSession {
    graph: Graph,
    bindings: HashMap<String, TensorData>,
    input_names: HashMap<NodeId, String>,
    next_input: usize,
}

impl Default for CpuSession {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuSession {
    /// Resolves the immutable inputs of a canonical static schedule without
    /// exposing session bindings or graph node IDs outside this module.
    #[allow(dead_code)]
    pub(crate) fn metal_schedule_values(
        &self,
        tensor: &Tensor,
    ) -> Result<(crate::Schedule, HashMap<u64, TensorData>)> {
        let output = self.node(tensor)?;
        let schedule = schedule(&self.graph, output).map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
        let mut values = HashMap::new();
        for item in &schedule.items {
            for binding in item.ordered_inputs() {
                if values.contains_key(&binding.desc.id) {
                    continue;
                }
                let value = match self.graph.op(binding.input_node)? {
                    Op::Input { name } => {
                        self.bindings
                            .get(name)
                            .cloned()
                            .ok_or_else(|| Error::SessionTraining {
                                reason: format!("missing session binding {name}"),
                            })?
                    }
                    Op::Constant(value) => value.clone(),
                    _ => continue,
                };
                if value.shape() != &binding.desc.shape
                    || value.dtype() != binding.desc.dtype
                    || value.len().checked_mul(value.dtype().itemsize()) != Some(binding.desc.bytes)
                {
                    return Err(Error::SessionTraining {
                        reason: "session Metal schedule descriptor mismatch".into(),
                    });
                }
                values.insert(binding.desc.id, value);
            }
        }
        Ok((schedule, values))
    }
    /// Creates a CPU-only session.
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            bindings: HashMap::new(),
            input_names: HashMap::new(),
            next_input: 0,
        }
    }

    /// Selects the backend explicitly. Non-CPU choices fail rather than falling back.
    pub fn on(device: SessionDevice) -> Result<Self> {
        match device {
            SessionDevice::Cpu => Ok(Self::new()),
            other => Err(Error::UnsupportedSessionDevice {
                device: other.name(),
            }),
        }
    }

    /// Creates an F32 graph constant from ordinary Rust values.
    pub fn tensor(
        &mut self,
        shape: impl Into<Shape>,
        values: impl IntoIterator<Item = f32>,
    ) -> Result<Tensor> {
        self.constant(TensorData::new(shape, values.into_iter().collect())?)
    }

    /// Creates a typed graph constant from exact scalar values.
    pub fn tensor_with_dtype(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        values: impl IntoIterator<Item = Scalar>,
    ) -> Result<Tensor> {
        self.constant(TensorData::from_scalars(shape, dtype, values)?)
    }

    /// Adds owned tensor data as a graph constant.
    pub fn constant(&mut self, value: TensorData) -> Result<Tensor> {
        let node = self.graph.constant(value);
        self.handle(node)
    }

    /// Adds an F32 input bound for CPU realization and reverse mode.
    pub fn variable(
        &mut self,
        shape: impl Into<Shape>,
        values: impl IntoIterator<Item = f32>,
    ) -> Result<Tensor> {
        self.variable_data(TensorData::new(shape, values.into_iter().collect())?)
    }

    /// Adds an owned input binding. Floating dtypes retain existing reverse mode.
    pub fn variable_data(&mut self, value: TensorData) -> Result<Tensor> {
        let name = format!("session_input_{}", self.next_input);
        self.next_input = self.next_input.checked_add(1).ok_or(Error::InvalidIndex)?;
        let node = self
            .graph
            .input_dtype(name.clone(), value.shape().clone(), value.dtype());
        self.bindings.insert(name.clone(), value);
        self.input_names.insert(node, name);
        self.handle(node)
    }

    /// Rebinds a session variable after exact shape and dtype validation.
    pub fn set(&mut self, tensor: &Tensor, value: TensorData) -> Result<()> {
        let node = self.node(tensor)?;
        let name = self
            .input_names
            .get(&node)
            .ok_or(Error::InvalidIndex)?
            .clone();
        if tensor.shape != *value.shape() {
            return Err(Error::InputShape {
                name,
                expected: tensor.shape.clone(),
                actual: value.shape().clone(),
            });
        }
        if tensor.dtype != value.dtype() {
            return Err(Error::InputDType {
                name,
                expected: tensor.dtype,
                actual: value.dtype(),
            });
        }
        self.bindings.insert(name, value);
        Ok(())
    }

    /// Adds two values using the graph's broadcast and promotion rules.
    pub fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.binary(lhs, rhs, Graph::add)
    }

    /// Subtracts two values using the graph's broadcast and promotion rules.
    pub fn sub(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.binary(lhs, rhs, Graph::sub)
    }

    /// Multiplies two values using the graph's broadcast and promotion rules.
    pub fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.binary(lhs, rhs, Graph::mul)
    }

    /// Divides two values using the graph's broadcast and promotion rules.
    pub fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.binary(lhs, rhs, Graph::div)
    }

    /// Applies the existing typed ReLU graph operation.
    pub fn relu(&mut self, input: &Tensor) -> Result<Tensor> {
        self.unary(input, Graph::relu)
    }

    /// Performs graph matmul with its checked shape and dtype contract.
    pub fn matmul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        self.binary(lhs, rhs, Graph::matmul)
    }

    /// Materializes a checked reshape through the existing graph operation.
    pub fn reshape(&mut self, input: &Tensor, shape: impl Into<Shape>) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.reshape(input, shape)?;
        self.handle(node)
    }

    /// Permutes axes using the graph's checked permutation contract.
    pub fn permute(&mut self, input: &Tensor, axes: impl Into<Vec<usize>>) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.permute(input, axes)?;
        self.handle(node)
    }

    /// Swaps two axes through the checked permutation operation.
    pub fn transpose(&mut self, input: &Tensor, first: usize, second: usize) -> Result<Tensor> {
        let input = self.node(input)?;
        let rank = self.graph.shape(input)?.rank();
        if first >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: first,
                rank,
            });
        }
        if second >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: second,
                rank,
            });
        }
        let mut axes = (0..rank).collect::<Vec<_>>();
        axes.swap(first, second);
        let node = self.graph.permute(input, axes)?;
        self.handle(node)
    }

    /// Takes checked half-open bounds for every axis.
    pub fn shrink(
        &mut self,
        input: &Tensor,
        bounds: impl Into<Vec<(usize, usize)>>,
    ) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.shrink(input, bounds)?;
        self.handle(node)
    }

    /// Applies Python-style signed static slices, including negative steps.
    pub fn slice(&mut self, input: &Tensor, slices: impl Into<Vec<Slice>>) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.slice(input, slices)?;
        self.handle(node)
    }

    /// Concatenates two or more session values along a checked axis.
    pub fn concat(&mut self, inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
        let nodes = inputs
            .iter()
            .map(|input| self.node(input))
            .collect::<Result<Vec<_>>>()?;
        let node = self.graph.concat(nodes, axis)?;
        self.handle(node)
    }

    /// Gathers values with an integer session tensor along a checked axis.
    pub fn gather(&mut self, input: &Tensor, index: &Tensor, axis: usize) -> Result<Tensor> {
        self.binary_axis(input, index, axis, Graph::gather)
    }

    /// Computes a numerically stable softmax over one signed static axis.
    pub fn softmax(&mut self, input: &Tensor, axis: isize) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.softmax(input, axis, None)?;
        self.handle(node)
    }

    /// Returns first-tie argmax indices as an I32 tensor with the reduced axis removed.
    pub fn argmax(&mut self, input: &Tensor, axis: isize) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.argmax(input, Some(axis), false)?;
        self.handle(node)
    }

    /// Reduces all axes to a scalar using established graph semantics.
    pub fn sum_all(&mut self, input: &Tensor) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.sum_all(input)?;
        self.handle(node)
    }

    /// Builds a first-order gradient node through the pure reverse-mode API.
    pub fn grad(&mut self, loss: &Tensor, wrt: &Tensor) -> Result<Tensor> {
        let loss = self.node(loss)?;
        let wrt = self.node(wrt)?;
        let node = self.graph.grad(loss, wrt)?;
        self.handle(node)
    }

    /// Realizes a tensor through the CPU semantic oracle and owned bindings.
    pub fn realize(&self, tensor: &Tensor) -> Result<TensorData> {
        CpuBackend.execute(&self.graph, self.node(tensor)?, &self.bindings)
    }

    /// Strict static Metal realization. It preflights the complete schedule
    /// before queue, cache, pipeline, or buffer creation and never falls back.
    pub fn realize_metal(
        &self,
        tensor: &Tensor,
        device: MetalDevice,
        renderer: MetalRenderer,
    ) -> Result<MetalSessionResult> {
        let node = self.node(tensor)?;
        if renderer.capabilities != device.info().capabilities {
            return Err(Error::SessionTraining {
                reason: "Metal preflight: renderer/device capability identity mismatch".into(),
            });
        }
        let capabilities = device.info().capabilities.clone();
        let (schedule, values) = self.metal_schedule_values(tensor)?;
        let plan = MetalPrefixPlan::plan(&schedule.items, renderer).map_err(|e| {
            Error::SessionTraining {
                reason: format!("Metal preflight: {e}"),
            }
        })?;
        let planned_item_ids = schedule.items.iter().map(|item| item.id).collect();
        if tensor.shape.numel().map_err(|_| Error::InvalidIndex)? == 0 {
            let trace = MetalSessionTrace::new(
                node,
                tensor,
                planned_item_ids,
                Vec::new(),
                capabilities,
                true,
            );
            return Ok(MetalSessionResult {
                output: TensorData::zeros_with_dtype(tensor.shape.clone(), tensor.dtype)?,
                cache_keys: Vec::new(),
                trace,
            });
        }
        let cache_keys = plan.cache_keys();
        let prepared =
            PreparedMetalPrefix::from_plan(device, plan).map_err(|e| Error::SessionTraining {
                reason: format!("Metal prepare: {e}"),
            })?;
        let mut values = values
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        prepared
            .execute(&mut values)
            .map_err(|e| Error::SessionTraining {
                reason: format!("Metal execute: {e}"),
            })?;
        let output =
            values
                .get(&(node.index() as u64))
                .cloned()
                .ok_or_else(|| Error::SessionTraining {
                    reason: "Metal output missing".into(),
                })?;
        let trace = MetalSessionTrace::new(
            node,
            tensor,
            planned_item_ids,
            cache_keys.clone(),
            capabilities,
            false,
        );
        Ok(MetalSessionResult {
            output,
            cache_keys,
            trace,
        })
    }

    /// Returns the deterministic graph trace for a session tensor.
    pub fn trace(&self, tensor: &Tensor) -> Result<CompileTrace> {
        self.graph.trace(self.node(tensor)?)
    }

    /// Returns immutable schedule and logical-memory facts for one session
    /// output without executing the graph or inspecting bound tensor bytes.
    pub fn execution_summary(
        &self,
        tensor: &Tensor,
        reuse_enabled: bool,
    ) -> Result<ExecutionPlanSummary> {
        let node = self.node(tensor)?;
        ExecutionPlanSummary::from_graph(&self.graph, &[node], reuse_enabled).map_err(|error| {
            Error::SessionTraining {
                reason: format!("execution summary: {error}"),
            }
        })
    }

    /// Exposes inspectable graph structure without exposing session bindings.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    fn binary(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        operation: fn(&mut Graph, NodeId, NodeId) -> Result<NodeId>,
    ) -> Result<Tensor> {
        let lhs = self.node(lhs)?;
        let rhs = self.node(rhs)?;
        let node = operation(&mut self.graph, lhs, rhs)?;
        self.handle(node)
    }

    fn unary(
        &mut self,
        input: &Tensor,
        operation: fn(&mut Graph, NodeId) -> Result<NodeId>,
    ) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = operation(&mut self.graph, input)?;
        self.handle(node)
    }

    fn binary_axis(
        &mut self,
        input: &Tensor,
        other: &Tensor,
        axis: usize,
        operation: fn(&mut Graph, NodeId, NodeId, usize) -> Result<NodeId>,
    ) -> Result<Tensor> {
        let input = self.node(input)?;
        let other = self.node(other)?;
        let node = operation(&mut self.graph, input, other, axis)?;
        self.handle(node)
    }

    fn node(&self, tensor: &Tensor) -> Result<NodeId> {
        if tensor.session != self.graph.id() {
            return Err(Error::SessionHandleMismatch {
                expected: self.graph.id(),
                actual: tensor.session,
            });
        }
        Ok(tensor.node)
    }

    fn handle(&self, node: NodeId) -> Result<Tensor> {
        Ok(Tensor {
            session: self.graph.id(),
            node,
            shape: self.graph.shape(node)?.clone(),
            dtype: self.graph.dtype(node)?,
        })
    }
}
