use crate::runtime::metal::{
    MetalCapabilities, MetalDevice, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix,
};
use crate::{
    Backend, BinaryOp, CompileTrace, CpuBackend, DType, DynamicInput, DynamicNodeId, Error,
    ExecutionPlanSummary, Graph, LiteralScalar, NodeId, Op, PendingRandomReservation, Result,
    Scalar, Shape, Slice, TensorData, UnaryOp, schedule,
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

/// A graph-owned exact-cardinality CPU value.
///
/// It can only be produced by [`CpuSession::masked_select_dynamic`] and then
/// consumed by this session's bounded F32 dynamic operations. It deliberately
/// has no static shape because its extent is fixed only after CPU realization.
#[derive(Clone, Debug)]
pub struct DynamicTensor {
    session: u64,
    node: DynamicNodeId,
    dtype: DType,
}

impl DynamicTensor {
    /// Storage dtype propagated by the exact runtime-buffer plan.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
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

    /// Adds a storage-less scalar literal, resolved against `lhs` before
    /// entering the ordinary graph binary path.
    pub fn add_literal(&mut self, lhs: &Tensor, rhs: LiteralScalar) -> Result<Tensor> {
        self.binary_literal(lhs, rhs, crate::BinaryOp::Add)
    }

    pub fn sub_literal(&mut self, lhs: &Tensor, rhs: LiteralScalar) -> Result<Tensor> {
        self.binary_literal(lhs, rhs, crate::BinaryOp::Sub)
    }

    pub fn mul_literal(&mut self, lhs: &Tensor, rhs: LiteralScalar) -> Result<Tensor> {
        self.binary_literal(lhs, rhs, crate::BinaryOp::Mul)
    }

    pub fn div_literal(&mut self, lhs: &Tensor, rhs: LiteralScalar) -> Result<Tensor> {
        self.binary_literal(lhs, rhs, crate::BinaryOp::Div)
    }

    /// Subtracts `rhs` from a storage-less scalar literal resolved against it.
    pub fn literal_sub(&mut self, lhs: LiteralScalar, rhs: &Tensor) -> Result<Tensor> {
        self.literal_binary(lhs, crate::BinaryOp::Sub, rhs)
    }

    /// Divides a storage-less scalar literal by `rhs` after resolution.
    pub fn literal_div(&mut self, lhs: LiteralScalar, rhs: &Tensor) -> Result<Tensor> {
        self.literal_binary(lhs, crate::BinaryOp::Div, rhs)
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

    /// Selects F32 values at a broadcast Bool mask into an exact rank-one CPU
    /// result. Its length is the row-major true-count at realization time.
    ///
    /// The dynamic value remains bounded to this session's `neg`, `square`,
    /// scalar `add`/`sub`/`mul`, `sum`, `mean`, and `realize_dynamic` methods.
    /// Capture, artifacts, native JIT, devices, arbitrary dynamic operations,
    /// and dynamic reverse mode remain deliberately unavailable.
    pub fn masked_select_dynamic(
        &mut self,
        input: &Tensor,
        mask: &Tensor,
    ) -> Result<DynamicTensor> {
        let input = self.node(input)?;
        let mask = self.node(mask)?;
        let dtype = self.graph.dtype(input)?;
        if dtype != DType::F32 {
            return Err(Error::InvalidElementwiseDType {
                op: "masked_select_dynamic",
                actual: dtype,
            });
        }
        let node = self.graph.masked_select_dynamic(input, mask)?;
        self.dynamic_handle(node)
    }

    /// Negates one bounded dynamic F32 value through its exact runtime buffer.
    pub fn dynamic_neg(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        self.dynamic_unary(input, UnaryOp::Neg)
    }

    /// Squares one bounded dynamic F32 value through its exact runtime buffer.
    pub fn dynamic_square(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        self.dynamic_unary(input, UnaryOp::Square)
    }

    /// Adds one static F32 scalar to every bounded dynamic F32 value.
    pub fn dynamic_add_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Add)
    }

    /// Subtracts one static F32 scalar from every bounded dynamic F32 value.
    pub fn dynamic_sub_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Sub)
    }

    /// Multiplies every bounded dynamic F32 value by one static F32 scalar.
    pub fn dynamic_mul_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Mul)
    }

    /// Reduces a bounded dynamic F32 value to its exact F32 sum scalar.
    pub fn dynamic_sum(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        let input = self.dynamic_node(input)?;
        let node = self.graph.dynamic_sum(input)?;
        self.dynamic_handle(node)
    }

    /// Reduces a bounded dynamic F32 value to its exact F32 mean scalar.
    pub fn dynamic_mean(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        let input = self.dynamic_node(input)?;
        let node = self.graph.dynamic_mean(input)?;
        self.dynamic_handle(node)
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

    /// Adds the explicit CPU-static distribution validation boundary while
    /// preserving the validated tensor value on successful realization.
    pub fn tensor_guard_distribution(&mut self, input: &Tensor, axis: isize) -> Result<Tensor> {
        let input = self.node(input)?;
        let node = self.graph.tensor_guard_distribution(input, axis)?;
        self.handle(node)
    }

    /// Draws from the released graph-owned implicit CPU Threefry stream.
    pub fn rand_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        let node = self.graph.rand_implicit(shape, dtype)?;
        self.handle(node)
    }

    /// Captures an unconsumed CPU implicit-uniform reservation gated by a
    /// TensorGuard tensor in this session. Capture/replay and non-CPU paths do
    /// not accept this continuation boundary.
    pub fn pending_uniform_after_guard(
        &self,
        guard: &Tensor,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<PendingRandomReservation> {
        let guard = self.node(guard)?;
        self.graph.pending_uniform_after_guard(guard, shape, dtype, 0)
    }

    /// Realizes only the guard and, when validation succeeds, atomically converts the
    /// pending candidate into an ordinary captured RandomStream graph node.
    pub fn commit_pending_uniform(
        &mut self,
        guard: &Tensor,
        pending: &mut PendingRandomReservation,
    ) -> Result<Tensor> {
        let guard_node = self.node(guard)?;
        let value = CpuBackend.execute(&self.graph, guard_node, &self.bindings)?;
        let _validated = value;
        let node = self
            .graph
            .commit_pending_uniform(
                pending,
                guard_node,
            )?;
        self.handle(node)
    }

    /// Samples I32 category indices through the CPU-static guarded implicit
    /// Threefry path. Invalid weights are realized and rejected before the
    /// pending reservation advances the graph-owned stream.
    pub fn multinomial_implicit(
        &mut self,
        input: &Tensor,
        samples: usize,
        axis: isize,
        replacement: bool,
    ) -> Result<Tensor> {
        let input = self.node(input)?;
        let plan = crate::ir::MultinomialPlan::new(
            &self.graph,
            input,
            samples,
            axis,
            replacement,
        )?;
        let guard = self.graph.tensor_guard_distribution(input, plan.axis as isize)?;
        let mut pending = self.graph.pending_uniform_after_guard(
            guard,
            plan.random_shape.clone(),
            plan.dtype,
            0,
        )?;
        CpuBackend.execute(&self.graph, guard, &self.bindings)?;
        let uniform = self.graph.commit_pending_uniform(&mut pending, guard)?;
        let output = self.graph.multinomial_from_uniform(guard, uniform, &plan)?;
        self.handle(output)
    }

    /// Realizes one bounded exact-cardinality result through the CPU oracle.
    pub fn realize_dynamic(&self, tensor: &DynamicTensor) -> Result<TensorData> {
        Ok(CpuBackend
            .execute_dynamic(&self.graph, self.dynamic_node(tensor)?, &self.bindings)?
            .output)
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

    fn binary_literal(
        &mut self,
        lhs: &Tensor,
        rhs: LiteralScalar,
        operation: crate::BinaryOp,
    ) -> Result<Tensor> {
        let lhs = self.node(lhs)?;
        let node = self.graph.binary_literal(operation, lhs, rhs)?;
        self.handle(node)
    }

    fn literal_binary(
        &mut self,
        lhs: LiteralScalar,
        operation: crate::BinaryOp,
        rhs: &Tensor,
    ) -> Result<Tensor> {
        let rhs = self.node(rhs)?;
        let node = self.graph.literal_binary(lhs, operation, rhs)?;
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

    fn dynamic_node(&self, tensor: &DynamicTensor) -> Result<DynamicNodeId> {
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

    fn dynamic_handle(&self, node: DynamicNodeId) -> Result<DynamicTensor> {
        Ok(DynamicTensor {
            session: self.graph.id(),
            node,
            dtype: self.graph.dynamic_node(node)?.dtype,
        })
    }

    fn dynamic_unary(
        &mut self,
        input: &DynamicTensor,
        operation: UnaryOp,
    ) -> Result<DynamicTensor> {
        let input = self.dynamic_node(input)?;
        let node = self.graph.dynamic_unary(input, operation)?;
        self.dynamic_handle(node)
    }

    fn dynamic_scalar_binary(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
        operation: BinaryOp,
    ) -> Result<DynamicTensor> {
        let input = self.dynamic_node(input)?;
        let scalar = self.node(scalar)?;
        let dtype = self.graph.dtype(scalar)?;
        if dtype != DType::F32 {
            return Err(Error::InvalidElementwiseDType {
                op: "dynamic_scalar_binary",
                actual: dtype,
            });
        }
        if self.graph.shape(scalar)?.numel()? != 1 {
            return Err(Error::InvalidIndex);
        }
        let node =
            self.graph
                .dynamic_binary(input, DynamicInput::StaticScalar(scalar), operation)?;
        self.dynamic_handle(node)
    }
}

#[cfg(test)]
mod literal_tests {
    use super::*;

    #[test]
    fn literal_methods_resolve_to_concrete_peer_dtypes_and_preserve_direction() {
        let cases = [
            (DType::Bool, LiteralScalar::Bool(true), DType::Bool),
            (DType::I8, LiteralScalar::I64(-1000), DType::I8),
            (DType::U16, LiteralScalar::U64(u64::MAX), DType::U16),
            (DType::F16, LiteralScalar::F64(-0.0), DType::F16),
            (DType::BF16, LiteralScalar::F64(f64::NAN), DType::BF16),
            (DType::F32, LiteralScalar::F64(1.0), DType::F32),
            (DType::F64, LiteralScalar::F64(1.0), DType::F64),
        ];
        for (dtype, literal, expected) in cases {
            let mut session = CpuSession::new();
            let input = session
                .tensor_with_dtype([1], dtype, [Scalar::I(1)])
                .unwrap();
            let output = session.add_literal(&input, literal).unwrap();
            assert_eq!(output.dtype(), expected, "{dtype:?}");
            let trace = session.trace(&output).unwrap().to_string();
            assert!(trace.contains("constant") && trace.contains("add"));
        }
    }

    #[test]
    fn reverse_literal_ops_keep_existing_graph_and_session_validation() {
        let mut session = CpuSession::new();
        let input = session.variable([1], [2.0]).unwrap();
        let sub = session.literal_sub(LiteralScalar::I64(5), &input).unwrap();
        let div = session
            .literal_div(LiteralScalar::F64(8.0), &input)
            .unwrap();
        assert_eq!(session.realize(&sub).unwrap().to_vec_f64(), vec![3.0]);
        assert_eq!(session.realize(&div).unwrap().to_vec_f64(), vec![4.0]);
        let shifted = session
            .add_literal(&input, LiteralScalar::F64(1.0))
            .unwrap();
        let loss = session.sum_all(&shifted).unwrap();
        let gradient = session.grad(&loss, &input).unwrap();
        assert_eq!(session.realize(&gradient).unwrap().to_vec_f64(), vec![1.0]);
        let foreign = CpuSession::new().variable([1], [1.0]).unwrap();
        assert!(matches!(
            session.add_literal(&foreign, LiteralScalar::I64(1)),
            Err(Error::SessionHandleMismatch { .. })
        ));
    }
}
