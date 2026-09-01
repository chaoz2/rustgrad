use crate::ir::PendingRandomReservation;
use crate::runtime::metal::{
    MetalCapabilities, MetalDevice, MetalPrefixPlan, MetalRenderer, PreparedMetalPrefix,
};
use crate::{
    Backend, BinaryOp, CompileTrace, CpuBackend, DType, DynamicInput, DynamicNodeId, Error,
    ExecutionPlanSummary, Graph, LiteralScalar, MappedTensor, MappedTensorError, MutableMappedFile,
    MutableMappedFileError, NodeId, Op, Result, Scalar, Shape, Slice, TensorData, UnaryOp,
    schedule,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

#[derive(Clone, Debug, PartialEq)]
struct CpuGradientSlot {
    shape: Shape,
    dtype: DType,
    value: TensorData,
}

/// Detached, session-authenticated persistent CPU gradients.
///
/// The store owns no graph nodes, aliases, or backend resources. A
/// [`CpuSession::backward`] transaction constructs lazy gradients on a private
/// graph candidate, realizes each unique requested target once, then commits
/// the complete detached result set here only after every descriptor and
/// accumulation has succeeded.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuGradientStore {
    session: u64,
    slots: BTreeMap<NodeId, CpuGradientSlot>,
}

impl CpuGradientStore {
    /// Number of unique targets with persistent gradient storage.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no target currently has persistent gradient storage.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// A graph-owned exact-cardinality CPU value.
///
/// Its concrete shape is produced by the runtime instruction DAG. The handle
/// exposes the exact shape expression and dtype, but no invented bound. Count
/// provenance is graph-local: it may be compared within this session, while
/// every operation still validates the owning session token.
#[derive(Clone, Debug)]
pub struct DynamicTensor {
    session: u64,
    node: DynamicNodeId,
    dtype: DType,
    shape: crate::DynamicOutputShape,
}

impl DynamicTensor {
    /// Storage dtype propagated by the exact runtime-buffer plan.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Exact runtime shape expression, including count provenance.
    pub fn shape_expression(&self) -> crate::DynamicOutputShape {
        self.shape
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

    /// Adds an immutable mapped source through its explicit owned CPU boundary.
    ///
    /// The mapping never becomes a `TensorData` storage alias: materialization
    /// finishes before this session mutates its Graph, and the resulting node is
    /// an ordinary constant with the existing no-autograd-input semantics.
    pub fn constant_mapped(&mut self, value: &MappedTensor) -> Result<Tensor> {
        let value = value.materialize_cpu().map_err(mapped_tensor_error)?;
        self.constant(value)
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
    /// The dynamic value composes through this session's pointwise unary,
    /// same-provenance binary, checked scalar, reduction, and realization
    /// methods. Its exact-cardinality first-order VJP is exposed separately;
    /// capture, artifacts, native JIT, devices, and general broadcasting remain
    /// deliberately unavailable.
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

    /// Returns source-compatible row-major nonzero coordinates with exact
    /// runtime shape `[count, input_rank]`.
    pub fn nonzero_dynamic(&mut self, input: &Tensor) -> Result<DynamicTensor> {
        let input = self.node(input)?;
        let node = self.graph.nonzero(input)?;
        self.dynamic_handle(node)
    }

    /// Negates one floating dynamic value through its exact runtime buffer.
    pub fn dynamic_neg(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        self.dynamic_unary(input, UnaryOp::Neg)
    }

    /// Squares one floating dynamic value through its exact runtime buffer.
    pub fn dynamic_square(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        self.dynamic_unary(input, UnaryOp::Square)
    }

    /// Adds one checked static scalar using source dtype promotion.
    pub fn dynamic_add_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Add)
    }

    /// Subtracts one checked static scalar using source dtype promotion.
    pub fn dynamic_sub_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Sub)
    }

    /// Multiplies by one checked static scalar using source dtype promotion.
    pub fn dynamic_mul_scalar(
        &mut self,
        input: &DynamicTensor,
        scalar: &Tensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_scalar_binary(input, scalar, BinaryOp::Mul)
    }

    /// Adds two dynamic values with the same runtime shape/count provenance.
    pub fn dynamic_add(
        &mut self,
        lhs: &DynamicTensor,
        rhs: &DynamicTensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_tensor_binary(lhs, rhs, BinaryOp::Add)
    }

    /// Subtracts same-cardinality dynamic values pointwise.
    pub fn dynamic_sub(
        &mut self,
        lhs: &DynamicTensor,
        rhs: &DynamicTensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_tensor_binary(lhs, rhs, BinaryOp::Sub)
    }

    /// Multiplies same-cardinality dynamic values pointwise.
    pub fn dynamic_mul(
        &mut self,
        lhs: &DynamicTensor,
        rhs: &DynamicTensor,
    ) -> Result<DynamicTensor> {
        self.dynamic_tensor_binary(lhs, rhs, BinaryOp::Mul)
    }

    /// Reduces a dynamic value to its exact typed sum scalar.
    pub fn dynamic_sum(&mut self, input: &DynamicTensor) -> Result<DynamicTensor> {
        let input = self.dynamic_node(input)?;
        let node = self.graph.dynamic_sum(input)?;
        self.dynamic_handle(node)
    }

    /// Reduces a dynamic value to its exact source-policy mean scalar.
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

    /// Creates an empty detached gradient store owned by this session.
    pub fn gradient_store(&self) -> CpuGradientStore {
        CpuGradientStore {
            session: self.graph.id(),
            slots: BTreeMap::new(),
        }
    }

    /// Constructs, realizes, accumulates, and atomically commits persistent
    /// gradients for an ordered target set.
    ///
    /// Exactly one [`Graph::gradient`] call builds the unique target gradients
    /// and exactly one [`CpuSession::realize_many`] call realizes their staged
    /// accumulations. Duplicate targets are projected back into request order
    /// without a second accumulation. `gradient` follows the source-facing
    /// explicit-seed contract; omitting it therefore requires a scalar loss.
    /// Targets are caller-supplied; the session has no ambient live-tensor
    /// registry and performs no automatic target discovery. Authentication is
    /// deterministic: store, loss, targets left-to-right, then optional seed.
    /// The derivative graph is private scratch state: success commits only
    /// detached values to `store`, while both success and failure leave this
    /// session's graph and bindings unchanged.
    pub fn backward(
        &self,
        store: &mut CpuGradientStore,
        loss: &Tensor,
        targets: &[&Tensor],
        gradient: Option<&Tensor>,
    ) -> Result<Vec<TensorData>> {
        self.validate_gradient_store(store)?;
        let loss = self.checked_gradient_node(loss)?;
        let ordered = targets
            .iter()
            .map(|target| self.checked_gradient_node(target))
            .collect::<Result<Vec<_>>>()?;
        let gradient = gradient
            .map(|value| self.checked_gradient_node(value))
            .transpose()?;
        let mut seen = BTreeSet::new();
        let unique = ordered
            .iter()
            .copied()
            .filter(|target| seen.insert(*target))
            .collect::<Vec<_>>();

        let mut candidate = self.transaction_candidate();
        let gradients = candidate.graph.gradient(loss, &unique, gradient)?;
        if gradients.len() != unique.len() {
            return Err(gradient_store_error(
                "gradient transform returned an incomplete target inventory",
            ));
        }

        let mut accumulated = Vec::with_capacity(unique.len());
        for (&target, gradient) in unique.iter().zip(gradients) {
            candidate.validate_gradient_node_descriptor(target, gradient)?;
            let accumulated_node = if let Some(slot) = store.slots.get(&target) {
                candidate.validate_gradient_slot(target, slot)?;
                let previous = candidate.graph.constant(slot.value.clone());
                let next = candidate.graph.add(previous, gradient)?;
                candidate.validate_gradient_node_descriptor(target, next)?;
                next
            } else {
                gradient
            };
            accumulated.push(candidate.handle(accumulated_node)?);
        }
        let accumulated_refs = accumulated.iter().collect::<Vec<_>>();
        let realized = candidate.realize_many(&accumulated_refs)?;
        if realized.len() != unique.len() {
            return Err(gradient_store_error(
                "gradient realization returned an incomplete target inventory",
            ));
        }

        let mut staged = store.clone();
        for (&target, value) in unique.iter().zip(realized) {
            candidate.validate_gradient_value(target, &value)?;
            staged.slots.insert(
                target,
                CpuGradientSlot {
                    shape: value.shape().clone(),
                    dtype: value.dtype(),
                    value,
                },
            );
        }
        candidate.validate_gradient_store(&staged)?;
        let projected = ordered
            .iter()
            .map(|target| {
                staged
                    .slots
                    .get(target)
                    .map(|slot| slot.value.clone())
                    .ok_or_else(|| gradient_store_error("staged gradient target is absent"))
            })
            .collect::<Result<Vec<_>>>()?;

        *store = staged;
        Ok(projected)
    }

    /// Returns detached stored gradients in target order. Duplicate targets
    /// repeat the same logical slot; missing targets remain `None`.
    pub fn gradients(
        &self,
        store: &CpuGradientStore,
        targets: &[&Tensor],
    ) -> Result<Vec<Option<TensorData>>> {
        self.validate_gradient_store(store)?;
        targets
            .iter()
            .map(|target| {
                let target = self.checked_gradient_node(target)?;
                Ok(store.slots.get(&target).map(|slot| slot.value.clone()))
            })
            .collect()
    }

    /// Clears every stored gradient in one commit, matching tinygrad's
    /// `grad = None` reset semantics.
    pub fn zero_grad(&self, store: &mut CpuGradientStore) -> Result<()> {
        self.clear_gradient_store(store)
    }

    fn clear_gradient_store(&self, store: &mut CpuGradientStore) -> Result<()> {
        self.validate_gradient_store(store)?;
        let mut staged = store.clone();
        staged.slots.clear();
        *store = staged;
        Ok(())
    }

    /// Realizes a tensor through the CPU semantic oracle and owned bindings.
    pub fn realize(&self, tensor: &Tensor) -> Result<TensorData> {
        CpuBackend.execute(&self.graph, self.node(tensor)?, &self.bindings)
    }

    /// Realizes an ordered tensor set through one shared scheduled CPU
    /// transaction. Every handle is authenticated before scheduling; shared
    /// producers execute once, while repeated handles retain their request
    /// positions in the returned values.
    pub fn realize_many(&self, tensors: &[&Tensor]) -> Result<Vec<TensorData>> {
        let outputs = tensors
            .iter()
            .map(|tensor| self.node(tensor))
            .collect::<Result<Vec<_>>>()?;
        CpuBackend
            .execute_many(&self.graph, &outputs, &self.bindings)
            .map(|realized| realized.outputs)
            .map_err(|error| Error::SessionRealization {
                reason: format!("shared CPU realization: {error}"),
            })
    }

    fn transaction_candidate(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            bindings: self.bindings.clone(),
            input_names: self.input_names.clone(),
            next_input: self.next_input,
        }
    }

    fn checked_gradient_node(&self, tensor: &Tensor) -> Result<NodeId> {
        let node = self.node(tensor)?;
        let shape = self.graph.shape(node)?;
        let dtype = self.graph.dtype(node)?;
        if shape != &tensor.shape || dtype != tensor.dtype {
            return Err(gradient_store_error(
                "tensor handle descriptor diverges from its graph node",
            ));
        }
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| {
                gradient_store_error("tensor handle descriptor byte extent overflows")
            })?;
        Ok(node)
    }

    fn validate_gradient_node_descriptor(&self, target: NodeId, gradient: NodeId) -> Result<()> {
        let target_shape = self.graph.shape(target)?;
        let target_dtype = self.graph.dtype(target)?;
        if self.graph.shape(gradient)? != target_shape
            || self.graph.dtype(gradient)? != target_dtype
        {
            return Err(gradient_store_error(
                "gradient node descriptor does not match its target",
            ));
        }
        target_shape
            .numel()?
            .checked_mul(target_dtype.itemsize())
            .ok_or_else(|| gradient_store_error("gradient node byte extent overflows"))?;
        Ok(())
    }

    fn validate_gradient_value(&self, target: NodeId, value: &TensorData) -> Result<()> {
        let shape = self.graph.shape(target)?;
        let dtype = self.graph.dtype(target)?;
        let bytes = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| gradient_store_error("gradient value byte extent overflows"))?;
        if value.shape() != shape
            || value.dtype() != dtype
            || value.len().checked_mul(value.dtype().itemsize()) != Some(bytes)
        {
            return Err(gradient_store_error(
                "realized gradient descriptor does not match its target",
            ));
        }
        Ok(())
    }

    fn validate_gradient_slot(&self, target: NodeId, slot: &CpuGradientSlot) -> Result<()> {
        if slot.value.shape() != &slot.shape || slot.value.dtype() != slot.dtype {
            return Err(gradient_store_error(
                "stored gradient metadata diverges from its value",
            ));
        }
        self.validate_gradient_value(target, &slot.value)
    }

    fn validate_gradient_store(&self, store: &CpuGradientStore) -> Result<()> {
        if store.session != self.graph.id() {
            return Err(gradient_store_error(
                "gradient store belongs to another CPU session",
            ));
        }
        for (&target, slot) in &store.slots {
            self.validate_gradient_slot(target, slot)?;
        }
        Ok(())
    }

    /// Realizes a CPU-session value into owned storage, then copies and syncs
    /// it through one checked mutable mapped-file window.
    ///
    /// Realization happens before the writer is touched. The writer validates
    /// the exact output shape, dtype, element offset, and byte extent before
    /// modifying its mapping; a validation or realization failure therefore
    /// leaves the mapped file unchanged. A later sync failure may leave copied
    /// mapped bytes dirty, but the exclusive owner remains usable for an
    /// explicit retry. This is owned CPU copying only: it creates no graph
    /// alias, autograd state, capture/artifact payload, or device backing.
    pub fn realize_to_mapped(
        &self,
        tensor: &Tensor,
        writer: &mut MutableMappedFile,
        offset_elements: usize,
    ) -> Result<()> {
        let value = self.realize(tensor)?;
        writer
            .write_tensor(
                offset_elements,
                value.shape().clone(),
                value.dtype(),
                &value,
            )
            .map_err(mapped_mutable_error)?;
        writer.sync().map_err(mapped_mutable_error)
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
        self.graph
            .pending_uniform_after_guard(guard, shape, dtype, 0)
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
        let node = self.graph.commit_pending_uniform(pending, guard_node)?;
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
        let plan = crate::ir::MultinomialPlan::new(&self.graph, input, samples, axis, replacement)?;
        let before = self.graph.node_count();
        let guard = self
            .graph
            .tensor_guard_distribution(input, plan.axis as isize)?;
        let mut pending = self.graph.pending_uniform_after_guard(
            guard,
            plan.random_shape.clone(),
            plan.dtype,
            0,
        )?;
        if let Err(error) = CpuBackend.execute(&self.graph, guard, &self.bindings) {
            self.graph.nodes.truncate(before);
            return Err(error);
        }
        let uniform = self.graph.commit_pending_uniform(&mut pending, guard)?;
        let output = self.graph.multinomial_from_uniform(guard, uniform, &plan)?;
        self.handle(output)
    }

    /// Realizes one exact-cardinality result through the CPU oracle.
    pub fn realize_dynamic(&self, tensor: &DynamicTensor) -> Result<TensorData> {
        Ok(CpuBackend
            .execute_dynamic(&self.graph, self.dynamic_node(tensor)?, &self.bindings)?
            .output)
    }

    /// Applies an exact realized upstream to one dynamic result and returns
    /// the first-order VJP in the requested static source descriptor.
    ///
    /// The output's count provenance remains dynamic: `upstream` must match
    /// the concrete result shape and dtype for this realization. Bool masks
    /// are cardinality inputs only and never receive a gradient.
    pub fn dynamic_vjp(
        &self,
        output: &DynamicTensor,
        upstream: &TensorData,
        target: &Tensor,
    ) -> Result<TensorData> {
        let output = self.dynamic_node(output)?;
        let target = self.node(target)?;
        let plan = self.graph.dynamic_vjp_plan(output, target)?;
        CpuBackend.execute_dynamic_vjp(&self.graph, &plan, upstream, &self.bindings)
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
        let plan =
            MetalPrefixPlan::plan_for_outputs(&schedule.items, &[node.index() as u64], renderer)
                .map_err(|e| Error::SessionTraining {
                    reason: format!("Metal preflight: {e}"),
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
        let dynamic = self.graph.dynamic_node(node)?;
        Ok(DynamicTensor {
            session: self.graph.id(),
            node,
            dtype: dynamic.dtype,
            shape: dynamic.output,
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
        if self.graph.shape(scalar)?.numel()? != 1 {
            return Err(Error::InvalidIndex);
        }
        let node =
            self.graph
                .dynamic_binary(input, DynamicInput::StaticScalar(scalar), operation)?;
        self.dynamic_handle(node)
    }

    fn dynamic_tensor_binary(
        &mut self,
        lhs: &DynamicTensor,
        rhs: &DynamicTensor,
        operation: BinaryOp,
    ) -> Result<DynamicTensor> {
        let lhs = self.dynamic_node(lhs)?;
        let rhs = self.dynamic_node(rhs)?;
        let node = self
            .graph
            .dynamic_binary(lhs, DynamicInput::Dynamic(rhs), operation)?;
        self.dynamic_handle(node)
    }
}

fn mapped_tensor_error(error: MappedTensorError) -> Error {
    Error::SessionTraining {
        reason: format!("mapped tensor materialization: {error:?}"),
    }
}

fn gradient_store_error(reason: impl Into<String>) -> Error {
    Error::SessionRealization {
        reason: format!("persistent gradient store: {}", reason.into()),
    }
}

fn mapped_mutable_error(error: MutableMappedFileError) -> Error {
    Error::SessionTraining {
        reason: format!("mapped tensor write: {error:?}"),
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

    #[test]
    fn mapped_constant_materializes_before_cpu_graph_insertion() {
        let path =
            std::env::temp_dir().join(format!("rustgrad-session-mapped-{}", std::process::id()));
        std::fs::write(&path, [0_u8, 0, 0x40, 0x40, 0, 0, 0x80, 0x40]).unwrap();
        let mapped = crate::MappedTensor::open(&path, [2], DType::F32).unwrap();
        let mut session = CpuSession::new();
        let input = session.constant_mapped(&mapped).unwrap();
        let output = session.add(&input, &input).unwrap();
        assert_eq!(
            session.realize(&output).unwrap().to_vec_f64(),
            vec![6.0, 8.0]
        );
        assert!(
            session
                .trace(&output)
                .unwrap()
                .to_string()
                .contains("constant")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cpu_realization_copies_to_checked_mapped_window_then_syncs() {
        let path = std::env::temp_dir().join(format!(
            "rustgrad-session-mapped-write-{}",
            std::process::id()
        ));
        let mut writer = crate::MutableMappedFile::create(&path, 12).unwrap();
        let mut session = CpuSession::new();
        let input = session.tensor([2], [1.0, 2.0]).unwrap();
        let output = session.add(&input, &input).unwrap();
        session.realize_to_mapped(&output, &mut writer, 1).unwrap();
        assert_eq!(
            writer.read_tensor(1, [2], DType::F32).unwrap().to_vec_f64(),
            vec![2.0, 4.0]
        );
        assert!(session.trace(&output).unwrap().to_string().contains("add"));
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn shared_cpu_realization_preserves_order_duplicates_and_raw_sources() {
        let mut session = CpuSession::new();
        let input = session.variable([2], [2.0, 3.0]).unwrap();
        let shared = session.mul(&input, &input).unwrap();
        let one = session.tensor([], [1.0]).unwrap();
        let left = session.add(&shared, &one).unwrap();
        let right = session.mul(&shared, &one).unwrap();
        let raw =
            TensorData::from_storage([2], crate::Storage::BF16(vec![0x8000, 0x7fc1])).unwrap();
        let source = session.variable_data(raw.clone()).unwrap();
        let empty = session
            .constant(TensorData::from_storage([0], crate::Storage::U32(vec![])).unwrap())
            .unwrap();

        let mut outputs = session
            .realize_many(&[&right, &source, &left, &right, &empty, &one])
            .unwrap();
        assert!(session.realize_many(&[]).unwrap().is_empty());
        assert_eq!(outputs.len(), 6);
        assert_eq!(outputs[0].to_vec_f64(), vec![4.0, 9.0]);
        assert_eq!(outputs[2].to_vec_f64(), vec![5.0, 10.0]);
        assert_eq!(outputs[0].storage(), outputs[3].storage());
        assert_eq!(outputs[1].storage(), raw.storage());
        assert_eq!(outputs[4].shape(), &Shape::new([0]));
        assert_eq!(outputs[4].dtype(), DType::U32);
        assert_eq!(outputs[5].shape(), &Shape::new([]));
        outputs[0]
            .assign(&TensorData::new([2], vec![11.0, 13.0]).unwrap())
            .unwrap();
        assert_eq!(outputs[0].to_vec_f64(), vec![11.0, 13.0]);
        assert_eq!(
            outputs[3].to_vec_f64(),
            vec![4.0, 9.0],
            "duplicate projections are detached owned values"
        );

        let requested = [
            right.node,
            source.node,
            left.node,
            right.node,
            empty.node,
            one.node,
        ];
        let schedule = crate::schedule_many(&session.graph, &requested).unwrap();
        assert_eq!(
            schedule
                .items
                .iter()
                .filter(|item| item.node == shared.node)
                .count(),
            1,
            "the shared producer has one scheduled execution"
        );
    }

    #[test]
    fn shared_cpu_realization_preflights_handles_and_publishes_no_partial_state() {
        let mut session = CpuSession::new();
        let input = session.variable([2], [2.0, 3.0]).unwrap();
        let good = session.mul(&input, &input).unwrap();
        let invalid_distribution = session.variable([2], [1.0, -1.0]).unwrap();
        let late_failure = session
            .tensor_guard_distribution(&invalid_distribution, 0)
            .unwrap();
        let node_count = session.graph.node_count();
        let bindings = session.bindings.clone();
        let error = session.realize_many(&[&good, &late_failure]).unwrap_err();
        assert!(matches!(&error, Error::SessionRealization { .. }));
        assert!(
            error
                .to_string()
                .starts_with("CPU session realization error:")
        );
        assert_eq!(session.graph.node_count(), node_count);
        assert_eq!(session.bindings, bindings);
        assert_eq!(session.realize(&good).unwrap().to_vec_f64(), vec![4.0, 9.0]);

        let foreign = CpuSession::new().tensor([1], [7.0]).unwrap();
        assert!(matches!(
            session.realize_many(&[&good, &foreign, &good]),
            Err(Error::SessionHandleMismatch { .. })
        ));
        assert_eq!(session.graph.node_count(), node_count);
        assert_eq!(session.bindings, bindings);
    }

    #[test]
    fn persistent_backward_accumulates_unique_targets_and_projects_aliases() {
        let mut session = CpuSession::new();
        let x = session.variable([2], [2.0, 3.0]).unwrap();
        let square = session.mul(&x, &x).unwrap();
        let loss = session.sum_all(&square).unwrap();
        let disconnected = session
            .constant(TensorData::zeros_with_dtype([0], DType::F16).unwrap())
            .unwrap();
        let mut store = session.gradient_store();
        let graph_nodes = session.graph.node_count();
        let bindings = session.bindings.clone();

        let first = session
            .backward(&mut store, &loss, &[&x, &x, &disconnected], None)
            .unwrap();
        assert_eq!(store.len(), 2, "duplicate targets own one slot");
        assert_eq!(first[0].to_vec_f64(), vec![4.0, 6.0]);
        assert_eq!(first[0], first[1]);
        assert_eq!(first[2].shape(), &Shape::new([0]));
        assert_eq!(first[2].dtype(), DType::F16);
        assert_eq!(session.graph.node_count(), graph_nodes);
        assert_eq!(session.bindings, bindings);
        let mut returned_snapshot = first[0].clone();
        returned_snapshot
            .assign(&TensorData::zeros([2]).unwrap())
            .unwrap();
        assert_eq!(returned_snapshot.to_vec_f64(), vec![0.0, 0.0]);
        assert_eq!(
            session
                .gradients(&store, &[&x])
                .unwrap()
                .remove(0)
                .unwrap()
                .to_vec_f64(),
            vec![4.0, 6.0],
            "returned values do not alias stored snapshots"
        );

        let disconnected_before = session
            .gradients(&store, &[&disconnected])
            .unwrap()
            .remove(0)
            .unwrap();
        let second = session
            .backward(&mut store, &loss, &[&x, &x], None)
            .unwrap();
        assert_eq!(second[0].to_vec_f64(), vec![8.0, 12.0]);
        assert_eq!(second[0], second[1]);
        assert_eq!(session.graph.node_count(), graph_nodes);
        assert_eq!(session.bindings, bindings);
        assert_eq!(
            session
                .gradients(&store, &[&disconnected])
                .unwrap()
                .remove(0)
                .unwrap(),
            disconnected_before,
            "unrequested stored gradients remain untouched"
        );
        let projected = session.gradients(&store, &[&x, &x, &disconnected]).unwrap();
        assert_eq!(projected[0], projected[1]);
        assert_eq!(projected[0].as_ref().unwrap().to_vec_f64(), vec![8.0, 12.0]);

        session.zero_grad(&mut store).unwrap();
        assert!(store.is_empty());
        assert_eq!(
            session.gradients(&store, &[&x, &disconnected]).unwrap(),
            vec![None, None]
        );

        let mut connected = CpuSession::new();
        let source = connected.variable([2], [2.0, 3.0]).unwrap();
        let connected_frozen = connected.tensor([2], [4.0, 5.0]).unwrap();
        let product = connected.mul(&source, &connected_frozen).unwrap();
        let loss = connected.sum_all(&product).unwrap();
        let mut store = connected.gradient_store();
        let gradient = connected
            .backward(&mut store, &loss, &[&connected_frozen], None)
            .unwrap();
        assert_eq!(gradient[0].to_vec_f64(), vec![2.0, 3.0]);

        let mut narrow = CpuSession::new();
        let x = narrow
            .variable_data(
                TensorData::from_scalars([2], DType::F16, [Scalar::F(2.0), Scalar::F(3.0)])
                    .unwrap(),
            )
            .unwrap();
        let square = narrow.mul(&x, &x).unwrap();
        let loss = narrow.sum_all(&square).unwrap();
        let mut store = narrow.gradient_store();
        let first = narrow.backward(&mut store, &loss, &[&x], None).unwrap();
        let second = narrow.backward(&mut store, &loss, &[&x], None).unwrap();
        assert_eq!(first[0].shape(), &Shape::new([2]));
        assert_eq!(first[0].dtype(), DType::F16);
        assert_eq!(second[0].dtype(), DType::F16);
        assert_eq!(second[0].to_vec_f64(), vec![8.0, 12.0]);
    }

    #[test]
    fn persistent_backward_preserves_seed_and_detach_zero_contracts() {
        let mut seeded = CpuSession::new();
        let x = seeded.variable([2], [2.0, 3.0]).unwrap();
        let output = seeded.mul(&x, &x).unwrap();
        let seed = seeded.tensor([2], [3.0, 4.0]).unwrap();
        let mut store = seeded.gradient_store();
        let gradient = seeded
            .backward(&mut store, &output, &[&x], Some(&seed))
            .unwrap();
        assert_eq!(gradient[0].to_vec_f64(), vec![12.0, 24.0]);

        let node_count = seeded.graph.node_count();
        let snapshot = store.clone();
        assert!(matches!(
            seeded.backward(&mut store, &output, &[&x], None),
            Err(Error::NonScalarLoss(shape)) if shape == Shape::new([2])
        ));
        assert_eq!(seeded.graph.node_count(), node_count);
        assert_eq!(store, snapshot);

        let scalar = seeded.sum_all(&output).unwrap();
        let empty_nodes = seeded.graph.node_count();
        let empty_snapshot = store.clone();
        assert_eq!(
            seeded.backward(&mut store, &scalar, &[], None).unwrap(),
            vec![]
        );
        assert_eq!(seeded.graph.node_count(), empty_nodes);
        assert_eq!(store, empty_snapshot);
        assert_eq!(
            seeded
                .backward(&mut store, &output, &[], Some(&seed))
                .unwrap(),
            vec![]
        );
        assert_eq!(seeded.graph.node_count(), empty_nodes);
        assert_eq!(store, empty_snapshot);

        let mut detached = CpuSession::new();
        let x = detached.variable([2], [5.0, 7.0]).unwrap();
        let detached_node = detached.graph.detach(x.node).unwrap();
        let detached_x = detached.handle(detached_node).unwrap();
        let square = detached.mul(&detached_x, &detached_x).unwrap();
        let loss = detached.sum_all(&square).unwrap();
        let mut store = detached.gradient_store();
        let gradient = detached
            .backward(&mut store, &loss, &[&detached_x, &x], None)
            .unwrap();
        assert_eq!(gradient[0].to_vec_f64(), vec![10.0, 14.0]);
        assert_eq!(gradient[1].to_vec_f64(), vec![0.0, 0.0]);
    }

    #[test]
    fn persistent_backward_and_store_lifecycle_fail_atomically() {
        let mut session = CpuSession::new();
        let x = session.variable([2], [2.0, 3.0]).unwrap();
        let square = session.mul(&x, &x).unwrap();
        let loss = session.sum_all(&square).unwrap();
        let mut store = session.gradient_store();
        session.backward(&mut store, &loss, &[&x], None).unwrap();

        let invalid = session.variable([1], [-1.0]).unwrap();
        let guard = session.tensor_guard_distribution(&invalid, 0).unwrap();
        let graph_nodes = session.graph.node_count();
        let bindings = session.bindings.clone();
        let snapshot = store.clone();
        assert!(matches!(
            session.backward(&mut store, &loss, &[&x], Some(&guard)),
            Err(Error::SessionRealization { .. })
        ));
        assert_eq!(session.graph.node_count(), graph_nodes);
        assert_eq!(session.bindings, bindings);
        assert_eq!(store, snapshot);

        let mut malformed = store.clone();
        malformed.slots.get_mut(&x.node).unwrap().shape = Shape::new([1, 2]);
        let malformed_snapshot = malformed.clone();
        assert!(matches!(
            session.zero_grad(&mut malformed),
            Err(Error::SessionRealization { .. })
        ));
        assert_eq!(malformed, malformed_snapshot);

        let mut foreign_session = CpuSession::new();
        let foreign = foreign_session.variable([2], [1.0, 1.0]).unwrap();
        let foreign_loss = foreign_session.sum_all(&foreign).unwrap();
        let mut foreign_store = foreign_session.gradient_store();
        let foreign_snapshot = foreign_store.clone();
        assert!(matches!(
            session.backward(&mut foreign_store, &loss, &[&x], None),
            Err(Error::SessionRealization { .. })
        ));
        assert_eq!(foreign_store, foreign_snapshot);
        assert!(matches!(
            session.gradients(&store, &[&foreign]),
            Err(Error::SessionHandleMismatch { .. })
        ));
        assert!(matches!(
            session.zero_grad(&mut foreign_store),
            Err(Error::SessionRealization { .. })
        ));

        let local_seed = session.tensor([], [1.0]).unwrap();
        assert!(matches!(
            session.backward(&mut store, &foreign_loss, &[&foreign], Some(&local_seed)),
            Err(Error::SessionHandleMismatch { actual, .. }) if actual == foreign_loss.session
        ));

        let mut seed_session = CpuSession::new();
        let foreign_seed = seed_session.tensor([], [1.0]).unwrap();
        assert_ne!(foreign.session, foreign_seed.session);
        assert!(matches!(
            session.backward(&mut store, &loss, &[&x, &foreign], Some(&foreign_seed)),
            Err(Error::SessionHandleMismatch { actual, .. }) if actual == foreign.session
        ));
    }

    #[test]
    fn cpu_to_mapped_preflight_failures_do_not_change_the_writer() {
        let path = std::env::temp_dir().join(format!(
            "rustgrad-session-mapped-write-failure-{}",
            std::process::id()
        ));
        let mut writer = crate::MutableMappedFile::create(&path, 4).unwrap();
        let mut session = CpuSession::new();
        let output = session.tensor([2], [1.0, 2.0]).unwrap();
        assert!(session.realize_to_mapped(&output, &mut writer, 0).is_err());
        assert_eq!(
            writer.read_tensor(0, [1], DType::F32).unwrap().to_vec_f64(),
            vec![0.0]
        );
        let mut foreign_session = CpuSession::new();
        let foreign = foreign_session.tensor([1], [3.0]).unwrap();
        assert!(session.realize_to_mapped(&foreign, &mut writer, 0).is_err());
        assert_eq!(
            writer.read_tensor(0, [1], DType::F32).unwrap().to_vec_f64(),
            vec![0.0]
        );
        drop(writer);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn dynamic_session_nonzero_exposes_exact_provenance_and_rejects_foreign_handles() {
        let mut session = CpuSession::new();
        let input = session.variable([3], [0.0, 2.0, 4.0]).unwrap();
        let output = session.nonzero_dynamic(&input).unwrap();
        assert!(matches!(
            output.shape_expression(),
            crate::DynamicOutputShape::CountRows { width: 1, .. }
        ));
        let realized = session.realize_dynamic(&output).unwrap();
        assert_eq!(realized.shape(), &Shape::from([2, 1]));
        assert_eq!(realized.dtype(), DType::I32);
        assert_eq!(realized.to_vec_f64(), vec![1.0, 2.0]);

        let foreign = CpuSession::new();
        assert!(matches!(
            foreign.realize_dynamic(&output),
            Err(Error::SessionHandleMismatch { .. })
        ));
    }

    #[test]
    fn dynamic_session_composes_shared_provenance_and_rejects_other_roots() {
        let mut session = CpuSession::new();
        let input = session.variable([3], [1.0, -2.0, 3.0]).unwrap();
        let mask = session
            .tensor_with_dtype(
                [3],
                DType::Bool,
                [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
            )
            .unwrap();
        let selected = session.masked_select_dynamic(&input, &mask).unwrap();
        let negated = session.dynamic_neg(&selected).unwrap();
        let squared = session.dynamic_square(&selected).unwrap();
        let combined = session.dynamic_add(&negated, &squared).unwrap();
        assert_eq!(
            session.realize_dynamic(&combined).unwrap().to_vec_f64(),
            vec![0.0, 6.0]
        );

        let unrelated = session.masked_select_dynamic(&input, &mask).unwrap();
        assert!(session.dynamic_add(&selected, &unrelated).is_err());

        let mut foreign = CpuSession::new();
        assert!(matches!(
            foreign.dynamic_neg(&selected),
            Err(Error::SessionHandleMismatch { .. })
        ));
    }

    #[test]
    fn dynamic_session_vjp_preserves_runtime_compaction_and_static_target_shape() {
        let mut session = CpuSession::new();
        let input = session
            .variable([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = session
            .tensor_with_dtype(
                [1, 3],
                DType::Bool,
                [Scalar::Bool(true), Scalar::Bool(false), Scalar::Bool(true)],
            )
            .unwrap();
        let selected = session.masked_select_dynamic(&input, &mask).unwrap();
        let upstream = TensorData::new([4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let gradient = session.dynamic_vjp(&selected, &upstream, &input).unwrap();
        assert_eq!(gradient.shape(), &Shape::from([2, 3]));
        assert_eq!(
            gradient.to_vec_f64(),
            vec![10.0, 0.0, 20.0, 30.0, 0.0, 40.0]
        );

        let foreign = CpuSession::new();
        assert!(matches!(
            foreign.dynamic_vjp(&selected, &upstream, &input),
            Err(Error::SessionHandleMismatch { .. })
        ));
    }

    #[test]
    fn session_grad_prunes_unrequested_nondifferentiable_branches() {
        let mut session = CpuSession::new();
        let input = session.variable([3], [2.0, 3.0, 5.0]).unwrap();
        let unrelated = session.variable([3], [0.2, 0.3, 0.5]).unwrap();
        let input_square = session.mul(&input, &input).unwrap();
        let input_loss = session.sum_all(&input_square).unwrap();
        let guarded = session.tensor_guard_distribution(&unrelated, 0).unwrap();
        let guarded_square = session.mul(&guarded, &guarded).unwrap();
        let guarded_loss = session.sum_all(&guarded_square).unwrap();
        let loss = session.add(&input_loss, &guarded_loss).unwrap();

        let gradient = session.grad(&loss, &input).unwrap();
        assert_eq!(
            session.realize(&gradient).unwrap().to_vec_f64(),
            vec![4.0, 6.0, 10.0]
        );
        assert!(matches!(
            session.grad(&loss, &unrelated),
            Err(Error::NonDifferentiableIndexing(
                "tensor guard gradient is not represented"
            ))
        ));
    }
}
