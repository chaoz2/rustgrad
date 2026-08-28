//! Typed, owned bindings and a portable interpreter for elementwise UOp kernels.
//!
//! This is intentionally not a backend: bindings clone their `TensorData`, so a
//! scheduled kernel cannot retain or alias a caller's storage.  Element offsets
//! are checked separately from byte offsets, which keeps the ABI boundary
//! explicit for future renderers.
use crate::{
    BinaryOp, CompareOp, DType, Error, Graph, LogicalOp, NodeId, Op, Result, Scalar, Shape,
    Storage, SymbolicShape, SymbolicVar, TensorData, UArg, UOp, UOpError, UOpKind, UType, UnaryOp,
};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum BufferRole {
    Input,
    Output,
    Constant,
    Temporary,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum KernelShape {
    Concrete(Shape),
    Symbolic(SymbolicShape),
}
impl KernelShape {
    pub fn bind(
        &self,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> std::result::Result<Shape, crate::SymbolicError> {
        match self {
            Self::Concrete(s) => Ok(s.clone()),
            Self::Symbolic(s) => s.bind(bindings),
        }
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct KernelBufferDesc {
    pub id: u64,
    pub role: BufferRole,
    pub dtype: DType,
    pub lanes: u16,
    pub shape: KernelShape,
    pub bytes: usize,
    pub alignment: usize,
    pub mutable: bool,
    pub address_space: crate::AddressSpace,
}
impl KernelBufferDesc {
    pub fn concrete(
        id: u64,
        role: BufferRole,
        shape: Shape,
        dtype: DType,
        mutable: bool,
    ) -> Result<Self> {
        let elements = shape.numel()?;
        let bytes = elements
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        Ok(Self {
            id,
            role,
            dtype,
            lanes: 1,
            shape: KernelShape::Concrete(shape),
            bytes,
            alignment: dtype.itemsize().max(1),
            mutable,
            address_space: crate::AddressSpace::Global,
        })
    }
    pub fn byte_offset(&self, element: usize) -> Result<usize> {
        let offset = element
            .checked_mul(self.dtype.itemsize())
            .ok_or(Error::InvalidIndex)?;
        if offset % self.alignment != 0 || offset >= self.bytes && self.bytes != 0 {
            return Err(Error::InvalidIndex);
        }
        Ok(offset)
    }
}

/// A normalized row-major output domain and a broadcasted input offset map.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IterationPlan {
    pub output: Shape,
    pub reduce_axes: Vec<usize>,
}

/// Separates the retained output coordinates from the axes traversed by a
/// reduction.  Both domains are row-major and remain meaningful for scalars
/// and zero-sized dimensions.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReductionPlan {
    pub input: Shape,
    pub output: Shape,
    pub axes: Vec<usize>,
    pub keepdim: bool,
    pub reduction: Shape,
}
impl ReductionPlan {
    pub fn new(input: Shape, output: Shape, axes: Vec<usize>, keepdim: bool) -> Result<Self> {
        if axes.windows(2).any(|w| w[0] >= w[1]) || axes.iter().any(|axis| *axis >= input.rank()) {
            return Err(Error::InvalidIndex);
        }
        let reduction = Shape::new(
            axes.iter()
                .map(|axis| input.dims()[*axis])
                .collect::<Vec<_>>(),
        );
        Ok(Self {
            input,
            output,
            axes,
            keepdim,
            reduction,
        })
    }
    pub fn output_len(&self) -> Result<usize> {
        self.output.numel()
    }
    pub fn reduction_len(&self) -> Result<usize> {
        self.reduction.numel()
    }
    pub fn input_linear(&self, output_linear: usize, reduce_linear: usize) -> Result<usize> {
        let output_coords = IterationPlan::new(self.output.clone()).coords(output_linear)?;
        let reduction_coords = IterationPlan::new(self.reduction.clone()).coords(reduce_linear)?;
        let mut input_coords = vec![0; self.input.rank()];
        let mut out_axis = 0;
        let mut reduce_axis = 0;
        for (axis, input_coord) in input_coords.iter_mut().enumerate() {
            if self.axes.contains(&axis) {
                *input_coord = reduction_coords[reduce_axis];
                reduce_axis += 1;
                if self.keepdim {
                    out_axis += 1;
                }
            } else {
                *input_coord = output_coords[out_axis];
                out_axis += 1;
            }
        }
        let mut linear = 0usize;
        for (coord, dim) in input_coords.iter().zip(self.input.dims()) {
            linear = linear
                .checked_mul(*dim)
                .and_then(|v| v.checked_add(*coord))
                .ok_or(Error::InvalidIndex)?;
        }
        Ok(linear)
    }
}
impl IterationPlan {
    pub fn new(output: Shape) -> Self {
        Self {
            output,
            reduce_axes: vec![],
        }
    }
    pub fn len(&self) -> Result<usize> {
        self.output.numel()
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    pub fn coords(&self, mut linear: usize) -> Result<Vec<usize>> {
        if linear >= self.len()? {
            return Err(Error::InvalidIndex);
        }
        let mut out = vec![0; self.output.rank()];
        for axis in (0..out.len()).rev() {
            let d = self.output.dims()[axis];
            if d != 0 {
                out[axis] = linear % d;
                linear /= d;
            }
        }
        Ok(out)
    }
    pub fn broadcast_offset(&self, input: &Shape, linear: usize) -> Result<usize> {
        if input.rank() > self.output.rank()
            || !input
                .dims()
                .iter()
                .rev()
                .zip(self.output.dims().iter().rev())
                .all(|(a, b)| *a == 1 || a == b)
        {
            return Err(Error::InvalidIndex);
        }
        let coords = self.coords(linear)?;
        let pad = self.output.rank() - input.rank();
        let mut offset = 0usize;
        for (axis, dim) in input.dims().iter().enumerate() {
            let coord = if *dim == 1 { 0 } else { coords[pad + axis] };
            offset = offset
                .checked_mul(*dim)
                .and_then(|x| x.checked_add(coord))
                .ok_or(Error::InvalidIndex)?;
        }
        Ok(offset)
    }
}

#[derive(Clone, Debug, Default)]
pub struct KernelBindings {
    values: BTreeMap<u64, TensorData>,
}
impl KernelBindings {
    pub fn insert(&mut self, desc: &KernelBufferDesc, value: TensorData) -> Result<()> {
        let shape = match &desc.shape {
            KernelShape::Concrete(s) => s,
            KernelShape::Symbolic(_) => {
                return Err(Error::Serialization {
                    reason: "unbound symbolic kernel buffer".into(),
                });
            }
        };
        if value.shape() != shape
            || value.dtype() != desc.dtype
            || value.len().checked_mul(value.dtype().itemsize()) != Some(desc.bytes)
        {
            return Err(Error::InvalidData {
                shape: shape.clone(),
                expected: shape.numel()?,
                actual: value.len(),
            });
        }
        self.values.insert(desc.id, value);
        Ok(())
    }
    pub fn get(&self, id: u64) -> Option<&TensorData> {
        self.values.get(&id)
    }
    pub fn into_buffer(self, id: u64) -> Option<TensorData> {
        self.values.get(&id).cloned()
    }
}

pub fn lower_graph_elementwise(
    graph: &Graph,
    output: NodeId,
) -> std::result::Result<UOp, UOpError> {
    lower_graph_elementwise_with_materialized(graph, output, &std::collections::BTreeSet::new())
}

/// Lowers a captured graph random source into a self-contained kernel.  This
/// is intentionally a source UOp rather than a load: the plan carries the
/// exact reservation selected when the graph node was created.
pub fn lower_graph_random(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    let Op::Random { kind, stream } = graph
        .op(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let plan = crate::random::plan::RandomKernelPlan::new(
        output,
        graph
            .shape(output)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .clone(),
        graph
            .dtype(output)
            .map_err(|_| UOpError::UseBeforeDefinition)?,
        *kind,
        *stream,
    )
    .map_err(|_| UOpError::InvalidArgument)?;
    let kernel = UOp::new(
        UOpKind::Random,
        Some(UType::scalar(plan.dtype)),
        vec![],
        UArg::Random(Box::new(plan)),
    );
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers one static generalized matmul into its authoritative typed UOp
/// semantic. The payload is already normalized and owns the pointer ABI.
pub fn lower_graph_matmul(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    let plan = crate::MatmulKernelPlan::from_graph(graph, output)
        .map_err(|_| UOpError::InvalidArgument)?;
    let dtype = plan.dtype;
    let target =
        crate::MatmulTargetCaps::conservative_ptx(80).map_err(|_| UOpError::InvalidArgument)?;
    let arg = match crate::TensorCoreMatmulPayload::select(plan.clone(), target.clone())
        .map_err(|_| UOpError::InvalidArgument)?
    {
        Some(payload) => {
            crate::plan_tensor_core_matmul_promotion(&payload)
                .map_err(|_| UOpError::InvalidArgument)?;
            UArg::TensorCoreMatmul(Box::new(payload))
        }
        None => match crate::TiledMatmulPayload::select(plan.clone(), target)
            .map_err(|_| UOpError::InvalidArgument)?
        {
            Some(payload) => {
                crate::plan_tiled_matmul_promotion(&payload)
                    .map_err(|_| UOpError::InvalidArgument)?;
                UArg::TiledMatmul(Box::new(payload))
            }
            None => UArg::Matmul(Box::new(plan)),
        },
    };
    let kernel = UOp::new(UOpKind::Matmul, Some(UType::scalar(dtype)), vec![], arg);
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers one materializing concat/gather/scatter operation into its validated
/// shared movement payload. Native renderers consume its ordered operand ABI.
pub fn lower_graph_movement(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    let plan = crate::MovementKernelPlan::from_graph(graph, output)
        .map_err(|_| UOpError::InvalidArgument)?;
    let kernel = UOp::new(
        UOpKind::Movement,
        Some(UType::scalar(plan.dtype)),
        vec![],
        UArg::Movement(Box::new(plan)),
    );
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers an elementwise region while treating already-scheduled producers as
/// typed loads. This preserves the UOp ABI and lets the schedule DAG prevent
/// duplicate computation of a shared producer.
pub(crate) fn lower_graph_elementwise_with_materialized(
    graph: &Graph,
    output: NodeId,
    materialized: &std::collections::BTreeSet<usize>,
) -> std::result::Result<UOp, UOpError> {
    let output_shape = graph
        .shape(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
        .clone();
    let output_ty = UType::scalar(
        graph
            .dtype(output)
            .map_err(|_| UOpError::UseBeforeDefinition)?,
    );
    let extent = output_shape
        .numel()
        .map_err(|_| UOpError::InvalidArgument)?;
    let extent_i64 = i64::try_from(extent).map_err(|_| UOpError::InvalidArgument)?;
    let range = UOp::new(
        UOpKind::Range,
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(extent_i64, UType::scalar(DType::I64))],
        UArg::RangeAxis(0),
    );
    fn load(
        graph: &Graph,
        id: NodeId,
        out: &Shape,
        range: &UOp,
        view: Option<crate::uop::AffineView>,
    ) -> std::result::Result<UOp, UOpError> {
        let shape = graph
            .shape(id)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .clone();
        let ty = UType::scalar(graph.dtype(id).map_err(|_| UOpError::UseBeforeDefinition)?);
        let elements = shape.numel().map_err(|_| UOpError::InvalidArgument)?;
        let address = UOp::new(
            UOpKind::DefineGlobal,
            Some(ty),
            vec![],
            UArg::Address {
                space: crate::AddressSpace::Global,
                name: format!("b{}", id.index()),
                element: ty,
            },
        );
        let index = UOp::new(
            UOpKind::Index,
            Some(ty),
            vec![address, range.clone()],
            match view {
                Some(view) => UArg::ViewBufferIndex {
                    buffer: id.index() as u64,
                    elements: view
                        .logical_shape
                        .numel()
                        .map_err(|_| UOpError::InvalidArgument)?,
                    input_shape: view.logical_shape.clone(),
                    output_shape: out.clone(),
                    view,
                },
                None => UArg::BufferIndex {
                    buffer: id.index() as u64,
                    elements,
                    input_shape: shape,
                    output_shape: out.clone(),
                },
            },
        );
        Ok(UOp::new(UOpKind::Load, Some(ty), vec![index], UArg::None))
    }
    fn lower(
        graph: &Graph,
        id: NodeId,
        out: &Shape,
        range: &UOp,
        memo: &mut HashMap<NodeId, UOp>,
        materialized: &std::collections::BTreeSet<usize>,
    ) -> std::result::Result<UOp, UOpError> {
        if let Some(v) = memo.get(&id) {
            return Ok(v.clone());
        }
        let ty = UType::scalar(graph.dtype(id).map_err(|_| UOpError::UseBeforeDefinition)?);
        let x = if materialized.contains(&id.index()) {
            load(graph, id, out, range, None)?
        } else {
            match graph.op(id).map_err(|_| UOpError::UseBeforeDefinition)? {
                Op::Input { .. } | Op::Constant(_) => load(graph, id, out, range, None)?,
                Op::Random { .. } => return Err(UOpError::InvalidArgument),
                // A reduction is a schedule materialization boundary.  The DAG
                // executor supplies its owned buffer under this stable node ID.
                Op::Reduce { .. } => load(graph, id, out, range, None)?,
                Op::Shrink { .. }
                | Op::Reshape { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
                | Op::Stride { .. } => {
                    let planned = crate::rangeify::static_view(graph, id)
                        .map_err(|_| UOpError::InvalidArgument)?;
                    load(graph, planned.source, out, range, Some(planned.view))?
                }
                Op::Cast { input, .. } => {
                    UOp::cast(lower(graph, *input, out, range, memo, materialized)?, ty)
                }
                Op::Unary { op, input } => UOp::new(
                    UOpKind::GraphUnary(*op),
                    Some(ty),
                    vec![lower(graph, *input, out, range, memo, materialized)?],
                    UArg::None,
                ),
                Op::Binary { op, lhs, rhs } => UOp::new(
                    UOpKind::GraphBinary(*op),
                    Some(ty),
                    vec![
                        lower(graph, *lhs, out, range, memo, materialized)?,
                        lower(graph, *rhs, out, range, memo, materialized)?,
                    ],
                    UArg::None,
                ),
                Op::Compare { op, lhs, rhs } => UOp::new(
                    UOpKind::GraphCompare(*op),
                    Some(ty),
                    vec![
                        lower(graph, *lhs, out, range, memo, materialized)?,
                        lower(graph, *rhs, out, range, memo, materialized)?,
                    ],
                    UArg::None,
                ),
                Op::Logical { op, lhs, rhs } => {
                    let mut s = vec![lower(graph, *lhs, out, range, memo, materialized)?];
                    if let Some(rhs) = rhs {
                        s.push(lower(graph, *rhs, out, range, memo, materialized)?);
                    }
                    UOp::new(UOpKind::GraphLogical(*op), Some(ty), s, UArg::None)
                }
                Op::Select {
                    condition,
                    on_true,
                    on_false,
                } => UOp::new(
                    UOpKind::Ternary(crate::uop::Ternary::Where),
                    Some(ty),
                    vec![
                        lower(graph, *condition, out, range, memo, materialized)?,
                        lower(graph, *on_true, out, range, memo, materialized)?,
                        lower(graph, *on_false, out, range, memo, materialized)?,
                    ],
                    UArg::None,
                ),
                _ => return Err(UOpError::InvalidArgument),
            }
        };
        memo.insert(id, x.clone());
        Ok(x)
    }
    let value = lower(
        graph,
        output,
        &output_shape,
        &range,
        &mut HashMap::new(),
        materialized,
    )?;
    let address = UOp::new(
        UOpKind::DefineGlobal,
        Some(output_ty),
        vec![],
        UArg::Address {
            space: crate::AddressSpace::Global,
            name: format!("b{}", output.index()),
            element: output_ty,
        },
    );
    let index = UOp::new(
        UOpKind::Index,
        Some(output_ty),
        vec![address, range.clone()],
        UArg::BufferIndex {
            buffer: output.index() as u64,
            elements: extent,
            input_shape: output_shape.clone(),
            output_shape,
        },
    );
    let store = UOp::new(UOpKind::Store, None, vec![index, value], UArg::None);
    Ok(UOp::sink(vec![
        store,
        UOp::new(UOpKind::EndRange, None, vec![range], UArg::None),
    ]))
}

/// Lowers a static reduction with a pure elementwise producer.  The accumulator UOps
/// make initialization, update and finalization visible even though this
/// portable interpreter executes their nested domains directly.
pub fn lower_graph_reduction(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    lower_graph_reduction_with_materialized(graph, output, &std::collections::BTreeSet::new())
}

pub(crate) fn lower_graph_reduction_with_materialized(
    graph: &Graph,
    output: NodeId,
    materialized: &std::collections::BTreeSet<usize>,
) -> std::result::Result<UOp, UOpError> {
    let Op::Reduce {
        input,
        kind,
        axes,
        keepdim,
    } = graph
        .op(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let producer = lower_graph_elementwise_with_materialized(graph, *input, materialized)?;
    let value = producer
        .sources()
        .first()
        .and_then(|store| store.sources().get(1))
        .cloned()
        .ok_or(UOpError::InvalidArgument)?;
    let output_shape = graph
        .shape(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
        .clone();
    let ty = UType::scalar(
        graph
            .dtype(output)
            .map_err(|_| UOpError::UseBeforeDefinition)?,
    );
    let extent = output_shape
        .numel()
        .map_err(|_| UOpError::InvalidArgument)?;
    let range = UOp::new(
        UOpKind::Range,
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(
            i64::try_from(extent).map_err(|_| UOpError::InvalidArgument)?,
            UType::scalar(DType::I64),
        )],
        UArg::RangeAxis(0),
    );
    let address = UOp::new(
        UOpKind::DefineGlobal,
        Some(ty),
        vec![],
        UArg::Address {
            space: crate::AddressSpace::Global,
            name: format!("b{}", output.index()),
            element: ty,
        },
    );
    let index = UOp::new(
        UOpKind::Index,
        Some(ty),
        vec![address, range.clone()],
        UArg::BufferIndex {
            buffer: output.index() as u64,
            elements: extent,
            input_shape: output_shape.clone(),
            output_shape,
        },
    );
    let init = UOp::new(
        UOpKind::ReduceInit,
        Some(ty),
        vec![],
        UArg::Reduction {
            input_shape: graph
                .shape(*input)
                .map_err(|_| UOpError::UseBeforeDefinition)?
                .clone(),
            output_shape: graph
                .shape(output)
                .map_err(|_| UOpError::UseBeforeDefinition)?
                .clone(),
            axes: axes.clone(),
            keepdim: *keepdim,
            kind: *kind,
            mean: matches!(kind, crate::ReduceKind::Mean),
        },
    );
    let update = UOp::new(
        UOpKind::ReduceAccumulate,
        Some(ty),
        vec![init, value],
        UArg::None,
    );
    let finalize = UOp::new(UOpKind::ReduceFinalize, Some(ty), vec![update], UArg::None);
    Ok(UOp::sink(vec![
        UOp::new(UOpKind::Store, None, vec![index, finalize], UArg::None),
        UOp::new(UOpKind::EndRange, None, vec![range], UArg::None),
    ]))
}

/// Executes the typed range/load/store UOp form without invoking `CpuBackend`.
pub fn execute_elementwise(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> Result<TensorData> {
    if matches!(graph.op(output)?, Op::Reduce { .. }) {
        return execute_reduction(graph, output, inputs);
    }
    let kernel = lower_graph_elementwise(graph, output).map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    kernel.validate().map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    let mut bindings = KernelBindings::default();
    for id in 0..=output.index() {
        let id = NodeId::from_index(id);
        let shape = graph.shape(id)?.clone();
        let dtype = graph.dtype(id)?;
        let (role, value) = match graph.op(id)? {
            Op::Input { name } => {
                let v = inputs
                    .get(name)
                    .ok_or_else(|| Error::MissingInput(name.clone()))?
                    .clone();
                (BufferRole::Input, v)
            }
            Op::Constant(v) => (BufferRole::Constant, v.clone()),
            _ if id == output => (
                BufferRole::Output,
                TensorData::from_scalars(
                    shape.clone(),
                    dtype,
                    (0..shape.numel()?).map(|_| Scalar::I(0)),
                )?,
            ),
            _ => continue,
        };
        let desc = KernelBufferDesc::concrete(
            id.index() as u64,
            role,
            shape,
            dtype,
            role == BufferRole::Output,
        )?;
        bindings.insert(&desc, value)?;
    }
    execute_lowered_elementwise(&kernel, &bindings)
}

/// Executes an already-lowered pure elementwise UOp with checked owned bindings.
/// This is crate-private so CUDA's test mock can use the same independent
/// semantic oracle without making host materialization part of a runtime path.
pub(crate) fn execute_lowered_elementwise(
    kernel: &UOp,
    bindings: &KernelBindings,
) -> Result<TensorData> {
    if matches!(kernel.kind(), UOpKind::Random)
        && let UArg::Random(plan) = kernel.arg()
    {
        return plan.execute();
    }
    if matches!(kernel.kind(), UOpKind::Matmul)
        && let Some(plan) = kernel.arg().matmul_plan()
    {
        let lhs = bindings
            .get(plan.lhs.index() as u64)
            .ok_or(Error::InvalidIndex)?;
        let rhs = bindings
            .get(plan.rhs.index() as u64)
            .ok_or(Error::InvalidIndex)?;
        return plan
            .execute(lhs, rhs)
            .map_err(|error| Error::Serialization {
                reason: error.to_string(),
            });
    }
    let store = kernel
        .sources()
        .iter()
        .find(|node| matches!(node.kind(), UOpKind::Store))
        .ok_or(Error::InvalidIndex)?;
    let index = store.sources().first().ok_or(Error::InvalidIndex)?;
    let UArg::BufferIndex { output_shape, .. } = index.arg() else {
        return Err(Error::InvalidIndex);
    };
    let output_dtype = index.ty().ok_or(Error::InvalidIndex)?.scalar;
    let output_shape = output_shape.clone();
    let plan = IterationPlan::new(output_shape.clone());
    let len = plan.len()?;
    if output_dtype == DType::BF16
        && let Some(raw) = direct_f32_to_bf16(store, bindings, &plan, len)?
    {
        return TensorData::from_storage(output_shape, Storage::BF16(raw));
    }
    let mut values = Vec::with_capacity(len);
    for linear in 0..len {
        values.push(eval_store_value(store, bindings, linear, &plan)?);
    }
    TensorData::from_scalars(output_shape, output_dtype, values)
}

fn direct_f32_to_bf16(
    store: &UOp,
    bindings: &KernelBindings,
    plan: &IterationPlan,
    len: usize,
) -> Result<Option<Vec<u16>>> {
    let value = store.sources().get(1).ok_or(Error::InvalidIndex)?;
    if !matches!(value.kind(), UOpKind::Cast)
        || value.ty().is_none_or(|ty| ty.scalar != DType::BF16)
    {
        return Ok(None);
    }
    let load = value.sources().first().ok_or(Error::InvalidIndex)?;
    if !matches!(load.kind(), UOpKind::Load) || load.ty().is_none_or(|ty| ty.scalar != DType::F32) {
        return Ok(None);
    }
    let index = load.sources().first().ok_or(Error::InvalidIndex)?;
    let (buffer, input_shape, view) = match index.arg() {
        UArg::BufferIndex {
            buffer,
            input_shape,
            ..
        } => (*buffer, input_shape, None),
        UArg::ViewBufferIndex {
            buffer,
            input_shape,
            view,
            ..
        } => (*buffer, input_shape, Some(view)),
        _ => return Ok(None),
    };
    let data = bindings.get(buffer).ok_or(Error::InvalidIndex)?;
    let Storage::F32(values) = data.storage() else {
        return Err(Error::InvalidIndex);
    };
    let mut output = Vec::with_capacity(len);
    for linear in 0..len {
        let logical = plan.broadcast_offset(input_shape, linear)?;
        let offset = match view {
            Some(view) => view
                .element_offset(logical)
                .map_err(|_| Error::InvalidIndex)?,
            None => i64::try_from(logical).map_err(|_| Error::InvalidIndex)?,
        };
        output.push(crate::tensor::f32_to_bf16(
            *values
                .get(usize::try_from(offset).map_err(|_| Error::InvalidIndex)?)
                .ok_or(Error::InvalidIndex)?,
        ));
    }
    Ok(Some(output))
}

/// Executes with a verified allocation plan. Requested results are always
/// materialized afresh, so reuse metadata cannot expose stale contents.
pub fn execute_with_memory_plan(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
    plan: &crate::MemoryPlan,
) -> Result<TensorData> {
    let _ = plan;
    execute_elementwise(graph, output, inputs)
}

fn execute_reduction(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> Result<TensorData> {
    let Op::Reduce { .. } = graph.op(output)? else {
        return Err(Error::InvalidIndex);
    };
    let kernel = lower_graph_reduction(graph, output).map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    kernel.validate().map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    let bindings = graph_bindings(graph, output, inputs)?;
    execute_lowered_elementwise(&kernel, &bindings)
}

fn graph_bindings(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> Result<KernelBindings> {
    let mut bindings = KernelBindings::default();
    for raw in 0..=output.index() {
        let id = NodeId::from_index(raw);
        let shape = graph.shape(id)?.clone();
        let dtype = graph.dtype(id)?;
        let (role, value) = match graph.op(id)? {
            Op::Input { name } => (
                BufferRole::Input,
                inputs
                    .get(name)
                    .ok_or_else(|| Error::MissingInput(name.clone()))?
                    .clone(),
            ),
            Op::Constant(v) => (BufferRole::Constant, v.clone()),
            _ => continue,
        };
        let desc = KernelBufferDesc::concrete(id.index() as u64, role, shape, dtype, false)?;
        bindings.insert(&desc, value)?;
    }
    Ok(bindings)
}

fn eval_store_value(
    store: &UOp,
    bindings: &KernelBindings,
    linear: usize,
    plan: &IterationPlan,
) -> Result<Scalar> {
    if !matches!(store.kind(), UOpKind::Store) || store.sources().len() != 2 {
        return Err(Error::InvalidIndex);
    }
    eval(&store.sources()[1], bindings, linear, plan)
}
fn eval(n: &UOp, bindings: &KernelBindings, linear: usize, plan: &IterationPlan) -> Result<Scalar> {
    match n.kind() {
        UOpKind::Const => match n.arg() {
            UArg::Int(v) => Ok(Scalar::I(*v)),
            UArg::Scalar { dtype, bits } => Ok(match dtype {
                DType::Bool => Scalar::Bool(*bits != 0),
                DType::I8 => Scalar::I(*bits as i8 as i64),
                DType::U8 => Scalar::U(*bits as u8 as u64),
                DType::I16 => Scalar::I(*bits as i16 as i64),
                DType::U16 => Scalar::U(*bits as u16 as u64),
                DType::I32 => Scalar::I(*bits as i32 as i64),
                DType::U32 => Scalar::U(*bits as u32 as u64),
                DType::I64 => Scalar::I(*bits as i64),
                DType::U64 => Scalar::U(*bits),
                DType::F16 => Scalar::F(crate::tensor::f16_to_f32(*bits as u16) as f64),
                DType::BF16 => Scalar::F(crate::tensor::bf16_to_f32(*bits as u16) as f64),
                DType::F32 => Scalar::F(f32::from_bits(*bits as u32) as f64),
                DType::F64 => Scalar::F(f64::from_bits(*bits)),
            }),
            _ => Err(Error::InvalidIndex),
        },
        UOpKind::Load => {
            let index = n.sources().first().ok_or(Error::InvalidIndex)?;
            let (buffer, input_shape, view) = match index.arg() {
                UArg::BufferIndex {
                    buffer,
                    input_shape,
                    ..
                } => (*buffer, input_shape, None),
                UArg::ViewBufferIndex {
                    buffer,
                    input_shape,
                    view,
                    ..
                } => (*buffer, input_shape, Some(view)),
                _ => return Err(Error::InvalidIndex),
            };
            let logical = plan.broadcast_offset(input_shape, linear)?;
            let offset = match view {
                Some(view) => view
                    .element_offset(logical)
                    .map_err(|_| Error::InvalidIndex)?,
                None => i64::try_from(logical).map_err(|_| Error::InvalidIndex)?,
            };
            bindings
                .get(buffer)
                .ok_or(Error::InvalidIndex)?
                .storage()
                .scalar(usize::try_from(offset).map_err(|_| Error::InvalidIndex)?)
                .pipe(Ok)
        }
        UOpKind::Cast => Ok(cast_scalar(
            eval(&n.sources()[0], bindings, linear, plan)?,
            n.ty().ok_or(Error::InvalidIndex)?.scalar,
        )),
        UOpKind::GraphUnary(op) => unary(
            eval(&n.sources()[0], bindings, linear, plan)?,
            n.ty().ok_or(Error::InvalidIndex)?.scalar,
            *op,
        ),
        UOpKind::GraphBinary(op) => binary(
            eval(&n.sources()[0], bindings, linear, plan)?,
            eval(&n.sources()[1], bindings, linear, plan)?,
            n.ty().ok_or(Error::InvalidIndex)?.scalar,
            *op,
        ),
        UOpKind::GraphCompare(op) => Ok(Scalar::Bool(compare(
            eval(&n.sources()[0], bindings, linear, plan)?,
            eval(&n.sources()[1], bindings, linear, plan)?,
            *op,
        ))),
        UOpKind::GraphLogical(op) => {
            let a = eval(&n.sources()[0], bindings, linear, plan)?.as_bool();
            Ok(Scalar::Bool(match op {
                LogicalOp::Not => !a,
                LogicalOp::And => a && eval(&n.sources()[1], bindings, linear, plan)?.as_bool(),
                LogicalOp::Or => a || eval(&n.sources()[1], bindings, linear, plan)?.as_bool(),
            }))
        }
        UOpKind::Ternary(crate::uop::Ternary::Where) => {
            if eval(&n.sources()[0], bindings, linear, plan)?.as_bool() {
                eval(&n.sources()[1], bindings, linear, plan)
            } else {
                eval(&n.sources()[2], bindings, linear, plan)
            }
        }
        UOpKind::ReduceFinalize => {
            let update = n.sources().first().ok_or(Error::InvalidIndex)?;
            let init = update.sources().first().ok_or(Error::InvalidIndex)?;
            let UArg::Reduction {
                input_shape,
                output_shape,
                axes,
                keepdim,
                kind,
                mean,
            } = init.arg()
            else {
                return Err(Error::InvalidIndex);
            };
            if &plan.output != output_shape {
                return Err(Error::InvalidIndex);
            }
            let reduction = ReductionPlan::new(
                input_shape.clone(),
                output_shape.clone(),
                axes.clone(),
                *keepdim,
            )?;
            let source_plan = IterationPlan::new(input_shape.clone());
            let dtype = n.ty().ok_or(Error::InvalidIndex)?.scalar;
            let value = update.sources().get(1).ok_or(Error::InvalidIndex)?;
            let reduction_len = reduction.reduction_len()?;
            let mut acc = match kind {
                crate::ReduceKind::Sum | crate::ReduceKind::Mean => Scalar::I(0),
                crate::ReduceKind::Product => Scalar::I(1),
                crate::ReduceKind::Max => Scalar::F(f64::NEG_INFINITY),
                crate::ReduceKind::Min => Scalar::F(f64::INFINITY),
            };
            for reduce_linear in 0..reduction_len {
                let next = eval(
                    value,
                    bindings,
                    reduction.input_linear(linear, reduce_linear)?,
                    &source_plan,
                )?;
                acc = match kind {
                    crate::ReduceKind::Sum | crate::ReduceKind::Mean => {
                        binary(acc, next, dtype, BinaryOp::Add)?
                    }
                    crate::ReduceKind::Product => binary(acc, next, dtype, BinaryOp::Mul)?,
                    crate::ReduceKind::Max
                        if !next.as_f64().is_nan() && next.as_f64() > acc.as_f64() =>
                    {
                        next
                    }
                    crate::ReduceKind::Min
                        if !next.as_f64().is_nan() && next.as_f64() < acc.as_f64() =>
                    {
                        next
                    }
                    crate::ReduceKind::Max | crate::ReduceKind::Min => acc,
                };
            }
            if *mean {
                acc = Scalar::F(if reduction_len == 0 {
                    f64::NAN
                } else {
                    acc.as_f64() / reduction_len as f64
                });
            }
            Ok(acc)
        }
        _ => Err(Error::InvalidIndex),
    }
}
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}
fn cast_scalar(x: Scalar, dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(x.as_bool()),
        // Narrow at the requested storage width exactly once. Going through
        // i64/u64 first changes Rust's saturating float-to-integer result for
        // NaN, infinities, and out-of-range values before the later storage
        // conversion can observe it.
        DType::I8 => Scalar::I(match x {
            Scalar::F(value) => value as i8 as i64,
            _ => x.as_i64(),
        }),
        DType::I16 => Scalar::I(match x {
            Scalar::F(value) => value as i16 as i64,
            _ => x.as_i64(),
        }),
        DType::I32 => Scalar::I(match x {
            Scalar::F(value) => value as i32 as i64,
            _ => x.as_i64(),
        }),
        DType::I64 => Scalar::I(x.as_i64()),
        DType::U8 => Scalar::U(match x {
            Scalar::F(value) => value as u8 as u64,
            _ => x.as_u64(),
        }),
        DType::U16 => Scalar::U(match x {
            Scalar::F(value) => value as u16 as u64,
            _ => x.as_u64(),
        }),
        DType::U32 => Scalar::U(match x {
            Scalar::F(value) => value as u32 as u64,
            _ => x.as_u64(),
        }),
        DType::U64 => Scalar::U(x.as_u64()),
        _ => Scalar::F(x.as_f64()),
    }
}
fn unary(x: Scalar, dtype: DType, op: UnaryOp) -> Result<Scalar> {
    if !dtype.is_float() {
        return Ok(match (dtype, op) {
            (_, UnaryOp::IsNan) => Scalar::Bool(false),
            (_, UnaryOp::IsInf) => Scalar::Bool(false),
            (_, UnaryOp::IsFinite) => Scalar::Bool(true),
            (DType::Bool, UnaryOp::Neg) => Scalar::Bool(!x.as_bool()),
            (
                DType::Bool,
                UnaryOp::Relu
                | UnaryOp::Step
                | UnaryOp::Abs
                | UnaryOp::Square
                | UnaryOp::Floor
                | UnaryOp::Ceil
                | UnaryOp::Trunc
                | UnaryOp::Round
                | UnaryOp::Sign,
            ) => Scalar::Bool(x.as_bool()),
            (DType::U8 | DType::U16 | DType::U32 | DType::U64, UnaryOp::Neg) => {
                Scalar::U(0u64.wrapping_sub(x.as_u64()))
            }
            (DType::U8 | DType::U16 | DType::U32 | DType::U64, UnaryOp::Square) => {
                Scalar::U(x.as_u64().wrapping_mul(x.as_u64()))
            }
            (DType::U8 | DType::U16 | DType::U32 | DType::U64, UnaryOp::Sign) => {
                Scalar::U(u64::from(x.as_u64() != 0))
            }
            (DType::U8 | DType::U16 | DType::U32 | DType::U64, _) => Scalar::U(x.as_u64()),
            (_, UnaryOp::Neg) => Scalar::I(x.as_i64().wrapping_neg()),
            (_, UnaryOp::Abs) => Scalar::I(x.as_i64().wrapping_abs()),
            (_, UnaryOp::Relu) => Scalar::I(x.as_i64().max(0)),
            (_, UnaryOp::Step) => Scalar::I(i64::from(x.as_i64() > 0)),
            (_, UnaryOp::Square) => Scalar::I(x.as_i64().wrapping_mul(x.as_i64())),
            (_, UnaryOp::Sign) => Scalar::I(x.as_i64().signum()),
            (_, _) => Scalar::I(x.as_i64()),
        });
    }
    let v = x.as_f64();
    Ok(match op {
        UnaryOp::Neg => Scalar::F(-v),
        UnaryOp::Abs => Scalar::F(v.abs()),
        UnaryOp::Relu => Scalar::F(v.max(0.)),
        UnaryOp::Square => Scalar::F(v * v),
        UnaryOp::Reciprocal => Scalar::F(v.recip()),
        UnaryOp::Sqrt => Scalar::F(v.sqrt()),
        UnaryOp::Rsqrt => Scalar::F(v.sqrt().recip()),
        UnaryOp::Exp => Scalar::F(v.exp()),
        UnaryOp::Log => Scalar::F(v.ln()),
        UnaryOp::Exp2 => Scalar::F(v.exp2()),
        UnaryOp::Log2 => Scalar::F(v.log2()),
        UnaryOp::Sin => Scalar::F(v.sin()),
        UnaryOp::Cos => Scalar::F(v.cos()),
        UnaryOp::Tan => Scalar::F(v.tan()),
        UnaryOp::Sinh => Scalar::F(v.sinh()),
        UnaryOp::Cosh => Scalar::F(v.cosh()),
        UnaryOp::Tanh => Scalar::F(v.tanh()),
        UnaryOp::Asin => Scalar::F(v.asin()),
        UnaryOp::Acos => Scalar::F(v.acos()),
        UnaryOp::Atan => Scalar::F(v.atan()),
        UnaryOp::Asinh => Scalar::F(v.asinh()),
        UnaryOp::Acosh => Scalar::F(v.acosh()),
        UnaryOp::Atanh => Scalar::F(v.atanh()),
        UnaryOp::Floor => Scalar::F(v.floor()),
        UnaryOp::Ceil => Scalar::F(v.ceil()),
        UnaryOp::Trunc => Scalar::F(v.trunc()),
        UnaryOp::Round => Scalar::F(v.round_ties_even()),
        // Match tinygrad's comparison composition: NaN is nonzero but not
        // ordered below zero, and both signed zeroes select canonical +0.
        UnaryOp::Sign => Scalar::F(if v == 0.0 {
            0.0
        } else if v < 0.0 {
            -1.0
        } else {
            1.0
        }),
        UnaryOp::Step => Scalar::F(f64::from(v > 0.)),
        UnaryOp::IsNan => Scalar::Bool(v.is_nan()),
        UnaryOp::IsInf => Scalar::Bool(v.is_infinite()),
        UnaryOp::IsFinite => Scalar::Bool(v.is_finite()),
        UnaryOp::Erf => Scalar::F(erf(v)),
        UnaryOp::Erfc => Scalar::F(1.0 - erf(v)),
    })
}
fn binary(a: Scalar, b: Scalar, d: DType, op: BinaryOp) -> Result<Scalar> {
    if matches!(
        op,
        BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::TruncDiv | BinaryOp::Mod | BinaryOp::FMod
    ) && !d.is_float()
        && b.as_u64() == 0
    {
        return Err(Error::DivisionByZero { op: op.name() });
    };
    if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
        let count = b.as_u64();
        if (!matches!(b, Scalar::U(_)) && b.as_i64() < 0) || count >= d.bits() as u64 {
            return Err(Error::InvalidShiftCount {
                count: count.min(i64::MAX as u64) as i64,
                bits: d.bits(),
            });
        }
    }
    if d.is_float() {
        let (a, b) = (a.as_f64(), b.as_f64());
        return Ok(Scalar::F(match op {
            BinaryOp::Add => a + b,
            BinaryOp::Sub => a - b,
            BinaryOp::Mul => a * b,
            BinaryOp::Div => a / b,
            BinaryOp::Pow => a.powf(b),
            BinaryOp::Maximum => if a < b { b } else { a },
            BinaryOp::Minimum => if a > b { b } else { a },
            BinaryOp::FloorDiv => (a / b).floor(),
            BinaryOp::TruncDiv => (a / b).trunc(),
            BinaryOp::Mod => a - (a / b).floor() * b,
            BinaryOp::FMod => a % b,
            BinaryOp::Atan2 => a.atan2(b),
            BinaryOp::Copysign => a.copysign(b),
            _ => {
                return Err(Error::InvalidElementwiseDType {
                    op: op.name(),
                    actual: d,
                });
            }
        }));
    };
    if matches!(d, DType::Bool) {
        let (a, b) = (a.as_bool(), b.as_bool());
        return Ok(Scalar::Bool(match op {
            BinaryOp::Add | BinaryOp::BitOr | BinaryOp::Maximum => a || b,
            BinaryOp::Sub | BinaryOp::BitXor => a ^ b,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::BitAnd | BinaryOp::Minimum => a && b,
            _ => {
                return Err(Error::InvalidElementwiseDType {
                    op: op.name(),
                    actual: d,
                });
            }
        }));
    }
    if matches!(d, DType::U8 | DType::U16 | DType::U32 | DType::U64) {
        let (a, b) = (a.as_u64(), b.as_u64());
        return Ok(Scalar::U(match op {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::TruncDiv => a / b,
            BinaryOp::Mod | BinaryOp::FMod => a % b,
            BinaryOp::Pow => a.wrapping_pow(b as u32),
            BinaryOp::Maximum => if a < b { b } else { a },
            BinaryOp::Minimum => if a > b { b } else { a },
            BinaryOp::BitAnd => a & b,
            BinaryOp::BitOr => a | b,
            BinaryOp::BitXor => a ^ b,
            BinaryOp::Shl => a.wrapping_shl(b as u32),
            BinaryOp::Shr => a.wrapping_shr(b as u32),
            _ => {
                return Err(Error::InvalidElementwiseDType {
                    op: op.name(),
                    actual: d,
                });
            }
        }));
    }
    let (a, b) = (a.as_i64(), b.as_i64());
    Ok(Scalar::I(match op {
        BinaryOp::Add => a.wrapping_add(b),
        BinaryOp::Sub => a.wrapping_sub(b),
        BinaryOp::Mul => a.wrapping_mul(b),
        BinaryOp::Div | BinaryOp::TruncDiv => a.wrapping_div(b),
        BinaryOp::FloorDiv => a.wrapping_div_euclid(b),
        BinaryOp::Mod => a.wrapping_rem_euclid(b),
        BinaryOp::FMod => a.wrapping_rem(b),
        BinaryOp::Maximum => if a < b { b } else { a },
        BinaryOp::Minimum => if a > b { b } else { a },
        BinaryOp::BitAnd => a & b,
        BinaryOp::BitOr => a | b,
        BinaryOp::BitXor => a ^ b,
        BinaryOp::Shl => a.wrapping_shl(b as u32),
        BinaryOp::Shr => a.wrapping_shr(b as u32),
        BinaryOp::Pow => a.wrapping_pow(b as u32),
        _ => {
            return Err(Error::InvalidElementwiseDType {
                op: op.name(),
                actual: d,
            });
        }
    }))
}
fn erf(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    let t = 1.0 / (1.0 + 0.327_591_1 * value.abs());
    let polynomial =
        ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t;
    value.signum() * (1.0 - polynomial * (-value * value).exp())
}

fn compare(a: Scalar, b: Scalar, op: CompareOp) -> bool {
    use std::cmp::Ordering;

    // Compare the evaluated Scalar variants, not a lossy floating projection.
    // GraphCompare carries both sources through the UOp, so same-width I64/U64
    // lanes above 2^53 remain distinguishable while mixed scalar kinds retain
    // the CPU evaluator's existing signed/unsigned ordering contract.
    let ordering = match (a, b) {
        (Scalar::F(a), b) => a.partial_cmp(&b.as_f64()),
        (a, Scalar::F(b)) => a.as_f64().partial_cmp(&b),
        (Scalar::I(a), Scalar::I(b)) => Some(a.cmp(&b)),
        (Scalar::U(a), Scalar::U(b)) => Some(a.cmp(&b)),
        (Scalar::I(a), Scalar::U(b)) => {
            if a < 0 { Some(Ordering::Less) } else { Some((a as u64).cmp(&b)) }
        }
        (Scalar::U(a), Scalar::I(b)) => {
            if b < 0 { Some(Ordering::Greater) } else { Some(a.cmp(&(b as u64))) }
        }
        (Scalar::Bool(a), Scalar::Bool(b)) => Some(a.cmp(&b)),
        (Scalar::Bool(a), b) => Some((a as u8 as i64).cmp(&b.as_i64())),
        (a, Scalar::Bool(b)) => Some(a.as_i64().cmp(&(b as u8 as i64))),
    };
    match op {
        CompareOp::Eq => ordering == Some(Ordering::Equal),
        CompareOp::Ne => ordering != Some(Ordering::Equal),
        CompareOp::Lt => ordering == Some(Ordering::Less),
        CompareOp::Le => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        CompareOp::Gt => ordering == Some(Ordering::Greater),
        CompareOp::Ge => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Shape, SymbolicExpr};

    #[test]
    fn iteration_plan_covers_scalar_zero_and_broadcast_offsets() {
        let scalar = IterationPlan::new(Shape::new([]));
        assert_eq!(scalar.coords(0).unwrap(), Vec::<usize>::new());
        assert_eq!(scalar.broadcast_offset(&Shape::new([]), 0).unwrap(), 0);
        let plan = IterationPlan::new(Shape::from([2, 3]));
        assert_eq!(plan.broadcast_offset(&Shape::from([1, 3]), 5).unwrap(), 2);
        assert_eq!(plan.broadcast_offset(&Shape::from([2, 1]), 5).unwrap(), 1);
        assert_eq!(IterationPlan::new(Shape::from([0, 3])).len().unwrap(), 0);
    }

    #[test]
    fn descriptor_checks_bytes_and_symbolic_specialization() {
        let d = KernelBufferDesc::concrete(
            7,
            BufferRole::Input,
            Shape::from([2, 3]),
            DType::F32,
            false,
        )
        .unwrap();
        assert_eq!(d.bytes, 24);
        assert_eq!(d.byte_offset(5).unwrap(), 20);
        assert!(d.byte_offset(6).is_err());
        let expr = SymbolicExpr::variable("n", 0, 8).unwrap();
        let var = expr.variables().into_iter().next().unwrap();
        let shape = KernelShape::Symbolic(SymbolicShape::new(vec![crate::SymbolicDim::new(expr)]));
        assert_eq!(
            shape.bind(&BTreeMap::from([(var, 3)])).unwrap(),
            Shape::from([3])
        );
    }

    #[test]
    fn fused_uop_execution_matches_cpu_for_broadcast_select_cast_and_zero_domain() {
        let mut graph = Graph::new();
        let x = graph.input("x", Shape::from([2, 1]));
        let y = graph.input("y", Shape::from([1, 3]));
        let sum = graph.add(x, y).unwrap();
        let two = graph.constant(TensorData::scalar(2.0));
        let cond = graph.gt(sum, two).unwrap();
        let neg = graph.neg(sum).unwrap();
        let out = graph.select(cond, sum, neg).unwrap();
        let inputs = HashMap::from([
            ("x".into(), TensorData::new([2, 1], vec![1., 3.]).unwrap()),
            (
                "y".into(),
                TensorData::new([1, 3], vec![0., 1., 2.]).unwrap(),
            ),
        ]);
        let expected = CpuBackend.execute(&graph, out, &inputs).unwrap();
        let actual = execute_elementwise(&graph, out, &inputs).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        assert_eq!(actual.to_vec_f64(), expected.to_vec_f64());
        let uop = lower_graph_elementwise(&graph, out).unwrap();
        uop.validate().unwrap();
        assert_eq!(
            format!("{uop}"),
            format!("{}", lower_graph_elementwise(&graph, out).unwrap())
        );

        let mut empty = Graph::new();
        let e = empty.input("e", Shape::from([0, 2]));
        let z = empty.neg(e).unwrap();
        let result = execute_elementwise(
            &empty,
            z,
            &HashMap::from([("e".into(), TensorData::new([0, 2], vec![]).unwrap())]),
        )
        .unwrap();
        assert!(result.is_empty());

        let mut integers = Graph::new();
        let a = integers.input_dtype("a", Shape::from([2]), DType::U64);
        let b = integers.input_dtype("b", Shape::from([2]), DType::U64);
        let sum = integers.add(a, b).unwrap();
        let exact_inputs = HashMap::from([
            (
                "a".into(),
                TensorData::from_scalars([2], DType::U64, [Scalar::U(u64::MAX), Scalar::U(7)])
                    .unwrap(),
            ),
            (
                "b".into(),
                TensorData::from_scalars([2], DType::U64, [Scalar::U(1), Scalar::U(9)]).unwrap(),
            ),
        ]);
        assert_eq!(
            execute_elementwise(&integers, sum, &exact_inputs)
                .unwrap()
                .storage(),
            CpuBackend
                .execute(&integers, sum, &exact_inputs)
                .unwrap()
                .storage()
        );
    }

    #[test]
    fn generic_compare_preserves_typed_wide_integer_ordering_and_float_boundaries() {
        let compare_all = |lhs_dtype, lhs_values: Vec<Scalar>, rhs_dtype, rhs_values: Vec<Scalar>| {
            for op in [
                CompareOp::Eq,
                CompareOp::Ne,
                CompareOp::Lt,
                CompareOp::Le,
                CompareOp::Gt,
                CompareOp::Ge,
            ] {
                let mut graph = Graph::new();
                let lhs = graph.input_dtype("lhs", [lhs_values.len()], lhs_dtype);
                let rhs = graph.input_dtype("rhs", [rhs_values.len()], rhs_dtype);
                let output = graph.compare(op, lhs, rhs).unwrap();
                let inputs = HashMap::from([
                    (
                        "lhs".into(),
                        TensorData::from_scalars([lhs_values.len()], lhs_dtype, lhs_values.clone()).unwrap(),
                    ),
                    (
                        "rhs".into(),
                        TensorData::from_scalars([rhs_values.len()], rhs_dtype, rhs_values.clone()).unwrap(),
                    ),
                ]);
                assert_eq!(
                    execute_elementwise(&graph, output, &inputs).unwrap().storage(),
                    CpuBackend.execute(&graph, output, &inputs).unwrap().storage(),
                    "{op:?} {lhs_dtype:?}/{rhs_dtype:?}",
                );
            }
        };

        let two_to_53 = 1_u64 << 53;
        compare_all(
            DType::U64,
            vec![Scalar::U(two_to_53), Scalar::U(u64::MAX)],
            DType::U64,
            vec![Scalar::U(two_to_53 + 1), Scalar::U(0)],
        );
        compare_all(
            DType::I64,
            vec![Scalar::I(-((1_i64) << 53)), Scalar::I(i64::MIN)],
            DType::I64,
            vec![Scalar::I(-((1_i64 << 53) + 1)), Scalar::I(i64::MAX)],
        );
        // GraphCompare itself has no promotion node, so actual mixed Scalar
        // kinds must retain the CPU evaluator's signed/unsigned ordering.
        compare_all(
            DType::I64,
            vec![Scalar::I(-1), Scalar::I((1_i64) << 53)],
            DType::U64,
            vec![Scalar::U(0), Scalar::U((1_u64 << 53) + 1)],
        );
        compare_all(
            DType::Bool,
            vec![Scalar::Bool(false), Scalar::Bool(true)],
            DType::Bool,
            vec![Scalar::Bool(true), Scalar::Bool(false)],
        );

        // Partial float order keeps NaN unordered, treats +/-0 as equal, and
        // retains ordinary infinity ordering across every CompareOp.
        for op in [
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [3], DType::F64);
            let rhs = graph.input_dtype("rhs", [3], DType::F64);
            let output = graph.compare(op, lhs, rhs).unwrap();
            let inputs = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::new([3], vec![f64::NAN, -0.0, f64::INFINITY]).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::new([3], vec![1.0, 0.0, f64::INFINITY]).unwrap(),
                ),
            ]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs).unwrap().storage(),
                CpuBackend.execute(&graph, output, &inputs).unwrap().storage(),
                "{op:?} float boundaries",
            );
        }
    }

    #[test]
    fn generic_extrema_match_cpu_ordered_selection_and_source_bridge() {
        for (op, label) in [
            (BinaryOp::Maximum, "maximum"),
            (BinaryOp::Minimum, "minimum"),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [5], DType::F64);
            let rhs = graph.input_dtype("rhs", [5], DType::F64);
            let output = graph.binary(op, lhs, rhs).unwrap();
            let inputs = HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(f64::NAN),
                            Scalar::F(-0.0),
                            Scalar::F(5.0),
                            Scalar::F(f64::NEG_INFINITY),
                            Scalar::F(f64::INFINITY),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::NAN),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::INFINITY),
                        ],
                    )
                    .unwrap(),
                ),
            ]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs).unwrap().storage(),
                CpuBackend.execute(&graph, output, &inputs).unwrap().storage(),
                "{label} ordered float selection",
            );
        }

        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [1], DType::I64);
        let rhs = graph.input_dtype("rhs", [1], DType::U64);
        let output = graph.maximum(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        let inputs = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([1], DType::I64, [Scalar::I(1_i64 << 53)]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([1], DType::U64, [Scalar::U((1_u64 << 53) + 1)])
                    .unwrap(),
            ),
        ]);
        assert_eq!(
            execute_elementwise(&graph, output, &inputs).unwrap().storage(),
            CpuBackend.execute(&graph, output, &inputs).unwrap().storage(),
        );
    }

    #[test]
    fn lowered_float_to_narrow_integer_casts_match_cpu_at_special_values() {
        let values = [
            Scalar::F(f32::NEG_INFINITY as f64),
            Scalar::F(f32::NAN as f64),
            Scalar::F(-1.5),
            Scalar::F(1.5),
            Scalar::F(f32::INFINITY as f64),
        ];

        for dtype in [
            DType::I8,
            DType::I16,
            DType::I32,
            DType::U8,
            DType::U16,
            DType::U32,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", Shape::from([values.len()]), DType::F32);
            let output = graph.cast(input, dtype).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars([values.len()], DType::F32, values).unwrap(),
            )]);

            assert_eq!(
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                "lowered {dtype:?} cast diverged from the CPU backend"
            );
        }
    }

    #[test]
    fn fused_sum_and_mean_reductions_match_cpu_across_domains() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([2, 3, 2]), DType::F16);
        let two = graph.constant(TensorData::scalar(2.0));
        let squared = graph.square(x).unwrap();
        let producer = graph.add(squared, two).unwrap();
        let sum = graph
            .reduce(producer, crate::ReduceKind::Sum, Some(vec![-1, 1]), true)
            .unwrap();
        let mean = graph
            .reduce(producer, crate::ReduceKind::Mean, None, false)
            .unwrap();
        let inputs = HashMap::from([(
            "x".into(),
            TensorData::from_scalars(
                [2, 3, 2],
                DType::F16,
                (0..12).map(|v| Scalar::F(v as f64 - 4.)),
            )
            .unwrap(),
        )]);
        for output in [sum, mean] {
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            let actual = execute_elementwise(&graph, output, &inputs).unwrap();
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(actual.storage(), expected.storage());
            lower_graph_reduction(&graph, output)
                .unwrap()
                .validate()
                .unwrap();
        }
        let mut empty = Graph::new();
        let x = empty.input("x", Shape::from([2, 0]));
        let sum = empty
            .reduce(x, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let inputs = HashMap::from([("x".into(), TensorData::new([2, 0], vec![]).unwrap())]);
        assert_eq!(
            execute_elementwise(&empty, sum, &inputs)
                .unwrap()
                .to_vec_f64(),
            CpuBackend
                .execute(&empty, sum, &inputs)
                .unwrap()
                .to_vec_f64()
        );
    }

    #[test]
    fn static_shrink_views_fuse_into_loads_and_match_cpu() {
        let mut graph = Graph::new();
        let x = graph.input_dtype("x", Shape::from([3, 4]), DType::I32);
        let first = graph.shrink(x, vec![(1, 3), (0, 4)]).unwrap();
        let view = graph.shrink(first, vec![(0, 2), (1, 3)]).unwrap();
        let rhs = graph.input_dtype("rhs", Shape::from([1, 2]), DType::I32);
        let out = graph.add(view, rhs).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::from_scalars([3, 4], DType::I32, (0..12).map(Scalar::I)).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([1, 2], DType::I32, [Scalar::I(10), Scalar::I(20)])
                    .unwrap(),
            ),
        ]);
        let expected = CpuBackend.execute(&graph, out, &inputs).unwrap();
        let actual = execute_elementwise(&graph, out, &inputs).unwrap();
        assert_eq!(actual.storage(), expected.storage());
        let lowered = lower_graph_elementwise(&graph, out).unwrap();
        lowered.validate().unwrap();
        assert!(
            lowered
                .topological()
                .unwrap()
                .iter()
                .any(|node| matches!(node.arg(), UArg::ViewBufferIndex { .. }))
        );
        let ptx = crate::PtxRenderer::new(80)
            .unwrap()
            .render(&lowered)
            .unwrap();
        assert!(ptx.source.contains("mad.lo.s64"));

        let empty = graph.shrink(x, vec![(0, 0), (0, 4)]).unwrap();
        let empty_result = execute_elementwise(&graph, empty, &inputs).unwrap();
        assert!(empty_result.is_empty());
    }
}
