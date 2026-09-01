//! Typed, owned bindings and a portable interpreter for elementwise UOp kernels.
//!
//! This is intentionally not a backend: bindings clone their `TensorData`, so a
//! scheduled kernel cannot retain or alias a caller's storage.  Element offsets
//! are checked separately from byte offsets, which keeps the ABI boundary
//! explicit for future renderers.
use crate::{
    AddressValue, BinaryOp, CompareOp, DType, Error, Graph, IndexValue, LiteralValue, LogicalOp,
    MatmulValue, MovementValue, NodeId, Op, Operation, PrefixScanValue, ReductionValue, Result,
    Scalar, Shape, SortValue, Storage, SymbolicShape, SymbolicVar, TensorData, TensorGuardValue,
    UOp, UOpError, UType, UnaryOp,
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
    let dtype = plan.dtype;
    let kernel = UOp::from_operation(
        Operation::Random(Box::new(plan)),
        Some(UType::scalar(dtype)),
        vec![],
    );
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers tinygrad's live packed-U64 Threefry operation as one typed,
/// dependency-bearing semantic. Unlike [`Operation::Random`], both operands
/// remain explicit schedule inputs and no stream reservation is involved.
pub fn lower_graph_threefry(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    let Op::Threefry { counter, key } = graph
        .op(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let value = crate::ThreefryValue {
        counter: *counter,
        key: *key,
        counter_shape: graph
            .shape(*counter)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .clone(),
        key_shape: graph
            .shape(*key)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .clone(),
        output,
        output_shape: graph
            .shape(output)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            .clone(),
    };
    value.validate()?;
    let kernel = UOp::from_operation(
        Operation::Threefry(value),
        Some(UType::scalar(DType::U64)),
        vec![],
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
    let operation = match crate::TensorCoreMatmulPayload::select(plan.clone(), target.clone())
        .map_err(|_| UOpError::InvalidArgument)?
    {
        Some(payload) => {
            crate::plan_tensor_core_matmul_promotion(&payload)
                .map_err(|_| UOpError::InvalidArgument)?;
            Operation::Matmul(MatmulValue::TensorCore(Box::new(payload)))
        }
        None => match crate::TiledMatmulPayload::select(plan.clone(), target)
            .map_err(|_| UOpError::InvalidArgument)?
        {
            Some(payload) => {
                crate::plan_tiled_matmul_promotion(&payload)
                    .map_err(|_| UOpError::InvalidArgument)?;
                Operation::Matmul(MatmulValue::Tiled(Box::new(payload)))
            }
            None => Operation::Matmul(MatmulValue::Serial(Box::new(plan))),
        },
    };
    let kernel = UOp::from_operation(operation, Some(UType::scalar(dtype)), vec![]);
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers only the legacy RGUA v10 validated static F32 NCHW 1x1 Conv2d
/// semantic. New public convolution graphs are compositional; broader legacy
/// nodes fail here before scheduling can create a callable ABI.
pub fn lower_graph_static_conv2d(
    graph: &Graph,
    output: NodeId,
) -> std::result::Result<UOp, UOpError> {
    let plan = crate::StaticConv2dPlan::from_graph(graph, output)
        .map_err(|_| UOpError::InvalidArgument)?;
    let kernel = UOp::from_operation(
        Operation::Conv2d(Box::new(plan)),
        Some(UType::scalar(DType::F32)),
        vec![],
    );
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers one materializing concat/gather/scatter operation into its validated
/// shared movement payload. Native renderers consume its ordered operand ABI.
pub fn lower_graph_movement(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    let plan = crate::MovementKernelPlan::from_scheduled_graph(graph, output)
        .map_err(|_| UOpError::InvalidArgument)?;
    let dtype = plan.dtype;
    let kernel = UOp::from_operation(
        Operation::Movement(MovementValue::Plan(Box::new(plan))),
        Some(UType::scalar(dtype)),
        vec![],
    );
    kernel.validate()?;
    Ok(kernel)
}

/// Lowers the narrow static computed-affine materialization boundary. The
/// result owns dense storage and can therefore feed contraction/reduction
/// plans without treating an aliasing view as an ABI buffer.
pub(crate) fn lower_graph_computed_affine_view(
    graph: &Graph,
    output: NodeId,
) -> std::result::Result<UOp, UOpError> {
    let plan = crate::MovementKernelPlan::from_computed_affine_view(graph, output)
        .map_err(|_| UOpError::InvalidArgument)?;
    let dtype = plan.dtype;
    let kernel = UOp::from_operation(
        Operation::Movement(MovementValue::Plan(Box::new(plan))),
        Some(UType::scalar(dtype)),
        vec![],
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
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(extent_i64, UType::scalar(DType::I64))],
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
        let address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: crate::AddressSpace::Global,
                name: format!("b{}", id.index()),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        let operation = match view {
            Some(view) => Operation::Index(IndexValue::View {
                buffer: id.index() as u64,
                elements: view
                    .logical_shape
                    .numel()
                    .map_err(|_| UOpError::InvalidArgument)?,
                input_shape: view.logical_shape.clone(),
                output_shape: out.clone(),
                view,
            }),
            None => Operation::Index(IndexValue::Buffer {
                buffer: id.index() as u64,
                elements,
                input_shape: shape,
                output_shape: out.clone(),
            }),
        };
        let index = UOp::from_operation(operation, Some(ty), vec![address, range.clone()]);
        Ok(UOp::from_operation(Operation::Load, Some(ty), vec![index]))
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
                Op::Input { .. } => load(graph, id, out, range, None)?,
                // A rank-0 graph constant is dependency-free once its typed
                // storage payload is present in the UOp. Keep all non-scalar
                // constants as buffer loads: their allocation, scheduling,
                // capture, and replay ownership remains unchanged.
                Op::Constant(data) if data.shape().rank() == 0 => {
                    UOp::scalar_constant(data.dtype(), crate::uop::raw_literal_bits(data)?, ty)
                }
                Op::Constant(_) => load(graph, id, out, range, None)?,
                Op::Random { .. } => return Err(UOpError::InvalidArgument),
                // A reduction is a schedule materialization boundary.  The DAG
                // executor supplies its owned buffer under this stable node ID.
                Op::Reduce { .. } | Op::PrefixScan { .. } => load(graph, id, out, range, None)?,
                Op::Shrink { .. }
                | Op::Reshape { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
                | Op::Stride { .. } => {
                    let planned = crate::rangeify::static_view(graph, id)
                        .or_else(|_| crate::rangeify::computed_view(graph, id))
                        .map_err(|_| UOpError::InvalidArgument)?;
                    load(graph, planned.source, out, range, Some(planned.view))?
                }
                Op::Cast { input, .. } => {
                    UOp::cast(lower(graph, *input, out, range, memo, materialized)?, ty)
                }
                // Graph bitcasts are materializing raw-byte movement roots.
                // A nested instance must have been scheduled already rather
                // than silently lowered as a numeric scalar cast.
                Op::Bitcast { .. } | Op::Contiguous { .. } => {
                    return Err(UOpError::InvalidArgument);
                }
                Op::ContiguousBackward { input } => {
                    lower(graph, *input, out, range, memo, materialized)?
                }
                // Detach is an autograd boundary, not a runtime value
                // transformation. Native lowering keeps the same typed value
                // while Graph reverse-mode traversal owns the gradient stop.
                Op::Detach { input } => lower(graph, *input, out, range, memo, materialized)?,
                Op::Unary { op, input } => UOp::from_operation(
                    Operation::GraphUnary(*op),
                    Some(ty),
                    vec![lower(graph, *input, out, range, memo, materialized)?],
                ),
                Op::Binary { op, lhs, rhs } => UOp::from_operation(
                    Operation::GraphBinary(*op),
                    Some(ty),
                    vec![
                        lower(graph, *lhs, out, range, memo, materialized)?,
                        lower(graph, *rhs, out, range, memo, materialized)?,
                    ],
                ),
                Op::Compare { op, lhs, rhs } => UOp::from_operation(
                    Operation::GraphCompare(*op),
                    Some(ty),
                    vec![
                        lower(graph, *lhs, out, range, memo, materialized)?,
                        lower(graph, *rhs, out, range, memo, materialized)?,
                    ],
                ),
                Op::Logical { op, lhs, rhs } => {
                    let mut s = vec![lower(graph, *lhs, out, range, memo, materialized)?];
                    if let Some(rhs) = rhs {
                        s.push(lower(graph, *rhs, out, range, memo, materialized)?);
                    }
                    UOp::from_operation(Operation::GraphLogical(*op), Some(ty), s)
                }
                Op::Select {
                    condition,
                    on_true,
                    on_false,
                } => UOp::from_operation(
                    Operation::Ternary(crate::uop::Ternary::Where),
                    Some(ty),
                    vec![
                        lower(graph, *condition, out, range, memo, materialized)?,
                        lower(graph, *on_true, out, range, memo, materialized)?,
                        lower(graph, *on_false, out, range, memo, materialized)?,
                    ],
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
    let address = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: crate::AddressSpace::Global,
            name: format!("b{}", output.index()),
            element: output_ty,
        }),
        Some(output_ty),
        vec![],
    );
    let index = UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer: output.index() as u64,
            elements: extent,
            input_shape: output_shape.clone(),
            output_shape,
        }),
        Some(output_ty),
        vec![address, range.clone()],
    );
    let store = UOp::from_operation(Operation::Store, None, vec![index, value]);
    Ok(UOp::sink(vec![
        store,
        UOp::from_operation(Operation::EndRange, None, vec![range]),
    ]))
}

/// Lowers a static reduction with a pure elementwise producer.  The accumulator UOps
/// make initialization, update and finalization visible even though this
/// portable interpreter executes their nested domains directly.
pub fn lower_graph_reduction(graph: &Graph, output: NodeId) -> std::result::Result<UOp, UOpError> {
    lower_graph_reduction_with_materialized(graph, output, &std::collections::BTreeSet::new())
}

/// Lowers a static inclusive prefix scan into one typed UOp. Unlike a
/// reduction, its output retains every input coordinate, so the normalized
/// scan axis is carried as an explicit payload rather than inferred from a
/// rank-changing loop nest.
pub fn lower_graph_prefix_scan(
    graph: &Graph,
    output: NodeId,
) -> std::result::Result<UOp, UOpError> {
    let Op::PrefixScan {
        input,
        axis,
        kind,
        output: scan_output,
    } = graph
        .op(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    Ok(UOp::from_operation(
        Operation::PrefixScan(PrefixScanValue {
            input: *input,
            destination: output,
            input_shape: graph
                .shape(*input)
                .map_err(|_| UOpError::UseBeforeDefinition)?
                .clone(),
            output_shape: graph
                .shape(output)
                .map_err(|_| UOpError::UseBeforeDefinition)?
                .clone(),
            axis: *axis,
            kind: *kind,
            output: *scan_output,
            input_dtype: graph
                .dtype(*input)
                .map_err(|_| UOpError::UseBeforeDefinition)?,
            dtype: graph
                .dtype(output)
                .map_err(|_| UOpError::UseBeforeDefinition)?,
        }),
        Some(UType::scalar(
            graph
                .dtype(output)
                .map_err(|_| UOpError::UseBeforeDefinition)?,
        )),
        vec![],
    ))
}

/// Lowers the coupled stable Sort producer. The UOp owns both output buffer
/// identities even though its scalar type remains the values dtype.
pub fn lower_graph_sort_pair(
    graph: &Graph,
    values: NodeId,
    indices: NodeId,
) -> std::result::Result<UOp, UOpError> {
    let Op::Sort {
        input,
        axis,
        descending,
        output: crate::SortOutput::Values,
        ..
    } = graph
        .op(values)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let Op::Sort {
        input: index_input,
        axis: index_axis,
        descending: index_descending,
        pair: index_pair,
        output: crate::SortOutput::Indices,
        ..
    } = graph
        .op(indices)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let Op::Sort { pair, .. } = graph
        .op(values)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    if input != index_input
        || axis != index_axis
        || descending != index_descending
        || pair != index_pair
    {
        return Err(UOpError::InvalidArgument);
    }
    let input_shape = graph
        .shape(*input)
        .map_err(|_| UOpError::UseBeforeDefinition)?
        .clone();
    if graph
        .shape(values)
        .map_err(|_| UOpError::UseBeforeDefinition)?
        != &input_shape
        || graph
            .shape(indices)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            != &input_shape
        || graph
            .dtype(indices)
            .map_err(|_| UOpError::UseBeforeDefinition)?
            != DType::I32
    {
        return Err(UOpError::InvalidArgument);
    }
    let dtype = graph
        .dtype(values)
        .map_err(|_| UOpError::UseBeforeDefinition)?;
    Ok(UOp::from_operation(
        Operation::Sort(SortValue {
            input: *input,
            input_shape,
            axis: *axis,
            descending: *descending,
            values,
            indices,
            dtype,
        }),
        Some(UType::scalar(dtype)),
        vec![],
    ))
}

pub fn lower_graph_tensor_guard(
    graph: &Graph,
    output: NodeId,
) -> std::result::Result<UOp, UOpError> {
    let Op::TensorGuard { input, axis } = graph
        .op(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?
    else {
        return Err(UOpError::InvalidArgument);
    };
    let input_shape = graph
        .shape(*input)
        .map_err(|_| UOpError::UseBeforeDefinition)?
        .clone();
    let dtype = graph
        .dtype(output)
        .map_err(|_| UOpError::UseBeforeDefinition)?;
    Ok(UOp::from_operation(
        Operation::TensorGuard(TensorGuardValue {
            input: *input,
            input_shape,
            axis: *axis,
            dtype,
        }),
        Some(UType::scalar(dtype)),
        vec![],
    ))
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
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(
            i64::try_from(extent).map_err(|_| UOpError::InvalidArgument)?,
            UType::scalar(DType::I64),
        )],
    );
    let address = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: crate::AddressSpace::Global,
            name: format!("b{}", output.index()),
            element: ty,
        }),
        Some(ty),
        vec![],
    );
    let index = UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer: output.index() as u64,
            elements: extent,
            input_shape: output_shape.clone(),
            output_shape,
        }),
        Some(ty),
        vec![address, range.clone()],
    );
    let init = UOp::from_operation(
        Operation::ReduceInit(ReductionValue {
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
        }),
        Some(ty),
        vec![],
    );
    let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, value]);
    let finalize = UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update]);
    Ok(UOp::sink(vec![
        UOp::from_operation(Operation::Store, None, vec![index, finalize]),
        UOp::from_operation(Operation::EndRange, None, vec![range]),
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

/// Executes an already-lowered pure kernel UOp with checked owned bindings.
/// Operation-scoped roots delegate to their canonical immutable plans;
/// ordinary fused roots retain the elementwise evaluator below. This is crate-private so
/// CUDA's test mock can use the same independent semantic oracle without
/// making host materialization part of a runtime path.
pub(crate) fn execute_lowered_elementwise(
    kernel: &UOp,
    bindings: &KernelBindings,
) -> Result<TensorData> {
    if let Operation::Movement(crate::MovementValue::Plan(plan)) = kernel.operation() {
        let operands = plan
            .input_operands()
            .into_iter()
            .map(|operand| {
                bindings
                    .get(operand.node.index() as u64)
                    .cloned()
                    .ok_or(Error::InvalidIndex)
            })
            .collect::<Result<Vec<_>>>()?;
        return plan
            .execute(&operands)
            .map_err(|error| Error::Serialization {
                reason: error.to_string(),
            });
    }
    if let Operation::Random(plan) = kernel.operation() {
        return plan.execute();
    }
    if let Operation::PrefixScan(plan) = kernel.operation() {
        let input = bindings
            .get(plan.input.index() as u64)
            .ok_or(Error::InvalidIndex)?;
        return crate::backend::execute_prefix_scan(
            input,
            plan.axis,
            plan.kind,
            plan.output,
            plan.dtype,
        );
    }
    if let Operation::Threefry(plan) = kernel.operation() {
        let counter = bindings
            .get(plan.counter.index() as u64)
            .ok_or(Error::InvalidIndex)?;
        let key = bindings
            .get(plan.key.index() as u64)
            .ok_or(Error::InvalidIndex)?;
        return crate::random::execute_live_threefry(counter, key, &plan.output_shape);
    }
    let matmul = match kernel.operation() {
        Operation::Matmul(MatmulValue::Serial(plan)) => Some(plan.as_ref()),
        Operation::Matmul(MatmulValue::Tiled(payload)) => Some(&payload.matmul),
        Operation::Matmul(MatmulValue::TensorCore(payload)) => Some(&payload.matmul),
        _ => None,
    };
    if let Some(plan) = matmul {
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
        .find(|node| matches!(node.operation(), Operation::Store))
        .ok_or(Error::InvalidIndex)?;
    let index = store.sources().first().ok_or(Error::InvalidIndex)?;
    let Operation::Index(IndexValue::Buffer { output_shape, .. }) = index.operation() else {
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
    TensorData::from_storage(output_shape, fused_storage(output_dtype, values))
}

fn fused_storage(dtype: DType, values: Vec<FusedValue>) -> Storage {
    match dtype {
        dtype @ (DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) => {
            Storage::Float8(crate::Float8Storage::from_raw(
                dtype.float8_format().expect("float8 dtype"),
                values
                    .into_iter()
                    .map(|value| match value {
                        FusedValue::F8(source, bits) if source == dtype => bits,
                        value => dtype
                            .float8_format()
                            .expect("float8 dtype")
                            .encode(value.scalar().as_f64()),
                    })
                    .collect(),
            ))
        }
        DType::F16 => Storage::F16(
            values
                .into_iter()
                .map(|value| match value {
                    FusedValue::F16(bits) => bits,
                    value => crate::tensor::f32_to_f16(value.scalar().as_f64() as f32),
                })
                .collect(),
        ),
        DType::BF16 => Storage::BF16(
            values
                .into_iter()
                .map(|value| match value {
                    FusedValue::BF16(bits) => bits,
                    value => crate::tensor::f32_to_bf16(value.scalar().as_f64() as f32),
                })
                .collect(),
        ),
        DType::F32 => Storage::F32(
            values
                .into_iter()
                .map(|value| match value {
                    FusedValue::F32(bits) => f32::from_bits(bits),
                    value => value.scalar().as_f64() as f32,
                })
                .collect(),
        ),
        DType::F64 => Storage::F64(
            values
                .into_iter()
                .map(|value| match value {
                    FusedValue::F64(bits) => f64::from_bits(bits),
                    value => value.scalar().as_f64(),
                })
                .collect(),
        ),
        _ => Storage::from_scalars(
            dtype,
            values.into_iter().map(|value| value.into_storage(dtype)),
        ),
    }
}

fn direct_f32_to_bf16(
    store: &UOp,
    bindings: &KernelBindings,
    plan: &IterationPlan,
    len: usize,
) -> Result<Option<Vec<u16>>> {
    let value = store.sources().get(1).ok_or(Error::InvalidIndex)?;
    if !matches!(value.operation(), Operation::Cast)
        || value.ty().is_none_or(|ty| ty.scalar != DType::BF16)
    {
        return Ok(None);
    }
    let load = value.sources().first().ok_or(Error::InvalidIndex)?;
    if !matches!(load.operation(), Operation::Load)
        || load.ty().is_none_or(|ty| ty.scalar != DType::F32)
    {
        return Ok(None);
    }
    let index = load.sources().first().ok_or(Error::InvalidIndex)?;
    let (buffer, input_shape, view) = match index.operation() {
        Operation::Index(IndexValue::Buffer {
            buffer,
            input_shape,
            ..
        }) => (*buffer, input_shape, None),
        Operation::Index(IndexValue::View {
            buffer,
            input_shape,
            view,
            ..
        }) => (*buffer, input_shape, Some(view)),
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
) -> Result<FusedValue> {
    if !matches!(store.operation(), Operation::Store) || store.sources().len() != 2 {
        return Err(Error::InvalidIndex);
    }
    eval(&store.sources()[1], bindings, linear, plan)
}
/// A fused lane retains exact floating storage only until an operation needs
/// its numeric value. This is deliberately private to the interpreter: UOps
/// and TensorData already carry typed raw constants/storage, while Scalar is
/// the public numeric conversion boundary.
#[derive(Clone, Copy, Debug)]
enum FusedValue {
    Scalar(Scalar),
    F8(DType, u8),
    F16(u16),
    BF16(u16),
    F32(u32),
    F64(u64),
}
impl FusedValue {
    fn typed(value: Scalar, dtype: DType) -> Self {
        match dtype {
            dtype @ (DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) => {
                Self::F8(
                    dtype,
                    dtype
                        .float8_format()
                        .expect("float8 dtype")
                        .encode(value.as_f64()),
                )
            }
            DType::F16 => Self::F16(crate::tensor::f32_to_f16(value.as_f64() as f32)),
            DType::BF16 => Self::BF16(crate::tensor::f32_to_bf16(value.as_f64() as f32)),
            DType::F32 => Self::F32((value.as_f64() as f32).to_bits()),
            DType::F64 => Self::F64(value.as_f64().to_bits()),
            _ => Self::Scalar(cast_scalar(value, dtype)),
        }
    }
    fn scalar(self) -> Scalar {
        match self {
            Self::Scalar(value) => value,
            Self::F8(dtype, bits) => {
                Scalar::F(dtype.float8_format().expect("float8 dtype").decode(bits))
            }
            Self::F16(bits) => Scalar::F(crate::tensor::f16_to_f32(bits) as f64),
            Self::BF16(bits) => Scalar::F(crate::tensor::bf16_to_f32(bits) as f64),
            Self::F32(bits) => Scalar::F(f32::from_bits(bits) as f64),
            Self::F64(bits) => Scalar::F(f64::from_bits(bits)),
        }
    }
    fn cast(self, dtype: DType) -> Self {
        // A same-width floating CAST is still an explicit source operation,
        // but its logical storage value is unchanged. Keep the raw encoding
        // rather than widening an F32 signaling NaN through Scalar::F.
        match (self, dtype) {
            (value @ Self::F8(source, _), target) if source == target => return value,
            (value @ Self::F16(_), DType::F16)
            | (value @ Self::BF16(_), DType::BF16)
            | (value @ Self::F32(_), DType::F32)
            | (value @ Self::F64(_), DType::F64) => return value,
            _ => {}
        }
        match dtype {
            dtype @ (DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) => {
                Self::F8(
                    dtype,
                    dtype
                        .float8_format()
                        .expect("float8 dtype")
                        .encode(self.scalar().as_f64()),
                )
            }
            DType::F16 => Self::F16(crate::tensor::f32_to_f16(self.scalar().as_f64() as f32)),
            DType::BF16 => {
                let source = match self {
                    // Do not route F32 through Scalar::F(f64): tinygrad's
                    // BF16 CAST sees the original F32 payload bits.
                    Self::F32(bits) => f32::from_bits(bits),
                    value => value.scalar().as_f64() as f32,
                };
                Self::BF16(crate::tensor::f32_to_bf16(source))
            }
            DType::F32 => Self::F32((self.scalar().as_f64() as f32).to_bits()),
            DType::F64 => Self::F64(self.scalar().as_f64().to_bits()),
            _ => Self::Scalar(cast_scalar(self.scalar(), dtype)),
        }
    }
    fn from_constant(dtype: DType, bits: u64) -> Self {
        match dtype {
            dtype @ (DType::F8E4M3 | DType::F8E5M2 | DType::F8E4M3FNUZ | DType::F8E5M2FNUZ) => {
                Self::F8(dtype, bits as u8)
            }
            DType::F16 => Self::F16(bits as u16),
            DType::BF16 => Self::BF16(bits as u16),
            DType::F32 => Self::F32(bits as u32),
            DType::F64 => Self::F64(bits),
            DType::Bool => Self::Scalar(Scalar::Bool(bits != 0)),
            DType::I8 => Self::Scalar(Scalar::I(bits as i8 as i64)),
            DType::U8 => Self::Scalar(Scalar::U(bits as u8 as u64)),
            DType::I16 => Self::Scalar(Scalar::I(bits as i16 as i64)),
            DType::U16 => Self::Scalar(Scalar::U(bits as u16 as u64)),
            DType::I32 => Self::Scalar(Scalar::I(bits as i32 as i64)),
            DType::U32 => Self::Scalar(Scalar::U(bits as u32 as u64)),
            DType::I64 => Self::Scalar(Scalar::I(bits as i64)),
            DType::U64 => Self::Scalar(Scalar::U(bits)),
        }
    }
    fn from_storage(storage: &Storage, index: usize) -> Self {
        match storage {
            Storage::Float8(values) => Self::F8(values.format().dtype(), values.as_raw()[index]),
            Storage::F16(values) => Self::F16(values[index]),
            Storage::BF16(values) => Self::BF16(values[index]),
            Storage::F32(values) => Self::F32(values[index].to_bits()),
            Storage::F64(values) => Self::F64(values[index].to_bits()),
            _ => Self::Scalar(storage.scalar(index)),
        }
    }
    fn into_storage(self, dtype: DType) -> Scalar {
        match (dtype, self) {
            (target, Self::F8(source, bits)) if target == source => {
                Scalar::F(source.float8_format().expect("float8 dtype").decode(bits))
            }
            (DType::F16, Self::F16(bits)) => Scalar::F(crate::tensor::f16_to_f32(bits) as f64),
            (DType::BF16, Self::BF16(bits)) => Scalar::F(crate::tensor::bf16_to_f32(bits) as f64),
            (DType::F32, Self::F32(bits)) => Scalar::F(f32::from_bits(bits) as f64),
            (DType::F64, Self::F64(bits)) => Scalar::F(f64::from_bits(bits)),
            (_, value) => value.scalar(),
        }
    }
}
fn eval(
    n: &UOp,
    bindings: &KernelBindings,
    linear: usize,
    plan: &IterationPlan,
) -> Result<FusedValue> {
    match n.operation() {
        Operation::Const(LiteralValue::Int(v)) => Ok(FusedValue::Scalar(Scalar::I(*v))),
        Operation::Const(LiteralValue::Scalar { dtype, bits }) => {
            Ok(FusedValue::from_constant(*dtype, *bits))
        }
        Operation::Load => {
            let index = n.sources().first().ok_or(Error::InvalidIndex)?;
            let (buffer, input_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    input_shape,
                    ..
                }) => (*buffer, input_shape, None),
                Operation::Index(IndexValue::View {
                    buffer,
                    input_shape,
                    view,
                    ..
                }) => (*buffer, input_shape, Some(view)),
                _ => return Err(Error::InvalidIndex),
            };
            let logical = plan.broadcast_offset(input_shape, linear)?;
            let offset = match view {
                Some(view) => view
                    .element_offset(logical)
                    .map_err(|_| Error::InvalidIndex)?,
                None => i64::try_from(logical).map_err(|_| Error::InvalidIndex)?,
            };
            Ok(FusedValue::from_storage(
                bindings.get(buffer).ok_or(Error::InvalidIndex)?.storage(),
                usize::try_from(offset).map_err(|_| Error::InvalidIndex)?,
            ))
        }
        Operation::Cast => Ok(eval(&n.sources()[0], bindings, linear, plan)?
            .cast(n.ty().ok_or(Error::InvalidIndex)?.scalar)),
        Operation::GraphUnary(op) => {
            let input = eval(&n.sources()[0], bindings, linear, plan)?;
            let input_dtype = n.sources()[0].ty().ok_or(Error::InvalidIndex)?.scalar;
            Ok(FusedValue::typed(
                unary(input.scalar(), input_dtype, *op)?,
                n.ty().ok_or(Error::InvalidIndex)?.scalar,
            ))
        }
        Operation::GraphBinary(op) => Ok(FusedValue::typed(
            binary(
                eval(&n.sources()[0], bindings, linear, plan)?.scalar(),
                eval(&n.sources()[1], bindings, linear, plan)?.scalar(),
                n.ty().ok_or(Error::InvalidIndex)?.scalar,
                *op,
            )?,
            n.ty().ok_or(Error::InvalidIndex)?.scalar,
        )),
        Operation::GraphCompare(op) => {
            let lhs = eval(&n.sources()[0], bindings, linear, plan)?.scalar();
            let rhs = eval(&n.sources()[1], bindings, linear, plan)?.scalar();
            let float8 = n
                .sources()
                .iter()
                .any(|source| source.ty().is_some_and(|ty| ty.scalar.is_float8()));
            Ok(FusedValue::Scalar(Scalar::Bool(if float8 {
                compare_float8(lhs.as_f64(), rhs.as_f64(), *op)
            } else {
                compare(lhs, rhs, *op)
            })))
        }
        Operation::GraphLogical(op) => {
            let lhs = eval(&n.sources()[0], bindings, linear, plan)?.scalar();
            Ok(FusedValue::Scalar(evaluate_constant_logical(
                lhs,
                *op,
                || {
                    n.sources()
                        .get(1)
                        .ok_or(Error::InvalidIndex)
                        .and_then(|rhs| eval(rhs, bindings, linear, plan))
                        .map(FusedValue::scalar)
                },
            )?))
        }
        Operation::Ternary(crate::uop::Ternary::Where) => {
            if eval(&n.sources()[0], bindings, linear, plan)?
                .scalar()
                .as_bool()
            {
                eval(&n.sources()[1], bindings, linear, plan)
            } else {
                eval(&n.sources()[2], bindings, linear, plan)
            }
        }
        Operation::ReduceFinalize => {
            let update = n.sources().first().ok_or(Error::InvalidIndex)?;
            let init = update.sources().first().ok_or(Error::InvalidIndex)?;
            let Operation::ReduceInit(ReductionValue {
                input_shape,
                output_shape,
                axes,
                keepdim,
                kind,
                mean,
            }) = init.operation()
            else {
                return Err(Error::InvalidIndex);
            };
            if &plan.output != output_shape {
                return Err(Error::InvalidIndex);
            }
            let reduction = ReductionPlan::new(
                input_shape.clone(),
                output_shape.clone(),
                axes.to_vec(),
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
                crate::ReduceKind::Any => Scalar::Bool(false),
                crate::ReduceKind::All => Scalar::Bool(true),
            };
            let mut extrema_seen = false;
            for reduce_linear in 0..reduction_len {
                let next = eval(
                    value,
                    bindings,
                    reduction.input_linear(linear, reduce_linear)?,
                    &source_plan,
                )?
                .scalar();
                acc = match kind {
                    crate::ReduceKind::Sum | crate::ReduceKind::Mean => {
                        binary(acc, next, dtype, BinaryOp::Add)?
                    }
                    crate::ReduceKind::Product => binary(acc, next, dtype, BinaryOp::Mul)?,
                    crate::ReduceKind::Max | crate::ReduceKind::Min => {
                        let replace = !extrema_seen
                            || reduction_extrema_is_better(
                                dtype,
                                matches!(kind, crate::ReduceKind::Max),
                                next,
                                acc,
                            );
                        extrema_seen = true;
                        if replace { next } else { acc }
                    }
                    crate::ReduceKind::Any => Scalar::Bool(acc.as_bool() || next.as_bool()),
                    crate::ReduceKind::All => Scalar::Bool(acc.as_bool() && next.as_bool()),
                };
            }
            if *mean {
                acc = Scalar::F(if reduction_len == 0 {
                    f64::NAN
                } else {
                    acc.as_f64() / reduction_len as f64
                });
            }
            Ok(FusedValue::typed(acc, dtype))
        }
        _ => Err(Error::InvalidIndex),
    }
}

/// Match the CPU oracle's stored-lane extrema ordering. A leading NaN and
/// equal signed-zero lanes keep their first payload, while I64/U64 lanes are
/// compared without a lossy floating projection.
fn reduction_extrema_is_better(dtype: DType, max: bool, candidate: Scalar, best: Scalar) -> bool {
    use std::cmp::Ordering;

    let ordering = if dtype.is_float() {
        candidate.as_f64().partial_cmp(&best.as_f64())
    } else if matches!(dtype.category(), crate::DTypeCategory::Unsigned) {
        Some(candidate.as_u64().cmp(&best.as_u64()))
    } else if dtype == DType::Bool {
        Some(candidate.as_bool().cmp(&best.as_bool()))
    } else {
        Some(candidate.as_i64().cmp(&best.as_i64()))
    };
    if max {
        ordering == Some(Ordering::Greater)
    } else {
        ordering == Some(Ordering::Less)
    }
}
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
            (DType::U8 | DType::U16 | DType::U32 | DType::U64, UnaryOp::Step) => {
                Scalar::U(u64::from(x.as_u64() > 0))
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
            BinaryOp::Maximum => {
                if a < b {
                    b
                } else {
                    a
                }
            }
            BinaryOp::Minimum => {
                if a > b {
                    b
                } else {
                    a
                }
            }
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
            BinaryOp::Maximum => {
                if a < b {
                    b
                } else {
                    a
                }
            }
            BinaryOp::Minimum => {
                if a > b {
                    b
                } else {
                    a
                }
            }
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
        BinaryOp::Maximum => {
            if a < b {
                b
            } else {
                a
            }
        }
        BinaryOp::Minimum => {
            if a > b {
                b
            } else {
                a
            }
        }
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
            if a < 0 {
                Some(Ordering::Less)
            } else {
                Some((a as u64).cmp(&b))
            }
        }
        (Scalar::U(a), Scalar::I(b)) => {
            if b < 0 {
                Some(Ordering::Greater)
            } else {
                Some(a.cmp(&(b as u64)))
            }
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

/// Shared exact scalar semantics for compile-time evaluation. Callers select
/// the operations that are safe to fold; this seam only prevents the
/// optimizer and interpreter from growing independent numeric evaluators.
pub(crate) fn evaluate_constant_unary(
    value: Scalar,
    dtype: DType,
    operation: UnaryOp,
) -> Result<Scalar> {
    unary(value, dtype, operation)
}

pub(crate) fn evaluate_constant_binary(
    lhs: Scalar,
    rhs: Scalar,
    dtype: DType,
    operation: BinaryOp,
) -> Result<Scalar> {
    binary(lhs, rhs, dtype, operation)
}

pub(crate) fn evaluate_constant_compare(lhs: Scalar, rhs: Scalar, operation: CompareOp) -> bool {
    compare(lhs, rhs, operation)
}

pub(crate) fn evaluate_constant_logical<F>(
    lhs: Scalar,
    operation: LogicalOp,
    rhs: F,
) -> Result<Scalar>
where
    F: FnOnce() -> Result<Scalar>,
{
    let lhs = lhs.as_bool();
    Ok(Scalar::Bool(match operation {
        LogicalOp::Not => !lhs,
        LogicalOp::And if !lhs => false,
        LogicalOp::Or if lhs => true,
        LogicalOp::And => rhs()?.as_bool(),
        LogicalOp::Or => rhs()?.as_bool(),
    }))
}

fn compare_float8(a: f64, b: f64, op: CompareOp) -> bool {
    use std::cmp::Ordering;

    // Float8 follows the public tinygrad comparison construction. Inclusive
    // comparisons are logical-not shells around the opposite strict compare,
    // so an unordered (NaN) lane is true for Le/Ge.
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => b.partial_cmp(&a) != Some(Ordering::Less),
        CompareOp::Gt => b < a,
        CompareOp::Ge => a.partial_cmp(&b) != Some(Ordering::Less),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Shape, SymbolicExpr};

    #[test]
    fn reciprocal_promotes_nonfloats_before_homogeneous_graph_unary_lowering() {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let output = graph.reciprocal(input).unwrap();

            // Public reciprocal explicitly promotes its nonfloat input to
            // F32, so the raw GraphUnary has matching operand/result types.
            let uop = lower_graph_elementwise(&graph, output).unwrap();
            uop.validate().unwrap();
        }
    }

    #[test]
    fn scalar_graph_constants_lower_to_typed_uop_payloads_without_buffer_dependencies() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::Bool);
        let truth = graph.constant(TensorData::scalar_with_dtype(
            Scalar::Bool(true),
            DType::Bool,
        ));
        let output = graph.ne(input, truth).unwrap();
        let uop = lower_graph_elementwise(&graph, output).unwrap();
        let nodes = uop.topological().unwrap();
        assert!(nodes.iter().any(|node| matches!(
            node.operation(),
            Operation::Const(LiteralValue::Scalar {
                dtype: DType::Bool,
                bits: 1
            })
        )));
        assert!(!nodes.iter().any(|node| matches!(
            node.operation(),
            Operation::Index(IndexValue::Buffer { buffer, .. }) if *buffer == truth.index() as u64
        )));

        // Exact scalar payloads are carried through the UOp rather than host
        // floating conversion, including an F32 NaN bit pattern.
        let mut payload = Graph::new();
        let input = payload.input_dtype("input", [], DType::F32);
        let nan = payload.constant(
            TensorData::from_storage(
                Shape::new([]),
                Storage::F32(vec![f32::from_bits(0x7f80_0001)]),
            )
            .unwrap(),
        );
        let output = payload.eq(input, nan).unwrap();
        let nodes = lower_graph_elementwise(&payload, output)
            .unwrap()
            .topological()
            .unwrap();
        assert!(nodes.iter().any(|node| matches!(
            node.operation(),
            Operation::Const(LiteralValue::Scalar {
                dtype: DType::F32,
                bits: 0x7f80_0001
            })
        )));

        // A rank-one singleton is not a scalar provenance value: it remains a
        // bound constant buffer and therefore preserves existing scheduling
        // and capture ownership.
        let mut nonscalar = Graph::new();
        let input = nonscalar.input_dtype("input", [2], DType::F32);
        let constant = nonscalar
            .constant(TensorData::from_scalars([1], DType::F32, [Scalar::F(1.0)]).unwrap());
        let output = nonscalar.add(input, constant).unwrap();
        let nodes = lower_graph_elementwise(&nonscalar, output)
            .unwrap()
            .topological()
            .unwrap();
        assert!(nodes.iter().any(|node| matches!(
            node.operation(),
            Operation::Index(IndexValue::Buffer { buffer, .. }) if *buffer == constant.index() as u64
        )));
    }

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
        let compare_all = |lhs_dtype,
                           lhs_values: Vec<Scalar>,
                           rhs_dtype,
                           rhs_values: Vec<Scalar>| {
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
                        TensorData::from_scalars([lhs_values.len()], lhs_dtype, lhs_values.clone())
                            .unwrap(),
                    ),
                    (
                        "rhs".into(),
                        TensorData::from_scalars([rhs_values.len()], rhs_dtype, rhs_values.clone())
                            .unwrap(),
                    ),
                ]);
                assert_eq!(
                    execute_elementwise(&graph, output, &inputs)
                        .unwrap()
                        .storage(),
                    CpuBackend
                        .execute(&graph, output, &inputs)
                        .unwrap()
                        .storage(),
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
                    TensorData::from_scalars(
                        [3],
                        DType::F64,
                        [
                            Scalar::F(f64::NAN),
                            Scalar::F(-0.0),
                            Scalar::F(f64::INFINITY),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [3],
                        DType::F64,
                        [Scalar::F(1.0), Scalar::F(0.0), Scalar::F(f64::INFINITY)],
                    )
                    .unwrap(),
                ),
            ]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
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
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
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
                TensorData::from_scalars([1], DType::U64, [Scalar::U((1_u64 << 53) + 1)]).unwrap(),
            ),
        ]);
        assert_eq!(
            execute_elementwise(&graph, output, &inputs)
                .unwrap()
                .storage(),
            CpuBackend
                .execute(&graph, output, &inputs)
                .unwrap()
                .storage(),
        );
    }

    #[test]
    fn captured_reductions_match_cpu_extrema_lane_order_for_floats_and_wide_integers() {
        let cases = [
            (
                DType::F32,
                crate::ReduceKind::Max,
                vec![Scalar::F(f32::NAN as f64), Scalar::F(f32::INFINITY as f64)],
            ),
            (
                DType::F32,
                crate::ReduceKind::Min,
                vec![Scalar::F(-0.0), Scalar::F(0.0)],
            ),
            (
                DType::U64,
                crate::ReduceKind::Max,
                vec![Scalar::U((1_u64 << 53) + 1), Scalar::U(1_u64 << 53)],
            ),
            (
                DType::I64,
                crate::ReduceKind::Min,
                vec![Scalar::I(-((1_i64 << 53) + 1)), Scalar::I(-(1_i64 << 53))],
            ),
        ];
        for (dtype, kind, values) in cases {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [values.len()], dtype);
            let output = graph.reduce(input, kind, Some(vec![0]), false).unwrap();
            let inputs = HashMap::from([(
                "input".into(),
                TensorData::from_scalars([values.len()], dtype, values).unwrap(),
            )]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                "{dtype:?} {kind:?}",
            );
        }
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
    fn fused_cast_values_keep_typed_storage_through_mul_and_select() {
        for (dtype, value, multiplier) in [
            (DType::F16, 1.0003, 1000.0),
            (DType::BF16, 1.003, 100.0),
            (DType::F32, 1.00000006, 100_000_000.0),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", Shape::from([1]), DType::F64);
            let cast = graph.cast(input, dtype).unwrap();
            let scale = graph
                .constant(TensorData::from_scalars([1], dtype, [Scalar::F(multiplier)]).unwrap());
            let output = graph.mul(cast, scale).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars([1], DType::F64, [Scalar::F(value)]).unwrap(),
            )]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                "{dtype:?} Cast->Mul storage boundary"
            );
        }

        // Select must retain the selected tagged value; otherwise this
        // payload-sensitive F32->BF16 Cast would regress after fusion.
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", Shape::from([1]), DType::F32);
        let condition = graph.input_dtype("condition", Shape::from([1]), DType::Bool);
        let cast = graph.cast(input, DType::BF16).unwrap();
        let fallback = graph
            .constant(TensorData::from_storage(Shape::from([1]), Storage::BF16(vec![0])).unwrap());
        let output = graph.select(condition, cast, fallback).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::from_storage(
                    Shape::from([1]),
                    Storage::F32(vec![f32::from_bits(0x7f80_0001)]),
                )
                .unwrap(),
            ),
            (
                "condition".into(),
                TensorData::from_scalars([1], DType::Bool, [Scalar::Bool(true)]).unwrap(),
            ),
        ]);
        assert_eq!(
            execute_elementwise(&graph, output, &inputs)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            CpuBackend
                .execute(&graph, output, &inputs)
                .unwrap()
                .to_le_bytes()
                .unwrap(),
            "F32->BF16 signaling-NaN payload through Select"
        );

        for (dtype, value) in [
            (DType::I64, Scalar::I((1_i64 << 53) + 1)),
            (DType::U64, Scalar::U((1_u64 << 53) + 1)),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", Shape::from([1]), dtype);
            let cast = graph.cast(input, DType::F32).unwrap();
            let scale = graph.constant(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(1.00000006)]).unwrap(),
            );
            let output = graph.mul(cast, scale).unwrap();
            let inputs = HashMap::from([(
                "x".into(),
                TensorData::from_scalars([1], dtype, [value]).unwrap(),
            )]);
            assert_eq!(
                execute_elementwise(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                CpuBackend
                    .execute(&graph, output, &inputs)
                    .unwrap()
                    .storage(),
                "large {dtype:?} to F32 Cast"
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
                .any(|node| matches!(node.operation(), Operation::Index(IndexValue::View { .. })))
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
