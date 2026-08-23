//! Typed, owned bindings and a portable interpreter for elementwise UOp kernels.
//!
//! This is intentionally not a backend: bindings clone their `TensorData`, so a
//! scheduled kernel cannot retain or alias a caller's storage.  Element offsets
//! are checked separately from byte offsets, which keeps the ABI boundary
//! explicit for future renderers.
use crate::{
    BinaryOp, CompareOp, DType, Error, Graph, LogicalOp, NodeId, Op, Result, Scalar, Shape,
    SymbolicShape, SymbolicVar, TensorData, UArg, UOp, UOpError, UOpKind, UType, UnaryOp,
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
            UArg::BufferIndex {
                buffer: id.index() as u64,
                elements,
                input_shape: shape,
                output_shape: out.clone(),
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
    ) -> std::result::Result<UOp, UOpError> {
        if let Some(v) = memo.get(&id) {
            return Ok(v.clone());
        }
        let ty = UType::scalar(graph.dtype(id).map_err(|_| UOpError::UseBeforeDefinition)?);
        let x = match graph.op(id).map_err(|_| UOpError::UseBeforeDefinition)? {
            Op::Input { .. } | Op::Constant(_) => load(graph, id, out, range)?,
            Op::Cast { input, .. } => UOp::cast(lower(graph, *input, out, range, memo)?, ty),
            Op::Unary { op, input } => UOp::new(
                UOpKind::GraphUnary(*op),
                Some(ty),
                vec![lower(graph, *input, out, range, memo)?],
                UArg::None,
            ),
            Op::Binary { op, lhs, rhs } => UOp::new(
                UOpKind::GraphBinary(*op),
                Some(ty),
                vec![
                    lower(graph, *lhs, out, range, memo)?,
                    lower(graph, *rhs, out, range, memo)?,
                ],
                UArg::None,
            ),
            Op::Compare { op, lhs, rhs } => UOp::new(
                UOpKind::GraphCompare(*op),
                Some(ty),
                vec![
                    lower(graph, *lhs, out, range, memo)?,
                    lower(graph, *rhs, out, range, memo)?,
                ],
                UArg::None,
            ),
            Op::Logical { op, lhs, rhs } => {
                let mut s = vec![lower(graph, *lhs, out, range, memo)?];
                if let Some(rhs) = rhs {
                    s.push(lower(graph, *rhs, out, range, memo)?);
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
                    lower(graph, *condition, out, range, memo)?,
                    lower(graph, *on_true, out, range, memo)?,
                    lower(graph, *on_false, out, range, memo)?,
                ],
                UArg::None,
            ),
            _ => return Err(UOpError::InvalidArgument),
        };
        memo.insert(id, x.clone());
        Ok(x)
    }
    let value = lower(graph, output, &output_shape, &range, &mut HashMap::new())?;
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

/// Executes the typed range/load/store UOp form without invoking `CpuBackend`.
pub fn execute_elementwise(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> Result<TensorData> {
    let kernel = lower_graph_elementwise(graph, output).map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    kernel.validate().map_err(|e| Error::Serialization {
        reason: e.to_string(),
    })?;
    let output_shape = graph.shape(output)?.clone();
    let output_dtype = graph.dtype(output)?;
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
    let plan = IterationPlan::new(output_shape.clone());
    let len = plan.len()?;
    let mut values = Vec::with_capacity(len);
    for linear in 0..len {
        values.push(eval_store_value(
            kernel.sources().first().ok_or(Error::InvalidIndex)?,
            &bindings,
            linear,
            &plan,
        )?);
    }
    TensorData::from_scalars(output_shape, output_dtype, values)
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
            _ => Err(Error::InvalidIndex),
        },
        UOpKind::Load => {
            let index = n.sources().first().ok_or(Error::InvalidIndex)?;
            let UArg::BufferIndex {
                buffer,
                input_shape,
                ..
            } = index.arg()
            else {
                return Err(Error::InvalidIndex);
            };
            let offset = plan.broadcast_offset(input_shape, linear)?;
            bindings
                .get(*buffer)
                .ok_or(Error::InvalidIndex)?
                .storage()
                .scalar(offset)
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
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I(x.as_i64()),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => Scalar::U(x.as_u64()),
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
        UnaryOp::Sign => Scalar::F(if v.is_nan() { f64::NAN } else { v.signum() }),
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
            BinaryOp::Maximum => a.max(b),
            BinaryOp::Minimum => a.min(b),
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
            BinaryOp::Maximum => a.max(b),
            BinaryOp::Minimum => a.min(b),
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
        BinaryOp::Maximum => a.max(b),
        BinaryOp::Minimum => a.min(b),
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
    let (a, b) = (a.as_f64(), b.as_f64());
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Ne => a != b,
        CompareOp::Lt => a < b,
        CompareOp::Le => a <= b,
        CompareOp::Gt => a > b,
        CompareOp::Ge => a >= b,
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
}
