//! Bounded portable node-table encoding for validated UOps.
use super::{AddressSpace, Binary, UArg, UOp, UOpKind, UType, Unary, ViewMap};
use crate::{
    BinaryOp, CompareOp, DType, LogicalOp, MatmulKernelPlan, NodeId, ReduceKind, Shape,
    SymbolicExpr, UnaryOp,
};
use std::{collections::BTreeMap, fmt};

const MAGIC: &[u8; 4] = b"RGUA";
const VERSION: u8 = 3;
const MAX_BYTES: usize = 64 << 20;
const MAX_NODES: usize = 1 << 20;
const MAX_SOURCES: usize = 1 << 20;
const MAX_COLLECTION: usize = 1 << 20;
const MAX_STRING: usize = 1 << 20;
const MAX_SYMBOLIC_DEPTH: usize = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactError {
    Format(&'static str),
    Unsupported,
    Checksum,
}
impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "artifact: {self:?}")
    }
}
impl std::error::Error for ArtifactError {}

/// Encodes one immutable UOp DAG. Node IDs are dense topological indices and
/// repeated source IDs preserve shared subgraphs.
pub fn encode(root: &UOp) -> Result<Vec<u8>, ArtifactError> {
    root.validate().map_err(|_| ArtifactError::Format("uop"))?;
    let nodes = root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?;
    if nodes.is_empty() || nodes.len() > MAX_NODES {
        return Err(ArtifactError::Format("count"));
    }
    let ids = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i as u32))
        .collect::<BTreeMap<_, _>>();
    let mut w = Writer::new();
    w.bytes(MAGIC)?;
    w.u8(VERSION)?;
    w.u32(nodes.len() as u32)?;
    w.u32((nodes.len() - 1) as u32)?;
    for (id, node) in nodes.iter().enumerate() {
        validate_fields(node.kind(), node.ty(), node.arg(), node.sources())?;
        w.u32(id as u32)?;
        write_kind(&mut w, node.kind())?;
        write_type(&mut w, node.ty())?;
        write_arg(&mut w, node.arg())?;
        if node.sources().len() > MAX_SOURCES {
            return Err(ArtifactError::Format("source limit"));
        }
        w.u32(node.sources().len() as u32)?;
        for source in node.sources() {
            let source_id = *ids.get(source).ok_or(ArtifactError::Format("source"))?;
            if source_id >= id as u32 {
                return Err(ArtifactError::Format("source order"));
            }
            w.u32(source_id)?;
        }
    }
    let sum = checksum(&w.out);
    w.u32(sum)?;
    Ok(w.out)
}

/// Decodes and completely validates an artifact before returning its root.
pub fn decode(bytes: &[u8]) -> Result<UOp, ArtifactError> {
    if bytes.len() > MAX_BYTES {
        return Err(ArtifactError::Format("byte limit"));
    }
    if bytes.len() < 17 {
        return Err(ArtifactError::Format("length"));
    }
    let body_len = bytes.len() - 4;
    let got = u32::from_le_bytes(
        bytes[body_len..]
            .try_into()
            .map_err(|_| ArtifactError::Format("checksum"))?,
    );
    if checksum(&bytes[..body_len]) != got {
        return Err(ArtifactError::Checksum);
    }
    let mut r = Reader::new(&bytes[..body_len]);
    if r.take(4)? != MAGIC {
        return Err(ArtifactError::Format("magic"));
    }
    let version = r.u8()?;
    if !matches!(version, 2 | VERSION) {
        return Err(ArtifactError::Format("version"));
    }
    let count = r.count(MAX_NODES)?;
    if count == 0 {
        return Err(ArtifactError::Format("count"));
    }
    let root = r.u32()? as usize;
    if root != count - 1 {
        return Err(ArtifactError::Format("root"));
    }
    let mut nodes: Vec<UOp> = Vec::with_capacity(count);
    for expected in 0..count {
        if r.u32()? as usize != expected {
            return Err(ArtifactError::Format("node id"));
        }
        let kind = read_kind(&mut r, version)?;
        let ty = read_type(&mut r)?;
        let arg = read_arg(&mut r, version)?;
        let source_count = r.count(MAX_SOURCES)?;
        let mut sources = Vec::with_capacity(source_count);
        for _ in 0..source_count {
            let source = r.u32()? as usize;
            if source >= expected {
                return Err(ArtifactError::Format("forward source"));
            }
            sources.push(nodes[source].clone());
        }
        validate_fields(&kind, ty, &arg, &sources)?;
        nodes.push(UOp::from_artifact(kind, ty, sources, arg));
    }
    if !r.done() {
        return Err(ArtifactError::Format("trailing bytes"));
    }
    let root = nodes[root].clone();
    root.validate().map_err(|_| ArtifactError::Format("uop"))?;
    if root
        .topological()
        .map_err(|_| ArtifactError::Format("dag"))?
        .len()
        != count
    {
        return Err(ArtifactError::Format("unreachable node"));
    }
    Ok(root)
}

fn validate_fields(
    kind: &UOpKind,
    ty: Option<UType>,
    arg: &UArg,
    sources: &[UOp],
) -> Result<(), ArtifactError> {
    if ty.is_some_and(|x| x.lanes == 0) {
        return Err(ArtifactError::Format("lane width"));
    }
    match arg {
        UArg::Scalar { dtype, bits } => {
            let used = dtype.bits() as usize;
            if used < 64 && bits >> used != 0 {
                return Err(ArtifactError::Format("scalar bits"));
            }
        }
        UArg::Variable { name, bounds } => {
            if name.is_empty() || bounds.bounds().is_err() {
                return Err(ArtifactError::Format("variable"));
            }
            let mut variables = BTreeMap::new();
            for variable in bounds.variables() {
                let metadata = (variable.name().to_owned(), variable.bounds());
                if variables
                    .insert(variable.id(), metadata.clone())
                    .is_some_and(|old| old != metadata)
                {
                    return Err(ArtifactError::Format("symbolic identity"));
                }
            }
        }
        UArg::Address { name, element, .. } => {
            if name.is_empty() || element.lanes == 0 {
                return Err(ArtifactError::Format("address"));
            }
        }
        UArg::BufferIndex {
            elements,
            input_shape,
            output_shape,
            ..
        } => validate_index(*elements, input_shape, output_shape, None)?,
        UArg::ViewBufferIndex {
            elements,
            input_shape,
            output_shape,
            view,
            ..
        } => validate_index(*elements, input_shape, output_shape, Some(view))?,
        UArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            ..
        } => {
            checked_shape(input_shape)?;
            checked_shape(output_shape)?;
            if axes.windows(2).any(|x| x[0] >= x[1])
                || axes.iter().any(|x| *x >= input_shape.rank())
            {
                return Err(ArtifactError::Format("reduction axes"));
            }
            let mut want = input_shape.dims().to_vec();
            if *keepdim {
                for axis in axes {
                    want[*axis] = 1;
                }
            } else {
                for axis in axes.iter().rev() {
                    want.remove(*axis);
                }
            }
            if output_shape.dims() != want {
                return Err(ArtifactError::Format("reduction shape"));
            }
        }
        UArg::Matmul(plan) => plan
            .validate()
            .map_err(|_| ArtifactError::Format("matmul plan"))?,
        _ => {}
    }
    let arg_ok = match kind {
        UOpKind::Const | UOpKind::VConst => matches!(arg, UArg::Int(_) | UArg::Scalar { .. }),
        UOpKind::DefineVar => matches!(arg, UArg::Variable { .. }),
        UOpKind::DefineGlobal | UOpKind::DefineLocal | UOpKind::DefineRegister => {
            matches!(arg, UArg::Address { .. })
        }
        UOpKind::Special => matches!(arg, UArg::Name(_)),
        UOpKind::Range => matches!(arg, UArg::RangeAxis(_)),
        UOpKind::Gep => matches!(arg, UArg::GepLane(_)),
        UOpKind::Index => matches!(arg, UArg::BufferIndex { .. } | UArg::ViewBufferIndex { .. }),
        UOpKind::ReduceInit => matches!(arg, UArg::Reduction { .. }),
        UOpKind::Matmul => matches!(arg, UArg::Matmul(_)),
        _ => matches!(arg, UArg::None),
    };
    if !arg_ok {
        return Err(ArtifactError::Format("kind argument"));
    }
    let arity_ok = match kind {
        UOpKind::Const
        | UOpKind::VConst
        | UOpKind::DefineVar
        | UOpKind::DefineGlobal
        | UOpKind::DefineLocal
        | UOpKind::DefineRegister
        | UOpKind::Special
        | UOpKind::Matmul
        | UOpKind::ReduceInit
        | UOpKind::Barrier => sources.is_empty(),
        UOpKind::Range
        | UOpKind::EndRange
        | UOpKind::If
        | UOpKind::EndIf
        | UOpKind::Unary(_)
        | UOpKind::GraphUnary(_)
        | UOpKind::ReduceFinalize
        | UOpKind::Cast
        | UOpKind::Bitcast
        | UOpKind::Gep
        | UOpKind::Load => sources.len() == 1,
        UOpKind::Binary(_)
        | UOpKind::GraphBinary(_)
        | UOpKind::GraphCompare(_)
        | UOpKind::ReduceAccumulate
        | UOpKind::Index
        | UOpKind::Store => sources.len() == 2,
        UOpKind::GraphLogical(LogicalOp::Not) => sources.len() == 1,
        UOpKind::GraphLogical(LogicalOp::And | LogicalOp::Or) => sources.len() == 2,
        UOpKind::Ternary(super::Ternary::Where) => sources.len() == 3,
        UOpKind::Vectorize => !sources.is_empty(),
        UOpKind::Sink => true,
    };
    if !arity_ok {
        return Err(ArtifactError::Format("kind sources"));
    }
    let type_ok = match kind {
        UOpKind::Const | UOpKind::VConst => ty.is_some_and(|node_ty| match arg {
            UArg::Scalar { dtype, .. } => node_ty.scalar == *dtype,
            UArg::Int(_) => true,
            _ => false,
        }),
        UOpKind::DefineGlobal => {
            matches!(arg, UArg::Address { space: AddressSpace::Global, element, .. } if ty == Some(*element))
        }
        UOpKind::DefineLocal => {
            matches!(arg, UArg::Address { space: AddressSpace::Local, element, .. } if ty == Some(*element))
        }
        UOpKind::DefineRegister => {
            matches!(arg, UArg::Address { space: AddressSpace::Register, element, .. } if ty == Some(*element))
        }
        UOpKind::Range => {
            ty.is_some_and(|x| x.scalar.is_integer())
                && sources.first().is_some_and(|x| x.ty() == ty)
        }
        UOpKind::Index => sources.first().is_some_and(|x| x.ty() == ty),
        UOpKind::Load => sources
            .first()
            .is_some_and(|x| x.ty() == ty && matches!(x.kind(), UOpKind::Index)),
        UOpKind::Store => {
            ty.is_none()
                && sources
                    .first()
                    .zip(sources.get(1))
                    .is_some_and(|(index, value)| {
                        matches!(index.kind(), UOpKind::Index) && index.ty() == value.ty()
                    })
        }
        UOpKind::GraphUnary(_) | UOpKind::Unary(_) => sources.first().is_some_and(|x| x.ty() == ty),
        UOpKind::GraphBinary(_) => ty.is_some() && sources.iter().all(|x| x.ty().is_some()),
        UOpKind::GraphCompare(_) => {
            ty == Some(UType::scalar(DType::Bool)) && sources.iter().all(|x| x.ty().is_some())
        }
        UOpKind::GraphLogical(_) => {
            ty == Some(UType::scalar(DType::Bool)) && sources.iter().all(|x| x.ty() == ty)
        }
        UOpKind::Matmul => {
            matches!(arg, UArg::Matmul(plan) if ty == Some(UType::scalar(plan.dtype)))
        }
        UOpKind::ReduceAccumulate => ty.is_some() && sources.iter().all(|x| x.ty() == ty),
        UOpKind::ReduceFinalize => sources.first().is_some_and(|x| x.ty() == ty),
        UOpKind::Ternary(super::Ternary::Where) => {
            sources
                .first()
                .is_some_and(|x| x.ty() == Some(UType::scalar(DType::Bool)))
                && sources
                    .get(1)
                    .zip(sources.get(2))
                    .is_some_and(|(a, b)| a.ty() == ty && b.ty() == ty)
        }
        UOpKind::Sink => ty.is_none(),
        _ => true,
    };
    if !type_ok {
        return Err(ArtifactError::Format("kind type"));
    }
    Ok(())
}

fn validate_index(
    elements: usize,
    input: &Shape,
    output: &Shape,
    view: Option<&ViewMap>,
) -> Result<(), ArtifactError> {
    checked_shape(input)?;
    checked_shape(output)?;
    if input.numel().ok() != Some(elements)
        || input.rank() > output.rank()
        || !input
            .dims()
            .iter()
            .rev()
            .zip(output.dims().iter().rev())
            .all(|(a, b)| *a == 1 || a == b)
    {
        return Err(ArtifactError::Format("index shape"));
    }
    if let Some(view) = view {
        validate_view(view)?;
        if &view.logical_shape != input {
            return Err(ArtifactError::Format("view logical shape"));
        }
    }
    Ok(())
}

pub(crate) fn validate_view(view: &ViewMap) -> Result<(), ArtifactError> {
    let source = checked_shape(&view.source_shape)?;
    checked_shape(&view.logical_shape)?;
    if view.strides.len() != view.logical_shape.rank() {
        return Err(ArtifactError::Format("view strides"));
    }
    if source == 0 {
        if view.logical_shape.numel().ok() != Some(0) || view.offset != 0 {
            return Err(ArtifactError::Format("empty view"));
        }
        return Ok(());
    }
    let mut max = view.offset;
    for (&dim, &stride) in view.logical_shape.dims().iter().zip(&view.strides) {
        if dim != 0 {
            max = max
                .checked_add(
                    (dim - 1)
                        .checked_mul(stride)
                        .ok_or(ArtifactError::Format("view overflow"))?,
                )
                .ok_or(ArtifactError::Format("view overflow"))?;
        }
    }
    if view.logical_shape.numel().ok() == Some(0) {
        if view.offset > source {
            return Err(ArtifactError::Format("empty view bounds"));
        }
    } else if max >= source {
        return Err(ArtifactError::Format("view bounds"));
    }
    Ok(())
}

fn checked_shape(shape: &Shape) -> Result<usize, ArtifactError> {
    if shape.rank() > MAX_COLLECTION {
        return Err(ArtifactError::Format("rank limit"));
    }
    shape
        .numel()
        .map_err(|_| ArtifactError::Format("shape overflow"))
}

fn write_type(w: &mut Writer, ty: Option<UType>) -> Result<(), ArtifactError> {
    match ty {
        None => w.u8(0),
        Some(ty) if ty.lanes != 0 => {
            w.u8(1)?;
            w.u8(dtype_tag(ty.scalar))?;
            w.u16(ty.lanes)
        }
        Some(_) => Err(ArtifactError::Format("lane width")),
    }
}
fn read_type(r: &mut Reader<'_>) -> Result<Option<UType>, ArtifactError> {
    match r.u8()? {
        0 => Ok(None),
        1 => {
            let scalar = dtype(r.u8()?)?;
            let lanes = r.u16()?;
            if lanes == 0 {
                Err(ArtifactError::Format("lane width"))
            } else {
                Ok(Some(UType { scalar, lanes }))
            }
        }
        _ => Err(ArtifactError::Format("type tag")),
    }
}

fn write_kind(w: &mut Writer, kind: &UOpKind) -> Result<(), ArtifactError> {
    use UOpKind::*;
    let (tag, sub) = match kind {
        Const => (0, None),
        VConst => (1, None),
        DefineVar => (2, None),
        DefineGlobal => (3, None),
        DefineLocal => (4, None),
        DefineRegister => (5, None),
        Special => (6, None),
        Range => (7, None),
        EndRange => (8, None),
        If => (9, None),
        EndIf => (10, None),
        Unary(x) => (11, Some(tag_unary(*x))),
        Binary(x) => (12, Some(tag_binary(*x))),
        GraphUnary(x) => (13, Some(tag_graph_unary(*x))),
        GraphBinary(x) => (14, Some(tag_graph_binary(*x))),
        GraphCompare(x) => (15, Some(tag_compare(*x))),
        GraphLogical(x) => (16, Some(tag_logical(*x))),
        ReduceInit => (17, None),
        ReduceAccumulate => (18, None),
        ReduceFinalize => (19, None),
        Ternary(super::Ternary::Where) => (20, Some(0)),
        Cast => (21, None),
        Bitcast => (22, None),
        Vectorize => (23, None),
        Gep => (24, None),
        Index => (25, None),
        Load => (26, None),
        Store => (27, None),
        Barrier => (28, None),
        Sink => (29, None),
        Matmul => (30, None),
    };
    w.u8(tag)?;
    if let Some(x) = sub {
        w.u8(x)?;
    }
    Ok(())
}
fn read_kind(r: &mut Reader<'_>, version: u8) -> Result<UOpKind, ArtifactError> {
    use UOpKind::*;
    Ok(match r.u8()? {
        0 => Const,
        1 => VConst,
        2 => DefineVar,
        3 => DefineGlobal,
        4 => DefineLocal,
        5 => DefineRegister,
        6 => Special,
        7 => Range,
        8 => EndRange,
        9 => If,
        10 => EndIf,
        11 => Unary(enum_unary(r.u8()?)?),
        12 => Binary(enum_binary(r.u8()?)?),
        13 => GraphUnary(enum_graph_unary(r.u8()?)?),
        14 => GraphBinary(enum_graph_binary(r.u8()?)?),
        15 => GraphCompare(enum_compare(r.u8()?)?),
        16 => GraphLogical(enum_logical(r.u8()?)?),
        17 => ReduceInit,
        18 => ReduceAccumulate,
        19 => ReduceFinalize,
        20 => Ternary(match r.u8()? {
            0 => super::Ternary::Where,
            _ => return Err(ArtifactError::Format("ternary")),
        }),
        21 => Cast,
        22 => Bitcast,
        23 => Vectorize,
        24 => Gep,
        25 => Index,
        26 => Load,
        27 => Store,
        28 => Barrier,
        29 => Sink,
        30 if version >= 3 => Matmul,
        _ => return Err(ArtifactError::Format("kind tag")),
    })
}

fn write_arg(w: &mut Writer, arg: &UArg) -> Result<(), ArtifactError> {
    match arg {
        UArg::None => w.u8(0),
        UArg::Int(x) => {
            w.u8(1)?;
            w.i64(*x)
        }
        UArg::Scalar { dtype, bits } => {
            w.u8(2)?;
            w.u8(dtype_tag(*dtype))?;
            w.u64(*bits)
        }
        UArg::Name(x) => {
            w.u8(3)?;
            w.string(x)
        }
        UArg::Variable { name, bounds } => {
            w.u8(4)?;
            w.string(name)?;
            write_symbolic(w, bounds, 0)
        }
        UArg::Address {
            space,
            name,
            element,
        } => {
            w.u8(5)?;
            w.u8(match space {
                AddressSpace::Global => 0,
                AddressSpace::Local => 1,
                AddressSpace::Register => 2,
            })?;
            w.string(name)?;
            write_type(w, Some(*element))
        }
        UArg::RangeAxis(x) => {
            w.u8(6)?;
            w.u32(*x)
        }
        UArg::GepLane(x) => {
            w.u8(7)?;
            w.u16(*x)
        }
        UArg::BufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
        } => {
            w.u8(8)?;
            w.u64(*buffer)?;
            w.usize(*elements)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)
        }
        UArg::ViewBufferIndex {
            buffer,
            elements,
            input_shape,
            output_shape,
            view,
        } => {
            w.u8(9)?;
            w.u64(*buffer)?;
            w.usize(*elements)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)?;
            write_view(w, view)
        }
        UArg::Reduction {
            input_shape,
            output_shape,
            axes,
            keepdim,
            kind,
            mean,
        } => {
            w.u8(10)?;
            write_shape(w, input_shape)?;
            write_shape(w, output_shape)?;
            w.usizes(axes)?;
            w.bool(*keepdim)?;
            w.u8(tag_reduce(*kind))?;
            w.bool(*mean)
        }
        UArg::Matmul(plan) => {
            w.u8(11)?;
            write_matmul(w, plan)
        }
    }
}
fn read_arg(r: &mut Reader<'_>, version: u8) -> Result<UArg, ArtifactError> {
    Ok(match r.u8()? {
        0 => UArg::None,
        1 => UArg::Int(r.i64()?),
        2 => UArg::Scalar {
            dtype: dtype(r.u8()?)?,
            bits: r.u64()?,
        },
        3 => UArg::Name(r.string()?),
        4 => UArg::Variable {
            name: r.string()?,
            bounds: read_symbolic(r, 0)?,
        },
        5 => UArg::Address {
            space: match r.u8()? {
                0 => AddressSpace::Global,
                1 => AddressSpace::Local,
                2 => AddressSpace::Register,
                _ => return Err(ArtifactError::Format("address space")),
            },
            name: r.string()?,
            element: read_type(r)?.ok_or(ArtifactError::Format("address type"))?,
        },
        6 => UArg::RangeAxis(r.u32()?),
        7 => UArg::GepLane(r.u16()?),
        8 => UArg::BufferIndex {
            buffer: r.u64()?,
            elements: r.usize()?,
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
        },
        9 => UArg::ViewBufferIndex {
            buffer: r.u64()?,
            elements: r.usize()?,
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
            view: read_view(r)?,
        },
        10 => UArg::Reduction {
            input_shape: read_shape(r)?,
            output_shape: read_shape(r)?,
            axes: r.usizes()?,
            keepdim: r.bool()?,
            kind: enum_reduce(r.u8()?)?,
            mean: r.bool()?,
        },
        11 if version >= 3 => UArg::Matmul(Box::new(read_matmul(r)?)),
        _ => return Err(ArtifactError::Format("argument tag")),
    })
}

fn write_matmul(w: &mut Writer, plan: &MatmulKernelPlan) -> Result<(), ArtifactError> {
    plan.validate()
        .map_err(|_| ArtifactError::Format("matmul plan"))?;
    w.u64(plan.lhs.index() as u64)?;
    w.u64(plan.rhs.index() as u64)?;
    w.u64(plan.output.index() as u64)?;
    write_shape(w, &plan.lhs_shape)?;
    write_shape(w, &plan.rhs_shape)?;
    write_shape(w, &plan.output_shape)?;
    w.u8(dtype_tag(plan.lhs_dtype))?;
    w.u8(dtype_tag(plan.rhs_dtype))?;
    w.u8(dtype_tag(plan.dtype))?;
    w.usizes(&plan.batch_shape)?;
    w.usize(plan.m)?;
    w.usize(plan.n)?;
    w.usize(plan.k)?;
    w.bool(plan.lhs_vector)?;
    w.bool(plan.rhs_vector)?;
    w.u64(plan.cache_key)
}

fn read_matmul(r: &mut Reader<'_>) -> Result<MatmulKernelPlan, ArtifactError> {
    let node = |id| {
        usize::try_from(id)
            .map(NodeId::from_index)
            .map_err(|_| ArtifactError::Format("matmul node"))
    };
    let plan = MatmulKernelPlan {
        lhs: node(r.u64()?)?,
        rhs: node(r.u64()?)?,
        output: node(r.u64()?)?,
        lhs_shape: read_shape(r)?,
        rhs_shape: read_shape(r)?,
        output_shape: read_shape(r)?,
        lhs_dtype: dtype(r.u8()?)?,
        rhs_dtype: dtype(r.u8()?)?,
        dtype: dtype(r.u8()?)?,
        batch_shape: r.usizes()?,
        m: r.usize()?,
        n: r.usize()?,
        k: r.usize()?,
        lhs_vector: r.bool()?,
        rhs_vector: r.bool()?,
        cache_key: r.u64()?,
    };
    plan.validate()
        .map_err(|_| ArtifactError::Format("matmul plan"))?;
    Ok(plan)
}

pub(crate) fn write_shape(w: &mut Writer, x: &Shape) -> Result<(), ArtifactError> {
    w.usizes(x.dims())
}
pub(crate) fn read_shape(r: &mut Reader<'_>) -> Result<Shape, ArtifactError> {
    let x = Shape::new(r.usizes()?);
    checked_shape(&x)?;
    Ok(x)
}
pub(crate) fn write_view(w: &mut Writer, x: &ViewMap) -> Result<(), ArtifactError> {
    validate_view(x)?;
    write_shape(w, &x.source_shape)?;
    write_shape(w, &x.logical_shape)?;
    w.usizes(&x.strides)?;
    w.usize(x.offset)
}
pub(crate) fn read_view(r: &mut Reader<'_>) -> Result<ViewMap, ArtifactError> {
    let x = ViewMap {
        source_shape: read_shape(r)?,
        logical_shape: read_shape(r)?,
        strides: r.usizes()?,
        offset: r.usize()?,
    };
    validate_view(&x)?;
    Ok(x)
}

fn write_symbolic(w: &mut Writer, x: &SymbolicExpr, depth: usize) -> Result<(), ArtifactError> {
    if depth >= MAX_SYMBOLIC_DEPTH {
        return Err(ArtifactError::Format("symbolic depth"));
    }
    use SymbolicExpr::*;
    match x {
        Const(v) => {
            w.u8(0)?;
            w.i64(*v)
        }
        Var(v) => {
            let (min, max) = v.bounds();
            w.u8(1)?;
            w.u64(v.id())?;
            w.string(v.name())?;
            w.i64(min)?;
            w.i64(max)
        }
        Add(xs) | Mul(xs) => {
            w.u8(if matches!(x, Add(_)) { 2 } else { 3 })?;
            w.u32_len(xs.len())?;
            for y in xs {
                write_symbolic(w, y, depth + 1)?;
            }
            Ok(())
        }
        Neg(a) | Not(a) => {
            w.u8(if matches!(x, Neg(_)) { 4 } else { 14 })?;
            write_symbolic(w, a, depth + 1)
        }
        FloorDiv(a, b)
        | Mod(a, b)
        | Min(a, b)
        | Max(a, b)
        | Eq(a, b)
        | Lt(a, b)
        | Le(a, b)
        | And(a, b)
        | Or(a, b) => {
            let tag = match x {
                FloorDiv(..) => 5,
                Mod(..) => 6,
                Min(..) => 7,
                Max(..) => 8,
                Eq(..) => 9,
                Lt(..) => 10,
                Le(..) => 11,
                And(..) => 12,
                Or(..) => 13,
                _ => unreachable!(),
            };
            w.u8(tag)?;
            write_symbolic(w, a, depth + 1)?;
            write_symbolic(w, b, depth + 1)
        }
        Where(a, b, c) => {
            w.u8(15)?;
            write_symbolic(w, a, depth + 1)?;
            write_symbolic(w, b, depth + 1)?;
            write_symbolic(w, c, depth + 1)
        }
    }
}
fn read_symbolic(r: &mut Reader<'_>, depth: usize) -> Result<SymbolicExpr, ArtifactError> {
    if depth >= MAX_SYMBOLIC_DEPTH {
        return Err(ArtifactError::Format("symbolic depth"));
    }
    use SymbolicExpr::*;
    let tag = r.u8()?;
    Ok(match tag {
        0 => Const(r.i64()?),
        1 => {
            let id = r.u64()?;
            let name = r.string()?;
            let min = r.i64()?;
            let max = r.i64()?;
            Var(crate::SymbolicVar::from_artifact(id, name, min, max)
                .map_err(|_| ArtifactError::Format("symbolic variable"))?)
        }
        2 | 3 => {
            let n = r.count(MAX_COLLECTION)?;
            let mut xs = Vec::with_capacity(n);
            for _ in 0..n {
                xs.push(read_symbolic(r, depth + 1)?);
            }
            if tag == 2 { Add(xs) } else { Mul(xs) }
        }
        4 => Neg(Box::new(read_symbolic(r, depth + 1)?)),
        tag @ 5..=13 => {
            let a = Box::new(read_symbolic(r, depth + 1)?);
            let b = Box::new(read_symbolic(r, depth + 1)?);
            match tag {
                5 => FloorDiv(a, b),
                6 => Mod(a, b),
                7 => Min(a, b),
                8 => Max(a, b),
                9 => Eq(a, b),
                10 => Lt(a, b),
                11 => Le(a, b),
                12 => And(a, b),
                13 => Or(a, b),
                _ => unreachable!(),
            }
        }
        14 => Not(Box::new(read_symbolic(r, depth + 1)?)),
        15 => Where(
            Box::new(read_symbolic(r, depth + 1)?),
            Box::new(read_symbolic(r, depth + 1)?),
            Box::new(read_symbolic(r, depth + 1)?),
        ),
        _ => return Err(ArtifactError::Format("symbolic tag")),
    })
}

pub(crate) fn checksum(x: &[u8]) -> u32 {
    x.iter().fold(0x811c9dc5u32, |h, b| {
        (h ^ u32::from(*b)).wrapping_mul(0x01000193)
    })
}
pub(crate) fn dtype_tag(d: DType) -> u8 {
    match d {
        DType::Bool => 0,
        DType::I8 => 1,
        DType::U8 => 2,
        DType::I16 => 3,
        DType::U16 => 4,
        DType::I32 => 5,
        DType::U32 => 6,
        DType::I64 => 7,
        DType::U64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
    }
}
pub(crate) fn dtype(t: u8) -> Result<DType, ArtifactError> {
    Ok(match t {
        0 => DType::Bool,
        1 => DType::I8,
        2 => DType::U8,
        3 => DType::I16,
        4 => DType::U16,
        5 => DType::I32,
        6 => DType::U32,
        7 => DType::I64,
        8 => DType::U64,
        9 => DType::F16,
        10 => DType::BF16,
        11 => DType::F32,
        12 => DType::F64,
        _ => return Err(ArtifactError::Format("dtype")),
    })
}
macro_rules! enum_codec{($encode:ident,$decode:ident,$ty:ty,[$($v:path),+$(,)?])=>{
    fn $encode(value:$ty)->u8{let all=[$($v),+];all.iter().position(|x|*x==value).expect("exhaustive enum codec") as u8}
    fn $decode(t:u8)->Result<$ty,ArtifactError>{let all=[$($v),+];all.get(t as usize).copied().ok_or(ArtifactError::Format("enum tag"))}
};}
enum_codec!(
    tag_unary,
    enum_unary,
    Unary,
    [Unary::Neg, Unary::Not, Unary::Abs]
);
enum_codec!(
    tag_binary,
    enum_binary,
    Binary,
    [
        Binary::Add,
        Binary::Sub,
        Binary::Mul,
        Binary::FloorDiv,
        Binary::Mod,
        Binary::Min,
        Binary::Max,
        Binary::Eq,
        Binary::Lt,
        Binary::Le,
        Binary::And,
        Binary::Or
    ]
);
enum_codec!(
    tag_graph_unary,
    enum_graph_unary,
    UnaryOp,
    [
        UnaryOp::Neg,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Relu,
        UnaryOp::Step,
        UnaryOp::Abs,
        UnaryOp::Reciprocal,
        UnaryOp::Square,
        UnaryOp::Sqrt,
        UnaryOp::Rsqrt,
        UnaryOp::Exp2,
        UnaryOp::Log2,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Tan,
        UnaryOp::Sinh,
        UnaryOp::Cosh,
        UnaryOp::Tanh,
        UnaryOp::Erf,
        UnaryOp::Erfc,
        UnaryOp::Asin,
        UnaryOp::Acos,
        UnaryOp::Atan,
        UnaryOp::Asinh,
        UnaryOp::Acosh,
        UnaryOp::Atanh,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Trunc,
        UnaryOp::Round,
        UnaryOp::Sign,
        UnaryOp::IsNan,
        UnaryOp::IsInf,
        UnaryOp::IsFinite
    ]
);
enum_codec!(
    tag_graph_binary,
    enum_graph_binary,
    BinaryOp,
    [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::FloorDiv,
        BinaryOp::TruncDiv,
        BinaryOp::Mod,
        BinaryOp::FMod,
        BinaryOp::BitAnd,
        BinaryOp::BitOr,
        BinaryOp::BitXor,
        BinaryOp::Shl,
        BinaryOp::Shr,
        BinaryOp::Atan2,
        BinaryOp::Copysign
    ]
);
enum_codec!(
    tag_compare,
    enum_compare,
    CompareOp,
    [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge
    ]
);
enum_codec!(
    tag_logical,
    enum_logical,
    LogicalOp,
    [LogicalOp::Not, LogicalOp::And, LogicalOp::Or]
);
enum_codec!(
    tag_reduce,
    enum_reduce,
    ReduceKind,
    [
        ReduceKind::Sum,
        ReduceKind::Mean,
        ReduceKind::Product,
        ReduceKind::Max,
        ReduceKind::Min
    ]
);

pub(crate) struct Writer {
    pub(crate) out: Vec<u8>,
}
impl Writer {
    pub(crate) fn new() -> Self {
        Self { out: Vec::new() }
    }
    fn reserve(&self, n: usize) -> Result<(), ArtifactError> {
        if self.out.len().checked_add(n).is_none_or(|x| x > MAX_BYTES) {
            Err(ArtifactError::Format("byte limit"))
        } else {
            Ok(())
        }
    }
    pub(crate) fn bytes(&mut self, x: &[u8]) -> Result<(), ArtifactError> {
        self.reserve(x.len())?;
        self.out.extend(x);
        Ok(())
    }
    pub(crate) fn u8(&mut self, x: u8) -> Result<(), ArtifactError> {
        self.bytes(&[x])
    }
    pub(crate) fn bool(&mut self, x: bool) -> Result<(), ArtifactError> {
        self.u8(u8::from(x))
    }
    pub(crate) fn u16(&mut self, x: u16) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn u32(&mut self, x: u32) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn u64(&mut self, x: u64) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn i64(&mut self, x: i64) -> Result<(), ArtifactError> {
        self.bytes(&x.to_le_bytes())
    }
    pub(crate) fn usize(&mut self, x: usize) -> Result<(), ArtifactError> {
        self.u64(x as u64)
    }
    fn u32_len(&mut self, n: usize) -> Result<(), ArtifactError> {
        if n > MAX_COLLECTION || n > u32::MAX as usize {
            Err(ArtifactError::Format("collection limit"))
        } else {
            self.u32(n as u32)
        }
    }
    pub(crate) fn string(&mut self, x: &str) -> Result<(), ArtifactError> {
        if x.len() > MAX_STRING {
            return Err(ArtifactError::Format("string limit"));
        }
        self.u32(x.len() as u32)?;
        self.bytes(x.as_bytes())
    }
    pub(crate) fn usizes(&mut self, x: &[usize]) -> Result<(), ArtifactError> {
        self.u32_len(x.len())?;
        for v in x {
            self.usize(*v)?;
        }
        Ok(())
    }
}
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
    last: u8,
}
impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            last: 0,
        }
    }
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], ArtifactError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(ArtifactError::Format("overflow"))?;
        let x = self
            .bytes
            .get(self.pos..end)
            .ok_or(ArtifactError::Format("truncated"))?;
        self.pos = end;
        Ok(x)
    }
    pub(crate) fn u8(&mut self) -> Result<u8, ArtifactError> {
        let x = self.take(1)?[0];
        self.last = x;
        Ok(x)
    }
    pub(crate) fn bool(&mut self) -> Result<bool, ArtifactError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ArtifactError::Format("bool")),
        }
    }
    pub(crate) fn u16(&mut self) -> Result<u16, ArtifactError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub(crate) fn u32(&mut self) -> Result<u32, ArtifactError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(crate) fn u64(&mut self) -> Result<u64, ArtifactError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn i64(&mut self) -> Result<i64, ArtifactError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(crate) fn usize(&mut self) -> Result<usize, ArtifactError> {
        usize::try_from(self.u64()?).map_err(|_| ArtifactError::Format("usize"))
    }
    pub(crate) fn count(&mut self, max: usize) -> Result<usize, ArtifactError> {
        let x = self.u32()? as usize;
        if x > max {
            Err(ArtifactError::Format("count limit"))
        } else {
            Ok(x)
        }
    }
    pub(crate) fn string(&mut self) -> Result<String, ArtifactError> {
        let n = self.count(MAX_STRING)?;
        let x = self.take(n)?;
        let s = std::str::from_utf8(x).map_err(|_| ArtifactError::Format("utf8"))?;
        Ok(s.to_owned())
    }
    pub(crate) fn usizes(&mut self) -> Result<Vec<usize>, ArtifactError> {
        let n = self.count(MAX_COLLECTION)?;
        let mut x = Vec::with_capacity(n);
        for _ in 0..n {
            x.push(self.usize()?);
        }
        Ok(x)
    }
    pub(crate) fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish(mut w: Writer) -> Vec<u8> {
        let sum = checksum(&w.out);
        w.u32(sum).unwrap();
        w.out
    }

    fn header(count: u32, root: u32) -> Writer {
        let mut w = Writer::new();
        w.bytes(MAGIC).unwrap();
        w.u8(VERSION).unwrap();
        w.u32(count).unwrap();
        w.u32(root).unwrap();
        w
    }
    #[test]
    fn exact_scalar_round_trip_is_deterministic() {
        for (d, b) in [
            (DType::U64, u64::MAX),
            (DType::F16, 0x8001),
            (DType::F32, 0x7fc01234),
            (DType::F64, 0x8000000000000000),
        ] {
            let x = UOp::scalar_constant(d, b, UType::scalar(d));
            let a = encode(&x).unwrap();
            assert_eq!(a, encode(&x).unwrap());
            assert_eq!(x, decode(&a).unwrap());
        }
    }
    #[test]
    fn shared_sources_remain_shared() {
        let x = UOp::constant(7, UType::scalar(DType::I64));
        let root = UOp::binary(Binary::Add, x.clone(), x);
        let decoded = decode(&encode(&root).unwrap()).unwrap();
        assert!(decoded.sources()[0].shares_node_with(&decoded.sources()[1]));
    }
    #[test]
    fn matmul_payload_round_trip_and_validation_are_exact() {
        let mut graph = crate::Graph::new();
        let lhs = graph.input_dtype("lhs", [2, 1, 2, 3], DType::F64);
        let rhs = graph.input_dtype("rhs", [1, 4, 3, 2], DType::F64);
        let output = graph.matmul(lhs, rhs).unwrap();
        let root = crate::lower_graph_matmul(&graph, output).unwrap();
        let bytes = encode(&root).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(bytes, encode(&decoded).unwrap());
        assert_eq!(root, decoded);
        let mut legacy_version = bytes;
        legacy_version[4] = 2;
        let body_len = legacy_version.len() - 4;
        let sum = checksum(&legacy_version[..body_len]);
        legacy_version[body_len..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&legacy_version).is_err());

        let UArg::Matmul(plan) = root.arg() else {
            panic!("matmul payload missing");
        };
        let mut malformed = plan.as_ref().clone();
        malformed.k += 1;
        let malformed = UOp::new(
            UOpKind::Matmul,
            Some(UType::scalar(DType::F64)),
            vec![],
            UArg::Matmul(Box::new(malformed)),
        );
        assert!(malformed.validate().is_err());
        assert!(encode(&malformed).is_err());
    }
    #[test]
    fn corruption_and_truncation_fail_closed() {
        let x = encode(&UOp::constant(1, UType::scalar(DType::I64))).unwrap();
        for n in 0..x.len() {
            assert!(decode(&x[..n]).is_err());
        }
        let mut bad = x;
        bad[4] = 99;
        let end = bad.len() - 4;
        let sum = checksum(&bad[..end]);
        bad[end..].copy_from_slice(&sum.to_le_bytes());
        assert!(decode(&bad).is_err());
    }

    #[test]
    fn malformed_node_tables_fail_closed() {
        let ty = Some(UType::scalar(DType::I64));

        let mut out_of_order = header(1, 0);
        out_of_order.u32(1).unwrap();
        write_kind(&mut out_of_order, &UOpKind::Const).unwrap();
        write_type(&mut out_of_order, ty).unwrap();
        write_arg(&mut out_of_order, &UArg::Int(1)).unwrap();
        out_of_order.u32(0).unwrap();
        assert!(decode(&finish(out_of_order)).is_err());

        let mut forward = header(2, 1);
        forward.u32(0).unwrap();
        write_kind(&mut forward, &UOpKind::Cast).unwrap();
        write_type(&mut forward, ty).unwrap();
        write_arg(&mut forward, &UArg::None).unwrap();
        forward.u32(1).unwrap();
        forward.u32(0).unwrap();
        forward.u32(1).unwrap();
        write_kind(&mut forward, &UOpKind::Const).unwrap();
        write_type(&mut forward, ty).unwrap();
        write_arg(&mut forward, &UArg::Int(1)).unwrap();
        forward.u32(0).unwrap();
        assert!(decode(&finish(forward)).is_err());

        let mut wrong_arg = header(1, 0);
        wrong_arg.u32(0).unwrap();
        write_kind(&mut wrong_arg, &UOpKind::Const).unwrap();
        write_type(&mut wrong_arg, ty).unwrap();
        write_arg(&mut wrong_arg, &UArg::None).unwrap();
        wrong_arg.u32(0).unwrap();
        assert!(decode(&finish(wrong_arg)).is_err());

        let mut count_limit = header(MAX_NODES as u32 + 1, 0);
        assert!(decode(&finish(count_limit)).is_err());
        count_limit = header(1, 0);
        count_limit.u32(0).unwrap();
        count_limit.u8(u8::MAX).unwrap();
        assert!(decode(&finish(count_limit)).is_err());

        let mut trailing = encode(&UOp::constant(1, UType::scalar(DType::I64))).unwrap();
        trailing.truncate(trailing.len() - 4);
        trailing.push(0);
        let sum = checksum(&trailing);
        trailing.extend(sum.to_le_bytes());
        assert!(decode(&trailing).is_err());
    }
}
