use crate::{
    Backend, BinaryOp, CompareOp, CpuBackend, DType, Graph, NodeId, ReduceKind, Scalar, Shape,
    Storage, TensorData, UnaryOp,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub(super) const MAX_RANK: usize = 4;
pub(super) const MAX_ELEMENTS: usize = 4096;
pub(super) const MAX_TENSOR_BYTES: usize = MAX_ELEMENTS * 8;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzBinaryOp {
    Add,
    Sub,
    Mul,
    Maximum,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzUnaryOp {
    Neg,
    Abs,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzCompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzLogicalOp {
    And,
    Or,
}

/// Closed raw movement scatter modes with a complete portable fuzz path.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzScatterOp {
    Replace,
    Add,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FuzzReduction {
    Sum,
    Mean,
    Product,
    Max,
    Min,
}

/// Portable little-endian tensor bytes used by generated cases and failures.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FuzzTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub bytes: Vec<u8>,
}

impl FuzzTensor {
    pub fn from_tensor(value: &TensorData) -> Self {
        let mut bytes = Vec::with_capacity(value.len() * value.dtype().itemsize());
        macro_rules! extend {
            ($values:expr) => {
                for value in $values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            };
        }
        match value.storage() {
            Storage::Bool(values) => bytes.extend(values.iter().map(|value| u8::from(*value))),
            Storage::I8(values) => bytes.extend(values.iter().map(|value| *value as u8)),
            Storage::U8(values) => bytes.extend_from_slice(values),
            Storage::I16(values) => extend!(values),
            Storage::U16(values) => extend!(values),
            Storage::I32(values) => extend!(values),
            Storage::U32(values) => extend!(values),
            Storage::I64(values) => extend!(values),
            Storage::U64(values) => extend!(values),
            Storage::F16(values) => extend!(values),
            Storage::BF16(values) => extend!(values),
            Storage::F32(values) => extend!(values),
            Storage::F64(values) => extend!(values),
        }
        Self {
            shape: value.shape().dims().to_vec(),
            dtype: value.dtype(),
            bytes,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.shape.len() > MAX_RANK {
            return Err("tensor rank exceeds fuzz bound".into());
        }
        let elements = Shape::new(self.shape.clone())
            .numel()
            .map_err(|error| error.to_string())?;
        if elements > MAX_ELEMENTS {
            return Err("tensor element count exceeds fuzz bound".into());
        }
        let expected = elements
            .checked_mul(self.dtype.itemsize())
            .ok_or("tensor byte overflow")?;
        if self.bytes.len() != expected || self.bytes.len() > MAX_TENSOR_BYTES {
            return Err("tensor byte length mismatch".into());
        }
        if self.dtype == DType::Bool && self.bytes.iter().any(|byte| *byte > 1) {
            return Err("boolean tensor contains non-canonical byte".into());
        }
        Ok(())
    }

    pub fn to_tensor(&self) -> Result<TensorData, String> {
        self.validate()?;
        macro_rules! parse {
            ($ty:ty, $variant:ident) => {{
                let values = self
                    .bytes
                    .chunks_exact(core::mem::size_of::<$ty>())
                    .map(|chunk| {
                        <$ty>::from_le_bytes(chunk.try_into().expect("validated chunk width"))
                    })
                    .collect();
                Storage::$variant(values)
            }};
        }
        let storage = match self.dtype {
            DType::Bool => Storage::Bool(self.bytes.iter().map(|byte| *byte == 1).collect()),
            DType::I8 => Storage::I8(self.bytes.iter().map(|byte| *byte as i8).collect()),
            DType::U8 => Storage::U8(self.bytes.clone()),
            DType::I16 => parse!(i16, I16),
            DType::U16 => parse!(u16, U16),
            DType::I32 => parse!(i32, I32),
            DType::U32 => parse!(u32, U32),
            DType::I64 => parse!(i64, I64),
            DType::U64 => parse!(u64, U64),
            DType::F16 => parse!(u16, F16),
            DType::BF16 => parse!(u16, BF16),
            DType::F32 => parse!(f32, F32),
            DType::F64 => parse!(f64, F64),
        };
        TensorData::from_storage(self.shape.clone(), storage).map_err(|error| error.to_string())
    }

    pub(super) fn zeroed(&self) -> Self {
        let mut value = self.clone();
        value.bytes.fill(0);
        value
    }

    pub(super) fn scalar_prefix(&self) -> Option<Self> {
        if self.bytes.is_empty() {
            return None;
        }
        Some(Self {
            shape: vec![],
            dtype: self.dtype,
            bytes: self.bytes[..self.dtype.itemsize()].to_vec(),
        })
    }
}

/// One valid, fully typed static semantic program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FuzzCase {
    Binary {
        op: FuzzBinaryOp,
        lhs: FuzzTensor,
        rhs: FuzzTensor,
    },
    Select {
        condition: FuzzTensor,
        on_true: FuzzTensor,
        on_false: FuzzTensor,
    },
    Cast {
        input: FuzzTensor,
        to: DType,
    },
    AffineView {
        input: FuzzTensor,
        start: usize,
        end: usize,
        expand: usize,
    },
    Reduction {
        input: FuzzTensor,
        reduction: FuzzReduction,
        axis: usize,
        keepdim: bool,
    },
    Concat {
        lhs: FuzzTensor,
        rhs: FuzzTensor,
        axis: usize,
    },
    Matmul {
        lhs: FuzzTensor,
        rhs: FuzzTensor,
    },
    Unary {
        op: FuzzUnaryOp,
        input: FuzzTensor,
    },
    Compare {
        op: FuzzCompareOp,
        lhs: FuzzTensor,
        rhs: FuzzTensor,
    },
    Logical {
        op: FuzzLogicalOp,
        lhs: FuzzTensor,
        rhs: FuzzTensor,
    },
    LogicalNot {
        input: FuzzTensor,
    },
    TensorT {
        input: FuzzTensor,
    },
    Pad {
        input: FuzzTensor,
        padding: Vec<(usize, usize)>,
        fill: FuzzTensor,
    },
    Gather {
        input: FuzzTensor,
        index: FuzzTensor,
        axis: usize,
    },
    Scatter {
        base: FuzzTensor,
        index: FuzzTensor,
        updates: FuzzTensor,
        axis: usize,
        op: FuzzScatterOp,
    },
}

pub(super) struct BuiltCase {
    pub graph: Graph,
    pub output: NodeId,
    pub ordered: BTreeMap<String, TensorData>,
    pub oracle: HashMap<String, TensorData>,
}

impl FuzzCase {
    pub fn validate(&self) -> Result<(), String> {
        self.build().map(|_| ())
    }

    pub(super) fn tensors(&self) -> Vec<&FuzzTensor> {
        match self {
            Self::Binary { lhs, rhs, .. }
            | Self::Concat { lhs, rhs, .. }
            | Self::Matmul { lhs, rhs }
            | Self::Compare { lhs, rhs, .. }
            | Self::Logical { lhs, rhs, .. }
            | Self::Gather {
                input: lhs,
                index: rhs,
                ..
            } => vec![lhs, rhs],
            Self::Scatter {
                base,
                index,
                updates,
                ..
            } => vec![base, index, updates],
            Self::Select {
                condition,
                on_true,
                on_false,
            } => vec![condition, on_true, on_false],
            Self::Cast { input, .. }
            | Self::AffineView { input, .. }
            | Self::Reduction { input, .. }
            | Self::Unary { input, .. }
            | Self::LogicalNot { input }
            | Self::TensorT { input } => vec![input],
            // Raw Graph::pad stores its fill in the movement plan, not as a
            // caller-owned graph buffer. `build` validates it explicitly.
            Self::Pad { input, .. } => vec![input],
        }
    }

    pub(super) fn build(&self) -> Result<BuiltCase, String> {
        for tensor in self.tensors() {
            tensor.validate()?;
        }
        let mut graph = Graph::new();
        let mut ordered = BTreeMap::new();
        let mut bind =
            |graph: &mut Graph, name: &str, tensor: &FuzzTensor| -> Result<NodeId, String> {
                let value = tensor.to_tensor()?;
                ordered.insert(name.to_string(), value);
                Ok(graph.input_dtype(name, tensor.shape.clone(), tensor.dtype))
            };
        let output = match self {
            Self::Binary { op, lhs, rhs } => {
                let lhs_id = bind(&mut graph, "lhs", lhs)?;
                let rhs_id = bind(&mut graph, "rhs", rhs)?;
                graph
                    .binary(
                        match op {
                            FuzzBinaryOp::Add => BinaryOp::Add,
                            FuzzBinaryOp::Sub => BinaryOp::Sub,
                            FuzzBinaryOp::Mul => BinaryOp::Mul,
                            FuzzBinaryOp::Maximum => BinaryOp::Maximum,
                        },
                        lhs_id,
                        rhs_id,
                    )
                    .map_err(|error| error.to_string())?
            }
            Self::Select {
                condition,
                on_true,
                on_false,
            } => {
                let condition = bind(&mut graph, "condition", condition)?;
                let on_true = bind(&mut graph, "on_true", on_true)?;
                let on_false = bind(&mut graph, "on_false", on_false)?;
                graph
                    .select(condition, on_true, on_false)
                    .map_err(|error| error.to_string())?
            }
            Self::Cast { input, to } => {
                let input = bind(&mut graph, "input", input)?;
                graph.cast(input, *to).map_err(|error| error.to_string())?
            }
            Self::AffineView {
                input,
                start,
                end,
                expand,
            } => {
                if input.shape.as_slice().get(1) != Some(&1)
                    || *start > *end
                    || *end > input.shape[0]
                {
                    return Err("invalid affine view geometry".into());
                }
                let input_id = bind(&mut graph, "input", input)?;
                let view = graph
                    .shrink(input_id, [(*start, *end), (0, 1)])
                    .and_then(|id| graph.expand(id, [end - start, *expand]))
                    .map_err(|error| error.to_string())?;
                let zero = graph.constant(TensorData::scalar_with_dtype(Scalar::I(0), input.dtype));
                graph.add(view, zero).map_err(|error| error.to_string())?
            }
            Self::Reduction {
                input,
                reduction,
                axis,
                keepdim,
            } => {
                let input_id = bind(&mut graph, "input", input)?;
                graph
                    .reduce(
                        input_id,
                        match reduction {
                            FuzzReduction::Sum => ReduceKind::Sum,
                            FuzzReduction::Mean => ReduceKind::Mean,
                            FuzzReduction::Product => ReduceKind::Product,
                            FuzzReduction::Max => ReduceKind::Max,
                            FuzzReduction::Min => ReduceKind::Min,
                        },
                        Some(vec![*axis as isize]),
                        *keepdim,
                    )
                    .map_err(|error| error.to_string())?
            }
            Self::Concat { lhs, rhs, axis } => {
                let lhs = bind(&mut graph, "lhs", lhs)?;
                let rhs = bind(&mut graph, "rhs", rhs)?;
                graph
                    .concat([lhs, rhs], *axis)
                    .map_err(|error| error.to_string())?
            }
            Self::Matmul { lhs, rhs } => {
                let lhs = bind(&mut graph, "lhs", lhs)?;
                let rhs = bind(&mut graph, "rhs", rhs)?;
                graph.matmul(lhs, rhs).map_err(|error| error.to_string())?
            }
            Self::Unary { op, input } => {
                let input = bind(&mut graph, "input", input)?;
                match op {
                    FuzzUnaryOp::Neg => graph.neg(input),
                    // `Graph::abs` is the source-level sign/mul composition.
                    // This portable fuzz family instead exercises the existing
                    // direct GraphUnary Abs path shared by captured/native
                    // replay, just as Neg does for its numeric inputs.
                    FuzzUnaryOp::Abs => graph.unary(UnaryOp::Abs, input),
                }
                .map_err(|error| error.to_string())?
            }
            Self::Compare { op, lhs, rhs } => {
                let lhs = bind(&mut graph, "lhs", lhs)?;
                let rhs = bind(&mut graph, "rhs", rhs)?;
                graph
                    .compare(
                        match op {
                            FuzzCompareOp::Eq => CompareOp::Eq,
                            FuzzCompareOp::Ne => CompareOp::Ne,
                            FuzzCompareOp::Lt => CompareOp::Lt,
                            FuzzCompareOp::Le => CompareOp::Le,
                            FuzzCompareOp::Gt => CompareOp::Gt,
                            FuzzCompareOp::Ge => CompareOp::Ge,
                        },
                        lhs,
                        rhs,
                    )
                    .map_err(|error| error.to_string())?
            }
            Self::Logical { op, lhs, rhs } => {
                let lhs = bind(&mut graph, "lhs", lhs)?;
                let rhs = bind(&mut graph, "rhs", rhs)?;
                match op {
                    FuzzLogicalOp::And => graph.logical_and(lhs, rhs),
                    FuzzLogicalOp::Or => graph.logical_or(lhs, rhs),
                }
                .map_err(|error| error.to_string())?
            }
            Self::LogicalNot { input } => {
                let input = bind(&mut graph, "input", input)?;
                graph.logical_not(input).map_err(|error| error.to_string())?
            }
            Self::TensorT { input } => {
                let input = bind(&mut graph, "input", input)?;
                graph.t_tinygrad(input).map_err(|error| error.to_string())?
            }
            Self::Pad {
                input,
                padding,
                fill,
            } => {
                fill.validate()?;
                if !fill.shape.is_empty() || fill.dtype != input.dtype {
                    return Err("pad fill must be a rank-zero tensor with the input dtype".into());
                }
                let fill = fill.to_tensor()?.scalar_at(0);
                let input = bind(&mut graph, "input", input)?;
                graph
                    .pad(input, padding.clone(), fill)
                    .map_err(|error| error.to_string())?
            }
            Self::Gather { input, index, axis } => {
                if !matches!(index.dtype, DType::I32 | DType::I64) {
                    return Err("raw fuzz gather index dtype must be I32 or I64".into());
                }
                if input.shape.is_empty()
                    || index.shape.len() != input.shape.len()
                    || *axis >= input.shape.len()
                    || input
                        .shape
                        .iter()
                        .zip(&index.shape)
                        .enumerate()
                        .any(|(dimension, (input, index))| dimension != *axis && index > input)
                {
                    return Err("invalid raw fuzz gather geometry".into());
                }
                let extent = input.shape[*axis];
                let index_value = index.to_tensor()?;
                if (0..index_value.len()).any(|position| match index_value.scalar_at(position) {
                    Scalar::I(value) => usize::try_from(value).map_or(true, |value| value >= extent),
                    _ => true,
                }) {
                    return Err("raw fuzz gather index is negative or out of range".into());
                }
                let input = bind(&mut graph, "input", input)?;
                let index = bind(&mut graph, "index", index)?;
                graph
                    .gather(input, index, *axis)
                    .map_err(|error| error.to_string())?
            }
            Self::Scatter {
                base,
                index,
                updates,
                axis,
                op,
            } => {
                if !matches!(index.dtype, DType::I32 | DType::I64) {
                    return Err("raw fuzz scatter index dtype must be I32 or I64".into());
                }
                if !matches!(base.dtype, DType::F32 | DType::I32 | DType::F16 | DType::Bool)
                    || updates.dtype != base.dtype
                {
                    return Err("raw fuzz scatter requires homogeneous portable data dtypes".into());
                }
                if *op == FuzzScatterOp::Add && base.dtype != DType::F32 {
                    return Err("raw fuzz scatter_add is portable for F32 only".into());
                }
                if base.shape.is_empty()
                    || index.shape.len() != base.shape.len()
                    || updates.shape.len() != index.shape.len()
                    || *axis >= base.shape.len()
                    || base
                        .shape
                        .iter()
                        .zip(&index.shape)
                        .enumerate()
                        .any(|(dimension, (base, index))| dimension != *axis && index > base)
                    || updates
                        .shape
                        .iter()
                        .zip(&index.shape)
                        .any(|(update, index)| update < index)
                {
                    return Err("invalid raw fuzz scatter geometry".into());
                }
                let extent = base.shape[*axis];
                let index_value = index.to_tensor()?;
                if (0..index_value.len()).any(|position| match index_value.scalar_at(position) {
                    Scalar::I(value) => usize::try_from(value).map_or(true, |value| value >= extent),
                    _ => true,
                }) {
                    return Err("raw fuzz scatter index is negative or out of range".into());
                }
                let base = bind(&mut graph, "base", base)?;
                let index = bind(&mut graph, "index", index)?;
                let updates = bind(&mut graph, "updates", updates)?;
                match op {
                    FuzzScatterOp::Replace => graph.scatter(base, index, updates, *axis),
                    FuzzScatterOp::Add => graph.scatter_add(base, index, updates, *axis),
                }
                .map_err(|error| error.to_string())?
            }
        };
        let oracle = ordered.clone().into_iter().collect();
        CpuBackend
            .execute(&graph, output, &oracle)
            .map_err(|error| format!("valid fuzz case rejected by CPU oracle: {error}"))?;
        Ok(BuiltCase {
            graph,
            output,
            ordered,
            oracle,
        })
    }
}
