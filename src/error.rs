use crate::{DType, NodeId, Shape};
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    InvalidData {
        shape: Shape,
        expected: usize,
        actual: usize,
    },
    ShapeOverflow(Shape),
    InvalidIndex,
    UnknownNode(NodeId),
    MissingInput(String),
    InputShape {
        name: String,
        expected: Shape,
        actual: Shape,
    },
    InputDType {
        name: String,
        expected: DType,
        actual: DType,
    },
    InvalidLogicalDType {
        op: &'static str,
        actual: DType,
    },
    InvalidElementwiseDType {
        op: &'static str,
        actual: DType,
    },
    DivisionByZero {
        op: &'static str,
    },
    InvalidShiftCount {
        count: i64,
        bits: u8,
    },
    ShapeMismatch {
        op: &'static str,
        lhs: Shape,
        rhs: Shape,
    },
    BroadcastMismatch {
        lhs: Shape,
        rhs: Shape,
    },
    InvalidAxis {
        node: NodeId,
        axis: usize,
        rank: usize,
    },
    InvalidReductionAxes {
        node: NodeId,
        axes: Vec<usize>,
        rank: usize,
    },
    EmptyReduction {
        op: &'static str,
        shape: Shape,
        axes: Vec<usize>,
    },
    InvalidArange {
        start: i64,
        end: i64,
        step: i64,
    },
    InvalidLinspace {
        steps: isize,
    },
    InvalidRandom {
        reason: &'static str,
    },
    InvalidReshape {
        from: Shape,
        to: Shape,
    },
    InvalidPermutation {
        shape: Shape,
        axes: Vec<usize>,
    },
    InvalidMatmul {
        lhs: Shape,
        rhs: Shape,
    },
    InvalidEinsum {
        equation: String,
        reason: &'static str,
    },
    EinsumOperandCount {
        expected: usize,
        actual: usize,
    },
    EinsumGradientPending,
    InvalidAttention {
        reason: &'static str,
    },
    UnsupportedDropout {
        probability_bits: u64,
    },
    InvalidConv2d {
        input: Shape,
        weight: Shape,
        reason: &'static str,
    },
    InvalidExpand {
        from: Shape,
        to: Shape,
    },
    InvalidSumTo {
        from: Shape,
        to: Shape,
    },
    InvalidMovementRank {
        op: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidBounds {
        axis: usize,
        start: usize,
        end: usize,
        dim: usize,
    },
    InvalidSliceStep {
        axis: usize,
    },
    InvalidConcat {
        axis: usize,
        shapes: Vec<Shape>,
    },
    InvalidIndexDType {
        op: &'static str,
        actual: DType,
    },
    InvalidIndexedShape {
        op: &'static str,
        input: Shape,
        index: Shape,
    },
    InvalidUpdateShape {
        index: Shape,
        updates: Shape,
    },
    IndexOutOfBounds {
        axis: usize,
        index: i64,
        dim: usize,
    },
    NonDifferentiableIndexing(&'static str),
    NonScalarLoss(Shape),
    NoGradient(NodeId),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidData {
                shape,
                expected,
                actual,
            } => {
                write!(f, "shape {shape} needs {expected} values, got {actual}")
            }
            Self::ShapeOverflow(shape) => write!(f, "shape {shape} overflows usize"),
            Self::InvalidIndex => write!(f, "invalid dense index or coordinate"),
            Self::UnknownNode(node) => write!(f, "unknown node %{node}"),
            Self::MissingInput(name) => write!(f, "missing input {name:?}"),
            Self::InputShape {
                name,
                expected,
                actual,
            } => {
                write!(f, "input {name:?} expected {expected}, got {actual}")
            }
            Self::InputDType {
                name,
                expected,
                actual,
            } => write!(f, "input {name:?} expected {expected:?}, got {actual:?}"),
            Self::InvalidLogicalDType { op, actual } => {
                write!(f, "{op} requires bool tensors, got {actual:?}")
            }
            Self::InvalidElementwiseDType { op, actual } => {
                write!(f, "{op} does not accept {actual:?} tensors")
            }
            Self::DivisionByZero { op } => write!(f, "{op} by zero"),
            Self::InvalidShiftCount { count, bits } => {
                write!(f, "shift count {count} is invalid for {bits}-bit values")
            }
            Self::ShapeMismatch { op, lhs, rhs } => {
                write!(f, "{op} requires equal shapes, got {lhs} and {rhs}")
            }
            Self::BroadcastMismatch { lhs, rhs } => {
                write!(f, "shapes {lhs} and {rhs} cannot be broadcast together")
            }
            Self::InvalidAxis { node, axis, rank } => {
                write!(f, "axis {axis} is invalid for rank-{rank} node %{node}")
            }
            Self::InvalidReductionAxes { node, axes, rank } => {
                write!(f, "axes {axes:?} are invalid for rank-{rank} node %{node}")
            }
            Self::EmptyReduction { op, shape, axes } => {
                write!(
                    f,
                    "{op} has no values to reduce in {shape} along axes {axes:?}"
                )
            }
            Self::InvalidArange { start, end, step } => {
                write!(f, "invalid arange({start}, {end}, {step})")
            }
            Self::InvalidLinspace { steps } => {
                write!(f, "linspace steps must be non-negative, got {steps}")
            }
            Self::InvalidRandom { reason } => write!(f, "invalid random operation: {reason}"),
            Self::InvalidReshape { from, to } => {
                write!(f, "cannot reshape {from} to {to}")
            }
            Self::InvalidPermutation { shape, axes } => {
                write!(f, "axes {axes:?} are not a permutation of shape {shape}")
            }
            Self::InvalidMatmul { lhs, rhs } => {
                write!(f, "matmul requires [M,K] @ [K,N], got {lhs} and {rhs}")
            }
            Self::InvalidEinsum { equation, reason } => {
                write!(f, "invalid einsum {equation:?}: {reason}")
            }
            Self::EinsumOperandCount { expected, actual } => {
                write!(f, "einsum expects {expected} operands, got {actual}")
            }
            Self::EinsumGradientPending => write!(
                f,
                "einsum gradients require the pending EinsumGrad operation"
            ),
            Self::InvalidAttention { reason } => {
                write!(f, "invalid scaled dot-product attention: {reason}")
            }
            Self::UnsupportedDropout { probability_bits } => write!(
                f,
                "scaled dot-product attention dropout_p={} requires RustGrad's random subsystem",
                f64::from_bits(*probability_bits)
            ),
            Self::InvalidConv2d {
                input,
                weight,
                reason,
            } => write!(
                f,
                "invalid conv2d ({reason}): input {input}, weight {weight}"
            ),
            Self::InvalidExpand { from, to } => write!(f, "cannot expand {from} to {to}"),
            Self::InvalidSumTo { from, to } => write!(f, "cannot reduce {from} to {to}"),
            Self::InvalidMovementRank {
                op,
                expected,
                actual,
            } => {
                write!(f, "{op} needs {expected} axis specifications, got {actual}")
            }
            Self::InvalidBounds {
                axis,
                start,
                end,
                dim,
            } => {
                write!(
                    f,
                    "axis {axis} bounds [{start}, {end}) are invalid for length {dim}"
                )
            }
            Self::InvalidSliceStep { axis } => {
                write!(f, "slice step on axis {axis} must not be zero")
            }
            Self::InvalidConcat { axis, shapes } => {
                write!(f, "cannot concatenate shapes {shapes:?} along axis {axis}")
            }
            Self::InvalidIndexDType { op, actual } => {
                write!(f, "{op} requires an integer index tensor, got {actual:?}")
            }
            Self::InvalidIndexedShape { op, input, index } => write!(
                f,
                "{op} cannot use index shape {index} with input shape {input}"
            ),
            Self::InvalidUpdateShape { index, updates } => {
                write!(f, "updates shape {updates} must cover index shape {index}")
            }
            Self::IndexOutOfBounds { axis, index, dim } => write!(
                f,
                "index {index} is out of bounds for axis {axis} with length {dim}"
            ),
            Self::NonDifferentiableIndexing(op) => write!(
                f,
                "{op} is deliberately nondifferentiable in the current graph"
            ),
            Self::NonScalarLoss(shape) => {
                write!(f, "backward requires a one-element loss, got {shape}")
            }
            Self::NoGradient(node) => write!(f, "node %{node} does not affect the loss"),
        }
    }
}

impl std::error::Error for Error {}
