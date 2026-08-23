use crate::{NodeId, Shape};
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
    UnknownNode(NodeId),
    MissingInput(String),
    InputShape {
        name: String,
        expected: Shape,
        actual: Shape,
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
    InvalidArange {
        start: i64,
        end: i64,
        step: i64,
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
    InvalidExpand {
        from: Shape,
        to: Shape,
    },
    InvalidSumTo {
        from: Shape,
        to: Shape,
    },
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
            Self::UnknownNode(node) => write!(f, "unknown node %{node}"),
            Self::MissingInput(name) => write!(f, "missing input {name:?}"),
            Self::InputShape {
                name,
                expected,
                actual,
            } => {
                write!(f, "input {name:?} expected {expected}, got {actual}")
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
            Self::InvalidArange { start, end, step } => {
                write!(f, "invalid arange({start}, {end}, {step})")
            }
            Self::InvalidReshape { from, to } => {
                write!(f, "cannot reshape {from} to {to}")
            }
            Self::InvalidPermutation { shape, axes } => {
                write!(f, "axes {axes:?} are not a permutation of shape {shape}")
            }
            Self::InvalidMatmul { lhs, rhs } => {
                write!(f, "matmul requires [M,K] @ [K,N], got {lhs} and {rhs}")
            }
            Self::InvalidExpand { from, to } => write!(f, "cannot expand {from} to {to}"),
            Self::InvalidSumTo { from, to } => write!(f, "cannot reduce {from} to {to}"),
            Self::NonScalarLoss(shape) => {
                write!(f, "backward requires a one-element loss, got {shape}")
            }
            Self::NoGradient(node) => write!(f, "node %{node} does not affect the loss"),
        }
    }
}

impl std::error::Error for Error {}
