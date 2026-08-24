//! Typed graph nodes whose concrete output extent is known only at realization.

use super::NodeId;
use crate::{DType, Error, Result, Shape};

/// Identifier in a graph's dynamic-result arena. It cannot be used where a
/// static [`NodeId`] is required.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DynamicNodeId(pub(crate) usize);

/// The static part of a data-dependent output shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynamicOutputShape {
    rank: usize,
}

impl DynamicOutputShape {
    pub const fn new(rank: usize) -> Self {
        Self { rank }
    }
    pub const fn rank(self) -> usize {
        self.rank
    }
    pub fn validate(self, shape: &Shape) -> Result<()> {
        if shape.rank() == self.rank {
            Ok(())
        } else {
            Err(Error::InvalidIndex)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DynamicOp {
    Nonzero { input: NodeId },
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicNode {
    pub op: DynamicOp,
    pub output: DynamicOutputShape,
    pub dtype: DType,
}

impl DynamicNode {
    pub(crate) fn nonzero(input: NodeId) -> Self {
        Self {
            op: DynamicOp::Nonzero { input },
            output: DynamicOutputShape::new(2),
            dtype: DType::I64,
        }
    }
}
