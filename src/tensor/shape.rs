use crate::{Error, Result};
use std::fmt;

#[derive(
    Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn numel(&self) -> Result<usize> {
        self.0.iter().try_fold(1usize, |n, dim| {
            n.checked_mul(*dim)
                .ok_or_else(|| Error::ShapeOverflow(self.clone()))
        })
    }

    pub fn without_axis(&self, axis: usize) -> Option<Self> {
        if axis >= self.rank() {
            None
        } else {
            let mut dims = self.0.clone();
            dims.remove(axis);
            Some(Self(dims))
        }
    }

    pub fn broadcast_with(&self, other: &Self) -> Result<Self> {
        let rank = self.rank().max(other.rank());
        let mut output = Vec::with_capacity(rank);
        for offset in (0..rank).rev() {
            let lhs = self
                .0
                .get(self.rank().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            let rhs = other
                .0
                .get(other.rank().wrapping_sub(1 + offset))
                .copied()
                .unwrap_or(1);
            if lhs != rhs && lhs != 1 && rhs != 1 {
                return Err(Error::BroadcastMismatch {
                    lhs: self.clone(),
                    rhs: other.clone(),
                });
            }
            output.push(if lhs == 0 || rhs == 0 {
                0
            } else {
                lhs.max(rhs)
            });
        }
        Ok(Self(output))
    }

    pub(crate) fn contiguous_strides(&self) -> Vec<usize> {
        let mut stride = 1;
        let mut strides = vec![0; self.rank()];
        for (index, dim) in self.0.iter().enumerate().rev() {
            strides[index] = stride;
            stride *= dim;
        }
        strides
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(value.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
