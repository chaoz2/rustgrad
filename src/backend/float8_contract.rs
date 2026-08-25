//! CPU-only contraction policy for narrow floating-point storage.
//!
//! tinygrad implements `dot` as multiply followed by `sum(-1)` and a final
//! cast to the least-upper operand dtype.  Float8 therefore widens each lane
//! to F32 for the reduction and encodes exactly once at the result boundary.

use crate::{DType, Scalar};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Float8ContractionPolicy {
    F32AccumulateThenNarrow,
}

impl Float8ContractionPolicy {
    /// Accumulates one decoded product at the source's F32 reduction boundary.
    pub(crate) fn accumulate(self, accumulator: Scalar, lhs: Scalar, rhs: Scalar) -> Scalar {
        match self {
            Self::F32AccumulateThenNarrow => Scalar::F(f64::from(
                accumulator.as_f64() as f32 + lhs.as_f64() as f32 * rhs.as_f64() as f32,
            )),
        }
    }
}

pub(crate) const fn matmul_policy(result: DType) -> Option<Float8ContractionPolicy> {
    if result.is_float8() {
        Some(Float8ContractionPolicy::F32AccumulateThenNarrow)
    } else {
        None
    }
}

/// Conv2d lowers to multiply followed by a spatial/channel sum in tinygrad.
pub(crate) const fn conv2d_policy(result: DType) -> Option<Float8ContractionPolicy> {
    matmul_policy(result)
}

/// tinygrad einsum is aligned elementwise multiplication followed by `sum`.
/// Its float8 result path therefore has the same F32 reduction boundary as dot.
pub(crate) const fn einsum_policy(result: DType) -> Option<Float8ContractionPolicy> {
    matmul_policy(result)
}
