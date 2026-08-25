//! CPU-only contraction policy for narrow floating-point storage.
//!
//! tinygrad implements `dot` as multiply followed by `sum(-1)` and a final
//! cast to the least-upper operand dtype.  Float8 therefore widens each lane
//! to F32 for the reduction and encodes exactly once at the result boundary.

use crate::DType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Float8ContractionPolicy {
    F32AccumulateThenNarrow,
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
