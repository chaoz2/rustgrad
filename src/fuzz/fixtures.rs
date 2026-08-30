use super::{
    FuzzBinaryOp, FuzzCase, FuzzCompareOp, FuzzLogicalOp, FuzzReduction, FuzzScatterOp, FuzzSlice,
    FuzzTensor, FuzzUnaryOp,
};
use crate::{DType, Float8Format, Float8Storage, Scalar, Storage, TensorData};

fn tensor(shape: impl Into<Vec<usize>>, storage: Storage) -> FuzzTensor {
    FuzzTensor::from_tensor(
        &TensorData::from_storage(shape.into(), storage).expect("fixture geometry"),
    )
}

/// Checked-in semantic edge cases used as a small replay corpus. These are
/// regression programs, not fabricated mismatch artifacts.
pub fn regression_cases() -> Vec<FuzzCase> {
    vec![
        FuzzCase::Binary {
            op: FuzzBinaryOp::Add,
            lhs: tensor(
                vec![17],
                Storage::I32(
                    (0..17)
                        .map(|value| if value == 0 { i32::MAX } else { value })
                        .collect(),
                ),
            ),
            rhs: tensor(vec![], Storage::I32(vec![1])),
        },
        FuzzCase::Binary {
            op: FuzzBinaryOp::Mul,
            lhs: tensor(vec![3], Storage::F32(vec![-0.0, f32::MIN_POSITIVE, -2.0])),
            rhs: tensor(vec![3], Storage::F32(vec![1.0, 0.5, -0.0])),
        },
        FuzzCase::Binary {
            op: FuzzBinaryOp::FloorDiv,
            lhs: tensor(vec![4], Storage::I64(vec![-7, -7, 7, 7])),
            rhs: tensor(vec![4], Storage::I64(vec![3, -3, 3, -3])),
        },
        FuzzCase::Binary {
            op: FuzzBinaryOp::FMod,
            lhs: tensor(vec![4], Storage::I64(vec![-7, -7, 7, 7])),
            rhs: tensor(vec![4], Storage::I64(vec![3, -3, 3, -3])),
        },
        FuzzCase::Binary {
            op: FuzzBinaryOp::Mod,
            lhs: tensor(vec![4], Storage::F64(vec![-7.5, -7.5, 7.5, 7.5])),
            rhs: tensor(vec![4], Storage::F64(vec![3.0, -3.0, 3.0, -3.0])),
        },
        FuzzCase::Cast {
            input: tensor(
                vec![5],
                Storage::F16(vec![0x0000, 0x8000, 0x3c00, 0x7c00, 0x7e00]),
            ),
            to: DType::F32,
        },
        FuzzCase::AffineView {
            input: tensor(vec![4, 1], Storage::F32(vec![0.0, 1.0, 2.0, 3.0])),
            start: 2,
            end: 2,
            expand: 8,
        },
        FuzzCase::Reduction {
            input: tensor(vec![3, 0], Storage::F32(vec![])),
            reduction: FuzzReduction::Sum,
            axis: 1,
            keepdim: false,
        },
        FuzzCase::Reduction {
            input: tensor(vec![1, 3], Storage::F32(vec![f32::NAN, -0.0, 2.0])),
            reduction: FuzzReduction::Product,
            axis: 1,
            keepdim: true,
        },
        FuzzCase::Reduction {
            // A non-leading NaN is filtered by raw Max; finite/infinite
            // candidates retain the source first-tie ordering contract.
            input: tensor(
                vec![1, 5],
                Storage::F32(vec![f32::NEG_INFINITY, f32::NAN, -0.0, 0.0, f32::INFINITY]),
            ),
            reduction: FuzzReduction::Max,
            axis: 1,
            keepdim: false,
        },
        FuzzCase::Reduction {
            // Equal signed zeros retain the first payload for raw Min.
            input: tensor(
                vec![1, 3],
                Storage::F32(vec![f32::from_bits(0x8000_0000), 0.0, f32::INFINITY]),
            ),
            reduction: FuzzReduction::Min,
            axis: 1,
            keepdim: true,
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Abs,
            input: tensor(
                vec![4],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::NEG_INFINITY,
                    2.0,
                ]),
            ),
        },
        FuzzCase::Unary {
            // Public Bool Neg is source logical_not, not a raw GraphUnary.
            op: FuzzUnaryOp::Neg,
            input: tensor(vec![2], Storage::Bool(vec![false, true])),
        },
        FuzzCase::Unary {
            // Direct Bool Abs is the raw storage identity path.
            op: FuzzUnaryOp::Abs,
            input: tensor(vec![2], Storage::Bool(vec![false, true])),
        },
        FuzzCase::Unary {
            // Signed minima use the exact wrapping_abs storage contract.
            op: FuzzUnaryOp::Abs,
            input: tensor(vec![2], Storage::I64(vec![i64::MIN, i64::MAX])),
        },
        FuzzCase::Unary {
            // This lane must not be rounded through f64 before raw Neg.
            op: FuzzUnaryOp::Neg,
            input: tensor(vec![1], Storage::U64(vec![(1u64 << 53) + 1])),
        },
        FuzzCase::Unary {
            // Half/BF16 retain sign and infinity semantics through the
            // established decode/encode boundary; arbitrary NaN payload
            // identity is intentionally not claimed by this fixture.
            op: FuzzUnaryOp::Abs,
            input: tensor(vec![3], Storage::F16(vec![0x8000, 0x7e01, 0x7c00])),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Abs,
            input: tensor(vec![3], Storage::BF16(vec![0x8000, 0x7fc1, 0x7f80])),
        },
        FuzzCase::Unary {
            // Full-width float storage retains observable special lanes.
            op: FuzzUnaryOp::Abs,
            input: tensor(
                vec![3],
                Storage::F64(vec![
                    f64::from_bits(0x8000_0000_0000_0000),
                    f64::from_bits(0x7ff8_0000_0000_0001),
                    f64::INFINITY,
                ]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Exp2,
            input: tensor(
                vec![5],
                Storage::F32(vec![f32::NEG_INFINITY, -0.0, 1.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Log2,
            input: tensor(
                vec![5],
                Storage::F64(vec![-0.0, 0.0, 1.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Sin,
            input: tensor(
                vec![4],
                Storage::F32(vec![-0.0, 1.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Cos,
            input: tensor(
                vec![4],
                Storage::F64(vec![-0.0, 1.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Tan,
            input: tensor(
                vec![4],
                Storage::F32(vec![-0.0, 1.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Log,
            input: tensor(
                vec![5],
                Storage::F64(vec![-0.0, 0.0, 1.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Trunc,
            input: tensor(
                vec![5],
                Storage::F32(vec![-0.0, -1.75, 1.75, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            // Exact integer Square uses storage-width wrapping rather than
            // signed C overflow or a lossy floating detour.
            op: FuzzUnaryOp::Square,
            input: tensor(vec![3], Storage::I64(vec![i64::MIN, -3, i64::MAX])),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Square,
            input: tensor(vec![2], Storage::U64(vec![(1u64 << 63) + 1, u64::MAX])),
        },
        FuzzCase::Unary {
            // Round is ties-to-even and retains the sign of a zero result.
            op: FuzzUnaryOp::Round,
            input: tensor(
                vec![8],
                Storage::F64(vec![
                    -2.5,
                    -1.5,
                    -0.5,
                    0.5,
                    1.5,
                    2.5,
                    f64::INFINITY,
                    f64::NAN,
                ]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Step,
            input: tensor(vec![5], Storage::I64(vec![i64::MIN, -1, 0, 1, i64::MAX])),
        },
        FuzzCase::Unary {
            // Unsigned Step is a source predicate, not an identity for lanes
            // above one; retain exact U64 values beyond f64 precision here.
            op: FuzzUnaryOp::Step,
            input: tensor(vec![3], Storage::U64(vec![0, 1, (1u64 << 53) + 1])),
        },
        FuzzCase::Unary {
            // One raw predicate fixture carries every floating classification.
            op: FuzzUnaryOp::IsFinite,
            input: tensor(
                vec![6],
                Storage::F32(vec![
                    -0.0,
                    1.0,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::NAN,
                    f32::MIN_POSITIVE,
                ]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Sqrt,
            input: tensor(
                vec![6],
                Storage::F64(vec![-1.0, -0.0, 0.0, 4.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Reciprocal,
            input: tensor(
                vec![6],
                Storage::F32(vec![-0.0, 0.0, -2.0, 2.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Sinh,
            input: tensor(vec![4], Storage::F16(vec![0x8000, 0x3c00, 0x7c00, 0x7e00])),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Cosh,
            input: tensor(
                vec![4],
                Storage::F64(vec![-0.0, 1.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Tanh,
            input: tensor(vec![4], Storage::BF16(vec![0xff80, 0x8000, 0x7f80, 0x7fc0])),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Erf,
            input: tensor(
                vec![5],
                Storage::F64(vec![f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Erfc,
            input: tensor(
                vec![5],
                Storage::F32(vec![f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Asin,
            input: tensor(
                vec![6],
                Storage::F64(vec![-1.0, -0.0, 0.0, 1.0, 2.0, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Acos,
            input: tensor(
                vec![6],
                Storage::F32(vec![-1.0, -0.0, 0.0, 1.0, 2.0, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Atan,
            input: tensor(
                vec![5],
                Storage::F64(vec![f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Asinh,
            input: tensor(
                vec![5],
                Storage::F32(vec![f32::NEG_INFINITY, -0.0, 0.0, f32::INFINITY, f32::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Acosh,
            input: tensor(
                vec![5],
                Storage::F64(vec![0.0, 1.0, 2.0, f64::INFINITY, f64::NAN]),
            ),
        },
        FuzzCase::Unary {
            op: FuzzUnaryOp::Atanh,
            input: tensor(
                vec![6],
                Storage::F32(vec![-1.0, -0.0, 0.0, 1.0, 2.0, f32::NAN]),
            ),
        },
        FuzzCase::Compare {
            // IEEE partial comparison makes NaN unequal to itself, while
            // signed zero remains equal to positive zero.
            op: FuzzCompareOp::Eq,
            lhs: tensor(
                vec![4],
                Storage::F32(vec![
                    f32::from_bits(0x7fc0_0001),
                    f32::from_bits(0x8000_0000),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ]),
            ),
            rhs: tensor(
                vec![4],
                Storage::F32(vec![f32::NAN, 0.0, f32::INFINITY, f32::INFINITY]),
            ),
        },
        FuzzCase::Compare {
            // Ordered comparisons retain the same NaN false-lane rule and
            // treat negative and positive zero as equal.
            op: FuzzCompareOp::Le,
            lhs: tensor(
                vec![3],
                Storage::F32(vec![f32::from_bits(0x8000_0000), f32::NAN, f32::INFINITY]),
            ),
            rhs: tensor(vec![3], Storage::F32(vec![0.0, 0.0, f32::INFINITY])),
        },
        FuzzCase::Compare {
            // E4M3 terminal payloads are NaN, while raw 0x80 is negative
            // zero and must compare equal to positive zero after decoding.
            op: FuzzCompareOp::Eq,
            lhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E4M3,
                    vec![0x80, 0x7f, 0x38],
                )),
            ),
            rhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E4M3,
                    vec![0x00, 0x7f, 0x38],
                )),
            ),
        },
        FuzzCase::Compare {
            // E5M2 retains infinities and unordered NaNs at its terminal
            // exponent, independent of their unsigned storage-byte order.
            op: FuzzCompareOp::Lt,
            lhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E5M2,
                    vec![0xfc, 0x7c, 0x7f],
                )),
            ),
            rhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E5M2,
                    vec![0x00, 0x7c, 0x00],
                )),
            ),
        },
        FuzzCase::Compare {
            // FNUZ reserves 0x80 as NaN rather than negative zero.
            op: FuzzCompareOp::Ne,
            lhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E4M3FNUZ,
                    vec![0x80, 0x00, 0x40],
                )),
            ),
            rhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E4M3FNUZ,
                    vec![0x80, 0x00, 0x40],
                )),
            ),
        },
        FuzzCase::Compare {
            // E5M2FNUZ uses bias 16; numeric order must be decoded rather
            // than inferred from the sign bit and raw magnitude byte.
            op: FuzzCompareOp::Ge,
            lhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E5M2FNUZ,
                    vec![0xc0, 0x40, 0x80],
                )),
            ),
            rhs: tensor(
                vec![3],
                Storage::Float8(crate::Float8Storage::from_raw(
                    crate::Float8Format::E5M2FNUZ,
                    vec![0x40, 0xc0, 0x00],
                )),
            ),
        },
        FuzzCase::Logical {
            // Full And truth table in row-major Bool storage.
            op: FuzzLogicalOp::And,
            lhs: tensor(vec![4], Storage::Bool(vec![true, true, false, false])),
            rhs: tensor(vec![4], Storage::Bool(vec![true, false, true, false])),
        },
        FuzzCase::Logical {
            // A scalar RHS exercises the same direct Or kernel's broadcast.
            op: FuzzLogicalOp::Or,
            lhs: tensor(vec![3], Storage::Bool(vec![false, true, false])),
            rhs: tensor(vec![], Storage::Bool(vec![true])),
        },
        FuzzCase::LogicalNot {
            // Source truthiness: +0/-0 are false; NaN, infinities, and every
            // nonzero fractional lane are true before the final Ne(true).
            input: tensor(
                vec![7],
                Storage::F32(vec![
                    0.0,
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    0.5,
                    -0.5,
                ]),
            ),
        },
        FuzzCase::LogicalNot {
            input: tensor(vec![3], Storage::Bool(vec![false, true, false])),
        },
        FuzzCase::LogicalNot {
            input: tensor(vec![3], Storage::I32(vec![0, -1, 2])),
        },
        FuzzCase::TensorT {
            // Row-major [2, 3] payload becomes [0, 3, 1, 4, 2, 5].
            input: tensor(vec![2, 3], Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
        },
        FuzzCase::TensorT {
            input: tensor(vec![0, 3], Storage::F16(vec![])),
        },
        FuzzCase::Permute {
            input: tensor(vec![], Storage::Bool(vec![true])),
            axes: vec![],
        },
        FuzzCase::Permute {
            input: tensor(vec![2, 2], Storage::I64(vec![1, 2, 3, 4])),
            axes: vec![0, 1],
        },
        FuzzCase::Permute {
            // [2, 1, 3] -> [3, 2, 1] makes source-order addressing visible.
            input: tensor(
                vec![2, 1, 3],
                Storage::F32(vec![-0.0, 1.0, f32::INFINITY, 3.0, 4.0, f32::NAN]),
            ),
            axes: vec![2, 0, 1],
        },
        FuzzCase::Permute {
            input: tensor(vec![0, 2, 3], Storage::F16(vec![])),
            axes: vec![2, 0, 1],
        },
        FuzzCase::Pad {
            // [2, 2] becomes [3, 4], retaining row-major placement and the
            // raw negative-zero fill at every padded lane.
            input: tensor(vec![2, 2], Storage::F32(vec![1.0, 2.0, 3.0, 4.0])),
            padding: vec![(1, 0), (0, 2)],
            fill: tensor(vec![], Storage::F32(vec![-0.0])),
        },
        FuzzCase::Pad {
            // A zero input domain can become a nonempty all-fill movement
            // result; this remains a raw Pad contract, not pad_signed.
            input: tensor(vec![0, 2], Storage::I32(vec![])),
            padding: vec![(1, 1), (1, 0)],
            fill: tensor(vec![], Storage::I32(vec![-7])),
        },
        FuzzCase::Pad {
            // The scalar bridge commits this raw F32 NaN fill through the
            // existing Graph::pad/MovementKernelPlan storage contract.
            input: tensor(vec![1], Storage::F32(vec![1.0])),
            padding: vec![(1, 1)],
            fill: tensor(vec![], Storage::F32(vec![f32::from_bits(0x7fc0_0001)])),
        },
        FuzzCase::Pad {
            // Rank-zero Pad has empty padding but still emits an Op::Pad.
            input: tensor(vec![], Storage::Bool(vec![true])),
            padding: vec![],
            fill: tensor(vec![], Storage::Bool(vec![false])),
        },
        FuzzCase::Pad {
            // Half scalar fills commit at storage width; the input lane stays
            // a raw movement operand rather than a scalar fill conversion.
            input: tensor(vec![1], Storage::F16(vec![0x3c00])),
            padding: vec![(1, 1)],
            fill: tensor(vec![], Storage::F16(vec![0x8000])),
        },
        FuzzCase::Pad {
            // BF16 NaN fill is committed through the scalar bridge; its raw
            // input storage is deliberately not conflated with fill identity.
            input: tensor(vec![1], Storage::BF16(vec![0x3f80])),
            padding: vec![(1, 1)],
            fill: tensor(vec![], Storage::BF16(vec![0x7fc1])),
        },
        FuzzCase::Pad {
            input: tensor(vec![1], Storage::F64(vec![f64::INFINITY])),
            padding: vec![(1, 0)],
            fill: tensor(vec![], Storage::F64(vec![-0.0])),
        },
        FuzzCase::Pad {
            input: tensor(vec![1], Storage::I64(vec![i64::MIN])),
            padding: vec![(1, 0)],
            fill: tensor(vec![], Storage::I64(vec![-1])),
        },
        FuzzCase::Pad {
            input: tensor(vec![1], Storage::U64(vec![u64::MAX])),
            padding: vec![(0, 1)],
            fill: tensor(vec![], Storage::U64(vec![u64::MAX])),
        },
        FuzzCase::Gather {
            // Axis-one duplicate/reorder selection is intentionally obvious.
            input: tensor(
                vec![2, 4],
                Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            ),
            index: tensor(vec![2, 3], Storage::I32(vec![3, 1, 1, 0, 2, 2])),
            axis: 1,
        },
        FuzzCase::Gather {
            input: tensor(vec![3], Storage::I32(vec![10, 20, 30])),
            index: tensor(vec![3], Storage::I64(vec![2, 0, 1])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(vec![2, 0], Storage::F16(vec![])),
            index: tensor(vec![2, 0], Storage::I32(vec![])),
            axis: 1,
        },
        FuzzCase::Gather {
            // Raw storage lanes retain signed zero and the NaN payload through
            // select_raw, with no scalar conversion on the Gather payload.
            input: tensor(
                vec![3],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                ]),
            ),
            index: tensor(vec![3], Storage::I32(vec![1, 0, 2])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(vec![3], Storage::F16(vec![0x8000, 0x7e01, 0x7c00])),
            index: tensor(vec![3], Storage::I64(vec![1, 0, 2])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(vec![3], Storage::BF16(vec![0x8000, 0x7fc1, 0x7f80])),
            index: tensor(vec![3], Storage::I32(vec![2, 1, 0])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(
                vec![3],
                Storage::F64(vec![
                    f64::from_bits(0x8000_0000_0000_0000),
                    f64::from_bits(0x7ff8_0000_0000_0001),
                    f64::INFINITY,
                ]),
            ),
            index: tensor(vec![3], Storage::I64(vec![1, 0, 2])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(vec![3], Storage::I8(vec![i8::MIN, -1, i8::MAX])),
            index: tensor(vec![3], Storage::I32(vec![2, 0, 1])),
            axis: 0,
        },
        FuzzCase::Gather {
            input: tensor(vec![3], Storage::U64(vec![0, 1, u64::MAX])),
            index: tensor(vec![3], Storage::I64(vec![2, 0, 1])),
            axis: 0,
        },
        FuzzCase::Scatter {
            // Row-major later duplicate updates replace the earlier lane.
            base: tensor(vec![1, 4], Storage::F32(vec![10.0, 20.0, 30.0, 40.0])),
            index: tensor(vec![1, 3], Storage::I32(vec![2, 1, 2])),
            updates: tensor(vec![1, 3], Storage::F32(vec![1.0, 2.0, 3.0])),
            axis: 1,
            op: FuzzScatterOp::Replace,
        },
        FuzzCase::Scatter {
            // Raw Add follows the same row-major duplicate order.
            base: tensor(vec![1, 3], Storage::F32(vec![1.0, 10.0, 100.0])),
            index: tensor(vec![1, 3], Storage::I32(vec![1, 1, 1])),
            updates: tensor(vec![1, 3], Storage::F32(vec![0.25, 0.5, 4.0])),
            axis: 1,
            op: FuzzScatterOp::Add,
        },
        FuzzCase::Scatter {
            // I64 indices and a zero scatter axis remain a valid empty plan.
            base: tensor(vec![2, 0], Storage::I32(vec![])),
            index: tensor(vec![2, 0], Storage::I64(vec![])),
            updates: tensor(vec![2, 0], Storage::I32(vec![])),
            axis: 1,
            op: FuzzScatterOp::Replace,
        },
        FuzzCase::Scatter {
            // Replacement preserves payload bits without scalar conversion.
            base: tensor(vec![3], Storage::F32(vec![0.0, 1.0, 2.0])),
            index: tensor(vec![3], Storage::I32(vec![2, 0, 1])),
            updates: tensor(
                vec![3],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                ]),
            ),
            axis: 0,
            op: FuzzScatterOp::Replace,
        },
        FuzzCase::Concat {
            lhs: tensor(vec![2, 0], Storage::I32(vec![])),
            rhs: tensor(vec![2, 3], Storage::I32(vec![0; 6])),
            axis: 1,
        },
        FuzzCase::ConcatMany {
            // Three inputs keep their source order across a zero-width middle.
            inputs: vec![
                tensor(vec![1, 2], Storage::I64(vec![1, 2])),
                tensor(vec![1, 0], Storage::I64(vec![])),
                tensor(vec![1, 2], Storage::I64(vec![3, 4])),
            ],
            axis: 1,
        },
        FuzzCase::ConcatMany {
            // Four raw F32 lanes retain negative zero, NaN, and infinity.
            inputs: vec![
                tensor(vec![1], Storage::F32(vec![f32::from_bits(0x8000_0000)])),
                tensor(vec![0], Storage::F32(vec![])),
                tensor(vec![1], Storage::F32(vec![f32::from_bits(0x7fc0_0001)])),
                tensor(vec![1], Storage::F32(vec![f32::INFINITY])),
            ],
            axis: 0,
        },
        FuzzCase::ConcatMany {
            // Half lanes retain their raw storage payload across a middle
            // zero-width input rather than a scalar conversion.
            inputs: vec![
                tensor(vec![1, 1], Storage::F16(vec![0x8000])),
                tensor(vec![1, 0], Storage::F16(vec![])),
                tensor(vec![1, 2], Storage::F16(vec![0x7e01, 0x7c00])),
            ],
            axis: 1,
        },
        FuzzCase::ConcatMany {
            // F64 remains a raw lane copy path, including its IEEE specials.
            inputs: vec![
                tensor(
                    vec![1],
                    Storage::F64(vec![f64::from_bits(0x8000_0000_0000_0000)]),
                ),
                tensor(vec![0], Storage::F64(vec![])),
                tensor(
                    vec![1],
                    Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001)]),
                ),
                tensor(vec![1], Storage::F64(vec![f64::INFINITY])),
            ],
            axis: 0,
        },
        FuzzCase::ConcatMany {
            // Zero non-axis geometry remains a valid rank-three movement plan.
            inputs: vec![
                tensor(vec![2, 0, 1], Storage::BF16(vec![])),
                tensor(vec![2, 0, 2], Storage::BF16(vec![])),
                tensor(vec![2, 0, 0], Storage::BF16(vec![])),
            ],
            axis: 2,
        },
        FuzzCase::ConcatMany {
            inputs: vec![
                tensor(vec![1, 2], Storage::Bool(vec![true, false])),
                tensor(vec![1, 1], Storage::Bool(vec![true])),
                tensor(vec![1, 1], Storage::Bool(vec![false])),
            ],
            axis: 1,
        },
        FuzzCase::ConcatMany {
            // Raw E4M3 lanes, including negative zero and a NaN encoding, are
            // copied without entering the numeric Float8 codec.
            inputs: vec![
                tensor(
                    vec![2],
                    Storage::Float8(Float8Storage::from_raw(
                        Float8Format::E4M3,
                        vec![0x80, 0x38],
                    )),
                ),
                tensor(
                    vec![0],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E4M3, vec![])),
                ),
                tensor(
                    vec![2],
                    Storage::Float8(Float8Storage::from_raw(
                        Float8Format::E4M3,
                        vec![0x7f, 0x7e],
                    )),
                ),
            ],
            axis: 0,
        },
        FuzzCase::ConcatMany {
            inputs: vec![
                tensor(
                    vec![1, 1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E5M2, vec![0x80])),
                ),
                tensor(
                    vec![1, 0],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E5M2, vec![])),
                ),
                tensor(
                    vec![1, 2],
                    Storage::Float8(Float8Storage::from_raw(
                        Float8Format::E5M2,
                        vec![0x7d, 0x7c],
                    )),
                ),
            ],
            axis: 1,
        },
        FuzzCase::ConcatMany {
            inputs: vec![
                tensor(
                    vec![1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E4M3FNUZ, vec![0x00])),
                ),
                tensor(
                    vec![1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E4M3FNUZ, vec![0x80])),
                ),
                tensor(
                    vec![1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E4M3FNUZ, vec![0xff])),
                ),
            ],
            axis: 0,
        },
        FuzzCase::ConcatMany {
            inputs: vec![
                tensor(
                    vec![1, 1, 1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E5M2FNUZ, vec![0x00])),
                ),
                tensor(
                    vec![1, 0, 1],
                    Storage::Float8(Float8Storage::from_raw(Float8Format::E5M2FNUZ, vec![])),
                ),
                tensor(
                    vec![1, 2, 1],
                    Storage::Float8(Float8Storage::from_raw(
                        Float8Format::E5M2FNUZ,
                        vec![0x80, 0xff],
                    )),
                ),
            ],
            axis: 1,
        },
        FuzzCase::Matmul {
            // F32 must round each product and running sum: the middle one is
            // lost before the final cancellation, unlike a double accumulator.
            lhs: tensor(vec![1, 3], Storage::F32(vec![1.0e10, 1.0, -1.0e10])),
            rhs: tensor(vec![3, 1], Storage::F32(vec![1.0, 1.0, 1.0])),
        },
        FuzzCase::Matmul {
            // Raw F64 remains its native double-width contraction path.
            lhs: tensor(vec![3], Storage::F64(vec![1.0, -2.0, 0.5])),
            rhs: tensor(vec![3], Storage::F64(vec![4.0, 8.0, 16.0])),
        },
        FuzzCase::Matmul {
            // Vector-matrix form preserves its rank-one lhs geometry.
            lhs: tensor(vec![3], Storage::F32(vec![1.0, 2.0, 3.0])),
            rhs: tensor(vec![3, 2], Storage::F32(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])),
        },
        FuzzCase::Matmul {
            // Right-aligned batch broadcasting exercises generalized output
            // geometry without changing the raw Matmul operation.
            lhs: tensor(vec![2, 1, 1, 2], Storage::F64(vec![1.0, 2.0, 3.0, 4.0])),
            rhs: tensor(
                vec![3, 2, 1],
                Storage::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
        },
        FuzzCase::Matmul {
            lhs: tensor(vec![3, 0], Storage::F32(vec![])),
            rhs: tensor(vec![0, 5], Storage::F32(vec![])),
        },
        FuzzCase::Stride {
            // Rank-zero slicing remains an explicit affine identity view.
            input: tensor(vec![], Storage::Bool(vec![true])),
            slices: vec![],
        },
        FuzzCase::Stride {
            // A full slice preserves wide integer lanes without conversion.
            input: tensor(
                vec![3],
                Storage::I64(vec![i64::MIN, -(1_i64 << 53) - 1, i64::MAX]),
            ),
            slices: vec![FuzzSlice {
                start: None,
                stop: None,
                step: 1,
            }],
        },
        FuzzCase::Stride {
            // Reverse stepping retains exact raw IEEE lane ordering.
            input: tensor(
                vec![4],
                Storage::F32(vec![
                    f32::from_bits(0x8000_0000),
                    f32::from_bits(0x7fc0_0001),
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ]),
            ),
            slices: vec![FuzzSlice {
                start: None,
                stop: None,
                step: -1,
            }],
        },
        FuzzCase::Stride {
            // Empty geometry and negative strides remain well-defined.
            input: tensor(vec![0, 3], Storage::BF16(vec![])),
            slices: vec![
                FuzzSlice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                FuzzSlice {
                    start: None,
                    stop: None,
                    step: -2,
                },
            ],
        },
        FuzzCase::Stride {
            // Storage-only views preserve Float8 payload bytes exactly.
            input: FuzzTensor {
                shape: vec![4],
                dtype: DType::F8E4M3,
                bytes: vec![0x80, 0x7f, 0x01, 0xff],
            },
            slices: vec![FuzzSlice {
                start: None,
                stop: None,
                step: -1,
            }],
        },
        FuzzCase::Select {
            condition: tensor(vec![3], Storage::Bool(vec![true, false, true])),
            on_true: tensor(vec![3], Storage::BF16(vec![0x3f80, 0x8000, 0x7fc1])),
            on_false: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
                Scalar::F(2.0),
                DType::BF16,
            )),
        },
        FuzzCase::Select {
            condition: tensor(vec![4], Storage::Bool(vec![true, false, false, true])),
            on_true: FuzzTensor {
                shape: vec![4],
                dtype: DType::F8E4M3,
                bytes: vec![0x80, 0x7f, 0x01, 0xff],
            },
            on_false: FuzzTensor {
                shape: vec![],
                dtype: DType::F8E4M3,
                bytes: vec![0xa5],
            },
        },
        FuzzCase::Select {
            condition: tensor(vec![4], Storage::Bool(vec![true, false, false, true])),
            on_true: FuzzTensor {
                shape: vec![4],
                dtype: DType::F8E5M2,
                bytes: vec![0x80, 0x7f, 0x01, 0xff],
            },
            on_false: FuzzTensor {
                shape: vec![],
                dtype: DType::F8E5M2,
                bytes: vec![0xa5],
            },
        },
        FuzzCase::Select {
            condition: tensor(vec![4], Storage::Bool(vec![true, false, false, true])),
            on_true: FuzzTensor {
                shape: vec![4],
                dtype: DType::F8E4M3FNUZ,
                bytes: vec![0x80, 0x7f, 0x01, 0xff],
            },
            on_false: FuzzTensor {
                shape: vec![],
                dtype: DType::F8E4M3FNUZ,
                bytes: vec![0xa5],
            },
        },
        FuzzCase::Select {
            condition: tensor(vec![4], Storage::Bool(vec![true, false, false, true])),
            on_true: FuzzTensor {
                shape: vec![4],
                dtype: DType::F8E5M2FNUZ,
                bytes: vec![0x80, 0x7f, 0x01, 0xff],
            },
            on_false: FuzzTensor {
                shape: vec![],
                dtype: DType::F8E5M2FNUZ,
                bytes: vec![0xa5],
            },
        },
    ]
}
