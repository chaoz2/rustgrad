use super::{
    FuzzBinaryOp, FuzzCase, FuzzCompareOp, FuzzLogicalOp, FuzzReduction, FuzzTensor,
    FuzzScatterOp, FuzzUnaryOp,
};
use crate::{DType, Scalar, Storage, TensorData};

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
        FuzzCase::Gather {
            // Axis-one duplicate/reorder selection is intentionally obvious.
            input: tensor(vec![2, 4], Storage::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])),
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
            input: tensor(vec![3], Storage::F32(vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0001), f32::INFINITY])),
            index: tensor(vec![3], Storage::I32(vec![1, 0, 2])),
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
            updates: tensor(vec![3], Storage::F32(vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0001), f32::INFINITY])),
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
                tensor(vec![1], Storage::F64(vec![f64::from_bits(0x8000_0000_0000_0000)])),
                tensor(vec![0], Storage::F64(vec![])),
                tensor(vec![1], Storage::F64(vec![f64::from_bits(0x7ff8_0000_0000_0001)])),
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
            rhs: tensor(vec![3, 2, 1], Storage::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        },
        FuzzCase::Matmul {
            lhs: tensor(vec![3, 0], Storage::F32(vec![])),
            rhs: tensor(vec![0, 5], Storage::F32(vec![])),
        },
        FuzzCase::Select {
            condition: tensor(vec![3], Storage::Bool(vec![true, false, true])),
            on_true: tensor(vec![3], Storage::BF16(vec![0x3f80, 0x8000, 0x7fc1])),
            on_false: FuzzTensor::from_tensor(&TensorData::scalar_with_dtype(
                Scalar::F(2.0),
                DType::BF16,
            )),
        },
    ]
}
