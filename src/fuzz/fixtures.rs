use super::{
    FuzzBinaryOp, FuzzCase, FuzzCompareOp, FuzzLogicalOp, FuzzReduction, FuzzTensor,
    FuzzUnaryOp,
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
        FuzzCase::Concat {
            lhs: tensor(vec![2, 0], Storage::I32(vec![])),
            rhs: tensor(vec![2, 3], Storage::I32(vec![0; 6])),
            axis: 1,
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
