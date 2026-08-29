use super::{
    FuzzBinaryOp, FuzzCase, FuzzCompareOp, FuzzLogicalOp, FuzzReduction, FuzzTensor,
    FuzzScatterOp, FuzzUnaryOp,
};
use crate::{DType, Scalar, Shape, TensorData};

#[derive(Clone, Copy)]
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    fn pick(&mut self, limit: usize) -> usize {
        (self.next() % limit as u64) as usize
    }
}

fn tensor(rng: &mut SplitMix64, shape: Vec<usize>, dtype: DType) -> FuzzTensor {
    let elements = Shape::new(shape.clone())
        .numel()
        .expect("bounded generated shape");
    let values = (0..elements).map(|index| {
        let raw = rng.next().wrapping_add(index as u64);
        match dtype {
            DType::Bool => Scalar::Bool(raw & 1 != 0),
            DType::I32 => Scalar::I((raw % 17) as i64 - 8),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                Scalar::F((raw % 33) as f64 / 4.0 - 4.0)
            }
            _ => Scalar::I((raw % 17) as i64 - 8),
        }
    });
    FuzzTensor::from_tensor(
        &TensorData::from_scalars(shape, dtype, values).expect("generated tensor geometry"),
    )
}

// Raw C casts outside this domain can differ from the TensorData oracle for
// non-finite/out-of-range float-to-int values or signed overflow. Keep the
// generated Cast matrix in values that every concrete storage conversion can
// represent exactly; focused acceptance covers finite truncation separately.
fn cast_tensor(rng: &mut SplitMix64, shape: Vec<usize>, dtype: DType) -> FuzzTensor {
    let elements = Shape::new(shape.clone())
        .numel()
        .expect("bounded generated cast shape");
    let values = (0..elements).map(|index| {
        let value = (rng.next().wrapping_add(index as u64) % 3) as i64;
        match dtype {
            DType::Bool => Scalar::Bool(value != 0),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => Scalar::F(value as f64),
            _ => Scalar::I(value),
        }
    });
    FuzzTensor::from_tensor(
        &TensorData::from_scalars(shape, dtype, values)
            .expect("generated cast tensor geometry"),
    )
}

fn static_shape(rng: &mut SplitMix64) -> Vec<usize> {
    [vec![], vec![0], vec![1], vec![3], vec![17], vec![2, 3]][rng.pick(6)].clone()
}

fn gather_index(
    rng: &mut SplitMix64,
    shape: Vec<usize>,
    axis_extent: usize,
    dtype: DType,
) -> FuzzTensor {
    let elements = Shape::new(shape.clone())
        .numel()
        .expect("bounded generated gather index shape");
    debug_assert!(elements == 0 || axis_extent != 0);
    FuzzTensor::from_tensor(
        &TensorData::from_scalars(
            shape,
            dtype,
            (0..elements).map(|_| Scalar::I(rng.pick(axis_extent) as i64)),
        )
        .expect("generated gather indices are in range"),
    )
}

/// Deterministically generates the `index`th valid bounded case for `seed`.
pub fn generate_case(seed: u64, index: u64) -> FuzzCase {
    let mut rng = SplitMix64(seed ^ index.wrapping_mul(0xd6e8_feb8_6659_fd93));
    match rng.pick(15) {
        0 => {
            let shape = static_shape(&mut rng);
            let dtype = [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64][rng.pick(13)];
            let rhs_shape = if rng.pick(3) == 0 {
                vec![]
            } else {
                shape.clone()
            };
            let op = [
                FuzzBinaryOp::Add,
                FuzzBinaryOp::Sub,
                FuzzBinaryOp::Mul,
                FuzzBinaryOp::Maximum,
            ][rng.pick(4)];
            FuzzCase::Binary {
                op,
                lhs: tensor(&mut rng, shape, dtype),
                rhs: tensor(&mut rng, rhs_shape, dtype),
            }
        }
        1 => {
            let shape = static_shape(&mut rng);
            let dtype = [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64][rng.pick(13)];
            let false_shape = if rng.pick(2) == 0 {
                vec![]
            } else {
                shape.clone()
            };
            let condition_shape = match rng.pick(3) {
                0 => vec![],
                1 if shape.len() >= 2 => vec![1, *shape.last().unwrap()],
                _ => shape.clone(),
            };
            FuzzCase::Select {
                condition: tensor(&mut rng, condition_shape, DType::Bool),
                on_true: tensor(&mut rng, shape.clone(), dtype),
                on_false: tensor(&mut rng, false_shape, dtype),
            }
        }
        2 => {
            let shape = static_shape(&mut rng);
            let choices = [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64];
            let from = choices[rng.pick(choices.len())];
            let to = choices[rng.pick(choices.len())];
            FuzzCase::Cast {
                input: cast_tensor(&mut rng, shape, from),
                to,
            }
        }
        3 => {
            let rows = [1, 3, 4, 8][rng.pick(4)];
            let start = rng.pick(rows + 1);
            let end = start + rng.pick(rows - start + 1);
            let expand = [1, 3, 8][rng.pick(3)];
            FuzzCase::AffineView {
                input: tensor(&mut rng, vec![rows, 1], DType::F32),
                start,
                end,
                expand,
            }
        }
        4 => {
            // All raw reduction kinds share the typed ReduceFinalize path.
            // Extrema require a nonempty reduced axis; Sum/Product/Mean keep
            // their defined zero-domain identities in this bounded family.
            let rank = 1 + rng.pick(3);
            let axis = rng.pick(rank);
            let shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let reduction = if shape[axis] == 0 {
                [FuzzReduction::Sum, FuzzReduction::Mean, FuzzReduction::Product][rng.pick(3)]
            } else {
                [
                    FuzzReduction::Sum,
                    FuzzReduction::Mean,
                    FuzzReduction::Product,
                    FuzzReduction::Max,
                    FuzzReduction::Min,
                ][rng.pick(5)]
            };
            FuzzCase::Reduction {
                input: tensor(&mut rng, shape, DType::F32),
                reduction,
                axis,
                keepdim: rng.pick(2) == 0,
            }
        }
        5 => {
            // Raw Concat is a homogeneous movement kernel in captured/native
            // replay. Preserve the legacy pair schema, but generate the
            // additive many-input surface across every local storage dtype.
            let rank = 1 + rng.pick(3);
            let axis = rng.pick(rank);
            let dtype = [
                DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
                DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let arity = 2 + rng.pick(3);
            let mut base_shape = Vec::with_capacity(rank);
            for dimension in 0..rank {
                base_shape.push(if dimension == axis { 0 } else { [0, 1, 2, 3][rng.pick(4)] });
            }
            FuzzCase::ConcatMany {
                inputs: (0..arity)
                    .map(|_| {
                        let mut shape = base_shape.clone();
                        shape[axis] = [0, 1, 2, 3][rng.pick(4)];
                        tensor(&mut rng, shape, dtype)
                    })
                    .collect(),
                axis,
            }
        }
        6 => {
            // Direct GraphUnary Neg/Abs have a complete bounded CPU,
            // captured, and strict-native path for F32 and small I32 lanes.
            // Do not claim Bool/narrow-float coverage through this raw path.
            let dtype = [DType::F32, DType::I32][rng.pick(2)];
            let op = [FuzzUnaryOp::Neg, FuzzUnaryOp::Abs][rng.pick(2)];
            let shape = static_shape(&mut rng);
            FuzzCase::Unary {
                op,
                input: tensor(&mut rng, shape, dtype),
            }
        }
        7 => {
            // Raw GraphCompare is portable for F32 and small I32 through the
            // CPU oracle, captured elementwise lowering, and strict native
            // replay. A scalar RHS deliberately exercises right broadcasting.
            let dtype = [DType::F32, DType::I32][rng.pick(2)];
            let shape = static_shape(&mut rng);
            let rhs_shape = if rng.pick(2) == 0 {
                vec![]
            } else {
                shape.clone()
            };
            let op = [
                FuzzCompareOp::Eq,
                FuzzCompareOp::Ne,
                FuzzCompareOp::Lt,
                FuzzCompareOp::Le,
                FuzzCompareOp::Gt,
                FuzzCompareOp::Ge,
            ][rng.pick(6)];
            FuzzCase::Compare {
                op,
                lhs: tensor(&mut rng, shape, dtype),
                rhs: tensor(&mut rng, rhs_shape, dtype),
            }
        }
        8 => {
            // Direct GraphLogical And/Or is a Bool-only elementwise kernel
            // through the CPU oracle, captured replay, and strict native path.
            let shape = static_shape(&mut rng);
            let rhs_shape = if rng.pick(2) == 0 {
                vec![]
            } else {
                shape.clone()
            };
            let op = [FuzzLogicalOp::And, FuzzLogicalOp::Or][rng.pick(2)];
            FuzzCase::Logical {
                op,
                lhs: tensor(&mut rng, shape, DType::Bool),
                rhs: tensor(&mut rng, rhs_shape, DType::Bool),
            }
        }
        9 => {
            // Source `logical_not` is Cast(Bool) then Ne(true), not the raw
            // GraphLogical Not opcode. This intersection is native-safe only
            // after Bool-target Cast adopted `!=0` truthiness.
            let dtype = [DType::Bool, DType::I32, DType::F32][rng.pick(3)];
            let shape = static_shape(&mut rng);
            FuzzCase::LogicalNot {
                input: tensor(&mut rng, shape, dtype),
            }
        }
        10 => {
            // Tensor.T is a rank-two literal Permute([1, 0]), never the
            // identity Input alias that blocks the broader Permute surface.
            let shape = vec![[0, 1, 2, 3][rng.pick(4)], [0, 1, 2, 3][rng.pick(4)]];
            let dtype = [DType::F32, DType::I32, DType::F16, DType::Bool][rng.pick(4)];
            FuzzCase::TensorT {
                input: tensor(&mut rng, shape, dtype),
            }
        }
        11 => {
            // Raw Pad is a self-contained homogeneous movement plan. It is
            // intentionally distinct from source-level signed/mode-aware pad.
            let rank = rng.pick(4);
            let dtype = [
                DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
                DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let padding = (0..rank)
                .map(|_| ([0, 1, 2][rng.pick(3)], [0, 1, 2][rng.pick(3)]))
                .collect::<Vec<_>>();
            FuzzCase::Pad {
                input: tensor(&mut rng, shape, dtype),
                padding,
                fill: tensor(&mut rng, vec![], dtype),
            }
        }
        12 => {
            // Raw Gather is a homogeneous movement kernel. Keep live indices
            // signed, nonnegative, in range, and same-rank by construction.
            let rank = 1 + rng.pick(3);
            let axis = rng.pick(rank);
            let dtype = [
                DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32,
                DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let input_shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let mut index_shape = Vec::with_capacity(rank);
            for dimension in 0..rank {
                if dimension == axis {
                    let source = input_shape[dimension];
                    index_shape.push(if source == 0 { 0 } else { rng.pick(4) });
                } else {
                    index_shape.push(rng.pick(input_shape[dimension] + 1));
                }
            }
            let index_dtype = [DType::I32, DType::I64][rng.pick(2)];
            let index = gather_index(
                &mut rng,
                index_shape,
                input_shape[axis],
                index_dtype,
            );
            FuzzCase::Gather {
                input: tensor(&mut rng, input_shape, dtype),
                index,
                axis,
            }
        }
        13 => {
            // Raw Scatter is a self-contained homogeneous movement kernel.
            // Replacement covers every portable movement dtype; Add is the
            // explicitly portable F32/F64 raw arithmetic subset.
            let rank = 1 + rng.pick(3);
            let axis = rng.pick(rank);
            let op = [FuzzScatterOp::Replace, FuzzScatterOp::Add][rng.pick(2)];
            let dtype = if op == FuzzScatterOp::Add {
                [DType::F32, DType::F64][rng.pick(2)]
            } else {
                [DType::Bool, DType::I8, DType::U8, DType::I16, DType::U16, DType::I32, DType::U32, DType::I64, DType::U64, DType::F16, DType::BF16, DType::F32, DType::F64][rng.pick(13)]
            };
            let base_shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let mut index_shape = Vec::with_capacity(rank);
            for dimension in 0..rank {
                if dimension == axis {
                    let source = base_shape[dimension];
                    index_shape.push(if source == 0 { 0 } else { rng.pick(4) });
                } else {
                    index_shape.push(rng.pick(base_shape[dimension] + 1));
                }
            }
            let updates_shape = index_shape
                .iter()
                .map(|extent| extent + rng.pick(2))
                .collect::<Vec<_>>();
            let index_dtype = [DType::I32, DType::I64][rng.pick(2)];
            let index = gather_index(
                &mut rng,
                index_shape,
                base_shape[axis],
                index_dtype,
            );
            FuzzCase::Scatter {
                base: tensor(&mut rng, base_shape, dtype),
                index,
                updates: tensor(&mut rng, updates_shape, dtype),
                axis,
                op,
            }
        }
        _ => {
            // Raw Matmul is portable only for homogeneous F32/F64. Exercise
            // every generalized rank form accepted by the static plan:
            // vectors, matrix/vector variants, matrices, and right-aligned
            // broadcast batches. All extents remain deliberately small.
            let m = [0, 1, 2, 3][rng.pick(4)];
            let n = [0, 1, 3, 5][rng.pick(4)];
            let k = [0, 1, 3, 8][rng.pick(4)];
            let dtype = [DType::F32, DType::F64][rng.pick(2)];
            let (lhs_shape, rhs_shape) = match rng.pick(5) {
                0 => (vec![k], vec![k]),
                1 => (vec![m, k], vec![k]),
                2 => (vec![k], vec![k, n]),
                3 => (vec![m, k], vec![k, n]),
                _ => {
                    let outer = [0, 1, 2][rng.pick(3)];
                    let inner = [0, 1, 2][rng.pick(3)];
                    if rng.pick(2) == 0 {
                        (vec![outer, 1, m, k], vec![inner, k, n])
                    } else {
                        (vec![outer, inner, m, k], vec![1, inner, k, n])
                    }
                }
            };
            FuzzCase::Matmul {
                lhs: tensor(&mut rng, lhs_shape, dtype),
                rhs: tensor(&mut rng, rhs_shape, dtype),
            }
        }
    }
}
