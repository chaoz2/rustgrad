use super::{
    FuzzBinaryOp, FuzzCase, FuzzCompareOp, FuzzLogicalOp, FuzzReduction, FuzzScatterOp, FuzzSlice,
    FuzzTensor, FuzzUnaryOp,
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

// Raw integer reduction in native C is exact for this bounded domain: it
// avoids signed overflow while still exercising each storage family and the
// graph's output/accumulator dtype policy. Product uses only -1/0/1 (or 0/1
// unsigned) so its complete reduction domain remains defined at every width.
fn reduction_tensor(
    rng: &mut SplitMix64,
    shape: Vec<usize>,
    dtype: DType,
    reduction: FuzzReduction,
) -> FuzzTensor {
    let elements = Shape::new(shape.clone())
        .numel()
        .expect("bounded generated reduction shape");
    let values = (0..elements).map(|index| {
        let raw = rng.next().wrapping_add(index as u64);
        match dtype {
            DType::Bool => Scalar::Bool(raw & 1 != 0),
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
                let value = if reduction == FuzzReduction::Product {
                    raw & 1
                } else {
                    raw % 3
                };
                Scalar::U(value)
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => Scalar::I((raw % 3) as i64 - 1),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => {
                let value = (raw % 3) as i64 - 1;
                Scalar::F(value as f64)
            }
            _ => unreachable!("float8 reduction fuzz is not generated"),
        }
    });
    FuzzTensor::from_tensor(
        &TensorData::from_scalars(shape, dtype, values)
            .expect("generated reduction tensor geometry"),
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
        &TensorData::from_scalars(shape, dtype, values).expect("generated cast tensor geometry"),
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
    match rng.pick(17) {
        0 => {
            let shape = static_shape(&mut rng);
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
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
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
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
            let choices = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ];
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
                [
                    FuzzReduction::Sum,
                    FuzzReduction::Mean,
                    FuzzReduction::Product,
                ][rng.pick(3)]
            } else {
                [
                    FuzzReduction::Sum,
                    FuzzReduction::Mean,
                    FuzzReduction::Product,
                    FuzzReduction::Max,
                    FuzzReduction::Min,
                ][rng.pick(5)]
            };
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
            FuzzCase::Reduction {
                input: reduction_tensor(&mut rng, shape, dtype, reduction),
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
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let arity = 2 + rng.pick(3);
            let mut base_shape = Vec::with_capacity(rank);
            for dimension in 0..rank {
                base_shape.push(if dimension == axis {
                    0
                } else {
                    [0, 1, 2, 3][rng.pick(4)]
                });
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
            // Neg/Abs retain their all-storage surface. The additional raw
            // transcendental/discrete operations are the exact F32/F64
            // CPU/captured/strict-native intersection proven by the native
            // renderer; narrow-float admission remains deliberately absent.
            let (op, dtype) = if rng.pick(2) == 0 {
                (
                    [FuzzUnaryOp::Neg, FuzzUnaryOp::Abs][rng.pick(2)],
                    [
                        DType::Bool,
                        DType::I8,
                        DType::U8,
                        DType::I16,
                        DType::U16,
                        DType::I32,
                        DType::U32,
                        DType::I64,
                        DType::U64,
                        DType::F16,
                        DType::BF16,
                        DType::F32,
                        DType::F64,
                    ][rng.pick(13)],
                )
            } else {
                (
                    [
                        FuzzUnaryOp::Exp2,
                        FuzzUnaryOp::Log2,
                        FuzzUnaryOp::Sin,
                        FuzzUnaryOp::Cos,
                        FuzzUnaryOp::Tan,
                        FuzzUnaryOp::Log,
                        FuzzUnaryOp::Trunc,
                    ][rng.pick(7)],
                    [DType::F32, DType::F64][rng.pick(2)],
                )
            };
            let shape = static_shape(&mut rng);
            FuzzCase::Unary {
                op,
                input: tensor(&mut rng, shape, dtype),
            }
        }
        7 => {
            // Raw GraphCompare is homogeneous here: CPU, captured UOps, and
            // C compare each stored kind directly, including I64/U64 rather
            // than projecting through f64. Exercise scalar, equal, and
            // right-aligned broadcast geometry without source promotion.
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let shape = [
                vec![],
                vec![0],
                vec![1],
                vec![3],
                vec![2, 3],
                vec![0, 3],
                vec![2, 1, 3],
                vec![0, 1, 3],
            ][rng.pick(8)]
            .clone();
            let rhs_shape = match rng.pick(3) {
                0 => vec![],
                1 => shape.clone(),
                _ if shape.len() >= 2 => vec![1, *shape.last().unwrap()],
                _ => shape.clone(),
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
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
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
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let input_shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let mut index_shape = Vec::with_capacity(rank);
            for (dimension, &source) in input_shape.iter().enumerate() {
                if dimension == axis {
                    index_shape.push(if source == 0 { 0 } else { rng.pick(4) });
                } else {
                    index_shape.push(rng.pick(source + 1));
                }
            }
            let index_dtype = [DType::I32, DType::I64][rng.pick(2)];
            let index = gather_index(&mut rng, index_shape, input_shape[axis], index_dtype);
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
                [
                    DType::Bool,
                    DType::I8,
                    DType::U8,
                    DType::I16,
                    DType::U16,
                    DType::I32,
                    DType::U32,
                    DType::I64,
                    DType::U64,
                    DType::F16,
                    DType::BF16,
                    DType::F32,
                    DType::F64,
                ][rng.pick(13)]
            };
            let base_shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let mut index_shape = Vec::with_capacity(rank);
            for (dimension, &source) in base_shape.iter().enumerate() {
                if dimension == axis {
                    index_shape.push(if source == 0 { 0 } else { rng.pick(4) });
                } else {
                    index_shape.push(rng.pick(source + 1));
                }
            }
            let updates_shape = index_shape
                .iter()
                .map(|extent| extent + rng.pick(2))
                .collect::<Vec<_>>();
            let index_dtype = [DType::I32, DType::I64][rng.pick(2)];
            let index = gather_index(&mut rng, index_shape, base_shape[axis], index_dtype);
            FuzzCase::Scatter {
                base: tensor(&mut rng, base_shape, dtype),
                index,
                updates: tensor(&mut rng, updates_shape, dtype),
                axis,
                op,
            }
        }
        14 => {
            // Raw Permute includes source identity and rank-zero passthrough
            // programs as well as ordinary affine views. Captured replay owns
            // passthrough Inputs directly rather than fabricating a kernel.
            let rank = rng.pick(4);
            let shape = (0..rank)
                .map(|_| [0, 1, 2, 3][rng.pick(4)])
                .collect::<Vec<_>>();
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(13)];
            let mut axes = (0..rank).collect::<Vec<_>>();
            for position in (1..rank).rev() {
                axes.swap(position, rng.pick(position + 1));
            }
            FuzzCase::Permute {
                input: tensor(&mut rng, shape, dtype),
                axes,
            }
        }
        15 => {
            // Raw Stride retains Python-style signed slicing as a source-backed
            // affine view. Exercise full/reverse/stepped/bounded slices across
            // scalar, empty, and ordinary ranks without malformed step zero.
            let rank = rng.pick(4);
            let shape = (0..rank)
                .map(|_| [0, 1, 2, 3, 5][rng.pick(5)])
                .collect::<Vec<_>>();
            let dtype = [
                DType::Bool,
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::I32,
                DType::U32,
                DType::I64,
                DType::U64,
                DType::F8E4M3,
                DType::F8E5M2,
                DType::F8E4M3FNUZ,
                DType::F8E5M2FNUZ,
                DType::F16,
                DType::BF16,
                DType::F32,
                DType::F64,
            ][rng.pick(17)];
            let slices = shape
                .iter()
                .map(|extent| match rng.pick(6) {
                    0 => FuzzSlice {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    1 => FuzzSlice {
                        start: None,
                        stop: None,
                        step: -1,
                    },
                    2 => FuzzSlice {
                        start: None,
                        stop: None,
                        step: 2,
                    },
                    3 => FuzzSlice {
                        start: None,
                        stop: None,
                        step: -2,
                    },
                    4 => FuzzSlice {
                        start: Some(0),
                        stop: Some(i64::try_from(*extent).expect("bounded stride extent")),
                        step: 1,
                    },
                    _ => FuzzSlice {
                        start: Some(-1),
                        stop: None,
                        step: -1,
                    },
                })
                .collect();
            FuzzCase::Stride {
                input: tensor(&mut rng, shape, dtype),
                slices,
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
