use super::{FuzzBinaryOp, FuzzCase, FuzzReduction, FuzzTensor};
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

fn static_shape(rng: &mut SplitMix64) -> Vec<usize> {
    [vec![], vec![0], vec![1], vec![3], vec![17], vec![2, 3]][rng.pick(6)].clone()
}

/// Deterministically generates the `index`th valid bounded case for `seed`.
pub fn generate_case(seed: u64, index: u64) -> FuzzCase {
    let mut rng = SplitMix64(seed ^ index.wrapping_mul(0xd6e8_feb8_6659_fd93));
    match rng.pick(6) {
        0 => {
            let shape = static_shape(&mut rng);
            let dtype = if rng.pick(2) == 0 {
                DType::F32
            } else {
                DType::I32
            };
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
            let dtype = if rng.pick(2) == 0 {
                DType::F32
            } else {
                DType::I32
            };
            let false_shape = if rng.pick(2) == 0 {
                vec![]
            } else {
                shape.clone()
            };
            FuzzCase::Select {
                condition: tensor(&mut rng, shape.clone(), DType::Bool),
                on_true: tensor(&mut rng, shape.clone(), dtype),
                on_false: tensor(&mut rng, false_shape, dtype),
            }
        }
        2 => {
            let shape = static_shape(&mut rng);
            let choices = [DType::Bool, DType::I32, DType::F16, DType::BF16, DType::F32];
            let from = choices[rng.pick(choices.len())];
            let mut to = choices[rng.pick(choices.len())];
            if to == from {
                to = DType::F32;
            }
            FuzzCase::Cast {
                input: tensor(&mut rng, shape, from),
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
            let rows = [0, 1, 2, 3][rng.pick(4)];
            let columns = [0, 1, 3, 8][rng.pick(4)];
            let reduction = if columns == 0 {
                [FuzzReduction::Sum, FuzzReduction::Product][rng.pick(2)]
            } else {
                [
                    FuzzReduction::Sum,
                    FuzzReduction::Mean,
                    FuzzReduction::Product,
                ][rng.pick(3)]
            };
            FuzzCase::Reduction {
                input: tensor(&mut rng, vec![rows, columns], DType::F32),
                reduction,
                axis: 1,
                keepdim: rng.pick(2) == 0,
            }
        }
        _ => {
            let m = [0, 1, 2, 3][rng.pick(4)];
            let n = [0, 1, 3, 5][rng.pick(4)];
            let k = [0, 1, 3, 8][rng.pick(4)];
            FuzzCase::Matmul {
                lhs: tensor(&mut rng, vec![m, k], DType::F32),
                rhs: tensor(&mut rng, vec![k, n], DType::F32),
            }
        }
    }
}
