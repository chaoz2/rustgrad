//! Immutable static matmul contract for future native renderers.
use crate::{DType, Graph, NodeId, Op, Scalar, Shape, TensorData};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MatmulKernelPlan {
    pub lhs: NodeId,
    pub rhs: NodeId,
    pub output: NodeId,
    pub lhs_shape: Shape,
    pub rhs_shape: Shape,
    pub output_shape: Shape,
    pub lhs_dtype: DType,
    pub rhs_dtype: DType,
    pub dtype: DType,
    pub batch_shape: Vec<usize>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_vector: bool,
    pub rhs_vector: bool,
    pub cache_key: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatmulPlanError {
    NotMatmul,
    InvalidGeometry,
    Overflow,
    DType,
}
impl fmt::Display for MatmulPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "matmul plan error: {self:?}")
    }
}
impl std::error::Error for MatmulPlanError {}
impl MatmulKernelPlan {
    pub fn from_graph(graph: &Graph, output: NodeId) -> Result<Self, MatmulPlanError> {
        let Op::Matmul { lhs, rhs } = graph
            .op(output)
            .map_err(|_| MatmulPlanError::InvalidGeometry)?
        else {
            return Err(MatmulPlanError::NotMatmul);
        };
        let lhs_shape = graph
            .shape(*lhs)
            .map_err(|_| MatmulPlanError::InvalidGeometry)?
            .clone();
        let rhs_shape = graph
            .shape(*rhs)
            .map_err(|_| MatmulPlanError::InvalidGeometry)?
            .clone();
        let output_shape = crate::ir::matmul_shape(&lhs_shape, &rhs_shape)
            .ok_or(MatmulPlanError::InvalidGeometry)?;
        let lhs_vector = lhs_shape.rank() == 1;
        let rhs_vector = rhs_shape.rank() == 1;
        let k = *lhs_shape
            .dims()
            .last()
            .ok_or(MatmulPlanError::InvalidGeometry)?;
        let rk = if rhs_vector {
            rhs_shape.dims()[0]
        } else {
            rhs_shape.dims()[rhs_shape.rank() - 2]
        };
        if k != rk {
            return Err(MatmulPlanError::InvalidGeometry);
        };
        let m = if lhs_vector {
            1
        } else {
            lhs_shape.dims()[lhs_shape.rank() - 2]
        };
        let n = if rhs_vector {
            1
        } else {
            *rhs_shape.dims().last().unwrap()
        };
        let lb = if lhs_vector {
            &[][..]
        } else {
            &lhs_shape.dims()[..lhs_shape.rank() - 2]
        };
        let rb = if rhs_vector {
            &[][..]
        } else {
            &rhs_shape.dims()[..rhs_shape.rank() - 2]
        };
        let rank = lb.len().max(rb.len());
        let mut batch = Vec::with_capacity(rank);
        for i in 0..rank {
            let a = lb
                .get(i + lb.len().saturating_sub(rank))
                .copied()
                .unwrap_or(1);
            let b = rb
                .get(i + rb.len().saturating_sub(rank))
                .copied()
                .unwrap_or(1);
            if a != b && a != 1 && b != 1 {
                return Err(MatmulPlanError::InvalidGeometry);
            }
            batch.push(a.max(b));
        }
        output_shape
            .numel()
            .map_err(|_| MatmulPlanError::Overflow)?;
        let lhs_dtype = graph.dtype(*lhs).map_err(|_| MatmulPlanError::DType)?;
        let rhs_dtype = graph.dtype(*rhs).map_err(|_| MatmulPlanError::DType)?;
        let dtype = lhs_dtype.promote(rhs_dtype);
        let mut p = Self {
            lhs: *lhs,
            rhs: *rhs,
            output,
            lhs_shape,
            rhs_shape,
            output_shape,
            lhs_dtype,
            rhs_dtype,
            dtype,
            batch_shape: batch,
            m,
            n,
            k,
            lhs_vector,
            rhs_vector,
            cache_key: 0,
        };
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        p.cache_key = h.finish();
        Ok(p)
    }
    pub fn abi_nodes(&self) -> [NodeId; 3] {
        [self.lhs, self.rhs, self.output]
    }
    pub fn execute(
        &self,
        lhs: &TensorData,
        rhs: &TensorData,
    ) -> Result<TensorData, MatmulPlanError> {
        if lhs.shape() != &self.lhs_shape
            || rhs.shape() != &self.rhs_shape
            || lhs.dtype() != self.lhs_dtype
            || rhs.dtype() != self.rhs_dtype
        {
            return Err(MatmulPlanError::InvalidGeometry);
        }
        let out_len = self
            .output_shape
            .numel()
            .map_err(|_| MatmulPlanError::Overflow)?;
        let mut out = Vec::with_capacity(out_len);
        for linear in 0..out_len {
            let c = coords(&self.output_shape, linear);
            let batch_len = self.batch_shape.len();
            let row = if self.lhs_vector { 0 } else { c[batch_len] };
            let col = if self.rhs_vector {
                0
            } else {
                c[batch_len + usize::from(!self.lhs_vector)]
            };
            let mut acc = Scalar::I(0);
            for inner in 0..self.k {
                let a = lhs.scalar_at(offset(
                    &self.lhs_shape,
                    &c[..batch_len],
                    row,
                    inner,
                    self.lhs_vector,
                    false,
                ));
                let b = rhs.scalar_at(offset(
                    &self.rhs_shape,
                    &c[..batch_len],
                    inner,
                    col,
                    false,
                    self.rhs_vector,
                ));
                acc = add(acc, mul(a, b, self.dtype), self.dtype);
            }
            out.push(acc);
        }
        TensorData::from_scalars(self.output_shape.clone(), self.dtype, out)
            .map_err(|_| MatmulPlanError::DType)
    }
}
fn coords(shape: &Shape, mut x: usize) -> Vec<usize> {
    let mut r = vec![0; shape.rank()];
    for (i, d) in shape.dims().iter().enumerate().rev() {
        r[i] = x % d;
        x /= d;
    }
    r
}
fn offset(
    shape: &Shape,
    batch: &[usize],
    row: usize,
    col: usize,
    _vector: bool,
    rhs_vector: bool,
) -> usize {
    if shape.rank() == 1 {
        return row.max(col);
    }
    let dims = shape.dims();
    let br = dims.len() - 2;
    let pad = batch.len() - br;
    let mut x = 0;
    for i in 0..br {
        x = x * dims[i] + if dims[i] == 1 { 0 } else { batch[i + pad] };
    }
    x = x * dims[br] + if rhs_vector { col } else { row };
    x * dims[br + 1] + if rhs_vector { 0 } else { col }
}
fn mul(a: Scalar, b: Scalar, d: DType) -> Scalar {
    if d.is_float() {
        Scalar::F(a.as_f64() * b.as_f64())
    } else if d == DType::Bool {
        Scalar::Bool(a.as_bool() && b.as_bool())
    } else if matches!(d.category(), crate::DTypeCategory::Unsigned) {
        Scalar::U(a.as_u64().wrapping_mul(b.as_u64()))
    } else {
        Scalar::I(a.as_i64().wrapping_mul(b.as_i64()))
    }
}
fn add(a: Scalar, b: Scalar, d: DType) -> Scalar {
    if d.is_float() {
        Scalar::F(a.as_f64() + b.as_f64())
    } else if d == DType::Bool {
        Scalar::Bool(a.as_bool() || b.as_bool())
    } else if matches!(d.category(), crate::DTypeCategory::Unsigned) {
        Scalar::U(a.as_u64().wrapping_add(b.as_u64()))
    } else {
        Scalar::I(a.as_i64().wrapping_add(b.as_i64()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend};
    use std::collections::HashMap;
    #[test]
    fn normalized_plan_matches_cpu() {
        for (a, b) in [
            (vec![3], vec![3]),
            (vec![2, 3], vec![3]),
            (vec![3], vec![3, 2]),
            (vec![2, 1, 3], vec![1, 3, 2]),
        ] {
            let mut g = Graph::new();
            let x = g.input_dtype("x", a.clone(), DType::I32);
            let y = g.input_dtype("y", b.clone(), DType::I32);
            let z = g.matmul(x, y).unwrap();
            let p = MatmulKernelPlan::from_graph(&g, z).unwrap();
            let tx = TensorData::from_scalars(
                a,
                DType::I32,
                (0..p.lhs_shape.numel().unwrap()).map(|i| Scalar::I(i as i64 - 2)),
            )
            .unwrap();
            let ty = TensorData::from_scalars(
                b,
                DType::I32,
                (0..p.rhs_shape.numel().unwrap()).map(|i| Scalar::I(i as i64 + 1)),
            )
            .unwrap();
            let got = p.execute(&tx, &ty).unwrap();
            let want = CpuBackend
                .execute(&g, z, &HashMap::from([("x".into(), tx), ("y".into(), ty)]))
                .unwrap();
            assert_eq!(got.storage(), want.storage());
        }
    }
}
