use super::{DType, Scalar, Shape, TensorData};
use crate::{Error, Result};

impl TensorData {
    /// Creates a dense tensor in an explicit dtype. Values are converted using
    /// Rust's saturating float-to-int and wrapping integer narrowing rules.
    pub fn full_with_dtype(shape: impl Into<Shape>, value: Scalar, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        Self::from_scalars(shape.clone(), dtype, vec![value; shape.numel()?])
    }

    /// Creates a dense tensor whose elements are all `value`.
    pub fn full(shape: impl Into<Shape>, value: f32) -> Result<Self> {
        let shape = shape.into();
        Self::new(shape.clone(), vec![value; shape.numel()?])
    }

    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, 0.0)
    }

    pub fn zeros_with_dtype(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::full_with_dtype(shape, Scalar::I(0), dtype)
    }

    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, 1.0)
    }

    pub fn ones_with_dtype(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::full_with_dtype(shape, Scalar::I(1), dtype)
    }

    /// RustGrad's deterministic `empty` contract: dense storage initialized to
    /// zero. Unlike tinygrad's allocator-oriented `empty`, it never exposes
    /// uninitialized bytes to an inspectable CPU graph.
    pub fn empty(shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        Self::zeros_with_dtype(shape, dtype)
    }

    /// Evenly spaced values including both endpoints.
    pub fn linspace(start: f64, stop: f64, steps: isize, dtype: DType) -> Result<Self> {
        if steps < 0 {
            return Err(Error::InvalidLinspace { steps });
        }
        if dtype == DType::Bool {
            return Err(Error::InvalidRandom {
                reason: "linspace does not support bool dtype",
            });
        }
        let steps = steps as usize;
        let values = match steps {
            0 => Vec::new(),
            1 => vec![Scalar::F(start)],
            _ => (0..steps)
                .map(|index| Scalar::F(start + (stop - start) * index as f64 / (steps - 1) as f64))
                .collect(),
        };
        Self::from_scalars([steps], dtype, values)
    }

    /// A rectangular identity matrix, with the diagonal represented exactly
    /// in every supported dense dtype.
    pub fn eye(rows: usize, columns: Option<usize>, dtype: DType) -> Result<Self> {
        let columns = columns.unwrap_or(rows);
        let count = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([rows, columns])))?;
        Self::from_scalars(
            [rows, columns],
            dtype,
            (0..count).map(|index| Scalar::I((index / columns == index % columns) as i64)),
        )
    }

    /// Integer arange with tinygrad/NumPy-style half-open bounds and exact i64
    /// storage (unless a caller subsequently casts it).
    pub fn arange(start: i64, end: i64, step: i64) -> Result<Self> {
        if step == 0 {
            return Err(Error::InvalidArange { start, end, step });
        }

        let mut values = Vec::new();
        let mut value = start;
        while (step > 0 && value < end) || (step < 0 && value > end) {
            values.push(Scalar::I(value));
            // A checked overflow here can only occur after the final valid
            // i64 value: the mathematical successor lies beyond the domain,
            // and therefore cannot satisfy the half-open bound.
            let Some(next) = value.checked_add(step) else {
                break;
            };
            value = next;
        }
        Self::from_scalars([values.len()], DType::I64, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_full_zeros_ones_and_arange() {
        assert_eq!(
            TensorData::full([2, 2], 3.5).unwrap(),
            TensorData::new([2, 2], vec![3.5; 4]).unwrap()
        );
        assert_eq!(
            TensorData::zeros([3]).unwrap(),
            TensorData::new([3], vec![0.0; 3]).unwrap()
        );
        assert_eq!(TensorData::ones([]).unwrap(), TensorData::scalar(1.0));
        assert_eq!(
            TensorData::arange(5, -2, -2).unwrap(),
            TensorData::from_scalars(
                [4],
                DType::I64,
                [Scalar::I(5), Scalar::I(3), Scalar::I(1), Scalar::I(-1)],
            )
            .unwrap()
        );
        assert_eq!(
            TensorData::zeros_with_dtype([2], DType::U16)
                .unwrap()
                .dtype(),
            DType::U16
        );
        assert_eq!(
            TensorData::arange(0, 4, 0),
            Err(Error::InvalidArange {
                start: 0,
                end: 4,
                step: 0
            })
        );
        assert_eq!(
            TensorData::arange(i64::MAX - 1, i64::MAX, 2)
                .unwrap()
                .to_vec_f64(),
            vec![(i64::MAX - 1) as f64]
        );
        assert_eq!(
            TensorData::arange(i64::MIN + 1, i64::MIN, -2)
                .unwrap()
                .to_vec_f64(),
            vec![(i64::MIN + 1) as f64]
        );
    }
}
