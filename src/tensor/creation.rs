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
            value = value
                .checked_add(step)
                .ok_or(Error::InvalidArange { start, end, step })?;
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
    }
}
