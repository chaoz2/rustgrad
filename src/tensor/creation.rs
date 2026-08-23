use super::{Shape, TensorData};
use crate::{Error, Result};

impl TensorData {
    /// Creates a dense tensor whose elements are all `value`.
    pub fn full(shape: impl Into<Shape>, value: f32) -> Result<Self> {
        let shape = shape.into();
        Self::new(shape.clone(), vec![value; shape.numel()?])
    }

    pub fn zeros(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, 0.0)
    }

    pub fn ones(shape: impl Into<Shape>) -> Result<Self> {
        Self::full(shape, 1.0)
    }

    /// Integer arange with tinygrad/NumPy-style half-open bounds.
    ///
    /// Integer parameters keep length calculation deterministic while the
    /// current storage oracle is f32-only. General dtype support will lift
    /// this restriction without changing the public tensor/compiler layers.
    pub fn arange(start: i64, end: i64, step: i64) -> Result<Self> {
        if step == 0 {
            return Err(Error::InvalidArange { start, end, step });
        }

        let mut values = Vec::new();
        let mut value = start;
        while (step > 0 && value < end) || (step < 0 && value > end) {
            values.push(value as f32);
            value = value
                .checked_add(step)
                .ok_or(Error::InvalidArange { start, end, step })?;
        }
        Self::new([values.len()], values)
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
            TensorData::new([4], vec![5.0, 3.0, 1.0, -1.0]).unwrap()
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
