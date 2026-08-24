use super::{dtype::DType, scalar::Scalar, shape::Shape, storage::Storage};
use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    shape: Shape,
    storage: Storage,
}

impl TensorData {
    pub fn new(shape: impl Into<Shape>, values: Vec<f32>) -> Result<Self> {
        Self::from_storage(shape, Storage::F32(values))
    }

    pub fn from_storage(shape: impl Into<Shape>, storage: Storage) -> Result<Self> {
        let shape = shape.into();
        let expected = shape.numel()?;
        if storage.len() != expected {
            return Err(Error::InvalidData {
                shape,
                expected,
                actual: storage.len(),
            });
        }
        Ok(Self { shape, storage })
    }

    pub fn from_scalars(
        shape: impl Into<Shape>,
        dtype: DType,
        values: impl IntoIterator<Item = Scalar>,
    ) -> Result<Self> {
        Self::from_storage(shape, Storage::from_scalars(dtype, values))
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            shape: Shape::new(Vec::new()),
            storage: Storage::F32(vec![value]),
        }
    }

    pub fn scalar_with_dtype(value: Scalar, dtype: DType) -> Self {
        Self {
            shape: Shape::new(Vec::new()),
            storage: Storage::from_scalars(dtype, [value]),
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn scalar_at(&self, index: usize) -> Scalar {
        self.storage.scalar(index)
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub(crate) fn resize_exact_splat(&self, shape: Shape) -> Result<Self> {
        let len = shape.numel()?;
        let storage = self
            .storage
            .repeat_exact_splat(len)
            .ok_or(Error::InvalidIndex)?;
        Self::from_storage(shape, storage)
    }

    pub fn values(&self) -> &[f32] {
        match &self.storage {
            Storage::F32(values) => values,
            _ => panic!("values() is only available for f32 TensorData; use scalar_at or storage"),
        }
    }

    pub fn cast(&self, dtype: DType) -> Self {
        let storage = match (&self.storage, dtype) {
            // Keep the source f32 payload in its original 32-bit form.  The
            // generic Scalar path widens through f64, which quiets signaling
            // NaNs on supported hosts before BF16 conversion can inspect the
            // original payload.
            (Storage::F32(values), DType::BF16) => Storage::BF16(
                values
                    .iter()
                    .map(|value| super::scalar::f32_to_bf16(*value))
                    .collect(),
            ),
            _ => Storage::from_scalars(dtype, (0..self.len()).map(|i| self.scalar_at(i))),
        };
        Self {
            shape: self.shape.clone(),
            storage,
        }
    }

    pub fn to_vec_f64(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| self.scalar_at(i).as_f64())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_preserves_integer_and_bool_values() {
        let x = TensorData::from_scalars(
            [3],
            DType::U64,
            [Scalar::U(u64::MAX), Scalar::U(1), Scalar::U(0)],
        )
        .unwrap();
        assert_eq!(x.storage(), &Storage::U64(vec![u64::MAX, 1, 0]));
        assert_eq!(
            x.cast(DType::Bool).storage(),
            &Storage::Bool(vec![true, true, false])
        );
    }

    #[test]
    fn casts_are_deterministic_and_half_storage_is_lossless() {
        let x = TensorData::from_scalars(
            [3],
            DType::F64,
            [Scalar::F(-1.9), Scalar::F(300.0), Scalar::F(f64::NAN)],
        )
        .unwrap();
        assert_eq!(x.cast(DType::U8).storage(), &Storage::U8(vec![0, 255, 0]));
        let half = TensorData::from_scalars([1], DType::F16, [Scalar::F(1.5)]).unwrap();
        assert_eq!(half.storage(), &Storage::F16(vec![0x3e00]));
        assert_eq!(half.to_vec_f64(), vec![1.5]);
    }

    #[test]
    fn f32_to_bf16_cast_preserves_adversarial_nan_payloads() {
        let bits = [
            0x0000_0000u32,
            0x8000_0000,
            0x0000_0001,
            0x007f_ffff,
            0x3f80_8000,
            0x3f81_8000,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0x7fff_ffff,
            0xff80_0001,
            0xffff_ffff,
        ];
        let bytes = bits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let input = TensorData::from_le_bytes([12], DType::F32, &bytes).unwrap();
        assert_eq!(
            input.cast(DType::BF16).storage(),
            &Storage::BF16(vec![
                0x0000, 0x8000, 0x0000, 0x0080, 0x3f80, 0x3f82, 0x7f80, 0xff80, 0x7f81, 0x7fff,
                0xff81, 0xffff,
            ])
        );
    }
}
