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
        // tinygrad treats a same-dtype cast as an identity. Retaining the
        // storage avoids quieting or otherwise rewriting a floating NaN
        // payload before a later fused consumer sees it.
        if self.dtype() == dtype {
            return self.clone();
        }
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

    /// Replaces this dense value from a same-dtype broadcast source.
    ///
    /// The source offsets are computed before storage is replaced, giving this
    /// CPU reference operation read-before-write snapshot semantics.  It owns
    /// no aliases; effectful graph/subbuffer lowering must use `EffectPlan`.
    pub fn assign_from(&mut self, source: &TensorData) -> Result<()> {
        if self.dtype() != source.dtype() {
            return Err(Error::InputDType {
                name: "assignment".into(),
                expected: self.dtype(),
                actual: source.dtype(),
            });
        }
        if source.shape.rank() > self.shape.rank()
            || !source
                .shape
                .dims()
                .iter()
                .rev()
                .zip(self.shape.dims().iter().rev())
                .all(|(source, target)| *source == 1 || source == target)
        {
            return Err(Error::ShapeMismatch {
                op: "assign",
                lhs: self.shape.clone(),
                rhs: source.shape.clone(),
            });
        }
        let offsets = (0..self.len())
            .map(|linear| broadcast_offset(&self.shape, &source.shape, linear))
            .collect::<Result<Vec<_>>>()?;
        // Snapshot all raw lanes before changing the destination. Matching
        // storage variants preserves narrow-float payloads and signed zero.
        self.storage = assigned_storage(&self.storage, &source.storage, &offsets)?;
        Ok(())
    }

    /// Materializes a logical read through the canonical affine descriptor.
    ///
    /// The descriptor is validated against this value's physical shape before
    /// any lane is selected. Selection copies the matching storage variant
    /// directly, preserving integer bits, narrow-float payloads, NaNs, and
    /// signed zero without widening through `Scalar`.
    pub fn affine_read(&self, view: &crate::AffineView) -> Result<Self> {
        if view.source_shape != self.shape {
            return Err(Error::InvalidIndex);
        }
        view.validate_read().map_err(|_| Error::InvalidIndex)?;
        let logical_len = view.logical_shape.numel()?;
        let offsets = (0..logical_len)
            .map(|index| {
                view.element_offset(index)
                    .map_err(|_| Error::InvalidIndex)
                    .and_then(|offset| usize::try_from(offset).map_err(|_| Error::InvalidIndex))
            })
            .collect::<Result<Vec<_>>>()?;
        let storage = assigned_storage(&self.storage, &self.storage, &offsets)?;
        Self::from_storage(view.logical_shape.clone(), storage)
    }

    /// Replaces only an injective affine logical region while preserving every
    /// untouched raw storage lane. This is the CPU oracle for effect views.
    pub(crate) fn assign_view_from(
        &mut self,
        view: &crate::AffineView,
        source: &TensorData,
    ) -> Result<()> {
        if view.source_shape != self.shape
            || view.logical_shape != *source.shape()
            || self.dtype() != source.dtype()
        {
            return Err(Error::InvalidIndex);
        }
        let offsets = (0..source.len())
            .map(|index| {
                view.element_offset(index)
                    .map_err(|_| Error::InvalidIndex)
                    .and_then(|offset| usize::try_from(offset).map_err(|_| Error::InvalidIndex))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut unique = std::collections::BTreeSet::new();
        if offsets.iter().any(|offset| !unique.insert(*offset)) {
            return Err(Error::InvalidIndex);
        }
        macro_rules! splice {
            ($base:ident, $source:ident, $variant:ident) => {{
                let mut result = $base.clone();
                for (destination, value) in offsets.iter().zip($source.iter()) {
                    result[*destination] = value.clone();
                }
                Storage::$variant(result)
            }};
        }
        self.storage = match (&self.storage, source.storage()) {
            (Storage::Bool(base), Storage::Bool(values)) => splice!(base, values, Bool),
            (Storage::I8(base), Storage::I8(values)) => splice!(base, values, I8),
            (Storage::U8(base), Storage::U8(values)) => splice!(base, values, U8),
            (Storage::I16(base), Storage::I16(values)) => splice!(base, values, I16),
            (Storage::U16(base), Storage::U16(values)) => splice!(base, values, U16),
            (Storage::I32(base), Storage::I32(values)) => splice!(base, values, I32),
            (Storage::U32(base), Storage::U32(values)) => splice!(base, values, U32),
            (Storage::I64(base), Storage::I64(values)) => splice!(base, values, I64),
            (Storage::U64(base), Storage::U64(values)) => splice!(base, values, U64),
            (Storage::F16(base), Storage::F16(values)) => splice!(base, values, F16),
            (Storage::BF16(base), Storage::BF16(values)) => splice!(base, values, BF16),
            (Storage::F32(base), Storage::F32(values)) => splice!(base, values, F32),
            (Storage::F64(base), Storage::F64(values)) => splice!(base, values, F64),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(())
    }

    /// Applies a normalized static replacement plan from an immutable source
    /// snapshot. Both the CPU graph oracle and the persistent effect runtime
    /// use this exact raw-storage path, including row-major last-writer-wins
    /// duplicate indices and RHS broadcasting.
    pub(crate) fn static_index_update_from(
        &mut self,
        plan: &crate::ir::indexing::StaticIndexPlan,
        source: &TensorData,
    ) -> Result<()> {
        if self.shape != *plan.source_shape() || self.dtype() != source.dtype() {
            return Err(Error::InvalidIndex);
        }
        if source.shape.rank() > plan.output_shape().rank()
            || source
                .shape
                .dims()
                .iter()
                .rev()
                .zip(plan.output_shape().dims().iter().rev())
                .any(|(source, target)| *source != 1 && source != target)
        {
            return Err(Error::InvalidIndex);
        }
        let targets = plan.source_offsets()?;
        let source_offsets = (0..targets.len())
            .map(|linear| broadcast_offset(plan.output_shape(), source.shape(), linear))
            .collect::<Result<Vec<_>>>()?;
        macro_rules! splice {
            ($base:ident, $values:ident, $variant:ident) => {{
                let mut result = $base.clone();
                for (target, value) in targets.iter().zip(source_offsets.iter()) {
                    result[*target] = $values[*value].clone();
                }
                Storage::$variant(result)
            }};
        }
        self.storage = match (&self.storage, source.storage()) {
            (Storage::Bool(base), Storage::Bool(values)) => splice!(base, values, Bool),
            (Storage::I8(base), Storage::I8(values)) => splice!(base, values, I8),
            (Storage::U8(base), Storage::U8(values)) => splice!(base, values, U8),
            (Storage::I16(base), Storage::I16(values)) => splice!(base, values, I16),
            (Storage::U16(base), Storage::U16(values)) => splice!(base, values, U16),
            (Storage::I32(base), Storage::I32(values)) => splice!(base, values, I32),
            (Storage::U32(base), Storage::U32(values)) => splice!(base, values, U32),
            (Storage::I64(base), Storage::I64(values)) => splice!(base, values, I64),
            (Storage::U64(base), Storage::U64(values)) => splice!(base, values, U64),
            (Storage::F16(base), Storage::F16(values)) => splice!(base, values, F16),
            (Storage::BF16(base), Storage::BF16(values)) => splice!(base, values, BF16),
            (Storage::F32(base), Storage::F32(values)) => splice!(base, values, F32),
            (Storage::F64(base), Storage::F64(values)) => splice!(base, values, F64),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(())
    }
}

fn broadcast_offset(target: &Shape, source: &Shape, mut linear: usize) -> Result<usize> {
    let mut coordinates = vec![0; target.rank()];
    for axis in (0..target.rank()).rev() {
        let dim = target.dims()[axis];
        if dim != 0 {
            coordinates[axis] = linear % dim;
            linear /= dim;
        }
    }
    let pad = target.rank() - source.rank();
    let mut offset = 0usize;
    for (axis, dim) in source.dims().iter().enumerate() {
        let coordinate = if *dim == 1 {
            0
        } else {
            coordinates[pad + axis]
        };
        offset = offset
            .checked_mul(*dim)
            .and_then(|value| value.checked_add(coordinate))
            .ok_or(Error::InvalidIndex)?;
    }
    Ok(offset)
}

fn assigned_storage(destination: &Storage, source: &Storage, offsets: &[usize]) -> Result<Storage> {
    macro_rules! copy {
        ($b:ident, $variant:ident) => {
            Ok(Storage::$variant(
                offsets.iter().map(|offset| $b[*offset].clone()).collect(),
            ))
        };
    }
    match (destination, source) {
        (Storage::Bool(_), Storage::Bool(values)) => copy!(values, Bool),
        (Storage::I8(_), Storage::I8(values)) => copy!(values, I8),
        (Storage::U8(_), Storage::U8(values)) => copy!(values, U8),
        (Storage::I16(_), Storage::I16(values)) => copy!(values, I16),
        (Storage::U16(_), Storage::U16(values)) => copy!(values, U16),
        (Storage::I32(_), Storage::I32(values)) => copy!(values, I32),
        (Storage::U32(_), Storage::U32(values)) => copy!(values, U32),
        (Storage::I64(_), Storage::I64(values)) => copy!(values, I64),
        (Storage::U64(_), Storage::U64(values)) => copy!(values, U64),
        (Storage::F16(_), Storage::F16(values)) => copy!(values, F16),
        (Storage::BF16(_), Storage::BF16(values)) => copy!(values, BF16),
        (Storage::F32(_), Storage::F32(values)) => copy!(values, F32),
        (Storage::F64(_), Storage::F64(values)) => copy!(values, F64),
        _ => Err(Error::InvalidIndex),
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

    #[test]
    fn same_dtype_float_cast_is_a_raw_storage_identity() {
        let input = TensorData::from_storage(
            [2],
            Storage::F32(vec![f32::from_bits(0x7f80_0001), f32::from_bits(0x8000_0000)]),
        )
        .unwrap();
        let Storage::F32(values) = input.cast(DType::F32).storage() else {
            panic!("same dtype cast changed F32 storage");
        };
        assert_eq!(
            values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            vec![0x7f80_0001, 0x8000_0000]
        );
    }

    #[test]
    fn dense_assignment_broadcasts_exact_raw_storage_and_is_transactional() {
        let mut dst = TensorData::from_storage([2, 3], Storage::U64(vec![0; 6])).unwrap();
        let src = TensorData::from_storage([1, 3], Storage::U64(vec![u64::MAX, 2, 3])).unwrap();
        dst.assign_from(&src).unwrap();
        assert_eq!(
            dst.storage(),
            &Storage::U64(vec![u64::MAX, 2, 3, u64::MAX, 2, 3])
        );
        let old = dst.clone();
        let wrong = TensorData::from_storage([2], Storage::I32(vec![1, 2])).unwrap();
        assert!(dst.assign_from(&wrong).is_err());
        assert_eq!(dst, old);
        let mut half = TensorData::from_storage([2], Storage::F16(vec![0, 0])).unwrap();
        half.assign_from(&TensorData::from_storage([1], Storage::F16(vec![0x7e01])).unwrap())
            .unwrap();
        assert_eq!(half.storage(), &Storage::F16(vec![0x7e01; 2]));
    }

    #[test]
    fn affine_read_preserves_raw_storage_for_signed_and_broadcast_maps() {
        let cases = [
            (
                DType::Bool,
                Storage::Bool(vec![true, false, true, false]),
                Storage::Bool(vec![false, true, false, true]),
            ),
            (
                DType::U64,
                Storage::U64(vec![0, u64::MAX, 7, 9]),
                Storage::U64(vec![9, 7, u64::MAX, 0]),
            ),
            (
                DType::F16,
                Storage::F16(vec![0x8000, 0x7e01, 0x3c00, 0xfc00]),
                Storage::F16(vec![0xfc00, 0x3c00, 0x7e01, 0x8000]),
            ),
            (
                DType::BF16,
                Storage::BF16(vec![0x8000, 0x7fc1, 0x3f80, 0xff80]),
                Storage::BF16(vec![0xff80, 0x3f80, 0x7fc1, 0x8000]),
            ),
            (
                DType::F32,
                Storage::F32(vec![0.0, f32::from_bits(0x7fc0_0001), -1.0, -0.0]),
                Storage::F32(vec![-0.0, -1.0, f32::from_bits(0x7fc0_0001), 0.0]),
            ),
            (
                DType::F64,
                Storage::F64(vec![0.0, f64::from_bits(0x7ff8_0000_0000_0001), -1.0, -0.0]),
                Storage::F64(vec![-0.0, -1.0, f64::from_bits(0x7ff8_0000_0000_0001), 0.0]),
            ),
        ];
        for (dtype, storage, expected) in cases {
            assert_eq!(storage.dtype(), dtype);
            let data = TensorData::from_storage([4], storage).unwrap();
            let flip = crate::AffineView::identity(Shape::from([4]))
                .flip(0)
                .unwrap();
            assert_eq!(
                data.affine_read(&flip).unwrap().to_le_bytes().unwrap(),
                TensorData::from_storage([4], expected)
                    .unwrap()
                    .to_le_bytes()
                    .unwrap(),
                "{dtype:?}"
            );
        }

        let scalar = TensorData::from_storage([], Storage::U64(vec![u64::MAX])).unwrap();
        let broadcast = crate::AffineView {
            source_shape: Shape::new([]),
            logical_shape: Shape::from([2, 3]),
            strides: vec![0, 0],
            offset: 0,
        };
        assert_eq!(
            scalar.affine_read(&broadcast).unwrap().storage(),
            &Storage::U64(vec![u64::MAX; 6])
        );
        let empty = crate::AffineView {
            source_shape: Shape::from([4]),
            logical_shape: Shape::from([0]),
            strides: vec![1],
            offset: 4,
        };
        assert!(data_for_empty().affine_read(&empty).unwrap().is_empty());
    }

    fn data_for_empty() -> TensorData {
        TensorData::from_storage([4], Storage::I32(vec![1, 2, 3, 4])).unwrap()
    }
}
