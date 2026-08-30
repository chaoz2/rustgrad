use super::{dtype::DType, scalar::Scalar, shape::Shape, storage::Storage};
use crate::{Error, Result};
use std::io::{self, Read, Seek, SeekFrom};

/// A recursive, row-major host representation for [`TensorData::tolist`].
///
/// Leaves retain RustGrad's typed [`Scalar`] boundary values. In particular,
/// half and bfloat16 storage is converted through its existing F32 decoding
/// path before becoming a floating scalar, matching tinygrad's Python-list
/// conversion rather than exposing raw storage bits.
#[derive(Clone, Debug)]
pub enum TensorList {
    Scalar(Scalar),
    List(Vec<TensorList>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    shape: Shape,
    storage: Storage,
}

/// A borrowed, read-only byte stream over a rank-one U8 [`TensorData`].
///
/// This mirrors tinygrad's `nn.state.TensorIO` at RustGrad's realized-value
/// boundary. Reads copy only the requested interval into the caller's buffer;
/// the adapter never exposes a storage view, realizes a graph value, or
/// supports writes.
#[derive(Debug)]
pub struct TensorDataReader<'a> {
    data: &'a TensorData,
    position: usize,
}

impl<'a> TensorDataReader<'a> {
    /// Creates a read-only TensorIO-compatible stream.
    ///
    /// The rank check intentionally precedes dtype validation, matching the
    /// left-to-right admission of tinygrad's `t.ndim != 1 or t.dtype != uint8`.
    pub fn new(data: &'a TensorData) -> Result<Self> {
        if data.shape.rank() != 1 {
            return Err(Error::InvalidTensorIo {
                reason: "TensorIO requires a rank-one TensorData",
            });
        }
        if data.dtype() != DType::U8 {
            return Err(Error::InvalidTensorIo {
                reason: "TensorIO requires U8 storage",
            });
        }
        Ok(Self { data, position: 0 })
    }

    /// Returns the clamped byte position maintained by this reader.
    pub fn position(&self) -> usize {
        self.position
    }

    fn source_bytes(&self) -> &[u8] {
        let Storage::U8(bytes) = self.data.storage() else {
            unreachable!("TensorDataReader validates U8 storage at construction")
        };
        bytes
    }

    fn seek_position(&self, offset: i128, origin: i128) -> usize {
        let end = self.source_bytes().len() as i128;
        (origin.saturating_add(offset).clamp(0, end)) as usize
    }
}

impl Read for TensorDataReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let position = self.position;
        let count = {
            let bytes = self.source_bytes();
            let count = bytes.len().saturating_sub(position).min(buffer.len());
            buffer[..count].copy_from_slice(&bytes[position..position + count]);
            count
        };
        self.position = position + count;
        Ok(count)
    }
}

impl Seek for TensorDataReader<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.position = match position {
            SeekFrom::Start(offset) => self.seek_position(i128::from(offset), 0),
            SeekFrom::Current(offset) => {
                self.seek_position(i128::from(offset), self.position as i128)
            }
            SeekFrom::End(offset) => {
                self.seek_position(i128::from(offset), self.source_bytes().len() as i128)
            }
        };
        Ok(self.position as u64)
    }
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

    /// Returns a borrowed, read-only TensorIO-compatible stream for this
    /// realized rank-one U8 value.
    pub fn byte_reader(&self) -> Result<TensorDataReader<'_>> {
        TensorDataReader::new(self)
    }

    pub fn scalar_at(&self, index: usize) -> Scalar {
        self.storage.scalar(index)
    }

    /// Returns the sole stored value as a typed scalar.
    ///
    /// Like tinygrad's `Tensor.item`, rank is irrelevant: scalar tensors and
    /// singleton tensors of any rank are accepted, while zero- and
    /// multi-element tensors fail without inspecting or modifying storage.
    pub fn item(&self) -> Result<Scalar> {
        if self.len() != 1 {
            return Err(Error::NonScalarItem(self.shape.clone()));
        }
        Ok(self.scalar_at(0))
    }

    /// Replaces this realized value's owned storage with an exact source
    /// clone. Shapes must match exactly, but the source storage family may
    /// differ. This is deliberately a dense-value operation: it does not
    /// represent a Graph/device/effectful replacement boundary.
    pub fn replace(&mut self, source: &TensorData) -> Result<&mut Self> {
        if self.shape != source.shape {
            return Err(Error::ShapeMismatch {
                op: "replace",
                lhs: self.shape.clone(),
                rhs: source.shape.clone(),
            });
        }
        self.storage = source.storage.clone();
        Ok(self)
    }

    /// Returns this already-realized dense value as tinygrad-style nested
    /// Python-list data. Rank zero is a single typed scalar leaf; every other
    /// rank nests one list per concrete shape dimension in row-major order.
    ///
    /// This is a read-only storage conversion. It neither realizes a graph
    /// value nor changes this value's shape, dtype, or backing storage.
    pub fn tolist(&self) -> TensorList {
        fn build<I: Iterator<Item = Scalar>>(dims: &[usize], values: &mut I) -> TensorList {
            match dims.split_first() {
                None => TensorList::Scalar(
                    values
                        .next()
                        .expect("TensorData shape/storage invariant validated at construction"),
                ),
                Some((&extent, tail)) => {
                    TensorList::List((0..extent).map(|_| build(tail, values)).collect())
                }
            }
        }

        build(
            self.shape.dims(),
            &mut (0..self.len()).map(|index| self.scalar_at(index)),
        )
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

    /// Reinterprets each canonical little-endian element payload without any
    /// numeric conversion. The source and target item sizes must agree.
    pub fn bitcast(&self, dtype: DType) -> Result<Self> {
        if self.dtype().itemsize() != dtype.itemsize() {
            return Err(Error::BitcastItemsize {
                from: self.dtype(),
                to: dtype,
            });
        }
        Self::from_le_bytes(self.shape.clone(), dtype, &self.to_le_bytes()?)
    }

    /// Internal raw-byte reinterpretation used by the typed Graph bitcast
    /// materialization contract. The source and destination must describe the
    /// same total byte extent. Bool storage is canonicalized from nonzero raw
    /// bytes because Rust's owned Bool lanes cannot retain noncanonical bytes.
    pub(crate) fn bitcast_with_shape(&self, shape: Shape, dtype: DType) -> Result<Self> {
        let source_bytes = self
            .len()
            .checked_mul(self.dtype().itemsize())
            .ok_or_else(|| Error::ShapeOverflow(self.shape.clone()))?;
        let output_bytes = shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if source_bytes != output_bytes {
            return Err(Error::InvalidBitcast {
                from: self.dtype(),
                to: dtype,
                shape,
            });
        }
        let mut bytes = self.to_le_bytes()?;
        if dtype == DType::Bool {
            bytes
                .iter_mut()
                .for_each(|byte| *byte = u8::from(*byte != 0));
        }
        Self::from_le_bytes(shape, dtype, &bytes)
    }

    pub fn to_vec_f64(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| self.scalar_at(i).as_f64())
            .collect()
    }

    /// Reorders dense lanes without decoding their storage representation.
    /// Movement and immutable indexing use this path for exact float8 payload
    /// transport (and retain the same guarantee for every other raw dtype).
    pub(crate) fn reorder_raw(&self, shape: Shape, offsets: &[usize]) -> Result<Self> {
        if shape.numel()? != offsets.len() || offsets.iter().any(|offset| *offset >= self.len()) {
            return Err(Error::InvalidIndex);
        }
        Ok(Self {
            shape,
            storage: selected_storage(&self.storage, offsets)?,
        })
    }

    /// Copies checked source lanes or one already-committed scalar fill into
    /// a dense output without decoding either storage representation.
    ///
    /// Padding uses this path so copied Float8 and floating NaN payloads are
    /// byte exact. The caller owns geometry construction; `None` selects the
    /// scalar fill and `Some(offset)` selects one source lane.
    pub(crate) fn pad_raw_offsets(
        &self,
        shape: Shape,
        offsets: &[Option<usize>],
        fill: &TensorData,
    ) -> Result<Self> {
        if shape.numel()? != offsets.len()
            || fill.shape.rank() != 0
            || fill.len() != 1
            || fill.dtype() != self.dtype()
            || offsets.iter().flatten().any(|offset| *offset >= self.len())
        {
            return Err(Error::InvalidIndex);
        }

        macro_rules! padded {
            ($source:ident, $fill:ident, $variant:ident) => {
                Storage::$variant(
                    offsets
                        .iter()
                        .map(|offset| offset.map_or($fill[0], |offset| $source[offset]))
                        .collect(),
                )
            };
        }
        let storage = match (&self.storage, fill.storage()) {
            (Storage::Bool(source), Storage::Bool(fill)) => padded!(source, fill, Bool),
            (Storage::I8(source), Storage::I8(fill)) => padded!(source, fill, I8),
            (Storage::U8(source), Storage::U8(fill)) => padded!(source, fill, U8),
            (Storage::I16(source), Storage::I16(fill)) => padded!(source, fill, I16),
            (Storage::U16(source), Storage::U16(fill)) => padded!(source, fill, U16),
            (Storage::I32(source), Storage::I32(fill)) => padded!(source, fill, I32),
            (Storage::U32(source), Storage::U32(fill)) => padded!(source, fill, U32),
            (Storage::I64(source), Storage::I64(fill)) => padded!(source, fill, I64),
            (Storage::U64(source), Storage::U64(fill)) => padded!(source, fill, U64),
            (Storage::Float8(source), Storage::Float8(fill))
                if source.format() == fill.format() =>
            {
                Storage::Float8(super::float8::Float8Storage::from_raw(
                    source.format(),
                    offsets
                        .iter()
                        .map(|offset| {
                            offset.map_or(fill.as_raw()[0], |offset| source.as_raw()[offset])
                        })
                        .collect(),
                ))
            }
            (Storage::F16(source), Storage::F16(fill)) => padded!(source, fill, F16),
            (Storage::BF16(source), Storage::BF16(fill)) => padded!(source, fill, BF16),
            (Storage::F32(source), Storage::F32(fill)) => padded!(source, fill, F32),
            (Storage::F64(source), Storage::F64(fill)) => padded!(source, fill, F64),
            _ => return Err(Error::InvalidIndex),
        };
        Self::from_storage(shape, storage)
    }

    /// Replaces checked destination lanes from checked source lanes without
    /// decoding storage. Row-major callers provide duplicate destinations in
    /// write order, so the final source wins deterministically.
    pub(crate) fn replace_raw_offsets(
        &mut self,
        source: &TensorData,
        destinations: &[usize],
        sources: &[usize],
    ) -> Result<()> {
        if self.dtype() != source.dtype()
            || destinations.len() != sources.len()
            || destinations.iter().any(|offset| *offset >= self.len())
            || sources.iter().any(|offset| *offset >= source.len())
        {
            return Err(Error::InvalidIndex);
        }
        macro_rules! splice {
            ($base:ident, $values:ident, $variant:ident) => {{
                let mut result = $base.clone();
                for (destination, source) in destinations.iter().zip(sources) {
                    result[*destination] = $values[*source];
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
            (Storage::Float8(base), Storage::Float8(values))
                if base.format() == values.format() =>
            {
                let mut result = base.as_raw().to_vec();
                for (destination, source) in destinations.iter().zip(sources) {
                    result[*destination] = values.as_raw()[*source];
                }
                Storage::Float8(super::float8::Float8Storage::from_raw(
                    base.format(),
                    result,
                ))
            }
            (Storage::F16(base), Storage::F16(values)) => splice!(base, values, F16),
            (Storage::BF16(base), Storage::BF16(values)) => splice!(base, values, BF16),
            (Storage::F32(base), Storage::F32(values)) => splice!(base, values, F32),
            (Storage::F64(base), Storage::F64(values)) => splice!(base, values, F64),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(())
    }

    /// Replaces this dense value from a same-dtype broadcast source.
    ///
    /// The source offsets are computed before storage is replaced, giving this
    /// CPU reference operation read-before-write snapshot semantics.  It owns
    /// no aliases; effectful graph/subbuffer lowering must use `EffectPlan`.
    pub fn assign_from(&mut self, source: &TensorData) -> Result<()> {
        // tinygrad resolves `_broadcast_to` before checking storage dtype, so
        // a source which is both wrongly shaped and wrongly typed reports the
        // shape failure without touching either dense value.
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
        if self.dtype() != source.dtype() {
            return Err(Error::InputDType {
                name: "assignment".into(),
                expected: self.dtype(),
                actual: source.dtype(),
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

    /// Source-shaped wrapper for dense realized assignment. This preserves
    /// `assign_from` for existing effect/runtime callers while returning the
    /// exact mutated receiver for direct TensorData use.
    pub fn assign(&mut self, source: &TensorData) -> Result<&mut Self> {
        self.assign_from(source)?;
        Ok(self)
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
            (Storage::Float8(base), Storage::Float8(values))
                if base.format() == values.format() =>
            {
                let mut result = base.as_raw().to_vec();
                for (destination, value) in offsets.iter().zip(values.as_raw()) {
                    result[*destination] = *value;
                }
                Storage::Float8(super::float8::Float8Storage::from_raw(
                    base.format(),
                    result,
                ))
            }
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
            (Storage::Float8(base), Storage::Float8(values))
                if base.format() == values.format() =>
            {
                let mut result = base.as_raw().to_vec();
                for (target, value) in targets.iter().zip(source_offsets.iter()) {
                    result[*target] = values.as_raw()[*value];
                }
                Storage::Float8(super::float8::Float8Storage::from_raw(
                    base.format(),
                    result,
                ))
            }
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
        (Storage::Float8(destination), Storage::Float8(source))
            if destination.format() == source.format() =>
        {
            Ok(Storage::Float8(super::float8::Float8Storage::from_raw(
                destination.format(),
                offsets
                    .iter()
                    .map(|offset| source.as_raw()[*offset])
                    .collect(),
            )))
        }
        (Storage::F16(_), Storage::F16(values)) => copy!(values, F16),
        (Storage::BF16(_), Storage::BF16(values)) => copy!(values, BF16),
        (Storage::F32(_), Storage::F32(values)) => copy!(values, F32),
        (Storage::F64(_), Storage::F64(values)) => copy!(values, F64),
        _ => Err(Error::InvalidIndex),
    }
}

fn selected_storage(storage: &Storage, offsets: &[usize]) -> Result<Storage> {
    assigned_storage(storage, storage, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    fn leaf(list: &TensorList) -> Scalar {
        let TensorList::Scalar(value) = list else {
            panic!("expected a scalar TensorList leaf");
        };
        *value
    }

    fn assert_same_scalar(actual: Scalar, expected: Scalar, dtype: DType) {
        match (actual, expected) {
            (Scalar::Bool(actual), Scalar::Bool(expected)) => {
                assert_eq!(actual, expected, "{dtype:?}")
            }
            (Scalar::I(actual), Scalar::I(expected)) => assert_eq!(actual, expected, "{dtype:?}"),
            (Scalar::U(actual), Scalar::U(expected)) => assert_eq!(actual, expected, "{dtype:?}"),
            (Scalar::F(actual), Scalar::F(expected)) => {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{dtype:?}")
            }
            (actual, expected) => {
                panic!("tolist changed scalar kind for {dtype:?}: {actual:?} != {expected:?}")
            }
        }
    }

    fn assert_same_storage_bits(actual: &Storage, expected: &Storage) {
        match (actual, expected) {
            (Storage::F32(actual), Storage::F32(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
            (Storage::F64(actual), Storage::F64(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn tensor_data_reader_reads_requested_ranges_and_read_to_end_without_mutation() {
        let data = TensorData::from_storage([5], Storage::U8(vec![3, 1, 4, 1, 5])).unwrap();
        let before = data.clone();
        let mut reader = data.byte_reader().unwrap();

        let mut prefix = [0u8; 3];
        assert_eq!(reader.read(&mut prefix).unwrap(), 3);
        assert_eq!(prefix, [3, 1, 4]);
        assert_eq!(reader.position(), 3);

        let mut suffix = Vec::new();
        assert_eq!(reader.read_to_end(&mut suffix).unwrap(), 2);
        assert_eq!(suffix, vec![1, 5]);
        assert_eq!(reader.read(&mut prefix).unwrap(), 0);
        assert_eq!(data, before);
    }

    #[test]
    fn tensor_data_reader_seek_clamps_all_origins_and_handles_empty_input() {
        let data = TensorData::from_storage([4], Storage::U8(vec![10, 20, 30, 40])).unwrap();
        let mut reader = TensorDataReader::new(&data).unwrap();
        let mut bytes = [0u8; 2];

        assert_eq!(reader.seek(SeekFrom::Start(2)).unwrap(), 2);
        assert_eq!(reader.read(&mut bytes).unwrap(), 2);
        assert_eq!(bytes, [30, 40]);
        assert_eq!(reader.seek(SeekFrom::Current(-99)).unwrap(), 0);
        assert_eq!(reader.read(&mut bytes).unwrap(), 2);
        assert_eq!(bytes, [10, 20]);
        assert_eq!(reader.seek(SeekFrom::End(-1)).unwrap(), 3);
        assert_eq!(reader.read(&mut bytes).unwrap(), 1);
        assert_eq!(bytes[0], 40);
        assert_eq!(reader.seek(SeekFrom::Start(u64::MAX)).unwrap(), 4);
        assert_eq!(reader.read(&mut bytes).unwrap(), 0);

        let empty = TensorData::from_storage([0], Storage::U8(vec![])).unwrap();
        let mut empty_reader = empty.byte_reader().unwrap();
        let mut end = Vec::new();
        assert_eq!(empty_reader.read_to_end(&mut end).unwrap(), 0);
        assert!(end.is_empty());
        assert_eq!(empty_reader.seek(SeekFrom::End(-1)).unwrap(), 0);
    }

    #[test]
    fn tensor_data_reader_validates_rank_before_dtype() {
        let rank_and_dtype = TensorData::from_storage([1, 1], Storage::F32(vec![1.0])).unwrap();
        assert_eq!(
            rank_and_dtype.byte_reader().unwrap_err(),
            Error::InvalidTensorIo {
                reason: "TensorIO requires a rank-one TensorData",
            }
        );

        let wrong_dtype = TensorData::from_storage([1], Storage::F32(vec![1.0])).unwrap();
        assert_eq!(
            TensorDataReader::new(&wrong_dtype).unwrap_err(),
            Error::InvalidTensorIo {
                reason: "TensorIO requires U8 storage",
            }
        );

        let scalar = TensorData::from_storage([], Storage::U8(vec![7])).unwrap();
        assert!(matches!(
            scalar.byte_reader(),
            Err(Error::InvalidTensorIo {
                reason: "TensorIO requires a rank-one TensorData"
            })
        ));
    }

    #[test]
    fn tolist_preserves_rank_zero_row_major_nesting_and_storage() {
        let scalar = TensorData::scalar_with_dtype(Scalar::I(-7), DType::I32);
        assert_same_scalar(leaf(&scalar.tolist()), Scalar::I(-7), DType::I32);

        let one_dim = TensorData::from_scalars(
            [3],
            DType::I32,
            [Scalar::I(10), Scalar::I(20), Scalar::I(30)],
        )
        .unwrap();
        let TensorList::List(values) = one_dim.tolist() else {
            panic!("rank-one TensorData must become a list");
        };
        assert_same_scalar(leaf(&values[0]), Scalar::I(10), DType::I32);
        assert_same_scalar(leaf(&values[1]), Scalar::I(20), DType::I32);
        assert_same_scalar(leaf(&values[2]), Scalar::I(30), DType::I32);

        let two_dim = TensorData::from_scalars(
            [2, 2],
            DType::I32,
            [Scalar::I(10), Scalar::I(20), Scalar::I(30), Scalar::I(40)],
        )
        .unwrap();
        let TensorList::List(rows) = two_dim.tolist() else {
            panic!("rank-two TensorData must become nested lists");
        };
        let TensorList::List(first_row) = &rows[0] else {
            panic!("rank-two TensorData must contain row lists");
        };
        let TensorList::List(second_row) = &rows[1] else {
            panic!("rank-two TensorData must contain row lists");
        };
        assert_same_scalar(leaf(&first_row[0]), Scalar::I(10), DType::I32);
        assert_same_scalar(leaf(&first_row[1]), Scalar::I(20), DType::I32);
        assert_same_scalar(leaf(&second_row[0]), Scalar::I(30), DType::I32);
        assert_same_scalar(leaf(&second_row[1]), Scalar::I(40), DType::I32);

        let data = TensorData::from_scalars([2, 2, 2], DType::I32, (0..8).map(Scalar::I)).unwrap();
        let before = data.clone();
        let TensorList::List(outer) = data.tolist() else {
            panic!("rank-three TensorData must become nested lists");
        };
        assert_eq!(outer.len(), 2);
        for (outer_index, expected) in [[0, 1, 2, 3], [4, 5, 6, 7]].into_iter().enumerate() {
            let TensorList::List(rows) = &outer[outer_index] else {
                panic!("rank-three outer lane must contain rows");
            };
            assert_eq!(rows.len(), 2);
            for (row_index, values) in expected.chunks_exact(2).enumerate() {
                let TensorList::List(columns) = &rows[row_index] else {
                    panic!("rank-three row must contain columns");
                };
                assert_eq!(columns.len(), 2);
                assert_same_scalar(leaf(&columns[0]), Scalar::I(values[0]), DType::I32);
                assert_same_scalar(leaf(&columns[1]), Scalar::I(values[1]), DType::I32);
            }
        }
        assert_eq!(data, before);
    }

    #[test]
    fn tolist_retains_zero_extent_nesting_and_every_dtype_leaf() {
        let first_zero = TensorData::from_scalars([0, 3], DType::U8, []).unwrap();
        let TensorList::List(values) = first_zero.tolist() else {
            panic!("rank-two empty TensorData must remain a list");
        };
        assert!(values.is_empty());

        let inner_zero = TensorData::from_scalars([2, 0, 3], DType::U8, []).unwrap();
        let TensorList::List(outer) = inner_zero.tolist() else {
            panic!("rank-three TensorData must remain a list");
        };
        assert_eq!(outer.len(), 2);
        for value in outer {
            let TensorList::List(inner) = value else {
                panic!("zero middle extent must preserve the outer dimension");
            };
            assert!(inner.is_empty());
        }

        let cases = [
            (DType::Bool, Scalar::Bool(true)),
            (DType::I8, Scalar::I(-7)),
            (DType::I16, Scalar::I(-300)),
            (DType::I32, Scalar::I(-70_000)),
            (DType::I64, Scalar::I(i64::MIN)),
            (DType::U8, Scalar::U(250)),
            (DType::U16, Scalar::U(60_000)),
            (DType::U32, Scalar::U(4_000_000_000)),
            (DType::U64, Scalar::U(u64::MAX)),
            (DType::F16, Scalar::F(1.5)),
            (DType::BF16, Scalar::F(1.5)),
            (DType::F32, Scalar::F(1.5)),
            (DType::F64, Scalar::F(1.5)),
        ];
        for (dtype, input) in cases {
            let data = TensorData::from_scalars([1], dtype, [input]).unwrap();
            let TensorList::List(values) = data.tolist() else {
                panic!("rank-one TensorData must become a list");
            };
            assert_eq!(values.len(), 1);
            assert_same_scalar(leaf(&values[0]), input, dtype);
        }
    }

    #[test]
    fn tolist_converts_half_and_bfloat16_leaves_without_storage_mutation() {
        for data in [
            TensorData::from_storage([2], Storage::F16(vec![0x8000, 0x7e01])).unwrap(),
            TensorData::from_storage([2], Storage::BF16(vec![0x8000, 0x7fc1])).unwrap(),
        ] {
            let before = data.clone();
            let TensorList::List(values) = data.tolist() else {
                panic!("rank-one half TensorData must become a list");
            };
            assert_eq!(leaf(&values[0]).as_f64().to_bits(), (-0.0f64).to_bits());
            assert!(leaf(&values[1]).as_f64().is_nan());
            assert_eq!(data, before);
        }
    }

    #[test]
    fn replace_changes_storage_family_and_returns_the_same_receiver() {
        let mut destination = TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap();
        let source = TensorData::from_storage([2], Storage::U64(vec![u64::MAX, 7])).unwrap();
        let source_before = source.clone();
        let destination_address: *mut TensorData = &mut destination;

        let returned = destination.replace(&source).unwrap();
        assert_eq!(returned as *mut TensorData, destination_address);
        assert_eq!(returned.dtype(), DType::U64);
        assert_eq!(returned.storage(), source.storage());
        assert_eq!(source, source_before);
    }

    #[test]
    fn replace_copies_raw_float_payloads_and_supports_scalar_and_empty_shapes() {
        for source in [
            TensorData::from_storage([2], Storage::F16(vec![0x8000, 0x7e01])).unwrap(),
            TensorData::from_storage([2], Storage::BF16(vec![0x8000, 0x7fc1])).unwrap(),
            TensorData::from_storage([2], Storage::F32(vec![-0.0, f32::from_bits(0x7f80_0001)]))
                .unwrap(),
            TensorData::from_storage(
                [2],
                Storage::F64(vec![-0.0, f64::from_bits(0x7ff0_0000_0000_0001)]),
            )
            .unwrap(),
        ] {
            let source_before = source.clone();
            let mut destination =
                TensorData::from_storage([2], Storage::Bool(vec![false, true])).unwrap();
            destination.replace(&source).unwrap();
            assert_same_storage_bits(destination.storage(), source.storage());
            assert_same_storage_bits(source.storage(), source_before.storage());
        }

        let scalar_source = TensorData::scalar_with_dtype(Scalar::I(-7), DType::I32);
        let mut scalar_destination =
            TensorData::scalar_with_dtype(Scalar::Bool(false), DType::Bool);
        scalar_destination.replace(&scalar_source).unwrap();
        assert_eq!(scalar_destination.storage(), scalar_source.storage());

        let empty_source = TensorData::from_storage([2, 0, 3], Storage::F16(vec![])).unwrap();
        let mut empty_destination =
            TensorData::from_storage([2, 0, 3], Storage::U8(vec![])).unwrap();
        empty_destination.replace(&empty_source).unwrap();
        assert_eq!(empty_destination.storage(), empty_source.storage());
    }

    #[test]
    fn replace_rejects_shape_mismatch_without_mutating_destination() {
        let mut destination = TensorData::from_storage([2, 1], Storage::I32(vec![1, 2])).unwrap();
        let source = TensorData::from_storage([2], Storage::U64(vec![3, 4])).unwrap();
        let before = destination.clone();

        assert_eq!(
            destination.replace(&source).unwrap_err(),
            Error::ShapeMismatch {
                op: "replace",
                lhs: Shape::new([2, 1]),
                rhs: Shape::new([2]),
            }
        );
        assert_eq!(destination, before);
    }

    #[test]
    fn item_returns_the_only_typed_value_for_every_storage_family() {
        let cases = [
            (DType::Bool, Scalar::Bool(true), Scalar::Bool(true)),
            (DType::I8, Scalar::I(-7), Scalar::I(-7)),
            (DType::I16, Scalar::I(-300), Scalar::I(-300)),
            (DType::I32, Scalar::I(-70_000), Scalar::I(-70_000)),
            (DType::I64, Scalar::I(i64::MIN), Scalar::I(i64::MIN)),
            (DType::U8, Scalar::U(250), Scalar::U(250)),
            (DType::U16, Scalar::U(60_000), Scalar::U(60_000)),
            (
                DType::U32,
                Scalar::U(4_000_000_000),
                Scalar::U(4_000_000_000),
            ),
            (DType::U64, Scalar::U(u64::MAX), Scalar::U(u64::MAX)),
            (DType::F16, Scalar::F(-0.0), Scalar::F(-0.0)),
            (DType::BF16, Scalar::F(-0.0), Scalar::F(-0.0)),
            (DType::F32, Scalar::F(-0.0), Scalar::F(-0.0)),
            (DType::F64, Scalar::F(-0.0), Scalar::F(-0.0)),
        ];
        for (dtype, input, expected) in cases {
            let data = TensorData::from_scalars([1, 1], dtype, [input]).unwrap();
            let actual = data.item().unwrap();
            match (actual, expected) {
                (Scalar::Bool(actual), Scalar::Bool(expected)) => {
                    assert_eq!(actual, expected, "{dtype:?}");
                }
                (Scalar::I(actual), Scalar::I(expected)) => {
                    assert_eq!(actual, expected, "{dtype:?}");
                }
                (Scalar::U(actual), Scalar::U(expected)) => {
                    assert_eq!(actual, expected, "{dtype:?}");
                }
                (Scalar::F(actual), Scalar::F(expected)) => {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{dtype:?}");
                }
                (actual, expected) => {
                    panic!("item changed scalar kind for {dtype:?}: {actual:?} != {expected:?}")
                }
            }
        }

        let scalar = TensorData::scalar_with_dtype(Scalar::I(42), DType::I32);
        assert!(matches!(scalar.item(), Ok(Scalar::I(42))));
    }

    #[test]
    fn item_preserves_float_nan_and_signed_zero_observability() {
        let half_nan = TensorData::from_storage([1], Storage::F16(vec![0x7e01])).unwrap();
        let bf16_nan = TensorData::from_storage([1], Storage::BF16(vec![0x7fc1])).unwrap();
        assert!(half_nan.item().unwrap().as_f64().is_nan());
        assert!(bf16_nan.item().unwrap().as_f64().is_nan());

        for data in [
            TensorData::from_storage([1], Storage::F16(vec![0x8000])).unwrap(),
            TensorData::from_storage([1], Storage::BF16(vec![0x8000])).unwrap(),
            TensorData::from_storage([1], Storage::F32(vec![-0.0])).unwrap(),
            TensorData::from_storage([1], Storage::F64(vec![-0.0])).unwrap(),
        ] {
            assert_eq!(data.item().unwrap().as_f64().to_bits(), (-0.0f64).to_bits());
        }
    }

    #[test]
    fn item_rejects_non_singletons_without_mutating_storage() {
        for data in [
            TensorData::from_storage([0], Storage::U8(vec![])).unwrap(),
            TensorData::from_storage([2], Storage::U8(vec![3, 4])).unwrap(),
        ] {
            let before = data.clone();
            let error = data.item().unwrap_err();
            assert_eq!(error, Error::NonScalarItem(data.shape().clone()));
            assert_eq!(
                error.to_string(),
                format!("item requires exactly one element, got {}", data.shape())
            );
            assert_eq!(data, before);
        }
    }

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
    fn bitcast_preserves_canonical_payloads_without_numeric_coercion() {
        let cases = [
            (DType::U16, DType::F16, vec![0x00, 0x80, 0x01, 0x7e]),
            (DType::I16, DType::BF16, vec![0x00, 0x80, 0xc1, 0x7f]),
            (DType::U32, DType::F32, vec![0x00, 0x00, 0x00, 0x80]),
            (DType::I64, DType::F64, vec![0, 0, 0, 0, 0, 0, 0xf0, 0x3f]),
        ];
        for (from, to, bytes) in cases {
            let value =
                TensorData::from_le_bytes([bytes.len() / from.itemsize()], from, &bytes).unwrap();
            let reinterpreted = value.bitcast(to).unwrap();
            assert_eq!(reinterpreted.shape(), value.shape());
            assert_eq!(reinterpreted.to_le_bytes().unwrap(), bytes);
            assert_eq!(
                reinterpreted.bitcast(from).unwrap().to_le_bytes().unwrap(),
                bytes
            );
        }
        let value = TensorData::from_le_bytes([], DType::U8, &[0x80]).unwrap();
        assert_eq!(
            value.bitcast(DType::F16),
            Err(Error::BitcastItemsize {
                from: DType::U8,
                to: DType::F16
            })
        );
        let empty = TensorData::from_le_bytes([0, 2], DType::U32, &[]).unwrap();
        assert!(empty.bitcast(DType::F32).unwrap().is_empty());
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
            Storage::F32(vec![
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0x8000_0000),
            ]),
        )
        .unwrap();
        let cast = input.cast(DType::F32);
        let Storage::F32(values) = cast.storage() else {
            panic!("same dtype cast changed F32 storage");
        };
        assert_eq!(
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0x7f80_0001, 0x8000_0000]
        );
    }

    #[test]
    fn raw_padding_preserves_float8_source_and_fill_bytes() {
        let input = TensorData::from_storage(
            [2],
            Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3FNUZ,
                vec![0x80, 0xff],
            )),
        )
        .unwrap();
        let fill = TensorData::from_storage(
            [],
            Storage::Float8(crate::Float8Storage::from_raw(
                crate::Float8Format::E4M3FNUZ,
                vec![0x7f],
            )),
        )
        .unwrap();
        let output = input
            .pad_raw_offsets(Shape::from([4]), &[None, Some(0), Some(1), None], &fill)
            .unwrap();
        let Storage::Float8(output) = output.storage() else {
            panic!("raw Float8 padding changed storage family");
        };
        assert_eq!(output.format(), crate::Float8Format::E4M3FNUZ);
        assert_eq!(output.as_raw(), &[0x7f, 0x80, 0xff, 0x7f]);

        let wrong_fill = TensorData::scalar_with_dtype(Scalar::I(0), DType::U8);
        assert!(
            input
                .pad_raw_offsets(Shape::from([1]), &[None], &wrong_fill)
                .is_err()
        );
        assert!(
            input
                .pad_raw_offsets(Shape::from([1]), &[Some(2)], &fill)
                .is_err()
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
    fn assign_returns_self_after_right_aligned_broadcast_without_changing_source() {
        let mut destination = TensorData::from_storage([2, 3], Storage::U64(vec![0; 6])).unwrap();
        let source = TensorData::from_storage([1, 3], Storage::U64(vec![u64::MAX, 2, 3])).unwrap();
        let source_before = source.clone();
        let destination_address: *mut TensorData = &mut destination;

        let returned = destination.assign(&source).unwrap();
        assert_eq!(returned as *mut TensorData, destination_address);
        assert_eq!(
            returned.storage(),
            &Storage::U64(vec![u64::MAX, 2, 3, u64::MAX, 2, 3])
        );
        assert_eq!(source, source_before);
    }

    #[test]
    fn assign_validates_shape_before_dtype_and_keeps_failures_atomic() {
        let mut destination = TensorData::from_storage([2, 3], Storage::F32(vec![1.0; 6])).unwrap();
        let before = destination.clone();
        let invalid_shape_and_dtype =
            TensorData::from_storage([4], Storage::I32(vec![1; 4])).unwrap();
        assert_eq!(
            destination.assign(&invalid_shape_and_dtype).unwrap_err(),
            Error::ShapeMismatch {
                op: "assign",
                lhs: Shape::new([2, 3]),
                rhs: Shape::new([4]),
            }
        );
        assert_eq!(destination, before);

        let valid_shape_wrong_dtype =
            TensorData::from_storage([1, 3], Storage::I32(vec![1; 3])).unwrap();
        assert_eq!(
            destination.assign(&valid_shape_wrong_dtype).unwrap_err(),
            Error::InputDType {
                name: "assignment".into(),
                expected: DType::F32,
                actual: DType::I32,
            }
        );
        assert_eq!(destination, before);
    }

    #[test]
    fn assign_preserves_raw_float_payloads_and_zero_extent_geometry() {
        for source in [
            TensorData::from_storage([1], Storage::F16(vec![0x8000])).unwrap(),
            TensorData::from_storage([1], Storage::BF16(vec![0x7fc1])).unwrap(),
            TensorData::from_storage([1], Storage::F32(vec![f32::from_bits(0x8000_0000)])).unwrap(),
            TensorData::from_storage([1], Storage::F64(vec![f64::from_bits(0x7ff0_0000_0001)]))
                .unwrap(),
        ] {
            let source_before = source.clone();
            let mut destination = TensorData::from_storage(
                [2],
                match source.dtype() {
                    DType::F16 => Storage::F16(vec![0; 2]),
                    DType::BF16 => Storage::BF16(vec![0; 2]),
                    DType::F32 => Storage::F32(vec![0.0; 2]),
                    DType::F64 => Storage::F64(vec![0.0; 2]),
                    _ => unreachable!("float fixture"),
                },
            )
            .unwrap();
            destination.assign(&source).unwrap();
            let expected = match source.storage() {
                Storage::F16(values) => Storage::F16(vec![values[0]; 2]),
                Storage::BF16(values) => Storage::BF16(vec![values[0]; 2]),
                Storage::F32(values) => Storage::F32(vec![values[0]; 2]),
                Storage::F64(values) => Storage::F64(vec![values[0]; 2]),
                _ => unreachable!("float fixture"),
            };
            assert_same_storage_bits(destination.storage(), &expected);
            assert_same_storage_bits(source.storage(), source_before.storage());
        }

        let scalar = TensorData::scalar_with_dtype(Scalar::I(-7), DType::I32);
        let mut scalar_destination = TensorData::scalar_with_dtype(Scalar::I(0), DType::I32);
        scalar_destination.assign(&scalar).unwrap();
        assert_eq!(scalar_destination.storage(), scalar.storage());

        let zero_source = TensorData::from_storage([1, 0, 3], Storage::U8(vec![])).unwrap();
        let mut zero_destination =
            TensorData::from_storage([2, 0, 3], Storage::U8(vec![])).unwrap();
        zero_destination.assign(&zero_source).unwrap();
        assert_eq!(zero_destination.storage(), &Storage::U8(vec![]));
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
