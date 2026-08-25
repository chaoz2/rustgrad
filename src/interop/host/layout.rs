use crate::{AffineView, DType, Shape, UOpError};
use std::{fmt, ops::Range};

/// Checked byte interval for one logical element. The range is always exactly
/// `dtype.itemsize()` bytes wide.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogicalByteRange {
    range: Range<usize>,
}

impl LogicalByteRange {
    pub fn start(&self) -> usize {
        self.range.start
    }
    pub fn end(&self) -> usize {
        self.range.end
    }
    pub fn as_range(&self) -> Range<usize> {
        self.range.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HostInteropError {
    Rank {
        shape: usize,
        strides: usize,
    },
    Overflow,
    Misaligned {
        value: isize,
        width: usize,
    },
    Bounds {
        start: isize,
        end: isize,
        bytes: usize,
    },
    LogicalIndex {
        index: usize,
        count: usize,
    },
    NonInjectiveWrite,
}

impl fmt::Display for HostInteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "host interop error: {self:?}")
    }
}
impl std::error::Error for HostInteropError {}

/// A signed, byte-addressed logical tensor layout. Construction validates the
/// rank and arithmetic representation; `validate_read` or `validate_write`
/// additionally binds it to a particular byte slice length.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostTensorLayout {
    dtype: DType,
    shape: Shape,
    byte_offset: isize,
    byte_strides: Vec<isize>,
}

impl HostTensorLayout {
    pub fn new(
        dtype: DType,
        shape: impl Into<Shape>,
        byte_offset: isize,
        byte_strides: Vec<isize>,
    ) -> Result<Self, HostInteropError> {
        let shape = shape.into();
        if shape.rank() != byte_strides.len() {
            return Err(HostInteropError::Rank {
                shape: shape.rank(),
                strides: byte_strides.len(),
            });
        }
        let width = dtype.itemsize();
        for value in std::iter::once(&byte_offset).chain(byte_strides.iter()) {
            if value.rem_euclid(width as isize) != 0 {
                return Err(HostInteropError::Misaligned {
                    value: *value,
                    width,
                });
            }
        }
        Ok(Self {
            dtype,
            shape,
            byte_offset,
            byte_strides,
        })
    }

    pub fn contiguous(dtype: DType, shape: impl Into<Shape>) -> Result<Self, HostInteropError> {
        let shape = shape.into();
        let width = isize::try_from(dtype.itemsize()).map_err(|_| HostInteropError::Overflow)?;
        let mut stride = width;
        let mut strides = vec![0; shape.rank()];
        for axis in (0..shape.rank()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(
                    isize::try_from(shape.dims()[axis]).map_err(|_| HostInteropError::Overflow)?,
                )
                .ok_or(HostInteropError::Overflow)?;
        }
        Self::new(dtype, shape, 0, strides)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn byte_offset(&self) -> isize {
        self.byte_offset
    }
    pub fn byte_strides(&self) -> &[isize] {
        &self.byte_strides
    }
    pub fn element_width(&self) -> usize {
        self.dtype.itemsize()
    }
    pub fn logical_len(&self) -> Result<usize, HostInteropError> {
        self.shape.numel().map_err(|_| HostInteropError::Overflow)
    }

    /// Deterministic descriptor identity, deliberately independent of backing
    /// allocation address and lifetime.
    pub fn identity(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut add = |bytes: &[u8]| {
            for byte in bytes {
                hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        add(&[self.dtype as u8]);
        add(&(self.shape.rank() as u64).to_le_bytes());
        for dim in self.shape.dims() {
            add(&(*dim as u64).to_le_bytes());
        }
        add(&(self.byte_offset as i128).to_le_bytes());
        for stride in &self.byte_strides {
            add(&(*stride as i128).to_le_bytes());
        }
        hash
    }

    /// Validates every readable element and returns the exact enclosing byte
    /// span. An empty logical view has an empty span at its checked offset.
    pub fn validate_read(&self, bytes: usize) -> Result<Range<usize>, HostInteropError> {
        let count = self.logical_len()?;
        self.checked_empty_offset(bytes)?;
        if count == 0 {
            return Ok(self.byte_offset as usize..self.byte_offset as usize);
        }
        let (min, max) = self.endpoint_offsets()?;
        let end = max
            .checked_add(
                isize::try_from(self.element_width()).map_err(|_| HostInteropError::Overflow)?,
            )
            .ok_or(HostInteropError::Overflow)?;
        if min < 0
            || usize::try_from(end)
                .ok()
                .filter(|end| *end <= bytes)
                .is_none()
        {
            return Err(HostInteropError::Bounds {
                start: min,
                end,
                bytes,
            });
        }
        Ok(min as usize..end as usize)
    }

    /// Read validation plus nonoverlapping element-sized destinations.
    pub fn validate_write(&self, bytes: usize) -> Result<Range<usize>, HostInteropError> {
        let span = self.validate_read(bytes)?;
        self.affine(bytes)?.validate_write().map_err(map_affine)?;
        Ok(span)
    }

    pub fn logical_byte_range(
        &self,
        bytes: usize,
        index: usize,
    ) -> Result<LogicalByteRange, HostInteropError> {
        self.validate_read(bytes)?;
        let count = self.logical_len()?;
        if index >= count {
            return Err(HostInteropError::LogicalIndex { index, count });
        }
        let start = self.logical_offset_signed(index)?;
        let end = start
            .checked_add(
                isize::try_from(self.element_width()).map_err(|_| HostInteropError::Overflow)?,
            )
            .ok_or(HostInteropError::Overflow)?;
        Ok(LogicalByteRange {
            range: usize::try_from(start).map_err(|_| HostInteropError::Overflow)?
                ..usize::try_from(end).map_err(|_| HostInteropError::Overflow)?,
        })
    }

    pub(crate) fn logical_offset_signed(&self, index: usize) -> Result<isize, HostInteropError> {
        let count = self.logical_len()?;
        if index >= count {
            return Err(HostInteropError::LogicalIndex { index, count });
        }
        let mut linear = index;
        let mut offset = self.byte_offset;
        for axis in (0..self.shape.rank()).rev() {
            let dim = self.shape.dims()[axis];
            let coordinate = linear % dim;
            linear /= dim;
            let term = isize::try_from(coordinate)
                .map_err(|_| HostInteropError::Overflow)?
                .checked_mul(self.byte_strides[axis])
                .ok_or(HostInteropError::Overflow)?;
            offset = offset.checked_add(term).ok_or(HostInteropError::Overflow)?;
        }
        Ok(offset)
    }

    fn checked_empty_offset(&self, bytes: usize) -> Result<(), HostInteropError> {
        if self.byte_offset < 0
            || usize::try_from(self.byte_offset)
                .ok()
                .filter(|offset| *offset <= bytes)
                .is_none()
        {
            return Err(HostInteropError::Bounds {
                start: self.byte_offset,
                end: self.byte_offset,
                bytes,
            });
        }
        Ok(())
    }

    fn endpoint_offsets(&self) -> Result<(isize, isize), HostInteropError> {
        let mut min = self.byte_offset;
        let mut max = self.byte_offset;
        for (dim, stride) in self.shape.dims().iter().zip(&self.byte_strides) {
            let extent = isize::try_from(dim.saturating_sub(1))
                .map_err(|_| HostInteropError::Overflow)?
                .checked_mul(*stride)
                .ok_or(HostInteropError::Overflow)?;
            if extent < 0 {
                min = min.checked_add(extent).ok_or(HostInteropError::Overflow)?;
            } else {
                max = max.checked_add(extent).ok_or(HostInteropError::Overflow)?;
            }
        }
        Ok((min, max))
    }

    fn affine(&self, bytes: usize) -> Result<AffineView, HostInteropError> {
        let width = self.element_width();
        let width_signed = isize::try_from(width).map_err(|_| HostInteropError::Overflow)?;
        let source = Shape::new([bytes / width]);
        let strides = self
            .byte_strides
            .iter()
            .map(|stride| {
                i64::try_from(*stride / width_signed).map_err(|_| HostInteropError::Overflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AffineView {
            source_shape: source,
            logical_shape: self.shape.clone(),
            strides,
            offset: i64::try_from(self.byte_offset / width_signed)
                .map_err(|_| HostInteropError::Overflow)?,
        })
    }
}

fn map_affine(_: UOpError) -> HostInteropError {
    HostInteropError::NonInjectiveWrite
}
