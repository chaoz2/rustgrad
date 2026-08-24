use super::{uniform_bf16_bits, uniform_f16_bits, uniform_word, words};
use crate::{DType, Error, RandomKind, RandomStream, Result, Scalar, Shape, TensorData};

/// Immutable, fully captured random kernel semantic.  It contains the exact
/// Threefry reservation; execution never consults the mutable stream registry.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RandomKernelPlan {
    pub output: crate::NodeId,
    pub shape: Shape,
    pub dtype: DType,
    pub kind: RandomKind,
    pub stream: RandomStream,
    pub word_count: usize,
}

impl RandomKernelPlan {
    pub fn new(
        output: crate::NodeId,
        shape: Shape,
        dtype: DType,
        kind: RandomKind,
        stream: RandomStream,
    ) -> Result<Self> {
        let count = shape.numel()?;
        let (source_dtype, source_elements) = match kind {
            RandomKind::Normal { .. } => (
                DType::F32,
                count
                    .checked_mul(2)
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
            ),
            _ if dtype.is_float() => (dtype, count),
            _ => (DType::F32, count),
        };
        let word_count = source_elements
            .checked_mul(source_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?
            .div_ceil(4);
        let plan = Self {
            output,
            shape,
            dtype,
            kind,
            stream,
            word_count,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        let count = self.shape.numel()?;
        match self.kind {
            RandomKind::Uniform { low, high }
                if !(low.is_finite() && high.is_finite() && low < high) =>
            {
                return Err(Error::InvalidRandom {
                    reason: "uniform requires finite low < high",
                });
            }
            RandomKind::Normal { mean, std }
                if !(self.dtype.is_float()
                    && mean.is_finite()
                    && std.is_finite()
                    && std >= 0.0) =>
            {
                return Err(Error::InvalidRandom {
                    reason: "normal requires floating dtype, finite mean and non-negative std",
                });
            }
            RandomKind::RandInt { low, high }
                if !self.dtype.is_integer() || low >= high || high.checked_sub(low).is_none() =>
            {
                return Err(Error::InvalidRandom {
                    reason: "randint requires integer dtype and non-overflowing low < high",
                });
            }
            _ => {}
        }
        let (_, elements) = match self.kind {
            RandomKind::Normal { .. } => (
                DType::F32,
                count
                    .checked_mul(2)
                    .ok_or_else(|| Error::ShapeOverflow(self.shape.clone()))?,
            ),
            _ if self.dtype.is_float() => (self.dtype, count),
            _ => (DType::F32, count),
        };
        let source = if matches!(self.kind, RandomKind::Normal { .. }) || !self.dtype.is_float() {
            DType::F32
        } else {
            self.dtype
        };
        let words = elements
            .checked_mul(source.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(self.shape.clone()))?
            .div_ceil(4);
        if words != self.word_count {
            return Err(Error::InvalidRandom {
                reason: "random word count mismatch",
            });
        }
        Ok(())
    }

    fn unit(bits: &[u32], index: usize, dtype: DType) -> f64 {
        match dtype {
            DType::F16 => {
                let packed = bits[index / 2];
                let raw = if index % 2 == 0 {
                    packed as u16
                } else {
                    (packed >> 16) as u16
                };
                f64::from(crate::tensor::f16_to_f32(uniform_f16_bits(raw))) - 1.0
            }
            DType::BF16 => {
                let packed = bits[index / 2];
                let raw = if index % 2 == 0 {
                    packed as u16
                } else {
                    (packed >> 16) as u16
                };
                f64::from(crate::tensor::bf16_to_f32(uniform_bf16_bits(raw))) - 1.0
            }
            DType::F64 => {
                f64::from_bits(
                    (((bits[index * 2 + 1] as u64) << 32 | bits[index * 2] as u64) >> 12)
                        | 0x3FF0_0000_0000_0000,
                ) - 1.0
            }
            _ => f64::from(uniform_word(bits[index])),
        }
    }

    pub(crate) fn execute(&self) -> Result<TensorData> {
        self.validate()?;
        let count = self.shape.numel()?;
        let source = if matches!(self.kind, RandomKind::Normal { .. }) || !self.dtype.is_float() {
            DType::F32
        } else {
            self.dtype
        };
        let bits = words(self.stream.key, self.stream.counter, self.word_count);
        let values = (0..count).map(|index| match self.kind {
            RandomKind::Uniform { low, high } => {
                Scalar::F(low + (high - low) * Self::unit(&bits, index, source))
            }
            RandomKind::Normal { mean, std } => {
                let i = index * 2;
                let u1 = Self::unit(&bits, i, source).max(f64::MIN_POSITIVE);
                let u2 = Self::unit(&bits, i + 1, source);
                Scalar::F(
                    mean + std * (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos(),
                )
            }
            RandomKind::RandInt { low, high } => {
                Scalar::F(low as f64 + (high - low) as f64 * Self::unit(&bits, index, DType::F32))
            }
        });
        TensorData::from_scalars(self.shape.clone(), self.dtype, values)
    }
}
