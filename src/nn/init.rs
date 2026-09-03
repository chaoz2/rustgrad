//! Deterministic parameter initialization helpers.

use crate::{DType, Result, Scalar, Shape, TensorData};

pub(super) fn uniform(shape: Shape, low: f32, high: f32, seed: u64) -> Result<TensorData> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }
    TensorData::from_scalars(
        shape.clone(),
        DType::F32,
        (0..shape.numel()?).map(|i| {
            Scalar::F(
                (low + (high - low)
                    * ((mix(seed.wrapping_add(i as u64)) >> 40) as f32 / (1u32 << 24) as f32))
                    as f64,
            )
        }),
    )
}

/// Monotone seed allocation for graph-independent module initialization.
///
/// Composite modules use one cursor instead of inventing per-layer offsets.
/// A cursor is local to a constructor rehearsal, so a failed preparation
/// consumes no ambient random state and publishes no parameters.
pub(super) struct InitCursor {
    next: u64,
}

pub(super) fn glorot_uniform_bound(shape: &Shape) -> Result<f32> {
    if shape.rank() == 0 {
        return Err(crate::Error::InvalidRandom {
            reason: "glorot_uniform requires rank at least one",
        });
    }
    let tail = shape.dims()[1..].iter().try_fold(1usize, |fan, &dim| {
        fan.checked_mul(dim)
            .ok_or_else(|| crate::Error::ShapeOverflow(shape.clone()))
    })?;
    let fan = shape.dims()[0]
        .checked_add(tail)
        .ok_or_else(|| crate::Error::ShapeOverflow(shape.clone()))?;
    if fan == 0 {
        return Err(crate::Error::InvalidRandom {
            reason: "glorot_uniform has zero fan",
        });
    }
    shape.numel()?;
    Ok((6.0 / fan as f64).sqrt() as f32)
}

impl InitCursor {
    pub(super) const fn new(seed: u64) -> Self {
        Self { next: seed }
    }

    fn take(&mut self) -> u64 {
        let seed = self.next;
        self.next = self.next.wrapping_add(1);
        seed
    }

    /// Host-owned F32 form of checked-in tinygrad's Glorot bound.
    pub(super) fn glorot_uniform(&mut self, shape: Shape) -> Result<TensorData> {
        let bound = glorot_uniform_bound(&shape)?;
        uniform(shape, -bound, bound, self.take())
    }

    /// One source-ordered host-uniform draw for composite module constructors.
    pub(super) fn uniform(&mut self, shape: Shape, low: f32, high: f32) -> Result<TensorData> {
        uniform(shape, low, high, self.take())
    }
}
