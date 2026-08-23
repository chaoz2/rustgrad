use super::{Graph, NodeId, Op, RandomKind};
use crate::{DType, Error, Result, Scalar, Shape, TensorData};
use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);
static GLOBAL_COUNTER: AtomicU64 = AtomicU64::new(0);

fn implicit_seed() -> u64 {
    GLOBAL_SEED.load(Ordering::Acquire)
        ^ GLOBAL_COUNTER
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl Graph {
    /// Compatibility façade for tinygrad's global seed. Explicit-seed methods
    /// remain the replayable core; this atomic stream serializes implicit calls.
    pub fn manual_seed(seed: u64) {
        GLOBAL_SEED.store(seed, Ordering::Release);
        GLOBAL_COUNTER.store(0, Ordering::Release);
    }
    pub fn full(&mut self, shape: impl Into<Shape>, value: f32) -> Result<NodeId> {
        Ok(self.constant(TensorData::full(shape, value)?))
    }

    pub fn full_with_dtype(
        &mut self,
        shape: impl Into<Shape>,
        value: Scalar,
        dtype: DType,
    ) -> Result<NodeId> {
        Ok(self.constant(TensorData::full_with_dtype(shape, value, dtype)?))
    }

    pub fn zeros(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::zeros(shape)?))
    }

    pub fn zeros_with_dtype(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::zeros_with_dtype(shape, dtype)?))
    }

    pub fn ones(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        Ok(self.constant(TensorData::ones(shape)?))
    }

    pub fn arange(&mut self, start: i64, end: i64, step: i64) -> Result<NodeId> {
        Ok(self.constant(TensorData::arange(start, end, step)?))
    }

    pub fn empty(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::empty(shape, dtype)?))
    }

    pub fn linspace(
        &mut self,
        start: f64,
        stop: f64,
        steps: isize,
        dtype: DType,
    ) -> Result<NodeId> {
        Ok(self.constant(TensorData::linspace(start, stop, steps, dtype)?))
    }

    pub fn eye(&mut self, rows: usize, columns: Option<usize>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::eye(rows, columns, dtype)?))
    }

    /// Uniform `[0, 1)` values from a stateless explicit seed. The sequence
    /// is intentionally RustGrad-specific (SplitMix64), not tinygrad's
    /// stateful per-device Threefry stream, so graph replay has no global RNG.
    pub fn rand(&mut self, shape: impl Into<Shape>, dtype: DType, seed: u64) -> Result<NodeId> {
        self.uniform(shape, 0.0, 1.0, dtype, seed)
    }

    pub fn rand_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.rand(shape, dtype, implicit_seed())
    }
    pub fn randn_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.randn(shape, dtype, implicit_seed())
    }

    pub fn uniform(
        &mut self,
        shape: impl Into<Shape>,
        low: f64,
        high: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "uniform requires a floating point dtype",
            });
        }
        if !(low.is_finite() && high.is_finite() && low < high) {
            return Err(Error::InvalidRandom {
                reason: "uniform requires finite low < high",
            });
        }
        self.random(shape.into(), dtype, RandomKind::Uniform { low, high }, seed)
    }

    pub fn randn(&mut self, shape: impl Into<Shape>, dtype: DType, seed: u64) -> Result<NodeId> {
        self.normal(shape, 0.0, 1.0, dtype, seed)
    }

    pub fn normal(
        &mut self,
        shape: impl Into<Shape>,
        mean: f64,
        std: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "normal requires a floating point dtype",
            });
        }
        if !(mean.is_finite() && std.is_finite() && std >= 0.0) {
            return Err(Error::InvalidRandom {
                reason: "normal requires finite mean and non-negative std",
            });
        }
        self.random(shape.into(), dtype, RandomKind::Normal { mean, std }, seed)
    }

    pub fn randint(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randint requires an integer dtype",
            });
        }
        if low >= high {
            return Err(Error::InvalidRandom {
                reason: "randint requires low < high",
            });
        }
        if high.checked_sub(low).is_none() {
            return Err(Error::InvalidRandom {
                reason: "randint range overflows i64",
            });
        }
        self.random(shape.into(), dtype, RandomKind::RandInt { low, high }, seed)
    }

    pub fn randint_implicit(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
    ) -> Result<NodeId> {
        self.randint(shape, low, high, dtype, implicit_seed())
    }

    pub fn full_like(
        &mut self,
        input: NodeId,
        value: Scalar,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        self.full_with_dtype(
            self.shape(input)?.clone(),
            value,
            dtype.unwrap_or(self.dtype(input)?),
        )
    }
    pub fn zeros_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.full_like(input, Scalar::I(0), dtype)
    }
    pub fn ones_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.full_like(input, Scalar::I(1), dtype)
    }
    pub fn empty_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.empty(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
        )
    }
    pub fn rand_like(&mut self, input: NodeId, dtype: Option<DType>, seed: u64) -> Result<NodeId> {
        self.rand(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
            seed,
        )
    }
    pub fn randn_like(&mut self, input: NodeId, dtype: Option<DType>, seed: u64) -> Result<NodeId> {
        self.randn(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
            seed,
        )
    }

    pub fn randperm(&mut self, count: usize, dtype: DType, seed: u64) -> Result<NodeId> {
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randperm requires an integer dtype",
            });
        }
        Ok(self.push(Op::RandomPermutation { seed }, Shape::new([count]), dtype))
    }
    pub fn randperm_implicit(&mut self, count: usize, dtype: DType) -> Result<NodeId> {
        self.randperm(count, dtype, implicit_seed())
    }

    pub fn scaled_uniform(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        let bound = (shape.numel()? as f64).sqrt().recip();
        self.uniform(shape, -bound, bound, dtype, seed)
    }
    pub fn glorot_uniform(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() == 0 {
            return Err(Error::InvalidRandom {
                reason: "glorot_uniform requires rank at least one",
            });
        }
        let fan = shape.dims()[0] + shape.dims()[1..].iter().product::<usize>();
        self.uniform(
            shape,
            -(6.0 / fan as f64).sqrt(),
            (6.0 / fan as f64).sqrt(),
            dtype,
            seed,
        )
    }
    pub fn kaiming_uniform(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() < 2 {
            return Err(Error::InvalidRandom {
                reason: "kaiming initializer requires rank at least two",
            });
        }
        let fan = shape.dims()[1..].iter().product::<usize>();
        let b = (6.0 / (1.0 + a * a) / fan as f64).sqrt();
        self.uniform(shape, -b, b, dtype, seed)
    }
    pub fn kaiming_normal(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
        seed: u64,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() < 2 {
            return Err(Error::InvalidRandom {
                reason: "kaiming initializer requires rank at least two",
            });
        }
        let fan = shape.dims()[1..].iter().product::<usize>();
        self.normal(
            shape,
            0.0,
            (2.0 / (1.0 + a * a) / fan as f64).sqrt(),
            dtype,
            seed,
        )
    }

    fn random(
        &mut self,
        shape: Shape,
        dtype: DType,
        kind: RandomKind,
        seed: u64,
    ) -> Result<NodeId> {
        shape.numel()?;
        Ok(self.push(Op::Random { kind, seed }, shape, dtype))
    }
}
