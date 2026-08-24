use super::{Graph, NodeId, Op, RandomKind, RandomStream};
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
    pub fn unsqueeze(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let mut dims = self.shape(input)?.dims().to_vec();
        let rank = dims.len() as isize + 1;
        let axis = if axis < 0 { axis + rank } else { axis };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        dims.insert(axis as usize, 1);
        self.reshape(input, Shape::new(dims))
    }

    pub fn squeeze(&mut self, input: NodeId, axis: Option<isize>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let mut dims = shape.dims().to_vec();
        if let Some(axis) = axis {
            let axis = if axis < 0 {
                axis + dims.len() as isize
            } else {
                axis
            };
            if axis < 0 || axis >= dims.len() as isize || dims[axis as usize] != 1 {
                return Err(Error::InvalidRandom {
                    reason: "squeeze axis must select a size-one dimension",
                });
            }
            dims.remove(axis as usize);
        } else {
            dims.retain(|dim| *dim != 1);
        }
        self.reshape(input, Shape::new(dims))
    }

    pub fn flatten(&mut self, input: NodeId, start: isize, end: isize) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let rank = shape.rank() as isize;
        let start = if start < 0 { start + rank } else { start };
        let end = if end < 0 { end + rank } else { end };
        if start < 0 || end < start || end >= rank {
            return Err(Error::InvalidRandom {
                reason: "invalid flatten dimensions",
            });
        }
        let mut dims = shape.dims()[..start as usize].to_vec();
        dims.push(
            shape.dims()[start as usize..=end as usize]
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
        );
        dims.extend_from_slice(&shape.dims()[end as usize + 1..]);
        self.reshape(input, Shape::new(dims))
    }

    pub fn stack(&mut self, inputs: impl Into<Vec<NodeId>>, axis: isize) -> Result<NodeId> {
        let inputs = inputs.into();
        if inputs.is_empty() {
            return Err(Error::InvalidRandom {
                reason: "stack requires at least one tensor",
            });
        }
        let rank = self.shape(inputs[0])?.rank() as isize + 1;
        let axis = if axis < 0 { axis + rank } else { axis };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidRandom {
                reason: "invalid stack axis",
            });
        }
        let mut expanded = Vec::with_capacity(inputs.len());
        for input in inputs {
            expanded.push(self.unsqueeze(input, axis)?);
        }
        self.concat(expanded, axis as usize)
    }

    pub fn one_hot(&mut self, input: NodeId, classes: usize) -> Result<NodeId> {
        if !self.dtype(input)?.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "one_hot requires integer indices",
            });
        }
        let mut dims = self.shape(input)?.dims().to_vec();
        dims.push(1);
        let values = self.reshape(input, Shape::new(dims.clone()))?;
        let classes_node = self.arange(0, classes as i64, 1)?;
        let mut class_shape = vec![1; dims.len()];
        *class_shape.last_mut().unwrap() = classes;
        let classes_node = self.reshape(classes_node, Shape::new(class_shape))?;
        let equal = self.eq(values, classes_node)?;
        let one = self.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), DType::I32));
        self.select(equal, one, zero)
    }

    pub fn meshgrid(
        &mut self,
        inputs: impl Into<Vec<NodeId>>,
        indexing: &str,
    ) -> Result<Vec<NodeId>> {
        let inputs = inputs.into();
        if !(indexing == "ij" || indexing == "xy") {
            return Err(Error::InvalidRandom {
                reason: "meshgrid indexing must be ij or xy",
            });
        }
        if inputs.len() <= 1 {
            return Ok(inputs);
        }
        let mut lengths = Vec::new();
        for input in &inputs {
            let shape = self.shape(*input)?;
            if shape.rank() > 1 {
                return Err(Error::InvalidRandom {
                    reason: "meshgrid inputs must be scalars or vectors",
                });
            }
            lengths.push(if shape.rank() == 0 {
                1
            } else {
                shape.dims()[0]
            });
        }
        let mut output = lengths.clone();
        if indexing == "xy" {
            output.swap(0, 1);
        }
        inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                let axis = if indexing == "xy" && index < 2 {
                    1 - index
                } else {
                    index
                };
                let mut shape = vec![1; output.len()];
                shape[axis] = lengths[index];
                let input = if self.shape(input)?.rank() == 0 {
                    self.unsqueeze(input, 0)?
                } else {
                    input
                };
                let input = self.reshape(input, Shape::new(shape))?;
                self.expand(input, Shape::new(output.clone()))
            })
            .collect()
    }
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

    /// Uniform `[0, 1)` values from an explicit Threefry stream key.
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
        Ok(self.push(
            Op::Random {
                kind,
                stream: RandomStream {
                    device: 0,
                    key: [0, seed as u32],
                    counter: [0, 0],
                },
            },
            shape,
            dtype,
        ))
    }
}
