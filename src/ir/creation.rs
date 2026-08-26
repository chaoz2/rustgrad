use super::{Graph, NodeId, Op, RandomKind, RandomStream};
use crate::random::reserve;
use crate::{DType, Error, Result, Scalar, Shape, TensorData};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct StreamRegistry {
    seed: u64,
    counters: BTreeMap<u32, [u32; 2]>,
}

/// An uncommitted implicit Threefry reservation owned by one graph and gated
/// by a scalar Bool node. Creating this value never mutates the global stream.
#[derive(Clone, Debug)]
pub struct PendingRandomReservation {
    graph: u64,
    guard: NodeId,
    shape: Shape,
    dtype: DType,
    device: u32,
    key: [u32; 2],
    expected_counter: [u32; 2],
    words: u64,
    committed: bool,
}

static STREAM_REGISTRY: OnceLock<Mutex<StreamRegistry>> = OnceLock::new();

fn stream_registry() -> &'static Mutex<StreamRegistry> {
    STREAM_REGISTRY.get_or_init(|| Mutex::new(StreamRegistry::default()))
}

fn stream_words(shape: &Shape, dtype: DType, multiplier: usize) -> Result<u64> {
    let elements = shape
        .numel()?
        .checked_mul(multiplier)
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    let bytes = elements
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
    Ok(bytes.div_ceil(4) as u64)
}

fn reserve_implicit_stream(device: u32, words: u64) -> RandomStream {
    // A mutex deliberately serializes implicit construction. Every node stores
    // the reservation it received, so later execution is schedule-independent.
    let mut registry = stream_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let start = reserve(registry.counters.entry(device).or_insert([0, 0]), words);
    RandomStream {
        device,
        // This is SHA256(0u32-be) narrowed to U32, matching tinygrad's first
        // device key. Further numeric devices use a deterministic distinct
        // derivation until RustGrad grows canonical backend device names.
        key: [device_key(device), registry.seed as u32],
        counter: start,
    }
}

fn checked_counter_end(counter: [u32; 2], words: u64) -> Result<()> {
    let start = u64::from(counter[0]) | (u64::from(counter[1]) << 32);
    start.checked_add(words).ok_or(Error::InvalidRandom {
        reason: "implicit random stream counter overflow",
    })?;
    Ok(())
}

fn device_key(device: u32) -> u32 {
    if device == 0 {
        0x14B8_1119
    } else {
        device.wrapping_mul(0x9E37_79B9).rotate_left(13) ^ 0xA5A5_5A5A
    }
}

fn validate_randperm_dtype(dtype: DType) -> Result<()> {
    if !dtype.is_integer() {
        return Err(Error::InvalidRandom {
            reason: "randperm requires an integer dtype",
        });
    }
    Ok(())
}

impl Graph {
    /// Captures, but does not consume, an implicit uniform reservation. The
    /// caller must commit it through the owning CPU session after `guard`
    /// validates successfully.
    pub fn pending_uniform_after_guard(
        &self,
        guard: NodeId,
        shape: impl Into<Shape>,
        dtype: DType,
        device: u32,
    ) -> Result<PendingRandomReservation> {
        if !matches!(self.op(guard)?, Op::TensorGuard { .. }) {
            return Err(Error::InvalidRandom {
                reason: "pending random guard requires a TensorGuard node",
            });
        }
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "pending uniform requires a floating point dtype",
            });
        }
        let shape = shape.into();
        let words = stream_words(&shape, dtype, 1)?;
        let registry = stream_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let expected_counter = *registry.counters.get(&device).unwrap_or(&[0, 0]);
        checked_counter_end(expected_counter, words)?;
        Ok(PendingRandomReservation {
            graph: self.id(),
            guard,
            shape,
            dtype,
            device,
            key: [device_key(device), registry.seed as u32],
            expected_counter,
            words,
            committed: false,
        })
    }

    pub(crate) fn commit_pending_uniform(
        &mut self,
        pending: &mut PendingRandomReservation,
        guard: NodeId,
    ) -> Result<NodeId> {
        if pending.graph != self.id() {
            return Err(Error::InvalidRandom {
                reason: "pending random reservation belongs to another graph",
            });
        }
        if pending.committed {
            return Err(Error::InvalidRandom {
                reason: "pending random reservation was already committed",
            });
        }
        if pending.guard != guard {
            return Err(Error::InvalidRandom {
                reason: "pending random reservation guard does not match",
            });
        }
        let mut registry = stream_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let counter = registry.counters.entry(pending.device).or_insert([0, 0]);
        if *counter != pending.expected_counter || registry.seed as u32 != pending.key[1] {
            return Err(Error::InvalidRandom {
                reason: "pending random reservation is stale",
            });
        }
        checked_counter_end(*counter, pending.words)?;
        let stream = RandomStream {
            device: pending.device,
            key: pending.key,
            counter: reserve(counter, pending.words),
        };
        let node = self.random_stream(
            pending.shape.clone(),
            pending.dtype,
            RandomKind::Uniform {
                low: 0.0,
                high: 1.0,
            },
            stream,
        )?;
        pending.committed = true;
        Ok(node)
    }
    /// Builds a square matrix with a rank-one input on its main diagonal.
    ///
    /// This is tinygrad's static `diag` composition: insert a singleton
    /// column, append typed zero padding, flatten, retain the square prefix,
    /// then reshape. All derived extents are checked before any graph node is
    /// emitted.
    pub fn diag(&mut self, input: NodeId) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        if shape.rank() != 1 {
            return Err(Error::InvalidDiagonal {
                reason: "diag requires a rank-one input",
            });
        }
        let extent = shape.dims()[0];
        let padded_width = extent
            .checked_add(1)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let square = extent
            .checked_mul(extent)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let column = self.unsqueeze(input, -1)?;
        let padded = self.pad(column, vec![(0, 0), (0, padded_width - 1)], Scalar::I(0))?;
        let flattened = self.flatten(padded, 0, 1)?;
        let square_prefix = self.shrink(flattened, vec![(0, square)])?;
        self.reshape(square_prefix, Shape::new(vec![extent, extent]))
    }

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
    /// Resets all implicit per-device Threefry streams. Existing graph nodes
    /// retain their captured reservations; only subsequently constructed nodes
    /// observe the new sequence.
    pub fn manual_seed(seed: u64) {
        let mut registry = stream_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        registry.seed = seed;
        registry.counters.clear();
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

    /// Creates a dense tensor of ones in an explicit storage dtype.
    pub fn ones_with_dtype(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        Ok(self.constant(TensorData::ones_with_dtype(shape, dtype)?))
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
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "rand requires a floating point dtype",
            });
        }
        self.uniform(shape, 0.0, 1.0, dtype, seed)
    }

    pub fn rand_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.rand_implicit_on_device(shape, dtype, 0)
    }

    /// Implicit `rand` from an isolated numeric device stream. Device `0` is
    /// the CPU-compatible default; accelerator lowering is not implemented.
    pub fn rand_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "rand requires a floating point dtype",
            });
        }
        let shape = shape.into();
        let stream = reserve_implicit_stream(device, stream_words(&shape, dtype, 1)?);
        self.random_stream(
            shape,
            dtype,
            RandomKind::Uniform {
                low: 0.0,
                high: 1.0,
            },
            stream,
        )
    }
    pub fn randn_implicit(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.randn_implicit_on_device(shape, dtype, 0)
    }

    pub fn randn_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        if !dtype.is_float() {
            return Err(Error::InvalidRandom {
                reason: "normal requires a floating point dtype",
            });
        }
        let shape = shape.into();
        // tinygrad's Box-Muller path consumes two F32 uniforms per output.
        let stream = reserve_implicit_stream(device, stream_words(&shape, DType::F32, 2)?);
        self.random_stream(
            shape,
            dtype,
            RandomKind::Normal {
                mean: 0.0,
                std: 1.0,
            },
            stream,
        )
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
        self.randint_implicit_on_device(shape, low, high, dtype, 0)
    }

    pub fn randint_implicit_on_device(
        &mut self,
        shape: impl Into<Shape>,
        low: i64,
        high: i64,
        dtype: DType,
        device: u32,
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
        let shape = shape.into();
        let stream = reserve_implicit_stream(device, stream_words(&shape, DType::F32, 1)?);
        self.random_stream(shape, dtype, RandomKind::RandInt { low, high }, stream)
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

    /// Creates a constant with the input's shape and, unless overridden,
    /// storage dtype. Like tinygrad's `const_like`, this is a leaf and does
    /// not create a gradient edge to `input`.
    pub fn const_like(
        &mut self,
        input: NodeId,
        value: Scalar,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        self.full_like(input, value, dtype)
    }

    pub fn zeros_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.const_like(input, Scalar::I(0), dtype)
    }
    pub fn ones_like(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.const_like(input, Scalar::I(1), dtype)
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

    /// Draws from the synchronized implicit Threefry stream with the input's
    /// shape and, unless overridden, dtype.
    pub fn rand_like_implicit(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.rand_implicit(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
        )
    }

    pub fn randn_like(&mut self, input: NodeId, dtype: Option<DType>, seed: u64) -> Result<NodeId> {
        self.randn(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
            seed,
        )
    }

    /// Draws a standard normal from the synchronized implicit Threefry stream
    /// with the input's shape and, unless overridden, dtype.
    pub fn randn_like_implicit(&mut self, input: NodeId, dtype: Option<DType>) -> Result<NodeId> {
        self.randn_implicit(
            self.shape(input)?.clone(),
            dtype.unwrap_or(self.dtype(input)?),
        )
    }

    pub fn randperm(&mut self, count: usize, dtype: DType, seed: u64) -> Result<NodeId> {
        validate_randperm_dtype(dtype)?;
        self.random_permutation(
            Shape::new([count]),
            dtype,
            RandomStream {
                device: 0,
                key: [0, seed as u32],
                counter: [0, 0],
            },
        )
    }

    pub fn randperm_implicit(&mut self, count: usize, dtype: DType) -> Result<NodeId> {
        self.randperm_implicit_on_device(count, dtype, 0)
    }

    /// Returns `rand(count).argsort()` from the named implicit Threefry stream.
    pub fn randperm_implicit_on_device(
        &mut self,
        count: usize,
        dtype: DType,
        device: u32,
    ) -> Result<NodeId> {
        validate_randperm_dtype(dtype)?;
        let shape = Shape::new([count]);
        let stream = reserve_implicit_stream(device, stream_words(&shape, DType::F32, 1)?);
        self.random_permutation(shape, dtype, stream)
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
        self.random_stream(
            shape,
            dtype,
            kind,
            RandomStream {
                device: 0,
                key: [0, seed as u32],
                counter: [0, 0],
            },
        )
    }

    fn random_stream(
        &mut self,
        shape: Shape,
        dtype: DType,
        kind: RandomKind,
        stream: RandomStream,
    ) -> Result<NodeId> {
        shape.numel()?;
        Ok(self.push(Op::Random { kind, stream }, shape, dtype))
    }

    fn random_permutation(
        &mut self,
        shape: Shape,
        dtype: DType,
        stream: RandomStream,
    ) -> Result<NodeId> {
        shape.numel()?;
        Ok(self.push(Op::RandomPermutation { stream }, shape, dtype))
    }
}

#[cfg(test)]
mod pending_random_tests {
    use super::checked_counter_end;

    #[test]
    fn pending_random_counter_overflow_is_rejected_at_the_private_snapshot_seam() {
        // Public callers cannot forge a counter snapshot; this directly covers
        // the checked arithmetic used before any reservation is committed.
        assert!(checked_counter_end([u32::MAX, u32::MAX], 1).is_err());
    }
}
