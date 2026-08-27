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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, SplitSections};
    use std::collections::HashMap;

    fn execute(graph: &Graph, output: NodeId, input: TensorData) -> TensorData {
        CpuBackend
            .execute(graph, output, &HashMap::from([("x".into(), input)]))
            .unwrap()
    }

    #[test]
    fn chunk_matches_tinygrad_uneven_tail_and_preserves_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let outputs = graph.chunk(input, 3, -1).unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(
            outputs
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([2, 2]), Shape::from([2, 2]), Shape::from([2, 1])]
        );
        let loss = graph.sum_all(outputs[1]).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 5], (0..10).map(|x| x as f32).collect()).unwrap();

        assert_eq!(
            execute(&graph, outputs[2], values.clone()),
            TensorData::new([2, 1], vec![4., 9.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 5], vec![0., 0., 1., 1., 0., 0., 0., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn chunk_of_a_zero_axis_returns_exactly_requested_empty_views() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0]);
        let outputs = graph.chunk(input, 3, 1).unwrap();
        assert_eq!(outputs.len(), 3);
        for output in outputs {
            assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 0]));
        }
    }

    #[test]
    fn chunk_rejects_invalid_count_or_axis_without_graph_growth() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let node_count = graph.node_count();

        assert!(graph.chunk(input, 0, 0).is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph.chunk(input, 2, 2).is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn split_preserves_explicit_sections_uniform_tails_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let explicit = graph
            .split(input, SplitSections::Explicit(vec![1, 3, 1]), -1)
            .unwrap();
        let uniform = graph.split(input, SplitSections::Uniform(2), 1).unwrap();
        assert_eq!(explicit.len(), 3);
        assert_eq!(uniform.len(), 3);
        assert_eq!(
            explicit
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([2, 1]), Shape::from([2, 3]), Shape::from([2, 1])]
        );
        let loss = graph.sum_all(explicit[1]).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 5], (0..10).map(|x| x as f32).collect()).unwrap();

        assert_eq!(
            execute(&graph, uniform[2], values.clone()),
            TensorData::new([2, 1], vec![4., 9.]).unwrap()
        );
        assert_eq!(
            execute(&graph, explicit[1], values.clone()),
            TensorData::new([2, 3], vec![1., 2., 3., 6., 7., 8.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 5], vec![0., 1., 1., 1., 0., 0., 1., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn split_preserves_tinygrad_zero_axis_forms() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0]);
        let uniform = graph.split(input, SplitSections::Uniform(0), 1).unwrap();
        let explicit = graph
            .split(input, SplitSections::Explicit(vec![0, 0]), 1)
            .unwrap();
        assert_eq!(uniform.len(), 1);
        assert_eq!(explicit.len(), 2);
        for output in uniform.into_iter().chain(explicit) {
            assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 0]));
        }
    }

    #[test]
    fn split_rejects_bad_sections_before_graph_growth() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let node_count = graph.node_count();

        assert!(graph
            .split(input, SplitSections::Uniform(0), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Explicit(vec![2, 2]), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Explicit(vec![usize::MAX, 1]), 1)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph
            .split(input, SplitSections::Uniform(1), isize::MIN)
            .is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn flip_uses_signed_axes_and_preserves_stride_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let flipped = graph.flip(input, [0isize, -1]).unwrap();
        let selected = graph.shrink(flipped, [(0, 1), (0, 2)]).unwrap();
        let loss = graph.sum_all(selected).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();

        assert_eq!(
            execute(&graph, flipped, values.clone()),
            TensorData::new([2, 3], vec![6., 5., 4., 3., 2., 1.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![0., 0., 0., 0., 1., 1.]).unwrap()
        );
    }

    #[test]
    fn flip_empty_axes_is_a_scalar_noop_and_bad_axes_do_not_grow_the_graph() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let node_count = graph.node_count();
        assert_eq!(graph.flip(scalar, Vec::<isize>::new()).unwrap(), scalar);
        assert_eq!(graph.node_count(), node_count);

        let input = graph.input("x", [2, 3]);
        let node_count = graph.node_count();
        assert!(graph.flip(input, [1isize, -1]).is_err());
        assert_eq!(graph.node_count(), node_count);
        assert!(graph.flip(input, [isize::MIN]).is_err());
        assert_eq!(graph.node_count(), node_count);
    }

    #[test]
    fn stack_preflights_all_inputs_before_constructing_unsqueezes() {
        let mut graph = Graph::new();
        let left = graph.input("left", [2]);
        let right = graph.input("right", [3]);
        let node_count = graph.node_count();

        assert!(graph.stack([left, right], 0).is_err());
        assert_eq!(graph.node_count(), node_count);

        let first = graph.input("first", [2]);
        let second = graph.input("second", [2]);
        let stacked = graph.stack([first, second], -1).unwrap();
        let loss = graph.sum_all(stacked).unwrap();
        let gradient = graph.grad(loss, first).unwrap();
        assert_eq!(graph.shape(stacked).unwrap(), &Shape::from([2, 2]));
        let bindings = HashMap::from([
            ("left".into(), TensorData::new([2], vec![0., 0.]).unwrap()),
            ("right".into(), TensorData::new([3], vec![0., 0., 0.]).unwrap()),
            ("first".into(), TensorData::new([2], vec![1., 2.]).unwrap()),
            ("second".into(), TensorData::new([2], vec![3., 4.]).unwrap()),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, stacked, &bindings).unwrap(),
            TensorData::new([2, 2], vec![1., 3., 2., 4.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            TensorData::new([2], vec![1., 1.]).unwrap()
        );
    }
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

fn device_key(device: u32) -> u32 {
    if device == 0 {
        0x14B8_1119
    } else {
        device.wrapping_mul(0x9E37_79B9).rotate_left(13) ^ 0xA5A5_5A5A
    }
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
            if axis < 0 || axis >= dims.len() as isize {
                return Err(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                });
            }
            if dims[axis as usize] != 1 {
                return Ok(input);
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
        let shapes = inputs
            .iter()
            .map(|&input| Ok(self.shape(input)?.clone()))
            .collect::<Result<Vec<_>>>()?;
        let rank = shapes[0].rank() as isize + 1;
        let axis = if axis < 0 {
            axis.checked_add(rank).ok_or(Error::InvalidAxis {
                node: inputs[0],
                axis: usize::MAX,
                rank: rank as usize,
            })?
        } else {
            axis
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: inputs[0],
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        if shapes.iter().any(|shape| shape != &shapes[0]) {
            return Err(Error::InvalidConcat {
                axis: axis as usize,
                shapes,
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
        if !dtype.is_integer() {
            return Err(Error::InvalidRandom {
                reason: "randperm requires an integer dtype",
            });
        }
        // `RandomPermutation` predates captured streams. Reserve the same F32
        // domain as tinygrad's `rand(n).argsort()` and derive its legacy seed
        // from that immutable reservation until permutation receives typed IR.
        let stream = reserve_implicit_stream(0, stream_words(&Shape::new([count]), DType::F32, 1)?);
        let seed = (u64::from(stream.counter[1]) << 32 | u64::from(stream.counter[0]))
            ^ (u64::from(stream.key[1]) << 1)
            ^ u64::from(stream.key[0]);
        self.randperm(count, dtype, seed)
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
}
