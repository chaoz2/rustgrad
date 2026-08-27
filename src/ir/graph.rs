use super::*;
use crate::nn::{ParameterId, ParameterSnapshot};
use crate::{
    CompileTrace, DType, EinsumPlan, Error, Result, Scalar, Shape, SymbolicShape, SymbolicVar,
    TensorData, TraceStep,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub op: Op,
    pub shape: Shape,
    pub dtype: DType,
    pub requires_grad: bool,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) dynamic_nodes: Vec<DynamicNode>,
    id: u64,
    pub(crate) grad_enabled: bool,
    parameter_bindings: BTreeMap<(ParameterId, u64), ParameterBinding>,
}

#[derive(Clone, Debug)]
struct ParameterBinding {
    node: NodeId,
    input_name: String,
    data: TensorData,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            dynamic_nodes: Vec::new(),
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            grad_enabled: true,
            parameter_bindings: BTreeMap::new(),
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stable graph identity used by diagnostics and graph-owned resources.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of graph nodes currently allocated.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns row-major coordinates of every nonzero input element. Its
    /// concrete shape is `[count, input_rank]` after realization.
    pub fn nonzero(&mut self, input: NodeId) -> Result<DynamicNodeId> {
        self.node(input)?;
        let id = DynamicNodeId(self.dynamic_nodes.len());
        self.dynamic_nodes.push(DynamicNode::nonzero(input));
        Ok(id)
    }

    /// Fixed-shape `nonzero(size=...)`, matching tinygrad's row-major
    /// pad/truncate form without introducing a new dynamic-result primitive.
    ///
    /// The result has shape `[size, input_rank]`. Every extent, the flattened
    /// selection length, and every `arange` endpoint are checked before this
    /// method appends its comparison, movement, or selection composition.
    pub fn nonzero_fixed(&mut self, input: NodeId, size: usize, fill: Scalar) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let count = shape.numel()?;
        let rank = shape.rank();
        let selection_len = size
            .checked_mul(rank)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        for &dimension in shape.dims() {
            i64::try_from(dimension).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        }
        if rank == 0 {
            return self.zeros_with_dtype([size, 0], DType::I32);
        }

        let zero = self.constant(TensorData::scalar_with_dtype(Scalar::I(0), dtype));
        let mask = self.ne(input, zero)?;
        let flattened_mask = self.reshape(mask, Shape::from([count]))?;
        let mut coordinates = Vec::with_capacity(rank);
        for (axis, &dimension) in shape.dims().iter().enumerate() {
            let range = self.arange(0, dimension as i64, 1)?;
            // tinygrad's default integer arange uses its default I32 width
            // until a coordinate no longer fits, then widens. Preserve that
            // observable index dtype rather than exposing Graph::arange's
            // legacy I64 storage for ordinary shapes.
            let range = if dimension <= i32::MAX as usize {
                self.cast(range, DType::I32)?
            } else {
                range
            };
            let mut coordinate_shape = vec![1; rank];
            coordinate_shape[axis] = dimension;
            let range = self.reshape(range, Shape::new(coordinate_shape))?;
            let range = self.expand(range, shape.clone())?;
            coordinates.push(self.reshape(range, Shape::from([count]))?);
        }
        let coordinates = self.stack(coordinates, -1)?;
        let expanded_mask = self.unsqueeze(flattened_mask, -1)?;
        let expanded_mask = self.expand(expanded_mask, Shape::from([count, rank]))?;
        let selected = self.masked_select(coordinates, expanded_mask, selection_len, fill)?;
        self.reshape(selected, Shape::from([size, rank]))
    }

    /// Unbounded, row-major boolean selection. Unlike [`Self::masked_select`]
    /// this result has runtime shape `[selected_count]`.
    pub fn masked_select_dynamic(&mut self, input: NodeId, mask: NodeId) -> Result<DynamicNodeId> {
        let source = self.node(input)?;
        let mask_node = self.node(mask)?;
        if mask_node.dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "masked_select_dynamic",
                actual: mask_node.dtype,
            });
        }
        if mask_node.shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidIndexedShape {
                op: "masked_select_dynamic",
                input: source.shape.clone(),
                index: mask_node.shape.clone(),
            });
        }
        let id = DynamicNodeId(self.dynamic_nodes.len());
        self.dynamic_nodes
            .push(DynamicNode::masked_select(input, mask, source.dtype));
        Ok(id)
    }

    /// Reduces a dynamic result to a scalar dynamic loss.
    pub fn dynamic_sum(&mut self, input: DynamicNodeId) -> Result<DynamicNodeId> {
        let dtype = self.dynamic_node(input)?.dtype;
        let id = DynamicNodeId(self.dynamic_nodes.len());
        self.dynamic_nodes.push(DynamicNode::sum(input, dtype));
        Ok(id)
    }

    pub(crate) fn dynamic_node(&self, id: DynamicNodeId) -> Result<&DynamicNode> {
        self.dynamic_nodes.get(id.0).ok_or(Error::InvalidIndex)
    }

    pub(crate) fn bind_parameter(&mut self, snapshot: ParameterSnapshot) -> Result<NodeId> {
        let key = (snapshot.identity, snapshot.version);
        if let Some(binding) = self.parameter_bindings.get(&key) {
            return Ok(binding.node);
        }
        let input_name = format!("{}_v{}", snapshot.input_name, snapshot.version);
        let node = self.input_dtype_requires_grad(
            input_name.clone(),
            snapshot.shape,
            snapshot.dtype,
            snapshot.trainable,
        );
        self.parameter_bindings.insert(
            key,
            ParameterBinding {
                node,
                input_name,
                data: snapshot.data,
            },
        );
        Ok(node)
    }

    pub(crate) fn bound_parameter_node(
        &self,
        identity: ParameterId,
        version: u64,
    ) -> Option<NodeId> {
        self.parameter_bindings
            .get(&(identity, version))
            .map(|binding| binding.node)
    }

    /// Returns every immutable parameter value captured by this graph.
    pub fn parameter_bindings(&self) -> HashMap<String, TensorData> {
        self.parameter_bindings
            .values()
            .map(|binding| (binding.input_name.clone(), binding.data.clone()))
            .collect()
    }

    pub(crate) fn parameter_bindings_for(
        &self,
        identities: &BTreeSet<ParameterId>,
    ) -> HashMap<String, TensorData> {
        self.parameter_bindings
            .iter()
            .filter(|((identity, _), _)| identities.contains(identity))
            .map(|(_, binding)| (binding.input_name.clone(), binding.data.clone()))
            .collect()
    }

    pub fn input(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> NodeId {
        self.input_dtype(name, shape, DType::F32)
    }

    /// Adds an input after explicitly specializing a symbolic shape.  This is
    /// intentionally a one-way boundary: graph nodes and CPU allocation retain
    /// the existing concrete `Shape` invariant.
    pub fn input_symbolic(
        &mut self,
        name: impl Into<String>,
        shape: &SymbolicShape,
        bindings: &BTreeMap<SymbolicVar, i64>,
    ) -> Result<NodeId> {
        Ok(self.input(name, shape.bind_for_graph(bindings)?))
    }

    pub fn input_dtype(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> NodeId {
        self.input_dtype_requires_grad(name, shape, dtype, dtype.is_float())
    }

    /// Adds an input leaf with an explicit gradient-tracking contract.
    pub fn input_dtype_requires_grad(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        self.push_with_grad(
            Op::Input { name: name.into() },
            shape.into(),
            dtype,
            requires_grad && dtype.is_float(),
        )
    }

    pub fn constant(&mut self, data: TensorData) -> NodeId {
        let shape = data.shape().clone();
        let dtype = data.dtype();
        self.push_with_grad(Op::Constant(data), shape, dtype, false)
    }

    /// Returns whether future graph operations record reverse-mode edges.
    pub fn grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// Runs a graph-building closure with reverse-mode recording disabled.
    /// The guard is stored on this graph only, so it is thread-safe and cannot
    /// leak to another graph instance.
    pub fn no_grad<T>(&mut self, build: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.grad_enabled;
        self.grad_enabled = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build(self)));
        self.grad_enabled = previous;
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Creates a value-sharing node that is a new gradient leaf.
    pub fn detach(&mut self, input: NodeId) -> Result<NodeId> {
        let node = self.node(input)?;
        Ok(self.push_with_grad(
            Op::Detach { input },
            node.shape.clone(),
            node.dtype,
            node.dtype.is_float(),
        ))
    }

    /// Returns the explicit gradient-tracking state of a graph node.
    pub fn requires_grad(&self, id: NodeId) -> Result<bool> {
        Ok(self.node(id)?.requires_grad)
    }

    pub fn sum(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        self.reduce(input, ReduceKind::Sum, Some(vec![axis as isize]), false)
    }
    pub fn reduce(
        &mut self,
        input: NodeId,
        kind: ReduceKind,
        axes: Option<Vec<isize>>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let axes = normalize_axes(input, source.shape.rank(), axes)?;
        let shape = reduction_shape(&source.shape, &axes, keepdim);
        if matches!(kind, ReduceKind::Max | ReduceKind::Min)
            && has_empty_reduction_domain(&source.shape, &shape, &axes)
        {
            return Err(Error::EmptyReduction {
                op: match kind {
                    ReduceKind::Max => "max",
                    ReduceKind::Min => "min",
                    _ => unreachable!(),
                },
                shape: source.shape.clone(),
                axes,
            });
        }
        let dtype = match kind {
            ReduceKind::Mean if !source.dtype.is_float() => DType::F32,
            ReduceKind::Sum => sum_dtype(source.dtype),
            _ => source.dtype,
        };
        Ok(self.push(
            Op::Reduce {
                input,
                kind,
                axes,
                keepdim,
            },
            shape,
            dtype,
        ))
    }
    pub fn argmax(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, true, axis, keepdim)
    }
    pub fn argmin(&mut self, input: NodeId, axis: Option<isize>, keepdim: bool) -> Result<NodeId> {
        self.arg_reduce(input, false, axis, keepdim)
    }

    /// Stable ordering along one signed axis. The two returned NodeIds are
    /// distinct selectors of one shared producer: values retain the input
    /// dtype and indices are I32 source positions.
    pub fn sort(
        &mut self,
        input: NodeId,
        axis: isize,
        descending: bool,
    ) -> Result<(NodeId, NodeId)> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        shape.numel()?;
        let axis = if shape.rank() == 0 {
            if !matches!(axis, -1 | 0) {
                return Err(Error::InvalidAxis {
                    node: input,
                    axis: usize::MAX,
                    rank: 0,
                });
            }
            0
        } else {
            normalize_axes(input, shape.rank(), Some(vec![axis]))?[0]
        };
        if shape.rank() != 0 && shape.dims()[axis] > i32::MAX as usize {
            return Err(Error::ShapeOverflow(shape));
        }
        let pair = self.nodes.len() as u64;
        let values = self.push(
            Op::Sort {
                input,
                axis,
                descending,
                pair,
                output: SortOutput::Values,
            },
            shape.clone(),
            dtype,
        );
        let indices = self.push(
            Op::Sort {
                input,
                axis,
                descending,
                pair,
                output: SortOutput::Indices,
            },
            shape,
            DType::I32,
        );
        Ok((values, indices))
    }

    /// Stable I32 source positions from [`Self::sort`].
    pub fn argsort(&mut self, input: NodeId, axis: isize, descending: bool) -> Result<NodeId> {
        Ok(self.sort(input, axis, descending)?.1)
    }

    fn arg_reduce(
        &mut self,
        input: NodeId,
        max: bool,
        axis: Option<isize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let axis = axis
            .map(|a| normalize_axes(input, source.shape.rank(), Some(vec![a])))
            .transpose()?
            .map(|v| v[0]);
        let axes = axis.map_or_else(|| (0..source.shape.rank()).collect(), |a| vec![a]);
        let shape = reduction_shape(&source.shape, &axes, keepdim);
        if has_empty_reduction_domain(&source.shape, &shape, &axes) {
            return Err(Error::EmptyReduction {
                op: if max { "argmax" } else { "argmin" },
                shape: source.shape.clone(),
                axes,
            });
        }
        Ok(self.push(
            Op::ArgReduce {
                input,
                max,
                axis,
                keepdim,
            },
            shape,
            DType::I32,
        ))
    }
    pub(crate) fn reduce_grad(
        &mut self,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let dtype = self.node(upstream)?.dtype;
        Ok(self.push(
            Op::ReduceGrad {
                input,
                upstream,
                kind,
                axes,
                keepdim,
            },
            shape,
            dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_grad_vjp(
        &mut self,
        cotangent: NodeId,
        input: NodeId,
        upstream: NodeId,
        kind: ReduceKind,
        axes: Vec<usize>,
        keepdim: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = match wrt {
            0 => self.node(input)?.shape.clone(),
            1 => self.node(upstream)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                kind,
                axes,
                keepdim,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn sum_to(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidSumTo {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::SumTo {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    pub fn reshape(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if source.shape.numel()? != shape.numel()? {
            return Err(Error::InvalidReshape {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::Reshape {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    pub fn permute(&mut self, input: NodeId, axes: impl Into<Vec<usize>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let axes = axes.into();
        let mut sorted = axes.clone();
        sorted.sort_unstable();
        if sorted != (0..source.shape.rank()).collect::<Vec<_>>() {
            return Err(Error::InvalidPermutation {
                shape: source.shape.clone(),
                axes,
            });
        }
        let shape = Shape::new(
            axes.iter()
                .map(|axis| source.shape.dims()[*axis])
                .collect::<Vec<_>>(),
        );
        Ok(self.push(Op::Permute { input, axes }, shape, source.dtype))
    }

    /// Swaps two signed axes, matching tinygrad's public `transpose(dim0,
    /// dim1)` composition over `permute`.
    ///
    /// Each axis is normalized independently so equal axes remain a source
    /// no-op rather than being rejected as a duplicate permutation. Both
    /// normalizations complete before a permutation node can be appended.
    pub fn transpose(&mut self, input: NodeId, dim0: isize, dim1: isize) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let dim0 = normalize_axes(input, rank, Some(vec![dim0]))?[0];
        let dim1 = normalize_axes(input, rank, Some(vec![dim1]))?[0];
        if dim0 == dim1 {
            return Ok(input);
        }
        let mut order = (0..rank).collect::<Vec<_>>();
        order.swap(dim0, dim1);
        self.permute(input, order)
    }

    /// Replaces one signed axis with concrete extents and, at most, one
    /// source-compatible inferred extent.
    ///
    /// This is tinygrad's `unflatten(dim, sizes)` expressed without a general
    /// negative-shape API. Every axis, inference, product, and output extent
    /// is checked before the final existing `reshape` node is appended.
    pub fn unflatten(
        &mut self,
        input: NodeId,
        dim: isize,
        sizes: impl Into<Vec<crate::UnflattenExtent>>,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        let axis = normalize_axes(input, shape.rank(), Some(vec![dim]))?[0];
        let sizes = sizes.into();
        let inferred = sizes
            .iter()
            .enumerate()
            .filter_map(|(index, size)| matches!(size, crate::UnflattenExtent::Infer).then_some(index))
            .collect::<Vec<_>>();
        if inferred.len() > 1 {
            return Err(Error::InvalidRandom {
                reason: "unflatten permits at most one inferred extent",
            });
        }
        let known_product = sizes.iter().try_fold(1usize, |product, size| match size {
            crate::UnflattenExtent::Exact(size) => product
                .checked_mul(*size)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone())),
            crate::UnflattenExtent::Infer => Ok(product),
        })?;
        let source_extent = shape.dims()[axis];
        let inferred_extent = if inferred.is_empty() {
            None
        } else {
            if known_product == 0 {
                return Err(Error::InvalidRandom {
                    reason: "unflatten cannot infer through zero extent product",
                });
            }
            Some(source_extent / known_product)
        };
        let output_capacity = shape
            .rank()
            .checked_sub(1)
            .and_then(|rank| rank.checked_add(sizes.len()))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut output = Vec::with_capacity(output_capacity);
        output.extend_from_slice(&shape.dims()[..axis]);
        output.extend(sizes.into_iter().map(|size| match size {
            crate::UnflattenExtent::Exact(size) => size,
            crate::UnflattenExtent::Infer => inferred_extent.expect("inferred extent prevalidated"),
        }));
        output.extend_from_slice(&shape.dims()[axis + 1..]);
        let output = Shape::new(output);
        if output.numel()? != shape.numel()? {
            return Err(Error::InvalidReshape {
                from: shape,
                to: output,
            });
        }
        self.reshape(input, output)
    }

    pub fn expand(&mut self, input: NodeId, shape: impl Into<Shape>) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = shape.into();
        if source.shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: source.shape.clone(),
                to: shape,
            });
        }
        Ok(self.push(
            Op::Expand {
                input,
                shape: shape.clone(),
            },
            shape,
            source.dtype,
        ))
    }

    /// Takes checked, half-open bounds for every input axis.
    pub fn shrink(
        &mut self,
        input: NodeId,
        bounds: impl Into<Vec<(usize, usize)>>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let bounds = bounds.into();
        if bounds.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "shrink",
                expected: source.shape.rank(),
                actual: bounds.len(),
            });
        }
        let mut dims = Vec::with_capacity(bounds.len());
        for (axis, ((start, end), dim)) in bounds.iter().zip(source.shape.dims()).enumerate() {
            if start > end || *end > *dim {
                return Err(Error::InvalidBounds {
                    axis,
                    start: *start,
                    end: *end,
                    dim: *dim,
                });
            }
            dims.push(end - start);
        }
        Ok(self.push(Op::Shrink { input, bounds }, Shape::new(dims), source.dtype))
    }

    /// Pads every axis with `(before, after)`. `fill` is deterministically
    /// converted to the input dtype; padding never changes tensor dtype.
    pub fn pad(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(usize, usize)>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let padding = padding.into();
        if padding.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "pad",
                expected: source.shape.rank(),
                actual: padding.len(),
            });
        }
        let dims = source
            .shape
            .dims()
            .iter()
            .zip(&padding)
            .map(|(dim, (before, after))| {
                dim.checked_add(*before)
                    .and_then(|x| x.checked_add(*after))
                    .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.push(
            Op::Pad {
                input,
                padding,
                fill,
            },
            Shape::new(dims),
            source.dtype,
        ))
    }

    /// Constant grouped padding with tinygrad-style signed crop support.
    ///
    /// Each pair is `(before, after)` in source axis order. Negative values
    /// crop first, then the remaining nonnegative padding is applied with the
    /// provided fill value. This intentionally covers tinygrad's default
    /// constant mode only; reflect, replicate, and circular require their
    /// own source-index constructions.
    pub fn pad_signed(
        &mut self,
        input: NodeId,
        padding: impl Into<Vec<(i64, i64)>>,
        fill: Scalar,
    ) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        let padding = padding.into();
        if padding.len() != shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "pad_signed",
                expected: shape.rank(),
                actual: padding.len(),
            });
        }
        let mut bounds = Vec::with_capacity(shape.rank());
        let mut positive = Vec::with_capacity(shape.rank());
        let mut cropped = false;
        let mut padded = false;
        for (axis, (&dimension, &(before, after))) in shape.dims().iter().zip(&padding).enumerate() {
            let dimension_i128 = dimension as i128;
            let start = (-(before as i128)).max(0);
            let end = (dimension_i128 + after as i128).min(dimension_i128);
            if end < 0 {
                return Err(Error::InvalidRandom {
                    reason: "signed padding crops beyond axis",
                });
            }
            let start = usize::try_from(start).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let end = usize::try_from(end).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            if start > end || end > dimension {
                return Err(Error::InvalidBounds {
                    axis,
                    start,
                    end,
                    dim: dimension,
                });
            }
            let before = usize::try_from(before.max(0))
                .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let after = usize::try_from(after.max(0))
                .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            let retained = end - start;
            retained
                .checked_add(before)
                .and_then(|value| value.checked_add(after))
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            cropped |= start != 0 || end != dimension;
            padded |= before != 0 || after != 0;
            bounds.push((start, end));
            positive.push((before, after));
        }
        let value = if cropped { self.shrink(input, bounds)? } else { input };
        if padded {
            self.pad(value, positive, fill)
        } else {
            Ok(value)
        }
    }

    /// Applies Python-style signed slices, including negative steps and flips.
    pub fn stride(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        let source = self.node(input)?;
        let slices = slices.into();
        if slices.len() != source.shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "stride",
                expected: source.shape.rank(),
                actual: slices.len(),
            });
        }
        let dims = slices
            .iter()
            .zip(source.shape.dims())
            .enumerate()
            .map(|(axis, (slice, dim))| {
                normalized_slice(*dim, *slice, axis).map(|(_, _, _, length)| length)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.push(Op::Stride { input, slices }, Shape::new(dims), source.dtype))
    }

    /// Alias for [`Graph::stride`], emphasizing ordinary slicing semantics.
    pub fn slice(&mut self, input: NodeId, slices: impl Into<Vec<Slice>>) -> Result<NodeId> {
        self.stride(input, slices)
    }

    /// Reverses each distinct signed axis through the existing checked stride
    /// view. An empty axis list is the same no-op as tinygrad's `flip(())`.
    pub fn flip(&mut self, input: NodeId, axes: impl Into<Vec<isize>>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let rank = shape.rank() as isize;
        let mut normalized = axes
            .into()
            .into_iter()
            .map(|axis| {
                let axis = if axis < 0 {
                    axis.checked_add(rank).ok_or(Error::InvalidAxis {
                        node: input,
                        axis: usize::MAX,
                        rank: rank as usize,
                    })?
                } else {
                    axis
                };
                if axis < 0 || axis >= rank {
                    return Err(Error::InvalidAxis {
                        node: input,
                        axis: usize::MAX,
                        rank: rank as usize,
                    });
                }
                Ok(axis as usize)
            })
            .collect::<Result<Vec<_>>>()?;
        normalized.sort_unstable();
        if normalized.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::InvalidRandom {
                reason: "flip axes must be unique",
            });
        }
        if normalized.is_empty() {
            return Ok(input);
        }
        let slices = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(axis, _)| Slice {
                start: None,
                stop: None,
                step: if normalized.binary_search(&axis).is_ok() {
                    -1
                } else {
                    1
                },
            })
            .collect::<Vec<_>>();
        self.stride(input, slices)
    }

    /// Splits a concrete axis into at most `chunks` ordered, contiguous
    /// shrink views. Uneven nonempty axes use the tinygrad tail rule; a
    /// zero-sized axis returns exactly `chunks` empty views.
    pub fn chunk(
        &mut self,
        input: NodeId,
        chunks: usize,
        axis: isize,
    ) -> Result<Vec<NodeId>> {
        if chunks == 0 {
            return Err(Error::InvalidRandom {
                reason: "chunk count must be positive",
            });
        }
        let shape = self.shape(input)?.clone();
        let rank = shape.rank() as isize;
        let axis = if axis < 0 {
            axis.checked_add(rank).ok_or(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            })?
        } else {
            axis
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        let axis = axis as usize;
        let axis_len = shape.dims()[axis];
        let ranges = if axis_len == 0 {
            vec![(0, 0); chunks]
        } else {
            let width = axis_len / chunks + if axis_len % chunks == 0 { 0 } else { 1 };
            (0..axis_len)
                .step_by(width)
                .map(|start| (start, start.saturating_add(width).min(axis_len)))
                .collect::<Vec<_>>()
        };
        // Every range is derived from the checked concrete source shape. Do
        // not construct a prefix of views until this entire inventory exists.
        let bounds = ranges
            .into_iter()
            .map(|(start, end)| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &size)| {
                        if dimension == axis {
                            (start, end)
                        } else {
                            (0, size)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        bounds
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// Splits a concrete axis into ordered contiguous shrink views, using the
    /// exact uniform-tail or explicit-coverage form selected by `sections`.
    pub fn split(
        &mut self,
        input: NodeId,
        sections: SplitSections,
        axis: isize,
    ) -> Result<Vec<NodeId>> {
        let shape = self.shape(input)?.clone();
        let rank = shape.rank() as isize;
        let axis = if axis < 0 {
            axis.checked_add(rank).ok_or(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            })?
        } else {
            axis
        };
        if axis < 0 || axis >= rank {
            return Err(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: rank as usize,
            });
        }
        let axis = axis as usize;
        let axis_len = shape.dims()[axis];
        let lengths = match sections {
            SplitSections::Uniform(size) => {
                if axis_len == 0 {
                    // tinygrad's `range(0, max(1, 0), max(1, size))`
                    // produces one empty section for any uniform size.
                    vec![0]
                } else if size == 0 {
                    return Err(Error::InvalidRandom {
                        reason: "uniform split size must be positive",
                    });
                } else {
                    (0..axis_len)
                        .step_by(size)
                        .map(|start| size.min(axis_len - start))
                        .collect()
                }
            }
            SplitSections::Explicit(lengths) => {
                let total = lengths.iter().try_fold(0usize, |total, &length| {
                    total.checked_add(length).ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                })?;
                if total != axis_len {
                    return Err(Error::InvalidBounds {
                        axis,
                        start: total,
                        end: total,
                        dim: axis_len,
                    });
                }
                lengths
            }
        };
        let mut start = 0usize;
        let ranges = lengths
            .into_iter()
            .map(|length| {
                let end = start
                    .checked_add(length)
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let range = (start, end);
                start = end;
                Ok(range)
            })
            .collect::<Result<Vec<_>>>()?;
        // Coverage and every endpoint are established before creating a view.
        // A rejected late section therefore cannot publish an earlier result.
        let bounds = ranges
            .into_iter()
            .map(|(start, end)| {
                shape
                    .dims()
                    .iter()
                    .enumerate()
                    .map(|(dimension, &size)| {
                        if dimension == axis {
                            (start, end)
                        } else {
                            (0, size)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        bounds
            .into_iter()
            .map(|bounds| self.shrink(input, bounds))
            .collect()
    }

    /// Concatenates at least two equally ranked tensors along `axis`.
    pub fn concat(&mut self, inputs: impl Into<Vec<NodeId>>, axis: usize) -> Result<NodeId> {
        let inputs = inputs.into();
        if inputs.len() < 2 {
            return Err(Error::InvalidConcat {
                axis,
                shapes: inputs
                    .iter()
                    .filter_map(|id| self.node(*id).ok().map(|n| n.shape.clone()))
                    .collect(),
            });
        }
        let first = self.node(inputs[0])?;
        if axis >= first.shape.rank() {
            return Err(Error::InvalidAxis {
                node: inputs[0],
                axis,
                rank: first.shape.rank(),
            });
        }
        let shape = first.shape.clone();
        let mut dtype = first.dtype;
        let mut total = 0usize;
        let shapes = inputs
            .iter()
            .map(|id| self.node(*id).map(|n| n.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        for (id, node_shape) in inputs.iter().zip(&shapes) {
            let node = self.node(*id)?;
            if node_shape.rank() != shape.rank()
                || node_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .any(|(i, dim)| i != axis && *dim != shape.dims()[i])
            {
                return Err(Error::InvalidConcat { axis, shapes });
            }
            total = total
                .checked_add(node_shape.dims()[axis])
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            dtype = dtype.promote(node.dtype);
        }
        let mut dims = shape.dims().to_vec();
        dims[axis] = total;
        let output = Shape::new(dims);
        output.numel()?;
        Ok(self.push(Op::Concat { inputs, axis }, output, dtype))
    }

    pub(crate) fn scatter_positions(
        &mut self,
        input: NodeId,
        shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        if starts.len() != shape.rank()
            || steps.len() != shape.rank()
            || source.shape.rank() != shape.rank()
        {
            return Err(Error::InvalidMovementRank {
                op: "scatter",
                expected: shape.rank(),
                actual: starts.len().min(steps.len()).min(source.shape.rank()),
            });
        }
        Ok(self.push(
            Op::ScatterPositions {
                input,
                shape: shape.clone(),
                starts,
                steps,
            },
            shape,
            source.dtype,
        ))
    }

    pub(crate) fn scatter_positions_vjp(
        &mut self,
        cotangent: NodeId,
        input_shape: Shape,
        starts: Vec<isize>,
        steps: Vec<isize>,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if starts.len() != input_shape.rank() || steps.len() != input_shape.rank() {
            return Err(Error::InvalidMovementRank {
                op: "scatter vjp",
                expected: input_shape.rank(),
                actual: starts.len().min(steps.len()),
            });
        }
        Ok(self.push(
            Op::ScatterPositionsVjp {
                cotangent,
                input_shape: input_shape.clone(),
                starts,
                steps,
            },
            input_shape,
            source.dtype,
        ))
    }

    /// Takes values from `input` at integer coordinates supplied by `index`.
    /// Index rank matches input rank and every non-axis index dimension must
    /// not exceed the corresponding input dimension. Negative indices are not
    /// accepted, matching tinygrad's gather contract.
    pub fn gather(&mut self, input: NodeId, index: NodeId, axis: usize) -> Result<NodeId> {
        let source = self.node(input)?;
        let index_node = self.node(index)?;
        validate_indexed("gather", source, index_node, axis)?;
        Ok(self.push(
            Op::Gather { input, index, axis },
            index_node.shape.clone(),
            source.dtype,
        ))
    }

    /// Signed-axis form of [`Self::gather`], matching tinygrad's public
    /// `gather(dim, index)` axis contract without changing the established
    /// unsigned Rust entry point.
    ///
    /// Axis normalization completes before the delegated gather validator can
    /// append a node. The existing gather route then validates index dtype,
    /// equal rank, and every non-axis extent before constructing the node.
    pub fn gather_signed(&mut self, input: NodeId, index: NodeId, axis: isize) -> Result<NodeId> {
        let rank = self.node(input)?.shape.rank();
        let axis = normalize_axes(input, rank, Some(vec![axis]))
            .map(|axes| axes[0])?;
        self.gather(input, index, axis)
    }

    /// Applies a checked, immutable mixed static index. Advanced indices are
    /// constant integer tensors represented by [`indexing::StaticIndex`];
    /// data-dependent boolean/nonzero indexing is intentionally not this API.
    pub fn static_index(
        &mut self,
        input: NodeId,
        specs: &[indexing::StaticIndex],
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let plan = indexing::StaticIndexPlan::new(source.shape.clone(), specs)?;
        Ok(self.push(
            Op::StaticIndex {
                input,
                plan: plan.clone(),
            },
            plan.output_shape().clone(),
            source.dtype,
        ))
    }

    pub(crate) fn static_index_grad(
        &mut self,
        cotangent: NodeId,
        input_shape: Shape,
        plan: indexing::StaticIndexPlan,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if source.shape != *plan.output_shape() {
            return Err(Error::InvalidIndex);
        }
        Ok(self.push(
            Op::StaticIndexGrad {
                cotangent,
                input_shape: input_shape.clone(),
                plan,
            },
            input_shape,
            source.dtype,
        ))
    }

    pub(crate) fn static_index_update_grad(
        &mut self,
        cotangent: NodeId,
        base_shape: Shape,
        value_shape: Shape,
        plan: indexing::StaticIndexPlan,
        wrt: super::StaticIndexUpdateWrt,
    ) -> Result<NodeId> {
        let source = self.node(cotangent)?;
        if source.dtype != DType::F32 || source.shape != base_shape {
            return Err(Error::NonDifferentiableIndexing(
                "static index update gradients require an F32 base cotangent",
            ));
        }
        if value_shape.broadcast_with(plan.output_shape()).as_ref() != Ok(plan.output_shape()) {
            return Err(Error::InvalidIndex);
        }
        let shape = match wrt {
            super::StaticIndexUpdateWrt::Base => base_shape.clone(),
            super::StaticIndexUpdateWrt::Value => value_shape.clone(),
        };
        Ok(self.push(
            Op::StaticIndexUpdateGrad {
                cotangent,
                base_shape,
                value_shape,
                plan,
                wrt,
            },
            shape,
            DType::F32,
        ))
    }

    /// Functional immutable replacement. RHS dtype must equal the snapshot
    /// dtype and broadcasts to the indexed result; duplicate target positions
    /// resolve in row-major update order with the final write winning.
    pub fn static_index_update(
        &mut self,
        base: NodeId,
        specs: &[indexing::StaticIndex],
        value: NodeId,
    ) -> Result<NodeId> {
        let base_node = self.node(base)?;
        let value_node = self.node(value)?;
        if value_node.dtype != base_node.dtype {
            return Err(Error::InvalidElementwiseDType {
                op: "static_index_update",
                actual: value_node.dtype,
            });
        }
        let plan = indexing::StaticIndexPlan::new(base_node.shape.clone(), specs)?;
        if value_node
            .shape
            .broadcast_with(plan.output_shape())
            .as_ref()
            != Ok(plan.output_shape())
        {
            return Err(Error::ShapeMismatch {
                op: "static_index_update",
                lhs: value_node.shape.clone(),
                rhs: plan.output_shape().clone(),
            });
        }
        Ok(self.push(
            Op::StaticIndexUpdate { base, value, plan },
            base_node.shape.clone(),
            base_node.dtype,
        ))
    }

    /// Replaces indexed base positions. Duplicate indices are deterministic:
    /// row-major later update coordinates win. Replacement scatter is
    /// deliberately non-differentiable; use [`Graph::scatter_add`] for a
    /// differentiable accumulation operation.
    pub fn scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, false)
    }

    /// Adds updates into indexed base positions. Duplicate coordinates are
    /// accumulated in row-major order and result dtype promotes base/updates.
    pub fn scatter_add(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Result<NodeId> {
        self.indexed_scatter(base, index, updates, axis, true)
    }

    fn indexed_scatter(
        &mut self,
        base: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
        add: bool,
    ) -> Result<NodeId> {
        let base_node = self.node(base)?;
        let index_node = self.node(index)?;
        let update_node = self.node(updates)?;
        validate_indexed("scatter", base_node, index_node, axis)?;
        if update_node.shape.rank() != index_node.shape.rank()
            || update_node
                .shape
                .dims()
                .iter()
                .zip(index_node.shape.dims())
                .any(|(update, index)| update < index)
        {
            return Err(Error::InvalidUpdateShape {
                index: index_node.shape.clone(),
                updates: update_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            },
            base_node.shape.clone(),
            base_node.dtype.promote(update_node.dtype),
        ))
    }

    /// Fixed-shape form of tinygrad's `masked_select(size=N)`. The mask must
    /// be bool and broadcastable to input; matches use row-major order.
    pub fn masked_select(
        &mut self,
        input: NodeId,
        mask: NodeId,
        size: usize,
        fill: Scalar,
    ) -> Result<NodeId> {
        let source = self.node(input)?;
        let mask_node = self.node(mask)?;
        if mask_node.dtype != DType::Bool {
            return Err(Error::InvalidLogicalDType {
                op: "masked_select",
                actual: mask_node.dtype,
            });
        }
        if mask_node.shape.broadcast_with(&source.shape).as_ref() != Ok(&source.shape) {
            return Err(Error::InvalidIndexedShape {
                op: "masked_select",
                input: source.shape.clone(),
                index: mask_node.shape.clone(),
            });
        }
        Ok(self.push(
            Op::MaskedSelect {
                input,
                mask,
                size,
                fill,
            },
            Shape::from([size]),
            source.dtype,
        ))
    }

    pub fn matmul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId> {
        let lhs_shape = &self.node(lhs)?.shape;
        let rhs_shape = &self.node(rhs)?.shape;
        let Some(shape) = matmul_shape(lhs_shape, rhs_shape) else {
            return Err(Error::InvalidMatmul {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
            });
        };
        shape.numel()?;
        let dtype = self.node(lhs)?.dtype.promote(self.node(rhs)?.dtype);
        Ok(self.push(Op::Matmul { lhs, rhs }, shape, dtype))
    }

    /// Applies tinygrad's functional linear composition.
    ///
    /// A rank-one weight is multiplied elementwise. Otherwise, the final two
    /// weight axes are exchanged before generalized matmul, so a conventional
    /// `[out_features, in_features]` weight contracts its final axis with the
    /// input's final axis. An optional `dtype` casts every operand before the
    /// composition, matching tinygrad's explicit accumulation-dtype path.
    pub fn linear(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        dtype: Option<DType>,
    ) -> Result<NodeId> {
        let input_shape = self.node(input)?.shape.clone();
        let weight_shape = self.node(weight)?.shape.clone();
        let bias_shape = bias.map(|bias| Ok(self.node(bias)?.shape.clone())).transpose()?;

        if weight_shape.rank() == 0 {
            return Err(Error::InvalidMatmul {
                lhs: input_shape,
                rhs: weight_shape,
            });
        }

        let (output_shape, weight_permutation) = if weight_shape.rank() == 1 {
            (input_shape.broadcast_with(&weight_shape)?, None)
        } else {
            let rank = weight_shape.rank();
            let mut permutation = (0..rank).collect::<Vec<_>>();
            permutation.swap(rank - 2, rank - 1);
            let mut transposed = weight_shape.dims().to_vec();
            transposed.swap(rank - 2, rank - 1);
            let transposed = Shape::new(transposed);
            let shape = matmul_shape(&input_shape, &transposed).ok_or_else(|| {
                Error::InvalidMatmul {
                    lhs: input_shape.clone(),
                    rhs: transposed,
                }
            })?;
            (shape, Some(permutation))
        };
        output_shape.numel()?;
        if let Some(bias_shape) = &bias_shape {
            output_shape.broadcast_with(bias_shape)?;
        }

        let input = dtype.map_or(Ok(input), |dtype| self.cast(input, dtype))?;
        let weight = dtype.map_or(Ok(weight), |dtype| self.cast(weight, dtype))?;
        let bias = bias
            .map(|bias| dtype.map_or(Ok(bias), |dtype| self.cast(bias, dtype)))
            .transpose()?;
        let output = match weight_permutation {
            Some(permutation) => {
                let weight = self.permute(weight, permutation)?;
                self.matmul(input, weight)?
            }
            None => self.mul(input, weight)?,
        };
        bias.map_or(Ok(output), |bias| self.add(output, bias))
    }

    /// Adds a static dense Einstein summation node with NumPy/tinygrad-style
    /// subscript grammar, including ellipses and repeated-label diagonals.
    pub fn einsum(&mut self, equation: &str, inputs: &[NodeId]) -> Result<NodeId> {
        let shapes = inputs
            .iter()
            .map(|id| Ok(self.node(*id)?.shape.clone()))
            .collect::<Result<Vec<_>>>()?;
        let plan = EinsumPlan::parse(equation, &shapes)?;
        let dtype = inputs.iter().try_fold(DType::Bool, |dtype, id| {
            Ok::<_, Error>(dtype.promote(self.node(*id)?.dtype))
        })?;
        Ok(self.push(
            Op::Einsum {
                inputs: inputs.to_vec(),
                plan: plan.clone(),
            },
            plan.output_shape(),
            dtype,
        ))
    }

    pub(crate) fn einsum_grad(
        &mut self,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
    ) -> Result<NodeId> {
        let target_id = *inputs.get(target).ok_or(Error::InvalidIndex)?;
        let target_node = self.node(target_id)?;
        let output_shape = plan.output_shape();
        if self.node(upstream)?.shape != output_shape {
            return Err(Error::ShapeMismatch {
                op: "einsum gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output_shape,
            });
        }
        if !target_node.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "einsum gradients require floating point target tensors",
            ));
        }
        Ok(self.push(
            Op::EinsumGrad {
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
            },
            target_node.shape.clone(),
            target_node.dtype,
        ))
    }

    pub(crate) fn einsum_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        inputs: &[NodeId],
        plan: EinsumPlan,
        target: usize,
        wrt: usize,
    ) -> Result<NodeId> {
        let output = if wrt == inputs.len() {
            plan.output_shape()
        } else {
            self.node(*inputs.get(wrt).ok_or(Error::InvalidIndex)?)?
                .shape
                .clone()
        };
        Ok(self.push(
            Op::EinsumGradVjp {
                cotangent,
                upstream,
                inputs: inputs.to_vec(),
                plan,
                target,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    /// Adds a first-class NCHW/OIHW 2D convolution node.
    pub fn conv2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
    ) -> Result<NodeId> {
        let input_node = self.node(input)?;
        let weight_node = self.node(weight)?;
        let shape = conv2d_shape(&input_node.shape, &weight_node.shape, options)?;
        if let Some(bias) = bias {
            let b = self.node(bias)?;
            if b.shape != Shape::from([weight_node.shape.dims()[0]]) {
                return Err(Error::InvalidConv2d {
                    input: input_node.shape.clone(),
                    weight: weight_node.shape.clone(),
                    reason: "bias must be [output_channels]",
                });
            }
        }
        let mut dtype = input_node.dtype.promote(weight_node.dtype);
        if let Some(bias) = bias {
            dtype = dtype.promote(self.node(bias)?.dtype);
        }
        Ok(self.push(
            Op::Conv2d {
                input,
                weight,
                bias,
                options,
            },
            shape,
            dtype,
        ))
    }
    pub fn conv_transpose2d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
    ) -> Result<NodeId> {
        let x = self.node(input)?;
        let w = self.node(weight)?;
        let shape = conv_transpose2d_shape(&x.shape, &w.shape, options)?;
        if let Some(b) = bias
            && self.node(b)?.shape != Shape::from([shape.dims()[1]])
        {
            return Err(Error::InvalidConv2d {
                input: x.shape.clone(),
                weight: w.shape.clone(),
                reason: "bias must be [output_channels]",
            });
        }
        let mut dtype = x.dtype.promote(w.dtype);
        if let Some(b) = bias {
            dtype = dtype.promote(self.node(b)?.dtype);
        }
        Ok(self.push(
            Op::ConvTranspose2d {
                input,
                weight,
                bias,
                options,
            },
            shape,
            dtype,
        ))
    }
    /// Lowers NCL/IOK transpose convolution through the singleton-height 2D core.
    pub fn conv_transpose1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose1dOptions,
    ) -> Result<NodeId> {
        let x = self.node(input)?.shape.clone();
        let w = self.node(weight)?.shape.clone();
        if x.rank() != 3
            || w.rank() != 3
            || options.stride == 0
            || options.dilation == 0
            || options.output_padding >= options.stride
        {
            return Err(Error::InvalidConv2d {
                input: x.clone(),
                weight: w.clone(),
                reason: "invalid 1d transpose convolution geometry",
            });
        }
        let x4_shape = Shape::new([x.dims()[0], x.dims()[1], 1, x.dims()[2]]);
        let w4_shape = Shape::new([w.dims()[0], w.dims()[1], 1, w.dims()[2]]);
        let options_2d = ConvTranspose2dOptions {
            groups: options.groups,
            stride: [1, options.stride],
            dilation: [1, options.dilation],
            padding: [0, 0, options.padding[0], options.padding[1]],
            output_padding: [0, options.output_padding],
        };
        let output_2d = conv_transpose2d_shape(&x4_shape, &w4_shape, options_2d)?;
        if let Some(bias) = bias
            && self.node(bias)?.shape != Shape::from([output_2d.dims()[1]])
        {
            return Err(Error::InvalidConv2d {
                input: x.clone(),
                weight: w.clone(),
                reason: "bias must be [output_channels]",
            });
        }
        let x4 = self.reshape(
            input,
            x4_shape,
        )?;
        let w4 = self.reshape(
            weight,
            w4_shape,
        )?;
        let y4 = self.conv_transpose2d(x4, w4, bias, options_2d)?;
        let y = self.node(y4)?.shape.clone();
        self.reshape(y4, Shape::new([y.dims()[0], y.dims()[1], y.dims()[3]]))
    }
    pub(crate) fn conv_transpose2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output =
            conv_transpose2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing transpose bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let n = self.node(node)?;
        if !n.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "transpose convolution gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            n.shape.clone(),
            n.dtype,
        ))
    }

    pub(crate) fn conv2d_grad(
        &mut self,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
    ) -> Result<NodeId> {
        let output = conv2d_shape(&self.node(input)?.shape, &self.node(weight)?.shape, options)?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "conv2d gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target_node = match target {
            0 => input,
            1 => weight,
            2 => bias.ok_or(Error::NonDifferentiableIndexing("missing conv2d bias"))?,
            _ => return Err(Error::InvalidIndex),
        };
        let target_data = self.node(target_node)?;
        if !target_data.dtype.is_float() {
            return Err(Error::NonDifferentiableIndexing(
                "conv2d gradients require floating point tensors",
            ));
        }
        Ok(self.push(
            Op::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                options,
                target,
            },
            target_data.shape.clone(),
            target_data.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: Conv2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_transpose2d_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        options: ConvTranspose2dOptions,
        target: u8,
        wrt: u8,
    ) -> Result<NodeId> {
        let shape = conv_vjp_shape(self, upstream, input, weight, bias, wrt)?;
        Ok(self.push(
            Op::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                options,
                target,
                wrt,
            },
            shape,
            self.node(cotangent)?.dtype,
        ))
    }

    pub(crate) fn matmul_grad(
        &mut self,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
    ) -> Result<NodeId> {
        let lhs_shape = self.node(lhs)?.shape.clone();
        let rhs_shape = self.node(rhs)?.shape.clone();
        let output = matmul_shape(&lhs_shape, &rhs_shape).ok_or(Error::InvalidMatmul {
            lhs: lhs_shape,
            rhs: rhs_shape,
        })?;
        if self.node(upstream)?.shape != output {
            return Err(Error::ShapeMismatch {
                op: "matmul gradient",
                lhs: self.node(upstream)?.shape.clone(),
                rhs: output,
            });
        }
        let target = if lhs_gradient { lhs } else { rhs };
        let shape = self.node(target)?.shape.clone();
        let dtype = self.node(target)?.dtype;
        Ok(self.push(
            Op::MatmulGrad {
                upstream,
                lhs,
                rhs,
                lhs_gradient,
            },
            shape,
            dtype,
        ))
    }

    pub(crate) fn matmul_grad_vjp(
        &mut self,
        cotangent: NodeId,
        upstream: NodeId,
        lhs: NodeId,
        rhs: NodeId,
        lhs_gradient: bool,
        wrt: u8,
    ) -> Result<NodeId> {
        let output = match wrt {
            0 => matmul_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape).ok_or_else(|| {
                Error::InvalidMatmul {
                    lhs: self.node(lhs).unwrap().shape.clone(),
                    rhs: self.node(rhs).unwrap().shape.clone(),
                }
            })?,
            1 => self.node(lhs)?.shape.clone(),
            2 => self.node(rhs)?.shape.clone(),
            _ => return Err(Error::InvalidIndex),
        };
        Ok(self.push(
            Op::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                lhs_gradient,
                wrt,
            },
            output,
            self.node(cotangent)?.dtype,
        ))
    }

    pub fn shape(&self, id: NodeId) -> Result<&Shape> {
        Ok(&self.node(id)?.shape)
    }

    pub fn dtype(&self, id: NodeId) -> Result<DType> {
        Ok(self.node(id)?.dtype)
    }

    /// Returns the typed operation for inspection without exposing graph
    /// storage internals.
    pub fn op(&self, id: NodeId) -> Result<&Op> {
        Ok(&self.node(id)?.op)
    }

    pub fn trace(&self, output: NodeId) -> Result<CompileTrace> {
        self.node(output)?;
        let steps = self.nodes[..=output.index()]
            .iter()
            .enumerate()
            .map(|(id, node)| TraceStep {
                node: NodeId(id),
                operation: node.op.label(),
                shape: node.shape.clone(),
                dtype: node.dtype,
            })
            .collect();
        Ok(CompileTrace { output, steps })
    }

    pub(crate) fn push(&mut self, op: Op, shape: Shape, dtype: DType) -> NodeId {
        let requires_grad =
            self.grad_enabled && dtype.is_float() && self.op_inputs_require_grad(&op);
        self.push_with_grad(op, shape, dtype, requires_grad)
    }

    fn push_with_grad(
        &mut self,
        op: Op,
        shape: Shape,
        dtype: DType,
        requires_grad: bool,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            op,
            shape,
            dtype,
            requires_grad,
        });
        id
    }

    fn op_inputs_require_grad(&self, op: &Op) -> bool {
        let mut tracked = |id: NodeId| {
            self.nodes
                .get(id.index())
                .is_some_and(|node| node.requires_grad)
        };
        match op {
            Op::Input { .. }
            | Op::Constant(_)
            | Op::Random { .. }
            | Op::RandomPermutation { .. }
            | Op::Sort { .. } => false,
            Op::Cast { input, .. }
            | Op::Unary { input, .. }
            | Op::Reduce { input, .. }
            | Op::ArgReduce { input, .. }
            | Op::SumTo { input, .. }
            | Op::Reshape { input, .. }
            | Op::Permute { input, .. }
            | Op::Expand { input, .. }
            | Op::Shrink { input, .. }
            | Op::Pad { input, .. }
            | Op::Stride { input, .. }
            | Op::ScatterPositions { input, .. }
            | Op::Gather { input, .. }
            | Op::StaticIndex { input, .. }
            | Op::MaskedSelect { input, .. } => tracked(*input),
            Op::StaticIndexUpdate { base, value, .. } => tracked(*base) || tracked(*value),
            Op::ScatterPositionsVjp { cotangent, .. }
            | Op::StaticIndexGrad { cotangent, .. }
            | Op::StaticIndexUpdateGrad { cotangent, .. } => tracked(*cotangent),
            Op::Detach { .. } => false,
            Op::Binary { lhs, rhs, .. } | Op::Compare { lhs, rhs, .. } => {
                tracked(*lhs) || tracked(*rhs)
            }
            Op::Logical { lhs, rhs, .. } => tracked(*lhs) || rhs.is_some_and(tracked),
            Op::Select {
                on_true, on_false, ..
            } => tracked(*on_true) || tracked(*on_false),
            Op::ReduceGrad {
                input, upstream, ..
            } => tracked(*input) || tracked(*upstream),
            Op::ReduceGradVjp {
                cotangent,
                input,
                upstream,
                ..
            } => tracked(*cotangent) || tracked(*input) || tracked(*upstream),
            Op::Concat { inputs, .. } | Op::Einsum { inputs, .. } => {
                inputs.iter().copied().any(&mut tracked)
            }
            Op::Scatter { base, updates, .. } => tracked(*base) || tracked(*updates),
            Op::Matmul { lhs, rhs } => tracked(*lhs) || tracked(*rhs),
            Op::EinsumGrad {
                upstream, inputs, ..
            } => tracked(*upstream) || inputs.iter().copied().any(&mut tracked),
            Op::EinsumGradVjp {
                cotangent,
                upstream,
                inputs,
                ..
            } => {
                tracked(*cotangent)
                    || tracked(*upstream)
                    || inputs.iter().copied().any(&mut tracked)
            }
            Op::MatmulGrad {
                upstream, lhs, rhs, ..
            } => tracked(*upstream) || tracked(*lhs) || tracked(*rhs),
            Op::MatmulGradVjp {
                cotangent,
                upstream,
                lhs,
                rhs,
                ..
            } => tracked(*cotangent) || tracked(*upstream) || tracked(*lhs) || tracked(*rhs),
            Op::Conv2d {
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2d {
                input,
                weight,
                bias,
                ..
            } => tracked(*input) || tracked(*weight) || bias.is_some_and(tracked),
            Op::Conv2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2dGrad {
                upstream,
                input,
                weight,
                bias,
                ..
            } => {
                tracked(*upstream)
                    || tracked(*input)
                    || tracked(*weight)
                    || bias.is_some_and(tracked)
            }
            Op::Conv2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            }
            | Op::ConvTranspose2dGradVjp {
                cotangent,
                upstream,
                input,
                weight,
                bias,
                ..
            } => {
                tracked(*cotangent)
                    || tracked(*upstream)
                    || tracked(*input)
                    || tracked(*weight)
                    || bias.is_some_and(tracked)
            }
        }
    }

    pub(crate) fn node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(id.index()).ok_or(Error::UnknownNode(id))
    }

    pub(super) fn broadcast_shape(&self, lhs: NodeId, rhs: NodeId) -> Result<Shape> {
        let lhs = &self.node(lhs)?.shape;
        let rhs = &self.node(rhs)?.shape;
        lhs.broadcast_with(rhs)
    }
}
