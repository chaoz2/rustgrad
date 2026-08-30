//! Source-literal gather lowering shared by tinygrad composite helpers.
//!
//! Tinygrad's public `gather` is not the raw indexed movement operation: it
//! crops non-index lanes, constructs a one-hot predicate, selects a typed
//! zero for invalid labels, and then performs a storage-width Sum.  Keeping
//! this descriptor independent of a particular loss helper also makes zero
//! class extents and invalid interpolation coordinates faithful without a
//! backend indexing exception.

use super::{Graph, NodeId, shape::normalize_axes};
use crate::{DType, Error, ReduceKind, ReductionDType, Result, Scalar, Shape, TensorData};

#[derive(Clone, Debug)]
pub(crate) struct SourceGatherPlan {
    bounds: Vec<(usize, usize)>,
    permutation: Vec<usize>,
    classes: usize,
    zero: TensorData,
    output_shape: Shape,
}

fn invalid(reason: &'static str) -> Error {
    Error::InvalidRandom { reason }
}

pub(crate) fn source_gather_plan(
    value_shape: &Shape,
    value_dtype: DType,
    index_shape: &Shape,
    index_dtype: DType,
    axis: usize,
) -> Result<SourceGatherPlan> {
    if !index_dtype.is_integer() {
        return Err(invalid("source gather requires integer indices"));
    }
    if value_shape.rank() != index_shape.rank() || axis >= value_shape.rank() {
        return Err(invalid("source gather rank or axis mismatch"));
    }
    let mut bounds = Vec::with_capacity(value_shape.rank());
    for (dimension, (&source, &requested)) in value_shape
        .dims()
        .iter()
        .zip(index_shape.dims())
        .enumerate()
    {
        if dimension != axis && requested > source {
            return Err(invalid("source gather index extent exceeds data"));
        }
        bounds.push(if dimension == axis {
            (0, source)
        } else {
            (0, requested)
        });
    }
    let rank = value_shape.rank();
    let mut moved_dims = bounds.iter().map(|(_, end)| *end).collect::<Vec<_>>();
    moved_dims.push(1);
    let mut permutation = (0..=rank).collect::<Vec<_>>();
    permutation.swap(axis, rank);
    let mut values_shape = moved_dims.clone();
    values_shape.swap(axis, rank);
    let values_shape = Shape::new(values_shape);
    let mut select_dims = index_shape.dims().to_vec();
    select_dims.push(value_shape.dims()[axis]);
    let select_shape = Shape::new(select_dims);
    let zero = TensorData::scalar_with_dtype(Scalar::I(0), value_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    for (shape, dtype) in [
        (value_shape, value_dtype),
        (index_shape, index_dtype),
        (&Shape::new(moved_dims), value_dtype),
        (&values_shape, value_dtype),
        (&select_shape, DType::Bool),
        (&select_shape, value_dtype),
        (index_shape, value_dtype),
        (zero.shape(), zero.dtype()),
    ] {
        extent(shape, dtype)?;
    }
    if values_shape.broadcast_with(&select_shape)? != select_shape
        || select_shape.broadcast_with(zero.shape())? != select_shape
        || zero.dtype() != value_dtype
    {
        return Err(invalid("source gather typed zero"));
    }
    Ok(SourceGatherPlan {
        bounds,
        permutation,
        classes: value_shape.dims()[axis],
        zero,
        output_shape: index_shape.clone(),
    })
}

pub(crate) fn lower_source_gather(
    graph: &mut Graph,
    value: NodeId,
    index: NodeId,
    plan: SourceGatherPlan,
) -> Result<NodeId> {
    let cropped = graph.shrink(value, plan.bounds)?;
    let expanded = graph.unsqueeze(cropped, -1)?;
    let values = graph.permute(expanded, plan.permutation)?;
    let predicate = graph.one_hot_bool(index, plan.classes)?;
    let zero = graph.constant(plan.zero);
    let selected = graph.select(predicate, values, zero)?;
    let dtype = graph.dtype(value)?;
    let output = graph.reduce_with_dtypes(
        selected,
        ReduceKind::Sum,
        Some(vec![-1]),
        false,
        ReductionDType::new(dtype, dtype),
    )?;
    debug_assert_eq!(
        graph.shape(output).expect("source gather preflighted"),
        &plan.output_shape
    );
    Ok(output)
}

pub(crate) fn source_gather(
    graph: &mut Graph,
    value: NodeId,
    index: NodeId,
    axis: usize,
) -> Result<NodeId> {
    let value_node = graph.node(value)?;
    let index_node = graph.node(index)?;
    let plan = source_gather_plan(
        &value_node.shape,
        value_node.dtype,
        &index_node.shape,
        index_node.dtype,
        axis,
    )?;
    lower_source_gather(graph, value, index, plan)
}

impl Graph {
    /// Source-literal public tinygrad `Tensor.gather(dim, index)`.
    ///
    /// This deliberately differs from raw [`Graph::gather`]: invalid live
    /// labels become all-false one-hot lanes selected to a typed zero rather
    /// than backend indexing errors. The clone rehearsal covers every later
    /// movement, lazy range, Select, and explicit storage-width Sum before a
    /// live constant or node is appended.
    pub fn gather_tinygrad(&mut self, value: NodeId, dim: isize, index: NodeId) -> Result<NodeId> {
        let value_node = self.node(value)?;
        let index_node = self.node(index)?;
        let axis = normalize_axes(value, value_node.shape.rank(), Some(vec![dim]))?[0];
        let plan = source_gather_plan(
            &value_node.shape,
            value_node.dtype,
            &index_node.shape,
            index_node.dtype,
            axis,
        )?;
        let mut rehearsal = self.clone();
        let rehearsed = lower_source_gather(&mut rehearsal, value, index, plan.clone())?;
        let output_shape = rehearsal.shape(rehearsed)?.clone();
        let output_dtype = rehearsal.dtype(rehearsed)?;
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        let output = lower_source_gather(self, value, index, plan)?;
        debug_assert_eq!(
            self.shape(output).expect("source gather preflighted"),
            &output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("source gather preflighted"),
            output_dtype
        );
        Ok(output)
    }
}
