//! Source-literal concrete interpolation for tinygrad's public Tensor helper.

use super::{Graph, NodeId, source_gather};
use crate::{DType, Error, Result, Scalar, Shape};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InterpolateMode {
    Linear,
    Nearest,
    NearestExact,
}

impl InterpolateMode {
    fn parse(mode: &str) -> Result<Self> {
        match mode {
            "linear" => Ok(Self::Linear),
            "nearest" => Ok(Self::Nearest),
            "nearest-exact" => Ok(Self::NearestExact),
            _ => Err(Error::InvalidRandom {
                reason: "interpolate mode must be linear, nearest, or nearest-exact",
            }),
        }
    }
}

/// Descriptor-first stage for one trailing source interpolation iteration.
///
/// Coordinate dtypes depend on the source-default range endpoint policy and
/// the preceding linear `lerp`; those descriptor details are rehearsed on a
/// cloned graph by [`interpolate_plan`] before the live graph is touched.
/// The stored movement descriptors make the source right-to-left stage order
/// explicit and independently preflight all source/output byte extents.
#[derive(Clone, Debug)]
struct InterpolateStagePlan {
    axis: usize,
    input_shape: Shape,
    output_shape: Shape,
    vector_shape: Shape,
    reshape_shape: Shape,
    input_extent_dtype: DType,
}

#[derive(Clone, Debug)]
struct InterpolatePlan {
    input_dtype: DType,
    stages: Vec<InterpolateStagePlan>,
    mode: InterpolateMode,
    align_corners: bool,
    output_shape: Shape,
}

fn extent(shape: &Shape, dtype: DType) -> Result<()> {
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
        .map(|_| ())
}

fn i64_extent(value: usize, shape: &Shape) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::ShapeOverflow(shape.clone()))
}

fn interpolate_plan(
    graph: &Graph,
    input: NodeId,
    size: &[usize],
    mode: InterpolateMode,
    align_corners: bool,
) -> Result<InterpolatePlan> {
    let node = graph.node(input)?;
    if size.is_empty() || size.len() > node.shape.rank() {
        return Err(Error::InvalidRandom {
            reason: "interpolate size must be a nonempty trailing dimension tuple",
        });
    }
    if align_corners && mode != InterpolateMode::Linear {
        return Err(Error::InvalidRandom {
            reason: "align_corners is only valid for linear interpolation",
        });
    }
    extent(&node.shape, node.dtype)?;
    let rank = node.shape.rank();
    let mut current = node.shape.clone();
    let mut stages = Vec::with_capacity(size.len());
    for offset in 0..size.len() {
        let axis = rank - 1 - offset;
        let out = size[size.len() - 1 - offset];
        let mut output_dims = current.dims().to_vec();
        output_dims[axis] = out;
        let output_shape = Shape::new(output_dims);
        let vector_shape = Shape::new([out]);
        let mut reshape_dims = vec![1; rank];
        reshape_dims[axis] = out;
        let reshape_shape = Shape::new(reshape_dims);
        // Exercise every static product now. The clone rehearsal below adds
        // the source LUB/cast/reduction descriptors that depend on the
        // evolving stage dtype.
        extent(&current, node.dtype)?;
        extent(&vector_shape, DType::I32)?;
        extent(&reshape_shape, DType::I32)?;
        extent(&output_shape, node.dtype)?;
        stages.push(InterpolateStagePlan {
            axis,
            input_shape: current.clone(),
            output_shape: output_shape.clone(),
            vector_shape,
            reshape_shape,
            input_extent_dtype: node.dtype,
        });
        current = output_shape;
    }

    // A private rehearsal is the complete descriptor pass for the composed
    // scalar arithmetic, lazy ranges, source gathers and Lerp stages.  It is
    // intentionally run on a clone before even a scalar constant can publish
    // into the caller's graph, just as the QR/Newton-Schulz plans do.
    let mut rehearsal = graph.clone();
    let output = lower_interpolate(&mut rehearsal, input, &stages, mode, align_corners)?;
    let output_shape = rehearsal.shape(output)?.clone();
    let stage_output_dtype = rehearsal.dtype(output)?;
    extent(&output_shape, stage_output_dtype)?;
    // The public source ends with `x.cast(self.dtype)` even after a linear
    // stage widened its work storage.
    extent(&output_shape, node.dtype)?;
    Ok(InterpolatePlan {
        input_dtype: node.dtype,
        stages,
        mode,
        align_corners,
        output_shape,
    })
}

fn expand_index(graph: &mut Graph, index: NodeId, stage: &InterpolateStagePlan) -> Result<NodeId> {
    let index = graph.reshape(index, stage.reshape_shape.clone())?;
    graph.expand(index, stage.output_shape.clone())
}

fn lower_linear_stage(
    graph: &mut Graph,
    value: NodeId,
    stage: &InterpolateStagePlan,
    align_corners: bool,
) -> Result<NodeId> {
    let input_size = i64_extent(stage.input_shape.dims()[stage.axis], &stage.input_shape)?;
    let output_size = i64_extent(stage.output_shape.dims()[stage.axis], &stage.output_shape)?;
    let arr = graph.lazy_arange_default_int(0, output_size, 1)?;
    let (num, denominator) = if align_corners {
        (
            graph.mul_scalar(
                arr,
                Scalar::I(
                    input_size
                        .checked_sub(1)
                        .ok_or_else(|| Error::ShapeOverflow(stage.input_shape.clone()))?,
                ),
            )?,
            output_size
                .checked_sub(1)
                .ok_or_else(|| Error::ShapeOverflow(stage.output_shape.clone()))?,
        )
    } else {
        let twice = graph.mul_scalar(arr, Scalar::I(2))?;
        let centered = graph.add_scalar(twice, Scalar::I(1))?;
        let scaled = graph.mul_scalar(centered, Scalar::I(input_size))?;
        (
            graph.sub_scalar(scaled, Scalar::I(output_size))?,
            output_size
                .checked_mul(2)
                .ok_or_else(|| Error::ShapeOverflow(stage.output_shape.clone()))?,
        )
    };
    let upper = input_size
        .checked_sub(1)
        .and_then(|value| value.checked_mul(denominator))
        .ok_or_else(|| Error::ShapeOverflow(stage.input_shape.clone()))?;
    let num = graph.clamp_with_scalars(num, Some(Scalar::I(0)), Some(Scalar::I(upper)))?;
    let low = graph.floor_div_scalar(num, Scalar::I(denominator))?;
    let high_numerator = graph.add_scalar(
        num,
        Scalar::I(
            denominator
                .checked_sub(1)
                .ok_or_else(|| Error::ShapeOverflow(stage.output_shape.clone()))?,
        ),
    )?;
    let high = graph.floor_div_scalar(high_numerator, Scalar::I(denominator))?;
    let percentage = graph.modulo_scalar(num, Scalar::I(denominator))?;
    let percentage = graph.cast(percentage, DType::F32)?;
    let percentage = graph.div_scalar(percentage, Scalar::I(denominator))?;
    let low = expand_index(graph, low, stage)?;
    let high = expand_index(graph, high, stage)?;
    let percentage = expand_index(graph, percentage, stage)?;
    let low = source_gather(graph, value, low, stage.axis)?;
    let high = source_gather(graph, value, high, stage.axis)?;
    graph.lerp(low, high, percentage)
}

fn lower_nearest_stage(
    graph: &mut Graph,
    value: NodeId,
    stage: &InterpolateStagePlan,
    exact: bool,
) -> Result<NodeId> {
    let input_size = stage.input_shape.dims()[stage.axis];
    let output_size = stage.output_shape.dims()[stage.axis];
    if output_size == 0 {
        // Source performs Python `in_sz / size` before creating the F32
        // arange, so nearest modes raise for a zero output extent.
        return Err(Error::InvalidRandom {
            reason: "nearest interpolate requires nonzero output extent",
        });
    }
    let range = graph.lazy_arange_f32(output_size)?;
    let scale = (input_size as f64) / (output_size as f64);
    // `nearest-exact` is literally `scale * (arr + .5)`: adding the half
    // before the weak scale commitment matters at F32 rounding boundaries.
    let range = if exact {
        graph.add_scalar(range, Scalar::F(0.5))?
    } else {
        range
    };
    let coordinate = graph.mul_scalar(range, Scalar::F(scale))?;
    let index = graph.cast(coordinate, DType::I32)?;
    let index = expand_index(graph, index, stage)?;
    source_gather(graph, value, index, stage.axis)
}

fn lower_interpolate(
    graph: &mut Graph,
    input: NodeId,
    stages: &[InterpolateStagePlan],
    mode: InterpolateMode,
    align_corners: bool,
) -> Result<NodeId> {
    let mut value = input;
    for stage in stages {
        debug_assert_eq!(
            graph.shape(value).expect("interpolate stage preflighted"),
            &stage.input_shape
        );
        let _ = stage.vector_shape.rank();
        let _ = stage.input_extent_dtype;
        value = match mode {
            InterpolateMode::Linear => lower_linear_stage(graph, value, stage, align_corners)?,
            InterpolateMode::Nearest => lower_nearest_stage(graph, value, stage, false)?,
            InterpolateMode::NearestExact => lower_nearest_stage(graph, value, stage, true)?,
        };
        debug_assert_eq!(
            graph.shape(value).expect("interpolate stage preflighted"),
            &stage.output_shape
        );
    }
    Ok(value)
}

impl Graph {
    /// Checked-in tinygrad's concrete trailing-axis interpolation helper.
    pub fn interpolate(
        &mut self,
        input: NodeId,
        size: impl AsRef<[usize]>,
        mode: &str,
        align_corners: bool,
    ) -> Result<NodeId> {
        let mode = InterpolateMode::parse(mode)?;
        let plan = interpolate_plan(self, input, size.as_ref(), mode, align_corners)?;
        let output = lower_interpolate(self, input, &plan.stages, plan.mode, plan.align_corners)?;
        let output = if self.dtype(output)? == plan.input_dtype {
            output
        } else {
            self.cast(output, plan.input_dtype)?
        };
        debug_assert_eq!(
            self.shape(output).expect("interpolate preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("interpolate preflighted"),
            plan.input_dtype
        );
        Ok(output)
    }

    /// Checked-in tinygrad's parameter-default interpolation surface.
    pub fn interpolate_default(
        &mut self,
        input: NodeId,
        size: impl AsRef<[usize]>,
    ) -> Result<NodeId> {
        self.interpolate(input, size, "linear", false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, NodeId, Op};

    #[test]
    fn interpolate_is_staged_source_literal_and_scalar_backed() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3, 4], DType::F16);
        let output = graph
            .interpolate(input, [5, 6], "linear", false)
            .expect("concrete source interpolation plans");
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 5, 6]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        // Each linear source stage has two one-hot Select gathers followed by
        // its live-weight Lerp. No dense coordinate payload may appear.
        assert!(
            graph
                .nodes
                .iter()
                .filter(|node| matches!(&node.op, Op::Select { .. }))
                .count()
                >= 4
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(&node.op, Op::Cast { .. }))
        );
        assert!(
            graph
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    Op::Constant(data) => Some(data.len()),
                    _ => None,
                })
                .all(|len| len == 1)
        );
        let loss = graph.sum_all(output).expect("interpolation output reduces");
        let gradient = graph
            .grad(loss, input)
            .expect("source gather and lerp compose");
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([2, 3, 4]));
    }

    #[test]
    fn interpolate_modes_and_storage_boundaries_are_checked_before_publication() {
        for (mode, align, dtype) in [
            ("linear", true, DType::BF16),
            ("nearest", false, DType::F32),
            ("nearest-exact", false, DType::F64),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("x", [2, 3], dtype);
            let output = graph.interpolate(input, [5], mode, align).unwrap();
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 5]));
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert!(
                graph
                    .nodes
                    .iter()
                    .filter_map(|node| match &node.op {
                        Op::Constant(data) => Some(data.len()),
                        _ => None,
                    })
                    .all(|len| len == 1)
            );
        }

        let mut integer = Graph::new();
        let input = integer.input_dtype("x", [3], DType::I16);
        let output = integer.interpolate(input, [7], "linear", false).unwrap();
        assert_eq!(integer.dtype(output).unwrap(), DType::I16);
        let default = integer.interpolate_default(input, [7]).unwrap();
        assert_eq!(integer.shape(default).unwrap(), &Shape::new([7]));

        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let nodes = scalar.node_count();
        assert!(scalar.interpolate(input, [1], "linear", false).is_err());
        assert_eq!(scalar.node_count(), nodes);

        let mut invalid = Graph::new();
        let input = invalid.input("x", [2]);
        let nodes = invalid.node_count();
        assert!(invalid.interpolate(input, [], "linear", false).is_err());
        assert!(invalid.interpolate(input, [2], "bad", false).is_err());
        assert!(invalid.interpolate(input, [2], "nearest", true).is_err());
        assert_eq!(invalid.node_count(), nodes);
    }

    #[test]
    fn interpolate_preserves_empty_source_gather_and_late_overflow_atomicity() {
        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0, 2], DType::F32);
        let output = empty
            .interpolate(input, [3], "nearest-exact", false)
            .unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
        let input = empty.input_dtype("linear_empty", [2], DType::F32);
        let output = empty.interpolate(input, [0], "linear", false).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));

        // The input itself fits but resizing the final stage does not. The
        // clone rehearsal rejects before a range, source-gather Select, or
        // final Cast is appended to the caller graph.
        let mut overflow = Graph::new();
        let input = overflow.input_dtype("x", [usize::MAX / 8, 1], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.interpolate(input, [3], "nearest", false),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
        assert!(overflow.node(NodeId(0)).is_ok());
    }
}
