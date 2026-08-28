use super::{shape::normalize_axes, Graph, NodeId, Op, RandomKind, RandomStream, RollDims, RollShifts};
use crate::random::reserve;
use crate::{
    DType, Error, ExpandExtent, ReshapeExtent, Result, Scalar, Shape, ShrinkRange, TensorData,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct StreamRegistry {
    seed: u64,
    counters: BTreeMap<u32, [u32; 2]>,
}

#[derive(Clone)]
pub(crate) struct LazyArangePlan {
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    pub(crate) step: TensorData,
    pub(crate) offset: TensorData,
}

/// Whole-operation descriptor for tinygrad's public `Tensor.one_hot`.
///
/// The literal first unsqueezes integer indices, compares them with a
/// source-default integer arange, then selects strong I32 one/zero scalars.
/// Keep every descriptor and storage boundary here so no movement, range, or
/// predicate node can leak from a malformed late shape or byte extent.
pub(crate) struct OneHotPlan {
    value_shape: Shape,
    range: LazyArangePlan,
    class_shape: Shape,
    comparison_dtype: DType,
    output_shape: Shape,
    one: Option<TensorData>,
    zero: Option<TensorData>,
}

/// Complete concrete descriptor contract for public tinygrad `Tensor.roll`.
/// The existing one-axis `roll`, `roll_axes`, and `roll_flattened` APIs remain
/// raw/backward-compatible building blocks; this records Python's scalar /
/// tuple / `None` dispatch before any movement can be appended.
#[derive(Clone, Debug)]
struct SourceRollPlan {
    shifts: Vec<i64>,
    axes: Vec<isize>,
    flattened: bool,
    flat_shape: Option<Shape>,
    output_shape: Shape,
}

fn source_roll_plan(
    graph: &Graph,
    input: NodeId,
    shifts: RollShifts,
    dims: RollDims,
) -> Result<SourceRollPlan> {
    let source = graph.node(input)?;
    let shape = source.shape.clone();
    let dtype = source.dtype;
    let extent = |shape: &Shape| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&shape)?;
    let shifts = shifts.into_vec();
    match dims.into_option_vec() {
        None => {
            // `self.flatten().roll(shifts, 0).reshape(self.shape)`: the
            // recursive scalar-dim call still applies Python make_tuple, so
            // a tuple shift is accepted iff it has exactly one item.
            if shifts.len() != 1 {
                return Err(Error::InvalidRepeat {
                    reason: "roll shifts and dims must have equal lengths",
                });
            }
            let flat_shape = Shape::new([shape.numel()?]);
            extent(&flat_shape)?;
            Ok(SourceRollPlan {
                shifts,
                axes: vec![0],
                flattened: true,
                flat_shape: Some(flat_shape),
                output_shape: shape,
            })
        }
        Some(axes) => {
            // Match `tuple(self._resolve_dim(d) for d in make_tuple(dims, 1))`
            // without using the ordinary duplicate-rejecting axis helper.
            let rank = shape.rank();
            let normalized = axes
                .into_iter()
                .map(|axis| {
                    let axis = if axis < 0 {
                        axis.checked_add(rank as isize).unwrap_or(isize::MIN)
                    } else {
                        axis
                    };
                    if axis < 0 || axis >= rank as isize {
                        Err(Error::InvalidAxis {
                            node: input,
                            axis: usize::try_from(axis).unwrap_or(usize::MAX),
                            rank,
                        })
                    } else {
                        Ok(axis as isize)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            if normalized.len() != shifts.len() {
                return Err(Error::InvalidRepeat {
                    reason: "roll shifts and dims must have equal lengths",
                });
            }
            // For a rank-zero tensor, `dims=(), shifts=()` reaches
            // `self.repeat(*())` in tinygrad, whose required `repeats`
            // positional argument is absent. Do not turn that Python call
            // failure into a silent identity node.
            if shape.rank() == 0 && normalized.is_empty() {
                return Err(Error::InvalidRepeat {
                    reason: "roll scalar requires a dimension",
                });
            }
            // Rehearsal below proves the source `repeat` and `shrink` byte
            // boundaries (including duplicated-axis ownership) before live
            // publication. Preserve zero extents as a validated identity.
            Ok(SourceRollPlan {
                shifts,
                axes: normalized,
                flattened: false,
                flat_shape: None,
                output_shape: shape,
            })
        }
    }
}

fn lower_source_roll(graph: &mut Graph, input: NodeId, plan: &SourceRollPlan) -> Result<NodeId> {
    if plan.flattened {
        let flat = graph.flatten(input, 0, -1)?;
        debug_assert_eq!(graph.shape(flat).expect("source roll preflighted"), plan.flat_shape.as_ref().expect("flatten plan"));
        let rolled = graph.roll_axes(flat, &plan.shifts, &plan.axes)?;
        graph.reshape(rolled, plan.output_shape.clone())
    } else {
        graph.roll_axes(input, &plan.shifts, &plan.axes)
    }
}

fn one_hot_source_lub(lhs: DType, rhs: DType) -> DType {
    if matches!(
        (lhs, rhs),
        (DType::I64, DType::U64)
            | (DType::U64, DType::I64)
            | (DType::U64, DType::I8 | DType::I16 | DType::I32)
            | (DType::I8 | DType::I16 | DType::I32, DType::U64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

pub(crate) fn one_hot_plan(graph: &Graph, input: NodeId, classes: usize) -> Result<OneHotPlan> {
    one_hot_plan_inner(graph, input, classes, true)
}

pub(crate) fn one_hot_bool_plan(graph: &Graph, input: NodeId, classes: usize) -> Result<OneHotPlan> {
    one_hot_plan_inner(graph, input, classes, false)
}

fn one_hot_plan_inner(graph: &Graph, input: NodeId, classes: usize, numeric: bool) -> Result<OneHotPlan> {
    let source = graph.node(input)?;
    let input_shape = source.shape.clone();
    let input_dtype = source.dtype;
    if !input_dtype.is_integer() {
        return Err(Error::InvalidRandom {
            reason: "one_hot requires integer indices",
        });
    }
    let class_end = i64::try_from(classes).map_err(|_| Error::InvalidRandom {
        reason: "one_hot class count exceeds the supported i64 range",
    })?;
    let range = lazy_arange_default_int_plan(0, class_end, 1)?;
    let mut value_dims = input_shape.dims().to_vec();
    value_dims.push(1);
    let value_shape = Shape::new(value_dims);
    let mut class_dims = vec![1; value_shape.rank()];
    *class_dims.last_mut().expect("one_hot unsqueeze has rank") = classes;
    let class_shape = Shape::new(class_dims);
    let mut output_dims = input_shape.dims().to_vec();
    output_dims.push(classes);
    let output_shape = Shape::new(output_dims);
    let comparison_dtype = one_hot_source_lub(input_dtype, range.dtype);
    let one = numeric.then(|| TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
    let zero = numeric.then(|| TensorData::scalar_with_dtype(Scalar::I(0), DType::I32));
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };

    // Original index input and its trailing singleton view.
    extent(&input_shape, input_dtype)?;
    extent(&value_shape, input_dtype)?;
    // `lazy_arange_default_int`: typed scalar step -> Expand -> typed
    // cumsum -> typed offset Add. Its declared range descriptor is also the
    // later class-axis reshape source.
    extent(range.step.shape(), range.step.dtype())?;
    extent(&range.shape, range.dtype)?;
    extent(&range.shape, range.dtype)?;
    extent(range.offset.shape(), range.offset.dtype())?;
    extent(&range.shape, range.dtype)?;
    extent(&class_shape, range.dtype)?;
    // Eq is Ne plus Bool logical-not after the source LUB; both casts,
    // comparison, and Bool stages must fit before either one is published.
    extent(&value_shape, comparison_dtype)?;
    extent(&class_shape, comparison_dtype)?;
    if value_shape.broadcast_with(&class_shape)? != output_shape {
        return Err(Error::InvalidElementwiseDType {
            op: "one_hot class broadcast",
            actual: comparison_dtype,
        });
    }
    extent(&output_shape, comparison_dtype)?;
    extent(&output_shape, DType::Bool)?;
    extent(&output_shape, DType::Bool)?;
    extent(&output_shape, DType::Bool)?;
    if let (Some(one), Some(zero)) = (&one, &zero) {
        // Public `one_hot.where(1, 0)` commits strong default-I32 values.
        extent(one.shape(), one.dtype())?;
        extent(zero.shape(), zero.dtype())?;
        if output_shape.broadcast_with(one.shape())? != output_shape
            || output_shape.broadcast_with(zero.shape())? != output_shape
            || one.dtype() != DType::I32 || zero.dtype() != DType::I32 {
            return Err(Error::InvalidElementwiseDType { op: "one_hot value commitment", actual: DType::I32 });
        }
        extent(&output_shape, DType::I32)?;
    }
    Ok(OneHotPlan {
        value_shape,
        range,
        class_shape,
        comparison_dtype,
        output_shape,
        one,
        zero,
    })
}

pub(crate) fn lazy_arange_default_int_plan(
    start: i64,
    end: i64,
    step: i64,
) -> Result<LazyArangePlan> {
    match lazy_arange_plan(start, end, step, DType::I32, true) {
        Ok(plan) => Ok(plan),
        Err(_) => lazy_arange_plan(start, end, step, DType::I64, true),
    }
}

fn lazy_arange_plan(
    start: i64,
    end: i64,
    step: i64,
    dtype: DType,
    source_checked: bool,
) -> Result<LazyArangePlan> {
    if step == 0 {
        return Err(Error::InvalidArange { start, end, step });
    }
    if !matches!(dtype, DType::I32 | DType::I64) {
        return Err(Error::InvalidElementwiseDType {
            op: "lazy arange",
            actual: dtype,
        });
    }

    // tinygrad checks the inclusive range endpoints before it creates the
    // buffer-free step fill. Keep those calculations wider than the host
    // inputs so an overflowing endpoint is rejected rather than wrapped.
    let start_wide = i128::from(start);
    let end_wide = i128::from(end);
    let step_wide = i128::from(step);
    let (lower, upper) = if step > 0 {
        (start_wide, end_wide - step_wide)
    } else {
        (end_wide - step_wide, start_wide)
    };
    let (minimum, maximum) = match dtype {
        DType::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
        DType::I64 => (i128::from(i64::MIN), i128::from(i64::MAX)),
        _ => unreachable!(),
    };
    if source_checked && (lower < minimum || upper > maximum) {
        return Err(Error::InvalidArange { start, end, step });
    }

    let requested_length = if step > 0 {
        if start >= end {
            0
        } else {
            (end_wide - start_wide + step_wide - 1) / step_wide
        }
    } else if start <= end {
        0
    } else {
        let stride = -step_wide;
        (start_wide - end_wide + stride - 1) / stride
    };
    // The legacy I64 API historically stops at the first nonrepresentable
    // successor. Source-typed ranges reject that case above instead. Retain
    // the legacy sequence length while sharing the same scalar-backed lowerer.
    let length = if source_checked || requested_length == 0 {
        requested_length
    } else if step > 0 {
        requested_length.min((i128::from(i64::MAX) - start_wide) / step_wide + 1)
    } else {
        requested_length.min((start_wide - i128::from(i64::MIN)) / -step_wide + 1)
    };
    let length = usize::try_from(length).map_err(|_| Error::InvalidArange { start, end, step })?;
    let shape = Shape::new([length]);
    shape
        .numel()?
        .checked_mul(dtype.itemsize())
        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;

    // These are typed scalar payloads. The source’s weak integer constants
    // are committed at the selected range width, including integer wrapping
    // at the step and offset storage boundaries.
    let typed = |value: i128| match dtype {
        DType::I32 => Scalar::I((value as i32).into()),
        DType::I64 => Scalar::I(value as i64),
        _ => unreachable!(),
    };
    Ok(LazyArangePlan {
        shape,
        dtype,
        step: TensorData::scalar_with_dtype(typed(step_wide), dtype),
        offset: TensorData::scalar_with_dtype(typed(start_wide - step_wide), dtype),
    })
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
    fn lazy_fill_and_range_are_scalar_backed_and_preflighted() {
        let mut graph = Graph::new();
        let fill = graph
            .lazy_full_with_dtype([2, 3], Scalar::I(-7), DType::I16)
            .unwrap();
        assert_eq!(graph.shape(fill).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(fill).unwrap(), DType::I16);
        let crate::Op::Expand { input, .. } = graph.op(fill).unwrap() else {
            panic!("expected scalar-backed fill expansion");
        };
        assert!(matches!(graph.op(*input).unwrap(), Op::Constant(data) if data.len() == 1));

        let positive = graph.lazy_arange_with_dtype(2, 8, 2, DType::I32).unwrap();
        assert_eq!(graph.shape(positive).unwrap(), &Shape::from([3]));
        assert_eq!(graph.dtype(positive).unwrap(), DType::I32);
        let crate::Op::Binary {
            op: crate::BinaryOp::Add,
            lhs,
            rhs,
        } = graph.op(positive).unwrap() else {
            panic!("expected cumulative range offset Add");
        };
        assert!(matches!(graph.op(*lhs).unwrap(), Op::Reduce { kind: crate::ReduceKind::Sum, .. }));
        assert!(matches!(graph.op(*rhs).unwrap(), Op::Constant(data) if data.len() == 1));

        let negative = graph.lazy_arange_with_dtype(5, -2, -3, DType::I64).unwrap();
        assert_eq!(graph.shape(negative).unwrap(), &Shape::from([3]));
        assert_eq!(graph.dtype(negative).unwrap(), DType::I64);
        let legacy = graph.arange(0, 3, 1).unwrap();
        assert_eq!(graph.dtype(legacy).unwrap(), DType::I64);
        let default_i32 = graph.lazy_arange_default_int(0, 3, 1).unwrap();
        let default_i64 = graph
            .lazy_arange_default_int(i64::from(i32::MAX) + 1, i64::from(i32::MAX) + 3, 1)
            .unwrap();
        assert_eq!(graph.dtype(default_i32).unwrap(), DType::I32);
        assert_eq!(graph.dtype(default_i64).unwrap(), DType::I64);

        let empty = graph.lazy_full_with_dtype([0], Scalar::F(1.0), DType::F64).unwrap();
        let empty_range = graph.lazy_arange_with_dtype(0, 0, 1, DType::I32).unwrap();
        assert_eq!(graph.shape(empty).unwrap(), &Shape::from([0]));
        assert_eq!(graph.shape(empty_range).unwrap(), &Shape::from([0]));
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let filled = graph.lazy_full_with_dtype([0], Scalar::I(1), dtype).unwrap();
            assert_eq!(graph.dtype(filled).unwrap(), dtype);
        }
        assert!(graph.nodes.iter().filter_map(|node| match &node.op {
            Op::Constant(data) => Some(data.len()),
            _ => None,
        }).all(|len| len == 1));

        let mut invalid = Graph::new();
        let before = invalid.node_count();
        assert!(matches!(
            invalid.lazy_arange_with_dtype(0, 2, 0, DType::I32),
            Err(Error::InvalidArange { .. })
        ));
        assert_eq!(invalid.node_count(), before);
        assert!(matches!(
            invalid.lazy_arange_with_dtype(i64::MAX, i64::MAX, 1, DType::I32),
            Err(Error::InvalidArange { .. })
        ));
        assert_eq!(invalid.node_count(), before);
        assert!(matches!(
            invalid.lazy_full_with_dtype([usize::MAX, 2], Scalar::I(0), DType::I64),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(invalid.node_count(), before);
    }

    #[test]
    fn ones_typed_and_like_match_tinygrad_full_boundaries() {
        let dtypes = [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ];
        let mut graph = Graph::new();
        for dtype in dtypes {
            let one = graph.ones_with_dtype([2, 3], dtype).unwrap();
            assert_eq!(graph.shape(one).unwrap(), &Shape::from([2, 3]));
            assert_eq!(graph.dtype(one).unwrap(), dtype);
            // `Tensor.ones(..., dtype=...)` is `full(..., 1.0)` with the
            // source default buffer, hence a materialized typed Constant—not
            // a scalar Expand alias.
            assert!(matches!(graph.op(one).unwrap(), Op::Constant(data)
                if data.len() == 6 && data.scalar_at(0).as_f64() == 1.0));
        }

        let input = graph.input_dtype("x", [2, 0, 3], DType::BF16);
        let inherited = graph.ones_like(input, None).unwrap();
        let override_dtype = graph.ones_like(input, Some(DType::U32)).unwrap();
        assert_ne!(inherited, input);
        assert_eq!(graph.shape(inherited).unwrap(), &Shape::from([2, 0, 3]));
        assert_eq!(graph.dtype(inherited).unwrap(), DType::BF16);
        assert_eq!(graph.dtype(override_dtype).unwrap(), DType::U32);
        assert!(matches!(graph.op(inherited).unwrap(), Op::Constant(data) if data.len() == 0));

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        assert!(matches!(
            malformed.ones_like(NodeId(usize::MAX), None),
            Err(Error::UnknownNode(_))
        ));
        assert_eq!(malformed.node_count(), before);
        assert!(matches!(
            malformed.ones_with_dtype([usize::MAX, 2], DType::I64),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), before);

        // Even if an override has a narrow output storage width, `*_like`
        // must validate the inspected input descriptor before publication.
        let source = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
        let before = malformed.node_count();
        assert!(matches!(
            malformed.ones_like(source, Some(DType::Bool)),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn uniform_implicit_is_source_f32_rand_mul_cast_add_and_preflighted() {
        Graph::manual_seed(42);
        let mut graph = Graph::new();
        let output = graph.uniform_implicit([2, 3], -2.0, 5.0, DType::F16).unwrap();
        let default = graph.uniform_default([]).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert!(!graph.node(output).unwrap().requires_grad);

        // The requested F16 cast sits between source-default-F32 Mul and the
        // final weak-low Add. It must not be replaced with one ranged Random.
        let Op::Binary { op: crate::BinaryOp::Add, lhs: cast, .. } = graph.op(output).unwrap() else {
            panic!("expected final source low Add");
        };
        let Op::Cast { input: multiply, dtype: DType::F16 } = graph.op(*cast).unwrap() else {
            panic!("expected requested dtype cast after source Mul");
        };
        let Op::Binary { op: crate::BinaryOp::Mul, lhs: scale, rhs: random } = graph.op(*multiply).unwrap() else {
            panic!("expected scalar-left source scale Mul");
        };
        assert!(matches!(graph.op(*scale).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == DType::F32));
        let Op::Random { kind: RandomKind::Uniform { low, high }, .. } = graph.op(*random).unwrap() else {
            panic!("expected source unit random stream");
        };
        assert_eq!((*low, *high), (0.0, 1.0));
        assert_eq!(graph.dtype(*random).unwrap(), DType::F32);

        // The literal ordered source validation admits NaN bounds because
        // Python's `nan >= high` is false; an integer cast is subsequently
        // lifted by the weak float `low` Add to source-default F32.
        let nonfinite = graph.uniform_implicit([0], f64::NAN, 1.0, DType::I16).unwrap();
        assert_eq!(graph.dtype(nonfinite).unwrap(), DType::F32);

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        assert!(malformed.uniform_implicit([2], 1.0, 1.0, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.uniform_implicit([usize::MAX, 2], 0.0, 1.0, DType::F64).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn normal_implicit_is_source_randn_mul_add_and_preflighted() {
        Graph::manual_seed(17);
        let mut graph = Graph::new();
        let output = graph.normal_implicit([2, 3], 4.0, 0.5, DType::BF16).unwrap();
        let default = graph.normal_default([]).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([2, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::BF16);
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert!(!graph.node(output).unwrap().requires_grad);

        // Source applies the requested randn cast before scalar-left std Mul,
        // then applies mean through a second weak-scalar Add.
        let Op::Binary { op: crate::BinaryOp::Add, lhs: multiply, .. } = graph.op(output).unwrap() else {
            panic!("expected final source mean Add");
        };
        let Op::Binary { op: crate::BinaryOp::Mul, lhs: std, rhs: cast } = graph.op(*multiply).unwrap() else {
            panic!("expected scalar-left source std Mul");
        };
        assert!(matches!(graph.op(*std).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == DType::BF16));
        let Op::Cast { input: random, dtype: DType::BF16 } = graph.op(*cast).unwrap() else {
            panic!("expected requested randn storage cast");
        };
        let Op::Random { kind: RandomKind::Normal { mean, std }, .. } = graph.op(*random).unwrap() else {
            panic!("expected source standard-normal stream");
        };
        assert_eq!((*mean, *std), (0.0, 1.0));
        assert_eq!(graph.dtype(*random).unwrap(), DType::F32);

        // Source's ordered `std < 0` test admits NaN, and an integer randn
        // cast is lifted by a weak floating mean only at the final Add.
        let nonfinite = graph.normal_implicit([0], f64::NAN, f64::NAN, DType::I16).unwrap();
        assert_eq!(graph.dtype(nonfinite).unwrap(), DType::F32);

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        assert!(malformed.normal_implicit([2], 0.0, -0.5, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.normal_implicit([usize::MAX, 2], 0.0, 1.0, DType::F64).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn glorot_uniform_implicit_preflights_fan_then_uses_source_uniform() {
        Graph::manual_seed(23);
        let mut graph = Graph::new();
        let output = graph.glorot_uniform_implicit([2, 3], DType::F16).unwrap();
        let default = graph.glorot_uniform_default([1]).unwrap();
        let empty = graph.glorot_uniform_implicit([0], DType::F64).unwrap();
        let lifted_integer = graph.glorot_uniform_implicit([1], DType::I16).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert_eq!(graph.shape(empty).unwrap(), &Shape::from([0]));
        // The Python-float bounds weakly lift a non-float Uniform result at
        // the source Add boundary, just as public `uniform` does.
        assert_eq!(graph.dtype(lifted_integer).unwrap(), DType::F32);
        assert!(!graph.node(output).unwrap().requires_grad);
        // The root is inherited literally from `uniform`: a weak lower-bound
        // Add over its separately visible unit-random scaling chain.
        assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: crate::BinaryOp::Add, .. }));
        assert!((0..graph.node_count()).any(|index| match graph.op(NodeId(index)).unwrap() {
            Op::Random { kind: RandomKind::Uniform { low, high }, .. } => *low == 0.0 && *high == 1.0,
            _ => false,
        }));

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        assert!(malformed.glorot_uniform_implicit([], DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        // rank-two zero shape has `fan_in + prod(fan_out) == 0`, whose Python
        // `6 / 0` fails before source uniform can reserve a stream.
        assert!(malformed.glorot_uniform_implicit([0, 0], DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.glorot_uniform_implicit([usize::MAX, 1], DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn kaiming_uniform_implicit_uses_empty_tail_identity_before_uniform() {
        Graph::manual_seed(29);
        let mut graph = Graph::new();
        // Unlike the old seeded compatibility API, source Kaiming permits a
        // scalar: `prod(shape[1:])` is Python's empty-product identity.
        let scalar = graph.kaiming_uniform_implicit([], 0.25, DType::F16).unwrap();
        let rank_one_empty = graph.kaiming_uniform_default_a([0], DType::F64).unwrap();
        let default = graph.kaiming_uniform_default([2, 3]).unwrap();
        let lifted_integer = graph.kaiming_uniform_implicit([1], f64::NAN, DType::I16).unwrap();
        assert_eq!(graph.shape(scalar).unwrap(), &Shape::new([]));
        assert_eq!(graph.shape(rank_one_empty).unwrap(), &Shape::from([0]));
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert_eq!(graph.dtype(lifted_integer).unwrap(), DType::F32);
        assert!(!graph.node(scalar).unwrap().requires_grad);
        assert!(matches!(graph.op(scalar).unwrap(), Op::Binary { op: crate::BinaryOp::Add, .. }));
        assert!((0..graph.node_count()).any(|index| match graph.op(NodeId(index)).unwrap() {
            Op::Random { kind: RandomKind::Uniform { low, high }, .. } => *low == 0.0 && *high == 1.0,
            _ => false,
        }));

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        // A zero tail fan and a zero bound from infinite `a` both make source
        // uniform reject before it can reserve a captured stream.
        assert!(malformed.kaiming_uniform_implicit([2, 0], 0.01, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.kaiming_uniform_implicit([2], f64::INFINITY, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.kaiming_uniform_implicit([1, usize::MAX, 2], 0.01, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn kaiming_normal_implicit_preflights_fan_then_uses_source_normal() {
        Graph::manual_seed(31);
        let mut graph = Graph::new();
        // Kaiming's tail product is empty for scalar and rank-one shapes.
        let scalar = graph.kaiming_normal_implicit([], 0.25, DType::F16).unwrap();
        let rank_one_empty = graph.kaiming_normal_default_a([0], DType::F64).unwrap();
        let default = graph.kaiming_normal_default([2, 3]).unwrap();
        let lifted_integer = graph.kaiming_normal_implicit([1], f64::NAN, DType::I16).unwrap();
        // Unlike Uniform, a zero Normal std is valid: infinite `a` still
        // invokes source Normal and therefore its ambient two-word stream.
        let zero_std = graph.kaiming_normal_implicit([1], f64::INFINITY, DType::F32).unwrap();
        assert_eq!(graph.shape(scalar).unwrap(), &Shape::new([]));
        assert_eq!(graph.shape(rank_one_empty).unwrap(), &Shape::from([0]));
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert_eq!(graph.dtype(lifted_integer).unwrap(), DType::F32);
        assert!(!graph.node(scalar).unwrap().requires_grad);

        let Op::Binary { op: crate::BinaryOp::Add, lhs: multiply, .. } = graph.op(scalar).unwrap() else {
            panic!("expected source mean Add");
        };
        let Op::Binary { op: crate::BinaryOp::Mul, .. } = graph.op(*multiply).unwrap() else {
            panic!("expected source scalar-left std Mul");
        };
        assert!((0..graph.node_count()).any(|index| match graph.op(NodeId(index)).unwrap() {
            Op::Random { kind: RandomKind::Normal { mean, std }, .. } => *mean == 0.0 && *std == 1.0,
            _ => false,
        }));
        assert!(matches!(graph.op(zero_std).unwrap(), Op::Binary { op: crate::BinaryOp::Add, .. }));

        let mut malformed = Graph::new();
        let before = malformed.node_count();
        assert!(malformed.kaiming_normal_implicit([2, 0], 0.01, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.kaiming_normal_implicit([1, usize::MAX, 2], 0.01, DType::F32).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn linspace_is_source_literal_f32_lazy_and_preflighted() {
        let mut graph = Graph::new();
        let empty = graph.linspace(2.0, 5.0, 0, DType::F32).unwrap();
        let singleton = graph.linspace(f64::NAN, 3.0, 1, DType::F16).unwrap();
        let line = graph.linspace(-1.0, f64::INFINITY, 3, DType::BF16).unwrap();
        let default = graph.linspace_default(0.0, 1.0, 4).unwrap();
        assert_eq!(graph.shape(empty).unwrap(), &Shape::new([0]));
        assert_eq!(graph.shape(singleton).unwrap(), &Shape::new([1]));
        assert_eq!(graph.dtype(line).unwrap(), DType::BF16);
        assert_eq!(graph.dtype(default).unwrap(), DType::F32);
        assert!((0..graph.node_count()).any(|n| matches!(graph.op(NodeId(n)).unwrap(), Op::Reduce { kind: crate::ReduceKind::Sum, .. })));
        assert!(graph.nodes.iter().filter_map(|node| match &node.op { Op::Constant(data)=>Some(data.len()), _=>None }).all(|len| len==1));

        for dtype in [DType::F32,DType::F16,DType::BF16,DType::F64,DType::I8,DType::U8,DType::I64,DType::U64] {
            let mut typed=Graph::new(); let output=typed.linspace(0.0,1.0,2,dtype).unwrap(); assert_eq!(typed.dtype(output).unwrap(),dtype);
        }
        let mut invalid=Graph::new(); let before=invalid.node_count();
        assert!(invalid.linspace(0.0,1.0,-1,DType::F32).is_err());
        assert!(invalid.linspace(0.0,1.0,2,DType::Bool).is_err()); assert_eq!(invalid.node_count(),before);
        let mut overflow=Graph::new(); let before=overflow.node_count();
        assert!(overflow.linspace(0.0,1.0,isize::MAX,DType::F64).is_err()); assert_eq!(overflow.node_count(),before);
    }

    #[test]
    fn eye_is_lazy_range_equality_and_preflighted() {
        let mut graph=Graph::new();
        let square=graph.eye_default(3,None).unwrap();
        let rectangular=graph.eye(2,Some(4),DType::I16).unwrap();
        let empty=graph.eye(0,Some(3),DType::F64).unwrap();
        assert_eq!(graph.shape(square).unwrap(),&Shape::new([3,3]));
        assert_eq!(graph.dtype(square).unwrap(),DType::F32);
        assert_eq!(graph.shape(rectangular).unwrap(),&Shape::new([2,4]));
        assert_eq!(graph.shape(empty).unwrap(),&Shape::new([0,3]));
        assert!((0..graph.node_count()).any(|n|matches!(graph.op(NodeId(n)).unwrap(),Op::Logical{..})));
        assert!(graph.nodes.iter().filter_map(|node|match &node.op{Op::Constant(data)=>Some(data.len()),_=>None}).all(|len|len==1));
        for dtype in [DType::Bool,DType::I8,DType::U8,DType::I16,DType::U16,DType::I32,DType::U32,DType::I64,DType::U64,DType::F16,DType::BF16,DType::F32,DType::F64] { let mut typed=Graph::new();let eye=typed.eye(1,None,dtype).unwrap();assert_eq!(typed.dtype(eye).unwrap(),dtype); }
        let mut overflow=Graph::new();let before=overflow.node_count();assert!(overflow.eye(usize::MAX,Some(2),DType::F64).is_err());assert_eq!(overflow.node_count(),before);
    }

    #[test]
    fn one_hot_plan_tracks_default_range_width_and_i64_u64_bridge_without_publication() {
        if usize::BITS < 64 {
            return;
        }
        let mut graph = Graph::new();
        let indices = graph.input_dtype("indices", [2], DType::U64);
        let narrow = one_hot_plan(&graph, indices, 3).unwrap();
        assert_eq!(narrow.range.dtype, DType::I32);
        assert_eq!(narrow.comparison_dtype, DType::F32);
        assert_eq!(narrow.output_shape, Shape::from([2, 3]));

        // The endpoint is exclusive: I32 remains valid through i32::MAX +
        // 1, and the next class forces tinygrad's source-default I64 range.
        let wide = one_hot_plan(
            &graph,
            indices,
            usize::try_from(i64::from(i32::MAX) + 2).unwrap(),
        )
        .unwrap();
        assert_eq!(wide.range.dtype, DType::I64);
        assert_eq!(wide.comparison_dtype, DType::F32);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn chunk_matches_tinygrad_uneven_tail_and_preserves_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let default_outputs = graph.chunk_default(input, 3).unwrap();
        let outputs = graph.chunk(input, 3, -1).unwrap();
        assert_eq!(default_outputs.len(), 2);
        assert_eq!(
            default_outputs
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([1, 5]), Shape::from([1, 5])]
        );
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
            execute(&graph, default_outputs[1], values.clone()),
            TensorData::new([1, 5], vec![5., 6., 7., 8., 9.]).unwrap()
        );
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
    fn unfold_matches_tinygrad_window_geometry_and_preflights() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let output = graph.unfold(input, -1, 2, 2).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 2, 2]));
        assert_eq!(
            execute(&graph, output, TensorData::new([2, 5], (0..10).map(|x| x as f32).collect()).unwrap()).to_vec_f64(),
            vec![0., 1., 2., 3., 5., 6., 7., 8.]
        );
        let mut invalid = Graph::new();
        let source = invalid.input("x", [3]);
        let nodes = invalid.node_count();
        assert!(invalid.unfold(source, 0, 4, 1).is_err());
        assert_eq!(invalid.node_count(), nodes);
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
    fn chunk_matches_source_overchunk_count_and_preflights_all_views() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [3], DType::I16);
        // `chunk(5)` first chooses ceildiv(3, 5) == 1, then `split(1)`.
        // Therefore source returns three views, not five padded empties.
        let outputs = graph.chunk_default(input, 5).unwrap();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|&output| graph.shape(output).unwrap() == &Shape::from([1])));
        assert!(outputs.iter().all(|&output| graph.dtype(output).unwrap() == DType::I16));

        let scalar = graph.input("scalar", []);
        let before_scalar = graph.node_count();
        assert!(graph.chunk_default(scalar, 1).is_err());
        assert_eq!(graph.node_count(), before_scalar);

        let mut overflow = Graph::new();
        let overflow_input = overflow.input_dtype(
            "overflow",
            [usize::MAX / DType::F64.itemsize() + 1],
            DType::F64,
        );
        let before_overflow = overflow.node_count();
        assert!(overflow.chunk_default(overflow_input, 1).is_err());
        assert_eq!(overflow.node_count(), before_overflow);
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
    fn triangular_helpers_match_tinygrad_diagonals_and_select_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let default_upper = graph.triu_default(input).unwrap();
        let default_lower = graph.tril_default(input).unwrap();
        let upper = graph.triu(input, 1).unwrap();
        let lower = graph.tril(input, -1).unwrap();
        let loss = graph.sum_all(upper).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();

        assert_eq!(
            execute(&graph, upper, values.clone()),
            TensorData::new([2, 3], vec![0., 2., 3., 0., 0., 6.]).unwrap()
        );
        assert_eq!(
            execute(&graph, lower, values.clone()),
            TensorData::new([2, 3], vec![0., 0., 0., 4., 0., 0.]).unwrap()
        );
        assert_eq!(
            execute(&graph, default_upper, values.clone()),
            TensorData::new([2, 3], vec![1., 2., 3., 0., 5., 6.]).unwrap()
        );
        assert_eq!(
            execute(&graph, default_lower, values.clone()),
            TensorData::new([2, 3], vec![1., 0., 0., 4., 5., 0.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![0., 1., 1., 0., 0., 1.]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [2, 0], DType::I8);
        let output = empty.tril(input, 0).unwrap();
        assert_eq!(empty.dtype(output).unwrap(), DType::I8);
        assert!(execute(
            &empty,
            output,
            TensorData::from_scalars([2, 0], DType::I8, []).unwrap(),
        )
        .to_vec_f64()
        .is_empty());
    }

    #[test]
    fn triangular_helpers_preflight_rank_extent_and_diagonal_before_nodes() {
        let mut graph = Graph::new();
        let vector = graph.input("vector", [3]);
        let before = graph.node_count();
        assert!(graph.triu(vector, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.tril(overflow, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let matrix = graph.input("matrix", [2, 2]);
        let before = graph.node_count();
        assert!(graph.tril(matrix, i64::MAX).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn diagonal_matches_tinygrad_offset_signed_dimensions_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [3, 4]);
        let default_diagonal = graph.diagonal_default(input).unwrap();
        let diagonal = graph.diagonal(input, 1, 0, 1).unwrap();
        let loss = graph.sum_all(diagonal).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new(
            [3, 4],
            (1..=12).map(|value| value as f32).collect(),
        )
        .unwrap();
        assert_eq!(graph.shape(default_diagonal).unwrap(), &Shape::from([3]));
        assert_eq!(
            execute(&graph, default_diagonal, values.clone()),
            TensorData::new([3], vec![1., 6., 11.]).unwrap()
        );
        assert_eq!(
            execute(&graph, diagonal, values.clone()),
            TensorData::new([3], vec![2., 7., 12.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new(
                [3, 4],
                vec![0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
            )
            .unwrap()
        );

        let mut signed = Graph::new();
        let input = signed.input("x", [2, 2, 3]);
        let diagonal = signed.diagonal(input, 1, -1, -3).unwrap();
        assert_eq!(signed.shape(diagonal).unwrap(), &Shape::from([2, 1]));
        assert_eq!(
            execute(
                &signed,
                diagonal,
                TensorData::new([2, 2, 3], (0..12).map(|value| value as f32).collect()).unwrap(),
            ),
            TensorData::new([2, 1], vec![6., 9.]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [2, 3], DType::I8);
        let diagonal = empty.diagonal(input, 3, 0, 1).unwrap();
        assert_eq!(empty.shape(diagonal).unwrap(), &Shape::from([0]));
        assert_eq!(empty.dtype(diagonal).unwrap(), DType::I8);
    }

    #[test]
    fn diagonal_preflights_axes_offsets_and_extents_before_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.diagonal(input, 0, 0, 0).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.diagonal(input, 4, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.diagonal(input, i64::MIN, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.diagonal(input, 0, 2, 1).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.diagonal(overflow, 0, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);

        let byte_overflow = graph.input_dtype(
            "byte_overflow",
            [usize::MAX / DType::F64.itemsize() + 1, 1],
            DType::F64,
        );
        let before = graph.node_count();
        assert!(graph.diagonal(byte_overflow, 0, 0, 1).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn diag_matches_tinygrad_literal_movement_composition_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [3]);
        let output = graph.diag(input).unwrap();
        let loss = graph.sum_all(output).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([3], vec![1., 2., 3.]).unwrap();

        assert_eq!(graph.shape(output).unwrap(), &Shape::from([3, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        assert_eq!(
            execute(&graph, output, values.clone()),
            TensorData::new([3, 3], vec![1., 0., 0., 0., 2., 0., 0., 0., 3.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([3], vec![1., 1., 1.]).unwrap()
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("x", [2], DType::I8);
        let output = integer.diag(input).unwrap();
        assert_eq!(integer.dtype(output).unwrap(), DType::I8);
        assert_eq!(
            execute(
                &integer,
                output,
                TensorData::from_scalars([2], DType::I8, [Scalar::I(-1), Scalar::I(2)]).unwrap(),
            ),
            TensorData::from_scalars(
                [2, 2],
                DType::I8,
                [Scalar::I(-1), Scalar::I(0), Scalar::I(0), Scalar::I(2)],
            )
            .unwrap()
        );
    }

    #[test]
    fn diag_preserves_empty_source_no_pad_identity_and_preflights_before_nodes() {
        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0], DType::BF16);
        let output = empty.diag(input).unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::from([0, 0]));
        assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
        assert!(empty
            .nodes
            .iter()
            .all(|node| !matches!(&node.op, Op::Pad { .. })));
        assert!(execute(
            &empty,
            output,
            TensorData::from_scalars([0], DType::BF16, []).unwrap(),
        )
        .to_vec_f64()
        .is_empty());

        let mut invalid = Graph::new();
        let scalar = invalid.input("scalar", []);
        let before = invalid.node_count();
        assert!(invalid.diag(scalar).is_err());
        assert_eq!(invalid.node_count(), before);
        let matrix = invalid.input("matrix", [1, 1]);
        let before = invalid.node_count();
        assert!(invalid.diag(matrix).is_err());
        assert_eq!(invalid.node_count(), before);

        let overflow = invalid.input_dtype("overflow", [usize::MAX], DType::U8);
        let before = invalid.node_count();
        assert!(invalid.diag(overflow).is_err());
        assert_eq!(invalid.node_count(), before);
    }

    #[test]
    fn roll_matches_tinygrad_signed_shift_axis_dtype_and_vjp_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 4]);
        let rolled = graph.roll(input, -1, -1).unwrap();
        let loss = graph.sum_all(rolled).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 4], (1..=8).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, rolled, values.clone()),
            TensorData::new([2, 4], vec![2., 3., 4., 1., 6., 7., 8., 5.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 4], vec![1.; 8]).unwrap()
        );

        let mut integer = Graph::new();
        let input = integer.input_dtype("x", [3], DType::I8);
        let rolled = integer.roll(input, 7, 0).unwrap();
        assert_eq!(integer.dtype(rolled).unwrap(), DType::I8);
        assert_eq!(
            execute(
                &integer,
                rolled,
                TensorData::from_scalars([3], DType::I8, [Scalar::I(1), Scalar::I(2), Scalar::I(3)])
                    .unwrap(),
            ),
            TensorData::from_scalars([3], DType::I8, [Scalar::I(3), Scalar::I(1), Scalar::I(2)])
                .unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [2, 0]);
        assert_eq!(empty.roll(input, i64::MIN, -1).unwrap(), input);
    }

    #[test]
    fn roll_axes_matches_tinygrad_tuple_repeat_shrink_and_duplicate_dim_contracts() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let rolled = graph.roll_axes(input, &[1, -1], &[0, 1]).unwrap();
        let loss = graph.sum_all(rolled).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], (0..6).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, rolled, values.clone()),
            TensorData::new([2, 3], vec![4., 5., 3., 1., 2., 0.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![1.; 6]).unwrap()
        );

        // tinygrad's `shrink_arg[d] = ...` assignment makes the final shift
        // win for a duplicated dim, while repeat remains a single doubling.
        let mut duplicate = Graph::new();
        let input = duplicate.input_dtype("x", [2, 3], DType::I8);
        let rolled = duplicate.roll_axes(input, &[1, -1], &[1, 1]).unwrap();
        assert_eq!(
            execute(
                &duplicate,
                rolled,
                TensorData::from_scalars(
                    [2, 3],
                    DType::I8,
                    [
                        Scalar::I(0),
                        Scalar::I(1),
                        Scalar::I(2),
                        Scalar::I(3),
                        Scalar::I(4),
                        Scalar::I(5),
                    ],
                )
                .unwrap(),
            ),
            TensorData::from_scalars(
                [2, 3],
                DType::I8,
                [
                    Scalar::I(1),
                    Scalar::I(2),
                    Scalar::I(0),
                    Scalar::I(4),
                    Scalar::I(5),
                    Scalar::I(3),
                ],
            )
            .unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0, 2], DType::BF16);
        assert_eq!(empty.roll_axes(input, &[i64::MIN], &[1]).unwrap(), input);
    }

    #[test]
    fn roll_axes_preflights_all_controls_and_repeated_bytes_before_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.roll_axes(input, &[1], &[0, 1]).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.roll_axes(input, &[1], &[2]).is_err());
        assert_eq!(graph.node_count(), before);

        let scalar = graph.input("scalar", []);
        let before = graph.node_count();
        assert!(graph.roll_axes(scalar, &[1], &[0]).is_err());
        assert_eq!(graph.node_count(), before);
        assert_eq!(graph.roll_axes(scalar, &[], &[]).unwrap(), scalar);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("x", [usize::MAX / 2 + 1], DType::U8);
        let before = overflow.node_count();
        assert!(overflow.roll_axes(input, &[1], &[0]).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn roll_preflights_scalar_axis_and_extent_before_nodes() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let before = graph.node_count();
        assert!(graph.roll(scalar, 1, 0).is_err());
        assert_eq!(graph.node_count(), before);

        let input = graph.input("input", [2, 3]);
        let before = graph.node_count();
        assert!(graph.roll(input, 1, 2).is_err());
        assert_eq!(graph.node_count(), before);

        let overflow = graph.input("overflow", [usize::MAX]);
        let before = graph.node_count();
        assert!(graph.roll(overflow, 1, 0).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn flattened_roll_matches_tinygrad_default_form_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let rolled = graph.roll_flattened(input, -1).unwrap();
        let loss = graph.sum_all(rolled).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], (1..=6).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, rolled, values.clone()),
            TensorData::new([2, 3], vec![2., 3., 4., 5., 6., 1.]).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![1.; 6]).unwrap()
        );

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::I8);
        assert_eq!(scalar.roll_flattened(input, i64::MIN).unwrap(), input);

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0, 2], DType::F16);
        assert_eq!(empty.roll_flattened(input, i64::MAX).unwrap(), input);
    }

    #[test]
    fn flattened_roll_preflights_extent_before_nodes() {
        let mut graph = Graph::new();
        let overflow = graph.input("overflow", [usize::MAX, 2]);
        let before = graph.node_count();
        assert!(graph.roll_flattened(overflow, 1).is_err());
        assert_eq!(graph.node_count(), before);
    }

    #[test]
    fn roll_tinygrad_dispatches_python_scalar_tuple_and_flattened_forms() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::BF16);
        let flattened = graph
            .roll_tinygrad_default_dims(input, RollShifts::Scalar(-1))
            .unwrap();
        assert_eq!(graph.shape(flattened).unwrap(), &Shape::new([2, 3]));
        assert_eq!(graph.dtype(flattened).unwrap(), DType::BF16);

        let tuple = graph
            .roll_tinygrad(
                input,
                RollShifts::Tuple(vec![1, -1]),
                RollDims::Tuple(vec![0, 1]),
            )
            .unwrap();
        assert_eq!(graph.shape(tuple).unwrap(), &Shape::new([2, 3]));
        // The source literal is Repeat followed by Shrink, not the legacy
        // one-axis concat shortcut.
        assert!(graph.nodes.iter().any(|node| matches!(&node.op, Op::Expand { .. })));
        assert!(graph.nodes.iter().any(|node| matches!(&node.op, Op::Shrink { .. })));
        assert!(graph.grad(graph.sum_all(tuple).unwrap(), input).is_ok());

        let tuple_one = graph
            .roll_tinygrad_default_dims(input, RollShifts::Tuple(vec![7]))
            .unwrap();
        assert_eq!(graph.shape(tuple_one).unwrap(), &Shape::new([2, 3]));
    }

    #[test]
    fn roll_tinygrad_preserves_duplicate_empty_scalar_and_atomic_controls() {
        let mut duplicate = Graph::new();
        let input = duplicate.input_dtype("x", [2, 3], DType::I8);
        let output = duplicate
            .roll_tinygrad(
                input,
                RollShifts::Tuple(vec![1, -1]),
                RollDims::Tuple(vec![1, 1]),
            )
            .unwrap();
        assert_eq!(duplicate.shape(output).unwrap(), &Shape::new([2, 3]));

        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        let output = scalar
            .roll_tinygrad_default_dims(input, RollShifts::Scalar(i64::MIN))
            .unwrap();
        assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(scalar.dtype(output).unwrap(), DType::F16);

        let mut empty = Graph::new();
        let input = empty.input_dtype("x", [0, 2], DType::U16);
        let output = empty
            .roll_tinygrad(input, RollShifts::Scalar(1), RollDims::Scalar(-1))
            .unwrap();
        assert_eq!(output, input);

        let mut invalid = Graph::new();
        let input = invalid.input("x", [2, 3]);
        let before = invalid.node_count();
        assert!(invalid
            .roll_tinygrad(
                input,
                RollShifts::Tuple(vec![1, 2]),
                RollDims::None,
            )
            .is_err());
        assert_eq!(invalid.node_count(), before);
        assert!(invalid
            .roll_tinygrad(input, RollShifts::Scalar(1), RollDims::Tuple(vec![]))
            .is_err());
        assert_eq!(invalid.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input_dtype("x", [usize::MAX / 2 + 1], DType::U8);
        let before = overflow.node_count();
        assert!(overflow
            .roll_tinygrad(input, RollShifts::Scalar(1), RollDims::Scalar(0))
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn flatten_matches_tinygrad_scalar_identity_and_signed_spans() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        let default_flattened = scalar.flatten_default(input).unwrap();
        let flattened = scalar.flatten(input, 0, -1).unwrap();
        assert_eq!(scalar.shape(default_flattened).unwrap(), &Shape::from([1]));
        assert_eq!(scalar.shape(flattened).unwrap(), &Shape::from([1]));
        assert_eq!(scalar.dtype(flattened).unwrap(), DType::F16);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3, 4]);
        let default_flattened = graph.flatten_default(input).unwrap();
        assert_eq!(graph.flatten(input, -2, -2).unwrap(), input);
        let flattened = graph.flatten(input, -3, -2).unwrap();
        assert_eq!(graph.shape(default_flattened).unwrap(), &Shape::from([24]));
        assert_eq!(
            graph.shape(flattened).unwrap(),
            &Shape::from([6, 4])
        );
        let loss = graph.sum_all(flattened).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3, 4], (0..24).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            execute(&graph, default_flattened, values.clone()),
            TensorData::new([24], (0..24).map(|value| value as f32).collect()).unwrap()
        );
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3, 4], vec![1f32; 24]).unwrap()
        );
    }

    #[test]
    fn flatten_preflights_invalid_scalar_axes_and_extents() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.flatten(input, 1, 0).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.flatten(input, 0, 1).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn squeeze_matches_tinygrad_scalar_and_identity_views() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::BF16);
        assert_eq!(scalar.squeeze(input, None).unwrap(), input);
        assert_eq!(scalar.squeeze(input, Some(-1)).unwrap(), input);
        assert_eq!(scalar.squeeze(input, Some(0)).unwrap(), input);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0, 3]);
        assert_eq!(graph.squeeze(input, None).unwrap(), input);
        assert_eq!(graph.squeeze(input, Some(-1)).unwrap(), input);

        let mut singleton_graph = Graph::new();
        let singleton = singleton_graph.input("x", [2, 1, 3]);
        let squeezed = singleton_graph.squeeze(singleton, Some(-2)).unwrap();
        assert_eq!(singleton_graph.shape(squeezed).unwrap(), &Shape::from([2, 3]));
        let loss = singleton_graph.sum_all(squeezed).unwrap();
        let gradient = singleton_graph.grad(loss, singleton).unwrap();
        let values = TensorData::new([2, 1, 3], vec![1f32; 6]).unwrap();
        assert_eq!(
            execute(&singleton_graph, gradient, values),
            TensorData::new([2, 1, 3], vec![1f32; 6]).unwrap()
        );
    }

    #[test]
    fn squeeze_preflights_invalid_scalar_axis_and_extent() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.squeeze(input, Some(1)).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.squeeze(input, None).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn unsqueeze_matches_tinygrad_single_signed_axis_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        let trailing = scalar.unsqueeze(input, -1).unwrap();
        let leading = scalar.unsqueeze(input, 0).unwrap();
        assert_eq!(scalar.shape(trailing).unwrap(), &Shape::from([1]));
        assert_eq!(scalar.shape(leading).unwrap(), &Shape::from([1]));

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 0, 3]);
        let unsqueezed = graph.unsqueeze(input, -2).unwrap();
        assert_eq!(graph.shape(unsqueezed).unwrap(), &Shape::from([2, 0, 1, 3]));
        let loss = graph.sum_all(unsqueezed).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        assert_eq!(
            execute(&graph, gradient, TensorData::new([2, 0, 3], Vec::<f32>::new()).unwrap()),
            TensorData::new([2, 0, 3], Vec::<f32>::new()).unwrap()
        );
    }

    #[test]
    fn unsqueeze_preflights_invalid_axis_and_extent() {
        let mut scalar = Graph::new();
        let input = scalar.input("x", []);
        let before = scalar.node_count();
        assert!(scalar.unsqueeze(input, 1).is_err());
        assert_eq!(scalar.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.unsqueeze(input, 0).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn permute_signed_matches_tinygrad_identity_scalar_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::F16);
        assert_eq!(scalar.permute_signed(input, Vec::<isize>::new()).unwrap(), input);

        let mut graph = Graph::new();
        let input = graph.input("x", [2, 2, 3]);
        assert_eq!(graph.permute_signed(input, [0, 1, 2]).unwrap(), input);
        let permuted = graph.permute_signed(input, [-1, -3, -2]).unwrap();
        assert_eq!(graph.shape(permuted).unwrap(), &Shape::from([3, 2, 2]));
        let loss = graph.sum_all(permuted).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 2, 3], vec![1f32; 12]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 2, 3], vec![1f32; 12]).unwrap()
        );

        let repeated = graph.input("repeated", [2, 2]);
        assert_ne!(graph.permute_signed(repeated, [1, 0]).unwrap(), repeated);
    }

    #[test]
    fn permute_signed_preflights_invalid_axes_and_extents() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.permute_signed(input, [0, 0]).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph.permute_signed(input, [isize::MIN, 1]).is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.permute_signed(input, [1, 0]).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn transpose_matches_tinygrad_defaults_equal_axes_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 2]);
        let transposed = graph.transpose_default(input).unwrap();
        assert_ne!(transposed, input);
        assert_eq!(graph.shape(transposed).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.transpose(input, -1, -1).unwrap(), input);
        let loss = graph.sum_all(transposed).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 2], vec![1f32; 4]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 2], vec![1f32; 4]).unwrap()
        );
    }

    #[test]
    fn transpose_default_preflights_rank_and_extent() {
        let mut vector = Graph::new();
        let input = vector.input("x", [2]);
        let before = vector.node_count();
        assert!(vector.transpose_default(input).is_err());
        assert_eq!(vector.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow.transpose_default(input).is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn reshape_with_extents_matches_tinygrad_infer_copy_identity_and_vjp() {
        let mut scalar = Graph::new();
        let input = scalar.input_dtype("x", [], DType::BF16);
        let reshaped = scalar.reshape_with_extents(input, [ReshapeExtent::Infer]).unwrap();
        assert_eq!(scalar.shape(reshaped).unwrap(), &Shape::from([1]));

        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F16);
        assert_eq!(
            graph
                .reshape_with_extents(input, [ReshapeExtent::Copy, ReshapeExtent::Copy])
                .unwrap(),
            input
        );
        let reshaped = graph
            .reshape_with_extents(input, [ReshapeExtent::Exact(3), ReshapeExtent::Infer])
            .unwrap();
        assert_eq!(graph.shape(reshaped).unwrap(), &Shape::from([3, 2]));
        let viewed = graph
            .view(input, [ReshapeExtent::Exact(3), ReshapeExtent::Infer])
            .unwrap();
        assert_eq!(graph.shape(viewed).unwrap(), &Shape::from([3, 2]));
        assert_eq!(graph.dtype(viewed).unwrap(), DType::F16);
        assert_eq!(
            graph.view(input, [ReshapeExtent::Copy, ReshapeExtent::Copy]).unwrap(),
            input
        );
        let loss = graph.sum_all(reshaped).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1f32; 6]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![1f32; 6]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [0, 3]);
        let reshaped = empty
            .reshape_with_extents(input, [ReshapeExtent::Exact(3), ReshapeExtent::Infer])
            .unwrap();
        assert_eq!(empty.shape(reshaped).unwrap(), &Shape::from([3, 0]));
    }

    #[test]
    fn reshape_with_extents_preflights_source_errors_without_nodes() {
        let mut graph = Graph::new();
        let input = graph.input("x", [0, 3]);
        let before = graph.node_count();
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Exact(0), ReshapeExtent::Infer])
            .is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Copy, ReshapeExtent::Copy, ReshapeExtent::Copy])
            .is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .reshape_with_extents(input, [ReshapeExtent::Infer, ReshapeExtent::Infer])
            .is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .view(input, [ReshapeExtent::Exact(0), ReshapeExtent::Infer])
            .is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow
            .reshape_with_extents(input, [ReshapeExtent::Infer])
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn expand_with_extents_matches_tinygrad_copy_alignment_identity_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [1, 3], DType::F16);
        assert_eq!(graph.expand(input, [3]).unwrap(), input);
        assert_eq!(
            graph
                .expand_with_extents(input, [ExpandExtent::Copy])
                .unwrap(),
            input
        );
        let expanded = graph
            .expand_with_extents(input, [ExpandExtent::Exact(2), ExpandExtent::Copy])
            .unwrap();
        assert_eq!(graph.shape(expanded).unwrap(), &Shape::from([2, 3]));
        let loss = graph.sum_all(expanded).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([1, 3], vec![1f32; 3]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([1, 3], vec![2f32; 3]).unwrap()
        );

        let mut empty = Graph::new();
        let input = empty.input("x", [1, 0]);
        let expanded = empty.expand_with_extents(input, [ExpandExtent::Exact(2), ExpandExtent::Copy]).unwrap();
        assert_eq!(empty.shape(expanded).unwrap(), &Shape::from([2, 0]));
    }

    #[test]
    fn expand_preflights_invalid_broadcast_and_extent() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.expand(input, [3]).is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow
            .expand_with_extents(input, [ExpandExtent::Copy, ExpandExtent::Copy])
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn shrink_with_ranges_matches_tinygrad_none_empty_identity_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F16);
        assert_eq!(
            graph
                .shrink_with_ranges(input, [ShrinkRange::Full, ShrinkRange::Full])
                .unwrap(),
            input
        );
        let shrunk = graph
            .shrink_with_ranges(
                input,
                [
                    ShrinkRange::Full,
                    ShrinkRange::Bounds { start: 1, end: 1 },
                ],
            )
            .unwrap();
        assert_eq!(graph.shape(shrunk).unwrap(), &Shape::from([2, 0]));
        let loss = graph.sum_all(shrunk).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let values = TensorData::new([2, 3], vec![1f32; 6]).unwrap();
        assert_eq!(
            execute(&graph, gradient, values),
            TensorData::new([2, 3], vec![0f32; 6]).unwrap()
        );
    }

    #[test]
    fn shrink_with_ranges_preflights_rank_bounds_and_extents() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 3]);
        let before = graph.node_count();
        assert!(graph.shrink_with_ranges(input, [ShrinkRange::Full]).is_err());
        assert_eq!(graph.node_count(), before);
        assert!(graph
            .shrink_with_ranges(
                input,
                [ShrinkRange::Full, ShrinkRange::Bounds { start: 2, end: 4 }],
            )
            .is_err());
        assert_eq!(graph.node_count(), before);

        let mut overflow = Graph::new();
        let input = overflow.input("x", [usize::MAX, 2]);
        let before = overflow.node_count();
        assert!(overflow
            .shrink_with_ranges(input, [ShrinkRange::Full, ShrinkRange::Full])
            .is_err());
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn shrink_to_matches_source_none_bounds_identity_and_preflight() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::I16);
        let output = graph.shrink_to(input, [Some(1), None]).unwrap();
        let zero = graph.shrink_to(input, [Some(0), Some(2)]).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::from([1, 3]));
        assert_eq!(graph.dtype(output).unwrap(), DType::I16);
        assert_eq!(graph.shape(zero).unwrap(), &Shape::from([0, 2]));
        assert_eq!(graph.shrink_to(input, [None, None]).unwrap(), input);
        let loss = graph.sum_all(output).unwrap();
        assert_eq!(graph.shape(graph.grad(loss, input).unwrap()).unwrap(), &Shape::from([2, 3]));

        let scalar = graph.input("scalar", []);
        assert_eq!(graph.shrink_to(scalar, []).unwrap(), scalar);

        let mut malformed = Graph::new();
        let input = malformed.input("x", [2, 3]);
        let before = malformed.node_count();
        assert!(malformed.shrink_to(input, [Some(1)]).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.shrink_to(input, [Some(3), None]).is_err());
        assert_eq!(malformed.node_count(), before);
        assert!(malformed.shrink_to(NodeId(usize::MAX), [None, None]).is_err());
        assert_eq!(malformed.node_count(), before);
        let overflow = malformed.input_dtype(
            "overflow",
            [usize::MAX / DType::F64.itemsize() + 1],
            DType::F64,
        );
        let before = malformed.node_count();
        assert!(malformed.shrink_to(overflow, [None]).is_err());
        assert_eq!(malformed.node_count(), before);
    }

    #[test]
    fn split_preserves_explicit_sections_uniform_tails_and_vjp() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2, 5]);
        let default_uniform = graph
            .split_default(input, SplitSections::Uniform(1))
            .unwrap();
        let default_explicit = graph
            .split_default(input, SplitSections::Explicit(vec![1, 1]))
            .unwrap();
        let explicit = graph
            .split(input, SplitSections::Explicit(vec![1, 3, 1]), -1)
            .unwrap();
        let uniform = graph.split(input, SplitSections::Uniform(2), 1).unwrap();
        assert_eq!(default_uniform.len(), 2);
        assert_eq!(default_explicit.len(), 2);
        assert_eq!(graph.shape(default_uniform[0]).unwrap(), &Shape::from([1, 5]));
        assert_eq!(graph.shape(default_explicit[1]).unwrap(), &Shape::from([1, 5]));
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
            execute(&graph, default_uniform[1], values.clone()),
            TensorData::new([1, 5], vec![5., 6., 7., 8., 9.]).unwrap()
        );
        assert_eq!(
            execute(&graph, default_explicit[0], values.clone()),
            TensorData::new([1, 5], vec![0., 1., 2., 3., 4.]).unwrap()
        );
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
    fn split_preserves_oversized_and_explicit_zero_sections_atomically() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [3], DType::U16);
        let oversized = graph
            .split_default(input, SplitSections::Uniform(9))
            .unwrap();
        let explicit = graph
            .split_default(input, SplitSections::Explicit(vec![0, 2, 0, 1]))
            .unwrap();
        assert_eq!(oversized.len(), 1);
        assert_eq!(graph.shape(oversized[0]).unwrap(), &Shape::from([3]));
        assert_eq!(
            explicit
                .iter()
                .map(|&output| graph.shape(output).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![Shape::from([0]), Shape::from([2]), Shape::from([0]), Shape::from([1])]
        );
        assert!(explicit.iter().all(|&output| graph.dtype(output).unwrap() == DType::U16));

        let scalar = graph.input("scalar", []);
        let before_scalar = graph.node_count();
        assert!(graph.split_default(scalar, SplitSections::Uniform(1)).is_err());
        assert_eq!(graph.node_count(), before_scalar);

        let mut overflow = Graph::new();
        let overflow_input = overflow.input_dtype(
            "overflow",
            [usize::MAX / DType::F64.itemsize() + 1],
            DType::F64,
        );
        let before_overflow = overflow.node_count();
        assert!(overflow
            .split_default(overflow_input, SplitSections::Explicit(vec![1, usize::MAX / DType::F64.itemsize()]))
            .is_err());
        assert_eq!(overflow.node_count(), before_overflow);
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

        let one = graph.input("one", [1]);
        assert_ne!(graph.flip(one, [0isize]).unwrap(), one);

        let mut overflow = Graph::new();
        let input = overflow.input("overflow", [usize::MAX, 2]);
        let node_count = overflow.node_count();
        assert!(overflow.flip(input, [0isize]).is_err());
        assert_eq!(overflow.node_count(), node_count);
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
        let default_stacked = graph.stack_default([first, second]).unwrap();
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
            CpuBackend.execute(&graph, default_stacked, &bindings).unwrap(),
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
        );
        assert_eq!(
            CpuBackend.execute(&graph, gradient, &bindings).unwrap(),
            TensorData::new([2], vec![1., 1.]).unwrap()
        );

        let mut singleton = Graph::new();
        let scalar = singleton.input_dtype("scalar", [], DType::F16);
        let output = singleton.stack_default([scalar]).unwrap();
        assert_eq!(singleton.shape(output).unwrap(), &Shape::from([1]));
        assert_eq!(singleton.dtype(output).unwrap(), DType::F16);
        let empty = singleton.input_dtype("empty", [0], DType::I8);
        let output = singleton.stack([empty], -1).unwrap();
        assert_eq!(singleton.shape(output).unwrap(), &Shape::from([0, 1]));

        let mut promoted = Graph::new();
        let signed = promoted.input_dtype("signed", [1], DType::I64);
        let unsigned = promoted.input_dtype("unsigned", [1], DType::U64);
        let output = promoted.stack_default([signed, unsigned]).unwrap();
        assert_eq!(promoted.shape(output).unwrap(), &Shape::from([2, 1]));
        assert_eq!(promoted.dtype(output).unwrap(), DType::F32);

        let mut overflow = Graph::new();
        let first = overflow.input_dtype("first", [usize::MAX], DType::Bool);
        let second = overflow.input_dtype("second", [usize::MAX], DType::Bool);
        let node_count = overflow.node_count();
        assert!(overflow.stack_default([first, second]).is_err());
        assert_eq!(overflow.node_count(), node_count);
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

fn checked_initializer_tail_fan(shape: &Shape) -> Result<usize> {
    shape.dims().get(1..).unwrap_or(&[]).iter().try_fold(1usize, |fan, &dimension| {
        fan.checked_mul(dimension)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    })
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

/// Concrete descriptor plan for tinygrad's `Tensor.stack` before it publishes
/// any inserted-axis view or concat node.
struct StackPlan {
    axis: usize,
    output_shape: Shape,
    output_dtype: DType,
    input_dtypes: Vec<DType>,
}

fn stack_source_lub(lhs: DType, rhs: DType) -> DType {
    if matches!(
        (lhs, rhs),
        (DType::I64, DType::U64) | (DType::U64, DType::I64)
    ) {
        DType::F32
    } else {
        lhs.promote(rhs)
    }
}

impl StackPlan {
    fn new(graph: &Graph, inputs: &[NodeId], axis: isize) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::InvalidRandom {
                reason: "stack requires at least one tensor",
            });
        }
        let descriptors = inputs
            .iter()
            .map(|&input| {
                let node = graph.node(input)?;
                node.shape
                    .numel()?
                    .checked_mul(node.dtype.itemsize())
                    .ok_or_else(|| Error::ShapeOverflow(node.shape.clone()))?;
                Ok((node.shape.clone(), node.dtype))
            })
            .collect::<Result<Vec<_>>>()?;
        let shapes = descriptors
            .iter()
            .map(|(shape, _)| shape.clone())
            .collect::<Vec<_>>();
        let output_rank = shapes[0]
            .rank()
            .checked_add(1)
            .ok_or_else(|| Error::ShapeOverflow(shapes[0].clone()))?;
        let rank = isize::try_from(output_rank)
            .map_err(|_| Error::ShapeOverflow(shapes[0].clone()))?;
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
        let axis = axis as usize;
        let mut output_dims = shapes[0].dims().to_vec();
        output_dims.insert(axis, inputs.len());
        let output_shape = Shape::new(output_dims);
        let output_dtype = descriptors
            .iter()
            .skip(1)
            .fold(descriptors[0].1, |dtype, (_, next)| {
                stack_source_lub(dtype, *next)
            });
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        Ok(Self {
            axis,
            output_shape,
            output_dtype,
            input_dtypes: descriptors.into_iter().map(|(_, dtype)| dtype).collect(),
        })
    }
}

impl Graph {
    pub fn unsqueeze(&mut self, input: NodeId, axis: isize) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut dims = shape.dims().to_vec();
        let rank = isize::try_from(dims.len())
            .ok()
            .and_then(|rank| rank.checked_add(1))
            .ok_or(Error::InvalidAxis {
                node: input,
                axis: usize::MAX,
                rank: usize::MAX,
            })?;
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
        dims.insert(axis as usize, 1);
        let output_shape = Shape::new(dims);
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        self.reshape(input, output_shape)
    }

    pub fn squeeze(&mut self, input: NodeId, axis: Option<isize>) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut dims = shape.dims().to_vec();
        if let Some(axis) = axis {
            // Tensor._resolve_dim accepts -1 and 0 for scalars, and the
            // explicit scalar path is a no-op.
            if dims.is_empty() {
                if matches!(axis, -1 | 0) {
                    return Ok(input);
                }
                return Err(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                });
            }
            let rank = isize::try_from(dims.len()).map_err(|_| Error::InvalidRandom {
                reason: "invalid squeeze axis",
            })?;
            let axis = if axis < 0 {
                axis.checked_add(rank).ok_or(Error::InvalidRandom {
                    reason: "invalid squeeze axis",
                })?
            } else {
                axis
            };
            if axis < 0 || axis >= rank {
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
        let output_shape = Shape::new(dims);
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        // Tensor.reshape returns self for both non-singleton explicit axes
        // and all-axis squeezes that leave the shape unchanged.
        if output_shape == shape {
            Ok(input)
        } else {
            self.reshape(input, output_shape)
        }
    }

    /// Flattens every dimension using tinygrad's default signed span.
    ///
    /// This is equivalent to `flatten(input, 0, -1)`.
    pub fn flatten_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.flatten(input, 0, -1)
    }

    /// Flattens an inclusive signed span of dimensions.
    pub fn flatten(&mut self, input: NodeId, start: isize, end: isize) -> Result<NodeId> {
        let shape = self.shape(input)?.clone();
        let dtype = self.dtype(input)?;
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let invalid = || Error::InvalidRandom {
            reason: "invalid flatten dimensions",
        };
        let rank = isize::try_from(shape.rank()).map_err(|_| invalid())?;
        // tinygrad resolves scalar dimensions against `max(1, ndim)`: every
        // accepted scalar span is empty and therefore reshapes `[]` to `[1]`.
        let output_shape = if rank == 0 {
            if !matches!(start, -1 | 0) || !matches!(end, -1 | 0) {
                return Err(invalid());
            }
            Shape::new([1])
        } else {
            let start = if start < 0 {
                start.checked_add(rank).ok_or_else(invalid)?
            } else {
                start
            };
            let end = if end < 0 {
                end.checked_add(rank).ok_or_else(invalid)?
            } else {
                end
            };
            if start < 0 || end < start || end >= rank {
                return Err(invalid());
            }
            let mut dims = shape.dims()[..start as usize].to_vec();
            dims.push(
                shape.dims()[start as usize..=end as usize]
                    .iter()
                    .try_fold(1usize, |n, d| n.checked_mul(*d))
                    .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?,
            );
            dims.extend_from_slice(&shape.dims()[end as usize + 1..]);
            Shape::new(dims)
        };
        output_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        // Tensor.reshape returns self when the view leaves the shape unchanged.
        if output_shape == shape {
            Ok(input)
        } else {
            self.reshape(input, output_shape)
        }
    }

    /// Stacks along tinygrad's default new leading axis.
    ///
    /// This is equivalent to `stack(inputs, 0)`.
    pub fn stack_default(&mut self, inputs: impl Into<Vec<NodeId>>) -> Result<NodeId> {
        self.stack(inputs, 0)
    }

    /// Stacks equal-shaped inputs along a signed newly inserted axis.
    pub fn stack(&mut self, inputs: impl Into<Vec<NodeId>>, axis: isize) -> Result<NodeId> {
        let inputs = inputs.into();
        let plan = StackPlan::new(self, &inputs, axis)?;
        let inputs = inputs
            .into_iter()
            .zip(plan.input_dtypes.iter().copied())
            .map(|(input, dtype)| {
                if dtype == plan.output_dtype {
                    Ok(input)
                } else {
                    self.cast(input, plan.output_dtype)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if inputs.len() == 1 {
            let output = self.unsqueeze(inputs[0], plan.axis as isize)?;
            debug_assert_eq!(self.shape(output).ok(), Some(&plan.output_shape));
            debug_assert_eq!(self.dtype(output).ok(), Some(plan.output_dtype));
            return Ok(output);
        }
        let mut expanded = Vec::with_capacity(inputs.len());
        for input in inputs {
            expanded.push(self.unsqueeze(input, plan.axis as isize)?);
        }
        let output = self.concat(expanded, plan.axis)?;
        debug_assert_eq!(self.shape(output).ok(), Some(&plan.output_shape));
        debug_assert_eq!(self.dtype(output).ok(), Some(plan.output_dtype));
        Ok(output)
    }

    /// Concrete public `Tensor.unfold(dim, size, step)` windows.
    ///
    /// tinygrad moves `dim` to the end, creates every fixed-stride window,
    /// then restores the original axes with the window-size lane trailing.
    /// Resolve all window bounds and the final permutation before the first
    /// Shrink/Stack node so invalid controls are atomic.
    pub fn unfold(&mut self, input: NodeId, dim: isize, size: usize, step: usize) -> Result<NodeId> {
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        if shape.rank() == 0 {
            return Err(Error::InvalidMovementRank { op: "unfold", expected: 1, actual: 0 });
        }
        if step == 0 {
            return Err(Error::InvalidRandom { reason: "unfold step must be positive" });
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![dim]))?[0];
        let extent = shape.dims()[axis];
        if size > extent {
            return Err(Error::InvalidBounds { axis, start: size, end: size, dim: extent });
        }
        let windows = extent
            .checked_sub(size)
            .and_then(|delta| delta.checked_add(1))
            .map(|span| span.div_ceil(step))
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let output_rank = shape.rank().checked_add(1).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let mut output_dims = shape.dims().to_vec();
        output_dims[axis] = windows;
        output_dims.push(size);
        let output_shape = Shape::new(output_dims);
        for (candidate, candidate_dtype) in [(&shape, dtype), (&output_shape, dtype)] {
            candidate.numel()?.checked_mul(candidate_dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(candidate.clone()))?;
        }
        let bounds = (0..windows)
            .map(|window| {
                let start = window.checked_mul(step).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                let end = start.checked_add(size).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
                if end > extent { return Err(Error::InvalidBounds { axis, start, end, dim: extent }); }
                let mut bound = shape.dims().iter().map(|&dim| (0, dim)).collect::<Vec<_>>();
                bound[axis] = (start, end);
                Ok(bound)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut permutation = (0..output_rank).collect::<Vec<_>>();
        let size_lane = axis + 1;
        permutation.remove(size_lane);
        permutation.push(size_lane);
        let windows = bounds.into_iter().map(|bound| self.shrink(input, bound)).collect::<Result<Vec<_>>>()?;
        let stacked = self.stack(windows, axis as isize)?;
        self.permute(stacked, permutation)
    }

    /// Internal `_one_hot_along_dim` Bool predicate used by tinygrad loss
    /// helpers. Public `Tensor.one_hot` continues below with I32 Select
    /// values, while this exact source branch stops at equality.
    pub(crate) fn one_hot_bool(&mut self, input: NodeId, classes: usize) -> Result<NodeId> {
        let plan = one_hot_bool_plan(self, input, classes)?;
        let values = self.reshape(input, plan.value_shape)?;
        let classes = self.lower_lazy_arange(plan.range)?;
        let classes = self.reshape(classes, plan.class_shape)?;
        debug_assert_eq!(
            one_hot_source_lub(
                self.dtype(values).expect("one_hot values preflighted"),
                self.dtype(classes).expect("one_hot classes preflighted"),
            ),
            plan.comparison_dtype,
        );
        let values = if self.dtype(values)? == plan.comparison_dtype {
            values
        } else {
            self.cast(values, plan.comparison_dtype)?
        };
        let classes = if self.dtype(classes)? == plan.comparison_dtype {
            classes
        } else {
            self.cast(classes, plan.comparison_dtype)?
        };
        let output = self.eq(values, classes)?;
        debug_assert_eq!(self.shape(output).expect("one_hot Bool preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("one_hot Bool preflighted"), DType::Bool);
        Ok(output)
    }

    pub fn one_hot(&mut self, input: NodeId, classes: usize) -> Result<NodeId> {
        let plan = one_hot_plan(self, input, classes)?;
        let values = self.reshape(input, plan.value_shape)?;
        let classes = self.lower_lazy_arange(plan.range)?;
        let classes = self.reshape(classes, plan.class_shape)?;
        let values = if self.dtype(values)? == plan.comparison_dtype { values } else { self.cast(values, plan.comparison_dtype)? };
        let classes = if self.dtype(classes)? == plan.comparison_dtype { classes } else { self.cast(classes, plan.comparison_dtype)? };
        let equal = self.eq(values, classes)?;
        let one = self.constant(plan.one.expect("numeric one_hot plan has one"));
        let zero = self.constant(plan.zero.expect("numeric one_hot plan has zero"));
        let output = self.select(equal, one, zero)?;
        debug_assert_eq!(self.shape(output).expect("one_hot preflighted"), &plan.output_shape);
        debug_assert_eq!(self.dtype(output).expect("one_hot preflighted"), DType::I32);
        Ok(output)
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
        if inputs.is_empty() {
            return Err(Error::InvalidRandom {
                reason: "meshgrid requires at least one input",
            });
        }

        // tinygrad literally reshapes every input to `(-1, 1, ...)`: an
        // input is therefore flattened, rather than restricted to a scalar
        // or vector. Build every flattened/intermediate/output descriptor
        // before the first reshape or expand can publish a view node.
        let mut lengths = Vec::with_capacity(inputs.len());
        let mut dtypes = Vec::with_capacity(inputs.len());
        for input in &inputs {
            let source = self.node(*input)?;
            let elements = source.shape.numel()?;
            elements
                .checked_mul(source.dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(source.shape.clone()))?;
            lengths.push(elements);
            dtypes.push(source.dtype);
        }
        // `self.meshgrid()` returns its receiver unchanged, including its
        // original (possibly non-vector) descriptor.
        if inputs.len() == 1 {
            return Ok(inputs);
        }
        let mut output = lengths.clone();
        if indexing == "xy" {
            output.swap(0, 1);
        }
        let output_shape = Shape::new(output);
        output_shape.numel()?;
        for &dtype in &dtypes {
            output_shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        }

        let basis = if indexing == "xy" {
            let mut basis = (0..inputs.len()).collect::<Vec<_>>();
            basis.swap(0, 1);
            basis
        } else {
            (0..inputs.len()).collect::<Vec<_>>()
        };
        let reshapes = lengths
            .iter()
            .zip(&basis)
            .map(|(&length, &axis)| {
                let trailing = inputs
                    .len()
                    .checked_sub(axis + 1)
                    .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
                let mut shape = Vec::with_capacity(trailing + 1);
                shape.push(length);
                shape.extend(std::iter::repeat(1).take(trailing));
                let shape = Shape::new(shape);
                if shape.numel()? != length {
                    return Err(Error::InvalidReshape {
                        from: Shape::new([length]),
                        to: shape,
                    });
                }
                Ok(shape)
            })
            .collect::<Result<Vec<_>>>()?;
        for (reshape, &dtype) in reshapes.iter().zip(&dtypes) {
            reshape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(reshape.clone()))?;
            if reshape.broadcast_with(&output_shape).as_ref() != Ok(&output_shape) {
                return Err(Error::InvalidExpand {
                    from: reshape.clone(),
                    to: output_shape.clone(),
                });
            }
        }

        inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| {
                let input = self.reshape(input, reshapes[index].clone())?;
                self.expand(input, output_shape.clone())
            })
            .collect()
    }

    /// Checked-in tinygrad `Tensor.meshgrid(*args)` defaulting to matrix-style
    /// `indexing="ij"`. The explicit helper already owns the complete
    /// descriptor/byte preflight, including one-input identity and the
    /// source-impossible zero-input Graph form.
    pub fn meshgrid_default(&mut self, inputs: impl Into<Vec<NodeId>>) -> Result<Vec<NodeId>> {
        self.meshgrid(inputs, "ij")
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

    /// Creates a graph-resident typed fill without materializing its payload.
    ///
    /// A single scalar constant is expanded to the requested shape. Both the
    /// scalar and expanded descriptors are validated before either node is
    /// published, including zero extents and byte-overflow rejection.
    pub fn lazy_full_with_dtype(
        &mut self,
        shape: impl Into<Shape>,
        value: Scalar,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let scalar = TensorData::scalar_with_dtype(value, dtype);
        let scalar_shape = scalar.shape().clone();
        scalar_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(scalar_shape.clone()))?;
        if scalar_shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: scalar_shape,
                to: shape,
            });
        }
        let scalar = self.constant(scalar);
        self.expand(scalar, shape)
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

    /// Creates a typed, materialized source-literal `Tensor.ones` constant.
    ///
    /// Tinygrad forwards `1.0` and an optional dtype to `Tensor.full` with its
    /// default `buffer=True`; retain that dense-constant boundary rather than
    /// substituting the buffer-free lazy-fill helper.
    pub fn ones_with_dtype(&mut self, shape: impl Into<Shape>, dtype: DType) -> Result<NodeId> {
        self.full_with_dtype(shape, Scalar::F(1.0), dtype)
    }

    pub fn arange(&mut self, start: i64, end: i64, step: i64) -> Result<NodeId> {
        let plan = lazy_arange_plan(start, end, step, DType::I64, false)?;
        self.lower_lazy_arange(plan)
    }

    /// Creates a typed graph-resident integer range without an eager payload.
    ///
    /// It mirrors tinygrad’s buffer-free Full → cumulative Add → offset
    /// composition. Only I32 and I64 are range storage types; the latter is
    /// used by the legacy [`Self::arange`] API for compatibility.
    pub fn lazy_arange_with_dtype(
        &mut self,
        start: i64,
        end: i64,
        step: i64,
        dtype: DType,
    ) -> Result<NodeId> {
        let plan = lazy_arange_plan(start, end, step, dtype, true)?;
        self.lower_lazy_arange(plan)
    }

    pub(crate) fn lower_lazy_arange(&mut self, plan: LazyArangePlan) -> Result<NodeId> {
        let step = self.lazy_full_with_dtype(plan.shape.clone(), plan.step.scalar_at(0), plan.dtype)?;
        let cumulative = self.cumsum(step, 0)?;
        let offset = self.constant(plan.offset);
        self.add(cumulative, offset)
    }

    /// Tinygrad’s default integer-range storage policy: I32 when the checked
    /// endpoints fit, otherwise I64. This is distinct from legacy
    /// [`Self::arange`], which deliberately remains I64.
    pub fn lazy_arange_default_int(
        &mut self,
        start: i64,
        end: i64,
        step: i64,
    ) -> Result<NodeId> {
        self.lower_lazy_arange(lazy_arange_default_int_plan(start, end, step)?)
    }

    // Literal `arange(n, dtype=default_float)` used by tinygrad linspace:
    // scalar F32 full, typed cumulative Add, then typed F32 offset. Keeping
    // this float throughout avoids an integer-range cast boundary for large
    // coordinates.
    pub(crate) fn lazy_arange_f32(&mut self, steps: usize) -> Result<NodeId> {
        let shape = Shape::new([steps]);
        shape.numel()?.checked_mul(DType::F32.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let step = self.lazy_full_with_dtype(shape, Scalar::F(1.0), DType::F32)?;
        let cumulative = self.cumsum(step, 0)?;
        let offset = self.constant(TensorData::scalar_with_dtype(Scalar::F(-1.0), DType::F32));
        self.add(cumulative, offset)
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
        if steps < 0 { return Err(Error::InvalidLinspace { steps }); }
        if dtype == DType::Bool { return Err(Error::InvalidRandom { reason: "linspace does not support bool dtype" }); }
        let steps = usize::try_from(steps).map_err(|_| Error::InvalidLinspace { steps })?;
        let shape = Shape::new([steps]);
        shape.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if steps == 1 { return self.lazy_full_with_dtype(shape, Scalar::F(start), dtype); }
        // tinygrad always constructs the coordinate range at default F32,
        // then applies its Python scale and start constants before final cast.
        let range = self.lazy_arange_f32(steps)?;
        let scale = Scalar::F((stop - start) / ((steps as isize - 1) as f64));
        let scaled = self.mul_scalar(range, scale)?;
        let shifted = self.add_scalar(scaled, Scalar::F(start))?;
        if self.dtype(shifted)? == dtype { Ok(shifted) } else { self.cast(shifted, dtype) }
    }

    pub fn linspace_default(&mut self, start: f64, stop: f64, steps: isize) -> Result<NodeId> {
        self.linspace(start, stop, steps, DType::F32)
    }

    pub fn eye(&mut self, rows: usize, columns: Option<usize>, dtype: DType) -> Result<NodeId> {
        let columns = columns.unwrap_or(rows);
        let output = Shape::new([rows, columns]);
        output.numel()?.checked_mul(dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(output.clone()))?;
        let row_plan = lazy_arange_default_int_plan(0, i64::try_from(rows).map_err(|_| Error::ShapeOverflow(output.clone()))?, 1)?;
        let column_plan = lazy_arange_default_int_plan(0, i64::try_from(columns).map_err(|_| Error::ShapeOverflow(output.clone()))?, 1)?;
        for plan in [&row_plan, &column_plan] {
            plan.shape.numel()?.checked_mul(plan.dtype.itemsize()).ok_or_else(|| Error::ShapeOverflow(plan.shape.clone()))?;
        }
        let comparison = if matches!((row_plan.dtype,column_plan.dtype),(DType::I64,DType::U64)|(DType::U64,DType::I64)) { DType::F32 } else { row_plan.dtype.promote(column_plan.dtype) };
        output.numel()?.checked_mul(comparison.itemsize()).ok_or_else(|| Error::ShapeOverflow(output.clone()))?;
        let rows = self.lower_lazy_arange(row_plan)?;
        let rows = self.unsqueeze(rows, -1)?;
        let columns = self.lower_lazy_arange(column_plan)?;
        let equal = self.eq(rows, columns)?;
        if dtype == DType::Bool { Ok(equal) } else { self.cast(equal, dtype) }
    }

    pub fn eye_default(&mut self, rows: usize, columns: Option<usize>) -> Result<NodeId> {
        self.eye(rows, columns, DType::F32)
    }

    /// Returns the upper triangular part of `input`, zeroing entries below
    /// `diagonal` in its final two dimensions.
    pub fn triu(&mut self, input: NodeId, diagonal: i64) -> Result<NodeId> {
        self.triangular(input, diagonal, false)
    }

    /// Checked-in tinygrad's `Tensor.triu()` default main diagonal.
    pub fn triu_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.triu(input, 0)
    }

    /// Returns the lower triangular part of `input`, zeroing entries above
    /// `diagonal` in its final two dimensions.
    pub fn tril(&mut self, input: NodeId, diagonal: i64) -> Result<NodeId> {
        self.triangular(input, diagonal, true)
    }

    /// Checked-in tinygrad's `Tensor.tril()` default main diagonal.
    pub fn tril_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.tril(input, 0)
    }

    /// The shared checked `Tensor._tri(...).where(...)` composition used by
    /// tinygrad's public triangular helpers. Every rank, index extent,
    /// diagonal shift, and broadcast is validated before this appends its I64
    /// index constants, comparison, zero, or select nodes.
    fn triangular(&mut self, input: NodeId, diagonal: i64, lower: bool) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        shape.numel()?;
        if shape.rank() < 2 {
            return Err(Error::InvalidMovementRank {
                op: "triangular",
                expected: 2,
                actual: shape.rank(),
            });
        }
        let rows = shape.dims()[shape.rank() - 2];
        let columns = shape.dims()[shape.rank() - 1];
        let rows_i64 = i64::try_from(rows).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let columns_i64 =
            i64::try_from(columns).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let shift = if lower {
            diagonal
                .checked_add(1)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?
        } else {
            diagonal
        };
        if rows != 0 {
            (rows_i64 - 1)
                .checked_add(shift)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        }
        let mask_shape = Shape::new([rows, columns]);
        mask_shape.numel()?;
        if mask_shape.broadcast_with(&shape).as_ref() != Ok(&shape) {
            return Err(Error::InvalidExpand {
                from: mask_shape,
                to: shape,
            });
        }

        let row = self.reshape(self.arange(0, rows_i64, 1)?, Shape::new([rows, 1]))?;
        let column = self.reshape(
            self.arange(0, columns_i64, 1)?,
            Shape::new([1, columns]),
        )?;
        let shift = self.full_with_dtype([], Scalar::I(shift), DType::I64)?;
        let outside = self.le(self.add(row, shift)?, column)?;
        let zero = self.zeros_with_dtype(shape, dtype)?;
        if lower {
            self.select(outside, zero, input)
        } else {
            self.select(outside, input, zero)
        }
    }

    /// Extracts the default main diagonal across the first two dimensions.
    ///
    /// This is tinygrad's parameter-free `diagonal()` form, equivalent to
    /// `diagonal(0, 0, 1)`.
    pub fn diagonal_default(&mut self, input: NodeId) -> Result<NodeId> {
        self.diagonal(input, 0, 0, 1)
    }

    /// Extracts an offset diagonal from two signed dimensions.
    ///
    /// This is tinygrad's movement-only `diagonal(offset, dim1, dim2)`
    /// composition. Axis normalization, crop bounds, every intermediate
    /// extent, and the final output shape are checked before it appends a
    /// permutation, movement node, or zero pad.
    pub fn diagonal(
        &mut self,
        input: NodeId,
        offset: i64,
        dim1: isize,
        dim2: isize,
    ) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        let checked_bytes = |descriptor: &Shape| -> Result<()> {
            descriptor
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(descriptor.clone()))?;
            Ok(())
        };
        checked_bytes(&shape)?;
        let rank = shape.rank();
        let dim1 = normalize_axes(input, rank, Some(vec![dim1]))?[0];
        let dim2 = normalize_axes(input, rank, Some(vec![dim2]))?[0];
        if dim1 == dim2 {
            return Err(Error::InvalidRandom {
                reason: "diagonal dimensions must differ",
            });
        }
        let rows = shape.dims()[dim1];
        let columns = shape.dims()[dim2];
        let (row_start, column_start) = if offset >= 0 {
            let column_start =
                usize::try_from(offset).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
            if column_start > columns {
                return Err(Error::InvalidBounds {
                    axis: dim2,
                    start: column_start,
                    end: columns,
                    dim: columns,
                });
            }
            (0, column_start)
        } else {
            let row_start = offset
                .checked_neg()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            if row_start > rows {
                return Err(Error::InvalidBounds {
                    axis: dim1,
                    start: row_start,
                    end: rows,
                    dim: rows,
                });
            }
            (row_start, 0)
        };
        let cropped_rows = rows - row_start;
        let cropped_columns = columns - column_start;
        let diagonal_extent = cropped_rows.min(cropped_columns);
        let mut order = (0..rank)
            .filter(|&axis| axis != dim1 && axis != dim2)
            .collect::<Vec<_>>();
        let leading_dims = order
            .iter()
            .map(|&axis| shape.dims()[axis])
            .collect::<Vec<_>>();
        order.extend([dim1, dim2]);

        let mut cropped_dims = leading_dims.clone();
        cropped_dims.extend([cropped_rows, cropped_columns]);
        checked_bytes(&Shape::new(cropped_dims))?;
        let mut output_dims = leading_dims.clone();
        output_dims.push(diagonal_extent);
        let output_shape = Shape::new(output_dims);
        checked_bytes(&output_shape)?;

        let unflatten_shape = if diagonal_extent == 0 {
            None
        } else {
            let square_extent = diagonal_extent
                .checked_mul(diagonal_extent)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let padded_extent = square_extent
                .checked_add(diagonal_extent)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let diagonal_plus_one = diagonal_extent
                .checked_add(1)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            let mut square_dims = leading_dims.clone();
            square_dims.extend([diagonal_extent, diagonal_extent]);
            checked_bytes(&Shape::new(square_dims))?;
            let mut flattened_dims = leading_dims.clone();
            flattened_dims.push(square_extent);
            checked_bytes(&Shape::new(flattened_dims))?;
            let mut padded_dims = leading_dims.clone();
            padded_dims.push(padded_extent);
            checked_bytes(&Shape::new(padded_dims))?;
            let mut unflatten_dims = leading_dims.clone();
            unflatten_dims.extend([diagonal_extent, diagonal_plus_one]);
            let unflatten_shape = Shape::new(unflatten_dims);
            checked_bytes(&unflatten_shape)?;
            let mut diagonal_dims = leading_dims.clone();
            diagonal_dims.extend([diagonal_extent, 1]);
            checked_bytes(&Shape::new(diagonal_dims))?;
            Some(unflatten_shape)
        };

        let permuted = self.permute(input, order)?;
        let mut crop_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        crop_bounds.extend([(row_start, rows), (column_start, columns)]);
        let cropped = self.shrink(permuted, crop_bounds)?;
        if diagonal_extent == 0 {
            return self.reshape(cropped, output_shape);
        }

        let mut square_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        square_bounds.extend([(0, diagonal_extent), (0, diagonal_extent)]);
        let square = self.shrink(cropped, square_bounds)?;
        let flattened = self.flatten(square, -2, -1)?;
        let mut padding = vec![(0, 0); leading_dims.len()];
        padding.push((0, diagonal_extent));
        let padded = self.pad(flattened, padding, Scalar::I(0))?;
        let unflattened = self.reshape(
            padded,
            unflatten_shape.expect("nonempty diagonal has a checked unflatten shape"),
        )?;
        let mut diagonal_bounds = leading_dims
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        diagonal_bounds.extend([(0, diagonal_extent), (0, 1)]);
        let diagonal = self.shrink(unflattened, diagonal_bounds)?;
        self.squeeze(diagonal, Some(-1))
    }

    /// Constructs a square diagonal matrix from a rank-one input.
    ///
    /// This follows tinygrad's literal `unsqueeze(-1).pad_to(...).flatten()
    /// .shrink_to(...).reshape(...)` construction. All concrete descriptors
    /// and byte extents are proven before the first view, pad, or constant can
    /// be published. The empty case deliberately omits Pad: tinygrad's
    /// `pad_to((0, 1))` returns the already matching unsqueezed view.
    pub fn diag(&mut self, input: NodeId) -> Result<NodeId> {
        let (shape, dtype) = {
            let source = self.node(input)?;
            (source.shape.clone(), source.dtype)
        };
        if shape.rank() != 1 {
            return Err(Error::InvalidMovementRank {
                op: "diag",
                expected: 1,
                actual: shape.rank(),
            });
        }
        let extent = shape.dims()[0];
        let checked_bytes = |shape: &Shape| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                .map(|_| ())
        };

        checked_bytes(&shape)?;
        let unsqueezed_shape = Shape::new([extent, 1]);
        checked_bytes(&unsqueezed_shape)?;
        let padded_width = extent
            .checked_add(1)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let padded_shape = Shape::new([extent, padded_width]);
        checked_bytes(&padded_shape)?;
        let flattened_extent = padded_shape.numel()?;
        let flattened_shape = Shape::new([flattened_extent]);
        checked_bytes(&flattened_shape)?;
        let output_extent = extent
            .checked_mul(extent)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        if output_extent > flattened_extent {
            return Err(Error::InvalidBounds {
                axis: 0,
                start: 0,
                end: output_extent,
                dim: flattened_extent,
            });
        }
        let output_shape = Shape::new([extent, extent]);
        checked_bytes(&output_shape)?;

        let unsqueezed = self.unsqueeze(input, -1)?;
        let padded = if extent == 0 {
            unsqueezed
        } else {
            self.pad(unsqueezed, vec![(0, 0), (0, extent)], Scalar::I(0))?
        };
        let flattened = self.flatten(padded, 0, -1)?;
        let cropped = self.shrink(flattened, vec![(0, output_extent)])?;
        self.reshape(cropped, output_shape)
    }

    /// Circularly rolls `input` by a signed shift along one signed axis.
    ///
    /// This is the one-axis branch of tinygrad's public `roll` helper. Its
    /// signed axis, empty-tensor no-op, and Euclidean shift normalization are
    /// resolved before the two source views or their concat are appended.
    pub fn roll(&mut self, input: NodeId, shift: i64, axis: isize) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        shape.numel()?;
        if shape.rank() == 0 {
            return Err(Error::InvalidMovementRank {
                op: "roll",
                expected: 1,
                actual: 0,
            });
        }
        let axis = normalize_axes(input, shape.rank(), Some(vec![axis]))?[0];
        if shape.dims().contains(&0) {
            return Ok(input);
        }
        let extent = shape.dims()[axis];
        let extent_i64 = i64::try_from(extent).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let normalized = shift.rem_euclid(extent_i64) as usize;
        if normalized == 0 {
            return Ok(input);
        }
        let split = extent - normalized;
        let tail = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                if dimension == axis {
                    (split, size)
                } else {
                    (0, size)
                }
            })
            .collect::<Vec<_>>();
        let head = shape
            .dims()
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                if dimension == axis {
                    (0, split)
                } else {
                    (0, size)
                }
            })
            .collect::<Vec<_>>();

        let tail = self.shrink(input, tail)?;
        let head = self.shrink(input, head)?;
        self.concat(vec![tail, head], axis)
    }

    /// Circularly rolls `input` by paired signed shifts and axes.
    ///
    /// This is tinygrad's tuple `roll(shifts, dims)` form.  It deliberately
    /// keeps duplicate axes: tinygrad's literal loop gives the final pair for
    /// an axis ownership of that axis's shrink bounds, while `repeat` still
    /// doubles each distinct selected axis only once.  All concrete movement
    /// descriptors are planned before the source-literal repeat and shrink
    /// composition can publish a node.
    pub fn roll_axes(
        &mut self,
        input: NodeId,
        shifts: &[i64],
        axes: &[isize],
    ) -> Result<NodeId> {
        if shifts.len() != axes.len() {
            return Err(Error::InvalidRepeat {
                reason: "roll shifts and axes must have equal lengths",
            });
        }

        let source = self.node(input)?;
        let shape = source.shape.clone();
        let dtype = source.dtype;
        let source_elements = shape.numel()?;
        source_elements
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let rank = shape.rank();

        let normalized_axes = axes
            .iter()
            .map(|&axis| {
                let axis = if axis < 0 {
                    axis.checked_add(rank as isize).unwrap_or(isize::MIN)
                } else {
                    axis
                };
                if axis < 0 || axis >= rank as isize {
                    Err(Error::InvalidAxis {
                        node: input,
                        axis: usize::try_from(axis).unwrap_or(usize::MAX),
                        rank,
                    })
                } else {
                    Ok(axis as usize)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // tinygrad checks the paired controls before its empty-tensor fast
        // path.  Its empty and no-axis compositions are identity views.
        if shape.dims().contains(&0) || normalized_axes.is_empty() {
            return Ok(input);
        }

        let mut repeats = vec![1isize; rank];
        let mut bounds = shape
            .dims()
            .iter()
            .map(|&extent| (0, extent))
            .collect::<Vec<_>>();
        for (&shift, &axis) in shifts.iter().zip(&normalized_axes) {
            let extent = shape.dims()[axis];
            let remainder = (shift as i128).rem_euclid(extent as i128) as usize;
            let start = extent - remainder;
            let end = start
                .checked_add(extent)
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
            repeats[axis] = 2;
            // Assignment, rather than accumulation, is source-literal for a
            // duplicated dim in tinygrad's `shrink_arg` loop.
            bounds[axis] = (start, end);
        }

        let repeated_shape = Shape::new(
            shape
                .dims()
                .iter()
                .zip(&repeats)
                .map(|(&extent, &repeat)| {
                    extent
                        .checked_mul(repeat as usize)
                        .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                })
                .collect::<Result<Vec<_>>>()?,
        );
        repeated_shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(repeated_shape.clone()))?;
        // `bounds` is intentionally shaped for the repeated tensor.  Its
        // output must be the original descriptor before the first movement
        // node is allowed to exist.
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;

        let repeated = self.repeat(input, &repeats)?;
        self.shrink(repeated, bounds)
    }

    /// Circularly rolls the flattened logical tensor, then restores its shape.
    ///
    /// This is tinygrad's public `roll(shifts)` form with `dims=None`, kept
    /// distinct from the explicit-axis API. Its flattened extent and signed
    /// shift are checked before flattening can append a movement node.
    pub fn roll_flattened(&mut self, input: NodeId, shift: i64) -> Result<NodeId> {
        let shape = self.node(input)?.shape.clone();
        let elements = shape.numel()?;
        if shape.rank() == 0 || elements == 0 {
            return Ok(input);
        }
        let elements_i64 =
            i64::try_from(elements).map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        if shift.rem_euclid(elements_i64) == 0 {
            return Ok(input);
        }
        let end = isize::try_from(shape.rank() - 1)
            .map_err(|_| Error::ShapeOverflow(shape.clone()))?;
        let flattened = self.flatten(input, 0, end)?;
        let rolled = self.roll(flattened, shift, 0)?;
        self.reshape(rolled, shape)
    }

    /// Source-literal public tinygrad `Tensor.roll(shifts, dims=None)`.
    ///
    /// This is deliberately distinct from the established one-axis
    /// [`Self::roll`] API. It retains Python's scalar/tuple control behavior,
    /// repeated-dimension final-wins rule, and flattening default while a
    /// cloned literal `flatten? -> repeat -> shrink -> reshape?` rehearsal
    /// proves every descriptor and byte extent before the caller graph moves.
    pub fn roll_tinygrad(
        &mut self,
        input: NodeId,
        shifts: RollShifts,
        dims: RollDims,
    ) -> Result<NodeId> {
        let plan = source_roll_plan(self, input, shifts, dims)?;
        let mut rehearsal = self.clone();
        let rehearsed = lower_source_roll(&mut rehearsal, input, &plan)?;
        let output_shape = rehearsal.shape(rehearsed)?.clone();
        let output_dtype = rehearsal.dtype(rehearsed)?;
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;
        debug_assert_eq!(output_shape, plan.output_shape);
        let output = lower_source_roll(self, input, &plan)?;
        debug_assert_eq!(self.shape(output).expect("source roll preflighted"), &plan.output_shape);
        Ok(output)
    }

    /// Public tinygrad default `dims=None` form of [`Self::roll_tinygrad`].
    pub fn roll_tinygrad_default_dims(
        &mut self,
        input: NodeId,
        shifts: RollShifts,
    ) -> Result<NodeId> {
        self.roll_tinygrad(input, shifts, RollDims::None)
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

    /// Source-literal ambient-stream `Tensor.uniform`.
    ///
    /// tinygrad does not create a ranged random buffer: it reserves its
    /// default-F32 `rand` stream, evaluates `(high - low) * rand`, casts that
    /// product at the requested boundary, and finally adds the weak Python
    /// `low` scalar.  Keep those storage boundaries visible instead of routing
    /// this through the seeded raw-range compatibility API.
    pub fn uniform_implicit(
        &mut self,
        shape: impl Into<Shape>,
        low: f64,
        high: f64,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        // Match Python's source predicate literally: NaN bypasses the ordered
        // rejection, while equal/reversed finite (and same-signed infinite)
        // bounds fail before a counter reservation.
        if low >= high {
            return Err(Error::InvalidRandom {
                reason: "uniform requires low < high",
            });
        }
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                .map(|_| ())
        };
        // `rand` is source default F32 even when uniform's eventual cast asks
        // for another storage type. Prove both independently before cloning.
        extent(DType::F32)?;
        extent(dtype)?;

        // Rehearse the full source composite on a cloned graph using an
        // explicit stream. This validates every scalar commitment, cast, and
        // final output descriptor without reserving/mutating the live ambient
        // stream on a malformed late stage.
        let mut rehearsal = self.clone();
        let unit = rehearsal.rand(shape.clone(), DType::F32, 0)?;
        let scaled = rehearsal.scalar_mul(Scalar::F(high - low), unit)?;
        let cast = if rehearsal.dtype(scaled)? == dtype {
            scaled
        } else {
            rehearsal.cast(scaled, dtype)?
        };
        let rehearsed = rehearsal.add_scalar(cast, Scalar::F(low))?;
        let output_shape = rehearsal.shape(rehearsed)?.clone();
        let output_dtype = rehearsal.dtype(rehearsed)?;
        if output_shape != shape {
            return Err(Error::InvalidRandom {
                reason: "uniform output shape changed during preflight",
            });
        }
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;

        // All fallible descriptor work has completed, so this is the first
        // operation that advances the graph's captured default-device stream.
        let unit = self.rand_implicit(shape, DType::F32)?;
        let scaled = self.scalar_mul(Scalar::F(high - low), unit)?;
        let cast = if self.dtype(scaled)? == dtype {
            scaled
        } else {
            self.cast(scaled, dtype)?
        };
        let output = self.add_scalar(cast, Scalar::F(low))?;
        debug_assert_eq!(self.shape(output).expect("uniform preflighted"), &output_shape);
        debug_assert_eq!(self.dtype(output).expect("uniform preflighted"), output_dtype);
        Ok(output)
    }

    /// Omits tinygrad `Tensor.uniform`'s low/high/dtype defaults.
    pub fn uniform_default(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        self.uniform_implicit(shape, 0.0, 1.0, DType::F32)
    }

    /// Source-literal ambient-stream `Tensor.glorot_uniform`.
    ///
    /// The source computes `sqrt(6 / (shape[0] + prod(shape[1:])))` before it
    /// invokes uniform, so all rank/fan/byte failures are resolved before an
    /// implicit stream reservation. The actual random graph is intentionally
    /// delegated to `uniform_implicit` rather than simplified to a ranged
    /// Random node.
    pub fn glorot_uniform_implicit(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        if shape.rank() == 0 {
            return Err(Error::InvalidRandom {
                reason: "glorot_uniform requires rank at least one",
            });
        }
        let fan_out = checked_initializer_tail_fan(&shape)?;
        let fan = shape.dims()[0]
            .checked_add(fan_out)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        // Python's `6 / 0` raises before `uniform` is called; do not turn it
        // into an infinity bound that would reserve a stream.
        if fan == 0 {
            return Err(Error::InvalidRandom {
                reason: "glorot_uniform has zero fan",
            });
        }
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let bound = (6.0 / fan as f64).sqrt();
        self.uniform_implicit(shape, -bound, bound, dtype)
    }

    /// Omits tinygrad `Tensor.glorot_uniform`'s dtype default.
    pub fn glorot_uniform_default(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        self.glorot_uniform_implicit(shape, DType::F32)
    }

    /// Source-literal ambient-stream `Tensor.kaiming_uniform`.
    ///
    /// tinygrad computes `(6 / (1 + a**2) / prod(shape[1:]))**0.5` before it
    /// calls `uniform`. In particular, scalar and rank-one shapes use the
    /// empty-tail identity of one, while a zero tail fan fails before any
    /// ambient stream reservation. The resulting random composition remains
    /// the public source `uniform` graph rather than a ranged Random node.
    pub fn kaiming_uniform_implicit(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        let fan = checked_initializer_tail_fan(&shape)?;
        // Python's ordered division raises for an integer zero denominator
        // before `uniform` runs. Preserve that stream-atomic failure instead
        // of allowing an infinite bound through the random helper.
        if fan == 0 {
            return Err(Error::InvalidRandom {
                reason: "kaiming_uniform has zero fan",
            });
        }
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let bound = (6.0 / (1.0 + a * a) / fan as f64).sqrt();
        self.uniform_implicit(shape, -bound, bound, dtype)
    }

    /// Omits tinygrad `Tensor.kaiming_uniform`'s `a=0.01` default.
    pub fn kaiming_uniform_default_a(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<NodeId> {
        self.kaiming_uniform_implicit(shape, 0.01, dtype)
    }

    /// Omits tinygrad `Tensor.kaiming_uniform`'s `a` and dtype defaults.
    pub fn kaiming_uniform_default(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        self.kaiming_uniform_default_a(shape, DType::F32)
    }

    /// Source-literal ambient-stream `Tensor.normal`.
    ///
    /// The checked-in helper is `std * randn(..., dtype=...) + mean`: randn's
    /// Box-Muller work is F32 internally but it reaches the requested storage
    /// boundary before either weak scalar is consumed. Do not fold mean/std
    /// into the raw seeded normal range used by legacy APIs.
    pub fn normal_implicit(
        &mut self,
        shape: impl Into<Shape>,
        mean: f64,
        std: f64,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        // Python's ordered predicate intentionally lets NaN through while
        // rejecting only values ordered below zero.
        if std < 0.0 {
            return Err(Error::InvalidRandom {
                reason: "normal requires std >= 0",
            });
        }
        let extent = |dtype: DType| {
            shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
                .map(|_| ())
        };
        // Box-Muller reserves two F32 words per output; requested randn
        // storage and the final promoted output are separately rehearsed.
        extent(DType::F32)?;
        extent(dtype)?;

        // Rehearse the exact source chain on an explicit stream before any
        // ambient counter reservation. This closes late scalar/cast/output
        // descriptor failures atomically for the live graph and stream.
        let mut rehearsal = self.clone();
        let standard = rehearsal.randn(shape.clone(), DType::F32, 0)?;
        let standard = if dtype == DType::F32 {
            standard
        } else {
            rehearsal.cast(standard, dtype)?
        };
        let scaled = rehearsal.scalar_mul(Scalar::F(std), standard)?;
        let rehearsed = rehearsal.add_scalar(scaled, Scalar::F(mean))?;
        let output_shape = rehearsal.shape(rehearsed)?.clone();
        let output_dtype = rehearsal.dtype(rehearsed)?;
        if output_shape != shape {
            return Err(Error::InvalidRandom {
                reason: "normal output shape changed during preflight",
            });
        }
        output_shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(output_shape.clone()))?;

        let standard = self.randn_implicit(shape, DType::F32)?;
        let standard = if dtype == DType::F32 {
            standard
        } else {
            self.cast(standard, dtype)?
        };
        let scaled = self.scalar_mul(Scalar::F(std), standard)?;
        let output = self.add_scalar(scaled, Scalar::F(mean))?;
        debug_assert_eq!(self.shape(output).expect("normal preflighted"), &output_shape);
        debug_assert_eq!(self.dtype(output).expect("normal preflighted"), output_dtype);
        Ok(output)
    }

    /// Omits tinygrad `Tensor.normal`'s mean/std/dtype defaults.
    pub fn normal_default(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        self.normal_implicit(shape, 0.0, 1.0, DType::F32)
    }

    /// Source-literal ambient-stream `Tensor.kaiming_normal`.
    ///
    /// tinygrad computes `(2 / (1 + a**2) / prod(shape[1:]))**0.5` before it
    /// invokes normal. Scalar and rank-one shapes use the empty-tail identity
    /// of one; a zero tail fan fails before the Normal stream is reserved.
    /// A zero standard deviation is valid for Normal (including from infinite
    /// `a`), so this intentionally delegates to `normal_implicit` unchanged.
    pub fn kaiming_normal_implicit(
        &mut self,
        shape: impl Into<Shape>,
        a: f64,
        dtype: DType,
    ) -> Result<NodeId> {
        let shape = shape.into();
        let fan = checked_initializer_tail_fan(&shape)?;
        if fan == 0 {
            return Err(Error::InvalidRandom {
                reason: "kaiming_normal has zero fan",
            });
        }
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        let std = (2.0 / (1.0 + a * a) / fan as f64).sqrt();
        self.normal_implicit(shape, 0.0, std, dtype)
    }

    /// Omits tinygrad `Tensor.kaiming_normal`'s `a=0.01` default.
    pub fn kaiming_normal_default_a(
        &mut self,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<NodeId> {
        self.kaiming_normal_implicit(shape, 0.01, dtype)
    }

    /// Omits tinygrad `Tensor.kaiming_normal`'s `a` and dtype defaults.
    pub fn kaiming_normal_default(&mut self, shape: impl Into<Shape>) -> Result<NodeId> {
        self.kaiming_normal_default_a(shape, DType::F32)
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
        let source = self.node(input)?;
        let shape = source.shape.clone();
        let output_dtype = dtype.unwrap_or(source.dtype);
        // `*_like` reads both the source descriptor and its selected output
        // dtype before `Tensor.full` stores the filled result.  Prove both
        // extents before the constant can be published, including an override
        // whose smaller element width would otherwise hide an invalid source.
        shape
            .numel()?
            .checked_mul(source.dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        shape
            .numel()?
            .checked_mul(output_dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
        self.full_with_dtype(shape, value, output_dtype)
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
        let fan = shape.dims()[0]
            .checked_add(checked_initializer_tail_fan(&shape)?)
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))?;
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
        let fan = checked_initializer_tail_fan(&shape)?;
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
        let fan = checked_initializer_tail_fan(&shape)?;
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
