//! Checked-in tinygrad loss helpers composed from inspectable graph operations.
use crate::ir::{
    SourceGatherPlan, logsigmoid_plan, lower_source_gather, one_hot_bool_plan, one_hot_plan,
    source_gather_plan, source_lub, source_weak_scalar_dtype, validate_log_softmax_plan,
};
use crate::{
    DType, Error, Graph, NodeId, ReduceKind, ReductionDType, Result, Scalar, Shape, TensorData,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    None,
    Sum,
    Mean,
}
#[derive(Clone, Copy, Debug)]
pub struct LossOptions {
    pub reduction: Reduction,
    pub class_axis: isize,
    pub ignore_index: Option<i64>,
    pub label_smoothing: f64,
}
impl Default for LossOptions {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            class_axis: 1,
            ignore_index: None,
            label_smoothing: 0.,
        }
    }
}

/// Source-compatible options for tinygrad's
/// [`Tensor.sparse_categorical_crossentropy`].  This intentionally does not
/// reuse [`LossOptions`]: tinygrad fixes the class axis to the final dimension,
/// whereas its public NLL helper fixes it to axis one.
#[derive(Clone, Copy, Debug)]
pub struct SparseCategoricalCrossEntropyOptions {
    /// A value of `-1` is tinygrad's sentinel for *no* ignore mask.
    pub ignore_index: i64,
    pub label_smoothing: f64,
    pub reduction: Reduction,
}
impl Default for SparseCategoricalCrossEntropyOptions {
    fn default() -> Self {
        Self {
            ignore_index: -1,
            label_smoothing: 0.,
            reduction: Reduction::Mean,
        }
    }
}

/// Descriptor-only contract for the literal tinygrad sparse categorical loss.
/// Every fallible shape, dtype, scalar, and byte fact is settled before a
/// helper creates the first view, constant, or graph node.
struct SparseCategoricalCrossEntropyPlan {
    logits_shape: Shape,
    target_shape: Shape,
    class_axis: usize,
    log_dtype: DType,
    selected_shape: Shape,
    selected_sum: ReductionDType,
    total_sum: ReductionDType,
    smoothing: TensorData,
    hard_weight: TensorData,
    ignored: Option<TensorData>,
}

fn sparse_categorical_cross_entropy_plan(
    graph: &Graph,
    logits: NodeId,
    target: NodeId,
    options: SparseCategoricalCrossEntropyOptions,
) -> Result<SparseCategoricalCrossEntropyPlan> {
    let logits_node = graph.node(logits)?;
    let target_node = graph.node(target)?;
    if !target_node.dtype.is_integer() {
        return Err(invalid("sparse targets must be integer"));
    }
    if !(0.0..=1.0).contains(&options.label_smoothing) {
        return Err(invalid("label smoothing must be in [0, 1]"));
    }
    if logits_node.shape.rank() == 0 {
        return Err(invalid("sparse logits require a final class axis"));
    }
    let class_axis = logits_node.shape.rank() - 1;
    let mut expected = logits_node.shape.dims().to_vec();
    let classes = expected.pop().expect("rank checked");
    let target_shape = Shape::new(expected);
    if target_node.shape != target_shape {
        return Err(invalid(
            "target shape must equal logits without final class axis",
        ));
    }
    // Reuse the exact lazy default-I32/I64-upgrade descriptor that backs
    // Graph::one_hot. This replaces the stale eager-I64 class-range sidecar
    // and proves every input/range/compare/Select byte extent before
    // LogSoftmax or any loss-local constant can publish.
    one_hot_plan(graph, target, classes)?;

    // `log_softmax` promotes exact inputs to F32, but preserves every floating
    // storage width. These are the widths subsequently observed by tinygrad's
    // weak smoothing constants and typed sums.
    let log_dtype = if logits_node.dtype.is_float() {
        logits_node.dtype
    } else {
        DType::F32
    };
    let selected_shape = target_shape.clone();
    let selected_sum = ReductionDType::sum_default(log_dtype);
    let total_sum = ReductionDType::sum_default(log_dtype);
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
    };
    // Inputs, log-softmax output, one-hot/mask construction, class reduction,
    // smoothing mean, and the final reduction/divisor all have concrete dense
    // extents before lowering begins.
    extent(&logits_node.shape, logits_node.dtype)?;
    extent(&target_shape, target_node.dtype)?;
    extent(&logits_node.shape, log_dtype)?;
    extent(&logits_node.shape, DType::I32)?; // one-hot/select output
    extent(&target_shape, DType::Bool)?;
    extent(&target_shape, log_dtype)?;
    extent(&selected_shape, selected_sum.accumulator)?;
    extent(&selected_shape, selected_sum.output)?;
    extent(&selected_shape, log_dtype)?; // mean and un-reduced loss
    extent(&Shape::new([]), total_sum.accumulator)?;
    extent(&Shape::new([]), total_sum.output)?;
    extent(&Shape::new([]), DType::I32)?; // Bool mask sum
    if logits_node.shape.broadcast_with(&Shape::new(
        logits_node
            .shape
            .dims()
            .iter()
            .enumerate()
            .map(|(i, &d)| if i == class_axis { 1 } else { d })
            .collect::<Vec<_>>(),
    ))? != logits_node.shape
    {
        return Err(invalid("final class reduction cannot broadcast"));
    }
    let smoothing = TensorData::scalar_with_dtype(Scalar::F(options.label_smoothing), log_dtype);
    let hard_weight =
        TensorData::scalar_with_dtype(Scalar::F(1. - options.label_smoothing), log_dtype);
    if smoothing.dtype() != log_dtype || hard_weight.dtype() != log_dtype {
        return Err(invalid("sparse smoothing scalar dtype mismatch"));
    }
    let ignored = (options.ignore_index != -1)
        .then(|| TensorData::scalar_with_dtype(Scalar::I(options.ignore_index), target_node.dtype));
    if ignored
        .as_ref()
        .is_some_and(|value| value.dtype() != target_node.dtype)
    {
        return Err(invalid("ignore-index scalar dtype mismatch"));
    }
    Ok(SparseCategoricalCrossEntropyPlan {
        logits_shape: logits_node.shape.clone(),
        target_shape,
        class_axis,
        log_dtype,
        selected_shape,
        selected_sum,
        total_sum,
        smoothing,
        hard_weight,
        ignored,
    })
}
fn invalid(reason: &'static str) -> Error {
    Error::InvalidAttention { reason }
}

/// Complete descriptor contract for tinygrad's public `Tensor.cross_entropy`.
/// Unlike the older generic free function, source selects hard labels solely
/// from shape inequality, then smooths both hard and probability targets by
/// the same literal two-stage affine expression.
struct TinygradCrossEntropyPlan {
    class_axis: usize,
    one_hot: bool,
    first_weight: TensorData,
    smooth: TensorData,
    class_loss_shape: Shape,
    class_loss_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
}

fn tinygrad_cross_entropy_plan(
    graph: &Graph,
    logits: NodeId,
    target: NodeId,
    reduction: Reduction,
    label_smoothing: f64,
) -> Result<TinygradCrossEntropyPlan> {
    if !(0.0..=1.0).contains(&label_smoothing) {
        return Err(invalid("label smoothing must be in [0, 1]"));
    }
    let logits_node = graph.node(logits)?;
    let target_node = graph.node(target)?;
    let logits_shape = logits_node.shape.clone();
    let target_original_shape = target_node.shape.clone();
    let class_axis = match logits_shape.rank() {
        0 => return Err(invalid("cross entropy logits require a class axis")),
        1 => 0,
        _ => 1,
    };
    let classes = logits_shape.dims()[class_axis];
    // `label_smoothing / int(Y.shape[classes_dim])` is evaluated even for
    // zero smoothing, so Python raises before a graph node can exist.
    if classes == 0 {
        return Err(Error::DivisionByZero {
            op: "cross entropy label smoothing",
        });
    }
    let one_hot = logits_shape != target_original_shape;
    if one_hot {
        let mut expected = logits_shape.dims().to_vec();
        expected.remove(class_axis);
        if target_original_shape != Shape::new(expected) {
            return Err(invalid("target shape must equal logits without class axis"));
        }
        one_hot_bool_plan(graph, target, classes)?;
    }
    let target_shape = if one_hot {
        logits_shape.clone()
    } else {
        target_original_shape.clone()
    };
    let target_dtype = if one_hot {
        DType::Bool
    } else {
        target_node.dtype
    };
    let log_dtype = if logits_node.dtype.is_float() {
        logits_node.dtype
    } else {
        DType::F32
    };
    validate_log_softmax_plan(
        logits,
        &logits_shape,
        logits_node.dtype,
        class_axis as isize,
        None,
    )?;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&logits_shape, logits_node.dtype)?;
    extent(&target_original_shape, target_node.dtype)?;
    extent(&target_shape, target_dtype)?;
    extent(&logits_shape, log_dtype)?;
    if target_shape != logits_shape {
        return Err(invalid("cross entropy one-hot shape mismatch"));
    }

    // `(1-smoothing) * Y` and `smoothing / classes` commit independent
    // Python floats at their individual source consumers.
    let first_dtype = source_weak_scalar_dtype(target_dtype, Scalar::F(1.0 - label_smoothing));
    let first_weight = TensorData::scalar_with_dtype(Scalar::F(1.0 - label_smoothing), first_dtype);
    let first_product_dtype = source_lub(first_dtype, target_dtype);
    let smooth_dtype = source_weak_scalar_dtype(
        first_product_dtype,
        Scalar::F(label_smoothing / classes as f64),
    );
    let smooth =
        TensorData::scalar_with_dtype(Scalar::F(label_smoothing / classes as f64), smooth_dtype);
    let target_dtype = source_lub(first_product_dtype, smooth_dtype);
    for (shape, dtype) in [
        (first_weight.shape(), first_weight.dtype()),
        (&target_shape, first_product_dtype),
        (smooth.shape(), smooth.dtype()),
        (&target_shape, target_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if first_weight.shape() != &Shape::new([])
        || smooth.shape() != &Shape::new([])
        || target_shape.broadcast_with(first_weight.shape())? != target_shape
        || target_shape.broadcast_with(smooth.shape())? != target_shape
    {
        return Err(invalid("cross entropy scalar promotion"));
    }

    let weighted_shape = logits_shape.broadcast_with(&target_shape)?;
    let weighted_dtype = source_lub(log_dtype, target_dtype);
    extent(&logits_shape, weighted_dtype)?;
    extent(&target_shape, weighted_dtype)?;
    extent(&weighted_shape, weighted_dtype)?;
    let mut class_dims = weighted_shape.dims().to_vec();
    class_dims.remove(class_axis);
    let class_loss_shape = Shape::new(class_dims);
    let class_sum = ReductionDType::sum_default(weighted_dtype);
    extent(&weighted_shape, class_sum.accumulator)?;
    extent(&class_loss_shape, class_sum.accumulator)?;
    extent(&class_loss_shape, class_sum.output)?;
    let class_loss_dtype = class_sum.output;
    let scalar = Shape::new([]);
    let (output_shape, output_dtype) = match reduction {
        Reduction::None => (class_loss_shape.clone(), class_loss_dtype),
        Reduction::Sum => {
            let dtypes = ReductionDType::sum_default(class_loss_dtype);
            extent(&class_loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, dtypes.output)?;
            (scalar, dtypes.output)
        }
        Reduction::Mean => {
            let dtypes = ReductionDType::sum_default(class_loss_dtype);
            let division_dtype = if dtypes.accumulator.is_float() {
                dtypes.accumulator
            } else {
                DType::F32
            };
            extent(&class_loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, division_dtype)?;
            extent(&scalar, class_loss_dtype)?;
            (scalar, class_loss_dtype)
        }
    };
    Ok(TinygradCrossEntropyPlan {
        class_axis,
        one_hot,
        first_weight,
        smooth,
        class_loss_shape,
        class_loss_dtype,
        output_shape,
        output_dtype,
    })
}
fn axis(graph: &Graph, node: NodeId, axis: isize) -> Result<usize> {
    let rank = graph.shape(node)?.rank() as isize;
    let a = if axis < 0 { rank + axis } else { axis };
    if a < 0 || a >= rank {
        return Err(invalid("invalid class axis"));
    }
    Ok(a as usize)
}
fn reduce(graph: &mut Graph, input: NodeId, reduction: Reduction) -> Result<NodeId> {
    match reduction {
        Reduction::None => Ok(input),
        Reduction::Sum => graph.reduce(input, ReduceKind::Sum, None, false),
        Reduction::Mean => graph.reduce(input, ReduceKind::Mean, None, false),
    }
}
fn target_shape(graph: &Graph, logits: NodeId, target: NodeId, axis: usize) -> Result<()> {
    let mut expected = graph.shape(logits)?.dims().to_vec();
    expected.remove(axis);
    if graph.shape(target)?.dims() != expected {
        return Err(invalid("target shape must equal logits without class axis"));
    }
    Ok(())
}
/// Descriptor-only literal for tinygrad `Tensor.binary_crossentropy`.
///
/// The two live inputs are independently promoted at every source consumer;
/// in particular the Python integer `1` is weakly committed separately for
/// `1 - Y` and `1 - self`, rather than published as an ambient F64 constant.
struct BinaryCrossEntropyPlan {
    loss_shape: Shape,
    loss_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
}

/// Descriptor-only literal for tinygrad `Tensor.binary_crossentropy_logits`.
/// It covers both independent LogSigmoid calls, the optional live
/// `pos_weight` branch, weak `1` commitments, and final source reduction
/// before a nested helper can publish any graph state.
struct BinaryCrossEntropyLogitsPlan {
    loss_shape: Shape,
    loss_dtype: DType,
    output_shape: Shape,
    output_dtype: DType,
}

fn binary_crossentropy_logits_plan(
    graph: &Graph,
    logits: NodeId,
    target: NodeId,
    pos_weight: Option<NodeId>,
    reduction: Reduction,
) -> Result<BinaryCrossEntropyLogitsPlan> {
    let logits_node = graph.node(logits)?;
    let target_node = graph.node(target)?;
    let logits_shape = logits_node.shape.clone();
    let target_shape = target_node.shape.clone();
    let logits_dtype = logits_node.dtype;
    let target_dtype = target_node.dtype;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&logits_shape, logits_dtype)?;
    extent(&target_shape, target_dtype)?;

    // `self.logsigmoid()` and `(-self).logsigmoid()`. The shared pure
    // LogSigmoid descriptor proves its own outer negation, Softplus literal,
    // typed weak constants, and final negation. The explicit source `-self`
    // between the two calls has the original storage width.
    logsigmoid_plan(&logits_shape, logits_dtype)?;
    extent(&logits_shape, logits_dtype)?;
    if logits_dtype == DType::Bool {
        extent(&Shape::new([]), DType::Bool)?;
        extent(&logits_shape, DType::Bool)?;
    }
    logsigmoid_plan(&logits_shape, logits_dtype)?;
    let log_dtype = if logits_dtype.is_float() {
        logits_dtype
    } else {
        DType::F32
    };
    extent(&logits_shape, log_dtype)?;

    // `(1 if pos_weight is None else pos_weight) * Y * log_p`.
    let (weight_shape, weight_dtype) = if let Some(weight) = pos_weight {
        let weight_node = graph.node(weight)?;
        extent(&weight_node.shape, weight_node.dtype)?;
        (weight_node.shape.clone(), weight_node.dtype)
    } else {
        let dtype = source_weak_scalar_dtype(target_dtype, Scalar::I(1));
        let one = TensorData::scalar_with_dtype(Scalar::I(1), dtype);
        extent(one.shape(), one.dtype())?;
        if one.shape() != &Shape::new([]) || one.dtype() != dtype {
            return Err(invalid("binary logits implicit pos_weight promotion"));
        }
        (Shape::new([]), dtype)
    };
    let weighted_target_shape = weight_shape.broadcast_with(&target_shape)?;
    let weighted_target_dtype = source_lub(weight_dtype, target_dtype);
    extent(&weight_shape, weighted_target_dtype)?;
    extent(&target_shape, weighted_target_dtype)?;
    extent(&weighted_target_shape, weighted_target_dtype)?;
    let positive_shape = weighted_target_shape.broadcast_with(&logits_shape)?;
    let positive_dtype = source_lub(weighted_target_dtype, log_dtype);
    extent(&weighted_target_shape, positive_dtype)?;
    extent(&logits_shape, positive_dtype)?;
    extent(&positive_shape, positive_dtype)?;

    // `(1-Y) * log_1_minus_p`, where this source `1` is a separate weak
    // scalar commitment from the absent-pos-weight one above.
    let complement_one_dtype = source_weak_scalar_dtype(target_dtype, Scalar::I(1));
    let complement_dtype = source_lub(complement_one_dtype, target_dtype);
    let complement_one = TensorData::scalar_with_dtype(Scalar::I(1), complement_one_dtype);
    for (shape, dtype) in [
        (complement_one.shape(), complement_one.dtype()),
        (&target_shape, complement_dtype),
    ] {
        extent(shape, dtype)?;
    }
    if complement_one.shape() != &Shape::new([])
        || complement_one.dtype() != complement_one_dtype
        || target_shape.broadcast_with(complement_one.shape())? != target_shape
    {
        return Err(invalid("binary logits complement promotion"));
    }
    let negative_shape = target_shape.broadcast_with(&logits_shape)?;
    let negative_dtype = source_lub(complement_dtype, log_dtype);
    extent(&target_shape, negative_dtype)?;
    extent(&logits_shape, negative_dtype)?;
    extent(&negative_shape, negative_dtype)?;

    // `-(positive + negative)` then `_do_reduction(reduction)`.
    let loss_shape = positive_shape.broadcast_with(&negative_shape)?;
    let loss_dtype = source_lub(positive_dtype, negative_dtype);
    extent(&positive_shape, loss_dtype)?;
    extent(&negative_shape, loss_dtype)?;
    extent(&loss_shape, loss_dtype)?; // ADD then outer Neg
    let scalar = Shape::new([]);
    let (output_shape, output_dtype) = match reduction {
        Reduction::None => (loss_shape.clone(), loss_dtype),
        Reduction::Sum => {
            let dtypes = ReductionDType::sum_default(loss_dtype);
            extent(&loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, dtypes.output)?;
            (scalar, dtypes.output)
        }
        Reduction::Mean => {
            let dtypes = ReductionDType::sum_default(loss_dtype);
            let division_dtype = if dtypes.accumulator.is_float() {
                dtypes.accumulator
            } else {
                DType::F32
            };
            let output_dtype = if loss_dtype.is_float() {
                loss_dtype
            } else {
                DType::F32
            };
            extent(&loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, division_dtype)?;
            extent(&scalar, output_dtype)?;
            let divisor = TensorData::scalar_with_dtype(
                Scalar::F(loss_shape.numel()? as f64),
                division_dtype,
            );
            extent(divisor.shape(), divisor.dtype())?;
            (scalar, output_dtype)
        }
    };
    Ok(BinaryCrossEntropyLogitsPlan {
        loss_shape,
        loss_dtype,
        output_shape,
        output_dtype,
    })
}

fn binary_crossentropy_plan(
    graph: &Graph,
    input: NodeId,
    target: NodeId,
    reduction: Reduction,
) -> Result<BinaryCrossEntropyPlan> {
    let input_node = graph.node(input)?;
    let target_node = graph.node(target)?;
    let input_shape = input_node.shape.clone();
    let target_shape = target_node.shape.clone();
    let input_dtype = input_node.dtype;
    let target_dtype = target_node.dtype;
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&input_shape, input_dtype)?;
    extent(&target_shape, target_dtype)?;

    // `-Y * self.log()`.
    let log_input_dtype = if input_dtype.is_float() {
        input_dtype
    } else {
        DType::F32
    };
    let left_shape = target_shape.broadcast_with(&input_shape)?;
    let left_dtype = source_lub(target_dtype, log_input_dtype);
    extent(&target_shape, target_dtype)?; // Neg target
    if target_dtype == DType::Bool {
        // `Tensor.neg()` is literal logical-not for Bool and publishes this
        // scalar only after the whole BCE plan has succeeded.
        extent(&Shape::new([]), DType::Bool)?;
        extent(&target_shape, DType::Bool)?;
    }
    extent(&input_shape, log_input_dtype)?; // Log2 and Log result
    extent(&Shape::new([]), log_input_dtype)?; // Log's typed ln(2)
    extent(&target_shape, left_dtype)?;
    extent(&input_shape, left_dtype)?;
    extent(&left_shape, left_dtype)?;

    // `(1 - Y) * (1 - self).log()`: each Python `1` is committed at its own
    // source `_broadcasted` consumer, then subtraction remains `a + (-b)`.
    let target_one_dtype = source_weak_scalar_dtype(target_dtype, Scalar::I(1));
    let target_complement_dtype = source_lub(target_one_dtype, target_dtype);
    let target_one = TensorData::scalar_with_dtype(Scalar::I(1), target_one_dtype);
    let input_one_dtype = source_weak_scalar_dtype(input_dtype, Scalar::I(1));
    let input_complement_dtype = source_lub(input_one_dtype, input_dtype);
    let input_one = TensorData::scalar_with_dtype(Scalar::I(1), input_one_dtype);
    for (shape, dtype) in [
        (target_one.shape(), target_one.dtype()),
        (&target_shape, target_complement_dtype),
        (input_one.shape(), input_one.dtype()),
        (&input_shape, input_complement_dtype),
    ] {
        extent(shape, dtype)?;
    }
    let log_complement_dtype = if input_complement_dtype.is_float() {
        input_complement_dtype
    } else {
        DType::F32
    };
    let right_shape = target_shape.broadcast_with(&input_shape)?;
    let right_dtype = source_lub(target_complement_dtype, log_complement_dtype);
    extent(&input_shape, log_complement_dtype)?; // Log2 and Log
    extent(&Shape::new([]), log_complement_dtype)?; // Log's typed ln(2)
    extent(&target_shape, right_dtype)?;
    extent(&input_shape, right_dtype)?;
    extent(&right_shape, right_dtype)?;

    // The outer spelling is a literal subtraction, not `neg(right) + left`.
    let loss_shape = left_shape.broadcast_with(&right_shape)?;
    let loss_dtype = source_lub(left_dtype, right_dtype);
    extent(&left_shape, loss_dtype)?;
    extent(&right_shape, loss_dtype)?;
    extent(&loss_shape, loss_dtype)?; // right negation and final ADD

    let scalar = Shape::new([]);
    let (output_shape, output_dtype) = match reduction {
        Reduction::None => (loss_shape.clone(), loss_dtype),
        Reduction::Sum => {
            let dtypes = ReductionDType::sum_default(loss_dtype);
            extent(&loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, dtypes.output)?;
            (scalar, dtypes.output)
        }
        Reduction::Mean => {
            let dtypes = ReductionDType::sum_default(loss_dtype);
            let division_dtype = if dtypes.accumulator.is_float() {
                dtypes.accumulator
            } else {
                DType::F32
            };
            let output_dtype = if loss_dtype.is_float() {
                loss_dtype
            } else {
                DType::F32
            };
            extent(&loss_shape, dtypes.accumulator)?;
            extent(&scalar, dtypes.accumulator)?;
            extent(&scalar, division_dtype)?; // cast/sum divisor/reciprocal/product
            extent(&scalar, output_dtype)?;
            let divisor = TensorData::scalar_with_dtype(
                Scalar::F(loss_shape.numel()? as f64),
                division_dtype,
            );
            extent(divisor.shape(), divisor.dtype())?;
            (scalar, output_dtype)
        }
    };
    if target_one.shape() != &Shape::new([])
        || input_one.shape() != &Shape::new([])
        || target_one.dtype() != target_one_dtype
        || input_one.dtype() != input_one_dtype
        || target_shape.broadcast_with(target_one.shape())? != target_shape
        || input_shape.broadcast_with(input_one.shape())? != input_shape
    {
        return Err(invalid("binary cross entropy scalar promotion"));
    }
    Ok(BinaryCrossEntropyPlan {
        loss_shape,
        loss_dtype,
        output_shape,
        output_dtype,
    })
}
fn probability_target(graph: &Graph, logits: NodeId, target: NodeId) -> Result<()> {
    if !graph.dtype(logits)?.is_float() || !graph.dtype(target)?.is_float() {
        return Err(invalid("logits and probability targets must be float"));
    }
    if graph.shape(logits)? != graph.shape(target)? {
        return Err(invalid("probability target must match logits shape"));
    }
    Ok(())
}
fn nll_inputs(
    graph: &Graph,
    log_probabilities: NodeId,
    target: NodeId,
    weight: Option<NodeId>,
    class_axis: isize,
) -> Result<usize> {
    if !graph.dtype(log_probabilities)?.is_float() {
        return Err(invalid("NLL log probabilities must be float"));
    }
    if !graph.dtype(target)?.is_integer() {
        return Err(invalid("NLL targets must be integer"));
    }
    let axis = axis(graph, log_probabilities, class_axis)?;
    target_shape(graph, log_probabilities, target, axis)?;
    if let Some(weight) = weight {
        if graph.shape(weight)?.dims() != [graph.shape(log_probabilities)?.dims()[axis]] {
            return Err(invalid("NLL weight must have class shape"));
        }
    }
    Ok(axis)
}
fn one_hot(graph: &mut Graph, logits: NodeId, target: NodeId, axis: usize) -> Result<NodeId> {
    let classes = graph.shape(logits)?.dims()[axis];
    let hot = graph.one_hot(target, classes)?;
    let rank = graph.shape(logits)?.rank();
    let mut axes = Vec::with_capacity(rank);
    for out in 0..rank {
        axes.push(if out == axis {
            rank - 1
        } else if out < axis {
            out
        } else {
            out - 1
        })
    }
    graph.permute(hot, axes)
}
fn one_hot_bool(graph: &mut Graph, logits: NodeId, target: NodeId, axis: usize) -> Result<NodeId> {
    let classes = graph.shape(logits)?.dims()[axis];
    let hot = graph.one_hot_bool(target, classes)?;
    let rank = graph.shape(logits)?.rank();
    let mut axes = Vec::with_capacity(rank);
    for out in 0..rank {
        axes.push(if out == axis {
            rank - 1
        } else if out < axis {
            out
        } else {
            out - 1
        });
    }
    graph.permute(hot, axes)
}
fn masked_reduce(
    graph: &mut Graph,
    loss: NodeId,
    mask: Option<NodeId>,
    reduction: Reduction,
) -> Result<NodeId> {
    if reduction != Reduction::Mean {
        return reduce(graph, loss, reduction);
    }
    let Some(mask) = mask else {
        return reduce(graph, loss, Reduction::Mean);
    };
    let sum = graph.reduce(loss, ReduceKind::Sum, None, false)?;
    let count = graph.reduce(mask, ReduceKind::Sum, None, false)?;
    let dtype = graph.dtype(sum)?;
    let denom = graph.cast(count, dtype)?;
    graph.div(sum, denom)
}
fn weighted_reduce(
    graph: &mut Graph,
    loss: NodeId,
    factor: NodeId,
    reduction: Reduction,
) -> Result<NodeId> {
    if reduction != Reduction::Mean {
        return reduce(graph, loss, reduction);
    }
    let sum = graph.reduce(loss, ReduceKind::Sum, None, false)?;
    let denominator = graph.reduce(factor, ReduceKind::Sum, None, false)?;
    let denominator = graph.cast(denominator, graph.dtype(sum)?)?;
    graph.div(sum, denominator)
}
impl Graph {
    /// Checked-in tinygrad `Tensor.binary_crossentropy(Y, reduction)`.
    ///
    /// This intentionally remains distinct from the stable logits loss. It
    /// preserves tinygrad's unclamped `log` composition and live-target
    /// broadcast/promotion behavior.
    pub fn binary_crossentropy(
        &mut self,
        input: NodeId,
        target: NodeId,
        reduction: Reduction,
    ) -> Result<NodeId> {
        let plan = binary_crossentropy_plan(self, input, target, reduction)?;
        let negative_target = self.neg(target)?;
        let log_input = self.log(input)?;
        let left = self.mul(negative_target, log_input)?;
        let complement_target = self.scalar_sub(Scalar::I(1), target)?;
        let complement_input = self.scalar_sub(Scalar::I(1), input)?;
        let log_complement = self.log(complement_input)?;
        let right = self.mul(complement_target, log_complement)?;
        let loss = self.sub(left, right)?;
        let output = match reduction {
            Reduction::None => loss,
            Reduction::Sum => self.sum_default(loss)?,
            Reduction::Mean => self.mean_default(loss)?,
        };
        debug_assert_eq!(
            self.shape(loss).expect("binary crossentropy preflighted"),
            &plan.loss_shape
        );
        debug_assert_eq!(
            self.dtype(loss).expect("binary crossentropy preflighted"),
            plan.loss_dtype
        );
        debug_assert_eq!(
            self.shape(output).expect("binary crossentropy preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("binary crossentropy preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// tinygrad's omitted `reduction` argument defaults to `"mean"`.
    pub fn binary_crossentropy_default(&mut self, input: NodeId, target: NodeId) -> Result<NodeId> {
        self.binary_crossentropy(input, target, Reduction::Mean)
    }

    /// Checked-in tinygrad `Tensor.binary_crossentropy_logits(Y, reduction,
    /// pos_weight)`. This is intentionally separate from probability BCE:
    /// its two source LogSigmoid branches operate on the unbounded logits.
    pub fn binary_crossentropy_logits(
        &mut self,
        logits: NodeId,
        target: NodeId,
        reduction: Reduction,
        pos_weight: Option<NodeId>,
    ) -> Result<NodeId> {
        let plan = binary_crossentropy_logits_plan(self, logits, target, pos_weight, reduction)?;
        let log_p = self.logsigmoid(logits)?;
        let neg_logits = self.neg(logits)?;
        let log_1_minus_p = self.logsigmoid(neg_logits)?;
        let weighted_target = match pos_weight {
            Some(weight) => self.mul(weight, target)?,
            None => self.scalar_mul(Scalar::I(1), target)?,
        };
        let positive = self.mul(weighted_target, log_p)?;
        let complement = self.scalar_sub(Scalar::I(1), target)?;
        let negative = self.mul(complement, log_1_minus_p)?;
        let sum = self.add(positive, negative)?;
        let loss = self.neg(sum)?;
        let output = match reduction {
            Reduction::None => loss,
            Reduction::Sum => self.sum_default(loss)?,
            Reduction::Mean => self.mean_default(loss)?,
        };
        debug_assert_eq!(
            self.shape(loss).expect("binary logits preflighted"),
            &plan.loss_shape
        );
        debug_assert_eq!(
            self.dtype(loss).expect("binary logits preflighted"),
            plan.loss_dtype
        );
        debug_assert_eq!(
            self.shape(output).expect("binary logits preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("binary logits preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// tinygrad's omitted logits-BCE reduction defaults to `"mean"`.
    pub fn binary_crossentropy_logits_default(
        &mut self,
        logits: NodeId,
        target: NodeId,
        pos_weight: Option<NodeId>,
    ) -> Result<NodeId> {
        self.binary_crossentropy_logits(logits, target, Reduction::Mean, pos_weight)
    }
}

/// Probability-target binary cross entropy, matching tinygrad's unclamped log contract.
///
/// Backward-compatible free-function spelling for the public graph method.
pub fn binary_cross_entropy(
    graph: &mut Graph,
    input: NodeId,
    target: NodeId,
    reduction: Reduction,
) -> Result<NodeId> {
    graph.binary_crossentropy(input, target, reduction)
}
/// Stable binary cross entropy from logits, optionally applying `pos_weight`.
pub fn binary_cross_entropy_with_logits(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    pos_weight: Option<NodeId>,
    reduction: Reduction,
) -> Result<NodeId> {
    graph.binary_crossentropy_logits(logits, target, reduction, pos_weight)
}
/// Sparse categorical CE with an explicitly selected class axis.
pub fn sparse_categorical_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    options: LossOptions,
) -> Result<NodeId> {
    if !graph.dtype(logits)?.is_float() || !graph.dtype(target)?.is_integer() {
        return Err(invalid("logits must be float and sparse targets integer"));
    }
    if !(0.0..=1.0).contains(&options.label_smoothing) {
        return Err(invalid("label smoothing must be in [0, 1]"));
    }
    let a = axis(graph, logits, options.class_axis)?;
    target_shape(graph, logits, target, a)?;
    let logp = graph.log_softmax(logits, a as isize, None)?;
    let hot = one_hot(graph, logits, target, a)?;
    let mask = if let Some(ignore) = options.ignore_index {
        let ignored = graph.constant(TensorData::scalar_with_dtype(
            Scalar::I(ignore),
            graph.dtype(target)?,
        ));
        Some(graph.ne(target, ignored)?)
    } else {
        None
    };
    let hot = if let Some(mask) = mask {
        let mut dims = graph.shape(mask)?.dims().to_vec();
        dims.insert(a, 1);
        let mask = graph.reshape(mask, Shape::new(dims))?;
        graph.mul(hot, mask)?
    } else {
        hot
    };
    let weighted = graph.mul(logp, hot)?;
    let picked = graph.reduce(weighted, ReduceKind::Sum, Some(vec![a as isize]), false)?;
    let loss = if options.label_smoothing == 0. {
        graph.neg(picked)?
    } else {
        let mean = graph.reduce(logp, ReduceKind::Mean, Some(vec![a as isize]), false)?;
        let one = graph.constant(TensorData::scalar((1. - options.label_smoothing) as f32));
        let smooth = graph.constant(TensorData::scalar(options.label_smoothing as f32));
        let hard = graph.mul(one, picked)?;
        let softened = graph.mul(smooth, mean)?;
        let combined = graph.add(hard, softened)?;
        graph.neg(combined)?
    };
    masked_reduce(graph, loss, mask, options.reduction)
}

impl Graph {
    /// Literal tinygrad `Tensor.sparse_categorical_crossentropy` lowering.
    ///
    /// The source always uses the final logits axis as classes, treats `-1`
    /// as a disabled ignore-mask sentinel, and evaluates the smoothing branch
    /// even when its scalar is zero. That latter branch is observable for
    /// empty class axes and NaN payloads.
    pub fn sparse_categorical_crossentropy(
        &mut self,
        logits: NodeId,
        target: NodeId,
        options: SparseCategoricalCrossEntropyOptions,
    ) -> Result<NodeId> {
        let plan = sparse_categorical_cross_entropy_plan(self, logits, target, options)?;

        // `Graph::log_softmax`, `one_hot`, `mean_with_axes`, and
        // `reduce_with_dtypes` each carry their own pure descriptor plans. The
        // enclosing plan above proves their cross-operation shapes, widths,
        // and scalar promotion contract before this first mutation.
        let logp = self.log_softmax(logits, -1, None)?;
        debug_assert_eq!(
            self.shape(logp).expect("sparse CE preflighted"),
            &plan.logits_shape
        );
        debug_assert_eq!(
            self.dtype(logp).expect("sparse CE preflighted"),
            plan.log_dtype
        );
        let hot = one_hot(self, logits, target, plan.class_axis)?;
        let mask = if let Some(ignored) = plan.ignored {
            let ignored = self.constant(ignored);
            self.ne(target, ignored)?
        } else {
            self.full_with_dtype(plan.target_shape.clone(), Scalar::Bool(true), DType::Bool)?
        };
        let mut mask_shape = plan.target_shape.dims().to_vec();
        mask_shape.insert(plan.class_axis, 1);
        let mask = self.reshape(mask, Shape::new(mask_shape))?;
        let masked_hot = self.mul(hot, mask)?;
        let weighted = self.mul(logp, masked_hot)?;
        let picked = self.reduce_with_dtypes(
            weighted,
            ReduceKind::Sum,
            Some(vec![plan.class_axis as isize]),
            false,
            plan.selected_sum,
        )?;
        let mean = self.mean_with_axes(logp, Some(vec![plan.class_axis as isize]), false)?;
        let flat_mask = self.reshape(mask, plan.target_shape.clone())?;
        let mean_masked = self.mul(mean, flat_mask)?;
        let hard_weight = self.constant(plan.hard_weight);
        let hard = self.mul(hard_weight, picked)?;
        let smoothing = self.constant(plan.smoothing);
        let smooth = self.mul(smoothing, mean_masked)?;
        let unreduced = self.add(hard, smooth)?;
        let output = match options.reduction {
            Reduction::None => self.neg(unreduced)?,
            Reduction::Sum => {
                let summed = self.reduce_with_dtypes(
                    unreduced,
                    ReduceKind::Sum,
                    None,
                    false,
                    plan.total_sum,
                )?;
                self.neg(summed)?
            }
            Reduction::Mean => {
                let numerator = self.reduce_with_dtypes(
                    unreduced,
                    ReduceKind::Sum,
                    None,
                    false,
                    plan.total_sum,
                )?;
                let denominator = self.reduce_with_dtypes(
                    flat_mask,
                    ReduceKind::Sum,
                    None,
                    false,
                    ReductionDType::sum_default(DType::Bool),
                )?;
                let numerator = self.neg(numerator)?;
                self.div(numerator, denominator)?
            }
        };
        let output_shape = match options.reduction {
            Reduction::None => plan.selected_shape,
            Reduction::Sum | Reduction::Mean => Shape::new([]),
        };
        debug_assert_eq!(
            self.shape(output).expect("sparse CE preflighted"),
            &output_shape
        );
        Ok(output)
    }

    /// tinygrad's omitted sparse categorical-loss arguments.
    pub fn sparse_categorical_crossentropy_default(
        &mut self,
        logits: NodeId,
        target: NodeId,
    ) -> Result<NodeId> {
        self.sparse_categorical_crossentropy(
            logits,
            target,
            SparseCategoricalCrossEntropyOptions::default(),
        )
    }
}

/// Backward-compatible free-function spelling for the source-literal Graph
/// surface.
pub fn sparse_categorical_cross_entropy_tinygrad(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    options: SparseCategoricalCrossEntropyOptions,
) -> Result<NodeId> {
    graph.sparse_categorical_crossentropy(logits, target, options)
}

impl Graph {
    /// Exact checked-in tinygrad `Tensor.cross_entropy` surface. Class labels
    /// are selected by shape, not dtype; this deliberately remains separate
    /// from the legacy configurable free function and sparse helper.
    pub fn cross_entropy(
        &mut self,
        logits: NodeId,
        target: NodeId,
        reduction: Reduction,
        label_smoothing: f64,
    ) -> Result<NodeId> {
        let plan = tinygrad_cross_entropy_plan(self, logits, target, reduction, label_smoothing)?;
        let target = if plan.one_hot {
            one_hot_bool(self, logits, target, plan.class_axis)?
        } else {
            target
        };
        let first_weight = self.constant(plan.first_weight);
        let target = self.mul(first_weight, target)?;
        let smooth = self.constant(plan.smooth);
        let target = self.add(target, smooth)?;
        let logp = self.log_softmax(logits, plan.class_axis as isize, None)?;
        let weighted = self.mul(logp, target)?;
        let summed =
            self.sum_with_options(weighted, Some(vec![plan.class_axis as isize]), false, None)?;
        let loss = self.neg(summed)?;
        let output = match reduction {
            Reduction::None => loss,
            Reduction::Sum => self.sum_default(loss)?,
            Reduction::Mean => self.mean_default(loss)?,
        };
        debug_assert_eq!(
            self.shape(loss).expect("cross entropy preflighted"),
            &plan.class_loss_shape
        );
        debug_assert_eq!(
            self.dtype(loss).expect("cross entropy preflighted"),
            plan.class_loss_dtype
        );
        debug_assert_eq!(
            self.shape(output).expect("cross entropy preflighted"),
            &plan.output_shape
        );
        debug_assert_eq!(
            self.dtype(output).expect("cross entropy preflighted"),
            plan.output_dtype
        );
        Ok(output)
    }

    /// tinygrad's omitted public cross-entropy arguments.
    pub fn cross_entropy_default(&mut self, logits: NodeId, target: NodeId) -> Result<NodeId> {
        self.cross_entropy(logits, target, Reduction::Mean, 0.0)
    }
}

/// Cross entropy accepts integer targets or probability targets of the logits shape.
pub fn cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    options: LossOptions,
) -> Result<NodeId> {
    if graph.dtype(target)?.is_integer() {
        return sparse_categorical_cross_entropy(graph, logits, target, options);
    }
    probability_target(graph, logits, target)?;
    if !(0.0..=1.0).contains(&options.label_smoothing) {
        return Err(invalid("label smoothing must be in [0, 1]"));
    }
    let a = axis(graph, logits, options.class_axis)?;
    let target = if options.label_smoothing == 0. {
        target
    } else {
        let classes = graph.shape(logits)?.dims()[a] as f32;
        let one = graph.constant(TensorData::scalar((1. - options.label_smoothing) as f32));
        let smooth = graph.constant(TensorData::scalar(
            (options.label_smoothing as f32) / classes,
        ));
        let scaled = graph.mul(one, target)?;
        graph.add(scaled, smooth)?
    };
    let logp = graph.log_softmax(logits, a as isize, None)?;
    let weighted = graph.mul(logp, target)?;
    let summed = graph.reduce(weighted, ReduceKind::Sum, Some(vec![a as isize]), false)?;
    let loss = graph.neg(summed)?;
    reduce(graph, loss, options.reduction)
}
/// NLL for log probabilities and sparse integer targets; optional class weights are rank-one.

struct NllLossPlan {
    main: SourceGatherPlan,
    weight: Option<SourceGatherPlan>,
    index_shape: Shape,
    flat_shape: Shape,
    target_shape: Shape,
}
fn nll_loss_plan(
    graph: &Graph,
    logp: NodeId,
    target: NodeId,
    weight: Option<NodeId>,
    ignore: Option<i64>,
    reduction: Reduction,
) -> Result<NllLossPlan> {
    let x = graph.node(logp)?;
    let y = graph.node(target)?;
    if x.shape.rank() < 2 || y.shape.rank() + 1 != x.shape.rank() || !y.dtype.is_integer() {
        return Err(invalid("NLL requires label tensor one rank below data"));
    }
    let mut idx = y.shape.dims().to_vec();
    idx.insert(1, 1);
    let index_shape = Shape::new(idx);
    let main = source_gather_plan(&x.shape, x.dtype, &index_shape, y.dtype, 1)?;
    let flat_shape = Shape::new([y.shape.numel()?]);
    let weight_plan = if let Some(weight) = weight {
        let w = graph.node(weight)?;
        if w.shape.rank() != 1 {
            return Err(invalid("NLL weight must be rank one"));
        }
        Some(source_gather_plan(
            &w.shape,
            w.dtype,
            &flat_shape,
            y.dtype,
            0,
        )?)
    } else {
        None
    };
    let extent = |shape: &Shape, dtype: DType| {
        shape
            .numel()?
            .checked_mul(dtype.itemsize())
            .ok_or_else(|| Error::ShapeOverflow(shape.clone()))
            .map(|_| ())
    };
    extent(&x.shape, x.dtype)?;
    extent(&y.shape, y.dtype)?;
    extent(&index_shape, y.dtype)?;
    extent(&flat_shape, y.dtype)?;
    let selected_dtype = x.dtype;
    let factor_dtype = weight
        .map(|w| graph.dtype(w))
        .transpose()?
        .unwrap_or(y.dtype);
    extent(&y.shape, selected_dtype)?;
    extent(&y.shape, factor_dtype)?;
    let product_dtype = source_lub(selected_dtype, factor_dtype);
    extent(&y.shape, product_dtype)?;
    if ignore.is_some() {
        extent(&y.shape, DType::Bool)?;
    }
    let scalar = Shape::new([]);
    match reduction {
        Reduction::None => {}
        Reduction::Sum => {
            let d = ReductionDType::sum_default(product_dtype);
            extent(&y.shape, d.accumulator)?;
            extent(&scalar, d.output)?;
        }
        Reduction::Mean => {
            let d = ReductionDType::sum_default(product_dtype);
            extent(&y.shape, d.accumulator)?;
            extent(&scalar, d.output)?;
            extent(&scalar, factor_dtype)?;
        }
    }
    Ok(NllLossPlan {
        main,
        weight: weight_plan,
        index_shape,
        flat_shape,
        target_shape: y.shape.clone(),
    })
}

impl Graph {
    pub fn nll_loss(
        &mut self,
        logp: NodeId,
        target: NodeId,
        weight: Option<NodeId>,
        ignore_index: Option<i64>,
        reduction: Reduction,
    ) -> Result<NodeId> {
        let plan = nll_loss_plan(self, logp, target, weight, ignore_index, reduction)?;
        let index = self.reshape(target, plan.index_shape)?;
        let selected = lower_source_gather(self, logp, index, plan.main)?;
        let selected = self.squeeze(selected, Some(1))?;
        let gathered_weight = if let (Some(weight), Some(weight_plan)) = (weight, plan.weight) {
            let flat = self.reshape(target, plan.flat_shape)?;
            let gathered = lower_source_gather(self, weight, flat, weight_plan)?;
            self.reshape(gathered, plan.target_shape)?
        } else {
            let target_dtype = self.dtype(target)?;
            self.lazy_full_with_dtype(plan.target_shape.clone(), Scalar::I(1), target_dtype)?
        };
        let factor = if let Some(ignore) = ignore_index {
            let target_dtype = self.dtype(target)?;
            let scalar = self.constant(TensorData::scalar_with_dtype(
                Scalar::I(ignore),
                target_dtype,
            ));
            let mask = self.ne(target, scalar)?;
            self.mul(gathered_weight, mask)?
        } else {
            gathered_weight
        };
        let selected = self.neg(selected)?;
        let loss = self.mul(selected, factor)?;
        match reduction {
            Reduction::None => Ok(loss),
            Reduction::Sum => self.sum_default(loss),
            Reduction::Mean => {
                let n = self.sum_default(loss)?;
                let d = self.sum_default(factor)?;
                self.div(n, d)
            }
        }
    }
    pub fn nll_loss_default(&mut self, logp: NodeId, target: NodeId) -> Result<NodeId> {
        self.nll_loss(logp, target, None, None, Reduction::Mean)
    }
}

pub fn nll_loss(
    graph: &mut Graph,
    log_probabilities: NodeId,
    target: NodeId,
    weight: Option<NodeId>,
    options: LossOptions,
) -> Result<NodeId> {
    let a = nll_inputs(graph, log_probabilities, target, weight, options.class_axis)?;
    let hot = one_hot(graph, log_probabilities, target, a)?;
    let weighted = graph.mul(log_probabilities, hot)?;
    let summed = graph.reduce(weighted, ReduceKind::Sum, Some(vec![a as isize]), false)?;
    let selected = graph.neg(summed)?;
    let mask = if let Some(ignore) = options.ignore_index {
        let x = graph.constant(TensorData::scalar_with_dtype(
            Scalar::I(ignore),
            graph.dtype(target)?,
        ));
        Some(graph.ne(target, x)?)
    } else {
        None
    };
    if let Some(weight) = weight {
        let mut dims = vec![1; graph.shape(log_probabilities)?.rank()];
        dims[a] = graph.shape(weight)?.dims()[0];
        let w = graph.reshape(weight, Shape::new(dims))?;
        let weighted = graph.mul(hot, w)?;
        let factor = graph.reduce(weighted, ReduceKind::Sum, Some(vec![a as isize]), false)?;
        let factor = if let Some(mask) = mask {
            graph.mul(factor, mask)?
        } else {
            factor
        };
        let selected = graph.mul(selected, factor)?;
        return weighted_reduce(graph, selected, factor, options.reduction);
    }
    let selected = if let Some(mask) = mask {
        graph.mul(selected, mask)?
    } else {
        selected
    };
    masked_reduce(graph, selected, mask, options.reduction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, CpuBackend, Storage};
    use std::collections::HashMap;
    fn values(data: TensorData) -> Vec<f32> {
        match data.storage() {
            Storage::F32(v) => v.clone(),
            _ => panic!("expected f32"),
        }
    }
    #[test]
    fn source_gather_and_public_nll_are_structurally_atomic() {
        let mut graph = Graph::new();
        let logp = graph.input_dtype("logp", [4, 3, 5], crate::DType::F32);
        let labels = graph.input_dtype("labels", [2, 4], crate::DType::I32);
        let weight = graph.input_dtype("weight", [3], crate::DType::F32);
        let loss = graph
            .nll_loss(logp, labels, Some(weight), Some(-1), Reduction::None)
            .unwrap();
        assert_eq!(graph.shape(loss).unwrap(), &Shape::new([2, 4]));
        assert!(graph.trace(loss).unwrap().to_string().contains("select"));
        assert!(
            (0..graph.node_count())
                .any(|n| matches!(graph.op(NodeId(n)).unwrap(), crate::Op::Select { .. }))
        );
        assert!((0..graph.node_count()).any(|n| matches!(
            graph.op(NodeId(n)).unwrap(),
            crate::Op::Reduce {
                kind: ReduceKind::Sum,
                ..
            }
        )));
        assert!(
            graph
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    crate::Op::Constant(data) => Some(data.len()),
                    _ => None,
                })
                .all(|len| len == 1)
        );
        let reduced_loss = graph.sum_default(loss).unwrap();
        assert!(matches!(graph.grad(reduced_loss, logp), Ok(_)));
        let reduced_loss = graph.sum_default(loss).unwrap();
        assert!(matches!(graph.grad(reduced_loss, labels), Err(_)));

        for reduction in [Reduction::Sum, Reduction::Mean] {
            let mut reduced = Graph::new();
            let x = reduced.input("x", [2, 3]);
            let y = reduced.input_dtype("y", [2], crate::DType::I32);
            let output = reduced.nll_loss(x, y, None, None, reduction).unwrap();
            assert_eq!(reduced.shape(output).unwrap(), &Shape::new([]));
        }
        let mut malformed = Graph::new();
        let x = malformed.input("x", [2]);
        let y = malformed.input_dtype("y", [], crate::DType::I32);
        let before = malformed.node_count();
        assert!(malformed.nll_loss_default(x, y).is_err());
        assert_eq!(malformed.node_count(), before);
        let mut overflow = Graph::new();
        let x = overflow.input_dtype("x", [usize::MAX / 4, 3], crate::DType::F32);
        let y = overflow.input_dtype("y", [usize::MAX / 4], crate::DType::I32);
        let before = overflow.node_count();
        assert!(overflow.nll_loss_default(x, y).is_err());
        assert_eq!(overflow.node_count(), before);
    }
    #[test]
    fn bce_and_logits_are_stable_and_reduce() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2]);
        let target = graph.input("y", [2]);
        let loss = binary_cross_entropy(&mut graph, input, target, Reduction::Mean).unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                loss,
                &HashMap::from([
                    ("x".into(), TensorData::new([2], vec![0.25, 0.75]).unwrap()),
                    ("y".into(), TensorData::new([2], vec![0., 1.]).unwrap()),
                ]),
            )
            .unwrap();
        assert!((values(output)[0] - 0.2876821).abs() < 1e-5);
        let mut graph = Graph::new();
        let logits = graph.input("x", [1]);
        let target = graph.input("y", [1]);
        let loss =
            binary_cross_entropy_with_logits(&mut graph, logits, target, None, Reduction::Mean)
                .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                loss,
                &HashMap::from([
                    ("x".into(), TensorData::new([1], vec![-100.]).unwrap()),
                    ("y".into(), TensorData::new([1], vec![1.]).unwrap()),
                ]),
            )
            .unwrap();
        assert!(values(output)[0] > 90.);
    }
    #[test]
    fn binary_crossentropy_preserves_live_target_promotion_broadcast_and_gradients() {
        let mut graph = Graph::new();
        let input = graph.input("x", [2]);
        let target = graph.input("y", [2]);
        let loss = binary_cross_entropy(&mut graph, input, target, Reduction::Mean).unwrap();
        let gradient = graph.grad(loss, input).unwrap();
        let inputs = HashMap::from([
            ("x".into(), TensorData::new([2], vec![0.25, 0.75]).unwrap()),
            ("y".into(), TensorData::new([2], vec![0., 1.]).unwrap()),
        ]);
        let grad = values(CpuBackend.execute(&graph, gradient, &inputs).unwrap());
        assert!((grad[0] - (2. / 3.)).abs() < 1e-6);
        assert!((grad[1] + (2. / 3.)).abs() < 1e-6);

        let mut graph = Graph::new();
        let logits = graph.input("x", [2]);
        let target = graph.input("y", [2]);
        let loss =
            binary_cross_entropy_with_logits(&mut graph, logits, target, None, Reduction::Mean)
                .unwrap();
        let gradient = graph.grad(loss, logits).unwrap();
        let inputs = HashMap::from([
            ("x".into(), TensorData::new([2], vec![-1., 2.]).unwrap()),
            ("y".into(), TensorData::new([2], vec![0., 1.]).unwrap()),
        ]);
        let grad = values(CpuBackend.execute(&graph, gradient, &inputs).unwrap());
        assert!((grad[0] - 0.13447072).abs() < 1e-6);
        assert!((grad[1] + 0.05960146).abs() < 1e-6);

        let mut graph = Graph::new();
        let probability = graph.input("x", [2, 1]);
        let broadcast_target = graph.input("y", [2]);
        let broadcast = graph
            .binary_crossentropy(probability, broadcast_target, Reduction::None)
            .unwrap();
        assert_eq!(graph.shape(broadcast).unwrap(), &Shape::from([2, 2]));
        let logits = graph.input("logits", [2]);
        let integer_target = graph.input_dtype("labels", [2], crate::DType::I32);
        let integer_loss = graph
            .binary_crossentropy_logits(logits, integer_target, Reduction::Mean, None)
            .unwrap();
        assert_eq!(graph.shape(integer_loss).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(integer_loss).unwrap(), DType::F32);
    }

    #[test]
    fn binary_crossentropy_is_literal_and_preflights_the_whole_live_contract() {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 1], DType::F16);
        let target = graph.input_dtype("target", [2], DType::I64);
        let loss = graph
            .binary_crossentropy(input, target, Reduction::None)
            .unwrap();
        assert_eq!(graph.shape(loss).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.dtype(loss).unwrap(), DType::F16);
        // The public source root is `left - right`, represented by the
        // source-literal ADD after negating its right branch.
        assert!(matches!(
            graph.op(loss).unwrap(),
            crate::Op::Binary {
                op: crate::BinaryOp::Add,
                ..
            }
        ));
        let sum = graph
            .binary_crossentropy(input, target, Reduction::Sum)
            .unwrap();
        let mean = graph
            .binary_crossentropy(input, target, Reduction::Mean)
            .unwrap();
        let default = graph.binary_crossentropy_default(input, target).unwrap();
        assert_eq!(graph.shape(sum).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(sum).unwrap(), DType::F16);
        assert_eq!(graph.shape(mean).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(mean).unwrap(), DType::F16);
        assert_eq!(graph.shape(default).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(default).unwrap(), DType::F16);
        let gradient = graph.grad(mean, input).unwrap();
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([2, 1]));

        let mut empty = Graph::new();
        let input = empty.input_dtype("input", [0], DType::F32);
        let target = empty.input_dtype("target", [0], DType::F32);
        let mean = empty
            .binary_crossentropy(input, target, Reduction::Mean)
            .unwrap();
        assert_eq!(empty.shape(mean).unwrap(), &Shape::new([]));
        assert_eq!(empty.dtype(mean).unwrap(), DType::F32);

        // The broadcast result overflows only after both independently valid
        // inputs; the pure BCE plan rejects it before a negation, log, or
        // weak-one constant can publish.
        let mut overflow = Graph::new();
        let input = overflow.input_dtype("input", [usize::MAX / 4, 1], DType::F32);
        let target = overflow.input_dtype("target", [1, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.binary_crossentropy(input, target, Reduction::None),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn logits_bce_preflights_pos_weight_and_preserves_broadcast_vjps() {
        let mut graph = Graph::new();
        let logits = graph.input("logits", [2]);
        let target = graph.input("target", [2]);
        let pos_weight = graph.input("pos_weight", []);
        let loss = binary_cross_entropy_with_logits(
            &mut graph,
            logits,
            target,
            Some(pos_weight),
            Reduction::Mean,
        )
        .unwrap();
        let logits_gradient = graph.grad(loss, logits).unwrap();
        let target_gradient = graph.grad(loss, target).unwrap();
        let pos_weight_gradient = graph.grad(loss, pos_weight).unwrap();
        let inputs = HashMap::from([
            ("logits".into(), TensorData::new([2], vec![0., 1.]).unwrap()),
            ("target".into(), TensorData::new([2], vec![1., 0.]).unwrap()),
            ("pos_weight".into(), TensorData::scalar(2.)),
        ]);
        assert!(
            (values(CpuBackend.execute(&graph, loss, &inputs).unwrap())[0] - 1.349_778).abs()
                < 1e-5
        );
        assert_eq!(
            values(
                CpuBackend
                    .execute(&graph, logits_gradient, &inputs)
                    .unwrap()
            ),
            vec![-0.5, 0.365_529_3]
        );
        let target_gradient = values(
            CpuBackend
                .execute(&graph, target_gradient, &inputs)
                .unwrap(),
        );
        assert!((target_gradient[0] - 0.346_573_6).abs() < 1e-6);
        assert!((target_gradient[1] + 0.343_369_2).abs() < 1e-6);
        assert!(
            (values(
                CpuBackend
                    .execute(&graph, pos_weight_gradient, &inputs)
                    .unwrap()
            )[0] - 0.346_573_6)
                .abs()
                < 1e-6
        );

        let mut malformed = Graph::new();
        let logits = malformed.input("logits", [2]);
        let target = malformed.input("target", [2]);
        let node_count = malformed.node_count();
        assert!(matches!(
            binary_cross_entropy_with_logits(
                &mut malformed,
                logits,
                target,
                Some(crate::NodeId(usize::MAX)),
                Reduction::Mean,
            ),
            Err(Error::UnknownNode(_))
        ));
        assert_eq!(malformed.node_count(), node_count);

        let pos_weight = malformed.input("bad_pos_weight", [3]);
        let node_count = malformed.node_count();
        assert!(matches!(
            binary_cross_entropy_with_logits(
                &mut malformed,
                logits,
                target,
                Some(pos_weight),
                Reduction::Mean,
            ),
            Err(Error::BroadcastMismatch { .. })
        ));
        assert_eq!(malformed.node_count(), node_count);
    }

    #[test]
    fn binary_crossentropy_logits_is_literal_and_preflights_live_promotion() {
        let mut graph = Graph::new();
        let logits = graph.input_dtype("logits", [2, 1], DType::F16);
        let target = graph.input_dtype("target", [2], DType::I64);
        let weight = graph.input_dtype("weight", [1, 2], DType::U64);
        let loss = graph
            .binary_crossentropy_logits(logits, target, Reduction::None, Some(weight))
            .unwrap();
        assert_eq!(graph.shape(loss).unwrap(), &Shape::from([2, 2]));
        assert_eq!(graph.dtype(loss).unwrap(), DType::F32);
        assert!(matches!(
            graph.op(loss).unwrap(),
            crate::Op::Unary {
                op: crate::UnaryOp::Neg,
                ..
            }
        ));
        let default = graph
            .binary_crossentropy_logits_default(logits, target, None)
            .unwrap();
        assert_eq!(graph.shape(default).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(default).unwrap(), DType::F16);
        let gradient = graph.grad(loss, logits).unwrap();
        assert_eq!(graph.shape(gradient).unwrap(), &Shape::from([2, 1]));

        let mut empty = Graph::new();
        let logits = empty.input_dtype("logits", [0], DType::F32);
        let target = empty.input_dtype("target", [0], DType::Bool);
        let output = empty
            .binary_crossentropy_logits(logits, target, Reduction::Sum, None)
            .unwrap();
        assert_eq!(empty.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(empty.dtype(output).unwrap(), DType::F32);

        // Inputs independently fit, but the final live broadcast cannot.
        // Planning must fail before either LogSigmoid, weak one, or cast node.
        let mut overflow = Graph::new();
        let logits = overflow.input_dtype("logits", [usize::MAX / 4, 1], DType::F32);
        let target = overflow.input_dtype("target", [1, 2], DType::F32);
        let nodes = overflow.node_count();
        assert!(matches!(
            overflow.binary_crossentropy_logits(logits, target, Reduction::None, None),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), nodes);
    }

    #[test]
    fn categorical_supports_sparse_probability_smoothing_and_gradients() {
        let mut graph = Graph::new();
        let logits = graph.input("x", [2, 3]);
        let target = graph.input_dtype("y", [2], crate::DType::I32);
        let loss =
            sparse_categorical_cross_entropy(&mut graph, logits, target, LossOptions::default())
                .unwrap();
        let gradient = graph.grad(loss, logits).unwrap();
        let inputs = HashMap::from([
            (
                "x".into(),
                TensorData::new([2, 3], vec![0., 1., 0., 0., 0., 1.]).unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_scalars([2], crate::DType::I32, [Scalar::I(1), Scalar::I(2)])
                    .unwrap(),
            ),
        ]);
        let output = CpuBackend.execute(&graph, loss, &inputs).unwrap();
        assert!(values(output)[0] < 0.6);
        let grad = values(CpuBackend.execute(&graph, gradient, &inputs).unwrap());
        assert!(grad[1] < 0. && grad[5] < 0.);
        let mut graph = Graph::new();
        let logits = graph.input("x", [1, 2]);
        let target = graph.input("y", [1, 2]);
        let loss = cross_entropy(
            &mut graph,
            logits,
            target,
            LossOptions {
                reduction: Reduction::None,
                class_axis: 1,
                ignore_index: None,
                label_smoothing: 0.1,
            },
        )
        .unwrap();
        let output = CpuBackend
            .execute(
                &graph,
                loss,
                &HashMap::from([
                    ("x".into(), TensorData::new([1, 2], vec![0., 0.]).unwrap()),
                    ("y".into(), TensorData::new([1, 2], vec![1., 0.]).unwrap()),
                ]),
            )
            .unwrap();
        assert!((values(output)[0] - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn tinygrad_sparse_categorical_defaults_to_final_axis_and_literal_mask_path() {
        let mut graph = Graph::new();
        let logits = graph.input_dtype("x", [2, 3, 4], crate::DType::F16);
        let target = graph.input_dtype("y", [2, 3], crate::DType::I32);
        let loss = sparse_categorical_cross_entropy_tinygrad(
            &mut graph,
            logits,
            target,
            SparseCategoricalCrossEntropyOptions::default(),
        )
        .unwrap();
        assert_eq!(graph.shape(loss).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(loss).unwrap(), crate::DType::F16);
        let trace = graph.trace(loss).unwrap().to_string();
        assert!(trace.contains("log_softmax") || trace.contains("exp2"));
        assert!(trace.contains("reduce"));
        assert!(!trace.contains("ne"));
        let dx = graph.grad(loss, logits).unwrap();
        assert_eq!(graph.shape(dx).unwrap(), &Shape::from([2, 3, 4]));

        let masked = sparse_categorical_cross_entropy_tinygrad(
            &mut graph,
            logits,
            target,
            SparseCategoricalCrossEntropyOptions {
                ignore_index: 0,
                label_smoothing: 0.25,
                reduction: Reduction::None,
            },
        )
        .unwrap();
        assert_eq!(graph.shape(masked).unwrap(), &Shape::from([2, 3]));
        assert!(graph.trace(masked).unwrap().to_string().contains("ne"));

        // The public Graph spelling retains the source default options and
        // delegates to the same final-axis, lazy-one-hot plan.
        let default = graph
            .sparse_categorical_crossentropy_default(logits, target)
            .unwrap();
        assert_eq!(graph.shape(default).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(default).unwrap(), crate::DType::F16);
    }

    #[test]
    fn tinygrad_sparse_categorical_preflights_before_publication() {
        let mut graph = Graph::new();
        let scalar = graph.input("scalar", []);
        let target = graph.input_dtype("target", [], crate::DType::I32);
        let nodes = graph.node_count();
        assert!(
            sparse_categorical_cross_entropy_tinygrad(
                &mut graph,
                scalar,
                target,
                SparseCategoricalCrossEntropyOptions::default(),
            )
            .is_err()
        );
        assert_eq!(graph.node_count(), nodes);

        let mut overflow = Graph::new();
        let logits = overflow.input("x", [usize::MAX, 2]);
        let target = overflow.input_dtype("y", [usize::MAX], crate::DType::I32);
        let nodes = overflow.node_count();
        assert!(
            sparse_categorical_cross_entropy_tinygrad(
                &mut overflow,
                logits,
                target,
                SparseCategoricalCrossEntropyOptions::default(),
            )
            .is_err()
        );
        assert_eq!(overflow.node_count(), nodes);
    }
    #[test]
    fn probability_cross_entropy_requires_float_logits_and_targets() {
        let mut graph = Graph::new();
        let logits = graph.input("logits", [1, 2]);
        let target = graph.input("target", [1, 2]);
        let loss = cross_entropy(&mut graph, logits, target, LossOptions::default()).unwrap();
        let gradient = graph.grad(loss, logits).unwrap();
        let inputs = HashMap::from([
            (
                "logits".into(),
                TensorData::new([1, 2], vec![0., 0.]).unwrap(),
            ),
            (
                "target".into(),
                TensorData::new([1, 2], vec![1., 0.]).unwrap(),
            ),
        ]);
        let output = values(CpuBackend.execute(&graph, loss, &inputs).unwrap());
        assert!((output[0] - std::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(
            values(CpuBackend.execute(&graph, gradient, &inputs).unwrap()),
            vec![-0.5, 0.5]
        );

        let mut graph = Graph::new();
        let integer_logits = graph.input_dtype("logits", [1, 2], crate::DType::I32);
        let probability_target = graph.input("target", [1, 2]);
        assert!(
            cross_entropy(
                &mut graph,
                integer_logits,
                probability_target,
                LossOptions::default(),
            )
            .is_err()
        );
        let logits = graph.input("other_logits", [1, 2]);
        let boolean_target = graph.input_dtype("other_target", [1, 2], crate::DType::Bool);
        assert!(cross_entropy(&mut graph, logits, boolean_target, LossOptions::default()).is_err());
    }

    #[test]
    fn tinygrad_cross_entropy_is_shape_driven_and_preflighted() {
        let mut hard = Graph::new();
        let logits = hard.input_dtype("x", [2, 3], crate::DType::F16);
        let labels = hard.input_dtype("y", [2], crate::DType::I32);
        let loss = hard.cross_entropy_default(logits, labels).unwrap();
        assert_eq!(hard.shape(loss).unwrap(), &Shape::new([]));
        // Hard labels are internal one-hot Bool, then the source's Python float
        // smoothing affine lifts them before LogSoftmax multiplication.
        assert_eq!(hard.dtype(loss).unwrap(), crate::DType::F32);
        assert!(matches!(hard.grad(loss, logits), Ok(_)));

        let mut probabilities = Graph::new();
        let logits = probabilities.input_dtype("x", [2, 3], crate::DType::I16);
        let target = probabilities.input_dtype("y", [2, 3], crate::DType::Bool);
        let loss = probabilities
            .cross_entropy(logits, target, Reduction::None, 0.25)
            .unwrap();
        assert_eq!(probabilities.shape(loss).unwrap(), &Shape::new([2]));
        assert_eq!(probabilities.dtype(loss).unwrap(), crate::DType::F32);
        assert!(matches!(
            probabilities.op(loss).unwrap(),
            crate::Op::Unary {
                op: crate::UnaryOp::Neg,
                ..
            }
        ));

        let mut malformed = Graph::new();
        let logits = malformed.input("x", [2, 3]);
        let target = malformed.input_dtype("y", [3], crate::DType::I32);
        let before = malformed.node_count();
        assert!(malformed.cross_entropy_default(logits, target).is_err());
        assert_eq!(malformed.node_count(), before);

        let mut zero_classes = Graph::new();
        let logits = zero_classes.input("x", [2, 0]);
        let target = zero_classes.input_dtype("y", [2], crate::DType::I32);
        let before = zero_classes.node_count();
        assert!(matches!(
            zero_classes.cross_entropy_default(logits, target),
            Err(Error::DivisionByZero { .. })
        ));
        assert_eq!(zero_classes.node_count(), before);

        let mut overflow = Graph::new();
        let extent = usize::MAX / 4;
        let logits = overflow.input_dtype("x", [extent, 2], crate::DType::F32);
        let target = overflow.input_dtype("y", [extent], crate::DType::I32);
        let before = overflow.node_count();
        assert!(matches!(
            overflow.cross_entropy_default(logits, target),
            Err(Error::ShapeOverflow(_))
        ));
        assert_eq!(overflow.node_count(), before);
    }

    #[test]
    fn nll_preflights_inputs_and_weights_before_building_its_sparse_graph() {
        let mut graph = Graph::new();
        let log_probabilities = graph.input("log_probabilities", [1, 2]);
        let target = graph.input_dtype("target", [1], crate::DType::I32);
        let loss = nll_loss(
            &mut graph,
            log_probabilities,
            target,
            None,
            LossOptions::default(),
        )
        .unwrap();
        let gradient = graph.grad(loss, log_probabilities).unwrap();
        let inputs = HashMap::from([
            (
                "log_probabilities".into(),
                TensorData::new([1, 2], vec![-1., -0.25]).unwrap(),
            ),
            (
                "target".into(),
                TensorData::from_scalars([1], crate::DType::I32, [Scalar::I(1)]).unwrap(),
            ),
        ]);
        assert_eq!(
            values(CpuBackend.execute(&graph, loss, &inputs).unwrap()),
            vec![0.25]
        );
        assert_eq!(
            values(CpuBackend.execute(&graph, gradient, &inputs).unwrap()),
            vec![0., -1.]
        );

        let mut graph = Graph::new();
        let log_probabilities = graph.input("weighted_log_probabilities", [2, 2]);
        let target = graph.input_dtype("weighted_target", [2], crate::DType::I32);
        let weight = graph.input("weight", [2]);
        let loss = nll_loss(
            &mut graph,
            log_probabilities,
            target,
            Some(weight),
            LossOptions::default(),
        )
        .unwrap();
        let gradient = graph.grad(loss, log_probabilities).unwrap();
        let inputs = HashMap::from([
            (
                "weighted_log_probabilities".into(),
                TensorData::new([2, 2], vec![-0.5, -1., -2., -0.25]).unwrap(),
            ),
            (
                "weighted_target".into(),
                TensorData::from_scalars([2], crate::DType::I32, [Scalar::I(0), Scalar::I(1)])
                    .unwrap(),
            ),
            ("weight".into(), TensorData::new([2], vec![2., 1.]).unwrap()),
        ]);
        assert!(
            (values(CpuBackend.execute(&graph, loss, &inputs).unwrap())[0] - 5. / 12.).abs() < 1e-6
        );
        let gradient = values(CpuBackend.execute(&graph, gradient, &inputs).unwrap());
        assert!((gradient[0] + 2. / 3.).abs() < 1e-6);
        assert_eq!(gradient[1], 0.);
        assert_eq!(gradient[2], 0.);
        assert!((gradient[3] + 1. / 3.).abs() < 1e-6);

        let mut graph = Graph::new();
        let integer_log_probabilities =
            graph.input_dtype("log_probabilities", [1, 2], crate::DType::I32);
        let target = graph.input_dtype("target", [1], crate::DType::I32);
        let before = graph.node_count();
        assert!(
            nll_loss(
                &mut graph,
                integer_log_probabilities,
                target,
                None,
                LossOptions::default(),
            )
            .is_err()
        );
        assert_eq!(graph.node_count(), before);

        let mut graph = Graph::new();
        let log_probabilities = graph.input("log_probabilities", [1, 2]);
        let float_target = graph.input("target", [1]);
        let before = graph.node_count();
        assert!(
            nll_loss(
                &mut graph,
                log_probabilities,
                float_target,
                None,
                LossOptions::default(),
            )
            .is_err()
        );
        assert_eq!(graph.node_count(), before);

        let mut graph = Graph::new();
        let log_probabilities = graph.input("log_probabilities", [1, 2]);
        let target = graph.input_dtype("target", [1], crate::DType::I32);
        let wrong_weight = graph.input("weight", [3]);
        let before = graph.node_count();
        assert!(
            nll_loss(
                &mut graph,
                log_probabilities,
                target,
                Some(wrong_weight),
                LossOptions::default(),
            )
            .is_err()
        );
        assert_eq!(graph.node_count(), before);
    }
}
