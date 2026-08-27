//! Checked-in tinygrad loss helpers composed from inspectable graph operations.
use crate::{Error, Graph, NodeId, ReduceKind, Result, Scalar, Shape, TensorData};

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
fn invalid(reason: &'static str) -> Error {
    Error::InvalidAttention { reason }
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
fn binary_target(graph: &Graph, input: NodeId, target: NodeId) -> Result<()> {
    if !graph.dtype(input)?.is_float() || !graph.dtype(target)?.is_float() {
        return Err(invalid("binary loss input and target must be float"));
    }
    if graph.shape(input)? != graph.shape(target)? {
        return Err(invalid("binary loss input and target shapes must match"));
    }
    Ok(())
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
/// Probability-target binary cross entropy, matching tinygrad's unclamped log contract.
pub fn binary_cross_entropy(
    graph: &mut Graph,
    input: NodeId,
    target: NodeId,
    reduction: Reduction,
) -> Result<NodeId> {
    binary_target(graph, input, target)?;
    let one = graph.constant(TensorData::scalar(1.));
    let negative_target = graph.neg(target)?;
    let log_input = graph.log(input)?;
    let left = graph.mul(negative_target, log_input)?;
    let complement_target = graph.sub(one, target)?;
    let negative_complement = graph.neg(complement_target)?;
    let complement_input = graph.sub(one, input)?;
    let log_complement = graph.log(complement_input)?;
    let right = graph.mul(negative_complement, log_complement)?;
    let loss = graph.add(left, right)?;
    reduce(graph, loss, reduction)
}
/// Stable binary cross entropy from logits, optionally applying `pos_weight`.
pub fn binary_cross_entropy_with_logits(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    pos_weight: Option<NodeId>,
    reduction: Reduction,
) -> Result<NodeId> {
    binary_target(graph, logits, target)?;
    let log_p = graph.logsigmoid(logits)?;
    let neg = graph.neg(logits)?;
    let log_q = graph.logsigmoid(neg)?;
    let pw = pos_weight.map_or(Ok(graph.constant(TensorData::scalar(1.))), Ok)?;
    let weighted_target = graph.mul(pw, target)?;
    let positive = graph.mul(weighted_target, log_p)?;
    let one = graph.constant(TensorData::scalar(1.));
    let complement = graph.sub(one, target)?;
    let negative = graph.mul(complement, log_q)?;
    let total = graph.add(positive, negative)?;
    let loss = graph.neg(total)?;
    reduce(graph, loss, reduction)
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
pub fn nll_loss(
    graph: &mut Graph,
    log_probabilities: NodeId,
    target: NodeId,
    weight: Option<NodeId>,
    options: LossOptions,
) -> Result<NodeId> {
    let a = nll_inputs(
        graph,
        log_probabilities,
        target,
        weight,
        options.class_axis,
    )?;
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
    fn binary_losses_require_paired_float_targets_and_preserve_gradients() {
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
        assert!(binary_cross_entropy(&mut graph, probability, broadcast_target, Reduction::Mean)
            .is_err());
        let logits = graph.input("logits", [2]);
        let integer_target = graph.input_dtype("labels", [2], crate::DType::I32);
        assert!(binary_cross_entropy_with_logits(
            &mut graph,
            logits,
            integer_target,
            None,
            Reduction::Mean,
        )
        .is_err());
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
    fn probability_cross_entropy_requires_float_logits_and_targets() {
        let mut graph = Graph::new();
        let logits = graph.input("logits", [1, 2]);
        let target = graph.input("target", [1, 2]);
        let loss = cross_entropy(&mut graph, logits, target, LossOptions::default()).unwrap();
        let gradient = graph.grad(loss, logits).unwrap();
        let inputs = HashMap::from([
            ("logits".into(), TensorData::new([1, 2], vec![0., 0.]).unwrap()),
            ("target".into(), TensorData::new([1, 2], vec![1., 0.]).unwrap()),
        ]);
        let output = values(CpuBackend.execute(&graph, loss, &inputs).unwrap());
        assert!((output[0] - std::f32::consts::LN_2).abs() < 1e-6);
        assert_eq!(values(CpuBackend.execute(&graph, gradient, &inputs).unwrap()), vec![-0.5, 0.5]);

        let mut graph = Graph::new();
        let integer_logits = graph.input_dtype("logits", [1, 2], crate::DType::I32);
        let probability_target = graph.input("target", [1, 2]);
        assert!(cross_entropy(
            &mut graph,
            integer_logits,
            probability_target,
            LossOptions::default(),
        )
        .is_err());
        let logits = graph.input("other_logits", [1, 2]);
        let boolean_target = graph.input_dtype("other_target", [1, 2], crate::DType::Bool);
        assert!(cross_entropy(&mut graph, logits, boolean_target, LossOptions::default()).is_err());
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
        assert_eq!(values(CpuBackend.execute(&graph, loss, &inputs).unwrap()), vec![0.25]);
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
        assert!((values(CpuBackend.execute(&graph, loss, &inputs).unwrap())[0] - 5. / 12.).abs() < 1e-6);
        let gradient = values(CpuBackend.execute(&graph, gradient, &inputs).unwrap());
        assert!((gradient[0] + 2. / 3.).abs() < 1e-6);
        assert_eq!(gradient[1], 0.);
        assert_eq!(gradient[2], 0.);
        assert!((gradient[3] + 1. / 3.).abs() < 1e-6);

        let mut graph = Graph::new();
        let integer_log_probabilities = graph.input_dtype("log_probabilities", [1, 2], crate::DType::I32);
        let target = graph.input_dtype("target", [1], crate::DType::I32);
        let before = graph.node_count();
        assert!(nll_loss(
            &mut graph,
            integer_log_probabilities,
            target,
            None,
            LossOptions::default(),
        )
        .is_err());
        assert_eq!(graph.node_count(), before);

        let mut graph = Graph::new();
        let log_probabilities = graph.input("log_probabilities", [1, 2]);
        let float_target = graph.input("target", [1]);
        let before = graph.node_count();
        assert!(nll_loss(
            &mut graph,
            log_probabilities,
            float_target,
            None,
            LossOptions::default(),
        )
        .is_err());
        assert_eq!(graph.node_count(), before);

        let mut graph = Graph::new();
        let log_probabilities = graph.input("log_probabilities", [1, 2]);
        let target = graph.input_dtype("target", [1], crate::DType::I32);
        let wrong_weight = graph.input("weight", [3]);
        let before = graph.node_count();
        assert!(nll_loss(
            &mut graph,
            log_probabilities,
            target,
            Some(wrong_weight),
            LossOptions::default(),
        )
        .is_err());
        assert_eq!(graph.node_count(), before);
    }
}
