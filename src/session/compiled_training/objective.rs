//! Objective and token-weight lowering for compiled AdamW programs.

use super::{
    CompiledAdamWConfig, CompiledAdamWIgnoreIndexContext, CompiledAdamWObjective,
    CompiledTokenWeightPolicy, MAX_EXACT_F32_INTEGER_COUNT, scalar_f32, training,
};
use crate::{CompareOp, DType, Graph, NodeId, Result, Scalar, Shape, TensorData};
use std::collections::BTreeMap;

pub(super) fn validate_token_weight_policy(
    inputs: &BTreeMap<String, (Shape, DType)>,
    policy: &CompiledTokenWeightPolicy,
    accumulation_steps: u64,
) -> Result<()> {
    if accumulation_steps == 0 {
        return Err(training(
            "compiled AdamW gradient accumulation steps must be positive",
        ));
    }
    let (shape, dtype) = policy.expected_descriptor(inputs)?;
    let token_elements = shape.numel()?;
    let valid_dtype = match policy {
        CompiledTokenWeightPolicy::ExplicitMask(_) => *dtype == DType::F32,
        CompiledTokenWeightPolicy::IgnoreIndex { .. } => *dtype == DType::I32,
    };
    if !valid_dtype || token_elements == 0 {
        let message = match policy {
            CompiledTokenWeightPolicy::ExplicitMask(_) => {
                "compiled AdamW token-weight mask must be nonempty fixed-shape F32"
            }
            CompiledTokenWeightPolicy::IgnoreIndex { .. } => {
                "compiled AdamW ignore-index target must be nonempty fixed-shape I32"
            }
        };
        return Err(training(message));
    }
    let token_elements = u64::try_from(token_elements)
        .map_err(|_| training("compiled AdamW token-weight element count overflows"))?;
    let maximum_count = token_elements
        .checked_mul(accumulation_steps)
        .ok_or_else(|| training("compiled AdamW token-weight count bound overflows"))?;
    if maximum_count > MAX_EXACT_F32_INTEGER_COUNT {
        return Err(training(
            "compiled AdamW token-weight count must remain exactly representable in F32",
        ));
    }
    Ok(())
}

pub(super) fn reject_token_weighted_scalar_loss(config: &CompiledAdamWConfig) -> Result<()> {
    if config.token_weight_policy.is_some() {
        return Err(training(
            "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
        ));
    }
    Ok(())
}

pub(super) fn lower_compiled_adamw_objective(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    objective: CompiledAdamWObjective,
) -> Result<NodeId> {
    lower_compiled_adamw_objective_for_policy(
        graph,
        inputs,
        objective,
        config.token_weight_policy.as_ref(),
        &config.inputs,
        config.allow_zero_valid_token_microbatches,
    )
}

pub(super) fn lower_compiled_adamw_objective_with_ignore_index_nodes(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    objective: CompiledAdamWObjective,
    nodes: CompiledAdamWIgnoreIndexContext,
) -> Result<NodeId> {
    lower_compiled_adamw_objective_for_ignore_index_policy(
        graph,
        objective,
        nodes,
        config.token_weight_policy.as_ref(),
        &config.inputs,
        config.allow_zero_valid_token_microbatches,
    )
}

pub(super) fn lower_compiled_adamw_objective_for_ignore_index_policy(
    graph: &mut Graph,
    objective: CompiledAdamWObjective,
    nodes: CompiledAdamWIgnoreIndexContext,
    policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    let policy = require_ignore_index_policy(policy)?;
    let (token_shape, _) = policy.expected_descriptor(input_descriptors)?;
    match objective {
        CompiledAdamWObjective::Scalar(_) => Err(training(
            "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
        )),
        CompiledAdamWObjective::TokenMean(losses) => lower_token_mean_loss(
            graph,
            losses,
            nodes.weight,
            token_shape,
            allow_zero_valid_token_microbatches,
        ),
    }
}

pub(super) fn lower_compiled_adamw_objective_for_policy(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    objective: CompiledAdamWObjective,
    token_weight_policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    match objective {
        CompiledAdamWObjective::Scalar(loss) => {
            if token_weight_policy.is_some() {
                return Err(training(
                    "compiled AdamW token-weighted accumulation requires the token-mean-loss compile surface",
                ));
            }
            Ok(loss)
        }
        CompiledAdamWObjective::TokenMean(losses) => {
            let policy = token_weight_policy.ok_or_else(|| {
                training("compiled AdamW token-mean-loss compilation requires token weighting")
            })?;
            let (token_shape, _) = policy.expected_descriptor(input_descriptors)?;
            let mask = lower_token_weight_mask(graph, inputs, policy)?;
            lower_token_mean_loss(
                graph,
                losses,
                mask,
                token_shape,
                allow_zero_valid_token_microbatches,
            )
        }
    }
}

pub(super) fn token_mean_loss_descriptor(config: &CompiledAdamWConfig) -> Result<(String, Shape)> {
    let mask_input = match config.token_weight_policy.as_ref().ok_or_else(|| {
        training("compiled AdamW token-mean-loss compilation requires token weighting")
    })? {
        CompiledTokenWeightPolicy::ExplicitMask(name) => name,
        CompiledTokenWeightPolicy::IgnoreIndex { .. } => {
            return Err(training(
                "compiled AdamW ignore-index weighting requires the unified token-mean compile surface",
            ));
        }
    };
    let (shape, dtype) = config
        .inputs
        .get(mask_input)
        .ok_or_else(|| training("compiled AdamW token-weight mask must name an existing input"))?;
    if *dtype != DType::F32 {
        return Err(training(
            "compiled AdamW token-weight mask must be nonempty fixed-shape F32",
        ));
    }
    Ok((mask_input.clone(), shape.clone()))
}

fn lower_token_weight_mask(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    policy: &CompiledTokenWeightPolicy,
) -> Result<NodeId> {
    match policy {
        CompiledTokenWeightPolicy::ExplicitMask(mask_input) => inputs
            .get(mask_input)
            .copied()
            .ok_or_else(|| training("compiled AdamW token-weight mask input is absent")),
        CompiledTokenWeightPolicy::IgnoreIndex {
            target_input,
            value,
        } => {
            let targets = inputs
                .get(target_input)
                .copied()
                .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
            let ignored =
                graph.full_with_dtype(Shape::from([]), Scalar::I(i64::from(*value)), DType::I32)?;
            let keep = graph.compare(CompareOp::Ne, targets, ignored)?;
            graph.cast(keep, DType::F32)
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct CompiledTokenBatchCount {
    pub(super) float: NodeId,
    pub(super) exact: NodeId,
}

pub(super) fn lower_token_batch_count(
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    token_weight: Option<NodeId>,
    policy: &CompiledTokenWeightPolicy,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
) -> Result<CompiledTokenBatchCount> {
    let weight = match token_weight {
        Some(weight) => weight,
        None => lower_token_weight_mask(graph, inputs, policy)?,
    };
    validate_token_weight_node(graph, weight, policy, input_descriptors)?;
    let float = graph.sum_all(weight)?;
    let exact = graph.cast(float, DType::U64)?;
    Ok(CompiledTokenBatchCount { float, exact })
}

pub(super) fn lower_ignore_index_nodes(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<CompiledAdamWIgnoreIndexContext> {
    lower_ignore_index_nodes_for_policy(
        config.token_weight_policy.as_ref(),
        &config.inputs,
        graph,
        inputs,
    )
}

pub(super) fn lower_ignore_index_nodes_for_policy(
    policy: Option<&CompiledTokenWeightPolicy>,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
) -> Result<CompiledAdamWIgnoreIndexContext> {
    let (policy, target_input, value) = match policy {
        Some(
            policy @ CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            },
        ) => (policy, target_input, value),
        _ => {
            return Err(training(
                "compiled AdamW ignore-index graph context requires ignore-index weighting",
            ));
        }
    };
    let targets = inputs
        .get(target_input)
        .copied()
        .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
    let ignored =
        graph.full_with_dtype(Shape::from([]), Scalar::I(i64::from(*value)), DType::I32)?;
    let validity = graph.compare(CompareOp::Ne, targets, ignored)?;
    let weight = graph.cast(validity, DType::F32)?;
    let nodes = CompiledAdamWIgnoreIndexContext {
        targets,
        validity,
        weight,
    };
    validate_ignore_index_nodes(policy, input_descriptors, graph, nodes)?;
    Ok(nodes)
}

fn require_ignore_index_policy(
    policy: Option<&CompiledTokenWeightPolicy>,
) -> Result<&CompiledTokenWeightPolicy> {
    match policy {
        Some(policy @ CompiledTokenWeightPolicy::IgnoreIndex { .. }) => Ok(policy),
        _ => Err(training(
            "compiled AdamW ignore-index graph context requires ignore-index weighting",
        )),
    }
}

fn validate_ignore_index_nodes(
    policy: &CompiledTokenWeightPolicy,
    input_descriptors: &BTreeMap<String, (Shape, DType)>,
    graph: &Graph,
    nodes: CompiledAdamWIgnoreIndexContext,
) -> Result<()> {
    let (shape, dtype) = policy.expected_descriptor(input_descriptors)?;
    if *dtype != DType::I32
        || graph.shape(nodes.targets)? != shape
        || graph.dtype(nodes.targets)? != DType::I32
        || graph.shape(nodes.validity)? != shape
        || graph.dtype(nodes.validity)? != DType::Bool
        || graph.shape(nodes.weight)? != shape
        || graph.dtype(nodes.weight)? != DType::F32
    {
        return Err(training(
            "compiled AdamW ignore-index graph context descriptor differs",
        ));
    }
    Ok(())
}

fn validate_token_weight_node(
    graph: &Graph,
    node: NodeId,
    policy: &CompiledTokenWeightPolicy,
    inputs: &BTreeMap<String, (Shape, DType)>,
) -> Result<()> {
    let (shape, _) = policy.expected_descriptor(inputs)?;
    if graph.shape(node)? != shape || graph.dtype(node)? != DType::F32 {
        return Err(training(
            "compiled AdamW token-weight graph context descriptor differs",
        ));
    }
    Ok(())
}

pub(super) fn lower_token_mean_loss(
    graph: &mut Graph,
    losses: NodeId,
    mask: NodeId,
    expected_shape: &Shape,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    if graph.dtype(losses)? != DType::F32 || graph.shape(losses)? != expected_shape {
        return Err(training(
            "compiled AdamW per-token losses must exactly match the token-weight mask descriptor",
        ));
    }
    if graph.dtype(mask)? != DType::F32 || graph.shape(mask)? != expected_shape {
        return Err(training(
            "compiled AdamW token-weight mask descriptor changed during compilation",
        ));
    }
    let weighted = if allow_zero_valid_token_microbatches {
        let zero = scalar_f32(graph, 0.0)?;
        let keep = graph.compare(CompareOp::Gt, mask, zero)?;
        graph.select(keep, losses, zero)?
    } else {
        graph.mul(losses, mask)?
    };
    let numerator = graph.sum_all(weighted)?;
    let denominator = graph.sum_all(mask)?;
    let denominator =
        safe_token_count_divisor(graph, denominator, allow_zero_valid_token_microbatches)?;
    graph.div(numerator, denominator)
}

pub(super) fn safe_token_count_divisor(
    graph: &mut Graph,
    count: NodeId,
    allow_zero_valid_token_microbatches: bool,
) -> Result<NodeId> {
    if !allow_zero_valid_token_microbatches {
        return Ok(count);
    }
    let zero = scalar_f32(graph, 0.0)?;
    let one = scalar_f32(graph, 1.0)?;
    let positive = graph.compare(CompareOp::Gt, count, zero)?;
    graph.select(positive, count, one)
}

pub(super) fn validate_token_weight(
    inputs: &BTreeMap<String, TensorData>,
    policy: Option<&CompiledTokenWeightPolicy>,
    allow_zero_valid_token_microbatches: bool,
) -> Result<u64> {
    let Some(policy) = policy else {
        return Ok(1);
    };
    let mut valid_tokens = 0_u64;
    match policy {
        CompiledTokenWeightPolicy::ExplicitMask(mask_input) => {
            let mask = inputs
                .get(mask_input)
                .ok_or_else(|| training("compiled AdamW token-weight mask input is absent"))?;
            for index in 0..mask.shape().numel()? {
                let value = mask.scalar_at(index).as_f64();
                if !value.is_finite() || (value != 0.0 && value != 1.0) {
                    return Err(training(
                        "compiled AdamW token-weight mask must contain finite binary values",
                    ));
                }
                valid_tokens = valid_tokens
                    .checked_add(u64::from(value == 1.0))
                    .ok_or_else(|| training("compiled AdamW token-weight count overflows"))?;
            }
        }
        CompiledTokenWeightPolicy::IgnoreIndex {
            target_input,
            value,
        } => {
            let targets = inputs
                .get(target_input)
                .ok_or_else(|| training("compiled AdamW ignore-index target input is absent"))?;
            for index in 0..targets.shape().numel()? {
                valid_tokens = valid_tokens
                    .checked_add(u64::from(
                        targets.scalar_at(index).as_i64() != i64::from(*value),
                    ))
                    .ok_or_else(|| training("compiled AdamW token-weight count overflows"))?;
            }
        }
    }
    if valid_tokens == 0 && !allow_zero_valid_token_microbatches {
        return Err(training(
            "compiled AdamW token-weight mask must contain at least one valid token",
        ));
    }
    Ok(valid_tokens)
}

pub(super) fn validate_retained_token_count(
    inputs: &BTreeMap<String, (Shape, DType)>,
    policy: &CompiledTokenWeightPolicy,
    accumulation_index: u64,
    count: u64,
    allow_zero_valid_token_microbatches: bool,
) -> Result<()> {
    let (shape, _) = policy.expected_descriptor(inputs)?;
    let token_elements = u64::try_from(shape.numel()?)
        .map_err(|_| training("compiled AdamW token-weight element count overflows"))?;
    let maximum_count = token_elements
        .checked_mul(accumulation_index)
        .ok_or_else(|| training("compiled AdamW retained token count bound overflows"))?;
    if (!allow_zero_valid_token_microbatches && count < accumulation_index) || count > maximum_count
    {
        return Err(training(
            "compiled AdamW retained token count is inconsistent with progress",
        ));
    }
    Ok(())
}
