//! Optimizer-specific graph lowering for compiled training programs.

use super::observation::{
    CompiledAdamWClipNodes, CompiledAdamWWindowLossNodes, CompiledTrainingObservationNode,
    adamw_observation_nodes,
};
use super::{AdamWGlobalState, AdamWParameterState, RecurrentStateKey, StateSpec};
use super::{
    CompiledAdamWConfig, CompiledLearningRatePolicy, CompiledMomentumSgdConfig,
    CompiledTrainingWindowTopology, lower_token_batch_count, materialize_compiled_output_alias,
    safe_token_count_divisor, state_dependent_zero, training,
};
use crate::{CompareOp, DType, Error, Graph, NodeId, Result, Scalar, Shape, TensorData};
use std::collections::BTreeMap;

pub(super) trait CompiledOptimizerProgram {
    fn name(&self) -> &'static str;
    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)>;
    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>>;
    fn gradients(
        &self,
        graph: &mut Graph,
        loss: NodeId,
        targets: &[NodeId],
    ) -> Result<Vec<NodeId>> {
        graph.gradient_default(loss, targets)
    }
    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering>;
}

pub(super) struct CompiledOptimizerLoweringContext<'a> {
    pub(super) loss: NodeId,
    pub(super) learning_rate: NodeId,
    pub(super) token_weight: Option<NodeId>,
    pub(super) inputs: &'a BTreeMap<String, NodeId>,
    pub(super) parameters: &'a BTreeMap<String, NodeId>,
    pub(super) gradients: &'a BTreeMap<String, NodeId>,
    pub(super) states: &'a BTreeMap<RecurrentStateKey, NodeId>,
}

pub(super) struct CompiledOptimizerLowering {
    pub(super) updates: BTreeMap<RecurrentStateKey, NodeId>,
    pub(super) sibling_updates: Option<BTreeMap<RecurrentStateKey, NodeId>>,
    pub(super) recurrent_store_groups: Vec<RecurrentStoreGroupSpec>,
    pub(super) observations: Vec<CompiledTrainingObservationNode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RecurrentStoreGroupSpec {
    pub(super) members: Vec<RecurrentStateKey>,
}

pub(super) fn adamw_recurrent_store_group_specs<'a>(
    parameters: impl Iterator<Item = &'a String>,
    topology: CompiledTrainingWindowTopology,
) -> Vec<RecurrentStoreGroupSpec> {
    if !topology.accumulating() {
        return Vec::new();
    }
    parameters
        .map(|name| RecurrentStoreGroupSpec {
            members: vec![
                RecurrentStateKey::parameter(name),
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment),
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment),
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator),
            ],
        })
        .collect()
}

pub(super) struct ClippedGradients {
    pub(super) gradients: BTreeMap<String, NodeId>,
    pub(super) report: Option<CompiledAdamWClipNodes>,
}

pub(super) struct MomentumProgram {
    pub(super) config: CompiledMomentumSgdConfig,
}

pub(super) struct AdamWProgram {
    pub(super) config: CompiledAdamWConfig,
}

impl CompiledOptimizerProgram for MomentumProgram {
    fn name(&self) -> &'static str {
        "momentum-SGD"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let mut specs = Vec::with_capacity(parameters.len() * 2);
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec::parameter(ordinal, name, value.clone()));
            specs.push(StateSpec::momentum(ordinal, name, value)?);
        }
        Ok(specs)
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering> {
        let CompiledOptimizerLoweringContext {
            learning_rate,
            parameters,
            gradients,
            states,
            ..
        } = context;
        let momentum = scalar_f32(graph, self.config.momentum)?;
        let mut updates = BTreeMap::new();
        for (name, parameter) in parameters {
            let momentum_key = RecurrentStateKey::momentum(name);
            let slot = states[&momentum_key];
            let retained = graph.mul(momentum, slot)?;
            let next_momentum = graph.add(retained, gradients[name])?;
            let scaled = graph.mul(learning_rate, next_momentum)?;
            let next_parameter = graph.sub(*parameter, scaled)?;
            validate_parameter_update(graph, *parameter, next_momentum)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(momentum_key, next_momentum);
            updates.insert(RecurrentStateKey::parameter(name), next_parameter);
        }
        Ok(CompiledOptimizerLowering {
            updates,
            sibling_updates: None,
            recurrent_store_groups: Vec::new(),
            observations: Vec::new(),
        })
    }
}

impl CompiledOptimizerProgram for AdamWProgram {
    fn name(&self) -> &'static str {
        "AdamW"
    }

    fn inputs(&self) -> &BTreeMap<String, (Shape, DType)> {
        &self.config.inputs
    }

    fn state_specs(&self, parameters: &BTreeMap<String, TensorData>) -> Result<Vec<StateSpec>> {
        let topology = CompiledTrainingWindowTopology::from_config(&self.config);
        let per_parameter = if topology.accumulating() { 4 } else { 3 };
        let mut specs = Vec::with_capacity(
            parameters.len() * per_parameter
                + 1
                + topology.accumulating() as usize
                + topology.retains_token_count() as usize
                + topology.retains_window_numerator() as usize,
        );
        for (ordinal, (name, value)) in parameters.iter().enumerate() {
            specs.push(StateSpec::parameter(ordinal, name, value.clone()));
            for state in [
                AdamWParameterState::FirstMoment,
                AdamWParameterState::SecondMoment,
            ] {
                specs.push(StateSpec::adamw_parameter(ordinal, name, value, state)?);
            }
            if topology.accumulating() {
                specs.push(StateSpec::adamw_parameter(
                    ordinal,
                    name,
                    value,
                    AdamWParameterState::GradientAccumulator,
                )?);
            }
        }
        specs.push(StateSpec::adamw_global(AdamWGlobalState::Step)?);
        if topology.accumulating() {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulationIndex,
            )?);
        }
        if topology.retains_token_count() {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulatedTokenCount,
            )?);
        }
        if topology.retains_window_numerator() {
            specs.push(StateSpec::adamw_global(
                AdamWGlobalState::AccumulatedLossNumerator,
            )?);
        }
        Ok(specs)
    }

    fn gradients(
        &self,
        graph: &mut Graph,
        loss: NodeId,
        targets: &[NodeId],
    ) -> Result<Vec<NodeId>> {
        if self.config.loss_scale == 1.0 {
            return graph.gradient_default(loss, targets);
        }
        let scale = scalar_f32(graph, self.config.loss_scale)?;
        let scaled_loss = graph.mul(loss, scale)?;
        graph
            .gradient_default(scaled_loss, targets)?
            .into_iter()
            .map(|gradient| graph.div(gradient, scale))
            .collect()
    }

    fn lower_updates(
        &self,
        graph: &mut Graph,
        context: CompiledOptimizerLoweringContext<'_>,
    ) -> Result<CompiledOptimizerLowering> {
        let CompiledOptimizerLoweringContext {
            loss,
            learning_rate,
            token_weight,
            inputs,
            parameters,
            gradients,
            states,
        } = context;
        let learning_rate = lower_adamw_learning_rate(&self.config, graph, learning_rate, states)?;
        let topology = CompiledTrainingWindowTopology::from_config(&self.config);
        if !topology.accumulating() {
            let token_count = self
                .config
                .window_loss_report
                .then_some(self.config.token_weight_policy.as_ref())
                .flatten()
                .map(|policy| {
                    lower_token_batch_count(
                        graph,
                        inputs,
                        token_weight,
                        policy,
                        &self.config.inputs,
                    )
                })
                .transpose()?;
            let clipped = clip_gradients_by_global_norm(&self.config, graph, gradients)?;
            let mut updates = lower_adamw_update_candidates(
                &self.config,
                graph,
                learning_rate,
                parameters,
                &clipped.gradients,
                states,
            )?;
            let window_loss_report = if self.config.window_loss_report {
                let numerator_key =
                    RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
                match token_count {
                    Some(token_count) => {
                        let zero = state_dependent_zero(graph, states[&numerator_key])?;
                        updates.insert(numerator_key, zero);
                        Some(CompiledAdamWWindowLossNodes {
                            mean_loss: materialize_compiled_output_alias(graph, loss)?,
                            loss_weight: token_count.exact,
                        })
                    }
                    None => {
                        let numerator = graph.add(states[&numerator_key], loss)?;
                        let zero = state_dependent_zero(graph, states[&numerator_key])?;
                        updates.insert(numerator_key, zero);
                        Some(CompiledAdamWWindowLossNodes {
                            mean_loss: numerator,
                            loss_weight: graph.full_with_dtype(
                                Shape::from([]),
                                Scalar::U(1),
                                DType::U64,
                            )?,
                        })
                    }
                }
            } else {
                None
            };
            let observations = adamw_observation_nodes(clipped.report, window_loss_report);
            return Ok(CompiledOptimizerLowering {
                updates,
                sibling_updates: None,
                recurrent_store_groups: Vec::new(),
                observations,
            });
        }

        let one_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)?;
        let zero_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?;
        let threshold = graph.full_with_dtype(
            Shape::from([]),
            Scalar::U(self.config.gradient_accumulation_steps),
            DType::U64,
        )?;
        let accumulation_index_key =
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex);
        let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
        let next_index = graph.add(states[&accumulation_index_key], one_u64)?;
        let commit = graph.compare(CompareOp::Eq, next_index, threshold)?;
        let reset_index = graph.select(commit, zero_u64, next_index)?;
        let weighted_count = self
            .config
            .token_weight_policy
            .as_ref()
            .map(|policy| {
                let batch_count = lower_token_batch_count(
                    graph,
                    inputs,
                    token_weight,
                    policy,
                    &self.config.inputs,
                )?;
                let count_key =
                    RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedTokenCount);
                let total_count = graph.add(states[&count_key], batch_count.exact)?;
                let divisor = graph.cast(total_count, DType::F32)?;
                Ok::<_, Error>((batch_count.float, count_key, total_count, divisor))
            })
            .transpose()?;
        let divisor = match &weighted_count {
            Some((_, _, _, divisor)) => *divisor,
            None => scalar_f32(graph, self.config.gradient_accumulation_steps as f32)?,
        };
        let divisor = safe_token_count_divisor(
            graph,
            divisor,
            self.config.allow_zero_valid_token_microbatches,
        )?;
        let window_loss = if self.config.window_loss_report {
            let numerator_key =
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
            let contribution = match &weighted_count {
                Some((batch_count, ..)) => graph.mul(loss, *batch_count)?,
                None => loss,
            };
            let numerator = graph.add(states[&numerator_key], contribution)?;
            let mean_loss = graph.div(numerator, divisor)?;
            let loss_weight = match &weighted_count {
                Some((_, _, total_count, _)) => *total_count,
                None => next_index,
            };
            Some((numerator_key, numerator, mean_loss, loss_weight))
        } else {
            None
        };

        let mut averaged_gradients = BTreeMap::new();
        let mut accumulated_gradients = BTreeMap::new();
        for (name, gradient) in gradients {
            let gradient = match &weighted_count {
                Some((batch_count, ..)) => graph.mul(*gradient, *batch_count)?,
                None => *gradient,
            };
            let key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulated = graph.add(states[&key], gradient)?;
            let averaged = graph.div(accumulated, divisor)?;
            accumulated_gradients.insert(name.clone(), accumulated);
            averaged_gradients.insert(name.clone(), averaged);
        }

        let clipped = clip_gradients_by_global_norm(&self.config, graph, &averaged_gradients)?;

        let candidates = lower_adamw_update_candidates(
            &self.config,
            graph,
            learning_rate,
            parameters,
            &clipped.gradients,
            states,
        )?;
        let mut accumulation_updates = states
            .iter()
            .map(|(key, value)| (key.clone(), *value))
            .collect::<BTreeMap<_, _>>();
        accumulation_updates.insert(accumulation_index_key.clone(), next_index);
        if let Some((_, count_key, total_count, _)) = &weighted_count {
            accumulation_updates.insert(count_key.clone(), *total_count);
        }
        if let Some((key, numerator, ..)) = &window_loss {
            accumulation_updates.insert(key.clone(), *numerator);
        }
        for name in parameters.keys() {
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            accumulation_updates.insert(accumulator_key, accumulated_gradients[name]);
        }

        let mut updates = BTreeMap::new();
        updates.insert(accumulation_index_key, reset_index);
        if let Some((_, count_key, total_count, _)) = weighted_count {
            updates.insert(count_key, graph.select(commit, zero_u64, total_count)?);
        }
        if let Some((key, numerator, ..)) = &window_loss {
            let zero = scalar_f32(graph, 0.0)?;
            updates.insert(key.clone(), graph.select(commit, zero, *numerator)?);
        }
        updates.insert(
            step_key.clone(),
            graph.select(commit, candidates[&step_key], states[&step_key])?,
        );
        for (name, parameter) in parameters {
            let first_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment);
            let second_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment);
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator_shape = graph.shape(accumulated_gradients[name])?.clone();
            let zero = graph.lazy_full_with_dtype(accumulator_shape, Scalar::I(0), DType::F32)?;
            let next_accumulator = graph.select(commit, zero, accumulated_gradients[name])?;
            let next_first = graph.select(commit, candidates[&first_key], states[&first_key])?;
            let next_second = graph.select(commit, candidates[&second_key], states[&second_key])?;
            let next_parameter = graph.select(
                commit,
                candidates[&RecurrentStateKey::parameter(name)],
                *parameter,
            )?;
            validate_parameter_update(graph, *parameter, next_accumulator)?;
            validate_parameter_update(graph, *parameter, next_first)?;
            validate_parameter_update(graph, *parameter, next_second)?;
            validate_parameter_update(graph, *parameter, next_parameter)?;
            updates.insert(accumulator_key, next_accumulator);
            updates.insert(first_key, next_first);
            updates.insert(second_key, next_second);
            updates.insert(RecurrentStateKey::parameter(name), next_parameter);
        }
        let observations = adamw_observation_nodes(
            clipped.report,
            window_loss.map(
                |(_, _, mean_loss, loss_weight)| CompiledAdamWWindowLossNodes {
                    mean_loss,
                    loss_weight,
                },
            ),
        );
        Ok(CompiledOptimizerLowering {
            updates,
            sibling_updates: Some(accumulation_updates),
            recurrent_store_groups: adamw_recurrent_store_group_specs(parameters.keys(), topology),
            observations,
        })
    }
}

pub(super) fn lower_adamw_learning_rate(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    external_learning_rate: NodeId,
    states: &BTreeMap<RecurrentStateKey, NodeId>,
) -> Result<NodeId> {
    let CompiledLearningRatePolicy::MultiStep(schedule) = &config.learning_rate else {
        return Ok(external_learning_rate);
    };
    let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
    let completed_step = states
        .get(&step_key)
        .copied()
        .ok_or_else(|| training("compiled AdamW optimizer step state is absent"))?;
    let mut learning_rate = scalar_f32(graph, schedule.base)?;
    if schedule.milestones.is_empty() {
        return Ok(learning_rate);
    }
    let gamma = scalar_f32(graph, schedule.gamma)?;
    for milestone in &schedule.milestones {
        let milestone =
            graph.full_with_dtype(Shape::from([]), Scalar::U(*milestone), DType::U64)?;
        let reached = graph.compare(CompareOp::Ge, completed_step, milestone)?;
        let decayed = graph.mul(learning_rate, gamma)?;
        learning_rate = graph.select(reached, decayed, learning_rate)?;
    }
    Ok(learning_rate)
}

pub(super) fn clip_gradients_by_global_norm(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    gradients: &BTreeMap<String, NodeId>,
) -> Result<ClippedGradients> {
    if config.max_gradient_norm.is_none() && !config.clip_report {
        return Ok(ClippedGradients {
            gradients: gradients.clone(),
            report: None,
        });
    }

    let gradients = materialize_gradient_vector(graph, gradients)?;

    let mut squared_norms = Vec::with_capacity(gradients.len());
    for gradient in gradients.values() {
        let squared = graph.mul(*gradient, *gradient)?;
        squared_norms.push(graph.sum_all(squared)?);
    }
    if squared_norms.len() == 1 {
        // Graph reductions intentionally elide extent-one axes. A trailing
        // positive zero keeps the single-parameter case on the same explicit
        // F32 reduction boundary without changing this nonnegative sum.
        squared_norms.push(scalar_f32(graph, 0.0)?);
    }
    // Stack the stable per-parameter F32 subtotals and reduce them through one
    // typed kernel. Its output storage is the exact commitment before `sqrt`:
    // native replay cannot widen a left-deep scalar addition chain, and no
    // scalar Contiguous fusion rehearsal can absorb the gradient graph.
    let squared_norms = graph.stack_default(squared_norms)?;
    let total = graph.sum_all(squared_norms)?;
    let norm = graph.sqrt(total)?;
    let scale = match config.max_gradient_norm {
        Some(max_norm) => {
            let max_norm = scalar_f32(graph, max_norm)?;
            // max(norm, limit) makes the scale exactly one below the limit,
            // while a NaN norm remains the ordered lhs and therefore
            // propagates instead of being silently treated as finite.
            let denominator = graph.maximum(norm, max_norm)?;
            graph.div(max_norm, denominator)?
        }
        None => scalar_f32(graph, 1.0)?,
    };
    let gradients = if config.max_gradient_norm.is_some() {
        gradients
            .iter()
            .map(|(name, gradient)| Ok((name.clone(), graph.mul(*gradient, scale)?)))
            .collect::<Result<_>>()?
    } else {
        gradients.clone()
    };
    Ok(ClippedGradients {
        gradients,
        report: config.clip_report.then_some(CompiledAdamWClipNodes {
            pre_clip_global_norm: norm,
            applied_scale: scale,
        }),
    })
}

fn materialize_gradient_vector(
    graph: &mut Graph,
    gradients: &BTreeMap<String, NodeId>,
) -> Result<BTreeMap<String, NodeId>> {
    if gradients.is_empty() {
        return Err(training("compiled AdamW has no gradients to materialize"));
    }
    let mut flattened = Vec::with_capacity(gradients.len().max(2));
    let mut ranges = Vec::with_capacity(gradients.len());
    let mut offset = 0usize;
    for (name, gradient) in gradients {
        let shape = graph.shape(*gradient)?.clone();
        if graph.dtype(*gradient)? != DType::F32 {
            return Err(training("compiled AdamW gradient must be F32"));
        }
        let elements = shape.numel()?;
        let end = offset
            .checked_add(elements)
            .ok_or_else(|| training("compiled AdamW gradient vector overflow"))?;
        flattened.push(graph.reshape(*gradient, [elements])?);
        ranges.push((name.clone(), shape, offset, end));
        offset = end;
    }
    if flattened.len() == 1 {
        // Concat requires two inputs. One empty F32 leaf forces the same real
        // materialization owner for a single parameter without adding a lane.
        flattened.push(graph.constant(TensorData::new([0], Vec::<f32>::new())?));
    }
    let vector = graph.concat(flattened, 0)?;
    ranges
        .into_iter()
        .map(|(name, shape, start, end)| {
            let slice = graph.shrink(vector, vec![(start, end)])?;
            let gradient = graph.reshape(slice, shape)?;
            Ok((name, gradient))
        })
        .collect()
}

pub(super) fn lower_adamw_update_candidates(
    config: &CompiledAdamWConfig,
    graph: &mut Graph,
    learning_rate: NodeId,
    parameters: &BTreeMap<String, NodeId>,
    gradients: &BTreeMap<String, NodeId>,
    states: &BTreeMap<RecurrentStateKey, NodeId>,
) -> Result<BTreeMap<RecurrentStateKey, NodeId>> {
    let one_u64 = graph.full_with_dtype(Shape::from([]), Scalar::U(1), DType::U64)?;
    let step_key = RecurrentStateKey::adamw_global(AdamWGlobalState::Step);
    let next_step = graph.add(states[&step_key], one_u64)?;
    let step_f32 = graph.cast(next_step, DType::F32)?;
    let one = scalar_f32(graph, 1.0)?;
    let beta1 = scalar_f32(graph, config.beta1)?;
    let beta2 = scalar_f32(graph, config.beta2)?;
    let one_minus_beta1 = scalar_f32(graph, 1.0 - config.beta1)?;
    let one_minus_beta2 = scalar_f32(graph, 1.0 - config.beta2)?;
    let eps = scalar_f32(graph, config.eps)?;
    let weight_decay = scalar_f32(graph, config.weight_decay)?;
    let beta1_power = graph.pow(beta1, step_f32)?;
    let beta2_power = graph.pow(beta2, step_f32)?;
    let first_correction = graph.sub(one, beta1_power)?;
    let second_correction = graph.sub(one, beta2_power)?;
    let decay = graph.mul(learning_rate, weight_decay)?;
    let decay_factor = graph.sub(one, decay)?;

    let mut updates = BTreeMap::from([(step_key, next_step)]);
    for (name, parameter) in parameters {
        let gradient = gradients[name];
        let first_key = RecurrentStateKey::adamw_parameter(name, AdamWParameterState::FirstMoment);
        let second_key =
            RecurrentStateKey::adamw_parameter(name, AdamWParameterState::SecondMoment);
        let retained_first = graph.mul(beta1, states[&first_key])?;
        let fresh_first = graph.mul(one_minus_beta1, gradient)?;
        let next_first = graph.add(retained_first, fresh_first)?;
        let retained_second = graph.mul(beta2, states[&second_key])?;
        let gradient_squared = graph.mul(gradient, gradient)?;
        let fresh_second = graph.mul(one_minus_beta2, gradient_squared)?;
        let next_second = graph.add(retained_second, fresh_second)?;
        let corrected_first = graph.div(next_first, first_correction)?;
        let corrected_second = graph.div(next_second, second_correction)?;
        let root = graph.sqrt(corrected_second)?;
        let denominator = graph.add(root, eps)?;
        let normalized = graph.div(corrected_first, denominator)?;
        let decayed = if config.weight_decay_exclusions.contains(name) {
            *parameter
        } else {
            graph.mul(*parameter, decay_factor)?
        };
        let scaled = graph.mul(learning_rate, normalized)?;
        let next_parameter = graph.sub(decayed, scaled)?;
        validate_parameter_update(graph, *parameter, next_first)?;
        validate_parameter_update(graph, *parameter, next_second)?;
        validate_parameter_update(graph, *parameter, next_parameter)?;
        updates.insert(first_key, next_first);
        updates.insert(second_key, next_second);
        updates.insert(RecurrentStateKey::parameter(name), next_parameter);
    }
    Ok(updates)
}

pub(super) fn scalar_f32(graph: &mut Graph, value: f32) -> Result<NodeId> {
    graph.full_with_dtype(Shape::from([]), Scalar::F(value as f64), DType::F32)
}

fn validate_parameter_update(graph: &Graph, parameter: NodeId, update: NodeId) -> Result<()> {
    if graph.shape(update)? != graph.shape(parameter)? || graph.dtype(update)? != DType::F32 {
        return Err(training("compiled optimizer update descriptor mismatch"));
    }
    Ok(())
}
