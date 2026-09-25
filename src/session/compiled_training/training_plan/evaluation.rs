//! Evaluation graph construction and replay for compiled training plans.

use super::*;

#[derive(Clone)]
pub(in crate::session::compiled_training) struct CompiledEvaluationPlan {
    pub(in crate::session::compiled_training) inference: CompiledEvaluationCapture,
    pub(in crate::session::compiled_training) inputs: BTreeMap<String, (Shape, DType)>,
    pub(in crate::session::compiled_training) output_names: Vec<String>,
    pub(in crate::session::compiled_training) parameter_inputs: BTreeMap<String, String>,
    pub(in crate::session::compiled_training) loss_weight_policy: Option<CompiledTokenWeightPolicy>,
    pub(in crate::session::compiled_training) allow_zero_valid_token_microbatches: bool,
    pub(in crate::session::compiled_training) capture_identity: u64,
}

#[derive(Clone)]
pub(in crate::session::compiled_training) struct CpuCompiledEvaluation {
    pub(in crate::session::compiled_training) plan: CompiledEvaluationPlan,
}

impl CompiledEvaluationPlan {
    pub(in crate::session::compiled_training) fn compile_with_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let (loss, outputs) = build(module, graph, inputs)?;
                Ok((loss, outputs, None))
            },
        )
    }

    pub(in crate::session::compiled_training) fn compile_graph_with_parameter_plan<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(&M, &mut Graph, &BTreeMap<String, NodeId>) -> Result<CompiledAdamWGraph>,
    {
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let (objective, outputs) = build(module, graph, inputs)?.into_parts();
                let loss_weight_policy = match objective {
                    CompiledAdamWObjective::Scalar(_) => None,
                    CompiledAdamWObjective::TokenMean(_) => {
                        training_plan.contract.token_weight_policy.clone()
                    }
                };
                let loss = lower_compiled_adamw_objective_for_policy(
                    graph,
                    inputs,
                    objective,
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    training_plan.contract.allow_zero_valid_token_microbatches,
                )?;
                Ok((loss, outputs, loss_weight_policy))
            },
        )
    }

    pub(in crate::session::compiled_training) fn compile_graph_with_ignore_index_parameter_plan<
        M,
        F,
    >(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
            CompiledAdamWIgnoreIndexContext,
        ) -> Result<CompiledAdamWGraph>,
    {
        Self::compile_with_parameter_plan_inner(
            module,
            training_plan,
            parameter_plan,
            |module, graph, inputs| {
                let nodes = lower_ignore_index_nodes_for_policy(
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    graph,
                    inputs,
                )?;
                let (objective, outputs) = build(module, graph, inputs, nodes)?.into_parts();
                let loss = lower_compiled_adamw_objective_for_ignore_index_policy(
                    graph,
                    objective,
                    nodes,
                    training_plan.contract.token_weight_policy.as_ref(),
                    &training_plan.inner.inputs,
                    training_plan.contract.allow_zero_valid_token_microbatches,
                )?;
                Ok((
                    loss,
                    outputs,
                    training_plan.contract.token_weight_policy.clone(),
                ))
            },
        )
    }

    pub(in crate::session::compiled_training) fn compile_with_parameter_plan_inner<M, F>(
        module: &M,
        training_plan: &CompiledAdamWPlan,
        parameter_plan: ModuleParameterPlan,
        build: F,
    ) -> Result<Self>
    where
        M: Module + ?Sized,
        F: FnOnce(
            &M,
            &mut Graph,
            &BTreeMap<String, NodeId>,
        ) -> Result<(
            NodeId,
            BTreeMap<String, NodeId>,
            Option<CompiledTokenWeightPolicy>,
        )>,
    {
        let mut graph = Graph::new();
        let inputs = training_plan
            .inner
            .inputs
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut parameters = BTreeMap::new();
        let mut parameter_inputs = BTreeMap::new();
        let mut residents = BTreeMap::new();
        for init in parameter_plan.initial_parameters()? {
            let key = RecurrentStateKey::parameter(init.name());
            let input_name = training_plan
                .inner
                .state_input_keys
                .iter()
                .find_map(|(input, candidate)| (candidate == &key).then(|| input.clone()))
                .ok_or_else(|| training("compiled evaluation parameter state is absent"))?;
            let value = training_plan
                .inner
                .state_values
                .get(&key)
                .cloned()
                .ok_or_else(|| training("compiled evaluation parameter value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                true,
            );
            parameters.insert(init.name().to_owned(), node);
            parameter_inputs.insert(init.name().to_owned(), input_name.clone());
            residents.insert(input_name, (node, value));
        }
        let mut loss_weight_policy = None;
        let (loss, outputs) = parameter_plan.lower(&mut graph, &parameters, |graph| {
            let (loss, outputs, mask_input) = build(module, graph, &inputs)?;
            loss_weight_policy = mask_input;
            Ok((loss, outputs))
        })?;
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            training_plan
                .inner
                .inputs
                .keys()
                .chain(parameter_inputs.keys()),
        )?;
        let requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .collect::<Vec<_>>();
        let requested = materialize_compiled_public_aliases(&mut graph, &requested)?;
        let inference =
            crate::CapturedInference::from_graph_residents(&graph, &requested, residents, &[])
                .map_err(captured_inference_error)?
                .with_authenticated_fixed_host_gathers(&training_plan.contract.host_token_inputs)
                .map_err(captured_inference_error)?;
        let transient_names = inference
            .transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        if transient_names
            != training_plan
                .inner
                .inputs
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        {
            return Err(training("compiled evaluation input inventory differs"));
        }
        let capture_identity = inference.capture().identity;
        Ok(Self {
            inference: CompiledEvaluationCapture::from_inference(inference),
            inputs: training_plan.inner.inputs.clone(),
            output_names: outputs.keys().cloned().collect(),
            parameter_inputs,
            loss_weight_policy,
            allow_zero_valid_token_microbatches: training_plan
                .contract
                .allow_zero_valid_token_microbatches,
            capture_identity,
        })
    }

    pub(in crate::session::compiled_training) fn bind(
        &self,
        mut inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<BTreeMap<String, TensorData>> {
        validate_evaluation_inputs(&self.inputs, &inputs)?;
        if parameters.keys().ne(self.parameter_inputs.keys()) {
            return Err(training("compiled evaluation parameter inventory differs"));
        }
        for (name, value) in parameters {
            let input = self
                .parameter_inputs
                .get(&name)
                .ok_or_else(|| training("compiled evaluation parameter is absent"))?;
            inputs.insert(input.clone(), value);
        }
        Ok(inputs)
    }

    pub(in crate::session::compiled_training) fn validate_loss_weight(
        &self,
        inputs: &BTreeMap<String, TensorData>,
    ) -> Result<u64> {
        validate_evaluation_inputs(&self.inputs, inputs)?;
        validate_token_weight(
            inputs,
            self.loss_weight_policy.as_ref(),
            self.allow_zero_valid_token_microbatches,
        )
    }

    pub(in crate::session::compiled_training) fn evaluate(
        &self,
        inputs: BTreeMap<String, TensorData>,
        parameters: BTreeMap<String, TensorData>,
    ) -> Result<CompiledEvaluationResult> {
        let loss_weight = self.validate_loss_weight(&inputs)?;
        let inputs = self.bind(inputs, parameters)?;
        let values = self
            .inference
            .capture()
            .replay(&inputs)
            .map_err(replay_error)?;
        evaluation_result(
            values,
            &self.output_names,
            loss_weight,
            self.capture_identity,
        )
    }

    pub(in crate::session::compiled_training) fn preflight_native(
        &self,
        parameters: BTreeMap<String, TensorData>,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<NativeCpuEvaluationPreparation> {
        let started = Instant::now();
        if parameter_buffers.keys().ne(self.parameter_inputs.keys()) {
            return Err(training(
                "compiled native CPU evaluation parameter buffer inventory differs",
            ));
        }
        let capture = self.inference.capture();
        let parameter_inputs = self
            .parameter_inputs
            .iter()
            .map(|(parameter, input_name)| {
                let value = parameters.get(parameter).ok_or_else(|| {
                    training("compiled native CPU evaluation parameter value is absent")
                })?;
                let input = capture
                    .inputs
                    .iter()
                    .find(|input| input.name == *input_name)
                    .ok_or_else(|| {
                        training("compiled native CPU evaluation parameter input is absent")
                    })?;
                let bytes = value
                    .len()
                    .checked_mul(value.dtype().itemsize())
                    .ok_or_else(|| {
                        training("compiled native CPU evaluation parameter bytes overflow")
                    })?;
                if value.shape() != &input.desc.shape
                    || value.dtype() != input.desc.dtype
                    || bytes != input.desc.bytes
                {
                    return Err(training(
                        "compiled native CPU evaluation parameter descriptor mismatch",
                    ));
                }
                Ok(PreparedNativeEvaluationParameterInput {
                    parameter: parameter.clone(),
                    input: input_name.clone(),
                    buffer: parameter_buffers[parameter],
                    shape: value.shape().clone(),
                    dtype: value.dtype(),
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let inputs = self.bind(zero_inputs(&self.inputs)?, parameters)?;
        Ok(NativeCpuEvaluationPreparation {
            inputs: Some(inputs),
            parameter_inputs,
            residual_wall_time: started.elapsed(),
        })
    }

    pub(in crate::session::compiled_training) fn finish_native(
        &self,
        preparation: NativeCpuEvaluationPreparation,
        parameter_buffers: &BTreeMap<String, u64>,
        plan: PlannedNativeItems,
    ) -> Result<PreparedNativeCpuEvaluation> {
        let capture = self.inference.capture();
        let module_preparation = plan.module_preparation();
        let work = NativeCpuPreparationWork::from_module(module_preparation);
        let execution_plan = ExecutionPlanSummary::from_capture(capture, true)
            .map_err(|error| training(format!("compiled native CPU summary: {error}")))?;
        let wall_time =
            native_preparation_wall_time(module_preparation, preparation.residual_wall_time)?;
        let report = NativeCpuProgramPreparationReport {
            capture_identity: self.capture_identity,
            native_identity: native_cpu_identity(
                self.capture_identity,
                plan.vectorized(),
                capture.items.iter().map(|item| item.cache_key),
            ),
            vectorized: plan.vectorized(),
            native_item_count: plan.item_count(),
            cache_hit_count: plan.cache_hit_count(),
            cache_miss_count: plan.cache_miss_count(),
            work,
            phases: NativeCpuPreparationPhases::from_module(module_preparation, wall_time)?,
            dispatch_segmentation: NativeCpuDispatchSegmentation::from_native(
                plan.dispatch_segmentation(),
            )?,
            execution_plan,
            wall_time,
        };
        let prepared = PreparedNativeCpuEvaluation {
            report,
            plan,
            parameter_inputs: preparation.parameter_inputs,
        };
        prepared.validate(self.capture_identity, capture, parameter_buffers)?;
        Ok(prepared)
    }

    pub(in crate::session::compiled_training) fn preflight_native_borrowed(
        &self,
        inputs: &BTreeMap<String, TensorData>,
        frontier: &[BufferState],
        prepared: &PreparedNativeCpuEvaluation,
        parameter_buffers: &BTreeMap<String, u64>,
    ) -> Result<(u64, Vec<BufferState>)> {
        let loss_weight = self.validate_loss_weight(inputs)?;
        let capture = self.inference.capture();
        prepared.validate(self.capture_identity, capture, parameter_buffers)?;
        let parameter_inputs = prepared
            .parameter_inputs
            .iter()
            .map(|binding| binding.input.as_str())
            .collect::<BTreeSet<_>>();
        for input in &capture.inputs {
            if parameter_inputs.contains(input.name.as_str()) {
                continue;
            }
            let value = inputs
                .get(&input.name)
                .ok_or_else(|| training("compiled evaluation input is absent"))?;
            crate::engine::validate_input_value(capture, input, value).map_err(replay_error)?;
        }
        Ok((loss_weight, prepared.active_parameter_states(frontier)?))
    }

    pub(in crate::session::compiled_training) fn evaluate_native_borrowed(
        &self,
        inputs: &BTreeMap<String, TensorData>,
        reads: &[crate::host_buffer::HostBufferRead<'_>],
        loss_weight: u64,
        executor: &CapturedReplayExecutor,
        prepared: &mut PreparedNativeCpuEvaluation,
    ) -> Result<(CompiledEvaluationResult, NativeCpuRunReport)> {
        if reads.len() != prepared.parameter_inputs.len() {
            return Err(training(
                "compiled native CPU evaluation active parameter cardinality mismatch",
            ));
        }
        let mut recurrent = BTreeMap::new();
        for (binding, read) in prepared.parameter_inputs.iter().zip(reads) {
            if read.ordinal() != recurrent.len() || read.buffer_id() != binding.buffer {
                return Err(training(
                    "compiled native CPU evaluation active parameter mapping mismatch",
                ));
            }
            if recurrent
                .insert(binding.input.clone(), read.tensor())
                .is_some()
            {
                return Err(training(
                    "compiled native CPU evaluation active parameter input is duplicated",
                ));
            }
        }
        let started = Instant::now();
        let capture = self.inference.capture();
        let executor_started = Instant::now();
        let executed = executor.execute_planned_native_items_with_recurrent_inputs(
            capture,
            inputs,
            &recurrent,
            &mut prepared.plan,
        );
        let executor_wall_time = executor_started.elapsed();
        let (values, traffic) = executed.map_err(replay_error)?;
        let outputs = values.requested(&capture.requested).map_err(replay_error)?;
        let schedule_cache_keys = prepared.plan.schedule_cache_keys().to_vec();
        let wall_time = started.elapsed();
        let report = NativeCpuRunReport {
            capture_identity: self.capture_identity,
            native_identity: prepared.report.native_identity,
            vectorized: prepared.plan.vectorized(),
            successful_invocation: 0,
            native_item_count: prepared.plan.item_count(),
            executed_native_item_count: traffic.executed_native_item_count,
            module_dispatch_count: traffic.module_dispatch_count,
            module_dispatched_native_item_count: traffic.module_dispatched_native_item_count,
            skipped_output_clear_count: traffic.skipped_output_clear_count,
            schedule_cache_keys,
            native_dispatcher_wall_time: traffic.native_dispatcher_wall_time,
            traffic: native_cpu_replay_traffic(traffic),
            executor_wall_time,
            wall_time,
        };
        debug_assert!(validate_native_cpu_run_report(&report).is_ok());
        Ok((
            evaluation_result(
                outputs,
                &self.output_names,
                loss_weight,
                self.capture_identity,
            )?,
            report,
        ))
    }
}
