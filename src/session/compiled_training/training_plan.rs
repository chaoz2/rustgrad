//! Backend-neutral compiled-training plan construction and phase captures.

use super::*;

mod evaluation;

pub(super) use evaluation::*;

/// Resource-free output of compiled training graph construction.
#[derive(Clone)]
pub(super) struct CompiledTrainingPlan {
    pub(super) capture: Arc<CapturedMixedSchedule>,
    pub(super) recurrent_capture: Arc<CompiledRecurrentCapture>,
    pub(super) inputs: BTreeMap<String, (Shape, DType)>,
    pub(super) phase_outputs: CompiledTrainingPhaseOutputSchema,
    pub(super) parameter_buffers: BTreeMap<String, u64>,
    pub(super) optimizer_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) workload_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) state_input_buffers: BTreeMap<String, u64>,
    pub(super) state_input_keys: BTreeMap<String, RecurrentStateKey>,
    pub(super) state_values: BTreeMap<RecurrentStateKey, TensorData>,
    pub(super) state_versions: BTreeMap<RecurrentStateKey, u64>,
    pub(super) recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    pub(super) frozen_parameter_nodes: BTreeSet<NodeId>,
    pub(super) step: u64,
    pub(super) accumulation: Option<CompiledTrainingSiblingPlan>,
}

#[derive(Clone)]
pub(super) struct CompiledTrainingSiblingPlan {
    pub(super) phase: CompiledRecurrentPhasePlan,
}

#[derive(Clone)]
pub(super) enum CompiledRecurrentPhaseAdmission {
    RetainUnchanged,
    Replace {
        store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
    },
}

#[derive(Clone)]
pub(super) struct CompiledRecurrentPhasePlan {
    pub(super) capture: Arc<CapturedMixedSchedule>,
    pub(super) recurrent_capture: CompiledRecurrentCapture,
    pub(super) state_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) cursor_projection: Arc<PreparedRecurrentCursorProjection>,
    pub(super) capture_identity: u64,
    pub(super) admission: CompiledRecurrentPhaseAdmission,
}

#[derive(Clone)]
pub(super) struct CompiledAdamWAuxiliaryPlan {
    pub(super) phase: CompiledRecurrentPhasePlan,
    pub(super) state_input_keys: BTreeMap<String, RecurrentStateKey>,
    pub(super) outputs: CompiledAdamWAuxiliaryOutputSchema,
}

impl CompiledRecurrentPhasePlan {
    pub(super) fn capture_identity(&self) -> u64 {
        self.capture_identity
    }

    pub(super) fn store_groups(&self) -> &[crate::engine::RecurrentStoreGroupManifest] {
        match &self.admission {
            CompiledRecurrentPhaseAdmission::RetainUnchanged => &[],
            CompiledRecurrentPhaseAdmission::Replace { store_groups } => store_groups,
        }
    }

    pub(super) fn retains_unchanged(&self) -> bool {
        matches!(
            &self.admission,
            CompiledRecurrentPhaseAdmission::RetainUnchanged
        )
    }
}

impl CompiledTrainingSiblingPlan {
    pub(super) fn phase(&self) -> &CompiledRecurrentPhasePlan {
        &self.phase
    }
}

impl CompiledAdamWAuxiliaryPlan {
    pub(super) fn phase(&self) -> &CompiledRecurrentPhasePlan {
        &self.phase
    }
}

/// Gives one public observation a distinct scheduled owner while preserving
/// its exact descriptor and raw value. Mixed capture rejects duplicate public
/// request identities even when two outputs intentionally report one value.
pub(super) fn materialize_compiled_output_alias(graph: &mut Graph, node: NodeId) -> Result<NodeId> {
    let shape = graph.shape(node)?.clone();
    let dtype = graph.dtype(node)?;
    checked_descriptor(&shape, dtype)?;
    Ok(graph.push(crate::Op::Contiguous { input: node }, shape, dtype))
}

/// Gives only terminal public aliases a concrete schedule owner before mixed
/// capture. RGSM stays owner-only, while ordinary already-materialized losses
/// and named outputs retain their existing graph and capture identities.
pub(super) fn materialize_compiled_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_requested_aliases(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                graph.contiguous(*node)
            } else {
                Ok(*node)
            }
        })
        .collect()
}

/// Gives ownerless public values and values that coincide with recurrent
/// inputs or successors distinct storage. Mixed capture requires every public
/// request to name a scheduled owner and deliberately rejects shared recurrent
/// node identity even when the value is otherwise already materialized.
pub(super) fn materialize_compiled_recurrent_public_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
    state_links: &[InferenceStateLink],
) -> Result<Vec<NodeId>> {
    let requested = materialize_compiled_public_aliases(graph, requested)?;
    let unowned = compiled_unowned_requests(graph, &requested)?;
    let state_nodes = state_links
        .iter()
        .flat_map(|link| [link.input(), link.output()])
        .collect::<BTreeSet<_>>();
    requested
        .into_iter()
        .map(|node| {
            if state_nodes.contains(&node) || unowned.contains(&node) {
                let shape = graph.shape(node)?.clone();
                let dtype = graph.dtype(node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: node }, shape, dtype))
            } else {
                Ok(node)
            }
        })
        .collect()
}

pub(super) fn compiled_requested_aliases(
    graph: &Graph,
    requested: &[NodeId],
) -> Result<BTreeSet<NodeId>> {
    Ok(schedule_many(graph, requested)
        .map_err(schedule_error)?
        .requested_passthroughs
        .iter()
        .map(|alias| alias.requested)
        .collect())
}

pub(super) fn compiled_unowned_requests(
    graph: &Graph,
    requested: &[NodeId],
) -> Result<BTreeSet<NodeId>> {
    let preview = schedule_many(graph, requested).map_err(schedule_error)?;
    let owners = preview
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .map(|output| output.id)
        .collect::<BTreeSet<_>>();
    Ok(requested
        .iter()
        .copied()
        .filter(|node| !owners.contains(&(node.index() as u64)))
        .collect())
}

/// Recurrent outputs must name storage produced by the captured transition,
/// even when their value is a constant or an existing buffer alias. Insert an
/// explicit copy only for those passthroughs; ordinary computed successors
/// retain their original identity and schedule.
pub(super) fn materialize_compiled_state_aliases(
    graph: &mut Graph,
    requested: &[NodeId],
) -> Result<Vec<NodeId>> {
    let aliases = compiled_unowned_requests(graph, requested)?;
    requested
        .iter()
        .map(|node| {
            if aliases.contains(node) {
                let shape = graph.shape(*node)?.clone();
                let dtype = graph.dtype(*node)?;
                checked_descriptor(&shape, dtype)?;
                Ok(graph.push(crate::Op::Contiguous { input: *node }, shape, dtype))
            } else {
                Ok(*node)
            }
        })
        .collect()
}

/// Builds an exact zero that retains the old recurrent value as an
/// authenticated dependency. Unlike subtraction, this remains zero for
/// non-finite floating-point state.
pub(super) fn state_dependent_zero(graph: &mut Graph, input: NodeId) -> Result<NodeId> {
    let shape = graph.shape(input)?.clone();
    let dtype = graph.dtype(input)?;
    let zero_value = if dtype == DType::F32 {
        Scalar::F(0.0)
    } else if dtype == DType::U64 {
        Scalar::U(0)
    } else {
        return Err(training(
            "compiled recurrent zero state dtype is unsupported",
        ));
    };
    let zero = graph.lazy_full_with_dtype(shape, zero_value, dtype)?;
    let false_condition = if dtype == DType::F32 {
        // F32 addition by +0 deliberately remains a graph operation because
        // it changes -0 to +0. Every ordered input compares equal to the
        // shifted value, while NaN makes ordered Lt false.
        let shifted = graph.add(input, zero)?;
        graph.compare(CompareOp::Lt, input, shifted)?
    } else if dtype == DType::U64 {
        // U64 subtraction is defined modulo 2^64. Casting its exact zero
        // avoids a native C self-comparison rejected by Apple Clang.
        let difference = graph.sub(input, input)?;
        graph.cast(difference, DType::Bool)?
    } else {
        unreachable!("state-dependent zero dtype was preflighted")
    };
    graph.select(false_condition, input, zero)
}

pub(super) fn resolve_recurrent_store_groups(
    groups: &[RecurrentStoreGroupSpec],
    updates: &BTreeMap<RecurrentStateKey, NodeId>,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
) -> Result<Vec<crate::engine::RecurrentStoreGroupManifest>> {
    groups
        .iter()
        .map(|group| {
            let members = group
                .members
                .iter()
                .map(|key| {
                    let output = updates
                        .get(key)
                        .ok_or_else(|| training("compiled recurrent store successor is absent"))?;
                    let state_buffer = state_buffers
                        .get(key)
                        .ok_or_else(|| training("compiled recurrent store state is absent"))?;
                    Ok(crate::engine::RecurrentStoreGroupMember {
                        output: output.index() as u64,
                        state_buffer: *state_buffer,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(crate::engine::RecurrentStoreGroupManifest { members })
        })
        .collect()
}

pub(super) struct CompiledTrainingPhaseCapture {
    pub(super) capture: CapturedMixedSchedule,
    pub(super) recurrent_capture: CompiledRecurrentCapture,
    pub(super) state_buffers: BTreeMap<RecurrentStateKey, u64>,
    pub(super) recurrent_store_groups: Vec<crate::engine::RecurrentStoreGroupManifest>,
}

pub(super) struct CompiledTrainingPhaseInput<'a> {
    pub(super) specs: &'a [StateSpec],
    pub(super) state_nodes: &'a BTreeMap<RecurrentStateKey, NodeId>,
    pub(super) state_values: &'a [(u64, TensorData)],
    pub(super) state_by_input: &'a BTreeMap<NodeId, BufferState>,
    pub(super) recurrent_store_groups: &'a [RecurrentStoreGroupSpec],
    pub(super) updates: &'a BTreeMap<RecurrentStateKey, NodeId>,
    pub(super) public_requested: &'a [NodeId],
    pub(super) external_input_names: &'a [String],
    pub(super) materialize_state_passthroughs: bool,
}

pub(super) fn capture_training_phase(
    graph: &mut Graph,
    input: CompiledTrainingPhaseInput<'_>,
) -> Result<CompiledTrainingPhaseCapture> {
    let CompiledTrainingPhaseInput {
        specs,
        state_nodes,
        state_values,
        state_by_input,
        recurrent_store_groups,
        updates,
        public_requested,
        external_input_names,
        materialize_state_passthroughs,
    } = input;
    if updates.len() != specs.len() || specs.iter().any(|spec| !updates.contains_key(&spec.key)) {
        return Err(training("compiled optimizer successor set mismatch"));
    }
    let ordered_updates = specs
        .iter()
        .map(|spec| updates[&spec.key])
        .collect::<Vec<_>>();
    let ordered_updates = if materialize_state_passthroughs {
        materialize_compiled_state_aliases(graph, &ordered_updates)?
    } else {
        ordered_updates
    };
    let updates = specs
        .iter()
        .zip(ordered_updates)
        .map(|(spec, update)| (spec.key.clone(), update))
        .collect::<BTreeMap<_, _>>();
    let state_links = specs
        .iter()
        .map(|spec| InferenceStateLink::new(state_nodes[&spec.key], updates[&spec.key]))
        .collect::<Vec<_>>();
    let public_requested =
        materialize_compiled_recurrent_public_aliases(graph, public_requested, &state_links)?;
    let initial_state = specs
        .iter()
        .map(|spec| (spec.input_name.clone(), spec.value.clone()))
        .collect();
    let public_output_count = public_requested.len();
    let mut requested = Vec::with_capacity(public_output_count + updates.len());
    requested.extend(public_requested.iter().copied());
    for spec in specs {
        requested.push(updates[&spec.key]);
    }
    for node in &requested {
        checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
    }
    let pure = schedule_many(graph, &requested).map_err(schedule_error)?;
    if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
        return Err(training(format!(
            "compiled pure prefix has an unsupported boundary at node {}",
            item.node.index()
        )));
    }
    let state_buffers = specs
        .iter()
        .zip(state_values)
        .map(|(spec, (buffer, _))| (spec.key.clone(), *buffer))
        .collect::<BTreeMap<_, _>>();
    let recurrent_store_groups =
        resolve_recurrent_store_groups(recurrent_store_groups, &updates, &state_buffers)?;
    let mut captured = CapturedSchedule::capture(graph, &pure, &requested[..public_output_count])
        .map_err(replay_error)?;
    if captured.requested.len() != public_output_count {
        return Err(training("compiled capture output count mismatch"));
    }

    let state_bindings = collect_state_bindings(&pure, state_by_input)?;
    let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
    let mut effects = EffectGraph::default();
    let mut effect_bindings = Vec::with_capacity(updates.len());
    for (ordinal, spec) in specs.iter().enumerate() {
        let next = updates[&spec.key];
        if next.index() as u64 >= STATE_BUFFER_BASE {
            return Err(training(
                "graph node identity overlaps persistent state namespace",
            ));
        }
        let buffer = state_values[ordinal].0;
        let destination = effects
            .insert(buffer, spec.value.clone())
            .map_err(effect_error)?;
        let source = effects
            .insert(
                next.index() as u64,
                TensorData::zeros_with_dtype(spec.value.shape().clone(), spec.value.dtype())?,
            )
            .map_err(effect_error)?;
        effects
            .assign(&destination, &source)
            .map_err(effect_error)?;
        let effect_index = u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?;
        effect_bindings.push(value_binding(&pure, next, effect_index)?);
    }
    let mixed = combine_mixed_schedules(
        pure,
        schedule_effects(&effects).map_err(schedule_error)?,
        effect_bindings,
    )
    .map_err(schedule_error)?;
    captured.items = mixed.items.clone();
    let states = effect_states(&effects)?;
    let capture =
        CapturedMixedSchedule::from_parts(captured, &mixed, states).map_err(replay_error)?;
    validate_external_binding_ownership(&capture, external_input_names.iter())?;
    let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
        graph,
        &capture,
        &public_requested,
        &state_links,
        initial_state,
    )?;
    Ok(CompiledTrainingPhaseCapture {
        capture,
        recurrent_capture,
        state_buffers,
        recurrent_store_groups,
    })
}

impl CompiledTrainingPlan {
    /// Compiles one exact static training program.
    ///
    /// `build` receives the declared external inputs and detached parameter
    /// graph inputs. It returns one scalar F32 loss and deterministically named
    /// detached outputs. All parameter gradients are constructed by exactly
    /// one [`Graph::gradient_default`] traversal.
    pub(super) fn compile<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<Self>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_observed(optimizer, parameters, build).map(|(plan, _)| plan)
    }

    pub(super) fn compile_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>)>,
    {
        Self::compile_with_workload_observed(
            optimizer,
            parameters,
            None,
            |graph, inputs, parameters, _| {
                let (loss, outputs) = build(graph, inputs, parameters)?;
                Ok((loss, outputs, None, None))
            },
        )
    }

    pub(super) fn compile_with_token_weight_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
        ) -> Result<(NodeId, BTreeMap<String, NodeId>, NodeId)>,
    {
        Self::compile_with_workload_observed(
            optimizer,
            parameters,
            None,
            |graph, inputs, parameters, _| {
                let (loss, outputs, token_weight) = build(graph, inputs, parameters)?;
                Ok((loss, outputs, None, Some(token_weight)))
            },
        )
    }

    pub(super) fn compile_with_workload_observed<F, O>(
        optimizer: O,
        parameters: impl IntoIterator<Item = TrainingParameterInit>,
        workload: Option<StateSpec>,
        build: F,
    ) -> Result<(Self, CompiledTrainingCompileObservation)>
    where
        O: CompiledOptimizerProgram,
        F: FnOnce(
            &mut Graph,
            &BTreeMap<String, NodeId>,
            &BTreeMap<String, NodeId>,
            Option<NodeId>,
        ) -> Result<(
            NodeId,
            BTreeMap<String, NodeId>,
            Option<NodeId>,
            Option<NodeId>,
        )>,
    {
        let parameters = canonical_parameters(parameters)?;
        if parameters.is_empty() {
            return Err(training(format!(
                "compiled {} needs at least one parameter",
                optimizer.name()
            )));
        }
        if parameters
            .keys()
            .any(|name| optimizer.inputs().contains_key(name))
        {
            return Err(training(
                "compiled parameter and input names must be globally unique",
            ));
        }

        let mut graph = Graph::new();
        let inputs = optimizer
            .inputs()
            .iter()
            .map(|(name, (shape, dtype))| {
                (
                    name.clone(),
                    graph.input_dtype_requires_grad(name.clone(), shape.clone(), *dtype, false),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );

        let mut specs = optimizer.state_specs(&parameters)?;
        let optimizer_spec_count = specs.len();
        if let Some(workload) = workload {
            if specs.iter().any(|spec| spec.key == workload.key) {
                return Err(training("compiled workload state key repeats"));
            }
            specs.push(workload);
        }
        let mut parameter_nodes = BTreeMap::new();
        let mut state_nodes = BTreeMap::new();
        let mut state_values = Vec::with_capacity(specs.len());
        let mut state_by_input = BTreeMap::new();
        let mut parameter_buffers = BTreeMap::new();
        let mut optimizer_buffers = BTreeMap::new();
        let mut workload_buffers = BTreeMap::new();
        let mut state_input_buffers = BTreeMap::new();
        let mut state_input_keys = BTreeMap::new();
        let optimizer_spec_count_u64 = u64::try_from(optimizer_spec_count)
            .map_err(|_| training("optimizer state count overflow"))?;
        for (ordinal, spec) in specs.iter().enumerate() {
            let ordinal = u64::try_from(ordinal).map_err(|_| training("parameter overflow"))?;
            let parameter_buffer = STATE_BUFFER_BASE
                .checked_add(ordinal)
                .ok_or_else(|| training("parameter buffer overflow"))?;
            let node = graph.input_dtype_requires_grad(
                spec.input_name.clone(),
                spec.value.shape().clone(),
                spec.value.dtype(),
                spec.requires_grad,
            );
            let state = state_for(parameter_buffer, &spec.value)?;
            state_nodes.insert(spec.key.clone(), node);
            state_by_input.insert(node, state);
            state_values.push((parameter_buffer, spec.value.clone()));
            state_input_buffers.insert(spec.input_name.clone(), parameter_buffer);
            state_input_keys.insert(spec.input_name.clone(), spec.key.clone());
            if let Some(name) = spec.key.parameter_name() {
                parameter_nodes.insert(name.to_string(), node);
                parameter_buffers.insert(name.to_string(), parameter_buffer);
            } else if ordinal < optimizer_spec_count_u64 {
                optimizer_buffers.insert(spec.key.clone(), parameter_buffer);
            } else {
                workload_buffers.insert(spec.key.clone(), parameter_buffer);
            }
        }
        if parameter_nodes.len() != parameters.len() {
            return Err(training("compiled optimizer omitted parameter state"));
        }

        let workload_node = specs
            .get(optimizer_spec_count)
            .map(|spec| state_nodes[&spec.key]);
        let objective_started = Instant::now();
        let (loss, outputs, workload_successor, token_weight) =
            build(&mut graph, &inputs, &parameter_nodes, workload_node)?;
        let objective_forward = CompiledTrainingCompilePhaseObservation::graph(
            objective_started.elapsed(),
            graph.node_count(),
        );
        validate_loss(&graph, loss)?;
        validate_outputs(
            loss,
            &outputs,
            optimizer.inputs().keys().chain(parameters.keys()),
        )?;

        let targets = parameter_nodes.values().copied().collect::<Vec<_>>();
        let autograd_started = Instant::now();
        let gradients = optimizer.gradients(&mut graph, loss, &targets)?;
        let autograd = CompiledTrainingCompilePhaseObservation::graph(
            autograd_started.elapsed(),
            graph.node_count(),
        );
        if gradients.len() != targets.len() {
            return Err(training("compiled gradient target count mismatch"));
        }
        let gradients = parameter_nodes
            .keys()
            .cloned()
            .zip(gradients)
            .collect::<BTreeMap<_, _>>();
        let optimizer_lowering_started = Instant::now();
        let CompiledOptimizerLowering {
            mut updates,
            mut sibling_updates,
            recurrent_store_groups,
            observations,
        } = optimizer.lower_updates(
            &mut graph,
            CompiledOptimizerLoweringContext {
                loss,
                learning_rate,
                token_weight,
                inputs: &inputs,
                parameters: &parameter_nodes,
                gradients: &gradients,
                states: &state_nodes,
            },
        )?;
        match (specs.get(optimizer_spec_count), workload_successor) {
            (Some(spec), Some(successor)) => {
                updates.insert(spec.key.clone(), successor);
                if let Some(sibling_updates) = &mut sibling_updates {
                    sibling_updates.insert(spec.key.clone(), successor);
                }
            }
            (None, None) => {}
            _ => return Err(training("compiled workload successor set mismatch")),
        }
        let observation_schema =
            CompiledTrainingObservationSchema::from_nodes(&graph, &observations)?;
        let optimizer_lowering = CompiledTrainingCompilePhaseObservation::graph(
            optimizer_lowering_started.elapsed(),
            graph.node_count(),
        );
        let main_public_requested = std::iter::once(loss)
            .chain(outputs.values().copied())
            .chain(observations.iter().map(|observation| observation.node))
            .collect::<Vec<_>>();
        let external_input_names = optimizer.inputs().keys().cloned().collect::<Vec<_>>();
        let main_started = Instant::now();
        let main = capture_training_phase(
            &mut graph,
            CompiledTrainingPhaseInput {
                specs: &specs,
                state_nodes: &state_nodes,
                state_values: &state_values,
                state_by_input: &state_by_input,
                recurrent_store_groups: &recurrent_store_groups,
                updates: &updates,
                public_requested: &main_public_requested,
                external_input_names: &external_input_names,
                materialize_state_passthroughs: false,
            },
        )?;
        let main_capture = CompiledTrainingCompilePhaseObservation::schedule(
            main_started.elapsed(),
            main.recurrent_capture.execution_plan().schedule_item_count,
        );
        let (accumulation, accumulation_capture) = if let Some(updates) = sibling_updates {
            let accumulation_started = Instant::now();
            let public_requested = std::iter::once(loss)
                .chain(outputs.values().copied())
                .collect::<Vec<_>>();
            let phase = capture_training_phase(
                &mut graph,
                CompiledTrainingPhaseInput {
                    specs: &specs,
                    state_nodes: &state_nodes,
                    state_values: &state_values,
                    state_by_input: &state_by_input,
                    recurrent_store_groups: &recurrent_store_groups,
                    updates: &updates,
                    public_requested: &public_requested,
                    external_input_names: &external_input_names,
                    materialize_state_passthroughs: true,
                },
            )?;
            let cursor_projection = PreparedRecurrentCursorProjection::prepare(
                &main.capture,
                &phase.capture,
                phase.state_buffers.values().copied(),
            )
            .map_err(replay_error)?;
            let capture_identity = cursor_projection.target_capture_identity();
            let schedule_item_count = phase.recurrent_capture.execution_plan().schedule_item_count;
            let plan = CompiledTrainingSiblingPlan {
                phase: CompiledRecurrentPhasePlan {
                    capture: Arc::new(phase.capture),
                    recurrent_capture: phase.recurrent_capture,
                    state_buffers: phase.state_buffers,
                    cursor_projection: Arc::new(cursor_projection),
                    capture_identity,
                    admission: CompiledRecurrentPhaseAdmission::RetainUnchanged,
                },
            };
            (
                Some(plan),
                Some(CompiledTrainingCompilePhaseObservation::schedule(
                    accumulation_started.elapsed(),
                    schedule_item_count,
                )),
            )
        } else {
            (None, None)
        };
        if let Some(accumulation) = &accumulation {
            let main_identity = main
                .capture
                .initial_recurrent_cursor()
                .map_err(replay_error)?
                .capture_identity();
            if accumulation.phase().capture_identity == main_identity {
                return Err(training(
                    "compiled accumulation capture identity is not distinct",
                ));
            }
        }

        let phase_outputs = CompiledTrainingPhaseOutputSchema {
            loss: CompiledTrainingLossOutput::ScalarF32,
            named_outputs: outputs.keys().cloned().collect(),
            observations: observation_schema,
        };
        let observation = CompiledTrainingCompileObservation::new(
            objective_forward,
            autograd,
            optimizer_lowering,
            main_capture,
            accumulation_capture,
        );
        Ok((
            Self {
                capture: Arc::new(main.capture),
                recurrent_capture: Arc::new(main.recurrent_capture),
                inputs: optimizer.inputs().clone(),
                phase_outputs,
                parameter_buffers,
                optimizer_buffers,
                workload_buffers,
                state_input_buffers,
                state_input_keys,
                state_values: specs
                    .iter()
                    .map(|spec| (spec.key.clone(), spec.value.clone()))
                    .collect(),
                state_versions: specs.iter().map(|spec| (spec.key.clone(), 0)).collect(),
                recurrent_store_groups: main.recurrent_store_groups,
                frozen_parameter_nodes: BTreeSet::new(),
                step: 0,
                accumulation,
            },
            observation,
        ))
    }

    pub(super) fn capture_identity(&self) -> Result<u64> {
        Ok(self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .capture_identity())
    }

    pub(super) fn recurrent_capture(&self) -> Result<CapturedStatefulInference> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((input.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.recurrent_capture.stateful(initial_state)
    }

    pub(super) fn metal_plan(
        &self,
        renderer: MetalRenderer,
        host_token_inputs: &BTreeMap<String, Shape>,
        evaluation: Option<CompiledEvaluationPlan>,
    ) -> Result<MetalCompiledTrainingPlan> {
        let recurrent = self.recurrent_capture()?;
        let recurrent = if self.recurrent_capture.is_portable() {
            recurrent
        } else {
            recurrent
                .with_authenticated_training_host_indices(
                    host_token_inputs,
                    &self.frozen_parameter_nodes,
                )
                .map_err(captured_inference_error)?
        };
        let inner = MetalStatefulInferencePlan::new(recurrent.clone(), renderer.clone()).map_err(
            |error| {
                let detail = if matches!(&error, MetalError::Unsupported(_)) {
                    recurrent
                        .capture()
                        .items
                        .iter()
                        .find_map(|item| {
                            renderer.render(&item.kernel).err().map(|item_error| {
                                format!(
                                    " at schedule item {} (node {}, {:?}): {item_error}",
                                    item.id,
                                    item.node.index(),
                                    item.kernel.operation()
                                )
                            })
                        })
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                training(format!("compiled Metal runtime: {error:?}{detail}"))
            },
        )?;
        let evaluation = evaluation
            .map(|evaluation| {
                let parameter_names = evaluation
                    .parameter_inputs
                    .values()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let resident_bindings = evaluation
                    .parameter_inputs
                    .iter()
                    .map(|(parameter, input)| {
                        let value = self
                            .state_values
                            .get(&RecurrentStateKey::parameter(parameter))
                            .cloned()
                            .ok_or_else(|| {
                                training("compiled evaluation parameter value is absent")
                            })?;
                        Ok((input.clone(), value))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                MetalFixedStateReadPlan::new(
                    evaluation.inference.inference(resident_bindings)?,
                    renderer,
                    &inner,
                    &parameter_names,
                )
                .map(|plan| (plan, evaluation.output_names, evaluation.capture_identity))
                .map_err(metal_training_error)
            })
            .transpose()?;
        Ok(MetalCompiledTrainingPlan {
            inner,
            inputs: self.inputs.clone(),
            output_names: self.phase_outputs.named_outputs.clone(),
            state_input_keys: self.state_input_keys.clone(),
            program_identity: self.capture_identity()?,
            evaluation,
        })
    }

    #[cfg(test)]
    pub(super) fn restore_frontier(
        self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<Self> {
        let versions = values.keys().cloned().map(|key| (key, step)).collect();
        self.restore_frontier_with_versions(step, values, versions)
    }

    pub(super) fn restore_frontier_with_versions(
        mut self,
        step: u64,
        values: BTreeMap<RecurrentStateKey, TensorData>,
        versions: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<Self> {
        let expected = self
            .parameter_buffers
            .keys()
            .map(RecurrentStateKey::parameter)
            .chain(self.optimizer_buffers.keys().cloned())
            .chain(self.workload_buffers.keys().cloned())
            .collect::<BTreeSet<_>>();
        if values.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state names mismatch"));
        }
        if versions.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(training("compiled checkpoint state versions mismatch"));
        }
        for (key, value) in &values {
            let current = self
                .state_values
                .get(key)
                .ok_or_else(|| training("compiled checkpoint state is absent"))?;
            if value.shape() != current.shape() || value.dtype() != current.dtype() {
                return Err(training("compiled checkpoint state descriptor mismatch"));
            }
            checked_bytes(value)?;
        }
        self.state_values = values;
        self.state_versions = versions;
        self.step = step;
        let frontier = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?
            .frontier()
            .iter()
            .cloned()
            .map(|mut state| {
                let key = self
                    .state_input_buffers
                    .iter()
                    .find_map(|(input, buffer)| {
                        (*buffer == state.buffer).then(|| self.state_input_keys[input].clone())
                    })
                    .ok_or_else(|| training("compiled checkpoint state buffer is absent"))?;
                state.version = self.state_versions[&key];
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        MixedReplayCursor::resume(&self.capture, frontier).map_err(replay_error)?;
        Ok(self)
    }

    pub(super) fn prepare_cpu(&self) -> Result<CpuCompiledTrainingProgram> {
        self.prepare_cpu_with_non_finite_policy(CpuNonFinitePolicy::Propagate)
    }

    pub(super) fn prepare_cpu_with_non_finite_policy(
        &self,
        non_finite_policy: CpuNonFinitePolicy,
    ) -> Result<CpuCompiledTrainingProgram> {
        if non_finite_policy == CpuNonFinitePolicy::RejectTransition {
            validate_finite_tensors(self.state_values.values(), "prepared recurrent state")?;
        }
        let initial_states = self
            .state_input_buffers
            .iter()
            .map(|(input, buffer)| {
                let key = self
                    .state_input_keys
                    .get(input)
                    .ok_or_else(|| training("compiled plan state key is absent"))?;
                let value = self
                    .state_values
                    .get(key)
                    .cloned()
                    .ok_or_else(|| training("compiled plan state value is absent"))?;
                Ok((*buffer, value))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut runtime = EffectRuntime::new();
        runtime
            .register_initial_states(initial_states)
            .map_err(runtime_error)?;
        let cursor = self
            .capture
            .initial_recurrent_cursor()
            .map_err(replay_error)?;
        let mut program = CpuCompiledTrainingProgram {
            capture: self.capture.clone(),
            recurrent_capture: self.recurrent_capture.clone(),
            runtime,
            cursor,
            inputs: self.inputs.clone(),
            phase_outputs: self.phase_outputs.clone(),
            parameter_buffers: self.parameter_buffers.clone(),
            optimizer_buffers: self.optimizer_buffers.clone(),
            workload_buffers: self.workload_buffers.clone(),
            state_input_buffers: self.state_input_buffers.clone(),
            state_input_keys: self.state_input_keys.clone(),
            recurrent_store_groups: self.recurrent_store_groups.clone(),
            frozen_parameter_nodes: self.frozen_parameter_nodes.clone(),
            step: 0,
            accumulation: self.accumulation.clone(),
        };
        if self.step != 0 || self.state_versions.values().any(|version| *version != 0) {
            program.restore_frontier(self.step, &self.state_values, &self.state_versions)?;
        }
        Ok(program)
    }
}

impl CompiledAdamWAuxiliaryPlan {
    pub(super) fn compile_partial_flush(
        training_plan: &CompiledTrainingPlan,
        config: &CompiledAdamWConfig,
    ) -> Result<Self> {
        let topology = CompiledTrainingWindowTopology::from_config(config);
        if !topology.accumulating() {
            return Err(training(
                "compiled AdamW partial flush requires gradient accumulation",
            ));
        }

        let state_buffers = training_plan
            .parameter_buffers
            .iter()
            .map(|(name, buffer)| (RecurrentStateKey::parameter(name), *buffer))
            .chain(
                training_plan
                    .optimizer_buffers
                    .iter()
                    .map(|(key, buffer)| (key.clone(), *buffer)),
            )
            .collect::<BTreeMap<_, _>>();
        let mut graph = Graph::new();
        let learning_rate = graph.input_dtype_requires_grad(
            LEARNING_RATE_INPUT,
            Shape::from([]),
            DType::F32,
            false,
        );
        let mut state_nodes = BTreeMap::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled partial flush state value is absent"))?;
            let node = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            state_nodes.insert(key.clone(), node);
            state_by_input.insert(node, state_for(buffer, &value)?);
            specs.push((input_name.clone(), key.clone(), value, node, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled partial flush state schema differs"));
        }

        let parameters = state_nodes
            .iter()
            .filter_map(|(key, node)| key.parameter_name().map(|name| (name.to_owned(), *node)))
            .collect::<BTreeMap<_, _>>();
        if parameters.len() != training_plan.parameter_buffers.len() {
            return Err(training("compiled partial flush parameter schema differs"));
        }
        let accumulation_index_key =
            RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulationIndex);
        let index = state_nodes
            .get(&accumulation_index_key)
            .copied()
            .ok_or_else(|| training("compiled partial flush accumulation index is absent"))?;
        let token_count_key =
            topology
                .retains_token_count()
                .then_some(RecurrentStateKey::adamw_global(
                    AdamWGlobalState::AccumulatedTokenCount,
                ));
        let divisor = match &token_count_key {
            Some(key) => {
                let count = state_nodes.get(key).copied().ok_or_else(|| {
                    training("compiled partial flush accumulated token count is absent")
                })?;
                graph.cast(count, DType::F32)?
            }
            None => graph.cast(index, DType::F32)?,
        };
        let divisor = safe_token_count_divisor(
            &mut graph,
            divisor,
            config.allow_zero_valid_token_microbatches,
        )?;
        let window_loss_report = if config.window_loss_report {
            let numerator_key =
                RecurrentStateKey::adamw_global(AdamWGlobalState::AccumulatedLossNumerator);
            let numerator = state_nodes.get(&numerator_key).copied().ok_or_else(|| {
                training("compiled partial flush accumulated loss numerator is absent")
            })?;
            let loss_weight = match &token_count_key {
                Some(key) => state_nodes[key],
                None => index,
            };
            Some((
                numerator_key,
                CompiledAdamWWindowLossNodes {
                    mean_loss: graph.div(numerator, divisor)?,
                    loss_weight,
                },
            ))
        } else {
            None
        };
        let mut gradients = BTreeMap::new();
        for name in parameters.keys() {
            let accumulator_key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator = state_nodes
                .get(&accumulator_key)
                .copied()
                .ok_or_else(|| training("compiled partial flush accumulator is absent"))?;
            gradients.insert(name.clone(), graph.div(accumulator, divisor)?);
        }
        let clipped = clip_gradients_by_global_norm(config, &mut graph, &gradients)?;
        let learning_rate =
            lower_adamw_learning_rate(config, &mut graph, learning_rate, &state_nodes)?;
        let mut updates = lower_adamw_update_candidates(
            config,
            &mut graph,
            learning_rate,
            &parameters,
            &clipped.gradients,
            &state_nodes,
        )?;
        // Token-weighted flush divides by the retained count instead of the
        // accumulation index. Derive the exact reset from the old index so
        // the strict recurrent capture still owns every state input.
        let zero_index = if token_count_key.is_some() {
            graph.sub(index, index)?
        } else {
            graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?
        };
        updates.insert(accumulation_index_key, zero_index);
        if let Some(key) = token_count_key {
            let zero_count = graph.full_with_dtype(Shape::from([]), Scalar::U(0), DType::U64)?;
            updates.insert(key, zero_count);
        }
        if let Some((key, _)) = &window_loss_report {
            updates.insert(key.clone(), scalar_f32(&mut graph, 0.0)?);
        }
        for name in parameters.keys() {
            let key =
                RecurrentStateKey::adamw_parameter(name, AdamWParameterState::GradientAccumulator);
            let accumulator = state_nodes
                .get(&key)
                .copied()
                .ok_or_else(|| training("compiled partial flush accumulator is absent"))?;
            let zero = state_dependent_zero(&mut graph, accumulator)?;
            updates.insert(key, zero);
        }
        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let successors = successor_keys
            .iter()
            .map(|key| updates[key])
            .collect::<Vec<_>>();
        let successors = materialize_compiled_state_aliases(&mut graph, &successors)?;
        for (key, successor) in successor_keys.into_iter().zip(successors) {
            updates.insert(key, successor);
        }
        if updates.len() != specs.len()
            || specs.iter().any(|(_, key, ..)| !updates.contains_key(key))
        {
            return Err(training("compiled partial flush successor schema differs"));
        }

        let state_links = specs
            .iter()
            .map(|(_, key, _, node, _)| InferenceStateLink::new(*node, updates[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let observations = adamw_observation_nodes(
            clipped.report,
            window_loss_report.as_ref().map(|(_, report)| *report),
        );
        let outputs = CompiledAdamWAuxiliaryOutputSchema::from_nodes(&graph, &observations)?;
        let public_requested = outputs.node_ids(&observations)?;
        let public_requested = materialize_compiled_recurrent_public_aliases(
            &mut graph,
            &public_requested,
            &state_links,
        )?;
        let public_output_count = public_requested.len();
        let mut requested = public_requested.clone();
        requested.extend(specs.iter().map(|(_, key, ..)| updates[key]));
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled partial flush has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured =
            CapturedSchedule::capture(&graph, &pure, &requested[..public_output_count])
                .map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = updates[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
            &graph,
            &capture,
            &public_requested,
            &state_links,
            initial_state,
        )?;
        let cursor_projection = PreparedRecurrentCursorProjection::prepare(
            &training_plan.capture,
            &capture,
            state_buffers.values().copied(),
        )
        .map_err(replay_error)?;
        let capture_identity = cursor_projection.target_capture_identity();
        let recurrent_store_groups = resolve_recurrent_store_groups(
            &adamw_recurrent_store_group_specs(parameters.keys(), topology),
            &updates,
            &state_buffers,
        )?;
        Ok(Self {
            phase: CompiledRecurrentPhasePlan {
                capture: Arc::new(capture),
                recurrent_capture,
                state_buffers,
                cursor_projection: Arc::new(cursor_projection),
                capture_identity,
                admission: CompiledRecurrentPhaseAdmission::Replace {
                    store_groups: recurrent_store_groups,
                },
            },
            state_input_keys: specs
                .into_iter()
                .map(|(input, key, ..)| (input, key))
                .collect(),
            outputs,
        })
    }

    pub(super) fn compile_zero_grad(
        training_plan: &CompiledTrainingPlan,
        topology: CompiledTrainingWindowTopology,
    ) -> Result<Self> {
        if !topology.accumulating() {
            return Err(training(
                "compiled AdamW zero-grad requires gradient accumulation",
            ));
        }
        let state_buffers = training_plan
            .optimizer_buffers
            .iter()
            .filter(|(key, _)| key.is_accumulation_reset_state())
            .map(|(key, buffer)| (key.clone(), *buffer))
            .collect::<BTreeMap<_, _>>();
        if state_buffers.is_empty() {
            return Err(training(
                "compiled AdamW zero-grad requires gradient accumulation",
            ));
        }

        Ok(Self {
            phase: CompiledRecurrentPhasePlan::compile_state_only_reset(
                training_plan,
                state_buffers,
            )?,
            state_input_keys: training_plan
                .state_input_keys
                .iter()
                .filter(|(_, key)| key.is_accumulation_reset_state())
                .map(|(input, key)| (input.clone(), key.clone()))
                .collect(),
            outputs: CompiledAdamWAuxiliaryOutputSchema::from_report_flags(false, false),
        })
    }
}

impl CompiledRecurrentPhasePlan {
    pub(super) fn compile_state_only_reset(
        training_plan: &CompiledTrainingPlan,
        state_buffers: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<Self> {
        let mut graph = Graph::new();
        let mut state_by_input = BTreeMap::new();
        let mut specs = Vec::with_capacity(state_buffers.len());
        let mut successors = BTreeMap::new();
        for (input_name, key) in &training_plan.state_input_keys {
            let Some(buffer) = state_buffers.get(key).copied() else {
                continue;
            };
            let value = training_plan
                .state_values
                .get(key)
                .cloned()
                .ok_or_else(|| training("compiled zero-grad state value is absent"))?;
            let input = graph.input_dtype_requires_grad(
                input_name.clone(),
                value.shape().clone(),
                value.dtype(),
                false,
            );
            let successor = state_dependent_zero(&mut graph, input)?;
            state_by_input.insert(input, state_for(buffer, &value)?);
            successors.insert(key.clone(), successor);
            specs.push((input_name.clone(), key.clone(), value, input, buffer));
        }
        if specs.len() != state_buffers.len() {
            return Err(training("compiled zero-grad state schema differs"));
        }

        let successor_keys = specs
            .iter()
            .map(|(_, key, ..)| key.clone())
            .collect::<Vec<_>>();
        let materialized = materialize_compiled_state_aliases(
            &mut graph,
            &successor_keys
                .iter()
                .map(|key| successors[key])
                .collect::<Vec<_>>(),
        )?;
        for (key, successor) in successor_keys.into_iter().zip(materialized) {
            successors.insert(key, successor);
        }
        let state_links = specs
            .iter()
            .map(|(_, key, _, input, _)| InferenceStateLink::new(*input, successors[key]))
            .collect::<Vec<_>>();
        let initial_state = specs
            .iter()
            .map(|(input, _, value, _, _)| (input.clone(), value.clone()))
            .collect();
        let requested = specs
            .iter()
            .map(|(_, key, ..)| successors[key])
            .collect::<Vec<_>>();
        for node in &requested {
            checked_descriptor(graph.shape(*node)?, graph.dtype(*node)?)?;
        }
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled zero-grad has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[]).map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let mut effects = EffectGraph::default();
        let mut effect_bindings = Vec::with_capacity(specs.len());
        for (ordinal, (_, key, value, _, buffer)) in specs.iter().enumerate() {
            let next = successors[key];
            if next.index() as u64 >= STATE_BUFFER_BASE {
                return Err(training(
                    "graph node identity overlaps persistent state namespace",
                ));
            }
            let destination = effects
                .insert(*buffer, value.clone())
                .map_err(effect_error)?;
            let source = effects
                .insert(
                    next.index() as u64,
                    TensorData::zeros_with_dtype(value.shape().clone(), value.dtype())?,
                )
                .map_err(effect_error)?;
            effects
                .assign(&destination, &source)
                .map_err(effect_error)?;
            effect_bindings.push(value_binding(
                &pure,
                next,
                u64::try_from(ordinal).map_err(|_| training("effect index overflow"))?,
            )?);
        }
        let mixed = combine_mixed_schedules(
            pure,
            schedule_effects(&effects).map_err(schedule_error)?,
            effect_bindings,
        )
        .map_err(schedule_error)?;
        captured.items = mixed.items.clone();
        let capture = CapturedMixedSchedule::from_parts(captured, &mixed, effect_states(&effects)?)
            .map_err(replay_error)?;
        validate_external_binding_ownership(&capture, std::iter::empty::<&String>())?;
        let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
            &graph,
            &capture,
            &[],
            &state_links,
            initial_state,
        )?;
        let cursor_projection = PreparedRecurrentCursorProjection::prepare(
            &training_plan.capture,
            &capture,
            state_buffers.values().copied(),
        )
        .map_err(replay_error)?;
        let capture_identity = cursor_projection.target_capture_identity();
        Ok(Self {
            capture: Arc::new(capture),
            recurrent_capture,
            state_buffers,
            cursor_projection: Arc::new(cursor_projection),
            capture_identity,
            admission: CompiledRecurrentPhaseAdmission::Replace {
                store_groups: Vec::new(),
            },
        })
    }
}

impl CompiledAdamWAuxiliaryPlan {
    pub(super) fn capture_identity(&self) -> u64 {
        self.phase.capture_identity()
    }

    pub(super) fn with_frontier(
        mut self,
        values: &BTreeMap<RecurrentStateKey, TensorData>,
    ) -> Result<Self> {
        let initial_state = self
            .state_input_keys
            .iter()
            .map(|(input, key)| {
                Ok((
                    input.clone(),
                    values
                        .get(key)
                        .cloned()
                        .ok_or_else(|| training("compiled auxiliary frontier is absent"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.phase
            .recurrent_capture
            .rebind_stateful(initial_state)?;
        Ok(self)
    }
}
