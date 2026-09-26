//! Recurrent phase ownership, mixed capture, and state-only resets.

use super::*;

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
    pub(super) capture_measurement: RecurrentCaptureStageMeasurement,
}

pub(super) struct RecurrentCaptureStageMeasurement {
    alias_planning_wall_time: Duration,
    preview_schedule_count: usize,
    final_schedule_wall_time: Duration,
    pure_capture_binding_wall_time: Duration,
    effect_assembly_sealing_wall_time: Duration,
    recurrent_authentication_wall_time: Duration,
    cursor_projection_wall_time: Option<Duration>,
    recurrent_state_count: usize,
}

impl RecurrentCaptureStageMeasurement {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        alias_planning_wall_time: Duration,
        preview_schedule_count: usize,
        final_schedule_wall_time: Duration,
        pure_capture_binding_wall_time: Duration,
        effect_assembly_sealing_wall_time: Duration,
        recurrent_authentication_wall_time: Duration,
        cursor_projection_wall_time: Option<Duration>,
        recurrent_state_count: usize,
    ) -> Self {
        Self {
            alias_planning_wall_time,
            preview_schedule_count,
            final_schedule_wall_time,
            pure_capture_binding_wall_time,
            effect_assembly_sealing_wall_time,
            recurrent_authentication_wall_time,
            cursor_projection_wall_time,
            recurrent_state_count,
        }
    }

    pub(super) fn with_cursor_projection(mut self, wall_time: Duration) -> Self {
        self.cursor_projection_wall_time = Some(wall_time);
        self
    }

    pub(super) fn finish(
        self,
        phase_wall_time: Duration,
    ) -> Result<CompiledTrainingRecurrentCaptureObservation> {
        CompiledTrainingRecurrentCaptureObservation::new(
            phase_wall_time,
            self.alias_planning_wall_time,
            self.preview_schedule_count,
            self.final_schedule_wall_time,
            self.pure_capture_binding_wall_time,
            self.effect_assembly_sealing_wall_time,
            self.recurrent_authentication_wall_time,
            self.cursor_projection_wall_time,
            self.recurrent_state_count,
        )
    }
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
    let mut alias_planning_wall_time = Duration::ZERO;
    let ordered_updates = if materialize_state_passthroughs {
        let state_alias_started = Instant::now();
        let materialized = materialize_compiled_state_aliases(graph, &ordered_updates)?;
        alias_planning_wall_time = state_alias_started.elapsed();
        materialized
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
    let public_alias_started = Instant::now();
    let public_requested =
        materialize_compiled_recurrent_public_aliases(graph, public_requested, &state_links)?;
    alias_planning_wall_time = alias_planning_wall_time
        .checked_add(public_alias_started.elapsed())
        .ok_or_else(|| training("compiled alias planning duration overflows"))?;
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
    let final_schedule_started = Instant::now();
    let pure = schedule_many(graph, &requested).map_err(schedule_error)?;
    let final_schedule_wall_time = final_schedule_started.elapsed();
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
    let pure_capture_binding_started = Instant::now();
    let mut captured = CapturedSchedule::capture(graph, &pure, &requested[..public_output_count])
        .map_err(replay_error)?;
    if captured.requested.len() != public_output_count {
        return Err(training("compiled capture output count mismatch"));
    }

    let state_bindings = collect_state_bindings(&pure, state_by_input)?;
    let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
    let pure_capture_binding_wall_time = pure_capture_binding_started.elapsed();
    let effect_assembly_sealing_started = Instant::now();
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
    let effect_assembly_sealing_wall_time = effect_assembly_sealing_started.elapsed();
    let recurrent_authentication_started = Instant::now();
    let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
        graph,
        &capture,
        &public_requested,
        &state_links,
        initial_state,
    )?;
    let recurrent_authentication_wall_time = recurrent_authentication_started.elapsed();
    Ok(CompiledTrainingPhaseCapture {
        capture,
        recurrent_capture,
        state_buffers,
        recurrent_store_groups,
        capture_measurement: RecurrentCaptureStageMeasurement::new(
            alias_planning_wall_time,
            if materialize_state_passthroughs { 3 } else { 2 },
            final_schedule_wall_time,
            pure_capture_binding_wall_time,
            effect_assembly_sealing_wall_time,
            recurrent_authentication_wall_time,
            None,
            specs.len(),
        ),
    })
}

impl CompiledRecurrentPhasePlan {
    pub(super) fn compile_state_only_reset(
        training_plan: &CompiledTrainingPlan,
        state_buffers: BTreeMap<RecurrentStateKey, u64>,
    ) -> Result<(Self, RecurrentCaptureStageMeasurement)> {
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
        let alias_planning_started = Instant::now();
        let materialized = materialize_compiled_state_aliases(
            &mut graph,
            &successor_keys
                .iter()
                .map(|key| successors[key])
                .collect::<Vec<_>>(),
        )?;
        let alias_planning_wall_time = alias_planning_started.elapsed();
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
        let final_schedule_started = Instant::now();
        let pure = schedule_many(&graph, &requested).map_err(schedule_error)?;
        let final_schedule_wall_time = final_schedule_started.elapsed();
        if let Some(item) = pure.items.iter().find(|item| item.boundary.is_some()) {
            return Err(training(format!(
                "compiled zero-grad has an unsupported boundary at node {}",
                item.node.index()
            )));
        }
        let pure_capture_binding_started = Instant::now();
        let mut captured = CapturedSchedule::capture(&graph, &pure, &[]).map_err(replay_error)?;
        let state_bindings = collect_state_bindings(&pure, &state_by_input)?;
        let pure = bind_schedule_states(pure, state_bindings).map_err(schedule_error)?;
        let pure_capture_binding_wall_time = pure_capture_binding_started.elapsed();
        let effect_assembly_sealing_started = Instant::now();
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
        let effect_assembly_sealing_wall_time = effect_assembly_sealing_started.elapsed();
        let recurrent_authentication_started = Instant::now();
        let recurrent_capture = CompiledRecurrentCapture::from_canonical_mixed(
            &graph,
            &capture,
            &[],
            &state_links,
            initial_state,
        )?;
        let recurrent_authentication_wall_time = recurrent_authentication_started.elapsed();
        let cursor_projection_started = Instant::now();
        let cursor_projection = PreparedRecurrentCursorProjection::prepare(
            &training_plan.capture,
            &capture,
            state_buffers.values().copied(),
        )
        .map_err(replay_error)?;
        let cursor_projection_wall_time = cursor_projection_started.elapsed();
        let capture_identity = cursor_projection.target_capture_identity();
        Ok((
            Self {
                capture: Arc::new(capture),
                recurrent_capture,
                state_buffers,
                cursor_projection: Arc::new(cursor_projection),
                capture_identity,
                admission: CompiledRecurrentPhaseAdmission::Replace {
                    store_groups: Vec::new(),
                },
            },
            RecurrentCaptureStageMeasurement::new(
                alias_planning_wall_time,
                1,
                final_schedule_wall_time,
                pure_capture_binding_wall_time,
                effect_assembly_sealing_wall_time,
                recurrent_authentication_wall_time,
                Some(cursor_projection_wall_time),
                specs.len(),
            ),
        ))
    }
}
