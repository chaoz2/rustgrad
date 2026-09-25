//! Runtime reconstruction primitives for admitted compiled-training wire data.

use super::*;

pub(super) fn decode_key_map(
    map: &BTreeMap<String, u64>,
) -> Result<BTreeMap<RecurrentStateKey, u64>> {
    map.iter()
        .map(|(key, buffer)| Ok((RecurrentStateKey::from_canonical(key)?, *buffer)))
        .collect()
}

pub(super) fn decode_input_key_map(
    map: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, RecurrentStateKey>> {
    map.iter()
        .map(|(input, key)| Ok((input.clone(), RecurrentStateKey::from_canonical(key)?)))
        .collect()
}

impl From<&CompiledTokenWeightPolicy> for TokenWeightWire {
    fn from(policy: &CompiledTokenWeightPolicy) -> Self {
        match policy {
            CompiledTokenWeightPolicy::ExplicitMask(name) => Self::ExplicitMask(name.clone()),
            CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            } => Self::IgnoreIndex {
                target_input: target_input.clone(),
                value: *value,
            },
        }
    }
}

impl From<&TokenWeightWire> for CompiledTokenWeightPolicy {
    fn from(policy: &TokenWeightWire) -> Self {
        match policy {
            TokenWeightWire::ExplicitMask(name) => Self::ExplicitMask(name.clone()),
            TokenWeightWire::IgnoreIndex {
                target_input,
                value,
            } => Self::IgnoreIndex {
                target_input: target_input.clone(),
                value: *value,
            },
        }
    }
}

pub(super) fn decode_manifest(
    wire: &AdamWManifestWire,
) -> Result<crate::engine::RecurrentStoreGroupManifest> {
    if wire
        .members
        .iter()
        .enumerate()
        .any(|(role, member)| usize::from(member.role) != role)
    {
        return Err(training(
            "compiled program artifact native AdamW role is invalid",
        ));
    }
    let members = wire
        .members
        .iter()
        .map(|member| crate::engine::RecurrentStoreGroupMember {
            output: member.output,
            state_buffer: member.state_buffer,
        })
        .collect();
    Ok(crate::engine::RecurrentStoreGroupManifest { members })
}

pub(super) fn validate_phase_capture(
    wire: &PhaseWire,
    capture: &CapturedMixedSchedule,
) -> Result<BTreeMap<RecurrentStateKey, u64>> {
    let state_buffers = decode_key_map(&wire.state_buffers)?;
    let frontier = capture
        .initial_recurrent_cursor()
        .map_err(replay_error)?
        .frontier()
        .iter()
        .map(|state| state.buffer)
        .collect::<BTreeSet<_>>();
    if state_buffers.len() != frontier.len()
        || state_buffers.values().copied().collect::<BTreeSet<_>>() != frontier
    {
        return Err(training(
            "compiled program artifact recurrent frontier differs",
        ));
    }
    let input_names = capture
        .schedule
        .inputs
        .iter()
        .map(|input| (input.node, input.name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let captured_inputs = capture.state_bindings.iter().try_fold(
        BTreeMap::<String, u64>::new(),
        |mut inputs, binding| {
            let name = input_names.get(&binding.input_node).ok_or_else(|| {
                training("compiled program artifact recurrent input owner is absent")
            })?;
            if let Some(previous) = inputs.insert((*name).to_owned(), binding.state.buffer)
                && previous != binding.state.buffer
            {
                return Err(training(
                    "compiled program artifact recurrent input aliases buffers",
                ));
            }
            Ok(inputs)
        },
    )?;
    let input_keys = decode_input_key_map(&wire.state_input_keys)?;
    if captured_inputs.keys().ne(input_keys.keys())
        || input_keys
            .iter()
            .any(|(name, key)| state_buffers.get(key) != captured_inputs.get(name))
    {
        return Err(training(
            "compiled program artifact recurrent input schema differs",
        ));
    }
    Ok(state_buffers)
}

pub(super) fn decode_auxiliary(
    main: &CapturedMixedSchedule,
    wire: &PhaseWire,
    capture: Arc<CapturedMixedSchedule>,
    admitted_recurrent: Option<CompiledRecurrentCapture>,
) -> Result<CompiledAdamWAuxiliaryPlan> {
    #[cfg(test)]
    update_decode_counts(|counts| counts.topology_phase_validations += 1);
    let state_buffers = validate_phase_capture(wire, capture.as_ref())?;
    let outputs = CompiledAdamWAuxiliaryOutputSchema::from_report_flags(
        wire.clip_report,
        wire.window_loss_report,
    );
    outputs.validate_report_flags(wire.clip_report, wire.window_loss_report)?;
    let cursor_projection = PreparedRecurrentCursorProjection::prepare(
        main,
        capture.as_ref(),
        state_buffers.values().copied(),
    )
    .map_err(replay_error)?;
    #[cfg(test)]
    update_decode_counts(|counts| counts.cursor_projections += 1);
    let capture_identity = cursor_projection.target_capture_identity();
    let recurrent_capture = match admitted_recurrent {
        Some(recurrent) => recurrent,
        None => CompiledRecurrentCapture::from_artifact(capture.as_ref(), None)?,
    };
    Ok(CompiledAdamWAuxiliaryPlan {
        phase: CompiledRecurrentPhasePlan {
            capture,
            recurrent_capture,
            state_buffers,
            cursor_projection: Arc::new(cursor_projection),
            capture_identity,
            admission: CompiledRecurrentPhaseAdmission::Replace {
                store_groups: wire
                    .adamw_native_updates
                    .iter()
                    .map(decode_manifest)
                    .collect::<Result<_>>()?,
            },
        },
        state_input_keys: decode_input_key_map(&wire.state_input_keys)?,
        outputs,
    })
}

pub(super) fn zero_frontier(
    capture: &CapturedMixedSchedule,
    state_buffers: &BTreeMap<RecurrentStateKey, u64>,
) -> Result<(
    BTreeMap<RecurrentStateKey, TensorData>,
    BTreeMap<RecurrentStateKey, u64>,
)> {
    let cursor = capture.initial_recurrent_cursor().map_err(replay_error)?;
    let descriptors = cursor
        .frontier()
        .iter()
        .map(|state| (state.buffer, state))
        .collect::<BTreeMap<_, _>>();
    let values = state_buffers
        .iter()
        .map(|(key, buffer)| {
            let state = descriptors
                .get(buffer)
                .ok_or_else(|| training("compiled artifact state descriptor is absent"))?;
            Ok((
                key.clone(),
                TensorData::zeros_with_dtype(state.shape.clone(), state.dtype)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let versions = values.keys().cloned().map(|key| (key, 0)).collect();
    Ok((values, versions))
}
