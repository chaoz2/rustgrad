//! Authenticated recurrent and evaluation captures for compiled training programs.

#[cfg(test)]
use super::program_artifact;
use super::{
    PortableCapturedInferenceRecipe, PortableInferenceHostPolicy, captured_inference_error,
    replay_error, schedule_error, training,
};
use crate::{
    CapturedMixedSchedule, CapturedSchedule, CapturedStatefulInference, ExecutionPlanSummary,
    Graph, InferenceStateLink, NodeId, Result, Shape, TensorData,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(super) struct CompiledRecurrentCapture {
    stateful: Option<CapturedStatefulInference>,
    portable: Option<Arc<PortableCapturedInferenceRecipe>>,
    execution_plan: Arc<ExecutionPlanSummary>,
}

impl CompiledRecurrentCapture {
    fn from_stateful(stateful: CapturedStatefulInference) -> Self {
        Self {
            execution_plan: Arc::new(stateful.execution_plan().clone()),
            stateful: Some(stateful),
            portable: None,
        }
    }

    pub(super) fn from_artifact(
        capture: &CapturedMixedSchedule,
        portable: Option<PortableCapturedInferenceRecipe>,
    ) -> Result<Self> {
        let (pure, execution_plan) = artifact_recurrent_execution_plan(capture)?;
        if let Some(portable) = &portable
            && portable.capture_bytes() != pure.to_bytes().map_err(replay_error)?
        {
            return Err(training(
                "compiled program artifact Metal recipe capture differs",
            ));
        }
        Ok(Self {
            stateful: None,
            portable: portable.map(Arc::new),
            execution_plan: Arc::new(execution_plan),
        })
    }

    pub(super) fn from_canonical_mixed(
        graph: &Graph,
        capture: &CapturedMixedSchedule,
        public_requested: &[NodeId],
        state_links: &[InferenceStateLink],
        initial_state: BTreeMap<String, TensorData>,
    ) -> Result<Self> {
        let prefix = AuthenticatedRecurrentPrefix::from_mixed(capture)?;
        #[cfg(test)]
        let reference_initial_state = initial_state.clone();
        let stateful = CapturedStatefulInference::from_captured_graph(
            graph,
            prefix.capture,
            prefix.execution_plan,
            public_requested,
            state_links,
            initial_state,
        )
        .map_err(captured_inference_error)?;
        #[cfg(test)]
        {
            record_canonical_recurrent_capture();
            if canonical_recurrent_reference_enabled() {
                let reference = CapturedStatefulInference::from_graph(
                    graph,
                    public_requested,
                    state_links,
                    reference_initial_state,
                )
                .map_err(captured_inference_error)?;
                let stateful_recipe = stateful
                    .portable_recipe(PortableInferenceHostPolicy::None)
                    .and_then(|recipe| recipe.to_bytes())
                    .map_err(captured_inference_error)?;
                let reference_recipe = reference
                    .portable_recipe(PortableInferenceHostPolicy::None)
                    .and_then(|recipe| recipe.to_bytes())
                    .map_err(captured_inference_error)?;
                if stateful.capture().to_bytes().map_err(replay_error)?
                    != reference.capture().to_bytes().map_err(replay_error)?
                    || stateful.execution_plan() != reference.execution_plan()
                    || stateful.public_output_count() != reference.public_output_count()
                    || stateful.state_links().ne(reference.state_links())
                    || stateful.initial_state() != reference.initial_state()
                    || stateful.deployment_identity() != reference.deployment_identity()
                    || stateful_recipe != reference_recipe
                {
                    return Err(training(
                        "canonical recurrent capture differs from reference construction",
                    ));
                }
                record_reference_recurrent_capture();
            }
        }
        Ok(Self::from_stateful(stateful))
    }

    pub(super) fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.execution_plan.as_ref()
    }

    pub(super) fn is_portable(&self) -> bool {
        self.portable.is_some()
    }

    pub(super) fn rebind_stateful(
        &mut self,
        initial_state: BTreeMap<String, TensorData>,
    ) -> Result<()> {
        if let Some(stateful) = self.stateful.take() {
            self.stateful = Some(
                stateful
                    .with_initial_state(initial_state)
                    .map_err(captured_inference_error)?,
            );
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn execution_plan_allocation_identity(&self) -> usize {
        Arc::as_ptr(&self.execution_plan) as usize
    }

    pub(super) fn stateful(
        &self,
        initial_state: BTreeMap<String, TensorData>,
    ) -> Result<CapturedStatefulInference> {
        if let Some(stateful) = &self.stateful {
            return stateful
                .clone()
                .with_initial_state(initial_state)
                .map_err(captured_inference_error);
        }
        self.portable
            .as_ref()
            .ok_or_else(|| {
                training("compiled AdamW program artifacts currently prepare on CPU only")
            })?
            .instantiate_stateful(initial_state)
            .map_err(captured_inference_error)
    }

    pub(super) fn portable_recipe(
        &self,
        policy: PortableInferenceHostPolicy,
    ) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.stateful
            .as_ref()
            .ok_or_else(|| training("compiled recurrent capture is absent"))?
            .portable_recipe(policy)
            .map_err(captured_inference_error)
    }

    pub(super) fn portable_training_recipe(
        &self,
        host_token_inputs: &BTreeMap<String, Shape>,
        frozen_parameter_nodes: &BTreeSet<NodeId>,
    ) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.stateful
            .as_ref()
            .ok_or_else(|| training("compiled recurrent capture is absent"))?
            .clone()
            .with_authenticated_training_host_indices(host_token_inputs, frozen_parameter_nodes)
            .map_err(captured_inference_error)?
            .portable_recipe(if host_token_inputs.is_empty() {
                PortableInferenceHostPolicy::None
            } else {
                PortableInferenceHostPolicy::Training
            })
            .map_err(captured_inference_error)
    }
}

fn artifact_recurrent_execution_plan(
    capture: &CapturedMixedSchedule,
) -> Result<(CapturedSchedule, ExecutionPlanSummary)> {
    #[cfg(test)]
    program_artifact::record_recurrent_execution_plan();
    let prefix = AuthenticatedRecurrentPrefix::from_mixed(capture)?;
    Ok((prefix.capture, prefix.execution_plan))
}

struct AuthenticatedRecurrentPrefix {
    capture: CapturedSchedule,
    execution_plan: ExecutionPlanSummary,
}

impl AuthenticatedRecurrentPrefix {
    fn from_mixed(capture: &CapturedMixedSchedule) -> Result<Self> {
        let split = capture
            .schedule
            .items
            .iter()
            .position(crate::ScheduleItem::is_effect)
            .ok_or_else(|| training("compiled artifact mixed capture has no effects"))?;
        if capture.schedule.items[split..]
            .iter()
            .any(|item| !item.is_effect())
        {
            return Err(training(
                "compiled artifact mixed capture does not have an ordered effect suffix",
            ));
        }
        let mut pure = capture.schedule.clone();
        pure.items.truncate(split);
        let split =
            u64::try_from(split).map_err(|_| training("compiled artifact split overflows"))?;
        for item in &mut pure.items {
            item.consumers.retain(|consumer| *consumer < split);
        }
        pure.requested.extend(
            capture
                .value_bindings
                .iter()
                .map(|binding| binding.producer_output.id),
        );
        let specialization = pure
            .specialized_from
            .as_ref()
            .map(|source| (source.source_identity, source.bindings.as_slice()));
        crate::schedule::rekey_schedule_items(&mut pure.items, &[], specialization)
            .map_err(schedule_error)?;
        pure.identity = crate::schedule::artifact::identity(&pure)
            .map_err(|error| training(format!("compiled artifact capture identity: {error}")))?;
        let pure = CapturedSchedule::from_bytes(&pure.to_bytes().map_err(replay_error)?)
            .map_err(replay_error)?;
        let execution_plan = ExecutionPlanSummary::from_capture(&pure, true)
            .map_err(|error| training(format!("compiled artifact execution summary: {error}")))?;
        Ok(Self {
            capture: pure,
            execution_plan,
        })
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CanonicalRecurrentCaptureCounts {
    pub(super) canonical: usize,
    pub(super) reference: usize,
}

#[cfg(test)]
std::thread_local! {
    static CANONICAL_RECURRENT_CAPTURE_COUNTS: std::cell::Cell<CanonicalRecurrentCaptureCounts> =
        const { std::cell::Cell::new(CanonicalRecurrentCaptureCounts {
            canonical: 0,
            reference: 0,
        }) };
    static CANONICAL_RECURRENT_REFERENCE_ENABLED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
struct CanonicalRecurrentReferenceGuard(bool);

#[cfg(test)]
impl Drop for CanonicalRecurrentReferenceGuard {
    fn drop(&mut self) {
        CANONICAL_RECURRENT_REFERENCE_ENABLED.with(|enabled| enabled.set(self.0));
    }
}

#[cfg(test)]
pub(super) fn with_canonical_recurrent_reference<T>(f: impl FnOnce() -> T) -> T {
    let previous = CANONICAL_RECURRENT_REFERENCE_ENABLED.with(|enabled| enabled.replace(true));
    let _guard = CanonicalRecurrentReferenceGuard(previous);
    f()
}

#[cfg(test)]
pub(super) fn canonical_recurrent_capture_counts() -> CanonicalRecurrentCaptureCounts {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_canonical_recurrent_capture() {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.canonical += 1;
        counts.set(next);
    });
}

#[cfg(test)]
fn canonical_recurrent_reference_enabled() -> bool {
    CANONICAL_RECURRENT_REFERENCE_ENABLED.with(std::cell::Cell::get)
}

#[cfg(test)]
fn record_reference_recurrent_capture() {
    CANONICAL_RECURRENT_CAPTURE_COUNTS.with(|counts| {
        let mut next = counts.get();
        next.reference += 1;
        counts.set(next);
    });
}

#[cfg(test)]
pub(super) fn canonical_recurrent_capture_delta(
    before: CanonicalRecurrentCaptureCounts,
    after: CanonicalRecurrentCaptureCounts,
) -> CanonicalRecurrentCaptureCounts {
    CanonicalRecurrentCaptureCounts {
        canonical: after.canonical - before.canonical,
        reference: after.reference - before.reference,
    }
}

#[derive(Clone, Debug)]
pub(super) struct CompiledEvaluationCapture {
    inference: Option<crate::CapturedInference>,
    portable: Option<Arc<PortableCapturedInferenceRecipe>>,
    capture: Arc<CapturedSchedule>,
    execution_plan: Arc<ExecutionPlanSummary>,
}

impl CompiledEvaluationCapture {
    pub(super) fn from_inference(inference: crate::CapturedInference) -> Self {
        Self {
            capture: Arc::new(inference.capture().clone()),
            execution_plan: Arc::new(inference.execution_plan().clone()),
            inference: Some(inference),
            portable: None,
        }
    }

    pub(super) fn from_artifact(
        capture: Arc<CapturedSchedule>,
        portable: Option<PortableCapturedInferenceRecipe>,
    ) -> Result<Self> {
        #[cfg(test)]
        program_artifact::record_evaluation_execution_plan();
        let execution_plan = ExecutionPlanSummary::from_capture(capture.as_ref(), true)
            .map_err(|error| training(format!("compiled evaluation artifact summary: {error}")))?;
        if let Some(portable) = &portable
            && portable.capture_bytes() != capture.to_bytes().map_err(replay_error)?
        {
            return Err(training(
                "compiled evaluation artifact Metal recipe capture differs",
            ));
        }
        Ok(Self {
            inference: None,
            portable: portable.map(Arc::new),
            capture,
            execution_plan: Arc::new(execution_plan),
        })
    }

    pub(super) fn capture(&self) -> &CapturedSchedule {
        self.capture.as_ref()
    }

    pub(super) fn execution_plan(&self) -> &ExecutionPlanSummary {
        self.execution_plan.as_ref()
    }

    #[cfg(test)]
    pub(super) fn topology_allocation_identities(&self) -> (usize, usize) {
        (
            Arc::as_ptr(&self.capture) as usize,
            Arc::as_ptr(&self.execution_plan) as usize,
        )
    }

    pub(super) fn inference(
        &self,
        resident_bindings: BTreeMap<String, TensorData>,
    ) -> Result<crate::CapturedInference> {
        if let Some(inference) = &self.inference {
            return Ok(inference.clone());
        }
        self.portable
            .as_ref()
            .ok_or_else(|| {
                training("compiled AdamW program artifacts currently prepare on CPU only")
            })?
            .instantiate(resident_bindings)
            .map_err(captured_inference_error)
    }

    pub(super) fn portable_recipe(&self) -> Result<PortableCapturedInferenceRecipe> {
        if let Some(recipe) = &self.portable {
            return Ok(recipe.as_ref().clone());
        }
        self.inference
            .as_ref()
            .ok_or_else(|| training("compiled evaluation capture is absent"))?
            .portable_recipe(PortableInferenceHostPolicy::FixedGathers)
            .map_err(captured_inference_error)
    }
}
