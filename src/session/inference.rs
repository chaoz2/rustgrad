//! Backend-neutral static module capture and fresh-graph CPU inference.
use crate::nn::{Module, ModuleForward, Parameter, module_input_node_bindings};
use crate::{
    Backend, CapturedReplayExecutor, CapturedReplayTrace, CapturedSchedule, CompileTrace,
    CpuBackend, DType, Error, ExecutionPlanSummary, ExecutionPlanSummaryError, Graph, NodeId, Op,
    ReplayError, ReplayInput, Result, Schedule, ScheduleError, Shape, TensorData, schedule,
    schedule_many,
};
use std::{
    collections::{BTreeMap, BTreeSet, hash_map::DefaultHasher},
    error, fmt,
    hash::{Hash, Hasher},
    time::{Duration, Instant},
};

/// Backend-neutral, resource-free ownership of one static module inference
/// capture and the immutable module values admitted by that capture.
#[derive(Clone, Debug)]
pub struct CapturedInference {
    capture: CapturedSchedule,
    execution_plan: ExecutionPlanSummary,
    resident_bindings: BTreeMap<String, TensorData>,
    transient_inputs: Vec<ReplayInput>,
    host_gathers: Vec<CapturedHostGather>,
    identity: u64,
}

/// Capture-authenticated permission for one internal Gather to omit its
/// device status path after its scalar host index has been checked.  This is
/// runtime policy, not part of the captured schedule or graph operation ABI.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CapturedHostGather {
    pub(crate) input: ReplayInput,
    pub(crate) index: u64,
    pub(crate) output: u64,
    pub(crate) axis: usize,
    pub(crate) axis_extent: usize,
    pub(crate) index_elements: usize,
}

/// One fixed-shape recurrent value whose produced output becomes the input of
/// the next successful inference invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InferenceStateLink {
    input: NodeId,
    output: NodeId,
}

/// One append-only recurrent state update. The output must be an exact raw
/// Scatter-replace of `updates` into `input` through the materialized I32
/// `index` tensor along `axis`; Metal may then authenticate and execute it in
/// place. `index` must be an exact reshape/expand/materialization of the scalar
/// I32 `position` input.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InferenceAppendStateLink {
    input: NodeId,
    output: NodeId,
    position: NodeId,
    index: NodeId,
    updates: NodeId,
    axis: usize,
}

impl InferenceAppendStateLink {
    pub const fn new(
        input: NodeId,
        output: NodeId,
        position: NodeId,
        index: NodeId,
        updates: NodeId,
        axis: usize,
    ) -> Self {
        Self {
            input,
            output,
            position,
            index,
            updates,
            axis,
        }
    }

    pub const fn input(self) -> NodeId {
        self.input
    }

    pub const fn output(self) -> NodeId {
        self.output
    }

    pub const fn index(self) -> NodeId {
        self.index
    }

    pub const fn position(self) -> NodeId {
        self.position
    }

    pub const fn updates(self) -> NodeId {
        self.updates
    }

    pub const fn axis(self) -> usize {
        self.axis
    }
}

impl InferenceStateLink {
    pub const fn new(input: NodeId, output: NodeId) -> Self {
        Self { input, output }
    }

    pub const fn input(self) -> NodeId {
        self.input
    }

    pub const fn output(self) -> NodeId {
        self.output
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedInferenceState {
    pub(crate) link: InferenceStateLink,
    pub(crate) input: ReplayInput,
    pub(crate) output: crate::BufferDesc,
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedInferenceAppendState {
    pub(crate) link: InferenceAppendStateLink,
    pub(crate) input: ReplayInput,
    pub(crate) position: ReplayInput,
    pub(crate) index: crate::BufferDesc,
    pub(crate) updates: crate::BufferDesc,
    pub(crate) output: crate::BufferDesc,
    pub(crate) axis_extent: usize,
    pub(crate) row_elements: usize,
    pub(crate) row_bytes: usize,
}

/// Resource-free authenticated capture with fixed-shape recurrent state.
/// State outputs are private protected owners and are never public results.
#[derive(Clone, Debug)]
pub struct CapturedStatefulInference {
    inference: CapturedInference,
    public_output_count: usize,
    states: Vec<CapturedInferenceState>,
    initial_state: BTreeMap<String, TensorData>,
    identity: u64,
}

/// Resource-free authenticated capture whose fixed F32 state is updated one
/// complete row at a time through a host-validated monotonic I32 index.
#[derive(Clone, Debug)]
pub struct CapturedAppendStateInference {
    inference: CapturedInference,
    public_output_count: usize,
    states: Vec<CapturedInferenceAppendState>,
    initial_state: BTreeMap<String, TensorData>,
    identity: u64,
}

/// Failure while constructing an owned inference capture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapturedInferenceError {
    State(Error),
    Schedule(ScheduleError),
    Capture(ReplayError),
    Summary(ExecutionPlanSummaryError),
    Binding(String),
}

impl fmt::Display for CapturedInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "captured inference error: {self:?}")
    }
}

impl error::Error for CapturedInferenceError {}

impl CapturedInference {
    /// Authenticates one static request and snapshots only graph-bound values
    /// belonging to `module` that remain inputs of the resulting capture.
    /// All validation completes before the owned capture is returned.
    pub fn from_module_graph(
        module: &(impl Module + ?Sized),
        graph: &Graph,
        requested: &[NodeId],
    ) -> std::result::Result<Self, CapturedInferenceError> {
        let module_bindings = module_input_node_bindings(module, graph)
            .map_err(CapturedInferenceError::State)?
            .into_iter()
            .collect::<BTreeMap<_, _>>();

        Self::from_graph_residents_impl(graph, requested, module_bindings, false, &[])
    }

    /// Captures an exact graph-owned dense and packed resident inventory.
    /// Unlike module traversal, this internal model-composition seam requires
    /// every declared owner to remain an input of the capture.
    pub(crate) fn from_graph_residents(
        graph: &Graph,
        requested: &[NodeId],
        residents: BTreeMap<String, (NodeId, TensorData)>,
        quantized: &[crate::engine::capture::QuantizedCaptureBinding],
    ) -> std::result::Result<Self, CapturedInferenceError> {
        Self::from_graph_residents_impl(graph, requested, residents, true, quantized)
    }

    fn from_graph_residents_impl(
        graph: &Graph,
        requested: &[NodeId],
        mut residents: BTreeMap<String, (NodeId, TensorData)>,
        require_complete_inventory: bool,
        quantized: &[crate::engine::capture::QuantizedCaptureBinding],
    ) -> std::result::Result<Self, CapturedInferenceError> {
        let schedule = schedule_many(graph, requested).map_err(CapturedInferenceError::Schedule)?;
        let capture = if quantized.is_empty() {
            CapturedSchedule::capture(graph, &schedule, requested)
        } else {
            CapturedSchedule::capture_with_quantized_bindings(
                graph, &schedule, requested, quantized,
            )
        }
        .map_err(CapturedInferenceError::Capture)?;
        let execution_plan = ExecutionPlanSummary::from_capture(&capture, true)
            .map_err(CapturedInferenceError::Summary)?;

        let mut resident_bindings = BTreeMap::new();
        let mut transient_inputs = Vec::new();
        for input in &capture.inputs {
            let Some((node, _)) = residents.get(&input.name) else {
                transient_inputs.push(input.clone());
                continue;
            };
            let node = *node;
            if node != input.node || input.desc.id != node.index() as u64 {
                if require_complete_inventory {
                    return Err(CapturedInferenceError::Binding(format!(
                        "resident input {} node identity mismatch",
                        input.name
                    )));
                }
                transient_inputs.push(input.clone());
                continue;
            }
            let (_, value) = residents
                .remove(&input.name)
                .expect("checked resident input binding exists");
            validate_captured_inference_binding(input, node, &value)?;
            resident_bindings.insert(input.name.clone(), value);
        }
        if require_complete_inventory && let Some((name, _)) = residents.first_key_value() {
            return Err(CapturedInferenceError::Binding(format!(
                "resident input {name} is absent from captured ownership"
            )));
        }
        let identity = captured_inference_identity(&capture, &resident_bindings)?;
        Ok(Self {
            capture,
            execution_plan,
            resident_bindings,
            transient_inputs,
            host_gathers: Vec::new(),
            identity,
        })
    }

    /// Adds a private, capture-derived status-free Gather policy. Every name
    /// must denote one dense scalar I32 transient whose only admitted Gather
    /// lineage is a value-preserving Reshape/Expand chain. The Gather itself
    /// must be an internal single-output raw movement owner.
    pub(crate) fn with_authenticated_host_gathers(
        mut self,
        names: &[&str],
    ) -> std::result::Result<Self, CapturedInferenceError> {
        if names.is_empty() {
            return Ok(self);
        }
        if !self.host_gathers.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "host Gather policy is already authenticated".into(),
            ));
        }
        let mut requested_names = names.iter().copied().collect::<BTreeSet<_>>();
        if requested_names.len() != names.len() {
            return Err(CapturedInferenceError::Binding(
                "host Gather transient names must be unique".into(),
            ));
        }
        let public = self
            .capture
            .requested
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let passthroughs = self
            .capture
            .requested_passthroughs
            .iter()
            .map(|alias| alias.requested.index() as u64)
            .collect::<BTreeSet<_>>();
        let mut matches = BTreeMap::<String, Vec<CapturedHostGather>>::new();
        for captured in self
            .transient_inputs
            .iter()
            .filter(|input| requested_names.contains(input.name.as_str()))
        {
            let name = &captured.name;
            let source = captured.node;
            if public.contains(&(source.index() as u64))
                || passthroughs.contains(&(source.index() as u64))
            {
                return Err(CapturedInferenceError::Binding(format!(
                    "host Gather input {name} cannot be a public output"
                )));
            }
            if captured.desc.id != source.index() as u64
                || captured.desc.dtype != DType::I32
                || captured
                    .desc
                    .shape
                    .numel()
                    .map_err(CapturedInferenceError::State)?
                    != 1
                || captured.desc.bytes != DType::I32.itemsize()
                || !captured.desc.read_only
            {
                return Err(CapturedInferenceError::Binding(format!(
                    "host Gather input {name} must be one dense scalar I32 transient"
                )));
            }
            for item in &self.capture.items {
                let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
                    item.kernel.operation()
                else {
                    continue;
                };
                let crate::MovementKernelKind::Gather { index, axis, .. } = &plan.kind else {
                    continue;
                };
                if item.outputs.len() != 1
                    || public.contains(&item.outputs.primary().id)
                    || passthroughs.contains(&item.outputs.primary().id)
                {
                    continue;
                }
                let portable =
                    crate::movement_plan::PortableIndexedMovement::new(plan).and_then(|portable| {
                        portable.validate_schedule_bindings(item.ordered_inputs())?;
                        Ok(portable)
                    });
                let Ok(portable) = portable else {
                    continue;
                };
                let link = CapturedHostGather {
                    input: captured.clone(),
                    index: index.node.index() as u64,
                    output: item.outputs.primary().id,
                    axis: *axis,
                    axis_extent: portable.axis_extent(),
                    index_elements: portable.index_elements(),
                };
                let static_link = crate::runtime::static_schedule::StaticHostGather {
                    input: link.input.desc.id,
                    input_desc: link.input.desc.clone(),
                    index: link.index,
                    output: link.output,
                    axis: link.axis,
                    axis_extent: link.axis_extent,
                    index_elements: link.index_elements,
                };
                if crate::runtime::static_schedule::authenticate_host_gather_lineage(
                    &self.capture.items,
                    &static_link,
                )
                .is_ok()
                {
                    matches.entry(name.clone()).or_default().push(link);
                }
            }
        }
        let mut host_gathers = Vec::with_capacity(names.len());
        while let Some(name) = requested_names.pop_first() {
            let candidates = matches.remove(name).unwrap_or_default();
            if candidates.len() != 1 {
                return Err(CapturedInferenceError::Binding(format!(
                    "host Gather input {name} has {} authenticated internal owners",
                    candidates.len()
                )));
            }
            host_gathers.push(candidates.into_iter().next().expect("one candidate"));
        }
        host_gathers.sort_by_key(|link| link.output);
        let mut hasher = DefaultHasher::new();
        "rustgrad-captured-host-gather-v1".hash(&mut hasher);
        self.identity.hash(&mut hasher);
        host_gathers.hash(&mut hasher);
        self.identity = hasher.finish();
        self.host_gathers = host_gathers;
        Ok(self)
    }

    /// Returns the deterministic capture plus resident-payload identity.
    pub const fn deployment_identity(&self) -> u64 {
        self.identity
    }

    /// Returns the exact authenticated schedule capture.
    pub const fn capture(&self) -> &CapturedSchedule {
        &self.capture
    }

    /// Returns the non-executing logical schedule and memory summary.
    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.execution_plan
    }

    /// Returns exact immutable values snapshotted from graph-bound module
    /// state. Mutating the module later cannot alter these values.
    pub const fn resident_bindings(&self) -> &BTreeMap<String, TensorData> {
        &self.resident_bindings
    }

    /// Returns named input schemas that remain caller-supplied per run.
    pub fn transient_inputs(&self) -> &[ReplayInput] {
        &self.transient_inputs
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        CapturedSchedule,
        ExecutionPlanSummary,
        BTreeMap<String, TensorData>,
        Vec<CapturedHostGather>,
        u64,
    ) {
        (
            self.capture,
            self.execution_plan,
            self.resident_bindings,
            self.host_gathers,
            self.identity,
        )
    }
}

impl CapturedStatefulInference {
    pub fn from_module_graph(
        module: &(impl Module + ?Sized),
        graph: &Graph,
        requested: &[NodeId],
        state_links: &[InferenceStateLink],
        initial_state: BTreeMap<String, TensorData>,
    ) -> std::result::Result<Self, CapturedInferenceError> {
        if state_links.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "stateful inference requires at least one state link".into(),
            ));
        }
        let public = requested.iter().copied().collect::<BTreeSet<_>>();
        let mut state_nodes = BTreeSet::new();
        let mut combined = requested.to_vec();
        for link in state_links {
            graph
                .op(link.input)
                .map_err(CapturedInferenceError::State)?;
            graph
                .op(link.output)
                .map_err(CapturedInferenceError::State)?;
            if link.input == link.output
                || public.contains(&link.input)
                || public.contains(&link.output)
                || !state_nodes.insert(link.input)
                || !state_nodes.insert(link.output)
            {
                return Err(CapturedInferenceError::Binding(
                    "state links and public outputs must own distinct nodes".into(),
                ));
            }
            combined.push(link.output);
        }
        let mut inference = CapturedInference::from_module_graph(module, graph, &combined)?;
        let mut states = Vec::with_capacity(state_links.len());
        let mut names = BTreeSet::new();
        for link in state_links {
            let Op::Input { name } = graph
                .op(link.input)
                .map_err(CapturedInferenceError::State)?
            else {
                return Err(CapturedInferenceError::Binding(
                    "state input must be a graph Input".into(),
                ));
            };
            if !names.insert(name.as_str()) || inference.resident_bindings.contains_key(name) {
                return Err(CapturedInferenceError::Binding(
                    "state input name aliases another owned binding".into(),
                ));
            }
            let input = captured_owned_input(&inference.capture, link.input, name, "state")?;
            let output = captured_owned_output(&inference.capture, link.output, "state")?;
            validate_state_descriptors(&input, &output, "state")?;
            let initial = initial_state.get(name).ok_or_else(|| {
                CapturedInferenceError::Binding(format!("missing initial state {name}"))
            })?;
            validate_captured_inference_binding(&input, link.input, initial)?;
            states.push(CapturedInferenceState {
                link: *link,
                input,
                output,
            });
        }
        if let Some(extra) = initial_state
            .keys()
            .find(|name| !names.contains(name.as_str()))
        {
            return Err(CapturedInferenceError::Binding(format!(
                "unexpected initial state {extra}"
            )));
        }
        inference
            .transient_inputs
            .retain(|input| !state_nodes.contains(&input.node));

        let mut hasher = DefaultHasher::new();
        "rustgrad-captured-stateful-inference-v1".hash(&mut hasher);
        inference.identity.hash(&mut hasher);
        requested.len().hash(&mut hasher);
        for state in &states {
            state.link.hash(&mut hasher);
            state.input.hash(&mut hasher);
            state.output.hash(&mut hasher);
            initial_state[&state.input.name]
                .to_le_bytes()
                .map_err(CapturedInferenceError::State)?
                .hash(&mut hasher);
        }
        let identity = hasher.finish();
        Ok(Self {
            inference,
            public_output_count: requested.len(),
            states,
            initial_state,
            identity,
        })
    }

    pub const fn deployment_identity(&self) -> u64 {
        self.identity
    }

    pub const fn capture(&self) -> &CapturedSchedule {
        &self.inference.capture
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.inference.execution_plan
    }

    /// Returns the ordered public-request prefix; state outputs follow it only
    /// in the private authenticated capture inventory.
    pub const fn public_output_count(&self) -> usize {
        self.public_output_count
    }

    pub fn resident_bindings(&self) -> &BTreeMap<String, TensorData> {
        &self.inference.resident_bindings
    }

    pub fn initial_state(&self) -> &BTreeMap<String, TensorData> {
        &self.initial_state
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        &self.inference.transient_inputs
    }

    pub fn state_links(&self) -> impl ExactSizeIterator<Item = InferenceStateLink> + '_ {
        self.states.iter().map(|state| state.link)
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        CapturedInference,
        usize,
        Vec<CapturedInferenceState>,
        BTreeMap<String, TensorData>,
        u64,
    ) {
        (
            self.inference,
            self.public_output_count,
            self.states,
            self.initial_state,
            self.identity,
        )
    }
}

fn authenticate_append_index_graph(
    graph: &Graph,
    link: InferenceAppendStateLink,
    index_shape: &Shape,
) -> std::result::Result<(), CapturedInferenceError> {
    let Op::Expand {
        input: reshaped,
        shape: expanded_shape,
    } = graph
        .op(link.index)
        .map_err(CapturedInferenceError::State)?
    else {
        return Err(CapturedInferenceError::Binding(
            "append index must be one scalar expansion".into(),
        ));
    };
    let Op::Reshape {
        input: position,
        shape: scalar_shape,
    } = graph.op(*reshaped).map_err(CapturedInferenceError::State)?
    else {
        return Err(CapturedInferenceError::Binding(
            "append index must reshape one scalar position".into(),
        ));
    };
    if *position != link.position
        || scalar_shape.rank() != index_shape.rank()
        || scalar_shape.dims().iter().any(|&dimension| dimension != 1)
        || expanded_shape != index_shape
        || graph
            .shape(link.position)
            .map_err(CapturedInferenceError::State)?
            .dims()
            != [1]
        || graph
            .dtype(link.position)
            .map_err(CapturedInferenceError::State)?
            != DType::I32
    {
        return Err(CapturedInferenceError::Binding(
            "append index position lineage is inconsistent".into(),
        ));
    }
    Ok(())
}

impl CapturedAppendStateInference {
    pub fn from_module_graph(
        module: &(impl Module + ?Sized),
        graph: &Graph,
        requested: &[NodeId],
        state_links: &[InferenceAppendStateLink],
        initial_state: BTreeMap<String, TensorData>,
    ) -> std::result::Result<Self, CapturedInferenceError> {
        Self::from_graph_with_capture(
            graph,
            requested,
            state_links,
            initial_state,
            &[],
            |combined| CapturedInference::from_module_graph(module, graph, combined),
        )
    }

    pub(crate) fn from_graph_residents(
        graph: &Graph,
        requested: &[NodeId],
        state_links: &[InferenceAppendStateLink],
        initial_state: BTreeMap<String, TensorData>,
        residents: BTreeMap<String, (NodeId, TensorData)>,
        quantized: &[crate::engine::capture::QuantizedCaptureBinding],
        host_gathers: &[&str],
    ) -> std::result::Result<Self, CapturedInferenceError> {
        Self::from_graph_with_capture(
            graph,
            requested,
            state_links,
            initial_state,
            host_gathers,
            |combined| {
                CapturedInference::from_graph_residents(graph, combined, residents, quantized)
            },
        )
    }

    fn from_graph_with_capture(
        graph: &Graph,
        requested: &[NodeId],
        state_links: &[InferenceAppendStateLink],
        initial_state: BTreeMap<String, TensorData>,
        host_gathers: &[&str],
        capture: impl FnOnce(
            &[NodeId],
        ) -> std::result::Result<CapturedInference, CapturedInferenceError>,
    ) -> std::result::Result<Self, CapturedInferenceError> {
        if state_links.is_empty() {
            return Err(CapturedInferenceError::Binding(
                "append-state inference requires at least one state link".into(),
            ));
        }
        let public = requested.iter().copied().collect::<BTreeSet<_>>();
        let mut owned = BTreeSet::new();
        let mut combined = requested.to_vec();
        for link in state_links {
            for id in [link.input, link.output, link.updates] {
                graph.op(id).map_err(CapturedInferenceError::State)?;
                if public.contains(&id) || !owned.insert(id) {
                    return Err(CapturedInferenceError::Binding(
                        "append state and public outputs must own distinct nodes".into(),
                    ));
                }
            }
            graph
                .op(link.index)
                .map_err(CapturedInferenceError::State)?;
            graph
                .op(link.position)
                .map_err(CapturedInferenceError::State)?;
            if public.contains(&link.index) || public.contains(&link.position) {
                return Err(CapturedInferenceError::Binding(
                    "append position/index cannot be a public output".into(),
                ));
            }
            combined.push(link.output);
        }
        let mut inference = capture(&combined)?.with_authenticated_host_gathers(host_gathers)?;
        if inference
            .capture
            .requested_passthroughs
            .iter()
            .any(|alias| {
                public.contains(&alias.requested)
                    && state_links.iter().any(|state| {
                        alias.source == state.input
                            || alias.source == state.output
                            || alias.source == state.updates
                    })
            })
        {
            return Err(CapturedInferenceError::Binding(
                "append state storage cannot escape through a public alias".into(),
            ));
        }
        let mut states = Vec::with_capacity(state_links.len());
        let mut state_names = BTreeSet::new();
        let mut shared_index = None;
        let mut shared_position = None;
        let mut shared_extent = None;
        for link in state_links {
            let Op::Input { name: state_name } = graph
                .op(link.input)
                .map_err(CapturedInferenceError::State)?
            else {
                return Err(CapturedInferenceError::Binding(
                    "append state input must be a graph Input".into(),
                ));
            };
            let Op::Input {
                name: position_name,
            } = graph
                .op(link.position)
                .map_err(CapturedInferenceError::State)?
            else {
                return Err(CapturedInferenceError::Binding(
                    "append position must be a graph Input".into(),
                ));
            };
            let Op::Scatter {
                base,
                index,
                updates,
                axis,
                add,
            } = graph
                .op(link.output)
                .map_err(CapturedInferenceError::State)?
            else {
                return Err(CapturedInferenceError::Binding(
                    "append state output must be raw Scatter-replace".into(),
                ));
            };
            if *base != link.input
                || *index != link.index
                || *updates != link.updates
                || *axis != link.axis
                || *add
            {
                return Err(CapturedInferenceError::Binding(
                    "append state output does not match its declared update".into(),
                ));
            }
            if graph
                .dtype(link.input)
                .map_err(CapturedInferenceError::State)?
                != DType::F32
                || graph
                    .dtype(link.index)
                    .map_err(CapturedInferenceError::State)?
                    != DType::I32
                || graph
                    .dtype(link.updates)
                    .map_err(CapturedInferenceError::State)?
                    != DType::F32
            {
                return Err(CapturedInferenceError::Binding(
                    "append state requires F32 state/updates and I32 indices".into(),
                ));
            }
            let state_shape = graph
                .shape(link.input)
                .map_err(CapturedInferenceError::State)?;
            let index_shape = graph
                .shape(link.index)
                .map_err(CapturedInferenceError::State)?;
            authenticate_append_index_graph(graph, *link, index_shape)?;
            if index_shape
                != graph
                    .shape(link.updates)
                    .map_err(CapturedInferenceError::State)?
                || index_shape.rank() != state_shape.rank()
                || link.axis >= state_shape.rank()
                || index_shape.dims()[link.axis] != 1
                || index_shape
                    .dims()
                    .iter()
                    .zip(state_shape.dims())
                    .enumerate()
                    .any(|(axis, (index, state))| axis != link.axis && index != state)
            {
                return Err(CapturedInferenceError::Binding(
                    "append update is not one complete state row".into(),
                ));
            }
            let axis_extent = state_shape.dims()[link.axis];
            if shared_index
                .replace(link.index)
                .is_some_and(|id| id != link.index)
                || shared_position
                    .replace(link.position)
                    .is_some_and(|id| id != link.position)
                || shared_extent
                    .replace(axis_extent)
                    .is_some_and(|extent| extent != axis_extent)
            {
                return Err(CapturedInferenceError::Binding(
                    "append state links must share one position and capacity".into(),
                ));
            }
            if !state_names.insert(state_name.as_str())
                || inference.resident_bindings.contains_key(state_name)
                || inference.resident_bindings.contains_key(position_name)
            {
                return Err(CapturedInferenceError::Binding(
                    "append state inputs must not alias immutable residents".into(),
                ));
            }
            let input =
                captured_owned_input(&inference.capture, link.input, state_name, "append state")?;
            let position = captured_owned_input(
                &inference.capture,
                link.position,
                position_name,
                "append position",
            )?;
            let index = captured_owned_output(&inference.capture, link.index, "append index")?;
            let updates =
                captured_owned_output(&inference.capture, link.updates, "append updates")?;
            let output = captured_owned_output(&inference.capture, link.output, "append state")?;
            validate_state_descriptors(&input, &output, "append state")?;
            if updates.view.is_some()
                || updates.read_only
                || index.view.is_some()
                || index.read_only
                || index.dtype != DType::I32
                || updates.shape != index.shape
                || updates.dtype != DType::F32
                || updates.bytes != index.bytes
            {
                return Err(CapturedInferenceError::Binding(
                    "append update is not an owned dense F32 row".into(),
                ));
            }
            let initial = initial_state.get(state_name).ok_or_else(|| {
                CapturedInferenceError::Binding(format!("missing initial state {state_name}"))
            })?;
            validate_captured_inference_binding(&input, link.input, initial)?;
            let row_elements = index_shape.numel().map_err(CapturedInferenceError::State)?;
            if (axis_extent == 0 && row_elements != 0) || axis_extent > i32::MAX as usize + 1 {
                return Err(CapturedInferenceError::Binding(
                    "append state capacity is outside the live I32 position contract".into(),
                ));
            }
            let row_bytes = row_elements
                .checked_mul(DType::F32.itemsize())
                .ok_or_else(|| {
                    CapturedInferenceError::State(Error::ShapeOverflow(index_shape.clone()))
                })?;
            states.push(CapturedInferenceAppendState {
                link: *link,
                input,
                position,
                index,
                updates,
                output,
                axis_extent,
                row_elements,
                row_bytes,
            });
        }
        let output_owners = inference
            .capture
            .items
            .iter()
            .flat_map(|item| item.outputs.iter().map(move |output| (output.id, item)))
            .collect::<BTreeMap<_, _>>();
        let append_owner_ids = states
            .iter()
            .map(|state| {
                output_owners
                    .get(&state.output.id)
                    .map(|item| item.id)
                    .ok_or_else(|| {
                        CapturedInferenceError::Binding("append state owner is absent".into())
                    })
            })
            .collect::<std::result::Result<BTreeSet<_>, _>>()?;
        let static_links = states
            .iter()
            .map(
                |state| crate::runtime::static_schedule::StaticAppendStateLink {
                    input: state.input.desc.id,
                    output: state.output.id,
                    position: state.position.desc.id,
                    index: state.index.id,
                    updates: state.updates.id,
                    axis: state.link.axis(),
                    axis_extent: state.axis_extent,
                    row_elements: state.row_elements,
                    row_bytes: state.row_bytes,
                },
            )
            .collect::<Vec<_>>();
        crate::runtime::static_schedule::authenticate_append_state_index_lineage(
            &inference.capture.items,
            &static_links,
        )
        .map_err(CapturedInferenceError::Binding)?;
        for state in &states {
            let owner = output_owners
                .get(&state.output.id)
                .copied()
                .ok_or_else(|| {
                    CapturedInferenceError::Binding("append state owner is absent".into())
                })?;
            let update_owner = output_owners
                .get(&state.updates.id)
                .copied()
                .ok_or_else(|| {
                    CapturedInferenceError::Binding("append update producer is absent".into())
                })?;
            if update_owner.id == owner.id || !owner.dependencies.contains(&update_owner.id) {
                return Err(CapturedInferenceError::Binding(
                    "append update producer is absent from owner dependencies".into(),
                ));
            }
            for id in [state.input.desc.id, state.updates.id] {
                let consumers = inference
                    .capture
                    .items
                    .iter()
                    .filter(|item| {
                        item.ordered_inputs()
                            .iter()
                            .any(|input| input.desc.id == id)
                    })
                    .collect::<Vec<_>>();
                if consumers.len() != 1 || consumers[0].id != owner.id {
                    return Err(CapturedInferenceError::Binding(
                        "append state inputs must be owned exclusively by their update".into(),
                    ));
                }
            }
            let index_consumers = inference
                .capture
                .items
                .iter()
                .filter(|item| {
                    item.ordered_inputs()
                        .iter()
                        .any(|input| input.desc.id == state.index.id)
                })
                .map(|item| item.id)
                .collect::<BTreeSet<_>>();
            if index_consumers != append_owner_ids {
                return Err(CapturedInferenceError::Binding(
                    "append index must be shared only by append-state owners".into(),
                ));
            }
        }
        if let Some(extra) = initial_state
            .keys()
            .find(|name| !state_names.contains(name.as_str()))
        {
            return Err(CapturedInferenceError::Binding(format!(
                "unexpected initial state {extra}"
            )));
        }
        inference
            .transient_inputs
            .retain(|input| !state_links.iter().any(|state| state.input == input.node));

        let identity = captured_append_state_identity(
            inference.identity,
            requested.len(),
            &states,
            &initial_state,
        )?;
        Ok(Self {
            inference,
            public_output_count: requested.len(),
            states,
            initial_state,
            identity,
        })
    }

    pub const fn deployment_identity(&self) -> u64 {
        self.identity
    }

    pub const fn capture(&self) -> &CapturedSchedule {
        &self.inference.capture
    }

    pub const fn execution_plan(&self) -> &ExecutionPlanSummary {
        &self.inference.execution_plan
    }

    pub const fn public_output_count(&self) -> usize {
        self.public_output_count
    }

    pub fn resident_bindings(&self) -> &BTreeMap<String, TensorData> {
        &self.inference.resident_bindings
    }

    pub fn initial_state(&self) -> &BTreeMap<String, TensorData> {
        &self.initial_state
    }

    pub fn transient_inputs(&self) -> &[ReplayInput] {
        &self.inference.transient_inputs
    }

    pub fn state_links(&self) -> impl ExactSizeIterator<Item = InferenceAppendStateLink> + '_ {
        self.states.iter().map(|state| state.link)
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        CapturedInference,
        usize,
        Vec<CapturedInferenceAppendState>,
        BTreeMap<String, TensorData>,
        u64,
    ) {
        (
            self.inference,
            self.public_output_count,
            self.states,
            self.initial_state,
            self.identity,
        )
    }
}

fn captured_append_state_identity(
    inference_identity: u64,
    public_output_count: usize,
    states: &[CapturedInferenceAppendState],
    initial_state: &BTreeMap<String, TensorData>,
) -> std::result::Result<u64, CapturedInferenceError> {
    let mut hasher = DefaultHasher::new();
    "rustgrad-captured-append-state-inference-v2".hash(&mut hasher);
    inference_identity.hash(&mut hasher);
    public_output_count.hash(&mut hasher);
    for state in states {
        state.link.hash(&mut hasher);
        state.input.hash(&mut hasher);
        state.position.hash(&mut hasher);
        state.index.hash(&mut hasher);
        state.updates.hash(&mut hasher);
        state.output.hash(&mut hasher);
        state.axis_extent.hash(&mut hasher);
        state.row_elements.hash(&mut hasher);
        initial_state[&state.input.name]
            .to_le_bytes()
            .map_err(CapturedInferenceError::State)?
            .hash(&mut hasher);
    }
    Ok(hasher.finish())
}

fn captured_owned_input(
    capture: &CapturedSchedule,
    node: NodeId,
    name: &str,
    label: &str,
) -> std::result::Result<ReplayInput, CapturedInferenceError> {
    capture
        .inputs
        .iter()
        .find(|input| input.node == node && input.name == name)
        .cloned()
        .ok_or_else(|| {
            CapturedInferenceError::Binding(format!(
                "{label} input is absent from captured ownership"
            ))
        })
}

fn captured_owned_output(
    capture: &CapturedSchedule,
    node: NodeId,
    label: &str,
) -> std::result::Result<crate::BufferDesc, CapturedInferenceError> {
    capture
        .items
        .iter()
        .flat_map(|item| item.outputs.iter())
        .find(|output| output.id == node.index() as u64)
        .cloned()
        .ok_or_else(|| {
            CapturedInferenceError::Binding(format!(
                "{label} output must be a directly produced capture owner"
            ))
        })
}

fn validate_state_descriptors(
    input: &ReplayInput,
    output: &crate::BufferDesc,
    label: &str,
) -> std::result::Result<(), CapturedInferenceError> {
    if output.view.is_some()
        || output.read_only
        || input.desc.view.is_some()
        || input.desc.shape != output.shape
        || input.desc.dtype != output.dtype
        || input.desc.bytes != output.bytes
        || input.desc.alignment != output.alignment
    {
        return Err(CapturedInferenceError::Binding(format!(
            "{label} input/output descriptors are not exact full-buffer peers"
        )));
    }
    Ok(())
}

fn validate_captured_inference_binding(
    input: &ReplayInput,
    node: NodeId,
    value: &TensorData,
) -> std::result::Result<(), CapturedInferenceError> {
    if input.node != node
        || input.desc.id != node.index() as u64
        || value.shape() != &input.desc.shape
        || value.dtype() != input.desc.dtype
    {
        return Err(CapturedInferenceError::Binding(format!(
            "resident input {} descriptor mismatch",
            input.name
        )));
    }
    let bytes = value.to_le_bytes().map_err(CapturedInferenceError::State)?;
    if bytes.len() != input.desc.bytes {
        return Err(CapturedInferenceError::Binding(format!(
            "resident input {} byte length mismatch",
            input.name
        )));
    }
    Ok(())
}

fn captured_inference_identity(
    capture: &CapturedSchedule,
    residents: &BTreeMap<String, TensorData>,
) -> std::result::Result<u64, CapturedInferenceError> {
    let mut hasher = DefaultHasher::new();
    "rustgrad-captured-inference-v1".hash(&mut hasher);
    capture.identity.hash(&mut hasher);
    for input in &capture.inputs {
        let Some(value) = residents.get(&input.name) else {
            continue;
        };
        input.hash(&mut hasher);
        value
            .to_le_bytes()
            .map_err(CapturedInferenceError::State)?
            .hash(&mut hasher);
    }
    Ok(hasher.finish())
}

#[derive(Clone, Debug)]
pub struct ModuleInferenceResult {
    output: TensorData,
    trace: CompileTrace,
    parameter_versions: BTreeMap<String, u64>,
}
#[derive(Clone, Debug)]
pub struct NativeModuleInferenceResult {
    output: TensorData,
    trace: CapturedReplayTrace,
    parameter_versions: BTreeMap<String, u64>,
    native_trace: NativeModuleInferenceTrace,
}

/// Immutable, opt-in local observations for one strict native module call.
///
/// Durations are current-thread wall-clock observations, not stable benchmarks
/// or hardware, allocator, RSS, device-memory, or per-kernel measurements.
/// They are deliberately excluded from `identity`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleExecutionReport {
    /// Deterministic identity of the static plan and native policy, excluding
    /// local durations and current cache-hit state.
    pub identity: u64,
    /// Canonical, non-executing logical schedule/memory facts. Strict native
    /// replay does not claim to consume this host allocation plan, so reuse is
    /// intentionally disabled in this summary.
    pub execution_plan: ExecutionPlanSummary,
    pub capture_identity: u64,
    pub native_trace_identity: u64,
    pub vectorized: bool,
    pub native_cache_keys: Vec<Option<String>>,
    pub graph_schedule_capture_duration: Duration,
    pub native_prepare_duration: Duration,
    pub native_execute_duration: Duration,
    pub native_item_count: usize,
    pub zero_pruned_item_count: usize,
    pub zero_materialized_item_count: usize,
    pub cache_hit_count: usize,
    pub cache_miss_count: usize,
}

/// The existing detached strict-native result plus opt-in execution
/// observations. The standard inference API intentionally does not construct
/// this report or measure durations.
#[derive(Clone, Debug)]
pub struct ReportedNativeModuleInferenceResult {
    inference: NativeModuleInferenceResult,
    report: NativeModuleExecutionReport,
}

impl ReportedNativeModuleInferenceResult {
    pub fn inference(&self) -> &NativeModuleInferenceResult {
        &self.inference
    }

    pub fn report(&self) -> &NativeModuleExecutionReport {
        &self.report
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeModuleInferenceTrace {
    pub identity: u64,
    pub capture_identity: u64,
    pub input_shape: crate::Shape,
    pub input_dtype: DType,
    pub parameter_versions: BTreeMap<String, u64>,
    pub vectorized: bool,
    pub renderer_version: &'static str,
    pub native_cache_keys: Vec<Option<String>>,
}
impl NativeModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CapturedReplayTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
    pub fn native_trace(&self) -> &NativeModuleInferenceTrace {
        &self.native_trace
    }
}
impl ModuleInferenceResult {
    pub fn output(&self) -> &TensorData {
        &self.output
    }
    pub fn trace(&self) -> &CompileTrace {
        &self.trace
    }
    pub fn parameter_versions(&self) -> &BTreeMap<String, u64> {
        &self.parameter_versions
    }
}

struct NativeModuleInferenceSetup {
    output: crate::NodeId,
    scheduled: Schedule,
    capture: CapturedSchedule,
    bindings: BTreeMap<String, TensorData>,
    input_shape: crate::Shape,
    parameters: Vec<(String, Parameter)>,
}

fn prepare_native_module_inference(
    module: &impl ModuleForward,
    input: TensorData,
) -> Result<NativeModuleInferenceSetup> {
    if input.dtype() != DType::F32 {
        return Err(Error::SessionTraining {
            reason: "module native CPU inference input must have dtype F32".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    let input_shape = input.shape().clone();
    bindings.insert("module_input".into(), input);
    let bindings = bindings.into_iter().collect::<BTreeMap<_, _>>();
    let scheduled = schedule(&graph, output).map_err(|error| Error::SessionTraining {
        reason: error.to_string(),
    })?;
    let capture = CapturedSchedule::capture(&graph, &scheduled, &[output]).map_err(|error| {
        Error::SessionTraining {
            reason: error.to_string(),
        }
    })?;
    Ok(NativeModuleInferenceSetup {
        output,
        scheduled,
        capture,
        bindings,
        input_shape,
        parameters,
    })
}

fn finish_native_module_inference(
    setup: NativeModuleInferenceSetup,
    replay: crate::CapturedReplayResult,
    vectorized: bool,
) -> Result<NativeModuleInferenceResult> {
    let parameter_versions: BTreeMap<String, u64> = setup
        .parameters
        .into_iter()
        .map(|(name, parameter)| Ok((name, parameter.version()?)))
        .collect::<Result<_>>()?;
    let native_cache_keys = replay
        .trace
        .items
        .iter()
        .map(|item| item.native_cache_key.clone())
        .collect::<Vec<_>>();
    let mut bytes = format!(
        "{}:{:?}:{}:{:?}",
        setup.capture.identity, setup.input_shape, vectorized, parameter_versions
    )
    .into_bytes();
    for key in &native_cache_keys {
        bytes.extend_from_slice(key.as_deref().unwrap_or("").as_bytes());
    }
    let identity = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    Ok(NativeModuleInferenceResult {
        output: replay
            .outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::SessionTraining {
                reason: "native inference missing output".into(),
            })?,
        trace: replay.trace,
        parameter_versions: parameter_versions.clone(),
        native_trace: NativeModuleInferenceTrace {
            identity,
            capture_identity: setup.capture.identity,
            input_shape: setup.input_shape,
            input_dtype: DType::F32,
            parameter_versions,
            vectorized,
            renderer_version: crate::cpu_jit::RENDERER_VERSION,
            native_cache_keys,
        },
    })
}
/// Builds and discards one fresh CPU graph for a one-input static module.
pub fn infer_module_cpu(
    module: &impl ModuleForward,
    input: TensorData,
) -> Result<ModuleInferenceResult> {
    if !module.accepts_input_dtype(input.dtype()) {
        return Err(Error::SessionTraining {
            reason: "module CPU inference input dtype is not accepted by the leading module".into(),
        });
    }
    let parameters = module.trainable_parameters()?;
    let mut graph = Graph::new();
    let node = graph.input_dtype("module_input", input.shape().clone(), input.dtype());
    let output = module.forward(&mut graph, node)?;
    let mut bindings = module.input_bindings(&graph)?;
    bindings.insert("module_input".into(), input);
    let value = CpuBackend.execute(&graph, output, &bindings)?;
    let parameter_versions: BTreeMap<String, u64> = parameters
        .into_iter()
        .map(|(n, p)| Ok((n, p.version()?)))
        .collect::<Result<_>>()?;
    Ok(ModuleInferenceResult {
        output: value,
        trace: graph.trace(output)?,
        parameter_versions,
    })
}

/// Fresh-graph strict native CPU inference. The caller owns the executor and
/// therefore its deterministic compilation cache; unsupported graphs fail
/// before a native item is executed and never fall back to interpretation.
pub fn infer_module_native_cpu(
    module: &impl ModuleForward,
    input: TensorData,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<NativeModuleInferenceResult> {
    let setup = prepare_native_module_inference(module, input)?;
    let replay = executor
        .replay_pruned_native(&setup.capture, &setup.bindings, vectorized)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    finish_native_module_inference(setup, replay, vectorized)
}

/// Fresh-graph strict native CPU inference with explicit local timing and
/// structural planning observations. This is not a benchmark or profiler.
pub fn infer_module_native_cpu_with_report(
    module: &impl ModuleForward,
    input: TensorData,
    executor: &CapturedReplayExecutor,
    vectorized: bool,
) -> Result<ReportedNativeModuleInferenceResult> {
    let graph_capture_start = Instant::now();
    let setup = prepare_native_module_inference(module, input)?;
    let graph_schedule_capture_duration = graph_capture_start.elapsed();
    let execution_plan =
        ExecutionPlanSummary::from_schedule(&setup.scheduled, &[setup.output], false).map_err(
            |error| Error::SessionTraining {
                reason: format!("native execution report summary: {error}"),
            },
        )?;

    let prepare_start = Instant::now();
    let prepared = executor
        .prepare_pruned_native(&setup.capture, &setup.bindings, vectorized)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    let native_prepare_duration = prepare_start.elapsed();
    let zero_pruned_item_count = prepared.zero_pruned_item_count();
    let zero_materialized_item_count = prepared.zero_materialized_item_count();

    let execute_start = Instant::now();
    let replay = executor
        .execute_prepared_pruned_native(&setup.capture, &setup.bindings, &prepared)
        .map_err(|error| Error::SessionTraining {
            reason: error.to_string(),
        })?;
    let native_execute_duration = execute_start.elapsed();
    let inference = finish_native_module_inference(setup, replay, vectorized)?;
    let native_item_count = inference.trace.items.len();
    let cache_items = inference
        .trace
        .items
        .iter()
        .filter(|item| item.native_cache_key.is_some())
        .collect::<Vec<_>>();
    let cache_hit_count = cache_items.iter().filter(|item| item.cache_hit).count();
    let cache_miss_count = cache_items.iter().filter(|item| !item.cache_hit).count();
    let native_cache_keys = inference.native_trace.native_cache_keys.clone();
    let mut bytes = format!(
        "{}:{}:{}:{:?}",
        execution_plan.identity, inference.native_trace.identity, vectorized, native_cache_keys
    )
    .into_bytes();
    bytes.extend_from_slice(&zero_pruned_item_count.to_le_bytes());
    bytes.extend_from_slice(&zero_materialized_item_count.to_le_bytes());
    let identity = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    let report = NativeModuleExecutionReport {
        identity,
        execution_plan,
        capture_identity: inference.native_trace.capture_identity,
        native_trace_identity: inference.native_trace.identity,
        vectorized,
        native_cache_keys,
        graph_schedule_capture_duration,
        native_prepare_duration,
        native_execute_duration,
        native_item_count,
        zero_pruned_item_count,
        zero_materialized_item_count,
        cache_hit_count,
        cache_miss_count,
    };
    Ok(ReportedNativeModuleInferenceResult { inference, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{
        AdaptiveAvgPool2d, Conv2d, Flatten, Linear, Module, ModuleForward, Parameter, ReLU,
        Sequential, StateKind,
    };
    use crate::{Conv2dOptions, NodeId, Scalar};

    fn append_index(graph: &mut Graph, name: &str, shape: impl Into<Shape>) -> (NodeId, NodeId) {
        let shape = shape.into();
        let position = graph.input_dtype(name, [1], DType::I32);
        let expanded = graph
            .reshape(position, vec![1; shape.rank()])
            .and_then(|value| graph.expand(value, shape))
            .unwrap();
        (position, expanded)
    }

    struct DuplicateTraversal {
        first: Parameter,
        second: Parameter,
    }

    impl Module for DuplicateTraversal {
        fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor("weight".into(), &self.first, StateKind::Parameter);
            visitor("weight".into(), &self.second, StateKind::Parameter);
        }
    }

    impl ModuleForward for DuplicateTraversal {
        fn forward(&self, _: &mut Graph, input: NodeId) -> Result<NodeId> {
            Ok(input)
        }
    }

    struct UnsupportedLater;

    impl Module for UnsupportedLater {
        fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
    }

    impl ModuleForward for UnsupportedLater {
        fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
            let supported = graph.relu(input)?;
            // Preserve a genuinely unsupported later schedule boundary now
            // that fixed masked selection is an ordinary native composition.
            graph.argmax(supported, Some(1), false)
        }
    }

    struct SingleParameter(Parameter);

    impl Module for SingleParameter {
        fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor("value".into(), &self.0, StateKind::Buffer);
        }
    }

    fn relu_mlp() -> (Sequential, Parameter) {
        let first = Linear::new_static(2, 2, true, 41).unwrap();
        first
            .weight
            .replace(TensorData::new([2, 2], vec![1., -1., 0.5, 2.]).unwrap())
            .unwrap();
        first
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.5, -1.]).unwrap())
            .unwrap();
        let second = Linear::new_static(2, 1, true, 42).unwrap();
        second
            .weight
            .replace(TensorData::new([1, 2], vec![3., -2.]).unwrap())
            .unwrap();
        second
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let output_weight = second.weight.clone();
        let mut model = Sequential::default();
        model.push(first);
        model.push(ReLU::new());
        model.push(second);
        (model, output_weight)
    }

    #[test]
    fn host_gather_policy_authenticates_only_captured_affine_provenance() {
        let module = Sequential::default();
        let mut graph = Graph::new();
        let table = graph.input_dtype("table", [4, 3], DType::F32);
        let token = graph.input_dtype("token", [], DType::I32);
        let row = graph.reshape(token, [1, 1]).unwrap();
        let indices = graph.expand(row, [1, 3]).unwrap();
        let gathered = graph.gather(table, indices, 0).unwrap();
        let output = graph.square(gathered).unwrap();
        let inference = CapturedInference::from_module_graph(&module, &graph, &[output]).unwrap();
        let captured_token = inference
            .transient_inputs()
            .iter()
            .find(|input| input.name == "token")
            .unwrap();
        let index_producer = inference
            .capture()
            .items
            .iter()
            .find(|item| item.node == indices)
            .unwrap();
        assert!(matches!(
            index_producer.kernel.operation(),
            crate::Operation::Sink
        ));
        assert_eq!(index_producer.ordered_inputs().len(), 1);
        assert_eq!(index_producer.ordered_inputs()[0].desc, captured_token.desc);
        assert!(captured_token.desc.view.is_some());
        assert!(
            inference
                .clone()
                .with_authenticated_host_gathers(&["token"])
                .is_ok()
        );

        // A separately constructed graph may reuse every NodeId, but it is not
        // an input to authorization. Only the already-owned captured items are
        // inspected, so graph-local identity coincidence cannot grant policy.
        let mut foreign = Graph::new();
        let foreign_table = foreign.input_dtype("table", [4, 3], DType::F32);
        let foreign_token = foreign.input_dtype("token", [], DType::I32);
        let changed = foreign.add(foreign_token, foreign_token).unwrap();
        let foreign_indices = foreign.expand(changed, [1, 3]).unwrap();
        let foreign_gathered = foreign.gather(foreign_table, foreign_indices, 0).unwrap();
        let foreign_output = foreign.square(foreign_gathered).unwrap();
        assert_eq!(foreign_gathered, gathered);
        let foreign_capture =
            CapturedInference::from_module_graph(&module, &foreign, &[foreign_output]).unwrap();
        assert!(
            foreign_capture
                .with_authenticated_host_gathers(&["token"])
                .is_err()
        );

        let mut missing_dependency = inference.clone();
        missing_dependency
            .capture
            .items
            .iter_mut()
            .find(|item| item.node == gathered)
            .unwrap()
            .dependencies
            .retain(|dependency| {
                *dependency
                    != inference
                        .capture
                        .items
                        .iter()
                        .find(|item| item.node == indices)
                        .unwrap()
                        .id
            });
        assert!(
            missing_dependency
                .with_authenticated_host_gathers(&["token"])
                .is_err()
        );

        let mut forged_producer = inference;
        forged_producer
            .capture
            .items
            .iter_mut()
            .find(|item| item.node == indices)
            .unwrap()
            .consumers
            .clear();
        assert!(
            forged_producer
                .with_authenticated_host_gathers(&["token"])
                .is_err()
        );
    }

    fn configured_cifar_classifier() -> (Sequential, Parameter) {
        let conv = Conv2d::new_static(3, 2, [1, 1], Conv2dOptions::default(), true, 81).unwrap();
        conv.weight
            .replace(TensorData::new([2, 3, 1, 1], vec![1., -1., 0.5, -0.5, 2., 1.]).unwrap())
            .unwrap();
        conv.bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.25, -0.75]).unwrap())
            .unwrap();
        let linear = Linear::new_static(2, 2, true, 82).unwrap();
        linear
            .weight
            .replace(TensorData::new([2, 2], vec![1., -2., 0.5, 3.]).unwrap())
            .unwrap();
        linear
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([2], vec![0.5, -1.]).unwrap())
            .unwrap();
        let output_weight = linear.weight.clone();
        let mut model = Sequential::default();
        model.push(conv);
        model.push(ReLU::new());
        model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
        model.push(Flatten::new(1));
        model.push(linear);
        (model, output_weight)
    }

    #[test]
    fn captured_inference_owns_only_admitted_module_state_and_payload_identity() {
        let (model, output_weight) = relu_mlp();
        let mut graph = Graph::new();
        let input = graph.input_dtype("module_input", [2, 2], DType::F32);
        let output = model.forward(&mut graph, input).unwrap();
        let inference = CapturedInference::from_module_graph(&model, &graph, &[output]).unwrap();
        assert_eq!(inference.capture().requested, [output.index() as u64]);
        assert_eq!(inference.execution_plan().requested_outputs.len(), 1);
        assert_eq!(inference.resident_bindings().len(), 4);
        assert_eq!(
            inference
                .transient_inputs()
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>(),
            ["module_input"]
        );
        let captured_residents = inference
            .resident_bindings()
            .iter()
            .map(|(name, value)| (name.clone(), value.to_le_bytes().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let capture_identity = inference.capture().identity;
        let deployment_identity = inference.deployment_identity();
        let output_snapshot = output_weight.snapshot().unwrap();
        let output_name = format!(
            "{}_v{}",
            output_snapshot.input_name, output_snapshot.version
        );
        let mut alternate_residents = inference.resident_bindings().clone();
        alternate_residents.insert(output_name, TensorData::new([1, 2], vec![9., 10.]).unwrap());
        assert_ne!(
            captured_inference_identity(inference.capture(), &alternate_residents).unwrap(),
            deployment_identity
        );

        output_weight
            .replace(TensorData::new([1, 2], vec![9., 10.]).unwrap())
            .unwrap();
        assert_eq!(
            inference
                .resident_bindings()
                .iter()
                .map(|(name, value)| (name.clone(), value.to_le_bytes().unwrap()))
                .collect::<BTreeMap<_, _>>(),
            captured_residents
        );

        let mut changed_graph = Graph::new();
        let changed_input = changed_graph.input_dtype("module_input", [2, 2], DType::F32);
        let changed_output = model.forward(&mut changed_graph, changed_input).unwrap();
        let changed =
            CapturedInference::from_module_graph(&model, &changed_graph, &[changed_output])
                .unwrap();
        assert_ne!(changed.capture().identity, capture_identity);
        assert_ne!(changed.deployment_identity(), deployment_identity);
        assert_eq!(changed.execution_plan(), inference.execution_plan());
    }

    #[test]
    fn stateful_capture_authenticates_distinct_full_buffer_pairs_before_publication() {
        let module = Sequential::default();
        let mut graph = Graph::new();
        let state = graph.input_dtype("state", [2], DType::F32);
        let transient = graph.input_dtype("transient", [2], DType::F32);
        let next = graph.add(state, transient).unwrap();
        let public = graph.square(next).unwrap();
        let initial = BTreeMap::from([(
            "state".into(),
            TensorData::new([2], vec![0.0, 1.0]).unwrap(),
        )]);
        let count = graph.node_count();
        let captured = CapturedStatefulInference::from_module_graph(
            &module,
            &graph,
            &[public],
            &[InferenceStateLink::new(state, next)],
            initial.clone(),
        )
        .unwrap();
        assert_eq!(
            captured.capture().requested,
            [public, next].map(|id| id.index() as u64)
        );
        assert_eq!(captured.transient_inputs()[0].name, "transient");
        assert_eq!(
            captured.state_links().collect::<Vec<_>>(),
            [InferenceStateLink::new(state, next)]
        );
        let changed = CapturedStatefulInference::from_module_graph(
            &module,
            &graph,
            &[public],
            &[InferenceStateLink::new(state, next)],
            BTreeMap::from([(
                "state".into(),
                TensorData::new([2], vec![0.0, 2.0]).unwrap(),
            )]),
        )
        .unwrap();
        assert_ne!(
            captured.deployment_identity(),
            changed.deployment_identity()
        );
        assert_eq!(graph.node_count(), count);

        assert!(matches!(
            CapturedStatefulInference::from_module_graph(
                &module,
                &graph,
                &[next],
                &[InferenceStateLink::new(state, next)],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        assert!(matches!(
            CapturedStatefulInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceStateLink::new(state, transient)],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        assert!(matches!(
            CapturedStatefulInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceStateLink::new(state, next)],
                BTreeMap::from([("state".into(), TensorData::scalar(0.0))]),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        assert_eq!(graph.node_count(), count);
    }

    #[test]
    fn append_state_capture_rejects_partial_additive_and_aliased_ownership() {
        let module = Sequential::default();
        let mut graph = Graph::new();
        let state = graph.input_dtype("cache", [2, 3], DType::F32);
        let (position, index) = append_index(&mut graph, "position", [1, 3]);
        let update_source = graph.input_dtype("update_source", [1, 3], DType::F32);
        let updates = graph.relu(update_source).unwrap();
        let output = graph.scatter(state, index, updates, 0).unwrap();
        let public = graph.square(output).unwrap();
        let initial = BTreeMap::from([(
            "cache".into(),
            TensorData::new([2, 3], vec![0.0; 6]).unwrap(),
        )]);
        let capture = CapturedAppendStateInference::from_module_graph(
            &module,
            &graph,
            &[public],
            &[InferenceAppendStateLink::new(
                state, output, position, index, updates, 0,
            )],
            initial.clone(),
        )
        .unwrap();
        assert_eq!(capture.state_links().count(), 1);
        assert_eq!(capture.transient_inputs().len(), 2);
        assert!(
            capture
                .transient_inputs()
                .iter()
                .any(|input| input.node == update_source)
        );
        assert!(
            capture
                .transient_inputs()
                .iter()
                .all(|input| input.node != updates)
        );
        assert!(
            capture
                .transient_inputs()
                .iter()
                .any(|input| input.node == position)
        );
        assert!(
            capture
                .transient_inputs()
                .iter()
                .all(|input| input.node != index)
        );
        let index_owner = capture
            .capture()
            .items
            .iter()
            .find(|item| {
                item.outputs
                    .iter()
                    .any(|output| output.id == index.index() as u64)
            })
            .unwrap();
        let append_output_id = output.index() as u64;
        let append_owner = capture
            .capture()
            .items
            .iter()
            .find(|item| {
                item.outputs
                    .iter()
                    .any(|descriptor| descriptor.id == append_output_id)
            })
            .unwrap();
        assert_eq!(index_owner.consumers, [append_owner.id]);
        assert!(append_owner.dependencies.contains(&index_owner.id));
        let changed = CapturedAppendStateInference::from_module_graph(
            &module,
            &graph,
            &[public],
            &[InferenceAppendStateLink::new(
                state, output, position, index, updates, 0,
            )],
            BTreeMap::from([(
                "cache".into(),
                TensorData::new([2, 3], vec![1.0; 6]).unwrap(),
            )]),
        )
        .unwrap();
        assert_ne!(capture.deployment_identity(), changed.deployment_identity());

        let (partial_position, partial_index) =
            append_index(&mut graph, "partial_position", [1, 1]);
        let partial_source = graph.input_dtype("partial_source", [1, 1], DType::F32);
        let partial_update = graph.relu(partial_source).unwrap();
        let partial = graph
            .scatter(state, partial_index, partial_update, 0)
            .unwrap();
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceAppendStateLink::new(
                    state,
                    partial,
                    partial_position,
                    partial_index,
                    partial_update,
                    0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));

        let raw_index = graph.input_dtype("raw_position_row", [1, 3], DType::I32);
        let raw_output = graph.scatter(state, raw_index, updates, 0).unwrap();
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceAppendStateLink::new(
                    state, raw_output, position, raw_index, updates, 0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));

        let unrelated_position = graph.input_dtype("unrelated_position", [1], DType::I32);
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceAppendStateLink::new(
                    state,
                    output,
                    unrelated_position,
                    index,
                    updates,
                    0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));

        let leaked_index = graph.cast(index, DType::F32).unwrap();
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[public, leaked_index],
                &[InferenceAppendStateLink::new(
                    state, output, position, index, updates, 0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        let additive = graph.scatter_add(state, index, updates, 0).unwrap();
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[public],
                &[InferenceAppendStateLink::new(
                    state, additive, position, index, updates, 0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        let state_alias = graph.reshape(state, [6]).unwrap();
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[state_alias],
                &[InferenceAppendStateLink::new(
                    state, output, position, index, updates, 0,
                )],
                initial.clone(),
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
        assert!(matches!(
            CapturedAppendStateInference::from_module_graph(
                &module,
                &graph,
                &[output],
                &[InferenceAppendStateLink::new(
                    state, output, position, index, updates, 0,
                )],
                initial,
            ),
            Err(CapturedInferenceError::Binding(_))
        ));
    }

    #[test]
    fn captured_inference_authenticates_parameter_nodes_across_name_collisions() {
        let model = SingleParameter(Parameter::new(
            TensorData::new([2], vec![9.0, 10.0]).unwrap(),
            false,
        ));
        let snapshot = model.0.snapshot().unwrap();
        let generated_name = format!("{}_v{}", snapshot.input_name, snapshot.version);
        let mut graph = Graph::new();
        let exposed = graph.input_dtype(generated_name, [2], DType::F32);
        let unused_parameter = model.0.bind(&mut graph).unwrap();
        assert_ne!(exposed, unused_parameter);

        let inference =
            CapturedInference::from_module_graph(&model, &graph, &[exposed, exposed]).unwrap();
        assert!(inference.resident_bindings().is_empty());
        assert_eq!(inference.transient_inputs().len(), 1);
        assert_eq!(inference.transient_inputs()[0].node, exposed);
        assert_eq!(
            inference.transient_inputs()[0].desc.id,
            exposed.index() as u64
        );
        assert_eq!(inference.execution_plan().schedule_item_count, 0);
        assert_eq!(inference.execution_plan().requested_outputs.len(), 2);
    }

    #[test]
    fn explicit_resident_capture_requires_exact_nodes_and_complete_ownership() {
        let mut graph = Graph::new();
        let resident = graph.input_dtype("resident", [2], DType::F32);
        let collision = graph.input_dtype("resident", [2], DType::F32);
        let transient = graph.input_dtype("transient", [2], DType::F32);
        let output = graph.add(resident, transient).unwrap();
        let value = TensorData::new([2], vec![1.0, 2.0]).unwrap();
        let captured = CapturedInference::from_graph_residents(
            &graph,
            &[output],
            BTreeMap::from([("resident".into(), (resident, value.clone()))]),
            &[],
        )
        .unwrap();
        assert_eq!(captured.resident_bindings()["resident"], value);
        assert_eq!(captured.transient_inputs()[0].name, "transient");

        assert!(matches!(
            CapturedInference::from_graph_residents(
                &graph,
                &[output],
                BTreeMap::from([("resident".into(), (collision, value.clone()))]),
                &[],
            ),
            Err(CapturedInferenceError::Binding(reason))
                if reason.contains("node identity mismatch")
        ));
        assert!(matches!(
            CapturedInference::from_graph_residents(
                &graph,
                &[output],
                BTreeMap::from([
                    ("resident".into(), (resident, value.clone())),
                    ("unused".into(), (collision, value)),
                ]),
                &[],
            ),
            Err(CapturedInferenceError::Binding(reason))
                if reason.contains("absent from captured ownership")
        ));
    }

    #[test]
    fn strict_native_configured_cifar_matches_cpu_and_preserves_contracts() {
        let (model, output_weight) = configured_cifar_classifier();
        let input = TensorData::new(
            [2, 3, 2, 2],
            (1..=24).map(|value| value as f32 / 8.).collect(),
        )
        .unwrap();
        let original_state = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let cold =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        let cache_len = executor.compile_cache_len(false);
        let warm =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(cold.inference().output(), cpu.output());
        assert_eq!(cold.inference().output(), warm.inference().output());
        assert_eq!(cold.report().identity, warm.report().identity);
        assert_eq!(cache_len, executor.compile_cache_len(false));
        assert_eq!(warm.report().cache_miss_count, 0);
        let vector =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, true).unwrap();
        assert_eq!(vector.inference().output(), cpu.output());
        assert!(vector.report().vectorized);
        assert_ne!(cold.report().identity, vector.report().identity);
        assert!(executor.compile_cache_len(true) > 0);
        assert!(
            cold.inference()
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        assert!(
            cold.inference()
                .native_trace()
                .parameter_versions
                .keys()
                .eq(["0.bias", "0.weight", "4.bias", "4.weight"])
        );
        assert_eq!(model.state_dict().unwrap(), original_state);

        let wider = TensorData::new([3, 3, 2, 2], vec![0.25; 36]).unwrap();
        let wider_native =
            infer_module_native_cpu(&model, wider.clone(), &executor, false).unwrap();
        assert_eq!(
            wider_native.output(),
            infer_module_cpu(&model, wider).unwrap().output()
        );
        assert_ne!(
            cold.inference().native_trace().identity,
            wider_native.native_trace().identity
        );
        output_weight
            .replace(TensorData::new([2, 2], vec![2., -2., 0.5, 3.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_ne!(
            cold.inference().native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(cold.inference().output(), changed.output());

        let empty = TensorData::new([0, 3, 2, 2], Vec::<f32>::new()).unwrap();
        let empty_cpu = infer_module_cpu(&model, empty.clone()).unwrap();
        let before_empty = executor.compile_cache_len(false);
        let empty_native = infer_module_native_cpu(&model, empty, &executor, false).unwrap();
        assert_eq!(empty_native.output(), empty_cpu.output());
        assert_eq!(empty_native.output().shape().dims(), &[0, 2]);
        assert_eq!(before_empty, executor.compile_cache_len(false));
        assert!(
            empty_native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn strict_native_static_conv_contract_matches_cpu_without_mutation() {
        let model = Conv2d::new_static(3, 2, [3, 3], Conv2dOptions::default(), false, 91).unwrap();
        let before = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();
        let input = TensorData::new([1, 3, 3, 3], vec![1.0f32; 27]).unwrap();
        let expected = infer_module_cpu(&model, input.clone()).unwrap();
        let native = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(native.output(), expected.output());
        assert!(
            native
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        let cached = executor.compile_cache_len(false);
        assert!(cached > 0);
        let replay = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(replay.output(), expected.output());
        assert_eq!(executor.compile_cache_len(false), cached);
        assert_eq!(model.state_dict().unwrap(), before);
    }

    #[test]
    fn inference_is_fresh_deterministic_and_nonmutating() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let before = model.state_dict().unwrap();
        let first = infer_module_cpu(&model, input.clone()).unwrap();
        let second = infer_module_cpu(&model, input.clone()).unwrap();
        assert_eq!(first.output().to_vec_f64(), vec![9., 19.]);
        assert_eq!(first.output(), second.output());
        assert_eq!(first.trace(), second.trace());
        assert_eq!(before, model.state_dict().unwrap());
        assert!(infer_module_cpu(&model, TensorData::new([1, 3], vec![0.; 3]).unwrap()).is_err());
        assert!(
            infer_module_cpu(
                &model,
                TensorData::from_scalars([1, 2], DType::F64, [crate::Scalar::F(0.); 2]).unwrap()
            )
            .is_err()
        );
        let empty =
            infer_module_cpu(&model, TensorData::new([0, 2], Vec::<f32>::new()).unwrap()).unwrap();
        assert_eq!(empty.output().shape().dims(), &[0, 1]);
    }

    #[test]
    fn strict_native_linear_matches_cpu_and_reuses_caller_cache() {
        let model = Linear::new_static(2, 1, true, 1).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let first = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        let cached = executor.compile_cache_len(false);
        let second = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), first.output());
        assert_eq!(first.output(), second.output());
        assert_eq!(first.native_trace(), second.native_trace());
        assert!(
            first
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
        assert_eq!(cached, executor.compile_cache_len(false));
        model
            .weight
            .replace(TensorData::new([1, 2], vec![4., 3.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            first.native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(first.output(), changed.output());
        let vector = infer_module_native_cpu(
            &model,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
            &executor,
            true,
        )
        .unwrap();
        assert_eq!(
            vector.output(),
            infer_module_cpu(
                &model,
                TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
            )
            .unwrap()
            .output()
        );
        assert!(vector.native_trace().vectorized);
        assert!(executor.compile_cache_len(true) > 0);
        let wider = infer_module_native_cpu(
            &model,
            TensorData::new([3, 2], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            &executor,
            false,
        )
        .unwrap();
        assert_ne!(
            changed.native_trace().identity,
            wider.native_trace().identity
        );
        assert_eq!(wider.output().shape().dims(), &[3, 1]);
    }

    #[test]
    fn opt_in_native_execution_report_correlates_with_warm_cache_and_static_plan() {
        let model = Linear::new_static(2, 1, true, 61).unwrap();
        model
            .weight
            .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
            .unwrap();
        model
            .bias
            .as_ref()
            .unwrap()
            .replace(TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
        let executor = CapturedReplayExecutor::default();
        let input = TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap();
        let original_input = input.clone();
        let original_state = model.state_dict().unwrap();

        let cold =
            infer_module_native_cpu_with_report(&model, input.clone(), &executor, false).unwrap();
        let cache_len = executor.compile_cache_len(false);
        let warm = infer_module_native_cpu_with_report(&model, input, &executor, false).unwrap();
        assert_eq!(cold.inference().output(), warm.inference().output());
        assert_eq!(cold.report().identity, warm.report().identity);
        assert_eq!(cold.report().execution_plan, warm.report().execution_plan);
        assert_eq!(cache_len, executor.compile_cache_len(false));
        assert_eq!(
            cold.report().native_cache_keys,
            cold.inference().native_trace().native_cache_keys
        );
        assert_eq!(
            cold.report().cache_hit_count + cold.report().cache_miss_count,
            cold.inference()
                .trace()
                .items
                .iter()
                .filter(|item| item.native_cache_key.is_some())
                .count()
        );
        assert_eq!(warm.report().cache_miss_count, 0);
        assert_eq!(
            warm.report().cache_hit_count,
            warm.inference()
                .trace()
                .items
                .iter()
                .filter(|item| item.native_cache_key.is_some())
                .count()
        );
        assert_eq!(
            cold.report().native_item_count,
            cold.inference().trace().items.len()
        );
        assert_eq!(cold.report().execution_plan.requested_outputs.len(), 1);
        let _ = (
            cold.report().graph_schedule_capture_duration,
            cold.report().native_prepare_duration,
            cold.report().native_execute_duration,
        );
        assert_eq!(
            original_input,
            TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap()
        );
        assert_eq!(
            model.state_dict().unwrap().tensors(),
            original_state.tensors()
        );
    }

    #[test]
    fn strict_native_sequential_matches_cpu() {
        let mut model = Sequential::default();
        model.push(Linear::new_static(2, 2, true, 1).unwrap());
        model.push(Linear::new_static(2, 1, true, 2).unwrap());
        let input = TensorData::new([1, 2], vec![1., -2.]).unwrap();
        let executor = CapturedReplayExecutor::default();
        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let native = infer_module_native_cpu(&model, input, &executor, false).unwrap();
        assert_eq!(cpu.output(), native.output());
        assert!(
            native
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );
    }

    #[test]
    fn strict_native_relu_mlp_matches_cpu_and_preserves_strict_contracts() {
        let (model, output_weight) = relu_mlp();
        let input = TensorData::new([2, 2], vec![1., -2., 3., 4.]).unwrap();
        let before = model.state_dict().unwrap();
        let executor = CapturedReplayExecutor::default();

        let cpu = infer_module_cpu(&model, input.clone()).unwrap();
        let first = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        let scalar_cache = executor.compile_cache_len(false);
        let second = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_eq!(first.output(), cpu.output());
        assert_eq!(first.output(), second.output());
        assert_eq!(first.native_trace(), second.native_trace());
        assert_eq!(scalar_cache, executor.compile_cache_len(false));
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("0.weight")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("0.bias")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("2.weight")
        );
        assert!(
            first
                .native_trace()
                .parameter_versions
                .contains_key("2.bias")
        );
        assert!(
            first
                .trace()
                .items
                .iter()
                .all(|item| item.backend == crate::ItemBackend::NativeJit)
        );

        let vector = infer_module_native_cpu(&model, input.clone(), &executor, true).unwrap();
        assert_eq!(vector.output(), cpu.output());
        assert!(vector.native_trace().vectorized);
        assert!(executor.compile_cache_len(true) > 0);

        let wider = TensorData::new([3, 2], vec![1., -2., 3., 4., -1., 2.]).unwrap();
        let wider_cpu = infer_module_cpu(&model, wider.clone()).unwrap();
        let wider_native = infer_module_native_cpu(&model, wider, &executor, false).unwrap();
        assert_eq!(wider_native.output(), wider_cpu.output());
        assert_ne!(
            first.native_trace().identity,
            wider_native.native_trace().identity
        );

        output_weight
            .replace(TensorData::new([1, 2], vec![2., 1.]).unwrap())
            .unwrap();
        let changed = infer_module_native_cpu(&model, input.clone(), &executor, false).unwrap();
        assert_ne!(
            first.native_trace().identity,
            changed.native_trace().identity
        );
        assert_ne!(first.output(), changed.output());
        assert_eq!(
            model.state_dict().unwrap().tensors().len(),
            before.tensors().len()
        );

        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let empty_cpu = infer_module_cpu(&model, empty.clone()).unwrap();
        let before_empty_cache = executor.compile_cache_len(false);
        let empty_native = infer_module_native_cpu(&model, empty, &executor, false).unwrap();
        assert_eq!(empty_native.output(), empty_cpu.output());
        assert_eq!(empty_native.output().shape().dims(), &[0, 1]);
        assert_eq!(before_empty_cache, executor.compile_cache_len(false));
        assert!(
            empty_native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );

        assert!(
            infer_module_native_cpu(
                &model,
                TensorData::from_scalars([1, 2], DType::F64, [Scalar::F(0.); 2]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
        assert!(
            infer_module_native_cpu(
                &model,
                TensorData::new([1, 3], vec![0.; 3]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn strict_native_module_rejects_later_unsupported_before_execution() {
        let executor = CapturedReplayExecutor::default();
        let before = executor.compile_cache_len(false);
        assert!(
            infer_module_native_cpu(
                &UnsupportedLater,
                TensorData::new([1, 2], vec![1., -1.]).unwrap(),
                &executor,
                false,
            )
            .is_err()
        );
        assert_eq!(before, executor.compile_cache_len(false));
    }

    #[test]
    fn strict_native_empty_modules_prune_dead_pure_work_without_native_cache_keys() {
        let linear = Linear::new_static(2, 1, true, 17).unwrap();
        let linear_executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let cpu = infer_module_cpu(&linear, empty.clone()).unwrap();
        let native = infer_module_native_cpu(&linear, empty, &linear_executor, false).unwrap();
        assert_eq!(native.output(), cpu.output());
        assert_eq!(native.output().shape().dims(), &[0, 1]);
        assert_eq!(linear_executor.compile_cache_len(false), 0);
        assert!(
            native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );

        let mut sequential = Sequential::default();
        sequential.push(Linear::new_static(2, 2, true, 18).unwrap());
        sequential.push(Linear::new_static(2, 1, true, 19).unwrap());
        let sequential_executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let cpu = infer_module_cpu(&sequential, empty.clone()).unwrap();
        let native =
            infer_module_native_cpu(&sequential, empty, &sequential_executor, false).unwrap();
        assert_eq!(native.output(), cpu.output());
        assert_eq!(native.output().shape().dims(), &[0, 1]);
        assert_eq!(sequential_executor.compile_cache_len(false), 0);
        assert!(
            native
                .native_trace()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn opt_in_report_keeps_empty_pruning_and_strict_preflight_honest() {
        let linear = Linear::new_static(2, 1, true, 62).unwrap();
        let executor = CapturedReplayExecutor::default();
        let empty = TensorData::new([0, 2], Vec::<f32>::new()).unwrap();
        let report = infer_module_native_cpu_with_report(&linear, empty, &executor, false).unwrap();
        assert_eq!(report.inference().output().shape().dims(), &[0, 1]);
        assert!(
            report
                .report()
                .native_cache_keys
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(report.report().cache_hit_count, 0);
        assert_eq!(report.report().cache_miss_count, 0);
        assert!(report.report().zero_materialized_item_count > 0);
        assert_eq!(executor.compile_cache_len(false), 0);

        let mut sequential = Sequential::default();
        sequential.push(Linear::new_static(2, 2, true, 63).unwrap());
        sequential.push(Linear::new_static(2, 1, true, 64).unwrap());
        let sequential_executor = CapturedReplayExecutor::default();
        let report = infer_module_native_cpu_with_report(
            &sequential,
            TensorData::new([0, 2], Vec::<f32>::new()).unwrap(),
            &sequential_executor,
            false,
        )
        .unwrap();
        assert_eq!(report.inference().output().shape().dims(), &[0, 1]);
        assert!(report.report().zero_pruned_item_count > 0);
        assert_eq!(sequential_executor.compile_cache_len(false), 0);

        let unsupported_executor = CapturedReplayExecutor::default();
        assert!(
            infer_module_native_cpu_with_report(
                &UnsupportedLater,
                TensorData::new([1, 2], vec![1., -1.]).unwrap(),
                &unsupported_executor,
                false,
            )
            .is_err()
        );
        assert_eq!(unsupported_executor.compile_cache_len(false), 0);
    }

    #[test]
    fn inference_rejects_poisoned_or_duplicate_modules_before_execution() {
        let poisoned = Linear::new_static(2, 1, true, 1).unwrap();
        let before = poisoned.bias.as_ref().unwrap().snapshot().unwrap();
        poisoned.weight.poison_for_test();
        assert!(matches!(
            infer_module_cpu(&poisoned, TensorData::new([1, 2], vec![0., 1.]).unwrap()),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert_eq!(
            poisoned.bias.as_ref().unwrap().snapshot().unwrap().data,
            before.data
        );
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            infer_module_native_cpu(
                &poisoned,
                TensorData::new([1, 2], vec![0., 1.]).unwrap(),
                &executor,
                false,
            ),
            Err(Error::ParameterLockPoisoned { .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);

        let duplicate = DuplicateTraversal {
            first: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(1.)]).unwrap(),
                true,
            ),
            second: Parameter::new(
                TensorData::from_scalars([1], DType::F32, [Scalar::F(2.)]).unwrap(),
                true,
            ),
        };
        let before = (
            duplicate.first.snapshot().unwrap(),
            duplicate.second.snapshot().unwrap(),
        );
        assert!(matches!(
            infer_module_cpu(&duplicate, TensorData::new([1, 1], vec![1.]).unwrap()),
            Err(Error::Serialization { .. })
        ));
        assert_eq!(duplicate.first.snapshot().unwrap().data, before.0.data);
        assert_eq!(duplicate.second.snapshot().unwrap().data, before.1.data);
        let executor = CapturedReplayExecutor::default();
        assert!(matches!(
            infer_module_native_cpu(
                &duplicate,
                TensorData::new([1, 1], vec![1.]).unwrap(),
                &executor,
                false,
            ),
            Err(Error::Serialization { .. })
        ));
        assert_eq!(executor.compile_cache_len(false), 0);
    }
}
