use super::module_adamw_checkpoint::{
    DecodedModuleAdamWCheckpoint, ModuleCheckpointState, ModuleCheckpointStateKind,
    ModuleCheckpointVisit,
};
use super::{CompiledAdamWConfig, checked_bytes, training, validate_user_name};
use crate::nn::{
    Parameter, ParameterRestore, ParameterSnapshot, StateKind, next_version, restore_parameters,
};
use crate::{DType, Graph, LoadReport, Module, NodeId, ParameterId, Result, TensorData};
use std::collections::{BTreeMap, BTreeSet};

/// Detached initial value for one compiled training parameter.
///
/// Construction does not create an [`crate::nn::Parameter`] or retain a live
/// module handle. The value is consumed by compilation and subsequently owned
/// only by the compiled session's [`EffectRuntime`].
#[derive(Clone, Debug)]
pub struct TrainingParameterInit {
    pub(super) name: String,
    pub(super) value: TensorData,
}

impl TrainingParameterInit {
    pub fn new(name: impl Into<String>, value: TensorData) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "parameter")?;
        if value.dtype() != DType::F32 {
            return Err(training("compiled training parameters must be F32"));
        }
        checked_bytes(&value)?;
        Ok(Self { name, value })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &TensorData {
        &self.value
    }
}

#[derive(Clone, Debug)]
struct ModuleParameterEntry {
    identity: ParameterId,
    name: String,
    value: TensorData,
    source_trainable: bool,
    trainable: bool,
    policy_frozen: bool,
}

/// Frozen snapshot of one module's parameter topology for compilation.
///
/// Trainable identities become recurrent optimizer-owned inputs. Frozen
/// parameters and buffers become immutable capture constants. Tied traversal
/// entries keep one identity and therefore resolve to exactly one graph node.
#[derive(Clone, Debug)]
pub(super) struct ModuleParameterPlan {
    entries: Vec<ModuleParameterEntry>,
    noncanonical_names: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SealedModuleVisit {
    name: String,
    identity: ParameterId,
    kind: StateKind,
    trainable: bool,
}

#[derive(Clone, Debug)]
pub(super) struct SealedModuleState {
    pub(super) name: String,
    parameter: Parameter,
    pub(super) snapshot: ParameterSnapshot,
    kind: StateKind,
    source_trainable: bool,
    trainable: bool,
    publication_value: Option<TensorData>,
}

/// Complete host-module state retained while an owned compiled session runs.
///
/// The seal is deliberately private: it is meaningful only together with the
/// exact module value retained by its compiled module plan or prepared session.
#[derive(Clone, Debug)]
pub(super) struct CompiledModuleSeal {
    visits: Vec<SealedModuleVisit>,
    pub(super) states: BTreeMap<ParameterId, SealedModuleState>,
    frozen_parameters: BTreeSet<String>,
}

impl CompiledModuleSeal {
    pub(super) fn capture(
        module: &(impl Module + ?Sized),
        frozen_parameters: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut names = BTreeSet::new();
        let mut visits = Vec::new();
        let mut states = BTreeMap::<ParameterId, SealedModuleState>::new();
        let mut error = None;
        module.visit("", &mut |name, parameter, kind| {
            if error.is_some() {
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(training("compiled module state names repeat"));
                return;
            }
            let snapshot = match parameter.snapshot() {
                Ok(snapshot) => snapshot,
                Err(err) => {
                    error = Some(err);
                    return;
                }
            };
            let trainable = snapshot.trainable && matches!(kind, StateKind::Parameter);
            visits.push(SealedModuleVisit {
                name: name.clone(),
                identity: snapshot.identity,
                kind,
                trainable,
            });
            if let Some(first) = states.get(&snapshot.identity) {
                let same_bytes = first.snapshot.data.to_le_bytes().and_then(|expected| {
                    snapshot.data.to_le_bytes().map(|actual| expected == actual)
                });
                match same_bytes {
                    Ok(true)
                        if first.kind == kind
                            && first.snapshot.trainable == snapshot.trainable
                            && first.snapshot.shape == snapshot.shape
                            && first.snapshot.dtype == snapshot.dtype
                            && first.snapshot.version == snapshot.version => {}
                    Ok(_) => {
                        error = Some(training(
                            "tied compiled module state has inconsistent kind or snapshot",
                        ));
                    }
                    Err(err) => error = Some(err),
                }
                return;
            }
            states.insert(
                snapshot.identity,
                SealedModuleState {
                    name,
                    parameter: parameter.clone(),
                    kind,
                    source_trainable: snapshot.trainable,
                    snapshot,
                    trainable,
                    publication_value: None,
                },
            );
        });
        match error {
            Some(error) => Err(error),
            None => {
                let mut seal = Self {
                    visits,
                    states,
                    frozen_parameters: frozen_parameters.clone(),
                };
                seal.apply_frozen_parameters()?;
                for state in seal.states.values().filter(|state| state.trainable) {
                    next_version(state.snapshot.version)?;
                }
                Ok(seal)
            }
        }
    }

    fn apply_frozen_parameters(&mut self) -> Result<()> {
        let noncanonical_names = self
            .visits
            .iter()
            .filter(|visit| self.states[&visit.identity].name != visit.name)
            .map(|visit| visit.name.clone())
            .collect::<BTreeSet<_>>();
        for name in &self.frozen_parameters {
            if noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW frozen parameter is a noncanonical tied alias",
                ));
            }
            match self.states.values_mut().find(|state| state.name == *name) {
                Some(state) if state.trainable => state.trainable = false,
                Some(_) => {
                    return Err(training(
                        "compiled AdamW frozen parameter is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training("compiled AdamW frozen parameter name is unknown"));
                }
            }
        }
        if self.states.values().all(|state| !state.trainable) {
            return Err(training(
                "compiled module needs at least one effectively trainable parameter",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_unchanged(&self, module: &(impl Module + ?Sized)) -> Result<()> {
        let current = Self::capture(module, &self.frozen_parameters)?;
        if current.visits != self.visits || current.states.keys().ne(self.states.keys()) {
            return Err(training("owned compiled module topology changed"));
        }
        for (identity, expected) in &self.states {
            let actual = &current.states[identity];
            if actual.name != expected.name
                || actual.kind != expected.kind
                || actual.trainable != expected.trainable
                || actual.snapshot.shape != expected.snapshot.shape
                || actual.snapshot.dtype != expected.snapshot.dtype
                || actual.snapshot.version != expected.snapshot.version
                || actual.snapshot.input_name != expected.snapshot.input_name
                || actual.snapshot.data.to_le_bytes()? != expected.snapshot.data.to_le_bytes()?
            {
                return Err(training("owned compiled module state changed while sealed"));
            }
        }
        Ok(())
    }

    pub(super) fn checkpoint_inventory(
        &self,
    ) -> (Vec<ModuleCheckpointState>, Vec<ModuleCheckpointVisit>) {
        let mut states = Vec::with_capacity(self.states.len());
        let mut visits = Vec::with_capacity(self.visits.len());
        let mut seen = BTreeSet::new();
        for visit in &self.visits {
            let state = &self.states[&visit.identity];
            visits.push(ModuleCheckpointVisit {
                name: visit.name.clone(),
                canonical_name: state.name.clone(),
            });
            if seen.insert(visit.identity) {
                states.push(ModuleCheckpointState {
                    name: state.name.clone(),
                    kind: match state.kind {
                        StateKind::Parameter => ModuleCheckpointStateKind::Parameter,
                        StateKind::Buffer => ModuleCheckpointStateKind::Buffer,
                    },
                    source_trainable: state.source_trainable,
                    policy_frozen: self.frozen_parameters.contains(&state.name),
                    value: (!state.trainable).then(|| {
                        state
                            .publication_value
                            .clone()
                            .unwrap_or_else(|| state.snapshot.data.clone())
                    }),
                });
            }
        }
        (states, visits)
    }

    pub(super) fn apply_module_checkpoint(
        &mut self,
        checkpoint: &DecodedModuleAdamWCheckpoint,
    ) -> Result<BTreeMap<String, TensorData>> {
        let (current_states, current_visits) = self.checkpoint_inventory();
        if current_visits != checkpoint.visits || current_states.len() != checkpoint.states.len() {
            return Err(training("compiled module checkpoint topology mismatch"));
        }
        let optimizer_parameters = &checkpoint.optimizer.decoded().parameters;
        let mut immutable_values = BTreeMap::new();
        for (current, saved) in current_states.iter().zip(&checkpoint.states) {
            if current.name != saved.name
                || current.kind != saved.kind
                || current.source_trainable != saved.source_trainable
                || current.policy_frozen != saved.policy_frozen
                || current.trainable() != saved.trainable()
            {
                return Err(training(
                    "compiled module checkpoint state topology mismatch",
                ));
            }
            let saved_value = match &saved.value {
                Some(value) => value,
                None => &optimizer_parameters[&saved.name],
            };
            let state = self
                .states
                .values_mut()
                .find(|state| state.name == saved.name)
                .ok_or_else(|| training("compiled module checkpoint state is absent"))?;
            if saved_value.shape() != &state.snapshot.shape
                || saved_value.dtype() != state.snapshot.dtype
            {
                return Err(training(
                    "compiled module checkpoint state descriptor mismatch",
                ));
            }
            checked_bytes(saved_value)?;
            if let Some(value) = &saved.value {
                next_version(state.snapshot.version)?;
                state.publication_value = Some(value.clone());
                immutable_values.insert(saved.name.clone(), value.clone());
            }
        }
        Ok(immutable_values)
    }

    pub(super) fn parameter_plan(
        &self,
        module: &(impl Module + ?Sized),
    ) -> Result<ModuleParameterPlan> {
        let immutable_values = self
            .checkpoint_inventory()
            .0
            .into_iter()
            .filter_map(|state| state.value.map(|value| (state.name, value)))
            .collect::<BTreeMap<_, _>>();
        ModuleParameterPlan::new(module, &self.frozen_parameters)?
            .with_immutable_values(&immutable_values)
    }

    pub(super) fn publish(
        &self,
        module: &(impl Module + ?Sized),
        parameters: &BTreeMap<String, TensorData>,
    ) -> Result<LoadReport> {
        self.validate_unchanged(module)?;
        let trainable = self
            .states
            .values()
            .filter(|state| state.trainable)
            .map(|state| (state.name.as_str(), state))
            .collect::<BTreeMap<_, _>>();
        if parameters.len() != trainable.len()
            || parameters
                .keys()
                .map(String::as_str)
                .ne(trainable.keys().copied())
        {
            return Err(training(
                "owned compiled module trainable parameter names changed",
            ));
        }

        let mut restores = Vec::with_capacity(self.states.len());
        let mut loaded_keys = Vec::with_capacity(trainable.len());
        for state in self.states.values() {
            let (data, restored_version) = if state.trainable {
                let value = &parameters[&state.name];
                if value.shape() != &state.snapshot.shape || value.dtype() != state.snapshot.dtype {
                    return Err(training(
                        "owned compiled module trainable parameter descriptor changed",
                    ));
                }
                loaded_keys.push(state.name.clone());
                (value.clone(), next_version(state.snapshot.version)?)
            } else {
                match &state.publication_value {
                    Some(value) => (value.clone(), next_version(state.snapshot.version)?),
                    None => (state.snapshot.data.clone(), state.snapshot.version),
                }
            };
            restores.push(ParameterRestore {
                parameter: state.parameter.clone(),
                data,
                expected_version: state.snapshot.version,
                restored_version,
            });
        }
        restore_parameters(restores)?;
        loaded_keys.sort();
        Ok(LoadReport {
            loaded_keys,
            ..LoadReport::default()
        })
    }
}

impl ModuleParameterPlan {
    pub(super) fn new(
        module: &(impl Module + ?Sized),
        frozen_parameters: &BTreeSet<String>,
    ) -> Result<Self> {
        let mut entries = Vec::<ModuleParameterEntry>::new();
        let mut identities = BTreeMap::<ParameterId, usize>::new();
        let mut names = BTreeSet::new();
        let mut noncanonical_names = BTreeSet::new();
        let mut error = None;
        module.visit("", &mut |name, parameter, kind| {
            if error.is_some() {
                return;
            }
            if !names.insert(name.clone()) {
                error = Some(training("compiled module state names repeat"));
                return;
            }
            let identity = parameter.id();
            if let Some(&index) = identities.get(&identity) {
                let is_parameter = matches!(kind, StateKind::Parameter);
                if entries[index].source_trainable != (parameter.is_trainable() && is_parameter) {
                    error = Some(training(
                        "tied compiled module state has inconsistent trainability",
                    ));
                } else {
                    noncanonical_names.insert(name);
                }
                return;
            }
            match parameter.snapshot() {
                Ok(snapshot) => {
                    let trainable = snapshot.trainable && matches!(kind, StateKind::Parameter);
                    identities.insert(identity, entries.len());
                    entries.push(ModuleParameterEntry {
                        identity,
                        name,
                        value: snapshot.data,
                        source_trainable: trainable,
                        trainable,
                        policy_frozen: false,
                    });
                }
                Err(err) => error = Some(err),
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        for name in frozen_parameters {
            if noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW frozen parameter is a noncanonical tied alias",
                ));
            }
            match entries.iter_mut().find(|entry| entry.name == *name) {
                Some(entry) if entry.trainable => {
                    entry.trainable = false;
                    entry.policy_frozen = true;
                }
                Some(_) => {
                    return Err(training(
                        "compiled AdamW frozen parameter is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training("compiled AdamW frozen parameter name is unknown"));
                }
            }
        }
        if entries.iter().all(|entry| !entry.trainable) {
            return Err(training(
                "compiled module needs at least one effectively trainable parameter",
            ));
        }
        Ok(Self {
            entries,
            noncanonical_names,
        })
    }

    pub(super) fn validate_weight_decay_exclusions(
        &self,
        config: &CompiledAdamWConfig,
    ) -> Result<()> {
        for name in &config.weight_decay_exclusions {
            if self.noncanonical_names.contains(name) {
                return Err(training(
                    "compiled AdamW weight-decay exclusion is a noncanonical tied alias",
                ));
            }
            match self
                .entries
                .iter()
                .find(|entry| entry.name == name.as_str())
            {
                Some(entry) if entry.trainable => {}
                Some(_) => {
                    return Err(training(
                        "compiled AdamW weight-decay exclusion is not a trainable parameter",
                    ));
                }
                None => {
                    return Err(training(
                        "compiled AdamW weight-decay exclusion name is unknown",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn with_immutable_values(
        mut self,
        values: &BTreeMap<String, TensorData>,
    ) -> Result<Self> {
        let expected = self
            .entries
            .iter()
            .filter(|entry| !entry.trainable)
            .map(|entry| entry.name.as_str())
            .collect::<BTreeSet<_>>();
        if expected != values.keys().map(String::as_str).collect::<BTreeSet<_>>() {
            return Err(training(
                "compiled module checkpoint immutable inventory mismatch",
            ));
        }
        for entry in self.entries.iter_mut().filter(|entry| !entry.trainable) {
            let value = &values[&entry.name];
            if value.shape() != entry.value.shape() || value.dtype() != entry.value.dtype() {
                return Err(training(
                    "compiled module checkpoint immutable descriptor mismatch",
                ));
            }
            entry.value = value.clone();
        }
        Ok(self)
    }

    pub(super) fn initial_parameters(&self) -> Result<Vec<TrainingParameterInit>> {
        self.entries
            .iter()
            .filter(|entry| entry.trainable)
            .map(|entry| TrainingParameterInit::new(entry.name.clone(), entry.value.clone()))
            .collect()
    }

    pub(super) fn lower<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        self.lower_impl(graph, parameters, None, build)
    }

    pub(super) fn lower_with_frozen_parameter_nodes<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        frozen_parameter_nodes: &mut BTreeSet<NodeId>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        self.lower_impl(graph, parameters, Some(frozen_parameter_nodes), build)
    }

    fn lower_impl<T>(
        &self,
        graph: &mut Graph,
        parameters: &BTreeMap<String, NodeId>,
        mut frozen_parameter_nodes: Option<&mut BTreeSet<NodeId>>,
        build: impl FnOnce(&mut Graph) -> Result<T>,
    ) -> Result<T> {
        let mut overrides = BTreeMap::new();
        for entry in &self.entries {
            let node = if entry.trainable {
                parameters
                    .get(&entry.name)
                    .copied()
                    .ok_or_else(|| training("compiled module parameter set mismatch"))?
            } else {
                let node = graph.constant(entry.value.clone());
                if entry.policy_frozen
                    && let Some(nodes) = frozen_parameter_nodes.as_deref_mut()
                {
                    nodes.insert(node);
                }
                node
            };
            if graph.shape(node)? != entry.value.shape()
                || graph.dtype(node)? != entry.value.dtype()
                || graph.requires_grad(node)? != entry.trainable
            {
                return Err(training("compiled module parameter descriptor mismatch"));
            }
            overrides.insert(
                entry.identity,
                (node, entry.source_trainable, entry.trainable),
            );
        }
        graph.with_parameter_overrides(overrides, build)
    }
}
