//! Deterministic module traversal and state loading.

use super::{Parameter, ParameterRestore, ParameterSnapshot, restore_parameters};
use crate::{Error, Graph, Result, TensorData};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub enum StateKind {
    Parameter,
    Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastPolicy {
    Exact,
    Allow,
}

/// Explicit execution mode. It is passed to stateful normalization forwards;
/// RustGrad deliberately has no process-global training flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Training,
    Eval,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub missing_keys: Vec<String>,
    pub unexpected_keys: Vec<String>,
    pub shape_mismatches: Vec<String>,
    pub dtype_mismatches: Vec<String>,
    pub loaded_keys: Vec<String>,
}
impl LoadReport {
    pub fn is_clean(&self) -> bool {
        self.missing_keys.is_empty()
            && self.unexpected_keys.is_empty()
            && self.shape_mismatches.is_empty()
            && self.dtype_mismatches.is_empty()
    }
}

/// A deterministic state map that converts directly to RustGrad safetensors maps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StateDict {
    tensors: BTreeMap<String, TensorData>,
}
impl StateDict {
    pub fn tensors(&self) -> &BTreeMap<String, TensorData> {
        &self.tensors
    }
    pub fn into_tensors(self) -> BTreeMap<String, TensorData> {
        self.tensors
    }
    pub fn insert(&mut self, name: impl Into<String>, value: TensorData) {
        self.tensors.insert(name.into(), value);
    }
}
impl From<BTreeMap<String, TensorData>> for StateDict {
    fn from(tensors: BTreeMap<String, TensorData>) -> Self {
        Self { tensors }
    }
}
impl From<StateDict> for BTreeMap<String, TensorData> {
    fn from(value: StateDict) -> Self {
        value.tensors
    }
}

/// Rust-native explicit state traversal. Implementors call `visit` for fields,
/// nested modules, vectors, and options in their declared deterministic order.
pub trait Module {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind));
    fn state_dict(&self) -> Result<StateDict> {
        let mut tensors = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |name, parameter, _| {
            if seen.insert(parameter.identity()) {
                match parameter.snapshot() {
                    Ok(snapshot) => {
                        tensors.insert(name, snapshot.data);
                    }
                    Err(err) => error = Some(err),
                }
            }
        });
        match error {
            Some(err) => Err(err),
            None => Ok(StateDict { tensors }),
        }
    }
    fn input_bindings(&self, graph: &Graph) -> Result<HashMap<String, TensorData>> {
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |_, parameter, _| match parameter.snapshot() {
            Ok(snapshot) => {
                seen.insert(snapshot.identity);
            }
            Err(err) => error = Some(err),
        });
        match error {
            Some(err) => Err(err),
            None => Ok(graph.parameter_bindings_for(&seen)),
        }
    }
    fn load_state_dict(
        &self,
        state: &StateDict,
        strict: bool,
        cast: CastPolicy,
    ) -> Result<LoadReport> {
        let mut entries = BTreeMap::<String, (Parameter, ParameterSnapshot)>::new();
        let mut seen = BTreeSet::new();
        let mut error = None;
        self.visit("", &mut |name, parameter, _| {
            if seen.insert(parameter.identity()) {
                match parameter.snapshot() {
                    Ok(snapshot) => {
                        entries.insert(name, (parameter.clone(), snapshot));
                    }
                    Err(err) => error = Some(err),
                }
            }
        });
        if let Some(err) = error {
            return Err(err);
        }
        let mut report = LoadReport::default();
        let mut restores = Vec::new();
        let mut loaded_keys = Vec::new();
        for (name, (parameter, snapshot)) in &entries {
            let Some(value) = state.tensors.get(name) else {
                report.missing_keys.push(name.clone());
                continue;
            };
            // tinygrad's loader admits only this one shape relaxation: a
            // scalar and a rank-one singleton carry the same single storage
            // lane, so it reshapes the incoming value to the parameter shape
            // before replacement.  Keep it in this preflight phase and clone
            // raw storage so narrow payloads, NaNs, and signed zero survive.
            let value = if value.shape() != &snapshot.shape {
                if singleton_scalar_rank_one_pair(value, &snapshot) {
                    TensorData::from_storage(snapshot.shape.clone(), value.storage().clone())?
                } else {
                    report.shape_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value.clone()
            };
            let value = if value.dtype() != snapshot.dtype {
                if cast == CastPolicy::Allow {
                    value.cast(snapshot.dtype)
                } else {
                    report.dtype_mismatches.push(name.clone());
                    continue;
                }
            } else {
                value
            };
            restores.push(ParameterRestore {
                parameter: parameter.clone(),
                data: value,
                expected_version: snapshot.version,
                restored_version: snapshot.version.wrapping_add(1),
            });
            loaded_keys.push(name.clone());
        }
        report.unexpected_keys = state
            .tensors
            .keys()
            .filter(|name| !entries.contains_key(*name))
            .cloned()
            .collect();
        if strict && !report.is_clean() {
            return Err(Error::Serialization {
                reason: format!(
                    "state_dict mismatch: missing={:?}, unexpected={:?}, shape={:?}, dtype={:?}",
                    report.missing_keys,
                    report.unexpected_keys,
                    report.shape_mismatches,
                    report.dtype_mismatches
                ),
            });
        }
        restore_parameters(restores)?;
        report.loaded_keys = loaded_keys;
        Ok(report)
    }
}

/// The only load-time shape adaptation accepted by tinygrad state loading.
/// Both descriptors have exactly one element, so rebuilding the descriptor
/// from cloned storage is a checked descriptor change rather than a broadcast
/// or a value conversion.
fn singleton_scalar_rank_one_pair(value: &TensorData, snapshot: &ParameterSnapshot) -> bool {
    (value.shape().rank() == 0
        && snapshot.shape.rank() == 1
        && snapshot.shape.dims() == [1])
        || (value.shape().rank() == 1
            && value.shape().dims() == [1]
            && snapshot.shape.rank() == 0)
}

pub(super) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.into()
    } else {
        format!("{prefix}.{name}")
    }
}
