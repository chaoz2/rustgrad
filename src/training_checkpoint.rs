//! In-process identity-preserving training checkpoints.

use crate::nn::StateDict;
use crate::optim::{LearningRateScheduler, Optimizer};
use crate::{Error, Module, ParameterId, Result, load_safetensors, save_safetensors};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(reason: &str) -> Error {
    Error::Serialization {
        reason: format!("optimizer: {reason}"),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParameterCheckpointStamp {
    identity: ParameterId,
    version: u64,
    trainable: bool,
}

/// Exact in-process checkpoint retaining the original module parameter identities.
#[derive(Clone, Debug)]
pub struct TrainingCheckpoint {
    module_safetensors: Vec<u8>,
    optimizer_state: StateDict,
    scheduler_state: StateDict,
    parameter_stamps: BTreeMap<String, ParameterCheckpointStamp>,
    optimizer_ownership: BTreeMap<String, ParameterId>,
}
impl TrainingCheckpoint {
    pub fn capture(
        module: &(impl Module + ?Sized),
        optimizer: &Optimizer,
        scheduler: &LearningRateScheduler,
    ) -> Result<Self> {
        let (module_state, parameter_stamps) = checkpoint_module_state(module)?;
        let optimizer_ownership = optimizer.checkpoint_ownership();
        validate_optimizer_ownership(&parameter_stamps, &optimizer_ownership)?;
        Ok(Self {
            module_safetensors: save_safetensors(module_state.tensors(), &BTreeMap::new())?,
            optimizer_state: optimizer.state_dict()?,
            scheduler_state: scheduler.state_dict()?,
            parameter_stamps,
            optimizer_ownership,
        })
    }
    pub fn resume(
        &self,
        module: &(impl Module + ?Sized),
        optimizer: &mut Optimizer,
        scheduler: &mut LearningRateScheduler,
    ) -> Result<()> {
        let (raw, metadata) = load_safetensors(&self.module_safetensors)?;
        if !metadata.is_empty() {
            return Err(invalid("module metadata must be empty"));
        }
        let (current, stamps) = checkpoint_module_state(module)?;
        if stamps != self.parameter_stamps {
            return Err(invalid("parameter identity or version mismatch"));
        }
        if current != StateDict::from(raw) {
            return Err(invalid("module value mismatch"));
        }
        let ownership = optimizer.checkpoint_ownership();
        if ownership != self.optimizer_ownership {
            return Err(invalid("optimizer ownership mismatch"));
        }
        validate_optimizer_ownership(&stamps, &ownership)?;
        let next_optimizer = optimizer.restore_candidate(&self.optimizer_state)?;
        let next_scheduler = scheduler.restore_candidate(&self.scheduler_state)?;
        *optimizer = next_optimizer;
        *scheduler = next_scheduler;
        Ok(())
    }
    pub fn module_safetensors(&self) -> &[u8] {
        &self.module_safetensors
    }
    pub fn optimizer_state(&self) -> &StateDict {
        &self.optimizer_state
    }
    pub fn scheduler_state(&self) -> &StateDict {
        &self.scheduler_state
    }
    pub fn parameter_versions(&self) -> BTreeMap<String, u64> {
        self.parameter_stamps
            .iter()
            .map(|(n, s)| (n.clone(), s.version))
            .collect()
    }
}
fn checkpoint_module_state(
    module: &(impl Module + ?Sized),
) -> Result<(StateDict, BTreeMap<String, ParameterCheckpointStamp>)> {
    let mut tensors = BTreeMap::new();
    let mut stamps = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut error = None;
    module.visit("", &mut |name, parameter, _| {
        if seen.insert(parameter.id()) {
            match parameter.snapshot() {
                Ok(s) => {
                    tensors.insert(name.clone(), s.data);
                    stamps.insert(
                        name,
                        ParameterCheckpointStamp {
                            identity: s.identity,
                            version: s.version,
                            trainable: s.trainable,
                        },
                    );
                }
                Err(e) => error = Some(e),
            }
        }
    });
    error.map_or_else(|| Ok((StateDict::from(tensors), stamps)), Err)
}
fn validate_optimizer_ownership(
    stamps: &BTreeMap<String, ParameterCheckpointStamp>,
    ownership: &BTreeMap<String, ParameterId>,
) -> Result<()> {
    for (name, id) in ownership {
        let stamp = stamps
            .get(name)
            .ok_or_else(|| invalid("optimizer parameter is absent from module checkpoint"))?;
        if !stamp.trainable || stamp.identity != *id {
            return Err(invalid("optimizer parameter identity mismatch"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;
    use crate::nn::Linear;
    use crate::optim::{LearningRateScheduler, Optimizer, SgdConfig};

    #[test]
    fn training_checkpoint_rejects_each_mismatched_part_atomically() {
        let mut graph = Graph::new();
        let linear = Linear::new(&mut graph, 1, 1, false, 5).unwrap();
        let optimizer = Optimizer::sgd(
            vec![("weight".into(), linear.weight.clone())],
            SgdConfig::default(),
        )
        .unwrap();
        let scheduler = LearningRateScheduler::multi_step(vec![0], 0.5).unwrap();
        let checkpoint = TrainingCheckpoint::capture(&linear, &optimizer, &scheduler).unwrap();
        let mut target = Optimizer::sgd(
            vec![("weight".into(), linear.weight.clone())],
            SgdConfig::default(),
        )
        .unwrap();
        let mut target_scheduler = LearningRateScheduler::multi_step(vec![0], 0.5).unwrap();
        let before = target.state_dict().unwrap();
        let mut bad = checkpoint.clone();
        let mut tensors = bad.optimizer_state.into_tensors();
        tensors.remove("optimizer.step");
        bad.optimizer_state = StateDict::from(tensors);
        assert!(
            bad.resume(&linear, &mut target, &mut target_scheduler)
                .is_err()
        );
        assert_eq!(target.state_dict().unwrap(), before);
    }
}
