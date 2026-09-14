use super::{
    CompiledAdamWConfig, CompiledDropoutState, CompiledLearningRatePolicy,
    CompiledTokenWeightPolicy, training,
};
use crate::{Result, Shape};
use std::collections::{BTreeMap, BTreeSet};

/// Immutable optimizer and workload policy shared by every compiled AdamW
/// representation. Runtime progress, captures, and prepared resources remain
/// owned by their respective plan or session.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CompiledAdamWContract {
    pub(super) gradient_accumulation_steps: u64,
    pub(super) token_weight_policy: Option<CompiledTokenWeightPolicy>,
    pub(super) allow_zero_valid_token_microbatches: bool,
    pub(super) max_gradient_norm: Option<f32>,
    pub(super) clip_report: bool,
    pub(super) window_loss_report: bool,
    pub(super) loss_scale: f32,
    pub(super) dropout: Option<CompiledDropoutState>,
    pub(super) host_token_inputs: BTreeMap<String, Shape>,
    pub(super) frozen_parameters: BTreeSet<String>,
    pub(super) learning_rate: CompiledLearningRatePolicy,
    pub(super) optimizer: CompiledAdamWPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CompiledAdamWPolicy {
    pub(super) beta1: f32,
    pub(super) beta2: f32,
    pub(super) eps: f32,
    pub(super) weight_decay: f32,
    pub(super) weight_decay_exclusions: BTreeSet<String>,
}

/// Metal-supported projection of the backend-neutral compiled AdamW policy.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct MetalAdamWContract {
    pub(super) gradient_accumulation_steps: u64,
    pub(super) max_gradient_norm: Option<f32>,
    pub(super) loss_scale: f32,
    pub(super) dropout: Option<CompiledDropoutState>,
    pub(super) frozen_parameters: BTreeSet<String>,
}

impl CompiledAdamWContract {
    pub(super) fn from_config(
        config: &CompiledAdamWConfig,
        dropout: Option<CompiledDropoutState>,
    ) -> Self {
        Self {
            gradient_accumulation_steps: config.gradient_accumulation_steps,
            token_weight_policy: config.token_weight_policy.clone(),
            allow_zero_valid_token_microbatches: config.allow_zero_valid_token_microbatches,
            max_gradient_norm: config.max_gradient_norm,
            clip_report: config.clip_report,
            window_loss_report: config.window_loss_report,
            loss_scale: config.loss_scale,
            dropout,
            host_token_inputs: config.host_token_inputs.clone(),
            frozen_parameters: config.frozen_parameters.clone(),
            learning_rate: config.learning_rate.clone(),
            optimizer: CompiledAdamWPolicy::from_config(config),
        }
    }

    pub(super) fn metal(&self) -> Result<MetalAdamWContract> {
        // Preserve the public admission order: reporting, token policy, then LR.
        if self.clip_report {
            return Err(training(
                "compiled AdamW clip reporting is currently CPU-only",
            ));
        }
        if self.window_loss_report {
            return Err(training(
                "compiled AdamW window-loss reporting is currently CPU-only",
            ));
        }
        if self.token_weight_policy.is_some() {
            return Err(training(
                "compiled AdamW token-weighted accumulation is currently CPU-only",
            ));
        }
        if matches!(
            &self.learning_rate,
            CompiledLearningRatePolicy::MultiStep(_)
        ) {
            return Err(training(
                "compiled MultiStep learning-rate policy is currently CPU-only",
            ));
        }
        Ok(MetalAdamWContract {
            gradient_accumulation_steps: self.gradient_accumulation_steps,
            max_gradient_norm: self.max_gradient_norm,
            loss_scale: self.loss_scale,
            dropout: self.dropout,
            frozen_parameters: self.frozen_parameters.clone(),
        })
    }
}

impl CompiledAdamWPolicy {
    fn from_config(config: &CompiledAdamWConfig) -> Self {
        Self {
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
            weight_decay_exclusions: config.weight_decay_exclusions.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn contract_centralizes_full_cpu_policy_and_metal_rejection_order() {
        let schedule = super::super::CompiledMultiStepLr::new(1e-3, 0.5, [2, 5]).unwrap();
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)
            .unwrap()
            .with_input("targets", [2, 3], DType::I32)
            .unwrap()
            .with_host_token_input("tokens", [2, 3])
            .unwrap()
            .with_gradient_accumulation(3)
            .unwrap()
            .with_token_weighted_ignore_index("targets", -100)
            .unwrap()
            .with_zero_valid_token_microbatches()
            .unwrap()
            .with_max_gradient_norm(0.25)
            .unwrap()
            .with_clip_report()
            .with_window_loss_report()
            .with_loss_scale(128.0)
            .unwrap()
            .with_frozen_parameters(["position.weight"])
            .unwrap()
            .with_weight_decay_exclusions(["norm.weight"])
            .unwrap()
            .with_captured_multi_step_lr(schedule.clone());
        let dropout = CompiledDropoutState {
            config: super::super::CompiledDropoutConfig::new(super::super::CompiledDropoutKey([
                7, 11,
            ])),
            blocks_per_replay: 13,
        };
        let contract = CompiledAdamWContract::from_config(&config, Some(dropout));

        assert_eq!(contract.gradient_accumulation_steps, 3);
        assert_eq!(
            contract.token_weight_policy,
            Some(CompiledTokenWeightPolicy::IgnoreIndex {
                target_input: "targets".into(),
                value: -100,
            })
        );
        assert!(contract.allow_zero_valid_token_microbatches);
        assert_eq!(contract.max_gradient_norm, Some(0.25));
        assert!(contract.clip_report);
        assert!(contract.window_loss_report);
        assert_eq!(contract.loss_scale, 128.0);
        assert_eq!(contract.dropout, Some(dropout));
        assert_eq!(contract.host_token_inputs["tokens"].dims(), &[2, 3]);
        assert!(contract.frozen_parameters.contains("position.weight"));
        assert_eq!(
            contract.learning_rate,
            CompiledLearningRatePolicy::MultiStep(schedule)
        );
        assert_eq!(contract.optimizer.beta1, 0.9);
        assert_eq!(contract.optimizer.beta2, 0.999);
        assert_eq!(contract.optimizer.eps, 1e-8);
        assert_eq!(contract.optimizer.weight_decay, 0.01);
        assert!(
            contract
                .optimizer
                .weight_decay_exclusions
                .contains("norm.weight")
        );

        let mut candidate = contract;
        for (expected, clear) in [
            ("clip reporting is currently CPU-only", 0_u8),
            ("window-loss reporting is currently CPU-only", 1),
            ("token-weighted accumulation is currently CPU-only", 2),
            ("MultiStep learning-rate policy is currently CPU-only", 3),
        ] {
            let error = candidate.metal().unwrap_err();
            assert!(error.to_string().contains(expected));
            match clear {
                0 => candidate.clip_report = false,
                1 => candidate.window_loss_report = false,
                2 => {
                    candidate.token_weight_policy = None;
                    candidate.allow_zero_valid_token_microbatches = false;
                }
                3 => candidate.learning_rate = CompiledLearningRatePolicy::External,
                _ => unreachable!(),
            }
        }
        let metal = candidate.metal().unwrap();
        assert_eq!(metal.gradient_accumulation_steps, 3);
        assert_eq!(metal.max_gradient_norm, Some(0.25));
        assert_eq!(metal.loss_scale, 128.0);
        assert_eq!(metal.dropout, Some(dropout));
        assert!(metal.frozen_parameters.contains("position.weight"));

        let default = CompiledAdamWConfig::new(0.0, 0.0, 1e-8, 0.0).unwrap();
        assert_eq!(
            CompiledAdamWContract::from_config(&default, None)
                .metal()
                .unwrap()
                .gradient_accumulation_steps,
            1
        );
    }
}
