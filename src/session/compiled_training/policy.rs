use super::{
    CompiledAdamWContract, CompiledInputBatch, CompiledInputPolicy, checked_descriptor, training,
    validate_token_weight_policy, validate_user_name,
};
use crate::{DType, Result, Shape};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
};

/// Static compilation policy for [`super::CpuCompiledMomentumSgd`].
#[derive(Clone, Debug)]
pub struct CompiledMomentumSgdConfig {
    pub(super) momentum: f32,
    pub(super) inputs: BTreeMap<String, (Shape, DType)>,
}

impl CompiledMomentumSgdConfig {
    /// Creates the source-style momentum rule `v = momentum*v + grad`.
    /// Tinygrad rejects only ordered negative momentum, so NaN and infinity
    /// remain ordinary graph constants rather than receiving an invented
    /// finite-value restriction here.
    pub fn new(momentum: f32) -> Result<Self> {
        if momentum < 0.0 {
            return Err(training(
                "compiled momentum-SGD momentum must be nonnegative",
            ));
        }
        Ok(Self {
            momentum,
            inputs: BTreeMap::new(),
        })
    }

    /// Adds one exact external input descriptor. Names are deterministic and
    /// may not overlap the session's private state/LR namespace.
    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }
}

/// An immutable learning-rate schedule captured by a compiled AdamW program.
///
/// The rate starts at `base` and is multiplied by `gamma` once for each
/// milestone less than or equal to the number of already completed optimizer
/// updates. Milestones are completed-update boundaries:
/// milestone one first changes the second update, independently of microbatch
/// accumulation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledMultiStepLr {
    pub(super) base: f32,
    pub(super) gamma: f32,
    pub(super) milestones: Vec<u64>,
}

impl CompiledMultiStepLr {
    pub fn new(base: f32, gamma: f32, milestones: impl IntoIterator<Item = u64>) -> Result<Self> {
        if !base.is_finite() || base < 0.0 {
            return Err(training(
                "compiled MultiStep learning-rate base must be finite and nonnegative",
            ));
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err(training(
                "compiled MultiStep learning-rate gamma must be finite and nonnegative",
            ));
        }
        let milestones = milestones.into_iter().collect::<Vec<_>>();
        if milestones
            .iter()
            .any(|milestone| *milestone == 0 || *milestone == u64::MAX)
        {
            return Err(training(
                "compiled MultiStep learning-rate milestones must be positive and below u64::MAX",
            ));
        }
        if milestones.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(training(
                "compiled MultiStep learning-rate milestones must be strictly increasing",
            ));
        }
        let mut rate = base;
        for _ in &milestones {
            rate *= gamma;
            if !rate.is_finite() {
                return Err(training(
                    "compiled MultiStep learning-rate values must remain finite",
                ));
            }
        }
        Ok(Self {
            base,
            gamma,
            milestones,
        })
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn milestones(&self) -> &[u64] {
        &self.milestones
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum CompiledLearningRatePolicy {
    External,
    MultiStep(CompiledMultiStepLr),
}

impl CompiledLearningRatePolicy {
    pub(super) fn require_external(&self) -> Result<()> {
        if matches!(self, Self::External) {
            Ok(())
        } else {
            Err(training(
                "compiled AdamW external learning-rate entrypoint requires external policy",
            ))
        }
    }

    pub(super) fn require_scheduled(&self) -> Result<()> {
        if matches!(self, Self::MultiStep(_)) {
            Ok(())
        } else {
            Err(training(
                "compiled AdamW scheduled learning-rate entrypoint requires compiled MultiStep policy",
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CompiledTokenWeightPolicy {
    ExplicitMask(String),
    IgnoreIndex { target_input: String, value: i32 },
}

impl CompiledTokenWeightPolicy {
    pub(super) fn input_name(&self) -> &str {
        match self {
            Self::ExplicitMask(name)
            | Self::IgnoreIndex {
                target_input: name, ..
            } => name,
        }
    }

    pub(super) fn expected_descriptor<'a>(
        &self,
        inputs: &'a BTreeMap<String, (Shape, DType)>,
    ) -> Result<&'a (Shape, DType)> {
        inputs.get(self.input_name()).ok_or_else(|| match self {
            Self::ExplicitMask(_) => {
                training("compiled AdamW token-weight mask must name an existing input")
            }
            Self::IgnoreIndex { .. } => {
                training("compiled AdamW ignore-index target must name an existing input")
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledTrainingWindowTopology {
    pub(super) accumulation_steps: NonZeroU64,
    pub(super) token_weighted: bool,
    pub(super) window_loss_report: bool,
}

impl CompiledTrainingWindowTopology {
    pub(super) fn from_validated_parts(
        accumulation_steps: u64,
        token_weighted: bool,
        window_loss_report: bool,
    ) -> Self {
        Self {
            accumulation_steps: NonZeroU64::new(accumulation_steps)
                .expect("compiled training accumulation steps were validated"),
            token_weighted,
            window_loss_report,
        }
    }

    pub(super) fn from_config(config: &CompiledAdamWConfig) -> Self {
        Self::from_validated_parts(
            config.gradient_accumulation_steps,
            config.token_weight_policy.is_some(),
            config.window_loss_report,
        )
    }

    pub(super) fn from_contract(contract: &CompiledAdamWContract) -> Self {
        Self::from_validated_parts(
            contract.gradient_accumulation_steps,
            contract.token_weight_policy.is_some(),
            contract.window_loss_report,
        )
    }

    pub(super) const fn accumulating(self) -> bool {
        self.accumulation_steps.get() > 1
    }

    pub(super) const fn retains_token_count(self) -> bool {
        self.accumulating() && self.token_weighted
    }

    pub(super) const fn retains_window_numerator(self) -> bool {
        self.window_loss_report
    }
}

/// Static compilation policy for [`super::CpuCompiledAdamW`].
#[derive(Clone, Debug)]
pub struct CompiledAdamWConfig {
    pub(super) beta1: f32,
    pub(super) beta2: f32,
    pub(super) eps: f32,
    pub(super) weight_decay: f32,
    pub(super) gradient_accumulation_steps: u64,
    pub(super) token_weight_policy: Option<CompiledTokenWeightPolicy>,
    pub(super) allow_zero_valid_token_microbatches: bool,
    pub(super) max_gradient_norm: Option<f32>,
    pub(super) clip_report: bool,
    pub(super) window_loss_report: bool,
    pub(super) loss_scale: f32,
    pub(super) frozen_parameters: BTreeSet<String>,
    pub(super) weight_decay_exclusions: BTreeSet<String>,
    pub(super) inputs: BTreeMap<String, (Shape, DType)>,
    pub(super) host_token_inputs: BTreeMap<String, Shape>,
    pub(super) learning_rate: CompiledLearningRatePolicy,
}

impl CompiledAdamWConfig {
    pub fn new(beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Result<Self> {
        if !(0.0..1.0).contains(&beta1)
            || !(0.0..1.0).contains(&beta2)
            || !eps.is_finite()
            || eps <= 0.0
            || !weight_decay.is_finite()
            || weight_decay < 0.0
        {
            return Err(training(
                "compiled AdamW requires beta1/beta2 in [0,1), positive finite epsilon, and finite nonnegative weight decay",
            ));
        }
        Ok(Self {
            beta1,
            beta2,
            eps,
            weight_decay,
            gradient_accumulation_steps: 1,
            token_weight_policy: None,
            allow_zero_valid_token_microbatches: false,
            max_gradient_norm: None,
            clip_report: false,
            window_loss_report: false,
            loss_scale: 1.0,
            frozen_parameters: BTreeSet::new(),
            weight_decay_exclusions: BTreeSet::new(),
            inputs: BTreeMap::new(),
            host_token_inputs: BTreeMap::new(),
            learning_rate: CompiledLearningRatePolicy::External,
        })
    }

    /// Accumulates gradients across exactly `steps` recurrent replays before
    /// committing one averaged AdamW update. Parameters, moments, the
    /// optimizer step, the partial gradient sums, and the accumulation cursor
    /// all remain inside the captured state frontier.
    pub fn with_gradient_accumulation(mut self, steps: u64) -> Result<Self> {
        if steps == 0 {
            return Err(training(
                "compiled AdamW gradient accumulation steps must be positive",
            ));
        }
        if let Some(policy) = &self.token_weight_policy {
            validate_token_weight_policy(&self.inputs, policy, steps)?;
        }
        self.gradient_accumulation_steps = steps;
        Ok(self)
    }

    /// Configures an existing fixed F32 binary mask for compiler-owned token
    /// mean loss and valid-token-weighted gradient accumulation.
    ///
    /// This CPU-first opt-in requires an explicit token-mean objective through
    /// [`super::CompiledAdamWPlan::compile_module_graph`] or its dropout variant. The
    /// compatibility token-mean constructor follows the same lowering. With
    /// accumulation, compilation weights each normalized microbatch gradient
    /// by its valid count and divides by the whole window count immediately
    /// before clipping and AdamW. It adds one recurrent U64 count only for a
    /// multi-microbatch window; mask padding layout remains a batch-level
    /// policy.
    pub fn with_token_weighted_gradient_accumulation(
        mut self,
        mask_input_name: impl Into<String>,
    ) -> Result<Self> {
        let mask_input_name = mask_input_name.into();
        if self.token_weight_policy.is_some() {
            return Err(training(
                "compiled AdamW token-weighted accumulation policy repeats",
            ));
        }
        validate_token_weight_policy(
            &self.inputs,
            &CompiledTokenWeightPolicy::ExplicitMask(mask_input_name.clone()),
            self.gradient_accumulation_steps,
        )?;
        self.token_weight_policy = Some(CompiledTokenWeightPolicy::ExplicitMask(mask_input_name));
        Ok(self)
    }

    /// Derives token-mean weighting from an existing fixed I32 target input.
    /// Target lanes equal to `ignore_index` contribute exact zero loss, token
    /// weight, and gradient; every other lane contributes one. This policy is
    /// mutually exclusive with an explicit F32 token-weight mask. Use
    /// [`super::CompiledAdamWPlan::compile_module_graph_with_ignore_index`] when the
    /// model graph also needs the compiler-owned validity node.
    pub fn with_token_weighted_ignore_index(
        mut self,
        target_input: impl Into<String>,
        ignore_index: i32,
    ) -> Result<Self> {
        let target_input = target_input.into();
        if self.token_weight_policy.is_some() {
            return Err(training(
                "compiled AdamW token-weighted accumulation policy repeats",
            ));
        }
        let policy = CompiledTokenWeightPolicy::IgnoreIndex {
            target_input,
            value: ignore_index,
        };
        validate_token_weight_policy(&self.inputs, &policy, self.gradient_accumulation_steps)?;
        self.token_weight_policy = Some(policy);
        Ok(self)
    }

    /// Allows a fixed-shape token-mean microbatch whose explicit mask or
    /// target-derived ignore-index policy selects no valid tokens. Its public
    /// loss, token weight, and gradient contribution are exact zero inside a
    /// multi-step window while replay and dropout progress advance normally.
    /// A completed window (including every `N=1` replay) or explicitly flushed
    /// window whose total token weight is still zero rejects atomically.
    ///
    /// This opt-in requires a compiler-owned token-weight policy and changes
    /// the compiled capture identity. The default continues to reject empty
    /// token selections before replay.
    pub fn with_zero_valid_token_microbatches(mut self) -> Result<Self> {
        if self.token_weight_policy.is_none() {
            return Err(training(
                "compiled AdamW zero-token microbatches require token-weighted accumulation",
            ));
        }
        self.allow_zero_valid_token_microbatches = true;
        Ok(self)
    }

    /// Clips the complete ordered parameter-gradient set to one global L2
    /// norm inside the compiled graph. With gradient accumulation, clipping is
    /// applied once to the averaged window immediately before AdamW updates;
    /// individual microbatch gradients are never clipped independently.
    /// The squared total is committed at F32 width before its square root so
    /// interpreter and strict-native CPU overflow semantics are identical; that
    /// storage boundary is part of the captured program identity.
    pub fn with_max_gradient_norm(mut self, max_norm: f32) -> Result<Self> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(training(
                "compiled AdamW maximum gradient norm must be positive and finite",
            ));
        }
        self.max_gradient_norm = Some(max_norm);
        Ok(self)
    }

    /// Requests CPU step and partial-flush results to report the complete
    /// pre-clip global gradient norm and the exact scale applied before AdamW.
    /// Accumulation-only steps and empty flushes report no completed window.
    /// Under [`super::CpuNonFinitePolicy::RejectTransition`], only reports belonging
    /// to a committing full window or explicit partial flush participate in
    /// admission. The default remains disabled and adds no graph outputs.
    pub fn with_clip_report(mut self) -> Self {
        self.clip_report = true;
        self
    }

    /// Retains the exact accumulated loss numerator inside the captured AdamW
    /// frontier and reports one aggregate mean only when a full or explicitly
    /// flushed window commits. Ordinary scalar objectives use equal
    /// microbatch weights; compiler-owned token means use the same validated
    /// token counts as gradient accumulation. `zero_grad` discards both
    /// gradients and the pending loss numerator atomically.
    pub fn with_window_loss_report(mut self) -> Self {
        self.window_loss_report = true;
        self
    }

    /// Scales the differentiation root by a fixed finite factor, then
    /// unscales the complete F32 parameter-gradient set before accumulation,
    /// clipping, and AdamW. The public loss remains the original unscaled
    /// scalar. A scale of one is canonical and adds no graph nodes.
    pub fn with_loss_scale(mut self, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(training(
                "compiled AdamW loss scale must be positive and finite",
            ));
        }
        self.loss_scale = scale;
        Ok(self)
    }

    /// Captures an immutable MultiStep learning-rate policy in the program.
    /// Scheduled CPU replay then uses the explicit no-learning-rate methods;
    /// the default remains a caller-supplied scalar on every replay.
    pub fn with_captured_multi_step_lr(mut self, schedule: CompiledMultiStepLr) -> Self {
        self.learning_rate = CompiledLearningRatePolicy::MultiStep(schedule);
        self
    }

    /// Freezes exact canonical module parameter names for this compilation.
    /// The policy is resolved by parameter identity without mutating the
    /// source module. Raw [`super::TrainingParameterInit`] compilation rejects a
    /// nonempty policy because it has no module topology to authenticate.
    pub fn with_frozen_parameters<I, S>(mut self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            validate_user_name(&name, "AdamW frozen parameter")?;
            if !self.frozen_parameters.insert(name) {
                return Err(training("duplicate compiled AdamW frozen parameter name"));
            }
        }
        Ok(self)
    }

    /// Excludes exact canonical trainable parameter names from decoupled
    /// weight decay. Names are accumulated across calls and duplicates are
    /// rejected; module compilation also rejects frozen state, buffers, tied
    /// aliases, and names outside the canonical trainable inventory.
    pub fn with_weight_decay_exclusions<I, S>(mut self, names: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in names {
            let name = name.into();
            validate_user_name(&name, "AdamW weight-decay exclusion")?;
            if !self.weight_decay_exclusions.insert(name) {
                return Err(training(
                    "duplicate compiled AdamW weight-decay exclusion name",
                ));
            }
        }
        Ok(self)
    }

    pub fn with_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        dtype: DType,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, dtype)?;
        if self.inputs.insert(name, (shape, dtype)).is_some() {
            return Err(training("duplicate compiled training input name"));
        }
        Ok(self)
    }

    /// Declares one nonempty fixed-shape rank-two I32 token batch whose exact
    /// raw F32 Gather and first-order ScatterAdd VJP may be authenticated for
    /// status-free Metal replay. Module compilation may instead authenticate
    /// a sole forward Gather when its data is an exact policy-frozen parameter.
    /// The input name is declared atomically, so it collides with
    /// [`Self::with_input`] in either call order.
    pub fn with_host_token_input(
        mut self,
        name: impl Into<String>,
        shape: impl Into<Shape>,
    ) -> Result<Self> {
        let name = name.into();
        validate_user_name(&name, "input")?;
        let shape = shape.into();
        checked_descriptor(&shape, DType::I32)?;
        if shape.rank() != 2 || shape.dims().contains(&0) {
            return Err(training(
                "compiled host token input must be nonempty fixed rank-two I32",
            ));
        }
        if self.inputs.contains_key(&name) || self.host_token_inputs.contains_key(&name) {
            return Err(training("duplicate compiled training input name"));
        }
        self.inputs
            .insert(name.clone(), (shape.clone(), DType::I32));
        self.host_token_inputs.insert(name, shape);
        Ok(self)
    }

    /// Declares the complete fixed external schema of a typed workload batch.
    pub fn with_input_batch<B>(mut self) -> Result<Self>
    where
        B: CompiledInputBatch,
    {
        for spec in B::schema() {
            self = match spec.policy {
                CompiledInputPolicy::External => {
                    self.with_input(spec.name, spec.shape.to_vec(), spec.dtype)?
                }
                CompiledInputPolicy::HostToken => {
                    self.with_host_token_input(spec.name, spec.shape.to_vec())?
                }
            };
        }
        Ok(self)
    }

    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    pub fn eps(&self) -> f32 {
        self.eps
    }

    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Returns canonical frozen parameter names in deterministic sorted order.
    pub fn frozen_parameters(&self) -> impl ExactSizeIterator<Item = &str> {
        self.frozen_parameters.iter().map(String::as_str)
    }

    /// Returns canonical exclusion names in deterministic sorted order.
    pub fn weight_decay_exclusions(&self) -> impl ExactSizeIterator<Item = &str> {
        self.weight_decay_exclusions.iter().map(String::as_str)
    }

    pub fn gradient_accumulation_steps(&self) -> u64 {
        self.gradient_accumulation_steps
    }

    /// Existing F32 input whose valid-token count weights each microbatch.
    /// Returns `None` for target-derived ignore-index weighting.
    pub fn token_weighted_gradient_accumulation_mask(&self) -> Option<&str> {
        match &self.token_weight_policy {
            Some(CompiledTokenWeightPolicy::ExplicitMask(name)) => Some(name),
            _ => None,
        }
    }

    /// I32 target input and sentinel used for compiler-owned token weighting.
    pub fn token_weighted_ignore_index(&self) -> Option<(&str, i32)> {
        match &self.token_weight_policy {
            Some(CompiledTokenWeightPolicy::IgnoreIndex {
                target_input,
                value,
            }) => Some((target_input, *value)),
            _ => None,
        }
    }

    /// Whether zero-valid-token selections may contribute zero inside a
    /// multi-step window. A zero-token completed window still rejects.
    pub fn zero_valid_token_microbatches_enabled(&self) -> bool {
        self.allow_zero_valid_token_microbatches
    }

    pub fn max_gradient_norm(&self) -> Option<f32> {
        self.max_gradient_norm
    }

    pub fn clip_report_enabled(&self) -> bool {
        self.clip_report
    }

    /// Whether completed-window loss aggregation is captured and reported.
    pub fn window_loss_report_enabled(&self) -> bool {
        self.window_loss_report
    }

    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    pub fn captured_multi_step_lr(&self) -> Option<&CompiledMultiStepLr> {
        match &self.learning_rate {
            CompiledLearningRatePolicy::External => None,
            CompiledLearningRatePolicy::MultiStep(schedule) => Some(schedule),
        }
    }

    pub fn inputs(&self) -> impl Iterator<Item = (&str, &Shape, DType)> {
        self.inputs
            .iter()
            .map(|(name, (shape, dtype))| (name.as_str(), shape, *dtype))
    }

    /// Returns authenticated host-token declarations in lexical name order.
    pub fn host_token_inputs(&self) -> impl ExactSizeIterator<Item = (&str, &Shape)> {
        self.host_token_inputs
            .iter()
            .map(|(name, shape)| (name.as_str(), shape))
    }
}
