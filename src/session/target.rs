//! Typed runtime selection for resource-free session plans.

use crate::CapturedReplayExecutor;
use crate::runtime::metal::{
    MetalDevice, MetalError, MetalPlanOptions, MetalRenderer, MetalScoreboardContext,
};

/// A concrete preparation target for one resource-free session plan.
///
/// Both the prepared session and its error remain plan-specific. Adding a
/// model or backend therefore requires an implementation in the owning module,
/// not another central dispatcher branch or a lowest-common-denominator enum.
pub trait SessionTarget<P> {
    type Session;
    type Error;

    fn prepare(&self, plan: P) -> std::result::Result<Self::Session, Self::Error>;
}

/// CPU admission policy for non-finite compiled-training transitions.
///
/// This is a runtime safety boundary rather than part of the captured program:
/// accepted transitions, capture identities, and checkpoint bytes are identical
/// under both policies.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CpuNonFinitePolicy {
    /// Preserve historical IEEE propagation behavior.
    #[default]
    Propagate,
    /// Reject a non-finite loss or F32 recurrent successor before state commit.
    RejectTransition,
}

/// Graph-free CPU replay target for compiled training plans.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuSessionTarget;

impl CpuSessionTarget {
    pub const fn new() -> Self {
        Self
    }

    /// Returns a configured CPU target without changing this historical unit
    /// target's propagation behavior or source-compatible construction.
    pub const fn with_non_finite_policy(
        self,
        policy: CpuNonFinitePolicy,
    ) -> ConfiguredCpuSessionTarget {
        ConfiguredCpuSessionTarget {
            non_finite_policy: policy,
        }
    }

    /// Prepares a plan implemented for the CPU target.
    pub fn prepare<P>(
        &self,
        plan: P,
    ) -> std::result::Result<<Self as SessionTarget<P>>::Session, <Self as SessionTarget<P>>::Error>
    where
        Self: SessionTarget<P>,
    {
        <Self as SessionTarget<P>>::prepare(self, plan)
    }
}

/// CPU session target carrying an explicit compiled-transition admission policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfiguredCpuSessionTarget {
    non_finite_policy: CpuNonFinitePolicy,
}

impl ConfiguredCpuSessionTarget {
    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.non_finite_policy
    }

    pub fn prepare<P>(
        &self,
        plan: P,
    ) -> std::result::Result<<Self as SessionTarget<P>>::Session, <Self as SessionTarget<P>>::Error>
    where
        Self: SessionTarget<P>,
    {
        <Self as SessionTarget<P>>::prepare(self, plan)
    }
}

/// Strict native-JIT CPU target for compiled training plans.
///
/// Preparation compiles every pure main, flush, and evaluation schedule before
/// returning a mutable session. Replay uses the supplied executor with
/// [`crate::JitFallback::Error`]'s strict admission contract; it never falls
/// back to the captured interpreter.
#[derive(Clone, Copy)]
pub struct NativeCpuSessionTarget<'a> {
    executor: &'a CapturedReplayExecutor,
    vectorized: bool,
    non_finite_policy: CpuNonFinitePolicy,
}

impl<'a> NativeCpuSessionTarget<'a> {
    pub const fn new(executor: &'a CapturedReplayExecutor) -> Self {
        Self {
            executor,
            vectorized: false,
            non_finite_policy: CpuNonFinitePolicy::Propagate,
        }
    }

    pub const fn vectorized(mut self, vectorized: bool) -> Self {
        self.vectorized = vectorized;
        self
    }

    pub const fn executor(&self) -> &'a CapturedReplayExecutor {
        self.executor
    }

    pub const fn is_vectorized(&self) -> bool {
        self.vectorized
    }

    /// Selects CPU-only admission for non-finite compiled-training transitions.
    pub const fn with_non_finite_policy(mut self, policy: CpuNonFinitePolicy) -> Self {
        self.non_finite_policy = policy;
        self
    }

    pub const fn non_finite_policy(&self) -> CpuNonFinitePolicy {
        self.non_finite_policy
    }

    pub fn prepare<P>(
        &self,
        plan: P,
    ) -> std::result::Result<<Self as SessionTarget<P>>::Session, <Self as SessionTarget<P>>::Error>
    where
        Self: SessionTarget<P>,
    {
        <Self as SessionTarget<P>>::prepare(self, plan)
    }
}

impl std::fmt::Debug for NativeCpuSessionTarget<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeCpuSessionTarget")
            .field("vectorized", &self.vectorized)
            .field("non_finite_policy", &self.non_finite_policy)
            .finish_non_exhaustive()
    }
}

/// Strict persistent Metal target for compiled training and inference plans.
///
/// Construction derives the renderer from the selected device, so capability
/// identity cannot drift between rendering and resource preparation. The
/// Target implementations decide whether a plan is reusable by reference or
/// consumed into its device resources. The optional scoreboard is bound before
/// resources are exposed by every supported plan.
#[derive(Clone)]
pub struct MetalSessionTarget {
    device: MetalDevice,
    renderer: MetalRenderer,
    scoreboard: Option<MetalScoreboardContext>,
}

impl MetalSessionTarget {
    pub fn new(device: MetalDevice, local_size: usize) -> std::result::Result<Self, MetalError> {
        let renderer = device.renderer(local_size)?;
        Ok(Self {
            device,
            renderer,
            scoreboard: None,
        })
    }

    /// Attaches fail-soft execution observation before resource preparation.
    pub fn with_scoreboard(mut self, context: MetalScoreboardContext) -> Self {
        self.scoreboard = Some(context);
        self
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn renderer(&self) -> &MetalRenderer {
        &self.renderer
    }

    pub fn scoreboard_context(&self) -> Option<&MetalScoreboardContext> {
        self.scoreboard.as_ref()
    }

    /// Returns the planning controls already authenticated by this target.
    pub const fn plan_options(&self) -> MetalPlanOptions {
        MetalPlanOptions::new(self.renderer.local_size)
    }

    /// Prepares a plan implemented for this exact Metal target.
    pub fn prepare<P>(
        &self,
        plan: P,
    ) -> std::result::Result<<Self as SessionTarget<P>>::Session, <Self as SessionTarget<P>>::Error>
    where
        Self: SessionTarget<P>,
    {
        <Self as SessionTarget<P>>::prepare(self, plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompiledAdamWConfig, CompiledAdamWPlan, DType, Scalar, Shape, TensorData,
        TrainingParameterInit,
    };
    use std::collections::BTreeMap;

    fn scalar_plan() -> CompiledAdamWPlan {
        let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01).unwrap();
        let parameter = TrainingParameterInit::new(
            "weight",
            TensorData::from_scalars(Shape::from([]), DType::F32, [Scalar::F(2.0)]).unwrap(),
        )
        .unwrap();
        CompiledAdamWPlan::compile(config, [parameter], |graph, _, parameters| {
            let loss = graph.mul(parameters["weight"], parameters["weight"])?;
            Ok((loss, BTreeMap::new()))
        })
        .unwrap()
    }

    #[test]
    fn cpu_target_prepares_independent_sessions_from_one_plan() {
        let plan = scalar_plan();
        let unit_compatible: CpuSessionTarget = CpuSessionTarget;
        let target = CpuSessionTarget::new();
        assert_eq!(target, unit_compatible);
        let first = target.prepare(&plan).unwrap();
        let second = target.prepare(&plan).unwrap();

        assert_eq!(first.capture_identity(), plan.capture_identity());
        assert_eq!(second.capture_identity(), plan.capture_identity());
        assert_eq!(first.step_count(), 0);
        assert_eq!(second.step_count(), 0);

        let guarded = target
            .with_non_finite_policy(CpuNonFinitePolicy::RejectTransition)
            .prepare(&plan)
            .unwrap();
        assert_eq!(guarded.capture_identity(), plan.capture_identity());
        assert_eq!(
            guarded.non_finite_policy(),
            CpuNonFinitePolicy::RejectTransition
        );
    }
}
