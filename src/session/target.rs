//! Typed runtime selection for resource-free session plans.

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

/// Graph-free CPU replay target for compiled training plans.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuSessionTarget;

impl CpuSessionTarget {
    pub const fn new() -> Self {
        Self
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
        let target = CpuSessionTarget::new();
        let first = target.prepare(&plan).unwrap();
        let second = target.prepare(&plan).unwrap();

        assert_eq!(first.capture_identity(), plan.capture_identity());
        assert_eq!(second.capture_identity(), plan.capture_identity());
        assert_eq!(first.step_count(), 0);
        assert_eq!(second.step_count(), 0);
    }
}
