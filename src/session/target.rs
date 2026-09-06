//! Backend selection for resource-free compiled session plans.

use super::{CompiledAdamWPlan, CpuCompiledAdamW, MetalCompiledAdamW};
use crate::Result;
use crate::runtime::metal::{MetalDevice, MetalError, MetalRenderer, MetalScoreboardContext};

/// A concrete preparation target for one resource-free compiled plan.
///
/// The plan type determines the prepared session type through this trait's
/// associated type. Adding a backend therefore requires an implementation,
/// not another central dispatcher branch.
pub trait CompiledSessionTarget<P> {
    type Session;

    fn prepare(&self, plan: &P) -> Result<Self::Session>;
}

/// Graph-free CPU replay target for compiled training plans.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuSessionTarget;

impl CpuSessionTarget {
    pub const fn new() -> Self {
        Self
    }
}

/// Strict persistent Metal target for compiled training plans.
///
/// Construction derives the renderer from the selected device, so capability
/// identity cannot drift between rendering and resource preparation. The
/// target is reusable: each preparation creates an independent device session
/// from the same authenticated compiled plan.
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
}

impl CompiledSessionTarget<CompiledAdamWPlan> for CpuSessionTarget {
    type Session = CpuCompiledAdamW;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        plan.prepare_cpu()
    }
}

impl CompiledSessionTarget<CompiledAdamWPlan> for MetalSessionTarget {
    type Session = MetalCompiledAdamW;

    fn prepare(&self, plan: &CompiledAdamWPlan) -> Result<Self::Session> {
        let rendered = plan.metal_plan(self.renderer.clone())?;
        match &self.scoreboard {
            Some(context) => rendered.prepare_with_scoreboard(self.device.clone(), context.clone()),
            None => rendered.prepare(self.device.clone()),
        }
    }
}

impl CompiledAdamWPlan {
    /// Prepares this authenticated plan through a concrete target.
    ///
    /// The target's associated session keeps backend-specific reports and
    /// controls statically available without a runtime backend enum or CPU
    /// fallback.
    pub fn prepare<T>(&self, target: &T) -> Result<T::Session>
    where
        T: CompiledSessionTarget<Self>,
    {
        target.prepare(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompiledAdamWConfig, DType, Scalar, Shape, TensorData, TrainingParameterInit};
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
        let first = plan.prepare(&target).unwrap();
        let second = plan.prepare(&target).unwrap();

        assert_eq!(first.capture_identity(), plan.capture_identity());
        assert_eq!(second.capture_identity(), plan.capture_identity());
        assert_eq!(first.step_count(), 0);
        assert_eq!(second.step_count(), 0);
    }
}
