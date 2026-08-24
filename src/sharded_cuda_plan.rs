//! Deterministic CUDA realization planning for graph-composed sharded tensors.
//!
//! This is deliberately a data-only Phase 3A plan. It neither enters a CUDA context
//! nor creates streams, allocations, modules, or Driver work.
use crate::collective::{
    CollectiveKind, CollectivePlan, CollectivePlanner, CollectiveRequest, DeviceGroup,
    DeviceId as SemanticDeviceId, Reduction,
};
use crate::{
    Capability, DType, Error, Graph, PrimaryContext, PtxRenderer, Shape, ShardedGraphTensor,
    schedule,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Caller-supplied owner/capability binding. Context resources stay outside the serializable plan.
#[derive(Clone)]
pub struct CudaPlanBinding {
    pub device: SemanticDeviceId,
    pub context: PrimaryContext,
    pub capability: Capability,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CudaPlanDiagnostic {
    Unsupported { node: usize, reason: String },
    CapabilityMismatch { reason: String },
    Trace { action: String, reason: String },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CudaPlanStage {
    Local {
        id: usize,
        device: SemanticDeviceId,
        owner_identity: usize,
        node: usize,
        shape: Shape,
        dtype: DType,
        inputs: Vec<u64>,
        output: u64,
        dependencies: Vec<usize>,
        source_key: String,
        module_key: String,
        diagnostic: Option<CudaPlanDiagnostic>,
    },
    Collective {
        id: usize,
        action: String,
        plan: CollectivePlan,
        dependencies: Vec<usize>,
    },
    Transfer {
        id: usize,
        action: String,
        routes: Vec<(SemanticDeviceId, SemanticDeviceId, usize)>,
        dependencies: Vec<usize>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardedCudaPlan {
    pub graph_id: u64,
    pub layout_key: String,
    pub bindings: Vec<(SemanticDeviceId, usize, u32)>,
    pub stages: Vec<CudaPlanStage>,
    pub diagnostics: Vec<CudaPlanDiagnostic>,
    pub cache_key: String,
}
pub struct ShardedCudaPlanner;
impl ShardedCudaPlanner {
    pub fn build(
        graph: &Graph,
        value: &ShardedGraphTensor,
        bindings: &[CudaPlanBinding],
    ) -> Result<ShardedCudaPlan, Error> {
        if value.graph_id() != graph.id() {
            return Err(err("sharded tensor belongs to another graph"));
        }
        let group = value.layout().group();
        validate_bindings(group, bindings)?;
        let mut stages = Vec::new();
        let mut diagnostics = Vec::new();
        let mut previous = Vec::new();
        for (rank, node) in value.nodes().iter().enumerate() {
            let binding = &bindings[rank];
            let owner_identity = binding.context.identity();
            let scheduled = schedule(graph, *node).map_err(|e| err(e.to_string()))?;
            let mut diagnostic = scheduled
                .items
                .first()
                .and_then(|item| item.boundary.as_ref())
                .map(|x| CudaPlanDiagnostic::Unsupported {
                    node: node.index(),
                    reason: format!("schedule boundary: {x:?}"),
                });
            let source_key = scheduled
                .items
                .first()
                .map(|x| format!("schedule:{}", x.cache_key))
                .unwrap_or_else(|| "schedule:empty".into());
            if diagnostic.is_none()
                && let Some(item) = scheduled.items.first()
                && PtxRenderer::new(binding.capability.sm())
                    .and_then(|r| r.render(&item.kernel))
                    .is_err()
            {
                diagnostic=Some(CudaPlanDiagnostic::Unsupported{node:node.index(),reason:"current PTX renderer accepts only elementwise/select/cast; reductions are Phase 3B1 diagnostics".into()});
            }
            if let Some(d) = diagnostic.clone() {
                diagnostics.push(d);
            }
            let id = stages.len();
            let item = scheduled.items.first();
            stages.push(CudaPlanStage::Local {
                id,
                device: binding.device.clone(),
                owner_identity,
                node: node.index(),
                shape: graph.shape(*node)?.clone(),
                dtype: graph.dtype(*node)?,
                inputs: item
                    .map(|x| x.inputs.iter().map(|b| b.id).collect())
                    .unwrap_or_default(),
                output: item.map(|x| x.output.id).unwrap_or(node.index() as u64),
                dependencies: previous.clone(),
                source_key: source_key.clone(),
                module_key: format!(
                    "owner:{}:sm{}:{source_key}",
                    owner_identity,
                    binding.capability.sm()
                ),
                diagnostic,
            });
            previous = vec![id];
        }
        for trace in &value.trace().steps {
            if trace.action.contains("all-reduce") {
                let plan = collective_plan(group, value.dtype(), value.layout())?;
                let id = stages.len();
                stages.push(CudaPlanStage::Collective {
                    id,
                    action: trace.action.into(),
                    plan,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            } else if trace.action == "redistribute" || trace.action == "gather-movement" {
                let id = stages.len();
                let elements = value.layout().global_shape().numel()?;
                let routes = group
                    .devices()
                    .iter()
                    .map(|device| (device.clone(), device.clone(), elements))
                    .collect();
                stages.push(CudaPlanStage::Transfer {
                    id,
                    action: trace.action.into(),
                    routes,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            }
        }
        let cache_key = format!(
            "sharded-cuda-plan:v1:{}:{}",
            value.layout().cache_key(),
            bindings
                .iter()
                .map(|b| format!("{}:{}", b.context.identity(), b.capability.sm()))
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(ShardedCudaPlan {
            graph_id: graph.id(),
            layout_key: value.layout().cache_key().into(),
            bindings: bindings
                .iter()
                .map(|b| (b.device.clone(), b.context.identity(), b.capability.sm()))
                .collect(),
            stages,
            diagnostics,
            cache_key,
        })
    }
}
fn validate_bindings(group: &DeviceGroup, bindings: &[CudaPlanBinding]) -> Result<(), Error> {
    if bindings.len() != group.len() {
        return Err(err("CUDA bindings do not match device group length"));
    }
    if bindings
        .iter()
        .map(|b| &b.device)
        .collect::<BTreeSet<_>>()
        .len()
        != bindings.len()
        || bindings
            .iter()
            .map(|b| b.context.identity())
            .collect::<BTreeSet<_>>()
            .len()
            != bindings.len()
    {
        return Err(err(
            "CUDA plan bindings require distinct semantic devices and owners",
        ));
    }
    for (expected, actual) in group.devices().iter().zip(bindings) {
        if expected != &actual.device {
            return Err(err("CUDA plan binding device order does not match layout"));
        }
        if actual.context.device() != actual.capability.device {
            return Err(err("CUDA capability device does not match primary context"));
        }
    }
    Ok(())
}
fn collective_plan(
    group: &DeviceGroup,
    dtype: DType,
    layout: &crate::ShardLayout,
) -> Result<CollectivePlan, Error> {
    let n = layout.global_shape().numel()?;
    CollectivePlanner::plan(CollectiveRequest {
        group: group.clone(),
        kind: CollectiveKind::AllReduce {
            reduction: Reduction::Sum,
        },
        dtype,
        input_lengths: vec![n; group.len()],
    })
}
fn err(reason: impl Into<String>) -> Error {
    Error::Collective {
        reason: reason.into(),
    }
}
