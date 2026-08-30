//! Deterministic CUDA realization planning for graph-composed sharded tensors.
//!
//! Planning is deliberately data-only. Phase 3B2 retains the typed all-reduce
//! buffer ABI here; execution owns contexts, streams, allocations, and Driver work
//! separately in `sharded_cuda_execute`.
use crate::collective::{
    CollectiveKind, CollectivePlan, CollectivePlanner, CollectiveRequest, DeviceGroup,
    DeviceId as SemanticDeviceId, Reduction,
};
use crate::sharded_cuda_execute::{BufferSubstitution, ShardedCudaPlanComposition};
use crate::{
    Capability, CollectiveBoundaryLifecycle, DType, Error, Graph, NodeId, Op, PrimaryContext,
    PtxRenderer, RenderedPtx, Shape, ShardedGraphTensor, UnaryOp, schedule,
    schedule_with_external_materializations,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Caller-supplied owner/capability binding. Context resources stay outside the serializable plan.
#[derive(Clone)]
pub struct CudaPlanBinding {
    pub device: SemanticDeviceId,
    pub context: PrimaryContext,
    pub capability: Capability,
}

/// Closed graph operation identity retained only by the graph-aware unary
/// downstream companion. The released v5 envelope remains unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GraphBackedDownstreamUnary {
    Neg,
    Abs,
    NegF64,
    AbsF64,
}

impl GraphBackedDownstreamUnary {
    pub(crate) fn op(self) -> UnaryOp {
        match self {
            Self::Neg => UnaryOp::Neg,
            Self::Abs => UnaryOp::Abs,
            Self::NegF64 => UnaryOp::Neg,
            Self::AbsF64 => UnaryOp::Abs,
        }
    }

    fn dtype(self) -> DType {
        match self {
            Self::Neg | Self::Abs => DType::F32,
            Self::NegF64 | Self::AbsF64 => DType::F64,
        }
    }

    fn cache_prefix(self) -> &'static str {
        match self {
            Self::Neg => "graph-backed-unary-neg:",
            Self::Abs => "graph-backed-unary-abs:",
            Self::NegF64 => "graph-backed-unary-f64-neg:",
            Self::AbsF64 => "graph-backed-unary-f64-abs:",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Neg => "Neg",
            Self::Abs => "Abs",
            Self::NegF64 => "F64 Neg",
            Self::AbsF64 => "F64 Abs",
        }
    }
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
        /// Typed computed input nodes explicitly supplied by a preceding stage.
        external_materializations: Vec<u64>,
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
        /// Ordered rank-local output buffers mutated in place by this plan.
        /// The order is the semantic `DeviceGroup` order and is never inferred
        /// from CUDA handles at execution time.
        #[serde(default)]
        buffers: Vec<u64>,
        dependencies: Vec<usize>,
    },
    Transfer {
        id: usize,
        action: String,
        routes: Vec<CudaTransferRoute>,
        dependencies: Vec<usize>,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CudaTransferRoute {
    pub source_rank: usize,
    pub source_device: SemanticDeviceId,
    pub source_buffer: u64,
    pub source_element_offset: usize,
    pub destination_rank: usize,
    pub destination_device: SemanticDeviceId,
    pub destination_buffer: u64,
    pub destination_element_offset: usize,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardedCudaPlan {
    pub graph_id: u64,
    pub layout_key: String,
    pub bindings: Vec<(SemanticDeviceId, usize, u32)>,
    pub stages: Vec<CudaPlanStage>,
    pub diagnostics: Vec<CudaPlanDiagnostic>,
    pub cache_key: String,
    /// Explicit v3-only schema: older raw/v1/v2 paths must reject this key
    /// rather than infer it through serde defaults.
    pub materializations: Vec<CollectiveResultMaterialization>,
}

/// Canonical, versioned data-only envelope for a sharded CUDA plan.
///
/// Runtime owners, streams, modules, leases, and capture state are never part
/// of this artifact. Version one is deliberately candidate-free: a future
/// collective transaction must introduce a new version rather than relying on
/// serde defaults to infer candidate buffers or commit boundaries.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardedCudaPlanArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
}

/// A transaction-owned rank-local buffer. This descriptor is data-only: CUDA
/// leases are created only after the complete transaction has been preflighted.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCandidateDescriptor {
    pub stage: usize,
    pub rank: usize,
    pub candidate_buffer: u64,
    pub source_buffer: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
}

/// Ordered copy from a transaction candidate into its declared final target.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCommitRecord {
    pub order: usize,
    pub rank: usize,
    pub candidate_buffer: u64,
    pub target_buffer: u64,
}

/// Logical candidate result binding reserved for a future local consumer.
/// It has no allocation or launch semantics by itself.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveResultMaterialization {
    pub boundary_key: String,
    pub replicated_result: usize,
    pub rank: usize,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub candidate_buffer: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub producer_stage: usize,
    pub first_consumer: usize,
    pub last_consumer: usize,
}

/// Explicit v4 lifecycle discriminator. It prevents a terminal artifact from
/// accidentally acquiring a downstream consumer through omitted/defaulted
/// fields.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CollectiveMaterializationLifecycle {
    Terminal,
    Downstream {
        first_consumer_stage: usize,
        lifetime_end_stage: usize,
    },
}

/// One rank-local ABI input that a future local stage is allowed to consume.
/// It is data-only and does not authorize PTX execution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveConsumerDescriptor {
    pub rank: usize,
    pub consumer_stage: usize,
    pub consumer_buffer: u64,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
}

/// V4 binding of a transaction candidate to an explicit result lifecycle.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveLifecycleMaterialization {
    pub materialization: CollectiveResultMaterialization,
    pub lifecycle: CollectiveMaterializationLifecycle,
    pub consumers: Vec<CollectiveConsumerDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveTransactionArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
}

/// Version-three envelope. Unlike v1/raw and v2, materializations are explicit
/// signed logical metadata and are never inferred during owner rebinding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveMaterializationArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
}

/// Version-four envelope for explicit downstream result lifetimes. V3 stays
/// terminal-only; no v4 metadata is accepted by a legacy decoder.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveLifecycleMaterializationArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
}
type LifecycleMaterializationArtifactParts = (
    ShardedCudaPlan,
    Vec<CollectiveCandidateDescriptor>,
    Vec<CollectiveCommitRecord>,
    Vec<CollectiveLifecycleMaterialization>,
);

/// Transaction-owned output of one explicitly declared downstream local stage.
/// This remains a data-only ABI until a later executor vertical can bind the
/// stage's PTX output without exposing the collective candidate as an alias.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveDownstreamOutputDescriptor {
    pub rank: usize,
    pub consumer_stage: usize,
    pub output_candidate_buffer: u64,
    pub source_candidate_buffer: u64,
    pub destination_buffer: u64,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub first_stage: usize,
    pub last_stage: usize,
}

/// Ordered transaction finalization of a local-stage output candidate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveDownstreamOutputCommitRecord {
    pub order: usize,
    pub rank: usize,
    pub output_candidate_buffer: u64,
    pub destination_buffer: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveGraphResultBinding {
    pub replicated_result: usize,
    pub rank: usize,
    pub candidate_buffer: u64,
    pub local_input_buffer: u64,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub first_consumer_stage: usize,
    pub lifetime_end_stage: usize,
}

impl CollectiveGraphResultBinding {
    /// Pure per-rank validation used by the graph-unary envelope.
    pub fn validate(&self) -> Result<(), Error> {
        if self.candidate_buffer == self.local_input_buffer
            || self.bytes
                != self
                    .shape
                    .numel()?
                    .checked_mul(self.dtype.itemsize())
                    .ok_or_else(|| err("graph result binding byte overflow"))?
            || self.first_consumer_stage > self.lifetime_end_stage
        {
            return Err(err("collective graph result binding is inconsistent"));
        }
        Ok(())
    }

    pub fn canonical_key(&self) -> (usize, usize, u64, u64) {
        (
            self.replicated_result,
            self.rank,
            self.candidate_buffer,
            self.local_input_buffer,
        )
    }
}

/// Graph-unary local ABI identity. This record names the distinct graph-schedule
/// input key that the executor substitutes with the collective candidate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveDownstreamConsumerAbi {
    pub replicated_result: usize,
    pub rank: usize,
    pub candidate_buffer: u64,
    pub local_input_buffer: u64,
    pub output_candidate_buffer: u64,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub consumer_stage: usize,
    pub lifetime_end_stage: usize,
}

impl CollectiveDownstreamConsumerAbi {
    pub fn validate(&self) -> Result<(), Error> {
        if self.candidate_buffer == self.local_input_buffer
            || self.output_candidate_buffer == self.candidate_buffer
            || self.output_candidate_buffer == self.local_input_buffer
            || self.consumer_stage > self.lifetime_end_stage
            || self.bytes
                != self
                    .shape
                    .numel()?
                    .checked_mul(self.dtype.itemsize())
                    .ok_or_else(|| err("v5 consumer ABI byte overflow"))?
        {
            return Err(err("v5 downstream consumer ABI is inconsistent"));
        }
        Ok(())
    }

    pub fn canonical_key(&self) -> (usize, usize, u64, u64, u64) {
        (
            self.replicated_result,
            self.rank,
            self.candidate_buffer,
            self.local_input_buffer,
            self.output_candidate_buffer,
        )
    }
}

/// Version-five envelope.  It is the first format that can describe owned
/// downstream output candidates and their ordered final commits; older
/// envelopes deliberately reject these keys rather than infer defaults.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveDownstreamOutputArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
    pub outputs: Vec<CollectiveDownstreamOutputDescriptor>,
    pub output_commits: Vec<CollectiveDownstreamOutputCommitRecord>,
}
type DownstreamOutputArtifactParts = (
    ShardedCudaPlan,
    Vec<CollectiveCandidateDescriptor>,
    Vec<CollectiveCommitRecord>,
    Vec<CollectiveLifecycleMaterialization>,
    Vec<CollectiveDownstreamOutputDescriptor>,
    Vec<CollectiveDownstreamOutputCommitRecord>,
);

/// Version-one envelope for the closed graph-backed downstream unary route.
/// It is deliberately distinct from the released v5 logical-output envelope:
/// graph schedule bindings cannot be inferred by an older decoder.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CollectiveGraphUnaryOutputArtifact {
    pub format_version: u32,
    pub fingerprint: String,
    pub plan: ShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
    pub graph_result_bindings: Vec<CollectiveGraphResultBinding>,
    pub consumer_abis: Vec<CollectiveDownstreamConsumerAbi>,
    pub outputs: Vec<CollectiveDownstreamOutputDescriptor>,
    pub output_commits: Vec<CollectiveDownstreamOutputCommitRecord>,
}

type GraphUnaryOutputArtifactParts = (
    ShardedCudaPlan,
    Vec<CollectiveCandidateDescriptor>,
    Vec<CollectiveCommitRecord>,
    Vec<CollectiveLifecycleMaterialization>,
    Vec<CollectiveGraphResultBinding>,
    Vec<CollectiveDownstreamConsumerAbi>,
    Vec<CollectiveDownstreamOutputDescriptor>,
    Vec<CollectiveDownstreamOutputCommitRecord>,
);

pub struct CollectiveGraphUnaryOutputComponents {
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
    pub graph_result_bindings: Vec<CollectiveGraphResultBinding>,
    pub consumer_abis: Vec<CollectiveDownstreamConsumerAbi>,
    pub outputs: Vec<CollectiveDownstreamOutputDescriptor>,
    pub output_commits: Vec<CollectiveDownstreamOutputCommitRecord>,
}

#[derive(Clone, Copy)]
struct DownstreamOutputArtifactComponentRefs<'a> {
    candidates: &'a [CollectiveCandidateDescriptor],
    commits: &'a [CollectiveCommitRecord],
    materializations: &'a [CollectiveLifecycleMaterialization],
    graph_result_bindings: &'a [CollectiveGraphResultBinding],
    consumer_abis: &'a [CollectiveDownstreamConsumerAbi],
    outputs: &'a [CollectiveDownstreamOutputDescriptor],
    output_commits: &'a [CollectiveDownstreamOutputCommitRecord],
}

impl CollectiveGraphUnaryOutputComponents {
    fn refs(&self) -> DownstreamOutputArtifactComponentRefs<'_> {
        DownstreamOutputArtifactComponentRefs {
            candidates: &self.candidates,
            commits: &self.commits,
            materializations: &self.materializations,
            graph_result_bindings: &self.graph_result_bindings,
            consumer_abis: &self.consumer_abis,
            outputs: &self.outputs,
            output_commits: &self.output_commits,
        }
    }
}

impl CollectiveDownstreamOutputArtifact {
    pub const FORMAT_VERSION: u32 = 5;

    pub fn encode(
        plan: &ShardedCudaPlan,
        candidates: Vec<CollectiveCandidateDescriptor>,
        commits: Vec<CollectiveCommitRecord>,
        materializations: Vec<CollectiveLifecycleMaterialization>,
        outputs: Vec<CollectiveDownstreamOutputDescriptor>,
        output_commits: Vec<CollectiveDownstreamOutputCommitRecord>,
    ) -> Result<Vec<u8>, Error> {
        validate_downstream_output_plan(
            plan,
            &candidates,
            &commits,
            &materializations,
            &outputs,
            &output_commits,
        )?;
        let fingerprint = downstream_output_fingerprint(
            plan,
            &candidates,
            &commits,
            &materializations,
            &outputs,
            &output_commits,
        )?;
        serde_json::to_vec(&Self {
            format_version: Self::FORMAT_VERSION,
            fingerprint,
            plan: plan.clone(),
            candidates,
            commits,
            materializations,
            outputs,
            output_commits,
        })
        .map_err(|error| {
            err(format!(
                "sharded CUDA downstream output artifact encode: {error}"
            ))
        })
    }

    pub fn decode(bytes: &[u8]) -> Result<DownstreamOutputArtifactParts, Error> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            err(format!(
                "sharded CUDA downstream output artifact JSON: {error}"
            ))
        })?;
        reject_unknown_envelope_fields(
            &value,
            &[
                "format_version",
                "fingerprint",
                "plan",
                "candidates",
                "commits",
                "materializations",
                "outputs",
                "output_commits",
            ],
        )?;
        reject_unknown_plan_fields(
            value
                .get("plan")
                .ok_or_else(|| err("v5 artifact plan is absent"))?,
        )?;
        let envelope: Self = serde_json::from_value(value).map_err(|error| {
            err(format!(
                "sharded CUDA downstream output artifact envelope: {error}"
            ))
        })?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err(
                "unsupported sharded CUDA downstream output artifact version",
            ));
        }
        validate_downstream_output_plan(
            &envelope.plan,
            &envelope.candidates,
            &envelope.commits,
            &envelope.materializations,
            &envelope.outputs,
            &envelope.output_commits,
        )?;
        if envelope.fingerprint
            != downstream_output_fingerprint(
                &envelope.plan,
                &envelope.candidates,
                &envelope.commits,
                &envelope.materializations,
                &envelope.outputs,
                &envelope.output_commits,
            )?
        {
            return Err(err(
                "sharded CUDA downstream output artifact fingerprint mismatch",
            ));
        }
        Ok((
            envelope.plan,
            envelope.candidates,
            envelope.commits,
            envelope.materializations,
            envelope.outputs,
            envelope.output_commits,
        ))
    }
}

impl CollectiveGraphUnaryOutputArtifact {
    pub const FORMAT_VERSION: u32 = 1;

    pub fn encode(
        plan: &ShardedCudaPlan,
        components: CollectiveGraphUnaryOutputComponents,
    ) -> Result<Vec<u8>, Error> {
        let refs = components.refs();
        validate_graph_unary_output_plan(plan, &refs)?;
        let fingerprint = graph_unary_output_fingerprint(plan, &refs)?;
        serde_json::to_vec(&Self {
            format_version: Self::FORMAT_VERSION,
            fingerprint,
            plan: plan.clone(),
            candidates: components.candidates,
            commits: components.commits,
            materializations: components.materializations,
            graph_result_bindings: components.graph_result_bindings,
            consumer_abis: components.consumer_abis,
            outputs: components.outputs,
            output_commits: components.output_commits,
        })
        .map_err(|error| err(format!("sharded CUDA graph unary artifact encode: {error}")))
    }

    pub fn decode(bytes: &[u8]) -> Result<GraphUnaryOutputArtifactParts, Error> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| err(format!("sharded CUDA graph unary artifact JSON: {error}")))?;
        reject_unknown_envelope_fields(
            &value,
            &[
                "format_version",
                "fingerprint",
                "plan",
                "candidates",
                "commits",
                "materializations",
                "graph_result_bindings",
                "consumer_abis",
                "outputs",
                "output_commits",
            ],
        )?;
        reject_unknown_plan_fields(
            value
                .get("plan")
                .ok_or_else(|| err("graph unary artifact plan is absent"))?,
        )?;
        let envelope: Self = serde_json::from_value(value).map_err(|error| {
            err(format!(
                "sharded CUDA graph unary artifact envelope: {error}"
            ))
        })?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err("unsupported sharded CUDA graph unary artifact version"));
        }
        let refs = DownstreamOutputArtifactComponentRefs {
            candidates: &envelope.candidates,
            commits: &envelope.commits,
            materializations: &envelope.materializations,
            graph_result_bindings: &envelope.graph_result_bindings,
            consumer_abis: &envelope.consumer_abis,
            outputs: &envelope.outputs,
            output_commits: &envelope.output_commits,
        };
        validate_graph_unary_output_plan(&envelope.plan, &refs)?;
        if envelope.fingerprint != graph_unary_output_fingerprint(&envelope.plan, &refs)? {
            return Err(err(
                "sharded CUDA graph unary artifact fingerprint mismatch",
            ));
        }
        Ok((
            envelope.plan,
            envelope.candidates,
            envelope.commits,
            envelope.materializations,
            envelope.graph_result_bindings,
            envelope.consumer_abis,
            envelope.outputs,
            envelope.output_commits,
        ))
    }
}

impl CollectiveLifecycleMaterializationArtifact {
    pub const FORMAT_VERSION: u32 = 4;

    pub fn encode(
        plan: &ShardedCudaPlan,
        candidates: Vec<CollectiveCandidateDescriptor>,
        commits: Vec<CollectiveCommitRecord>,
        materializations: Vec<CollectiveLifecycleMaterialization>,
    ) -> Result<Vec<u8>, Error> {
        validate_lifecycle_materialization_plan(plan, &candidates, &commits, &materializations)?;
        let fingerprint =
            lifecycle_materialization_fingerprint(plan, &candidates, &commits, &materializations)?;
        serde_json::to_vec(&Self {
            format_version: Self::FORMAT_VERSION,
            fingerprint,
            plan: plan.clone(),
            candidates,
            commits,
            materializations,
        })
        .map_err(|error| {
            err(format!(
                "sharded CUDA lifecycle materialization artifact encode: {error}"
            ))
        })
    }

    pub fn decode(bytes: &[u8]) -> Result<LifecycleMaterializationArtifactParts, Error> {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            err(format!(
                "sharded CUDA lifecycle materialization artifact JSON: {error}"
            ))
        })?;
        reject_downstream_output_metadata(&value)?;
        reject_unknown_envelope_fields(
            &value,
            &[
                "format_version",
                "fingerprint",
                "plan",
                "candidates",
                "commits",
                "materializations",
            ],
        )?;
        reject_unknown_plan_fields(
            value
                .get("plan")
                .ok_or_else(|| err("v4 artifact plan is absent"))?,
        )?;
        let envelope: Self = serde_json::from_value(value).map_err(|error| {
            err(format!(
                "sharded CUDA lifecycle materialization artifact envelope: {error}"
            ))
        })?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err(
                "unsupported sharded CUDA lifecycle materialization artifact version",
            ));
        }
        validate_lifecycle_materialization_plan(
            &envelope.plan,
            &envelope.candidates,
            &envelope.commits,
            &envelope.materializations,
        )?;
        if envelope.fingerprint
            != lifecycle_materialization_fingerprint(
                &envelope.plan,
                &envelope.candidates,
                &envelope.commits,
                &envelope.materializations,
            )?
        {
            return Err(err(
                "sharded CUDA lifecycle materialization artifact fingerprint mismatch",
            ));
        }
        Ok((
            envelope.plan,
            envelope.candidates,
            envelope.commits,
            envelope.materializations,
        ))
    }
}

impl CollectiveMaterializationArtifact {
    pub const FORMAT_VERSION: u32 = 3;

    pub fn encode(
        plan: &ShardedCudaPlan,
        candidates: Vec<CollectiveCandidateDescriptor>,
        commits: Vec<CollectiveCommitRecord>,
    ) -> Result<Vec<u8>, Error> {
        validate_materialization_plan(plan, &candidates, &commits)?;
        let fingerprint = materialization_fingerprint(plan, &candidates, &commits)?;
        serde_json::to_vec(&Self {
            format_version: Self::FORMAT_VERSION,
            fingerprint,
            plan: plan.clone(),
            candidates,
            commits,
        })
        .map_err(|error| {
            err(format!(
                "sharded CUDA materialization artifact encode: {error}"
            ))
        })
    }

    /// V3 is intentionally the only artifact route that accepts logical
    /// materializations.  The fingerprint is verified before a caller can
    /// bind owners, populate a cache, or allocate a runtime lease.
    pub fn decode(
        bytes: &[u8],
    ) -> Result<
        (
            ShardedCudaPlan,
            Vec<CollectiveCandidateDescriptor>,
            Vec<CollectiveCommitRecord>,
        ),
        Error,
    > {
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
            err(format!(
                "sharded CUDA materialization artifact JSON: {error}"
            ))
        })?;
        reject_downstream_output_metadata(&value)?;
        reject_unknown_envelope_fields(
            &value,
            &[
                "format_version",
                "fingerprint",
                "plan",
                "candidates",
                "commits",
            ],
        )?;
        reject_unknown_plan_fields(
            value
                .get("plan")
                .ok_or_else(|| err("sharded CUDA materialization artifact plan is absent"))?,
        )?;
        let envelope: Self = serde_json::from_value(value).map_err(|error| {
            err(format!(
                "sharded CUDA materialization artifact envelope: {error}"
            ))
        })?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err(
                "unsupported sharded CUDA materialization artifact version",
            ));
        }
        validate_materialization_plan(&envelope.plan, &envelope.candidates, &envelope.commits)?;
        if envelope.fingerprint
            != materialization_fingerprint(&envelope.plan, &envelope.candidates, &envelope.commits)?
        {
            return Err(err(
                "sharded CUDA materialization artifact fingerprint mismatch",
            ));
        }
        Ok((envelope.plan, envelope.candidates, envelope.commits))
    }
}

impl CollectiveTransactionArtifact {
    pub const FORMAT_VERSION: u32 = 2;

    pub fn encode(
        plan: &ShardedCudaPlan,
        candidates: Vec<CollectiveCandidateDescriptor>,
        commits: Vec<CollectiveCommitRecord>,
    ) -> Result<Vec<u8>, Error> {
        validate_transaction_plan(plan, &candidates, &commits)?;
        let fingerprint = transaction_fingerprint(plan, &candidates, &commits)?;
        serde_json::to_vec(&serde_json::json!({
            "format_version": Self::FORMAT_VERSION,
            "fingerprint": fingerprint,
            "plan": legacy_candidate_free_plan_value(plan)?,
            "candidates": candidates,
            "commits": commits,
        }))
        .map_err(|error| err(format!("sharded CUDA transaction artifact encode: {error}")))
    }

    pub fn decode(
        bytes: &[u8],
    ) -> Result<
        (
            ShardedCudaPlan,
            Vec<CollectiveCandidateDescriptor>,
            Vec<CollectiveCommitRecord>,
        ),
        Error,
    > {
        let mut value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| err(format!("sharded CUDA transaction artifact JSON: {error}")))?;
        reject_downstream_output_metadata(&value)?;
        reject_materialization_metadata(&value)?;
        inject_empty_legacy_materializations(&mut value, true)?;
        let envelope: Self = serde_json::from_value(value)
            .map_err(|error| err(format!("sharded CUDA transaction artifact JSON: {error}")))?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err("unsupported sharded CUDA transaction artifact version"));
        }
        validate_transaction_plan(&envelope.plan, &envelope.candidates, &envelope.commits)?;
        if envelope.fingerprint
            != transaction_fingerprint(&envelope.plan, &envelope.candidates, &envelope.commits)?
        {
            return Err(err(
                "sharded CUDA transaction artifact fingerprint mismatch",
            ));
        }
        Ok((envelope.plan, envelope.candidates, envelope.commits))
    }
}

impl ShardedCudaPlanArtifact {
    pub const FORMAT_VERSION: u32 = 1;

    pub fn encode(plan: &ShardedCudaPlan) -> Result<Vec<u8>, Error> {
        validate_candidate_free_plan(plan)?;
        let fingerprint = plan_fingerprint(plan)?;
        serde_json::to_vec(&serde_json::json!({
            "format_version": Self::FORMAT_VERSION,
            "fingerprint": fingerprint,
            "plan": legacy_candidate_free_plan_value(plan)?,
        }))
        .map_err(|error| err(format!("sharded CUDA artifact encode: {error}")))
    }

    /// Decodes either the v1 envelope or a released raw plan. Raw plans retain
    /// their candidate-free behavior only; transaction keys are rejected before
    /// deserialization, cache insertion, owner binding, or execution.
    pub fn decode(bytes: &[u8]) -> Result<ShardedCudaPlan, Error> {
        let mut value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| err(format!("sharded CUDA artifact JSON: {error}")))?;
        reject_downstream_output_metadata(&value)?;
        reject_transaction_metadata(&value)?;
        reject_materialization_metadata(&value)?;
        if value.get("format_version").is_none() {
            inject_empty_legacy_materializations(&mut value, false)?;
            let plan = serde_json::from_value(value)
                .map_err(|error| err(format!("legacy sharded CUDA plan: {error}")))?;
            validate_candidate_free_plan(&plan)?;
            return Ok(plan);
        }
        inject_empty_legacy_materializations(&mut value, true)?;
        let envelope: Self = serde_json::from_value(value)
            .map_err(|error| err(format!("sharded CUDA artifact envelope: {error}")))?;
        if envelope.format_version != Self::FORMAT_VERSION {
            return Err(err("unsupported sharded CUDA artifact version"));
        }
        validate_candidate_free_plan(&envelope.plan)?;
        if envelope.fingerprint != plan_fingerprint(&envelope.plan)? {
            return Err(err("sharded CUDA artifact fingerprint mismatch"));
        }
        Ok(envelope.plan)
    }
}

fn plan_fingerprint(plan: &ShardedCudaPlan) -> Result<String, Error> {
    let canonical = serde_json::to_vec(&legacy_candidate_free_plan_value(plan)?)
        .map_err(|error| err(format!("sharded CUDA artifact canonicalize: {error}")))?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn legacy_candidate_free_plan_value(plan: &ShardedCudaPlan) -> Result<serde_json::Value, Error> {
    validate_candidate_free_plan(plan)?;
    let mut value = serde_json::to_value(plan)
        .map_err(|error| err(format!("sharded CUDA artifact canonicalize: {error}")))?;
    value
        .as_object_mut()
        .ok_or_else(|| err("sharded CUDA artifact plan must be an object"))?
        .remove("materializations");
    Ok(value)
}

fn reject_transaction_metadata(value: &serde_json::Value) -> Result<(), Error> {
    if !value.is_object() {
        return Err(err("sharded CUDA artifact must be an object"));
    }
    if contains_transaction_metadata(value) {
        return Err(err(
            "candidate transaction metadata requires a newer artifact version",
        ));
    }
    Ok(())
}

fn reject_materialization_metadata(value: &serde_json::Value) -> Result<(), Error> {
    if !value.is_object() {
        return Err(err("sharded CUDA artifact must be an object"));
    }
    if contains_materialization_metadata(value) {
        return Err(err(
            "collective result materialization metadata requires the v3 artifact envelope",
        ));
    }
    Ok(())
}

fn reject_downstream_output_metadata(value: &serde_json::Value) -> Result<(), Error> {
    if !value.is_object() {
        return Err(err("sharded CUDA artifact must be an object"));
    }
    if contains_downstream_output_metadata(value) {
        return Err(err(
            "transaction-owned downstream output metadata requires the v5 artifact envelope",
        ));
    }
    Ok(())
}

/// Earlier envelopes predate the required v3 field.  We add an explicit empty
/// value only after rejecting materialization metadata, instead of using a
/// serde default that could silently reinterpret an untrusted artifact.
fn inject_empty_legacy_materializations(
    value: &mut serde_json::Value,
    envelope: bool,
) -> Result<(), Error> {
    let plan = if envelope {
        value
            .get_mut("plan")
            .ok_or_else(|| err("sharded CUDA artifact plan is absent"))?
    } else {
        value
    };
    let object = plan
        .as_object_mut()
        .ok_or_else(|| err("sharded CUDA artifact plan must be an object"))?;
    if object.contains_key("materializations") {
        return Err(err(
            "legacy sharded CUDA artifact unexpectedly contains materializations",
        ));
    }
    object.insert("materializations".into(), serde_json::Value::Array(vec![]));
    Ok(())
}

fn reject_unknown_envelope_fields(value: &serde_json::Value, fields: &[&str]) -> Result<(), Error> {
    let object = value
        .as_object()
        .ok_or_else(|| err("sharded CUDA artifact envelope must be an object"))?;
    if object.keys().any(|key| !fields.contains(&key.as_str())) {
        return Err(err("sharded CUDA artifact envelope has unknown fields"));
    }
    Ok(())
}

fn reject_unknown_plan_fields(value: &serde_json::Value) -> Result<(), Error> {
    reject_unknown_envelope_fields(
        value,
        &[
            "graph_id",
            "layout_key",
            "bindings",
            "stages",
            "diagnostics",
            "cache_key",
            "materializations",
        ],
    )
}

fn contains_transaction_metadata(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => object.iter().any(|(key, value)| {
            key.contains("candidate")
                || key.contains("commit")
                || contains_transaction_metadata(value)
        }),
        serde_json::Value::Array(values) => values.iter().any(contains_transaction_metadata),
        _ => false,
    }
}

fn contains_materialization_metadata(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => object.iter().any(|(key, value)| {
            // `external_materializations` is a released local-stage ABI field,
            // not v3/v4 collective metadata. Only the exact logical field is
            // version-gated before legacy deserialization.
            key == "materializations" || contains_materialization_metadata(value)
        }),
        serde_json::Value::Array(values) => values.iter().any(contains_materialization_metadata),
        _ => false,
    }
}

fn contains_downstream_output_metadata(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => object.iter().any(|(key, value)| {
            key == "graph_result_bindings"
                || key == "consumer_abis"
                || key == "outputs"
                || key == "output_commits"
                || contains_downstream_output_metadata(value)
        }),
        serde_json::Value::Array(values) => values.iter().any(contains_downstream_output_metadata),
        _ => false,
    }
}

fn validate_candidate_free_plan(plan: &ShardedCudaPlan) -> Result<(), Error> {
    if !plan.materializations.is_empty() {
        return Err(err(
            "collective result materializations require the v3 artifact envelope",
        ));
    }
    if plan.cache_key.is_empty() {
        return Err(err("sharded CUDA artifact cache key is empty"));
    }
    let mut ids = BTreeSet::new();
    for (expected, stage) in plan.stages.iter().enumerate() {
        let (id, dependencies) = match stage {
            CudaPlanStage::Local {
                id, dependencies, ..
            }
            | CudaPlanStage::Collective {
                id, dependencies, ..
            }
            | CudaPlanStage::Transfer {
                id, dependencies, ..
            } => (*id, dependencies),
        };
        if id != expected
            || !ids.insert(id)
            || dependencies.iter().any(|dependency| *dependency >= id)
        {
            return Err(err(
                "sharded CUDA artifact stage order or dependency is noncanonical",
            ));
        }
    }
    Ok(())
}

fn transaction_fingerprint(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
) -> Result<String, Error> {
    let canonical =
        serde_json::to_vec(&(legacy_candidate_free_plan_value(plan)?, candidates, commits))
            .map_err(|error| err(format!("sharded CUDA transaction canonicalize: {error}")))?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn validate_transaction_plan(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
) -> Result<(), Error> {
    validate_candidate_free_plan(plan)?;
    if candidates.is_empty() || commits.is_empty() {
        return Err(err("transaction artifact requires candidates and commits"));
    }
    let mut candidate_keys = BTreeSet::new();
    let mut candidate_sources = BTreeSet::new();
    for candidate in candidates {
        let Some(CudaPlanStage::Collective { buffers, .. }) = plan.stages.get(candidate.stage)
        else {
            return Err(err("candidate stage is not a collective"));
        };
        let source = *buffers
            .get(candidate.rank)
            .ok_or_else(|| err("candidate rank is outside collective buffers"))?;
        if source != candidate.source_buffer
            || candidate.candidate_buffer == candidate.source_buffer
            || candidate.bytes
                != candidate
                    .shape
                    .numel()?
                    .checked_mul(candidate.dtype.itemsize())
                    .ok_or_else(|| err("candidate byte overflow"))?
            || !candidate_keys.insert((candidate.rank, candidate.candidate_buffer))
            || !candidate_sources.insert((candidate.stage, candidate.rank, candidate.source_buffer))
        {
            return Err(err("candidate descriptor is duplicate or inconsistent"));
        }
    }
    let mut targets = BTreeSet::new();
    for (expected, commit) in commits.iter().enumerate() {
        if commit.order != expected
            || !candidate_keys.contains(&(commit.rank, commit.candidate_buffer))
            || commit.candidate_buffer == commit.target_buffer
            || !targets.insert((commit.rank, commit.target_buffer))
        {
            return Err(err(
                "transaction commit order, source, or target is invalid",
            ));
        }
    }
    if commits.len() != candidates.len() {
        return Err(err("transaction commits do not cover every candidate"));
    }
    Ok(())
}

fn materialization_fingerprint(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
) -> Result<String, Error> {
    let canonical = serde_json::to_vec(&(
        CollectiveMaterializationArtifact::FORMAT_VERSION,
        plan,
        candidates,
        commits,
    ))
    .map_err(|error| {
        err(format!(
            "sharded CUDA materialization canonicalize: {error}"
        ))
    })?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn lifecycle_materialization_fingerprint(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
    materializations: &[CollectiveLifecycleMaterialization],
) -> Result<String, Error> {
    let canonical = serde_json::to_vec(&(
        CollectiveLifecycleMaterializationArtifact::FORMAT_VERSION,
        plan,
        candidates,
        commits,
        materializations,
    ))
    .map_err(|error| {
        err(format!(
            "sharded CUDA lifecycle materialization canonicalize: {error}"
        ))
    })?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn downstream_output_fingerprint(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
    materializations: &[CollectiveLifecycleMaterialization],
    outputs: &[CollectiveDownstreamOutputDescriptor],
    output_commits: &[CollectiveDownstreamOutputCommitRecord],
) -> Result<String, Error> {
    let canonical = serde_json::to_vec(&(
        CollectiveDownstreamOutputArtifact::FORMAT_VERSION,
        plan,
        candidates,
        commits,
        materializations,
        outputs,
        output_commits,
    ))
    .map_err(|error| {
        err(format!(
            "sharded CUDA downstream output canonicalize: {error}"
        ))
    })?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn graph_unary_output_fingerprint(
    plan: &ShardedCudaPlan,
    components: &DownstreamOutputArtifactComponentRefs<'_>,
) -> Result<String, Error> {
    let DownstreamOutputArtifactComponentRefs {
        candidates,
        commits,
        materializations,
        graph_result_bindings,
        consumer_abis,
        outputs,
        output_commits,
    } = *components;
    let canonical = serde_json::to_vec(&(
        CollectiveGraphUnaryOutputArtifact::FORMAT_VERSION,
        plan,
        candidates,
        commits,
        materializations,
        graph_result_bindings,
        consumer_abis,
        outputs,
        output_commits,
    ))
    .map_err(|error| {
        err(format!(
            "sharded CUDA graph unary output canonicalize: {error}"
        ))
    })?;
    let hash = canonical
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x1000_0000_01b3)
        });
    Ok(format!("fnv1a64:{hash:016x}"))
}

fn validate_materialization_plan(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
) -> Result<(), Error> {
    let mut transaction_plan = plan.clone();
    transaction_plan.materializations.clear();
    validate_transaction_plan(&transaction_plan, candidates, commits)?;
    if plan.materializations.is_empty() {
        return Err(err("v3 materialization artifact requires materializations"));
    }
    let mut keys = BTreeSet::new();
    let mut candidate_keys = BTreeSet::new();
    for materialization in &plan.materializations {
        let binding = plan
            .bindings
            .get(materialization.rank)
            .ok_or_else(|| err("materialization rank is outside bindings"))?;
        let candidate = candidates
            .iter()
            .find(|candidate| {
                candidate.rank == materialization.rank
                    && candidate.candidate_buffer == materialization.candidate_buffer
            })
            .ok_or_else(|| err("materialization candidate linkage is absent"))?;
        let Some(CudaPlanStage::Collective { .. }) =
            plan.stages.get(materialization.producer_stage)
        else {
            return Err(err("materialization producer is not a collective stage"));
        };
        if materialization.boundary_key.is_empty()
            || materialization.device != binding.0
            || materialization.owner_identity != binding.1
            || materialization.dtype != candidate.dtype
            || materialization.shape != candidate.shape
            || materialization.bytes != candidate.bytes
            || materialization.producer_stage != candidate.stage
            // The v3 bridge only records terminal materializations. A future
            // local consumer must introduce its own execution vertical rather
            // than being accidentally accepted here.
            || materialization.first_consumer != plan.stages.len()
            || materialization.last_consumer != plan.stages.len()
            || !keys.insert((materialization.boundary_key.as_str(), materialization.rank))
            || !candidate_keys.insert((materialization.rank, materialization.candidate_buffer))
        {
            return Err(err(
                "materialization descriptor is duplicate or inconsistent",
            ));
        }
    }
    let expected = candidates
        .iter()
        .map(|candidate| (candidate.rank, candidate.candidate_buffer))
        .collect::<BTreeSet<_>>();
    if keys.len() != candidates.len() || candidate_keys != expected {
        return Err(err("materialization rank coverage is incomplete"));
    }
    Ok(())
}

/// V4 keeps terminal materializations out of the plan body. This makes the
/// lifecycle discriminator mandatory and prevents v3 terminal metadata from
/// being silently reinterpreted as a downstream alias.
fn validate_lifecycle_materialization_plan(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
    materializations: &[CollectiveLifecycleMaterialization],
) -> Result<(), Error> {
    validate_lifecycle_materialization_components(plan, candidates, commits, materializations, true)
}

fn validate_lifecycle_materialization_components(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
    materializations: &[CollectiveLifecycleMaterialization],
    require_shared_boundary_lifetime: bool,
) -> Result<(), Error> {
    validate_transaction_plan(plan, candidates, commits)?;
    if !plan.materializations.is_empty() {
        return Err(err(
            "v4 lifecycle artifact must not contain v3 materializations",
        ));
    }
    if materializations.is_empty() {
        return Err(err("v4 lifecycle artifact requires materializations"));
    }
    let mut bindings = BTreeSet::new();
    let mut candidate_keys = BTreeSet::new();
    let mut consumer_buffers = BTreeSet::new();
    let mut downstream_boundaries = BTreeMap::new();
    for record in materializations {
        let binding = &record.materialization;
        let owner = plan
            .bindings
            .get(binding.rank)
            .ok_or_else(|| err("v4 materialization rank is outside bindings"))?;
        let candidate = candidates
            .iter()
            .find(|candidate| {
                candidate.rank == binding.rank
                    && candidate.candidate_buffer == binding.candidate_buffer
            })
            .ok_or_else(|| err("v4 materialization candidate linkage is absent"))?;
        let Some(CudaPlanStage::Collective { .. }) = plan.stages.get(binding.producer_stage) else {
            return Err(err("v4 materialization producer is not a collective stage"));
        };
        if binding.boundary_key.is_empty()
            || binding.device != owner.0
            || binding.owner_identity != owner.1
            || binding.dtype != candidate.dtype
            || binding.shape != candidate.shape
            || binding.bytes != candidate.bytes
            || binding.producer_stage != candidate.stage
            || !bindings.insert((binding.boundary_key.as_str(), binding.rank))
            || !candidate_keys.insert((binding.rank, binding.candidate_buffer))
        {
            return Err(err(
                "v4 materialization binding is duplicate or inconsistent",
            ));
        }
        match &record.lifecycle {
            CollectiveMaterializationLifecycle::Terminal => {
                if !record.consumers.is_empty()
                    || binding.first_consumer != plan.stages.len()
                    || binding.last_consumer != plan.stages.len()
                {
                    return Err(err("terminal v4 materialization has downstream fields"));
                }
            }
            CollectiveMaterializationLifecycle::Downstream {
                first_consumer_stage,
                lifetime_end_stage,
            } => {
                if *first_consumer_stage <= binding.producer_stage
                    || *first_consumer_stage >= plan.stages.len()
                    || *lifetime_end_stage < *first_consumer_stage
                    || *lifetime_end_stage >= plan.stages.len()
                    || binding.first_consumer != *first_consumer_stage
                    || binding.last_consumer != *lifetime_end_stage
                    || record.consumers.len() != 1
                {
                    return Err(err("downstream v4 materialization lifecycle is invalid"));
                }
                let consumer = &record.consumers[0];
                if require_shared_boundary_lifetime
                    && let Some((first, last)) = downstream_boundaries.insert(
                        binding.boundary_key.as_str(),
                        (*first_consumer_stage, *lifetime_end_stage),
                    )
                    && (first != *first_consumer_stage || last != *lifetime_end_stage)
                {
                    return Err(err(
                        "downstream v4 boundary has inconsistent rank-local lifetime",
                    ));
                }
                let Some(CudaPlanStage::Local {
                    id,
                    inputs,
                    dependencies,
                    ..
                }) = plan.stages.get(*first_consumer_stage)
                else {
                    return Err(err("downstream v4 consumer is not a local stage"));
                };
                if *id != *first_consumer_stage
                    || consumer.rank != binding.rank
                    || consumer.consumer_stage != *first_consumer_stage
                    || consumer.consumer_buffer != binding.candidate_buffer
                    || consumer.device != binding.device
                    || consumer.owner_identity != binding.owner_identity
                    || consumer.dtype != binding.dtype
                    || consumer.shape != binding.shape
                    || consumer.bytes != binding.bytes
                    || !inputs.contains(&consumer.consumer_buffer)
                    || !dependencies.contains(&binding.producer_stage)
                    || !consumer_buffers.insert((
                        consumer.rank,
                        consumer.consumer_stage,
                        consumer.consumer_buffer,
                    ))
                {
                    return Err(err("downstream v4 consumer descriptor is inconsistent"));
                }
            }
        }
    }
    let expected = candidates
        .iter()
        .map(|candidate| (candidate.rank, candidate.candidate_buffer))
        .collect::<BTreeSet<_>>();
    if bindings.len() != candidates.len() || candidate_keys != expected {
        return Err(err("v4 materialization rank coverage is incomplete"));
    }
    Ok(())
}

/// V5 adds transaction-owned downstream outputs while retaining the exact v4
/// lifecycle proof.  It validates the complete future commit table before any
/// concrete owner rebinding, cache insertion, allocation, or driver work.
fn validate_downstream_output_plan(
    plan: &ShardedCudaPlan,
    candidates: &[CollectiveCandidateDescriptor],
    commits: &[CollectiveCommitRecord],
    materializations: &[CollectiveLifecycleMaterialization],
    outputs: &[CollectiveDownstreamOutputDescriptor],
    output_commits: &[CollectiveDownstreamOutputCommitRecord],
) -> Result<(), Error> {
    validate_lifecycle_materialization_plan(plan, candidates, commits, materializations)?;
    if outputs.is_empty() || outputs.len() != output_commits.len() {
        return Err(err("v5 downstream output coverage is invalid"));
    }
    let mut output_keys = BTreeSet::new();
    let mut destinations = BTreeSet::new();
    let mut source_keys = BTreeSet::new();
    for output in outputs {
        let owner = plan
            .bindings
            .get(output.rank)
            .ok_or_else(|| err("v5 downstream output rank is outside bindings"))?;
        let materialization = materializations
            .iter()
            .find(|record| {
                record.materialization.rank == output.rank
                    && record.materialization.candidate_buffer == output.source_candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream output provenance is absent"))?;
        let CollectiveMaterializationLifecycle::Downstream {
            first_consumer_stage,
            lifetime_end_stage,
        } = &materialization.lifecycle
        else {
            return Err(err(
                "v5 downstream output requires downstream materialization",
            ));
        };
        let Some(CudaPlanStage::Local {
            id,
            device,
            owner_identity,
            output: declared_output,
            dependencies,
            ..
        }) = plan.stages.get(output.consumer_stage)
        else {
            return Err(err("v5 downstream output consumer is not local"));
        };
        if output.consumer_stage != *first_consumer_stage
            || output.first_stage != *first_consumer_stage
            || output.last_stage != *lifetime_end_stage
            || output.last_stage < output.first_stage
            || *id != output.consumer_stage
            || output.device != *device
            || output.owner_identity != *owner_identity
            || output.device != owner.0
            || output.owner_identity != owner.1
            || output.dtype != materialization.materialization.dtype
            || output.shape != materialization.materialization.shape
            || output.bytes != materialization.materialization.bytes
            || output.bytes
                != output
                    .shape
                    .numel()?
                    .checked_mul(output.dtype.itemsize())
                    .ok_or_else(|| err("v5 downstream output byte overflow"))?
            || output.output_candidate_buffer == output.source_candidate_buffer
            || output.output_candidate_buffer == output.destination_buffer
            || *declared_output != output.destination_buffer
            || !dependencies.contains(&materialization.materialization.producer_stage)
            || !output_keys.insert((output.rank, output.output_candidate_buffer))
            || !destinations.insert((output.rank, output.destination_buffer))
            || !source_keys.insert((output.rank, output.source_candidate_buffer))
        {
            return Err(err(
                "v5 downstream output descriptor is duplicate or inconsistent",
            ));
        }
    }
    let mut committed = BTreeSet::new();
    for (expected, commit) in output_commits.iter().enumerate() {
        let output = outputs
            .iter()
            .find(|output| {
                output.rank == commit.rank
                    && output.output_candidate_buffer == commit.output_candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream output commit source is absent"))?;
        if commit.order != expected
            || commit.destination_buffer != output.destination_buffer
            || !committed.insert((commit.rank, commit.output_candidate_buffer))
        {
            return Err(err(
                "v5 downstream output commit is duplicate or inconsistent",
            ));
        }
    }
    if committed.len() != output_keys.len() || source_keys.len() != materializations.len() {
        return Err(err(
            "v5 downstream output commits or provenance are incomplete",
        ));
    }
    Ok(())
}

fn validate_graph_unary_output_plan(
    plan: &ShardedCudaPlan,
    components: &DownstreamOutputArtifactComponentRefs<'_>,
) -> Result<(), Error> {
    let DownstreamOutputArtifactComponentRefs {
        candidates,
        commits,
        materializations,
        graph_result_bindings,
        consumer_abis,
        outputs,
        output_commits,
    } = *components;
    let lifecycle_plan = downstream_output_lifecycle_projection(plan, graph_result_bindings)?;
    validate_lifecycle_materialization_components(
        &lifecycle_plan,
        candidates,
        commits,
        materializations,
        false,
    )?;
    if graph_result_bindings.len() != materializations.len() {
        return Err(err("v5 graph result binding coverage is incomplete"));
    }
    let mut graph_keys = BTreeSet::new();
    for binding in graph_result_bindings {
        binding.validate()?;
        let materialization = materializations
            .iter()
            .find(|record| {
                record.materialization.rank == binding.rank
                    && record.materialization.candidate_buffer == binding.candidate_buffer
            })
            .ok_or_else(|| err("v5 graph result binding candidate linkage is absent"))?;
        if binding.replicated_result != materialization.materialization.replicated_result
            || binding.device != materialization.materialization.device
            || binding.owner_identity != materialization.materialization.owner_identity
            || binding.dtype != materialization.materialization.dtype
            || binding.shape != materialization.materialization.shape
            || binding.bytes != materialization.materialization.bytes
            || binding.first_consumer_stage != materialization.materialization.first_consumer
            || binding.lifetime_end_stage != materialization.materialization.last_consumer
            || !graph_keys.insert(binding.canonical_key())
        {
            return Err(err("v5 graph result binding is duplicate or inconsistent"));
        }
    }
    let expected_graph_keys = materializations
        .iter()
        .map(|record| {
            (
                record.materialization.replicated_result,
                record.materialization.rank,
                record.materialization.candidate_buffer,
            )
        })
        .collect::<BTreeSet<_>>();
    let actual_graph_keys = graph_result_bindings
        .iter()
        .map(|binding| {
            (
                binding.replicated_result,
                binding.rank,
                binding.candidate_buffer,
            )
        })
        .collect::<BTreeSet<_>>();
    if actual_graph_keys != expected_graph_keys {
        return Err(err("v5 graph result binding rank coverage is incomplete"));
    }
    if consumer_abis.len() != materializations.len() {
        return Err(err("v5 downstream consumer ABI coverage is incomplete"));
    }
    let mut consumer_keys = BTreeSet::new();
    let mut local_inputs = BTreeSet::new();
    let mut output_candidates = BTreeSet::new();
    let mut previous_consumer_key = None;
    for abi in consumer_abis {
        abi.validate()?;
        let key = abi.canonical_key();
        if let Some(previous) = previous_consumer_key
            && previous >= key
        {
            return Err(err("v5 downstream consumer ABI ordering is not canonical"));
        }
        previous_consumer_key = Some(key);
        let binding = graph_result_bindings
            .iter()
            .find(|binding| {
                binding.rank == abi.rank && binding.candidate_buffer == abi.candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream consumer ABI graph binding is absent"))?;
        let materialization = materializations
            .iter()
            .find(|record| {
                record.materialization.rank == abi.rank
                    && record.materialization.candidate_buffer == abi.candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream consumer ABI materialization is absent"))?;
        let output = outputs
            .iter()
            .find(|output| {
                output.rank == abi.rank && output.source_candidate_buffer == abi.candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream consumer ABI output linkage is absent"))?;
        if abi.replicated_result != binding.replicated_result
            || abi.local_input_buffer != binding.local_input_buffer
            || abi.consumer_stage != binding.first_consumer_stage
            || abi.lifetime_end_stage != binding.lifetime_end_stage
            || abi.replicated_result != materialization.materialization.replicated_result
            || abi.device != materialization.materialization.device
            || abi.owner_identity != materialization.materialization.owner_identity
            || abi.dtype != materialization.materialization.dtype
            || abi.shape != materialization.materialization.shape
            || abi.bytes != materialization.materialization.bytes
            || abi.consumer_stage != materialization.materialization.first_consumer
            || abi.lifetime_end_stage != materialization.materialization.last_consumer
            || abi.output_candidate_buffer != output.output_candidate_buffer
            || abi.consumer_stage != output.consumer_stage
            || abi.lifetime_end_stage != output.last_stage
            || abi.device != output.device
            || abi.owner_identity != output.owner_identity
            || abi.dtype != output.dtype
            || abi.shape != output.shape
            || abi.bytes != output.bytes
            || !consumer_keys.insert((abi.replicated_result, abi.rank, abi.candidate_buffer))
            || !local_inputs.insert((abi.rank, abi.local_input_buffer))
            || !output_candidates.insert((abi.rank, abi.output_candidate_buffer))
        {
            return Err(err(
                "v5 downstream consumer ABI is duplicate or inconsistent",
            ));
        }
    }
    let expected_consumer_keys = materializations
        .iter()
        .map(|record| {
            (
                record.materialization.replicated_result,
                record.materialization.rank,
                record.materialization.candidate_buffer,
            )
        })
        .collect::<BTreeSet<_>>();
    if consumer_keys != expected_consumer_keys {
        return Err(err(
            "v5 downstream consumer ABI rank coverage is incomplete",
        ));
    }
    if outputs.is_empty() || outputs.len() != output_commits.len() {
        return Err(err("v5 downstream output coverage is invalid"));
    }
    let mut output_keys = BTreeSet::new();
    let mut destinations = BTreeSet::new();
    let mut source_keys = BTreeSet::new();
    for output in outputs {
        let owner = plan
            .bindings
            .get(output.rank)
            .ok_or_else(|| err("v5 downstream output rank is outside bindings"))?;
        let materialization = materializations
            .iter()
            .find(|record| {
                record.materialization.rank == output.rank
                    && record.materialization.candidate_buffer == output.source_candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream output provenance is absent"))?;
        let CollectiveMaterializationLifecycle::Downstream {
            first_consumer_stage,
            lifetime_end_stage,
        } = &materialization.lifecycle
        else {
            return Err(err(
                "v5 downstream output requires downstream materialization",
            ));
        };
        let Some(CudaPlanStage::Local {
            id,
            device,
            owner_identity,
            output: declared_output,
            dependencies,
            ..
        }) = plan.stages.get(output.consumer_stage)
        else {
            return Err(err("v5 downstream output consumer is not local"));
        };
        if output.consumer_stage != *first_consumer_stage
            || output.first_stage != *first_consumer_stage
            || output.last_stage != *lifetime_end_stage
            || output.last_stage < output.first_stage
            || *id != output.consumer_stage
            || output.device != *device
            || output.owner_identity != *owner_identity
            || output.device != owner.0
            || output.owner_identity != owner.1
            || output.dtype != materialization.materialization.dtype
            || output.shape != materialization.materialization.shape
            || output.bytes != materialization.materialization.bytes
            || output.bytes
                != output
                    .shape
                    .numel()?
                    .checked_mul(output.dtype.itemsize())
                    .ok_or_else(|| err("v5 downstream output byte overflow"))?
            || output.output_candidate_buffer == output.source_candidate_buffer
            || output.output_candidate_buffer == output.destination_buffer
            || *declared_output != output.destination_buffer
            || !dependencies.contains(&materialization.materialization.producer_stage)
            || !output_keys.insert((output.rank, output.output_candidate_buffer))
            || !destinations.insert((output.rank, output.destination_buffer))
            || !source_keys.insert((output.rank, output.source_candidate_buffer))
        {
            return Err(err(
                "v5 downstream output descriptor is duplicate or inconsistent",
            ));
        }
    }
    let mut committed = BTreeSet::new();
    for (expected, commit) in output_commits.iter().enumerate() {
        let output = outputs
            .iter()
            .find(|output| {
                output.rank == commit.rank
                    && output.output_candidate_buffer == commit.output_candidate_buffer
            })
            .ok_or_else(|| err("v5 downstream output commit source is absent"))?;
        if commit.order != expected
            || commit.destination_buffer != output.destination_buffer
            || !committed.insert((commit.rank, commit.output_candidate_buffer))
        {
            return Err(err(
                "v5 downstream output commit is duplicate or inconsistent",
            ));
        }
    }
    let expected_sources = materializations
        .iter()
        .map(|record| {
            (
                record.materialization.rank,
                record.materialization.candidate_buffer,
            )
        })
        .collect::<BTreeSet<_>>();
    if committed.len() != output_keys.len()
        || source_keys != expected_sources
        || output_candidates != output_keys
    {
        return Err(err(
            "v5 downstream output commits or provenance are incomplete",
        ));
    }
    Ok(())
}

/// The graph-unary envelope retains the rendered local-stage ABI while
/// projecting its result binding to the candidate ABI required by the released
/// v4 lifecycle proof.
fn downstream_output_lifecycle_projection(
    plan: &ShardedCudaPlan,
    graph_result_bindings: &[CollectiveGraphResultBinding],
) -> Result<ShardedCudaPlan, Error> {
    let mut projection = plan.clone();
    for binding in graph_result_bindings {
        let Some(CudaPlanStage::Local { inputs, .. }) =
            projection.stages.get_mut(binding.first_consumer_stage)
        else {
            return Err(err("v5 graph result binding consumer is not a local stage"));
        };
        if inputs.contains(&binding.candidate_buffer) {
            continue;
        }
        let Some(input) = inputs
            .iter_mut()
            .find(|input| **input == binding.local_input_buffer)
        else {
            return Err(err("v5 graph result binding local input is absent"));
        };
        *input = binding.candidate_buffer;
    }
    Ok(projection)
}
/// Non-serializable execution companion retaining exact PTX ABI artifacts and primary owners.
///
/// `ShardedCudaPlan` is the data-only replay record. This companion deliberately
/// has no capture/replay serialization path: primary contexts, streams, modules,
/// leases, and peer-access state must be rebound and preflighted by the caller.
pub struct ExecutableShardedCudaPlan {
    pub logical: ShardedCudaPlan,
    pub owners: Vec<PrimaryContext>,
    pub kernels: Vec<Option<RenderedPtx>>,
    pub buffers: Vec<ExecutableBuffer>,
}
/// A v2 artifact rebound to concrete graph owners. Runtime preflight still
/// verifies every candidate source and commit target before allocating a lease.
pub struct ExecutableCollectiveTransaction {
    pub plan: ExecutableShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
}
/// A v3 artifact rebound to concrete owners. `CollectiveResult` entries are
/// logical lifetime descriptors only: downstream execution remains rejected
/// until a later stage can consume them without exposing an alias.
pub struct ExecutableCollectiveMaterialization {
    pub plan: ExecutableShardedCudaPlan,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveResultMaterialization>,
}
/// A v4 artifact rebound to concrete owners without rendering or allocating
/// CUDA resources.  `Downstream` entries deliberately stop here: this records
/// the exact future ABI while the executor remains fail-closed.
pub struct ExecutableCollectiveLifecycleMaterialization {
    pub logical: ShardedCudaPlan,
    pub owners: Vec<PrimaryContext>,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
    pub buffers: Vec<ExecutableBuffer>,
}
/// A v5 artifact rebound to concrete owners without allocation or rendering.
/// The output candidate and commit table are immutable execution inputs; a
/// later local-stage vertical must explicitly authorize their launch path.
pub struct ExecutableCollectiveDownstreamOutput {
    pub logical: ShardedCudaPlan,
    pub owners: Vec<PrimaryContext>,
    pub candidates: Vec<CollectiveCandidateDescriptor>,
    pub commits: Vec<CollectiveCommitRecord>,
    pub materializations: Vec<CollectiveLifecycleMaterialization>,
    pub outputs: Vec<CollectiveDownstreamOutputDescriptor>,
    pub output_commits: Vec<CollectiveDownstreamOutputCommitRecord>,
    pub buffers: Vec<ExecutableBuffer>,
}

/// Dedicated executable companion for the graph-unary envelope. The released
/// v5 companion remains exactly source-compatible for external struct literals.
pub struct ExecutableCollectiveGraphUnaryOutput {
    pub downstream: ExecutableCollectiveDownstreamOutput,
    pub graph_result_bindings: Vec<CollectiveGraphResultBinding>,
    pub consumer_abis: Vec<CollectiveDownstreamConsumerAbi>,
    /// Graph-backed rank-local unary nodes retained only after the strict
    /// constructor has proven their exact correspondence to the artifact.
    pub(crate) consumer_nodes: Vec<NodeId>,
    /// Exact closed unary operation identity retained with the non-serializable
    /// executable companion. The serialized artifact cache key is validated
    /// against this identity before graph rebinding.
    pub(crate) unary_op: Option<GraphBackedDownstreamUnary>,
    /// Per-rank graph-schedule ABI substitutions retained for the dedicated
    /// execution entrypoint; generic substitution execution stays closed.
    pub(crate) substitutions: Vec<BufferSubstitution>,
    /// Retained only by the graph-aware unary constructor; execution must still
    /// rehydrate the exact graph schedules from these checked owner bindings.
    pub(crate) unary_bindings: Option<Vec<CudaPlanBinding>>,
}

impl std::ops::Deref for ExecutableCollectiveGraphUnaryOutput {
    type Target = ExecutableCollectiveDownstreamOutput;

    fn deref(&self) -> &Self::Target {
        &self.downstream
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutableBufferRole {
    External,
    Output,
    CollectiveResult,
    TransactionOutput,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableBuffer {
    pub rank: usize,
    pub device: SemanticDeviceId,
    pub owner_identity: usize,
    pub buffer: u64,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: usize,
    pub producer: Option<usize>,
    pub consumers: Vec<usize>,
    pub first_stage: usize,
    pub last_stage: usize,
    pub role: ExecutableBufferRole,
}
impl ExecutableShardedCudaPlan {
    /// Pure preflight of the canonical map and exact transfer endpoints; it has no CUDA side effects.
    pub fn validate(&self) -> Result<(), Error> {
        if self.logical.materializations.iter().any(|binding| {
            !self.buffers.iter().any(|buffer| {
                buffer.rank == binding.rank
                    && buffer.buffer == binding.candidate_buffer
                    && matches!(buffer.role, ExecutableBufferRole::CollectiveResult)
                    && buffer.device == binding.device
                    && buffer.owner_identity == binding.owner_identity
                    && buffer.dtype == binding.dtype
                    && buffer.shape == binding.shape
                    && buffer.bytes == binding.bytes
                    && buffer.producer == Some(binding.producer_stage)
                    && buffer.first_stage == binding.producer_stage
                    && buffer.last_stage == binding.last_consumer
                    && buffer.consumers == vec![binding.first_consumer]
            })
        }) {
            return Err(err(
                "collective result materialization is absent from executable map",
            ));
        }
        for stage in &self.logical.stages {
            if let CudaPlanStage::Collective { plan, buffers, .. } = stage {
                plan.validate()?;
                if buffers.len() != self.owners.len()
                    || plan.request.group.devices().len() != self.owners.len()
                    || plan.request.input_lengths.len() != buffers.len()
                {
                    return Err(err("collective buffer/group arity mismatch"));
                }
                for (rank, &buffer) in buffers.iter().enumerate() {
                    let descriptor = self
                        .buffers
                        .iter()
                        .find(|entry| entry.rank == rank && entry.buffer == buffer)
                        .ok_or_else(|| err("collective buffer is absent from canonical map"))?;
                    if descriptor.dtype != plan.request.dtype
                        || descriptor.shape.numel()? != plan.request.input_lengths[rank]
                        || descriptor.bytes
                            != plan.request.input_lengths[rank]
                                .checked_mul(plan.request.dtype.itemsize())
                                .ok_or_else(|| err("collective buffer byte overflow"))?
                    {
                        return Err(err("collective buffer descriptor mismatch"));
                    }
                }
            }
        }
        if self.kernels.len() != self.logical.stages.len() {
            return Err(err(
                "retained PTX artifacts do not match canonical stage count",
            ));
        }
        let mut canonical_buffers = BTreeSet::new();
        for buffer in &self.buffers {
            if !canonical_buffers.insert((buffer.rank, buffer.buffer)) {
                return Err(err("duplicate canonical executable buffer identity"));
            }
            let owner = self
                .owners
                .get(buffer.rank)
                .ok_or_else(|| err("canonical buffer rank exceeds retained owners"))?;
            if buffer.owner_identity != owner.identity() {
                return Err(err("canonical buffer owner does not match retained owner"));
            }
        }
        for (stage_index, stage) in self.logical.stages.iter().enumerate() {
            if let CudaPlanStage::Transfer { routes, .. } = stage {
                let mut destination_coverage = BTreeMap::<(usize, u64), Vec<(usize, usize)>>::new();
                for route in routes {
                    let source = self
                        .buffers
                        .iter()
                        .find(|buffer| {
                            buffer.rank == route.source_rank && buffer.buffer == route.source_buffer
                        })
                        .ok_or_else(|| {
                            err("transfer source buffer is absent from canonical map")
                        })?;
                    let destination = self
                        .buffers
                        .iter()
                        .find(|buffer| {
                            buffer.rank == route.destination_rank
                                && buffer.buffer == route.destination_buffer
                        })
                        .ok_or_else(|| {
                            err("transfer destination buffer is absent from canonical map")
                        })?;
                    if source.device != route.source_device
                        || destination.device != route.destination_device
                        || source.dtype != route.dtype
                        || destination.dtype != route.dtype
                    {
                        return Err(err("transfer route owner/device/dtype mismatch"));
                    }
                    let source_end = route
                        .source_element_offset
                        .checked_mul(route.dtype.itemsize())
                        .and_then(|x| x.checked_add(route.bytes))
                        .ok_or_else(|| err("transfer source range overflow"))?;
                    let destination_end = route
                        .destination_element_offset
                        .checked_mul(route.dtype.itemsize())
                        .and_then(|x| x.checked_add(route.bytes))
                        .ok_or_else(|| err("transfer destination range overflow"))?;
                    if source_end > source.bytes
                        || destination_end > destination.bytes
                        || route.bytes
                            != route
                                .elements
                                .checked_mul(route.dtype.itemsize())
                                .ok_or_else(|| err("transfer byte overflow"))?
                    {
                        return Err(err("transfer range exceeds canonical buffer"));
                    }
                    destination_coverage
                        .entry((route.destination_rank, route.destination_buffer))
                        .or_default()
                        .push((
                            route
                                .destination_element_offset
                                .checked_mul(route.dtype.itemsize())
                                .ok_or_else(|| err("transfer destination range overflow"))?,
                            destination_end,
                        ));
                }
                for buffer in self.buffers.iter().filter(|buffer| {
                    buffer.producer == Some(stage_index)
                        && matches!(buffer.role, ExecutableBufferRole::Output)
                }) {
                    let key = (buffer.rank, buffer.buffer);
                    let ranges = destination_coverage
                        .get_mut(&key)
                        .ok_or_else(|| err("transfer output has no canonical destination route"))?;
                    ranges.sort_unstable();
                    if buffer.bytes == 0 {
                        if ranges.len() != 1 || ranges[0] != (0, 0) {
                            return Err(err("logical-zero transfer output has ambiguous routes"));
                        }
                        continue;
                    }
                    let mut cursor = 0;
                    for &(start, end) in ranges.iter() {
                        if start != cursor || end < start {
                            return Err(err("transfer routes do not exactly cover output buffer"));
                        }
                        cursor = end;
                    }
                    if cursor != buffer.bytes {
                        return Err(err("transfer routes do not exactly cover output buffer"));
                    }
                }
            }
        }
        Ok(())
    }
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
        value
            .trace()
            .validate_collective_provenance(group, value.nodes())
            .map_err(|error| err(error.to_string()))?;
        if value.trace().steps.iter().any(|step| {
            matches!(
                step.collective.as_ref().map(|boundary| &boundary.lifecycle),
                Some(CollectiveBoundaryLifecycle::Downstream { .. })
            )
        }) {
            return Err(err(
                "non-terminal collective provenance is representation-only; native execution is unsupported",
            ));
        }
        let terminal_collective = value
            .trace()
            .steps
            .last()
            .filter(|trace| trace.collective.is_some() || trace.action.contains("all-reduce"));
        if value
            .trace()
            .steps
            .iter()
            .take(value.trace().steps.len().saturating_sub(1))
            .any(|trace| trace.collective.is_some() || trace.action.contains("all-reduce"))
        {
            return Err(err(
                "Phase 3B2 supports one terminal all-reduce provenance step",
            ));
        }
        let execution_nodes = if let Some(trace) = terminal_collective {
            let inputs = trace
                .collective
                .as_ref()
                .map(|boundary| boundary.ordered_inputs.as_slice())
                .unwrap_or(trace.collective_inputs.as_slice());
            if inputs.len() != group.len() {
                return Err(err("collective provenance rank count mismatch"));
            }
            inputs
        } else {
            value.nodes()
        };
        let mut stages = Vec::new();
        let mut diagnostics = Vec::new();
        let mut previous = Vec::new();
        for (rank, node) in execution_nodes.iter().enumerate() {
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
                && let Err(error) =
                    PtxRenderer::new(binding.capability.sm()).and_then(|r| r.render(&item.kernel))
            {
                diagnostic = Some(CudaPlanDiagnostic::Unsupported {
                    node: node.index(),
                    reason: format!("PTX renderer: {error}"),
                });
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
                external_materializations: vec![],
                output: item
                    .map(|x| x.primary_output().id)
                    .unwrap_or(node.index() as u64),
                dependencies: previous.clone(),
                source_key: source_key.clone(),
                module_key: format!(
                    "owner:{}:sm{}:{source_key}",
                    owner_identity,
                    binding.capability.sm()
                ),
                diagnostic,
            });
            previous.push(id);
        }
        for trace in &value.trace().steps {
            if trace.collective.is_some() || trace.action.contains("all-reduce") {
                let plan = collective_plan(group, value.dtype(), graph.shape(execution_nodes[0])?)?;
                let id = stages.len();
                let buffers = stages
                    .iter()
                    .take(group.len())
                    .map(|stage| match stage {
                        CudaPlanStage::Local { output, .. } => Ok(*output),
                        _ => Err(err("collective local producer is absent")),
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                stages.push(CudaPlanStage::Collective {
                    id,
                    action: trace.action.to_string(),
                    plan,
                    buffers,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            } else if trace.action == "redistribute" || trace.action == "gather-movement" {
                let id = stages.len();
                if trace.routes.is_empty() {
                    return Err(err("redistribution trace has no concrete routes"));
                }
                let routes = trace
                    .routes
                    .iter()
                    .map(|route| {
                        let bytes = route
                            .elements
                            .checked_mul(value.dtype().itemsize())
                            .ok_or_else(|| err("redistribution byte overflow"))?;
                        Ok(CudaTransferRoute {
                            source_rank: route.source_rank,
                            source_device: route.source_device.clone(),
                            source_buffer: route.source_node.index() as u64,
                            source_element_offset: route.source_offset,
                            destination_rank: route.destination_rank,
                            destination_device: route.destination_device.clone(),
                            destination_buffer: route.destination_node.index() as u64,
                            destination_element_offset: route.destination_offset,
                            elements: route.elements,
                            bytes,
                            dtype: value.dtype(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                stages.push(CudaPlanStage::Transfer {
                    id,
                    action: trace.action.into(),
                    routes,
                    dependencies: previous.clone(),
                });
                previous = vec![id];
            }
        }
        let stage_identity = stages
            .iter()
            .map(|stage| match stage {
                CudaPlanStage::Local {
                    source_key, output, ..
                } => format!("local:{source_key}:{output}"),
                CudaPlanStage::Collective { plan, buffers, .. } => format!(
                    "collective:{}:{}",
                    plan.cache_key,
                    buffers
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                CudaPlanStage::Transfer { action, routes, .. } => {
                    format!("transfer:{action}:{}", routes.len())
                }
            })
            .collect::<Vec<_>>()
            .join("|");
        let cache_key = format!(
            "sharded-cuda-plan:v2:{}:{}:{stage_identity}",
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
            materializations: vec![],
        })
    }
    /// Builds the proven transfer-then-local executable composition directly
    /// from typed graph provenance.  The transfer destination is the only
    /// explicitly materialized computed input; no operation label or graph
    /// walk is used to discover substitutions.
    pub fn executable_fused(
        graph: &Graph,
        value: &ShardedGraphTensor,
        bindings: &[CudaPlanBinding],
    ) -> Result<ShardedCudaPlanComposition, Error> {
        if value.graph_id() != graph.id() {
            return Err(err("sharded tensor belongs to another graph"));
        }
        validate_bindings(value.layout().group(), bindings)?;
        // With one rank, redistribution is an identity layout transition.  The
        // normal retained local plan is the exact executable artifact; there
        // is no computed transfer output to substitute or materialize.
        if value.layout().group().len() == 1 {
            let local = Self::executable(graph, Self::build(graph, value, bindings)?, bindings)?;
            return Ok(ShardedCudaPlanComposition {
                plan: local,
                substitutions: vec![],
            });
        }
        let local_step = value
            .trace()
            .steps
            .last()
            .ok_or_else(|| err("fused plan requires a local provenance step"))?;
        if local_step.local_inputs.len() != value.nodes().len() {
            return Err(err("local provenance rank count mismatch"));
        }
        let mut substitutions = Vec::new();
        let mut local_stages = Vec::new();
        let mut diagnostics = Vec::new();
        for (rank, node) in value.nodes().iter().enumerate() {
            let provenance = local_step
                .local_inputs
                .get(rank)
                .ok_or_else(|| err("local provenance rank missing"))?;
            if provenance.rank != rank || provenance.consumer_local_node != *node {
                return Err(err("local provenance rank or consumer mismatch"));
            }
            let external = provenance
                .ordered_inputs
                .iter()
                .filter_map(|operand| operand.producer_redistribution_destination)
                .collect::<Vec<_>>();
            if external.len()
                != external
                    .iter()
                    .map(|node| node.index())
                    .collect::<BTreeSet<_>>()
                    .len()
            {
                return Err(err("duplicate redistribution destination provenance"));
            }
            let scheduled = schedule_with_external_materializations(graph, &[*node], &external)
                .map_err(|e| err(e.to_string()))?;
            let item = scheduled
                .items
                .first()
                .ok_or_else(|| err("local stage schedule missing"))?;
            item.validate_input_bindings()
                .map_err(|e| err(e.to_string()))?;
            if item.external_materializations != external {
                return Err(err("schedule external materialization provenance mismatch"));
            }
            if item.ordered_inputs().len() != provenance.ordered_inputs.len() {
                return Err(err("local provenance/ABI input count mismatch"));
            }
            for (operand, abi) in provenance.ordered_inputs.iter().zip(item.ordered_inputs()) {
                if abi.abi_index >= item.ordered_inputs().len() || !item.inputs.contains(&abi.desc)
                {
                    return Err(err("local provenance/ABI descriptor mismatch"));
                }
                if let Some(destination) = operand.producer_redistribution_destination {
                    if destination != abi.input_node || destination.index() as u64 != abi.desc.id {
                        return Err(err("redistribution destination ABI mismatch"));
                    }
                    substitutions.push(BufferSubstitution {
                        rank,
                        local_buffer: abi.desc.id,
                        transfer_buffer: destination.index() as u64,
                    });
                } else if operand.input_node != abi.input_node && abi.desc.view.is_none() {
                    // Static local shrink operands deliberately retain the
                    // original backing buffer in the ABI.  Any other node-id
                    // mismatch would lose the ordered provenance contract.
                    return Err(err("local provenance/ABI ordering or node mismatch"));
                }
            }
            let binding = &bindings[rank];
            let diagnostic =
                item.boundary
                    .as_ref()
                    .map(|boundary| CudaPlanDiagnostic::Unsupported {
                        node: node.index(),
                        reason: format!("schedule boundary: {boundary:?}"),
                    });
            let diagnostic = if diagnostic.is_none()
                && let Err(error) = PtxRenderer::new(binding.capability.sm())
                    .and_then(|renderer| renderer.render(&item.kernel))
            {
                Some(CudaPlanDiagnostic::Unsupported {
                    node: node.index(),
                    reason: format!("PTX renderer: {error}"),
                })
            } else {
                diagnostic
            };
            if let Some(diagnostic) = diagnostic.clone() {
                diagnostics.push(diagnostic);
            }
            let source_key = format!("schedule:{}", item.cache_key);
            local_stages.push(CudaPlanStage::Local {
                id: rank,
                device: binding.device.clone(),
                owner_identity: binding.context.identity(),
                node: node.index(),
                shape: graph.shape(*node)?.clone(),
                dtype: graph.dtype(*node)?,
                inputs: item.inputs.iter().map(|desc| desc.id).collect(),
                external_materializations: external
                    .iter()
                    .map(|node| node.index() as u64)
                    .collect(),
                output: item.primary_output().id,
                dependencies: vec![],
                source_key: source_key.clone(),
                module_key: format!(
                    "owner:{}:sm{}:{source_key}",
                    binding.context.identity(),
                    binding.capability.sm()
                ),
                diagnostic,
            });
        }
        let local_logical = ShardedCudaPlan {
            graph_id: graph.id(),
            layout_key: value.layout().cache_key().into(),
            bindings: bindings
                .iter()
                .map(|binding| {
                    (
                        binding.device.clone(),
                        binding.context.identity(),
                        binding.capability.sm(),
                    )
                })
                .collect(),
            stages: local_stages,
            diagnostics,
            cache_key: format!("sharded-cuda-local-fused:{}", value.layout().cache_key()),
            materializations: vec![],
        };
        let local = Self::executable(graph, local_logical, bindings)?;
        if substitutions.is_empty() {
            return Err(err("fused plan has no redistribution-produced local input"));
        }
        let transfer = transfer_from_provenance(graph, value, bindings, &substitutions)?;
        ShardedCudaPlanComposition::compose(&transfer, &local, substitutions)
    }
    /// Rehydrates only the exact local graph nodes named by the logical plan and verifies
    /// their schedule identity before retaining their rendered ABI. It never infers work
    /// from trace labels and performs no Driver operation.
    pub fn executable(
        graph: &Graph,
        logical: ShardedCudaPlan,
        bindings: &[CudaPlanBinding],
    ) -> Result<ExecutableShardedCudaPlan, Error> {
        if logical.graph_id != graph.id() || logical.bindings.len() != bindings.len() {
            return Err(err("logical plan graph or binding mismatch"));
        }
        let mut owners = Vec::with_capacity(bindings.len());
        for (record, binding) in logical.bindings.iter().zip(bindings) {
            if record.0 != binding.device
                || record.1 != binding.context.identity()
                || record.2 != binding.capability.sm()
            {
                return Err(err("logical plan owner/capability mismatch"));
            }
            owners.push(binding.context.clone());
        }
        let mut kernels = Vec::with_capacity(logical.stages.len());
        for stage in &logical.stages {
            match stage {
                CudaPlanStage::Local {
                    node,
                    owner_identity,
                    source_key,
                    diagnostic,
                    external_materializations,
                    ..
                } => {
                    let binding = bindings
                        .iter()
                        .find(|binding| binding.context.identity() == *owner_identity)
                        .ok_or_else(|| err("local stage owner missing"))?;
                    let materialized = external_materializations
                        .iter()
                        .map(|node| crate::NodeId::from_index(*node as usize))
                        .collect::<Vec<_>>();
                    let item = schedule_with_external_materializations(
                        graph,
                        &[crate::NodeId::from_index(*node)],
                        &materialized,
                    )
                    .map_err(|e| err(e.to_string()))?
                    .items
                    .into_iter()
                    .next()
                    .ok_or_else(|| err("local stage schedule missing"))?;
                    if source_key != &format!("schedule:{}", item.cache_key) {
                        return Err(err("local stage schedule identity mismatch"));
                    }
                    kernels.push(if diagnostic.is_none() {
                        Some(
                            PtxRenderer::new(binding.capability.sm())
                                .and_then(|renderer| renderer.render(&item.kernel))
                                .map_err(|e| err(e.to_string()))?,
                        )
                    } else {
                        None
                    });
                }
                _ => kernels.push(None),
            }
        }
        let mut buffers = Vec::new();
        for (stage_index, stage) in logical.stages.iter().enumerate() {
            if let CudaPlanStage::Local {
                device,
                owner_identity,
                node,
                external_materializations,
                ..
            } = stage
            {
                let rank = owners
                    .iter()
                    .position(|owner| owner.identity() == *owner_identity)
                    .ok_or_else(|| err("buffer owner missing"))?;
                let materialized = external_materializations
                    .iter()
                    .map(|node| crate::NodeId::from_index(*node as usize))
                    .collect::<Vec<_>>();
                let item = schedule_with_external_materializations(
                    graph,
                    &[crate::NodeId::from_index(*node)],
                    &materialized,
                )
                .map_err(|e| err(e.to_string()))?
                .items
                .into_iter()
                .next()
                .ok_or_else(|| err("local stage schedule missing"))?;
                for descriptor in item.inputs.iter().chain(item.outputs.iter()) {
                    let buffer = descriptor.id;
                    let producer = item
                        .outputs
                        .iter()
                        .any(|output| buffer == output.id)
                        .then_some(stage_index);
                    let bytes = descriptor.bytes;
                    if let Some(entry) = buffers.iter_mut().find(|entry: &&mut ExecutableBuffer| {
                        entry.rank == rank && entry.buffer == buffer
                    }) {
                        if entry.dtype != descriptor.dtype
                            || entry.shape != descriptor.shape
                            || entry.bytes != bytes
                        {
                            return Err(err("incompatible canonical buffer descriptor"));
                        }
                        entry.last_stage = stage_index;
                        entry.consumers.push(stage_index);
                        if producer.is_some() {
                            entry.producer = producer;
                            entry.role = ExecutableBufferRole::Output;
                        }
                    } else {
                        buffers.push(ExecutableBuffer {
                            rank,
                            device: device.clone(),
                            owner_identity: *owner_identity,
                            buffer,
                            dtype: descriptor.dtype,
                            shape: descriptor.shape.clone(),
                            bytes,
                            producer,
                            consumers: vec![stage_index],
                            first_stage: stage_index,
                            last_stage: stage_index,
                            role: if producer.is_some() {
                                ExecutableBufferRole::Output
                            } else {
                                ExecutableBufferRole::External
                            },
                        });
                    }
                }
            }
        }
        for stage in &logical.stages {
            if let CudaPlanStage::Transfer { routes, .. } = stage {
                for route in routes {
                    for (rank, device, buffer) in [
                        (route.source_rank, &route.source_device, route.source_buffer),
                        (
                            route.destination_rank,
                            &route.destination_device,
                            route.destination_buffer,
                        ),
                    ] {
                        if buffers
                            .iter()
                            .any(|entry| entry.rank == rank && entry.buffer == buffer)
                        {
                            continue;
                        }
                        let owner = logical
                            .bindings
                            .get(rank)
                            .ok_or_else(|| err("transfer rank outside bindings"))?;
                        let shape = graph
                            .shape(crate::NodeId::from_index(buffer as usize))?
                            .clone();
                        let dtype = graph.dtype(crate::NodeId::from_index(buffer as usize))?;
                        let bytes = shape
                            .numel()?
                            .checked_mul(dtype.itemsize())
                            .ok_or_else(|| err("transfer buffer byte overflow"))?;
                        buffers.push(ExecutableBuffer {
                            rank,
                            device: device.clone(),
                            owner_identity: owner.1,
                            buffer,
                            dtype,
                            shape,
                            bytes,
                            producer: None,
                            consumers: vec![],
                            first_stage: 0,
                            last_stage: logical.stages.len(),
                            role: ExecutableBufferRole::External,
                        });
                    }
                }
            }
        }
        for (stage_index, stage) in logical.stages.iter().enumerate() {
            if let CudaPlanStage::Collective {
                plan, buffers: ids, ..
            } = stage
            {
                if ids.len() != owners.len() || plan.request.input_lengths.len() != ids.len() {
                    return Err(err("collective buffer/group arity mismatch"));
                }
                for (rank, &buffer) in ids.iter().enumerate() {
                    let entry = buffers
                        .iter_mut()
                        .find(|entry| entry.rank == rank && entry.buffer == buffer)
                        .ok_or_else(|| err("collective output buffer is absent"))?;
                    if entry.dtype != plan.request.dtype
                        || entry.shape.numel()? != plan.request.input_lengths[rank]
                    {
                        return Err(err("collective output descriptor mismatch"));
                    }
                    entry.consumers.push(stage_index);
                    entry.last_stage = stage_index;
                }
            }
        }
        Ok(ExecutableShardedCudaPlan {
            logical,
            owners,
            kernels,
            buffers,
        })
    }
    /// Strictly decodes a fingerprinted v2 artifact, then rebinds its logical
    /// plan through the same owner/capability and provenance checks as a fresh
    /// plan. Raw/v1 artifacts deliberately have no route here.
    pub fn executable_transaction_artifact(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveTransaction, Error> {
        let (logical, candidates, commits) = CollectiveTransactionArtifact::decode(bytes)?;
        let plan = Self::executable(graph, logical, bindings)?;
        Ok(ExecutableCollectiveTransaction {
            plan,
            candidates,
            commits,
        })
    }

    /// Strict v3 decode and owner rebinding. This only constructs immutable
    /// buffer/lifetime metadata; execution rejects it before allocator/cache
    /// work because a downstream collective-result consumer is not released.
    pub fn executable_materialization_artifact(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveMaterialization, Error> {
        let (logical, candidates, commits) = CollectiveMaterializationArtifact::decode(bytes)?;
        let materializations = logical.materializations.clone();
        let mut plan = Self::executable(graph, logical, bindings)?;
        // V3 binds virtual candidate result descriptors. The released v2
        // runtime preflight intentionally requires a candidate lease to be
        // absent from the map before allocation, so it cannot be reused here:
        // decode has already validated the immutable v2 candidate/commit
        // schema, while this rebind only verifies concrete owners and builds
        // logical lifetime roles without allocating a lease.
        for materialization in &materializations {
            let candidate = candidates
                .iter()
                .find(|candidate| {
                    candidate.rank == materialization.rank
                        && candidate.candidate_buffer == materialization.candidate_buffer
                })
                .ok_or_else(|| err("materialization candidate linkage is absent at rebind"))?;
            if plan.buffers.iter().any(|buffer| {
                buffer.rank == materialization.rank
                    && buffer.buffer == materialization.candidate_buffer
            }) {
                return Err(err(
                    "materialization candidate collides with canonical buffer",
                ));
            }
            plan.buffers.push(ExecutableBuffer {
                rank: materialization.rank,
                device: materialization.device.clone(),
                owner_identity: materialization.owner_identity,
                buffer: materialization.candidate_buffer,
                dtype: materialization.dtype,
                shape: materialization.shape.clone(),
                bytes: materialization.bytes,
                producer: Some(candidate.stage),
                consumers: vec![materialization.first_consumer],
                first_stage: materialization.producer_stage,
                last_stage: materialization.last_consumer,
                role: ExecutableBufferRole::CollectiveResult,
            });
        }
        plan.logical.materializations = materializations.clone();
        plan.validate()?;
        Ok(ExecutableCollectiveMaterialization {
            plan,
            candidates,
            commits,
            materializations,
        })
    }

    /// Rebinds a validated v4 artifact before any schedule/cache, allocator,
    /// renderer, driver, or launch work.  A later execution vertical must
    /// explicitly consume `CollectiveResult` roles; this method never turns a
    /// downstream record into a host or PTX fallback.
    pub fn rebind_lifecycle_materialization_artifact(
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveLifecycleMaterialization, Error> {
        let (logical, candidates, commits, materializations) =
            CollectiveLifecycleMaterializationArtifact::decode(bytes)?;
        if bindings.len() != logical.bindings.len()
            || bindings
                .iter()
                .zip(&logical.bindings)
                .any(|(binding, expected)| {
                    binding.device != expected.0
                        || binding.context.identity() != expected.1
                        || binding.capability.sm() != expected.2
                        || binding.context.device() != binding.capability.device
                })
        {
            return Err(err(
                "v4 lifecycle artifact owner or capability binding mismatch",
            ));
        }
        if bindings
            .iter()
            .map(|binding| (&binding.device, binding.context.identity()))
            .collect::<BTreeSet<_>>()
            .len()
            != bindings.len()
        {
            return Err(err("v4 lifecycle artifact bindings are not unique"));
        }
        let mut buffers = Vec::with_capacity(materializations.len());
        for record in &materializations {
            let binding = &record.materialization;
            let consumers = match &record.lifecycle {
                CollectiveMaterializationLifecycle::Terminal => vec![],
                CollectiveMaterializationLifecycle::Downstream { .. } => record
                    .consumers
                    .iter()
                    .map(|consumer| consumer.consumer_stage)
                    .collect(),
            };
            buffers.push(ExecutableBuffer {
                rank: binding.rank,
                device: binding.device.clone(),
                owner_identity: binding.owner_identity,
                buffer: binding.candidate_buffer,
                dtype: binding.dtype,
                shape: binding.shape.clone(),
                bytes: binding.bytes,
                producer: Some(binding.producer_stage),
                consumers,
                first_stage: binding.producer_stage,
                last_stage: binding.last_consumer,
                role: ExecutableBufferRole::CollectiveResult,
            });
        }
        Ok(ExecutableCollectiveLifecycleMaterialization {
            logical,
            owners: bindings
                .iter()
                .map(|binding| binding.context.clone())
                .collect(),
            candidates,
            commits,
            materializations,
            buffers,
        })
    }

    /// Rebinds the dedicated fingerprinted graph-unary envelope before cache,
    /// allocation, renderer, driver, or launch work.
    fn rebind_graph_unary_output_artifact(
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        let (
            logical,
            candidates,
            commits,
            materializations,
            graph_result_bindings,
            consumer_abis,
            outputs,
            output_commits,
        ) = CollectiveGraphUnaryOutputArtifact::decode(bytes)?;
        if bindings.len() != logical.bindings.len()
            || bindings
                .iter()
                .zip(&logical.bindings)
                .any(|(binding, expected)| {
                    binding.device != expected.0
                        || binding.context.identity() != expected.1
                        || binding.capability.sm() != expected.2
                        || binding.context.device() != binding.capability.device
                })
        {
            return Err(err(
                "graph unary output owner or capability binding mismatch",
            ));
        }
        if bindings
            .iter()
            .map(|binding| (&binding.device, binding.context.identity()))
            .collect::<BTreeSet<_>>()
            .len()
            != bindings.len()
        {
            return Err(err("graph unary output bindings are not unique"));
        }
        let owners = bindings
            .iter()
            .map(|binding| binding.context.clone())
            .collect();
        let mut buffers = materializations
            .iter()
            .map(|record| {
                let binding = &record.materialization;
                ExecutableBuffer {
                    rank: binding.rank,
                    device: binding.device.clone(),
                    owner_identity: binding.owner_identity,
                    buffer: binding.candidate_buffer,
                    dtype: binding.dtype,
                    shape: binding.shape.clone(),
                    bytes: binding.bytes,
                    producer: Some(binding.producer_stage),
                    consumers: record
                        .consumers
                        .iter()
                        .map(|consumer| consumer.consumer_stage)
                        .collect(),
                    first_stage: binding.producer_stage,
                    last_stage: binding.last_consumer,
                    role: ExecutableBufferRole::CollectiveResult,
                }
            })
            .collect::<Vec<_>>();
        for output in &outputs {
            if buffers.iter().any(|buffer| {
                buffer.rank == output.rank
                    && (buffer.buffer == output.output_candidate_buffer
                        || buffer.buffer == output.destination_buffer)
            }) {
                return Err(err("v5 downstream output collides with collective role"));
            }
            buffers.push(ExecutableBuffer {
                rank: output.rank,
                device: output.device.clone(),
                owner_identity: output.owner_identity,
                buffer: output.output_candidate_buffer,
                dtype: output.dtype,
                shape: output.shape.clone(),
                bytes: output.bytes,
                producer: Some(output.consumer_stage),
                consumers: vec![],
                first_stage: output.first_stage,
                last_stage: output.last_stage,
                role: ExecutableBufferRole::TransactionOutput,
            });
        }
        Ok(ExecutableCollectiveGraphUnaryOutput {
            downstream: ExecutableCollectiveDownstreamOutput {
                logical,
                owners,
                candidates,
                commits,
                materializations,
                outputs,
                output_commits,
                buffers,
            },
            graph_result_bindings,
            consumer_abis,
            consumer_nodes: vec![],
            substitutions: vec![],
            unary_op: None,
            unary_bindings: None,
        })
    }

    /// Rebinds the released v5 logical-output artifact without authorizing its
    /// local stage. Graph-aware execution uses the distinct unary envelope.
    pub fn rebind_downstream_output_artifact(
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveDownstreamOutput, Error> {
        let (logical, candidates, commits, materializations, outputs, output_commits) =
            CollectiveDownstreamOutputArtifact::decode(bytes)?;
        let v4 = CollectiveLifecycleMaterializationArtifact::encode(
            &logical,
            candidates.clone(),
            commits.clone(),
            materializations.clone(),
        )?;
        let rebound = Self::rebind_lifecycle_materialization_artifact(bindings, &v4)?;
        let mut buffers = rebound.buffers;
        for output in &outputs {
            if buffers.iter().any(|buffer| {
                buffer.rank == output.rank
                    && (buffer.buffer == output.output_candidate_buffer
                        || buffer.buffer == output.destination_buffer)
            }) {
                return Err(err("v5 downstream output collides with collective role"));
            }
            buffers.push(ExecutableBuffer {
                rank: output.rank,
                device: output.device.clone(),
                owner_identity: output.owner_identity,
                buffer: output.output_candidate_buffer,
                dtype: output.dtype,
                shape: output.shape.clone(),
                bytes: output.bytes,
                producer: Some(output.consumer_stage),
                consumers: vec![],
                first_stage: output.first_stage,
                last_stage: output.last_stage,
                role: ExecutableBufferRole::TransactionOutput,
            });
        }
        Ok(ExecutableCollectiveDownstreamOutput {
            logical,
            owners: rebound.owners,
            candidates,
            commits,
            materializations,
            outputs,
            output_commits,
            buffers,
        })
    }

    /// Graph-aware, still non-executing rebind for exactly one collective
    /// result consumed by one rank-local closed-set F32 unary per rank. The
    /// generic downstream executor deliberately remains fail-closed; this only retains
    /// verified node identities for the dedicated transaction execution path.
    pub(crate) fn rebind_downstream_output_artifact_for_unary(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
        unary_op: GraphBackedDownstreamUnary,
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        let mut rebound = Self::rebind_graph_unary_output_artifact(bindings, bytes)?;
        if rebound.logical.graph_id != graph.id()
            || !rebound
                .logical
                .cache_key
                .starts_with(unary_op.cache_prefix())
            || rebound.materializations.len() != bindings.len()
            || rebound.consumer_abis.len() != bindings.len()
            || rebound.outputs.len() != bindings.len()
            || rebound
                .logical
                .stages
                .iter()
                .filter(|stage| matches!(stage, CudaPlanStage::Collective { .. }))
                .count()
                != 1
        {
            return Err(err(
                "v5 graph-backed unary rebind requires one collective and one local stage per rank",
            ));
        }
        let boundary_keys = rebound
            .materializations
            .iter()
            .map(|record| record.materialization.boundary_key.as_str())
            .collect::<BTreeSet<_>>();
        if boundary_keys.len() != 1 {
            return Err(err(
                "v5 graph-backed unary rebind requires one collective boundary",
            ));
        }
        let collective_stage = rebound
            .logical
            .stages
            .iter()
            .position(|stage| matches!(stage, CudaPlanStage::Collective { .. }))
            .ok_or_else(|| err("v5 graph-backed unary collective stage is absent"))?;
        let consumer_stages = rebound
            .consumer_abis
            .iter()
            .map(|abi| abi.consumer_stage)
            .collect::<BTreeSet<_>>();
        if rebound
            .logical
            .stages
            .iter()
            .enumerate()
            .skip(collective_stage + 1)
            .any(|(index, stage)| {
                !consumer_stages.contains(&index) || !matches!(stage, CudaPlanStage::Local { .. })
            })
        {
            return Err(err(
                "v5 graph-backed unary has an unauthorized post-collective stage",
            ));
        }
        let mut consumer_nodes = Vec::with_capacity(bindings.len());
        let mut substitutions = Vec::with_capacity(bindings.len());
        for rank in 0..bindings.len() {
            let abi = rebound
                .consumer_abis
                .iter()
                .find(|abi| abi.rank == rank)
                .ok_or_else(|| err("v5 graph-backed unary rebind rank ABI is absent"))?;
            let output = rebound
                .outputs
                .iter()
                .find(|output| output.rank == rank)
                .ok_or_else(|| err("v5 graph-backed unary rebind rank output is absent"))?;
            let node = rebound
                .logical
                .stages
                .get(abi.consumer_stage)
                .and_then(|stage| match stage {
                    CudaPlanStage::Local {
                        node,
                        inputs,
                        external_materializations,
                        ..
                    } if external_materializations == &vec![abi.replicated_result as u64]
                        && inputs.contains(&abi.local_input_buffer) =>
                    {
                        Some(NodeId::from_index(*node))
                    }
                    _ => None,
                })
                .ok_or_else(|| {
                    err("v5 graph-backed unary rebind local ABI does not match graph result binding")
                })?;
            if graph.dtype(node)? != unary_op.dtype()
                || graph.shape(node)? != &abi.shape
                || !matches!(graph.op(node)?, Op::Unary { op, input } if *op == unary_op.op() && input.index() == abi.replicated_result)
                || output.consumer_stage != abi.consumer_stage
                || output.output_candidate_buffer != abi.output_candidate_buffer
            {
                return Err(err(format!(
                    "v5 {} rebind graph operation or layout is unsupported",
                    unary_op.name()
                )));
            }
            consumer_nodes.push(node);
            substitutions.push(BufferSubstitution {
                rank,
                local_buffer: abi.local_input_buffer,
                transfer_buffer: abi.candidate_buffer,
            });
        }
        rebound.consumer_nodes = consumer_nodes;
        rebound.substitutions = substitutions;
        rebound.unary_op = Some(unary_op);
        rebound.unary_bindings = Some(bindings.to_vec());
        Ok(rebound)
    }

    /// Compatibility entrypoint for the released graph-backed F32 Neg route.
    pub fn rebind_downstream_output_artifact_for_neg(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        Self::rebind_downstream_output_artifact_for_unary(
            graph,
            bindings,
            bytes,
            GraphBackedDownstreamUnary::Neg,
        )
    }

    /// Dedicated graph-aware v5 F32 Abs authorization. Generic downstream
    /// execution remains unavailable.
    pub fn rebind_downstream_output_artifact_for_abs(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        Self::rebind_downstream_output_artifact_for_unary(
            graph,
            bindings,
            bytes,
            GraphBackedDownstreamUnary::Abs,
        )
    }

    /// Authorizes the exact graph-backed F64 Neg companion after complete
    /// artifact, graph, owner, and local-schedule validation.
    pub fn rebind_downstream_output_artifact_for_f64_neg(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        Self::rebind_downstream_output_artifact_for_unary(
            graph,
            bindings,
            bytes,
            GraphBackedDownstreamUnary::NegF64,
        )
    }

    /// Authorizes the exact graph-backed F64 Abs companion after complete
    /// artifact, graph, owner, and local-schedule validation.
    pub fn rebind_downstream_output_artifact_for_f64_abs(
        graph: &Graph,
        bindings: &[CudaPlanBinding],
        bytes: &[u8],
    ) -> Result<ExecutableCollectiveGraphUnaryOutput, Error> {
        Self::rebind_downstream_output_artifact_for_unary(
            graph,
            bindings,
            bytes,
            GraphBackedDownstreamUnary::AbsF64,
        )
    }
}

fn transfer_from_provenance(
    graph: &Graph,
    value: &ShardedGraphTensor,
    bindings: &[CudaPlanBinding],
    substitutions: &[BufferSubstitution],
) -> Result<ExecutableShardedCudaPlan, Error> {
    let wanted = substitutions
        .iter()
        .map(|substitution| (substitution.rank, substitution.transfer_buffer))
        .collect::<BTreeSet<_>>();
    let trace = value
        .trace()
        .steps
        .iter()
        .find(|step| {
            let destinations = step
                .routes
                .iter()
                .map(|route| {
                    (
                        route.destination_rank,
                        route.destination_node.index() as u64,
                    )
                })
                .collect::<BTreeSet<_>>();
            !step.routes.is_empty() && wanted.is_subset(&destinations)
        })
        .ok_or_else(|| err("provenance redistribution routes are absent"))?;
    let mut routes = Vec::new();
    let mut buffers = BTreeMap::new();
    for route in &trace.routes {
        let bytes = route
            .elements
            .checked_mul(graph.dtype(route.source_node)?.itemsize())
            .ok_or_else(|| err("provenance route byte overflow"))?;
        let dtype = graph.dtype(route.source_node)?;
        if dtype != graph.dtype(route.destination_node)? {
            return Err(err("provenance route dtype mismatch"));
        }
        let source_shape = graph.shape(route.source_node)?.clone();
        let destination_shape = graph.shape(route.destination_node)?.clone();
        for (rank, device, node, shape, role, producer) in [
            (
                route.source_rank,
                route.source_device.clone(),
                route.source_node,
                source_shape,
                ExecutableBufferRole::External,
                None,
            ),
            (
                route.destination_rank,
                route.destination_device.clone(),
                route.destination_node,
                destination_shape,
                ExecutableBufferRole::Output,
                Some(0),
            ),
        ] {
            let binding = bindings
                .get(rank)
                .ok_or_else(|| err("route rank outside bindings"))?;
            if binding.device != device {
                return Err(err("provenance route device/rank mismatch"));
            }
            let bytes = shape
                .numel()?
                .checked_mul(dtype.itemsize())
                .ok_or_else(|| err("provenance buffer byte overflow"))?;
            let key = (rank, node.index() as u64);
            let entry = ExecutableBuffer {
                rank,
                device,
                owner_identity: binding.context.identity(),
                buffer: key.1,
                dtype,
                shape,
                bytes,
                producer,
                consumers: vec![0],
                first_stage: 0,
                last_stage: 0,
                role,
            };
            if let Some(existing) = buffers.get(&key) {
                if existing != &entry {
                    return Err(err("provenance transfer buffer descriptor mismatch"));
                }
            } else {
                buffers.insert(key, entry);
            }
        }
        routes.push(CudaTransferRoute {
            source_rank: route.source_rank,
            source_device: route.source_device.clone(),
            source_buffer: route.source_node.index() as u64,
            source_element_offset: route.source_offset,
            destination_rank: route.destination_rank,
            destination_device: route.destination_device.clone(),
            destination_buffer: route.destination_node.index() as u64,
            destination_element_offset: route.destination_offset,
            elements: route.elements,
            bytes,
            dtype,
        });
    }
    let logical = ShardedCudaPlan {
        graph_id: graph.id(),
        layout_key: value.layout().cache_key().into(),
        bindings: bindings
            .iter()
            .map(|binding| {
                (
                    binding.device.clone(),
                    binding.context.identity(),
                    binding.capability.sm(),
                )
            })
            .collect(),
        stages: vec![CudaPlanStage::Transfer {
            id: 0,
            action: "redistribute".into(),
            routes,
            dependencies: vec![],
        }],
        diagnostics: vec![],
        cache_key: format!(
            "sharded-cuda-provenance-transfer:{}",
            value.layout().cache_key()
        ),
        materializations: vec![],
    };
    Ok(ExecutableShardedCudaPlan {
        logical,
        owners: bindings
            .iter()
            .map(|binding| binding.context.clone())
            .collect(),
        kernels: vec![None],
        buffers: buffers.into_values().collect(),
    })
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
    local_shape: &Shape,
) -> Result<CollectivePlan, Error> {
    let n = local_shape.numel()?;
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

#[cfg(test)]
mod artifact_tests {
    use super::*;

    fn plan() -> ShardedCudaPlan {
        ShardedCudaPlan {
            graph_id: 7,
            layout_key: "artifact-layout".into(),
            bindings: vec![],
            stages: vec![],
            diagnostics: vec![],
            cache_key: "artifact-cache".into(),
            materializations: vec![],
        }
    }

    #[test]
    fn graph_result_binding_is_checked_and_canonically_ordered() {
        let binding = CollectiveGraphResultBinding {
            replicated_result: 7,
            rank: 0,
            candidate_buffer: 11,
            local_input_buffer: 12,
            device: SemanticDeviceId::new("CUDA:0").unwrap(),
            owner_identity: 41,
            dtype: DType::F32,
            shape: Shape::from([2]),
            bytes: 2 * DType::F32.itemsize(),
            first_consumer_stage: 3,
            lifetime_end_stage: 4,
        };
        assert_eq!(binding.canonical_key(), (7, 0, 11, 12));
        binding.validate().unwrap();
        let mut malformed = binding.clone();
        malformed.local_input_buffer = malformed.candidate_buffer;
        assert!(malformed.validate().is_err());
        let mut malformed = binding;
        malformed.lifetime_end_stage = 2;
        assert!(malformed.validate().is_err());
        let abi = CollectiveDownstreamConsumerAbi {
            replicated_result: 7,
            rank: 0,
            candidate_buffer: 11,
            local_input_buffer: 12,
            output_candidate_buffer: 13,
            device: SemanticDeviceId::new("CUDA:0").unwrap(),
            owner_identity: 41,
            dtype: DType::F32,
            shape: Shape::from([2]),
            bytes: 2 * DType::F32.itemsize(),
            consumer_stage: 3,
            lifetime_end_stage: 4,
        };
        abi.validate().unwrap();
        assert_eq!(abi.canonical_key(), (7, 0, 11, 12, 13));
        let mut malformed = abi;
        malformed.local_input_buffer = malformed.candidate_buffer;
        assert!(malformed.validate().is_err());
    }

    #[test]
    fn graph_backed_downstream_unary_scope_is_exactly_f32_f64_neg_abs() {
        for (unary, dtype, op, prefix) in [
            (
                GraphBackedDownstreamUnary::Neg,
                DType::F32,
                UnaryOp::Neg,
                "graph-backed-unary-neg:",
            ),
            (
                GraphBackedDownstreamUnary::Abs,
                DType::F32,
                UnaryOp::Abs,
                "graph-backed-unary-abs:",
            ),
            (
                GraphBackedDownstreamUnary::NegF64,
                DType::F64,
                UnaryOp::Neg,
                "graph-backed-unary-f64-neg:",
            ),
            (
                GraphBackedDownstreamUnary::AbsF64,
                DType::F64,
                UnaryOp::Abs,
                "graph-backed-unary-f64-abs:",
            ),
        ] {
            assert_eq!(unary.dtype(), dtype);
            assert_eq!(unary.op(), op);
            assert_eq!(unary.cache_prefix(), prefix);
        }
    }

    #[test]
    fn released_v5_executable_companion_keeps_its_eight_public_fields() {
        let executable = ExecutableCollectiveDownstreamOutput {
            logical: plan(),
            owners: vec![],
            candidates: vec![],
            commits: vec![],
            materializations: vec![],
            outputs: vec![],
            output_commits: vec![],
            buffers: vec![],
        };
        assert!(executable.owners.is_empty());
        assert!(executable.outputs.is_empty());
    }

    #[test]
    fn versioned_artifact_roundtrips_with_stable_identity_and_legacy_raw_is_candidate_free() {
        let plan = plan();
        let first = ShardedCudaPlanArtifact::encode(&plan).unwrap();
        let second = ShardedCudaPlanArtifact::encode(&plan).unwrap();
        assert_eq!(first, second);
        assert_eq!(ShardedCudaPlanArtifact::decode(&first).unwrap(), plan);
        let mut raw = serde_json::to_value(&plan).unwrap();
        raw.as_object_mut().unwrap().remove("materializations");
        let raw = serde_json::to_vec(&raw).unwrap();
        assert_eq!(ShardedCudaPlanArtifact::decode(&raw).unwrap(), plan);
    }

    #[test]
    fn artifact_rejects_tampering_unknown_versions_and_transaction_metadata() {
        let plan = plan();
        let encoded = ShardedCudaPlanArtifact::encode(&plan).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["fingerprint"] = serde_json::Value::String("fnv1a64:0000000000000000".into());
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&value).unwrap()).is_err());
        value["format_version"] = serde_json::Value::from(99_u32);
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut raw = serde_json::to_value(&plan).unwrap();
        raw["candidate_buffers"] = serde_json::json!([]);
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&raw).unwrap()).is_err());
        let mut raw = serde_json::to_value(&plan).unwrap();
        raw["materializations"] = serde_json::json!([]);
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&raw).unwrap()).is_err());
        let mut raw = serde_json::to_value(&plan).unwrap();
        raw["outputs"] = serde_json::json!([]);
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&raw).unwrap()).is_err());
        let mut raw = serde_json::to_value(&plan).unwrap();
        raw["consumer_abis"] = serde_json::json!([]);
        assert!(ShardedCudaPlanArtifact::decode(&serde_json::to_vec(&raw).unwrap()).is_err());
    }

    fn materialization_parts() -> (
        ShardedCudaPlan,
        Vec<CollectiveCandidateDescriptor>,
        Vec<CollectiveCommitRecord>,
    ) {
        let device = SemanticDeviceId::new("CUDA:0").unwrap();
        let group = DeviceGroup::new([device.clone()]).unwrap();
        let shape = Shape::from([1]);
        let collective = collective_plan(&group, DType::F32, &shape).unwrap();
        let plan = ShardedCudaPlan {
            graph_id: 9,
            layout_key: "v3-layout".into(),
            bindings: vec![(device.clone(), 41, 80)],
            stages: vec![CudaPlanStage::Collective {
                id: 0,
                action: "all-reduce".into(),
                plan: collective,
                buffers: vec![7],
                dependencies: vec![],
            }],
            diagnostics: vec![],
            cache_key: "v3-cache".into(),
            materializations: vec![CollectiveResultMaterialization {
                boundary_key: "terminal-all-reduce".into(),
                replicated_result: 0,
                rank: 0,
                device,
                owner_identity: 41,
                candidate_buffer: 8,
                dtype: DType::F32,
                shape: shape.clone(),
                bytes: DType::F32.itemsize(),
                producer_stage: 0,
                // `stages.len()` is the explicit terminal commit boundary;
                // a true downstream stage remains out of scope for v3.
                first_consumer: 1,
                last_consumer: 1,
            }],
        };
        (
            plan,
            vec![CollectiveCandidateDescriptor {
                stage: 0,
                rank: 0,
                candidate_buffer: 8,
                source_buffer: 7,
                dtype: DType::F32,
                shape,
                bytes: DType::F32.itemsize(),
            }],
            vec![CollectiveCommitRecord {
                order: 0,
                rank: 0,
                candidate_buffer: 8,
                target_buffer: 7,
            }],
        )
    }

    #[test]
    fn v3_materialization_roundtrips_stably_and_legacy_routes_reject_it() {
        let (plan, candidates, commits) = materialization_parts();
        let first =
            CollectiveMaterializationArtifact::encode(&plan, candidates.clone(), commits.clone())
                .unwrap();
        assert_eq!(
            first,
            CollectiveMaterializationArtifact::encode(&plan, candidates.clone(), commits.clone())
                .unwrap()
        );
        assert_eq!(
            CollectiveMaterializationArtifact::decode(&first).unwrap(),
            (plan.clone(), candidates, commits)
        );
        assert!(ShardedCudaPlanArtifact::encode(&plan).is_err());
        let raw = serde_json::to_vec(&plan).unwrap();
        assert!(ShardedCudaPlanArtifact::decode(&raw).is_err());
    }

    #[test]
    fn v3_materialization_tamper_and_invalid_linkage_reject_before_rebind() {
        let (plan, candidates, commits) = materialization_parts();
        let encoded =
            CollectiveMaterializationArtifact::encode(&plan, candidates, commits).unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["fingerprint"] = serde_json::Value::String("fnv1a64:0000000000000000".into());
        assert!(
            CollectiveMaterializationArtifact::decode(&serde_json::to_vec(&value).unwrap())
                .is_err()
        );
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["plan"]["materializations"][0]["candidate_buffer"] = serde_json::Value::from(99_u64);
        assert!(
            CollectiveMaterializationArtifact::decode(&serde_json::to_vec(&value).unwrap())
                .is_err()
        );
        let mut value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        value["format_version"] = serde_json::Value::from(2_u32);
        assert!(
            CollectiveMaterializationArtifact::decode(&serde_json::to_vec(&value).unwrap())
                .is_err()
        );
    }

    fn v4_downstream_parts() -> (
        ShardedCudaPlan,
        Vec<CollectiveCandidateDescriptor>,
        Vec<CollectiveCommitRecord>,
        Vec<CollectiveLifecycleMaterialization>,
    ) {
        let (mut plan, candidates, commits) = materialization_parts();
        let base = plan.materializations.pop().unwrap();
        let device = base.device.clone();
        plan.materializations = vec![];
        plan.stages.push(CudaPlanStage::Local {
            id: 1,
            device: device.clone(),
            owner_identity: base.owner_identity,
            node: 1,
            shape: base.shape.clone(),
            dtype: base.dtype,
            inputs: vec![base.candidate_buffer],
            external_materializations: vec![base.candidate_buffer],
            output: 9,
            dependencies: vec![0],
            source_key: "v4-downstream-source".into(),
            module_key: "v4-downstream-module".into(),
            diagnostic: None,
        });
        let materialization = CollectiveLifecycleMaterialization {
            materialization: CollectiveResultMaterialization {
                first_consumer: 1,
                last_consumer: 1,
                ..base
            },
            lifecycle: CollectiveMaterializationLifecycle::Downstream {
                first_consumer_stage: 1,
                lifetime_end_stage: 1,
            },
            consumers: vec![CollectiveConsumerDescriptor {
                rank: 0,
                consumer_stage: 1,
                consumer_buffer: 8,
                device,
                owner_identity: 41,
                dtype: DType::F32,
                shape: Shape::from([1]),
                bytes: DType::F32.itemsize(),
            }],
        };
        (plan, candidates, commits, vec![materialization])
    }

    #[test]
    fn v4_downstream_roundtrips_stably_and_legacy_envelopes_reject_it() {
        let (plan, candidates, commits, materializations) = v4_downstream_parts();
        let first = CollectiveLifecycleMaterializationArtifact::encode(
            &plan,
            candidates.clone(),
            commits.clone(),
            materializations.clone(),
        )
        .unwrap();
        assert_eq!(
            first,
            CollectiveLifecycleMaterializationArtifact::encode(
                &plan,
                candidates.clone(),
                commits.clone(),
                materializations.clone(),
            )
            .unwrap()
        );
        assert_eq!(
            CollectiveLifecycleMaterializationArtifact::decode(&first).unwrap(),
            (plan.clone(), candidates, commits, materializations)
        );
        assert!(CollectiveMaterializationArtifact::decode(&first).is_err());
        assert!(CollectiveTransactionArtifact::decode(&first).is_err());
        assert!(ShardedCudaPlanArtifact::decode(&first).is_err());
    }

    #[test]
    fn v4_downstream_tamper_and_malformed_lifetimes_reject_before_rebind() {
        let (plan, candidates, commits, materializations) = v4_downstream_parts();
        let encoded = CollectiveLifecycleMaterializationArtifact::encode(
            &plan,
            candidates,
            commits,
            materializations,
        )
        .unwrap();
        let mut tampered: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        tampered["fingerprint"] = serde_json::Value::String("fnv1a64:0000000000000000".into());
        assert!(
            CollectiveLifecycleMaterializationArtifact::decode(
                &serde_json::to_vec(&tampered).unwrap()
            )
            .is_err()
        );
        let mut malformed: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        malformed["materializations"][0]["consumers"][0]["consumer_stage"] =
            serde_json::Value::from(0_u64);
        assert!(
            CollectiveLifecycleMaterializationArtifact::decode(
                &serde_json::to_vec(&malformed).unwrap()
            )
            .is_err()
        );
        let mut terminal: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
        terminal["materializations"][0]["lifecycle"] = serde_json::json!("Terminal");
        assert!(
            CollectiveLifecycleMaterializationArtifact::decode(
                &serde_json::to_vec(&terminal).unwrap()
            )
            .is_err()
        );
    }
}
