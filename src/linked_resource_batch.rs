//! Versioned, payload-free proof for exactly two independent linked F32 Exp
//! capture items.  It deliberately has no execution entrypoint.

use crate::{
    cuda::LinkInput,
    linked_resource::{LinkedF32ExpResourceBinding, LinkedF32ExpResourceDescriptor},
    ptx::{KernelSemanticProgram, LinkedF32ExpRequest},
    BufferDesc, CapturedSchedule, DType, PrimaryContext,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const LINKED_F32_EXP_BATCH_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkedF32ExpBatchSlot {
    pub key: String,
    pub item_id: u64,
    pub request_identity: String,
    pub resource_identity: String,
    pub rendered_identity: String,
    pub input: BufferDesc,
    pub output: BufferDesc,
    pub owner_device: u32,
    pub sm: u32,
}

/// A distinct v2 envelope.  v1 sidecar bytes decode only through their v1
/// decoder and can never acquire batch metadata by a serde default.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkedF32ExpBatchArtifact {
    pub schema_version: u32,
    pub artifact_identity: String,
    pub capture_identity: u64,
    pub slots: Vec<LinkedF32ExpBatchSlot>,
}

#[derive(Clone, Debug)]
pub struct BoundLinkedF32ExpBatchResources {
    artifact: LinkedF32ExpBatchArtifact,
    bindings: BTreeMap<String, LinkedF32ExpResourceBinding>,
}

/// Data-only two-consumer preparation.  `candidate_id` and `target_id` are
/// logical transaction records, never caller-provided writable candidates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedLinkedF32ExpBatchCapture {
    pub capture_identity: u64,
    pub artifact_identity: String,
    pub slots: Vec<PreparedLinkedF32ExpBatchSlot>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedLinkedF32ExpBatchSlot {
    pub key: String,
    pub item_id: u64,
    pub input: BufferDesc,
    pub candidate_id: u64,
    pub target: BufferDesc,
    pub request_identity: String,
    pub resource_identity: String,
}

impl LinkedF32ExpBatchArtifact {
    pub fn from_capture_requests(
        capture: &CapturedSchedule,
        primary: &PrimaryContext,
        sm: u32,
        records: &[(LinkedF32ExpRequest, LinkedF32ExpResourceDescriptor)],
    ) -> Result<Self, crate::ptx::PtxError> {
        if capture.items.len() != 2 || capture.requested.len() != 2 || records.len() != 2 {
            return Err(invalid("linked Exp batch cardinality"));
        }
        let mut slots = Vec::with_capacity(2);
        for (item, (request, descriptor)) in capture.items.iter().zip(records) {
            validate_item(item, request, descriptor, primary, sm)?;
            let input = item.ordered_inputs()[0].desc.clone();
            slots.push(LinkedF32ExpBatchSlot {
                key: format!("linked-f32-exp-batch/{}/{}", item.id, request.identity()),
                item_id: item.id,
                request_identity: request.identity().into(),
                resource_identity: descriptor.resource_identity.clone(),
                rendered_identity: request.rendered().cache_key.clone(),
                input,
                output: item.primary_output().clone(),
                owner_device: primary.device().0,
                sm,
            });
        }
        let mut artifact = Self { schema_version: LINKED_F32_EXP_BATCH_SCHEMA_VERSION, artifact_identity: String::new(), capture_identity: capture.identity, slots };
        artifact.validate()?;
        artifact.artifact_identity = artifact.identity()?;
        Ok(artifact)
    }
    pub fn encode(&self) -> Result<Vec<u8>, crate::ptx::PtxError> { self.validate()?; serde_json::to_vec(self).map_err(|_| invalid("linked Exp batch encoding")) }
    pub fn decode(bytes: &[u8]) -> Result<Self, crate::ptx::PtxError> {
        let value = serde_json::from_slice::<Self>(bytes).map_err(|_| invalid("linked Exp batch encoding"))?;
        if serde_json::to_vec(&value).map_err(|_| invalid("linked Exp batch encoding"))? != bytes { return Err(invalid("linked Exp batch noncanonical")); }
        value.validate()?; Ok(value)
    }
    pub fn rebind(
        &self, capture: &CapturedSchedule, primary: &PrimaryContext, sm: u32,
        records: &BTreeMap<String, (LinkedF32ExpRequest, LinkedF32ExpResourceDescriptor, Vec<LinkInput>)>,
    ) -> Result<BoundLinkedF32ExpBatchResources, crate::ptx::PtxError> {
        self.validate()?;
        if capture.identity != self.capture_identity || records.len() != 2 { return Err(invalid("linked Exp batch rebind inventory")); }
        let mut bindings = BTreeMap::new();
        for slot in &self.slots {
            let (request, descriptor, payload) = records.get(&slot.key).ok_or_else(|| invalid("linked Exp batch slot"))?;
            if request.identity() != slot.request_identity || descriptor.resource_identity != slot.resource_identity || descriptor.device != slot.owner_device || descriptor.sm != slot.sm { return Err(invalid("linked Exp batch resource linkage")); }
            bindings.insert(slot.key.clone(), descriptor.rebind(primary, sm, payload)?);
        }
        Ok(BoundLinkedF32ExpBatchResources { artifact: self.clone(), bindings })
    }
    fn validate(&self) -> Result<(), crate::ptx::PtxError> {
        if self.schema_version != LINKED_F32_EXP_BATCH_SCHEMA_VERSION || self.capture_identity == 0 || self.slots.len() != 2 { return Err(invalid("linked Exp batch schema")); }
        let mut keys = BTreeSet::new(); let mut items = BTreeSet::new(); let mut buffers = BTreeSet::new();
        for slot in &self.slots {
            if slot.key.is_empty() || !keys.insert(&slot.key) || !items.insert(slot.item_id) || slot.request_identity.is_empty() || slot.resource_identity.is_empty() || slot.rendered_identity.is_empty() || slot.owner_device > i32::MAX as u32 || slot.sm == 0 || slot.input.dtype != DType::F32 || slot.output.dtype != DType::F32 || slot.input.shape != slot.output.shape || slot.input.bytes == 0 || slot.input.bytes != slot.output.bytes || !slot.input.read_only || slot.output.read_only || slot.input.view.is_some() || slot.output.view.is_some() || !buffers.insert(slot.input.id) || !buffers.insert(slot.output.id) { return Err(invalid("linked Exp batch slot")); }
        }
        if self.slots[0].item_id >= self.slots[1].item_id || self.slots[0].owner_device != self.slots[1].owner_device || self.slots[0].sm != self.slots[1].sm { return Err(invalid("linked Exp batch ordering/owner")); }
        if !self.artifact_identity.is_empty() && self.artifact_identity != self.identity()? { return Err(invalid("linked Exp batch identity")); }
        Ok(())
    }
    fn identity(&self) -> Result<String, crate::ptx::PtxError> { let mut canonical = self.clone(); canonical.artifact_identity.clear(); let bytes = serde_json::to_vec(&canonical).map_err(|_| invalid("linked Exp batch encoding"))?; let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |h,b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)); Ok(format!("linked-f32-exp-batch-v{}:{hash:016x}", self.schema_version)) }
}

impl PreparedLinkedF32ExpBatchCapture {
    pub fn prepare(capture: &CapturedSchedule, artifact: &LinkedF32ExpBatchArtifact, bound: &BoundLinkedF32ExpBatchResources) -> Result<Self, crate::ptx::PtxError> {
        if capture.identity != artifact.capture_identity || bound.artifact.artifact_identity != artifact.artifact_identity || bound.bindings.len() != 2 { return Err(invalid("linked Exp batch prepared linkage")); }
        let mut slots = Vec::new();
        for resource in &artifact.slots {
            let item = capture.items.iter().find(|item| item.id == resource.item_id).ok_or_else(|| invalid("linked Exp batch item"))?;
            if !item.dependencies.is_empty() || !item.consumers.is_empty() || item.ordered_inputs().len() != 1 || item.primary_output() != &resource.output || item.ordered_inputs()[0].desc != resource.input { return Err(invalid("linked Exp batch dependency/ABI")); }
            let candidate_id = resource.output.id.checked_add(0x8000_0000_0000_0000).ok_or_else(|| invalid("linked Exp batch candidate"))?;
            slots.push(PreparedLinkedF32ExpBatchSlot { key: resource.key.clone(), item_id: item.id, input: resource.input.clone(), candidate_id, target: resource.output.clone(), request_identity: resource.request_identity.clone(), resource_identity: resource.resource_identity.clone() });
        }
        Ok(Self { capture_identity: capture.identity, artifact_identity: artifact.artifact_identity.clone(), slots })
    }
}

fn validate_item(item: &crate::ScheduleItem, request: &LinkedF32ExpRequest, descriptor: &LinkedF32ExpResourceDescriptor, primary: &PrimaryContext, sm: u32) -> Result<(), crate::ptx::PtxError> {
    if !item.dependencies.is_empty() || !item.consumers.is_empty() || item.ordered_inputs().len() != 1 || !item.outputs.is_single() || item.ordered_inputs()[0].desc.view.is_some() || item.primary_output().view.is_some() || descriptor.request_identity != request.identity() || descriptor.device != primary.device().0 || descriptor.sm != sm || item.ordered_inputs()[0].desc.dtype != DType::F32 || item.primary_output().dtype != DType::F32 || item.ordered_inputs()[0].desc.shape != item.primary_output().shape || item.primary_output().bytes == 0 { return Err(invalid("linked Exp batch item")); }
    match request.rendered().semantic_program.as_ref() { Some(KernelSemanticProgram::UOp(program)) if program.as_ref() == &item.kernel => Ok(()), _ => Err(invalid("linked Exp batch UOp")) }
}
fn invalid(message: &str) -> crate::ptx::PtxError { crate::ptx::PtxError::InvalidBinding(message.into()) }
