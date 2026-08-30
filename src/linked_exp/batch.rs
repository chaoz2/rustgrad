//! Versioned, payload-free proof and dedicated launcher for exactly two
//! independent linked F32 Exp capture items. Caller-output publication uses
//! best-effort rollback and exposes possible partial mutation when restoration
//! itself fails.

use crate::{
    BufferDesc, CapturedSchedule, DType, PrimaryContext,
    cuda::{BufferView, DeviceBuffer, LinkInput, PrimaryOutputCommit},
    linked_exp::{LinkedF32ExpResourceBinding, LinkedF32ExpResourceDescriptor},
    ptx::{
        KernelSemanticProgram, LinkedF32ExpRequest, PrimaryLinkedRenderedKernelCache, PtxBinding,
    },
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
    #[serde(with = "buffer_desc_serde")]
    pub input: BufferDesc,
    #[serde(with = "buffer_desc_serde")]
    pub output: BufferDesc,
    pub owner_device: u32,
    pub sm: u32,
}

mod buffer_desc_serde {
    use crate::{BufferDesc, DType, Shape};
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    #[derive(Deserialize, Serialize)]
    #[serde(deny_unknown_fields)]
    struct WireBufferDesc {
        id: u64,
        shape: Vec<usize>,
        dtype: DType,
        bytes: usize,
        alignment: usize,
        read_only: bool,
    }

    pub fn serialize<S>(desc: &BufferDesc, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if desc.view.is_some() {
            return Err(serde::ser::Error::custom(
                "linked Exp batch descriptors cannot contain views",
            ));
        }
        WireBufferDesc {
            id: desc.id,
            shape: desc.shape.dims().to_vec(),
            dtype: desc.dtype,
            bytes: desc.bytes,
            alignment: desc.alignment,
            read_only: desc.read_only,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BufferDesc, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WireBufferDesc::deserialize(deserializer)?;
        let desc = BufferDesc {
            id: wire.id,
            shape: Shape::new(wire.shape),
            dtype: wire.dtype,
            bytes: wire.bytes,
            alignment: wire.alignment,
            read_only: wire.read_only,
            view: None,
        };
        crate::schedule::validate_buffer_desc(&desc)
            .map_err(|_| D::Error::custom("invalid linked Exp batch buffer descriptor"))?;
        Ok(desc)
    }
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

/// Data-only two-consumer preparation. `candidate_id` and `target_id` are
/// logical staging records, never caller-provided writable candidates.
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

/// Caller-lease proof for the v2 batch.  It is intentionally data-only: the
/// candidate identities are logical and no allocation, stream, cache, or
/// driver operation occurs during rebind.
pub struct BoundPreparedLinkedF32ExpBatchCapture<'a> {
    prepared: &'a PreparedLinkedF32ExpBatchCapture,
    inputs: [BufferView<'a>; 2],
    targets: [BufferView<'a>; 2],
}

impl LinkedF32ExpBatchArtifact {
    pub fn from_capture_requests(
        capture: &CapturedSchedule,
        primary: &PrimaryContext,
        sm: u32,
        records: &[(LinkedF32ExpRequest, LinkedF32ExpResourceDescriptor)],
    ) -> Result<Self, crate::ptx::PtxError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked Exp batch captured schedule"))?;
        if capture.items.len() != 2 || capture.requested.len() != 2 || records.len() != 2 {
            return Err(invalid("linked Exp batch cardinality"));
        }
        if !capture
            .requested
            .iter()
            .copied()
            .eq(capture.items.iter().map(|item| item.primary_output().id))
        {
            return Err(invalid("linked Exp batch requested outputs"));
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
        let mut artifact = Self {
            schema_version: LINKED_F32_EXP_BATCH_SCHEMA_VERSION,
            artifact_identity: String::new(),
            capture_identity: capture.identity,
            slots,
        };
        artifact.validate()?;
        artifact.artifact_identity = artifact.identity()?;
        Ok(artifact)
    }
    pub fn encode(&self) -> Result<Vec<u8>, crate::ptx::PtxError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|_| invalid("linked Exp batch encoding"))
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, crate::ptx::PtxError> {
        let value = serde_json::from_slice::<Self>(bytes)
            .map_err(|_| invalid("linked Exp batch encoding"))?;
        if serde_json::to_vec(&value).map_err(|_| invalid("linked Exp batch encoding"))? != bytes {
            return Err(invalid("linked Exp batch noncanonical"));
        }
        value.validate()?;
        Ok(value)
    }
    pub fn rebind(
        &self,
        capture: &CapturedSchedule,
        primary: &PrimaryContext,
        sm: u32,
        records: &BTreeMap<
            String,
            (
                LinkedF32ExpRequest,
                LinkedF32ExpResourceDescriptor,
                Vec<LinkInput>,
            ),
        >,
    ) -> Result<BoundLinkedF32ExpBatchResources, crate::ptx::PtxError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked Exp batch captured schedule"))?;
        self.validate()?;
        if capture.identity != self.capture_identity
            || records.len() != 2
            || !capture
                .requested
                .iter()
                .copied()
                .eq(self.slots.iter().map(|slot| slot.output.id))
        {
            return Err(invalid("linked Exp batch rebind inventory"));
        }
        let mut bindings = BTreeMap::new();
        for slot in &self.slots {
            let (request, descriptor, payload) = records
                .get(&slot.key)
                .ok_or_else(|| invalid("linked Exp batch slot"))?;
            if request.identity() != slot.request_identity
                || descriptor.resource_identity != slot.resource_identity
                || descriptor.device != slot.owner_device
                || descriptor.sm != slot.sm
            {
                return Err(invalid("linked Exp batch resource linkage"));
            }
            bindings.insert(slot.key.clone(), descriptor.rebind(primary, sm, payload)?);
        }
        Ok(BoundLinkedF32ExpBatchResources {
            artifact: self.clone(),
            bindings,
        })
    }
    fn validate(&self) -> Result<(), crate::ptx::PtxError> {
        if self.schema_version != LINKED_F32_EXP_BATCH_SCHEMA_VERSION
            || self.capture_identity == 0
            || self.slots.len() != 2
        {
            return Err(invalid("linked Exp batch schema"));
        }
        let mut keys = BTreeSet::new();
        let mut items = BTreeSet::new();
        let mut buffers = BTreeSet::new();
        for slot in &self.slots {
            if slot.key.is_empty()
                || !keys.insert(&slot.key)
                || !items.insert(slot.item_id)
                || slot.request_identity.is_empty()
                || slot.resource_identity.is_empty()
                || slot.rendered_identity.is_empty()
                || slot.owner_device > i32::MAX as u32
                || slot.sm == 0
                || slot.input.dtype != DType::F32
                || slot.output.dtype != DType::F32
                || slot.input.shape != slot.output.shape
                || slot.input.bytes == 0
                || slot.input.bytes != slot.output.bytes
                || !slot.input.read_only
                || slot.output.read_only
                || slot.input.view.is_some()
                || slot.output.view.is_some()
                || !buffers.insert(slot.input.id)
                || !buffers.insert(slot.output.id)
            {
                return Err(invalid("linked Exp batch slot"));
            }
        }
        if self.slots[0].item_id >= self.slots[1].item_id
            || self.slots[0].owner_device != self.slots[1].owner_device
            || self.slots[0].sm != self.slots[1].sm
        {
            return Err(invalid("linked Exp batch ordering/owner"));
        }
        if !self.artifact_identity.is_empty() && self.artifact_identity != self.identity()? {
            return Err(invalid("linked Exp batch identity"));
        }
        Ok(())
    }
    fn identity(&self) -> Result<String, crate::ptx::PtxError> {
        let mut canonical = self.clone();
        canonical.artifact_identity.clear();
        let bytes =
            serde_json::to_vec(&canonical).map_err(|_| invalid("linked Exp batch encoding"))?;
        let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        Ok(format!(
            "linked-f32-exp-batch-v{}:{hash:016x}",
            self.schema_version
        ))
    }
}

impl PreparedLinkedF32ExpBatchCapture {
    pub fn prepare(
        capture: &CapturedSchedule,
        artifact: &LinkedF32ExpBatchArtifact,
        bound: &BoundLinkedF32ExpBatchResources,
    ) -> Result<Self, crate::ptx::PtxError> {
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked Exp batch captured schedule"))?;
        if capture.identity != artifact.capture_identity
            || bound.artifact.artifact_identity != artifact.artifact_identity
            || bound.bindings.len() != 2
            || !capture
                .requested
                .iter()
                .copied()
                .eq(artifact.slots.iter().map(|slot| slot.output.id))
        {
            return Err(invalid("linked Exp batch prepared linkage"));
        }
        let mut slots = Vec::new();
        for resource in &artifact.slots {
            let item = capture
                .items
                .iter()
                .find(|item| item.id == resource.item_id)
                .ok_or_else(|| invalid("linked Exp batch item"))?;
            if !item.dependencies.is_empty()
                || !item.consumers.is_empty()
                || item.ordered_inputs().len() != 1
                || item.primary_output() != &resource.output
                || item.ordered_inputs()[0].desc != resource.input
            {
                return Err(invalid("linked Exp batch dependency/ABI"));
            }
            let candidate_id = resource
                .output
                .id
                .checked_add(0x8000_0000_0000_0000)
                .ok_or_else(|| invalid("linked Exp batch candidate"))?;
            slots.push(PreparedLinkedF32ExpBatchSlot {
                key: resource.key.clone(),
                item_id: item.id,
                input: resource.input.clone(),
                candidate_id,
                target: resource.output.clone(),
                request_identity: resource.request_identity.clone(),
                resource_identity: resource.resource_identity.clone(),
            });
        }
        Ok(Self {
            capture_identity: capture.identity,
            artifact_identity: artifact.artifact_identity.clone(),
            slots,
        })
    }
    pub fn rebind_leases<'a>(
        &'a self,
        primary: &PrimaryContext,
        leases: &BTreeMap<String, (&'a DeviceBuffer, &'a DeviceBuffer)>,
    ) -> Result<BoundPreparedLinkedF32ExpBatchCapture<'a>, crate::ptx::PtxError> {
        if self.slots.len() != 2 || leases.len() != 2 {
            return Err(invalid("linked Exp batch lease inventory"));
        }
        let mut pairs = Vec::with_capacity(2);
        for slot in &self.slots {
            let (input, target) = leases
                .get(&slot.key)
                .ok_or_else(|| invalid("linked Exp batch lease slot"))?;
            let input = input.view();
            let target = target.view();
            if !input.belongs_to_primary(primary)
                || !target.belongs_to_primary(primary)
                || input.device() != primary.device()
                || target.device() != primary.device()
                || input.len() != slot.input.bytes
                || target.len() != slot.target.bytes
                || input.is_empty()
                || input.device_ptr().map_err(crate::ptx::PtxError::Cuda)?
                    == target.device_ptr().map_err(crate::ptx::PtxError::Cuda)?
            {
                return Err(invalid("linked Exp batch lease ABI"));
            }
            pairs.push((input, target));
        }
        if pairs[0]
            .0
            .device_ptr()
            .map_err(crate::ptx::PtxError::Cuda)?
            == pairs[1]
                .0
                .device_ptr()
                .map_err(crate::ptx::PtxError::Cuda)?
            || pairs[0]
                .1
                .device_ptr()
                .map_err(crate::ptx::PtxError::Cuda)?
                == pairs[1]
                    .1
                    .device_ptr()
                    .map_err(crate::ptx::PtxError::Cuda)?
        {
            return Err(invalid("linked Exp batch lease alias"));
        }
        Ok(BoundPreparedLinkedF32ExpBatchCapture {
            prepared: self,
            inputs: [pairs[0].0, pairs[1].0],
            targets: [pairs[0].1, pairs[1].1],
        })
    }
}
impl BoundPreparedLinkedF32ExpBatchCapture<'_> {
    pub fn prepared(&self) -> &PreparedLinkedF32ExpBatchCapture {
        self.prepared
    }
    pub fn inputs(&self) -> [BufferView<'_>; 2] {
        self.inputs
    }
    pub fn targets(&self) -> [BufferView<'_>; 2] {
        self.targets
    }
}

/// Executes only the checked two-independent-consumer v2 proof.  Generic
/// capture replay never calls this entrypoint.
pub fn execute_prepared_linked_f32_exp_batch(
    bound: &BoundPreparedLinkedF32ExpBatchCapture<'_>,
    primary: &PrimaryContext,
    sm: u32,
    requests: &[LinkedF32ExpRequest],
    cache: &PrimaryLinkedRenderedKernelCache,
) -> Result<(), crate::ptx::PtxError> {
    let prepared = bound.prepared();
    if prepared.slots.len() != 2 || requests.len() != 2 || sm == 0 {
        return Err(invalid("linked Exp batch execution inventory"));
    }
    for ((slot, request), (input, target)) in prepared
        .slots
        .iter()
        .zip(requests)
        .zip(bound.inputs().into_iter().zip(bound.targets()))
    {
        if request.identity() != slot.request_identity
            || slot.input.dtype != DType::F32
            || slot.target.dtype != DType::F32
            || slot.input.shape != slot.target.shape
            || slot.input.bytes == 0
            || slot.input.bytes != slot.target.bytes
            || input.len() != slot.input.bytes
            || target.len() != slot.target.bytes
            || !input.belongs_to_primary(primary)
            || !target.belongs_to_primary(primary)
            || input.device() != primary.device()
            || target.device() != primary.device()
        {
            return Err(invalid("linked Exp batch execution binding"));
        }
    }
    let sizes = prepared
        .slots
        .iter()
        .map(|slot| {
            std::num::NonZeroUsize::new(slot.target.bytes)
                .ok_or_else(|| invalid("linked Exp batch zero output"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // All proof checks complete before any driver-visible operation.
    let first = primary.allocate(sizes[0])?;
    let second = primary.allocate(sizes[1])?;
    let stream = primary.stream()?;
    let candidates = [first, second];
    for ((request, input), candidate) in requests.iter().zip(bound.inputs()).zip(candidates.iter())
    {
        let kernel = request.load(primary, cache)?;
        kernel.launch(
            &stream,
            &[
                PtxBinding {
                    buffer: input,
                    dtype: DType::F32,
                    mutable: false,
                },
                PtxBinding {
                    buffer: candidate.view(),
                    dtype: DType::F32,
                    mutable: true,
                },
            ],
            true,
        )?;
    }
    primary
        .commit_caller_owned_outputs_with_rollback(
            &stream,
            &[
                PrimaryOutputCommit::new(candidates[0].view(), bound.targets()[0]),
                PrimaryOutputCommit::new(candidates[1].view(), bound.targets()[1]),
            ],
        )
        .map_err(|error| invalid(&format!("linked Exp batch output commit: {error}")))
}

fn validate_item(
    item: &crate::ScheduleItem,
    request: &LinkedF32ExpRequest,
    descriptor: &LinkedF32ExpResourceDescriptor,
    primary: &PrimaryContext,
    sm: u32,
) -> Result<(), crate::ptx::PtxError> {
    if !item.dependencies.is_empty()
        || !item.consumers.is_empty()
        || item.ordered_inputs().len() != 1
        || !item.outputs.is_single()
        || item.ordered_inputs()[0].desc.view.is_some()
        || item.primary_output().view.is_some()
        || descriptor.request_identity != request.identity()
        || descriptor.device != primary.device().0
        || descriptor.sm != sm
        || item.ordered_inputs()[0].desc.dtype != DType::F32
        || item.primary_output().dtype != DType::F32
        || item.ordered_inputs()[0].desc.shape != item.primary_output().shape
        || item.primary_output().bytes == 0
    {
        return Err(invalid("linked Exp batch item"));
    }
    match request.rendered().semantic_program.as_ref() {
        Some(KernelSemanticProgram::UOp(program)) if program.as_ref() == &item.kernel => Ok(()),
        _ => Err(invalid("linked Exp batch UOp")),
    }
}
fn invalid(message: &str) -> crate::ptx::PtxError {
    crate::ptx::PtxError::InvalidBinding(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Driver, Graph, PtxRenderer, Shape,
        cuda::{LinkInput, NvvmExportContract, NvvmProducerContract, NvvmPrototype},
    };
    use std::sync::Arc;

    type FixtureRecord = (
        LinkedF32ExpRequest,
        LinkedF32ExpResourceDescriptor,
        Vec<LinkInput>,
    );
    type Fixture = (
        Arc<crate::cuda::tests::Mock>,
        CapturedSchedule,
        PrimaryContext,
        LinkedF32ExpBatchArtifact,
        Vec<FixtureRecord>,
    );

    fn fixture() -> Fixture {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut graph = Graph::new();
        let left_input = graph.input("left", Shape::from([2]));
        let right_input = graph.input("right", Shape::from([3]));
        let left = graph.unary(crate::UnaryOp::Exp, left_input).unwrap();
        let right = graph.unary(crate::UnaryOp::Exp, right_input).unwrap();
        let schedule = crate::schedule_many(&graph, &[left, right]).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[left, right]).unwrap();
        assert_eq!(capture.items.len(), 2);
        let mut records = Vec::new();
        for (index, item) in capture.items.iter().enumerate() {
            let payload = format!("attested-nvvm-{index}").into_bytes();
            let export =
                NvvmExportContract::new("__nv_expf".into(), NvvmPrototype::F32ToF32).unwrap();
            let contract =
                NvvmProducerContract::new(11, 4, 1, 20, 90, vec![export], &payload).unwrap();
            let link =
                LinkInput::nvvm(&format!("libdevice-{index}.bc"), payload, contract).unwrap();
            let renderer = PtxRenderer::new(80).unwrap();
            let entry = renderer
                .render_linked_f32_exp(&item.kernel, std::slice::from_ref(&link))
                .unwrap()
                .entry;
            let request =
                LinkedF32ExpRequest::new(renderer, &item.kernel, vec![link.clone()], &entry, 32)
                    .unwrap();
            let descriptor =
                LinkedF32ExpResourceDescriptor::from_request(&request, crate::DeviceId(0), 80)
                    .unwrap();
            records.push((request, descriptor, vec![link]));
        }
        let pairs = records
            .iter()
            .map(|(request, descriptor, _)| (request.clone(), descriptor.clone()))
            .collect::<Vec<_>>();
        let artifact =
            LinkedF32ExpBatchArtifact::from_capture_requests(&capture, &primary, 80, &pairs)
                .unwrap();
        (mock, capture, primary, artifact, records)
    }

    #[test]
    fn linked_exp_batch_v2_round_trips_real_independent_capture_without_driver_work() {
        let (mock, capture, primary, artifact, records) = fixture();
        let before = mock.calls().len();
        let bytes = artifact.encode().unwrap();
        assert_eq!(bytes, artifact.encode().unwrap());
        assert_eq!(LinkedF32ExpBatchArtifact::decode(&bytes).unwrap(), artifact);
        let witnesses = artifact
            .slots
            .iter()
            .zip(records)
            .map(|(slot, (request, descriptor, payload))| {
                (slot.key.clone(), (request, descriptor, payload))
            })
            .collect::<BTreeMap<_, _>>();
        let bound = artifact.rebind(&capture, &primary, 80, &witnesses).unwrap();
        let prepared =
            PreparedLinkedF32ExpBatchCapture::prepare(&capture, &artifact, &bound).unwrap();
        assert_eq!(prepared.slots.len(), 2);
        assert_ne!(prepared.slots[0].input.shape, prepared.slots[1].input.shape);
        assert_ne!(
            prepared.slots[0].candidate_id,
            prepared.slots[1].candidate_id
        );
        assert_eq!(mock.calls().len(), before);
    }

    #[test]
    fn linked_exp_batch_v2_rejects_noncanonical_cardinality_order_and_tamper_preflight() {
        let (mock, capture, primary, artifact, records) = fixture();
        let before = mock.calls().len();
        let pairs = records
            .iter()
            .map(|(request, descriptor, _)| (request.clone(), descriptor.clone()))
            .collect::<Vec<_>>();
        let mut swapped_requested = capture.clone();
        swapped_requested.requested.reverse();
        swapped_requested.identity =
            crate::schedule::artifact::identity(&swapped_requested).unwrap();
        assert!(crate::schedule::artifact::validate_capture(&swapped_requested).is_ok());
        assert!(
            LinkedF32ExpBatchArtifact::from_capture_requests(
                &swapped_requested,
                &primary,
                80,
                &pairs,
            )
            .is_err()
        );
        let mut reversed = artifact.clone();
        reversed.slots.reverse();
        assert!(reversed.encode().is_err());
        let mut one = artifact.clone();
        one.slots.pop();
        assert!(one.encode().is_err());
        let mut three = artifact.clone();
        three.slots.push(three.slots[0].clone());
        assert!(three.encode().is_err());
        let mut tampered = artifact.clone();
        tampered.slots[0].sm = 81;
        assert!(tampered.encode().is_err());
        assert!(LinkedF32ExpBatchArtifact::decode(b"{}").is_err());
        let mut witnesses = artifact
            .slots
            .iter()
            .zip(records)
            .map(|(slot, (request, descriptor, payload))| {
                (slot.key.clone(), (request, descriptor, payload))
            })
            .collect::<BTreeMap<_, _>>();
        let missing = witnesses.keys().next().cloned().unwrap();
        witnesses.remove(&missing);
        assert!(artifact.rebind(&capture, &primary, 80, &witnesses).is_err());
        assert_eq!(mock.calls().len(), before);
    }

    #[test]
    fn linked_exp_batch_v2_rebinds_exact_two_caller_leases_without_driver_work() {
        use std::num::NonZeroUsize;
        let (mock, capture, primary, artifact, records) = fixture();
        let witnesses = artifact
            .slots
            .iter()
            .zip(records)
            .map(|(slot, (request, descriptor, payload))| {
                (slot.key.clone(), (request, descriptor, payload))
            })
            .collect::<BTreeMap<_, _>>();
        let resources = artifact.rebind(&capture, &primary, 80, &witnesses).unwrap();
        let prepared =
            PreparedLinkedF32ExpBatchCapture::prepare(&capture, &artifact, &resources).unwrap();
        let mut leases = BTreeMap::new();
        let mut owned = Vec::new();
        for slot in &prepared.slots {
            let input = primary
                .allocate(NonZeroUsize::new(slot.input.bytes).unwrap())
                .unwrap();
            let target = primary
                .allocate(NonZeroUsize::new(slot.target.bytes).unwrap())
                .unwrap();
            owned.push((input, target));
        }
        for (slot, pair) in prepared.slots.iter().zip(&owned) {
            leases.insert(slot.key.clone(), (&pair.0, &pair.1));
        }
        let before = mock.calls().len();
        let bound = prepared.rebind_leases(&primary, &leases).unwrap();
        assert_eq!(bound.prepared().slots, prepared.slots);
        assert_eq!(bound.inputs()[0].len(), prepared.slots[0].input.bytes);
        assert_eq!(bound.targets()[1].len(), prepared.slots[1].target.bytes);
        assert_eq!(mock.calls().len(), before);
        let missing = prepared.slots[0].key.clone();
        leases.remove(&missing);
        assert!(prepared.rebind_leases(&primary, &leases).is_err());
        leases.insert(missing.clone(), (&owned[0].0, &owned[0].1));
        leases.insert("extra".into(), (&owned[0].0, &owned[0].1));
        assert!(prepared.rebind_leases(&primary, &leases).is_err());
        leases.remove("extra");
        leases.insert(prepared.slots[1].key.clone(), (&owned[1].0, &owned[0].1));
        assert!(prepared.rebind_leases(&primary, &leases).is_err());
        assert_eq!(mock.calls().len(), before);
    }

    #[test]
    fn linked_exp_batch_launcher_exposes_partial_state_when_rollback_fails() {
        use crate::cuda::PrimaryOutputCommitPhase;
        use std::num::NonZeroUsize;
        let (mock, capture, primary, artifact, records) = fixture();
        let requests = records
            .iter()
            .map(|(request, _, _)| request.clone())
            .collect::<Vec<_>>();
        let witnesses = artifact
            .slots
            .iter()
            .zip(records)
            .map(|(slot, (request, descriptor, payload))| {
                (slot.key.clone(), (request, descriptor, payload))
            })
            .collect::<BTreeMap<_, _>>();
        let resources = artifact.rebind(&capture, &primary, 80, &witnesses).unwrap();
        let prepared =
            PreparedLinkedF32ExpBatchCapture::prepare(&capture, &artifact, &resources).unwrap();
        let values = [vec![-1.0_f32, 0.0], vec![0.5_f32, 1.0, 2.0]];
        let mut owned = Vec::new();
        let mut leases = BTreeMap::new();
        for (slot, values) in prepared.slots.iter().zip(&values) {
            let input = primary
                .allocate(NonZeroUsize::new(slot.input.bytes).unwrap())
                .unwrap();
            let target = primary
                .allocate(NonZeroUsize::new(slot.target.bytes).unwrap())
                .unwrap();
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            input.copy_from(0, &bytes).unwrap();
            target.copy_from(0, &vec![0x5a; bytes.len()]).unwrap();
            owned.push((input, target, bytes));
        }
        for (slot, pair) in prepared.slots.iter().zip(&owned) {
            leases.insert(slot.key.clone(), (&pair.0, &pair.1));
        }
        let bound = prepared.rebind_leases(&primary, &leases).unwrap();
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let baseline = mock.live_allocation_count(primary.owner());
        execute_prepared_linked_f32_exp_batch(&bound, &primary, 80, &requests, &cache).unwrap();
        for ((input, target, source), values) in owned.iter().zip(&values) {
            let mut actual_input = vec![0; source.len()];
            input.copy_to(0, &mut actual_input).unwrap();
            assert_eq!(&actual_input, source);
            let mut actual = vec![0; source.len()];
            target.copy_to(0, &mut actual).unwrap();
            for (got, want) in actual
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .zip(values.iter().copied().map(f32::exp))
            {
                assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0));
            }
        }

        for (_, target, source) in &owned {
            target.copy_from(0, &vec![0x5a; source.len()]).unwrap();
        }
        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Commit, 0, 71);
        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Restore, 0, 72);
        let error = execute_prepared_linked_f32_exp_batch(&bound, &primary, 80, &requests, &cache)
            .unwrap_err();
        assert!(error.to_string().contains("partially modified"));
        let mut first = vec![0; owned[0].2.len()];
        owned[0].1.copy_to(0, &mut first).unwrap();
        assert_ne!(first, vec![0x5a; owned[0].2.len()]);
        let mut second = vec![0; owned[1].2.len()];
        owned[1].1.copy_to(0, &mut second).unwrap();
        assert_eq!(second, vec![0x5a; owned[1].2.len()]);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
    }
}
