//! Data-only preparation for the sole opt-in linked F32 Exp capture route.
//!
//! This is deliberately not a `CapturedReplayExecutor` backend. It verifies
//! the decoded capture ABI against the typed linked request and bound sidecar
//! before a future dedicated launcher may access a linked cache or driver.

use crate::{
    linked_resource_artifact::{BoundLinkedF32ExpResources, LinkedF32ExpResourceArtifact},
    ptx::{KernelSemanticProgram, LinkedF32ExpRequest},
    BufferDesc, CapturedSchedule, DType, PrimaryContext,
};

/// A validated, non-executable single-item capture ABI. The input and output
/// descriptors remain schedule-owned; this proof owns no buffers or payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedLinkedF32ExpCapture {
    capture_identity: u64,
    sidecar_identity: String,
    slot_key: String,
    item_id: u64,
    item_node: crate::NodeId,
    input: BufferDesc,
    output: BufferDesc,
    rendered_identity: String,
    request_identity: String,
}

impl PreparedLinkedF32ExpCapture {
    /// Validates all resource, owner, consumer, UOp, and ABI linkage without
    /// creating a stream, allocation, cache entry, or driver call.
    pub fn prepare(
        capture: &CapturedSchedule,
        sidecar: &LinkedF32ExpResourceArtifact,
        bound: &BoundLinkedF32ExpResources,
        primary: &PrimaryContext,
        sm: u32,
        request: &LinkedF32ExpRequest,
    ) -> Result<Self, crate::ptx::PtxError> {
        let sidecar_bytes = sidecar.encode()?;
        let canonical = LinkedF32ExpResourceArtifact::decode(&sidecar_bytes)?;
        if &canonical != sidecar
            || bound.artifact() != sidecar
            || capture.identity != sidecar.capture_identity
            || sidecar.slots.len() != 1
        {
            return Err(invalid("linked Exp capture resource identity"));
        }
        let slot = &sidecar.slots[0];
        if slot.sm != sm
            || slot.owner_device != primary.device().0
            || slot.consumer_request_identity != request.identity()
            || bound.len() != 1
        {
            return Err(invalid("linked Exp capture resource owner"));
        }
        let binding = bound
            .binding(&slot.key)
            .ok_or_else(|| invalid("linked Exp capture resource slot"))?;
        binding.validate_owner(primary)?;
        if capture.items.len() != 1
            || capture.requested.len() != 1
            || capture.inputs.len() != 1
        {
            return Err(invalid("linked Exp capture must contain one kernel/input/output"));
        }
        let item = &capture.items[0];
        if item.id != 0
            || item.is_effect()
            || item.boundary.is_some()
            || !item.dependencies.is_empty()
            || !item.consumers.is_empty()
            || !item.external_materializations.is_empty()
            || !item.ordered_quantized_inputs().is_empty()
            || item.ordered_inputs().len() != 1
            || !item.outputs.is_single()
            || capture.requested[0] != item.primary_output().id
        {
            return Err(invalid("linked Exp capture schedule shape"));
        }
        let input = &item.ordered_inputs()[0].desc;
        let output = item.primary_output();
        if capture.inputs[0].node != item.ordered_inputs()[0].input_node
            || capture.inputs[0].desc != *input
            || input.dtype != DType::F32
            || output.dtype != DType::F32
            || input.shape != output.shape
            || input.bytes != output.bytes
            || !input.read_only
            || output.read_only
        {
            return Err(invalid("linked Exp capture buffer ABI"));
        }
        let rendered = request.rendered();
        let Some(KernelSemanticProgram::UOp(program)) = rendered.semantic_program.as_ref() else {
            return Err(invalid("linked Exp capture semantic program"));
        };
        if program.as_ref() != &item.kernel
            || rendered.buffers.len() != 2
            || rendered.extent != input.shape.numel().map_err(|_| crate::ptx::PtxError::Overflow)?
        {
            return Err(invalid("linked Exp capture rendered UOp"));
        }
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        let rendered_output = &rendered.buffers[1];
        if !rendered_output.mutable
            || rendered_output.id != output.id
            || rendered_output.dtype != output.dtype
            || rendered_output.source_shape != output.shape
            || rendered_output.elements.checked_mul(rendered_output.dtype.itemsize())
                != Some(output.bytes)
        {
            return Err(invalid("linked Exp capture rendered output ABI"));
        }
        Ok(Self {
            capture_identity: capture.identity,
            sidecar_identity: sidecar.artifact_identity.clone(),
            slot_key: slot.key.clone(),
            item_id: item.id,
            item_node: item.node,
            input: input.clone(),
            output: output.clone(),
            rendered_identity: rendered.cache_key.clone(),
            request_identity: request.identity().to_owned(),
        })
    }

    pub fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
    pub fn sidecar_identity(&self) -> &str {
        &self.sidecar_identity
    }
    pub fn slot_key(&self) -> &str {
        &self.slot_key
    }
    pub fn item_id(&self) -> u64 {
        self.item_id
    }
    pub fn item_node(&self) -> crate::NodeId {
        self.item_node
    }
    pub fn input(&self) -> &BufferDesc {
        &self.input
    }
    pub fn output(&self) -> &BufferDesc {
        &self.output
    }
    pub fn rendered_identity(&self) -> &str {
        &self.rendered_identity
    }
    /// The typed linked request that was validated with this schedule proof.
    /// Prepared proofs from before this field existed are intentionally not
    /// execution-capable: no decoder manufactures this identity.
    pub fn request_identity(&self) -> &str {
        &self.request_identity
    }
}

fn invalid(message: &str) -> crate::ptx::PtxError {
    crate::ptx::PtxError::InvalidBinding(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cuda::{LinkInput, NvvmExportContract, NvvmProducerContract, NvvmPrototype},
        Driver, Graph, PtxRenderer, Shape,
    };
    use std::{collections::BTreeMap, sync::Arc};

    fn fixture() -> (
        Arc<crate::cuda::tests::Mock>,
        CapturedSchedule,
        PrimaryContext,
        LinkedF32ExpRequest,
        LinkedF32ExpResourceArtifact,
        BoundLinkedF32ExpResources,
    ) {
        let mock = Arc::new(crate::cuda::tests::Mock::default());
        let primary = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::from([3]));
        let output = graph.exp(input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let payload = b"attested-nvvm".to_vec();
        let export = NvvmExportContract::new("__nv_expf".into(), NvvmPrototype::F32ToF32)
            .unwrap();
        let contract = NvvmProducerContract::new(11, 4, 1, 20, 90, vec![export], &payload)
            .unwrap();
        let link = LinkInput::nvvm("libdevice.bc", payload, contract).unwrap();
        let renderer = PtxRenderer::new(80).unwrap();
        let entry = renderer
            .render_linked_f32_exp(&capture.items[0].kernel, std::slice::from_ref(&link))
            .unwrap()
            .entry;
        let request = LinkedF32ExpRequest::new(
            renderer,
            &capture.items[0].kernel,
            vec![link.clone()],
            &entry,
            32,
        )
        .unwrap();
        let descriptor = crate::LinkedF32ExpResourceDescriptor::from_request(
            &request,
            crate::DeviceId(0),
            80,
        )
        .unwrap();
        let sidecar = LinkedF32ExpResourceArtifact::from_capture_request(
            &capture,
            &descriptor,
            &request,
        )
        .unwrap();
        let key = sidecar.slots[0].key.clone();
        let bound = sidecar
            .rebind(
                &capture,
                &primary,
                80,
                &request,
                &BTreeMap::from([(key.clone(), descriptor)]),
                &BTreeMap::from([(key, vec![link])]),
            )
            .unwrap();
        (mock, capture, primary, request, sidecar, bound)
    }

    #[test]
    fn linked_exp_capture_prepare_is_exact_and_preflight_only() {
        let (mock, capture, primary, request, sidecar, bound) = fixture();
        let before = mock.calls().len();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture, &sidecar, &bound, &primary, 80, &request,
        )
        .unwrap();
        assert_eq!(prepared.capture_identity(), capture.identity);
        assert_eq!(prepared.request_identity(), request.identity());
        assert_eq!(prepared.input(), &capture.items[0].ordered_inputs()[0].desc);
        assert_eq!(prepared.output(), capture.items[0].primary_output());
        assert_eq!(mock.calls().len(), before);

        let mut wrong_capture = capture.clone();
        wrong_capture.identity ^= 1;
        assert!(PreparedLinkedF32ExpCapture::prepare(
            &wrong_capture,
            &sidecar,
            &bound,
            &primary,
            80,
            &request,
        )
        .is_err());
        assert!(PreparedLinkedF32ExpCapture::prepare(
            &capture, &sidecar, &bound, &primary, 81, &request,
        )
        .is_err());
        assert_eq!(mock.calls().len(), before);
    }
}
