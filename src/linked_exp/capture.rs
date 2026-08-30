//! Preparation and dedicated execution for the sole opt-in linked F32 Exp route.
//!
//! This is deliberately not a `CapturedReplayExecutor` backend. It verifies
//! the decoded capture ABI against the typed linked request and bound sidecar,
//! rebinds caller-owned leases, then exposes only the dedicated launcher
//! below. Caller-output publication uses best-effort rollback and reports
//! possible partial mutation when restoration itself fails.

use crate::{
    BufferDesc, CapturedSchedule, DType, PrimaryContext,
    cuda::{BufferView, DeviceBuffer, PrimaryOutputCommit},
    linked_exp::{BoundLinkedF32ExpResources, LinkedF32ExpResourceArtifact},
    ptx::{
        KernelSemanticProgram, LinkedF32ExpRequest, PrimaryLinkedRenderedKernelCache, PtxBinding,
    },
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

/// A validated, data-only single-item capture ABI. The input and output
/// descriptors remain schedule-owned; this proof owns no buffers or payloads
/// and becomes executable only after explicit lease rebind.
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
        crate::schedule::artifact::validate_capture(capture)
            .map_err(|_| invalid("linked Exp captured schedule"))?;
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
        if capture.items.len() != 1 || capture.requested.len() != 1 || capture.inputs.len() != 1 {
            return Err(invalid(
                "linked Exp capture must contain one kernel/input/output",
            ));
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
            || input.view.is_some()
            || output.view.is_some()
        {
            return Err(invalid("linked Exp capture buffer ABI"));
        }
        let rendered = request.rendered();
        let Some(KernelSemanticProgram::UOp(program)) = rendered.semantic_program.as_ref() else {
            return Err(invalid("linked Exp capture semantic program"));
        };
        if program.as_ref() != &item.kernel
            || rendered.buffers.len() != 2
            || rendered.extent
                != input
                    .shape
                    .numel()
                    .map_err(|_| crate::ptx::PtxError::Overflow)?
        {
            return Err(invalid("linked Exp capture rendered UOp"));
        }
        rendered.validate_schedule_bindings(item.ordered_inputs())?;
        let rendered_output = &rendered.buffers[1];
        if !rendered_output.mutable
            || rendered_output.id != output.id
            || rendered_output.dtype != output.dtype
            || rendered_output.source_shape != output.shape
            || rendered_output
                .elements
                .checked_mul(rendered_output.dtype.itemsize())
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

/// Closed external role inventory for the prepared one-consumer route. The
/// candidate is intentionally a logical staging identity only: callers
/// never provide a writable candidate lease, and the future launcher must
/// allocate it after this data-only rebind succeeds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PreparedLinkedF32ExpExternalRole {
    Input,
    FinalTarget,
    StagedCandidate,
}

/// Immutable payload-free external binding schema for one prepared Exp item.
/// It does not serialize CUDA pointers or caller payloads.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PreparedLinkedF32ExpBindingTable {
    version: u32,
    capture_identity: u64,
    request_identity: String,
    resource_slot: String,
    item_id: u64,
    input_id: u64,
    output_id: u64,
    owner_device: u32,
    sm: u32,
    input_role: PreparedLinkedF32ExpExternalRole,
    target_role: PreparedLinkedF32ExpExternalRole,
    candidate_role: PreparedLinkedF32ExpExternalRole,
    identity: String,
}

/// Borrowed, caller-owned views proved against a binding table. No allocation,
/// cache, stream, or driver operation occurs while creating this value.
pub struct BoundPreparedLinkedF32ExpCapture<'a> {
    prepared: &'a PreparedLinkedF32ExpCapture,
    input: BufferView<'a>,
    target: BufferView<'a>,
    identity: String,
}

impl PreparedLinkedF32ExpBindingTable {
    const VERSION: u32 = 1;

    pub fn from_prepared(
        prepared: &PreparedLinkedF32ExpCapture,
        primary: &PrimaryContext,
        sm: u32,
    ) -> Result<Self, crate::ptx::PtxError> {
        if prepared.request_identity().is_empty() || prepared.input.id == prepared.output.id {
            return Err(invalid("linked Exp prepared binding identity"));
        }
        let mut table = Self {
            version: Self::VERSION,
            capture_identity: prepared.capture_identity,
            request_identity: prepared.request_identity.clone(),
            resource_slot: prepared.slot_key.clone(),
            item_id: prepared.item_id,
            input_id: prepared.input.id,
            output_id: prepared.output.id,
            owner_device: primary.device().0,
            sm,
            input_role: PreparedLinkedF32ExpExternalRole::Input,
            target_role: PreparedLinkedF32ExpExternalRole::FinalTarget,
            candidate_role: PreparedLinkedF32ExpExternalRole::StagedCandidate,
            identity: String::new(),
        };
        table.identity = table.canonical_identity();
        Ok(table)
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    fn canonical_identity(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.version.hash(&mut hasher);
        self.capture_identity.hash(&mut hasher);
        self.request_identity.hash(&mut hasher);
        self.resource_slot.hash(&mut hasher);
        self.item_id.hash(&mut hasher);
        self.input_id.hash(&mut hasher);
        self.output_id.hash(&mut hasher);
        self.owner_device.hash(&mut hasher);
        self.sm.hash(&mut hasher);
        self.input_role.hash(&mut hasher);
        self.target_role.hash(&mut hasher);
        self.candidate_role.hash(&mut hasher);
        format!(
            "linked-exp-prepared-bindings-v{}:{:016x}",
            self.version,
            hasher.finish()
        )
    }
    pub fn rebind<'a>(
        &'a self,
        prepared: &'a PreparedLinkedF32ExpCapture,
        primary: &PrimaryContext,
        sm: u32,
        request: &LinkedF32ExpRequest,
        input: &'a DeviceBuffer,
        target: &'a DeviceBuffer,
    ) -> Result<BoundPreparedLinkedF32ExpCapture<'a>, crate::ptx::PtxError> {
        if self.version != Self::VERSION
            || self.identity != self.canonical_identity()
            || self.capture_identity != prepared.capture_identity
            || self.request_identity != prepared.request_identity
            || self.resource_slot != prepared.slot_key
            || self.item_id != prepared.item_id
            || self.input_id != prepared.input.id
            || self.output_id != prepared.output.id
            || self.owner_device != primary.device().0
            || self.sm != sm
            || self.input_role != PreparedLinkedF32ExpExternalRole::Input
            || self.target_role != PreparedLinkedF32ExpExternalRole::FinalTarget
            || self.candidate_role != PreparedLinkedF32ExpExternalRole::StagedCandidate
            || request.identity() != prepared.request_identity
        {
            return Err(invalid("linked Exp prepared binding linkage"));
        }
        let input = input.view();
        let target = target.view();
        if !input.belongs_to_primary(primary)
            || !target.belongs_to_primary(primary)
            || input.len() != prepared.input.bytes
            || target.len() != prepared.output.bytes
            || input.is_empty()
            || input.device() != primary.device()
            || target.device() != primary.device()
            || input.device_ptr().map_err(crate::ptx::PtxError::Cuda)?
                == target.device_ptr().map_err(crate::ptx::PtxError::Cuda)?
        {
            return Err(invalid("linked Exp prepared external lease ABI"));
        }
        Ok(BoundPreparedLinkedF32ExpCapture {
            prepared,
            input,
            target,
            identity: self.identity.clone(),
        })
    }
}
impl BoundPreparedLinkedF32ExpCapture<'_> {
    pub fn prepared(&self) -> &PreparedLinkedF32ExpCapture {
        self.prepared
    }
    pub fn input(&self) -> BufferView<'_> {
        self.input
    }
    pub fn target(&self) -> BufferView<'_> {
        self.target
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
}

/// Executes exactly the prepared, caller-attested F32 Exp capture route.
/// Generic captured replay deliberately never calls this entrypoint.
pub fn execute_prepared_linked_f32_exp(
    bound: &BoundPreparedLinkedF32ExpCapture<'_>,
    primary: &PrimaryContext,
    sm: u32,
    request: &LinkedF32ExpRequest,
    cache: &PrimaryLinkedRenderedKernelCache,
) -> Result<(), crate::ptx::PtxError> {
    let prepared = bound.prepared();
    if prepared.request_identity() != request.identity()
        || prepared.input.dtype != DType::F32
        || prepared.output.dtype != DType::F32
        || prepared.input.shape != prepared.output.shape
        || prepared.input.bytes != prepared.output.bytes
        || bound.input().len() != prepared.input.bytes
        || bound.target().len() != prepared.output.bytes
        || !bound.input().belongs_to_primary(primary)
        || !bound.target().belongs_to_primary(primary)
        || bound.input().device() != primary.device()
        || bound.target().device() != primary.device()
        || sm == 0
    {
        return Err(invalid("prepared linked F32 Exp execution binding"));
    }
    let bytes = std::num::NonZeroUsize::new(prepared.output.bytes)
        .ok_or_else(|| invalid("prepared linked F32 Exp zero output"))?;
    // Only after the proof and all caller leases are validated do we access a
    // stream, cache, or allocation. The candidate is never caller-owned.
    let candidate = primary.allocate(bytes)?;
    let stream = primary.stream()?;
    let kernel = request.load(primary, cache)?;
    kernel.launch(
        &stream,
        &[
            PtxBinding {
                buffer: bound.input(),
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
    primary
        .commit_caller_owned_outputs_with_rollback(
            &stream,
            &[PrimaryOutputCommit::new(candidate.view(), bound.target())],
        )
        .map_err(|error| invalid(&format!("prepared linked F32 Exp output commit: {error}")))
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
    use std::{collections::BTreeMap, sync::Arc};

    fn fixture() -> (
        Arc<crate::cuda::tests::Mock>,
        CapturedSchedule,
        PrimaryContext,
        LinkedF32ExpRequest,
        LinkedF32ExpResourceArtifact,
        BoundLinkedF32ExpResources,
    ) {
        fixture_for(Shape::from([3]))
    }

    fn fixture_for(
        shape: Shape,
    ) -> (
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
        let input = graph.input("input", shape);
        let output = graph.unary(crate::UnaryOp::Exp, input).unwrap();
        let schedule = crate::schedule(&graph, output).unwrap();
        let capture = CapturedSchedule::capture(&graph, &schedule, &[output]).unwrap();
        let payload = b"attested-nvvm".to_vec();
        let export = NvvmExportContract::new("__nv_expf".into(), NvvmPrototype::F32ToF32).unwrap();
        let contract = NvvmProducerContract::new(11, 4, 1, 20, 90, vec![export], &payload).unwrap();
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
        let descriptor = crate::linked_exp::LinkedF32ExpResourceDescriptor::from_request(
            &request,
            crate::DeviceId(0),
            80,
        )
        .unwrap();
        let sidecar =
            LinkedF32ExpResourceArtifact::from_capture_request(&capture, &descriptor, &request)
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
        assert!(
            request
                .rendered()
                .source
                .contains(".extern .func (.param .b32 func_retval0) __nv_expf")
        );
        assert!(
            request
                .rendered()
                .source
                .contains("call.uni (exp_ret), __nv_expf, (exp_arg);")
        );
        assert!(
            PtxRenderer::new(80)
                .unwrap()
                .render(&capture.items[0].kernel)
                .is_err()
        );
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
        assert!(
            PreparedLinkedF32ExpCapture::prepare(
                &wrong_capture,
                &sidecar,
                &bound,
                &primary,
                80,
                &request,
            )
            .is_err()
        );
        let mut tampered_item = capture.clone();
        tampered_item.items[0].id ^= 1;
        assert_eq!(tampered_item.identity, capture.identity);
        assert!(
            PreparedLinkedF32ExpCapture::prepare(
                &tampered_item,
                &sidecar,
                &bound,
                &primary,
                80,
                &request,
            )
            .is_err()
        );
        assert!(
            PreparedLinkedF32ExpCapture::prepare(
                &capture, &sidecar, &bound, &primary, 81, &request,
            )
            .is_err()
        );
        assert_eq!(mock.calls().len(), before);
    }

    #[test]
    fn linked_exp_renderer_rejects_noncanonical_index_and_range_graphs() {
        use crate::{AddressSpace, AddressValue, Operation, UOp, UType};

        let (_, capture, _, request, _, _) = fixture();
        let renderer = PtxRenderer::new(80).unwrap();
        let canonical = &capture.items[0].kernel;
        let store = &canonical.sources()[0];
        let end_range = &canonical.sources()[1];
        let output_index = &store.sources()[0];
        let exp = &store.sources()[1];
        let load = &exp.sources()[0];
        let input_index = &load.sources()[0];
        let input_address = input_index.sources()[0].clone();
        let output_address = output_index.sources()[0].clone();
        let range = input_index.sources()[1].clone();
        let rebuild = |input_address: UOp,
                       input_offset: UOp,
                       output_address: UOp,
                       output_offset: UOp,
                       ended_range: UOp| {
            let input_index = UOp::from_operation(
                input_index.operation().clone(),
                input_index.ty(),
                vec![input_address, input_offset],
            );
            let output_index = UOp::from_operation(
                output_index.operation().clone(),
                output_index.ty(),
                vec![output_address, output_offset],
            );
            let load = UOp::from_operation(Operation::Load, load.ty(), vec![input_index]);
            let exp = UOp::from_operation(
                Operation::GraphUnary(crate::UnaryOp::Exp),
                exp.ty(),
                vec![load],
            );
            UOp::sink(vec![
                UOp::from_operation(Operation::Store, None, vec![output_index, exp]),
                UOp::from_operation(Operation::EndRange, None, vec![ended_range]),
            ])
        };

        let constant_index = rebuild(
            input_address.clone(),
            UOp::constant(0, UType::scalar(DType::I64)),
            output_address.clone(),
            range.clone(),
            range.clone(),
        );
        let wrong_bound = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![UOp::constant(4, UType::scalar(DType::I64))],
        );
        let out_of_bounds = rebuild(
            input_address.clone(),
            wrong_bound.clone(),
            output_address.clone(),
            wrong_bound.clone(),
            wrong_bound,
        );
        let wrong_address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: AddressSpace::Global,
                name: "b999".into(),
                element: UType::scalar(DType::F32),
            }),
            Some(UType::scalar(DType::F32)),
            vec![],
        );
        let wrong_buffer_identity = rebuild(
            wrong_address,
            range.clone(),
            output_address.clone(),
            range.clone(),
            range.clone(),
        );
        let different_end = UOp::from_operation(
            Operation::Range(1),
            Some(UType::scalar(DType::I64)),
            vec![UOp::constant(3, UType::scalar(DType::I64))],
        );
        let mismatched_end = rebuild(
            input_address,
            range.clone(),
            output_address,
            range,
            different_end,
        );
        let separate_equal_range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            input_index.sources()[1].sources().to_vec(),
        );
        assert_eq!(separate_equal_range, input_index.sources()[1]);
        assert!(!separate_equal_range.shares_node_with(&input_index.sources()[1]));
        let unshared_range = rebuild(
            input_index.sources()[0].clone(),
            input_index.sources()[1].clone(),
            output_index.sources()[0].clone(),
            separate_equal_range.clone(),
            separate_equal_range,
        );
        for malformed in [
            constant_index,
            out_of_bounds,
            wrong_buffer_identity,
            mismatched_end,
            unshared_range,
        ] {
            assert!(
                renderer
                    .render_linked_f32_exp(&malformed, request.link_inputs())
                    .is_err()
            );
        }
        assert!(matches!(end_range.operation(), Operation::EndRange));
    }

    #[test]
    fn linked_exp_block_size_is_part_of_request_and_cache_identity() {
        use std::num::NonZeroUsize;

        let (mock, capture, primary, request, _, _) = fixture();
        let renderer = PtxRenderer::new(80).unwrap();
        let alternate = LinkedF32ExpRequest::new(
            renderer,
            &capture.items[0].kernel,
            request.link_inputs().to_vec(),
            &request.rendered().entry,
            64,
        )
        .unwrap();
        assert_ne!(request.identity(), alternate.identity());
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let first = request.load(&primary, &cache).unwrap();
        let second = alternate.load(&primary, &cache).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(first.block_size(), 32);
        assert_eq!(second.block_size(), 64);
        assert_eq!(cache.len(), 2);
        // The lower linked module/function is independent of launch geometry
        // and is intentionally shared by both block-size-specific wrappers.
        assert_eq!(mock.generic_kernel_count(), 1);

        let bytes = capture.items[0].primary_output().bytes;
        let input = primary.allocate(NonZeroUsize::new(bytes).unwrap()).unwrap();
        let output = primary.allocate(NonZeroUsize::new(bytes).unwrap()).unwrap();
        input.copy_from(0, &vec![0; bytes]).unwrap();
        output.copy_from(0, &vec![0x5a; bytes]).unwrap();
        let stream = primary.stream().unwrap();
        drop(cache);
        drop(first);
        assert_eq!(mock.generic_kernel_count(), 1);
        second
            .launch(
                &stream,
                &[
                    PtxBinding {
                        buffer: input.view(),
                        dtype: DType::F32,
                        mutable: false,
                    },
                    PtxBinding {
                        buffer: output.view(),
                        dtype: DType::F32,
                        mutable: true,
                    },
                ],
                true,
            )
            .unwrap();
        let mut actual = vec![0; bytes];
        output.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, [1.0_f32.to_le_bytes(); 3].concat());
        drop(second);
        assert_eq!(mock.generic_kernel_count(), 0);
    }

    #[test]
    fn linked_exp_prepared_identity_distinguishes_dense_shapes() {
        let (
            scalar_mock,
            scalar_capture,
            scalar_primary,
            scalar_request,
            scalar_sidecar,
            scalar_bound,
        ) = fixture_for(Shape::from([]));
        let (_, vector_capture, vector_primary, vector_request, vector_sidecar, vector_bound) =
            fixture_for(Shape::from([5]));
        let calls = scalar_mock.calls().len();
        let scalar = PreparedLinkedF32ExpCapture::prepare(
            &scalar_capture,
            &scalar_sidecar,
            &scalar_bound,
            &scalar_primary,
            80,
            &scalar_request,
        )
        .unwrap();
        let vector = PreparedLinkedF32ExpCapture::prepare(
            &vector_capture,
            &vector_sidecar,
            &vector_bound,
            &vector_primary,
            80,
            &vector_request,
        )
        .unwrap();
        assert_ne!(scalar.request_identity(), vector.request_identity());
        assert_ne!(scalar.rendered_identity(), vector.rendered_identity());
        assert_eq!(scalar.input().shape, Shape::from([]));
        assert_eq!(vector.input().shape, Shape::from([5]));
        assert_eq!(scalar_mock.calls().len(), calls);
    }

    #[test]
    fn prepared_linked_exp_launcher_executes_scalar_and_dense_vector_shapes() {
        use std::num::NonZeroUsize;
        for (shape, values) in [
            (Shape::from([]), vec![1.0_f32]),
            (Shape::from([5]), vec![-1.0, -0.5, 0.0, 0.5, 1.0]),
        ] {
            let (mock, capture, primary, request, sidecar, resources) = fixture_for(shape);
            let prepared = PreparedLinkedF32ExpCapture::prepare(
                &capture, &sidecar, &resources, &primary, 80, &request,
            )
            .unwrap();
            let table =
                PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
            let bytes = NonZeroUsize::new(prepared.input().bytes).unwrap();
            let input = primary.allocate(bytes).unwrap();
            let target = primary.allocate(bytes).unwrap();
            let source = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            input.copy_from(0, &source).unwrap();
            target.copy_from(0, &vec![0x7f; source.len()]).unwrap();
            let bound = table
                .rebind(&prepared, &primary, 80, &request, &input, &target)
                .unwrap();
            let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
            let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
            let cache = PrimaryLinkedRenderedKernelCache::new(functions);
            let baseline = mock.live_allocation_count(primary.owner());
            execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();
            let mut unchanged = vec![0; source.len()];
            input.copy_to(0, &mut unchanged).unwrap();
            assert_eq!(unchanged, source);
            let mut output = vec![0; source.len()];
            target.copy_to(0, &mut output).unwrap();
            for (got, want) in output
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .zip(values.iter().copied().map(f32::exp))
            {
                assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0));
            }
            assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
        }
    }

    #[test]
    fn prepared_linked_exp_launcher_commits_only_after_candidate_execution() {
        use std::num::NonZeroUsize;

        let (mock, capture, primary, request, sidecar, bound_resources) = fixture();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture,
            &sidecar,
            &bound_resources,
            &primary,
            80,
            &request,
        )
        .unwrap();
        let table =
            PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
        let input = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let target = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let values = [-1.0_f32, 0.0, 1.0];
        input
            .copy_from(
                0,
                &values
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        target.copy_from(0, &[0x5a; 12]).unwrap();
        let bound = table
            .rebind(&prepared, &primary, 80, &request, &input, &target)
            .unwrap();
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let calls = mock.calls().len();
        execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();
        let mut input_bytes = [0; 12];
        input.copy_to(0, &mut input_bytes).unwrap();
        assert_eq!(
            input_bytes,
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>()
                .as_slice()
        );
        let mut actual = [0; 12];
        target.copy_to(0, &mut actual).unwrap();
        for (got, want) in actual
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .zip(values.map(f32::exp))
        {
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0));
        }
        let trace = mock.calls();
        assert!(trace[calls..].contains(&"launch"));
        assert!(trace[calls..].contains(&"dtod_async"));
        let before_bad = mock.calls().len();
        assert!(
            table
                .rebind(&prepared, &primary, 81, &request, &input, &target)
                .is_err()
        );
        assert_eq!(mock.calls().len(), before_bad);
    }

    #[test]
    fn prepared_linked_exp_launcher_recoverable_failures_preserve_target() {
        use std::num::NonZeroUsize;

        let (mock, capture, primary, request, sidecar, bound_resources) = fixture();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture,
            &sidecar,
            &bound_resources,
            &primary,
            80,
            &request,
        )
        .unwrap();
        let table =
            PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
        let input = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let target = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        input.copy_from(0, &[0; 12]).unwrap();
        target.copy_from(0, &[0x6d; 12]).unwrap();
        let bound = table
            .rebind(&prepared, &primary, 80, &request, &input, &target)
            .unwrap();
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let baseline = mock.live_allocation_count(primary.owner());
        let snapshot = [0x6d; 12];

        mock.set_allocation_failure(true);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        mock.set_allocation_failure(false);
        let mut actual = [0; 12];
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);

        mock.fail_generic_kernel_launch_after(0, 2);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);

        mock.fail_generic_kernel_sync_after(0, 3);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);

        // One backup submission then the final candidate-to-target submission
        // fails. This one-shot failure permits the best-effort restoration to
        // complete before the candidate and backup allocations are released.
        mock.fail_dtod_after(1, 91);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
        execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
    }

    #[test]
    fn prepared_linked_exp_launcher_evicts_failed_function_loads() {
        use std::num::NonZeroUsize;
        let (mock, capture, primary, request, sidecar, bound_resources) = fixture();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture,
            &sidecar,
            &bound_resources,
            &primary,
            80,
            &request,
        )
        .unwrap();
        let table =
            PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
        let input = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let target = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        input.copy_from(0, &[0; 12]).unwrap();
        target.copy_from(0, &[0x42; 12]).unwrap();
        let bound = table
            .rebind(&prepared, &primary, 80, &request, &input, &target)
            .unwrap();
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let baseline = mock.live_allocation_count(primary.owner());
        mock.set_function_result(73);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        mock.set_function_result(0);
        let mut target_bytes = [0; 12];
        target.copy_to(0, &mut target_bytes).unwrap();
        assert_eq!(target_bytes, [0x42; 12]);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
        execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
    }

    #[test]
    fn prepared_linked_exp_launcher_reports_best_effort_rollback_boundaries() {
        use crate::cuda::PrimaryOutputCommitPhase;
        use std::num::NonZeroUsize;
        let (mock, capture, primary, request, sidecar, resources) = fixture();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture, &sidecar, &resources, &primary, 80, &request,
        )
        .unwrap();
        let table =
            PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
        let input = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let target = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        input.copy_from(0, &[0; 12]).unwrap();
        target.copy_from(0, &[0x31; 12]).unwrap();
        let bound = table
            .rebind(&prepared, &primary, 80, &request, &input, &target)
            .unwrap();
        let modules = Arc::new(crate::cuda::PrimaryLinkedModuleCache::new());
        let functions = Arc::new(crate::cuda::PrimaryLinkedKernelCache::new(modules));
        let cache = PrimaryLinkedRenderedKernelCache::new(functions);
        let baseline = mock.live_allocation_count(primary.owner());
        let mut snapshot = [0; 12];
        target.copy_to(0, &mut snapshot).unwrap();

        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Backup, 0, 61);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        let mut actual = [0; 12];
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
        execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();

        target.copy_from(0, &snapshot).unwrap();
        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Commit, 0, 62);
        assert!(execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).is_err());
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(actual, snapshot);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);

        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Commit, 0, 63);
        mock.fail_output_commit_phase_after(PrimaryOutputCommitPhase::Restore, 0, 64);
        let error =
            execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap_err();
        assert!(error.to_string().contains("rollback"));
        assert!(error.to_string().contains("partially modified"));
        target.copy_to(0, &mut actual).unwrap();
        assert_eq!(
            actual,
            [
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f,
            ]
        );
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
        target.copy_from(0, &snapshot).unwrap();
        execute_prepared_linked_f32_exp(&bound, &primary, 80, &request, &cache).unwrap();
    }

    #[test]
    fn prepared_linked_exp_rebind_rejects_malformed_external_leases_without_driver_work() {
        use std::num::NonZeroUsize;
        let (mock, capture, primary, request, sidecar, resources) = fixture();
        let prepared = PreparedLinkedF32ExpCapture::prepare(
            &capture, &sidecar, &resources, &primary, 80, &request,
        )
        .unwrap();
        let table =
            PreparedLinkedF32ExpBindingTable::from_prepared(&prepared, &primary, 80).unwrap();
        let input = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let target = primary.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let small = primary.allocate(NonZeroUsize::new(4).unwrap()).unwrap();
        let other = Driver::from_dispatch(mock.clone())
            .unwrap()
            .device(crate::DeviceId(0))
            .unwrap()
            .retain_primary_context()
            .unwrap();
        let foreign = other.allocate(NonZeroUsize::new(12).unwrap()).unwrap();
        let calls = mock.calls().len();
        let baseline = mock.live_allocation_count(primary.owner());
        for result in [
            table.rebind(&prepared, &primary, 81, &request, &input, &target),
            table.rebind(&prepared, &primary, 80, &request, &small, &target),
            table.rebind(&prepared, &primary, 80, &request, &input, &input),
            table.rebind(&prepared, &primary, 80, &request, &foreign, &target),
        ] {
            assert!(result.is_err());
        }
        assert_eq!(mock.calls().len(), calls);
        assert_eq!(mock.live_allocation_count(primary.owner()), baseline);
    }
}
