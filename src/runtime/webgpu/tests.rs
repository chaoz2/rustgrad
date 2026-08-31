use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::runtime::scalar_lane::emit_scalar_lane;
use crate::{
    AddressSpace, AddressValue, Backend, BinaryOp, BufferRole, CapturedMixedBatch,
    CapturedReplayExecutor, CompareOp, CpuBackend, DType, EffectBatchStep, EffectRuntime, Graph,
    IndexValue, KernelBindings, KernelBufferDesc, LaneInstruction, LiteralValue, NodeId, Operation,
    ReduceKind, Scalar, Shape, Slice, Storage, TensorData, TypedValue, UOp, UType, ViewMap,
    schedule,
};
use dispatch::{
    CopyRegion, Dispatch, KernelSemantics, LaunchGeometry, RawAdapter, RawBuffer, RawCommand,
    RawDevice, RawInstance, RawPipeline, RawQueue, RawShader,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    rc::Rc,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct Failures {
    instance: Option<&'static str>,
    adapters: Option<&'static str>,
    device: Option<&'static str>,
    queue: Option<&'static str>,
    buffer: Option<&'static str>,
    buffer_after: Option<(usize, &'static str)>,
    write: Option<&'static str>,
    read: Option<&'static str>,
    read_after: Option<(usize, &'static str)>,
    copy: Option<&'static str>,
    build: Option<String>,
    pipeline: Option<&'static str>,
    launch: Option<&'static str>,
    query: Option<&'static str>,
    wait: Option<&'static str>,
}

#[derive(Default)]
struct State {
    calls: Vec<String>,
    owners: BTreeSet<u64>,
    next_buffer: usize,
    next_shader: usize,
    next_pipeline: usize,
    next_command: usize,
    buffers: BTreeMap<(u64, usize), Vec<u8>>,
    shaders: BTreeMap<(u64, usize), String>,
    semantics: BTreeMap<(u64, usize), Arc<KernelSemantics>>,
    commands: BTreeMap<(u64, usize), bool>,
    fault_order: Vec<usize>,
    failures: Failures,
}

#[derive(Default)]
struct MockDispatch {
    state: Mutex<State>,
}

#[test]
fn mixed_batch_webgpu_mock_is_prepared_atomic_and_retryable() {
    let (first, first_next) = crate::engine::mixed_batch::test_support::pure_add_capture(700);
    let (second, second_next) = crate::engine::mixed_batch::test_support::pure_add_capture(700);
    let batch = CapturedMixedBatch::new(vec![first.clone(), second]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = WgslRenderer::new(8, capabilities()).unwrap();
    let inputs = vec![
        BTreeMap::from([
            (
                "x".into(),
                crate::engine::mixed_batch::test_support::data(vec![1., 2.]),
            ),
            (
                "y".into(),
                crate::engine::mixed_batch::test_support::data(vec![3., 4.]),
            ),
        ]),
        BTreeMap::from([
            (
                "x".into(),
                crate::engine::mixed_batch::test_support::data(vec![5., 6.]),
            ),
            (
                "y".into(),
                crate::engine::mixed_batch::test_support::data(vec![7., 8.]),
            ),
        ]),
    ];
    let mut runtime = EffectRuntime::new();
    runtime
        .register(
            700,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    runtime
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    assert!(
        batch
            .replay_webgpu(
                &mut runtime,
                &inputs,
                device.clone(),
                renderer.clone(),
                Some(EffectBatchStep { entry: 1, step: 0 }),
            )
            .is_err()
    );
    assert_eq!(
        runtime
            .snapshot(&crate::BufferState {
                version: 0,
                ..first_next.clone()
            })
            .unwrap()
            .tensor()
            .storage(),
        &Storage::F32(vec![0., 0.])
    );
    let result = batch
        .replay_webgpu(&mut runtime, &inputs, device, renderer, None)
        .unwrap();
    assert_eq!(result.trace.identity, batch.identity());
    assert_eq!(
        runtime
            .snapshot(&crate::BufferState {
                version: 2,
                ..second_next
            })
            .unwrap()
            .tensor()
            .storage(),
        &Storage::F32(vec![12., 14.])
    );
    assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn mixed_batch_webgpu_signed_state_input_matches_interpreter_and_native() {
    let (capture, end) = crate::engine::mixed_batch::test_support::signed_state_add_capture();
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let supplied = BTreeMap::from([(
        "bias".into(),
        crate::engine::mixed_batch::test_support::data(vec![10., 20., 30., 40.]),
    )]);
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let mut webgpu = EffectRuntime::new();
    webgpu
        .register(
            90,
            crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
        )
        .unwrap();
    webgpu
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0.; 4]),
        )
        .unwrap();
    batch
        .replay_webgpu(
            &mut webgpu,
            std::slice::from_ref(&supplied),
            device,
            WgslRenderer::new(8, capabilities()).unwrap(),
            None,
        )
        .unwrap();
    let mut interpreter = EffectRuntime::new();
    interpreter
        .register(
            90,
            crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
        )
        .unwrap();
    interpreter
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0.; 4]),
        )
        .unwrap();
    batch
        .replay(&mut interpreter, std::slice::from_ref(&supplied), None)
        .unwrap();
    let mut native = EffectRuntime::new();
    native
        .register(
            90,
            crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
        )
        .unwrap();
    native
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0.; 4]),
        )
        .unwrap();
    batch
        .replay_native(
            &mut native,
            &[supplied],
            &CapturedReplayExecutor::default(),
            false,
            None,
        )
        .unwrap();
    let expected = &Storage::F32(vec![14., 23., 32., 41.]);
    assert_eq!(webgpu.snapshot(&end).unwrap().tensor().storage(), expected);
    assert_eq!(
        webgpu.snapshot(&end).unwrap().tensor().storage(),
        interpreter.snapshot(&end).unwrap().tensor().storage()
    );
    assert_eq!(
        webgpu.snapshot(&end).unwrap().tensor().storage(),
        native.snapshot(&end).unwrap().tensor().storage()
    );
    assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn mixed_batch_webgpu_rejects_later_unsupported_before_submission() {
    let (first, first_end) = crate::engine::mixed_batch::test_support::pure_add_capture(91);
    let (mut later, _) = crate::engine::mixed_batch::test_support::pure_add_capture(92);
    crate::engine::mixed_batch::test_support::mark_first_prefix_unsupported(&mut later, "test");
    let batch = CapturedMixedBatch::new(vec![first, later]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let mut runtime = EffectRuntime::new();
    runtime
        .register(
            91,
            crate::engine::mixed_batch::test_support::data(vec![9., 9.]),
        )
        .unwrap();
    runtime
        .register(
            92,
            crate::engine::mixed_batch::test_support::data(vec![8., 8.]),
        )
        .unwrap();
    runtime
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    assert!(
        batch
            .replay_webgpu(
                &mut runtime,
                &[
                    crate::engine::mixed_batch::test_support::add_inputs(),
                    crate::engine::mixed_batch::test_support::add_inputs(),
                ],
                device,
                WgslRenderer::new(8, capabilities()).unwrap(),
                None,
            )
            .is_err()
    );
    assert!(mock.calls().iter().all(|call| !call.starts_with("launch:")));
    assert_eq!(
        runtime
            .snapshot(&crate::BufferState {
                version: 0,
                ..first_end
            })
            .unwrap()
            .tensor()
            .storage(),
        &Storage::F32(vec![9., 9.])
    );
}

#[test]
fn mixed_batch_webgpu_empty_prefix_skips_submission_and_commits() {
    let (capture, end) = crate::engine::mixed_batch::test_support::zero_extent_add_capture();
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let mut runtime = EffectRuntime::new();
    runtime
        .register(93, crate::engine::mixed_batch::test_support::data(vec![]))
        .unwrap();
    runtime
        .register(2, crate::engine::mixed_batch::test_support::data(vec![]))
        .unwrap();
    batch
        .replay_webgpu(
            &mut runtime,
            &[BTreeMap::from([
                (
                    "x".into(),
                    crate::engine::mixed_batch::test_support::data(vec![]),
                ),
                (
                    "y".into(),
                    crate::engine::mixed_batch::test_support::data(vec![]),
                ),
            ])],
            device,
            WgslRenderer::new(8, capabilities()).unwrap(),
            None,
        )
        .unwrap();
    assert!(mock.calls().iter().all(|call| !call.starts_with("launch:")));
    assert_eq!(
        runtime.snapshot(&end).unwrap().tensor().storage(),
        &Storage::F32(vec![])
    );
}

#[test]
fn mixed_batch_webgpu_reuses_prepared_keys_for_equivalent_replays() {
    let (capture, next) = crate::engine::mixed_batch::test_support::pure_add_capture(94);
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = WgslRenderer::new(8, capabilities()).unwrap();
    let inputs = vec![crate::engine::mixed_batch::test_support::add_inputs()];
    let mut first = EffectRuntime::new();
    first
        .register(
            94,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    first
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    let first_result = batch
        .replay_webgpu(&mut first, &inputs, device.clone(), renderer.clone(), None)
        .unwrap();
    let compiled = mock
        .calls()
        .iter()
        .filter(|call| call.starts_with("pipeline_create:"))
        .count();
    assert_eq!(compiled, 1);
    let mut second = EffectRuntime::new();
    second
        .register(
            94,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    second
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    let second_result = batch
        .replay_webgpu(&mut second, &inputs, device, renderer, None)
        .unwrap();
    assert_eq!(first_result.trace, second_result.trace);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("pipeline_create:"))
            .count(),
        compiled
    );
    assert_eq!(
        second.snapshot(&next).unwrap().tensor().storage(),
        &Storage::F32(vec![4., 6.])
    );
}

#[test]
fn mixed_batch_webgpu_launch_failure_preserves_state_and_reuses_preparation() {
    let (capture, next) = crate::engine::mixed_batch::test_support::pure_add_capture(95);
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = WgslRenderer::new(8, capabilities()).unwrap();
    let mut runtime = EffectRuntime::new();
    runtime
        .register(
            95,
            crate::engine::mixed_batch::test_support::data(vec![9., 9.]),
        )
        .unwrap();
    runtime
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0., 0.]),
        )
        .unwrap();
    mock.state.lock().unwrap().failures.launch = Some("mixed batch launch");
    assert!(
        batch
            .replay_webgpu(
                &mut runtime,
                &[crate::engine::mixed_batch::test_support::add_inputs()],
                device.clone(),
                renderer.clone(),
                None
            )
            .is_err()
    );
    assert_eq!(
        runtime
            .snapshot(&crate::BufferState {
                version: 0,
                ..next.clone()
            })
            .unwrap()
            .tensor()
            .storage(),
        &Storage::F32(vec![9., 9.])
    );
    let compiled = mock
        .calls()
        .iter()
        .filter(|call| call.starts_with("pipeline_create:"))
        .count();
    mock.clear_failures();
    let result = batch
        .replay_webgpu(
            &mut runtime,
            &[crate::engine::mixed_batch::test_support::add_inputs()],
            device,
            renderer,
            None,
        )
        .unwrap();
    assert_eq!(result.trace.identity, batch.identity());
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("pipeline_create:"))
            .count(),
        compiled
    );
    assert_eq!(
        runtime.snapshot(&next).unwrap().tensor().storage(),
        &Storage::F32(vec![4., 6.])
    );
}

impl MockDispatch {
    fn calls(&self) -> Vec<String> {
        self.state.lock().unwrap().calls.clone()
    }
    fn clear_failures(&self) {
        self.state.lock().unwrap().failures = Failures::default();
    }

    fn command(state: &mut State, owner: u64) -> RawCommand {
        state.next_command += 1;
        let raw = RawCommand(700 + state.next_command);
        state.commands.insert((owner, raw.0), false);
        raw
    }

    fn failure(operation: &'static str, detail: &'static str) -> WebGpuError {
        WebGpuError::Driver {
            operation,
            detail: detail.into(),
        }
    }
}

impl Dispatch for MockDispatch {
    fn instance_create(&self) -> Result<RawInstance, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.instance.take() {
            return Err(Self::failure("instance_create", detail));
        }
        state.calls.push("instance_create".into());
        Ok(RawInstance(1))
    }

    fn instance_release(&self, instance: RawInstance) {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("instance_release:{}", instance.0));
    }

    fn adapters(&self, _instance: RawInstance) -> Result<Vec<RawAdapter>, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.adapters.take() {
            return Err(Self::failure("adapters", detail));
        }
        state.calls.push("adapters".into());
        Ok(vec![RawAdapter(2), RawAdapter(1)])
    }

    fn adapter_info(&self, adapter: RawAdapter) -> Result<WebGpuAdapterInfo, WebGpuError> {
        Ok(WebGpuAdapterInfo {
            name: format!("Mock WebGPU {}", adapter.0),
            backend: if adapter.0 == 1 {
                WebGpuBackend::Metal
            } else {
                WebGpuBackend::Vulkan
            },
            vendor: adapter.0 as u32,
            device: 100 + adapter.0 as u32,
            driver: format!("mock-{}", adapter.0),
            capabilities: capabilities(),
        })
    }

    fn adapter_release(&self, adapter: RawAdapter) {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("adapter_release:{}", adapter.0));
    }

    fn device_create(&self, _adapter: RawAdapter, owner: u64) -> Result<RawDevice, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.device.take() {
            return Err(Self::failure("device_create", detail));
        }
        state.owners.insert(owner);
        state.calls.push(format!("device_create:{owner}"));
        Ok(RawDevice(10))
    }

    fn device_release(&self, _device: RawDevice, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.owners.remove(&owner);
        state.calls.push(format!("device_release:{owner}"));
    }

    fn queue_create(&self, _device: RawDevice, owner: u64) -> Result<RawQueue, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.queue.take() {
            return Err(Self::failure("queue_create", detail));
        }
        state.calls.push(format!("queue_create:{owner}"));
        Ok(RawQueue(20))
    }

    fn queue_release(&self, _queue: RawQueue, owner: u64) {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("queue_release:{owner}"));
    }

    fn buffer_create(
        &self,
        _device: RawDevice,
        physical_bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.buffer.take() {
            return Err(Self::failure("buffer_create", detail));
        }
        if let Some((remaining, detail)) = state.failures.buffer_after {
            if remaining == 0 {
                state.failures.buffer_after = None;
                return Err(Self::failure("buffer_create", detail));
            }
            state.failures.buffer_after = Some((remaining - 1, detail));
        }
        state.next_buffer += 1;
        let raw = RawBuffer(100 + state.next_buffer);
        state
            .buffers
            .insert((owner, raw.0), vec![0; physical_bytes]);
        state
            .calls
            .push(format!("buffer_create:{owner}:{}:{physical_bytes}", raw.0));
        Ok(raw)
    }

    fn buffer_release(&self, buffer: RawBuffer, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.buffers.remove(&(owner, buffer.0));
        state
            .calls
            .push(format!("buffer_release:{owner}:{}", buffer.0));
    }

    fn buffer_write(
        &self,
        _queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.write.take() {
            return Err(Self::failure("write", detail));
        }
        let storage = state
            .buffers
            .get_mut(&(owner, buffer.0))
            .ok_or(WebGpuError::OwnerMismatch)?;
        storage[offset..offset + bytes.len()].copy_from_slice(bytes);
        state.calls.push(format!("write:{owner}:{}", buffer.0));
        Ok(())
    }

    fn buffer_read(
        &self,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        owner: u64,
    ) -> Result<(), WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.read.take() {
            return Err(Self::failure("read", detail));
        }
        if let Some((remaining, detail)) = state.failures.read_after {
            if remaining == 0 {
                state.failures.read_after = None;
                return Err(Self::failure("read", detail));
            }
            state.failures.read_after = Some((remaining - 1, detail));
        }
        let storage = state
            .buffers
            .get(&(owner, buffer.0))
            .ok_or(WebGpuError::OwnerMismatch)?;
        bytes.copy_from_slice(&storage[offset..offset + bytes.len()]);
        state.calls.push(format!("read:{owner}:{}", buffer.0));
        Ok(())
    }

    fn buffer_copy(
        &self,
        _queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: CopyRegion,
        owner: u64,
    ) -> Result<RawCommand, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.copy.take() {
            return Err(Self::failure("copy", detail));
        }
        let value = state
            .buffers
            .get(&(owner, src.0))
            .ok_or(WebGpuError::OwnerMismatch)?
            [region.src_offset..region.src_offset + region.bytes]
            .to_vec();
        state
            .buffers
            .get_mut(&(owner, dst.0))
            .ok_or(WebGpuError::OwnerMismatch)?
            [region.dst_offset..region.dst_offset + region.bytes]
            .copy_from_slice(&value);
        state.calls.push(format!("copy:{owner}"));
        Ok(Self::command(&mut state, owner))
    }

    fn shader_create(
        &self,
        _device: RawDevice,
        source: &str,
        owner: u64,
    ) -> Result<RawShader, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(diagnostic) = state.failures.build.take() {
            return Err(WebGpuError::Build { diagnostic });
        }
        state.next_shader += 1;
        let raw = RawShader(300 + state.next_shader);
        state.shaders.insert((owner, raw.0), source.into());
        state.calls.push(format!("shader_create:{owner}"));
        Ok(raw)
    }

    fn shader_release(&self, shader: RawShader, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.shaders.remove(&(owner, shader.0));
        state.calls.push(format!("shader_release:{owner}"));
    }

    fn pipeline_create(
        &self,
        _device: RawDevice,
        _shader: RawShader,
        _entry: &str,
        storage_bindings: usize,
        owner: u64,
    ) -> Result<RawPipeline, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.pipeline.take() {
            return Err(Self::failure("pipeline_create", detail));
        }
        if storage_bindings == 0 {
            return Err(WebGpuError::InvalidBinding(
                "mock pipeline has no buffers".into(),
            ));
        }
        state.next_pipeline += 1;
        let raw = RawPipeline(500 + state.next_pipeline);
        state
            .calls
            .push(format!("pipeline_create:{owner}:{storage_bindings}"));
        Ok(raw)
    }

    fn pipeline_release(&self, pipeline: RawPipeline, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.semantics.remove(&(owner, pipeline.0));
        state.calls.push(format!("pipeline_release:{owner}"));
    }

    fn launch(
        &self,
        _queue: RawQueue,
        pipeline: RawPipeline,
        buffers: &[RawBuffer],
        geometry: LaunchGeometry,
        owner: u64,
    ) -> Result<RawCommand, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.launch.take() {
            return Err(Self::failure("launch", detail));
        }
        let semantics = state
            .semantics
            .get(&(owner, pipeline.0))
            .cloned()
            .ok_or_else(|| WebGpuError::InvalidBinding("mock semantics absent".into()))?;
        let transaction = semantics.transaction.as_ref();
        let expected_buffers = semantics.buffers.len() + usize::from(transaction.is_some());
        if buffers.len() != expected_buffers
            || geometry.extent as usize != semantics.extent
            || geometry.local == 0
            || geometry.workgroups != geometry.extent.div_ceil(geometry.local)
            || geometry.extent_binding != semantics.buffers.len()
            || geometry.status_binding
                != transaction.map(|_| semantics.buffers.len().saturating_add(1))
        {
            return Err(WebGpuError::InvalidArgument("invalid mock launch geometry"));
        }
        let mut bindings = KernelBindings::default();
        let mut output = None;
        for (position, (raw, abi)) in buffers.iter().zip(&semantics.buffers).enumerate() {
            let logical = abi.logical_bytes()?;
            let bytes = state
                .buffers
                .get(&(owner, raw.0))
                .ok_or(WebGpuError::OwnerMismatch)?;
            if bytes.len() != logical.div_ceil(4) * 4 {
                return Err(WebGpuError::InvalidBinding(format!(
                    "mock buffer {position} length mismatch"
                )));
            }
            let value =
                TensorData::from_le_bytes(abi.source_shape.clone(), abi.dtype, &bytes[..logical])
                    .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
            let role = if abi.mutable {
                BufferRole::Output
            } else {
                BufferRole::Input
            };
            let desc = KernelBufferDesc::concrete(
                abi.id,
                role,
                abi.source_shape.clone(),
                abi.dtype,
                abi.mutable,
            )
            .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
            bindings
                .insert(&desc, value)
                .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
            if abi.mutable {
                output = Some((*raw, logical));
            }
        }
        if let Some(transaction) = transaction {
            let stored = semantics
                .buffers
                .iter()
                .enumerate()
                .map(|(position, abi)| {
                    let bytes = state
                        .buffers
                        .get(&(owner, buffers[position].0))
                        .ok_or(WebGpuError::OwnerMismatch)?;
                    Ok((abi.clone(), bytes.clone()))
                })
                .collect::<Result<Vec<_>, WebGpuError>>()?;
            let order = if state.fault_order.is_empty() {
                (0..semantics.extent).collect::<Vec<_>>()
            } else {
                state.fault_order.clone()
            };
            if order.len() != semantics.extent
                || order.iter().copied().collect::<BTreeSet<_>>().len() != semantics.extent
                || order.iter().any(|&index| index >= semantics.extent)
            {
                return Err(WebGpuError::InvalidBinding(
                    "mock fault order is not an extent permutation".into(),
                ));
            }
            let mut status = transaction::CLEAN_STATUS;
            for logical in order {
                if let Some(id) =
                    transaction::first_fault_at(transaction, logical, |arg, dtype, logical| {
                        let buffer_id = match arg {
                            IndexValue::Buffer { buffer, .. } | IndexValue::View { buffer, .. } => {
                                *buffer
                            }
                        };
                        let (abi, bytes) = stored
                            .iter()
                            .find(|(abi, _)| abi.id == buffer_id)
                            .ok_or_else(|| {
                                WebGpuError::InvalidBinding("mock transaction buffer absent".into())
                            })?;
                        if abi.dtype != dtype {
                            return Err(WebGpuError::InvalidBinding(
                                "mock transaction dtype mismatch".into(),
                            ));
                        }
                        let offset = transaction::logical_offset(arg, logical)?;
                        let start = offset
                            .checked_mul(dtype.itemsize())
                            .ok_or(WebGpuError::Overflow)?;
                        decode_mock_scalar(dtype, &bytes[start..start + dtype.itemsize()])
                    })?
                {
                    status = status.min(transaction.key(logical, id)?);
                }
            }
            let status_raw = buffers
                .last()
                .ok_or_else(|| WebGpuError::InvalidBinding("mock status absent".into()))?;
            state
                .buffers
                .get_mut(&(owner, status_raw.0))
                .ok_or(WebGpuError::OwnerMismatch)?[..4]
                .copy_from_slice(&status.to_le_bytes());
            if status != transaction::CLEAN_STATUS {
                state.calls.push(format!(
                    "launch:{owner}:{}:{}",
                    geometry.workgroups, geometry.local
                ));
                return Ok(Self::command(&mut state, owner));
            }
        }
        // Independent typed lowered-UOp execution: this is not `CpuBackend`.
        let result = match semantics.program.as_ref() {
            dispatch::KernelSemanticProgram::UOp(program) => {
                execute_webgpu_semantics(program, &bindings)
            }
            dispatch::KernelSemanticProgram::Random(plan) => plan.execute(),
        }
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?
        .to_le_bytes()
        .map_err(|error| WebGpuError::InvalidBinding(error.to_string()))?;
        let (output, logical) =
            output.ok_or_else(|| WebGpuError::InvalidBinding("mock output absent".into()))?;
        if result.len() != logical {
            return Err(WebGpuError::InvalidBinding(
                "mock output length mismatch".into(),
            ));
        }
        state
            .buffers
            .get_mut(&(owner, output.0))
            .ok_or(WebGpuError::OwnerMismatch)?[..logical]
            .copy_from_slice(&result);
        state.calls.push(format!(
            "launch:{owner}:{}:{}",
            geometry.workgroups, geometry.local
        ));
        Ok(Self::command(&mut state, owner))
    }

    fn command_query(&self, command: RawCommand, owner: u64) -> Result<bool, WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.query.take() {
            return Err(Self::failure("query", detail));
        }
        state
            .commands
            .get(&(owner, command.0))
            .copied()
            .ok_or(WebGpuError::OwnerMismatch)
    }

    fn command_wait(&self, command: RawCommand, owner: u64) -> Result<(), WebGpuError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.wait.take() {
            return Err(Self::failure("wait", detail));
        }
        *state
            .commands
            .get_mut(&(owner, command.0))
            .ok_or(WebGpuError::OwnerMismatch)? = true;
        state.calls.push(format!("wait:{owner}"));
        Ok(())
    }

    fn command_release(&self, command: RawCommand, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.commands.remove(&(owner, command.0));
        state.calls.push(format!("command_release:{owner}"));
    }

    fn register_kernel_semantics(
        &self,
        owner: u64,
        pipeline: RawPipeline,
        semantics: Arc<KernelSemantics>,
    ) -> Result<(), WebGpuError> {
        self.state
            .lock()
            .unwrap()
            .semantics
            .insert((owner, pipeline.0), semantics);
        Ok(())
    }

    fn unregister_kernel_semantics(&self, owner: u64, pipeline: RawPipeline) {
        self.state
            .lock()
            .unwrap()
            .semantics
            .remove(&(owner, pipeline.0));
    }
}

fn capabilities() -> WebGpuCapabilities {
    WebGpuCapabilities {
        max_buffer_size: 1 << 20,
        max_storage_buffers_per_shader_stage: 8,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroups_per_dimension: 65_535,
        timestamp_query: true,
        shader_f16: false,
    }
}

fn setup(mock: Arc<MockDispatch>) -> (WebGpuDevice, WebGpuQueue) {
    let runtime = WebGpuRuntime::from_dispatch(mock);
    let instance = runtime.create_instance().unwrap();
    let mut adapters = instance.adapters().unwrap();
    assert_eq!(adapters[0].info().backend, WebGpuBackend::Vulkan);
    let device = adapters.remove(0).request_device().unwrap();
    let queue = device.create_queue().unwrap();
    (device, queue)
}

fn materialized_values(
    graph: &Graph,
    rendered: &RenderedWgsl,
    inputs: &HashMap<String, TensorData>,
) -> BTreeMap<u64, TensorData> {
    rendered
        .buffers
        .iter()
        .filter(|abi| !abi.mutable)
        .map(|abi| {
            let node = NodeId::from_index(abi.id as usize);
            (abi.id, CpuBackend.execute(graph, node, inputs).unwrap())
        })
        .collect()
}

fn allocate_rendered(
    device: &WebGpuDevice,
    queue: &WebGpuQueue,
    rendered: &RenderedWgsl,
    values: &BTreeMap<u64, TensorData>,
) -> Vec<WebGpuBuffer> {
    rendered
        .buffers
        .iter()
        .map(|abi| {
            let buffer = device.allocate_typed(abi.elements, abi.dtype).unwrap();
            if let Some(value) = values.get(&abi.id) {
                queue
                    .write(&buffer, 0, &value.to_le_bytes().unwrap())
                    .unwrap();
            }
            buffer
        })
        .collect()
}

fn execute_mock(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> (TensorData, Arc<MockDispatch>) {
    let item = schedule(graph, output).unwrap().items.pop().unwrap();
    let rendered = WgslRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let values = materialized_values(graph, &rendered, inputs);
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let buffers = allocate_rendered(&device, &queue, &rendered, &values);
    let cache = device.cache();
    let pipeline = cache.load(&rendered).unwrap();
    assert!(Rc::ptr_eq(&pipeline, &cache.load(&rendered).unwrap()));
    assert_eq!(cache.len(), 1);
    let refs = buffers.iter().collect::<Vec<_>>();
    if rendered.transaction.is_some() {
        let transaction = pipeline.launch_transactional(&queue, &refs).unwrap();
        assert_eq!(transaction.query().unwrap(), rendered.extent == 0);
        let completion = transaction.collect().unwrap();
        assert_eq!(completion.extent, rendered.extent);
    } else if let Some(command) = pipeline.launch(&queue, &refs).unwrap() {
        assert!(!command.query().unwrap());
        let completion = command.collect().unwrap();
        assert_eq!(completion.extent, rendered.extent);
        assert_eq!(completion.retained_resources, rendered.buffers.len());
    }
    let output_abi = rendered.buffers.last().unwrap();
    let mut bytes = vec![0; output_abi.logical_bytes().unwrap()];
    queue.read(buffers.last().unwrap(), 0, &mut bytes).unwrap();
    let result =
        TensorData::from_le_bytes(output_abi.source_shape.clone(), output_abi.dtype, &bytes)
            .unwrap();
    (result, mock)
}

#[test]
fn signed_affine_flip_lowers_and_mock_matches_cpu_without_adapter_submission() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3], DType::F32);
    let flipped = graph
        .stride(
            input,
            vec![
                Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
            ],
        )
        .unwrap();
    let output = graph.add(flipped, flipped).unwrap();
    let tensor =
        TensorData::from_scalars([2, 3], DType::F32, [1., 2., 3., 4., 5., 6.].map(Scalar::F))
            .unwrap();
    let rendered = WgslRenderer::new(8, capabilities())
        .unwrap()
        .render(&crate::kernel::lower_graph_elementwise(&graph, output).unwrap())
        .unwrap();
    assert!(rendered.source.contains("* -1i"), "{}", rendered.source);
    let (actual, calls) = execute_mock(
        &graph,
        output,
        &HashMap::from([("x".into(), tensor.clone())]),
    );
    let expected = CpuBackend
        .execute(&graph, output, &HashMap::from([("x".into(), tensor)]))
        .unwrap();
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );
    assert!(calls.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn captured_random_plans_render_and_mock_execute_without_stream_state() {
    let renderer = WgslRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let uniform = graph.uniform([5], -1.25, 2.5, DType::F32, 1337).unwrap();
    let normal = graph.randn([3], DType::F32, 1338).unwrap();
    let signed = graph.randint([5], -7, 19, DType::I32, 1339).unwrap();
    let unsigned = graph.randint([5], 3, 19, DType::U32, 1340).unwrap();
    for output in [uniform, normal, signed, unsigned] {
        let root = crate::kernel::lower_graph_random(&graph, output).unwrap();
        let rendered = renderer.render(&root).unwrap();
        let Operation::Random(plan) = root.operation() else {
            panic!("missing random plan")
        };
        let expected = plan.execute().unwrap().to_le_bytes().unwrap();
        let mock = Arc::new(MockDispatch::default());
        let (device, queue) = setup(mock.clone());
        let buffer = device.allocate_typed(rendered.extent, plan.dtype).unwrap();
        let cache = device.cache();
        let pipeline = cache.load(&rendered).unwrap();
        assert!(Rc::ptr_eq(&pipeline, &cache.load(&rendered).unwrap()));
        pipeline
            .launch(&queue, &[&buffer])
            .unwrap()
            .unwrap()
            .collect()
            .unwrap();
        let mut actual = vec![0; expected.len()];
        queue.read(&buffer, 0, &mut actual).unwrap();
        assert_eq!(actual, expected, "{:?}", plan.kind);
        assert!(rendered.source.contains("captured-threefry"));
        assert!(rendered.source.contains("let chunk=i/maxw"));
        assert!(rendered.source_map.contains_key(&plan.output.index()));
        assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
    }
}

#[test]
fn captured_webgpu_random_rejects_packed_and_wide_storage_before_submission() {
    let renderer = WgslRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let narrow = graph.rand([3], DType::F16, 3).unwrap();
    let wide = graph.randint([3], -3, 5, DType::I64, 4).unwrap();
    let empty = graph.rand([0], DType::F32, 5).unwrap();
    assert!(matches!(
        renderer.render(&crate::kernel::lower_graph_random(&graph, narrow).unwrap()),
        Err(WebGpuError::Unsupported(_))
    ));
    assert!(matches!(
        renderer.render(&crate::kernel::lower_graph_random(&graph, wide).unwrap()),
        Err(WebGpuError::Unsupported(_))
    ));
    let rendered = renderer
        .render(&crate::kernel::lower_graph_random(&graph, empty).unwrap())
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let output = device.allocate_typed(0, DType::F32).unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(pipeline.launch(&queue, &[&output]).unwrap().is_none());
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch:")));
}

fn ints(values: &[i32]) -> TensorData {
    TensorData::from_scalars(
        [values.len()],
        DType::I32,
        values.iter().map(|&v| Scalar::I(v as i64)),
    )
    .unwrap()
}

fn uints(values: &[u32]) -> TensorData {
    TensorData::from_scalars(
        [values.len()],
        DType::U32,
        values.iter().map(|&v| Scalar::U(v as u64)),
    )
    .unwrap()
}

fn narrow_data(dtype: DType, shape: impl Into<Shape>, bits: &[u16]) -> TensorData {
    let bytes = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    TensorData::from_le_bytes(shape, dtype, &bytes).unwrap()
}

fn execute_webgpu_semantics(program: &UOp, bindings: &KernelBindings) -> crate::Result<TensorData> {
    let uses_narrow = program
        .topological()
        .map_err(|error| crate::Error::Serialization {
            reason: error.to_string(),
        })?
        .iter()
        .any(|node| {
            node.ty()
                .is_some_and(|ty| matches!(ty.scalar, DType::F16 | DType::BF16))
        });
    if !uses_narrow {
        return execute_lowered_elementwise(program, bindings);
    }
    let store = program
        .sources()
        .iter()
        .find(|node| matches!(node.operation(), Operation::Store))
        .ok_or(crate::Error::InvalidIndex)?;
    let index = store.sources().first().ok_or(crate::Error::InvalidIndex)?;
    let Operation::Index(IndexValue::Buffer { output_shape, .. }) = index.operation() else {
        return Err(crate::Error::InvalidIndex);
    };
    let dtype = index.ty().ok_or(crate::Error::InvalidIndex)?.scalar;
    if dtype == DType::BF16
        && store.sources()[1].operation() == &Operation::Cast
        && store.sources()[1]
            .sources()
            .first()
            .and_then(UOp::ty)
            .is_some_and(|ty| ty.scalar == DType::F32)
    {
        return execute_lowered_elementwise(program, bindings);
    }
    let elements = output_shape.numel()?;
    let values = (0..elements)
        .map(|linear| eval_webgpu_narrow(&store.sources()[1], bindings, linear, output_shape))
        .collect::<crate::Result<Vec<_>>>()?;
    TensorData::from_scalars(output_shape.clone(), dtype, values)
}

fn eval_webgpu_narrow(
    node: &UOp,
    bindings: &KernelBindings,
    linear: usize,
    output_shape: &Shape,
) -> crate::Result<Scalar> {
    let dtype = node.ty().ok_or(crate::Error::InvalidIndex)?.scalar;
    let child = |position| {
        eval_webgpu_narrow(
            node.sources()
                .get(position)
                .ok_or(crate::Error::InvalidIndex)?,
            bindings,
            linear,
            output_shape,
        )
    };
    match node.operation() {
        Operation::Const(LiteralValue::Scalar { dtype, bits }) => quantize_webgpu_scalar(
            *dtype,
            match dtype {
                DType::Bool => Scalar::Bool(*bits != 0),
                DType::I32 => Scalar::I(*bits as i32 as i64),
                DType::U32 => Scalar::U(*bits as u32 as u64),
                DType::F16 => Scalar::F(crate::tensor::f16_to_f32(*bits as u16) as f64),
                DType::BF16 => Scalar::F(crate::tensor::bf16_to_f32(*bits as u16) as f64),
                DType::F32 => Scalar::F(f32::from_bits(*bits as u32) as f64),
                _ => return Err(crate::Error::InvalidIndex),
            },
        ),
        Operation::Load => {
            let index = node.sources().first().ok_or(crate::Error::InvalidIndex)?;
            let (buffer, input_shape, view) = match index.operation() {
                Operation::Index(IndexValue::Buffer {
                    buffer,
                    input_shape,
                    ..
                }) => (*buffer, input_shape, None),
                Operation::Index(IndexValue::View {
                    buffer,
                    input_shape,
                    view,
                    ..
                }) => (*buffer, input_shape, Some(view)),
                _ => return Err(crate::Error::InvalidIndex),
            };
            let logical = semantic_broadcast_offset(input_shape, output_shape, linear)?;
            let offset = match view {
                Some(view) => view
                    .element_offset(logical)
                    .map_err(|_| crate::Error::InvalidIndex)
                    .and_then(|offset| {
                        usize::try_from(offset).map_err(|_| crate::Error::InvalidIndex)
                    })?,
                None => logical,
            };
            Ok(bindings
                .get(buffer)
                .ok_or(crate::Error::InvalidIndex)?
                .storage()
                .scalar(offset))
        }
        Operation::Cast => quantize_webgpu_scalar(dtype, child(0)?),
        Operation::GraphBinary(op) => {
            let lhs = child(0)?;
            let rhs = child(1)?;
            quantize_webgpu_scalar(dtype, webgpu_binary(lhs, rhs, dtype, *op)?)
        }
        Operation::Binary(op) => {
            use crate::uop::Binary::{Add, Eq, Le, Lt, Mul, Sub};
            let lhs = child(0)?;
            let rhs = child(1)?;
            match op {
                Add => quantize_webgpu_scalar(
                    dtype,
                    webgpu_binary(lhs, rhs, dtype, crate::BinaryOp::Add)?,
                ),
                Sub => quantize_webgpu_scalar(
                    dtype,
                    webgpu_binary(lhs, rhs, dtype, crate::BinaryOp::Sub)?,
                ),
                Mul => quantize_webgpu_scalar(
                    dtype,
                    webgpu_binary(lhs, rhs, dtype, crate::BinaryOp::Mul)?,
                ),
                Eq => Ok(Scalar::Bool(lhs.as_f64() == rhs.as_f64())),
                Lt => Ok(Scalar::Bool(lhs.as_f64() < rhs.as_f64())),
                Le => Ok(Scalar::Bool(lhs.as_f64() <= rhs.as_f64())),
                _ => Err(crate::Error::InvalidIndex),
            }
        }
        Operation::GraphCompare(op) => {
            let lhs = child(0)?.as_f64();
            let rhs = child(1)?.as_f64();
            Ok(Scalar::Bool(match op {
                crate::CompareOp::Eq => lhs == rhs,
                crate::CompareOp::Ne => lhs != rhs,
                crate::CompareOp::Lt => lhs < rhs,
                crate::CompareOp::Le => lhs <= rhs,
                crate::CompareOp::Gt => lhs > rhs,
                crate::CompareOp::Ge => lhs >= rhs,
            }))
        }
        Operation::GraphLogical(op) => {
            let lhs = child(0)?.as_bool();
            Ok(Scalar::Bool(match op {
                crate::LogicalOp::Not => !lhs,
                crate::LogicalOp::And => lhs && child(1)?.as_bool(),
                crate::LogicalOp::Or => lhs || child(1)?.as_bool(),
            }))
        }
        Operation::Ternary(crate::uop::Ternary::Where) => {
            let selected = if child(0)?.as_bool() {
                child(1)?
            } else {
                child(2)?
            };
            quantize_webgpu_scalar(dtype, selected)
        }
        _ => Err(crate::Error::InvalidIndex),
    }
}

fn semantic_broadcast_offset(input: &Shape, output: &Shape, linear: usize) -> crate::Result<usize> {
    if input.rank() > output.rank() {
        return Err(crate::Error::InvalidIndex);
    }
    let input_strides = input.contiguous_strides();
    let output_strides = output.contiguous_strides();
    let pad = output.rank() - input.rank();
    let mut offset = 0usize;
    for axis in 0..input.rank() {
        let dim = input.dims()[axis];
        let output_dim = output.dims()[pad + axis];
        if dim != 1 && dim != output_dim {
            return Err(crate::Error::InvalidIndex);
        }
        if dim != 1 {
            let coordinate = (linear / output_strides[pad + axis]) % dim;
            offset = offset
                .checked_add(
                    coordinate
                        .checked_mul(input_strides[axis])
                        .ok_or(crate::Error::InvalidIndex)?,
                )
                .ok_or(crate::Error::InvalidIndex)?;
        }
    }
    Ok(offset)
}

fn quantize_webgpu_scalar(dtype: DType, value: Scalar) -> crate::Result<Scalar> {
    Ok(TensorData::from_scalars([], dtype, [value])?
        .storage()
        .scalar(0))
}

fn webgpu_binary(
    lhs: Scalar,
    rhs: Scalar,
    dtype: DType,
    op: crate::BinaryOp,
) -> crate::Result<Scalar> {
    use crate::BinaryOp::{Add, Mul, Sub};
    Ok(match dtype {
        DType::F16 | DType::BF16 | DType::F32 => {
            let lhs = lhs.as_f64();
            let rhs = rhs.as_f64();
            Scalar::F(match op {
                Add => lhs + rhs,
                Sub => lhs - rhs,
                Mul => lhs * rhs,
                _ => return Err(crate::Error::InvalidIndex),
            })
        }
        DType::I32 => {
            let lhs = lhs.as_i64() as i32;
            let rhs = rhs.as_i64() as i32;
            Scalar::I(match op {
                Add => lhs.wrapping_add(rhs),
                Sub => lhs.wrapping_sub(rhs),
                Mul => lhs.wrapping_mul(rhs),
                _ => return Err(crate::Error::InvalidIndex),
            } as i64)
        }
        DType::U32 => {
            let lhs = lhs.as_u64() as u32;
            let rhs = rhs.as_u64() as u32;
            Scalar::U(match op {
                Add => lhs.wrapping_add(rhs),
                Sub => lhs.wrapping_sub(rhs),
                Mul => lhs.wrapping_mul(rhs),
                _ => return Err(crate::Error::InvalidIndex),
            } as u64)
        }
        DType::Bool => Scalar::Bool(match op {
            Add => lhs.as_bool() || rhs.as_bool(),
            Sub => lhs.as_bool() != rhs.as_bool(),
            Mul => lhs.as_bool() && rhs.as_bool(),
            _ => return Err(crate::Error::InvalidIndex),
        }),
        _ => return Err(crate::Error::InvalidIndex),
    })
}

fn decode_mock_scalar(dtype: DType, bytes: &[u8]) -> Result<Scalar, WebGpuError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(bytes == [1]),
        DType::I32 => {
            Scalar::I(i32::from_le_bytes(bytes.try_into().map_err(|_| WebGpuError::Bounds)?) as i64)
        }
        DType::U32 => {
            Scalar::U(u32::from_le_bytes(bytes.try_into().map_err(|_| WebGpuError::Bounds)?) as u64)
        }
        _ => {
            return Err(WebGpuError::InvalidBinding(
                "mock transaction storage dtype".into(),
            ));
        }
    })
}

#[test]
fn renderer_mock_matches_cpu_for_affine_broadcast_select_and_casts() {
    let mut graph = Graph::new();
    let storage = graph.input("storage", Shape::from([4, 2]));
    let view = graph.shrink(storage, [(1, 3), (0, 2)]).unwrap();
    let row = graph.input("row", Shape::from([1, 2]));
    let sum = graph.add(view, row).unwrap();
    let two = graph.constant(TensorData::scalar(2.0));
    let product = graph.mul(sum, two).unwrap();
    let nine = graph.constant(TensorData::scalar(9.0));
    let condition = graph.gt(product, nine).unwrap();
    let truth = graph.cast(product, DType::Bool).unwrap();
    let round_trip = graph.cast(truth, DType::F32).unwrap();
    let output = graph.select(condition, product, round_trip).unwrap();
    let inputs = HashMap::from([
        (
            "storage".into(),
            TensorData::new([4, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
        ),
        (
            "row".into(),
            TensorData::new([1, 2], vec![0.5, -1.0]).unwrap(),
        ),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = WgslRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.source.contains("@builtin(global_invocation_id)"));
    assert!(rendered.source.contains("var<uniform> rg_extent"));
    assert!(rendered.source.contains("2u +"));
    let expected_order = item
        .ordered_inputs()
        .iter()
        .map(|binding| binding.desc.id)
        .chain([output.index() as u64])
        .collect::<Vec<_>>();
    assert_eq!(
        rendered
            .buffers
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        expected_order
    );
    let (actual, mock) = execute_mock(&graph, output, &inputs);
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("shader_create"))
            .count(),
        1
    );
}

#[test]
fn affine_forms_bool_packing_and_integer_wrapping_match_cpu_oracle() {
    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([1, 6]));
    let reshaped = graph.reshape(input, [1, 2, 3]).unwrap();
    let expanded = graph.expand(reshaped, [2, 2, 3]).unwrap();
    let permuted = graph.permute(expanded, vec![1, 0, 2]).unwrap();
    let strided = graph
        .stride(
            permuted,
            [
                Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                Slice {
                    start: Some(0),
                    stop: None,
                    step: 2,
                },
            ],
        )
        .unwrap();
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.add(strided, one).unwrap();
    let inputs = HashMap::from([(
        "input".into(),
        TensorData::new([1, 6], vec![0., 1., 2., 3., 4., 5.]).unwrap(),
    )]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let (actual, _) = execute_mock(&graph, output, &inputs);
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );

    for dtype in [DType::I32, DType::U32] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [4], dtype);
        let rhs = graph.input_dtype("rhs", [4], dtype);
        let added = graph.add(lhs, rhs).unwrap();
        let multiplied = graph.mul(added, rhs).unwrap();
        let wrapped = graph.sub(multiplied, lhs).unwrap();
        let condition = graph.gt(wrapped, lhs).unwrap();
        let condition_as_value = graph.cast(condition, dtype).unwrap();
        let output = graph
            .select(condition, wrapped, condition_as_value)
            .unwrap();
        let inputs = if dtype == DType::I32 {
            HashMap::from([
                ("lhs".into(), ints(&[i32::MAX, i32::MIN, -1, 7])),
                ("rhs".into(), ints(&[2, -1, i32::MAX, -9])),
            ])
        } else {
            HashMap::from([
                ("lhs".into(), uints(&[u32::MAX, 0, 1, 7])),
                ("rhs".into(), uints(&[2, u32::MAX, u32::MAX, 9])),
            ])
        };
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let item = &schedule(&graph, output).unwrap().items[0];
        let rendered = WgslRenderer::new(4, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap();
        if dtype == DType::I32 {
            assert!(rendered.source.contains("bitcast<i32>(bitcast<u32>"));
        }
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{dtype:?}"
        );
    }

    let mut bool_graph = Graph::new();
    let lhs = bool_graph.input_dtype("lhs", [5], DType::Bool);
    let rhs = bool_graph.input_dtype("rhs", [5], DType::Bool);
    let added = bool_graph.add(lhs, rhs).unwrap();
    let subtracted = bool_graph.sub(lhs, rhs).unwrap();
    let bool_output = bool_graph.mul(added, subtracted).unwrap();
    let bool_inputs = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [5],
                DType::Bool,
                [true, true, false, false, true].map(Scalar::Bool),
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [5],
                DType::Bool,
                [true, false, true, false, false].map(Scalar::Bool),
            )
            .unwrap(),
        ),
    ]);
    let bool_expected = CpuBackend
        .execute(&bool_graph, bool_output, &bool_inputs)
        .unwrap();
    let item = &schedule(&bool_graph, bool_output).unwrap().items[0];
    let rendered = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.source.contains("array<atomic<u32>>"));
    assert!(rendered.source.contains("atomicAnd"));
    assert!(rendered.source.contains("atomicOr"));
    let (bool_actual, _) = execute_mock(&bool_graph, bool_output, &bool_inputs);
    assert_eq!(
        bool_actual.to_le_bytes().unwrap(),
        bool_expected.to_le_bytes().unwrap()
    );

    let ordered = bool_graph.lt(lhs, rhs).unwrap();
    let ordered_expected = CpuBackend
        .execute(&bool_graph, ordered, &bool_inputs)
        .unwrap();
    let ordered_item = &schedule(&bool_graph, ordered).unwrap().items[0];
    let ordered_rendered = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&ordered_item.kernel)
        .unwrap();
    assert!(ordered_rendered.source.contains("select(0u, 1u"));
    let (ordered_actual, _) = execute_mock(&bool_graph, ordered, &bool_inputs);
    assert_eq!(
        ordered_actual.to_le_bytes().unwrap(),
        ordered_expected.to_le_bytes().unwrap()
    );
}

#[test]
fn complete_supported_cast_matrix_matches_oracle_bytes_and_wgsl_contract() {
    let cases = [
        (
            "i32_to_u32",
            DType::I32,
            DType::U32,
            ints(&[-1, 0, 1, i32::MIN]),
            "bitcast<u32>",
        ),
        (
            "u32_to_i32",
            DType::U32,
            DType::I32,
            uints(&[0, 1, i32::MAX as u32 + 1, u32::MAX]),
            "bitcast<i32>",
        ),
        (
            "i32_to_f32",
            DType::I32,
            DType::F32,
            ints(&[-16_777_217, -1, 0, 16_777_217]),
            "f32(",
        ),
        (
            "u32_to_f32",
            DType::U32,
            DType::F32,
            uints(&[0, 1, 16_777_217, u32::MAX]),
            "f32(",
        ),
        (
            "f32_to_i32",
            DType::F32,
            DType::I32,
            TensorData::new([4], vec![f32::NEG_INFINITY, f32::NAN, 2.75, f32::INFINITY]).unwrap(),
            "rg_f32_to_i32(",
        ),
        (
            "f32_to_u32",
            DType::F32,
            DType::U32,
            TensorData::new([4], vec![-1.0, f32::NAN, 2.75, f32::INFINITY]).unwrap(),
            "rg_f32_to_u32(",
        ),
    ];
    for (name, source, target, value, marker) in cases {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [4], source);
        let output = graph.cast(input, target).unwrap();
        let inputs = HashMap::from([("input".into(), value)]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let item = &schedule(&graph, output).unwrap().items[0];
        let rendered = WgslRenderer::new(4, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap();
        assert!(rendered.source.contains(marker), "{name}");
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn packed_narrow_storage_views_arithmetic_and_selection_match_cpu_bytes() {
    let cases = [
        (
            DType::F16,
            [
                0x0000u16, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e01,
            ],
            "rg_f16_to_f32",
            "rg_f32_to_f16",
        ),
        (
            DType::BF16,
            [
                0x0000u16, 0x8000, 0x0001, 0x007f, 0x0080, 0x3f80, 0x7f7f, 0x7f80, 0xff80, 0x7fc1,
            ],
            "rg_bf16_to_f32",
            "rg_f32_to_bf16",
        ),
    ];
    for (dtype, bits, decode, encode) in cases {
        let mut graph = Graph::new();
        let storage = graph.input_dtype("storage", [2, 5], dtype);
        let viewed = graph.shrink(storage, [(1, 2), (0, 5)]).unwrap();
        let viewed = graph.reshape(viewed, [5]).unwrap();
        let rhs = graph.input_dtype("rhs", [1], dtype);
        let sum = graph.add(viewed, rhs).unwrap();
        let condition = graph.input_dtype("condition", [5], DType::Bool);
        let output = graph.select(condition, sum, viewed).unwrap();
        let inputs = HashMap::from([
            ("storage".into(), narrow_data(dtype, [2, 5], &bits)),
            ("rhs".into(), narrow_data(dtype, [1], &[0x0001])),
            (
                "condition".into(),
                TensorData::from_scalars(
                    [5],
                    DType::Bool,
                    [true, false, true, false, true].map(Scalar::Bool),
                )
                .unwrap(),
            ),
        ]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let item = &schedule(&graph, output).unwrap().items[0];
        let rendered = WgslRenderer::new(4, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap();
        assert!(rendered.source.contains(decode), "{dtype:?}");
        assert!(rendered.source.contains(encode), "{dtype:?}");
        assert!(rendered.source.contains("array<atomic<u32>>"));
        assert!(rendered.source.contains("(gid & 1u) * 16u"));
        assert!(rendered.source.contains("atomicAnd"));
        assert!(rendered.source.contains("atomicOr"));
        assert!(!rendered.source.contains("enable f16"));
        let output_abi = rendered.buffers.last().unwrap();
        assert_eq!(output_abi.logical_bytes().unwrap(), 10);
        assert_eq!(output_abi.physical_bytes().unwrap(), 12);
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{dtype:?}"
        );
    }
}

#[test]
fn narrow_f32_and_cross_narrow_casts_match_cpu_raw_bytes() {
    let f32_bits = [
        0x00000000u32,
        0x80000000,
        0x33800000,
        0x387fc000,
        0x3f801000,
        0x7f800000,
        0xff800000,
        0x7fc12000,
    ];
    let f32_bytes = f32_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    let cases = [
        (
            DType::F32,
            DType::F16,
            TensorData::from_le_bytes([8], DType::F32, &f32_bytes).unwrap(),
        ),
        (
            DType::F32,
            DType::BF16,
            TensorData::from_le_bytes([8], DType::F32, &f32_bytes).unwrap(),
        ),
        (
            DType::F16,
            DType::F32,
            narrow_data(
                DType::F16,
                [8],
                &[
                    0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7c00, 0xfc00, 0x7e09,
                ],
            ),
        ),
        (
            DType::BF16,
            DType::F32,
            narrow_data(
                DType::BF16,
                [8],
                &[
                    0x0000, 0x8000, 0x0001, 0x007f, 0x3f80, 0x7f80, 0xff80, 0x7fc9,
                ],
            ),
        ),
        (
            DType::F16,
            DType::BF16,
            narrow_data(
                DType::F16,
                [8],
                &[
                    0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0x7c00, 0xfc00, 0x7e09,
                ],
            ),
        ),
        (
            DType::BF16,
            DType::F16,
            narrow_data(
                DType::BF16,
                [8],
                &[
                    0x0000, 0x8000, 0x0001, 0x007f, 0x3f80, 0x7f80, 0xff80, 0x7fc9,
                ],
            ),
        ),
    ];
    for (source, target, value) in cases {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [8], source);
        let output = graph.cast(input, target).unwrap();
        let inputs = HashMap::from([("input".into(), value)]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let item = schedule(&graph, output).unwrap().items.pop().unwrap();
        let rendered = WgslRenderer::new(4, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap();
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{source:?}->{target:?}\n{}",
            rendered.source
        );
    }
}

#[test]
fn f32_to_bf16_source_and_mock_preserve_nan_payloads_exactly() {
    let input_bits = [
        0x0000_0000u32,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x3f80_8000,
        0x3f81_8000,
        0xbf80_8000,
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0001,
        0x7f80_7fff,
        0x7f81_0000,
        0x7fc0_0000,
        0x7fff_ffff,
        0xff80_0001,
        0xffff_ffff,
    ];
    let expected = [
        0x0000u16, 0x8000, 0x0000, 0x0080, 0x3f80, 0x3f82, 0xbf80, 0x7f80, 0xff80, 0x7f81, 0x7f81,
        0x7f81, 0x7fc0, 0x7fff, 0xff81, 0xffff,
    ];
    let bytes = input_bits
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let value = TensorData::from_le_bytes([16], DType::F32, &bytes).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [16], DType::F32);
    let output = graph.cast(input, DType::BF16).unwrap();
    let inputs = HashMap::from([("input".into(), value)]);
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let rendered = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(
        rendered
            .source
            .contains("(bits & 0x7f800000u) == 0x7f800000u")
    );
    assert!(rendered.source.contains("return upper | 1u"));
    let (actual, _) = execute_mock(&graph, output, &inputs);
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        expected
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
fn fused_narrow_nodes_round_at_each_typed_graph_boundary() {
    for (dtype, lhs_bits, increment_bits, scale_bits, marker) in [
        (DType::F16, 0x3c00, 0x1000, 0x63fe, "rg_f32_to_f16"),
        (DType::BF16, 0x3f80, 0x3b80, 0x42fe, "rg_f32_to_bf16"),
    ] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [1], dtype);
        let increment = graph.input_dtype("increment", [1], dtype);
        let scale = graph.input_dtype("scale", [1], dtype);
        let rounded_sum = graph.add(lhs, increment).unwrap();
        let output = graph.mul(rounded_sum, scale).unwrap();
        let inputs = HashMap::from([
            ("lhs".into(), narrow_data(dtype, [1], &[lhs_bits])),
            (
                "increment".into(),
                narrow_data(dtype, [1], &[increment_bits]),
            ),
            ("scale".into(), narrow_data(dtype, [1], &[scale_bits])),
        ]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        assert_eq!(
            expected.to_le_bytes().unwrap(),
            scale_bits.to_le_bytes(),
            "adversarial fixture must distinguish fused from typed rounding"
        );
        let item = &schedule(&graph, output).unwrap().items[0];
        let rendered = WgslRenderer::new(1, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap();
        assert!(rendered.source.matches(marker).count() >= 3, "{dtype:?}");
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{dtype:?}"
        );
    }
}

#[test]
fn narrow_scalar_literals_preserve_raw_bits_at_the_storage_boundary() {
    for (dtype, bits, marker) in [
        (DType::F16, 0x8001u64, "rg_f16_to_f32(0x8001u)"),
        (DType::BF16, 0x7fc1u64, "rg_bf16_to_f32(0x7fc1u)"),
    ] {
        let ty = UType::scalar(dtype);
        let shape = Shape::new([]);
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(UType::scalar(DType::I64)),
            vec![UOp::constant(1, UType::scalar(DType::I64))],
        );
        let address = UOp::from_operation(
            Operation::DefineGlobal(AddressValue {
                space: AddressSpace::Global,
                name: "literal".into(),
                element: ty,
            }),
            Some(ty),
            vec![],
        );
        let index = UOp::from_operation(
            Operation::Index(IndexValue::Buffer {
                buffer: 77,
                elements: 1,
                input_shape: shape.clone(),
                output_shape: shape,
            }),
            Some(ty),
            vec![address, range.clone()],
        );
        let rendered = WgslRenderer::new(1, capabilities())
            .unwrap()
            .render(&UOp::sink(vec![
                UOp::from_operation(
                    Operation::Store,
                    None,
                    vec![index, UOp::scalar_constant(dtype, bits, ty)],
                ),
                UOp::from_operation(Operation::EndRange, None, vec![range]),
            ]))
            .unwrap();
        assert!(rendered.source.contains(marker), "{dtype:?}");
        assert!(rendered.source.contains("& 0xffffu"), "{dtype:?}");
    }
}

#[test]
fn narrow_capability_packing_cache_and_pre_submission_rejections_are_exact() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [3], DType::F16);
    let rhs = graph.input_dtype("rhs", [3], DType::F16);
    let output = graph.add(lhs, rhs).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let software = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let mut native_feature = capabilities();
    native_feature.shader_f16 = true;
    let advertised = WgslRenderer::new(4, native_feature)
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert_ne!(software.cache_key, advertised.cache_key);
    assert_eq!(software.source, advertised.source);
    assert!(software.source.contains(&format!(
        "ABI {WEBGPU_ABI_VERSION} STATUS {WEBGPU_STATUS_VERSION} NARROW {WEBGPU_NARROW_ABI_VERSION}"
    )));
    assert_eq!(
        software.buffers.last().unwrap().physical_bytes().unwrap(),
        8
    );

    let mut too_small = capabilities();
    too_small.max_buffer_size = 7;
    assert!(matches!(
        WgslRenderer::new(4, too_small).unwrap().render(&item.kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("buffer limit")
    ));

    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let mut malformed = software.clone();
    malformed.buffers.last_mut().unwrap().elements = 4;
    assert!(matches!(
        device.compile(&malformed),
        Err(WebGpuError::InvalidBinding(reason)) if reason.contains("storage metadata")
    ));
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("shader_create"))
    );
    let mut malformed_view = software.clone();
    malformed_view.buffers[0].view = Some(
        ViewMap {
            source_shape: Shape::new([3]),
            logical_shape: Shape::new([3]),
            strides: vec![2],
            offset: 0,
        }
        .into(),
    );
    assert!(matches!(
        device.compile(&malformed_view),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("invalid signed affine")
    ));
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("shader_create"))
    );

    let values = BTreeMap::from([
        (
            lhs.index() as u64,
            narrow_data(DType::F16, [3], &[0x3c00; 3]),
        ),
        (
            rhs.index() as u64,
            narrow_data(DType::F16, [3], &[0x4000; 3]),
        ),
    ]);
    let buffers = allocate_rendered(&device, &queue, &software, &values);
    let wrong_dtype = device.allocate_typed(3, DType::BF16).unwrap();
    let mut wrong_bindings = buffers.iter().collect::<Vec<_>>();
    wrong_bindings[0] = &wrong_dtype;
    let pipeline = device.cache().load(&software).unwrap();
    assert!(matches!(
        pipeline.launch(&queue, &wrong_bindings),
        Err(WebGpuError::InvalidBinding(reason)) if reason.contains("dtype mismatch")
    ));
    let narrow_copy = device.allocate_typed(3, DType::F16).unwrap();
    assert!(matches!(
        queue.copy(&buffers[0], &narrow_copy, 0, 0, 6),
        Err(WebGpuError::InvalidArgument(reason)) if reason.contains("four-byte aligned")
    ));
    let output_buffer = buffers.last().unwrap();
    queue.write(output_buffer, 0, &[0x5a; 6]).unwrap();
    let generation = output_buffer.generation();
    mock.state.lock().unwrap().failures.launch = Some("narrow dispatch");
    assert!(matches!(
        pipeline.launch(&queue, &buffers.iter().collect::<Vec<_>>()),
        Err(WebGpuError::Driver {
            operation: "launch",
            ..
        })
    ));
    assert_eq!(output_buffer.generation(), generation);
    let mut bytes = [0u8; 6];
    queue.read(output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x5a; 6]);

    let mut ordinary = Graph::new();
    let values = ordinary.input_dtype("values", [3], DType::F32);
    let divisors = ordinary.input_dtype("divisors", [3], DType::F32);
    let divided = ordinary.binary(BinaryOp::Div, values, divisors).unwrap();
    let item = &schedule(&ordinary, divided).unwrap().items[0];
    let rendered = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.transaction.is_none());
    assert!(rendered.source.contains(" / "));
}

#[test]
fn zero_extent_narrow_storage_has_no_physical_allocation_or_submission() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [0], DType::BF16);
    let output = graph.cast(input, DType::F16).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let rendered = WgslRenderer::new(1, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    assert!(buffers.iter().all(WebGpuBuffer::is_empty));
    assert!(
        device
            .cache()
            .load(&rendered)
            .unwrap()
            .launch(&queue, &buffers.iter().collect::<Vec<_>>())
            .unwrap()
            .is_none()
    );
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch")));
}

#[test]
fn guarded_i32_u32_operation_matrix_matches_cpu_bytes() {
    use crate::BinaryOp;
    let operations = [
        BinaryOp::Div,
        BinaryOp::FloorDiv,
        BinaryOp::TruncDiv,
        BinaryOp::Mod,
        BinaryOp::FMod,
        BinaryOp::Shl,
        BinaryOp::Shr,
    ];
    for dtype in [DType::I32, DType::U32] {
        for operation in operations {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [4], dtype);
            let rhs = graph.input_dtype("rhs", [4], dtype);
            let output = graph.binary(operation, lhs, rhs).unwrap();
            let lhs_value = if dtype == DType::I32 {
                ints(&[-9, -7, 8, i32::MIN])
            } else {
                uints(&[9, 7, 8, u32::MAX])
            };
            let rhs_value = if matches!(operation, BinaryOp::Shl | BinaryOp::Shr) {
                if dtype == DType::I32 {
                    ints(&[1, 2, 3, 1])
                } else {
                    uints(&[1, 2, 3, 1])
                }
            } else if dtype == DType::I32 {
                ints(&[2, -3, 2, -1])
            } else {
                uints(&[2, 3, 2, 1])
            };
            let inputs = HashMap::from([("lhs".into(), lhs_value), ("rhs".into(), rhs_value)]);
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            let item = &schedule(&graph, output).unwrap().items[0];
            let rendered = WgslRenderer::new(4, capabilities())
                .unwrap()
                .render(&item.kernel)
                .unwrap();
            assert_eq!(
                rendered.transaction.as_ref().unwrap().guards[0].operation,
                GuardedIntegerOp::from_binary(operation).unwrap(),
                "{dtype:?} {operation:?}"
            );
            assert!(rendered.source.contains("atomicMin(&rg_status.value"));
            let (actual, _) = execute_mock(&graph, output, &inputs);
            assert_eq!(
                actual.to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap(),
                "{dtype:?} {operation:?}"
            );
        }
    }
}

#[test]
fn renderer_identity_and_unsupported_work_are_pre_submission() {
    let mut graph = Graph::new();
    let input = graph.input("x", [4]);
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.sub(input, one).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let first = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let second = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert_eq!(first.source, second.source);
    assert_eq!(first.cache_key, second.cache_key);
    let mut changed = capabilities();
    changed.timestamp_query = false;
    assert_ne!(
        first.cache_key,
        WgslRenderer::new(4, changed)
            .unwrap()
            .render(&item.kernel)
            .unwrap()
            .cache_key
    );

    let reduced = graph
        .reduce(input, ReduceKind::Sum, Some(vec![0]), false)
        .unwrap();
    assert!(matches!(
        WgslRenderer::new(4, capabilities()).unwrap().render(&schedule(&graph, reduced).unwrap().items[0].kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("reductions")
    ));
    let mut int_graph = Graph::new();
    let lhs = int_graph.input_dtype("lhs", [4], DType::I32);
    let rhs = int_graph.input_dtype("rhs", [4], DType::I32);
    let divided_node = int_graph.binary(BinaryOp::Div, lhs, rhs).unwrap();
    let floored_node = int_graph.binary(BinaryOp::FloorDiv, lhs, rhs).unwrap();
    let divided = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&schedule(&int_graph, divided_node).unwrap().items[0].kernel)
        .unwrap();
    let floored = WgslRenderer::new(4, capabilities())
        .unwrap()
        .render(&schedule(&int_graph, floored_node).unwrap().items[0].kernel)
        .unwrap();
    assert!(divided.transaction.is_some());
    assert!(divided.source.contains("atomicMin(&rg_status.value"));
    assert!(divided.source.contains(&format!(
        "ABI {WEBGPU_ABI_VERSION} STATUS {WEBGPU_STATUS_VERSION}"
    )));
    assert_ne!(divided.cache_key, floored.cache_key);
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let mut malformed = divided.clone();
    malformed.transaction.as_mut().unwrap().version = 0;
    assert!(matches!(
        device.compile(&malformed),
        Err(WebGpuError::InvalidBinding(reason)) if reason.contains("transaction metadata")
    ));
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("shader_create"))
    );
    let mut too_few = capabilities();
    too_few.max_storage_buffers_per_shader_stage = 1;
    assert!(matches!(
        WgslRenderer::new(4, too_few).unwrap().render(&item.kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("storage-buffer limit")
    ));
}

#[test]
fn shared_scalar_lane_intrinsics_division_bitwise_and_narrow_commit_render_structurally() {
    let renderer = WgslRenderer::new(4, capabilities()).unwrap();
    let dialect = renderer::WgslScalarDialect;
    let typed = |register: &str, dtype| TypedValue {
        register: register.to_string(),
        ty: UType::scalar(dtype),
    };
    let mixed_bitwise = LaneInstruction::GraphBinary {
        output: typed("out", DType::I32),
        lhs: typed("lhs", DType::Bool),
        rhs: typed("rhs", DType::I32),
        op: BinaryOp::BitOr,
    };
    let mixed_add = LaneInstruction::GraphBinary {
        output: typed("out", DType::F32),
        lhs: typed("lhs", DType::I32),
        rhs: typed("rhs", DType::F32),
        op: BinaryOp::Add,
    };
    let mixed_compare = LaneInstruction::Compare {
        output: typed("out", DType::Bool),
        lhs: typed("lhs", DType::I32),
        rhs: typed("rhs", DType::F32),
        op: CompareOp::Lt,
    };
    let bitwise = emit_scalar_lane(&dialect, &mixed_bitwise).unwrap();
    let add = emit_scalar_lane(&dialect, &mixed_add).unwrap();
    let compare_error = emit_scalar_lane(&dialect, &mixed_compare).unwrap_err();
    assert!(bitwise.contains("select(0i, 1i, lhs)") && bitwise.contains(" | "));
    assert!(add.contains("f32(lhs)") && add.contains(" + "));
    assert!(compare_error.contains("compare dtype"));

    for dtype in [DType::F16, DType::BF16, DType::F32] {
        for (name, operation) in [
            ("sqrt", crate::UnaryOp::Sqrt),
            ("exp2", crate::UnaryOp::Exp2),
            ("log2", crate::UnaryOp::Log2),
            ("sin", crate::UnaryOp::Sin),
            ("trunc", crate::UnaryOp::Trunc),
        ] {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2], dtype);
            let output = match operation {
                crate::UnaryOp::Sqrt => graph.sqrt(input),
                crate::UnaryOp::Exp2 => graph.exp2(input),
                crate::UnaryOp::Log2 => graph.log2(input),
                crate::UnaryOp::Sin => graph.sin(input),
                crate::UnaryOp::Trunc => graph.trunc(input),
                _ => unreachable!(),
            }
            .unwrap();
            let scheduled = schedule(&graph, output).unwrap();
            let item = scheduled
                .items
                .iter()
                .find(|item| item.node == output)
                .unwrap();
            let rendered = renderer.render(&item.kernel).unwrap();
            let intrinsic = format!("{name}(");
            assert!(
                rendered.source.contains(intrinsic.as_str()),
                "{dtype:?} {operation:?}"
            );
            assert!(rendered.source.contains(WGSL_RENDERER_VERSION));
            if dtype == DType::F16 {
                assert!(rendered.source.matches("rg_f32_to_f16").count() >= 2);
            }
            if dtype == DType::BF16 {
                assert!(rendered.source.matches("rg_f32_to_bf16").count() >= 2);
            }
        }
    }

    for dtype in [DType::I32, DType::U32] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let output = graph.sin(input).unwrap();
        let scheduled = schedule(&graph, output).unwrap();
        let item = scheduled
            .items
            .iter()
            .find(|item| item.node == output)
            .unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        assert!(rendered.source.contains("sin(f32("), "{dtype:?}");
    }

    let mut graph = Graph::new();
    let narrow = graph.input_dtype("narrow", [2], DType::F16);
    let exp = graph.exp2(narrow).unwrap();
    let chained = graph.log2(exp).unwrap();
    let scheduled = schedule(&graph, chained).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == chained)
        .unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    assert!(rendered.source.matches("rg_f32_to_f16").count() >= 3);

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::F32);
    let rhs = graph.input_dtype("rhs", [2], DType::F32);
    let output = graph.binary(BinaryOp::Div, lhs, rhs).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert!(
        renderer
            .render(&item.kernel)
            .unwrap()
            .source
            .contains(" / ")
    );

    for dtype in [DType::Bool, DType::I32, DType::U32] {
        for op in [BinaryOp::BitAnd, BinaryOp::BitOr, BinaryOp::BitXor] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2], dtype);
            let rhs = graph.input_dtype("rhs", [2], dtype);
            let output = graph.binary(op, lhs, rhs).unwrap();
            let scheduled = schedule(&graph, output).unwrap();
            let item = scheduled
                .items
                .iter()
                .find(|item| item.node == output)
                .unwrap();
            renderer.render(&item.kernel).unwrap();
        }
    }

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let rhs = graph.input_dtype("rhs", [2], DType::I32);
    let divided = graph.binary(BinaryOp::Div, lhs, rhs).unwrap();
    let output = graph.neg(divided).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    let guarded = renderer.render(&item.kernel).unwrap();
    assert_eq!(guarded.transaction.as_ref().unwrap().guards.len(), 1);
    assert!(guarded.source.contains("if (rg_ok)"));
    assert!(
        guarded.source.find("atomicMin(&rg_status.value").unwrap()
            < guarded.source.rfind("0u -").unwrap()
    );

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F64);
    let output = graph.sqrt(input).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    assert!(matches!(
        renderer.render(&item.kernel),
        Err(WebGpuError::Unsupported(_))
    ));
}

#[test]
fn nested_guards_choose_earliest_fault_and_commit_only_clean_generation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [4], DType::I32);
    let divisor = graph.input_dtype("divisor", [4], DType::I32);
    let count_lhs = graph.input_dtype("count_lhs", [4], DType::I32);
    let count_rhs = graph.input_dtype("count_rhs", [1], DType::I32);
    let quotient = graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let quotient = graph.cast(quotient, DType::U32).unwrap();
    let quotient = graph.cast(quotient, DType::I32).unwrap();
    let count = graph.add(count_lhs, count_rhs).unwrap();
    let shifted = graph.binary(BinaryOp::Shl, quotient, count).unwrap();
    let output = graph.add(shifted, lhs).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let rendered = WgslRenderer::new(2, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let abi = rendered.transaction.as_ref().unwrap();
    assert_eq!(abi.version, WEBGPU_TRANSACTION_ABI_VERSION);
    assert_eq!(
        abi.guards
            .iter()
            .map(|guard| guard.operation)
            .collect::<Vec<_>>(),
        [GuardedIntegerOp::Div, GuardedIntegerOp::Shl]
    );
    assert!(rendered.source.contains("gid * 2u + 0u"));
    assert!(rendered.source.contains("gid * 2u + 1u"));

    let mock = Arc::new(MockDispatch::default());
    mock.state.lock().unwrap().fault_order = vec![3, 1, 0, 2];
    let (device, queue) = setup(mock.clone());
    let values = BTreeMap::from([
        (lhs.index() as u64, ints(&[8, 9, 10, 11])),
        (divisor.index() as u64, ints(&[1, 0, 2, 1])),
        (count_lhs.index() as u64, ints(&[39, 0, 0, 0])),
        (count_rhs.index() as u64, ints(&[1])),
    ]);
    let buffers = allocate_rendered(&device, &queue, &rendered, &values);
    let positions = rendered
        .buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let refs = buffers.iter().collect::<Vec<_>>();
    let output_buffer = &buffers[abi.output_abi_index];
    queue.write(output_buffer, 0, &[0x5a; 16]).unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(matches!(
        pipeline.launch(&queue, &refs),
        Err(WebGpuError::InvalidArgument(
            "guarded kernel requires transactional launch"
        ))
    ));

    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(WebGpuError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 0,
            count: Some(40),
            bits: 32,
        })
    ));
    let mut unchanged = [0; 16];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 16]);
    assert_eq!(output_buffer.generation(), 1);

    queue
        .write(
            &buffers[positions[&(divisor.index() as u64)]],
            0,
            &ints(&[0, 1, 2, 1]).to_le_bytes().unwrap(),
        )
        .unwrap();
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(WebGpuError::IntegerFault {
            operation: GuardedIntegerOp::Div,
            index: 0,
            count: None,
            bits: 32,
        })
    ));

    queue
        .write(
            &buffers[positions[&(divisor.index() as u64)]],
            0,
            &ints(&[1, 1, 2, 1]).to_le_bytes().unwrap(),
        )
        .unwrap();
    mock.state.lock().unwrap().failures.read_after = Some((1, "detail"));
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs)
            .unwrap()
            .wait(),
        Err(WebGpuError::Driver { operation: "read", detail }) if detail == "detail"
    ));
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 16]);

    queue
        .write(
            &buffers[positions[&(count_lhs.index() as u64)]],
            0,
            &ints(&[0, 1, 2, 0]).to_le_bytes().unwrap(),
        )
        .unwrap();
    let generation = output_buffer.generation();
    let first = pipeline.launch_transactional(&queue, &refs).unwrap();
    let stale = pipeline.launch_transactional(&queue, &refs).unwrap();
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 16]);
    first.wait().unwrap();
    assert_eq!(output_buffer.generation(), generation + 1);
    assert!(matches!(
        stale.wait(),
        Err(WebGpuError::StaleGeneration { expected, actual })
            if expected == generation && actual == generation + 1
    ));
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                ("lhs".into(), ints(&[8, 9, 10, 11])),
                ("divisor".into(), ints(&[1, 1, 2, 1])),
                ("count_lhs".into(), ints(&[0, 1, 2, 0])),
                ("count_rhs".into(), ints(&[1])),
            ]),
        )
        .unwrap();
    let mut actual = [0; 16];
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected.to_le_bytes().unwrap());
}

#[test]
fn transaction_failures_zero_domain_retry_and_capability_preflight_preserve_visibility() {
    let mut graph = Graph::new();
    let condition = graph.input_dtype("condition", [2], DType::Bool);
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let divisor = graph.input_dtype("divisor", [2], DType::I32);
    let count = graph.input_dtype("count", [2], DType::I32);
    let quotient = graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let shifted = graph.binary(BinaryOp::Shl, lhs, count).unwrap();
    let output = graph.select(condition, quotient, shifted).unwrap();
    let rendered = WgslRenderer::new(2, capabilities())
        .unwrap()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    assert!(rendered.source.contains("else if (rg_ok)"));
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let values = BTreeMap::from([
        (
            condition.index() as u64,
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(false), Scalar::Bool(true)])
                .unwrap(),
        ),
        (lhs.index() as u64, ints(&[4, 8])),
        (divisor.index() as u64, ints(&[0, 2])),
        (count.index() as u64, ints(&[1, 99])),
    ]);
    let buffers = allocate_rendered(&device, &queue, &rendered, &values);
    let refs = buffers.iter().collect::<Vec<_>>();
    let output_buffer = &buffers[rendered.transaction.as_ref().unwrap().output_abi_index];
    let cache = device.cache();
    let pipeline = cache.load(&rendered).unwrap();
    pipeline
        .launch_transactional(&queue, &refs)
        .unwrap()
        .wait()
        .unwrap();
    let mut exact = [0; 8];
    queue.read(output_buffer, 0, &mut exact).unwrap();
    assert_eq!(
        exact,
        [8i32.to_le_bytes(), 4i32.to_le_bytes()].concat().as_slice()
    );

    let sentinel = [0x3c; 8];
    queue.write(output_buffer, 0, &sentinel).unwrap();
    let generation = output_buffer.generation();
    let baseline_buffers = mock.state.lock().unwrap().buffers.len();

    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs[..refs.len() - 1]),
        Err(WebGpuError::InvalidBinding(reason)) if reason.contains("count")
    ));
    assert_eq!(mock.state.lock().unwrap().buffers.len(), baseline_buffers);

    mock.state.lock().unwrap().failures.launch = Some("submit");
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs),
        Err(WebGpuError::Driver { operation: "launch", detail }) if detail == "submit"
    ));
    mock.state.lock().unwrap().failures.wait = Some("compute");
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs)
            .unwrap()
            .wait(),
        Err(WebGpuError::Driver { operation: "wait", detail }) if detail == "compute"
    ));
    mock.state.lock().unwrap().failures.read = Some("status");
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs)
            .unwrap()
            .wait(),
        Err(WebGpuError::Driver { operation: "read", detail }) if detail == "status"
    ));
    mock.state.lock().unwrap().failures.query = Some("nonblocking");
    let token = pipeline.launch_transactional(&queue, &refs).unwrap();
    assert!(matches!(
        token.query(),
        Err(WebGpuError::Driver { operation: "query", detail }) if detail == "nonblocking"
    ));
    drop(token);

    mock.state.lock().unwrap().failures.buffer = Some("candidate");
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs),
        Err(WebGpuError::Driver { operation: "buffer_create", detail }) if detail == "candidate"
    ));
    mock.state.lock().unwrap().failures.buffer_after = Some((1, "status allocation"));
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs),
        Err(WebGpuError::Driver { operation: "buffer_create", detail }) if detail == "status allocation"
    ));
    mock.state.lock().unwrap().failures.write = Some("status initialize");
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs),
        Err(WebGpuError::Driver { operation: "write", detail }) if detail == "status initialize"
    ));
    let mut unchanged = [0; 8];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, sentinel);
    assert_eq!(output_buffer.generation(), generation);
    assert_eq!(mock.state.lock().unwrap().buffers.len(), baseline_buffers);

    pipeline
        .launch_transactional(&queue, &refs)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(output_buffer.generation(), generation + 1);
    assert_eq!(mock.state.lock().unwrap().buffers.len(), baseline_buffers);
    assert_eq!(cache.len(), 1);

    let mut empty = Graph::new();
    let empty_lhs = empty.input_dtype("lhs", [0], DType::U32);
    let empty_rhs = empty.input_dtype("rhs", [0], DType::U32);
    let empty_output = empty.binary(BinaryOp::Div, empty_lhs, empty_rhs).unwrap();
    let empty_rendered = WgslRenderer::new(1, capabilities())
        .unwrap()
        .render(&schedule(&empty, empty_output).unwrap().items[0].kernel)
        .unwrap();
    let empty_buffers = empty_rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let empty_refs = empty_buffers.iter().collect::<Vec<_>>();
    let empty_pipeline = cache.load(&empty_rendered).unwrap();
    let before = empty_buffers.last().unwrap().generation();
    let token = empty_pipeline
        .launch_transactional(&queue, &empty_refs)
        .unwrap();
    assert!(token.query().unwrap());
    token.wait().unwrap();
    assert_eq!(empty_buffers.last().unwrap().generation(), before + 1);

    let mut insufficient = capabilities();
    insufficient.max_storage_buffers_per_shader_stage = 3;
    let mut simple = Graph::new();
    let lhs = simple.input_dtype("lhs", [1], DType::I32);
    let rhs = simple.input_dtype("rhs", [1], DType::I32);
    let divided = simple.binary(BinaryOp::Div, lhs, rhs).unwrap();
    assert!(matches!(
        WgslRenderer::new(1, insufficient)
            .unwrap()
            .render(&schedule(&simple, divided).unwrap().items[0].kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("transaction status")
    ));
}

#[test]
fn lazy_logical_guards_and_affine_shift_detail_match_cpu_contract() {
    let mut and_graph = Graph::new();
    let mask = and_graph.input_dtype("mask", [2], DType::Bool);
    let lhs = and_graph.input_dtype("lhs", [2], DType::I32);
    let divisor = and_graph.input_dtype("divisor", [2], DType::I32);
    let zero =
        and_graph.constant(TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap());
    let quotient = and_graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let positive = and_graph.gt(quotient, zero).unwrap();
    let output = and_graph.logical_and(mask, positive).unwrap();
    let inputs = HashMap::from([
        (
            "mask".into(),
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(false), Scalar::Bool(true)])
                .unwrap(),
        ),
        ("lhs".into(), ints(&[4, 8])),
        ("divisor".into(), ints(&[0, 2])),
    ]);
    let (actual, _) = execute_mock(&and_graph, output, &inputs);
    assert_eq!(actual.to_le_bytes().unwrap(), [0, 1]);

    let mut or_graph = Graph::new();
    let mask = or_graph.input_dtype("mask", [2], DType::Bool);
    let lhs = or_graph.input_dtype("lhs", [2], DType::I32);
    let count = or_graph.input_dtype("count", [2], DType::I32);
    let zero =
        or_graph.constant(TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap());
    let shifted = or_graph.binary(BinaryOp::Shl, lhs, count).unwrap();
    let positive = or_graph.gt(shifted, zero).unwrap();
    let output = or_graph.logical_or(mask, positive).unwrap();
    let inputs = HashMap::from([
        (
            "mask".into(),
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(true), Scalar::Bool(false)])
                .unwrap(),
        ),
        ("lhs".into(), ints(&[4, 8])),
        ("count".into(), ints(&[99, 1])),
    ]);
    let (actual, _) = execute_mock(&or_graph, output, &inputs);
    assert_eq!(actual.to_le_bytes().unwrap(), [1, 1]);

    let mut view_graph = Graph::new();
    let lhs = view_graph.input_dtype("lhs", [2, 2], DType::I32);
    let rhs_storage = view_graph.input_dtype("rhs", [2, 4], DType::I32);
    let rhs = view_graph.shrink(rhs_storage, [(0, 2), (1, 3)]).unwrap();
    let output = view_graph.binary(BinaryOp::Shl, lhs, rhs).unwrap();
    let rendered = WgslRenderer::new(2, capabilities())
        .unwrap()
        .render(&schedule(&view_graph, output).unwrap().items[0].kernel)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock);
    let buffers = allocate_rendered(
        &device,
        &queue,
        &rendered,
        &BTreeMap::from([
            (lhs.index() as u64, ints(&[1, 2, 3, 4])),
            (
                rhs_storage.index() as u64,
                TensorData::from_scalars(
                    [2, 4],
                    DType::I32,
                    [9, 1, 2, 9, 9, -1, 3, 9].into_iter().map(Scalar::I),
                )
                .unwrap(),
            ),
        ]),
    );
    let refs = buffers.iter().collect::<Vec<_>>();
    let output_buffer = &buffers[rendered.transaction.as_ref().unwrap().output_abi_index];
    queue.write(output_buffer, 0, &[0x77; 16]).unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(WebGpuError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 2,
            count: Some(-1),
            bits: 32,
        })
    ));
    let mut unchanged = [0; 16];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x77; 16]);
}

#[test]
fn checked_transfers_retention_owner_and_generation_contracts_hold() {
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let bools = device.allocate_typed(5, DType::Bool).unwrap();
    assert_eq!(bools.len(), 5);
    assert_eq!(bools.physical_len(), 8);
    assert_eq!(bools.generation(), 1);
    queue.write(&bools, 0, &[1, 0, 1, 0, 1]).unwrap();
    let mut actual = [0; 5];
    queue.read(&bools, 0, &mut actual).unwrap();
    assert_eq!(actual, [1, 0, 1, 0, 1]);
    assert!(matches!(
        queue.copy(&bools, &bools, 0, 0, 5),
        Err(WebGpuError::InvalidArgument(_))
    ));

    let src = device.allocate_typed(4, DType::F32).unwrap();
    let dst = device.allocate_typed(4, DType::F32).unwrap();
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    queue.write(&src, 0, &bytes).unwrap();
    assert!(matches!(
        queue.copy(&src, &src, 0, 0, bytes.len()),
        Err(WebGpuError::InvalidArgument(reason)) if reason.contains("same buffer")
    ));
    let command = queue.copy(&src, &dst, 0, 0, bytes.len()).unwrap().unwrap();
    assert!(!command.query().unwrap());
    drop(src);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_release"))
            .count(),
        0
    );
    command.collect().unwrap();
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_release"))
            .count(),
        1
    );
    let mut copied = vec![0; bytes.len()];
    queue.read(&dst, 0, &mut copied).unwrap();
    assert_eq!(copied, bytes);

    let other_mock = Arc::new(MockDispatch::default());
    let (other_device, other_queue) = setup(other_mock);
    let other = other_device.allocate_typed(4, DType::F32).unwrap();
    assert!(matches!(
        other_queue.copy(&dst, &other, 0, 0, 16),
        Err(WebGpuError::OwnerMismatch)
    ));
    assert!(matches!(
        queue.read(&dst, 16, &mut [0; 1]),
        Err(WebGpuError::Bounds)
    ));
}

#[test]
fn raii_children_retain_parents_and_release_in_dependency_order() {
    let mock = Arc::new(MockDispatch::default());
    let runtime = WebGpuRuntime::from_dispatch(mock.clone());
    let instance = runtime.create_instance().unwrap();
    let mut adapters = instance.adapters().unwrap();
    let adapter = adapters.remove(0);
    drop(adapters);
    let device = adapter.request_device().unwrap();
    let queue = device.create_queue().unwrap();
    let buffer = device.allocate_typed(1, DType::F32).unwrap();
    drop(adapter);
    drop(instance);
    drop(device);
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("device_release"))
    );
    drop(buffer);
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("device_release"))
    );
    drop(queue);
    let calls = mock.calls();
    let buffer_release = calls
        .iter()
        .position(|call| call.starts_with("buffer_release"))
        .unwrap();
    let queue_release = calls
        .iter()
        .position(|call| call.starts_with("queue_release"))
        .unwrap();
    let device_release = calls
        .iter()
        .position(|call| call.starts_with("device_release"))
        .unwrap();
    let selected_adapter_release = calls
        .iter()
        .position(|call| call == "adapter_release:2")
        .unwrap();
    let instance_release = calls
        .iter()
        .position(|call| call == "instance_release:1")
        .unwrap();
    assert!(buffer_release < queue_release);
    assert!(queue_release < device_release);
    assert!(device_release < selected_adapter_release);
    assert!(selected_adapter_release < instance_release);
}

#[test]
fn discovery_resource_build_pipeline_launch_query_wait_failures_are_distinct() {
    let mock = Arc::new(MockDispatch::default());
    mock.state.lock().unwrap().failures.instance = Some("instance");
    assert!(matches!(
        WebGpuRuntime::from_dispatch(mock.clone()).create_instance(),
        Err(WebGpuError::Driver {
            operation: "instance_create",
            ..
        })
    ));
    mock.clear_failures();
    let runtime = WebGpuRuntime::from_dispatch(mock.clone());
    let instance = runtime.create_instance().unwrap();
    mock.state.lock().unwrap().failures.adapters = Some("enumerate");
    assert!(matches!(
        instance.adapters(),
        Err(WebGpuError::Driver {
            operation: "adapters",
            ..
        })
    ));
    mock.clear_failures();
    let adapter = instance.adapters().unwrap().remove(0);
    mock.state.lock().unwrap().failures.device = Some("request");
    assert!(matches!(
        adapter.request_device(),
        Err(WebGpuError::Driver {
            operation: "device_create",
            ..
        })
    ));
    mock.clear_failures();
    let device = adapter.request_device().unwrap();
    mock.state.lock().unwrap().failures.queue = Some("queue");
    assert!(matches!(
        device.create_queue(),
        Err(WebGpuError::Driver {
            operation: "queue_create",
            ..
        })
    ));
    mock.clear_failures();
    let queue = device.create_queue().unwrap();
    mock.state.lock().unwrap().failures.buffer = Some("oom");
    assert!(matches!(
        device.allocate(4),
        Err(WebGpuError::Driver {
            operation: "buffer_create",
            ..
        })
    ));
    mock.clear_failures();
    let src = device.allocate_typed(1, DType::F32).unwrap();
    let dst = device.allocate_typed(1, DType::F32).unwrap();
    mock.state.lock().unwrap().failures.write = Some("upload");
    assert!(matches!(
        queue.write(&src, 0, &1.0f32.to_le_bytes()),
        Err(WebGpuError::Driver {
            operation: "write",
            ..
        })
    ));
    mock.state.lock().unwrap().failures.copy = Some("encode copy");
    assert!(matches!(
        queue.copy(&src, &dst, 0, 0, 4),
        Err(WebGpuError::Driver {
            operation: "copy",
            ..
        })
    ));

    let mut graph = Graph::new();
    let input = graph.input("x", [1]);
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.add(input, one).unwrap();
    let rendered = WgslRenderer::new(1, capabilities())
        .unwrap()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    mock.state.lock().unwrap().failures.build = Some("line 7: bad WGSL".into());
    assert!(
        matches!(device.compile(&rendered), Err(WebGpuError::Build { diagnostic }) if diagnostic.contains("bad WGSL"))
    );
    mock.state.lock().unwrap().failures.build = Some("x".repeat(70_000));
    assert!(matches!(
        device.compile(&rendered),
        Err(WebGpuError::Build { diagnostic }) if diagnostic.len() < 66_000 && diagnostic.ends_with("[truncated]")
    ));
    let shader = device.compile(&rendered).unwrap();
    mock.state.lock().unwrap().failures.pipeline = Some("layout");
    assert!(matches!(
        shader.create_pipeline(),
        Err(WebGpuError::Driver {
            operation: "pipeline_create",
            ..
        })
    ));
    let pipeline = shader.create_pipeline().unwrap();
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let untyped = device
        .allocate(rendered.buffers[0].logical_bytes().unwrap())
        .unwrap();
    let mut untyped_refs = buffers.iter().collect::<Vec<_>>();
    untyped_refs[0] = &untyped;
    assert!(matches!(
        pipeline.launch(&queue, &untyped_refs),
        Err(WebGpuError::InvalidBinding(reason)) if reason.contains("requires typed")
    ));
    queue.write(&buffers[0], 0, &2.0f32.to_le_bytes()).unwrap();
    let refs = buffers.iter().collect::<Vec<_>>();
    mock.state.lock().unwrap().failures.launch = Some("dispatch");
    assert!(matches!(
        pipeline.launch(&queue, &refs),
        Err(WebGpuError::Driver {
            operation: "launch",
            ..
        })
    ));
    let command = pipeline.launch(&queue, &refs).unwrap().unwrap();
    mock.state.lock().unwrap().failures.query = Some("poll");
    assert!(matches!(
        command.query(),
        Err(WebGpuError::Driver {
            operation: "query",
            ..
        })
    ));
    mock.state.lock().unwrap().failures.wait = Some("device lost");
    assert!(matches!(
        command.collect(),
        Err(WebGpuError::Driver {
            operation: "wait",
            ..
        })
    ));
    let stale = pipeline.launch(&queue, &refs).unwrap().unwrap();
    assert_eq!(
        buffers
            .last()
            .unwrap()
            .replace_generation_for_test()
            .unwrap(),
        2
    );
    assert!(matches!(
        stale.collect(),
        Err(WebGpuError::StaleGeneration {
            expected: 1,
            actual: 2
        })
    ));
    mock.state.lock().unwrap().failures.read = Some("map");
    assert!(matches!(
        queue.read(buffers.last().unwrap(), 0, &mut [0; 4]),
        Err(WebGpuError::Driver {
            operation: "read",
            ..
        })
    ));
}

#[test]
fn zero_extent_uses_no_physical_buffer_or_submission() {
    let mut graph = Graph::new();
    let input = graph.input("x", [0]);
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.add(input, one).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let rendered = WgslRenderer::new(1, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    assert!(buffers.last().unwrap().is_empty());
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(
        pipeline
            .launch(&queue, &buffers.iter().collect::<Vec<_>>())
            .unwrap()
            .is_none()
    );
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch")));
}

#[test]
#[ignore = "requires a pinned wgpu-native/Dawn C ABI and an available adapter"]
fn live_webgpu_discovery_compile_transfer_launch_query_collect_smoke() {
    let runtime = match WebGpuRuntime::load() {
        Ok(runtime) => runtime,
        Err(
            error @ (WebGpuError::LibraryUnavailable { .. }
            | WebGpuError::NativeAbiUnsupported { .. }),
        ) => {
            eprintln!("live WebGPU unavailable: {error}");
            return;
        }
        Err(error) => panic!("unexpected WebGPU loader error: {error}"),
    };
    let instance = runtime.create_instance().unwrap();
    let adapter = instance.adapters().unwrap().remove(0);
    let device = adapter.request_device().unwrap();
    let queue = device.create_queue().unwrap();
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [4]);
    let rhs = graph.input("rhs", [4]);
    let output = graph.add(lhs, rhs).unwrap();
    let rendered = WgslRenderer::new(4, device.info().capabilities.clone())
        .unwrap()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    queue
        .write(
            &buffers[0],
            0,
            &TensorData::new([4], vec![1., 2., 3., 4.])
                .unwrap()
                .to_le_bytes()
                .unwrap(),
        )
        .unwrap();
    queue
        .write(
            &buffers[1],
            0,
            &TensorData::new([4], vec![5., 6., 7., 8.])
                .unwrap()
                .to_le_bytes()
                .unwrap(),
        )
        .unwrap();
    let mirror = device.allocate_typed(4, DType::F32).unwrap();
    queue
        .copy(&buffers[0], &mirror, 0, 0, 16)
        .unwrap()
        .unwrap()
        .collect()
        .unwrap();
    let command = device
        .cache()
        .load(&rendered)
        .unwrap()
        .launch(&queue, &buffers.iter().collect::<Vec<_>>())
        .unwrap()
        .unwrap();
    let _ = command.query().unwrap();
    command.collect().unwrap();
    let mut bytes = [0; 16];
    queue.read(buffers.last().unwrap(), 0, &mut bytes).unwrap();
    assert_eq!(
        TensorData::from_le_bytes([4], DType::F32, &bytes)
            .unwrap()
            .values(),
        &[6., 8., 10., 12.]
    );
}
