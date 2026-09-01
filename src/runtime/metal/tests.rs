use super::renderer::{
    METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION, METAL_RAW_COPY_RENDERER_VERSION,
    METAL_STATIC_POSITION_RENDERER_VERSION,
};
use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::runtime::scalar_lane::emit_scalar_lane;
use crate::{
    Backend, BinaryOp, BufferRole, CapturedMixedBatch, CapturedReplayExecutor, CompareOp,
    CpuBackend, CpuSession, DType, EffectBatchStep, EffectRuntime, Graph, IndexValue,
    KernelBindings, KernelBufferDesc, LaneInstruction, MovementKernelKind, MovementValue, NodeId,
    Operation, ReduceKind, Scalar, Shape, Slice, Storage, TensorData, TypedValue, UType, schedule,
};
use dispatch::{
    CopyRegion, Dispatch, KernelSemantics, LaunchGeometry, RawBuffer, RawCommand, RawDevice,
    RawLibrary, RawPipeline, RawQueue,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    rc::Rc,
    sync::{Arc, Mutex},
};

#[test]
fn portable_f32_matmul_metal_renders_shared_geometry_and_executes_zero_k() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 2, 3], DType::F32);
    let rhs = graph.input_dtype("rhs", [1, 3, 2], DType::F32);
    let output = graph.matmul(lhs, rhs).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let rendered = renderer.render(&scheduled.items[0].kernel).unwrap();
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION)
    );
    assert!(rendered.source.contains("#pragma clang fp contract(off)"));
    assert!(rendered.source.contains("const float rg_product"));
    assert!(rendered.source.contains("rg_acc = rg_acc + rg_product"));

    let mut aliased = Graph::new();
    let input = aliased.input_dtype("input", [2, 2], DType::F32);
    let output = aliased.matmul(input, input).unwrap();
    let item = schedule(&aliased, output).unwrap().items.pop().unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert_eq!(rendered.buffers.len(), 2);

    let mut cancellation = Graph::new();
    let lhs = cancellation.input_dtype("lhs", [3], DType::F32);
    let rhs = cancellation.input_dtype("rhs", [3], DType::F32);
    let output = cancellation.matmul(lhs, rhs).unwrap();
    let (actual, _) = execute_mock(
        &cancellation,
        output,
        &HashMap::from([
            (
                "lhs".into(),
                TensorData::from_storage([3], Storage::F32(vec![16_777_216.0, 1.0, -16_777_216.0]))
                    .unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_storage([3], Storage::F32(vec![1.0; 3])).unwrap(),
            ),
        ]),
    );
    assert_eq!(actual.storage(), &Storage::F32(vec![0.0]));

    let mut zero = Graph::new();
    let lhs = zero.input_dtype("lhs", [2, 0], DType::F32);
    let rhs = zero.input_dtype("rhs", [0, 3], DType::F32);
    let output = zero.matmul(lhs, rhs).unwrap();
    let items = schedule(&zero, output).unwrap().items;
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let prefix = PreparedMetalPrefix::prepare(device, &items, renderer.clone()).unwrap();
    let mut values = BTreeMap::from([
        (
            lhs.index() as u64,
            TensorData::from_storage([2, 0], Storage::F32(Vec::new())).unwrap(),
        ),
        (
            rhs.index() as u64,
            TensorData::from_storage([0, 3], Storage::F32(Vec::new())).unwrap(),
        ),
    ]);
    prefix.execute(&mut values).unwrap();
    assert_eq!(
        values[&(output.index() as u64)].storage(),
        &Storage::F32(vec![0.0; 6])
    );
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_create:") && call.ends_with(":4"))
            .count(),
        2
    );
    drop(prefix);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_release:"))
            .count(),
        3
    );

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 4], DType::F32);
    let rhs = empty.input_dtype("rhs", [4, 3], DType::F32);
    let output = empty.matmul(lhs, rhs).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let prefix =
        PreparedMetalPrefix::prepare(device, &schedule(&empty, output).unwrap().items, renderer)
            .unwrap();
    let mut values = BTreeMap::from([
        (
            lhs.index() as u64,
            TensorData::from_storage([0, 4], Storage::F32(Vec::new())).unwrap(),
        ),
        (
            rhs.index() as u64,
            TensorData::from_storage([4, 3], Storage::F32(vec![1.0; 12])).unwrap(),
        ),
    ]);
    prefix.execute(&mut values).unwrap();
    assert_eq!(
        values[&(output.index() as u64)].storage(),
        &Storage::F32(Vec::new())
    );
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch:")));
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("buffer_create:"))
    );
}

#[test]
fn prepared_metal_prefix_reuses_disjoint_exact_temporary_slots() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let first_value = graph.square(input).unwrap();
    let first = graph.contiguous(first_value).unwrap();
    let second_value = graph.square(first).unwrap();
    let second = graph.contiguous(second_value).unwrap();
    let third_value = graph.square(second).unwrap();
    let third = graph.contiguous(third_value).unwrap();
    let output_value = graph.square(third).unwrap();
    let output = graph.contiguous(output_value).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    assert_eq!(scheduled.items.len(), 4);
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let plan =
        MetalPrefixPlan::plan_for_outputs(&scheduled.items, &[output.index() as u64], renderer)
            .unwrap();
    let prefix = PreparedMetalPrefix::from_plan(device, plan).unwrap();
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_create:"))
            .count(),
        4
    );
    let mut values = BTreeMap::from([(
        input.index() as u64,
        TensorData::from_storage([2], Storage::F32(vec![1.0, -1.0])).unwrap(),
    )]);
    prefix.execute(&mut values).unwrap();
    assert_eq!(
        values[&(output.index() as u64)].storage(),
        &Storage::F32(vec![1.0, 1.0])
    );
    drop(prefix);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("buffer_release:"))
            .count(),
        4
    );
}

#[test]
fn reduction_epilogue_renders_one_metal_kernel() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", Shape::from([2, 3]), DType::F32);
    let reduced = graph.sum(input, 1).unwrap();
    let output = graph.relu(reduced).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&scheduled.items[0].kernel)
        .unwrap();
    assert!(rendered.source.contains("rg_acc"));
    assert!(rendered.source.contains(" ? "));
    assert!(
        rendered
            .buffers
            .iter()
            .all(|buffer| buffer.id != reduced.index() as u64)
    );
}

#[derive(Default)]
struct Failures {
    buffer_create: Option<&'static str>,
    buffer_create_after: Option<(usize, &'static str)>,
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

#[test]
fn mixed_batch_metal_mock_is_prepared_atomic_and_retryable() {
    let (first, first_next) = crate::engine::mixed_batch::test_support::pure_add_capture(700);
    let (second, second_next) = crate::engine::mixed_batch::test_support::pure_add_capture(700);
    let batch = CapturedMixedBatch::new(vec![first.clone(), second]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let inputs = vec![
        BTreeMap::from([
            (
                "x".into(),
                TensorData::from_storage([2], Storage::F32(vec![1.0, 2.0])).unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_storage([2], Storage::F32(vec![3.0, 4.0])).unwrap(),
            ),
        ]),
        BTreeMap::from([
            (
                "x".into(),
                TensorData::from_storage([2], Storage::F32(vec![5.0, 6.0])).unwrap(),
            ),
            (
                "y".into(),
                TensorData::from_storage([2], Storage::F32(vec![7.0, 8.0])).unwrap(),
            ),
        ]),
    ];
    let mut runtime = EffectRuntime::new();
    runtime
        .register(
            700,
            TensorData::from_storage([2], Storage::F32(vec![0.0, 0.0])).unwrap(),
        )
        .unwrap();
    runtime
        .register(
            2,
            TensorData::from_storage([2], Storage::F32(vec![0.0, 0.0])).unwrap(),
        )
        .unwrap();
    assert!(
        batch
            .replay_metal(
                &mut runtime,
                &inputs,
                device.clone(),
                renderer.clone(),
                Some(EffectBatchStep { entry: 1, step: 0 })
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
        &Storage::F32(vec![0.0, 0.0])
    );
    let result = batch
        .replay_metal(&mut runtime, &inputs, device, renderer, None)
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
        &Storage::F32(vec![12.0, 14.0])
    );
    assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn mixed_batch_metal_signed_state_input_matches_interpreter_and_native() {
    let (capture, end) = crate::engine::mixed_batch::test_support::signed_state_add_capture();
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let supplied = BTreeMap::from([(
        "bias".into(),
        crate::engine::mixed_batch::test_support::data(vec![10., 20., 30., 40.]),
    )]);
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();

    let mut metal = EffectRuntime::new();
    metal
        .register(
            90,
            crate::engine::mixed_batch::test_support::data(vec![1., 2., 3., 4.]),
        )
        .unwrap();
    metal
        .register(
            2,
            crate::engine::mixed_batch::test_support::data(vec![0.; 4]),
        )
        .unwrap();
    batch
        .replay_metal(
            &mut metal,
            std::slice::from_ref(&supplied),
            device,
            renderer,
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
    assert_eq!(metal.snapshot(&end).unwrap().tensor().storage(), expected);
    assert_eq!(
        metal.snapshot(&end).unwrap().tensor().storage(),
        interpreter.snapshot(&end).unwrap().tensor().storage()
    );
    assert_eq!(
        metal.snapshot(&end).unwrap().tensor().storage(),
        native.snapshot(&end).unwrap().tensor().storage()
    );
    assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn mixed_batch_metal_rejects_later_unsupported_before_submission() {
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
            .replay_metal(
                &mut runtime,
                &[
                    crate::engine::mixed_batch::test_support::add_inputs(),
                    crate::engine::mixed_batch::test_support::add_inputs(),
                ],
                device,
                MetalRenderer::new(8, capabilities()).unwrap(),
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
fn mixed_batch_metal_empty_prefix_skips_submission_and_commits() {
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
        .replay_metal(
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
            MetalRenderer::new(8, capabilities()).unwrap(),
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
fn mixed_batch_metal_reuses_prepared_keys_for_equivalent_replays() {
    let (capture, next) = crate::engine::mixed_batch::test_support::pure_add_capture(94);
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
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
        .replay_metal(&mut first, &inputs, device.clone(), renderer.clone(), None)
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
        .replay_metal(&mut second, &inputs, device, renderer, None)
        .unwrap();
    assert_eq!(first_result.trace.identity, batch.identity());
    assert_eq!(first_result.trace, second_result.trace);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("pipeline_create:"))
            .count(),
        compiled,
        "equivalent logical batches reuse the same device-scoped pipeline"
    );
    assert_eq!(
        second.snapshot(&next).unwrap().tensor().storage(),
        &Storage::F32(vec![4., 6.])
    );
}

#[test]
fn mixed_batch_metal_launch_failure_preserves_state_and_reuses_preparation() {
    let (capture, next) = crate::engine::mixed_batch::test_support::pure_add_capture(95);
    let batch = CapturedMixedBatch::new(vec![capture]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
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
            .replay_metal(
                &mut runtime,
                &[crate::engine::mixed_batch::test_support::add_inputs()],
                device.clone(),
                renderer.clone(),
                None,
            )
            .is_err()
    );
    assert!(
        mock.calls()
            .iter()
            .any(|call| call.starts_with("pipeline_create:"))
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
        .replay_metal(
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

#[derive(Default)]
struct State {
    calls: Vec<String>,
    owners: BTreeSet<u64>,
    next_buffer: usize,
    next_library: usize,
    next_pipeline: usize,
    next_command: usize,
    buffers: BTreeMap<(u64, usize), Vec<u8>>,
    libraries: BTreeMap<(u64, usize), String>,
    semantics: BTreeMap<(u64, usize), Arc<KernelSemantics>>,
    commands: BTreeMap<(u64, usize), bool>,
    failures: Failures,
    fault_order: Vec<usize>,
}

#[derive(Default)]
struct MockDispatch {
    state: Mutex<State>,
}

impl MockDispatch {
    fn calls(&self) -> Vec<String> {
        self.state.lock().unwrap().calls.clone()
    }

    fn clear_calls(&self) {
        self.state.lock().unwrap().calls.clear();
    }

    fn clear_failures(&self) {
        self.state.lock().unwrap().failures = Failures::default();
    }

    fn command(state: &mut State, owner: u64) -> RawCommand {
        state.next_command += 1;
        let raw = RawCommand(500 + state.next_command);
        state.commands.insert((owner, raw.0), false);
        raw
    }

    fn failure(operation: &'static str, detail: &'static str) -> MetalError {
        MetalError::Driver {
            operation,
            detail: detail.into(),
        }
    }
}

impl Dispatch for MockDispatch {
    fn devices(&self) -> Result<Vec<RawDevice>, MetalError> {
        self.state.lock().unwrap().calls.push("devices".into());
        Ok(vec![RawDevice(2), RawDevice(1)])
    }

    fn device_info(&self, device: RawDevice) -> Result<MetalDeviceInfo, MetalError> {
        Ok(MetalDeviceInfo {
            name: format!("Mock Metal {}", device.0),
            registry_id: device.0 as u64,
            capabilities: MetalCapabilities {
                max_buffer_length: 1 << 20,
                unified_memory: true,
                family: "MockApple9".into(),
            },
        })
    }

    fn device_release(&self, device: RawDevice) {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("device_release:{}", device.0));
    }

    fn queue_create(&self, _device: RawDevice, owner: u64) -> Result<RawQueue, MetalError> {
        let mut state = self.state.lock().unwrap();
        state.owners.insert(owner);
        state.calls.push(format!("queue_create:{owner}"));
        Ok(RawQueue(10))
    }

    fn queue_release(&self, _queue: RawQueue, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.calls.push(format!("queue_release:{owner}"));
        state.owners.remove(&owner);
    }

    fn buffer_create(
        &self,
        _device: RawDevice,
        bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.buffer_create.take() {
            return Err(Self::failure("buffer_create", detail));
        }
        if let Some((remaining, detail)) = state.failures.buffer_create_after.as_mut() {
            if *remaining == 0 {
                let detail = *detail;
                state.failures.buffer_create_after = None;
                return Err(Self::failure("buffer_create", detail));
            }
            *remaining -= 1;
        }
        state.next_buffer += 1;
        let raw = RawBuffer(100 + state.next_buffer);
        state.buffers.insert((owner, raw.0), vec![0; bytes]);
        state
            .calls
            .push(format!("buffer_create:{owner}:{}:{bytes}", raw.0));
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
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.write.take() {
            return Err(Self::failure("write", detail));
        }
        let storage = state
            .buffers
            .get_mut(&(owner, buffer.0))
            .ok_or(MetalError::OwnerMismatch)?;
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
    ) -> Result<(), MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.read.take() {
            return Err(Self::failure("read", detail));
        }
        if let Some((remaining, detail)) = state.failures.read_after.as_mut() {
            if *remaining == 0 {
                let detail = *detail;
                state.failures.read_after = None;
                return Err(Self::failure("read", detail));
            }
            *remaining -= 1;
        }
        let storage = state
            .buffers
            .get(&(owner, buffer.0))
            .ok_or(MetalError::OwnerMismatch)?;
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
    ) -> Result<RawCommand, MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.copy.take() {
            return Err(Self::failure("copy", detail));
        }
        let value = state
            .buffers
            .get(&(owner, src.0))
            .ok_or(MetalError::OwnerMismatch)?[region.src_offset..region.src_offset + region.bytes]
            .to_vec();
        state
            .buffers
            .get_mut(&(owner, dst.0))
            .ok_or(MetalError::OwnerMismatch)?[region.dst_offset..region.dst_offset + region.bytes]
            .copy_from_slice(&value);
        state.calls.push(format!("copy:{owner}"));
        Ok(Self::command(&mut state, owner))
    }

    fn library_compile(
        &self,
        _device: RawDevice,
        source: &str,
        owner: u64,
    ) -> Result<RawLibrary, MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(diagnostic) = state.failures.build.take() {
            return Err(MetalError::Build { diagnostic });
        }
        state.next_library += 1;
        let raw = RawLibrary(200 + state.next_library);
        state.libraries.insert((owner, raw.0), source.into());
        state.calls.push(format!("library_compile:{owner}"));
        Ok(raw)
    }

    fn library_release(&self, library: RawLibrary, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.libraries.remove(&(owner, library.0));
        state.calls.push(format!("library_release:{owner}"));
    }

    fn pipeline_create(
        &self,
        _device: RawDevice,
        _library: RawLibrary,
        _entry: &str,
        owner: u64,
    ) -> Result<(RawPipeline, usize), MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.pipeline.take() {
            return Err(Self::failure("pipeline_create", detail));
        }
        state.next_pipeline += 1;
        let raw = RawPipeline(300 + state.next_pipeline);
        state.calls.push(format!("pipeline_create:{owner}"));
        Ok((raw, 128))
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
    ) -> Result<RawCommand, MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.launch.take() {
            return Err(Self::failure("launch", detail));
        }
        let semantics = state
            .semantics
            .get(&(owner, pipeline.0))
            .cloned()
            .ok_or_else(|| MetalError::InvalidBinding("mock semantics absent".into()))?;
        let transaction = semantics.transaction.as_ref();
        let expected_buffers = semantics.buffers.len() + usize::from(transaction.is_some());
        if geometry.extent as usize != semantics.extent
            || geometry.extent_index != semantics.buffers.len()
            || geometry.local == 0
            || geometry.global < semantics.extent
            || !geometry.global.is_multiple_of(geometry.local)
            || buffers.len() != expected_buffers
        {
            return Err(MetalError::InvalidArgument("invalid mock launch geometry"));
        }
        let mut bindings = KernelBindings::default();
        let mut output = None;
        for (position, (raw, abi)) in buffers.iter().zip(&semantics.buffers).enumerate() {
            let logical = abi
                .elements
                .checked_mul(abi.dtype.itemsize())
                .ok_or(MetalError::Overflow)?;
            let physical = if semantics.extent != 0 && logical == 0 {
                DType::F32.itemsize()
            } else {
                logical
            };
            let bytes = state
                .buffers
                .get(&(owner, raw.0))
                .ok_or(MetalError::OwnerMismatch)?;
            if bytes.len() != physical {
                return Err(MetalError::InvalidBinding(format!(
                    "mock buffer {position} length mismatch"
                )));
            }
            let value =
                TensorData::from_le_bytes(abi.source_shape.clone(), abi.dtype, &bytes[..logical])
                    .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
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
            .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
            bindings
                .insert(&desc, value)
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
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
                        .ok_or(MetalError::OwnerMismatch)?;
                    Ok((abi.clone(), bytes.clone()))
                })
                .collect::<Result<Vec<_>, MetalError>>()?;
            let order = if state.fault_order.is_empty() {
                (0..semantics.extent).collect::<Vec<_>>()
            } else {
                state.fault_order.clone()
            };
            let mut status = transaction::CLEAN_STATUS;
            for logical in order {
                if logical >= semantics.extent {
                    return Err(MetalError::InvalidBinding(
                        "mock fault order exceeds extent".into(),
                    ));
                }
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
                                MetalError::InvalidBinding("mock transaction buffer absent".into())
                            })?;
                        if abi.dtype != dtype {
                            return Err(MetalError::InvalidBinding(
                                "mock transaction dtype mismatch".into(),
                            ));
                        }
                        let offset = transaction::logical_offset(arg, logical)?;
                        let start = offset
                            .checked_mul(dtype.itemsize())
                            .ok_or(MetalError::Overflow)?;
                        decode_mock_scalar(dtype, &bytes[start..start + dtype.itemsize()])
                    })?
                {
                    status = status.min(transaction.key(logical, id)?);
                }
            }
            let status_raw = buffers
                .last()
                .ok_or_else(|| MetalError::InvalidBinding("mock status absent".into()))?;
            state
                .buffers
                .get_mut(&(owner, status_raw.0))
                .ok_or(MetalError::OwnerMismatch)?
                .copy_from_slice(&status.to_le_bytes());
            if status != transaction::CLEAN_STATUS {
                state.calls.push(format!(
                    "launch:{owner}:{}:{}",
                    geometry.global, geometry.local
                ));
                return Ok(Self::command(&mut state, owner));
            }
        }
        // This is RustGrad's retained semantic artifact, not CpuBackend or
        // native Metal. Captured random stays graph-free and immutable.
        let result = match semantics.program.as_ref() {
            dispatch::KernelSemanticProgram::UOp(program) => {
                execute_lowered_elementwise(program, &bindings)
                    .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
            }
            dispatch::KernelSemanticProgram::Random(plan) => plan
                .execute()
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?,
        }
        .to_le_bytes()
        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
        let (output, expected) =
            output.ok_or_else(|| MetalError::InvalidBinding("mock output absent".into()))?;
        if result.len() != expected {
            return Err(MetalError::InvalidBinding(
                "mock semantic output length mismatch".into(),
            ));
        }
        state
            .buffers
            .get_mut(&(owner, output.0))
            .ok_or(MetalError::OwnerMismatch)?
            .copy_from_slice(&result);
        state.calls.push(format!(
            "launch:{owner}:{}:{}",
            geometry.global, geometry.local
        ));
        Ok(Self::command(&mut state, owner))
    }

    fn command_query(&self, command: RawCommand, owner: u64) -> Result<bool, MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.query.take() {
            return Err(Self::failure("query", detail));
        }
        state
            .commands
            .get(&(owner, command.0))
            .copied()
            .ok_or(MetalError::OwnerMismatch)
    }

    fn command_wait(&self, command: RawCommand, owner: u64) -> Result<(), MetalError> {
        let mut state = self.state.lock().unwrap();
        if let Some(detail) = state.failures.wait.take() {
            return Err(Self::failure("wait", detail));
        }
        *state
            .commands
            .get_mut(&(owner, command.0))
            .ok_or(MetalError::OwnerMismatch)? = true;
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
    ) -> Result<(), MetalError> {
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

fn decode_mock_scalar(dtype: DType, bytes: &[u8]) -> Result<Scalar, MetalError> {
    Ok(match dtype {
        DType::Bool => Scalar::Bool(bytes == [1]),
        DType::I32 => {
            Scalar::I(i32::from_le_bytes(bytes.try_into().map_err(|_| MetalError::Bounds)?) as i64)
        }
        DType::U32 => {
            Scalar::U(u32::from_le_bytes(bytes.try_into().map_err(|_| MetalError::Bounds)?) as u64)
        }
        _ => return Err(MetalError::InvalidBinding("mock detail dtype".into())),
    })
}

fn capabilities() -> MetalCapabilities {
    MetalCapabilities {
        max_buffer_length: 1 << 20,
        unified_memory: true,
        family: "MockApple9".into(),
    }
}

fn setup(mock: Arc<MockDispatch>) -> (MetalDevice, MetalCommandQueue) {
    let runtime = MetalRuntime::from_dispatch(mock);
    let mut devices = runtime.devices().unwrap();
    assert_eq!(devices[0].info().registry_id, 1);
    let device = devices.remove(0);
    let queue = device.create_queue().unwrap();
    (device, queue)
}

fn test_device(mock: Arc<MockDispatch>) -> MetalDevice {
    let runtime = MetalRuntime::from_dispatch(mock);
    runtime.devices().unwrap().remove(0)
}

#[test]
fn typed_discovery_reports_devices_without_queue_creation() {
    let mock = Arc::new(MockDispatch::default());
    let runtime = MetalRuntime::from_dispatch(mock.clone());
    let MetalDiscovery::Devices(devices) = runtime.discover().unwrap() else {
        panic!("mock must expose deterministic devices");
    };
    assert_eq!(devices.len(), 2);
    assert!(mock.calls().contains(&"devices".into()));
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("queue_create:"))
    );
}

#[test]
fn cpu_session_metal_public_path_matches_cpu_and_reuses_owner_cache() {
    let mut session = CpuSession::new();
    let input = session.variable([2], [1.0, 2.0]).unwrap();
    let bias = session.tensor([2], [3.0, 4.0]).unwrap();
    let output = session.add(&input, &bias).unwrap();
    let expected = session.realize(&output).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();

    let first = session
        .realize_metal(&output, device.clone(), renderer.clone())
        .unwrap();
    assert_eq!(first.output(), &expected);
    assert!(!first.cache_keys.is_empty());
    assert!(!first.trace.zero_domain_skipped);
    assert_eq!(device.cache().len(), first.cache_keys.len());
    let second = session
        .realize_metal(&output, device.clone(), renderer)
        .unwrap();
    assert_eq!(second.output(), &expected);
    assert_eq!(first.trace, second.trace);
    assert_eq!(device.cache().len(), first.cache_keys.len());
    assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn cpu_session_metal_zero_domain_preflights_without_resources() {
    let mut session = CpuSession::new();
    let input = session.variable([0, 2], Vec::<f32>::new()).unwrap();
    let bias = session.tensor([2], [3.0, 4.0]).unwrap();
    let output = session.add(&input, &bias).unwrap();
    let expected = session.realize(&output).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let result = session
        .realize_metal(
            &output,
            device.clone(),
            MetalRenderer::new(8, capabilities()).unwrap(),
        )
        .unwrap();
    assert_eq!(result.output(), &expected);
    assert!(result.trace.zero_domain_skipped);
    assert!(result.cache_keys.is_empty());
    assert!(result.trace.cache_keys.is_empty());
    assert_eq!(device.cache().len(), 0);
    assert!(mock.calls().is_empty());
}

#[test]
fn cpu_session_metal_unsupported_preflight_has_no_resource_side_effect() {
    let mut session = CpuSession::new();
    let input = session
        .variable_data(TensorData::from_storage([2], Storage::F64(vec![1.0, -2.0])).unwrap())
        .unwrap();
    let weight = session
        .tensor_with_dtype([2, 1], DType::F64, [Scalar::F(1.0), Scalar::F(2.0)])
        .unwrap();
    let output = session.matmul(&input, &weight).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let error = session
        .realize_metal(
            &output,
            device.clone(),
            MetalRenderer::new(8, capabilities()).unwrap(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("Metal preflight"));
    assert_eq!(device.cache().len(), 0);
    assert!(mock.calls().is_empty());
}

#[test]
#[ignore = "requires an Apple Metal device"]
fn live_cpu_session_metal_static_elementwise_smoke() {
    let runtime = MetalRuntime::load().unwrap();
    let device = runtime.devices().unwrap().remove(0);
    let renderer = MetalRenderer::new(64, device.info().capabilities.clone()).unwrap();
    let mut session = CpuSession::new();
    let input = session.variable([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
    let bias = session.tensor([2], [10.0, 20.0]).unwrap();
    let output = session.add(&input, &bias).unwrap();
    let cpu = session.realize(&output).unwrap();
    let metal = session.realize_metal(&output, device, renderer).unwrap();
    assert_eq!(metal.output(), &cpu);
    assert!(!metal.cache_keys.is_empty());
}

fn materialized_values(
    graph: &Graph,
    rendered: &RenderedMetal,
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

fn execute_mock(
    graph: &Graph,
    output: NodeId,
    inputs: &HashMap<String, TensorData>,
) -> (TensorData, Arc<MockDispatch>) {
    let item = schedule(graph, output).unwrap().items.pop().unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let values = materialized_values(graph, &rendered, inputs);
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let buffers = rendered
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
        .collect::<Vec<_>>();
    let cache = device.cache();
    let pipeline = cache.load(&rendered).unwrap();
    assert!(Rc::ptr_eq(&pipeline, &cache.load(&rendered).unwrap()));
    assert_eq!(cache.len(), 1);
    let refs = buffers.iter().collect::<Vec<_>>();
    let completion = if rendered.transaction.is_some() {
        let transaction = pipeline.launch_transactional(&queue, &refs, 8).unwrap();
        assert!(!transaction.query().unwrap());
        transaction.collect().unwrap()
    } else {
        let command = pipeline.launch(&queue, &refs, 8).unwrap().unwrap();
        assert!(!command.query().unwrap());
        command.collect().unwrap()
    };
    assert_eq!(completion.extent, rendered.extent);
    assert_eq!(
        completion.retained_resources,
        rendered.buffers.len() + usize::from(rendered.transaction.is_some()) * 2
    );
    let output_abi = rendered.buffers.last().unwrap();
    let mut bytes = vec![0; output_abi.elements * output_abi.dtype.itemsize()];
    queue.read(buffers.last().unwrap(), 0, &mut bytes).unwrap();
    let result =
        TensorData::from_le_bytes(output_abi.source_shape.clone(), output_abi.dtype, &bytes)
            .unwrap();
    (result, mock)
}

#[test]
fn raw_movement_copy_metal_executes_affine_and_dense_storage_contracts() {
    let mut affine = Graph::new();
    let input = affine.input_dtype("input", [4], DType::F32);
    let viewed = affine
        .stride(
            input,
            [Slice {
                start: None,
                stop: None,
                step: -1,
            }],
        )
        .unwrap();
    let output = affine.contiguous(viewed).unwrap();
    let item = schedule(&affine, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert!(rendered.source.contains(METAL_RAW_COPY_RENDERER_VERSION));
    assert!(
        rendered
            .source
            .contains("rg_axis_0 = (ulong)3ul - rg_axis_0")
    );
    assert!(rendered.source.contains("b1[gid] = b0[rg_source]"));
    assert_eq!(
        MetalRenderer::new(8, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap()
            .cache_key,
        rendered.cache_key
    );
    let raw = [0x8000_0000_u32, 0x7fc1_2345, 0x7f80_0000, 0xff80_0000];
    let value = TensorData::from_storage(
        [4],
        Storage::F32(raw.into_iter().map(f32::from_bits).collect()),
    )
    .unwrap();
    let (actual, calls) = execute_mock(&affine, output, &HashMap::from([("input".into(), value)]));
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        raw.into_iter()
            .rev()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert!(calls.calls().iter().any(|call| call.starts_with("launch:")));

    let mut dense = Graph::new();
    let input = dense.input_dtype("input", [2], DType::I32);
    let producer = dense.square(input).unwrap();
    let output = dense.contiguous(producer).unwrap();
    let item = crate::schedule_many(&dense, &[producer, output])
        .unwrap()
        .items
        .into_iter()
        .find(|item| item.node == output)
        .unwrap();
    assert!(matches!(
        item.kernel.operation(),
        Operation::Movement(MovementValue::Plan(plan))
            if matches!(&plan.kind, MovementKernelKind::Contiguous { input } if input.node == producer)
    ));
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.source.contains("b1[gid] = b0[gid]"));
    let redirected = schedule(&dense, output).unwrap().items.pop().unwrap();
    assert!(matches!(redirected.kernel.operation(), Operation::Sink));
    MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&redirected.kernel)
        .unwrap();
    let (actual, _) = execute_mock(
        &dense,
        output,
        &HashMap::from([(
            "input".into(),
            TensorData::from_storage([2], Storage::I32(vec![3, -4])).unwrap(),
        )]),
    );
    assert_eq!(actual.storage(), &Storage::I32(vec![9, 16]));

    let reshaped = dense.reshape(producer, [2, 1]).unwrap();
    let viewed = dense.expand(reshaped, [2, 3]).unwrap();
    let affine_output = dense.contiguous(viewed).unwrap();
    let affine_item = crate::schedule_many(&dense, &[producer, affine_output])
        .unwrap()
        .items
        .into_iter()
        .find(|item| item.node == affine_output)
        .unwrap();
    assert!(matches!(
        affine_item.kernel.operation(),
        Operation::Movement(MovementValue::Plan(plan))
            if matches!(&plan.kind, MovementKernelKind::AffineCopy { input, .. } if input.node == producer)
    ));
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&affine_item.kernel)
        .unwrap();
    assert_eq!(rendered.buffers[0].elements, 2);
    assert_eq!(rendered.buffers[1].elements, 6);
    assert!(rendered.source.contains("b1[gid] = b0[rg_source]"));
}

#[test]
fn static_positions_metal_zeroes_holes_and_preserves_raw_payloads() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F32);
    let output = graph
        .scatter_positions(input, Shape::from([5]), vec![4], vec![-2])
        .unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert!(
        rendered
            .source
            .contains(METAL_STATIC_POSITION_RENDERER_VERSION)
    );
    assert!(rendered.source.contains("rg_delta_0 % (ulong)2ul"));
    assert!(rendered.source.contains("rg_mapped = false"));
    assert_eq!(rendered.buffers[0].elements, 2);
    assert_eq!(rendered.buffers[1].elements, 5);
    let raw = [0x7fc1_2345_u32, 0x8000_0000];
    let value = TensorData::from_storage(
        [2],
        Storage::F32(raw.into_iter().map(f32::from_bits).collect()),
    )
    .unwrap();
    let (actual, calls) = execute_mock(&graph, output, &HashMap::from([("input".into(), value)]));
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        [0, 0, raw[1], 0, raw[0]]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
    assert!(calls.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn raw_movement_copy_metal_preserves_zero_domain_and_rejects_other_movements() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("empty", [0, 2], DType::Bool);
    let viewed = graph.permute(input, [1, 0]).unwrap();
    let output = graph.contiguous(viewed).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    assert_eq!(rendered.extent, 0);
    assert_eq!(rendered.buffers[0].elements, 0);

    let scalar = graph.input_dtype("scalar", [], DType::U64);
    let producer = graph.detach(scalar).unwrap();
    let output = graph.contiguous(producer).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    assert_eq!(renderer.render(&item.kernel).unwrap().extent, 1);

    let pad_input = graph.input_dtype("pad", [2], DType::U64);
    let padded = graph.pad(pad_input, [(1, 1)], Scalar::U(0)).unwrap();
    let padded_item = schedule(&graph, padded).unwrap().items.pop().unwrap();
    assert!(matches!(
        renderer.render(&padded_item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("only raw AffineCopy and Contiguous")
    ));
}

#[test]
fn signed_affine_flip_lowers_and_mock_matches_cpu_without_native_submission() {
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
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&crate::kernel::lower_graph_elementwise(&graph, output).unwrap())
        .unwrap();
    assert!(rendered.source.contains("* -1l"), "{}", rendered.source);
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

fn ints(values: &[i32]) -> TensorData {
    TensorData::from_scalars(
        [values.len()],
        DType::I32,
        values.iter().map(|&value| Scalar::I(value as i64)),
    )
    .unwrap()
}

fn uints(values: &[u32]) -> TensorData {
    TensorData::from_scalars(
        [values.len()],
        DType::U32,
        values.iter().map(|&value| Scalar::U(value as u64)),
    )
    .unwrap()
}

fn allocate_rendered(
    device: &MetalDevice,
    queue: &MetalCommandQueue,
    rendered: &RenderedMetal,
    values: &BTreeMap<u64, TensorData>,
) -> Vec<MetalBuffer> {
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

#[test]
fn captured_random_plans_render_and_mock_execute_without_stream_state() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let uniform = graph.uniform([5], -1.25, 2.5, DType::F32, 1337).unwrap();
    let normal = graph.randn([3], DType::F32, 1338).unwrap();
    let randint_i32 = graph.randint([5], -7, 19, DType::I32, 1339).unwrap();
    let randint_u32 = graph.randint([5], 3, 19, DType::U32, 1340).unwrap();
    for output in [uniform, normal, randint_i32, randint_u32] {
        let root = crate::kernel::lower_graph_random(&graph, output).unwrap();
        let rendered = renderer.render(&root).unwrap();
        let Operation::Random(plan) = root.operation() else {
            panic!("missing random plan")
        };
        let expected = plan.execute().unwrap();
        let mock = Arc::new(MockDispatch::default());
        let (device, queue) = setup(mock.clone());
        let output_buffer = device.allocate_typed(rendered.extent, plan.dtype).unwrap();
        let cache = device.cache();
        let pipeline = cache.load(&rendered).unwrap();
        assert!(Rc::ptr_eq(&pipeline, &cache.load(&rendered).unwrap()));
        pipeline
            .launch(&queue, &[&output_buffer], 8)
            .unwrap()
            .unwrap()
            .collect()
            .unwrap();
        let mut bytes = vec![0; expected.to_le_bytes().unwrap().len()];
        queue.read(&output_buffer, 0, &mut bytes).unwrap();
        assert_eq!(bytes, expected.to_le_bytes().unwrap(), "{:?}", plan.kind);
        assert_eq!(rendered.buffers.len(), 1);
        assert!(rendered.source.contains("captured-threefry"));
        assert!(rendered.source.contains("ulong chunk=i/maxw"));
        assert!(rendered.source_map.contains_key(&plan.output.index()));
        assert!(mock.calls().iter().any(|call| call.starts_with("launch:")));
    }
}

#[test]
fn captured_metal_random_rejects_unsupported_storage_and_empty_launch_is_safe() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let narrow = graph.rand([3], DType::F16, 4).unwrap();
    let wide = graph.randint([3], -3, 5, DType::I64, 5).unwrap();
    let empty = graph.rand([0], DType::F32, 6).unwrap();
    assert!(matches!(
        renderer.render(&crate::kernel::lower_graph_random(&graph, narrow).unwrap()),
        Err(MetalError::Unsupported(_))
    ));
    assert!(matches!(
        renderer.render(&crate::kernel::lower_graph_random(&graph, wide).unwrap()),
        Err(MetalError::Unsupported(_))
    ));
    let rendered = renderer
        .render(&crate::kernel::lower_graph_random(&graph, empty).unwrap())
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let output = device.allocate_typed(0, DType::F32).unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(pipeline.launch(&queue, &[&output], 8).unwrap().is_none());
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn captured_random_owner_and_launch_failures_preserve_visible_bytes() {
    let mut graph = Graph::new();
    let output = graph.randint([3], -7, 19, DType::I32, 91).unwrap();
    let other = graph.randint([3], -7, 19, DType::I32, 92).unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let rendered = renderer
        .render(&crate::kernel::lower_graph_random(&graph, output).unwrap())
        .unwrap();
    let other_rendered = renderer
        .render(&crate::kernel::lower_graph_random(&graph, other).unwrap())
        .unwrap();
    assert_ne!(rendered.cache_key, other_rendered.cache_key);
    let mock = Arc::new(MockDispatch::default());
    let runtime = MetalRuntime::from_dispatch(mock.clone());
    let mut devices = runtime.devices().unwrap();
    let first = devices.remove(0);
    let second = devices.remove(0);
    let first_queue = first.create_queue().unwrap();
    let second_queue = second.create_queue().unwrap();
    let output_buffer = first.allocate_typed(3, DType::I32).unwrap();
    let original = [0x5au8; 12];
    first_queue.write(&output_buffer, 0, &original).unwrap();
    let pipeline = first.cache().load(&rendered).unwrap();
    assert!(matches!(
        pipeline.launch(&second_queue, &[&output_buffer], 8),
        Err(MetalError::OwnerMismatch)
    ));
    mock.state.lock().unwrap().failures.launch = Some("random launch");
    assert!(matches!(
        pipeline.launch(&first_queue, &[&output_buffer], 8),
        Err(MetalError::Driver {
            operation: "launch",
            ..
        })
    ));
    let mut actual = [0u8; 12];
    first_queue.read(&output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual, original);
}

#[test]
fn renderer_mock_matches_cpu_for_affine_broadcast_select_and_casts() {
    let mut graph = Graph::new();
    let storage = graph.input("storage", Shape::from([4, 2]));
    let view = graph.shrink(storage, [(1, 3), (0, 2)]).unwrap();
    let row = graph.input("row", Shape::from([1, 2]));
    let sum = graph.add(view, row).unwrap();
    let scale = graph.constant(TensorData::scalar(2.0));
    let product = graph.mul(sum, scale).unwrap();
    let threshold = graph.constant(TensorData::scalar(9.0));
    let condition = graph.gt(product, threshold).unwrap();
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
    let rendered = MetalRenderer::new(8, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.source.contains("thread_position_in_grid"));
    assert!(rendered.source.contains("[[buffer(0)]]"));
    assert!(rendered.source.contains("2ul +"));
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
            .filter(|call| call.starts_with("library_compile"))
            .count(),
        1
    );
}

#[test]
fn all_source_backed_affine_forms_and_bool_alu_match_cpu_oracle() {
    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([1, 6]));
    let reshaped = graph.reshape(input, [1, 2, 3]).unwrap();
    let expanded = graph.expand(reshaped, [2, 2, 3]).unwrap();
    let permuted = graph.permute(expanded, vec![1, 0, 2]).unwrap();
    let shrunk = graph.shrink(permuted, [(0, 2), (0, 2), (0, 3)]).unwrap();
    let strided = graph
        .stride(
            shrunk,
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
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let view = rendered
        .buffers
        .iter()
        .find(|abi| abi.id == input.index() as u64)
        .unwrap()
        .view
        .as_ref()
        .unwrap();
    assert_eq!(view.logical_shape, Shape::from([2, 2, 2]));
    assert_eq!(view.strides, vec![3, 0, 2]);
    let (actual, _) = execute_mock(&graph, output, &inputs);
    assert_eq!(
        actual.to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );

    let mut bool_graph = Graph::new();
    let lhs = bool_graph.input_dtype("lhs", Shape::from([4]), DType::Bool);
    let rhs = bool_graph.input_dtype("rhs", Shape::from([4]), DType::Bool);
    let added = bool_graph.add(lhs, rhs).unwrap();
    let subtracted = bool_graph.sub(lhs, rhs).unwrap();
    let bool_output = bool_graph.mul(added, subtracted).unwrap();
    let bool_inputs = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [4],
                DType::Bool,
                [true, true, false, false].map(Scalar::Bool),
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [4],
                DType::Bool,
                [true, false, true, false].map(Scalar::Bool),
            )
            .unwrap(),
        ),
    ]);
    let bool_expected = CpuBackend
        .execute(&bool_graph, bool_output, &bool_inputs)
        .unwrap();
    let (bool_actual, _) = execute_mock(&bool_graph, bool_output, &bool_inputs);
    assert_eq!(
        bool_actual.to_le_bytes().unwrap(),
        bool_expected.to_le_bytes().unwrap()
    );
}

#[test]
fn renderer_identity_and_unsupported_boundaries_are_pre_submission() {
    let mut graph = Graph::new();
    let input = graph.input("x", Shape::from([2, 2]));
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.sub(input, one).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let first = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert_eq!(
        first.source,
        MetalRenderer::new(4, capabilities())
            .unwrap()
            .render(&item.kernel)
            .unwrap()
            .source
    );
    let mut changed = capabilities();
    changed.family = "MockApple10".into();
    assert_ne!(
        first.cache_key,
        MetalRenderer::new(4, changed)
            .unwrap()
            .render(&item.kernel)
            .unwrap()
            .cache_key
    );

    let reduced = graph
        .reduce(input, ReduceKind::Sum, Some(vec![1]), false)
        .unwrap();
    let reduction_item = schedule(&graph, reduced).unwrap().items.pop().unwrap();
    let reduction = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&reduction_item.kernel)
        .unwrap();
    assert!(reduction.source.contains("for (ulong rg_r"));

    let mut unsupported = Graph::new();
    let f64_input = unsupported.input_dtype("x", [2, 2], DType::F64);
    let f64_reduced = unsupported
        .reduce(f64_input, ReduceKind::Sum, Some(vec![1]), false)
        .unwrap();
    let f64_item = schedule(&unsupported, f64_reduced)
        .unwrap()
        .items
        .pop()
        .unwrap();
    assert!(matches!(
        MetalRenderer::new(4, capabilities())
            .unwrap()
            .render(&f64_item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("F64")
    ));

    let mut integer_graph = Graph::new();
    let lhs = integer_graph.input_dtype("lhs", Shape::from([2]), DType::I32);
    let rhs = integer_graph.input_dtype("rhs", Shape::from([2]), DType::I32);
    let integer_output = integer_graph.add(lhs, rhs).unwrap();
    let integer_item = schedule(&integer_graph, integer_output)
        .unwrap()
        .items
        .pop()
        .unwrap();
    let integer_rendered = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&integer_item.kernel)
        .unwrap();
    assert!(integer_rendered.source.contains("as_type<uint>"));
    assert!(integer_rendered.transaction.is_none());

    let divided = integer_graph.binary(BinaryOp::Div, lhs, rhs).unwrap();
    let floored = integer_graph.binary(BinaryOp::FloorDiv, lhs, rhs).unwrap();
    let divided = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&schedule(&integer_graph, divided).unwrap().items[0].kernel)
        .unwrap();
    let floored = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&schedule(&integer_graph, floored).unwrap().items[0].kernel)
        .unwrap();
    assert_ne!(divided.cache_key, floored.cache_key);
    assert_ne!(divided.transaction, floored.transaction);
}

#[test]
fn typed_reduction_recurrence_executes_through_metal_mock() {
    let cases = [
        (
            DType::F32,
            ReduceKind::Mean,
            TensorData::from_storage([3], Storage::F32(vec![1.0, 2.0, 4.0])).unwrap(),
        ),
        (
            DType::I32,
            ReduceKind::Sum,
            TensorData::from_storage([2], Storage::I32(vec![i32::MAX, 1])).unwrap(),
        ),
        (
            DType::Bool,
            ReduceKind::Max,
            TensorData::from_storage([3], Storage::Bool(vec![false, true, false])).unwrap(),
        ),
    ];
    for (dtype, kind, input) in cases {
        let mut graph = Graph::new();
        let source = graph.input_dtype("x", input.shape().clone(), dtype);
        let output = graph.reduce(source, kind, Some(vec![0]), false).unwrap();
        let inputs = HashMap::from([("x".into(), input)]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap(),
            "{dtype:?} {kind:?}"
        );
    }
}

#[test]
fn shared_scalar_lane_intrinsics_division_and_bitwise_render_structurally() {
    let renderer = MetalRenderer::new(4, capabilities()).unwrap();
    let dialect = renderer::MetalScalarDialect;
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
    assert!(bitwise.contains("(int)(lhs)") && bitwise.contains(" | "));
    assert!(add.contains("(float)(lhs)") && add.contains(" + "));
    assert!(compare_error.contains("compare dtype"));

    for (name, operation) in [
        ("sqrt", crate::UnaryOp::Sqrt),
        ("exp2", crate::UnaryOp::Exp2),
        ("log2", crate::UnaryOp::Log2),
        ("precise::sin", crate::UnaryOp::Sin),
        ("trunc", crate::UnaryOp::Trunc),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], DType::F32);
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
            "{operation:?}"
        );
        assert!(rendered.source.contains(METAL_RENDERER_VERSION));
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
        assert!(
            rendered.source.contains("precise::sin((float)("),
            "{dtype:?}"
        );
    }

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
        guarded.source.find("atomic_fetch_min_explicit").unwrap()
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
        Err(MetalError::Unsupported(_))
    ));
}

#[test]
fn checked_copies_and_command_retention_preserve_resources() {
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let src = device.allocate_typed(4, DType::F32).unwrap();
    let dst = device.allocate_typed(4, DType::F32).unwrap();
    assert_eq!(src.generation(), 1);
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    queue.write(&src, 0, &bytes).unwrap();
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
    let mut actual = vec![0; bytes.len()];
    queue.read(&dst, 0, &mut actual).unwrap();
    assert_eq!(actual, bytes);
    let dropped_src = device.allocate_typed(4, DType::F32).unwrap();
    let dropped_dst = device.allocate_typed(4, DType::F32).unwrap();
    let dropped_command = queue
        .copy(&dropped_src, &dropped_dst, 0, 0, bytes.len())
        .unwrap()
        .unwrap();
    drop(dropped_src);
    let waits_before_drop = mock
        .calls()
        .iter()
        .filter(|call| call.starts_with("wait"))
        .count();
    drop(dropped_command);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("wait"))
            .count(),
        waits_before_drop + 1
    );
    assert!(matches!(
        queue.read(&dst, bytes.len(), &mut [0; 1]),
        Err(MetalError::Bounds)
    ));
    let bool_buffer = device.allocate_typed(bytes.len(), DType::Bool).unwrap();
    assert!(matches!(
        queue.copy(&dst, &bool_buffer, 0, 0, bytes.len()),
        Err(MetalError::InvalidBinding(reason)) if reason.contains("D2D copy dtype")
    ));
    let mut graph = Graph::new();
    let input = graph.input("x", Shape::from([4]));
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.add(input, one).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(4, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    // The scalar constant is embedded directly in the typed UOp, so the ABI
    // owns only the input and output buffers.
    assert_eq!(rendered.buffers.len(), 2);
    let wrong_buffers = rendered
        .buffers
        .iter()
        .map(|abi| {
            device
                .allocate_typed(
                    if abi.id == input.index() as u64 {
                        abi.elements * DType::F32.itemsize()
                    } else {
                        abi.elements
                    },
                    if abi.id == input.index() as u64 {
                        DType::Bool
                    } else {
                        abi.dtype
                    },
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let wrong_refs = wrong_buffers.iter().collect::<Vec<_>>();
    assert!(matches!(
        pipeline.launch(&queue, &wrong_refs, 4),
        Err(MetalError::InvalidBinding(reason)) if reason.contains("dtype")
    ));
}

#[test]
fn resource_copy_build_launch_and_event_failures_are_distinct() {
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    mock.state.lock().unwrap().failures.buffer_create = Some("oom");
    assert!(matches!(
        device.allocate(4),
        Err(MetalError::Driver {
            operation: "buffer_create",
            ..
        })
    ));
    mock.clear_failures();
    let src = device.allocate_typed(1, DType::F32).unwrap();
    let dst = device.allocate_typed(1, DType::F32).unwrap();
    mock.state.lock().unwrap().failures.write = Some("mapping");
    assert!(matches!(
        queue.write(&src, 0, &1.0f32.to_le_bytes()),
        Err(MetalError::Driver {
            operation: "write",
            ..
        })
    ));
    mock.state.lock().unwrap().failures.copy = Some("blit");
    assert!(matches!(
        queue.copy(&src, &dst, 0, 0, 4),
        Err(MetalError::Driver {
            operation: "copy",
            ..
        })
    ));

    let mut graph = Graph::new();
    let input = graph.input("x", Shape::from([1]));
    let one = graph.constant(TensorData::scalar(1.0));
    let output = graph.add(input, one).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(1, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    mock.state.lock().unwrap().failures.build = Some("line 7: bad source".into());
    assert!(matches!(
        device.compile(&rendered),
        Err(MetalError::Build { diagnostic }) if diagnostic == "line 7: bad source"
    ));
    let library = device.compile(&rendered).unwrap();
    mock.state.lock().unwrap().failures.pipeline = Some("pipeline");
    assert!(matches!(
        library.create_pipeline(),
        Err(MetalError::Driver {
            operation: "pipeline_create",
            ..
        })
    ));
    let pipeline = library.create_pipeline().unwrap();
    let launch_buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    queue
        .write(&launch_buffers[0], 0, &2.0f32.to_le_bytes())
        .unwrap();
    let launch_refs = launch_buffers.iter().collect::<Vec<_>>();
    mock.state.lock().unwrap().failures.launch = Some("encode");
    assert!(matches!(
        pipeline.launch(&queue, &launch_refs, 1),
        Err(MetalError::Driver {
            operation: "launch",
            ..
        })
    ));
    let command = pipeline.launch(&queue, &launch_refs, 1).unwrap().unwrap();
    mock.state.lock().unwrap().failures.query = Some("status");
    assert!(matches!(
        command.query(),
        Err(MetalError::Driver {
            operation: "query",
            ..
        })
    ));
    mock.state.lock().unwrap().failures.wait = Some("gpu fault");
    assert!(matches!(
        command.collect(),
        Err(MetalError::Driver {
            operation: "wait",
            ..
        })
    ));
    mock.state.lock().unwrap().failures.read = Some("mapping");
    assert!(matches!(
        queue.read(launch_buffers.last().unwrap(), 0, &mut [0; 4]),
        Err(MetalError::Driver {
            operation: "read",
            ..
        })
    ));
}

#[test]
fn exact_i32_u32_arithmetic_guard_matrix_matches_cpu() {
    use crate::BinaryOp;
    let guarded = [
        BinaryOp::Div,
        BinaryOp::FloorDiv,
        BinaryOp::TruncDiv,
        BinaryOp::Mod,
        BinaryOp::FMod,
        BinaryOp::Shl,
        BinaryOp::Shr,
    ];
    for dtype in [DType::I32, DType::U32] {
        for operation in guarded {
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
            let inputs = HashMap::from([
                ("lhs".into(), lhs_value.clone()),
                ("rhs".into(), rhs_value.clone()),
            ]);
            let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
            let (actual, _) = execute_mock(&graph, output, &inputs);
            assert_eq!(
                actual.to_le_bytes().unwrap(),
                expected.to_le_bytes().unwrap(),
                "{dtype:?} {operation:?}"
            );
        }
    }

    for dtype in [DType::I32, DType::U32] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [4], dtype);
        let rhs = graph.input_dtype("rhs", [4], dtype);
        let added = graph.add(lhs, rhs).unwrap();
        let multiplied = graph.mul(added, rhs).unwrap();
        let wrapped = graph.sub(multiplied, lhs).unwrap();
        let compared = graph.gt(wrapped, lhs).unwrap();
        let as_integer = graph.cast(compared, dtype).unwrap();
        let output = graph.select(compared, wrapped, as_integer).unwrap();
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
        let (actual, _) = execute_mock(&graph, output, &inputs);
        assert_eq!(
            actual.to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );
    }
}

#[test]
fn nested_guard_order_detail_rollback_retry_and_stale_swap_are_exact() {
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
    let rendered = MetalRenderer::new(2, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let abi = rendered.transaction.as_ref().unwrap();
    assert_eq!(abi.version, METAL_TRANSACTION_ABI_VERSION);
    assert_eq!(
        abi.guards
            .iter()
            .map(|guard| guard.operation)
            .collect::<Vec<_>>(),
        [GuardedIntegerOp::Div, GuardedIntegerOp::Shl]
    );
    assert!(rendered.source.contains("atomic_fetch_min_explicit"));
    assert!(rendered.source.contains("(uint)gid * 2u + 0u"));
    assert!(rendered.source.contains("(uint)gid * 2u + 1u"));

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
        pipeline.launch(&queue, &refs, 2),
        Err(MetalError::InvalidArgument(
            "guarded kernel requires transactional launch"
        ))
    ));

    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::IntegerFault {
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
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::IntegerFault {
            operation: GuardedIntegerOp::Div,
            index: 0,
            ..
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
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::Driver { operation: "read", detail }) if detail == "detail"
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
    let first = pipeline.launch_transactional(&queue, &refs, 2).unwrap();
    let stale = pipeline.launch_transactional(&queue, &refs, 2).unwrap();
    first.wait().unwrap();
    assert_eq!(output_buffer.generation(), generation + 1);
    assert!(matches!(
        stale.wait(),
        Err(MetalError::StaleGeneration { expected, actual })
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
fn transaction_failures_lazy_branches_zero_domain_and_cleanup_preserve_visibility() {
    let mut graph = Graph::new();
    let condition = graph.input_dtype("condition", [2], DType::Bool);
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let divisor = graph.input_dtype("divisor", [2], DType::I32);
    let count = graph.input_dtype("count", [2], DType::I32);
    let quotient = graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let shifted = graph.binary(BinaryOp::Shl, lhs, count).unwrap();
    let output = graph.select(condition, quotient, shifted).unwrap();
    let rendered = MetalRenderer::new(2, capabilities())
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
    queue.write(output_buffer, 0, &[0x6d; 8]).unwrap();
    let cache = device.cache();
    let pipeline = cache.load(&rendered).unwrap();
    pipeline
        .launch_transactional(&queue, &refs, 2)
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
    for stage in ["encode", "submit"] {
        mock.state.lock().unwrap().failures.launch = Some(stage);
        assert!(matches!(
            pipeline.launch_transactional(&queue, &refs, 2),
            Err(MetalError::Driver { operation: "launch", detail }) if detail == stage
        ));
        assert_eq!(output_buffer.generation(), generation);
    }
    mock.state.lock().unwrap().failures.wait = Some("compute");
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::Driver { operation: "wait", detail }) if detail == "compute"
    ));
    mock.state.lock().unwrap().failures.read = Some("status");
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::Driver { operation: "read", detail }) if detail == "status"
    ));
    mock.state.lock().unwrap().failures.query = Some("nonblocking");
    let token = pipeline.launch_transactional(&queue, &refs, 2).unwrap();
    assert!(matches!(
        token.query(),
        Err(MetalError::Driver { operation: "query", detail }) if detail == "nonblocking"
    ));
    drop(token);

    mock.state.lock().unwrap().failures.buffer_create = Some("candidate");
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs, 2),
        Err(MetalError::Driver { operation: "buffer_create", detail }) if detail == "candidate"
    ));
    mock.state.lock().unwrap().failures.buffer_create_after = Some((1, "status allocation"));
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs, 2),
        Err(MetalError::Driver { operation: "buffer_create", detail }) if detail == "status allocation"
    ));
    mock.state.lock().unwrap().failures.write = Some("status initialize");
    assert!(matches!(
        pipeline.launch_transactional(&queue, &refs, 2),
        Err(MetalError::Driver { operation: "write", detail }) if detail == "status initialize"
    ));
    let mut unchanged = [0; 8];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, sentinel);
    assert_eq!(output_buffer.generation(), generation);
    assert_eq!(mock.state.lock().unwrap().buffers.len(), buffers.len());

    let mut empty_graph = Graph::new();
    let empty_lhs = empty_graph.input_dtype("lhs", [0], DType::U32);
    let empty_rhs = empty_graph.input_dtype("rhs", [0], DType::U32);
    let empty_output = empty_graph
        .binary(BinaryOp::Div, empty_lhs, empty_rhs)
        .unwrap();
    let empty_rendered = MetalRenderer::new(1, capabilities())
        .unwrap()
        .render(&schedule(&empty_graph, empty_output).unwrap().items[0].kernel)
        .unwrap();
    let empty_buffers = empty_rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let empty_pipeline = cache.load(&empty_rendered).unwrap();
    let empty_refs = empty_buffers.iter().collect::<Vec<_>>();
    let before = empty_buffers.last().unwrap().generation();
    let token = empty_pipeline
        .launch_transactional(&queue, &empty_refs, 1)
        .unwrap();
    assert!(token.query().unwrap());
    token.wait().unwrap();
    assert_eq!(empty_buffers.last().unwrap().generation(), before + 1);

    for dtype in [DType::I64, DType::U64] {
        let mut unsupported = Graph::new();
        let lhs = unsupported.input_dtype("lhs", [1], dtype);
        let rhs = unsupported.input_dtype("rhs", [1], dtype);
        let output = unsupported.div(lhs, rhs).unwrap();
        let item = &schedule(&unsupported, output).unwrap().items[0];
        assert!(matches!(
            MetalRenderer::new(1, capabilities()).unwrap().render(&item.kernel),
            Err(MetalError::Unsupported(reason)) if reason.contains("I64") || reason.contains("U64")
        ));
    }
}

#[test]
fn lazy_logical_branches_and_affine_shift_detail_are_exact() {
    let mut and_graph = Graph::new();
    let mask = and_graph.input_dtype("mask", [2], DType::Bool);
    let lhs = and_graph.input_dtype("lhs", [2], DType::I32);
    let divisor = and_graph.input_dtype("divisor", [2], DType::I32);
    let zero =
        and_graph.constant(TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap());
    let quotient = and_graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let positive = and_graph.gt(quotient, zero).unwrap();
    let and_output = and_graph.logical_and(mask, positive).unwrap();
    let and_inputs = HashMap::from([
        (
            "mask".into(),
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(false), Scalar::Bool(true)])
                .unwrap(),
        ),
        ("lhs".into(), ints(&[4, 8])),
        ("divisor".into(), ints(&[0, 2])),
    ]);
    let (actual, _) = execute_mock(&and_graph, and_output, &and_inputs);
    assert_eq!(actual.to_le_bytes().unwrap(), [0, 1]);

    let mut or_graph = Graph::new();
    let mask = or_graph.input_dtype("mask", [2], DType::Bool);
    let lhs = or_graph.input_dtype("lhs", [2], DType::I32);
    let count = or_graph.input_dtype("count", [2], DType::I32);
    let zero =
        or_graph.constant(TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap());
    let shifted = or_graph.binary(BinaryOp::Shl, lhs, count).unwrap();
    let positive = or_graph.gt(shifted, zero).unwrap();
    let or_output = or_graph.logical_or(mask, positive).unwrap();
    let or_inputs = HashMap::from([
        (
            "mask".into(),
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(true), Scalar::Bool(false)])
                .unwrap(),
        ),
        ("lhs".into(), ints(&[4, 8])),
        ("count".into(), ints(&[99, 1])),
    ]);
    let (actual, _) = execute_mock(&or_graph, or_output, &or_inputs);
    assert_eq!(actual.to_le_bytes().unwrap(), [1, 1]);

    let mut view_graph = Graph::new();
    let lhs = view_graph.input_dtype("lhs", [2, 2], DType::I32);
    let rhs_storage = view_graph.input_dtype("rhs", [2, 4], DType::I32);
    let rhs = view_graph.shrink(rhs_storage, [(0, 2), (1, 3)]).unwrap();
    let view_output = view_graph.binary(BinaryOp::Shl, lhs, rhs).unwrap();
    let rendered = MetalRenderer::new(2, capabilities())
        .unwrap()
        .render(&schedule(&view_graph, view_output).unwrap().items[0].kernel)
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
    let output_buffer = buffers.last().unwrap();
    queue.write(output_buffer, 0, &[0x77; 16]).unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(matches!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait(),
        Err(MetalError::IntegerFault {
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

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires an Apple Metal device"]
fn live_metal_discovery_compile_transfer_launch_wait_smoke() {
    let runtime = MetalRuntime::load().unwrap();
    let device = runtime.devices().unwrap().remove(0);
    let queue = device.create_queue().unwrap();
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", Shape::from([4]));
    let rhs = graph.input("rhs", Shape::from([4]));
    let output = graph.add(lhs, rhs).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(4, device.info().capabilities.clone())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let lhs_bytes = [1.0f32, -2.0, 3.5, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let rhs_bytes = [2.0f32, 1.0, -0.5, -0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let lhs_staging = device.allocate_typed(4, DType::F32).unwrap();
    let rhs_staging = device.allocate_typed(4, DType::F32).unwrap();
    queue.write(&lhs_staging, 0, &lhs_bytes).unwrap();
    queue.write(&rhs_staging, 0, &rhs_bytes).unwrap();
    queue
        .copy(&lhs_staging, &buffers[0], 0, 0, lhs_bytes.len())
        .unwrap()
        .unwrap()
        .collect()
        .unwrap();
    queue
        .copy(&rhs_staging, &buffers[1], 0, 0, rhs_bytes.len())
        .unwrap()
        .unwrap()
        .collect()
        .unwrap();
    let pipeline = device.cache().load(&rendered).unwrap();
    let refs = buffers.iter().collect::<Vec<_>>();
    let command = pipeline.launch(&queue, &refs, 4).unwrap().unwrap();
    let _ = command.query().unwrap();
    command.collect().unwrap();
    let mut actual = vec![0; 16];
    queue.read(buffers.last().unwrap(), 0, &mut actual).unwrap();
    let expected = [3.0f32, -1.0, 3.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires an Apple Metal device"]
fn live_metal_i32_transaction_success_and_fault_rollback_smoke() {
    let runtime = MetalRuntime::load().unwrap();
    let device = runtime.devices().unwrap().remove(0);
    let queue = device.create_queue().unwrap();
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let divisor = graph.input_dtype("divisor", [2], DType::I32);
    let count = graph.input_dtype("count", [2], DType::I32);
    let quotient = graph.binary(BinaryOp::Div, lhs, divisor).unwrap();
    let shifted_left = graph.binary(BinaryOp::Shl, quotient, count).unwrap();
    let output = graph.binary(BinaryOp::Shr, shifted_left, count).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let rendered = MetalRenderer::new(2, device.info().capabilities.clone())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| device.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let positions = rendered
        .buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let write = |id: u64, value: &[i32]| {
        queue
            .write(
                &buffers[positions[&id]],
                0,
                &value
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    };
    write(lhs.index() as u64, &[8, -9]);
    write(divisor.index() as u64, &[2, 3]);
    write(count.index() as u64, &[1, 2]);
    let output_buffer = &buffers[rendered.transaction.as_ref().unwrap().output_abi_index];
    let refs = buffers.iter().collect::<Vec<_>>();
    let pipeline = device.cache().load(&rendered).unwrap();
    let transaction = pipeline.launch_transactional(&queue, &refs, 2).unwrap();
    let _ = transaction.query().unwrap();
    transaction.wait().unwrap();
    let mut actual = [0; 8];
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(
        actual,
        [4i32.to_le_bytes(), (-3i32).to_le_bytes()]
            .concat()
            .as_slice()
    );

    let sentinel = [0x5a; 8];
    queue.write(output_buffer, 0, &sentinel).unwrap();
    let generation = output_buffer.generation();
    write(divisor.index() as u64, &[0, 3]);
    assert_eq!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait()
            .unwrap_err(),
        MetalError::IntegerFault {
            operation: GuardedIntegerOp::Div,
            index: 0,
            count: None,
            bits: 32,
        }
    );
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual, sentinel);
    assert_eq!(output_buffer.generation(), generation);

    write(divisor.index() as u64, &[2, 3]);
    write(count.index() as u64, &[32, 2]);
    assert_eq!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait()
            .unwrap_err(),
        MetalError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 0,
            count: Some(32),
            bits: 32,
        }
    );
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual, sentinel);
    assert_eq!(output_buffer.generation(), generation);
}
