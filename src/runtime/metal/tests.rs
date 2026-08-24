use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::{
    Backend, BufferRole, CpuBackend, DType, Graph, KernelBindings, KernelBufferDesc, NodeId,
    ReduceKind, Scalar, Shape, Slice, TensorData, schedule,
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

#[derive(Default)]
struct Failures {
    buffer_create: Option<&'static str>,
    write: Option<&'static str>,
    read: Option<&'static str>,
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
    next_library: usize,
    next_pipeline: usize,
    next_command: usize,
    buffers: BTreeMap<(u64, usize), Vec<u8>>,
    libraries: BTreeMap<(u64, usize), String>,
    semantics: BTreeMap<(u64, usize), Arc<KernelSemantics>>,
    commands: BTreeMap<(u64, usize), bool>,
    failures: Failures,
}

#[derive(Default)]
struct MockDispatch {
    state: Mutex<State>,
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
        if geometry.extent as usize != semantics.extent
            || geometry.local == 0
            || geometry.global < semantics.extent
            || !geometry.global.is_multiple_of(geometry.local)
            || buffers.len() != semantics.buffers.len()
        {
            return Err(MetalError::InvalidArgument("invalid mock launch geometry"));
        }
        let mut bindings = KernelBindings::default();
        let mut output = None;
        for (position, (raw, abi)) in buffers.iter().zip(&semantics.buffers).enumerate() {
            let expected = abi
                .elements
                .checked_mul(abi.dtype.itemsize())
                .ok_or(MetalError::Overflow)?;
            let bytes = state
                .buffers
                .get(&(owner, raw.0))
                .ok_or(MetalError::OwnerMismatch)?;
            if bytes.len() != expected {
                return Err(MetalError::InvalidBinding(format!(
                    "mock buffer {position} length mismatch"
                )));
            }
            let value = TensorData::from_le_bytes(abi.source_shape.clone(), abi.dtype, bytes)
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
                output = Some((*raw, expected));
            }
        }
        // This is RustGrad's typed UOp interpreter, not CpuBackend or native
        // Metal. It independently executes the retained semantic artifact.
        let result = execute_lowered_elementwise(&semantics.program, &bindings)
            .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
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
    let command = pipeline.launch(&queue, &refs, 8).unwrap().unwrap();
    assert!(!command.query().unwrap());
    let completion = command.collect().unwrap();
    assert_eq!(completion.extent, rendered.extent);
    assert_eq!(completion.retained_resources, rendered.buffers.len());
    let output_abi = rendered.buffers.last().unwrap();
    let mut bytes = vec![0; output_abi.elements * output_abi.dtype.itemsize()];
    queue.read(buffers.last().unwrap(), 0, &mut bytes).unwrap();
    let result =
        TensorData::from_le_bytes(output_abi.source_shape.clone(), output_abi.dtype, &bytes)
            .unwrap();
    (result, mock)
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
    assert!(matches!(
        MetalRenderer::new(4, capabilities())
            .unwrap()
            .render(&reduction_item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("reductions")
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
    assert!(matches!(
        MetalRenderer::new(4, capabilities())
            .unwrap()
            .render(&integer_item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("I32")
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
    assert_eq!(rendered.buffers.len(), 3);
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
