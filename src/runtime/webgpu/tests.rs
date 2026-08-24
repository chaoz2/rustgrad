use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::{
    Backend, BufferRole, CpuBackend, DType, Graph, KernelBindings, KernelBufferDesc, NodeId,
    ReduceKind, Scalar, Shape, Slice, TensorData, schedule,
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
    next_shader: usize,
    next_pipeline: usize,
    next_command: usize,
    buffers: BTreeMap<(u64, usize), Vec<u8>>,
    shaders: BTreeMap<(u64, usize), String>,
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
        if buffers.len() != semantics.buffers.len()
            || geometry.extent as usize != semantics.extent
            || geometry.local == 0
            || geometry.workgroups != geometry.extent.div_ceil(geometry.local)
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
        // Independent typed lowered-UOp execution: this is not `CpuBackend`.
        let result = execute_lowered_elementwise(&semantics.program, &bindings)
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
    if let Some(command) = pipeline.launch(&queue, &refs).unwrap() {
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
    let divided = int_graph.div(lhs, rhs).unwrap();
    assert!(matches!(
        WgslRenderer::new(4, capabilities()).unwrap().render(&schedule(&int_graph, divided).unwrap().items[0].kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("transactional")
    ));
    let mut too_few = capabilities();
    too_few.max_storage_buffers_per_shader_stage = 1;
    assert!(matches!(
        WgslRenderer::new(4, too_few).unwrap().render(&item.kernel),
        Err(WebGpuError::Unsupported(reason)) if reason.contains("storage-buffer limit")
    ));
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
