use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::{
    Backend, BufferRole, CpuBackend, DType, Graph, KernelBindings, KernelBufferDesc, Shape,
    TensorData, schedule,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    rc::Rc,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
enum Arg {
    Buffer(RawBuffer),
    U64(u64),
}

#[derive(Default)]
struct Failures {
    device_info: Option<i32>,
    buffer_create: Option<i32>,
    copy: Option<i32>,
    build: Option<(i32, String)>,
    launch: Option<i32>,
}

#[derive(Default)]
struct State {
    calls: Vec<String>,
    owners: BTreeSet<u64>,
    next_buffer: BTreeMap<u64, usize>,
    next_event: BTreeMap<u64, usize>,
    buffers: BTreeMap<(u64, usize), Vec<u8>>,
    programs: BTreeMap<(u64, usize), String>,
    kernels: BTreeMap<(u64, usize), String>,
    semantics: BTreeMap<(u64, usize), Arc<dispatch::KernelSemantics>>,
    args: BTreeMap<(u64, usize), Vec<Arg>>,
    events: BTreeMap<(u64, usize), bool>,
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

    fn set_build_failure(&self, code: i32, log: String) {
        self.state.lock().unwrap().failures.build = Some((code, log));
    }

    fn set_copy_failure(&self, code: i32) {
        self.state.lock().unwrap().failures.copy = Some(code);
    }

    fn set_launch_failure(&self, code: i32) {
        self.state.lock().unwrap().failures.launch = Some(code);
    }

    fn clear_failures(&self) {
        self.state.lock().unwrap().failures = Failures::default();
    }

    fn event(state: &mut State, owner: u64) -> RawEvent {
        let next = state.next_event.entry(owner).or_insert(60);
        let raw = RawEvent(*next);
        *next += 1;
        state.events.insert((owner, raw.0), false);
        raw
    }

    fn check_owner(state: &State, owner: u64) -> Result<(), OpenClError> {
        if state.owners.contains(&owner) {
            Ok(())
        } else {
            Err(OpenClError::OwnerMismatch)
        }
    }
}

impl Dispatch for MockDispatch {
    fn platforms(&self) -> Result<Vec<RawPlatform>, OpenClError> {
        self.state.lock().unwrap().calls.push("platforms".into());
        Ok(vec![RawPlatform(1)])
    }

    fn platform_name(&self, _platform: RawPlatform) -> Result<String, OpenClError> {
        Ok("Mock OpenCL".into())
    }

    fn devices(&self, _platform: RawPlatform) -> Result<Vec<RawDevice>, OpenClError> {
        self.state.lock().unwrap().calls.push("devices".into());
        Ok(vec![RawDevice(2)])
    }

    fn device_info(&self, _device: RawDevice) -> Result<DeviceInfo, OpenClError> {
        let state = self.state.lock().unwrap();
        if let Some(code) = state.failures.device_info {
            return Err(OpenClError::Driver {
                operation: "device_info",
                code,
            });
        }
        Ok(DeviceInfo {
            name: "Mock Device".into(),
            max_work_group_size: 128,
        })
    }

    fn context_create(&self, _device: RawDevice, owner: u64) -> Result<RawContext, OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.owners.insert(owner);
        state.calls.push(format!("context_create:{owner}"));
        Ok(RawContext(10))
    }

    fn context_release(&self, _context: RawContext, owner: u64) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.calls.push(format!("context_release:{owner}"));
        state.owners.remove(&owner);
        Ok(())
    }

    fn queue_create(
        &self,
        _context: RawContext,
        _device: RawDevice,
        owner: u64,
    ) -> Result<RawQueue, OpenClError> {
        let mut state = self.state.lock().unwrap();
        Self::check_owner(&state, owner)?;
        state.calls.push(format!("queue_create:{owner}"));
        Ok(RawQueue(20))
    }

    fn queue_release(&self, _queue: RawQueue, owner: u64) -> Result<(), OpenClError> {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("queue_release:{owner}"));
        Ok(())
    }

    fn queue_finish(&self, _queue: RawQueue, owner: u64) -> Result<(), OpenClError> {
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("finish:{owner}"));
        Ok(())
    }

    fn buffer_create(
        &self,
        _context: RawContext,
        bytes: usize,
        owner: u64,
    ) -> Result<RawBuffer, OpenClError> {
        let mut state = self.state.lock().unwrap();
        Self::check_owner(&state, owner)?;
        if let Some(code) = state.failures.buffer_create {
            return Err(OpenClError::Driver {
                operation: "buffer_create",
                code,
            });
        }
        let next = state.next_buffer.entry(owner).or_insert(30);
        let raw = RawBuffer(*next);
        *next += 1;
        state.buffers.insert((owner, raw.0), vec![0; bytes]);
        state.calls.push(format!("buffer_create:{owner}:{bytes}"));
        Ok(raw)
    }

    fn buffer_release(&self, buffer: RawBuffer, owner: u64) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.buffers.remove(&(owner, buffer.0));
        state
            .calls
            .push(format!("buffer_release:{owner}:{}", buffer.0));
        Ok(())
    }

    fn buffer_write(
        &self,
        _queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &[u8],
        owner: u64,
    ) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        let storage = state
            .buffers
            .get_mut(&(owner, buffer.0))
            .ok_or(OpenClError::OwnerMismatch)?;
        storage[offset..offset + bytes.len()].copy_from_slice(bytes);
        state
            .calls
            .push(format!("write:{owner}:{}:{}", buffer.0, bytes.len()));
        Ok(())
    }

    fn buffer_read(
        &self,
        _queue: RawQueue,
        buffer: RawBuffer,
        offset: usize,
        bytes: &mut [u8],
        owner: u64,
    ) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        let storage = state
            .buffers
            .get(&(owner, buffer.0))
            .ok_or(OpenClError::OwnerMismatch)?;
        bytes.copy_from_slice(&storage[offset..offset + bytes.len()]);
        state
            .calls
            .push(format!("read:{owner}:{}:{}", buffer.0, bytes.len()));
        Ok(())
    }

    fn buffer_copy(
        &self,
        _queue: RawQueue,
        src: RawBuffer,
        dst: RawBuffer,
        region: BufferCopyRegion,
        owner: u64,
    ) -> Result<RawEvent, OpenClError> {
        let mut state = self.state.lock().unwrap();
        if let Some(code) = state.failures.copy.take() {
            return Err(OpenClError::Driver {
                operation: "copy",
                code,
            });
        }
        let copied = state
            .buffers
            .get(&(owner, src.0))
            .ok_or(OpenClError::OwnerMismatch)?
            [region.src_offset..region.src_offset + region.bytes]
            .to_vec();
        state
            .buffers
            .get_mut(&(owner, dst.0))
            .ok_or(OpenClError::OwnerMismatch)?
            [region.dst_offset..region.dst_offset + region.bytes]
            .copy_from_slice(&copied);
        state
            .calls
            .push(format!("copy:{owner}:{}:{}:{}", src.0, dst.0, region.bytes));
        Ok(Self::event(&mut state, owner))
    }

    fn program_create(
        &self,
        _context: RawContext,
        source: &str,
        owner: u64,
    ) -> Result<RawProgram, OpenClError> {
        let mut state = self.state.lock().unwrap();
        let raw = RawProgram(40 + state.programs.keys().filter(|(o, _)| *o == owner).count());
        state.programs.insert((owner, raw.0), source.into());
        state.calls.push(format!("program_create:{owner}"));
        Ok(raw)
    }

    fn program_build(
        &self,
        _program: RawProgram,
        _device: RawDevice,
        options: &str,
        owner: u64,
    ) -> Result<(), OpenClError> {
        let state = self.state.lock().unwrap();
        if let Some((code, _)) = &state.failures.build {
            return Err(OpenClError::Driver {
                operation: "build",
                code: *code,
            });
        }
        drop(state);
        self.state
            .lock()
            .unwrap()
            .calls
            .push(format!("build:{owner}:{options}"));
        Ok(())
    }

    fn program_build_info(
        &self,
        _program: RawProgram,
        _device: RawDevice,
        _owner: u64,
    ) -> Result<BuildInfo, OpenClError> {
        Ok(BuildInfo {
            log: self
                .state
                .lock()
                .unwrap()
                .failures
                .build
                .as_ref()
                .map(|x| x.1.clone())
                .unwrap_or_else(|| "mock build ok".into()),
        })
    }

    fn program_release(&self, program: RawProgram, owner: u64) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.programs.remove(&(owner, program.0));
        state.calls.push(format!("program_release:{owner}"));
        Ok(())
    }

    fn kernel_create(
        &self,
        _program: RawProgram,
        entry: &str,
        owner: u64,
    ) -> Result<RawKernel, OpenClError> {
        let mut state = self.state.lock().unwrap();
        let raw = RawKernel(50 + state.kernels.keys().filter(|(o, _)| *o == owner).count());
        state.kernels.insert((owner, raw.0), entry.into());
        state.calls.push(format!("kernel_create:{owner}:{entry}"));
        Ok(raw)
    }

    fn kernel_release(&self, kernel: RawKernel, owner: u64) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.kernels.remove(&(owner, kernel.0));
        state.args.remove(&(owner, kernel.0));
        state.calls.push(format!("kernel_release:{owner}"));
        Ok(())
    }

    fn kernel_arg_buffer(
        &self,
        kernel: RawKernel,
        index: u32,
        buffer: RawBuffer,
        owner: u64,
    ) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        if !state.buffers.contains_key(&(owner, buffer.0)) {
            return Err(OpenClError::OwnerMismatch);
        }
        let args = state.args.entry((owner, kernel.0)).or_default();
        let index = index as usize;
        if args.len() <= index {
            args.resize(index + 1, Arg::U64(0));
        }
        args[index] = Arg::Buffer(buffer);
        Ok(())
    }

    fn kernel_arg_u64(
        &self,
        kernel: RawKernel,
        index: u32,
        value: u64,
        owner: u64,
    ) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        let args = state.args.entry((owner, kernel.0)).or_default();
        let index = index as usize;
        if args.len() <= index {
            args.resize(index + 1, Arg::U64(0));
        }
        args[index] = Arg::U64(value);
        Ok(())
    }

    fn kernel_launch(
        &self,
        _queue: RawQueue,
        kernel: RawKernel,
        global: usize,
        local: usize,
        owner: u64,
    ) -> Result<RawEvent, OpenClError> {
        let mut state = self.state.lock().unwrap();
        if let Some(code) = state.failures.launch.take() {
            return Err(OpenClError::Driver {
                operation: "launch",
                code,
            });
        }
        let semantics = state
            .semantics
            .get(&(owner, kernel.0))
            .cloned()
            .ok_or_else(|| OpenClError::InvalidBinding("mock kernel semantics absent".into()))?;
        let args = state
            .args
            .get(&(owner, kernel.0))
            .cloned()
            .ok_or_else(|| OpenClError::InvalidBinding("mock kernel args absent".into()))?;
        if global < semantics.extent || local == 0 || global % local != 0 {
            return Err(OpenClError::InvalidArgument("invalid mock launch geometry"));
        }
        let Some(Arg::U64(extent)) = args.get(semantics.buffers.len()) else {
            return Err(OpenClError::InvalidBinding("extent scalar absent".into()));
        };
        if usize::try_from(*extent).map_err(|_| OpenClError::Overflow)? != semantics.extent {
            return Err(OpenClError::InvalidBinding("extent scalar mismatch".into()));
        }
        let mut bindings = KernelBindings::default();
        let mut output = None;
        for (index, abi) in semantics.buffers.iter().enumerate() {
            let Some(Arg::Buffer(raw)) = args.get(index) else {
                return Err(OpenClError::InvalidBinding(format!(
                    "buffer arg {index} absent"
                )));
            };
            let expected = abi
                .elements
                .checked_mul(abi.dtype.itemsize())
                .ok_or(OpenClError::Overflow)?;
            let bytes = state
                .buffers
                .get(&(owner, raw.0))
                .ok_or(OpenClError::OwnerMismatch)?;
            if bytes.len() < expected {
                return Err(OpenClError::InvalidBinding(format!(
                    "buffer arg {index} too small"
                )));
            }
            let data =
                TensorData::from_le_bytes(abi.source_shape.clone(), abi.dtype, &bytes[..expected])
                    .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
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
            .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
            bindings
                .insert(&desc, data)
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
            if abi.mutable {
                output = Some((raw, expected));
            }
        }
        let result = execute_lowered_elementwise(&semantics.program, &bindings)
            .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        let result = result
            .to_le_bytes()
            .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
        let (raw, expected) =
            output.ok_or_else(|| OpenClError::InvalidBinding("mutable output absent".into()))?;
        if result.len() != expected {
            return Err(OpenClError::InvalidBinding(
                "semantic output size mismatch".into(),
            ));
        }
        state.buffers.get_mut(&(owner, raw.0)).unwrap()[..expected].copy_from_slice(&result);
        state.calls.push(format!("launch:{owner}:{global}:{local}"));
        Ok(Self::event(&mut state, owner))
    }

    fn event_query(&self, event: RawEvent, owner: u64) -> Result<bool, OpenClError> {
        self.state
            .lock()
            .unwrap()
            .events
            .get(&(owner, event.0))
            .copied()
            .ok_or(OpenClError::OwnerMismatch)
    }

    fn event_wait(&self, event: RawEvent, owner: u64) -> Result<(), OpenClError> {
        *self
            .state
            .lock()
            .unwrap()
            .events
            .get_mut(&(owner, event.0))
            .ok_or(OpenClError::OwnerMismatch)? = true;
        Ok(())
    }

    fn event_release(&self, event: RawEvent, owner: u64) -> Result<(), OpenClError> {
        let mut state = self.state.lock().unwrap();
        state.events.remove(&(owner, event.0));
        state.calls.push(format!("event_release:{owner}"));
        Ok(())
    }

    fn register_kernel_semantics(
        &self,
        owner: u64,
        kernel: RawKernel,
        semantics: Arc<dispatch::KernelSemantics>,
    ) -> Result<(), OpenClError> {
        self.state
            .lock()
            .unwrap()
            .semantics
            .insert((owner, kernel.0), semantics);
        Ok(())
    }

    fn unregister_kernel_semantics(&self, owner: u64, kernel: RawKernel) {
        self.state
            .lock()
            .unwrap()
            .semantics
            .remove(&(owner, kernel.0));
    }
}

fn setup(mock: Arc<MockDispatch>) -> (OpenClContext, OpenClQueue) {
    let icd = OpenClIcd::from_dispatch(mock);
    let platform = icd.platforms().unwrap().remove(0);
    assert_eq!(platform.name(), "Mock OpenCL");
    let device = platform.devices().unwrap().remove(0);
    assert_eq!(device.info().name, "Mock Device");
    let context = device.create_context().unwrap();
    let queue = context.create_queue().unwrap();
    (context, queue)
}

#[test]
fn discovery_and_allocation_failures_preserve_driver_status() {
    let mock = Arc::new(MockDispatch::default());
    mock.state.lock().unwrap().failures.device_info = Some(-33);
    let icd = OpenClIcd::from_dispatch(mock.clone());
    let platform = icd.platforms().unwrap().remove(0);
    assert!(matches!(
        platform.devices(),
        Err(OpenClError::Driver {
            operation: "device_info",
            code: -33
        })
    ));
    mock.clear_failures();
    let device = platform.devices().unwrap().remove(0);
    let context = device.create_context().unwrap();
    mock.state.lock().unwrap().failures.buffer_create = Some(-4);
    assert!(matches!(
        context.allocate(4),
        Err(OpenClError::Driver {
            operation: "buffer_create",
            code: -4
        })
    ));
}

#[test]
fn renderer_and_semantic_mock_execute_broadcast_select_in_schedule_order() {
    let mut graph = Graph::new();
    // Reverse NodeId order relative to use order to prove ABI ordering.
    let right = graph.input("right", Shape::from([1, 3]));
    let left = graph.input("left", Shape::from([2, 1]));
    let zero = graph.constant(TensorData::scalar(0.0));
    let condition = graph.gt(left, zero).unwrap();
    let output = graph.select(condition, left, right).unwrap();
    let scheduled = schedule(&graph, output).unwrap();
    let item = scheduled
        .items
        .iter()
        .find(|item| item.node == output)
        .unwrap();
    let renderer = OpenClRenderer::new(64).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let mut expected_abi = item
        .ordered_inputs()
        .iter()
        .map(|binding| binding.desc.id)
        .collect::<Vec<_>>();
    expected_abi.push(output.index() as u64);
    assert_eq!(
        rendered
            .buffers
            .iter()
            .map(|buffer| buffer.id)
            .collect::<Vec<_>>(),
        expected_abi
    );
    assert!(rendered.source.contains("get_global_id(0)"));
    assert!(rendered.source.contains("% 3ul"));
    assert_eq!(
        renderer.render(&item.kernel).unwrap().source,
        rendered.source
    );
    assert_eq!(
        renderer.render(&item.kernel).unwrap().cache_key,
        rendered.cache_key
    );

    let inputs = HashMap::from([
        (
            "left".into(),
            TensorData::new([2, 1], vec![1.0, 3.0]).unwrap(),
        ),
        (
            "right".into(),
            TensorData::new([1, 3], vec![0.0, 1.0, 2.0]).unwrap(),
        ),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let condition_value = CpuBackend.execute(&graph, condition, &inputs).unwrap();
    let values = BTreeMap::from([
        (left.index() as u64, inputs["left"].clone()),
        (right.index() as u64, inputs["right"].clone()),
        (condition.index() as u64, condition_value),
    ]);
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| {
            let bytes = abi.elements * abi.dtype.itemsize();
            let buffer = context.allocate(bytes).unwrap();
            if let Some(value) = values.get(&abi.id) {
                queue
                    .write(&buffer, 0, &value.to_le_bytes().unwrap())
                    .unwrap();
            }
            buffer
        })
        .collect::<Vec<_>>();
    let cache = context.cache();
    let kernel = cache
        .load(&rendered, "-cl-std=CL1.2", renderer.local_size)
        .unwrap();
    let again = cache
        .load(&rendered, "-cl-std=CL1.2", renderer.local_size)
        .unwrap();
    assert!(Rc::ptr_eq(&kernel, &again));
    assert_eq!(cache.len(), 1);
    let binding_refs = buffers.iter().collect::<Vec<_>>();
    let event = kernel.launch(&queue, &binding_refs).unwrap().unwrap();
    assert!(!event.query().unwrap());
    event.wait().unwrap();
    assert!(event.query().unwrap());
    let mut actual = vec![0; 24];
    queue.read(buffers.last().unwrap(), 0, &mut actual).unwrap();
    assert_eq!(actual, expected.to_le_bytes().unwrap());
    assert_eq!(kernel.build_info().log, "mock build ok");
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("program_create"))
            .count(),
        1
    );
}

#[test]
fn preflight_zero_owner_copy_and_failure_paths_are_non_mutating() {
    let mock = Arc::new(MockDispatch::default());
    let (first, first_queue) = setup(mock.clone());
    let (second, second_queue) = setup(mock.clone());
    let a = first.allocate(4).unwrap();
    let b = first.allocate(4).unwrap();
    let foreign = second.allocate(4).unwrap();
    first_queue.write(&a, 0, &[1, 2, 3, 4]).unwrap();
    let before = mock.calls().len();
    assert!(matches!(
        first_queue.write(&a, 3, &[5, 6]),
        Err(OpenClError::Bounds)
    ));
    assert!(matches!(
        first_queue.copy(&a, &foreign, 0, 0, 4),
        Err(OpenClError::OwnerMismatch)
    ));
    assert_eq!(mock.calls().len(), before);

    mock.set_copy_failure(-5);
    assert!(matches!(
        first_queue.copy(&a, &b, 0, 0, 4),
        Err(OpenClError::Driver { code: -5, .. })
    ));
    let mut bytes = [9; 4];
    first_queue.read(&b, 0, &mut bytes).unwrap();
    assert_eq!(bytes, [0; 4]);
    mock.clear_failures();
    let copy = first_queue.copy(&a, &b, 0, 0, 4).unwrap().unwrap();
    copy.wait().unwrap();
    first_queue.read(&b, 0, &mut bytes).unwrap();
    assert_eq!(bytes, [1, 2, 3, 4]);

    let zero = first.allocate(0).unwrap();
    let calls = mock.calls().len();
    assert!(first_queue.copy(&zero, &zero, 0, 0, 0).unwrap().is_none());
    assert_eq!(mock.calls().len(), calls);

    let mut graph = Graph::new();
    let x = graph.input("x", Shape::from([0, 2]));
    let output = graph.neg(x).unwrap();
    let rendered = OpenClRenderer::default()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let kernel = first.cache().load(&rendered, "", 64).unwrap();
    let output_zero = first.allocate(0).unwrap();
    let calls = mock.calls().len();
    assert!(
        kernel
            .launch(&first_queue, &[&zero, &output_zero])
            .unwrap()
            .is_none()
    );
    assert_eq!(mock.calls().len(), calls);
    assert!(matches!(
        kernel.launch(&second_queue, &[&zero, &output_zero]),
        Err(OpenClError::OwnerMismatch)
    ));
}

#[test]
fn build_launch_and_cleanup_failures_are_structured_and_retryable() {
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([1]));
    let output = graph.neg(input).unwrap();
    let rendered = OpenClRenderer::default()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    mock.set_build_failure(-11, "x".repeat(70_000));
    let cache = context.cache();
    let error = match cache.load(&rendered, "", 64) {
        Ok(_) => panic!("configured build failure unexpectedly succeeded"),
        Err(error) => error,
    };
    assert!(matches!(error, OpenClError::Build { code: -11, ref log } if log.len() == 65_536));
    assert!(
        mock.calls()
            .iter()
            .any(|call| call.starts_with("program_release"))
    );
    mock.clear_failures();
    let kernel = cache.load(&rendered, "", 64).unwrap();
    let input_buffer = context.allocate(4).unwrap();
    let output_buffer = context.allocate(4).unwrap();
    queue
        .write(&input_buffer, 0, &1.5f32.to_le_bytes())
        .unwrap();
    mock.set_launch_failure(-6);
    assert!(matches!(
        kernel.launch(&queue, &[&input_buffer, &output_buffer]),
        Err(OpenClError::Driver { code: -6, .. })
    ));
    let mut bytes = [0; 4];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(f32::from_le_bytes(bytes), 0.0);
    let event = kernel
        .launch(&queue, &[&input_buffer, &output_buffer])
        .unwrap()
        .unwrap();
    event.wait().unwrap();
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(f32::from_le_bytes(bytes), -1.5);

    drop(event);
    drop(kernel);
    drop(cache);
    drop(output_buffer);
    drop(input_buffer);
    drop(queue);
    drop(context);
    let calls = mock.calls();
    let kernel_release = calls
        .iter()
        .position(|call| call.starts_with("kernel_release"))
        .unwrap();
    let program_release = calls
        .iter()
        .rposition(|call| call.starts_with("program_release"))
        .unwrap();
    let context_release = calls
        .iter()
        .rposition(|call| call.starts_with("context_release"))
        .unwrap();
    assert!(kernel_release < program_release && program_release < context_release);
}

#[test]
fn renderer_rejects_unsupported_work_before_icd_calls() {
    let mock = Arc::new(MockDispatch::default());
    let before = mock.calls().len();
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::I32);
    let output = graph.neg(input).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    assert!(matches!(
        OpenClRenderer::default().render(&item.kernel),
        Err(OpenClError::Unsupported(_))
    ));
    assert_eq!(mock.calls().len(), before);

    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([2, 2]));
    let reduced = graph.sum(input, 1).unwrap();
    let item = &schedule(&graph, reduced).unwrap().items[0];
    assert!(matches!(
        OpenClRenderer::default().render(&item.kernel),
        Err(OpenClError::Unsupported(_))
    ));
    assert_eq!(mock.calls().len(), before);
}

#[test]
#[ignore = "requires a live OpenCL ICD and device"]
fn live_opencl_discovery_smoke() {
    let icd = OpenClIcd::load().unwrap();
    let platform = icd.platforms().unwrap().remove(0);
    let device = platform.devices().unwrap().remove(0);
    let context = device.create_context().unwrap();
    let queue = context.create_queue().unwrap();

    let mut graph = Graph::new();
    let lhs = graph.input("lhs", Shape::from([4]));
    let rhs = graph.input("rhs", Shape::from([4]));
    let output = graph.add(lhs, rhs).unwrap();
    let inputs = HashMap::from([
        (
            "lhs".into(),
            TensorData::new([4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::new([4], vec![0.5, -1.0, 2.0, 8.0]).unwrap(),
        ),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let renderer = OpenClRenderer::new(4).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let lhs_buffer = context.allocate(16).unwrap();
    let rhs_buffer = context.allocate(16).unwrap();
    let output_buffer = context.allocate(16).unwrap();
    queue
        .write(&lhs_buffer, 0, &inputs["lhs"].to_le_bytes().unwrap())
        .unwrap();
    queue
        .write(&rhs_buffer, 0, &inputs["rhs"].to_le_bytes().unwrap())
        .unwrap();
    let kernel = context
        .cache()
        .load(&rendered, "-cl-std=CL1.2", renderer.local_size)
        .unwrap();
    kernel
        .launch(&queue, &[&lhs_buffer, &rhs_buffer, &output_buffer])
        .unwrap()
        .unwrap()
        .wait()
        .unwrap();
    let mut bytes = vec![0; 16];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, expected.to_le_bytes().unwrap());
}
