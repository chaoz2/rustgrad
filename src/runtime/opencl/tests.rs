use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::{
    AddressSpace, Backend, BinaryOp, BufferRole, CpuBackend, DType, Graph, KernelBindings,
    KernelBufferDesc, Scalar, Shape, TensorData, UArg, UOp, UOpKind, UType, schedule,
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
    read: Option<i32>,
    wait: Option<i32>,
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
    device_capabilities: Option<OpenClCapabilities>,
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

    fn set_read_failure(&self, code: i32) {
        self.state.lock().unwrap().failures.read = Some(code);
    }

    fn set_wait_failure(&self, code: i32) {
        self.state.lock().unwrap().failures.wait = Some(code);
    }

    fn clear_failures(&self) {
        self.state.lock().unwrap().failures = Failures::default();
    }

    fn set_device_capabilities(&self, capabilities: OpenClCapabilities) {
        self.state.lock().unwrap().device_capabilities = Some(capabilities);
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
            capabilities: state
                .device_capabilities
                .unwrap_or(OpenClCapabilities::FULL),
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
        if let Some(code) = state.failures.read.take() {
            return Err(OpenClError::Driver {
                operation: "read",
                code,
            });
        }
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
        let status_raw = if semantics.transaction.is_some() {
            let Some(Arg::Buffer(raw)) = args.get(semantics.buffers.len() + 1) else {
                return Err(OpenClError::InvalidBinding(
                    "transaction status absent".into(),
                ));
            };
            Some(*raw)
        } else {
            None
        };
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
        if let Some(transaction) = &semantics.transaction {
            let rhs = &semantics.buffers[transaction.rhs_abi_index];
            let Some(Arg::Buffer(rhs_raw)) = args.get(transaction.rhs_abi_index) else {
                return Err(OpenClError::InvalidBinding("transaction RHS absent".into()));
            };
            let rhs_bytes = state
                .buffers
                .get(&(owner, rhs_raw.0))
                .ok_or(OpenClError::OwnerMismatch)?;
            let rhs_data =
                TensorData::from_le_bytes(rhs.source_shape.clone(), rhs.dtype, rhs_bytes)
                    .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?;
            let mut first = None;
            for logical in (0..semantics.extent).rev() {
                let offset = resource::transaction_rhs_offset(&transaction.rhs_index, logical)?;
                let scalar = rhs_data.scalar_at(offset);
                let invalid = if transaction.operation.is_shift() {
                    match rhs.dtype.category() {
                        crate::DTypeCategory::Signed => {
                            scalar.as_i64() < 0
                                || scalar.as_u64() >= transaction.dtype.bits() as u64
                        }
                        _ => scalar.as_u64() >= transaction.dtype.bits() as u64,
                    }
                } else {
                    scalar.as_u64() == 0
                };
                if invalid {
                    first = Some(first.map_or(logical, |old: usize| old.min(logical)));
                }
            }
            if let Some(index) = first {
                let status_raw = status_raw.expect("transaction status");
                state
                    .buffers
                    .get_mut(&(owner, status_raw.0))
                    .ok_or(OpenClError::OwnerMismatch)?[..4]
                    .copy_from_slice(&(index as u32).to_le_bytes());
                state.calls.push(format!("launch:{owner}:{global}:{local}"));
                return Ok(Self::event(&mut state, owner));
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
        let mut state = self.state.lock().unwrap();
        if let Some(code) = state.failures.wait.take() {
            return Err(OpenClError::Driver {
                operation: "wait",
                code,
            });
        }
        *state
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

fn execute_mock_rendered(
    rendered: &RenderedOpenCl,
    renderer: OpenClRenderer,
    values: &BTreeMap<u64, TensorData>,
) -> (Vec<u8>, Vec<String>) {
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
        .load(rendered, "-cl-std=CL1.2", renderer.local_size)
        .unwrap();
    let refs = buffers.iter().collect::<Vec<_>>();
    if rendered.transaction.is_some() {
        kernel
            .launch_transactional(&queue, &refs)
            .unwrap()
            .wait()
            .unwrap();
    } else if let Some(event) = kernel.launch(&queue, &refs).unwrap() {
        event.wait().unwrap();
    }
    let output = rendered.buffers.last().unwrap();
    let mut bytes = vec![0; output.elements * output.dtype.itemsize()];
    queue.read(buffers.last().unwrap(), 0, &mut bytes).unwrap();
    (bytes, mock.calls())
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
fn static_view_and_core_dtype_semantics_match_cpu_oracle() {
    let renderer = OpenClRenderer::with_capabilities(4, OpenClCapabilities::FULL).unwrap();

    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([4, 2]));
    let view = graph.shrink(input, [(1, 3), (0, 2)]).unwrap();
    let scalar = graph.constant(TensorData::scalar(2.0));
    let output = graph.add(view, scalar).unwrap();
    let value = TensorData::new([4, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
    let inputs = HashMap::from([("input".into(), value.clone())]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert_eq!(rendered.buffers[0].source_shape, Shape::from([4, 2]));
    assert_eq!(rendered.buffers[0].view.as_ref().unwrap().offset, 2);
    assert!(rendered.source.contains("2ul +"));
    let (actual, _) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([
            (input.index() as u64, value),
            (scalar.index() as u64, TensorData::scalar(2.0)),
        ]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    let cases = [
        (
            "i32 wrapping",
            DType::I32,
            [i32::MAX.to_le_bytes(), 7i32.to_le_bytes()].concat(),
            [1i32.to_le_bytes(), (-9i32).to_le_bytes()].concat(),
        ),
        (
            "u32 wrapping",
            DType::U32,
            [u32::MAX.to_le_bytes(), 7u32.to_le_bytes()].concat(),
            [1u32.to_le_bytes(), 9u32.to_le_bytes()].concat(),
        ),
        (
            "i64 wrapping",
            DType::I64,
            [i64::MAX.to_le_bytes(), (-7i64).to_le_bytes()].concat(),
            [1i64.to_le_bytes(), 9i64.to_le_bytes()].concat(),
        ),
        (
            "u64 high bit",
            DType::U64,
            [u64::MAX.to_le_bytes(), (1u64 << 63).to_le_bytes()].concat(),
            [1u64.to_le_bytes(), (1u64 << 63).to_le_bytes()].concat(),
        ),
        (
            "f64",
            DType::F64,
            [1.25f64.to_le_bytes(), (-0.0f64).to_le_bytes()].concat(),
            [2.5f64.to_le_bytes(), 0.0f64.to_le_bytes()].concat(),
        ),
    ];
    for (name, dtype, lhs_bytes, rhs_bytes) in cases {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], dtype);
        let rhs = graph.input_dtype("rhs", [2], dtype);
        let output = graph.add(lhs, rhs).unwrap();
        let lhs_value = TensorData::from_le_bytes([2], dtype, &lhs_bytes).unwrap();
        let rhs_value = TensorData::from_le_bytes([2], dtype, &rhs_bytes).unwrap();
        let inputs = HashMap::from([
            ("lhs".into(), lhs_value.clone()),
            ("rhs".into(), rhs_value.clone()),
        ]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let rendered = renderer
            .render(&schedule(&graph, output).unwrap().items[0].kernel)
            .unwrap();
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([
                (lhs.index() as u64, lhs_value),
                (rhs.index() as u64, rhs_value),
            ]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
    }
}

#[test]
fn narrow_float_storage_literals_views_and_casts_are_exact() {
    let renderer = OpenClRenderer::with_capabilities(4, OpenClCapabilities::FULL).unwrap();

    let literal_source = |dtype, bits| {
        let ty = UType::scalar(dtype);
        let shape = Shape::new([]);
        let range = UOp::new(
            UOpKind::Range,
            Some(UType::scalar(DType::I64)),
            vec![UOp::constant(1, UType::scalar(DType::I64))],
            UArg::RangeAxis(0),
        );
        let address = UOp::new(
            UOpKind::DefineGlobal,
            Some(ty),
            vec![],
            UArg::Address {
                space: AddressSpace::Global,
                name: "literal".into(),
                element: ty,
            },
        );
        let index = UOp::new(
            UOpKind::Index,
            Some(ty),
            vec![address, range.clone()],
            UArg::BufferIndex {
                buffer: 77,
                elements: 1,
                input_shape: shape.clone(),
                output_shape: shape,
            },
        );
        renderer
            .render(&UOp::sink(vec![
                UOp::new(
                    UOpKind::Store,
                    None,
                    vec![index, UOp::scalar_constant(dtype, bits, ty)],
                    UArg::None,
                ),
                UOp::new(UOpKind::EndRange, None, vec![range], UArg::None),
            ]))
            .unwrap()
            .source
    };
    assert!(literal_source(DType::F16, 0x8001).contains("((ushort)0x8001u)"));
    assert!(literal_source(DType::BF16, 0x7fc1).contains("((ushort)0x7fc1u)"));

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [3, 2], DType::F16);
    let view = graph.shrink(input, [(0, 3), (1, 2)]).unwrap();
    let output = graph.neg(view).unwrap();
    let value = TensorData::from_le_bytes(
        [3, 2],
        DType::F16,
        &[
            0x0000u16.to_le_bytes(),
            0x8000u16.to_le_bytes(),
            0x0001u16.to_le_bytes(),
            0x7c00u16.to_le_bytes(),
            0x7e01u16.to_le_bytes(),
            0x3c00u16.to_le_bytes(),
        ]
        .concat(),
    )
    .unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let rendered_f16 = renderer.render(&item.kernel).unwrap();
    rendered_f16
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert!(rendered_f16.source.contains("rg_f16_to_f32"));
    assert!(rendered_f16.source.contains("rg_f32_to_f16"));
    assert!(rendered_f16.source.contains("* 2ul"));
    assert!(!rendered_f16.source.contains("cl_khr_fp16"));
    let (actual, _) = execute_mock_rendered(
        &rendered_f16,
        renderer,
        &BTreeMap::from([(input.index() as u64, value)]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::BF16);
    let one_value = TensorData::from_le_bytes([], DType::BF16, &0x3f80u16.to_le_bytes()).unwrap();
    let one = graph.constant(one_value.clone());
    let output = graph.add(input, one).unwrap();
    let value = TensorData::from_le_bytes(
        [4],
        DType::BF16,
        &[
            0x8000u16.to_le_bytes(),
            0x0001u16.to_le_bytes(),
            0x7f80u16.to_le_bytes(),
            0x7fc1u16.to_le_bytes(),
        ]
        .concat(),
    )
    .unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let rendered_bf16 = renderer.render(&item.kernel).unwrap();
    rendered_bf16
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert!(rendered_bf16.source.contains("rg_bf16_to_f32"));
    assert!(rendered_bf16.source.contains("rg_bf16_to_f32(b"));
    assert_ne!(rendered_bf16.cache_key, rendered_f16.cache_key);
    let (actual, _) = execute_mock_rendered(
        &rendered_bf16,
        renderer,
        &BTreeMap::from([
            (input.index() as u64, value),
            (one.index() as u64, one_value),
        ]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    let mut graph = Graph::new();
    let input = graph.input("input", [3]);
    let output = graph.cast(input, DType::F16).unwrap();
    let value = TensorData::new([3], vec![1.0004, -0.0, f32::INFINITY]).unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let rendered = renderer.render(&item.kernel).unwrap();
    assert!(rendered.source.contains("rg_f32_to_f16((float)"));
    let (actual, _) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([(input.index() as u64, value)]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    let mock = Arc::new(MockDispatch::default());
    let before = mock.calls().len();
    assert!(matches!(
        OpenClRenderer::default().render(&item.kernel),
        Err(OpenClError::Unsupported(reason)) if reason.contains("narrow-float")
    ));
    assert_eq!(mock.calls().len(), before);
}

#[test]
fn narrow_float_reductions_match_cpu_raw_storage_contracts() {
    let renderer = OpenClRenderer::with_capabilities(4, OpenClCapabilities::FULL).unwrap();
    let cases = [
        (
            "f16 sum",
            DType::F16,
            crate::ReduceKind::Sum,
            vec![0x3c00u16, 0x4000, 0xc200],
        ),
        (
            "bf16 mean",
            DType::BF16,
            crate::ReduceKind::Mean,
            vec![0x3f80u16, 0x4000, 0x4040],
        ),
        (
            "f16 product",
            DType::F16,
            crate::ReduceKind::Product,
            vec![0x4000u16, 0xc200, 0x3800],
        ),
        (
            "f16 nan-ignore first negative zero min",
            DType::F16,
            crate::ReduceKind::Min,
            vec![0x7e01u16, 0x8000, 0x0000],
        ),
        (
            "bf16 nan-ignore first positive zero max",
            DType::BF16,
            crate::ReduceKind::Max,
            vec![0x7fc1u16, 0x0000, 0x8000],
        ),
    ];
    let mut keys = BTreeSet::new();
    for (name, dtype, kind, words) in cases {
        let bytes = words
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let value = TensorData::from_le_bytes([3], dtype, &bytes).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [3], dtype);
        let output = graph.reduce(input, kind, Some(vec![0]), false).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), value.clone())]),
            )
            .unwrap();
        let item = schedule(&graph, output).unwrap().items.remove(0);
        let rendered = renderer.render(&item.kernel).unwrap();
        rendered
            .validate_schedule_bindings(item.ordered_inputs())
            .unwrap();
        assert!(keys.insert(rendered.cache_key.clone()), "{name}");
        assert!(rendered.source.contains("double acc"), "{name}");
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([(input.index() as u64, value)]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
    }

    for (name, dtype, kind, expected_word) in [
        (
            "f16 empty sum",
            DType::F16,
            crate::ReduceKind::Sum,
            0x0000u16,
        ),
        (
            "bf16 empty mean",
            DType::BF16,
            crate::ReduceKind::Mean,
            0x7fc0u16,
        ),
        (
            "f16 empty product",
            DType::F16,
            crate::ReduceKind::Product,
            0x3c00u16,
        ),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0], dtype);
        let output = graph.reduce(input, kind, Some(vec![0]), false).unwrap();
        let value = TensorData::from_le_bytes([0], dtype, &[]).unwrap();
        let item = schedule(&graph, output).unwrap().items.remove(0);
        let rendered = renderer.render(&item.kernel).unwrap();
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([(input.index() as u64, value)]),
        );
        assert_eq!(actual, expected_word.to_le_bytes(), "{name}");
    }
}

#[test]
fn serial_sum_mean_reductions_match_cpu_and_gate_fp64() {
    let renderer = OpenClRenderer::with_capabilities(8, OpenClCapabilities::FULL).unwrap();
    for (name, kind, shape, axes, keepdim, values) in [
        (
            "multi-axis sum",
            crate::ReduceKind::Sum,
            Shape::from([2, 2, 2]),
            vec![0, 2],
            true,
            vec![1., 2., 3., 4., 5., 6., 7., 8.],
        ),
        (
            "mean",
            crate::ReduceKind::Mean,
            Shape::from([2, 2]),
            vec![1],
            false,
            vec![1., 3., 5., 7.],
        ),
        (
            "empty sum",
            crate::ReduceKind::Sum,
            Shape::from([2, 0]),
            vec![1],
            false,
            vec![],
        ),
        (
            "empty mean",
            crate::ReduceKind::Mean,
            Shape::from([2, 0]),
            vec![1],
            false,
            vec![],
        ),
    ] {
        let mut graph = Graph::new();
        let input = graph.input("input", shape.clone());
        let output = graph
            .reduce(
                input,
                kind,
                Some(axes.into_iter().map(|axis| axis as isize).collect()),
                keepdim,
            )
            .unwrap();
        let value = TensorData::new(shape, values).unwrap();
        let inputs = HashMap::from([("input".into(), value.clone())]);
        let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
        let item = schedule(&graph, output).unwrap().items.remove(0);
        assert!(matches!(
            OpenClRenderer::default().render(&item.kernel),
            Err(OpenClError::Unsupported(_))
        ));
        let rendered = renderer.render(&item.kernel).unwrap();
        rendered
            .validate_schedule_bindings(item.ordered_inputs())
            .unwrap();
        assert!(
            rendered.source.contains("double acc")
                || rendered.source.contains("7fc00000")
                || rendered.source.contains("as_float((uint)0u)"),
            "{name}: {}",
            rendered.source
        );
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([(input.index() as u64, value)]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
    }
}

#[test]
fn serial_product_extrema_match_cpu_raw_storage_contracts() {
    let renderer = OpenClRenderer::with_capabilities(8, OpenClCapabilities::FULL).unwrap();
    let cases = vec![
        (
            "bool product",
            DType::Bool,
            crate::ReduceKind::Product,
            vec![1, 1, 0],
        ),
        (
            "i32 wrapping product",
            DType::I32,
            crate::ReduceKind::Product,
            [
                i32::MAX.to_le_bytes(),
                2i32.to_le_bytes(),
                (-1i32).to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "i64 wrapping product",
            DType::I64,
            crate::ReduceKind::Product,
            [
                i64::MAX.to_le_bytes(),
                2i64.to_le_bytes(),
                (-1i64).to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "u32 minimum",
            DType::U32,
            crate::ReduceKind::Min,
            [
                u32::MAX.to_le_bytes(),
                0u32.to_le_bytes(),
                1u32.to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "u64 f64-projection tie",
            DType::U64,
            crate::ReduceKind::Max,
            [(1u64 << 63).to_le_bytes(), ((1u64 << 63) + 1).to_le_bytes()].concat(),
        ),
        (
            "f32 nan-ignore first signed-zero tie",
            DType::F32,
            crate::ReduceKind::Min,
            [
                f32::NAN.to_le_bytes(),
                (-0.0f32).to_le_bytes(),
                0.0f32.to_le_bytes(),
            ]
            .concat(),
        ),
        (
            "f64 nan-ignore first signed-zero tie",
            DType::F64,
            crate::ReduceKind::Max,
            [
                f64::NAN.to_le_bytes(),
                0.0f64.to_le_bytes(),
                (-0.0f64).to_le_bytes(),
            ]
            .concat(),
        ),
    ];
    let mut keys = BTreeSet::new();
    for (name, dtype, kind, bytes) in cases {
        let elements = bytes.len() / dtype.itemsize();
        let value = TensorData::from_le_bytes([elements], dtype, &bytes).unwrap();
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [elements], dtype);
        let output = graph.reduce(input, kind, Some(vec![0]), false).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), value.clone())]),
            )
            .unwrap();
        let item = schedule(&graph, output).unwrap().items.remove(0);
        let rendered = renderer.render(&item.kernel).unwrap();
        rendered
            .validate_schedule_bindings(item.ordered_inputs())
            .unwrap();
        assert!(keys.insert(rendered.cache_key.clone()), "{name}");
        assert!(rendered.source.contains("for (ulong r"), "{name}");
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([(input.index() as u64, value)]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
    }

    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i32", DType::I32),
        ("u64", DType::U64),
        ("f32", DType::F32),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [0], dtype);
        let output = graph
            .reduce(input, crate::ReduceKind::Product, Some(vec![0]), false)
            .unwrap();
        let value = TensorData::from_le_bytes([0], dtype, &[]).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), value.clone())]),
            )
            .unwrap();
        let item = schedule(&graph, output).unwrap().items.remove(0);
        let rendered = renderer.render(&item.kernel).unwrap();
        rendered
            .validate_schedule_bindings(item.ordered_inputs())
            .unwrap();
        assert_eq!(rendered.buffers.len(), 1, "{name}");
        let (actual, calls) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([(input.index() as u64, value)]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{name}");
        assert!(
            calls.iter().any(|call| call.starts_with("launch:")),
            "{name}"
        );
    }
}

#[test]
fn strided_view_and_capability_preflight_are_exact() {
    let mock = Arc::new(MockDispatch::default());
    let before = mock.calls().len();
    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([3, 3]));
    let view = graph.shrink(input, [(0, 3), (1, 2)]).unwrap();
    let output = graph.neg(view).unwrap();
    let value = TensorData::new([3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let renderer = OpenClRenderer::new(4).unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    assert!(rendered.source.contains("* 3ul"));
    let (actual, _) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([(input.index() as u64, value)]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1], DType::U64);
    let rhs = graph.input_dtype("rhs", [1], DType::U64);
    let output = graph.add(lhs, rhs).unwrap();
    assert!(matches!(
        OpenClRenderer::default().render(&schedule(&graph, output).unwrap().items[0].kernel),
        Err(OpenClError::Unsupported(reason)) if reason.contains("64-bit")
    ));
    assert_eq!(mock.calls().len(), before);

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1], DType::F64);
    let rhs = graph.input_dtype("rhs", [1], DType::F64);
    let output = graph.add(lhs, rhs).unwrap();
    let full = OpenClRenderer::with_capabilities(1, OpenClCapabilities::FULL).unwrap();
    let rendered = full
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let mut f32_graph = Graph::new();
    let f32_input = f32_graph.input("input", [1]);
    let f32_output = f32_graph.neg(f32_input).unwrap();
    let core = OpenClRenderer::new(1)
        .unwrap()
        .render(&schedule(&f32_graph, f32_output).unwrap().items[0].kernel)
        .unwrap();
    let full_f32 = full
        .render(&schedule(&f32_graph, f32_output).unwrap().items[0].kernel)
        .unwrap();
    assert_ne!(full_f32.cache_key, core.cache_key);
    mock.set_device_capabilities(OpenClCapabilities::CORE_32);
    let (context, _) = setup(mock.clone());
    let calls = mock.calls().len();
    assert!(matches!(
        context.cache().load(&rendered, "", 1),
        Err(OpenClError::Unsupported(reason)) if reason.contains("capabilities")
    ));
    assert_eq!(mock.calls().len(), calls);

    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::U64);
    let output = graph
        .reduce(input, crate::ReduceKind::Max, Some(vec![0]), false)
        .unwrap();
    let int64_only = OpenClRenderer::with_capabilities(
        1,
        OpenClCapabilities {
            int64: true,
            fp64: false,
        },
    )
    .unwrap();
    assert!(matches!(
        int64_only.render(&schedule(&graph, output).unwrap().items[0].kernel),
        Err(OpenClError::Unsupported(reason)) if reason.contains("extrema")
    ));
    assert_eq!(mock.calls().len(), calls);
}

#[test]
fn renderer_rejects_unsupported_work_before_icd_calls() {
    let mock = Arc::new(MockDispatch::default());
    let before = mock.calls().len();
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", Shape::from([2]), DType::I32);
    let output = graph.neg(input).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    assert!(OpenClRenderer::default().render(&item.kernel).is_ok());
    assert_eq!(mock.calls().len(), before);

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let rhs = graph.input_dtype("rhs", [2], DType::I32);
    let output = graph.div(lhs, rhs).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let rendered = OpenClRenderer::default().render(&item.kernel).unwrap();
    assert!(rendered.transaction.is_some());
    assert!(rendered.source.contains("atomic_min(rg_status"));
    assert_eq!(mock.calls().len(), before);

    let mut graph = Graph::new();
    let input = graph.input("input", Shape::from([2, 2]));
    let reduced = graph.sum(input, 1).unwrap();
    let item = &schedule(&graph, reduced).unwrap().items[0];
    assert!(matches!(
        OpenClRenderer::default().render(&item.kernel),
        Err(OpenClError::Unsupported(_))
    ));
    assert!(
        OpenClRenderer::with_capabilities(64, OpenClCapabilities::FULL)
            .unwrap()
            .render(&item.kernel)
            .is_ok()
    );
    assert_eq!(mock.calls().len(), before);
}

#[test]
fn guarded_integer_launch_stages_earliest_fault_and_commits_only_success() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [4], DType::I32);
    let rhs = graph.input_dtype("rhs", [4], DType::I32);
    let output = graph.floor_div(lhs, rhs).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let renderer = OpenClRenderer::new(2).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    let transaction = rendered.transaction.as_ref().unwrap();
    assert_eq!(transaction.operation, GuardedIntegerOp::FloorDiv);
    assert!(rendered.source.contains("atomic_min(rg_status"));
    assert!(rendered.source.contains("rustgrad-opencl-static-v3 ABI 3"));

    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let lhs_buffer = context.allocate(16).unwrap();
    let rhs_buffer = context.allocate(16).unwrap();
    let output_buffer = context.allocate(16).unwrap();
    let ints = |values: &[i32]| {
        TensorData::from_scalars(
            [values.len()],
            DType::I32,
            values.iter().map(|&x| Scalar::I(x as i64)),
        )
        .unwrap()
    };
    let lhs_value = ints(&[i32::MIN, -7, 8, 9]);
    queue
        .write(&lhs_buffer, 0, &lhs_value.to_le_bytes().unwrap())
        .unwrap();
    queue.write(&output_buffer, 0, &[0x5a; 16]).unwrap();
    let cache = context.cache();
    let kernel = cache.load(&rendered, "-cl-std=CL1.2", 2).unwrap();
    let bindings = [&lhs_buffer, &rhs_buffer, &output_buffer];

    let bad_rhs = ints(&[-1, 0, 2, 0]);
    queue
        .write(&rhs_buffer, 0, &bad_rhs.to_le_bytes().unwrap())
        .unwrap();
    let token = kernel.launch_transactional(&queue, &bindings).unwrap();
    assert!(!token.query().unwrap());
    assert!(matches!(
        token.wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::FloorDiv,
            index: 1,
            count: None,
            bits: 32,
        })
    ));
    let mut unchanged = vec![0; 16];
    queue.read(&output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, vec![0x5a; 16]);

    let good_rhs = ints(&[-1, 3, 2, -4]);
    queue
        .write(&rhs_buffer, 0, &good_rhs.to_le_bytes().unwrap())
        .unwrap();
    let generation = output_buffer.generation();
    let token = kernel.launch_transactional(&queue, &bindings).unwrap();
    let mut before_collect = vec![0; 16];
    queue.read(&output_buffer, 0, &mut before_collect).unwrap();
    assert_eq!(before_collect, vec![0x5a; 16]);
    assert_eq!(output_buffer.generation(), generation);
    token.wait().unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("lhs".into(), lhs_value), ("rhs".into(), good_rhs)]),
        )
        .unwrap();
    let mut actual = vec![0; 16];
    queue.read(&output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual, expected.to_le_bytes().unwrap());
    assert_eq!(output_buffer.generation(), generation + 1);
    assert!(!mock.calls().iter().any(|call| call.starts_with("copy:")));

    let shared_generation = output_buffer.generation();
    let first = kernel.launch_transactional(&queue, &bindings).unwrap();
    let stale = kernel.launch_transactional(&queue, &bindings).unwrap();
    first.wait().unwrap();
    assert_eq!(output_buffer.generation(), shared_generation + 1);
    assert!(matches!(
        stale.wait(),
        Err(OpenClError::StaleGeneration { expected, actual })
            if expected == shared_generation && actual == shared_generation + 1
    ));
    assert_eq!(output_buffer.generation(), shared_generation + 1);

    let sentinel = [0x3cu8; 16];
    let failure_generation = output_buffer.generation();
    let assert_unchanged = |queue: &OpenClQueue, output: &OpenClBuffer| {
        let mut actual = [0u8; 16];
        queue.read(output, 0, &mut actual).unwrap();
        assert_eq!(actual, sentinel);
    };
    queue.write(&output_buffer, 0, &sentinel).unwrap();
    mock.set_launch_failure(-6);
    assert!(matches!(
        kernel.launch_transactional(&queue, &bindings),
        Err(OpenClError::Driver {
            operation: "launch",
            code: -6
        })
    ));
    assert_unchanged(&queue, &output_buffer);
    assert_eq!(output_buffer.generation(), failure_generation);

    mock.set_wait_failure(-14);
    let token = kernel.launch_transactional(&queue, &bindings).unwrap();
    assert!(matches!(
        token.wait(),
        Err(OpenClError::Driver {
            operation: "wait",
            code: -14
        })
    ));
    assert_unchanged(&queue, &output_buffer);
    assert_eq!(output_buffer.generation(), failure_generation);

    mock.set_read_failure(-5);
    let token = kernel.launch_transactional(&queue, &bindings).unwrap();
    assert!(matches!(
        token.wait(),
        Err(OpenClError::Driver {
            operation: "read",
            code: -5
        })
    ));
    assert_unchanged(&queue, &output_buffer);
    assert_eq!(output_buffer.generation(), failure_generation);

    mock.state.lock().unwrap().failures.buffer_create = Some(-4);
    assert!(matches!(
        kernel.launch_transactional(&queue, &bindings),
        Err(OpenClError::Driver {
            operation: "buffer_create",
            code: -4
        })
    ));
    mock.clear_failures();
    assert_unchanged(&queue, &output_buffer);
    assert_eq!(output_buffer.generation(), failure_generation);

    assert_eq!(
        mock.state.lock().unwrap().buffers.len(),
        3,
        "each terminal transaction releases scratch and status"
    );
}

#[test]
fn visible_generation_releases_old_allocation_after_retained_event() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let rhs = graph.input_dtype("rhs", [2], DType::I32);
    let output = graph.div(lhs, rhs).unwrap();
    let rendered = OpenClRenderer::new(2)
        .unwrap()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let lhs_buffer = context.allocate_typed(2, DType::I32).unwrap();
    let rhs_buffer = context.allocate_typed(2, DType::I32).unwrap();
    let output_buffer = context.allocate_typed(2, DType::I32).unwrap();
    let observer = context.allocate_typed(2, DType::I32).unwrap();
    queue
        .write(&lhs_buffer, 0, &1i32.to_le_bytes().repeat(2))
        .unwrap();
    queue
        .write(&rhs_buffer, 0, &1i32.to_le_bytes().repeat(2))
        .unwrap();
    let kernel = context.cache().load(&rendered, "", 2).unwrap();
    {
        let wrong = context.allocate_typed(2, DType::U32).unwrap();
        let before = mock.calls().len();
        assert!(matches!(
            kernel.launch_transactional(&queue, &[&wrong, &rhs_buffer, &output_buffer]),
            Err(OpenClError::InvalidBinding(reason)) if reason.contains("dtype")
        ));
        assert_eq!(mock.calls().len(), before);
    }
    let retained = queue
        .copy(&output_buffer, &observer, 0, 0, 8)
        .unwrap()
        .unwrap();
    let generation = output_buffer.generation();
    kernel
        .launch_transactional(&queue, &[&lhs_buffer, &rhs_buffer, &output_buffer])
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(output_buffer.generation(), generation + 1);
    assert_eq!(mock.state.lock().unwrap().buffers.len(), 5);
    drop(retained);
    assert_eq!(mock.state.lock().unwrap().buffers.len(), 4);
    let calls = mock.calls();
    let event_release = calls
        .iter()
        .rposition(|call| call.starts_with("event_release:"))
        .unwrap();
    let buffer_release = calls
        .iter()
        .rposition(|call| call.starts_with("buffer_release:"))
        .unwrap();
    assert!(event_release < buffer_release);
}

#[test]
fn guarded_integer_operation_and_width_matrix_matches_cpu() {
    let operations = [
        BinaryOp::Div,
        BinaryOp::FloorDiv,
        BinaryOp::TruncDiv,
        BinaryOp::Mod,
        BinaryOp::FMod,
        BinaryOp::Shl,
        BinaryOp::Shr,
    ];
    for dtype in [DType::I32, DType::U32, DType::I64, DType::U64] {
        for operation in operations {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [4], dtype);
            let rhs = graph.input_dtype("rhs", [4], dtype);
            let output = match operation {
                BinaryOp::Div => graph.div(lhs, rhs),
                BinaryOp::FloorDiv => graph.floor_div(lhs, rhs),
                BinaryOp::TruncDiv => graph.trunc_div(lhs, rhs),
                BinaryOp::Mod => graph.modulo(lhs, rhs),
                BinaryOp::FMod => graph.fmod(lhs, rhs),
                BinaryOp::Shl => graph.shl(lhs, rhs),
                BinaryOp::Shr => graph.shr(lhs, rhs),
                _ => unreachable!(),
            }
            .unwrap();
            let signed = matches!(dtype, DType::I32 | DType::I64);
            let lhs_values = if signed {
                [
                    -9i64,
                    -7,
                    8,
                    if dtype == DType::I32 {
                        i32::MIN as i64
                    } else {
                        i64::MIN
                    },
                ]
                .into_iter()
                .map(Scalar::I)
                .collect::<Vec<_>>()
            } else {
                [9u64, 7, 8, u32::MAX as u64]
                    .into_iter()
                    .map(Scalar::U)
                    .collect::<Vec<_>>()
            };
            let rhs_values = if operation == BinaryOp::Shl || operation == BinaryOp::Shr {
                vec![Scalar::U(1), Scalar::U(2), Scalar::U(3), Scalar::U(1)]
            } else if signed {
                vec![Scalar::I(2), Scalar::I(-3), Scalar::I(2), Scalar::I(-1)]
            } else {
                vec![Scalar::U(2), Scalar::U(3), Scalar::U(2), Scalar::U(1)]
            };
            let lhs_value = TensorData::from_scalars([4], dtype, lhs_values).unwrap();
            let rhs_value = TensorData::from_scalars([4], dtype, rhs_values).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([
                        ("lhs".into(), lhs_value.clone()),
                        ("rhs".into(), rhs_value.clone()),
                    ]),
                )
                .unwrap();
            let renderer = OpenClRenderer::with_capabilities(4, OpenClCapabilities::FULL).unwrap();
            let rendered = renderer
                .render(&schedule(&graph, output).unwrap().items[0].kernel)
                .unwrap();
            let (actual, _) = execute_mock_rendered(
                &rendered,
                renderer,
                &BTreeMap::from([
                    (lhs.index() as u64, lhs_value),
                    (rhs.index() as u64, rhs_value),
                ]),
            );
            assert_eq!(
                actual,
                expected.to_le_bytes().unwrap(),
                "{dtype:?} {operation:?}"
            );
        }
    }

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [0], DType::U32);
    let rhs = graph.input_dtype("rhs", [0], DType::U32);
    let output = graph.div(lhs, rhs).unwrap();
    let renderer = OpenClRenderer::new(1).unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let lhs_buffer = context.allocate(0).unwrap();
    let rhs_buffer = context.allocate(0).unwrap();
    let output_buffer = context.allocate(0).unwrap();
    let kernel = context.cache().load(&rendered, "", 1).unwrap();
    let before = mock.calls();
    kernel
        .launch_transactional(&queue, &[&lhs_buffer, &rhs_buffer, &output_buffer])
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(mock.calls(), before);
}

#[test]
fn guarded_shift_reconstructs_count_through_static_view() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 2], DType::I32);
    let rhs_storage = graph.input_dtype("rhs", [2, 4], DType::I32);
    let rhs = graph.shrink(rhs_storage, [(0, 2), (1, 3)]).unwrap();
    let output = graph.shl(lhs, rhs).unwrap();
    let rendered = OpenClRenderer::new(2)
        .unwrap()
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    assert!(matches!(
        rendered.transaction.as_ref().unwrap().rhs_index,
        UArg::ViewBufferIndex { .. }
    ));
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock);
    let lhs_buffer = context.allocate(16).unwrap();
    let rhs_buffer = context.allocate(32).unwrap();
    let output_buffer = context.allocate(16).unwrap();
    let ints = |values: &[i32]| {
        TensorData::from_scalars(
            [values.len()],
            DType::I32,
            values.iter().map(|&x| Scalar::I(x as i64)),
        )
        .unwrap()
        .to_le_bytes()
        .unwrap()
    };
    queue.write(&lhs_buffer, 0, &ints(&[1, 2, 3, 4])).unwrap();
    queue
        .write(&rhs_buffer, 0, &ints(&[9, 1, 2, 9, 9, -1, 3, 9]))
        .unwrap();
    queue.write(&output_buffer, 0, &[0x77; 16]).unwrap();
    let kernel = context.cache().load(&rendered, "", 2).unwrap();
    assert!(matches!(
        kernel
            .launch_transactional(&queue, &[&lhs_buffer, &rhs_buffer, &output_buffer])
            .unwrap()
            .wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 2,
            count: Some(-1),
            bits: 32,
        })
    ));
    let mut bytes = [0u8; 16];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, [0x77; 16]);
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

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [4], DType::I32);
    let rhs = graph.input_dtype("rhs", [4], DType::I32);
    let output = graph.floor_div(lhs, rhs).unwrap();
    let ints = |values: &[i32]| {
        TensorData::from_scalars(
            [values.len()],
            DType::I32,
            values.iter().map(|&x| Scalar::I(x as i64)),
        )
        .unwrap()
    };
    let lhs_value = ints(&[i32::MIN, -7, 8, 9]);
    let rhs_value = ints(&[-1, 3, 2, -4]);
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                ("lhs".into(), lhs_value.clone()),
                ("rhs".into(), rhs_value.clone()),
            ]),
        )
        .unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let lhs_buffer = context.allocate(16).unwrap();
    let rhs_buffer = context.allocate(16).unwrap();
    let output_buffer = context.allocate(16).unwrap();
    queue
        .write(&lhs_buffer, 0, &lhs_value.to_le_bytes().unwrap())
        .unwrap();
    queue
        .write(&rhs_buffer, 0, &rhs_value.to_le_bytes().unwrap())
        .unwrap();
    queue.write(&output_buffer, 0, &[0x5a; 16]).unwrap();
    let kernel = context.cache().load(&rendered, "-cl-std=CL1.2", 4).unwrap();
    let bindings = [&lhs_buffer, &rhs_buffer, &output_buffer];
    let token = kernel.launch_transactional(&queue, &bindings).unwrap();
    let mut before_collect = vec![0; 16];
    queue.read(&output_buffer, 0, &mut before_collect).unwrap();
    assert_eq!(before_collect, vec![0x5a; 16]);
    token.wait().unwrap();
    let mut bytes = vec![0; 16];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, expected.to_le_bytes().unwrap());
}

#[test]
#[ignore = "requires a live OpenCL ICD and device"]
fn live_opencl_static_view_and_reduction_smoke() {
    let icd = OpenClIcd::load().unwrap();
    let device = icd
        .platforms()
        .unwrap()
        .remove(0)
        .devices()
        .unwrap()
        .remove(0);
    let capabilities = device.info().capabilities;
    let context = device.create_context().unwrap();
    let queue = context.create_queue().unwrap();
    let renderer = OpenClRenderer::with_capabilities(4, capabilities).unwrap();

    let mut graph = Graph::new();
    let input = graph.input("input", [4, 2]);
    let view = graph.shrink(input, [(0, 4), (1, 2)]).unwrap();
    let output = graph.neg(view).unwrap();
    let value = TensorData::new([4, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let input_buffer = context.allocate(32).unwrap();
    let output_buffer = context.allocate(16).unwrap();
    queue
        .write(&input_buffer, 0, &value.to_le_bytes().unwrap())
        .unwrap();
    context
        .cache()
        .load(&rendered, "-cl-std=CL1.2", 4)
        .unwrap()
        .launch(&queue, &[&input_buffer, &output_buffer])
        .unwrap()
        .unwrap()
        .wait()
        .unwrap();
    let mut bytes = vec![0; 16];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, expected.to_le_bytes().unwrap());

    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let output = graph
        .reduce(input, crate::ReduceKind::Max, Some(vec![1]), false)
        .unwrap();
    let value = TensorData::new([2, 2], vec![f32::NAN, -0.0, 0.0, -5.0]).unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([("input".into(), value.clone())]),
        )
        .unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let input_buffer = context.allocate(16).unwrap();
    let output_buffer = context.allocate(8).unwrap();
    queue
        .write(&input_buffer, 0, &value.to_le_bytes().unwrap())
        .unwrap();
    context
        .cache()
        .load(&rendered, "-cl-std=CL1.2", 4)
        .unwrap()
        .launch(&queue, &[&input_buffer, &output_buffer])
        .unwrap()
        .unwrap()
        .wait()
        .unwrap();
    let mut bytes = vec![0; 8];
    queue.read(&output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(bytes, expected.to_le_bytes().unwrap());

    if capabilities.fp64 {
        let mut graph = Graph::new();
        let input = graph.input("input", [2, 2]);
        let output = graph
            .reduce(input, crate::ReduceKind::Product, Some(vec![1]), false)
            .unwrap();
        let value = TensorData::new([2, 2], vec![2., 3., -4., 0.5]).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("input".into(), value.clone())]),
            )
            .unwrap();
        let rendered = renderer
            .render(&schedule(&graph, output).unwrap().items[0].kernel)
            .unwrap();
        let input_buffer = context.allocate(16).unwrap();
        let output_buffer = context.allocate(8).unwrap();
        queue
            .write(&input_buffer, 0, &value.to_le_bytes().unwrap())
            .unwrap();
        context
            .cache()
            .load(&rendered, "-cl-std=CL1.2", 4)
            .unwrap()
            .launch(&queue, &[&input_buffer, &output_buffer])
            .unwrap()
            .unwrap()
            .wait()
            .unwrap();
        let mut bytes = vec![0; 8];
        queue.read(&output_buffer, 0, &mut bytes).unwrap();
        assert_eq!(bytes, expected.to_le_bytes().unwrap());

        for (dtype, words) in [
            (DType::F16, vec![0x8000u16, 0x0001, 0x7c00, 0x7e01]),
            (DType::BF16, vec![0x8000u16, 0x0001, 0x7f80, 0x7fc1]),
        ] {
            let raw = words
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>();
            let value = TensorData::from_le_bytes([4], dtype, &raw).unwrap();
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [4], dtype);
            let output = graph.neg(input).unwrap();
            let expected = CpuBackend
                .execute(
                    &graph,
                    output,
                    &HashMap::from([("input".into(), value.clone())]),
                )
                .unwrap();
            let rendered = renderer
                .render(&schedule(&graph, output).unwrap().items[0].kernel)
                .unwrap();
            let input_buffer = context.allocate(8).unwrap();
            let output_buffer = context.allocate(8).unwrap();
            queue
                .write(&input_buffer, 0, &value.to_le_bytes().unwrap())
                .unwrap();
            context
                .cache()
                .load(&rendered, "-cl-std=CL1.2", 4)
                .unwrap()
                .launch(&queue, &[&input_buffer, &output_buffer])
                .unwrap()
                .unwrap()
                .wait()
                .unwrap();
            let mut bytes = vec![0; 8];
            queue.read(&output_buffer, 0, &mut bytes).unwrap();
            assert_eq!(bytes, expected.to_le_bytes().unwrap(), "{dtype:?}");
        }
    }
}
