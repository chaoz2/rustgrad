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
    read_after: Option<(usize, i32)>,
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
    work_item_order: Option<Vec<usize>>,
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

    fn set_read_failure_after(&self, successful_reads: usize, code: i32) {
        self.state.lock().unwrap().failures.read_after = Some((successful_reads, code));
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

    fn set_work_item_order(&self, order: Vec<usize>) {
        self.state.lock().unwrap().work_item_order = Some(order);
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
        if let Some((remaining, code)) = state.failures.read_after.as_mut() {
            if *remaining == 0 {
                let code = *code;
                state.failures.read_after = None;
                return Err(OpenClError::Driver {
                    operation: "read",
                    code,
                });
            }
            *remaining -= 1;
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
            let mut first = None::<u32>;
            let transaction_extent = transaction.domain.extent()?;
            let order = state
                .work_item_order
                .clone()
                .unwrap_or_else(|| (0..transaction_extent).rev().collect());
            if order.len() != transaction_extent
                || order.iter().copied().collect::<BTreeSet<_>>().len() != transaction_extent
                || order.iter().any(|&index| index >= transaction_extent)
            {
                return Err(OpenClError::InvalidBinding(
                    "mock work-item order is not a permutation".into(),
                ));
            }
            for logical in order {
                let fault =
                    transaction::first_fault_at(transaction, logical, |arg, dtype, logical| {
                        let buffer = match arg {
                            UArg::BufferIndex { buffer, .. }
                            | UArg::ViewBufferIndex { buffer, .. } => *buffer,
                            _ => {
                                return Err(OpenClError::InvalidBinding(
                                    "semantic load index".into(),
                                ));
                            }
                        };
                        let data = bindings.get(buffer).ok_or_else(|| {
                            OpenClError::InvalidBinding("semantic input absent".into())
                        })?;
                        if data.dtype() != dtype {
                            return Err(OpenClError::InvalidBinding(
                                "semantic input dtype mismatch".into(),
                            ));
                        }
                        Ok(data.scalar_at(transaction::logical_offset(arg, logical)?))
                    })?;
                if let Some(id) = fault {
                    let key = transaction.key(logical, id)?;
                    first = Some(first.map_or(key, |old| old.min(key)));
                }
            }
            if let Some(key) = first {
                let status_raw = status_raw.expect("transaction status");
                state
                    .buffers
                    .get_mut(&(owner, status_raw.0))
                    .ok_or(OpenClError::OwnerMismatch)?[..4]
                    .copy_from_slice(&key.to_le_bytes());
                state.calls.push(format!("launch:{owner}:{global}:{local}"));
                return Ok(Self::event(&mut state, owner));
            }
        }
        let result = match semantics.program.as_ref() {
            dispatch::KernelSemanticProgram::UOp(program) => {
                execute_lowered_elementwise(program, &bindings)
                    .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?
            }
            dispatch::KernelSemanticProgram::Random(plan) => plan
                .execute()
                .map_err(|error| OpenClError::InvalidBinding(error.to_string()))?,
        };
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
fn signed_affine_flip_lowers_and_mock_matches_cpu_without_icd_fallback() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3], DType::F32);
    let flipped = graph
        .stride(
            input,
            vec![
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
                crate::Slice {
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
    let renderer = OpenClRenderer::new(8).unwrap();
    let rendered = renderer
        .render(&crate::kernel::lower_graph_elementwise(&graph, output).unwrap())
        .unwrap();
    assert!(rendered.source.contains("long"));
    let (actual, calls) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([(input.index() as u64, tensor.clone())]),
    );
    let expected = CpuBackend
        .execute(&graph, output, &HashMap::from([("x".into(), tensor)]))
        .unwrap()
        .to_le_bytes()
        .unwrap();
    assert_eq!(actual, expected);
    assert!(calls.iter().any(|call| call.starts_with("launch:")));
}

#[test]
fn captured_random_plans_render_and_mock_execute_without_stream_state() {
    let renderer = OpenClRenderer::with_capabilities(32, OpenClCapabilities::FULL).unwrap();
    let mut graph = Graph::new();
    let uniform = graph.uniform([5], -1.25, 2.5, DType::F16, 1337).unwrap();
    let normal = graph.randn([5], DType::BF16, 1338).unwrap();
    let randint = graph.randint([5], -7, 19, DType::I64, 1339).unwrap();
    for output in [uniform, normal, randint] {
        let root = crate::kernel::lower_graph_random(&graph, output).unwrap();
        let rendered = renderer.render(&root).unwrap();
        let UArg::Random(plan) = root.arg() else {
            panic!("missing random plan")
        };
        let expected = plan.execute().unwrap().to_le_bytes().unwrap();
        let (actual, calls) = execute_mock_rendered(&rendered, renderer, &BTreeMap::new());
        assert_eq!(actual, expected, "{:?}", plan.dtype);
        assert_eq!(rendered.buffers.len(), 1);
        assert!(rendered.buffers[0].mutable);
        assert!(rendered.source_map.contains_key(&plan.output.index()));
        assert!(rendered.source.contains("captured-threefry"));
        assert!(rendered.source.contains("ulong chunk=i/maxw"));
        assert!(calls.iter().any(|call| call.starts_with("launch:")));
    }
}

#[test]
fn captured_random_plan_capabilities_zero_domain_and_cache_identity_are_checked() {
    let mut graph = Graph::new();
    let f64 = graph.rand([3], DType::F64, 7).unwrap();
    let empty = graph.rand([0], DType::F32, 8).unwrap();
    let other = graph.rand([3], DType::F32, 9).unwrap();
    let core = OpenClRenderer::new(16).unwrap();
    assert!(matches!(
        core.render(&crate::kernel::lower_graph_random(&graph, f64).unwrap()),
        Err(OpenClError::Unsupported(_))
    ));
    let rendered_empty = core
        .render(&crate::kernel::lower_graph_random(&graph, empty).unwrap())
        .unwrap();
    assert_eq!(rendered_empty.extent, 0);
    let rendered_other = core
        .render(&crate::kernel::lower_graph_random(&graph, other).unwrap())
        .unwrap();
    assert_ne!(rendered_empty.cache_key, rendered_other.cache_key);
    let mock = Arc::new(MockDispatch::default());
    let (context, queue) = setup(mock.clone());
    let cache = context.cache();
    let kernel = cache.load(&rendered_empty, "", 16).unwrap();
    let same = cache.load(&rendered_empty, "", 16).unwrap();
    assert!(Rc::ptr_eq(&kernel, &same));
    let output = context.allocate_typed(0, DType::F32).unwrap();
    assert!(kernel.launch(&queue, &[&output]).unwrap().is_none());
    assert!(!mock.calls().iter().any(|call| call.starts_with("launch:")));
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
    let item = schedule(&graph, output).unwrap().items.remove(0);
    let renderer = OpenClRenderer::with_capabilities(4, OpenClCapabilities::FULL).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    assert!(
        rendered
            .source
            .contains("(bits & 0x7f800000u) == 0x7f800000u")
    );
    assert!(rendered.source.contains("upper | 1u"));
    let (actual, _) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([(input.index() as u64, value)]),
    );
    assert_eq!(
        actual,
        expected
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    );
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

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 2], DType::I32);
    let rhs = graph.input_dtype("rhs", [2, 2], DType::I32);
    let divided = graph.div(lhs, rhs).unwrap();
    let reduced = graph.sum(divided, 1).unwrap();
    let item = &schedule(&graph, reduced).unwrap().items[0];
    let rendered = OpenClRenderer::with_capabilities(1, OpenClCapabilities::FULL)
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    assert!(rendered.transaction.is_some());
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
    assert_eq!(transaction.guards.len(), 1);
    assert_eq!(transaction.guards[0].operation, GuardedIntegerOp::FloorDiv);
    assert!(rendered.source.contains("atomic_min(rg_status"));
    assert!(rendered.source.contains("rustgrad-opencl-static-v4 ABI 4"));

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
fn nested_guards_order_faults_and_reconstruct_computed_shift_counts() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [4], DType::I32);
    let divisor = graph.input_dtype("divisor", [4], DType::I32);
    let count_lhs = graph.input_dtype("count_lhs", [4], DType::I32);
    let count_rhs = graph.input_dtype("count_rhs", [1], DType::I32);
    let quotient = graph.div(lhs, divisor).unwrap();
    let quotient = graph.cast(quotient, DType::U32).unwrap();
    let quotient = graph.cast(quotient, DType::I32).unwrap();
    let count = graph.add(count_lhs, count_rhs).unwrap();
    let shifted = graph.shl(quotient, count).unwrap();
    let output = graph.add(shifted, lhs).unwrap();
    let item = &schedule(&graph, output).unwrap().items[0];
    let renderer = OpenClRenderer::new(2).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    let transaction = rendered.transaction.as_ref().unwrap();
    assert_eq!(transaction.version, OPENCL_TRANSACTION_ABI_VERSION);
    assert_eq!(transaction.guards.len(), 2);
    assert_eq!(transaction.guards[0].operation, GuardedIntegerOp::Div);
    assert_eq!(transaction.guards[1].operation, GuardedIntegerOp::Shl);
    assert!(rendered.source.contains("(uint)gid * 2u + 0u"));
    assert!(rendered.source.contains("(uint)gid * 2u + 1u"));
    assert!(rendered.source.contains("if (rg_ok)"));

    let ints = |values: &[i32]| {
        TensorData::from_scalars(
            [values.len()],
            DType::I32,
            values.iter().map(|&value| Scalar::I(value as i64)),
        )
        .unwrap()
    };
    let scalar = |value: i32| ints(&[value]);
    let mock = Arc::new(MockDispatch::default());
    mock.set_work_item_order(vec![3, 1, 0, 2]);
    let (context, queue) = setup(mock.clone());
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| context.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let positions = rendered
        .buffers
        .iter()
        .enumerate()
        .map(|(index, abi)| (abi.id, index))
        .collect::<BTreeMap<_, _>>();
    let write = |id: u64, value: &TensorData| {
        queue
            .write(&buffers[positions[&id]], 0, &value.to_le_bytes().unwrap())
            .unwrap();
    };
    write(lhs.index() as u64, &ints(&[8, 9, 10, 11]));
    write(divisor.index() as u64, &ints(&[1, 0, 2, 1]));
    write(count_lhs.index() as u64, &ints(&[39, 0, 0, 0]));
    write(count_rhs.index() as u64, &scalar(1));
    let output_buffer = &buffers[transaction.output_abi_index];
    queue.write(output_buffer, 0, &[0x5a; 16]).unwrap();
    let kernel = context.cache().load(&rendered, "", 2).unwrap();
    let refs = buffers.iter().collect::<Vec<_>>();

    // Logical index wins before guard ID: lane zero's second guard precedes
    // lane one's first guard, and its computed RHS is reconstructed exactly.
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 0,
            count: Some(40),
            bits: 32,
        })
    ));
    let mut unchanged = [0; 16];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 16]);

    // Two faults in the same lane select the earlier producer guard.
    write(divisor.index() as u64, &ints(&[0, 1, 2, 1]));
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::Div,
            index: 0,
            count: None,
            bits: 32,
        })
    ));

    // A detail read failure is terminal and cannot expose candidate bytes.
    write(divisor.index() as u64, &ints(&[1, 1, 2, 1]));
    mock.set_read_failure_after(1, -5);
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::Driver {
            operation: "read",
            code: -5,
        })
    ));
    mock.clear_failures();
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 16]);

    // A clean retry commits the same fused DAG and matches the CPU oracle.
    write(count_lhs.index() as u64, &ints(&[0, 1, 2, 0]));
    kernel
        .launch_transactional(&queue, &refs)
        .unwrap()
        .wait()
        .unwrap();
    let inputs = HashMap::from([
        ("lhs".into(), ints(&[8, 9, 10, 11])),
        ("divisor".into(), ints(&[1, 1, 2, 1])),
        ("count_lhs".into(), ints(&[0, 1, 2, 0])),
        ("count_rhs".into(), scalar(1)),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let mut actual = [0; 16];
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected.to_le_bytes().unwrap());
}

#[test]
fn transactional_select_does_not_evaluate_the_unselected_guard() {
    let mut graph = Graph::new();
    let condition = graph.input_dtype("condition", [2], DType::Bool);
    let lhs = graph.input_dtype("lhs", [2], DType::I32);
    let divisor = graph.input_dtype("divisor", [2], DType::I32);
    let count = graph.input_dtype("count", [2], DType::I32);
    let quotient = graph.div(lhs, divisor).unwrap();
    let shifted = graph.shl(lhs, count).unwrap();
    let output = graph.select(condition, quotient, shifted).unwrap();
    let renderer = OpenClRenderer::new(2).unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    assert_eq!(rendered.transaction.as_ref().unwrap().guards.len(), 2);
    assert!(rendered.source.contains("else if (rg_ok)"));
    let values = BTreeMap::from([
        (
            condition.index() as u64,
            TensorData::from_scalars([2], DType::Bool, [Scalar::Bool(false), Scalar::Bool(true)])
                .unwrap(),
        ),
        (
            lhs.index() as u64,
            TensorData::from_scalars([2], DType::I32, [Scalar::I(4), Scalar::I(8)]).unwrap(),
        ),
        (
            divisor.index() as u64,
            TensorData::from_scalars([2], DType::I32, [Scalar::I(0), Scalar::I(2)]).unwrap(),
        ),
        (
            count.index() as u64,
            TensorData::from_scalars([2], DType::I32, [Scalar::I(1), Scalar::I(99)]).unwrap(),
        ),
    ]);
    let (actual, _) = execute_mock_rendered(&rendered, renderer, &values);
    assert_eq!(actual, [8i32.to_le_bytes(), 4i32.to_le_bytes()].concat());
}

#[test]
fn guarded_reductions_order_source_faults_and_swap_only_clean_results() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 3], DType::I32);
    let divisor = graph.input_dtype("divisor", [2, 3], DType::I32);
    let counts = graph.input_dtype("counts", [2, 3], DType::I32);
    let quotient = graph.div(lhs, divisor).unwrap();
    let shifted = graph.shl(quotient, counts).unwrap();
    let output = graph
        .reduce(shifted, crate::ReduceKind::Sum, Some(vec![1]), false)
        .unwrap();
    let renderer = OpenClRenderer::with_capabilities(2, OpenClCapabilities::FULL).unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    let transaction = rendered.transaction.as_ref().unwrap();
    assert!(matches!(
        transaction.domain,
        super::OpenClGuardDomain::ReductionSource { ref shape } if shape == &Shape::from([2, 3])
    ));
    assert_eq!(transaction.guards.len(), 2);
    assert!(rendered.source.contains("(uint)src_gid * 2u + 0u"));
    assert!(rendered.source.contains("(uint)src_gid * 2u + 1u"));
    assert!(rendered.source.contains("if (rg_ok) acc +="));

    let ints = |values: &[i32]| {
        TensorData::from_scalars(
            [2, 3],
            DType::I32,
            values.iter().map(|&value| Scalar::I(value as i64)),
        )
        .unwrap()
    };
    let mock = Arc::new(MockDispatch::default());
    mock.set_work_item_order(vec![4, 5, 3, 2, 1, 0]);
    let (context, queue) = setup(mock.clone());
    let buffers = rendered
        .buffers
        .iter()
        .map(|abi| context.allocate_typed(abi.elements, abi.dtype).unwrap())
        .collect::<Vec<_>>();
    let positions = rendered
        .buffers
        .iter()
        .enumerate()
        .map(|(position, abi)| (abi.id, position))
        .collect::<BTreeMap<_, _>>();
    let write = |id: u64, value: &TensorData| {
        queue
            .write(&buffers[positions[&id]], 0, &value.to_le_bytes().unwrap())
            .unwrap();
    };
    let lhs_value = ints(&[8, 9, 10, 11, 12, 13]);
    write(lhs.index() as u64, &lhs_value);
    let output_buffer = &buffers[transaction.output_abi_index];
    queue.write(output_buffer, 0, &[0x5a; 8]).unwrap();
    let kernel = context.cache().load(&rendered, "", 2).unwrap();
    let refs = buffers.iter().collect::<Vec<_>>();

    // Source position wins globally across different reduction outputs.  The
    // mock visits the second output first, but source two precedes source four.
    write(divisor.index() as u64, &ints(&[1, 1, 0, 1, 0, 1]));
    write(counts.index() as u64, &ints(&[1, 1, 1, 1, 1, 1]));
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::Div,
            index: 2,
            count: None,
            bits: 32,
        })
    ));
    let mut unchanged = [0; 8];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 8]);

    // At one source position producer order breaks the tie between guards.
    write(divisor.index() as u64, &ints(&[1, 0, 1, 1, 1, 1]));
    write(counts.index() as u64, &ints(&[40, 1, 1, 1, 1, 1]));
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::IntegerFault {
            operation: GuardedIntegerOp::Shl,
            index: 0,
            count: Some(40),
            bits: 32,
        })
    ));

    // A failed bounded detail read cannot expose the provisional generation.
    mock.set_read_failure_after(1, -5);
    assert!(matches!(
        kernel.launch_transactional(&queue, &refs).unwrap().wait(),
        Err(OpenClError::Driver {
            operation: "read",
            code: -5,
        })
    ));
    mock.clear_failures();
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, [0x5a; 8]);

    let divisor_value = ints(&[1, 3, 2, 1, 3, 1]);
    let count_value = ints(&[1, 2, 1, 0, 1, 2]);
    write(divisor.index() as u64, &divisor_value);
    write(counts.index() as u64, &count_value);
    let base_generation = output_buffer.generation();
    let clean = kernel.launch_transactional(&queue, &refs).unwrap();
    let stale = kernel.launch_transactional(&queue, &refs).unwrap();
    clean.wait().unwrap();
    assert!(matches!(
        stale.wait(),
        Err(OpenClError::StaleGeneration { expected, actual })
            if expected == base_generation && actual == base_generation + 1
    ));
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                ("lhs".into(), lhs_value),
                ("divisor".into(), divisor_value),
                ("counts".into(), count_value),
            ]),
        )
        .unwrap();
    let mut actual = [0; 8];
    queue.read(output_buffer, 0, &mut actual).unwrap();
    assert_eq!(actual.as_slice(), expected.to_le_bytes().unwrap());
}

#[test]
fn guarded_reduction_kind_and_integer_width_matrix_matches_cpu() {
    let renderer = OpenClRenderer::with_capabilities(3, OpenClCapabilities::FULL).unwrap();
    let mut cache_keys = BTreeSet::new();
    for dtype in [DType::I32, DType::U32, DType::I64, DType::U64] {
        for kind in [
            crate::ReduceKind::Sum,
            crate::ReduceKind::Mean,
            crate::ReduceKind::Product,
            crate::ReduceKind::Min,
            crate::ReduceKind::Max,
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 3], dtype);
            let rhs = graph.input_dtype("rhs", [1], dtype);
            let quotient = graph.div(lhs, rhs).unwrap();
            let output = graph.reduce(quotient, kind, Some(vec![1]), true).unwrap();
            let scalar = |value: u64| {
                if matches!(dtype, DType::I32 | DType::I64) {
                    Scalar::I(value as i64)
                } else {
                    Scalar::U(value)
                }
            };
            let lhs_value = TensorData::from_scalars(
                [2, 3],
                dtype,
                [12, 9, 24, 30, 36, 42].into_iter().map(scalar),
            )
            .unwrap();
            let rhs_value = TensorData::from_scalars([1], dtype, [scalar(3)]).unwrap();
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
            let item = schedule(&graph, output).unwrap().items.remove(0);
            let rendered = renderer.render(&item.kernel).unwrap();
            rendered
                .validate_schedule_bindings(item.ordered_inputs())
                .unwrap();
            assert!(rendered.transaction.is_some(), "{dtype:?} {kind:?}");
            assert!(cache_keys.insert(rendered.cache_key.clone()));
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
                "{dtype:?} {kind:?}"
            );
        }
    }

    // A static view and scalar splat use the source logical index domain once.
    let mut graph = Graph::new();
    let storage = graph.input_dtype("storage", [2, 4], DType::I32);
    let view = graph.shrink(storage, [(0, 2), (1, 4)]).unwrap();
    let rhs = graph.input_dtype("rhs", [1], DType::I32);
    let quotient = graph.div(view, rhs).unwrap();
    let output = graph
        .reduce(quotient, crate::ReduceKind::Product, Some(vec![1]), false)
        .unwrap();
    let storage_value = TensorData::from_scalars(
        [2, 4],
        DType::I32,
        [99, 2, 3, 4, 88, 5, 6, 7].into_iter().map(Scalar::I),
    )
    .unwrap();
    let rhs_value = TensorData::from_scalars([1], DType::I32, [Scalar::I(1)]).unwrap();
    let expected = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                ("storage".into(), storage_value.clone()),
                ("rhs".into(), rhs_value.clone()),
            ]),
        )
        .unwrap();
    let rendered = renderer
        .render(&schedule(&graph, output).unwrap().items[0].kernel)
        .unwrap();
    assert!(rendered.source.contains("src_gid"));
    assert!(rendered.source.contains("* 4ul"));
    let (actual, _) = execute_mock_rendered(
        &rendered,
        renderer,
        &BTreeMap::from([
            (storage.index() as u64, storage_value),
            (rhs.index() as u64, rhs_value),
        ]),
    );
    assert_eq!(actual, expected.to_le_bytes().unwrap());

    for (shape, expect_transaction) in [([2, 0], false), ([0, 3], true)] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", shape, DType::I32);
        let rhs = graph.input_dtype("rhs", shape, DType::I32);
        let quotient = graph.div(lhs, rhs).unwrap();
        let output = graph
            .reduce(quotient, crate::ReduceKind::Sum, Some(vec![1]), false)
            .unwrap();
        let value = TensorData::from_scalars(shape, DType::I32, []).unwrap();
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([("lhs".into(), value.clone()), ("rhs".into(), value.clone())]),
            )
            .unwrap();
        let rendered = renderer
            .render(&schedule(&graph, output).unwrap().items[0].kernel)
            .unwrap();
        assert_eq!(rendered.transaction.is_some(), expect_transaction);
        let (actual, calls) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([
                (lhs.index() as u64, value.clone()),
                (rhs.index() as u64, value),
            ]),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap());
        if shape[0] == 0 {
            assert!(!calls.iter().any(|call| call.starts_with("launch:")));
        }
    }
}

#[test]
fn transactional_logical_and_or_skip_inactive_reduction_guards() {
    for (is_and, condition_values, expected) in [
        (true, [false, true, true, false], false),
        (false, [true, false, false, true], true),
    ] {
        let mut graph = Graph::new();
        let condition = graph.input_dtype("condition", [4], DType::Bool);
        let lhs = graph.input_dtype("lhs", [4], DType::I32);
        let divisor = graph.input_dtype("divisor", [4], DType::I32);
        let quotient = graph.div(lhs, divisor).unwrap();
        let zero =
            graph.constant(TensorData::from_scalars([], DType::I32, [Scalar::I(0)]).unwrap());
        let positive = graph.gt(quotient, zero).unwrap();
        let logical = if is_and {
            graph.logical_and(condition, positive)
        } else {
            graph.logical_or(condition, positive)
        }
        .unwrap();
        let output = graph
            .reduce(logical, crate::ReduceKind::Product, Some(vec![0]), false)
            .unwrap();
        let renderer = OpenClRenderer::new(2).unwrap();
        let rendered = renderer
            .render(&schedule(&graph, output).unwrap().items[0].kernel)
            .unwrap();
        assert_eq!(rendered.transaction.as_ref().unwrap().guards.len(), 1);
        assert!(rendered.source.contains("else if (rg_ok)") || is_and);
        let condition_value = TensorData::from_scalars(
            [4],
            DType::Bool,
            condition_values.into_iter().map(Scalar::Bool),
        )
        .unwrap();
        let lhs_value =
            TensorData::from_scalars([4], DType::I32, [4, 8, 6, 10].into_iter().map(Scalar::I))
                .unwrap();
        // Zero divisors occur only where the left logical operand determines
        // the result, so neither generated code nor semantic detail may touch them.
        let divisor_value =
            TensorData::from_scalars([4], DType::I32, [0, 2, 3, 0].into_iter().map(Scalar::I))
                .unwrap();
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &BTreeMap::from([
                (condition.index() as u64, condition_value),
                (lhs.index() as u64, lhs_value),
                (divisor.index() as u64, divisor_value),
                (
                    zero.index() as u64,
                    TensorData::from_scalars([], DType::I32, [Scalar::I(0)]).unwrap(),
                ),
            ]),
        );
        assert_eq!(actual, vec![u8::from(expected)]);
    }
}

#[test]
fn nested_guarded_integer_widths_match_cpu_oracle() {
    let renderer = OpenClRenderer::with_capabilities(2, OpenClCapabilities::FULL).unwrap();
    let mut keys = BTreeSet::new();
    for dtype in [DType::I32, DType::U32, DType::I64, DType::U64] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], dtype);
        let divisor = graph.input_dtype("divisor", [2], dtype);
        let shifted = graph.input_dtype("shifted", [2], dtype);
        let count = graph.input_dtype("count", [1], dtype);
        let quotient = graph.div(lhs, divisor).unwrap();
        let shifted_value = graph.shl(shifted, count).unwrap();
        let output = graph.add(quotient, shifted_value).unwrap();
        let rendered = renderer
            .render(&schedule(&graph, output).unwrap().items[0].kernel)
            .unwrap();
        assert_eq!(rendered.transaction.as_ref().unwrap().guards.len(), 2);
        assert!(keys.insert(rendered.cache_key.clone()));
        let tensor = |values: &[u64]| {
            TensorData::from_scalars(
                [values.len()],
                dtype,
                values.iter().map(|&value| {
                    if matches!(dtype, DType::I32 | DType::I64) {
                        Scalar::I(value as i64)
                    } else {
                        Scalar::U(value)
                    }
                }),
            )
            .unwrap()
        };
        let values = [
            ("lhs", lhs, tensor(&[8, 12])),
            ("divisor", divisor, tensor(&[2, 3])),
            ("shifted", shifted, tensor(&[1, 2])),
            ("count", count, tensor(&[2])),
        ];
        let expected = CpuBackend
            .execute(
                &graph,
                output,
                &values
                    .iter()
                    .map(|(name, _, value)| ((*name).into(), value.clone()))
                    .collect(),
            )
            .unwrap();
        let (actual, _) = execute_mock_rendered(
            &rendered,
            renderer,
            &values
                .into_iter()
                .map(|(_, node, value)| (node.index() as u64, value))
                .collect(),
        );
        assert_eq!(actual, expected.to_le_bytes().unwrap(), "{dtype:?}");
    }
    assert_eq!(keys.len(), 4);
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
        rendered.transaction.as_ref().unwrap().guards[0]
            .rhs
            .sources()[0]
            .arg(),
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

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 2], DType::I32);
    let rhs = graph.input_dtype("rhs", [2, 2], DType::I32);
    let quotient = graph.div(lhs, rhs).unwrap();
    let output = graph
        .reduce(quotient, crate::ReduceKind::Sum, Some(vec![1]), false)
        .unwrap();
    let lhs_value = TensorData::from_scalars(
        [2, 2],
        DType::I32,
        [8, 9, 10, 12].into_iter().map(Scalar::I),
    )
    .unwrap();
    let rhs_value =
        TensorData::from_scalars([2, 2], DType::I32, [2, 3, 5, 4].into_iter().map(Scalar::I))
            .unwrap();
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
    let output_buffer = context.allocate(8).unwrap();
    queue
        .write(&lhs_buffer, 0, &lhs_value.to_le_bytes().unwrap())
        .unwrap();
    queue
        .write(&rhs_buffer, 0, &rhs_value.to_le_bytes().unwrap())
        .unwrap();
    context
        .cache()
        .load(&rendered, "-cl-std=CL1.2", 4)
        .unwrap()
        .launch_transactional(&queue, &[&lhs_buffer, &rhs_buffer, &output_buffer])
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
