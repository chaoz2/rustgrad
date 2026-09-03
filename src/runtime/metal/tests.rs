use super::renderer::{
    METAL_HOST_GATHER_RENDERER_VERSION, METAL_INDEXED_MOVEMENT_RENDERER_VERSION,
    METAL_PORTABLE_BITCAST_RENDERER_VERSION, METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION,
    METAL_PORTABLE_F32_MATMUL_RENDERER_VERSION, METAL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION,
    METAL_PORTABLE_SORT_RENDERER_VERSION, METAL_PORTABLE_THREEFRY_RENDERER_VERSION,
    METAL_RAW_COPY_RENDERER_VERSION, METAL_RENDERER_VERSION,
    METAL_STATIC_POSITION_RENDERER_VERSION,
};

#[test]
fn captured_scalar_host_gather_is_direct_atomic_and_fail_closed() {
    let mut graph = Graph::new();
    let table = graph.input_dtype("table", [4, 3], DType::F32);
    let token = graph.input_dtype("token", [], DType::I32);
    let token_row = graph.reshape(token, [1, 1]).unwrap();
    let indices = graph.expand(token_row, [1, 3]).unwrap();
    let gathered = graph.gather(table, indices, 0).unwrap();
    let output = graph.square(gathered).unwrap();
    let ordinary =
        CapturedInference::from_module_graph(&IdentityModule, &graph, &[output]).unwrap();
    let frozen_capture = ordinary.capture().to_bytes().unwrap();
    let ordinary_identity = ordinary.deployment_identity();
    let inference = ordinary
        .with_authenticated_host_gathers(&["token"])
        .unwrap();
    assert_eq!(inference.capture().to_bytes().unwrap(), frozen_capture);
    assert_ne!(inference.deployment_identity(), ordinary_identity);

    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let gather_item = inference
        .capture()
        .items
        .iter()
        .find(|item| item.node == gathered)
        .unwrap();
    let ordinary_rendered = renderer.render(&gather_item.kernel).unwrap();
    assert!(ordinary_rendered.indexed_movement().is_some());
    assert!(ordinary_rendered.source.contains("rg_status"));
    assert!(matches!(
        renderer.render_host_gather(
            &gather_item.kernel,
            &crate::runtime::static_schedule::StaticHostGather {
                input: token.index() as u64,
                index: indices.index() as u64,
                output: gathered.index() as u64,
                axis: 0,
                axis_extent: 5,
                index_elements: 3,
            },
        ),
        Err(MetalError::InvalidBinding(_))
    ));

    let (captured, _, _, links, _) = inference.clone().into_parts();
    let mut forged = crate::runtime::static_schedule::StaticHostGather {
        input: links[0].input.desc.id,
        index: links[0].index,
        output: links[0].output,
        axis: links[0].axis,
        axis_extent: links[0].axis_extent,
        index_elements: links[0].index_elements,
    };
    forged.input = table.index() as u64;
    assert!(matches!(
        MetalPrefixPlan::plan_with_output_policy(
            &captured.items,
            &[output.index() as u64],
            &[output.index() as u64],
            &[],
            &[forged],
            renderer.clone(),
        ),
        Err(MetalError::InvalidBinding(_))
    ));

    let plan = MetalInferencePlan::new(inference, renderer).unwrap();
    let direct = plan
        .rendered_items()
        .find(|rendered| rendered.source.contains(METAL_HOST_GATHER_RENDERER_VERSION))
        .expect("authenticated Gather renderer");
    assert!(direct.indexed_movement().is_none());
    assert!(direct.transaction.is_none());
    assert!(!direct.source.contains("rg_status"));
    assert_ne!(direct.cache_key, ordinary_rendered.cache_key);

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut session = plan.prepare(device).unwrap();
    let table_value = TensorData::new(
        [4, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let invocation = |selected: i32| {
        BTreeMap::from([
            ("table".into(), table_value.clone()),
            (
                "token".into(),
                TensorData::from_scalars([], DType::I32, [Scalar::I(i64::from(selected))]).unwrap(),
            ),
        ])
    };
    for invalid in [-1, 4] {
        mock.clear_calls();
        assert!(matches!(
            session.run(&invocation(invalid)),
            Err(MetalError::IndexOutOfBounds {
                axis: 0,
                index: 0,
                value,
                dim: 4,
            }) if value == invalid
        ));
        assert!(mock.calls().is_empty());
        assert_eq!(session.successful_run_count(), 0);
    }
    for stage in ["write", "launch", "wait", "read"] {
        mock.clear_calls();
        let successful = session.successful_run_count();
        {
            let mut state = mock.state.lock().unwrap();
            match stage {
                "write" => state.failures.write = Some("host Gather transient upload"),
                "launch" => state.failures.launch = Some("host Gather launch"),
                "wait" => state.failures.wait = Some("host Gather wait"),
                "read" => state.failures.read = Some("host Gather final read"),
                _ => unreachable!(),
            }
        }
        assert!(session.run(&invocation(2)).is_err());
        assert_eq!(session.successful_run_count(), successful);
        mock.clear_failures();
        mock.clear_calls();
        let run = session.run(&invocation(2)).unwrap();
        assert_eq!(
            run.outputs(),
            &[TensorData::new([1, 3], vec![49.0, 64.0, 81.0]).unwrap()]
        );
        assert_eq!(session.successful_run_count(), successful + 1);
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("read:"))
                .count(),
            1
        );
    }
}

#[test]
fn captured_scalar_host_gather_rejects_ambiguous_transformed_and_public_indices() {
    let capture = |graph: &Graph, output| {
        CapturedInference::from_module_graph(&IdentityModule, graph, &[output]).unwrap()
    };

    let mut wrong_name = Graph::new();
    let table = wrong_name.input_dtype("table", [3, 2], DType::F32);
    let token = wrong_name.input_dtype("token", [], DType::I32);
    let row = wrong_name.reshape(token, [1, 1]).unwrap();
    let indices = wrong_name.expand(row, [1, 2]).unwrap();
    let gathered = wrong_name.gather(table, indices, 0).unwrap();
    let output = wrong_name.square(gathered).unwrap();
    assert!(matches!(
        capture(&wrong_name, output).with_authenticated_host_gathers(&["missing"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("0 authenticated internal owners")
    ));

    let mut ambiguous = Graph::new();
    let table = ambiguous.input_dtype("table", [3, 2], DType::F32);
    let token = ambiguous.input_dtype("token", [], DType::I32);
    let row = ambiguous.reshape(token, [1, 1]).unwrap();
    let indices = ambiguous.expand(row, [1, 2]).unwrap();
    let first = ambiguous.gather(table, indices, 0).unwrap();
    let second = ambiguous.gather(table, indices, 0).unwrap();
    let output = ambiguous.add(first, second).unwrap();
    assert!(matches!(
        capture(&ambiguous, output).with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("2 authenticated internal owners")
    ));

    let mut transformed = Graph::new();
    let table = transformed.input_dtype("table", [3, 2], DType::F32);
    let token = transformed.input_dtype("token", [], DType::I32);
    let one = transformed.constant(TensorData::scalar_with_dtype(Scalar::I(1), DType::I32));
    let changed = transformed.add(token, one).unwrap();
    let row = transformed.reshape(changed, [1, 1]).unwrap();
    let indices = transformed.expand(row, [1, 2]).unwrap();
    let gathered = transformed.gather(table, indices, 0).unwrap();
    let output = transformed.square(gathered).unwrap();
    assert!(matches!(
        capture(&transformed, output).with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("0 authenticated internal owners")
    ));

    let mut multiple = Graph::new();
    let table = multiple.input_dtype("table", [3, 2], DType::F32);
    let token = multiple.input_dtype("token", [2], DType::I32);
    let indices = multiple.reshape(token, [1, 2]).unwrap();
    let gathered = multiple.gather(table, indices, 0).unwrap();
    let output = multiple.square(gathered).unwrap();
    assert!(matches!(
        capture(&multiple, output).with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("one dense scalar I32 transient")
    ));

    let mut wrong_dtype = Graph::new();
    let table = wrong_dtype.input_dtype("table", [3, 2], DType::F32);
    let token = wrong_dtype.input_dtype("token", [], DType::I64);
    let row = wrong_dtype.reshape(token, [1, 1]).unwrap();
    let indices = wrong_dtype.expand(row, [1, 2]).unwrap();
    let gathered = wrong_dtype.gather(table, indices, 0).unwrap();
    let output = wrong_dtype.square(gathered).unwrap();
    assert!(matches!(
        capture(&wrong_dtype, output).with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("one dense scalar I32 transient")
                || reason.contains("0 authenticated internal owners")
    ));

    let position = Parameter::new(
        TensorData::from_scalars([], DType::I32, [Scalar::I(1)]).unwrap(),
        false,
    );
    let resident_name = position.snapshot().unwrap().input_name;
    let module = DirectParameterModule(position.clone());
    let mut resident = Graph::new();
    let table = resident.input_dtype("table", [3, 2], DType::F32);
    let token = position.bind(&mut resident).unwrap();
    let row = resident.reshape(token, [1, 1]).unwrap();
    let indices = resident.expand(row, [1, 2]).unwrap();
    let gathered = resident.gather(table, indices, 0).unwrap();
    let output = resident.square(gathered).unwrap();
    assert!(matches!(
        CapturedInference::from_module_graph(&module, &resident, &[output])
            .unwrap()
            .with_authenticated_host_gathers(&[resident_name.as_str()]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("0 authenticated internal owners")
    ));

    let mut public = Graph::new();
    let table = public.input_dtype("table", [3, 2], DType::F32);
    let token = public.input_dtype("token", [], DType::I32);
    let row = public.reshape(token, [1, 1]).unwrap();
    let indices = public.expand(row, [1, 2]).unwrap();
    let gathered = public.gather(table, indices, 0).unwrap();
    let output = public.square(gathered).unwrap();
    let schedule = crate::schedule_many(&public, &[output, token]).unwrap();
    let captured = CapturedSchedule::capture(&public, &schedule, &[output, token]).unwrap();
    let inference =
        CapturedInference::from_module_graph(&IdentityModule, &public, &[output, token]).unwrap();
    assert_eq!(inference.capture().identity, captured.identity);
    assert!(matches!(
        inference.with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(_))
    ));
}

#[test]
fn captured_scalar_host_gather_zero_domain_is_addressless() {
    let mut graph = Graph::new();
    let table = graph.input_dtype("table", [0, 3], DType::F32);
    let token = graph.input_dtype("token", [], DType::I32);
    let row = graph.reshape(token, [1, 1]).unwrap();
    let indices = graph.expand(row, [0, 3]).unwrap();
    let gathered = graph.gather(table, indices, 0).unwrap();
    let output = graph.square(gathered).unwrap();
    let inference = CapturedInference::from_module_graph(&IdentityModule, &graph, &[output])
        .unwrap()
        .with_authenticated_host_gathers(&["token"])
        .unwrap();
    let plan =
        MetalInferencePlan::new(inference, MetalRenderer::new(8, capabilities()).unwrap()).unwrap();
    assert_eq!(plan.summary().nonzero_item_count, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    assert!(mock.calls().is_empty());
    let run = session
        .run(&BTreeMap::from([
            (
                "table".into(),
                TensorData::from_storage([0, 3], Storage::F32(Vec::new())).unwrap(),
            ),
            (
                "token".into(),
                TensorData::from_scalars([], DType::I32, [Scalar::I(-1)]).unwrap(),
            ),
        ]))
        .unwrap();
    assert!(run.outputs()[0].is_empty());
    assert!(mock.calls().is_empty());
}

#[test]
fn predicated_projected_metal_preserves_raw_lanes_and_guards_addressless_reads() {
    assert_eq!(METAL_RENDERER_VERSION, "rustgrad-metal-static-v8");
    for (shape, values, expected) in [
        (
            Shape::from([2]),
            Storage::F32(vec![-0.0, f32::from_bits(0x7fc0_1234)]),
            vec![0, 0x8000_0000u32, 0x7fc0_1234, 0],
        ),
        (Shape::from([0]), Storage::F32(vec![]), vec![0; 2]),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", shape.clone(), DType::F32);
        let output = graph.pad(input, [(1, 1)], Scalar::F(0.0)).unwrap();
        let output = graph.detach(output).unwrap();
        let item = schedule(&graph, output).unwrap().items.pop().unwrap();
        let renderer = MetalRenderer::new(8, capabilities()).unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        assert!(rendered.source.contains(METAL_RENDERER_VERSION));
        assert!(rendered.source.contains(" ? "));
        let (actual, _) = execute_mock(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_storage(shape, values).unwrap(),
            )]),
        );
        let Storage::F32(actual) = actual.storage() else {
            panic!("predicated Metal F32 output")
        };
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn portable_dense_metal_uses_checked_multi_input_raw_storage_abi() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 2], DType::Bool);
    let rhs = graph.input_dtype("rhs", [1, 1], DType::Bool);
    let output = graph.concat([lhs, rhs, lhs], 1).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert_eq!((rendered.extent, rendered.buffers.len()), (5, 3));
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION)
    );
    assert!(rendered.source.contains("device const uchar* b0"));
    assert!(rendered.source.contains("device uchar* b2"));

    let empty = graph.input_dtype("empty", [0, 2], DType::F32);
    let output = graph.pad(empty, [(0, 0), (1, 0)], Scalar::F(-0.0)).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    assert_eq!(renderer.render(&item.kernel).unwrap().extent, 0);
}

#[test]
fn portable_bitcast_metal_preserves_raw_payload_and_shape_change() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("bytes", [2, 4], DType::U8);
    let output = graph.bitcast(input, DType::U32).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    assert_eq!((rendered.extent, rendered.buffers[1].elements), (8, 2));
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_BITCAST_RENDERER_VERSION)
    );
    let (actual, _) = execute_mock(
        &graph,
        output,
        &HashMap::from([(
            "bytes".into(),
            TensorData::from_storage([2, 4], Storage::U8(vec![1, 2, 3, 4, 0, 0x80, 0xff, 1]))
                .unwrap(),
        )]),
    );
    assert_eq!(
        actual.storage(),
        &Storage::U32(vec![0x0403_0201, 0x01ff_8000])
    );
}

#[test]
fn portable_threefry_metal_renders_and_executes_broadcast_bits() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let counter = graph.input_dtype("counter", [2, 1], DType::U64);
    let key = graph.input_dtype("key", [1, 3], DType::U64);
    let output = graph.threefry(counter, key).unwrap();
    let items = schedule(&graph, output).unwrap().items;
    let rendered = renderer.render(&items[0].kernel).unwrap();
    rendered
        .validate_schedule_bindings(items[0].ordered_inputs())
        .unwrap();
    assert_eq!((rendered.extent, rendered.buffers.len()), (6, 3));
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_THREEFRY_RENDERER_VERSION)
    );
    let counter_value = TensorData::from_storage(
        [2, 1],
        Storage::U64(vec![0x0000_0007_0000_0001, 0x0000_000d_ffff_ffff]),
    )
    .unwrap();
    let key_value = TensorData::from_storage(
        [1, 3],
        Storage::U64(vec![0x0000_0539_0000_0000, 5, 0x0000_0001_ffff_ffff]),
    )
    .unwrap();
    let expected = crate::random::execute_live_threefry(
        &counter_value,
        &key_value,
        graph.shape(output).unwrap(),
    )
    .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock);
    let prefix = PreparedMetalPrefix::prepare(device, &items, renderer).unwrap();
    let mut realized = BTreeMap::from([
        (counter.index() as u64, counter_value),
        (key.index() as u64, key_value),
    ]);
    prefix.execute(&mut realized).unwrap();
    assert_eq!(
        realized[&(output.index() as u64)].to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );

    let mut chained = Graph::new();
    let source = chained.input_dtype("source", [1, 2], DType::U64);
    let viewed = chained.permute(source, [1, 0]).unwrap();
    let counter = chained.contiguous(viewed).unwrap();
    let key = chained.input_dtype("key", [1, 3], DType::U64);
    let output = chained.threefry(counter, key).unwrap();
    let scheduled = schedule(&chained, output).unwrap();
    assert_eq!(scheduled.items.len(), 2);
    assert_eq!(scheduled.items[1].dependencies, vec![scheduled.items[0].id]);
    let source_value = TensorData::from_storage(
        [1, 2],
        Storage::U64(vec![0x0000_0007_0000_0001, 0x0000_000d_ffff_ffff]),
    )
    .unwrap();
    let counter_value = TensorData::from_storage(
        [2, 1],
        Storage::U64(vec![0x0000_0007_0000_0001, 0x0000_000d_ffff_ffff]),
    )
    .unwrap();
    let key_value = TensorData::from_storage([1, 3], Storage::U64(vec![0, 1, 2])).unwrap();
    let expected = crate::random::execute_live_threefry(
        &counter_value,
        &key_value,
        chained.shape(output).unwrap(),
    )
    .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock);
    let prefix = PreparedMetalPrefix::prepare(
        device,
        &scheduled.items,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    let mut realized = BTreeMap::from([
        (source.index() as u64, source_value),
        (key.index() as u64, key_value),
    ]);
    prefix.execute(&mut realized).unwrap();
    assert_eq!(
        realized[&(output.index() as u64)].to_le_bytes().unwrap(),
        expected.to_le_bytes().unwrap()
    );
}
use super::*;
use crate::kernel::execute_lowered_elementwise;
use crate::nn::{Linear, Module, Parameter, StateKind};
use crate::runtime::scalar_lane::emit_scalar_lane;
use crate::{
    Backend, BinaryOp, BufferRole, CapturedAppendStateInference, CapturedInference,
    CapturedMixedBatch, CapturedReplayExecutor, CapturedSchedule, CapturedStatefulInference,
    CompareOp, CpuBackend, CpuSession, DType, EffectBatchStep, EffectRuntime, Graph, IndexValue,
    InferenceAppendStateLink, InferenceStateLink, KernelBindings, KernelBufferDesc,
    LaneInstruction, Mode, MovementKernelKind, MovementValue, NodeId, Operation, ReduceKind,
    ResNet, ResNetConfig, Scalar, Shape, Slice, Storage, TensorData, TypedValue, UType, schedule,
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

fn captured_metal_residual_block() -> (Graph, CapturedSchedule, NodeId, NodeId) {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 4], DType::F32);
    let scale = graph.input_dtype("scale", [2, 4], DType::F32);
    let bias = graph
        .constant(TensorData::new([2, 4], vec![1.0, -1.0, 0.5, 2.0, 3.0, 0.0, -2.0, 1.0]).unwrap());
    let residual_scale = graph.input_dtype("residual_scale", [2, 4], DType::F32);
    let scaled = graph.mul(input, scale).unwrap();
    let shifted = graph.add(scaled, bias).unwrap();
    let hidden = graph.relu(shifted).unwrap();
    let residual = graph.mul(hidden, residual_scale).unwrap();
    let skip = graph.add(residual, input).unwrap();
    let output = graph.relu(skip).unwrap();
    let requested = [output, scale, output];
    let scheduled = crate::schedule_many(&graph, &requested).unwrap();
    let capture = CapturedSchedule::capture(&graph, &scheduled, &requested).unwrap();
    (graph, capture, input, output)
}

fn residual_residents() -> BTreeMap<String, TensorData> {
    BTreeMap::from([
        (
            "scale".into(),
            TensorData::new([2, 4], vec![0.5, 1.0, 1.5, 2.0, -0.5, 0.25, 2.0, 1.0]).unwrap(),
        ),
        (
            "residual_scale".into(),
            TensorData::new([2, 4], vec![1.0, 2.0, 0.5, 0.25, 1.5, 1.0, 0.75, 2.0]).unwrap(),
        ),
    ])
}

struct IdentityModule;

impl Module for IdentityModule {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl crate::nn::ModuleForward for IdentityModule {
    fn forward(&self, _: &mut Graph, input: NodeId) -> crate::Result<NodeId> {
        Ok(input)
    }
}

struct DirectParameterModule(Parameter);

impl Module for DirectParameterModule {
    fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        visitor("value".into(), &self.0, StateKind::Buffer);
    }
}

fn zero_trainable_module_parameters(module: &impl Module) {
    let mut parameters = Vec::new();
    module.visit("", &mut |_, parameter, kind| {
        if matches!(kind, StateKind::Parameter) {
            parameters.push(parameter.clone());
        }
    });
    for parameter in parameters {
        let snapshot = parameter.snapshot().unwrap();
        parameter
            .replace(TensorData::zeros(snapshot.shape).unwrap())
            .unwrap();
    }
}

#[test]
fn metal_device_session_reuses_residents_and_reports_exact_driver_activity() {
    let (graph, capture, _, output) = captured_metal_residual_block();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let plan = MetalDeviceSessionPlan::from_capture(
        CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap(),
        ["scale".into(), "residual_scale".into()],
        renderer,
    )
    .unwrap();
    assert_eq!(plan.summary().capture_identity, capture.identity);
    assert_eq!(plan.resident_inputs().len(), 2);
    assert_eq!(plan.transient_inputs().len(), 1);
    assert_eq!(plan.summary().constant_count, 1);
    assert_eq!(plan.summary().requested_output_count, 3);
    assert_eq!(plan.summary().fallback_count, 0);
    assert!(plan.rendered_items().next().is_some());
    assert_eq!(
        plan.summary().rendered_cache_keys.len(),
        plan.summary().nonzero_item_count
    );

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device, residual_residents()).unwrap();
    assert_eq!(session.device_info().registry_id, 1);
    assert_ne!(session.device_owner_id(), 0);
    let preparation_calls = mock.calls();
    let preparation = session.preparation_report();
    assert_eq!(
        preparation.resident_h2d_calls,
        preparation_calls
            .iter()
            .filter(|call| call.starts_with("write:"))
            .count()
    );
    assert_eq!(
        preparation.pipeline_cache_miss_count,
        preparation_calls
            .iter()
            .filter(|call| call.starts_with("library_compile:"))
            .count()
    );

    for (invocation, input_values) in [
        vec![1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -3.0],
        vec![-4.0, 1.0, 2.0, 0.0, 3.0, -2.0, 1.5, 2.5],
    ]
    .into_iter()
    .enumerate()
    {
        let input_value = TensorData::new([2, 4], input_values).unwrap();
        let mut oracle = HashMap::from([("input".into(), input_value.clone())]);
        oracle.extend(residual_residents());
        let expected = CpuBackend.execute(&graph, output, &oracle).unwrap();
        mock.clear_calls();
        let run = session
            .run(&BTreeMap::from([("input".into(), input_value)]))
            .unwrap();
        assert_eq!(run.outputs().len(), 3);
        assert_eq!(
            run.outputs()[0].to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );
        assert_eq!(
            run.outputs()[2].to_le_bytes().unwrap(),
            expected.to_le_bytes().unwrap()
        );
        assert_eq!(
            run.outputs()[1].to_le_bytes().unwrap(),
            residual_residents()["scale"].to_le_bytes().unwrap()
        );
        let calls = mock.calls();
        assert!(!calls.iter().any(|call| {
            call.starts_with("buffer_create:")
                || call.starts_with("library_compile:")
                || call.starts_with("pipeline_create:")
                || call.starts_with("queue_create:")
        }));
        let report = run.report();
        assert_eq!(report.successful_invocation, invocation as u64 + 1);
        assert_eq!(report.first_successful_run, invocation == 0);
        assert_eq!(report.transient_h2d_calls, 1);
        assert_eq!(report.transient_h2d_bytes, 8 * DType::F32.itemsize());
        assert_eq!(report.retained_d2h_calls, 1);
        assert_eq!(report.retained_d2h_bytes, 8 * DType::F32.itemsize());
        assert_eq!(
            report.kernel_launch_count,
            calls
                .iter()
                .filter(|call| call.starts_with("launch:"))
                .count()
        );
        assert_eq!(
            report.kernel_launch_count,
            session.summary().nonzero_item_count
        );
        assert_eq!(report.output_count, 3);
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("write:"))
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("read:"))
                .count(),
            1
        );
    }
    assert_eq!(session.successful_run_count(), 2);
}

#[test]
fn metal_stateful_inference_commits_only_after_public_projection_and_retries() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("cache", [2], DType::F32);
    let token = graph.input_dtype("token", [2], DType::F32);
    let next = graph.add(state, token).unwrap();
    let public = graph.square(next).unwrap();
    let captured = CapturedStatefulInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[public, public],
        &[InferenceStateLink::new(state, next)],
        BTreeMap::from([(
            "cache".into(),
            TensorData::new([2], vec![1.0, 2.0]).unwrap(),
        )]),
    )
    .unwrap();
    let identity = captured.deployment_identity();
    let plan =
        MetalStatefulInferencePlan::new(captured, MetalRenderer::new(8, capabilities()).unwrap())
            .unwrap();
    assert_eq!(plan.deployment_identity(), identity);
    assert_eq!(plan.state_inputs()[0].name, "cache");
    assert_eq!(plan.summary().state_pair_count, 1);
    assert_eq!(plan.summary().state_bank_count, 2);
    assert_eq!(plan.summary().logical_state_bytes, 8);
    assert_eq!(plan.summary().state_device_bytes, 16);
    assert_eq!(plan.summary().requested_output_count, 2);
    assert_eq!(plan.rendered_items().count(), 2);
    assert_eq!(
        plan.transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["token"]
    );

    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    assert_eq!(session.preparation_report().initial_state_h2d_calls, 1);
    assert_eq!(session.preparation_report().initial_state_h2d_bytes, 8);
    assert!(!session.state_epoch());

    let token = BTreeMap::from([(
        "token".into(),
        TensorData::new([2], vec![2.0, 3.0]).unwrap(),
    )]);
    mock.state.lock().unwrap().failures.read = Some("public read");
    assert!(session.run(&token).is_err());
    assert_eq!(session.successful_run_count(), 0);
    assert!(!session.state_epoch());
    mock.clear_failures();

    let first = session.run(&token).unwrap();
    assert_eq!(
        first.outputs(),
        &[
            TensorData::new([2], vec![9.0, 25.0]).unwrap(),
            TensorData::new([2], vec![9.0, 25.0]).unwrap(),
        ]
    );
    assert_eq!(first.report().retained_d2h_calls, 1);
    assert_eq!(first.report().committed_state_pair_count, 1);
    assert_eq!(first.report().committed_state_bytes, 8);
    assert!(session.state_epoch());

    let second = session
        .run(&BTreeMap::from([(
            "token".into(),
            TensorData::new([2], vec![1.0, 1.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(
        second.outputs()[0],
        TensorData::new([2], vec![16.0, 36.0]).unwrap()
    );
    assert_eq!(session.successful_run_count(), 2);
    assert!(!session.state_epoch());
}

#[test]
fn metal_stateful_inference_zero_work_owns_no_native_resources() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("empty_cache", [0], DType::F32);
    let token = graph.input_dtype("empty_token", [0], DType::F32);
    let next = graph.add(state, token).unwrap();
    let captured = CapturedStatefulInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[token, token],
        &[InferenceStateLink::new(state, next)],
        BTreeMap::from([(
            "empty_cache".into(),
            TensorData::new([0], Vec::new()).unwrap(),
        )]),
    )
    .unwrap();
    let plan =
        MetalStatefulInferencePlan::new(captured, MetalRenderer::new(8, capabilities()).unwrap())
            .unwrap();
    assert_eq!(plan.summary().state_pair_count, 1);
    assert_eq!(plan.summary().logical_state_bytes, 0);
    assert_eq!(plan.summary().state_bank_count, 2);
    assert_eq!(plan.summary().state_device_bytes, 0);
    assert_eq!(plan.summary().planned_device_bytes, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    assert_eq!(
        mock.calls(),
        vec!["devices".to_owned(), "device_release:2".to_owned()]
    );
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    assert!(mock.calls().is_empty());
    let run = session
        .run(&BTreeMap::from([(
            "empty_token".into(),
            TensorData::new([0], Vec::new()).unwrap(),
        )]))
        .unwrap();
    assert_eq!(run.outputs().len(), 2);
    assert_eq!(run.report().kernel_launch_count, 0);
    assert_eq!(run.report().committed_state_pair_count, 1);
    assert_eq!(run.report().committed_state_bytes, 0);
    assert!(session.state_epoch());
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_append_state_is_one_bank_sparse_monotonic_and_retryable() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("cache", [2, 3], DType::F32);
    let position = graph.input_dtype("position", [1, 3], DType::I32);
    let update_source = graph.input_dtype("updates", [1, 3], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, position, updates, 0).unwrap();
    let attention_read = graph.square(next).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[attention_read, attention_read],
        &[InferenceAppendStateLink::new(
            state, next, position, updates, 0,
        )],
        BTreeMap::from([(
            "cache".into(),
            TensorData::new([2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        )]),
    )
    .unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let plan = MetalAppendStateInferencePlan::new(captured, renderer.clone()).unwrap();
    assert_eq!(plan.summary().state_pair_count, 1);
    assert_eq!(plan.summary().state_bank_count, 1);
    assert_eq!(plan.summary().logical_state_bytes, 24);
    assert_eq!(plan.summary().state_device_bytes, 24);
    assert_eq!(plan.summary().append_state_row_bytes, 12);
    assert_eq!(plan.summary().append_state_work_items, 3);
    let append_item = plan
        .rendered_items()
        .find(|rendered| rendered.append_state().is_some())
        .unwrap();
    let append = append_item.append_state().unwrap();
    assert_eq!(append.axis, 0);
    assert_eq!(append.axis_extent, 2);
    assert_eq!(append.row_elements, 3);
    assert!(append_item.source.contains("rg_metal_append_state_f32_i32"));
    assert!(!append_item.source.contains("rg_value = b0[gid]"));
    let ordinary = renderer
        .render(
            &plan
                .capture()
                .items
                .iter()
                .find(|item| {
                    item.outputs
                        .iter()
                        .any(|output| output.id == next.index() as u64)
                })
                .unwrap()
                .kernel,
        )
        .unwrap();
    assert_ne!(append_item.cache_key, ordinary.cache_key);

    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.preparation_report().initial_state_h2d_bytes, 24);

    let invocation = |position_value: i32, updates: Vec<f32>| {
        BTreeMap::from([
            (
                "position".into(),
                TensorData::from_scalars(
                    [1, 3],
                    DType::I32,
                    [Scalar::I(i64::from(position_value)); 3],
                )
                .unwrap(),
            ),
            ("updates".into(), TensorData::new([1, 3], updates).unwrap()),
        ])
    };
    mock.state.lock().unwrap().failures.read = Some("public read");
    assert!(session.run(&invocation(0, vec![1.0, 2.0, 3.0])).is_err());
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.successful_run_count(), 0);
    mock.clear_failures();

    mock.clear_launch_bindings();
    let first = session.run(&invocation(0, vec![4.0, 5.0, 6.0])).unwrap();
    assert_eq!(
        first.outputs(),
        &[
            TensorData::new([2, 3], vec![16.0, 25.0, 36.0, 1600.0, 2500.0, 3600.0]).unwrap(),
            TensorData::new([2, 3], vec![16.0, 25.0, 36.0, 1600.0, 2500.0, 3600.0]).unwrap(),
        ]
    );
    assert_eq!(first.report().committed_state_bytes, 12);
    assert_eq!(first.report().committed_state_work_items, 3);
    assert_eq!(first.report().transient_h2d_calls, 2);
    assert_eq!(first.report().transient_h2d_bytes, 24);
    assert_eq!(first.report().retained_d2h_calls, 1);
    assert_eq!(first.report().retained_d2h_bytes, 24);
    assert_eq!(session.committed_state_position(), Some(1));
    let bindings = mock.launch_bindings();
    let append_ordinal = bindings
        .iter()
        .position(|item| item.len() == 4)
        .expect("append launch has its exact three-input/one-output ABI");
    assert_eq!(bindings[append_ordinal][0], bindings[append_ordinal][3]);
    assert_eq!(bindings[append_ordinal + 1][0], bindings[append_ordinal][0]);

    let before = mock.calls().len();
    assert!(session.run(&invocation(0, vec![7.0, 8.0, 9.0])).is_err());
    assert_eq!(mock.calls().len(), before);
    assert_eq!(session.committed_state_position(), Some(1));

    let second = session.run(&invocation(1, vec![7.0, 8.0, 9.0])).unwrap();
    assert_eq!(
        second.outputs()[0],
        TensorData::new([2, 3], vec![16.0, 25.0, 36.0, 49.0, 64.0, 81.0]).unwrap()
    );
    assert_eq!(session.committed_state_position(), Some(2));
    let before = mock.calls().len();
    assert!(session.run(&invocation(2, vec![10.0, 11.0, 12.0])).is_err());
    assert_eq!(mock.calls().len(), before);
    assert_eq!(session.successful_run_count(), 2);
    assert!(!mock.calls().iter().any(|call| call.starts_with("copy:")));
}

#[test]
fn metal_append_state_kv_links_commit_together_after_partial_failure() {
    let mut graph = Graph::new();
    let key_state = graph.input_dtype("key_cache", [2, 2], DType::F32);
    let value_state = graph.input_dtype("value_cache", [2, 2], DType::F32);
    let position = graph.input_dtype("position", [1, 2], DType::I32);
    let key_source = graph.input_dtype("key_updates", [1, 2], DType::F32);
    let value_source = graph.input_dtype("value_updates", [1, 2], DType::F32);
    let key_updates = graph.relu(key_source).unwrap();
    let value_updates = graph.relu(value_source).unwrap();
    let next_keys = graph.scatter(key_state, position, key_updates, 0).unwrap();
    let next_values = graph
        .scatter(value_state, position, value_updates, 0)
        .unwrap();
    let attention_read = graph.add(next_keys, next_values).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[attention_read],
        &[
            InferenceAppendStateLink::new(key_state, next_keys, position, key_updates, 0),
            InferenceAppendStateLink::new(value_state, next_values, position, value_updates, 0),
        ],
        BTreeMap::from([
            (
                "key_cache".into(),
                TensorData::new([2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
            ),
            (
                "value_cache".into(),
                TensorData::new([2, 2], vec![50.0, 60.0, 70.0, 80.0]).unwrap(),
            ),
        ]),
    )
    .unwrap();
    let plan = MetalAppendStateInferencePlan::new(
        captured,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().state_pair_count, 2);
    assert_eq!(plan.summary().state_bank_count, 1);
    assert_eq!(plan.summary().state_device_bytes, 32);
    assert_eq!(plan.summary().append_state_row_bytes, 16);
    assert_eq!(plan.summary().append_state_work_items, 4);
    let append_launches = plan
        .rendered_items()
        .filter(|item| item.extent != 0)
        .enumerate()
        .filter_map(|(ordinal, item)| item.append_state().map(|_| ordinal))
        .collect::<Vec<_>>();
    assert_eq!(append_launches.len(), 2);

    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    let invocation = |position_value: i32, keys: Vec<f32>, values: Vec<f32>| {
        BTreeMap::from([
            (
                "position".into(),
                TensorData::from_scalars(
                    [1, 2],
                    DType::I32,
                    [Scalar::I(i64::from(position_value)); 2],
                )
                .unwrap(),
            ),
            ("key_updates".into(), TensorData::new([1, 2], keys).unwrap()),
            (
                "value_updates".into(),
                TensorData::new([1, 2], values).unwrap(),
            ),
        ])
    };
    mock.state.lock().unwrap().failures.launch_after = Some((append_launches[1], "second append"));
    assert!(
        session
            .run(&invocation(0, vec![1.0, 2.0], vec![3.0, 4.0]))
            .is_err()
    );
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.successful_run_count(), 0);

    let retry = session
        .run(&invocation(0, vec![5.0, 6.0], vec![7.0, 8.0]))
        .unwrap();
    assert_eq!(
        retry.outputs()[0],
        TensorData::new([2, 2], vec![12.0, 14.0, 100.0, 120.0]).unwrap()
    );
    assert_eq!(retry.report().committed_state_pair_count, 2);
    assert_eq!(retry.report().committed_state_bytes, 16);
    assert_eq!(retry.report().committed_state_work_items, 4);
    assert_eq!(session.committed_state_position(), Some(1));
    assert_eq!(session.successful_run_count(), 1);

    let second = session
        .run(&invocation(1, vec![9.0, 10.0], vec![11.0, 12.0]))
        .unwrap();
    assert_eq!(
        second.outputs()[0],
        TensorData::new([2, 2], vec![12.0, 14.0, 20.0, 22.0]).unwrap()
    );
    assert_eq!(session.committed_state_position(), Some(2));
    assert_eq!(session.successful_run_count(), 2);
    assert!(!mock.calls().iter().any(|call| call.starts_with("copy:")));
}

#[test]
fn metal_append_state_empty_rows_are_addressless_but_advance_logically() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("empty_cache", [2, 0], DType::F32);
    let position = graph.input_dtype("empty_position", [1, 0], DType::I32);
    let update_source = graph.input_dtype("empty_updates", [1, 0], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, position, updates, 0).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[update_source],
        &[InferenceAppendStateLink::new(
            state, next, position, updates, 0,
        )],
        BTreeMap::from([(
            "empty_cache".into(),
            TensorData::new([2, 0], Vec::new()).unwrap(),
        )]),
    )
    .unwrap();
    let plan = MetalAppendStateInferencePlan::new(
        captured,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().state_bank_count, 1);
    assert_eq!(plan.summary().state_device_bytes, 0);
    assert_eq!(plan.summary().append_state_work_items, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    assert!(mock.calls().is_empty());
    let values = BTreeMap::from([
        (
            "empty_position".into(),
            TensorData::from_scalars([1, 0], DType::I32, std::iter::empty()).unwrap(),
        ),
        (
            "empty_updates".into(),
            TensorData::new([1, 0], Vec::new()).unwrap(),
        ),
    ]);
    session.run(&values).unwrap();
    assert_eq!(session.committed_state_position(), Some(1));
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_inference_plan_freezes_module_residents_and_preserves_inspection() {
    let model = Linear::new_static(2, 2, true, 701).unwrap();
    model
        .weight
        .replace(TensorData::new([2, 2], vec![1., -2., 0.5, 3.]).unwrap())
        .unwrap();
    model
        .bias
        .as_ref()
        .unwrap()
        .replace(TensorData::new([2], vec![0.25, -0.75]).unwrap())
        .unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("features", [2, 2], DType::F32);
    let output = model.forward(&mut graph, input).unwrap();
    let input_value = TensorData::new([2, 2], vec![1., 2., -1., 0.5]).unwrap();
    let mut oracle_bindings = model.input_bindings(&graph).unwrap();
    oracle_bindings.insert("features".into(), input_value.clone());
    let expected = CpuBackend
        .execute(&graph, output, &oracle_bindings)
        .unwrap();

    let inference = CapturedInference::from_module_graph(&model, &graph, &[output]).unwrap();
    let deployment_identity = inference.deployment_identity();
    let capture_identity = inference.capture().identity;
    let logical_identity = inference.execution_plan().identity;
    let mock = Arc::new(MockDispatch::default());
    let runtime = MetalRuntime::from_dispatch(mock.clone());
    let device = runtime.device(0).unwrap();
    let renderer = device.renderer(8).unwrap();
    assert_eq!(renderer.local_size, 8);
    assert_eq!(renderer.capabilities, device.info().capabilities);
    mock.clear_calls();
    let plan = MetalInferencePlan::new(inference, renderer).unwrap();
    assert!(mock.calls().is_empty());
    assert_eq!(plan.deployment_identity(), deployment_identity);
    assert_eq!(plan.capture().identity, capture_identity);
    assert_eq!(plan.execution_plan().identity, logical_identity);
    assert_eq!(plan.resident_inputs().len(), 2);
    assert_eq!(
        plan.transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["features"]
    );
    assert_eq!(
        plan.rendered_items().count(),
        plan.summary().nonzero_item_count
    );
    assert_eq!(plan.summary().fallback_count, 0);

    // A deployment owns the values admitted by its graph even when the live
    // module changes before device preparation.
    model
        .weight
        .replace(TensorData::zeros([2, 2]).unwrap())
        .unwrap();
    let mut session = plan.prepare(device).unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 2);
    for invocation in 1..=2 {
        mock.clear_calls();
        let run = session
            .run(&BTreeMap::from([("features".into(), input_value.clone())]))
            .unwrap();
        assert_eq!(run.outputs(), std::slice::from_ref(&expected));
        assert_eq!(run.report().successful_invocation, invocation);
        assert_eq!(run.report().transient_h2d_calls, 1);
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert!(!mock.calls().iter().any(|call| {
            call.starts_with("buffer_create:")
                || call.starts_with("library_compile:")
                || call.starts_with("pipeline_create:")
                || call.starts_with("queue_create:")
        }));
    }
}

#[test]
fn metal_scoreboard_records_only_one_session_success_prefix() {
    let model = Linear::new_static(2, 2, true, 811).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("features", [1, 2], DType::F32);
    let output = model.forward(&mut graph, input).unwrap();
    let inference = CapturedInference::from_module_graph(&model, &graph, &[output]).unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let plan = MetalInferencePlan::new(inference.clone(), renderer.clone()).unwrap();
    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new("linear-1x2", "test-revision", "semantic mock").unwrap(),
        &plan,
    );
    assert_eq!(
        scoreboard.report().unwrap_err(),
        MetalScoreboardError::NotBound
    );

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut session = plan.prepare(device.clone()).unwrap();
    mock.clear_calls();
    scoreboard.bind(&session).unwrap();
    assert!(mock.calls().is_empty());
    assert_eq!(
        scoreboard.bind(&session).unwrap_err(),
        MetalScoreboardError::AlreadyBound
    );
    let empty = scoreboard.report().unwrap();
    assert_eq!(empty.successful_run_count, 0);
    assert_eq!(empty.fallback_count, 0);
    assert_eq!(empty.inputs.len(), 3);
    assert_eq!(
        empty
            .inputs
            .iter()
            .filter(|input| input.kind == MetalScoreboardInputKind::Resident)
            .count(),
        2
    );

    let other_plan = MetalInferencePlan::new(inference, renderer).unwrap();
    let mut other_session = other_plan.prepare(device).unwrap();
    let values = BTreeMap::from([(
        "features".into(),
        TensorData::new([1, 2], vec![2.0, -1.0]).unwrap(),
    )]);
    let other_run = other_session.run(&values).unwrap();
    assert_eq!(
        scoreboard.record(&other_run).unwrap_err(),
        MetalScoreboardError::WrongSession
    );

    let malformed = BTreeMap::new();
    assert!(session.run(&malformed).is_err());
    assert_eq!(scoreboard.report().unwrap().successful_run_count, 0);
    let first = session.run(&values).unwrap();
    scoreboard.record(&first).unwrap();
    assert_eq!(
        scoreboard.record(&first).unwrap_err(),
        MetalScoreboardError::OutOfOrder {
            expected: 2,
            actual: 1,
        }
    );
    assert!(session.run(&malformed).is_err());
    assert_eq!(scoreboard.report().unwrap().successful_run_count, 1);
    let second = session.run(&values).unwrap();
    let third = session.run(&values).unwrap();
    assert_eq!(
        scoreboard.record(&third).unwrap_err(),
        MetalScoreboardError::OutOfOrder {
            expected: 2,
            actual: 3,
        }
    );
    scoreboard.record(&second).unwrap();
    scoreboard.record(&third).unwrap();
    let report = scoreboard.report().unwrap();
    assert_eq!(report.successful_run_count, 3);
    assert_eq!(
        report.first_run_host_wall_time,
        Some(first.report().run_wall_time)
    );
    assert_eq!(
        report.steady_run_host_wall_times,
        vec![second.report().run_wall_time, third.report().run_wall_time]
    );
    assert_eq!(
        report.transient_host_api_h2d_calls,
        first.report().transient_h2d_calls
            + second.report().transient_h2d_calls
            + third.report().transient_h2d_calls
    );
    assert_eq!(
        report.retained_host_api_d2h_bytes,
        first.report().retained_d2h_bytes
            + second.report().retained_d2h_bytes
            + third.report().retained_d2h_bytes
    );
    assert_eq!(
        report.host_api_h2d_calls,
        report.resident_host_api_h2d_calls + report.transient_host_api_h2d_calls
    );
    assert_eq!(
        report.host_api_d2h_calls,
        report.retained_host_api_d2h_calls
    );
    assert_eq!(report.kernel_launch_count, 3 * report.planned_kernel_count);
    assert_eq!(report.fallback_count, 0);
}

#[test]
fn metal_scoreboard_keeps_zero_resource_passthrough_facts_exact() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("empty", [0], DType::F32);
    let output = graph.relu(input).unwrap();
    let inference =
        CapturedInference::from_module_graph(&IdentityModule, &graph, &[output, output]).unwrap();
    let plan =
        MetalInferencePlan::new(inference, MetalRenderer::new(8, capabilities()).unwrap()).unwrap();
    let mut scoreboard = MetalSessionScoreboard::new(
        MetalScoreboardContext::new("empty-identity", "test-revision", "semantic mock").unwrap(),
        &plan,
    );
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    scoreboard.bind(&session).unwrap();
    let run = session
        .run(&BTreeMap::from([(
            "empty".into(),
            TensorData::new([0], Vec::<f32>::new()).unwrap(),
        )]))
        .unwrap();
    scoreboard.record(&run).unwrap();
    let report = scoreboard.report().unwrap();
    assert_eq!(report.planned_static_tensor_slot_count, 0);
    assert_eq!(report.planned_static_tensor_slot_bytes, 0);
    assert_eq!(report.planned_kernel_count, 0);
    assert_eq!(report.planned_zero_item_count, 1);
    assert_eq!(report.kernel_launch_count, 0);
    assert_eq!(report.zero_item_count, 1);
    assert_eq!(report.resident_host_api_h2d_calls, 0);
    assert_eq!(report.transient_host_api_h2d_calls, 0);
    assert_eq!(report.retained_host_api_d2h_calls, 0);
    assert_eq!(report.fallback_count, 0);
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_inference_plan_runs_identity_and_direct_parameter_without_resources() {
    let mut identity_graph = Graph::new();
    let input = identity_graph.input_dtype("features", [2], DType::F32);
    let output =
        crate::nn::ModuleForward::forward(&IdentityModule, &mut identity_graph, input).unwrap();
    let identity =
        CapturedInference::from_module_graph(&IdentityModule, &identity_graph, &[output, output])
            .unwrap();
    assert_eq!(identity.execution_plan().schedule_item_count, 0);
    assert_eq!(identity.execution_plan().requested_outputs.len(), 2);
    let identity_plan =
        MetalInferencePlan::new(identity, MetalRenderer::new(8, capabilities()).unwrap()).unwrap();
    assert_eq!(identity_plan.summary().planned_slot_count, 0);
    assert_eq!(identity_plan.summary().nonzero_item_count, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut identity_session = identity_plan.prepare(device).unwrap();
    assert!(mock.calls().is_empty());
    let input_value = TensorData::new([2], vec![1.25, -3.5]).unwrap();
    let identity_run = identity_session
        .run(&BTreeMap::from([("features".into(), input_value.clone())]))
        .unwrap();
    assert_eq!(identity_run.outputs(), &[input_value.clone(), input_value]);
    assert_eq!(identity_run.report().kernel_launch_count, 0);
    assert_eq!(identity_run.report().transient_h2d_calls, 0);
    assert_eq!(identity_run.report().retained_d2h_calls, 0);
    assert!(mock.calls().is_empty());

    let parameter_value = TensorData::new([2], vec![7.0, -2.0]).unwrap();
    let parameter = DirectParameterModule(Parameter::new(parameter_value.clone(), false));
    let mut parameter_graph = Graph::new();
    let parameter_output = parameter.0.bind(&mut parameter_graph).unwrap();
    let inference =
        CapturedInference::from_module_graph(&parameter, &parameter_graph, &[parameter_output])
            .unwrap();
    assert_eq!(inference.resident_bindings().len(), 1);
    assert!(inference.transient_inputs().is_empty());
    let plan =
        MetalInferencePlan::new(inference, MetalRenderer::new(8, capabilities()).unwrap()).unwrap();
    assert_eq!(plan.summary().planned_device_bytes, 0);
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 0);
    assert!(mock.calls().is_empty());
    let run = session.run(&BTreeMap::new()).unwrap();
    assert_eq!(run.outputs(), std::slice::from_ref(&parameter_value));
    assert_eq!(run.report().kernel_launch_count, 0);
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_device_session_rejects_malformed_calls_before_driver_and_retries_failures() {
    let (_, capture, _, _) = captured_metal_residual_block();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        ["scale".into(), "residual_scale".into()],
        renderer,
    )
    .unwrap();
    let mut session = plan.prepare(device, residual_residents()).unwrap();
    mock.clear_calls();
    for malformed in [
        BTreeMap::new(),
        BTreeMap::from([("extra".into(), TensorData::scalar(1.0))]),
        BTreeMap::from([("input".into(), TensorData::new([1], vec![1.0]).unwrap())]),
    ] {
        assert!(session.run(&malformed).is_err());
        assert!(mock.calls().is_empty());
        assert_eq!(session.successful_run_count(), 0);
    }

    let input = BTreeMap::from([(
        "input".into(),
        TensorData::new([2, 4], vec![1.0; 8]).unwrap(),
    )]);
    for (stage, install) in [("launch", 0usize), ("wait", 1usize), ("read", 2usize)] {
        mock.clear_calls();
        let mut state = mock.state.lock().unwrap();
        match install {
            0 => state.failures.launch = Some(stage),
            1 => state.failures.wait = Some(stage),
            _ => state.failures.read = Some(stage),
        }
        drop(state);
        assert!(session.run(&input).is_err());
        assert_eq!(session.successful_run_count(), install as u64);
        mock.clear_failures();
        let run = session.run(&input).unwrap();
        assert_eq!(run.report().successful_invocation, install as u64 + 1);
        assert_eq!(session.successful_run_count(), install as u64 + 1);
    }
}

#[test]
fn metal_device_session_preparation_is_unpublished_on_validation_or_upload_failure() {
    let (_, capture, _, _) = captured_metal_residual_block();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut tampered = capture.clone();
    tampered.identity ^= 1;
    assert!(matches!(
        MetalDeviceSessionPlan::from_capture(
            tampered,
            ["scale".into(), "residual_scale".into()],
            renderer.clone(),
        ),
        Err(MetalError::InvalidBinding(_))
    ));
    assert!(matches!(
        MetalDeviceSessionPlan::from_capture(capture.clone(), ["unknown".into()], renderer.clone(),),
        Err(MetalError::InvalidBinding(_))
    ));
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let missing = MetalDeviceSessionPlan::from_capture(
        capture.clone(),
        ["scale".into(), "residual_scale".into()],
        renderer.clone(),
    )
    .unwrap();
    assert!(missing.prepare(device.clone(), BTreeMap::new()).is_err());
    assert!(mock.calls().is_empty());

    let extra = MetalDeviceSessionPlan::from_capture(
        capture.clone(),
        ["scale".into(), "residual_scale".into()],
        renderer.clone(),
    )
    .unwrap();
    let mut extra_values = residual_residents();
    extra_values.insert("extra".into(), TensorData::scalar(1.0));
    assert!(extra.prepare(device.clone(), extra_values).is_err());
    assert!(mock.calls().is_empty());

    let malformed = MetalDeviceSessionPlan::from_capture(
        capture.clone(),
        ["scale".into(), "residual_scale".into()],
        renderer.clone(),
    )
    .unwrap();
    let mut malformed_values = residual_residents();
    malformed_values.insert("scale".into(), TensorData::new([1], vec![1.0]).unwrap());
    assert!(malformed.prepare(device.clone(), malformed_values).is_err());
    assert!(mock.calls().is_empty());

    let mut mismatched_capabilities = capabilities();
    mismatched_capabilities.family = "OtherMetalFamily".into();
    let mismatch = MetalDeviceSessionPlan::from_capture(
        capture.clone(),
        ["scale".into(), "residual_scale".into()],
        MetalRenderer::new(8, mismatched_capabilities).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        mismatch.prepare(device.clone(), residual_residents()),
        Err(MetalError::InvalidBinding(_))
    ));
    assert!(mock.calls().is_empty());

    let upload = MetalDeviceSessionPlan::from_capture(
        capture.clone(),
        ["scale".into(), "residual_scale".into()],
        renderer.clone(),
    )
    .unwrap();
    mock.state.lock().unwrap().failures.write = Some("resident upload");
    assert!(
        upload
            .prepare(device.clone(), residual_residents())
            .is_err()
    );
    assert!(
        mock.calls()
            .iter()
            .any(|call| call.starts_with("buffer_release:"))
    );
    mock.clear_failures();

    let retry = MetalDeviceSessionPlan::from_capture(
        capture,
        ["scale".into(), "residual_scale".into()],
        renderer,
    )
    .unwrap();
    let session = retry.prepare(device, residual_residents()).unwrap();
    assert_eq!(session.successful_run_count(), 0);
}

#[test]
fn metal_device_session_zero_work_is_resource_free_and_projects_duplicates() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [0, 4], DType::F32);
    let output = graph.square(input).unwrap();
    let requested = [output, output];
    let scheduled = crate::schedule_many(&graph, &requested).unwrap();
    let capture = CapturedSchedule::capture(&graph, &scheduled, &requested).unwrap();
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        std::iter::empty(),
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().nonzero_item_count, 0);
    assert_eq!(plan.summary().planned_slot_count, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
    assert!(mock.calls().is_empty());
    assert_eq!(session.preparation_report().pipeline_cache_request_count, 0);
    assert_eq!(session.preparation_report().pipeline_cache_hit_count, 0);
    assert_eq!(session.preparation_report().pipeline_cache_miss_count, 0);
    let run = session
        .run(&BTreeMap::from([(
            "input".into(),
            TensorData::new([0, 4], vec![]).unwrap(),
        )]))
        .unwrap();
    assert!(mock.calls().is_empty());
    assert_eq!(run.outputs().len(), 2);
    assert_eq!(run.outputs()[0].shape(), &Shape::from([0, 4]));
    assert_eq!(run.outputs()[0].to_le_bytes().unwrap(), Vec::<u8>::new());
    assert_eq!(
        run.outputs()[0].to_le_bytes().unwrap(),
        run.outputs()[1].to_le_bytes().unwrap()
    );
    assert_eq!(run.report().kernel_launch_count, 0);
    assert_eq!(run.report().transient_h2d_calls, 0);
    assert_eq!(run.report().retained_d2h_calls, 0);
}

#[test]
fn metal_device_session_zero_contraction_uses_pointer_sentinels_without_empty_writes() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 0], DType::F32);
    let rhs = graph.input_dtype("rhs", [0, 3], DType::F32);
    let output = graph.matmul(lhs, rhs).unwrap();
    let scheduled = crate::schedule(&graph, output).unwrap();
    let capture = CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        ["lhs".into(), "rhs".into()],
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().nonzero_item_count, 1);
    assert_eq!(plan.summary().zero_byte_sentinel_count, 2);

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan
        .prepare(
            device,
            BTreeMap::from([
                (
                    "lhs".into(),
                    TensorData::from_storage([2, 0], Storage::F32(Vec::new())).unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_storage([0, 3], Storage::F32(Vec::new())).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 0);
    assert_eq!(session.preparation_report().resident_h2d_bytes, 0);
    assert!(!mock.calls().iter().any(|call| call.starts_with("write:")));

    mock.clear_calls();
    let run = session.run(&BTreeMap::new()).unwrap();
    assert_eq!(run.report().transient_h2d_calls, 0);
    assert_eq!(run.report().kernel_launch_count, 1);
    assert_eq!(run.report().retained_d2h_calls, 1);
    assert_eq!(run.outputs()[0].storage(), &Storage::F32(vec![0.0; 6]));
    let calls = mock.calls();
    assert!(!calls.iter().any(|call| call.starts_with("write:")));
    assert_eq!(
        calls
            .iter()
            .filter(|call| call.starts_with("launch:"))
            .count(),
        1
    );
}

#[test]
fn default_resnet18_is_one_boundary_free_resident_metal_session() {
    let model = ResNet::new_static(ResNetConfig::default(), 19).unwrap();
    zero_trainable_module_parameters(&model);
    let mut graph = Graph::new();
    let image = graph.input_dtype("image", [1, 3, 224, 224], DType::F32);
    let forward = model.forward_mode(&mut graph, image, Mode::Eval).unwrap();
    assert!(forward.pending.is_empty());
    let logits = forward.output.logits().unwrap();
    assert_eq!(graph.shape(logits).unwrap(), &Shape::new([1, 1000]));

    let scheduled = crate::schedule(&graph, logits).unwrap();
    assert!(!scheduled.items.is_empty());
    assert!(scheduled.items.iter().all(|item| item.boundary.is_none()));

    let stem_iteration_elements = 64usize * 112 * 112 * 3 * 7 * 7;
    let stem_projection = scheduled
        .items
        .iter()
        .flat_map(|item| item.kernel.topological().unwrap())
        .filter(crate::projected_index::ProjectedIndexPlan::is_predicated)
        .find_map(|index| {
            let plan = crate::projected_index::ProjectedIndexPlan::from_index(&index).ok()?;
            (plan.buffer == image.index() as u64
                && plan.elements == 3 * 224 * 224
                && plan.output_elements == stem_iteration_elements
                && plan.is_guarded())
            .then_some(plan)
        })
        .expect("stem convolution must retain its authenticated padded input projection");
    assert!(!stem_projection.valid(0).unwrap());

    let maxpool_window_shape = Shape::new([1, 64, 56, 56, 9]);
    let maxpool_windows = scheduled
        .items
        .iter()
        .find(|item| {
            item.primary_output().shape == maxpool_window_shape
                && matches!(
                    item.kernel.operation(),
                    Operation::Movement(MovementValue::Plan(plan))
                        if matches!(&plan.kind, MovementKernelKind::Concat { .. })
                )
        })
        .expect("stem maxpool must materialize its nine source windows");
    let maxpool = scheduled
        .items
        .iter()
        .find(|item| {
            item.primary_output().shape == Shape::new([1, 64, 56, 56])
                && item.dependencies.contains(&maxpool_windows.id)
                && item.kernel.topological().is_ok_and(|nodes| {
                    nodes
                        .iter()
                        .any(|node| matches!(node.operation(), Operation::ReduceFinalize))
                })
        })
        .expect("stem maxpool reduction must depend on its materialized window buffer");
    assert!(maxpool.inputs.iter().any(|input| {
        input.id == maxpool_windows.primary_output().id && input.shape == maxpool_window_shape
    }));

    let capture = CapturedSchedule::capture(&graph, &scheduled, &[logits]).unwrap();
    let capture_identity = capture.identity;
    let capture = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
    assert_eq!(capture.identity, capture_identity);
    let residents = model
        .input_bindings(&graph)
        .unwrap()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let resident_names = residents.keys().cloned().collect::<Vec<_>>();
    assert!(!resident_names.is_empty());
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        resident_names,
        MetalRenderer::new(64, virtual_conformance_capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().capture_identity, capture_identity);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(plan.summary().zero_item_count, 0);
    assert_eq!(plan.rendered_items().count(), scheduled.items.len());
    assert_eq!(
        plan.summary().rendered_cache_keys.len(),
        plan.summary().nonzero_item_count
    );
    assert_eq!(plan.resident_inputs().len(), residents.len());
    assert_eq!(
        plan.transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["image"]
    );

    // Full host evaluation is intentionally excluded here: the semantic mock
    // validates every registered kernel/slot ABI but virtualizes large device
    // allocations. With zero trainable parameters, exact source logits are
    // zero for any finite image, so zero-filled retained output is observable
    // source truth without billions of host convolution operations.
    let mock = Arc::new(MockDispatch::virtual_zero_execution());
    let device = test_device(mock.clone());
    let mut session = plan.prepare(device, residents).unwrap();
    assert!(mock.registered_semantic_program_count() > 0);
    assert!(mock.registered_semantic_program_count() <= session.summary().nonzero_item_count);
    assert_eq!(
        session.compiled_kernels().count(),
        session.summary().nonzero_item_count
    );
    let prepared_owner = session.device_owner_id();
    let planned_slots = session.summary().planned_slot_count;

    let mut observed_bindings = None;
    for invocation in 0..2 {
        mock.clear_calls();
        mock.clear_launch_bindings();
        let run = session
            .run(&BTreeMap::from([(
                "image".into(),
                TensorData::zeros([1, 3, 224, 224]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(run.outputs(), &[TensorData::zeros([1, 1000]).unwrap()]);
        assert_eq!(run.report().successful_invocation, invocation as u64 + 1);
        assert_eq!(
            run.report().kernel_launch_count,
            session.summary().nonzero_item_count
        );
        assert_eq!(run.report().transient_h2d_calls, 1);
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert_eq!(session.device_owner_id(), prepared_owner);
        assert_eq!(session.summary().planned_slot_count, planned_slots);
        assert!(!mock.calls().iter().any(|call| {
            call.starts_with("buffer_create:")
                || call.starts_with("library_compile:")
                || call.starts_with("pipeline_create:")
                || call.starts_with("queue_create:")
        }));
        let bindings = mock.launch_bindings();
        assert_eq!(bindings.len(), session.summary().nonzero_item_count);
        if let Some(previous) = &observed_bindings {
            assert_eq!(&bindings, previous);
        }
        observed_bindings = Some(bindings);
    }
}

#[test]
fn captured_static_metal_preserves_requested_order_and_passthrough_storage() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2], DType::F32);
    let squared = graph.square(input).unwrap();
    let source_value =
        TensorData::from_storage([2], Storage::F32(vec![-0.0, f32::from_bits(0x7fc0_1234)]))
            .unwrap();
    let source = graph.constant(source_value.clone());
    let viewed = graph.permute(source, [0]).unwrap();
    let schedule = crate::schedule_many(&graph, &[viewed, squared]).unwrap();
    let capture = crate::CapturedSchedule::capture(&graph, &schedule, &[viewed, squared]).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let prepared = PreparedMetalPrefix::prepare_capture(device, &capture, renderer).unwrap();
    let outputs = prepared
        .execute(&BTreeMap::from([(
            "x".into(),
            TensorData::new([2], vec![2.0, -3.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].to_le_bytes().unwrap(),
        source_value.to_le_bytes().unwrap()
    );
    assert_eq!(outputs[1].storage(), &Storage::F32(vec![4.0, 9.0]));
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("read:"))
            .count(),
        1
    );
}

#[test]
fn portable_sort_metal_executes_coupled_outputs_and_preserves_storage_bits() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    for dtype in [DType::Bool, DType::I32, DType::U32, DType::F32] {
        let mut matrix = Graph::new();
        let input = matrix.input_dtype("x", [2, 3, 2], dtype);
        let (values, indices) = matrix.sort(input, 1, true).unwrap();
        let item = crate::schedule_many(&matrix, &[values, indices])
            .unwrap()
            .items
            .pop()
            .unwrap();
        let rendered = renderer.render(&item.kernel).unwrap();
        rendered
            .validate_schedule_bindings(item.ordered_inputs())
            .unwrap();
        assert_eq!(rendered.extent, 4);
    }
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3], DType::F32);
    let (values_id, indices_id) = graph.sort(input, 1, false).unwrap();
    let items = crate::schedule_many(&graph, &[values_id, indices_id])
        .unwrap()
        .items;
    let rendered = renderer.render(&items[0].kernel).unwrap();
    rendered
        .validate_schedule_bindings(items[0].ordered_inputs())
        .unwrap();
    assert_eq!((rendered.extent, rendered.buffers.len()), (2, 3));
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_SORT_RENDERER_VERSION)
    );
    let input_value = TensorData::from_storage(
        [2, 3],
        Storage::F32(vec![-0.0, 0.0, f32::NAN, 3.0, 1.0, 1.0]),
    )
    .unwrap();
    let expected = crate::backend::stable_sort_pair(&input_value, 1, false).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock);
    let prefix = PreparedMetalPrefix::prepare(device, &items, renderer.clone()).unwrap();
    let mut realized = BTreeMap::from([(input.index() as u64, input_value)]);
    prefix.execute(&mut realized).unwrap();
    assert_eq!(
        realized[&(values_id.index() as u64)].to_le_bytes().unwrap(),
        expected.0.to_le_bytes().unwrap()
    );
    assert_eq!(
        realized[&(indices_id.index() as u64)]
            .to_le_bytes()
            .unwrap(),
        expected.1.to_le_bytes().unwrap()
    );

    let mut unsupported = Graph::new();
    let input = unsupported.input_dtype("x", [3], DType::F64);
    let (values, indices) = unsupported.sort(input, 0, false).unwrap();
    let item = crate::schedule_many(&unsupported, &[values, indices])
        .unwrap()
        .items
        .pop()
        .unwrap();
    assert!(matches!(
        renderer.render(&item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("Bool/I32/U32/F32")
    ));
}

#[test]
fn portable_prefix_scan_metal_renders_common_matrix_and_executes_first_match_indices() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    for dtype in [DType::Bool, DType::I32, DType::U32, DType::F32] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], dtype);
        let sum = graph.cumsum(input, 1).unwrap();
        let product = graph.cumprod(input, 1).unwrap();
        let (maximum, maximum_indices) = graph.cummax(input, 1).unwrap();
        let (minimum, minimum_indices) = graph.cummin(input, 1).unwrap();
        for output in [
            sum,
            product,
            maximum,
            maximum_indices,
            minimum,
            minimum_indices,
        ] {
            let item = schedule(&graph, output).unwrap().items.pop().unwrap();
            let rendered = renderer.render(&item.kernel).unwrap();
            rendered
                .validate_schedule_bindings(item.ordered_inputs())
                .unwrap();
            assert_eq!(rendered.extent, 2);
            assert_eq!(rendered.buffers.len(), 2);
            assert!(
                rendered
                    .source
                    .contains(METAL_PORTABLE_PREFIX_SCAN_RENDERER_VERSION)
            );
            if output == maximum
                || output == maximum_indices
                || output == minimum
                || output == minimum_indices
            {
                assert!(rendered.source.contains("rg_equal_before"));
            } else {
                assert!(rendered.source.contains("rg_acc ="));
            }
            if dtype == DType::I32 && (output == sum || output == product) {
                assert!(
                    rendered
                        .source
                        .contains("as_type<int>(as_type<uint>(rg_acc)")
                );
            }
            if output == product && dtype == DType::I32 {
                assert!(rendered.source.contains("int rg_acc = (int)1;"));
            }
            if output == product && dtype == DType::U32 {
                assert!(rendered.source.contains("uint rg_acc = 1u;"));
            }
        }
    }

    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3], DType::F32);
    let output = graph.cummax(input, 1).unwrap().1;
    let (actual, _) = execute_mock(
        &graph,
        output,
        &HashMap::from([(
            "x".into(),
            TensorData::from_storage(
                [2, 3],
                Storage::F32(vec![1.0, 3.0, 3.0, f32::NAN, 5.0, 4.0]),
            )
            .unwrap(),
        )]),
    );
    assert_eq!(actual.storage(), &Storage::I32(vec![0, 1, 1, 3, 1, 1]));

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("x", [2], DType::F16);
    let output = narrow.cumsum(input, 0).unwrap();
    let item = schedule(&narrow, output).unwrap().items.pop().unwrap();
    assert!(matches!(
        renderer.render(&item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("Bool/I32/U32/F32")
    ));

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("x", [], DType::F32);
    let output = scalar.cumsum(input, 0).unwrap();
    let item = schedule(&scalar, output).unwrap().items.pop().unwrap();
    assert!(
        renderer
            .render(&item.kernel)
            .unwrap()
            .source
            .contains("as_type<float>(as_type<uint>(b0[0]))")
    );
}

#[test]
fn portable_prefix_scan_metal_zero_domain_skips_buffers_queue_and_launch() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 0, 3], DType::F32);
    let output = graph.cumsum(input, 1).unwrap();
    let items = schedule(&graph, output).unwrap().items;
    let mock = Arc::new(MockDispatch::default());
    let (device, _) = setup(mock.clone());
    let before = mock.calls();
    let prefix = PreparedMetalPrefix::prepare(device, &items, renderer).unwrap();
    let mut values = BTreeMap::from([(
        input.index() as u64,
        TensorData::from_storage([2, 0, 3], Storage::F32(Vec::new())).unwrap(),
    )]);
    prefix.execute(&mut values).unwrap();
    assert!(values[&(output.index() as u64)].is_empty());
    assert!(!mock.calls()[before.len()..].iter().any(|call| {
        call.starts_with("buffer_create:")
            || call.starts_with("queue_create:")
            || call.starts_with("launch:")
    }));
}

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
    launch_after: Option<(usize, &'static str)>,
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
    buffer_lengths: BTreeMap<(u64, usize), usize>,
    libraries: BTreeMap<(u64, usize), String>,
    semantics: BTreeMap<(u64, usize), Arc<KernelSemantics>>,
    commands: BTreeMap<(u64, usize), bool>,
    failures: Failures,
    fault_order: Vec<usize>,
    launch_bindings: Vec<Vec<usize>>,
    virtual_zero_execution: bool,
}

#[derive(Default)]
struct MockDispatch {
    state: Mutex<State>,
}

impl MockDispatch {
    fn virtual_zero_execution() -> Self {
        Self {
            state: Mutex::new(State {
                virtual_zero_execution: true,
                ..State::default()
            }),
        }
    }

    fn calls(&self) -> Vec<String> {
        self.state.lock().unwrap().calls.clone()
    }

    fn clear_calls(&self) {
        self.state.lock().unwrap().calls.clear();
    }

    fn clear_launch_bindings(&self) {
        self.state.lock().unwrap().launch_bindings.clear();
    }

    fn launch_bindings(&self) -> Vec<Vec<usize>> {
        self.state.lock().unwrap().launch_bindings.clone()
    }

    fn registered_semantic_program_count(&self) -> usize {
        self.state.lock().unwrap().semantics.len()
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
        let max_buffer_length = if self.state.lock().unwrap().virtual_zero_execution {
            1usize << 40
        } else {
            1 << 20
        };
        Ok(MetalDeviceInfo {
            name: format!("Mock Metal {}", device.0),
            registry_id: device.0 as u64,
            capabilities: MetalCapabilities {
                max_buffer_length,
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
        state.buffer_lengths.insert((owner, raw.0), bytes);
        let storage = if state.virtual_zero_execution {
            Vec::new()
        } else {
            vec![0; bytes]
        };
        state.buffers.insert((owner, raw.0), storage);
        state
            .calls
            .push(format!("buffer_create:{owner}:{}:{bytes}", raw.0));
        Ok(raw)
    }

    fn buffer_release(&self, buffer: RawBuffer, owner: u64) {
        let mut state = self.state.lock().unwrap();
        state.buffers.remove(&(owner, buffer.0));
        state.buffer_lengths.remove(&(owner, buffer.0));
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
        if state.virtual_zero_execution {
            let limit = *state
                .buffer_lengths
                .get(&(owner, buffer.0))
                .ok_or(MetalError::OwnerMismatch)?;
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > limit)
            {
                return Err(MetalError::Bounds);
            }
            state.calls.push(format!("write:{owner}:{}", buffer.0));
            return Ok(());
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
        if state.virtual_zero_execution {
            let limit = *state
                .buffer_lengths
                .get(&(owner, buffer.0))
                .ok_or(MetalError::OwnerMismatch)?;
            if offset
                .checked_add(bytes.len())
                .is_none_or(|end| end > limit)
            {
                return Err(MetalError::Bounds);
            }
            bytes.fill(0);
            state.calls.push(format!("read:{owner}:{}", buffer.0));
            return Ok(());
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
        if state.virtual_zero_execution {
            let src_len = *state
                .buffer_lengths
                .get(&(owner, src.0))
                .ok_or(MetalError::OwnerMismatch)?;
            let dst_len = *state
                .buffer_lengths
                .get(&(owner, dst.0))
                .ok_or(MetalError::OwnerMismatch)?;
            if region
                .src_offset
                .checked_add(region.bytes)
                .is_none_or(|end| end > src_len)
                || region
                    .dst_offset
                    .checked_add(region.bytes)
                    .is_none_or(|end| end > dst_len)
            {
                return Err(MetalError::Bounds);
            }
            state.calls.push(format!("copy:{owner}"));
            return Ok(Self::command(&mut state, owner));
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
        if let Some((remaining, detail)) = state.failures.launch_after.as_mut() {
            if *remaining == 0 {
                let detail = *detail;
                state.failures.launch_after = None;
                return Err(Self::failure("launch", detail));
            }
            *remaining -= 1;
        }
        let semantics = state
            .semantics
            .get(&(owner, pipeline.0))
            .cloned()
            .ok_or_else(|| MetalError::InvalidBinding("mock semantics absent".into()))?;
        let transaction = semantics.transaction.as_ref();
        let indexed_movement = semantics.indexed_movement.as_ref();
        let expected_buffers = semantics.buffers.len()
            + usize::from(transaction.is_some() || indexed_movement.is_some());
        if geometry.extent as usize != semantics.extent
            || geometry.extent_index != semantics.buffers.len()
            || geometry.local == 0
            || geometry.global < semantics.extent
            || !geometry.global.is_multiple_of(geometry.local)
            || buffers.len() != expected_buffers
        {
            return Err(MetalError::InvalidArgument("invalid mock launch geometry"));
        }
        state
            .launch_bindings
            .push(buffers.iter().map(|buffer| buffer.0).collect());
        if state.virtual_zero_execution {
            if transaction.is_some() {
                return Err(MetalError::InvalidBinding(
                    "virtual zero execution does not admit guarded kernels".into(),
                ));
            }
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
                if state.buffer_lengths.get(&(owner, raw.0)) != Some(&physical) {
                    return Err(MetalError::InvalidBinding(format!(
                        "virtual mock buffer {position} length mismatch"
                    )));
                }
            }
            state.calls.push(format!(
                "launch:{owner}:{}:{}",
                geometry.global, geometry.local
            ));
            return Ok(Self::command(&mut state, owner));
        }
        let mut bindings = KernelBindings::default();
        let mut outputs = Vec::new();
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
                outputs.push((*raw, logical));
            }
        }
        if transaction.is_some() || indexed_movement.is_some() {
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
            let fault_extent =
                indexed_movement.map_or(semantics.extent, |indexed| indexed.index_elements);
            let order = if state.fault_order.is_empty() {
                (0..fault_extent).collect::<Vec<_>>()
            } else {
                state.fault_order.clone()
            };
            let mut status = transaction::CLEAN_STATUS;
            for logical in order {
                if logical >= fault_extent {
                    return Err(MetalError::InvalidBinding(
                        "mock fault order exceeds extent".into(),
                    ));
                }
                if let Some(indexed) = indexed_movement {
                    let (abi, bytes) = stored.get(indexed.index_abi_index).ok_or_else(|| {
                        MetalError::InvalidBinding("mock indexed buffer absent".into())
                    })?;
                    if abi.dtype != DType::I32 || abi.elements != indexed.index_elements {
                        return Err(MetalError::InvalidBinding(
                            "mock indexed buffer descriptor mismatch".into(),
                        ));
                    }
                    let start = logical.checked_mul(4).ok_or(MetalError::Overflow)?;
                    let selected = i32::from_le_bytes(
                        bytes[start..start + 4]
                            .try_into()
                            .map_err(|_| MetalError::Bounds)?,
                    );
                    if selected < 0 || selected as usize >= indexed.axis_extent {
                        status =
                            status.min(u32::try_from(logical).map_err(|_| MetalError::Overflow)?);
                    }
                    continue;
                }
                let transaction = transaction.expect("integer transaction present");
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
        let results = match semantics.program.as_ref() {
            dispatch::KernelSemanticProgram::UOp(program) => match program.operation() {
                crate::Operation::Sort(value) => {
                    let input = bindings.get(value.input.index() as u64).ok_or_else(|| {
                        MetalError::InvalidBinding("sort semantic input absent".into())
                    })?;
                    let (values, indices) = crate::portable_sort::PortableSortPair::new(value)
                        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?
                        .execute(input)
                        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
                    vec![values, indices]
                }
                _ => vec![
                    execute_lowered_elementwise(program, &bindings)
                        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?,
                ],
            },
            dispatch::KernelSemanticProgram::Random(plan) => vec![
                plan.execute()
                    .map_err(|error| MetalError::InvalidBinding(error.to_string()))?,
            ],
        };
        if outputs.is_empty() || results.len() != outputs.len() {
            return Err(MetalError::InvalidBinding(
                "mock output ABI mismatch".into(),
            ));
        }
        for (result, (output, expected)) in results.into_iter().zip(outputs) {
            let result = result
                .to_le_bytes()
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
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
        }
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

fn virtual_conformance_capabilities() -> MetalCapabilities {
    MetalCapabilities {
        max_buffer_length: 1usize << 40,
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
fn indexed_device_selection_and_renderer_policy_create_no_execution_resources() {
    let mock = Arc::new(MockDispatch::default());
    let runtime = MetalRuntime::from_dispatch(mock.clone());
    let device = runtime.device(0).unwrap();
    assert_eq!(device.info().registry_id, 1);
    let renderer = device.renderer(8).unwrap();
    assert_eq!(renderer.local_size, 8);
    assert_eq!(renderer.capabilities, device.info().capabilities);
    assert!(matches!(
        device.renderer(0),
        Err(MetalError::InvalidArgument("zero local size"))
    ));
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("queue_create:")
            || call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
    }));

    mock.clear_calls();
    assert!(matches!(
        runtime.device(2),
        Err(MetalError::InvalidArgument("device index is out of range"))
    ));
    assert!(mock.calls().contains(&"devices".into()));
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("queue_create:")
            || call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
    }));
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
            let buffer = device
                .allocate_static(crate::runtime::static_schedule::StaticBufferAllocation {
                    elements: abi.elements,
                    bytes: abi.elements * abi.dtype.itemsize(),
                    dtype: abi.dtype,
                    requires_native_handle: rendered.extent != 0,
                })
                .unwrap();
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
    let transactional = rendered.transaction.is_some() || rendered.indexed_movement().is_some();
    let completion = if transactional {
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
        rendered.buffers.len() + usize::from(transactional) * 2
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
    let rendered = renderer.render(&padded_item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(padded_item.ordered_inputs())
        .unwrap();
    assert!(
        rendered
            .source
            .contains(METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION)
    );
    assert_eq!((rendered.extent, rendered.buffers.len()), (4, 2));

    let unsupported_base = graph.input_dtype("unsupported_base", [2], DType::U64);
    let unsupported_index = graph.input_dtype("unsupported_index", [2], DType::I32);
    let gathered = graph
        .gather(unsupported_base, unsupported_index, 0)
        .unwrap();
    let gathered_item = schedule(&graph, gathered).unwrap().items.pop().unwrap();
    assert!(matches!(
        renderer.render(&gathered_item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("F32 values and I32 indices")
    ));
}

#[test]
fn indexed_movement_metal_is_checked_f32_i32_and_row_major() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut gather_graph = Graph::new();
    let input = gather_graph.input_dtype("input", [2, 3], DType::F32);
    let index = gather_graph.input_dtype("index", [2, 2], DType::I32);
    let output = gather_graph.gather(input, index, 1).unwrap();
    let item = schedule(&gather_graph, output)
        .unwrap()
        .items
        .pop()
        .unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    rendered
        .validate_schedule_bindings(item.ordered_inputs())
        .unwrap();
    let mut malformed_bindings = item.ordered_inputs().to_vec();
    malformed_bindings[0].desc.id ^= 1;
    assert!(matches!(
        rendered.validate_schedule_bindings(&malformed_bindings),
        Err(MetalError::InvalidBinding(_))
    ));
    let indexed = rendered.indexed_movement().unwrap();
    assert_eq!(indexed.version, METAL_INDEXED_MOVEMENT_ABI_VERSION);
    assert_eq!(
        (indexed.axis, indexed.axis_extent, indexed.index_elements),
        (1, 3, 4)
    );
    assert_eq!((rendered.extent, rendered.buffers.len()), (4, 3));
    assert!(
        rendered
            .source
            .contains(METAL_INDEXED_MOVEMENT_RENDERER_VERSION)
    );
    assert!(rendered.source.contains("atomic_fetch_min_explicit"));
    assert!(rendered.source.contains("device const int* b1"));
    assert_eq!(
        renderer.render(&item.kernel).unwrap().cache_key,
        rendered.cache_key
    );
    let (actual, _) = execute_mock(
        &gather_graph,
        output,
        &HashMap::from([
            (
                "input".into(),
                TensorData::new([2, 3], vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0]).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_storage([2, 2], Storage::I32(vec![2, 0, 1, 1])).unwrap(),
            ),
        ]),
    );
    assert_eq!(
        actual,
        TensorData::new([2, 2], vec![12.0, 10.0, 21.0, 21.0]).unwrap()
    );

    for add in [false, true] {
        let mut graph = Graph::new();
        let base = graph.input_dtype("base", [1, 4], DType::F32);
        let index = graph.input_dtype("index", [1, 3], DType::I32);
        let updates = graph.input_dtype("updates", [1, 3], DType::F32);
        let output = if add {
            graph.scatter_add(base, index, updates, 1).unwrap()
        } else {
            graph.scatter(base, index, updates, 1).unwrap()
        };
        let (actual, _) = execute_mock(
            &graph,
            output,
            &HashMap::from([
                (
                    "base".into(),
                    TensorData::new([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ),
                (
                    "index".into(),
                    TensorData::from_storage([1, 3], Storage::I32(vec![2, 1, 2])).unwrap(),
                ),
                (
                    "updates".into(),
                    TensorData::new([1, 3], vec![10.0, 20.0, 30.0]).unwrap(),
                ),
            ]),
        );
        let expected = if add {
            vec![1.0, 22.0, 43.0, 4.0]
        } else {
            vec![1.0, 20.0, 30.0, 4.0]
        };
        assert_eq!(actual, TensorData::new([1, 4], expected).unwrap());
    }

    let mut graph = Graph::new();
    let base = graph.input_dtype("base", [1], DType::F32);
    let index = graph.input_dtype("index", [2], DType::I32);
    let updates = graph.input_dtype("updates", [2], DType::F32);
    let output = graph.scatter_add(base, index, updates, 0).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    assert!(rendered.source.contains("float rg_value = b0[gid]"));
    assert!(rendered.source.contains("rg_value +="));
    let (actual, _) = execute_mock(
        &graph,
        output,
        &HashMap::from([
            (
                "base".into(),
                TensorData::new([1], vec![100_000_000.0]).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_storage([2], Storage::I32(vec![0, 0])).unwrap(),
            ),
            (
                "updates".into(),
                TensorData::new([2], vec![1.0, -100_000_000.0]).unwrap(),
            ),
        ]),
    );
    assert_eq!(actual, TensorData::new([1], vec![0.0]).unwrap());
}

#[test]
fn indexed_movement_metal_rejects_bad_indices_before_publication_and_retries() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [3], DType::F32);
    let index = graph.input_dtype("index", [2], DType::I32);
    let output = graph.gather(input, index, 0).unwrap();
    let item = schedule(&graph, output).unwrap().items.pop().unwrap();
    let rendered = MetalRenderer::new(2, capabilities())
        .unwrap()
        .render(&item.kernel)
        .unwrap();
    let indexed = rendered.indexed_movement().unwrap();
    let mock = Arc::new(MockDispatch::default());
    let (device, queue) = setup(mock.clone());
    let values = BTreeMap::from([
        (
            input.index() as u64,
            TensorData::new([3], vec![4.0, 5.0, 6.0]).unwrap(),
        ),
        (
            index.index() as u64,
            TensorData::from_storage([2], Storage::I32(vec![3, -1])).unwrap(),
        ),
    ]);
    mock.state.lock().unwrap().fault_order = vec![1, 0];
    let buffers = allocate_rendered(&device, &queue, &rendered, &values);
    let refs = buffers.iter().collect::<Vec<_>>();
    let output_buffer = &buffers[indexed.output_abi_index];
    let sentinel = [0x5a; 8];
    queue.write(output_buffer, 0, &sentinel).unwrap();
    let generation = output_buffer.generation();
    let pipeline = device.cache().load(&rendered).unwrap();
    assert!(matches!(
        pipeline.launch(&queue, &refs, 2),
        Err(MetalError::InvalidArgument(
            "guarded kernel requires transactional launch"
        ))
    ));
    assert_eq!(
        pipeline
            .launch_transactional(&queue, &refs, 2)
            .unwrap()
            .wait()
            .unwrap_err(),
        MetalError::IndexOutOfBounds {
            axis: 0,
            index: 0,
            value: 3,
            dim: 3,
        }
    );
    let mut unchanged = [0; 8];
    queue.read(output_buffer, 0, &mut unchanged).unwrap();
    assert_eq!(unchanged, sentinel);
    assert_eq!(output_buffer.generation(), generation);

    queue
        .write(
            &buffers[indexed.index_abi_index],
            0,
            &TensorData::from_storage([2], Storage::I32(vec![1, 2]))
                .unwrap()
                .to_le_bytes()
                .unwrap(),
        )
        .unwrap();
    pipeline
        .launch_transactional(&queue, &refs, 2)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(output_buffer.generation(), generation + 1);
    let mut bytes = [0; 8];
    queue.read(output_buffer, 0, &mut bytes).unwrap();
    assert_eq!(
        bytes,
        [5.0f32.to_le_bytes(), 6.0f32.to_le_bytes()]
            .concat()
            .as_slice()
    );
}

#[test]
fn captured_indexed_movement_metal_is_atomic_and_projects_duplicate_outputs() {
    let mut graph = Graph::new();
    let source = graph.input_dtype("source", [3], DType::F32);
    let input = graph.square(source).unwrap();
    let index = graph.input_dtype("index", [2], DType::I32);
    let output = graph.gather(input, index, 0).unwrap();
    let requested = [output, output];
    let scheduled = crate::schedule_many(&graph, &requested).unwrap();
    let capture = CapturedSchedule::capture(&graph, &scheduled, &requested).unwrap();
    let capture = CapturedSchedule::from_bytes(&capture.to_bytes().unwrap()).unwrap();
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        std::iter::empty(),
        MetalRenderer::new(2, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().requested_output_count, 2);
    assert_eq!(plan.rendered_items().count(), 2);
    assert!(
        plan.rendered_items()
            .any(|rendered| rendered.indexed_movement().is_some())
    );
    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock), BTreeMap::new()).unwrap();
    let input_value = TensorData::new([3], vec![4.0, 5.0, 6.0]).unwrap();
    let bad = BTreeMap::from([
        ("source".into(), input_value.clone()),
        (
            "index".into(),
            TensorData::from_storage([2], Storage::I32(vec![3, 0])).unwrap(),
        ),
    ]);
    assert_eq!(
        session.run(&bad).err().expect("invalid index must fail"),
        MetalError::IndexOutOfBounds {
            axis: 0,
            index: 0,
            value: 3,
            dim: 3,
        }
    );
    assert_eq!(session.successful_run_count(), 0);
    let run = session
        .run(&BTreeMap::from([
            ("source".into(), input_value),
            (
                "index".into(),
                TensorData::from_storage([2], Storage::I32(vec![2, 0])).unwrap(),
            ),
        ]))
        .unwrap();
    assert_eq!(session.successful_run_count(), 1);
    assert_eq!(run.outputs().len(), 2);
    assert_eq!(
        run.outputs()[0],
        TensorData::new([2], vec![36.0, 16.0]).unwrap()
    );
    assert_eq!(run.outputs()[0], run.outputs()[1]);
    assert_eq!(run.report().kernel_launch_count, 2);
}

#[test]
fn indexed_movement_metal_zero_domain_is_resource_free_and_dtypes_fail_closed() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 4], DType::F32);
    let index = empty.input_dtype("index", [0, 2], DType::I32);
    let output = empty.gather(input, index, 1).unwrap();
    let requested = [output, output];
    let scheduled = crate::schedule_many(&empty, &requested).unwrap();
    let capture = CapturedSchedule::capture(&empty, &scheduled, &requested).unwrap();
    let plan = MetalDeviceSessionPlan::from_capture(capture, std::iter::empty(), renderer.clone())
        .unwrap();
    assert_eq!(plan.summary().nonzero_item_count, 0);
    assert_eq!(plan.summary().zero_item_count, 1);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
    assert!(mock.calls().is_empty());
    let run = session
        .run(&BTreeMap::from([
            (
                "input".into(),
                TensorData::from_storage([0, 4], Storage::F32(Vec::new())).unwrap(),
            ),
            (
                "index".into(),
                TensorData::from_storage([0, 2], Storage::I32(Vec::new())).unwrap(),
            ),
        ]))
        .unwrap();
    assert!(mock.calls().is_empty());
    assert_eq!(run.outputs().len(), 2);
    assert!(run.outputs()[0].is_empty());
    assert_eq!(run.report().kernel_launch_count, 0);

    let mut i64_indexed = Graph::new();
    let input = i64_indexed.input_dtype("input", [1], DType::F32);
    let index = i64_indexed.input_dtype("index", [1], DType::I64);
    let output = i64_indexed.gather(input, index, 0).unwrap();
    let item = schedule(&i64_indexed, output).unwrap().items.pop().unwrap();
    assert!(matches!(
        renderer.render(&item.kernel),
        Err(MetalError::Unsupported(reason)) if reason.contains("F32 values and I32 indices")
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
