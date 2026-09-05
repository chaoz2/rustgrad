use super::renderer::{
    METAL_FIXED_HOST_GATHER_RENDERER_VERSION, METAL_HOST_GATHER_RENDERER_VERSION,
    METAL_INDEXED_MOVEMENT_RENDERER_VERSION, METAL_PORTABLE_BITCAST_RENDERER_VERSION,
    METAL_PORTABLE_DENSE_MATERIALIZATION_RENDERER_VERSION,
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
    let token_desc = ordinary
        .capture()
        .inputs
        .iter()
        .find(|input| input.node == token)
        .unwrap()
        .desc
        .clone();
    let frozen_capture = ordinary.capture().to_bytes().unwrap();
    let ordinary_identity = ordinary.deployment_identity();
    let inference = ordinary
        .with_authenticated_host_gathers(&["token"])
        .unwrap();
    assert_eq!(inference.capture().to_bytes().unwrap(), frozen_capture);
    assert_ne!(inference.deployment_identity(), ordinary_identity);
    let (_, _, _, scalar_links, _) = inference.clone().into_parts();
    let mut legacy_identity = std::collections::hash_map::DefaultHasher::new();
    "rustgrad-captured-host-gather-v1".hash(&mut legacy_identity);
    ordinary_identity.hash(&mut legacy_identity);
    scalar_links.hash(&mut legacy_identity);
    assert_eq!(inference.deployment_identity(), legacy_identity.finish());

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
                input_desc: token_desc,
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
        input_desc: links[0].input.desc.clone(),
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
fn captured_fixed_host_gather_checks_every_lane_before_driver_work() {
    let mut graph = Graph::new();
    let table = graph.input_dtype("table", [1, 4, 2], DType::F32);
    let token = graph.input_dtype("token", [1, 3], DType::I32);
    let token_rows = graph.reshape(token, [1, 3, 1]).unwrap();
    let indices = graph.expand(token_rows, [1, 3, 2]).unwrap();
    let gathered = graph.gather(table, indices, 1).unwrap();
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
    let plan = MetalInferencePlan::new(inference, renderer).unwrap();
    let direct = plan
        .rendered_items()
        .find(|rendered| {
            rendered
                .source
                .contains(METAL_FIXED_HOST_GATHER_RENDERER_VERSION)
        })
        .expect("authenticated fixed-cardinality Gather renderer");
    assert_eq!(direct.entry, "rg_metal_host_gather_fixed_f32_i32");
    assert!(direct.indexed_movement().is_none());
    assert!(direct.transaction.is_none());
    assert!(!direct.source.contains("rg_status"));

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut session = plan.prepare(device).unwrap();
    let table_value =
        TensorData::new([1, 4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let invocation = |selected: [i32; 3]| {
        BTreeMap::from([
            ("table".into(), table_value.clone()),
            (
                "token".into(),
                TensorData::from_scalars(
                    [1, 3],
                    DType::I32,
                    selected.map(|value| Scalar::I(i64::from(value))),
                )
                .unwrap(),
            ),
        ])
    };
    for (selected, lane, value) in [([-1, 1, 2], 0, -1), ([0, 4, 2], 1, 4), ([0, 1, -2], 2, -2)] {
        mock.clear_calls();
        assert!(matches!(
            session.run(&invocation(selected)),
            Err(MetalError::IndexOutOfBounds {
                axis: 1,
                index,
                value: actual,
                dim: 4,
            }) if index == lane && actual == value
        ));
        assert!(mock.calls().is_empty());
        assert_eq!(session.successful_run_count(), 0);
    }
    mock.clear_calls();
    let run = session.run(&invocation([3, 0, 2])).unwrap();
    assert_eq!(
        run.outputs(),
        &[TensorData::new([1, 3, 2], vec![49.0, 64.0, 1.0, 4.0, 25.0, 36.0]).unwrap()]
    );
    assert_eq!(run.report().transient_h2d_calls, 2);
    assert_eq!(
        run.report().transient_h2d_bytes,
        8 * DType::F32.itemsize() + 3 * DType::I32.itemsize()
    );
    assert_eq!(run.report().retained_d2h_calls, 1);
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
    // Sharing one scalar index producer across two Gather owners violates the
    // authenticated sole-consumer lineage, so neither owner is admissible.
    assert!(matches!(
        capture(&ambiguous, output).with_authenticated_host_gathers(&["token"]),
        Err(crate::CapturedInferenceError::Binding(reason))
            if reason.contains("0 authenticated internal owners")
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
            if reason.contains("one dense scalar or batch-one fixed I32 transient")
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
            if reason.contains("one dense scalar or batch-one fixed I32 transient")
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
    assert_eq!(METAL_RENDERER_VERSION, "rustgrad-metal-static-v9");
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
use super::resource::MetalBatchItem;
use super::*;
use crate::engine::capture::QuantizedCaptureBinding;
use crate::kernel::execute_lowered_elementwise;
use crate::models::transformer::{
    LLAMA_SIMPLE_CHAT_TEMPLATE, LlamaChatMessage, LlamaChatRole, LlamaGenerator,
    LlamaMetalGreedyPlan, LlamaMetalStepPlan, LlamaPromptWorkflow, LlamaSampling,
    packed_metal_fixture_models, packed_metal_workflow_bytes,
};
use crate::nn::{Linear, Module, Parameter, StateKind};
use crate::runtime::scalar_lane::emit_scalar_lane;
use crate::{
    Backend, BinaryOp, BufferRole, CapturedAppendStateInference, CapturedInference,
    CapturedMixedBatch, CapturedReplayExecutor, CapturedSchedule, CapturedStatefulInference,
    CompareOp, CpuBackend, CpuSession, DType, EffectBatchStep, EffectRuntime, GgmlType, Graph,
    IndexValue, InferenceAppendStateLink, InferenceStateLink, KernelBindings, KernelBufferDesc,
    LaneInstruction, MovementKernelKind, MovementValue, NodeId, Operation, QuantizedTensorData,
    ReduceKind, ResNet, ResNetConfig, ResNetMetalError, ResNetMetalPlan, Scalar, Shape, Slice,
    Storage, TensorData, TypedValue, UOp, UType, schedule,
};

fn packed_ones(kind: GgmlType, rows: usize) -> QuantizedTensorData {
    let (columns, block_bytes) = match kind {
        GgmlType::Q4_0 => (32, 18),
        GgmlType::Q8_0 => (32, 34),
        GgmlType::Q4K => (256, 144),
        GgmlType::Q6K => (256, 210),
        _ => panic!("test packed format"),
    };
    let mut bytes = vec![0u8; rows * block_bytes];
    for block in bytes.chunks_exact_mut(block_bytes) {
        match kind {
            GgmlType::Q4_0 => {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                block[2..].fill(0x99);
            }
            GgmlType::Q8_0 => {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                block[2..].fill(1);
            }
            GgmlType::Q4K => {
                block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                block[4..8].fill(1);
                block[12..16].fill(1);
                block[16..].fill(0x11);
            }
            GgmlType::Q6K => {
                block[..128].fill(0x11);
                block[128..192].fill(0xaa);
                block[192..208].fill(1);
                block[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
            }
            _ => unreachable!("test packed format"),
        }
    }
    QuantizedTensorData::new(kind, Shape::from([rows, columns]), bytes).unwrap()
}
use dispatch::{
    CopyRegion, Dispatch, KernelSemantics, LaunchGeometry, RawBuffer, RawCommand, RawDevice,
    RawLibrary, RawPipeline, RawQueue,
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    rc::Rc,
    sync::{Arc, Mutex},
};

fn append_position(graph: &mut Graph, name: &str, shape: impl Into<Shape>) -> (NodeId, NodeId) {
    let shape = shape.into();
    let position = graph.input_dtype(name, [1], DType::I32);
    let expanded = graph
        .reshape(position, vec![1; shape.rank()])
        .and_then(|value| graph.expand(value, shape))
        .unwrap();
    (position, expanded)
}

fn append_span_position(
    graph: &mut Graph,
    name: &str,
    updates: NodeId,
    axis: usize,
) -> (NodeId, NodeId) {
    let shape = graph.shape(updates).unwrap().clone();
    let scalar_position = graph.input_dtype(name, [1], DType::I32);
    let position = graph
        .reshape(scalar_position, vec![1; shape.rank()])
        .and_then(|value| graph.expand(value, shape.clone()))
        .unwrap();
    let iota = graph.shape_iota(updates, axis).unwrap();
    let mut iota_shape = vec![1; shape.rank()];
    iota_shape[axis] = shape.dims()[axis];
    let iota = graph
        .reshape(iota, iota_shape)
        .and_then(|value| graph.expand(value, shape))
        .unwrap();
    let index = graph.add(position, iota).unwrap();
    (scalar_position, index)
}

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
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(37)));
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
        assert_eq!(report.command_submission_count, 1);
        assert_eq!(report.command_wait_count, 1);
        assert_eq!(
            report.gpu_command_execution_time,
            Some(std::time::Duration::from_nanos(37))
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
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("batch_submit:"))
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|call| call.starts_with("wait:"))
                .count(),
            1
        );
        let write = calls
            .iter()
            .position(|call| call.starts_with("write:"))
            .unwrap();
        let submit = calls
            .iter()
            .position(|call| call.starts_with("batch_submit:"))
            .unwrap();
        let wait = calls
            .iter()
            .position(|call| call.starts_with("wait:"))
            .unwrap();
        let read = calls
            .iter()
            .position(|call| call.starts_with("read:"))
            .unwrap();
        assert!(write < submit && submit < wait && wait < read);
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
    let (position, index) = append_position(&mut graph, "position", [1, 3]);
    let update_source = graph.input_dtype("updates", [1, 3], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let attention_read = graph.square(next).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[attention_read, attention_read],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
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
    let mut scoreboard = MetalSessionScoreboard::new_append_state(
        MetalScoreboardContext::new("append-cache", "test-revision", "semantic mock").unwrap(),
        &plan,
    );
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

    let expected_kernel_launches = plan.summary().nonzero_item_count;
    let mock = Arc::new(MockDispatch::default());
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(11)));
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    let calls_after_prepare = mock.calls();
    scoreboard.bind(&session).unwrap();
    assert_eq!(mock.calls(), calls_after_prepare);
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.preparation_report().initial_state_h2d_bytes, 24);
    let empty_report = scoreboard.report().unwrap();
    assert_eq!(
        empty_report.state_policy,
        MetalScoreboardStatePolicy::Append
    );
    assert_eq!(empty_report.committed_state_position, Some(0));
    assert_eq!(empty_report.successful_run_count, 0);
    assert_eq!(empty_report.initial_state_host_api_h2d_calls, 1);
    assert_eq!(empty_report.initial_state_host_api_h2d_bytes, 24);
    assert_eq!(
        empty_report
            .inputs
            .iter()
            .filter(|input| input.kind == MetalScoreboardInputKind::State)
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["cache"]
    );

    let invocation = |position_value: i32, updates: Vec<f32>| {
        BTreeMap::from([
            (
                "position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(i64::from(position_value))])
                    .unwrap(),
            ),
            ("updates".into(), TensorData::new([1, 3], updates).unwrap()),
        ])
    };
    mock.clear_calls();
    assert!(
        session
            .run_without_host_outputs(&invocation(0, vec![1.0, 2.0, 3.0]))
            .is_err()
    );
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.successful_run_count(), 0);
    assert!(mock.calls().is_empty());

    mock.state.lock().unwrap().failures.read = Some("public read");
    assert!(session.run(&invocation(0, vec![1.0, 2.0, 3.0])).is_err());
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.successful_run_count(), 0);
    assert_eq!(scoreboard.report().unwrap(), empty_report);
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
    assert_eq!(first.report().transient_h2d_bytes, 16);
    assert_eq!(first.report().retained_d2h_calls, 1);
    assert_eq!(first.report().retained_d2h_bytes, 24);
    assert_eq!(first.report().kernel_launch_count, expected_kernel_launches);
    assert_eq!(
        first.report().gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(11))
    );
    assert_eq!(first.report().committed_state_position, Some(1));
    assert_eq!(session.committed_state_position(), Some(1));
    scoreboard.record(&first).unwrap();
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
    assert_eq!(scoreboard.report().unwrap().successful_run_count, 1);

    let second = session.run(&invocation(1, vec![7.0, 8.0, 9.0])).unwrap();
    assert_eq!(
        second.outputs()[0],
        TensorData::new([2, 3], vec![16.0, 25.0, 36.0, 49.0, 64.0, 81.0]).unwrap()
    );
    assert_eq!(session.committed_state_position(), Some(2));
    assert_eq!(second.report().committed_state_position, Some(2));
    scoreboard.record(&second).unwrap();
    let report = scoreboard.report().unwrap();
    assert_eq!(report.format_version, 7);
    assert_eq!(report.successful_run_count, 2);
    assert_eq!(
        report.gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(22))
    );
    assert_eq!(
        report.successful_runs[0].gpu_command_execution_time,
        first.report().gpu_command_execution_time
    );
    assert_eq!(report.committed_state_position, Some(2));
    assert_eq!(report.state_pair_count, 1);
    assert_eq!(report.logical_state_bytes, 24);
    assert_eq!(report.state_bank_count, 1);
    assert_eq!(report.state_device_bytes, 24);
    assert_eq!(report.append_state_row_bytes, 12);
    assert_eq!(report.append_state_work_items, 3);
    assert_eq!(report.committed_state_pair_count, 2);
    assert_eq!(report.committed_state_bytes, 24);
    assert_eq!(report.committed_state_work_items, 6);
    assert_eq!(report.successful_runs[0].committed_state_position, Some(1));
    assert_eq!(report.successful_runs[1].committed_state_position, Some(2));
    assert_eq!(
        report.first_run_host_wall_time,
        Some(first.report().run_wall_time)
    );
    assert_eq!(
        report.steady_run_host_wall_times,
        [second.report().run_wall_time]
    );
    assert_eq!(
        report.host_api_h2d_calls,
        report.resident_host_api_h2d_calls
            + report.initial_state_host_api_h2d_calls
            + report.transient_host_api_h2d_calls
    );
    assert_eq!(
        report.host_api_h2d_bytes,
        report.resident_host_api_h2d_bytes
            + report.initial_state_host_api_h2d_bytes
            + report.transient_host_api_h2d_bytes
    );
    assert_eq!(
        report.kernel_launch_count,
        report.planned_kernel_count * report.successful_runs.len()
    );
    assert_eq!(report.fallback_count, 0);
    let json =
        serde_json::from_slice::<serde_json::Value>(&report.to_json_bytes().unwrap()).unwrap();
    assert_eq!(json["state_policy"], "append");
    assert_eq!(json["committed_state_position"], 2);
    assert_eq!(json["successful_runs"][0]["committed_state_position"], 1);
    let before = mock.calls().len();
    assert!(session.run(&invocation(2, vec![10.0, 11.0, 12.0])).is_err());
    assert_eq!(mock.calls().len(), before);
    assert_eq!(session.successful_run_count(), 2);
    assert_eq!(scoreboard.report().unwrap(), report);
    assert!(!mock.calls().iter().any(|call| call.starts_with("copy:")));
}

#[test]
fn shared_append_preparation_rejects_a_source_after_state_advances() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("cache", [2, 1], DType::F32);
    let (position, index) = append_position(&mut graph, "position", [1, 1]);
    let update_source = graph.input_dtype("updates", [1, 1], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
        )],
        BTreeMap::from([("cache".into(), TensorData::zeros([2, 1]).unwrap())]),
    )
    .unwrap()
    .seal_committed_position()
    .unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let source_plan =
        MetalAppendStateInferencePlan::new(captured.clone(), renderer.clone()).unwrap();
    let target_plan = MetalAppendStateInferencePlan::new(captured, renderer).unwrap();
    let proof = target_plan.authenticate_shared_from(&source_plan).unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut source = source_plan.prepare(device.clone()).unwrap();
    source
        .run(&BTreeMap::from([(
            "updates".into(),
            TensorData::new([1, 1], vec![4.0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(source.committed_state_position(), Some(1));
    mock.clear_calls();
    assert!(matches!(
        target_plan.prepare_shared(device, &source, proof),
        Err(MetalError::InvalidBinding(reason))
            if reason == "shared Metal session proof does not belong to these deployments"
    ));
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_append_scoreboard_reports_packed_constants_exactly() {
    let packed = packed_ones(GgmlType::Q4_0, 2);
    let packed_bytes = packed.bytes().len();
    let mut graph = Graph::new();
    let activation = graph.input_dtype("activation", [1, 32], DType::F32);
    let weight = graph.input_dtype("weight", [2, 32], DType::F32);
    let transposed = graph.permute(weight, [1, 0]).unwrap();
    let updates = graph.matmul(activation, transposed).unwrap();
    let state = graph.input_dtype("cache", [2, 2], DType::F32);
    let (position, index) = append_position(&mut graph, "position", [1, 2]);
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let requested = graph.square(next).unwrap();
    let capture = CapturedAppendStateInference::from_graph_residents(
        &graph,
        &[requested],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
        )],
        BTreeMap::from([("cache".into(), TensorData::zeros([2, 2]).unwrap())]),
        BTreeMap::new(),
        &[QuantizedCaptureBinding::Matmul {
            output: updates,
            activation,
            weight,
            value: packed,
        }],
        &[],
    )
    .unwrap();
    let plan =
        MetalAppendStateInferencePlan::new(capture, MetalRenderer::new(8, capabilities()).unwrap())
            .unwrap();
    let summary = plan.summary().clone();
    assert_eq!(summary.constant_count, 0);
    assert_eq!(summary.constant_bytes, 0);
    assert_eq!(summary.quantized_constant_count, 1);
    assert_eq!(summary.quantized_constant_bytes, packed_bytes);
    let mut scoreboard = MetalSessionScoreboard::new_append_state(
        MetalScoreboardContext::new("packed-append", "test-revision", "semantic mock").unwrap(),
        &plan,
    );
    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock)).unwrap();
    scoreboard.bind(&session).unwrap();
    let run = session
        .run(&BTreeMap::from([
            (
                "activation".into(),
                TensorData::new([1, 32], vec![1.0; 32]).unwrap(),
            ),
            (
                "position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap(),
            ),
        ]))
        .unwrap();
    scoreboard.record(&run).unwrap();
    let report = scoreboard.report().unwrap();
    assert_eq!(report.captured_constant_count, summary.constant_count);
    assert_eq!(report.captured_constant_bytes, summary.constant_bytes);
    assert_eq!(
        report.captured_quantized_constant_count,
        summary.quantized_constant_count
    );
    assert_eq!(
        report.captured_quantized_constant_bytes,
        summary.quantized_constant_bytes
    );
    assert_eq!(
        report.resident_host_api_h2d_bytes,
        report.captured_quantized_constant_bytes
    );
    assert_eq!(report.initial_state_host_api_h2d_bytes, 16);
    assert_eq!(
        report
            .captured_constant_bytes
            .checked_add(report.captured_quantized_constant_bytes)
            .unwrap(),
        packed_bytes
    );
    assert!(
        report.planned_physical_static_tensor_slot_bytes
            >= report.captured_quantized_constant_bytes
    );
    let encoded = report.to_json_bytes().unwrap();
    assert_eq!(encoded, report.to_json_bytes().unwrap());
    let json = serde_json::from_slice::<serde_json::Value>(&encoded).unwrap();
    assert_eq!(json["state_policy"], "append");
    assert_eq!(json["captured_constant_count"], 0);
    assert_eq!(json["captured_constant_bytes"], 0);
    assert_eq!(json["captured_quantized_constant_count"], 1);
    assert_eq!(
        json["captured_quantized_constant_bytes"],
        u64::try_from(packed_bytes).unwrap()
    );
}

#[test]
fn ordinary_i32_shape_iota_renders_only_its_authenticated_store_iteration() {
    let mut graph = Graph::new();
    let source = graph.input_dtype("source", [3], DType::F32);
    let iota = graph.shape_iota(source, 0).unwrap();
    let item = schedule(&graph, iota).unwrap().items.pop().unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let rendered = renderer.render(&item.kernel).unwrap();
    assert_eq!(rendered.extent, 3);
    assert_eq!(rendered.buffers.len(), 1);
    assert_eq!(rendered.buffers[0].dtype, DType::I32);
    assert!(rendered.source.contains("b0[gid] = ((int)((ulong)gid));"));
    assert!(!rendered.source.contains("device long*"));

    let store = &item.kernel.sources()[0];
    let index = &store.sources()[0];
    let address = index.sources()[0].clone();
    let wrong_iteration = UOp::from_operation(
        Operation::Range(0),
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(4, UType::scalar(DType::I64))],
    );
    let wrong_index = UOp::from_operation(
        index.operation().clone(),
        index.ty(),
        vec![address, wrong_iteration.clone()],
    );
    let malformed = UOp::sink(vec![
        UOp::from_operation(
            Operation::Store,
            None,
            vec![
                wrong_index,
                UOp::cast(wrong_iteration.clone(), UType::scalar(DType::I32)),
            ],
        ),
        UOp::from_operation(Operation::EndRange, None, vec![wrong_iteration]),
    ]);
    assert!(matches!(
        renderer.render(&malformed),
        Err(MetalError::Unsupported(reason))
            if reason.contains("dtype I64 is outside the exact Metal static subset")
    ));

    let mut wide_graph = Graph::new();
    let wide_source = wide_graph.input_dtype(
        "wide_source",
        [usize::try_from(i32::MAX).unwrap() + 1],
        DType::F32,
    );
    let wide_iota = wide_graph.shape_iota(wide_source, 0).unwrap();
    assert_eq!(wide_graph.dtype(wide_iota).unwrap(), DType::I64);
    let wide_item = schedule(&wide_graph, wide_iota)
        .unwrap()
        .items
        .pop()
        .unwrap();
    assert!(matches!(
        renderer.render(&wide_item.kernel),
        Err(MetalError::Unsupported(reason))
            if reason.contains("dtype I64 is outside the exact Metal static subset")
    ));
}

#[test]
fn metal_append_state_fixed_span_commits_consecutive_rows_and_retries_tail() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("cache", [7, 2], DType::F32);
    let update_source = graph.input_dtype("updates", [3, 2], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let (position, index) = append_span_position(&mut graph, "position", updates, 0);
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let visible = graph.square(next).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[visible],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
        )],
        BTreeMap::from([("cache".into(), TensorData::zeros([7, 2]).unwrap())]),
    )
    .unwrap();
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let plan = MetalAppendStateInferencePlan::new(captured, renderer.clone()).unwrap();
    assert_eq!(plan.append_span_rows(), 3);
    assert_eq!(plan.summary().append_state_row_bytes, 24);
    assert_eq!(plan.summary().append_state_work_items, 6);
    let rendered = plan
        .rendered_items()
        .find(|rendered| rendered.append_state().is_some())
        .unwrap();
    assert_eq!(rendered.entry, "rg_metal_append_span_f32_i32");
    assert!(rendered.source.contains("rustgrad-metal-append-span-v1"));
    assert!(!rendered.source.contains("rustgrad-metal-append-state-v1"));
    let iota_rendered = plan
        .rendered_items()
        .find(|rendered| rendered.entry == "rg_metal_append_span_iota_i32")
        .expect("authenticated fixed-span ShapeIota renderer");
    assert!(
        iota_rendered
            .source
            .contains("rustgrad-metal-append-span-iota-i32-v1")
    );
    assert!(iota_rendered.source.contains("device int* b0"));
    assert!(iota_rendered.source.contains("b0[gid] = (int)rg_gid;"));
    assert!(!iota_rendered.source.contains("device long*"));
    assert_eq!(iota_rendered.buffers.len(), 1);
    assert_eq!(iota_rendered.buffers[0].dtype, DType::I32);
    assert_eq!(iota_rendered.extent, 3);
    assert_ne!(iota_rendered.cache_key, rendered.cache_key);
    let ordinary_iota = plan
        .capture()
        .items
        .iter()
        .find(|item| item.outputs.primary().id == iota_rendered.buffers[0].id)
        .expect("captured ShapeIota producer");
    let ordinary_iota = renderer.render(&ordinary_iota.kernel).unwrap();
    assert_eq!(ordinary_iota.buffers.len(), 1);
    assert_eq!(ordinary_iota.buffers[0].dtype, DType::I32);
    assert!(
        ordinary_iota
            .source
            .contains("b0[gid] = ((int)((ulong)gid));")
    );
    assert!(!ordinary_iota.source.contains("device long*"));

    let context =
        || MetalScoreboardContext::new("append-span", "test-revision", "semantic mock").unwrap();
    let error = MetalSessionScoreboard::try_new_append_state_v4(context(), &plan)
        .err()
        .unwrap();
    assert_eq!(
        error,
        MetalScoreboardError::UnsupportedAppendSpan { span_rows: 3 }
    );
    assert_eq!(
        error.to_string(),
        "Metal token-step scoreboard requires one appended row per successful invocation; got a span of 3 rows"
    );
    let mut scoreboard = MetalSessionScoreboard::new_append_state(context(), &plan);
    let mut rejected = MetalSessionScoreboard::new_append_state(context(), &plan);
    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    scoreboard.bind(&session).unwrap();
    rejected.bind(&session).unwrap();

    let invocation = |position: i32, values: [f32; 6]| {
        BTreeMap::from([
            (
                "position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(i64::from(position))])
                    .unwrap(),
            ),
            (
                "updates".into(),
                TensorData::new([3, 2], values.to_vec()).unwrap(),
            ),
        ])
    };
    let first = session
        .run(&invocation(0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .unwrap();
    assert_eq!(first.report().committed_state_position, Some(3));
    assert_eq!(first.report().committed_state_bytes, 24);
    assert_eq!(first.report().committed_state_work_items, 6);
    assert_eq!(session.committed_state_position(), Some(3));
    assert_eq!(
        rejected.record_from_position(&first, 1),
        Err(MetalScoreboardError::StateOutOfOrder {
            expected: Some(4),
            actual: Some(3),
        })
    );
    assert_eq!(rejected.report().unwrap().successful_run_count, 0);
    assert_eq!(
        rejected.record_from_position(&first, usize::MAX),
        Err(MetalScoreboardError::Overflow)
    );
    assert_eq!(rejected.report().unwrap().successful_run_count, 0);
    scoreboard.record(&first).unwrap();
    let second = session
        .run(&invocation(3, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]))
        .unwrap();
    assert_eq!(second.report().committed_state_position, Some(6));
    scoreboard.record(&second).unwrap();
    let scoreboard_report = scoreboard.report().unwrap();
    assert_eq!(scoreboard_report.append_span_rows, 3);
    assert_eq!(scoreboard_report.append_state_row_bytes, 24);
    assert_eq!(scoreboard_report.append_state_work_items, 6);
    assert_eq!(scoreboard_report.successful_run_count, 2);
    assert_eq!(scoreboard_report.committed_state_position, Some(6));
    assert_eq!(scoreboard_report.committed_state_bytes, 48);
    assert_eq!(scoreboard_report.committed_state_work_items, 12);
    assert_eq!(
        second.outputs()[0],
        TensorData::new(
            [7, 2],
            vec![
                1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 0.0, 0.0,
            ],
        )
        .unwrap()
    );
    let calls = mock.calls().len();
    assert!(
        session
            .run(&invocation(6, [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]))
            .is_err()
    );
    assert_eq!(mock.calls().len(), calls);
    assert_eq!(session.committed_state_position(), Some(6));
    assert_eq!(session.successful_run_count(), 2);
    assert_eq!(scoreboard.report().unwrap(), scoreboard_report);
}

#[test]
fn metal_append_state_kv_links_commit_together_after_partial_failure() {
    let mut graph = Graph::new();
    let key_state = graph.input_dtype("key_cache", [2, 2], DType::F32);
    let value_state = graph.input_dtype("value_cache", [2, 2], DType::F32);
    let (position, index) = append_position(&mut graph, "position", [1, 2]);
    let key_source = graph.input_dtype("key_updates", [1, 2], DType::F32);
    let value_source = graph.input_dtype("value_updates", [1, 2], DType::F32);
    let key_updates = graph.relu(key_source).unwrap();
    let value_updates = graph.relu(value_source).unwrap();
    let next_keys = graph.scatter(key_state, index, key_updates, 0).unwrap();
    let next_values = graph.scatter(value_state, index, value_updates, 0).unwrap();
    let attention_read = graph.add(next_keys, next_values).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[attention_read],
        &[
            InferenceAppendStateLink::new(key_state, next_keys, position, index, key_updates, 0),
            InferenceAppendStateLink::new(
                value_state,
                next_values,
                position,
                index,
                value_updates,
                0,
            ),
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
    let expected_kernel_launches = plan.summary().nonzero_item_count;

    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    let invocation = |position_value: i32, keys: Vec<f32>, values: Vec<f32>| {
        BTreeMap::from([
            (
                "position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(i64::from(position_value))])
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
    assert_eq!(retry.report().transient_h2d_calls, 3);
    assert_eq!(retry.report().transient_h2d_bytes, 20);
    assert_eq!(retry.report().retained_d2h_calls, 1);
    assert_eq!(retry.report().retained_d2h_bytes, 16);
    assert_eq!(retry.report().kernel_launch_count, expected_kernel_launches);
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
fn metal_append_state_empty_fixed_span_is_addressless_but_advances_rows() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("empty_cache", [6, 0], DType::F32);
    let (position, index) = append_position(&mut graph, "empty_position", [3, 0]);
    let update_source = graph.input_dtype("empty_updates", [3, 0], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let visible = graph.square(next).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[visible],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
        )],
        BTreeMap::from([("empty_cache".into(), TensorData::zeros([6, 0]).unwrap())]),
    )
    .unwrap();
    let plan = MetalAppendStateInferencePlan::new(
        captured,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.append_span_rows(), 3);
    assert_eq!(plan.summary().append_state_row_bytes, 0);
    assert_eq!(plan.summary().append_state_work_items, 0);
    assert_eq!(plan.summary().nonzero_item_count, 0);
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
        .run(&BTreeMap::from([
            (
                "empty_position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap(),
            ),
            ("empty_updates".into(), TensorData::zeros([3, 0]).unwrap()),
        ]))
        .unwrap();
    assert_eq!(run.outputs(), &[TensorData::zeros([6, 0]).unwrap()]);
    assert_eq!(run.report().committed_state_position, Some(3));
    assert_eq!(run.report().kernel_launch_count, 0);
    assert_eq!(run.report().command_submission_count, 0);
    assert_eq!(run.report().command_wait_count, 0);
    assert_eq!(run.report().gpu_command_execution_time, None);
    assert_eq!(run.report().transient_h2d_calls, 0);
    assert_eq!(run.report().runtime_control_h2d_calls, 0);
    assert_eq!(run.report().retained_d2h_calls, 0);
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_append_state_fixed_span_two_link_failure_retries_the_same_rows() {
    let mut graph = Graph::new();
    let key_state = graph.input_dtype("span_key_cache", [6, 1], DType::F32);
    let value_state = graph.input_dtype("span_value_cache", [6, 1], DType::F32);
    let key_source = graph.input_dtype("span_key_updates", [3, 1], DType::F32);
    let value_source = graph.input_dtype("span_value_updates", [3, 1], DType::F32);
    let key_updates = graph.relu(key_source).unwrap();
    let value_updates = graph.relu(value_source).unwrap();
    let (position, index) = append_span_position(&mut graph, "span_position", key_updates, 0);
    let next_keys = graph.scatter(key_state, index, key_updates, 0).unwrap();
    let next_values = graph.scatter(value_state, index, value_updates, 0).unwrap();
    let visible = graph.add(next_keys, next_values).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[visible],
        &[
            InferenceAppendStateLink::new(key_state, next_keys, position, index, key_updates, 0),
            InferenceAppendStateLink::new(
                value_state,
                next_values,
                position,
                index,
                value_updates,
                0,
            ),
        ],
        BTreeMap::from([
            ("span_key_cache".into(), TensorData::zeros([6, 1]).unwrap()),
            (
                "span_value_cache".into(),
                TensorData::zeros([6, 1]).unwrap(),
            ),
        ]),
    )
    .unwrap();
    let plan = MetalAppendStateInferencePlan::new(
        captured,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    let append_launches = plan
        .rendered_items()
        .filter(|item| item.extent != 0)
        .enumerate()
        .filter_map(|(ordinal, item)| item.append_state().map(|_| ordinal))
        .collect::<Vec<_>>();
    assert_eq!(append_launches.len(), 2);
    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    let invocation = |keys: [f32; 3], values: [f32; 3]| {
        BTreeMap::from([
            (
                "span_position".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap(),
            ),
            (
                "span_key_updates".into(),
                TensorData::new([3, 1], keys.to_vec()).unwrap(),
            ),
            (
                "span_value_updates".into(),
                TensorData::new([3, 1], values.to_vec()).unwrap(),
            ),
        ])
    };
    mock.state.lock().unwrap().failures.launch_after =
        Some((append_launches[1], "second span append"));
    assert!(matches!(
        session.run(&invocation([9.0; 3], [8.0; 3])),
        Err(MetalError::Driver { operation: "launch", detail })
            if detail == "second span append"
    ));
    assert!(mock.state.lock().unwrap().failures.launch_after.is_none());
    assert_eq!(session.committed_state_position(), Some(0));
    assert_eq!(session.successful_run_count(), 0);

    let retry = session
        .run(&invocation([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]))
        .unwrap();
    assert_eq!(
        retry.outputs(),
        &[TensorData::new([6, 1], vec![5.0, 7.0, 9.0, 0.0, 0.0, 0.0]).unwrap()]
    );
    assert_eq!(retry.report().committed_state_pair_count, 2);
    assert_eq!(retry.report().committed_state_position, Some(3));
    assert_eq!(session.committed_state_position(), Some(3));
    assert_eq!(session.successful_run_count(), 1);
}

#[test]
fn metal_append_state_empty_rows_are_addressless_but_advance_logically() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("empty_cache", [2, 0], DType::F32);
    let (position, index) = append_position(&mut graph, "empty_position", [1, 0]);
    let update_source = graph.input_dtype("empty_updates", [1, 0], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[update_source],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
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
    let mut scoreboard = MetalSessionScoreboard::new_append_state(
        MetalScoreboardContext::new("empty-append", "test-revision", "semantic mock").unwrap(),
        &plan,
    );
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device).unwrap();
    scoreboard.bind(&session).unwrap();
    assert!(mock.calls().is_empty());
    let values = BTreeMap::from([
        (
            "empty_position".into(),
            TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap(),
        ),
        (
            "empty_updates".into(),
            TensorData::new([1, 0], Vec::new()).unwrap(),
        ),
    ]);
    let run = session.run(&values).unwrap();
    assert_eq!(session.committed_state_position(), Some(1));
    assert_eq!(run.report().transient_h2d_calls, 0);
    assert_eq!(run.report().transient_h2d_bytes, 0);
    assert_eq!(run.report().kernel_launch_count, 0);
    scoreboard.record(&run).unwrap();
    let report = scoreboard.report().unwrap();
    assert_eq!(report.committed_state_position, Some(1));
    assert_eq!(report.initial_state_host_api_h2d_calls, 0);
    assert_eq!(report.host_api_h2d_calls, 0);
    assert_eq!(report.host_api_d2h_calls, 0);
    assert_eq!(report.kernel_launch_count, 0);
    assert_eq!(report.committed_state_bytes, 0);
    assert_eq!(report.committed_state_work_items, 0);
    assert!(mock.calls().is_empty());
}

#[test]
fn bounded_i32_device_proof_rejects_sentinel_and_upper_bound_before_commit() {
    let mut graph = Graph::new();
    let state = graph.input_dtype("proof_cache", [2, 1], DType::F32);
    let (position, index) = append_position(&mut graph, "proof_position", [1, 1]);
    let update_source = graph.input_dtype("proof_update", [1, 1], DType::F32);
    let updates = graph.relu(update_source).unwrap();
    let next = graph.scatter(state, index, updates, 0).unwrap();
    let candidate = graph.input_dtype("proof_token", [1], DType::I32);
    let zero = graph.constant(TensorData::from_scalars([1], DType::I32, [Scalar::I(0)]).unwrap());
    let candidate = graph.add(candidate, zero).unwrap();
    let captured = CapturedAppendStateInference::from_module_graph(
        &IdentityModule,
        &graph,
        &[candidate],
        &[InferenceAppendStateLink::new(
            state, next, position, index, updates, 0,
        )],
        BTreeMap::from([(
            "proof_cache".into(),
            TensorData::new([2, 1], vec![0.0, 0.0]).unwrap(),
        )]),
    )
    .unwrap()
    .seal_committed_position()
    .unwrap();
    let plan = MetalAppendStateInferencePlan::new(
        captured,
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    let mut session = plan
        .prepare(test_device(Arc::new(MockDispatch::default())))
        .unwrap();
    let inputs = |token: i64| {
        BTreeMap::from([
            (
                "proof_token".into(),
                TensorData::from_scalars([1], DType::I32, [Scalar::I(token)]).unwrap(),
            ),
            (
                "proof_update".into(),
                TensorData::new([1, 1], vec![1.0]).unwrap(),
            ),
        ])
    };

    for (output, upper_exclusive) in [(1, 4), (0, 0)] {
        assert!(matches!(
            session.run_at_requiring_bounded_i32(&inputs(2), 0, output, upper_exclusive,),
            Err(MetalError::InvalidDeviceProof(_))
        ));
        assert_eq!(session.committed_state_position(), Some(0));
        assert_eq!(session.successful_run_count(), 0);
    }

    for invalid in [-1, 4] {
        assert!(matches!(
            session.run_at_requiring_bounded_i32(&inputs(invalid), 0, 0, 4),
            Err(MetalError::InvalidDeviceProof(_))
        ));
        assert_eq!(session.committed_state_position(), Some(0));
        assert_eq!(session.successful_run_count(), 0);
    }

    let retry = session
        .run_at_requiring_bounded_i32(&inputs(2), 0, 0, 4)
        .unwrap();
    assert_eq!(retry.outputs()[0].dtype(), DType::I32);
    assert_eq!(retry.outputs()[0].scalar_at(0).as_i64(), 2);
    assert_eq!(retry.report().retained_d2h_calls, 1);
    assert_eq!(retry.report().retained_d2h_bytes, 4);
    assert_eq!(session.committed_state_position(), Some(1));
    assert_eq!(session.successful_run_count(), 1);
}

#[test]
fn packed_llama_token_session_matches_dense_oracle_and_retries_atomically() {
    let (packed, _, dense, _) = packed_metal_fixture_models();
    let plan =
        LlamaMetalStepPlan::new(&packed, MetalRenderer::new(8, capabilities()).unwrap()).unwrap();
    let control_name = &plan.summary().runtime_control_input_names[0];
    let control = plan
        .capture()
        .inputs
        .iter()
        .find(|input| &input.name == control_name)
        .expect("runtime control remains an exact captured input");
    assert!(control.desc.view.is_some());
    let stable_summary = plan.summary().clone();
    let mock = Arc::new(MockDispatch::default());
    let mut session = plan.prepare(test_device(mock.clone())).unwrap();
    let stable_owner = session.metal_session().device_owner_id();
    let stable_compiled = session.metal_session().compiled_kernels().count();
    let expected_last = |tokens: &[u32]| {
        let all = dense.forward(tokens).unwrap();
        let vocab = packed.config().schema().vocab_size();
        let start = (tokens.len() - 1) * vocab;
        TensorData::new([1, vocab], all.values()[start..start + vocab].to_vec()).unwrap()
    };
    let assert_logits = |actual: &TensorData, expected: &TensorData| {
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), DType::F32);
        for (index, (actual, expected)) in actual.values().iter().zip(expected.values()).enumerate()
        {
            assert!(
                actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-3,
                "logit {index}: packed Metal {actual} != dense oracle {expected}"
            );
        }
    };

    mock.clear_calls();
    assert!(
        session
            .commit_token(packed.config().schema().vocab_size() as u32)
            .is_err()
    );
    assert_eq!(session.position(), 0);
    assert!(mock.calls().is_empty());

    mock.clear_calls();
    mock.state.lock().unwrap().failures.launch = Some("packed Llama launch");
    assert!(session.run_token(3).is_err());
    assert_eq!(session.position(), 0);
    assert_eq!(session.metal_session().successful_run_count(), 0);
    mock.clear_failures();

    mock.clear_calls();
    let first = session.commit_token(3).unwrap();
    assert_eq!(first.position(), 0);
    assert_eq!(session.position(), 1);
    assert_eq!(first.report().transient_h2d_calls, 1);
    assert_eq!(first.report().transient_h2d_bytes, 4);
    assert_eq!(first.report().runtime_control_h2d_calls, 1);
    assert_eq!(first.report().runtime_control_h2d_bytes, 4);
    assert_eq!(first.report().retained_d2h_calls, 0);
    assert_eq!(first.report().retained_d2h_bytes, 0);
    assert_eq!(first.report().output_count, 0);
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
            || call.starts_with("queue_create:")
    }));

    let second = session.commit_token(4).unwrap();
    assert_eq!(second.position(), 1);
    assert_eq!(second.report().retained_d2h_calls, 0);
    assert_eq!(session.position(), 2);

    mock.state.lock().unwrap().failures.read = Some("packed Llama logits read");
    assert!(session.run_token(5).is_err());
    assert_eq!(session.position(), 2);
    assert_eq!(session.metal_session().successful_run_count(), 2);
    mock.clear_failures();

    mock.clear_calls();
    let retry = session.run_token(5).unwrap();
    assert_logits(retry.logits(), &expected_last(&[3, 4, 5]));
    assert_eq!(retry.position(), 2);
    assert_eq!(session.position(), 3);
    assert_eq!(retry.report().transient_h2d_calls, 1);
    assert_eq!(retry.report().transient_h2d_bytes, 4);
    assert_eq!(retry.report().runtime_control_h2d_calls, 1);
    assert_eq!(retry.report().runtime_control_h2d_bytes, 4);
    assert_eq!(retry.report().retained_d2h_calls, 1);
    assert_eq!(session.metal_session().device_owner_id(), stable_owner);
    assert_eq!(session.metal_session().summary(), &stable_summary);
    assert_eq!(
        session.metal_session().compiled_kernels().count(),
        stable_compiled
    );
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
            || call.starts_with("queue_create:")
    }));
}

#[test]
fn llama_metal_prompt_facade_prefills_with_one_read_and_matches_cpu() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (cpu_model, cpu_tokenizer) =
        crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let prompt = [3, 4, 5];
    let expected = cpu_model.forward(&prompt).unwrap();
    let vocab = cpu_model.config().schema().vocab_size();
    let expected_last = TensorData::new(
        [1, vocab],
        expected.values()[expected.values().len() - vocab..].to_vec(),
    )
    .unwrap();

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let plan = crate::models::transformer::LlamaMetalPlan::from_workflow(
        workflow,
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap();
    assert_eq!(plan.selected_device_owner_id(), device.owner_id());
    assert_eq!(plan.transient_inputs().len(), 1);
    assert_eq!(plan.runtime_control_inputs().len(), 1);
    assert_eq!(plan.summary().fallback_count, 0);
    let stable_identity = plan.step_deployment_identity();
    let stable_capture = plan.capture().identity;
    let mut session = plan.prepare().unwrap();
    assert!(session.execution_scoreboard().is_none());
    assert!(session.scoreboard_recording_error().is_none());
    let stable_kernels = session.compiled_kernels().count();
    let calls_after_prepare = mock.calls();

    let zero = session
        .generate_ids(&prompt, 0, LlamaSampling::Greedy)
        .unwrap();
    assert!(zero.reports().is_empty());
    let zero_evidence = zero.workload_evidence();
    assert_eq!(zero_evidence.plan, *session.summary());
    assert_eq!(
        zero_evidence.token_step_preparation,
        *session.preparation_report()
    );
    assert!(zero_evidence.fixed_prefill_preparation.is_none());
    assert!(zero_evidence.first_successful_run.is_none());
    assert_eq!(zero_evidence.prompt_prefill.token_count, 0);
    assert_eq!(zero_evidence.prompt_prefill.successful_invocation_count, 0);
    assert_eq!(zero_evidence.prompt_prefill.host_tokens_per_second(), None);
    assert_eq!(zero_evidence.steady_decode.token_count, 0);
    assert_eq!(zero_evidence.steady_decode.successful_invocation_count, 0);
    assert_eq!(zero_evidence.steady_decode.host_tokens_per_second(), None);
    assert_eq!(session.position(), 0);
    assert_eq!(mock.calls(), calls_after_prepare);

    let chat = session
        .generate_chat(
            &[LlamaChatMessage::new(LlamaChatRole::User, "a").unwrap()],
            0,
            LlamaSampling::Greedy,
        )
        .unwrap();
    assert!(chat.rendered_prompt().contains("assistant"));
    assert!(chat.generation().reports().is_empty());
    assert_eq!(session.position(), 0);
    assert_eq!(mock.calls(), calls_after_prepare);

    mock.clear_calls();
    let prefill = session.prefill_ids(&prompt).unwrap();
    assert_eq!(prefill.logits().shape(), expected_last.shape());
    for (actual, expected) in prefill.logits().values().iter().zip(expected_last.values()) {
        assert!(actual.is_finite() && (actual - expected).abs() <= 1e-3);
    }
    assert_eq!(prefill.reports().len(), 3);
    assert_eq!(prefill.reports()[0].retained_d2h_calls, 0);
    assert_eq!(prefill.reports()[1].retained_d2h_calls, 0);
    assert_eq!(prefill.reports()[2].retained_d2h_calls, 1);
    let evidence = prefill.workload_evidence();
    assert_eq!(evidence.plan, *session.summary());
    assert_eq!(evidence.plan.fallback_count, 0);
    assert_eq!(evidence.plan.nonzero_item_count, stable_kernels);
    assert!(evidence.plan.planned_device_bytes > 0);
    assert!(evidence.plan.resident_input_bytes > 0);
    assert_eq!(
        evidence.token_step_preparation,
        *session.preparation_report()
    );
    assert!(evidence.fixed_prefill_preparation.is_none());
    assert_eq!(evidence.prompt_prefill.token_count, prompt.len());
    assert_eq!(evidence.prompt_prefill.successful_invocation_count, 3);
    assert_eq!(evidence.prompt_prefill.transient_h2d_calls, 3);
    assert_eq!(evidence.prompt_prefill.transient_h2d_bytes, 12);
    assert_eq!(evidence.prompt_prefill.runtime_control_h2d_calls, 3);
    assert_eq!(evidence.prompt_prefill.runtime_control_h2d_bytes, 12);
    assert_eq!(evidence.prompt_prefill.retained_d2h_calls, 1);
    assert_eq!(
        evidence.prompt_prefill.kernel_launch_count,
        prefill
            .reports()
            .iter()
            .map(|report| report.kernel_launch_count)
            .sum::<usize>()
    );
    let first = evidence.first_successful_run.as_ref().unwrap();
    assert_eq!(first.token_count, 1);
    assert_eq!(first.successful_invocation_count, 1);
    assert_eq!(
        first.kernel_launch_count,
        prefill.reports()[0].kernel_launch_count
    );
    assert_eq!(evidence.steady_decode.successful_invocation_count, 0);
    assert!(
        prefill
            .reports()
            .iter()
            .all(|report| report.transient_h2d_calls == 1
                && report.transient_h2d_bytes == 4
                && report.runtime_control_h2d_calls == 1
                && report.runtime_control_h2d_bytes == 4)
    );
    assert_eq!(session.position(), prompt.len());
    assert_eq!(session.compiled_kernels().count(), stable_kernels);
    assert_eq!(session.capture().identity, stable_capture);
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
            || call.starts_with("queue_create:")
    }));

    let expected_generation = LlamaGenerator::new(&cpu_model, &cpu_tokenizer)
        .generate_ids(&prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let fresh = crate::models::transformer::LlamaMetalPlan::from_workflow(
        workflow,
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap();
    assert_eq!(fresh.step_deployment_identity(), stable_identity);
    let mut fresh = fresh.prepare().unwrap();
    let generation = fresh
        .generate_ids(&prompt, 2, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(generation.generation(), &expected_generation);
    let evidence = generation.workload_evidence();
    let decode_invocations = generation.reports().len() - prompt.len();
    assert_eq!(evidence.prompt_prefill.token_count, prompt.len());
    assert_eq!(
        evidence.prompt_prefill.successful_invocation_count,
        prompt.len()
    );
    assert_eq!(evidence.steady_decode.token_count, decode_invocations);
    assert_eq!(
        evidence.steady_decode.successful_invocation_count,
        decode_invocations
    );
    assert_eq!(
        decode_invocations,
        generation.generated_ids().len().saturating_sub(1)
    );
    assert_ne!(
        evidence.steady_decode.successful_invocation_count,
        generation.generated_ids().len()
    );
    assert_eq!(
        fresh.position(),
        prompt.len()
            + generation
                .generation()
                .generated_ids()
                .len()
                .saturating_sub(1)
    );
}

#[test]
fn llama_metal_fixed_span_prefill_shares_state_and_preserves_t1_tail() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (cpu_model, _) = crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let prompt = [3, 4, 5, 6];
    let expected = cpu_model.forward(&prompt).unwrap();
    let vocab = cpu_model.config().schema().vocab_size();
    let expected_last = TensorData::new(
        [1, vocab],
        expected.values()[expected.values().len() - vocab..].to_vec(),
    )
    .unwrap();

    let mock = Arc::new(MockDispatch::default());
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(5)));
    let device = test_device(mock.clone());
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let plan = crate::models::transformer::LlamaMetalPlan::from_workflow_with_prefill_span(
        workflow,
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(3).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.prefill_span_rows().unwrap().get(), 3);
    assert_eq!(plan.prefill_summary().unwrap().requested_output_count, 0);
    assert_eq!(
        plan.prefill_capture().unwrap().requested.len(),
        plan.prefill_summary().unwrap().state_pair_count
    );
    assert!(!plan.prefill_execution_plan().unwrap().items.is_empty());
    assert!(plan.prefill_rendered_items().unwrap().len() > 0);

    let mut session = plan.prepare().unwrap();
    assert_eq!(session.prefill_span_rows().unwrap().get(), 3);
    assert!(session.compiled_prefill_kernels().unwrap().count() > 0);
    let prefill_preparation = session.prefill_preparation_report().unwrap();
    assert_eq!(prefill_preparation.resident_h2d_calls, 0);
    assert_eq!(prefill_preparation.resident_h2d_bytes, 0);
    assert_eq!(prefill_preparation.initial_state_h2d_calls, 0);
    assert_eq!(prefill_preparation.initial_state_h2d_bytes, 0);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("queue_create:"))
            .count(),
        1
    );
    let calls_after_prepare = mock.calls();

    let prefill = session.prefill_ids(&prompt).unwrap();
    assert_eq!(prefill.logits().shape(), expected_last.shape());
    for (actual, expected) in prefill.logits().values().iter().zip(expected_last.values()) {
        assert!(actual.is_finite() && (actual - expected).abs() <= 1e-3);
    }
    assert_eq!(prefill.reports().len(), 2);
    assert_eq!(prefill.reports()[0].successful_invocation, 1);
    assert!(prefill.reports()[0].first_successful_run);
    assert_eq!(prefill.reports()[0].committed_state_position, Some(3));
    assert_eq!(prefill.reports()[0].retained_d2h_calls, 0);
    assert_eq!(prefill.reports()[0].output_count, 0);
    assert_eq!(prefill.reports()[1].successful_invocation, 2);
    assert!(!prefill.reports()[1].first_successful_run);
    assert_eq!(prefill.reports()[1].committed_state_position, Some(4));
    assert_eq!(prefill.reports()[1].retained_d2h_calls, 1);
    let evidence = prefill.workload_evidence();
    assert_eq!(evidence.plan, *session.summary());
    assert_eq!(evidence.plan.fallback_count, 0);
    assert!(evidence.plan.planned_device_bytes > 0);
    assert!(evidence.plan.resident_input_bytes > 0);
    assert_eq!(
        evidence.token_step_preparation,
        *session.preparation_report()
    );
    assert_eq!(
        evidence.fixed_prefill_preparation.as_ref(),
        session.prefill_preparation_report()
    );
    assert_eq!(evidence.prompt_prefill.token_count, prompt.len());
    assert_eq!(evidence.prompt_prefill.successful_invocation_count, 2);
    assert_eq!(
        evidence.prompt_prefill.gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(10))
    );
    assert_eq!(
        evidence
            .first_successful_run
            .as_ref()
            .unwrap()
            .gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(5))
    );
    assert_eq!(evidence.prompt_prefill.retained_d2h_calls, 1);
    assert_eq!(
        evidence.prompt_prefill.retained_d2h_bytes,
        std::mem::size_of_val(prefill.logits().values())
    );
    assert_eq!(
        evidence.first_successful_run.as_ref().unwrap().token_count,
        3
    );
    assert_eq!(evidence.steady_decode.successful_invocation_count, 0);
    assert_eq!(session.position(), 4);
    assert_eq!(session.successful_invocation_count(), 2);
    assert!(
        !mock.calls()[calls_after_prepare.len()..]
            .iter()
            .any(|call| {
                call.starts_with("buffer_create:")
                    || call.starts_with("library_compile:")
                    || call.starts_with("pipeline_create:")
                    || call.starts_with("queue_create:")
            })
    );

    let next = session.run_token(7).unwrap();
    assert_eq!(next.position(), 4);
    assert_eq!(next.report().successful_invocation, 3);
    assert_eq!(session.position(), 5);
    assert_eq!(session.successful_invocation_count(), 3);
}

#[test]
fn llama_metal_device_greedy_dense_prefill_matches_cpu_with_one_i32_read() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (cpu_model, cpu_tokenizer) =
        crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let prompt = [3, 4, 5, 6];
    let expected = LlamaGenerator::new(&cpu_model, &cpu_tokenizer)
        .generate_ids(&prompt, 3, LlamaSampling::Greedy)
        .unwrap();

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let plan = LlamaMetalGreedyPlan::from_workflow_with_prefill_span(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(3).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.summary().requested_output_count, 1);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(plan.prefill_summary().unwrap().requested_output_count, 0);
    assert_eq!(plan.prefill_summary().unwrap().fallback_count, 0);
    let token_identity = plan.step_deployment_identity();
    let prefill_identity = plan.prefill_deployment_identity().unwrap();
    assert_ne!(token_identity, 0);
    assert_ne!(prefill_identity, 0);

    let mut session = plan
        .prepare_with_scoreboard(
            MetalScoreboardContext::new(
                "llama-device-greedy-scoreboard",
                "test-revision",
                "semantic mock",
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        session
            .execution_scoreboard_report()
            .unwrap()
            .unwrap()
            .successful_run_count,
        0
    );
    let prefill_preparation = session.prefill_preparation_report().unwrap();
    assert_eq!(prefill_preparation.resident_h2d_calls, 0);
    assert_eq!(prefill_preparation.initial_state_h2d_calls, 0);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("queue_create:"))
            .count(),
        1
    );
    let actual = session.generate_ids(&prompt, 3).unwrap();
    assert_eq!(actual.generation(), &expected);
    assert!(
        actual
            .generated_ids()
            .iter()
            .all(|token| (*token as usize) < session.vocab_size())
    );
    for report in actual.reports() {
        match report.output_count {
            0 => {
                assert_eq!(report.retained_d2h_calls, 0);
                assert_eq!(report.retained_d2h_bytes, 0);
            }
            1 => {
                assert_eq!(report.retained_d2h_calls, 1);
                assert_eq!(report.retained_d2h_bytes, 4);
            }
            count => panic!("unexpected device-greedy output count {count}"),
        }
    }
    let evidence = actual.workload_evidence();
    assert_eq!(
        evidence.token_step_deployment_identity,
        session.token_step_deployment_identity()
    );
    assert_eq!(
        evidence.fixed_prefill_deployment_identity,
        session.fixed_prefill_deployment_identity()
    );
    assert_eq!(evidence.token_step_deployment_identity, token_identity);
    assert_eq!(
        evidence.fixed_prefill_deployment_identity,
        Some(prefill_identity)
    );
    assert_eq!(evidence.plan.fallback_count, 0);
    assert_eq!(
        evidence.fixed_prefill_plan.as_ref().unwrap().fallback_count,
        0
    );
    assert_eq!(evidence.prompt_prefill.token_count, prompt.len());
    assert_eq!(evidence.prompt_prefill.successful_invocation_count, 2);
    assert_eq!(evidence.prompt_prefill.retained_d2h_calls, 1);
    assert_eq!(evidence.prompt_prefill.retained_d2h_bytes, 4);
    assert_eq!(
        evidence.steady_decode.retained_d2h_bytes,
        evidence.steady_decode.successful_invocation_count * 4
    );
    let scoreboard = session.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(
        scoreboard.successful_run_count,
        u64::try_from(actual.reports().len()).unwrap()
    );
    assert_eq!(
        scoreboard.prompt_prefill.committed_token_count,
        prompt.len()
    );
    assert_eq!(scoreboard.prompt_prefill.successful_invocation_count, 2);
    assert_eq!(
        scoreboard.steady_decode.committed_token_count,
        actual.reports().len() - 2
    );
    assert_eq!(
        scoreboard.steady_decode.successful_invocation_count,
        u64::try_from(actual.reports().len() - 2).unwrap()
    );
    assert_eq!(scoreboard.standalone.successful_invocation_count, 0);
    let [fixed_prompt, token_prompt, decode @ ..] = scoreboard.successful_runs.as_slice() else {
        panic!("device-greedy scoreboard omitted prompt records");
    };
    assert_eq!(
        (
            fixed_prompt.program,
            fixed_prompt.phase,
            fixed_prompt.program_successful_invocation,
        ),
        (
            crate::LlamaMetalScoreboardProgram::FixedPrefill,
            crate::LlamaMetalScoreboardPhase::PromptPrefill,
            1,
        )
    );
    assert_eq!(
        (
            token_prompt.program,
            token_prompt.phase,
            token_prompt.program_successful_invocation,
        ),
        (
            crate::LlamaMetalScoreboardProgram::TokenStep,
            crate::LlamaMetalScoreboardPhase::PromptPrefill,
            1,
        )
    );
    assert!(fixed_prompt.first_successful_run);
    assert!(!token_prompt.first_successful_run);
    assert!(scoreboard.fixed_prefill.as_ref().unwrap().successful_runs[0].first_successful_run);
    assert!(scoreboard.token_step.successful_runs[0].first_successful_run);
    for (index, run) in decode.iter().enumerate() {
        assert_eq!(run.program, crate::LlamaMetalScoreboardProgram::TokenStep);
        assert_eq!(run.phase, crate::LlamaMetalScoreboardPhase::SteadyDecode);
        assert_eq!(
            run.program_successful_invocation,
            u64::try_from(index + 2).unwrap()
        );
    }
    assert_eq!(
        scoreboard.token_step.retained_host_api_d2h_bytes,
        usize::try_from(scoreboard.token_step.successful_run_count).unwrap() * 4
    );
    assert_eq!(
        scoreboard
            .fixed_prefill
            .as_ref()
            .unwrap()
            .retained_host_api_d2h_bytes,
        0
    );
    assert!(session.scoreboard_recording_error().is_none());
    assert_eq!(
        session.scoreboard_record_attempts(),
        Some(usize::try_from(scoreboard.token_step.successful_run_count).unwrap())
    );
}

#[test]
fn llama_metal_device_greedy_packed_matches_cpu_without_logits_download() {
    let bytes = packed_metal_workflow_bytes();
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (cpu_model, cpu_tokenizer) =
        crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let expected = LlamaGenerator::new(&cpu_model, &cpu_tokenizer)
        .generate_ids(&[3], 2, LlamaSampling::Greedy)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock);
    let plan = LlamaMetalGreedyPlan::from_workflow(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap();
    assert_eq!(plan.summary().requested_output_count, 1);
    assert_eq!(plan.summary().fallback_count, 0);
    assert!(plan.summary().quantized_constant_count > 0);
    let mut session = plan.prepare().unwrap();
    assert!(session.execution_scoreboard().is_none());
    assert_eq!(session.execution_scoreboard_report(), Ok(None));
    assert!(session.scoreboard_recording_error().is_none());
    let actual = session.generate_ids(&[3], 2).unwrap();
    assert_eq!(actual.generation(), &expected);
    assert!(actual.reports().iter().all(|report| {
        report.output_count == 1 && report.retained_d2h_calls == 1 && report.retained_d2h_bytes == 4
    }));
}

#[test]
fn llama_metal_device_greedy_scoreboard_failure_is_fail_soft_and_freezes_prefix() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        8,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let device = test_device(Arc::new(MockDispatch::default()));
    let mut session = LlamaMetalGreedyPlan::from_workflow(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap()
    .prepare_with_scoreboard(
        MetalScoreboardContext::new(
            "llama-device-greedy-fail-soft",
            "test-revision",
            "semantic mock",
        )
        .unwrap(),
    )
    .unwrap();
    session.inject_scoreboard_recording_error(MetalScoreboardError::Overflow);

    let generation = session.generate_ids(&[3], 2).unwrap();
    assert_eq!(generation.reports().len(), 2);
    assert_eq!(session.position(), 2);
    assert_eq!(session.successful_invocation_count(), 2);
    assert_eq!(session.scoreboard_record_attempts(), Some(0));
    assert_eq!(
        session.scoreboard_recording_error(),
        Some(&MetalScoreboardError::Overflow)
    );
    assert_eq!(
        session
            .execution_scoreboard()
            .unwrap()
            .report()
            .unwrap()
            .successful_run_count,
        0
    );
    assert_eq!(
        session.execution_scoreboard_report(),
        Err(MetalScoreboardError::Overflow)
    );
}

#[test]
fn llama_metal_device_greedy_returns_eos_without_appending_it() {
    let base = crate::models::transformer::model_tests::serialized_model_with_template(8, None);
    let file = crate::gguf::read_gguf(&base).unwrap();
    let (model, tokenizer) = crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let selected = LlamaGenerator::new(&model, &tokenizer)
        .generate_ids(&[3], 1, LlamaSampling::Greedy)
        .unwrap()
        .generated_ids()[0];
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template_and_eos(
        8,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
        selected,
    );
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let expected = LlamaGenerator::new(&model, &tokenizer)
        .generate_ids(&[3], 4, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(expected.generated_ids(), [selected]);
    assert!(expected.stopped());

    let device = test_device(Arc::new(MockDispatch::default()));
    let mut session = LlamaMetalGreedyPlan::from_workflow(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap()
    .prepare()
    .unwrap();
    let actual = session.generate_ids(&[3], 4).unwrap();
    assert_eq!(actual.generation(), &expected);
    assert_eq!(actual.reports().len(), 1);
    assert_eq!(session.position(), 1);
    assert_eq!(session.successful_invocation_count(), 1);
}

#[test]
fn llama_metal_workload_phase_uses_measured_time_and_saturating_counters() {
    fn report(
        successful_invocation: u64,
        first_successful_run: bool,
        run_wall_time: std::time::Duration,
        transaction_wall_time: std::time::Duration,
        counter: usize,
    ) -> MetalDeviceRunReport {
        MetalDeviceRunReport {
            successful_invocation,
            first_successful_run,
            run_wall_time,
            synchronous_transaction_wall_time: transaction_wall_time,
            transient_h2d_calls: counter,
            transient_h2d_bytes: counter,
            runtime_control_h2d_calls: counter,
            runtime_control_h2d_bytes: counter,
            retained_d2h_calls: counter,
            retained_d2h_bytes: counter,
            kernel_launch_count: counter,
            command_submission_count: counter,
            command_wait_count: counter,
            gpu_command_execution_time: (counter != 0).then_some(run_wall_time),
            zero_item_count: 0,
            output_count: 0,
            committed_state_pair_count: 0,
            committed_state_bytes: 0,
            committed_state_work_items: 0,
            committed_state_position: None,
        }
    }

    let no_elapsed = crate::models::transformer::LlamaMetalWorkloadPhase::from_reports(
        4,
        &[report(
            1,
            true,
            std::time::Duration::ZERO,
            std::time::Duration::ZERO,
            0,
        )],
    );
    assert_eq!(no_elapsed.host_tokens_per_second(), None);
    assert_eq!(no_elapsed.gpu_command_execution_time, None);

    let reports = [
        report(
            1,
            true,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(500),
            usize::MAX,
        ),
        report(
            2,
            false,
            std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(500),
            1,
        ),
    ];
    let measured = crate::models::transformer::LlamaMetalWorkloadPhase::from_reports(4, &reports);
    assert_eq!(measured.successful_invocation_count, 2);
    assert_eq!(
        measured.host_run_wall_time,
        std::time::Duration::from_secs(2)
    );
    assert_eq!(
        measured.host_synchronous_transaction_wall_time,
        std::time::Duration::from_secs(1)
    );
    assert_eq!(measured.host_tokens_per_second(), Some(2.0));
    assert_eq!(measured.kernel_launch_count, usize::MAX);
    assert_eq!(measured.command_submission_count, usize::MAX);
    assert_eq!(measured.command_wait_count, usize::MAX);
    assert_eq!(
        measured.gpu_command_execution_time,
        Some(std::time::Duration::from_secs(2))
    );
    assert_eq!(measured.transient_h2d_calls, usize::MAX);
    assert_eq!(measured.transient_h2d_bytes, usize::MAX);
    assert_eq!(measured.runtime_control_h2d_calls, usize::MAX);
    assert_eq!(measured.runtime_control_h2d_bytes, usize::MAX);
    assert_eq!(measured.retained_d2h_calls, usize::MAX);
    assert_eq!(measured.retained_d2h_bytes, usize::MAX);

    let mut unavailable = reports.clone();
    unavailable[1].gpu_command_execution_time = None;
    assert_eq!(
        crate::models::transformer::LlamaMetalWorkloadPhase::from_reports(4, &unavailable)
            .gpu_command_execution_time,
        None
    );
    let mut overflowing = reports.clone();
    overflowing[0].gpu_command_execution_time = Some(std::time::Duration::MAX);
    overflowing[1].gpu_command_execution_time = Some(std::time::Duration::from_nanos(1));
    assert_eq!(
        crate::models::transformer::LlamaMetalWorkloadPhase::from_reports(4, &overflowing)
            .gpu_command_execution_time,
        None
    );
}

#[test]
fn llama_metal_prefill_span_one_is_the_exact_t1_plan() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        8,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let device = test_device(Arc::new(MockDispatch::default()));
    let ordinary = crate::models::transformer::LlamaMetalPlan::from_workflow(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap();
    let span_one = crate::models::transformer::LlamaMetalPlan::from_workflow_with_prefill_span(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(1).unwrap(),
    )
    .unwrap();
    assert!(span_one.prefill_span_rows().is_none());
    assert!(span_one.prefill_capture().is_none());
    assert_eq!(span_one.capture().identity, ordinary.capture().identity);
    assert_eq!(
        span_one.step_deployment_identity(),
        ordinary.step_deployment_identity()
    );
    assert_eq!(span_one.summary(), ordinary.summary());
}

#[test]
fn llama_metal_prefill_chunk_failure_is_typed_atomic_and_retryable() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        8,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut session = crate::models::transformer::LlamaMetalPlan::from_workflow_with_prefill_span(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(3).unwrap(),
    )
    .unwrap()
    .prepare()
    .unwrap();
    mock.state.lock().unwrap().failures.launch = Some("fixed prefill launch");
    let failure = session.prefill_ids(&[3, 4, 5, 6]).unwrap_err();
    assert!(matches!(
        failure,
        crate::models::transformer::LlamaMetalGenerationError::PrefillChunkExecution {
            progress,
            token_offset: 0,
            span_rows: 3,
            ..
        } if progress.committed_position() == 0 && progress.reports().is_empty()
    ));
    assert_eq!(session.position(), 0);
    assert_eq!(session.successful_invocation_count(), 0);
    mock.clear_failures();
    let retry = session.prefill_ids(&[3, 4, 5, 6]).unwrap();
    assert_eq!(retry.reports().len(), 2);
    assert_eq!(retry.reports()[0].successful_invocation, 1);
    assert_eq!(session.position(), 4);
}

#[test]
fn llama_metal_fixed_span_scoreboard_preserves_component_identity_and_global_order() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut session = crate::models::transformer::LlamaMetalPlan::from_workflow_with_prefill_span(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(3).unwrap(),
    )
    .unwrap()
    .prepare_with_scoreboard(
        MetalScoreboardContext::new("llama-fixed-prefill", "test-revision", "semantic mock")
            .unwrap(),
    )
    .unwrap();

    let empty = session.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(empty.format_version, 2);
    assert_eq!(empty.successful_run_count, 0);
    assert_eq!(empty.committed_state_position, 0);
    assert_eq!(empty.token_step.append_span_rows, 1);
    assert_eq!(empty.token_step.successful_run_count, 0);
    assert_eq!(empty.fixed_prefill.as_ref().unwrap().append_span_rows, 3);
    assert_eq!(
        empty.fixed_prefill.as_ref().unwrap().successful_run_count,
        0
    );
    assert_ne!(
        empty.token_step.deployment_identity,
        empty.fixed_prefill.as_ref().unwrap().deployment_identity
    );
    for phase in [
        &empty.prompt_prefill,
        &empty.steady_decode,
        &empty.standalone,
    ] {
        assert_eq!(phase.committed_token_count, 0);
        assert_eq!(phase.successful_invocation_count, 0);
        assert_eq!(phase.gpu_command_execution_time, None);
        assert_eq!(phase.host_tokens_per_second(), None);
        assert_eq!(phase.gpu_command_tokens_per_second(), None);
    }

    mock.state.lock().unwrap().failures.launch = Some("fixed prefill launch");
    assert!(session.prefill_ids(&[3, 4, 5, 6]).is_err());
    assert_eq!(session.position(), 0);
    assert_eq!(
        session.execution_scoreboard_report().unwrap().unwrap(),
        empty
    );

    mock.clear_failures();
    let prefill = session.prefill_ids(&[3, 4, 5, 6]).unwrap();
    assert_eq!(prefill.reports().len(), 2);
    let report = session.execution_scoreboard_report().unwrap().unwrap();
    let fixed = report.fixed_prefill.as_ref().unwrap();
    assert_eq!(report.successful_run_count, 2);
    assert_eq!(report.committed_state_position, 4);
    assert_eq!(report.fallback_count, 0);
    assert_eq!(fixed.successful_run_count, 1);
    assert_eq!(fixed.committed_state_position, Some(3));
    assert_eq!(report.token_step.successful_run_count, 1);
    assert_eq!(report.token_step.committed_state_position, Some(4));
    assert_eq!(fixed.successful_runs[0].successful_invocation, 1);
    assert!(fixed.successful_runs[0].first_successful_run);
    assert_eq!(
        report.token_step.successful_runs[0].successful_invocation,
        1
    );
    assert!(report.token_step.successful_runs[0].first_successful_run);
    assert_eq!(report.prompt_prefill.committed_token_count, 4);
    assert_eq!(report.prompt_prefill.successful_invocation_count, 2);
    assert_eq!(report.prompt_prefill.gpu_command_execution_time, None);
    assert_eq!(report.steady_decode.successful_invocation_count, 0);
    assert_eq!(report.standalone.successful_invocation_count, 0);
    assert_eq!(
        report
            .successful_runs
            .iter()
            .map(|run| (
                run.successful_invocation,
                run.first_successful_run,
                run.program,
                run.phase,
                run.program_successful_invocation,
                run.append_span_rows,
                run.committed_state_position,
            ))
            .collect::<Vec<_>>(),
        [
            (
                1,
                true,
                crate::LlamaMetalScoreboardProgram::FixedPrefill,
                crate::LlamaMetalScoreboardPhase::PromptPrefill,
                1,
                3,
                3,
            ),
            (
                2,
                false,
                crate::LlamaMetalScoreboardProgram::TokenStep,
                crate::LlamaMetalScoreboardPhase::PromptPrefill,
                1,
                1,
                4,
            ),
        ]
    );
    assert_eq!(
        report.successful_runs[0].committed_state_bytes,
        fixed.append_state_row_bytes
    );
    assert_eq!(
        report.successful_runs[0].committed_state_work_items,
        fixed.append_state_work_items
    );
    assert_eq!(
        report.successful_runs[1].committed_state_bytes,
        report.token_step.append_state_row_bytes
    );
    assert_eq!(
        report.successful_runs[1].committed_state_work_items,
        report.token_step.append_state_work_items
    );
    assert_eq!(
        report.committed_state_bytes,
        prefill
            .reports()
            .iter()
            .map(|run| run.committed_state_bytes)
            .sum::<usize>()
    );
    assert_eq!(
        report.committed_state_work_items,
        prefill
            .reports()
            .iter()
            .map(|run| run.committed_state_work_items)
            .sum::<usize>()
    );
    let json: serde_json::Value = serde_json::from_slice(&report.to_json_bytes().unwrap()).unwrap();
    assert_eq!(json["format_version"], 2);
    assert_eq!(json["successful_runs"][0]["program"], "fixed_prefill");
    assert_eq!(json["successful_runs"][0]["phase"], "prompt_prefill");
    assert_eq!(json["successful_runs"][0]["append_span_rows"], 3);
    assert_eq!(json["successful_runs"][1]["program"], "token_step");
    assert_eq!(json["successful_runs"][1]["committed_state_position"], 4);
    assert_eq!(json["prompt_prefill"]["committed_token_count"], 4);
    assert_eq!(json["steady_decode"]["successful_invocation_count"], 0);
    assert_eq!(json["standalone"]["successful_invocation_count"], 0);

    let mut wrong_span = fixed.clone();
    wrong_span.append_span_rows = 2;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(wrong_span),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut wrong_position = report.successful_runs.clone();
    wrong_position[0].committed_state_position = 4;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_position,
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut wrong_bytes = report.successful_runs.clone();
    wrong_bytes[0].committed_state_bytes += 1;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_bytes,
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut wrong_work_items = report.successful_runs.clone();
    wrong_work_items[1].committed_state_work_items += 1;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_work_items,
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut wrong_order = report.successful_runs.clone();
    wrong_order.swap(0, 1);
    assert!(matches!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_order,
        ),
        Err(MetalScoreboardError::OutOfOrder { .. })
    ));
    let mut overflow_fixed = fixed.clone();
    overflow_fixed.append_span_rows = usize::MAX;
    overflow_fixed.successful_runs[0].committed_state_position = Some(usize::MAX);
    overflow_fixed.committed_state_position = Some(usize::MAX);
    let mut overflow_runs = report.successful_runs.clone();
    overflow_runs[0].append_span_rows = usize::MAX;
    overflow_runs[0].committed_state_position = usize::MAX;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(overflow_fixed),
            overflow_runs,
        ),
        Err(MetalScoreboardError::Overflow)
    );
    let mut wrong_phase = report.successful_runs.clone();
    wrong_phase[0].phase = crate::LlamaMetalScoreboardPhase::SteadyDecode;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_phase,
        ),
        Err(MetalScoreboardError::PlanMismatch)
    );
    let mut wrong_program = report.successful_runs.clone();
    wrong_program[1].program = crate::LlamaMetalScoreboardProgram::FixedPrefill;
    assert!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            report.fixed_prefill.clone(),
            wrong_program,
        )
        .is_err()
    );
    let mut wrong_component = fixed.clone();
    wrong_component.successful_runs[0].first_successful_run = false;
    assert!(matches!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(wrong_component),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::OutOfOrder { .. })
    ));
    let mut wrong_component = fixed.clone();
    wrong_component.committed_state_position = Some(4);
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(wrong_component),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut wrong_component = fixed.clone();
    wrong_component.successful_runs[0].kernel_launch_count += 1;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(wrong_component),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::StateCommitMismatch)
    );
    let mut fallback_component = fixed.clone();
    fallback_component.fallback_count = 1;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            report.token_step.clone(),
            Some(fallback_component),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::PlanMismatch)
    );
    let mut overflow_token = report.token_step.clone();
    overflow_token.successful_runs[0].run_wall_time = std::time::Duration::from_nanos(1);
    overflow_token.first_run_host_wall_time = Some(std::time::Duration::from_nanos(1));
    let mut overflow_fixed = fixed.clone();
    overflow_fixed.successful_runs[0].run_wall_time = std::time::Duration::MAX;
    overflow_fixed.first_run_host_wall_time = Some(std::time::Duration::MAX);
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            overflow_token,
            Some(overflow_fixed),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::Overflow)
    );
    let mut overflow_token = report.token_step.clone();
    overflow_token.successful_runs[0].kernel_launch_count = 1;
    overflow_token.kernel_launch_count = 1;
    let mut overflow_fixed = fixed.clone();
    overflow_fixed.successful_runs[0].kernel_launch_count = usize::MAX;
    overflow_fixed.kernel_launch_count = usize::MAX;
    assert_eq!(
        crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
            overflow_token,
            Some(overflow_fixed),
            report.successful_runs.clone(),
        ),
        Err(MetalScoreboardError::Overflow)
    );
    assert!(session.scoreboard_recording_error().is_none());

    let direct = session.run_token(7).unwrap();
    assert_eq!(direct.report().successful_invocation, 3);
    let direct_report = session.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(direct_report.prompt_prefill.committed_token_count, 4);
    assert_eq!(direct_report.standalone.committed_token_count, 1);
    assert_eq!(direct_report.standalone.successful_invocation_count, 1);
    assert_eq!(
        direct_report.successful_runs[2].phase,
        crate::LlamaMetalScoreboardPhase::Standalone
    );

    session.prefill_ids(&[3, 4, 5, 6]).unwrap();
    let interleaved = session.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(interleaved.successful_run_count, 5);
    assert_eq!(interleaved.committed_state_position, 9);
    assert_eq!(interleaved.prompt_prefill.committed_token_count, 8);
    assert_eq!(interleaved.standalone.committed_token_count, 1);
    assert_eq!(interleaved.token_step.successful_run_count, 3);
    assert_eq!(
        interleaved
            .token_step
            .successful_runs
            .iter()
            .map(|run| run.committed_state_position.unwrap())
            .collect::<Vec<_>>(),
        [4, 5, 9]
    );
    assert_eq!(
        interleaved
            .fixed_prefill
            .as_ref()
            .unwrap()
            .successful_runs
            .iter()
            .map(|run| run.committed_state_position.unwrap())
            .collect::<Vec<_>>(),
        [3, 8]
    );
}

#[test]
fn llama_metal_scoreboard_aggregates_mixed_prompt_and_decode_from_physical_runs() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let mock = Arc::new(MockDispatch::default());
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(5)));
    let device = test_device(mock);
    let mut session = crate::models::transformer::LlamaMetalPlan::from_workflow_with_prefill_span(
        LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap(),
        &device,
        MetalPlanOptions::new(8),
        NonZeroUsize::new(3).unwrap(),
    )
    .unwrap()
    .prepare_with_scoreboard(
        MetalScoreboardContext::new("llama-phase-scoreboard", "test-revision", "semantic mock")
            .unwrap(),
    )
    .unwrap();

    let vocab = session.vocab_size();
    let mut uniforms = vec![0.0; 2 * vocab];
    uniforms[4] = 1.0 - f32::EPSILON;
    uniforms[vocab + 5] = 1.0 - f32::EPSILON;
    let generation = session
        .generate_ids(
            &[3, 4, 5, 6],
            2,
            LlamaSampling::GumbelMax {
                temperature: f32::MAX,
                uniforms: &uniforms,
            },
        )
        .unwrap();
    assert_eq!(generation.generated_ids(), [4, 5]);
    let report = session.execution_scoreboard_report().unwrap().unwrap();
    let decode_invocations = generation.reports().len() - 2;
    assert_eq!(decode_invocations, 1);
    assert_eq!(report.prompt_prefill.committed_token_count, 4);
    assert_eq!(report.prompt_prefill.successful_invocation_count, 2);
    let prompt_runs = &generation.reports()[..2];
    assert_eq!(
        report.prompt_prefill.host_run_wall_time,
        prompt_runs.iter().map(|run| run.run_wall_time).sum()
    );
    assert_eq!(
        report.prompt_prefill.host_synchronous_transaction_wall_time,
        prompt_runs
            .iter()
            .map(|run| run.synchronous_transaction_wall_time)
            .sum()
    );
    let prompt_counters = prompt_runs.iter().fold([0usize; 9], |mut totals, run| {
        totals[0] += run.kernel_launch_count;
        totals[1] += run.command_submission_count;
        totals[2] += run.command_wait_count;
        totals[3] += run.transient_h2d_calls;
        totals[4] += run.transient_h2d_bytes;
        totals[5] += run.runtime_control_h2d_calls;
        totals[6] += run.runtime_control_h2d_bytes;
        totals[7] += run.retained_d2h_calls;
        totals[8] += run.retained_d2h_bytes;
        totals
    });
    assert_eq!(
        [
            report.prompt_prefill.kernel_launch_count,
            report.prompt_prefill.command_submission_count,
            report.prompt_prefill.command_wait_count,
            report.prompt_prefill.transient_host_api_h2d_calls,
            report.prompt_prefill.transient_host_api_h2d_bytes,
            report.prompt_prefill.runtime_control_host_api_h2d_calls,
            report.prompt_prefill.runtime_control_host_api_h2d_bytes,
            report.prompt_prefill.retained_host_api_d2h_calls,
            report.prompt_prefill.retained_host_api_d2h_bytes,
        ],
        prompt_counters
    );
    assert_eq!(
        report.prompt_prefill.gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(10))
    );
    let mut rate_fixture = report.prompt_prefill.clone();
    rate_fixture.host_run_wall_time = std::time::Duration::from_secs(2);
    assert_eq!(rate_fixture.host_tokens_per_second(), Some(2.0));
    rate_fixture.host_run_wall_time = std::time::Duration::ZERO;
    assert_eq!(rate_fixture.host_tokens_per_second(), None);
    rate_fixture.gpu_command_execution_time = Some(std::time::Duration::ZERO);
    assert_eq!(rate_fixture.gpu_command_tokens_per_second(), None);
    assert!(
        report
            .prompt_prefill
            .gpu_command_tokens_per_second()
            .is_some()
    );
    assert_eq!(
        report.steady_decode.successful_invocation_count,
        u64::try_from(decode_invocations).unwrap()
    );
    assert_eq!(
        report.steady_decode.committed_token_count,
        decode_invocations
    );
    assert_eq!(
        report.steady_decode.gpu_command_execution_time,
        (decode_invocations != 0).then_some(std::time::Duration::from_nanos(
            5 * u64::try_from(decode_invocations).unwrap()
        ))
    );
    assert_eq!(report.standalone.successful_invocation_count, 0);
    assert!(
        report.successful_runs[..2]
            .iter()
            .all(|run| { run.phase == crate::LlamaMetalScoreboardPhase::PromptPrefill })
    );
    assert!(report.successful_runs[2..].iter().all(|run| {
        run.phase == crate::LlamaMetalScoreboardPhase::SteadyDecode
            && run.program == crate::LlamaMetalScoreboardProgram::TokenStep
    }));

    let mut unavailable_token = report.token_step.clone();
    unavailable_token.successful_runs[1].gpu_command_execution_time = None;
    unavailable_token.gpu_command_execution_time = None;
    let unavailable = crate::models::transformer::LlamaMetalExecutionScoreboardReport::new(
        unavailable_token,
        report.fixed_prefill.clone(),
        report.successful_runs.clone(),
    )
    .unwrap();
    assert_eq!(
        unavailable.prompt_prefill.gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(10))
    );
    assert_eq!(unavailable.steady_decode.gpu_command_execution_time, None);
    assert_eq!(
        unavailable.steady_decode.gpu_command_tokens_per_second(),
        None
    );
}

#[test]
fn llama_metal_prompt_facade_preflights_and_reports_partial_commits() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let prepare = |mock: Arc<MockDispatch>| {
        let device = test_device(mock);
        let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
        crate::models::transformer::LlamaMetalPlan::from_workflow(
            workflow,
            &device,
            MetalPlanOptions::new(8),
        )
        .unwrap()
        .prepare()
        .unwrap()
    };

    let mock = Arc::new(MockDispatch::default());
    let mut session = prepare(mock.clone());
    mock.clear_calls();
    for invalid in [
        session.generate_ids(&[], 1, LlamaSampling::Greedy),
        session.generate_ids(&[u32::MAX], 1, LlamaSampling::Greedy),
        session.generate_ids(&[3, u32::MAX], 1, LlamaSampling::Greedy),
        session.generate_ids(&[3; 16], 1, LlamaSampling::Greedy),
        session.generate_ids(
            &[3],
            1,
            LlamaSampling::GumbelMax {
                temperature: 1.0,
                uniforms: &[],
            },
        ),
    ] {
        assert!(invalid.is_err());
        assert_eq!(session.position(), 0);
        assert!(mock.calls().is_empty());
    }

    for stage in ["launch", "wait"] {
        match stage {
            "launch" => mock.state.lock().unwrap().failures.launch = Some("prompt launch"),
            "wait" => mock.state.lock().unwrap().failures.wait = Some("prompt wait"),
            _ => unreachable!(),
        }
        let failure = session.prefill_ids(&[3, 4, 5]).unwrap_err();
        assert!(matches!(
            failure,
            crate::models::transformer::LlamaMetalGenerationError::Execution {
                progress,
                stage: crate::models::transformer::LlamaMetalGenerationStage::Prompt,
                token_offset: 0,
                token: 3,
                ..
            } if progress.committed_position() == 0 && progress.reports().is_empty()
        ));
        assert_eq!(session.position(), 0);
        mock.clear_failures();
    }

    mock.state.lock().unwrap().failures.read = Some("prompt final read");
    let failure = session.prefill_ids(&[3, 4, 5]).unwrap_err();
    assert!(matches!(
        failure,
        crate::models::transformer::LlamaMetalGenerationError::Execution {
            progress,
            stage: crate::models::transformer::LlamaMetalGenerationStage::Prompt,
            token_offset: 2,
            token: 5,
            ..
        } if progress.committed_position() == 2 && progress.reports().len() == 2
    ));
    assert_eq!(session.position(), 2);
    mock.clear_failures();
    let retry = session.run_token(5).unwrap();
    assert_eq!(retry.position(), 2);
    assert_eq!(session.position(), 3);

    mock.clear_calls();
    assert!(matches!(
        session.generate_ids(&[3], 1, LlamaSampling::Greedy),
        Err(
            crate::models::transformer::LlamaMetalGenerationError::FreshSessionRequired {
                position: 3
            }
        )
    ));
    assert!(mock.calls().is_empty());

    let mock = Arc::new(MockDispatch::default());
    let mut gumbel_session = prepare(mock);
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (model, tokenizer) = crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let vocab = model.config().schema().vocab_size();
    let uniforms = (0..2 * vocab)
        .map(|index| (index as f32 + 0.5) / (2 * vocab) as f32)
        .collect::<Vec<_>>();
    let sampling = LlamaSampling::GumbelMax {
        temperature: 0.75,
        uniforms: &uniforms,
    };
    let expected = LlamaGenerator::new(&model, &tokenizer)
        .generate_ids(&[3], 2, sampling)
        .unwrap();
    let actual = gumbel_session.generate_ids(&[3], 2, sampling).unwrap();
    assert_eq!(actual.generation(), &expected);

    for stop_token in [1usize, 2] {
        let mock = Arc::new(MockDispatch::default());
        let mut stop_session = prepare(mock.clone());
        let vocab = stop_session.vocab_size();
        let mut uniforms = vec![0.0; 4 * vocab];
        uniforms[stop_token] = 1.0 - f32::EPSILON;
        mock.clear_calls();
        let stopped = stop_session
            .generate_ids(
                &[3],
                4,
                LlamaSampling::GumbelMax {
                    temperature: f32::MAX,
                    uniforms: &uniforms,
                },
            )
            .unwrap();
        assert_eq!(stopped.generated_ids(), &[stop_token as u32]);
        assert!(stopped.stopped());
        assert_eq!(stopped.reports().len(), 1);
        assert_eq!(stop_session.position(), 1);
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("launch:"))
                .count(),
            stopped.reports()[0].kernel_launch_count
        );
    }

    let mock = Arc::new(MockDispatch::default());
    let mut failed_decode = prepare(mock.clone());
    let vocab = failed_decode.vocab_size();
    let kernels_per_token = failed_decode.summary().nonzero_item_count;
    let mut uniforms = vec![0.0; 2 * vocab];
    uniforms[4] = 1.0 - f32::EPSILON;
    mock.state.lock().unwrap().failures.launch_after =
        Some((kernels_per_token, "generated token launch"));
    let failure = failed_decode
        .generate_ids(
            &[3],
            2,
            LlamaSampling::GumbelMax {
                temperature: f32::MAX,
                uniforms: &uniforms,
            },
        )
        .unwrap_err();
    assert!(matches!(
        failure,
        crate::models::transformer::LlamaMetalGenerationError::Execution {
            progress,
            stage: crate::models::transformer::LlamaMetalGenerationStage::Decode,
            token_offset: 1,
            token: 4,
            ..
        } if progress.prompt_ids() == [3]
            && progress.generated_ids() == [4]
            && progress.reports().len() == 1
            && progress.committed_position()
                == progress.start_position() + progress.reports().len()
    ));
    assert_eq!(failed_decode.position(), 1);
    mock.clear_failures();
    let retry = failed_decode.run_token(4).unwrap();
    assert_eq!(retry.position(), 1);
    assert_eq!(retry.report().successful_invocation, 2);
    assert_eq!(failed_decode.position(), 2);
}

#[test]
fn llama_metal_prompt_facade_preserves_packed_and_tied_ownership() {
    let bytes = packed_metal_workflow_bytes();
    let file = crate::gguf::read_gguf(&bytes).unwrap();
    let (oracle, tokenizer) = crate::models::transformer::LlamaModel::from_gguf(&file).unwrap();
    let expected = LlamaGenerator::new(&oracle, &tokenizer)
        .generate_ids(&[3], 1, LlamaSampling::Greedy)
        .unwrap();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let plan = crate::models::transformer::LlamaMetalPlan::from_workflow(
        workflow,
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap();
    assert_eq!(plan.summary().quantized_constant_count, 1 + 2 * 7);
    assert_eq!(
        plan.capture()
            .quantized_constants
            .values()
            .map(|value| value.descriptor().ggml_type)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K])
    );
    let stable_owner = plan.selected_device_owner_id();
    let mut session = plan.prepare().unwrap();
    mock.clear_calls();
    let actual = session
        .generate_ids(&[3], 1, LlamaSampling::Greedy)
        .unwrap();
    assert_eq!(actual.generation(), &expected);
    assert_eq!(session.device_owner_id(), stable_owner);
    assert_eq!(actual.reports().len(), 1);
    assert_eq!(actual.reports()[0].retained_d2h_calls, 1);
    assert!(!mock.calls().iter().any(|call| {
        call.starts_with("buffer_create:")
            || call.starts_with("library_compile:")
            || call.starts_with("pipeline_create:")
            || call.starts_with("queue_create:")
    }));
}

#[test]
fn llama_metal_scoreboard_records_exact_token_execution_prefix_fail_soft() {
    let bytes = crate::models::transformer::model_tests::serialized_model_with_template(
        16,
        Some(LLAMA_SIMPLE_CHAT_TEMPLATE),
    );
    let make_plan = |device: &MetalDevice| {
        let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
        crate::models::transformer::LlamaMetalPlan::from_workflow(
            workflow,
            device,
            MetalPlanOptions::new(8),
        )
        .unwrap()
    };
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let ordinary = make_plan(&device);
    let ordinary_identity = ordinary.step_deployment_identity();
    let ordinary_cache_keys = ordinary
        .rendered_items()
        .map(|rendered| rendered.cache_key.clone())
        .collect::<Vec<_>>();
    let scored = make_plan(&device);
    assert_eq!(scored.step_deployment_identity(), ordinary_identity);
    assert_eq!(
        scored
            .rendered_items()
            .map(|rendered| rendered.cache_key.clone())
            .collect::<Vec<_>>(),
        ordinary_cache_keys
    );
    let mut session = scored
        .prepare_with_scoreboard(
            MetalScoreboardContext::new("llama-token-execution", "test-revision", "semantic mock")
                .unwrap(),
        )
        .unwrap();
    let empty = session.execution_scoreboard().unwrap().report().unwrap();
    assert_eq!(empty.format_version, 7);
    assert_eq!(empty.successful_run_count, 0);
    assert_eq!(empty.committed_state_position, Some(0));
    assert!(session.scoreboard_recording_error().is_none());

    let calls_after_prepare = mock.calls();
    let no_work = session
        .generate_ids(&[3, 4], 0, LlamaSampling::Greedy)
        .unwrap();
    assert!(no_work.reports().is_empty());
    assert_eq!(session.position(), 0);
    assert_eq!(mock.calls(), calls_after_prepare);
    assert_eq!(
        session.execution_scoreboard().unwrap().report().unwrap(),
        empty
    );

    let vocab = session.vocab_size();
    let mut uniforms = vec![0.0; 2 * vocab];
    uniforms[4] = 1.0 - f32::EPSILON;
    uniforms[vocab + 5] = 1.0 - f32::EPSILON;
    let generation = session
        .generate_ids(
            &[3, 4, 5],
            2,
            LlamaSampling::GumbelMax {
                temperature: f32::MAX,
                uniforms: &uniforms,
            },
        )
        .unwrap();
    assert_eq!(generation.generated_ids(), [4, 5]);
    assert_eq!(generation.reports().len(), 4);
    assert_eq!(session.position(), 4);
    assert_eq!(
        session.position(),
        generation.prompt_ids().len() + generation.generated_ids().len() - 1
    );
    let report = session.execution_scoreboard().unwrap().report().unwrap();
    assert_eq!(report.successful_run_count, 4);
    assert_eq!(report.committed_state_position, Some(4));
    assert_eq!(
        report
            .successful_runs
            .iter()
            .map(|run| run.retained_host_api_d2h_calls)
            .collect::<Vec<_>>(),
        [0, 0, 1, 1]
    );
    for (recorded, executed) in report.successful_runs.iter().zip(generation.reports()) {
        assert_eq!(
            recorded.successful_invocation,
            executed.successful_invocation
        );
        assert_eq!(recorded.kernel_launch_count, executed.kernel_launch_count);
        assert_eq!(
            recorded.committed_state_position,
            executed.committed_state_position
        );
    }
    assert!(session.scoreboard_recording_error().is_none());

    let stop_mock = Arc::new(MockDispatch::default());
    let stop_device = test_device(stop_mock);
    let mut stop = make_plan(&stop_device)
        .prepare_with_scoreboard(
            MetalScoreboardContext::new("llama-stop", "test-revision", "semantic mock").unwrap(),
        )
        .unwrap();
    let mut stop_uniforms = vec![0.0; 4 * stop.vocab_size()];
    stop_uniforms[1] = 1.0 - f32::EPSILON;
    let stopped = stop
        .generate_ids(
            &[3],
            4,
            LlamaSampling::GumbelMax {
                temperature: f32::MAX,
                uniforms: &stop_uniforms,
            },
        )
        .unwrap();
    assert_eq!(stopped.generated_ids(), [1]);
    assert!(stopped.stopped());
    assert_eq!(stopped.reports().len(), 1);
    assert_eq!(stop.position(), 1);
    assert_eq!(
        stop.execution_scoreboard()
            .unwrap()
            .report()
            .unwrap()
            .successful_run_count,
        1
    );

    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    let mut retry = make_plan(&device)
        .prepare_with_scoreboard(
            MetalScoreboardContext::new("llama-retry", "test-revision", "semantic mock").unwrap(),
        )
        .unwrap();
    mock.state.lock().unwrap().failures.launch = Some("token launch");
    assert!(retry.run_token(3).is_err());
    assert_eq!(retry.position(), 0);
    assert_eq!(
        retry
            .execution_scoreboard()
            .unwrap()
            .report()
            .unwrap()
            .successful_run_count,
        0
    );
    let empty_outer = retry.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(empty_outer.successful_run_count, 0);
    assert_eq!(empty_outer.standalone.successful_invocation_count, 0);
    mock.clear_failures();
    assert_eq!(retry.run_token(3).unwrap().position(), 0);
    assert_eq!(retry.position(), 1);
    assert_eq!(retry.scoreboard_record_attempts(), Some(1));
    assert_eq!(
        retry
            .execution_scoreboard()
            .unwrap()
            .report()
            .unwrap()
            .successful_run_count,
        1
    );
    let retry_outer = retry.execution_scoreboard_report().unwrap().unwrap();
    assert_eq!(retry_outer.standalone.committed_token_count, 1);
    assert_eq!(retry_outer.standalone.successful_invocation_count, 1);
    assert_eq!(
        retry_outer.successful_runs[0].phase,
        crate::LlamaMetalScoreboardPhase::Standalone
    );
    retry.inject_scoreboard_recording_error(MetalScoreboardError::Overflow);
    assert_eq!(retry.run_token(4).unwrap().position(), 1);
    assert_eq!(retry.run_token(5).unwrap().position(), 2);
    assert_eq!(retry.position(), 3);
    assert_eq!(retry.scoreboard_record_attempts(), Some(1));
    assert_eq!(
        retry.scoreboard_recording_error(),
        Some(&MetalScoreboardError::Overflow)
    );
    assert_eq!(
        retry
            .execution_scoreboard()
            .unwrap()
            .report()
            .unwrap()
            .successful_run_count,
        1
    );
    assert_eq!(
        retry.execution_scoreboard_report(),
        Err(MetalScoreboardError::Overflow)
    );
}

#[test]
fn llama_metal_scoreboard_preserves_packed_execution_evidence() {
    let bytes = packed_metal_workflow_bytes();
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock);
    let workflow = LlamaPromptWorkflow::from_gguf_bytes(&bytes).unwrap();
    let mut session = crate::models::transformer::LlamaMetalPlan::from_workflow(
        workflow,
        &device,
        MetalPlanOptions::new(8),
    )
    .unwrap()
    .prepare_with_scoreboard(
        MetalScoreboardContext::new("packed-llama", "test-revision", "semantic mock").unwrap(),
    )
    .unwrap();
    let step = session.run_token(3).unwrap();
    let report = session.execution_scoreboard().unwrap().report().unwrap();
    assert_eq!(report.successful_run_count, 1);
    assert_eq!(report.committed_state_position, Some(1));
    assert_eq!(
        report.captured_quantized_constant_count,
        session.summary().quantized_constant_count
    );
    assert_eq!(
        report.captured_quantized_constant_bytes,
        session.summary().quantized_constant_bytes
    );
    assert_eq!(
        report.successful_runs[0].successful_invocation,
        step.report().successful_invocation
    );
    assert_eq!(report.successful_runs[0].retained_host_api_d2h_calls, 1);
    assert!(session.scoreboard_recording_error().is_none());
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
    let logical_schedule_item_count = plan.execution_plan().schedule_item_count;
    let peak_logical_temporary_allocation_count = plan.execution_plan().peak_logical_allocations;
    let peak_logical_temporary_bytes = plan.execution_plan().peak_logical_bytes;
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
    assert!(empty.successful_runs.is_empty());
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
    assert_eq!(
        other_session
            .preparation_report()
            .cache_miss_pipeline_build_wall_time,
        std::time::Duration::ZERO
    );
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
    assert_eq!(report.successful_runs.len(), 3);
    for (recorded, observed) in report.successful_runs.iter().zip([&first, &second, &third]) {
        assert_eq!(
            recorded.successful_invocation,
            observed.report().successful_invocation
        );
        assert_eq!(recorded.run_wall_time, observed.report().run_wall_time);
        assert_eq!(
            recorded.transient_host_api_h2d_calls,
            observed.report().transient_h2d_calls
        );
        assert_eq!(
            recorded.transient_host_api_h2d_bytes,
            observed.report().transient_h2d_bytes
        );
        assert_eq!(
            recorded.retained_host_api_d2h_calls,
            observed.report().retained_d2h_calls
        );
        assert_eq!(
            recorded.retained_host_api_d2h_bytes,
            observed.report().retained_d2h_bytes
        );
    }
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
    assert_eq!(
        report.transient_host_api_h2d_calls,
        report
            .successful_runs
            .iter()
            .map(|run| run.transient_host_api_h2d_calls)
            .sum::<usize>()
    );
    assert_eq!(
        report.retained_host_api_d2h_bytes,
        report
            .successful_runs
            .iter()
            .map(|run| run.retained_host_api_d2h_bytes)
            .sum::<usize>()
    );
    assert_eq!(
        report.logical_schedule_item_count,
        logical_schedule_item_count
    );
    assert_eq!(
        report.peak_logical_temporary_allocation_count,
        peak_logical_temporary_allocation_count
    );
    assert_eq!(
        report.planned_physical_static_tensor_slot_bytes,
        session.summary().planned_device_bytes
    );
    assert_eq!(
        report.peak_logical_temporary_bytes,
        peak_logical_temporary_bytes
    );
    assert!(
        session
            .preparation_report()
            .cache_miss_pipeline_build_wall_time
            <= session.preparation_report().native_prepare_wall_time
    );
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
    assert_eq!(report.planned_physical_static_tensor_slot_count, 0);
    assert_eq!(report.planned_physical_static_tensor_slot_bytes, 0);
    assert_eq!(report.planned_kernel_count, 0);
    assert_eq!(report.planned_zero_item_count, 1);
    assert_eq!(report.kernel_launch_count, 0);
    assert_eq!(report.zero_item_count, 1);
    assert_eq!(report.resident_host_api_h2d_calls, 0);
    assert_eq!(report.transient_host_api_h2d_calls, 0);
    assert_eq!(report.retained_host_api_d2h_calls, 0);
    assert_eq!(report.fallback_count, 0);
    assert_eq!(report.successful_runs.len(), 1);
    assert_eq!(report.successful_runs[0].zero_item_count, 1);
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
        assert_eq!(run.report().command_submission_count, 1);
        assert_eq!(run.report().command_wait_count, 1);
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
    assert_eq!(run.report().command_submission_count, 0);
    assert_eq!(run.report().command_wait_count, 0);
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
    let mock = Arc::new(MockDispatch::virtual_zero_execution());
    let device = test_device(mock.clone());
    let plan =
        ResNetMetalPlan::eval_f32(&model, &device, [1, 3, 224, 224], MetalPlanOptions::new(64))
            .unwrap();
    let graph = plan.graph();
    let image = plan.image_input().node;
    let logits = plan.logits_node();
    assert_eq!(graph.shape(logits).unwrap(), &Shape::new([1, 1000]));
    assert_eq!(plan.logits_output().shape, Shape::new([1, 1000]));
    assert_eq!(plan.selected_device_owner_id(), device.owner_id());
    assert_eq!(plan.selected_device_info(), device.info());

    let scheduled = plan.capture();
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

    let capture_identity = plan.capture().identity;
    let roundtrip = CapturedSchedule::from_bytes(&plan.capture().to_bytes().unwrap()).unwrap();
    assert_eq!(roundtrip.identity, capture_identity);
    assert_eq!(plan.summary().capture_identity, capture_identity);
    assert_eq!(plan.summary().fallback_count, 0);
    assert_eq!(plan.summary().zero_item_count, 0);
    assert!(plan.summary().nonzero_item_count > 1);
    assert_eq!(plan.rendered_items().count(), scheduled.items.len());
    assert_eq!(
        plan.summary().rendered_cache_keys.len(),
        plan.summary().nonzero_item_count
    );
    assert!(!plan.resident_inputs().is_empty());
    assert_eq!(
        plan.transient_inputs()
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["image"]
    );
    let stable_summary = plan.summary().clone();
    let stable_resident_schema = plan.resident_inputs().to_vec();
    let cache = device.cache();
    assert!(cache.is_empty());

    // The plan owns the captured resident snapshots, not live module handles.
    let mut first_parameter = None;
    model.visit("", &mut |_, parameter, kind| {
        if first_parameter.is_none() && matches!(kind, StateKind::Parameter) {
            first_parameter = Some(parameter.clone());
        }
    });
    let changed = first_parameter.expect("ResNet has trainable parameters");
    let snapshot = changed.snapshot().unwrap();
    let changed_elements = snapshot.shape.numel().unwrap();
    changed
        .replace(TensorData::new(snapshot.shape, vec![1.0; changed_elements]).unwrap())
        .unwrap();
    drop(model);

    // Full host evaluation is intentionally excluded here: the semantic mock
    // validates every registered kernel/slot ABI but virtualizes large device
    // allocations. With zero trainable parameters, exact source logits are
    // zero for any finite image, so zero-filled retained output is observable
    // source truth without billions of host convolution operations.
    let mut session = plan.prepare().unwrap();
    assert!(mock.registered_semantic_program_count() > 0);
    assert!(
        mock.registered_semantic_program_count()
            <= session.metal_session().summary().nonzero_item_count
    );
    assert_eq!(
        session.metal_session().compiled_kernels().count(),
        session.metal_session().summary().nonzero_item_count
    );
    let prepared_owner = session.metal_session().device_owner_id();
    let planned_slots = session.metal_session().summary().planned_slot_count;
    let stable_cache_len = cache.len();
    assert_eq!(session.metal_session().summary(), &stable_summary);
    assert_eq!(
        session.metal_session().resident_inputs(),
        stable_resident_schema.as_slice()
    );

    mock.clear_calls();
    assert!(matches!(
        session.run(TensorData::zeros([1, 3, 223, 224]).unwrap()),
        Err(ResNetMetalError::InvalidImage { .. })
    ));
    assert!(matches!(
        session.run(TensorData::zeros_with_dtype([1, 3, 224, 224], DType::I32).unwrap()),
        Err(ResNetMetalError::InvalidImage { .. })
    ));
    assert!(mock.calls().is_empty());
    assert_eq!(session.metal_session().successful_run_count(), 0);

    mock.state.lock().unwrap().failures.launch = Some("ResNet facade retry");
    assert!(
        session
            .run(TensorData::zeros([1, 3, 224, 224]).unwrap())
            .is_err()
    );
    assert_eq!(session.metal_session().successful_run_count(), 0);
    mock.clear_failures();

    let mut observed_bindings = None;
    for invocation in 0..2 {
        mock.clear_calls();
        mock.clear_launch_bindings();
        let run = session
            .run(TensorData::zeros([1, 3, 224, 224]).unwrap())
            .unwrap();
        assert_eq!(run.logits(), &TensorData::zeros([1, 1000]).unwrap());
        assert_eq!(run.report().successful_invocation, invocation as u64 + 1);
        assert_eq!(
            run.report().kernel_launch_count,
            session.metal_session().summary().nonzero_item_count
        );
        assert_eq!(run.report().command_submission_count, 1);
        assert_eq!(run.report().command_wait_count, 1);
        assert_eq!(run.report().transient_h2d_calls, 1);
        assert_eq!(run.report().retained_d2h_calls, 1);
        assert_eq!(session.metal_session().device_owner_id(), prepared_owner);
        assert_eq!(cache.len(), stable_cache_len);
        assert_eq!(session.metal_session().summary(), &stable_summary);
        assert_eq!(
            session.metal_session().resident_inputs(),
            stable_resident_schema.as_slice()
        );
        assert_eq!(
            session.metal_session().summary().planned_slot_count,
            planned_slots
        );
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("batch_submit:"))
                .count(),
            1
        );
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("wait:"))
                .count(),
            1
        );
        assert!(!mock.calls().iter().any(|call| {
            call.starts_with("buffer_create:")
                || call.starts_with("library_compile:")
                || call.starts_with("pipeline_create:")
                || call.starts_with("queue_create:")
        }));
        let bindings = mock.launch_bindings();
        assert_eq!(
            bindings.len(),
            session.metal_session().summary().nonzero_item_count
        );
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
fn packed_gguf_constants_upload_once_and_execute_direct_metal_matmul() {
    assert_eq!(
        METAL_QUANTIZED_MATMUL_RENDERER_VERSION,
        "rustgrad-metal-quantized-matmul-f32-v1"
    );
    assert_eq!(
        METAL_QUANTIZED_ROW_GATHER_RENDERER_VERSION,
        "rustgrad-metal-quantized-row-gather-v1"
    );
    let mut format_cache_keys = BTreeSet::new();
    for kind in [GgmlType::Q4_0, GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K] {
        let weight = packed_ones(kind, 2);
        let columns = weight.descriptor().logical_shape.dims()[1];
        let capture = CapturedSchedule::capture_quantized_matmul(
            "activation",
            NodeId::from_index(80),
            NodeId::from_index(81),
            NodeId::from_index(82),
            Shape::from([1, columns]),
            weight.clone(),
        )
        .unwrap();
        let artifact = capture.to_bytes().unwrap();
        assert_eq!(
            CapturedSchedule::from_bytes(&artifact)
                .unwrap()
                .to_bytes()
                .unwrap(),
            artifact
        );
        let renderer = MetalRenderer::new(8, capabilities()).unwrap();
        let plan = MetalDeviceSessionPlan::from_capture(
            capture.clone(),
            Vec::<String>::new(),
            renderer.clone(),
        )
        .unwrap();
        let rendered = plan.rendered_items().next().unwrap();
        assert!(rendered.source.contains("device const uchar* b1"));
        assert!(
            rendered
                .source
                .contains(METAL_QUANTIZED_MATMUL_RENDERER_VERSION)
        );
        match kind {
            GgmlType::Q4_0 => assert!(rendered.source.contains("rg_lane < 16ul")),
            GgmlType::Q8_0 => assert!(rendered.source.contains("as_type<char>")),
            GgmlType::Q4K => {
                assert!(rendered.source.contains("rg_g < 4ul"));
                assert!(rendered.source.contains("rg_dm"));
            }
            GgmlType::Q6K => {
                assert!(rendered.source.contains("rg_b + 208ul"));
                assert!(rendered.source.contains("rg_scale"));
            }
            _ => unreachable!("admitted packed format"),
        }
        assert!(format_cache_keys.insert(rendered.cache_key.clone()));
        assert_eq!(&rendered.quantized_buffers[0].desc, weight.descriptor());
        assert_eq!(plan.summary().quantized_constant_count, 1);
        assert_eq!(plan.summary().fallback_count, 0);
        assert_eq!(
            plan.summary().quantized_constant_bytes,
            weight.bytes().len()
        );
        assert_eq!(plan.summary().planned_slot_count, 3);
        assert_eq!(
            plan.summary().planned_device_bytes,
            columns * DType::F32.itemsize() + 2 * DType::F32.itemsize() + weight.bytes().len()
        );

        let mock = Arc::new(MockDispatch::default());
        let device = test_device(mock.clone());
        mock.clear_calls();
        let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
        assert_eq!(session.preparation_report().resident_h2d_calls, 1);
        assert_eq!(
            session.preparation_report().resident_h2d_bytes,
            weight.bytes().len()
        );
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("write:"))
                .count(),
            1
        );
        let cache_keys = session
            .compiled_kernels()
            .map(|kernel| kernel.cache_key.clone())
            .collect::<Vec<_>>();
        mock.clear_calls();
        mock.clear_launch_bindings();
        let activation = TensorData::new([1, columns], vec![1.0; columns]).unwrap();
        let inputs = BTreeMap::from([("activation".into(), activation)]);
        let first = session.run(&inputs).unwrap();
        let second = session.run(&inputs).unwrap();
        for run in [&first, &second] {
            let Storage::F32(values) = run.outputs()[0].storage() else {
                panic!("packed Metal matmul output")
            };
            assert_eq!(values.len(), 2);
            assert!(values.iter().all(|value| {
                (*value - columns as f32).abs() <= (columns as f32 * 1.0e-5).max(1.0e-5)
            }));
            assert_eq!(run.report().transient_h2d_calls, 1);
            assert_eq!(run.report().retained_d2h_calls, 1);
            assert_eq!(run.report().kernel_launch_count, 1);
        }
        assert_eq!(
            session
                .compiled_kernels()
                .map(|kernel| kernel.cache_key.as_str())
                .collect::<Vec<_>>(),
            cache_keys.iter().map(String::as_str).collect::<Vec<_>>()
        );
        assert_eq!(session.successful_run_count(), 2);
        assert_eq!(mock.launch_bindings().len(), 2);
        assert_eq!(mock.launch_bindings()[0], mock.launch_bindings()[1]);
        assert_eq!(
            mock.calls()
                .iter()
                .filter(|call| call.starts_with("write:"))
                .count(),
            2,
            "only the two transient activations are uploaded after preparation"
        );

        let plain_mock = Arc::new(MockDispatch::default());
        let plain_device = test_device(plain_mock.clone());
        plain_mock.clear_calls();
        assert!(matches!(
            PreparedMetalPrefix::prepare(plain_device.clone(), &capture.items, renderer),
            Err(MetalError::Unsupported(_))
        ));
        assert!(plain_mock.calls().is_empty());
        drop(plain_device);
        assert_eq!(plain_mock.calls(), vec!["device_release:1".to_owned()]);
    }
    assert_eq!(format_cache_keys.len(), 4);

    let first = packed_ones(GgmlType::Q4_0, 1);
    let mut changed_bytes = first.bytes().to_vec();
    *changed_bytes.last_mut().unwrap() ^= 1;
    let changed =
        QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([1, 32]), changed_bytes).unwrap();
    let plans = [first, changed]
        .into_iter()
        .map(|weight| {
            let capture = CapturedSchedule::capture_quantized_matmul(
                "activation",
                NodeId::from_index(120),
                NodeId::from_index(121),
                NodeId::from_index(122),
                Shape::from([1, 32]),
                weight,
            )
            .unwrap();
            let identity = capture.identity;
            let plan = MetalDeviceSessionPlan::from_capture(
                capture,
                Vec::<String>::new(),
                MetalRenderer::new(8, capabilities()).unwrap(),
            )
            .unwrap();
            (
                identity,
                plan.rendered_items().next().unwrap().cache_key.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_ne!(plans[0].0, plans[1].0);
    assert_ne!(plans[0].1, plans[1].1);

    let unsupported =
        QuantizedTensorData::new(GgmlType::Q4_1, Shape::from([1, 32]), vec![0; 20]).unwrap();
    let capture = CapturedSchedule::capture_quantized_matmul(
        "activation",
        NodeId::from_index(83),
        NodeId::from_index(84),
        NodeId::from_index(85),
        Shape::from([1, 32]),
        unsupported,
    )
    .unwrap();
    assert!(matches!(
        MetalDeviceSessionPlan::from_capture(
            capture,
            Vec::<String>::new(),
            MetalRenderer::new(8, capabilities()).unwrap(),
        ),
        Err(MetalError::Unsupported(_))
    ));
}

#[test]
fn packed_row_gather_preflights_indices_and_retries_without_reuploading_weight() {
    let weight = packed_ones(GgmlType::Q4_0, 3);
    let capture = CapturedSchedule::capture_quantized_row_gather(
        "indices",
        NodeId::from_index(90),
        NodeId::from_index(91),
        NodeId::from_index(92),
        Shape::from([1, 3]),
        DType::I32,
        weight.clone(),
    )
    .unwrap();
    let mut tampered = capture.clone();
    tampered
        .quantized_constants
        .insert(91, packed_ones(GgmlType::Q8_0, 3));
    assert!(matches!(
        MetalDeviceSessionPlan::from_capture(
            tampered,
            Vec::<String>::new(),
            MetalRenderer::new(8, capabilities()).unwrap(),
        ),
        Err(MetalError::InvalidBinding(_))
    ));
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let plan =
        MetalDeviceSessionPlan::from_capture(capture.clone(), Vec::<String>::new(), renderer)
            .unwrap();
    let rendered = plan.rendered_items().next().unwrap();
    assert!(
        rendered
            .source
            .contains(METAL_QUANTIZED_ROW_GATHER_RENDERER_VERSION)
    );
    assert!(!rendered.source.contains("status"));
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    mock.state.lock().unwrap().failures.write = Some("packed resident upload");
    assert!(plan.prepare(device.clone(), BTreeMap::new()).is_err());
    assert!(
        mock.calls()
            .iter()
            .any(|call| call.starts_with("buffer_release:"))
    );
    mock.clear_failures();
    mock.clear_calls();
    let plan = MetalDeviceSessionPlan::from_capture(
        capture,
        Vec::<String>::new(),
        MetalRenderer::new(8, capabilities()).unwrap(),
    )
    .unwrap();
    let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 1);
    mock.clear_calls();

    let invalid = BTreeMap::from([(
        "indices".into(),
        TensorData::from_storage([1, 3], Storage::I32(vec![0, -1, 2])).unwrap(),
    )]);
    assert!(matches!(
        session.run(&invalid),
        Err(MetalError::IndexOutOfBounds {
            axis: 0,
            index: 1,
            value: -1,
            dim: 3
        })
    ));
    assert!(mock.calls().is_empty());
    assert_eq!(session.successful_run_count(), 0);

    let valid = BTreeMap::from([(
        "indices".into(),
        TensorData::from_storage([1, 3], Storage::I32(vec![2, 0, 1])).unwrap(),
    )]);
    mock.state.lock().unwrap().failures.launch = Some("packed gather retry");
    assert!(session.run(&valid).is_err());
    assert_eq!(session.successful_run_count(), 0);
    mock.clear_failures();
    mock.clear_calls();
    let run = session.run(&valid).unwrap();
    let Storage::F32(values) = run.outputs()[0].storage() else {
        panic!("packed Metal gather output")
    };
    assert_eq!(values, &vec![1.0; 96]);
    assert_eq!(run.report().transient_h2d_calls, 1);
    assert_eq!(run.report().retained_d2h_calls, 1);
    assert_eq!(run.report().kernel_launch_count, 1);
    assert_eq!(session.successful_run_count(), 1);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("write:"))
            .count(),
        1,
        "retry uploads only indices, never the immutable packed weight"
    );
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("read:"))
            .count(),
        1,
        "row-gather reads only its requested dense result"
    );
    assert!(
        mock.calls()
            .iter()
            .all(|call| !call.starts_with("buffer_create:")),
        "row-gather run owns no candidate or status allocation"
    );
}

#[test]
fn packed_zero_work_is_resource_free_and_zero_inner_dimension_uses_a_sentinel() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let weight = packed_ones(GgmlType::Q4_0, 2);
    let capture = CapturedSchedule::capture_quantized_matmul(
        "activation",
        NodeId::from_index(100),
        NodeId::from_index(101),
        NodeId::from_index(102),
        Shape::from([0, 32]),
        weight,
    )
    .unwrap();
    let plan =
        MetalDeviceSessionPlan::from_capture(capture, Vec::<String>::new(), renderer.clone())
            .unwrap();
    assert_eq!(plan.summary().zero_item_count, 1);
    assert_eq!(plan.summary().planned_slot_count, 0);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 0);
    let run = session
        .run(&BTreeMap::from([(
            "activation".into(),
            TensorData::new([0, 32], vec![]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(run.outputs()[0].shape(), &Shape::from([0, 2]));
    assert_eq!(run.report().kernel_launch_count, 0);
    assert!(mock.calls().is_empty());

    let empty_weight =
        QuantizedTensorData::new(GgmlType::Q4_0, Shape::from([2, 0]), vec![]).unwrap();
    let capture = CapturedSchedule::capture_quantized_matmul(
        "activation",
        NodeId::from_index(110),
        NodeId::from_index(111),
        NodeId::from_index(112),
        Shape::from([1, 0]),
        empty_weight,
    )
    .unwrap();
    let plan =
        MetalDeviceSessionPlan::from_capture(capture, Vec::<String>::new(), renderer).unwrap();
    assert_eq!(plan.summary().zero_byte_sentinel_count, 2);
    let mock = Arc::new(MockDispatch::default());
    let device = test_device(mock.clone());
    mock.clear_calls();
    let mut session = plan.prepare(device, BTreeMap::new()).unwrap();
    assert_eq!(session.preparation_report().resident_h2d_calls, 0);
    let run = session
        .run(&BTreeMap::from([(
            "activation".into(),
            TensorData::new([1, 0], vec![]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(run.outputs()[0].storage(), &Storage::F32(vec![0.0; 2]));
    assert_eq!(run.report().transient_h2d_calls, 0);
    assert_eq!(run.report().kernel_launch_count, 1);
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
    gpu_command_execution_time: Option<std::time::Duration>,
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

    fn set_gpu_command_execution_time(&self, value: Option<std::time::Duration>) {
        self.state.lock().unwrap().gpu_command_execution_time = value;
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
        let expected_buffers = semantics.pointer_order.len()
            + usize::from(transaction.is_some() || indexed_movement.is_some());
        if geometry.extent as usize != semantics.extent
            || geometry.extent_index != semantics.pointer_order.len()
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
            for (position, (raw, pointer)) in
                buffers.iter().zip(&semantics.pointer_order).enumerate()
            {
                let logical = match pointer {
                    renderer::MetalPointerAbi::Dense(index) => semantics.buffers[*index]
                        .elements
                        .checked_mul(semantics.buffers[*index].dtype.itemsize())
                        .ok_or(MetalError::Overflow)?,
                    renderer::MetalPointerAbi::Quantized(index) => {
                        semantics.quantized_buffers[*index].desc.bytes
                    }
                };
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
        let mut quantized = BTreeMap::<u64, QuantizedTensorData>::new();
        let mut outputs = Vec::new();
        for (position, (raw, pointer)) in buffers.iter().zip(&semantics.pointer_order).enumerate() {
            let (logical, dense) = match pointer {
                renderer::MetalPointerAbi::Dense(index) => {
                    let abi = &semantics.buffers[*index];
                    (
                        abi.elements
                            .checked_mul(abi.dtype.itemsize())
                            .ok_or(MetalError::Overflow)?,
                        Some(abi),
                    )
                }
                renderer::MetalPointerAbi::Quantized(index) => {
                    (semantics.quantized_buffers[*index].desc.bytes, None)
                }
            };
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
            let Some(abi) = dense else {
                let packed_abi = match pointer {
                    renderer::MetalPointerAbi::Quantized(index) => {
                        &semantics.quantized_buffers[*index]
                    }
                    renderer::MetalPointerAbi::Dense(_) => unreachable!("dense handled above"),
                };
                let value = QuantizedTensorData::new(
                    packed_abi.desc.ggml_type,
                    packed_abi.desc.logical_shape.clone(),
                    bytes[..logical].to_vec(),
                )
                .map_err(|error| MetalError::InvalidBinding(error.to_string()))?;
                if value.descriptor() != &packed_abi.desc
                    || quantized.insert(packed_abi.id, value).is_some()
                {
                    return Err(MetalError::InvalidBinding(
                        "mock packed buffer descriptor mismatch".into(),
                    ));
                }
                continue;
            };
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
        // This is RustGrad's retained or authenticated typed semantic
        // artifact, not CpuBackend or native Metal. Captured random stays
        // graph-free and immutable.
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
            dispatch::KernelSemanticProgram::QuantizedMatmul(plan) => {
                let activation = bindings
                    .get(plan.activation.index() as u64)
                    .ok_or_else(|| {
                        MetalError::InvalidBinding("quantized matmul activation absent".into())
                    })?;
                let weight = quantized
                    .get(&(plan.weight.index() as u64))
                    .ok_or_else(|| {
                        MetalError::InvalidBinding("quantized matmul weight absent".into())
                    })?;
                vec![
                    plan.execute(activation, weight)
                        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?,
                ]
            }
            dispatch::KernelSemanticProgram::QuantizedRowGather(plan) => {
                let indices = bindings.get(plan.indices.index() as u64).ok_or_else(|| {
                    MetalError::InvalidBinding("quantized gather indices absent".into())
                })?;
                let weight = quantized
                    .get(&(plan.weight.index() as u64))
                    .ok_or_else(|| {
                        MetalError::InvalidBinding("quantized gather weight absent".into())
                    })?;
                vec![
                    plan.execute(indices, weight)
                        .map_err(|error| MetalError::InvalidBinding(error.to_string()))?,
                ]
            }
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

    fn launch_batch(
        &self,
        queue: RawQueue,
        launches: &[dispatch::BatchLaunch],
        owner: u64,
    ) -> Result<RawCommand, MetalError> {
        if launches.is_empty() {
            return Err(MetalError::InvalidArgument("empty mock Metal launch batch"));
        }
        let mut encoded = Vec::with_capacity(launches.len());
        for launch in launches {
            match self.launch(
                queue,
                launch.pipeline,
                &launch.buffers,
                launch.geometry,
                owner,
            ) {
                Ok(command) => encoded.push(command),
                Err(error) => {
                    let mut state = self.state.lock().unwrap();
                    for command in encoded {
                        state.commands.remove(&(owner, command.0));
                    }
                    state.calls.push(format!("batch_abort:{owner}"));
                    return Err(error);
                }
            }
        }
        let mut state = self.state.lock().unwrap();
        for command in encoded {
            state.commands.remove(&(owner, command.0));
        }
        state
            .calls
            .push(format!("batch_submit:{owner}:{}", launches.len()));
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

    fn command_gpu_duration(&self, command: RawCommand, owner: u64) -> Option<std::time::Duration> {
        let mut state = self.state.lock().unwrap();
        if state.commands.get(&(owner, command.0)) != Some(&true) {
            return None;
        }
        state.calls.push(format!("gpu_time:{owner}"));
        state.gpu_command_execution_time
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
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(7)));
    let mut session = plan
        .prepare(test_device(mock.clone()), BTreeMap::new())
        .unwrap();
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
    assert_eq!(run.report().command_submission_count, 2);
    assert_eq!(run.report().command_wait_count, 2);
    assert_eq!(
        run.report().gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(14))
    );
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("batch_submit:"))
    );
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
    assert_eq!(run.report().gpu_command_execution_time, None);

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
fn metal_batch_validates_every_launch_before_one_ordered_submission_and_retains_resources() {
    let renderer = MetalRenderer::new(8, capabilities()).unwrap();
    let mut graph = Graph::new();
    let first = graph.uniform([2], -1.0, 1.0, DType::F32, 41).unwrap();
    let second = graph.randint([3], -4, 7, DType::I32, 42).unwrap();
    let first_root = crate::kernel::lower_graph_random(&graph, first).unwrap();
    let second_root = crate::kernel::lower_graph_random(&graph, second).unwrap();
    let first_rendered = renderer.render(&first_root).unwrap();
    let second_rendered = renderer.render(&second_root).unwrap();
    let mock = Arc::new(MockDispatch::default());
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(19)));
    let (device, queue) = setup(mock.clone());
    let first_pipeline = device.cache().load(&first_rendered).unwrap();
    let second_pipeline = device.cache().load(&second_rendered).unwrap();
    let first_output = device.allocate_typed(2, DType::F32).unwrap();
    let second_output = device.allocate_typed(3, DType::I32).unwrap();
    let first_bindings = [&first_output];
    let second_bindings = [&second_output];

    mock.clear_calls();
    let command = queue
        .launch_batch(&[
            MetalBatchItem {
                pipeline: first_pipeline.as_ref(),
                bindings: &first_bindings,
                local_size: 8,
                capture_initialized: false,
            },
            MetalBatchItem {
                pipeline: second_pipeline.as_ref(),
                bindings: &second_bindings,
                local_size: 8,
                capture_initialized: false,
            },
        ])
        .unwrap()
        .unwrap();
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("launch:"))
            .count(),
        2
    );
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("batch_submit:"))
            .count(),
        1
    );
    assert!(!mock.calls().iter().any(|call| call.starts_with("wait:")));
    let completion = command.collect().unwrap();
    assert_eq!(completion.extent, 5);
    assert_eq!(completion.retained_resources, 2);
    assert_eq!(
        completion.gpu_command_execution_time,
        Some(std::time::Duration::from_nanos(19))
    );
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("wait:"))
            .count(),
        1
    );
    let calls = mock.calls();
    let wait = calls
        .iter()
        .position(|call| call.starts_with("wait:"))
        .unwrap();
    let gpu_time = calls
        .iter()
        .position(|call| call.starts_with("gpu_time:"))
        .unwrap();
    let release = calls
        .iter()
        .position(|call| call.starts_with("command_release:"))
        .unwrap();
    assert!(wait < gpu_time && gpu_time < release);

    mock.set_gpu_command_execution_time(None);
    let unavailable = queue
        .launch_batch(&[MetalBatchItem {
            pipeline: first_pipeline.as_ref(),
            bindings: &first_bindings,
            local_size: 8,
            capture_initialized: false,
        }])
        .unwrap()
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(unavailable.gpu_command_execution_time, None);

    let empty_bindings: [&MetalBuffer; 0] = [];
    mock.clear_calls();
    assert!(matches!(
        queue.launch_batch(&[
            MetalBatchItem {
                pipeline: first_pipeline.as_ref(),
                bindings: &first_bindings,
                local_size: 8,
                capture_initialized: false,
            },
            MetalBatchItem {
                pipeline: second_pipeline.as_ref(),
                bindings: &empty_bindings,
                local_size: 8,
                capture_initialized: false,
            },
        ]),
        Err(MetalError::InvalidBinding(_))
    ));
    assert!(mock.calls().is_empty());
}

#[test]
fn metal_gpu_timestamp_intervals_reject_unavailable_and_invalid_values() {
    use super::dispatch::gpu_duration_from_seconds;

    assert_eq!(
        gpu_duration_from_seconds(1.25, 1.5),
        Some(std::time::Duration::from_millis(250))
    );
    assert_eq!(gpu_duration_from_seconds(0.0, 1.0), None);
    assert_eq!(gpu_duration_from_seconds(-1.0, 1.0), None);
    assert_eq!(gpu_duration_from_seconds(2.0, 1.0), None);
    assert_eq!(gpu_duration_from_seconds(f64::NAN, 1.0), None);
    assert_eq!(gpu_duration_from_seconds(1.0, f64::INFINITY), None);
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
fn ordered_graph_maximum_renders_i32_and_preserves_lhs_on_ties_and_nan() {
    let renderer = MetalRenderer::new(4, capabilities()).unwrap();
    let mut integer_graph = Graph::new();
    let integer_lhs = integer_graph.input_dtype("lhs", [3], DType::I32);
    let integer_rhs = integer_graph.input_dtype("rhs", [3], DType::I32);
    let integer_maximum = integer_graph.maximum(integer_lhs, integer_rhs).unwrap();
    let integer_item = schedule(&integer_graph, integer_maximum)
        .unwrap()
        .items
        .pop()
        .unwrap();
    let integer_rendered = renderer.render(&integer_item.kernel).unwrap();
    assert!(integer_rendered.source.contains(" < "));
    assert!(integer_rendered.source.contains(" ? "));

    let mut float_graph = Graph::new();
    let float_lhs = float_graph.input_dtype("lhs", [3], DType::F32);
    let float_rhs = float_graph.input_dtype("rhs", [3], DType::F32);
    let float_maximum = float_graph.maximum(float_lhs, float_rhs).unwrap();
    let leading_nan = f32::from_bits(0x7fc0_1234);
    let (actual, _) = execute_mock(
        &float_graph,
        float_maximum,
        &HashMap::from([
            (
                "lhs".into(),
                TensorData::new([3], vec![leading_nan, -0.0, 1.0]).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::new([3], vec![9.0, 0.0, 2.0]).unwrap(),
            ),
        ]),
    );
    assert!(actual.scalar_at(0).as_f64().is_nan());
    assert_eq!(actual.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(actual.scalar_at(2).as_f64(), 2.0);
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
    let maximum = LaneInstruction::GraphBinary {
        output: typed("out", DType::I32),
        lhs: typed("lhs", DType::I32),
        rhs: typed("rhs", DType::I32),
        op: BinaryOp::Maximum,
    };
    let mixed_maximum = LaneInstruction::GraphBinary {
        output: typed("out", DType::F32),
        lhs: typed("lhs", DType::I32),
        rhs: typed("rhs", DType::F32),
        op: BinaryOp::Maximum,
    };
    let minimum = LaneInstruction::GraphBinary {
        output: typed("out", DType::I32),
        lhs: typed("lhs", DType::I32),
        rhs: typed("rhs", DType::I32),
        op: BinaryOp::Minimum,
    };
    let bitwise = emit_scalar_lane(&dialect, &mixed_bitwise).unwrap();
    let add = emit_scalar_lane(&dialect, &mixed_add).unwrap();
    let compare_error = emit_scalar_lane(&dialect, &mixed_compare).unwrap_err();
    let maximum = emit_scalar_lane(&dialect, &maximum).unwrap();
    let mixed_maximum_error = emit_scalar_lane(&dialect, &mixed_maximum).unwrap_err();
    let minimum_error = emit_scalar_lane(&dialect, &minimum).unwrap_err();
    assert!(bitwise.contains("(int)(lhs)") && bitwise.contains(" | "));
    assert!(add.contains("(float)(lhs)") && add.contains(" + "));
    assert!(compare_error.contains("compare dtype"));
    assert_eq!(maximum, "(((lhs) < (rhs)) ? (rhs) : (lhs))");
    assert!(mixed_maximum_error.contains("binary dtype"));
    assert!(minimum_error.contains("Minimum"));

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
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(29)));
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
    let completion = command.collect().unwrap();
    assert_eq!(completion.gpu_command_execution_time, None);
    assert!(
        !mock
            .calls()
            .iter()
            .any(|call| call.starts_with("gpu_time:"))
    );
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
    mock.set_gpu_command_execution_time(Some(std::time::Duration::from_nanos(23)));
    let gpu_queries_before = mock
        .calls()
        .iter()
        .filter(|call| call.starts_with("gpu_time:"))
        .count();
    mock.state.lock().unwrap().failures.wait = Some("gpu fault");
    assert!(matches!(
        command.collect(),
        Err(MetalError::Driver {
            operation: "wait",
            ..
        })
    ));
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|call| call.starts_with("gpu_time:"))
            .count(),
        gpu_queries_before
    );
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
