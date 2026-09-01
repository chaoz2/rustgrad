use super::*;
use crate::{
    Backend, CapturedReplayExecutor, CapturedReplayOptions, CpuBackend, DType, Graph, NodeId,
    Storage, TensorData, save_safetensors,
};
use std::collections::BTreeMap;

fn f32s(value: &TensorData) -> Vec<f32> {
    match value.storage() {
        Storage::F32(values) => values.clone(),
        other => panic!("expected F32 storage, got {other:?}"),
    }
}

fn fixture_lstm(layers: usize, dropout: f64) -> LSTM {
    let lstm = LSTM::new_static(1, 1, layers, dropout, 41).unwrap();
    for layer in 0..layers {
        let cell = lstm.cell(layer).unwrap();
        cell.weight_ih
            .replace(TensorData::new([4, 1], vec![0.2, -0.1, 0.3, 0.4]).unwrap())
            .unwrap();
        cell.weight_hh
            .replace(TensorData::new([4, 1], vec![0.1, 0.2, -0.2, 0.3]).unwrap())
            .unwrap();
        for bias in [&cell.bias_ih, &cell.bias_hh] {
            bias.as_ref()
                .unwrap()
                .replace(TensorData::new([4], vec![0.0; 4]).unwrap())
                .unwrap();
        }
    }
    lstm
}

fn execute(
    graph: &Graph,
    output: NodeId,
    module: &impl Module,
    input: (&str, TensorData),
) -> TensorData {
    let mut bindings = module.input_bindings(graph).unwrap();
    bindings.insert(input.0.into(), input.1);
    CpuBackend.execute(graph, output, &bindings).unwrap()
}

#[test]
fn lstm_cell_fixture_zero_state_and_traversal() {
    let mut g = Graph::new();
    let cell = LSTMCell::new(&mut g, 1, 1, true, 1).unwrap();
    cell.weight_ih
        .replace(TensorData::new([4, 1], vec![0., 0., 1., 0.]).unwrap())
        .unwrap();
    cell.weight_hh
        .replace(TensorData::new([4, 1], vec![0.; 4]).unwrap())
        .unwrap();
    for b in [&cell.bias_ih, &cell.bias_hh] {
        b.as_ref()
            .unwrap()
            .replace(TensorData::new([4], vec![0.; 4]).unwrap())
            .unwrap();
    }
    let x = g.input("x", [1, 1]);
    let (h, c) = cell.forward(&mut g, x, None).unwrap();
    let input = TensorData::new([1, 1], vec![1.]).unwrap();
    let hv = execute(&g, h, &cell, ("x", input.clone()))
        .scalar_at(0)
        .as_f64();
    let cv = execute(&g, c, &cell, ("x", input)).scalar_at(0).as_f64();
    let expected_c = 0.5 * 1f64.tanh();
    assert!((cv - expected_c).abs() < 1e-6 && (hv - (0.5 * expected_c.tanh())).abs() < 1e-6);
    assert_eq!(
        cell.state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["bias_hh", "bias_ih", "weight_hh", "weight_ih"]
    );
    let bad = g.input("bad", [1, 2]);
    assert!(cell.forward(&mut g, bad, None).is_err());
}

#[test]
fn lstm_cell_threads_state_and_omits_disabled_biases() {
    let mut g = Graph::new();
    let cell = LSTMCell::new(&mut g, 1, 1, false, 3).unwrap();
    cell.weight_ih
        .replace(TensorData::new([4, 1], vec![0.2, -0.1, 0.3, 0.4]).unwrap())
        .unwrap();
    cell.weight_hh
        .replace(TensorData::new([4, 1], vec![0.1, 0.2, -0.2, 0.3]).unwrap())
        .unwrap();
    assert!(cell.bias_ih.is_none() && cell.bias_hh.is_none());
    assert_eq!(
        cell.state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["weight_hh", "weight_ih"]
    );
    let x1 = g.input("x1", [1, 1]);
    let (h1, c1) = cell.forward(&mut g, x1, None).unwrap();
    let x2 = g.input("x2", [1, 1]);
    let (h2, c2) = cell.forward(&mut g, x2, Some((h1, c1))).unwrap();
    let binds = cell
        .input_bindings(&g)
        .unwrap()
        .into_iter()
        .chain([
            (
                String::from("x1"),
                TensorData::new([1, 1], vec![0.5]).unwrap(),
            ),
            (
                String::from("x2"),
                TensorData::new([1, 1], vec![-0.25]).unwrap(),
            ),
        ])
        .collect();
    let h = CpuBackend
        .execute(&g, h2, &binds)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let c = CpuBackend
        .execute(&g, c2, &binds)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let step = |x: f64, h: f64, c: f64| {
        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        let i = sigmoid(0.2 * x + 0.1 * h);
        let f = sigmoid(-0.1 * x + 0.2 * h);
        let z = (0.3 * x - 0.2 * h).tanh();
        let o = sigmoid(0.4 * x + 0.3 * h);
        let nc = f * c + i * z;
        (o * nc.tanh(), nc)
    };
    let (eh1, ec1) = step(0.5, 0., 0.);
    let (eh2, ec2) = step(-0.25, eh1, ec1);
    assert!((h - eh2).abs() < 1e-6 && (c - ec2).abs() < 1e-6);
}

#[test]
fn lstm_cell_static_constructor_preserves_legacy_state_and_forward_contract() {
    let mut legacy_graph = Graph::new();
    let legacy = LSTMCell::new(&mut legacy_graph, 2, 3, true, 19).unwrap();
    let fresh = LSTMCell::new_static(2, 3, true, 19).unwrap();
    assert_eq!(legacy.state_dict().unwrap(), fresh.state_dict().unwrap());
    assert_ne!(
        legacy
            .trainable_parameters()
            .unwrap()
            .into_iter()
            .map(|(_, parameter)| parameter.id())
            .collect::<Vec<_>>(),
        fresh
            .trainable_parameters()
            .unwrap()
            .into_iter()
            .map(|(_, parameter)| parameter.id())
            .collect::<Vec<_>>()
    );

    let source = fresh.state_dict().unwrap();
    let restored = LSTMCell::new_static(2, 3, true, 23).unwrap();
    restored.load_state_dict_strict(&source).unwrap();
    assert_eq!(restored.state_dict().unwrap(), source);

    let mut graph = Graph::new();
    let input = graph.input("input", [1, 2]);
    let (output, state) = restored.forward(&mut graph, input, None).unwrap();
    let input_value = TensorData::new([1, 2], vec![0.25, -0.5]).unwrap();
    let first = execute(&graph, output, &restored, ("input", input_value.clone()));
    let second = execute(&graph, state, &restored, ("input", input_value));
    assert_eq!(first.shape().dims(), &[1, 3]);
    assert_eq!(second.shape().dims(), &[1, 3]);
    assert!(LSTMCell::new_static(0, 1, true, 1).is_err());
    assert!(LSTMCell::new_static(1, 0, true, 1).is_err());

    let before = restored.state_dict().unwrap();
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("unexpected".into(), TensorData::new([1], vec![1.]).unwrap());
    assert!(
        restored
            .load_state_dict_strict(&StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(restored.state_dict().unwrap(), before);
}

#[test]
fn leaf_modules_round_trip_state_through_safetensors() {
    let mut g = Graph::new();
    let ln = LayerNorm2d::new(&mut g, 2, 1e-5, true).unwrap();
    let cell = LSTMCell::new(&mut g, 1, 1, true, 9).unwrap();
    let ln_state = ln.state_dict().unwrap();
    let cell_state = cell.state_dict().unwrap();
    for (module, state) in [
        (&ln as &dyn Module, ln_state),
        (&cell as &dyn Module, cell_state),
    ] {
        let bytes = save_safetensors(&state.clone().into_tensors(), &BTreeMap::new()).unwrap();
        let (raw, _) = crate::load_safetensors(&bytes).unwrap();
        assert!(
            module
                .load_state_dict(&StateDict::from(raw), true, CastPolicy::Exact)
                .unwrap()
                .is_clean()
        );
    }
}

#[test]
fn lstm_cell_input_and_weight_gradients_match_central_differences() {
    fn loss(input: f32, weight: f32) -> f64 {
        let mut g = Graph::new();
        let cell = LSTMCell::new(&mut g, 1, 1, false, 1).unwrap();
        cell.weight_ih
            .replace(TensorData::new([4, 1], vec![weight, -0.2, 0.3, 0.1]).unwrap())
            .unwrap();
        cell.weight_hh
            .replace(TensorData::new([4, 1], vec![0.1, -0.1, 0.2, 0.05]).unwrap())
            .unwrap();
        let x = g.input("x", [1, 1]);
        let (h, c) = cell.forward(&mut g, x, None).unwrap();
        let y = g.add(h, c).unwrap();
        execute(
            &g,
            y,
            &cell,
            ("x", TensorData::new([1, 1], vec![input]).unwrap()),
        )
        .scalar_at(0)
        .as_f64()
    }
    let input = 0.25f32;
    let weight = 0.15f32;
    let mut g = Graph::new();
    let cell = LSTMCell::new(&mut g, 1, 1, false, 1).unwrap();
    cell.weight_ih
        .replace(TensorData::new([4, 1], vec![weight, -0.2, 0.3, 0.1]).unwrap())
        .unwrap();
    cell.weight_hh
        .replace(TensorData::new([4, 1], vec![0.1, -0.1, 0.2, 0.05]).unwrap())
        .unwrap();
    let x = g.input("x", [1, 1]);
    let (h, c) = cell.forward(&mut g, x, None).unwrap();
    let y = g.add(h, c).unwrap();
    let loss_node = g.reduce(y, crate::ReduceKind::Sum, None, false).unwrap();
    let dx = g.grad(loss_node, x).unwrap();
    let dw = g.grad(loss_node, cell.weight_ih.node(&g).unwrap()).unwrap();
    let bindings = cell
        .input_bindings(&g)
        .unwrap()
        .into_iter()
        .chain([(
            String::from("x"),
            TensorData::new([1, 1], vec![input]).unwrap(),
        )])
        .collect();
    let analytic_x = CpuBackend
        .execute(&g, dx, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let analytic_w = CpuBackend
        .execute(&g, dw, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let eps = 1e-3f32;
    let numeric_x = (loss(input + eps, weight) - loss(input - eps, weight)) / (2. * eps as f64);
    let numeric_w = (loss(input, weight + eps) - loss(input, weight - eps)) / (2. * eps as f64);
    assert!(
        (analytic_x - numeric_x).abs() < 2e-3,
        "input analytic={analytic_x} numeric={numeric_x}"
    );
    assert!(
        (analytic_w - numeric_w).abs() < 2e-3,
        "weight analytic={analytic_w} numeric={numeric_w}"
    );
}

#[test]
fn stacked_lstm_has_typed_shapes_state_and_cell_traversal() {
    for layers in [1, 2] {
        let mut graph = Graph::new();
        let lstm = fixture_lstm(layers, 0.0);
        assert_eq!(lstm.layers(), layers);
        assert_eq!(lstm.input_size(), 1);
        assert_eq!(lstm.hidden_size(), 1);
        assert_eq!(lstm.dropout(), 0.0);
        assert_eq!(lstm.seed(), 41);
        assert!(lstm.cell(layers).is_none());
        let input = graph.input_dtype("x", [3, 2, 1], DType::F32);
        let output = lstm.forward(&mut graph, input, None, Mode::Eval).unwrap();
        assert_eq!(graph.shape(output.sequence()).unwrap().dims(), &[3, 2, 1]);
        assert_eq!(
            graph.shape(output.state().hidden()).unwrap().dims(),
            &[layers, 2, 1]
        );
        assert_eq!(
            graph.shape(output.state().cell()).unwrap().dims(),
            &[layers, 2, 1]
        );
        let names = lstm
            .state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let expected = (0..layers)
            .flat_map(|layer| {
                ["bias_hh", "bias_ih", "weight_hh", "weight_ih"]
                    .map(move |name| format!("cells.{layer}.{name}"))
            })
            .collect::<Vec<_>>();
        assert_eq!(names, expected);

        let mut bindings = lstm.input_bindings(&graph).unwrap();
        bindings.insert(
            "x".into(),
            TensorData::new([3, 2, 1], vec![0.25, -0.5, 0.1, 0.2, -0.3, 0.4]).unwrap(),
        );
        let realized = CpuBackend
            .execute(&graph, output.sequence(), &bindings)
            .unwrap();
        assert_eq!(realized.shape().dims(), &[3, 2, 1]);
        assert!(f32s(&realized).iter().all(|value| value.is_finite()));
    }
}

#[test]
fn stacked_lstm_zero_state_matches_explicit_and_carried_chunks_match_full_sequence() {
    let mut graph = Graph::new();
    let lstm = fixture_lstm(2, 0.0);
    let full_input = graph.input_dtype("full", [2, 1, 1], DType::F32);
    let full = lstm
        .forward(&mut graph, full_input, None, Mode::Eval)
        .unwrap();

    let first_input = graph.input_dtype("first", [1, 1, 1], DType::F32);
    let zeros = LSTMState::new(
        graph.zeros_with_dtype([2, 1, 1], DType::F32).unwrap(),
        graph.zeros_with_dtype([2, 1, 1], DType::F32).unwrap(),
    );
    let first = lstm
        .forward(&mut graph, first_input, Some(zeros), Mode::Eval)
        .unwrap();
    let second_input = graph.input_dtype("second", [1, 1, 1], DType::F32);
    let second = lstm
        .forward(&mut graph, second_input, Some(first.state()), Mode::Eval)
        .unwrap();

    let mut bindings = lstm.input_bindings(&graph).unwrap();
    bindings.insert(
        "full".into(),
        TensorData::new([2, 1, 1], vec![0.25, -0.5]).unwrap(),
    );
    bindings.insert(
        "first".into(),
        TensorData::new([1, 1, 1], vec![0.25]).unwrap(),
    );
    bindings.insert(
        "second".into(),
        TensorData::new([1, 1, 1], vec![-0.5]).unwrap(),
    );
    let full_values = f32s(
        &CpuBackend
            .execute(&graph, full.sequence(), &bindings)
            .unwrap(),
    );
    let chunk_values = [first.sequence(), second.sequence()]
        .into_iter()
        .flat_map(|node| f32s(&CpuBackend.execute(&graph, node, &bindings).unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(full_values, chunk_values);
    for (full_state, chunk_state) in [
        (full.state().hidden(), second.state().hidden()),
        (full.state().cell(), second.state().cell()),
    ] {
        assert_eq!(
            f32s(&CpuBackend.execute(&graph, full_state, &bindings).unwrap()),
            f32s(&CpuBackend.execute(&graph, chunk_state, &bindings).unwrap())
        );
    }
}

#[test]
fn stacked_lstm_is_atomic_for_malformed_descriptors_and_supports_empty_batch() {
    assert!(LSTM::new_static(0, 1, 1, 0.0, 1).is_err());
    assert!(LSTM::new_static(1, 0, 1, 0.0, 1).is_err());
    assert!(LSTM::new_static(1, 1, 0, 0.0, 1).is_err());
    assert!(LSTM::new_static(1, 1, 1, -0.1, 1).is_err());
    assert!(LSTM::new_static(1, 1, 1, 1.1, 1).is_err());

    let lstm = fixture_lstm(2, 0.0);
    let mut graph = Graph::new();
    for bad in [
        graph.input_dtype("rank", [2, 1], DType::F32),
        graph.input_dtype("feature", [2, 1, 2], DType::F32),
        graph.input_dtype("dtype", [2, 1, 1], DType::I32),
        graph.input_dtype("time", [0, 1, 1], DType::F32),
    ] {
        let before = graph.node_count();
        assert!(lstm.forward(&mut graph, bad, None, Mode::Eval).is_err());
        assert_eq!(graph.node_count(), before);
    }
    let input = graph.input_dtype("state_input", [2, 1, 1], DType::F32);
    let bad_state = LSTMState::new(
        graph.zeros_with_dtype([1, 1, 1], DType::F32).unwrap(),
        graph.zeros_with_dtype([2, 1, 1], DType::F32).unwrap(),
    );
    let before = graph.node_count();
    assert!(
        lstm.forward(&mut graph, input, Some(bad_state), Mode::Eval)
            .is_err()
    );
    assert_eq!(graph.node_count(), before);

    let mut empty = Graph::new();
    let input = empty.input_dtype("empty", [2, 0, 1], DType::F32);
    let output = lstm
        .forward(&mut empty, input, None, Mode::Training)
        .unwrap();
    let mut bindings = lstm.input_bindings(&empty).unwrap();
    bindings.insert(
        "empty".into(),
        TensorData::from_storage([2, 0, 1], Storage::F32(Vec::new())).unwrap(),
    );
    let value = CpuBackend
        .execute(&empty, output.sequence(), &bindings)
        .unwrap();
    assert_eq!(value.shape().dims(), &[2, 0, 1]);
    assert!(f32s(&value).is_empty());

    let broken = fixture_lstm(1, 0.0);
    broken.cell(0).unwrap().weight_ih.poison_for_test();
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [1, 1, 1], DType::F32);
    let before = graph.node_count();
    assert!(broken.forward(&mut graph, input, None, Mode::Eval).is_err());
    assert_eq!(graph.node_count(), before);
}

#[test]
fn stacked_lstm_dropout_gradients_and_captured_interpreter_are_compositional() {
    let lstm = fixture_lstm(1, 1.0);
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("x", [2, 1, 1], DType::F32, true);
    let training = lstm
        .forward(&mut graph, input, None, Mode::Training)
        .unwrap();
    let evaluation = lstm.forward(&mut graph, input, None, Mode::Eval).unwrap();
    let loss = graph
        .reduce(evaluation.sequence(), crate::ReduceKind::Sum, None, false)
        .unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let weight = lstm.cell(0).unwrap().weight_ih.node(&graph).unwrap();
    let weight_gradient = graph.grad(loss, weight).unwrap();

    let mut bindings = lstm.input_bindings(&graph).unwrap();
    bindings.insert(
        "x".into(),
        TensorData::new([2, 1, 1], vec![0.25, -0.5]).unwrap(),
    );
    assert_eq!(
        f32s(
            &CpuBackend
                .execute(&graph, training.sequence(), &bindings)
                .unwrap()
        ),
        vec![0.0, 0.0]
    );
    assert!(
        f32s(
            &CpuBackend
                .execute(&graph, evaluation.sequence(), &bindings)
                .unwrap()
        )
        .iter()
        .any(|value| *value != 0.0)
    );
    assert!(
        f32s(
            &CpuBackend
                .execute(&graph, input_gradient, &bindings)
                .unwrap()
        )
        .iter()
        .all(|value| value.is_finite())
    );
    assert!(
        f32s(
            &CpuBackend
                .execute(&graph, weight_gradient, &bindings)
                .unwrap()
        )
        .iter()
        .all(|value| value.is_finite())
    );

    let requested = [
        evaluation.sequence(),
        evaluation.state().hidden(),
        evaluation.state().cell(),
    ];
    let schedule = crate::schedule_many(&graph, &requested).unwrap();
    let capture = crate::CapturedSchedule::capture(&graph, &schedule, &requested).unwrap();
    let provided = bindings.clone().into_iter().collect::<BTreeMap<_, _>>();
    let replay = CapturedReplayExecutor::default()
        .replay(&capture, &provided, CapturedReplayOptions::default())
        .unwrap();
    for (node, replayed) in requested.into_iter().zip(replay.outputs) {
        assert_eq!(
            replayed.storage(),
            CpuBackend
                .execute(&graph, node, &bindings)
                .unwrap()
                .storage()
        );
    }
}
