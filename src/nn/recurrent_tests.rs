use super::*;
use crate::{Backend, CpuBackend, Graph, NodeId, TensorData, save_safetensors};
use std::collections::BTreeMap;

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
