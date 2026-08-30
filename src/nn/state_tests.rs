use super::state::join;
use super::*;
use crate::{Backend, CpuBackend, Error, Graph, NodeId, Storage, TensorData, save_safetensors};
use std::collections::BTreeMap;

fn f32s(data: &TensorData) -> Vec<f32> {
    match data.storage() {
        Storage::F32(v) => v.clone(),
        _ => panic!("expected f32"),
    }
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
fn linear_is_a_graph_leaf_and_replacement_is_versioned() {
    let mut graph = Graph::new();
    let linear = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
    linear
        .weight
        .replace(TensorData::new([1, 2], vec![2., 3.]).unwrap())
        .unwrap();
    linear
        .bias
        .as_ref()
        .unwrap()
        .replace(TensorData::new([1], vec![1.]).unwrap())
        .unwrap();
    let input = graph.input("x", [2, 2]);
    let output = linear.forward(&mut graph, input).unwrap();
    assert_eq!(
        f32s(&execute(
            &graph,
            output,
            &linear,
            ("x", TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap())
        )),
        vec![9., 19.]
    );
    assert!(
        linear
            .weight
            .replace(TensorData::new([2], vec![1., 2.]).unwrap())
            .is_err()
    );
    assert_eq!(linear.weight.version(), Ok(1));
    let loss = graph
        .reduce(output, crate::ReduceKind::Sum, None, false)
        .unwrap();
    let gradient = graph
        .grad(loss, linear.weight.node(&graph).unwrap())
        .unwrap();
    assert_eq!(
        f32s(&execute(
            &graph,
            gradient,
            &linear,
            ("x", TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap())
        )),
        vec![4., 6.]
    );
}

struct OneParameter(Parameter);
impl Module for OneParameter {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "value"), &self.0, StateKind::Parameter)
    }
}

struct ScalarParameter(Parameter);
impl Module for ScalarParameter {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "value"), &self.0, StateKind::Parameter)
    }
}

#[test]
fn parameter_binding_is_graph_local_versioned_and_captures_values() {
    let parameter = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let module = OneParameter(parameter.clone());

    let mut first = Graph::new();
    let first_node = parameter.bind(&mut first).unwrap();
    assert_eq!(parameter.bind(&mut first).unwrap(), first_node);
    assert_eq!(first.node_count(), 1);
    assert!(matches!(
        first.op(first_node).unwrap(),
        crate::Op::Input { name } if name.ends_with("_v0")
    ));

    let second = Graph::new();
    assert!(matches!(
        parameter.node(&second),
        Err(Error::ParameterGraphMismatch)
    ));
    let mut second = second;
    let second_node = parameter.bind(&mut second).unwrap();
    assert_eq!(parameter.node(&second).unwrap(), second_node);
    assert_ne!(first.id(), second.id());
    assert_eq!(second.node_count(), 1);

    let stale_gradient =
        crate::Gradient::for_parameter(&parameter, TensorData::new([1], vec![1.]).unwrap())
            .unwrap();
    let mut optimizer = crate::Optimizer::sgd(
        vec![("value".into(), parameter.clone())],
        crate::SgdConfig::default(),
    )
    .unwrap();
    optimizer
        .step(&BTreeMap::from([("value".into(), stale_gradient.clone())]))
        .unwrap();
    assert_eq!(parameter.version().unwrap(), 1);
    assert!(matches!(
        parameter.node(&first),
        Err(Error::ParameterGraphMismatch)
    ));

    let new_node = parameter.bind(&mut first).unwrap();
    assert_ne!(new_node, first_node);
    assert_eq!(first.node_count(), 2);
    assert_eq!(parameter.bind(&mut first).unwrap(), new_node);
    assert!(matches!(
        first.op(new_node).unwrap(),
        crate::Op::Input { name } if name.ends_with("_v1")
    ));

    let cpu = CpuBackend;
    let old_bindings = module.input_bindings(&first).unwrap();
    assert_eq!(old_bindings.len(), 2);
    assert_eq!(
        cpu.execute(&first, first_node, &old_bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        2.
    );
    let current = cpu
        .execute(&first, new_node, &old_bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    assert!((current - 1.999).abs() < 1e-6);

    assert!(
        optimizer
            .step(&BTreeMap::from([("value".into(), stale_gradient)]))
            .is_err()
    );
}

#[test]
fn tied_parameter_handles_share_identity_and_one_bound_leaf() {
    let parameter = Parameter::new(TensorData::new([2], vec![1., 2.]).unwrap(), true);
    let tied = parameter.clone();
    assert_eq!(parameter.id(), tied.id());
    let mut graph = Graph::new();
    let left = parameter.bind(&mut graph).unwrap();
    let right = tied.bind(&mut graph).unwrap();
    assert_eq!(left, right);
    assert_eq!(graph.node_count(), 1);
}

struct Tied {
    left: Linear,
    right: Parameter,
    running: Parameter,
}
impl Module for Tied {
    fn visit(&self, p: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        self.left.visit(&join(p, "layers.0"), v);
        v(
            join(p, "layers.1.weight"),
            &self.right,
            StateKind::Parameter,
        );
        v(join(p, "running"), &self.running, StateKind::Buffer)
    }
}

struct Stateless;
impl Module for Stateless {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

struct DuplicateNames {
    first: Parameter,
    replacement: Parameter,
    middle: Parameter,
}
impl Module for DuplicateNames {
    fn visit(&self, _: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v("first".into(), &self.first, StateKind::Parameter);
        v("duplicate".into(), &self.first, StateKind::Parameter);
        v("middle".into(), &self.middle, StateKind::Buffer);
        v("duplicate".into(), &self.replacement, StateKind::Parameter);
    }
}

#[test]
fn live_get_state_dict_preserves_prefix_order_buffers_and_tied_liveness() {
    let mut graph = Graph::new();
    let left = Linear::new(&mut graph, 2, 2, false, 1).unwrap();
    let running = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), false);
    let tied = Tied {
        right: left.weight.clone(),
        left,
        running: running.clone(),
    };
    let state = get_state_dict(&tied, "root.");
    assert_eq!(state.len(), 3);
    assert!(!state.is_empty());
    assert_eq!(
        state.keys().collect::<Vec<_>>(),
        vec![
            "root.layers.0.weight",
            "root.layers.1.weight",
            "root.running"
        ]
    );
    assert_eq!(
        state
            .entries()
            .map(|(name, parameter)| (name, parameter.id()))
            .collect::<Vec<_>>(),
        vec![
            ("root.layers.0.weight", tied.left.weight.id()),
            ("root.layers.1.weight", tied.right.id()),
            ("root.running", running.id()),
        ]
    );
    assert_eq!(
        state.get("root.layers.0.weight").unwrap().id(),
        state.get("root.layers.1.weight").unwrap().id()
    );
    tied.left
        .weight
        .replace(TensorData::new([2, 2], vec![3.; 4]).unwrap())
        .unwrap();
    assert_eq!(
        state.get("root.layers.1.weight").unwrap().value().unwrap(),
        TensorData::new([2, 2], vec![3.; 4]).unwrap()
    );

    let one = OneParameter(Parameter::new(
        TensorData::new([1], vec![1.]).unwrap(),
        true,
    ));
    assert_eq!(
        get_state_dict(&one, "root").keys().collect::<Vec<_>>(),
        vec!["rootvalue"]
    );
    assert_eq!(
        get_state_dict(&one, "root.").keys().collect::<Vec<_>>(),
        vec!["root.value"]
    );
    let first = Parameter::new(TensorData::new([1], vec![1.]).unwrap(), true);
    let second = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let mut nested = Sequential::default();
    nested.push(OneParameter(second));
    let mut sequence = Sequential::default();
    sequence.push(OneParameter(first));
    sequence.push(nested);
    assert_eq!(
        get_state_dict(&sequence, "").keys().collect::<Vec<_>>(),
        vec!["0.value", "1.0.value"]
    );
    assert!(get_state_dict(&Stateless, "root.").is_empty());
}

#[test]
fn live_get_state_dict_keeps_duplicate_key_position_and_refactors_get_parameters() {
    let first = Parameter::new(TensorData::new([1], vec![1.]).unwrap(), true);
    let replacement = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let middle = Parameter::new(TensorData::new([1], vec![3.]).unwrap(), false);
    let duplicate = DuplicateNames {
        first: first.clone(),
        replacement: replacement.clone(),
        middle: middle.clone(),
    };
    let state = get_state_dict(&duplicate, "");
    assert_eq!(
        state.keys().collect::<Vec<_>>(),
        vec!["first", "duplicate", "middle"]
    );
    assert_eq!(state.get("duplicate").unwrap().id(), replacement.id());
    assert_eq!(
        state.values().map(Parameter::id).collect::<Vec<_>>(),
        vec![first.id(), replacement.id(), middle.id()]
    );
    assert_eq!(
        state
            .clone()
            .into_entries()
            .into_iter()
            .map(|(name, parameter)| (name, parameter.id()))
            .collect::<Vec<_>>(),
        vec![
            ("first".into(), first.id()),
            ("duplicate".into(), replacement.id()),
            ("middle".into(), middle.id()),
        ]
    );
    assert_eq!(
        get_parameters(&duplicate)
            .iter()
            .map(Parameter::id)
            .collect::<Vec<_>>(),
        state.values().map(Parameter::id).collect::<Vec<_>>()
    );
}

#[test]
fn live_get_state_dict_never_snapshots_locks_or_mutates_handles() {
    let parameter = Parameter::new(TensorData::new([1], vec![4.]).unwrap(), true);
    let module = OneParameter(parameter.clone());
    let version = parameter.version().unwrap();
    let value = parameter.value().unwrap();
    let state = get_state_dict(&module, "");
    assert_eq!(state.get("value").unwrap().id(), parameter.id());
    assert_eq!(parameter.version().unwrap(), version);
    assert_eq!(parameter.value().unwrap(), value);

    parameter.poison_for_test();
    let poisoned = get_state_dict(&module, "");
    assert_eq!(poisoned.len(), 1);
    assert_eq!(poisoned.get("value").unwrap().id(), parameter.id());
    assert!(matches!(
        parameter.snapshot(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
}

#[test]
fn get_parameters_preserves_declared_order_buffers_and_tied_handles() {
    let mut graph = Graph::new();
    let left = Linear::new(&mut graph, 2, 2, false, 1).unwrap();
    let running = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), false);
    let tied = Tied {
        right: left.weight.clone(),
        left,
        running: running.clone(),
    };
    let handles = get_parameters(&tied);
    assert_eq!(handles.len(), 3);
    assert_eq!(handles[0].id(), tied.left.weight.id());
    assert_eq!(handles[1].id(), tied.right.id());
    assert_eq!(handles[2].id(), running.id());
    assert_eq!(handles[0].id(), handles[1].id());
    assert!(!handles[2].is_trainable());

    let first = Parameter::new(TensorData::new([1], vec![1.]).unwrap(), true);
    let second = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let third = Parameter::new(TensorData::new([1], vec![3.]).unwrap(), true);
    let mut nested = Sequential::default();
    nested.push(OneParameter(second.clone()));
    nested.push(OneParameter(third.clone()));
    let mut sequence = Sequential::default();
    sequence.push(OneParameter(first.clone()));
    sequence.push(nested);
    assert_eq!(
        get_parameters(&sequence)
            .iter()
            .map(Parameter::id)
            .collect::<Vec<_>>(),
        vec![first.id(), second.id(), third.id()]
    );

    assert!(get_parameters(&Stateless).is_empty());
}

#[test]
fn get_parameters_clones_handles_without_snapshotting_locking_or_mutation() {
    let parameter = Parameter::new(TensorData::new([1], vec![4.]).unwrap(), true);
    let module = OneParameter(parameter.clone());
    let version = parameter.version().unwrap();
    let value = parameter.value().unwrap();
    let handles = get_parameters(&module);
    assert_eq!(handles.len(), 1);
    assert_eq!(handles[0].id(), parameter.id());
    assert_eq!(parameter.version().unwrap(), version);
    assert_eq!(parameter.value().unwrap(), value);

    parameter.poison_for_test();
    let poisoned_handles = get_parameters(&module);
    assert_eq!(poisoned_handles.len(), 1);
    assert_eq!(poisoned_handles[0].id(), parameter.id());
    assert!(matches!(
        parameter.snapshot(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
}

#[test]
fn state_is_deterministic_shared_and_safetensors_portable() {
    let mut graph = Graph::new();
    let left = Linear::new(&mut graph, 2, 2, false, 1).unwrap();
    let running = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), false);
    let tied = Tied {
        right: left.weight.clone(),
        left,
        running,
    };
    let state = tied.state_dict().unwrap();
    assert_eq!(
        state.tensors().keys().cloned().collect::<Vec<_>>(),
        vec!["layers.0.weight", "running"]
    );
    let bytes = save_safetensors(&state.clone().into_tensors(), &BTreeMap::new()).unwrap();
    let (raw, _) = crate::load_safetensors(&bytes).unwrap();
    let report = tied
        .load_state_dict(&StateDict::from(raw), true, CastPolicy::Exact)
        .unwrap();
    assert_eq!(report.loaded_keys, vec!["layers.0.weight", "running"]);
    let mut changed = state.clone().into_tensors();
    changed.insert("unexpected".into(), TensorData::scalar(1.));
    let report = tied
        .load_state_dict(&StateDict::from(changed), false, CastPolicy::Exact)
        .unwrap();
    assert_eq!(report.unexpected_keys, vec!["unexpected"]);
}

#[test]
fn trainable_parameters_are_canonical_and_fail_before_optimizer_allocation() {
    let mut graph = Graph::new();
    let left = Linear::new(&mut graph, 2, 2, false, 17).unwrap();
    let tied = Tied {
        right: left.weight.clone(),
        left,
        running: Parameter::new(TensorData::new([1], vec![0.]).unwrap(), false),
    };
    assert_eq!(
        tied.trainable_parameters()
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        vec!["layers.0.weight"]
    );
    let optimizer = crate::Optimizer::sgd_for_module(&tied, crate::SgdConfig::default()).unwrap();
    assert_eq!(optimizer.parameter_names(), vec!["layers.0.weight"]);

    let empty = OneParameter(Parameter::new(
        TensorData::new([1], vec![0.]).unwrap(),
        false,
    ));
    assert!(crate::Optimizer::sgd_for_module(&empty, crate::SgdConfig::default()).is_err());

    struct Duplicate(Parameter, Parameter);
    impl Module for Duplicate {
        fn visit(&self, _: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor("weight".into(), &self.0, StateKind::Parameter);
            visitor("weight".into(), &self.1, StateKind::Parameter);
        }
    }
    let duplicate = Duplicate(
        Parameter::new(TensorData::new([1], vec![0.]).unwrap(), true),
        Parameter::new(TensorData::new([1], vec![0.]).unwrap(), true),
    );
    assert!(matches!(
        duplicate.trainable_parameters(),
        Err(Error::Serialization { .. })
    ));

    let poisoned = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), true);
    poisoned.poison_for_test();
    assert!(matches!(
        OneParameter(poisoned).trainable_parameters(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
}

#[test]
fn strict_state_loading_is_exact_transactional_and_rejects_tied_aliases() {
    let mut construction = Graph::new();
    let linear = Linear::new(&mut construction, 2, 1, true, 7).unwrap();
    let before = linear.state_dict().unwrap();
    let mut cases = Vec::new();

    let mut missing = before.clone().into_tensors();
    missing.remove("weight");
    cases.push(("missing", StateDict::from(missing)));

    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("extra".into(), TensorData::scalar(1.));
    cases.push(("unexpected", StateDict::from(unexpected)));

    let mut bad_shape = before.clone().into_tensors();
    bad_shape.insert("weight".into(), TensorData::new([1], vec![9.]).unwrap());
    cases.push(("shape", StateDict::from(bad_shape)));

    // `bias` sorts before `weight`; this catches the historical partial-update
    // hazard where a valid earlier value was committed before a later mismatch.
    let mut bad_dtype = before.clone().into_tensors();
    bad_dtype.insert("bias".into(), TensorData::new([1], vec![9.]).unwrap());
    bad_dtype.insert(
        "weight".into(),
        TensorData::from_le_bytes([1, 2], crate::DType::I32, &[1, 0, 0, 0, 2, 0, 0, 0]).unwrap(),
    );
    cases.push(("dtype", StateDict::from(bad_dtype)));

    for (name, state) in cases {
        assert!(linear.load_state_dict_strict(&state).is_err(), "{name}");
        assert_eq!(linear.state_dict().unwrap(), before, "{name}");
    }

    let tied_parameter = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    struct Aliased(Parameter, Parameter);
    impl Module for Aliased {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor(join(prefix, "canonical"), &self.0, StateKind::Parameter);
            visitor(join(prefix, "alias"), &self.1, StateKind::Parameter);
        }
    }
    let tied = Aliased(tied_parameter.clone(), tied_parameter);
    let before = tied.state_dict().unwrap();
    assert_eq!(
        before.tensors().keys().cloned().collect::<Vec<_>>(),
        ["canonical"]
    );
    let mut conflicting = before.clone().into_tensors();
    conflicting.insert("alias".into(), TensorData::new([1], vec![3.]).unwrap());
    assert!(
        tied.load_state_dict_strict(&StateDict::from(conflicting))
            .is_err()
    );
    assert_eq!(tied.state_dict().unwrap(), before);
}

#[test]
fn strict_loading_lock_failure_leaves_other_parameters_unchanged() {
    struct Pair(Parameter, Parameter);
    impl Module for Pair {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            visitor(join(prefix, "first"), &self.0, StateKind::Parameter);
            visitor(join(prefix, "second"), &self.1, StateKind::Parameter);
        }
    }
    let first = Parameter::new(TensorData::new([1], vec![1.]).unwrap(), true);
    let second = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let module = Pair(first.clone(), second.clone());
    let first_before = first.value().unwrap();
    second.poison_for_test();
    let state = StateDict::from(BTreeMap::from([
        ("first".into(), TensorData::new([1], vec![9.]).unwrap()),
        ("second".into(), TensorData::new([1], vec![8.]).unwrap()),
    ]));
    assert!(matches!(
        module.load_state_dict_strict(&state),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert_eq!(first.value().unwrap(), first_before);
}

#[test]
fn strict_state_load_preflights_every_container_parameter_before_replacement() {
    let mut graph = Graph::new();
    let linear = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
    let weight_before = linear.weight.snapshot().unwrap();
    let bias = linear.bias.as_ref().unwrap();
    let bias_before = bias.snapshot().unwrap();

    let mut malformed = linear.state_dict().unwrap().into_tensors();
    malformed.insert("bias".into(), TensorData::new([1], vec![5.]).unwrap());
    malformed.insert("weight".into(), TensorData::new([1], vec![7.]).unwrap());
    assert!(
        linear
            .load_state_dict(&StateDict::from(malformed), true, CastPolicy::Exact)
            .is_err()
    );
    let weight_after_failure = linear.weight.snapshot().unwrap();
    let bias_after_failure = bias.snapshot().unwrap();
    assert_eq!(weight_after_failure.data, weight_before.data);
    assert_eq!(weight_after_failure.version, weight_before.version);
    assert_eq!(bias_after_failure.data, bias_before.data);
    assert_eq!(bias_after_failure.version, bias_before.version);

    let mut valid = linear.state_dict().unwrap().into_tensors();
    valid.insert("bias".into(), TensorData::new([1], vec![5.]).unwrap());
    valid.insert(
        "weight".into(),
        TensorData::new([1, 2], vec![7., 8.]).unwrap(),
    );
    let report = linear
        .load_state_dict(&StateDict::from(valid), true, CastPolicy::Exact)
        .unwrap();
    assert!(report.is_clean());
    let weight_after_success = linear.weight.snapshot().unwrap();
    let bias_after_success = bias.snapshot().unwrap();
    assert_eq!(f32s(&weight_after_success.data), vec![7., 8.]);
    assert_eq!(f32s(&bias_after_success.data), vec![5.]);
    assert_eq!(
        weight_after_success.version,
        weight_before.version.checked_add(1).unwrap()
    );
    assert_eq!(
        bias_after_success.version,
        bias_before.version.checked_add(1).unwrap()
    );
}

#[test]
fn state_load_admits_only_the_tinygrad_scalar_singleton_shape_bridge() {
    let vector_parameter = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), true);
    let vector = OneParameter(vector_parameter.clone());
    let report = vector
        .load_state_dict(
            &StateDict::from(BTreeMap::from([("value".into(), TensorData::scalar(-0.0))])),
            true,
            CastPolicy::Exact,
        )
        .unwrap();
    assert!(report.is_clean());
    let vector_snapshot = vector_parameter.snapshot().unwrap();
    assert_eq!(vector_snapshot.shape.dims(), [1]);
    assert_eq!(
        f32s(&vector_snapshot.data)[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(vector_snapshot.version, 1);

    let scalar_parameter = Parameter::new(TensorData::scalar(0.0), true);
    let scalar = ScalarParameter(scalar_parameter.clone());
    let report = scalar
        .load_state_dict(
            &StateDict::from(BTreeMap::from([(
                "value".into(),
                TensorData::new([1], vec![3.5]).unwrap(),
            )])),
            true,
            CastPolicy::Exact,
        )
        .unwrap();
    assert!(report.is_clean());
    let scalar_snapshot = scalar_parameter.snapshot().unwrap();
    assert!(scalar_snapshot.shape.dims().is_empty());
    assert_eq!(f32s(&scalar_snapshot.data), vec![3.5]);
    assert_eq!(scalar_snapshot.version, 1);
}

#[test]
fn parameter_version_overflow_is_preflighted_without_any_publication() {
    let parameter = Parameter::new(TensorData::new([1], vec![2.]).unwrap(), true);
    let tied = parameter.clone();
    parameter.set_version_for_test(u64::MAX).unwrap();
    let before = parameter.snapshot().unwrap();
    assert!(matches!(
        parameter.replace(TensorData::new([1], vec![3.]).unwrap()),
        Err(Error::ParameterVersionOverflow { version: u64::MAX })
    ));
    assert_eq!(parameter.snapshot().unwrap().data, before.data);
    assert_eq!(parameter.snapshot().unwrap().version, before.version);
    assert_eq!(tied.snapshot().unwrap().data, before.data);
    assert_eq!(tied.snapshot().unwrap().version, before.version);

    let mut graph = Graph::new();
    let linear = Linear::new(&mut graph, 2, 1, true, 7).unwrap();
    let bias = linear.bias.as_ref().unwrap();
    linear.weight.set_version_for_test(u64::MAX).unwrap();
    let weight_before = linear.weight.snapshot().unwrap();
    let bias_before = bias.snapshot().unwrap();
    let mut state = linear.state_dict().unwrap().into_tensors();
    state.insert("bias".into(), TensorData::new([1], vec![5.]).unwrap());
    state.insert(
        "weight".into(),
        TensorData::new([1, 2], vec![7., 8.]).unwrap(),
    );
    assert!(matches!(
        linear.load_state_dict(&StateDict::from(state), true, CastPolicy::Exact),
        Err(Error::ParameterVersionOverflow { version: u64::MAX })
    ));
    assert_eq!(linear.weight.snapshot().unwrap().data, weight_before.data);
    assert_eq!(
        linear.weight.snapshot().unwrap().version,
        weight_before.version
    );
    assert_eq!(bias.snapshot().unwrap().data, bias_before.data);
    assert_eq!(bias.snapshot().unwrap().version, bias_before.version);
}

#[test]
fn parameters_are_send_sync_and_snapshots_are_concurrent() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Parameter>();
    assert_send_sync::<Linear>();
    assert_send_sync::<Conv1d>();
    assert_send_sync::<Conv2d>();

    let mut graph = Graph::new();
    let linear = std::sync::Arc::new(Linear::new(&mut graph, 2, 2, false, 3).unwrap());
    let mut workers = Vec::new();
    for _ in 0..4 {
        let linear = linear.clone();
        workers.push(std::thread::spawn(move || {
            let graph = Graph::new();
            for _ in 0..32 {
                assert_eq!(linear.state_dict().unwrap().tensors().len(), 1);
                // No forward was built in this graph, so there are no captured leaves.
                assert_eq!(linear.input_bindings(&graph).unwrap().len(), 0);
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn conflicting_snapshot_writes_report_a_version_conflict() {
    let parameter = Parameter::new(TensorData::new([1], vec![0.]).unwrap(), true);
    let first = parameter.snapshot().unwrap();
    parameter
        .replace_expected(TensorData::new([1], vec![1.]).unwrap(), Some(first.version))
        .unwrap();
    assert!(matches!(
        parameter.replace_expected(TensorData::new([1], vec![2.]).unwrap(), Some(first.version)),
        Err(Error::ParameterVersionConflict { .. })
    ));
}

#[test]
fn poisoned_parameter_returns_errors_without_panicking() {
    let mut graph = Graph::new();
    let linear = Linear::new(&mut graph, 1, 1, false, 1).unwrap();
    linear.weight.poison_for_test();
    assert!(matches!(
        linear.weight.snapshot(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.weight.shape(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.weight.dtype(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.weight.value(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.weight.version(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear
            .weight
            .replace(TensorData::new([1, 1], vec![1.]).unwrap()),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.state_dict(),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.input_bindings(&graph),
        Err(Error::ParameterLockPoisoned { .. })
    ));
    assert!(matches!(
        linear.load_state_dict(&StateDict::default(), false, CastPolicy::Exact),
        Err(Error::ParameterLockPoisoned { .. })
    ));
}
