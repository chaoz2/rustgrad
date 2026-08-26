use super::*;
use crate::{
    Backend, CpuBackend, DType, Error, Graph, NodeId, Result, Shape, Storage, TensorData,
    infer_module_cpu,
};
use std::collections::HashMap;

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
fn batchnorm_training_commit_and_eval_match_tinygrad_statistics() {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 2, 1e-5, true, true, 0.1).unwrap();
    let input = graph.input("x", [2, 2]);
    let result = norm.forward(&mut graph, input, Mode::Training).unwrap();
    let token = result.pending.expect("training token");
    let mut bindings = norm.input_bindings(&graph).unwrap();
    bindings.insert(
        "x".into(),
        TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let mean = CpuBackend.execute(&graph, token.mean, &bindings).unwrap();
    let variance = CpuBackend
        .execute(&graph, token.variance, &bindings)
        .unwrap();
    token.commit_stats(&norm, mean, variance).unwrap();
    assert_eq!(
        f32s(&norm.running_mean.as_ref().unwrap().value().unwrap()),
        vec![0.2, 0.3]
    );
    assert_eq!(
        f32s(&norm.running_var.as_ref().unwrap().value().unwrap()),
        vec![1.1, 1.1]
    );
    assert_eq!(
        norm.num_batches_tracked
            .value()
            .unwrap()
            .scalar_at(0)
            .as_u64(),
        1
    );
    assert!(matches!(
        token.commit_stats(
            &norm,
            TensorData::new([2], vec![2., 3.]).unwrap(),
            TensorData::new([2], vec![1., 1.]).unwrap()
        ),
        Err(Error::BatchNormToken { .. })
    ));

    let x = graph.input("eval_x", [1, 2]);
    let eval = norm.forward(&mut graph, x, Mode::Eval).unwrap();
    assert!(eval.pending.is_none());
    let mut bindings = norm.input_bindings(&graph).unwrap();
    bindings.insert(
        "x".into(),
        TensorData::new([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    bindings.insert(
        "eval_x".into(),
        TensorData::new([1, 2], vec![0.2, 0.3]).unwrap(),
    );
    let output = CpuBackend.execute(&graph, eval.output, &bindings).unwrap();
    assert!(f32s(&output).iter().all(|x| x.abs() < 1e-5));
}

#[test]
fn normalization_modules_have_group_and_instance_fixtures() {
    let mut graph = Graph::new();
    let group = GroupNorm::new(&mut graph, 2, 4, 1e-5, false).unwrap();
    let input = graph.input("x", [1, 4, 1]);
    let output = group.forward(&mut graph, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([1, 4, 1], vec![1., 3., 10., 14.]).unwrap(),
    )]);
    let output = CpuBackend.execute(&graph, output, &bindings).unwrap();
    let values = f32s(&output);
    assert!((values[0] + 1.).abs() < 1e-4 && (values[1] - 1.).abs() < 1e-4);
    assert!((values[2] + 1.).abs() < 1e-4 && (values[3] - 1.).abs() < 1e-4);
    assert!(GroupNorm::new(&mut graph, 3, 4, 1e-5, true).is_err());
    let instance = InstanceNorm::new(&mut graph, 2, 1e-5, false).unwrap();
    let x = graph.input("i", [1, 2, 2]);
    let output = instance.forward(&mut graph, x).unwrap();
    let mut bindings = HashMap::from([(
        "i".into(),
        TensorData::new([1, 2, 2], vec![1., 3., 10., 14.]).unwrap(),
    )]);
    bindings.insert(
        "x".into(),
        TensorData::new([1, 4, 1], vec![1., 3., 10., 14.]).unwrap(),
    );
    let output = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(f32s(&output).len(), 4);
}

#[test]
fn batchnorm_tokens_are_send_sync_and_reject_wrong_modules() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BatchNorm>();
    assert_send_sync::<PendingBatchNormStats>();
    let mut graph = Graph::new();
    let left = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1).unwrap();
    let right = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1).unwrap();
    let input = graph.input("x", [2, 1]);
    let result = left.forward(&mut graph, input, Mode::Training).unwrap();
    let token = result.pending.unwrap();
    assert!(matches!(
        token.commit_stats(
            &right,
            TensorData::new([1], vec![1.]).unwrap(),
            TensorData::new([1], vec![1.]).unwrap()
        ),
        Err(Error::BatchNormToken { .. })
    ));
    let mut bindings = left.input_bindings(&graph).unwrap();
    bindings.extend(right.input_bindings(&graph).unwrap());
    bindings.insert("x".into(), TensorData::new([2, 1], vec![1., 3.]).unwrap());
    let mean = CpuBackend.execute(&graph, token.mean, &bindings).unwrap();
    let variance = CpuBackend
        .execute(&graph, token.variance, &bindings)
        .unwrap();
    token.commit_stats(&left, mean, variance).unwrap();
}

#[test]
fn groupnorm_affine_and_input_gradients_are_finite() {
    let mut graph = Graph::new();
    let norm = GroupNorm::new(&mut graph, 1, 2, 1e-5, true).unwrap();
    let input = graph.input("x", [1, 2, 2]);
    let output = norm.forward(&mut graph, input).unwrap();
    let loss = graph
        .reduce(output, crate::ReduceKind::Sum, None, false)
        .unwrap();
    let input_grad = graph.grad(loss, input).unwrap();
    let weight_grad = graph
        .grad(loss, norm.weight.as_ref().unwrap().node(&graph).unwrap())
        .unwrap();
    let mut bindings = norm.input_bindings(&graph).unwrap();
    bindings.insert(
        "x".into(),
        TensorData::new([1, 2, 2], vec![1., 2., 4., 8.]).unwrap(),
    );
    let input_grad = CpuBackend.execute(&graph, input_grad, &bindings).unwrap();
    let weight_grad = CpuBackend.execute(&graph, weight_grad, &bindings).unwrap();
    assert!(f32s(&input_grad).iter().all(|x| x.is_finite()));
    assert!(f32s(&weight_grad).iter().all(|x| x.is_finite()));
}

#[test]
fn layernorm2d_matches_channelwise_fixture_and_state() {
    let mut g = Graph::new();
    let norm = LayerNorm2d::new(&mut g, 2, 0.0, true).unwrap();
    norm.inner
        .weight
        .as_ref()
        .unwrap()
        .replace(TensorData::new([2], vec![2., 3.]).unwrap())
        .unwrap();
    norm.inner
        .bias
        .as_ref()
        .unwrap()
        .replace(TensorData::new([2], vec![1., -1.]).unwrap())
        .unwrap();
    let x = g.input("x", [1, 2, 1, 2]);
    let y = norm.forward(&mut g, x).unwrap();
    let out = execute(
        &g,
        y,
        &norm,
        (
            "x",
            TensorData::new([1, 2, 1, 2], vec![1., 3., 5., 7.]).unwrap(),
        ),
    );
    assert_eq!(out.shape().dims(), &[1, 2, 1, 2]);
    assert_eq!(f32s(&out), vec![-1., -1., 2., 2.]);
    assert_eq!(
        norm.state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["bias", "weight"]
    );
    let bad = g.input("bad", [1, 2, 2]);
    assert!(norm.forward(&mut g, bad).is_err());
}

fn layernorm_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let linear = Linear::new_static(2, 2, false, seed)?;
    let norm = LayerNorm::new_static(Shape::new([2]), 0.0, true)?;
    if fixed {
        linear
            .weight
            .replace(TensorData::new([2, 2], vec![1., 0., 0., 1.])?)?;
        norm.weight
            .as_ref()
            .expect("configured affine weight")
            .replace(TensorData::new([2], vec![2., 3.])?)?;
        norm.bias
            .as_ref()
            .expect("configured affine bias")
            .replace(TensorData::new([2], vec![1., -1.])?)?;
    }
    let mut model = Sequential::default();
    model.push(linear);
    model.push(norm);
    Ok(model)
}

#[test]
fn layernorm_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = LayerNorm::new(&mut legacy_graph, Shape::new([2]), 1e-5, true)?;
    let graph_free = LayerNorm::new_static(Shape::new([2]), 1e-5, true)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);
    assert!(LayerNorm::new_static(Shape::new([]), 1e-5, true).is_err());

    let source = layernorm_classifier(101, true)?;
    let target = layernorm_classifier(103, false)?;
    let source_state = source.state_dict()?;
    let source_parameters = source
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| (name, parameter.id()))
        .collect::<Vec<_>>();
    let target_parameters = target
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| (name, parameter.id()))
        .collect::<Vec<_>>();
    assert_eq!(
        source_parameters
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        vec!["0.weight", "1.bias", "1.weight"]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 2], vec![1., 3., 3., 1.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(f32s(first.output()), vec![-1., 2., 3., -4.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([
            ("0.weight".into(), 1),
            ("1.bias".into(), 1),
            ("1.weight".into(), 1),
        ])
    );
    Ok(())
}

#[test]
fn layernorm_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = layernorm_classifier(107, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 2]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 2], vec![1.; 2])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 3], vec![1.; 3])?).is_err());

    let norm = LayerNorm::new_static(Shape::new([2]), 1e-5, true)?;
    let norm_before = norm.state_dict()?;
    assert!(infer_module_cpu(&norm, TensorData::scalar(1.0f32)).is_err());
    assert!(infer_module_cpu(&norm, TensorData::new([1, 3], vec![1.; 3])?).is_err());
    assert_eq!(norm.state_dict()?, norm_before);

    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("1.unexpected".into(), TensorData::new([1], vec![1.])?);
    assert!(
        model
            .load_state_dict_strict(&crate::nn::StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict()?, before);
    Ok(())
}

fn rmsnorm_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let linear = Linear::new_static(2, 2, false, seed)?;
    let norm = RMSNorm::new_static(2, 0.0, true)?;
    if fixed {
        linear
            .weight
            .replace(TensorData::new([2, 2], vec![1., 0., 0., 1.])?)?;
        norm.weight
            .as_ref()
            .expect("configured affine weight")
            .replace(TensorData::new([2], vec![2., 3.])?)?;
    }
    let mut model = Sequential::default();
    model.push(linear);
    model.push(norm);
    Ok(model)
}

#[test]
fn rmsnorm_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = RMSNorm::new(&mut legacy_graph, 2, 1e-5, true)?;
    let graph_free = RMSNorm::new_static(2, 1e-5, true)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);
    assert!(RMSNorm::new_static(0, 1e-5, true).is_err());

    let source = rmsnorm_classifier(109, true)?;
    let target = rmsnorm_classifier(113, false)?;
    let source_state = source.state_dict()?;
    let source_parameters = source
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| (name, parameter.id()))
        .collect::<Vec<_>>();
    let target_parameters = target
        .trainable_parameters()?
        .into_iter()
        .map(|(name, parameter)| (name, parameter.id()))
        .collect::<Vec<_>>();
    assert_eq!(
        source_parameters
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        vec!["0.weight", "1.weight"]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 2], vec![3., 4., 0., 5.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    let expected = [1.697_056_3, 3.394_112_6, 0., 4.242_640_5];
    for (actual, expected) in f32s(first.output()).iter().zip(expected) {
        assert!((*actual - expected).abs() < 1e-5);
    }
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([("0.weight".into(), 1), ("1.weight".into(), 1),])
    );
    Ok(())
}

#[test]
fn rmsnorm_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = rmsnorm_classifier(127, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 2]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 2], vec![1.; 2])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 3], vec![1.; 3])?).is_err());

    let norm = RMSNorm::new_static(2, 1e-5, true)?;
    let norm_before = norm.state_dict()?;
    assert!(infer_module_cpu(&norm, TensorData::scalar(1.0f32)).is_err());
    assert!(infer_module_cpu(&norm, TensorData::new([1, 3], vec![1.; 3])?).is_err());
    assert_eq!(norm.state_dict()?, norm_before);

    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("1.unexpected".into(), TensorData::new([1], vec![1.])?);
    assert!(
        model
            .load_state_dict_strict(&crate::nn::StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict()?, before);
    Ok(())
}
