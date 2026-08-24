use super::*;
use crate::{Backend, CpuBackend, Error, Graph, NodeId, Storage, TensorData};
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
