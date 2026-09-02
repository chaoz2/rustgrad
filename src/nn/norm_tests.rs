use super::*;
use crate::Conv2dOptions;
use crate::Scalar;
use crate::nn::{AdaptiveAvgPool2d, Conv2d, Flatten, Linear, ReLU, Sequential};
use crate::{
    Backend, CpuBackend, DType, Error, Graph, NodeId, Result, Shape, Storage, TensorData,
    TrainingContext, infer_module_cpu,
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
fn pending_mode_effects_commit_batchnorm_buffers_atomically_and_retry() -> Result<()> {
    let mut graph = Graph::new();
    let first = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1)?;
    let second = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1)?;
    let input = graph.input("x", [2, 1]);
    let first_forward = first.forward_mode(&mut graph, input, Mode::Training)?;
    let second_forward = second.forward_mode(&mut graph, first_forward.output, Mode::Training)?;
    let mut effects = first_forward.pending;
    effects.append(second_forward.pending);
    let nodes = effects.batchnorm_stat_nodes();
    assert_eq!(nodes.len(), 2);

    let mut bindings = first.input_bindings(&graph)?;
    bindings.extend(second.input_bindings(&graph)?);
    bindings.insert("x".into(), TensorData::new([2, 1], vec![1., 3.])?);
    let output = CpuBackend.execute(&graph, second_forward.output, &bindings)?;
    assert_eq!(output.shape().dims(), &[2, 1]);
    let stats = nodes
        .iter()
        .map(|&(mean, variance)| {
            Ok(crate::RealizedBatchNormStats {
                mean: CpuBackend.execute(&graph, mean, &bindings)?,
                variance: CpuBackend.execute(&graph, variance, &bindings)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let before_first = first.state_dict()?;
    let before_second = second.state_dict()?;
    let malformed = vec![
        crate::RealizedBatchNormStats {
            mean: stats[0].mean.clone(),
            variance: stats[0].variance.clone(),
        },
        crate::RealizedBatchNormStats {
            mean: TensorData::scalar(0.0f32),
            variance: stats[1].variance.clone(),
        },
    ];
    assert!(matches!(
        effects.commit_batchnorm(malformed),
        Err(Error::BatchNormToken { .. })
    ));
    assert_eq!(first.state_dict()?, before_first);
    assert_eq!(second.state_dict()?, before_second);

    effects.commit_batchnorm(stats)?;
    assert_ne!(first.state_dict()?, before_first);
    assert_ne!(second.state_dict()?, before_second);
    assert!(matches!(
        effects.commit_batchnorm(Vec::new()),
        Err(Error::BatchNormToken { .. })
    ));
    Ok(())
}

#[test]
fn ambient_batchnorm_keeps_state_commit_explicit_atomic_and_retryable() -> Result<()> {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 2, 1e-5, false, true, 0.1)?;
    let input = graph.input("x", [2, 2]);
    let training = TrainingContext::training();
    let forward = norm.forward_ambient(&mut graph, input)?;
    drop(training);
    assert_eq!(forward.pending.batchnorm_stat_nodes().len(), 1);

    let mut bindings = norm.input_bindings(&graph)?;
    bindings.insert(
        "x".into(),
        TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 6.0])?,
    );
    let stats = forward
        .pending
        .batchnorm_stat_nodes()
        .into_iter()
        .map(|(mean, variance)| {
            Ok(RealizedBatchNormStats {
                mean: CpuBackend.execute(&graph, mean, &bindings)?,
                variance: CpuBackend.execute(&graph, variance, &bindings)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let before = norm.state_dict()?;
    assert!(matches!(
        forward
            .pending
            .commit_batchnorm(vec![RealizedBatchNormStats {
                mean: TensorData::scalar(0.0),
                variance: stats[0].variance.clone(),
            }]),
        Err(Error::BatchNormToken { .. })
    ));
    assert_eq!(norm.state_dict()?, before);

    forward.pending.commit_batchnorm(stats)?;
    assert_ne!(norm.state_dict()?, before);
    assert_eq!(norm.num_batches_tracked.value()?.scalar_at(0).as_u64(), 1);
    Ok(())
}

#[test]
fn mode_aware_batchnorm_eval_is_read_only_and_has_no_pending_effects() -> Result<()> {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1)?;
    let input = graph.input("x", [2, 1]);
    let forward = norm.forward_mode(&mut graph, input, Mode::Eval)?;
    assert!(forward.pending.is_empty());
    let before = norm.state_dict()?;
    let mut bindings = norm.input_bindings(&graph)?;
    bindings.insert("x".into(), TensorData::new([2, 1], vec![1., 3.])?);
    let first = CpuBackend.execute(&graph, forward.output, &bindings)?;
    let second = CpuBackend.execute(&graph, forward.output, &bindings)?;
    assert_eq!(first, second);
    assert_eq!(norm.state_dict()?, before);
    Ok(())
}

#[test]
fn pending_mode_effects_reject_duplicate_batchnorm_targets_before_mutation() -> Result<()> {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1)?;
    let input = graph.input("x", [2, 1]);
    let first = norm.forward_mode(&mut graph, input, Mode::Training)?;
    let second = norm.forward_mode(&mut graph, first.output, Mode::Training)?;
    let mut effects = first.pending;
    effects.append(second.pending);
    let nodes = effects.batchnorm_stat_nodes();
    let mut bindings = norm.input_bindings(&graph)?;
    bindings.insert("x".into(), TensorData::new([2, 1], vec![1., 3.])?);
    let values = nodes
        .iter()
        .map(|&(mean, variance)| {
            Ok(crate::RealizedBatchNormStats {
                mean: CpuBackend.execute(&graph, mean, &bindings)?,
                variance: CpuBackend.execute(&graph, variance, &bindings)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let before = norm.state_dict()?;
    assert!(matches!(
        effects.commit_batchnorm(values),
        Err(Error::BatchNormToken {
            reason: "duplicate pending mode effect target"
        })
    ));
    assert_eq!(norm.state_dict()?, before);
    Ok(())
}

#[test]
fn batchnorm3d_is_the_batchnorm_alias_with_rank_three_module_surface() {
    let mut graph = Graph::new();
    let norm: BatchNorm3d = BatchNorm3d::new(&mut graph, 2, 1e-5, true, true, 0.1).unwrap();
    let _: &BatchNorm = &norm;

    let input = graph.input("input", [1, 2, 3]);
    let output = norm.forward(&mut graph, input, Mode::Eval).unwrap().output;
    assert_eq!(graph.shape(output).unwrap().dims(), &[1, 2, 3]);

    let state = get_state_dict(&norm, "");
    let names: Vec<_> = state.keys().collect();
    assert_eq!(
        names,
        [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ]
    );
}

#[test]
fn batchnorm_stale_statistics_preflight_leaves_other_running_buffers_unchanged() {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1).unwrap();
    let input = graph.input("x", [2, 1]);
    let token = norm
        .forward(&mut graph, input, Mode::Training)
        .unwrap()
        .pending
        .unwrap();
    let mean_before = norm.running_mean.as_ref().unwrap().snapshot().unwrap();
    let batch_before = norm.num_batches_tracked.snapshot().unwrap();
    let var = norm.running_var.as_ref().unwrap();
    var.replace(var.value().unwrap()).unwrap();
    assert!(matches!(
        token.commit_stats(
            &norm,
            TensorData::new([1], vec![2.]).unwrap(),
            TensorData::new([1], vec![3.]).unwrap(),
        ),
        Err(Error::BatchNormToken { .. })
    ));
    assert_eq!(
        norm.running_mean.as_ref().unwrap().snapshot().unwrap().data,
        mean_before.data
    );
    assert_eq!(
        norm.num_batches_tracked.snapshot().unwrap().data,
        batch_before.data
    );
}

#[test]
fn batchnorm_counter_and_parameter_version_overflow_preflight_every_statistic() {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 1, 1e-5, false, true, 0.1).unwrap();
    norm.num_batches_tracked
        .replace(TensorData::scalar_with_dtype(
            Scalar::U(u64::MAX),
            DType::U64,
        ))
        .unwrap();
    let input = graph.input("x", [2, 1]);
    let token = norm
        .forward(&mut graph, input, Mode::Training)
        .unwrap()
        .pending
        .unwrap();
    let mean_before = norm.running_mean.as_ref().unwrap().snapshot().unwrap();
    let var_before = norm.running_var.as_ref().unwrap().snapshot().unwrap();
    let batches_before = norm.num_batches_tracked.snapshot().unwrap();
    assert!(matches!(
        token.commit_stats(
            &norm,
            TensorData::new([1], vec![2.]).unwrap(),
            TensorData::new([1], vec![3.]).unwrap(),
        ),
        Err(Error::BatchNormToken {
            reason: "batch counter overflow"
        })
    ));
    assert_eq!(
        norm.running_mean.as_ref().unwrap().snapshot().unwrap().data,
        mean_before.data
    );
    assert_eq!(
        norm.running_mean
            .as_ref()
            .unwrap()
            .snapshot()
            .unwrap()
            .version,
        mean_before.version
    );
    assert_eq!(
        norm.running_var.as_ref().unwrap().snapshot().unwrap().data,
        var_before.data
    );
    assert_eq!(
        norm.running_var
            .as_ref()
            .unwrap()
            .snapshot()
            .unwrap()
            .version,
        var_before.version
    );
    assert_eq!(
        norm.num_batches_tracked.snapshot().unwrap().data,
        batches_before.data
    );
    assert_eq!(
        norm.num_batches_tracked.snapshot().unwrap().version,
        batches_before.version
    );
}

#[test]
fn batchnorm_preflights_configured_channels_before_binding_or_staging_statistics() {
    let mut graph = Graph::new();
    let norm = BatchNorm::new(&mut graph, 2, 1e-5, true, true, 0.1).unwrap();
    let mean_before = norm.running_mean.as_ref().unwrap().snapshot().unwrap();
    let var_before = norm.running_var.as_ref().unwrap().snapshot().unwrap();
    let batches_before = norm.num_batches_tracked.snapshot().unwrap();
    let wrong_channels = graph.input("wrong_channels", [1, 3]);
    assert!(matches!(
        norm.forward(&mut graph, wrong_channels, Mode::Eval),
        Err(Error::InvalidReshape { .. })
    ));
    assert!(graph.parameter_bindings().is_empty());
    assert_eq!(
        norm.running_mean.as_ref().unwrap().snapshot().unwrap().data,
        mean_before.data
    );
    assert_eq!(
        norm.running_var.as_ref().unwrap().snapshot().unwrap().data,
        var_before.data
    );
    assert_eq!(
        norm.num_batches_tracked.snapshot().unwrap().data,
        batches_before.data
    );

    let valid = graph.input("valid", [1, 2]);
    assert!(
        norm.forward(&mut graph, valid, Mode::Training)
            .unwrap()
            .pending
            .is_some()
    );

    let mut stateless_graph = Graph::new();
    let stateless = BatchNorm::new(&mut stateless_graph, 2, 1e-5, false, false, 0.1).unwrap();
    let stateless_wrong_channels = stateless_graph.input("stateless_wrong_channels", [1, 3]);
    assert!(matches!(
        stateless.forward(
            &mut stateless_graph,
            stateless_wrong_channels,
            Mode::Training
        ),
        Err(Error::InvalidReshape { .. })
    ));
    assert!(stateless_graph.parameter_bindings().is_empty());
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

fn groupnorm_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let conv = Conv2d::new_static(1, 4, [1, 1], Conv2dOptions::default(), true, seed)?;
    let norm = GroupNorm::new_static(2, 4, 1e-5, true)?;
    let linear = Linear::new_static(4, 2, true, seed.wrapping_add(1))?;
    if fixed {
        conv.weight
            .replace(TensorData::new([4, 1, 1, 1], vec![1., -1., 0.5, 2.])?)?;
        conv.bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([4], vec![0.5, -0.5, 1., -1.])?)?;
        norm.weight
            .as_ref()
            .expect("configured affine weight")
            .replace(TensorData::new([4], vec![1.5, 0.5, 2., 1.])?)?;
        norm.bias
            .as_ref()
            .expect("configured affine bias")
            .replace(TensorData::new([4], vec![0.25, -0.25, 0.5, -0.5])?)?;
        linear.weight.replace(TensorData::new(
            [2, 4],
            vec![1., -1., 0.5, 2., -0.5, 1., 2., -1.],
        )?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([2], vec![0.25, -0.5])?)?;
    }
    let mut model = Sequential::default();
    model.push(conv);
    model.push(norm);
    model.push(ReLU::new());
    model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
    model.push(Flatten::new(1));
    model.push(linear);
    Ok(model)
}

#[test]
fn groupnorm_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = GroupNorm::new(&mut legacy_graph, 2, 4, 1e-5, true)?;
    let graph_free = GroupNorm::new_static(2, 4, 1e-5, true)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);
    assert!(GroupNorm::new_static(0, 4, 1e-5, true).is_err());
    assert!(GroupNorm::new_static(3, 4, 1e-5, true).is_err());
    assert!(GroupNorm::new_static(2, 4, f32::NAN, true).is_err());

    let source = groupnorm_classifier(149, true)?;
    let target = groupnorm_classifier(151, false)?;
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
        vec![
            "0.bias", "0.weight", "1.bias", "1.weight", "5.bias", "5.weight"
        ]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 2, 2], vec![1., 2., 3., 4., 2., 4., 6., 8.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([
            ("0.bias".into(), 1),
            ("0.weight".into(), 1),
            ("1.bias".into(), 1),
            ("1.weight".into(), 1),
            ("5.bias".into(), 1),
            ("5.weight".into(), 1),
        ])
    );
    Ok(())
}

#[test]
fn groupnorm_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = groupnorm_classifier(157, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 2, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 2]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 2, 2], vec![1.; 4])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 2, 2, 2], vec![1.; 8])?).is_err());
    assert!(infer_module_cpu(&model, TensorData::new([1, 1, 2], vec![1.; 2])?).is_err());
    assert_eq!(model.state_dict()?, before);

    let norm = GroupNorm::new_static(2, 4, 1e-5, true)?;
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

fn instancenorm_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let conv = Conv2d::new_static(1, 4, [1, 1], Conv2dOptions::default(), true, seed)?;
    let norm = InstanceNorm::new_static(4, 1e-5, true)?;
    let linear = Linear::new_static(4, 2, true, seed.wrapping_add(1))?;
    if fixed {
        conv.weight
            .replace(TensorData::new([4, 1, 1, 1], vec![1., -1., 0.5, 2.])?)?;
        conv.bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([4], vec![0.5, -0.5, 1., -1.])?)?;
        linear.weight.replace(TensorData::new(
            [2, 4],
            vec![1., -1., 0.5, 2., -0.5, 1., 2., -1.],
        )?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([2], vec![0.25, -0.5])?)?;
    }
    let mut model = Sequential::default();
    model.push(conv);
    model.push(norm);
    model.push(ReLU::new());
    model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
    model.push(Flatten::new(1));
    model.push(linear);
    Ok(model)
}

#[test]
fn instancenorm_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = InstanceNorm::new(&mut legacy_graph, 4, 1e-5, true)?;
    let graph_free = InstanceNorm::new_static(4, 1e-5, true)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);
    assert!(InstanceNorm::new_static(0, 1e-5, true).is_err());
    assert!(InstanceNorm::new_static(4, f32::NAN, true).is_err());

    let source = instancenorm_classifier(163, true)?;
    let target = instancenorm_classifier(167, false)?;
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
        vec![
            "0.bias", "0.weight", "1.bias", "1.weight", "5.bias", "5.weight"
        ]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 2, 2], vec![1., 2., 3., 4., 2., 4., 6., 8.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(first.parameter_versions(), second.parameter_versions());

    let legacy_input = legacy_graph.input("legacy", [1, 4, 2]);
    let legacy_output = legacy.forward(&mut legacy_graph, legacy_input)?;
    let mut legacy_bindings = legacy.input_bindings(&legacy_graph)?;
    legacy_bindings.insert(
        "legacy".into(),
        TensorData::new([1, 4, 2], vec![1., 3., 10., 14., 2., 4., 8., 16.])?,
    );
    let expected = CpuBackend.execute(&legacy_graph, legacy_output, &legacy_bindings)?;
    let actual = infer_module_cpu(
        &graph_free,
        TensorData::new([1, 4, 2], vec![1., 3., 10., 14., 2., 4., 8., 16.])?,
    )?;
    assert_eq!(actual.output(), &expected);
    Ok(())
}

#[test]
fn instancenorm_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = instancenorm_classifier(173, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 2, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 2]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 2, 2], vec![1.; 4])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 2, 2, 2], vec![1.; 8])?).is_err());
    assert!(infer_module_cpu(&model, TensorData::new([1, 1, 2], vec![1.; 2])?).is_err());
    assert_eq!(model.state_dict()?, before);

    let norm = InstanceNorm::new_static(4, 1e-5, true)?;
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

#[test]
fn groupnorm_preflights_grouped_extent_before_lowering() {
    let mut malformed = Graph::new();
    let norm = GroupNorm::new(&mut malformed, 1, 2, 1e-5, false).unwrap();
    let input = malformed.input("input", [1, 2, usize::MAX]);
    let original_nodes = malformed.node_count();
    assert!(matches!(
        norm.forward(&mut malformed, input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), original_nodes);
    assert!(malformed.parameter_bindings().is_empty());

    let mut valid = Graph::new();
    let norm = GroupNorm::new(&mut valid, 2, 4, 1e-5, false).unwrap();
    let input = valid.input("input", [1, 4, 1]);
    let output = norm.forward(&mut valid, input).unwrap();
    let output = CpuBackend
        .execute(
            &valid,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::new([1, 4, 1], vec![1., 3., 10., 14.]).unwrap(),
            )]),
        )
        .unwrap();
    let values = f32s(&output);
    assert!((values[0] + 1.).abs() < 1e-4 && (values[1] - 1.).abs() < 1e-4);
    assert!((values[2] + 1.).abs() < 1e-4 && (values[3] - 1.).abs() < 1e-4);
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

fn layernorm2d_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let conv = Conv2d::new_static(1, 2, [1, 1], Conv2dOptions::default(), true, seed)?;
    let norm = LayerNorm2d::new_static(2, 1e-5, true)?;
    let linear = Linear::new_static(2, 2, true, seed.wrapping_add(1))?;
    if fixed {
        conv.weight
            .replace(TensorData::new([2, 1, 1, 1], vec![1., -1.])?)?;
        conv.bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([2], vec![0.5, -0.5])?)?;
        norm.inner
            .weight
            .as_ref()
            .expect("configured affine weight")
            .replace(TensorData::new([2], vec![1.5, 0.5])?)?;
        norm.inner
            .bias
            .as_ref()
            .expect("configured affine bias")
            .replace(TensorData::new([2], vec![0.25, -0.25])?)?;
        linear
            .weight
            .replace(TensorData::new([2, 2], vec![1., -1., 0.5, 2.])?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([2], vec![0.25, -0.5])?)?;
    }
    let mut model = Sequential::default();
    model.push(conv);
    model.push(norm);
    model.push(ReLU::new());
    model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
    model.push(Flatten::new(1));
    model.push(linear);
    Ok(model)
}

#[test]
fn layernorm2d_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = LayerNorm2d::new(&mut legacy_graph, 2, 1e-5, true)?;
    let graph_free = LayerNorm2d::new_static(2, 1e-5, true)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);
    assert!(LayerNorm2d::new_static(0, 1e-5, true).is_err());
    assert!(LayerNorm2d::new_static(2, f32::NAN, true).is_err());

    let source = layernorm2d_classifier(131, true)?;
    let target = layernorm2d_classifier(137, false)?;
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
        vec![
            "0.bias", "0.weight", "1.bias", "1.weight", "5.bias", "5.weight"
        ]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 2, 2], vec![1., 2., 3., 4., 2., 4., 6., 8.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([
            ("0.bias".into(), 1),
            ("0.weight".into(), 1),
            ("1.bias".into(), 1),
            ("1.weight".into(), 1),
            ("5.bias".into(), 1),
            ("5.weight".into(), 1),
        ])
    );
    Ok(())
}

#[test]
fn layernorm2d_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = layernorm2d_classifier(139, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 2, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 2]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 2, 2], vec![1.; 4])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 2, 2, 2], vec![1.; 8])?).is_err());
    assert!(infer_module_cpu(&model, TensorData::new([1, 1, 2], vec![1.; 2])?).is_err());
    assert_eq!(model.state_dict()?, before);

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
    let zero = RMSNorm::new_static(0, 1e-5, true)?;
    assert_eq!(zero.state_dict()?.tensors()["weight"].shape().dims(), &[0]);

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

#[test]
fn layernorm2d_preflights_channels_before_layout_lowering() {
    let mut malformed = Graph::new();
    let norm = LayerNorm2d::new(&mut malformed, 2, 1e-5, true).unwrap();
    let input = malformed.input("input", [1, 3, 1, 1]);
    let original_nodes = malformed.node_count();
    assert!(matches!(
        norm.forward(&mut malformed, input),
        Err(Error::InvalidReshape { .. })
    ));
    assert_eq!(malformed.node_count(), original_nodes);
    assert!(malformed.parameter_bindings().is_empty());

    let mut valid = Graph::new();
    let norm = LayerNorm2d::new(&mut valid, 2, 0.0, false).unwrap();
    let input = valid.input("input", [1, 2, 1, 2]);
    let output = norm.forward(&mut valid, input).unwrap();
    let output = execute(
        &valid,
        output,
        &norm,
        (
            "input",
            TensorData::new([1, 2, 1, 2], vec![1., 3., 5., 7.]).unwrap(),
        ),
    );
    assert_eq!(f32s(&output), vec![-1., -1., 1., 1.]);
}

#[test]
fn layernorm_constructor_preflights_nonempty_checked_normalized_geometry() {
    let mut graph = Graph::new();
    assert!(LayerNorm::new(&mut graph, Shape::new([0]), 1e-5, false).is_err());
    assert!(LayerNorm::new(&mut graph, Shape::new([usize::MAX, 2]), 1e-5, false,).is_err());
    assert!(graph.parameter_bindings().is_empty());

    let norm = LayerNorm::new(&mut graph, Shape::new([2]), 1e-5, true).unwrap();
    assert_eq!(
        norm.state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["bias", "weight"]
    );
}
