use super::{Conv1d, Conv1dOptions, ConvTranspose2d, Flatten, Linear, Module, Sequential};
use crate::{ConvTranspose2dOptions, DType, Error, Graph, Result, TensorData, infer_module_cpu};

fn classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let conv = Conv1d::new_static(1, 1, 2, Conv1dOptions::default(), true, seed)?;
    let linear = Linear::new_static(2, 1, true, seed.wrapping_add(1))?;
    if fixed {
        conv.weight
            .replace(TensorData::new([1, 1, 2], vec![2., 1.])?)?;
        conv.bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([1], vec![0.])?)?;
        linear
            .weight
            .replace(TensorData::new([1, 2], vec![1., 2.])?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([1], vec![1.])?)?;
    }
    let mut model = Sequential::default();
    model.push(conv);
    model.push(Flatten::new(1));
    model.push(linear);
    Ok(model)
}

#[test]
fn conv1d_static_constructor_and_module_forward_compose_deterministically() -> Result<()> {
    let mut legacy_graph = Graph::new();
    let legacy = Conv1d::new(
        &mut legacy_graph,
        1,
        1,
        2,
        Conv1dOptions::default(),
        true,
        41,
    )?;
    let graph_free = Conv1d::new_static(1, 1, 2, Conv1dOptions::default(), true, 41)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);

    let source = classifier(43, true)?;
    let target = classifier(47, false)?;
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
        vec!["0.bias", "0.weight", "2.bias", "2.weight"]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 3], vec![1., 2., 3., 3., 2., 1.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output().to_vec_f64(), [19., 19.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([
            ("0.bias".into(), 1),
            ("0.weight".into(), 1),
            ("2.bias".into(), 1),
            ("2.weight".into(), 1),
        ])
    );
    Ok(())
}

#[test]
fn conv1d_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = classifier(53, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 3], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 1]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 3], vec![1.; 3])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 1], vec![1.])?).is_err());
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("1.weight".into(), TensorData::new([1], vec![1.])?);
    assert!(
        model
            .load_state_dict_strict(&crate::nn::StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict()?, before);
    Ok(())
}

fn transpose_classifier(seed: u64, fixed: bool) -> Result<Sequential> {
    let transpose =
        ConvTranspose2d::new_static(1, 1, [1, 1], ConvTranspose2dOptions::default(), true, seed)?;
    let linear = Linear::new_static(1, 1, true, seed.wrapping_add(1))?;
    if fixed {
        transpose
            .weight
            .replace(TensorData::new([1, 1, 1, 1], vec![2.])?)?;
        transpose
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([1], vec![1.])?)?;
        linear.weight.replace(TensorData::new([1, 1], vec![3.])?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([1], vec![-1.])?)?;
    }
    let mut model = Sequential::default();
    model.push(transpose);
    model.push(Flatten::new(1));
    model.push(linear);
    Ok(model)
}

#[test]
fn conv_transpose2d_static_constructor_and_module_forward_compose_deterministically() -> Result<()>
{
    let mut legacy_graph = Graph::new();
    let legacy = ConvTranspose2d::new(
        &mut legacy_graph,
        1,
        1,
        [1, 1],
        ConvTranspose2dOptions::default(),
        true,
        61,
    )?;
    let graph_free =
        ConvTranspose2d::new_static(1, 1, [1, 1], ConvTranspose2dOptions::default(), true, 61)?;
    assert_eq!(legacy.state_dict()?, graph_free.state_dict()?);

    let source = transpose_classifier(63, true)?;
    let target = transpose_classifier(67, false)?;
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
        vec!["0.bias", "0.weight", "2.bias", "2.weight"]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 1, 1], vec![3., 4.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output().to_vec_f64(), [20., 26.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([
            ("0.bias".into(), 1),
            ("0.weight".into(), 1),
            ("2.bias".into(), 1),
            ("2.weight".into(), 1),
        ])
    );
    Ok(())
}

#[test]
fn conv_transpose2d_module_forward_preserves_empty_and_preflight_failure_contracts() -> Result<()> {
    let model = transpose_classifier(71, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 1, 1], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 1]);

    let before = model.state_dict()?;
    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 1, 1], vec![1.])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 1, 1], vec![1.])?).is_err());
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("1.weight".into(), TensorData::new([1], vec![1.])?);
    assert!(
        model
            .load_state_dict_strict(&crate::nn::StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict()?, before);
    Ok(())
}
