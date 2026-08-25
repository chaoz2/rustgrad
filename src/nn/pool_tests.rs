use super::{Flatten, Linear, MaxPool2d, Module, Sequential};
use crate::{DType, Error, Pool2dOptions, Result, TensorData, infer_module_cpu};

fn classifier(seed: u64, fixed_linear: bool) -> Result<Sequential> {
    let pool = MaxPool2d::new(Pool2dOptions::default());
    let flatten = Flatten::new(1);
    let linear = Linear::new_static(1, 1, true, seed)?;
    if fixed_linear {
        linear.weight.replace(TensorData::new([1, 1], vec![2.])?)?;
        linear
            .bias
            .as_ref()
            .expect("configured bias")
            .replace(TensorData::new([1], vec![1.])?)?;
    }
    let mut module = Sequential::default();
    module.push(pool);
    module.push(flatten);
    module.push(linear);
    Ok(module)
}

#[test]
fn max_pool2d_composes_statelessly_in_fresh_static_modules() -> Result<()> {
    let source = classifier(7, true)?;
    let target = classifier(11, false)?;
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
        vec!["2.bias", "2.weight"]
    );
    assert_ne!(source_parameters, target_parameters);
    target.load_state_dict_strict(&source_state)?;
    assert_eq!(target.state_dict()?, source_state);

    let input = TensorData::new([2, 1, 2, 2], vec![1., 4., 3., 2., -1., -3., -2., -4.])?;
    let first = infer_module_cpu(&target, input.clone())?;
    let second = infer_module_cpu(&target, input)?;
    assert_eq!(first.output().to_vec_f64(), [9., -1.]);
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(
        first.parameter_versions(),
        &std::collections::BTreeMap::from([("2.bias".into(), 1), ("2.weight".into(), 1)])
    );
    Ok(())
}

#[test]
fn max_pool2d_module_forward_keeps_static_empty_and_error_contracts() -> Result<()> {
    let pool = MaxPool2d::new(Pool2dOptions::default());
    assert!(pool.state_dict()?.tensors().is_empty());
    assert!(pool.trainable_parameters()?.is_empty());

    let model = classifier(17, true)?;
    let empty = infer_module_cpu(&model, TensorData::new([0, 1, 2, 2], Vec::<f32>::new())?)?;
    assert_eq!(empty.output().shape().dims(), &[0, 1]);
    assert_eq!(empty.parameter_versions().len(), 2);

    assert!(matches!(
        infer_module_cpu(
            &model,
            TensorData::new([1, 1, 2, 2], vec![1.; 4])?.cast(DType::F64)
        ),
        Err(Error::SessionTraining { .. })
    ));
    assert!(infer_module_cpu(&model, TensorData::new([1, 1, 2], vec![1.; 2])?).is_err());

    let before = model.state_dict()?;
    let mut unexpected = before.clone().into_tensors();
    unexpected.insert("0.weight".into(), TensorData::new([1], vec![1.])?);
    assert!(
        model
            .load_state_dict_strict(&crate::nn::StateDict::from(unexpected))
            .is_err()
    );
    assert_eq!(model.state_dict()?, before);
    Ok(())
}
