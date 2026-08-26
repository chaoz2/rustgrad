//! Public fresh-graph binary-logit module workflow acceptance.

use rustgrad::nn::Linear;
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    CpuBinaryModuleTrainer, DType, Module, ModuleBinaryCrossEntropy, Reduction, Result, Scalar,
    TensorData,
};

fn model() -> Linear {
    Linear::new_static(2, 1, true, 811).unwrap()
}

fn optimizer(model: &Linear) -> Result<Optimizer> {
    Optimizer::sgd_for_module(
        model,
        SgdConfig {
            lr: 0.2,
            ..SgdConfig::default()
        },
    )
}

fn batch() -> Result<(TensorData, TensorData)> {
    Ok((
        TensorData::new([4, 2], [-2., -1., -1., 1., 1., -1., 2., 1.].to_vec())?,
        TensorData::new([4, 1], [0., 0., 1., 1.].to_vec())?,
    ))
}

#[test]
fn cpu_binary_module_trainer_runs_fresh_graph_training_and_read_only_evaluation() -> Result<()> {
    let model = model();
    let mut optimizer = optimizer(&model)?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let (input, target) = batch()?;
    let before = model.state_dict()?;
    let initial = CpuBinaryModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleBinaryCrossEntropy::default(),
    )?
    .evaluate(input.clone(), target.clone())?;
    let mut trainer = CpuBinaryModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleBinaryCrossEntropy::default(),
    )?;
    let trained = trainer.train_step(input.clone(), target.clone())?;
    assert_eq!(trained.logits().shape().dims(), &[4, 1]);
    assert_eq!(trained.logits().dtype(), DType::F32);
    assert!(!trained.trace().steps.is_empty());
    assert_eq!(trained.optimizer_step(), 1);
    assert_eq!(trained.scheduler_epoch(), 1);
    assert_ne!(model.state_dict()?, before);
    let after_train = model.state_dict()?;
    let evaluated = trainer.evaluate(input, target)?;
    assert_eq!(model.state_dict()?, after_train);
    assert!(evaluated.loss() <= initial.loss());
    Ok(())
}

#[test]
fn cpu_binary_module_trainer_rejects_nonscalar_or_nonfloat_contracts_without_mutation() -> Result<()>
{
    let model = model();
    let mut optimizer = optimizer(&model)?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let before = (
        model.state_dict()?,
        optimizer.state_dict()?,
        scheduler.state_dict()?,
    );
    assert!(
        CpuBinaryModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleBinaryCrossEntropy {
                reduction: Reduction::None,
            },
        )
        .is_err()
    );
    let mut trainer = CpuBinaryModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleBinaryCrossEntropy::default(),
    )?;
    let (input, target) = batch()?;
    assert!(
        trainer
            .train_step(
                TensorData::from_scalars(
                    input.shape().clone(),
                    DType::I32,
                    std::iter::repeat_n(Scalar::I(0), input.shape().numel()?),
                )?,
                target.clone(),
            )
            .is_err()
    );
    assert!(
        trainer
            .train_step(input, TensorData::new([4], [0., 0., 1., 1.].to_vec())?)
            .is_err()
    );
    assert_eq!(model.state_dict()?, before.0);
    assert_eq!(trainer.optimizer().state_dict()?, before.1);
    assert_eq!(trainer.scheduler().state_dict()?, before.2);
    Ok(())
}
