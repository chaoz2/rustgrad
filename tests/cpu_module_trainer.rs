//! Public fresh-graph module training bridge acceptance.

use rustgrad::nn::Linear;
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    BatchIter, CpuModuleTrainer, DType, Error, Module, ModuleCrossEntropy,
    PortableTrainingCheckpoint, Result, Scalar, TensorData,
};

fn model() -> Linear {
    let mut graph = rustgrad::Graph::new();
    Linear::new(&mut graph, 2, 2, true, 61).unwrap()
}

fn make_optimizer(model: &Linear) -> Optimizer {
    Optimizer::sgd(
        vec![
            ("weight".into(), model.weight.clone()),
            ("bias".into(), model.bias.clone().unwrap()),
        ],
        SgdConfig {
            lr: 0.4,
            ..SgdConfig::default()
        },
    )
    .unwrap()
}

fn batch(indices: &[usize]) -> Result<(TensorData, TensorData)> {
    const INPUTS: [[f32; 2]; 4] = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]];
    const TARGETS: [u8; 4] = [0, 0, 1, 1];
    Ok((
        TensorData::new(
            [indices.len(), 2],
            indices.iter().flat_map(|&index| INPUTS[index]).collect(),
        )?,
        TensorData::from_scalars(
            [indices.len()],
            DType::U8,
            indices
                .iter()
                .map(|&index| Scalar::U(TARGETS[index] as u64)),
        )?,
    ))
}

fn evaluate(
    model: &Linear,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
) -> Result<(f64, Vec<u8>)> {
    let (input, target) = batch(&[0, 1, 2, 3])?;
    let trainer =
        CpuModuleTrainer::new(model, optimizer, scheduler, ModuleCrossEntropy::default())?;
    let result = trainer.evaluate(input, target)?;
    assert_eq!(result.logits().shape().dims(), &[4, 2]);
    assert_eq!(result.logits().dtype(), DType::F32);
    assert!(!result.trace().steps.is_empty());
    Ok((result.loss(), result.logits().to_le_bytes()?))
}

#[test]
fn cpu_module_trainer_trains_resumes_and_evaluates_without_raw_graph_plumbing() -> Result<()> {
    let batches = BatchIter::new(4, 3, 97, true, false)?.collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);

    let baseline = model();
    let mut baseline_optimizer = make_optimizer(&baseline);
    let mut baseline_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    let initial = evaluate(&baseline, &mut baseline_optimizer, &mut baseline_scheduler)?;
    let mut losses = Vec::new();
    for indices in batches.iter().cycle().take(8) {
        let (input, target) = batch(indices)?;
        let mut trainer = CpuModuleTrainer::new(
            &baseline,
            &mut baseline_optimizer,
            &mut baseline_scheduler,
            ModuleCrossEntropy::default(),
        )?;
        let step = trainer.train_step(input, target)?;
        assert_eq!(step.optimizer_step(), losses.len() as u64 + 1);
        assert_eq!(step.scheduler_epoch(), losses.len() as u64 + 1);
        assert_eq!(
            step.parameter_versions()["weight"],
            losses.len() as u64 + 1,
            "one successful step advances each parameter version once"
        );
        losses.push(step.loss());
    }
    let expected = evaluate(&baseline, &mut baseline_optimizer, &mut baseline_scheduler)?;
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "losses: {losses:?}"
    );
    assert!(expected.0 < initial.0);

    let source = model();
    let mut source_optimizer = make_optimizer(&source);
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    for indices in batches.iter().cycle().take(4) {
        let (input, target) = batch(indices)?;
        CpuModuleTrainer::new(
            &source,
            &mut source_optimizer,
            &mut source_scheduler,
            ModuleCrossEntropy::default(),
        )?
        .train_step(input, target)?;
    }
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler)?;
    let resumed = model();
    let mut resumed_optimizer = make_optimizer(&resumed);
    let mut resumed_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    checkpoint.restore(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)?;
    for indices in batches.iter().cycle().skip(4).take(4) {
        let (input, target) = batch(indices)?;
        CpuModuleTrainer::new(
            &resumed,
            &mut resumed_optimizer,
            &mut resumed_scheduler,
            ModuleCrossEntropy::default(),
        )?
        .train_step(input, target)?;
    }
    assert_eq!(baseline.state_dict()?, resumed.state_dict()?);
    assert_eq!(
        baseline_optimizer.state_dict()?,
        resumed_optimizer.state_dict()?
    );
    assert_eq!(
        baseline_scheduler.state_dict()?,
        resumed_scheduler.state_dict()?
    );
    assert_eq!(
        expected,
        evaluate(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)?
    );
    let before = (
        resumed.state_dict()?,
        resumed_optimizer.state_dict()?,
        resumed_scheduler.state_dict()?,
    );
    assert_eq!(
        expected,
        evaluate(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)?
    );
    assert_eq!(before.0, resumed.state_dict()?);
    assert_eq!(before.1, resumed_optimizer.state_dict()?);
    assert_eq!(before.2, resumed_scheduler.state_dict()?);
    Ok(())
}

#[test]
fn cpu_module_trainer_rejects_invalid_contracts_before_mutation() -> Result<()> {
    let model = model();
    let mut optimizer = make_optimizer(&model);
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let before = (
        model.state_dict()?,
        optimizer.state_dict()?,
        scheduler.state_dict()?,
    );
    {
        let trainer = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )?;
        let (_, target) = batch(&[0])?;
        assert!(matches!(
            trainer.evaluate(
                TensorData::from_scalars([1, 2], DType::F64, [Scalar::F(0.); 2])?,
                target
            ),
            Err(Error::SessionTraining { .. })
        ));
        let (input, _) = batch(&[0])?;
        assert!(matches!(
            trainer.evaluate(input, TensorData::new([1], vec![0f32])?),
            Err(Error::SessionTraining { .. })
        ));
    }
    let (input, target) = batch(&[0])?;
    let invalid_axis = ModuleCrossEntropy {
        options: rustgrad::LossOptions {
            class_axis: 3,
            ..rustgrad::LossOptions::default()
        },
    };
    assert!(
        CpuModuleTrainer::new(&model, &mut optimizer, &mut scheduler, invalid_axis)?
            .train_step(input, target)
            .is_err()
    );
    assert_eq!(before.0, model.state_dict()?);
    assert_eq!(before.1, optimizer.state_dict()?);
    assert_eq!(before.2, scheduler.state_dict()?);

    let mut wrong_optimizer = Optimizer::sgd(
        vec![("wrong".into(), model.weight.clone())],
        SgdConfig::default(),
    )?;
    assert!(matches!(
        CpuModuleTrainer::new(
            &model,
            &mut wrong_optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default()
        ),
        Err(Error::SessionTraining { .. })
    ));
    let mut plateau = LearningRateScheduler::reduce_on_plateau(
        rustgrad::optim::PlateauMode::Min,
        0.5,
        1,
        0.,
        rustgrad::optim::ThresholdMode::Absolute,
    )?;
    assert!(
        CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut plateau,
            ModuleCrossEntropy::default()
        )
        .is_err()
    );

    let mut fresh_optimizer = make_optimizer(&model);
    let mut fresh_scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let mut trainer = CpuModuleTrainer::new(
        &model,
        &mut fresh_optimizer,
        &mut fresh_scheduler,
        ModuleCrossEntropy::default(),
    )?;
    let replacement = model.weight.snapshot()?.data;
    model.weight.replace(replacement)?;
    let stale_before = (
        model.state_dict()?,
        trainer.optimizer().state_dict()?,
        trainer.scheduler().state_dict()?,
    );
    let (input, target) = batch(&[0])?;
    assert!(trainer.train_step(input, target).is_err());
    assert_eq!(stale_before.0, model.state_dict()?);
    assert_eq!(stale_before.1, trainer.optimizer().state_dict()?);
    assert_eq!(stale_before.2, trainer.scheduler().state_dict()?);
    Ok(())
}
