//! Public acceptance for the documented CPU training workflow.

use rustgrad::nn::Linear;
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, BatchIter, CpuBackend, DType, Error, Graph, LossOptions, Module,
    PortableTrainingCheckpoint, Reduction, Result, TensorData, cross_entropy,
};
use std::collections::BTreeMap;

struct Classifier(Linear);

impl Classifier {
    fn new() -> Self {
        let mut graph = Graph::new();
        Self(Linear::new(&mut graph, 2, 2, true, 7).unwrap())
    }

    fn optimizer(&self) -> Optimizer {
        Optimizer::sgd(
            vec![
                ("weight".into(), self.0.weight.clone()),
                ("bias".into(), self.0.bias.clone().unwrap()),
            ],
            SgdConfig {
                lr: 0.25,
                ..SgdConfig::default()
            },
        )
        .unwrap()
    }
}

impl Module for Classifier {
    fn visit(
        &self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &rustgrad::Parameter, rustgrad::nn::StateKind),
    ) {
        self.0.visit(prefix, visitor);
    }
}

fn data(indices: &[usize]) -> (TensorData, TensorData) {
    const X: [[f32; 2]; 4] = [[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]];
    const Y: [u8; 4] = [0, 0, 1, 1];
    (
        TensorData::new(
            [indices.len(), 2],
            indices.iter().flat_map(|&index| X[index]).collect(),
        )
        .unwrap(),
        TensorData::from_scalars(
            [indices.len()],
            DType::U8,
            indices
                .iter()
                .map(|&index| rustgrad::Scalar::U(Y[index] as u64)),
        )
        .unwrap(),
    )
}

fn step(
    model: &Classifier,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    indices: &[usize],
) -> Result<f64> {
    let (inputs, targets) = data(indices);
    let mut graph = Graph::new();
    let input = graph.input("input", [indices.len(), 2]);
    let target = graph.input_dtype("target", [indices.len()], DType::U8);
    let logits = model.0.forward(&mut graph, input)?;
    let loss = cross_entropy(
        &mut graph,
        logits,
        target,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )?;
    let weight_grad = graph.grad(loss, model.0.weight.node(&graph)?)?;
    let bias = model.0.bias.as_ref().unwrap();
    let bias_grad = graph.grad(loss, bias.node(&graph)?)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    let loss_value = cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64();
    optimizer.step(&BTreeMap::from([
        (
            "weight".into(),
            Gradient::for_parameter(
                &model.0.weight,
                cpu.execute(&graph, weight_grad, &bindings)?,
            )?,
        ),
        (
            "bias".into(),
            Gradient::for_parameter(bias, cpu.execute(&graph, bias_grad, &bindings)?)?,
        ),
    ]))?;
    scheduler.step(optimizer)?;
    Ok(loss_value)
}

fn evaluate(model: &Classifier) -> Result<(f64, Vec<u8>)> {
    let indices = [0, 1, 2, 3];
    let (inputs, targets) = data(&indices);
    let mut graph = Graph::new();
    let input = graph.input("input", [4, 2]);
    let target = graph.input_dtype("target", [4], DType::U8);
    let logits = model.0.forward(&mut graph, input)?;
    let loss = cross_entropy(&mut graph, logits, target, LossOptions::default())?;
    let predictions = graph.argmax(logits, Some(-1), false)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    Ok((
        cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64(),
        cpu.execute(&graph, predictions, &bindings)?.to_le_bytes()?,
    ))
}

#[test]
fn documented_cpu_train_checkpoint_resume_and_evaluate_is_exact() {
    let batches = BatchIter::new(4, 3, 19, true, false)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);
    assert!(
        BatchIter::new(0, 1, 19, false, false)
            .unwrap()
            .next()
            .is_none()
    );
    assert!(matches!(
        BatchIter::new(1, 0, 0, false, false),
        Err(Error::Dataset { .. })
    ));

    let uninterrupted = Classifier::new();
    let mut uninterrupted_optimizer = uninterrupted.optimizer();
    let mut uninterrupted_scheduler = LearningRateScheduler::multi_step(vec![3], 0.5).unwrap();
    let initial = evaluate(&uninterrupted).unwrap();
    let mut losses = Vec::new();
    for indices in batches.iter().cycle().take(20) {
        losses.push(
            step(
                &uninterrupted,
                &mut uninterrupted_optimizer,
                &mut uninterrupted_scheduler,
                indices,
            )
            .unwrap(),
        );
    }
    let expected = evaluate(&uninterrupted).unwrap();
    assert!(losses.last().unwrap() < losses.first().unwrap());
    assert!(expected.0 < initial.0);
    assert!(
        expected
            .1
            .iter()
            .zip([0, 0, 1, 1])
            .filter(|(actual, expected)| **actual == *expected)
            .count()
            > initial
                .1
                .iter()
                .zip([0, 0, 1, 1])
                .filter(|(actual, expected)| **actual == *expected)
                .count()
    );

    let source = Classifier::new();
    let mut source_optimizer = source.optimizer();
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![3], 0.5).unwrap();
    for indices in batches.iter().cycle().take(10) {
        step(
            &source,
            &mut source_optimizer,
            &mut source_scheduler,
            indices,
        )
        .unwrap();
    }
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler).unwrap();
    let decoded = PortableTrainingCheckpoint::from_bytes(checkpoint.into_bytes()).unwrap();
    let resumed = Classifier::new();
    assert_ne!(source.0.weight.id(), resumed.0.weight.id());
    let mut resumed_optimizer = resumed.optimizer();
    let mut resumed_scheduler = LearningRateScheduler::multi_step(vec![3], 0.5).unwrap();
    decoded
        .restore(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)
        .unwrap();
    for indices in batches.iter().cycle().skip(10).take(10) {
        step(
            &resumed,
            &mut resumed_optimizer,
            &mut resumed_scheduler,
            indices,
        )
        .unwrap();
    }
    assert_eq!(
        uninterrupted.state_dict().unwrap(),
        resumed.state_dict().unwrap()
    );
    assert_eq!(
        uninterrupted_optimizer.state_dict().unwrap(),
        resumed_optimizer.state_dict().unwrap()
    );
    assert_eq!(
        uninterrupted_scheduler.state_dict().unwrap(),
        resumed_scheduler.state_dict().unwrap()
    );
    assert_eq!(expected, evaluate(&resumed).unwrap());

    let state_before = resumed.state_dict().unwrap();
    let optimizer_before = resumed_optimizer.state_dict().unwrap();
    let scheduler_before = resumed_scheduler.state_dict().unwrap();
    assert_eq!(evaluate(&resumed).unwrap(), expected);
    assert_eq!(resumed.state_dict().unwrap(), state_before);
    assert_eq!(resumed_optimizer.state_dict().unwrap(), optimizer_before);
    assert_eq!(resumed_scheduler.state_dict().unwrap(), scheduler_before);
}
