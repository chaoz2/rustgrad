//! Deterministic CPU-only train, portable checkpoint, resume, and evaluation.
//!
//! Run with `cargo run --example cpu_train_resume`.

use rustgrad::nn::Linear;
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, BatchIter, CpuBackend, DType, Graph, LossOptions, Module, PortableTrainingCheckpoint,
    Reduction, Result, TensorData, cross_entropy,
};
use std::collections::BTreeMap;

struct Classifier {
    linear: Linear,
}

impl Classifier {
    fn new() -> Result<Self> {
        let mut construction = Graph::new();
        Ok(Self {
            linear: Linear::new(&mut construction, 2, 2, true, 7)?,
        })
    }

    fn optimizer(&self) -> Result<Optimizer> {
        Optimizer::sgd(
            vec![
                ("weight".into(), self.linear.weight.clone()),
                (
                    "bias".into(),
                    self.linear.bias.clone().expect("classifier has a bias"),
                ),
            ],
            SgdConfig {
                lr: 0.25,
                momentum: 0.0,
                dampening: 0.0,
                nesterov: false,
                weight_decay: 0.0,
            },
        )
    }

    fn forward(&self, graph: &mut Graph, input: rustgrad::NodeId) -> Result<rustgrad::NodeId> {
        self.linear.forward(graph, input)
    }
}

impl Module for Classifier {
    fn visit(
        &self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &rustgrad::Parameter, rustgrad::nn::StateKind),
    ) {
        self.linear.visit(prefix, visitor);
    }
}

fn batch_data(indices: &[usize]) -> Result<(TensorData, TensorData)> {
    const FEATURES: [[f32; 2]; 4] = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]];
    const LABELS: [u8; 4] = [0, 0, 1, 1];
    let features = indices
        .iter()
        .flat_map(|&index| FEATURES[index])
        .collect::<Vec<_>>();
    let labels = indices
        .iter()
        .map(|&index| LABELS[index])
        .collect::<Vec<_>>();
    Ok((
        TensorData::new([indices.len(), 2], features)?,
        TensorData::from_scalars(
            [indices.len()],
            DType::U8,
            labels
                .into_iter()
                .map(|label| rustgrad::Scalar::U(label as u64)),
        )?,
    ))
}

fn train_step(
    model: &Classifier,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    indices: &[usize],
) -> Result<f64> {
    let (inputs, targets) = batch_data(indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [indices.len(), 2]);
    let target = graph.input_dtype("target", [indices.len()], DType::U8);
    let logits = model.forward(&mut graph, input)?;
    let loss = cross_entropy(
        &mut graph,
        logits,
        target,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )?;
    let weight_grad = graph.grad(loss, model.linear.weight.node(&graph)?)?;
    let bias = model.linear.bias.as_ref().expect("classifier has a bias");
    let bias_grad = graph.grad(loss, bias.node(&graph)?)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    let value = cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64();
    let gradients = BTreeMap::from([
        (
            "weight".into(),
            Gradient::for_parameter(
                &model.linear.weight,
                cpu.execute(&graph, weight_grad, &bindings)?,
            )?,
        ),
        (
            "bias".into(),
            Gradient::for_parameter(bias, cpu.execute(&graph, bias_grad, &bindings)?)?,
        ),
    ]);
    optimizer.step(&gradients)?;
    scheduler.step(optimizer)?;
    Ok(value)
}

fn evaluate(model: &Classifier) -> Result<(f64, usize)> {
    let indices = [0, 1, 2, 3];
    let (inputs, targets) = batch_data(&indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [4, 2]);
    let target = graph.input_dtype("target", [4], DType::U8);
    let logits = model.forward(&mut graph, input)?;
    let loss = cross_entropy(&mut graph, logits, target, LossOptions::default())?;
    let prediction = graph.argmax(logits, Some(-1), false)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    let predictions = cpu.execute(&graph, prediction, &bindings)?.to_vec_f64();
    Ok((
        cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64(),
        predictions
            .iter()
            .zip([0.0, 0.0, 1.0, 1.0])
            .filter(|(actual, expected)| **actual == *expected)
            .count(),
    ))
}

fn main() -> Result<()> {
    let batches = BatchIter::new(4, 3, 19, true, false)?.collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);

    let source = Classifier::new()?;
    let mut source_optimizer = source.optimizer()?;
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![3], 0.5)?;
    let mut losses = Vec::new();
    for indices in batches.iter().cycle().take(3) {
        losses.push(train_step(
            &source,
            &mut source_optimizer,
            &mut source_scheduler,
            indices,
        )?);
    }
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler)?;

    let restored = Classifier::new()?;
    let mut restored_optimizer = restored.optimizer()?;
    let mut restored_scheduler = LearningRateScheduler::multi_step(vec![3], 0.5)?;
    checkpoint.restore(&restored, &mut restored_optimizer, &mut restored_scheduler)?;
    for indices in batches.iter().cycle().skip(3).take(3) {
        losses.push(train_step(
            &restored,
            &mut restored_optimizer,
            &mut restored_scheduler,
            indices,
        )?);
    }
    let (evaluation_loss, correct) = evaluate(&restored)?;
    println!(
        "first loss: {:.6}, final batch loss: {:.6}",
        losses[0],
        losses.last().unwrap()
    );
    println!("evaluation loss: {evaluation_loss:.6}, accuracy: {correct}/4");
    Ok(())
}
