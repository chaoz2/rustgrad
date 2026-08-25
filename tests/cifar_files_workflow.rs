//! Public local-CIFAR file to CPU Conv training acceptance.

use rustgrad::nn::{AdaptiveAvgPool2d, Conv2d, Linear};
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, BatchIter, Conv2dOptions, CpuBackend, DType, Graph, LossOptions, Module,
    PortableTrainingCheckpoint, Reduction, Result, Scalar, TensorData, cross_entropy,
    load_cifar10_files,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct TinyConvClassifier {
    conv: Conv2d,
    pool: AdaptiveAvgPool2d,
    linear: Linear,
}

impl TinyConvClassifier {
    fn new() -> Self {
        let mut construction = Graph::new();
        Self {
            conv: Conv2d::new(
                &mut construction,
                3,
                2,
                [1, 1],
                Conv2dOptions::default(),
                true,
                31,
            )
            .unwrap(),
            pool: AdaptiveAvgPool2d::new([Some(1), Some(1)]),
            linear: Linear::new(&mut construction, 2, 2, true, 32).unwrap(),
        }
    }

    fn optimizer(&self) -> Optimizer {
        Optimizer::sgd(
            vec![
                ("conv.weight".into(), self.conv.weight.clone()),
                ("conv.bias".into(), self.conv.bias.clone().unwrap()),
                ("linear.weight".into(), self.linear.weight.clone()),
                ("linear.bias".into(), self.linear.bias.clone().unwrap()),
            ],
            SgdConfig {
                lr: 0.2,
                ..SgdConfig::default()
            },
        )
        .unwrap()
    }

    fn forward(&self, graph: &mut Graph, input: rustgrad::NodeId) -> Result<rustgrad::NodeId> {
        let convolved = self.conv.forward(graph, input)?;
        let pooled = self.pool.forward(graph, convolved)?;
        let batch = graph.shape(input)?.dims()[0];
        let flattened = graph.reshape(pooled, [batch, 2])?;
        self.linear.forward(graph, flattened)
    }

    fn parameters(&self) -> [(&str, &rustgrad::Parameter); 4] {
        [
            ("conv.weight", &self.conv.weight),
            ("conv.bias", self.conv.bias.as_ref().unwrap()),
            ("linear.weight", &self.linear.weight),
            ("linear.bias", self.linear.bias.as_ref().unwrap()),
        ]
    }
}

impl Module for TinyConvClassifier {
    fn visit(
        &self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &rustgrad::Parameter, rustgrad::nn::StateKind),
    ) {
        let name = |child: &str| {
            if prefix.is_empty() {
                child.to_string()
            } else {
                format!("{prefix}.{child}")
            }
        };
        self.conv.visit(&name("conv"), visitor);
        self.linear.visit(&name("linear"), visitor);
    }
}

fn cifar_record(label: u8, value: u8) -> Vec<u8> {
    let mut output = vec![label];
    output.extend(std::iter::repeat_n(value, 3 * 32 * 32));
    output
}

fn write_fixture() -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
    let root = std::env::temp_dir().join(format!(
        "rustgrad-cifar-workflow-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let first = root.join("first.bin");
    let second = root.join("second.bin");
    std::fs::write(&first, [cifar_record(0, 0), cifar_record(1, 255)].concat()).unwrap();
    std::fs::write(&second, [cifar_record(0, 0), cifar_record(1, 255)].concat()).unwrap();
    (root, vec![first, second])
}

fn batch(dataset: &rustgrad::Cifar10, indices: &[usize]) -> Result<(TensorData, TensorData)> {
    Ok((
        TensorData::from_scalars(
            [indices.len(), 3, 32, 32],
            DType::F32,
            indices.iter().flat_map(|&row| {
                (0..3 * 32 * 32).map(move |column| {
                    Scalar::F(
                        dataset
                            .images
                            .scalar_at(row * 3 * 32 * 32 + column)
                            .as_f64()
                            / 255.,
                    )
                })
            }),
        )?,
        TensorData::from_scalars(
            [indices.len()],
            DType::U8,
            indices
                .iter()
                .map(|&row| Scalar::U(dataset.labels.scalar_at(row).as_f64() as u64)),
        )?,
    ))
}

fn step(
    model: &TinyConvClassifier,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    dataset: &rustgrad::Cifar10,
    indices: &[usize],
) -> Result<f64> {
    let (inputs, targets) = batch(dataset, indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [indices.len(), 3, 32, 32]);
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
    let gradients = model
        .parameters()
        .into_iter()
        .map(|(name, parameter)| Ok((name.to_string(), graph.grad(loss, parameter.node(&graph)?)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    let value = cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64();
    let gradients = gradients
        .into_iter()
        .map(|(name, node)| {
            let parameter = model
                .parameters()
                .into_iter()
                .find(|(candidate, _)| *candidate == name)
                .unwrap()
                .1;
            Ok((
                name,
                Gradient::for_parameter(parameter, cpu.execute(&graph, node, &bindings)?)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    optimizer.step(&gradients)?;
    scheduler.step(optimizer)?;
    Ok(value)
}

fn evaluate(model: &TinyConvClassifier, dataset: &rustgrad::Cifar10) -> Result<(f64, Vec<u8>)> {
    let indices = (0..dataset.labels.len()).collect::<Vec<_>>();
    let (inputs, targets) = batch(dataset, &indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [indices.len(), 3, 32, 32]);
    let target = graph.input_dtype("target", [indices.len()], DType::U8);
    let logits = model.forward(&mut graph, input)?;
    let loss = cross_entropy(&mut graph, logits, target, LossOptions::default())?;
    let prediction = graph.argmax(logits, Some(-1), false)?;
    let mut bindings = model.input_bindings(&graph)?;
    bindings.insert("input".into(), inputs);
    bindings.insert("target".into(), targets);
    let cpu = CpuBackend;
    Ok((
        cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64(),
        cpu.execute(&graph, prediction, &bindings)?.to_le_bytes()?,
    ))
}

#[test]
fn local_cifar_files_conv_train_resume_and_evaluate_without_mutation() -> Result<()> {
    let (root, paths) = write_fixture();
    let dataset = load_cifar10_files(&paths).unwrap();
    assert_eq!(dataset.images.shape().dims(), &[4, 3, 32, 32]);
    assert_eq!(dataset.labels.to_le_bytes().unwrap(), vec![0, 1, 0, 1]);
    let batches = BatchIter::new(4, 3, 41, true, false)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);
    assert_eq!(
        batches,
        BatchIter::new(4, 3, 41, true, false)
            .unwrap()
            .collect::<Vec<_>>()
    );

    let baseline = TinyConvClassifier::new();
    let mut baseline_optimizer = baseline.optimizer();
    let mut baseline_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    let initial = evaluate(&baseline, &dataset).unwrap();
    let losses = batches
        .iter()
        .cycle()
        .take(8)
        .map(|indices| {
            step(
                &baseline,
                &mut baseline_optimizer,
                &mut baseline_scheduler,
                &dataset,
                indices,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let expected = evaluate(&baseline, &dataset).unwrap();
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "losses: {losses:?}"
    );
    assert!(expected.0 < initial.0);

    let source = TinyConvClassifier::new();
    let mut source_optimizer = source.optimizer();
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    for indices in batches.iter().cycle().take(4) {
        step(
            &source,
            &mut source_optimizer,
            &mut source_scheduler,
            &dataset,
            indices,
        )?;
    }
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler)?;
    let resumed = TinyConvClassifier::new();
    let mut resumed_optimizer = resumed.optimizer();
    let mut resumed_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5)?;
    checkpoint.restore(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)?;
    for indices in batches.iter().cycle().skip(4).take(4) {
        step(
            &resumed,
            &mut resumed_optimizer,
            &mut resumed_scheduler,
            &dataset,
            indices,
        )?;
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
    assert_eq!(expected, evaluate(&resumed, &dataset)?);
    let before = (
        resumed.state_dict()?,
        resumed_optimizer.state_dict()?,
        resumed_scheduler.state_dict()?,
    );
    assert_eq!(expected, evaluate(&resumed, &dataset)?);
    assert_eq!(before.0, resumed.state_dict()?);
    assert_eq!(before.1, resumed_optimizer.state_dict()?);
    assert_eq!(before.2, resumed_scheduler.state_dict()?);
    std::fs::remove_dir_all(root).unwrap();
    Ok(())
}
