//! Public local-IDX workflow acceptance.

use rustgrad::nn::Linear;
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, BatchIter, CpuBackend, DType, Graph, LossOptions, MnistIdx, Module,
    PortableTrainingCheckpoint, Reduction, Result, Scalar, TensorData, cross_entropy,
    load_mnist_idx_files,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Classifier(Linear);

impl Classifier {
    fn new() -> Self {
        let mut construction = Graph::new();
        Self(Linear::new(&mut construction, 28 * 28, 2, true, 13).unwrap())
    }

    fn optimizer(&self) -> Optimizer {
        Optimizer::sgd(
            vec![
                ("weight".into(), self.0.weight.clone()),
                ("bias".into(), self.0.bias.clone().unwrap()),
            ],
            SgdConfig {
                lr: 0.5,
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

fn write_fixture() -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "rustgrad-mnist-workflow-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let mut images = Vec::new();
    images.extend_from_slice(&2051u32.to_be_bytes());
    images.extend_from_slice(&4u32.to_be_bytes());
    images.extend_from_slice(&28u32.to_be_bytes());
    images.extend_from_slice(&28u32.to_be_bytes());
    for (index, value) in [0u8, 0, 255, 255].into_iter().enumerate() {
        let mut image = vec![0; 28 * 28];
        image[index] = value;
        image[28 + index] = value;
        images.extend(image);
    }
    let mut labels = Vec::new();
    labels.extend_from_slice(&2049u32.to_be_bytes());
    labels.extend_from_slice(&4u32.to_be_bytes());
    labels.extend_from_slice(&[0, 0, 1, 1]);
    let image_path = root.join("train-images.idx3-ubyte");
    let label_path = root.join("train-labels.idx1-ubyte");
    std::fs::write(&image_path, images).unwrap();
    std::fs::write(&label_path, labels).unwrap();
    (image_path, label_path)
}

fn batch(dataset: &MnistIdx, indices: &[usize]) -> Result<(TensorData, TensorData)> {
    let pixels = dataset.rows * dataset.cols;
    Ok((
        TensorData::from_scalars(
            [indices.len(), pixels],
            DType::F32,
            indices.iter().flat_map(|&row| {
                (0..pixels).map(move |column| {
                    Scalar::F(dataset.images.scalar_at(row * pixels + column).as_f64() / 255.)
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
    model: &Classifier,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    dataset: &MnistIdx,
    indices: &[usize],
) -> Result<f64> {
    let (inputs, targets) = batch(dataset, indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [indices.len(), 28 * 28]);
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
    let value = cpu.execute(&graph, loss, &bindings)?.scalar_at(0).as_f64();
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
    Ok(value)
}

fn evaluate(model: &Classifier, dataset: &MnistIdx) -> Result<(f64, Vec<u8>)> {
    let indices = [0, 1, 2, 3];
    let (inputs, targets) = batch(dataset, &indices)?;
    let mut graph = Graph::new();
    let input = graph.input("input", [4, 28 * 28]);
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
fn local_idx_files_train_resume_and_evaluate_without_mutation() {
    let (images, labels) = write_fixture();
    let dataset = load_mnist_idx_files(&images, &labels).unwrap();
    assert_eq!(dataset.images.shape().dims(), &[4, 1, 28, 28]);
    let batches = BatchIter::new(4, 3, 29, true, false)
        .unwrap()
        .collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);
    assert_eq!(
        batches,
        BatchIter::new(4, 3, 29, true, false)
            .unwrap()
            .collect::<Vec<_>>()
    );

    let baseline = Classifier::new();
    let mut baseline_optimizer = baseline.optimizer();
    let mut baseline_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5).unwrap();
    let initial = evaluate(&baseline, &dataset).unwrap();
    let mut losses = Vec::new();
    for indices in batches.iter().cycle().take(8) {
        losses.push(
            step(
                &baseline,
                &mut baseline_optimizer,
                &mut baseline_scheduler,
                &dataset,
                indices,
            )
            .unwrap(),
        );
    }
    let expected = evaluate(&baseline, &dataset).unwrap();
    assert!(losses.last().unwrap() < losses.first().unwrap());
    assert!(expected.0 < initial.0);

    let source = Classifier::new();
    let mut source_optimizer = source.optimizer();
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5).unwrap();
    for indices in batches.iter().cycle().take(4) {
        step(
            &source,
            &mut source_optimizer,
            &mut source_scheduler,
            &dataset,
            indices,
        )
        .unwrap();
    }
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler).unwrap();
    let resumed = Classifier::new();
    let mut resumed_optimizer = resumed.optimizer();
    let mut resumed_scheduler = LearningRateScheduler::multi_step(vec![4], 0.5).unwrap();
    checkpoint
        .restore(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)
        .unwrap();
    for indices in batches.iter().cycle().skip(4).take(4) {
        step(
            &resumed,
            &mut resumed_optimizer,
            &mut resumed_scheduler,
            &dataset,
            indices,
        )
        .unwrap();
    }
    assert_eq!(
        baseline.state_dict().unwrap(),
        resumed.state_dict().unwrap()
    );
    assert_eq!(
        baseline_optimizer.state_dict().unwrap(),
        resumed_optimizer.state_dict().unwrap()
    );
    assert_eq!(
        baseline_scheduler.state_dict().unwrap(),
        resumed_scheduler.state_dict().unwrap()
    );
    assert_eq!(expected, evaluate(&resumed, &dataset).unwrap());
    let before = (
        resumed.state_dict().unwrap(),
        resumed_optimizer.state_dict().unwrap(),
        resumed_scheduler.state_dict().unwrap(),
    );
    assert_eq!(evaluate(&resumed, &dataset).unwrap(), expected);
    assert_eq!(before.0, resumed.state_dict().unwrap());
    assert_eq!(before.1, resumed_optimizer.state_dict().unwrap());
    assert_eq!(before.2, resumed_scheduler.state_dict().unwrap());
    std::fs::remove_dir_all(images.parent().unwrap()).unwrap();
}
