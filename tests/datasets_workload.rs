//! Public dataset and training acceptance workloads.

use rustgrad::nn::{Conv2d, Linear, MaxPool2d};
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, BatchIter, Conv2dOptions, CpuBackend, DType, Graph, LossOptions, MnistIdx, Module,
    Pool2dOptions, Reduction, Scalar, Shape, TensorData, TrainingCheckpoint, cross_entropy,
    parse_cifar10, parse_mnist_idx,
};
use std::collections::BTreeMap;

fn cifar_record(label: u8, red: u8, green: u8, blue: u8) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(3073);
    bytes.push(label);
    bytes.extend(std::iter::repeat_n(red, 1024));
    bytes.extend(std::iter::repeat_n(green, 1024));
    bytes.extend(std::iter::repeat_n(blue, 1024));
    bytes
}

#[test]
fn cifar_parser_composes_with_conv_pool_and_linear() {
    let dataset = parse_cifar10(&cifar_record(4, 51, 102, 153), 1).unwrap();
    let input = dataset.normalized_f32([0.; 3], [1.; 3]).unwrap();

    let mut graph = Graph::new();
    let x = graph.input("x", [1, 3, 32, 32]);
    let conv = Conv2d::new(&mut graph, 3, 1, [1, 1], Conv2dOptions::default(), false, 9).unwrap();
    conv.weight
        .replace(TensorData::from_scalars([1, 3, 1, 1], DType::F32, [Scalar::F(1.); 3]).unwrap())
        .unwrap();
    let convolved = conv.forward(&mut graph, x).unwrap();
    let pooled = MaxPool2d::new(Pool2dOptions::default())
        .forward(&mut graph, convolved)
        .unwrap();
    let flattened = graph.reshape(pooled, [1, 256]).unwrap();
    let linear = Linear::new(&mut graph, 256, 2, false, 10).unwrap();
    let mut weights = vec![0.; 512];
    weights[..256].fill(1.);
    linear
        .weight
        .replace(TensorData::new([2, 256], weights).unwrap())
        .unwrap();
    let output = linear.forward(&mut graph, flattened).unwrap();
    let mut bindings = conv.input_bindings(&graph).unwrap();
    bindings.extend(linear.input_bindings(&graph).unwrap());
    bindings.insert("x".into(), input);
    let actual = CpuBackend.execute(&graph, output, &bindings).unwrap();

    assert_eq!(dataset.labels.to_le_bytes().unwrap(), vec![4]);
    assert_eq!(actual.shape().dims(), &[1, 2]);
    assert!((actual.values()[0] - 307.2).abs() < 1e-3);
    assert_eq!(actual.values()[1], 0.);
}
#[test]
fn fixed_synthetic_idx_mlp_rebuilds_bindings_and_decreases_loss() {
    let mut image_bytes = Vec::new();
    image_bytes.extend_from_slice(&2051u32.to_be_bytes());
    image_bytes.extend_from_slice(&4u32.to_be_bytes());
    image_bytes.extend_from_slice(&1u32.to_be_bytes());
    image_bytes.extend_from_slice(&4u32.to_be_bytes());
    image_bytes.extend_from_slice(&[255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255]);
    let mut label_bytes = Vec::new();
    label_bytes.extend_from_slice(&2049u32.to_be_bytes());
    label_bytes.extend_from_slice(&4u32.to_be_bytes());
    label_bytes.extend_from_slice(&[0, 1, 0, 1]);
    let dataset = parse_mnist_idx(&image_bytes, &label_bytes).unwrap();
    assert_eq!(
        BatchIter::new(4, 2, 17, true, false)
            .unwrap()
            .collect::<Vec<_>>(),
        BatchIter::new(4, 2, 17, true, false)
            .unwrap()
            .collect::<Vec<_>>()
    );

    let mut graph = Graph::new();
    let first = Linear::new(&mut graph, 4, 4, true, 3).unwrap();
    let second = Linear::new(&mut graph, 4, 2, true, 4).unwrap();
    let parameters: Vec<(String, rustgrad::Parameter)> = vec![
        ("first.weight".into(), first.weight.clone()),
        ("first.bias".into(), first.bias.clone().unwrap()),
        ("second.weight".into(), second.weight.clone()),
        ("second.bias".into(), second.bias.clone().unwrap()),
    ];
    let mut optimizer = Optimizer::sgd(
        parameters,
        SgdConfig {
            lr: 0.4,
            momentum: 0.,
            dampening: 0.,
            nesterov: false,
            weight_decay: 0.,
        },
    )
    .unwrap();
    let mut scheduler = LearningRateScheduler::multi_step(vec![6], 0.5).unwrap();
    let cpu = CpuBackend;
    let mut losses = Vec::new();
    for step in 0..12 {
        let mut graph = Graph::new();
        let x = graph.input("x", [4, 4]);
        let target = graph.input_dtype("target", [4], DType::U8);
        let first_output = first.forward(&mut graph, x).unwrap();
        let hidden = graph.relu(first_output).unwrap();
        let logits = second.forward(&mut graph, hidden).unwrap();
        let loss = cross_entropy(
            &mut graph,
            logits,
            target,
            LossOptions {
                reduction: Reduction::Mean,
                ..LossOptions::default()
            },
        )
        .unwrap();
        let grad_nodes = parameters_for_test_names(&first, &second)
            .into_iter()
            .map(|(name, parameter)| {
                (
                    name,
                    graph.grad(loss, parameter.node(&graph).unwrap()).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut bindings = first.input_bindings(&graph).unwrap();
        bindings.extend(second.input_bindings(&graph).unwrap());
        bindings.insert(
            "x".into(),
            TensorData::from_scalars(
                Shape::new([4, 4]),
                DType::F32,
                (0..dataset.images.len())
                    .map(|i| rustgrad::Scalar::F(dataset.images.scalar_at(i).as_f64() / 255.)),
            )
            .unwrap(),
        );
        bindings.insert("target".into(), dataset.labels.clone());
        losses.push(
            cpu.execute(&graph, loss, &bindings)
                .unwrap()
                .scalar_at(0)
                .as_f64(),
        );
        let gradients = grad_nodes
            .iter()
            .map(|(name, node)| {
                (
                    name.clone(),
                    Gradient::for_parameter(
                        parameters_for_test(&first, &second, name),
                        cpu.execute(&graph, *node, &bindings).unwrap(),
                    )
                    .unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        optimizer.step(&gradients).unwrap();
        scheduler.step(&mut optimizer).unwrap();
        assert_eq!(optimizer.step_count(), (step + 1) as u64);
    }
    assert!(losses.last().unwrap() < losses.first().unwrap());
}

struct SyntheticMlp {
    first: Linear,
    second: Linear,
}

impl SyntheticMlp {
    fn new() -> Self {
        let mut construction_graph = Graph::new();
        Self {
            first: Linear::new(&mut construction_graph, 4, 4, true, 3).unwrap(),
            second: Linear::new(&mut construction_graph, 4, 2, true, 4).unwrap(),
        }
    }

    fn forward(
        &self,
        graph: &mut Graph,
        input: rustgrad::NodeId,
    ) -> rustgrad::Result<rustgrad::NodeId> {
        let hidden = self.first.forward(graph, input)?;
        let hidden = graph.relu(hidden)?;
        self.second.forward(graph, hidden)
    }

    fn optimizer(&self) -> Optimizer {
        Optimizer::sgd(
            vec![
                ("first.weight".into(), self.first.weight.clone()),
                ("first.bias".into(), self.first.bias.clone().unwrap()),
                ("second.weight".into(), self.second.weight.clone()),
                ("second.bias".into(), self.second.bias.clone().unwrap()),
            ],
            SgdConfig {
                lr: 0.4,
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
                weight_decay: 0.,
            },
        )
        .unwrap()
    }

    fn named_parameters(&self) -> Vec<(String, &rustgrad::Parameter)> {
        parameters_for_test_names(&self.first, &self.second)
    }

    fn versions(&self) -> BTreeMap<String, u64> {
        self.named_parameters()
            .into_iter()
            .map(|(name, parameter)| (name, parameter.version().unwrap()))
            .collect()
    }
}

impl Module for SyntheticMlp {
    fn visit(
        &self,
        prefix: &str,
        visitor: &mut dyn FnMut(String, &rustgrad::Parameter, rustgrad::nn::StateKind),
    ) {
        let child = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        self.first.visit(&child("first"), visitor);
        self.second.visit(&child("second"), visitor);
    }
}

fn synthetic_dataset() -> MnistIdx {
    let mut images = Vec::new();
    images.extend_from_slice(&2051u32.to_be_bytes());
    images.extend_from_slice(&4u32.to_be_bytes());
    images.extend_from_slice(&1u32.to_be_bytes());
    images.extend_from_slice(&4u32.to_be_bytes());
    images.extend_from_slice(&[255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255]);
    let mut labels = Vec::new();
    labels.extend_from_slice(&2049u32.to_be_bytes());
    labels.extend_from_slice(&4u32.to_be_bytes());
    labels.extend_from_slice(&[0, 1, 0, 1]);
    parse_mnist_idx(&images, &labels).unwrap()
}

fn synthetic_inputs(dataset: &MnistIdx) -> TensorData {
    TensorData::from_scalars(
        Shape::new([4, 4]),
        DType::F32,
        (0..dataset.images.len())
            .map(|i| rustgrad::Scalar::F(dataset.images.scalar_at(i).as_f64() / 255.)),
    )
    .unwrap()
}

fn train_mlp_step(
    model: &SyntheticMlp,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
    dataset: &MnistIdx,
) -> f64 {
    let mut graph = Graph::new();
    let x = graph.input("x", [4, 4]);
    let target = graph.input_dtype("target", [4], DType::U8);
    let logits = model.forward(&mut graph, x).unwrap();
    let loss = cross_entropy(
        &mut graph,
        logits,
        target,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )
    .unwrap();
    let grad_nodes = model
        .named_parameters()
        .into_iter()
        .map(|(name, parameter)| {
            (
                name,
                graph.grad(loss, parameter.node(&graph).unwrap()).unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert("x".into(), synthetic_inputs(dataset));
    bindings.insert("target".into(), dataset.labels.clone());
    let cpu = CpuBackend;
    let value = cpu
        .execute(&graph, loss, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let gradients = grad_nodes
        .into_iter()
        .map(|(name, node)| {
            let parameter = model
                .named_parameters()
                .into_iter()
                .find(|(candidate, _)| candidate == &name)
                .unwrap()
                .1;
            (
                name,
                Gradient::for_parameter(parameter, cpu.execute(&graph, node, &bindings).unwrap())
                    .unwrap(),
            )
        })
        .collect();
    optimizer.step(&gradients).unwrap();
    scheduler.step(optimizer).unwrap();
    value
}

fn infer_mlp(
    model: &SyntheticMlp,
    dataset: &MnistIdx,
) -> (TensorData, TensorData, TensorData, BTreeMap<String, u64>) {
    let mut graph = Graph::new();
    let x = graph.input("x", [4, 4]);
    let target = graph.input_dtype("target", [4], DType::U8);
    let logits = model.forward(&mut graph, x).unwrap();
    let predictions = graph.argmax(logits, Some(-1), false).unwrap();
    let loss = cross_entropy(
        &mut graph,
        logits,
        target,
        LossOptions {
            reduction: Reduction::Mean,
            ..LossOptions::default()
        },
    )
    .unwrap();
    let versions = model.versions();
    for (name, parameter) in model.named_parameters() {
        let node = parameter.node(&graph).unwrap();
        assert!(matches!(
            graph.op(node).unwrap(),
            rustgrad::Op::Input { name: input_name }
                if input_name.ends_with(&format!("_v{}", versions[&name]))
        ));
    }
    assert_eq!(graph.parameter_bindings().len(), 4);
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert("x".into(), synthetic_inputs(dataset));
    bindings.insert("target".into(), dataset.labels.clone());
    let cpu = CpuBackend;
    (
        cpu.execute(&graph, loss, &bindings).unwrap(),
        cpu.execute(&graph, logits, &bindings).unwrap(),
        cpu.execute(&graph, predictions, &bindings).unwrap(),
        versions,
    )
}

#[test]
fn synthetic_idx_mlp_checkpoint_resume_is_bit_exact() {
    let dataset = synthetic_dataset();
    let baseline = SyntheticMlp::new();
    let mut baseline_optimizer = baseline.optimizer();
    let mut baseline_scheduler = LearningRateScheduler::multi_step(vec![6], 0.5).unwrap();
    for _ in 0..12 {
        train_mlp_step(
            &baseline,
            &mut baseline_optimizer,
            &mut baseline_scheduler,
            &dataset,
        );
    }

    let resumed = SyntheticMlp::new();
    let mut midpoint_optimizer = resumed.optimizer();
    let mut midpoint_scheduler = LearningRateScheduler::multi_step(vec![6], 0.5).unwrap();
    for _ in 0..6 {
        train_mlp_step(
            &resumed,
            &mut midpoint_optimizer,
            &mut midpoint_scheduler,
            &dataset,
        );
    }
    let checkpoint =
        TrainingCheckpoint::capture(&resumed, &midpoint_optimizer, &midpoint_scheduler).unwrap();
    let (serialized_module, metadata) =
        rustgrad::load_safetensors(checkpoint.module_safetensors()).unwrap();
    assert!(metadata.is_empty());
    assert_eq!(
        rustgrad::nn::StateDict::from(serialized_module),
        resumed.state_dict().unwrap()
    );
    assert_eq!(checkpoint.parameter_versions(), resumed.versions());

    let mut resumed_optimizer = resumed.optimizer();
    let mut resumed_scheduler = LearningRateScheduler::multi_step(vec![6], 0.5).unwrap();
    checkpoint
        .resume(&resumed, &mut resumed_optimizer, &mut resumed_scheduler)
        .unwrap();
    assert_eq!(
        resumed_optimizer.state_dict().unwrap(),
        checkpoint.optimizer_state().clone()
    );
    assert_eq!(
        resumed_scheduler.state_dict().unwrap(),
        checkpoint.scheduler_state().clone()
    );
    for _ in 6..12 {
        train_mlp_step(
            &resumed,
            &mut resumed_optimizer,
            &mut resumed_scheduler,
            &dataset,
        );
    }

    let baseline_state = baseline.state_dict().unwrap();
    let resumed_state = resumed.state_dict().unwrap();
    assert_eq!(baseline_state, resumed_state);
    for name in baseline_state.tensors().keys() {
        assert_eq!(
            baseline_state.tensors()[name].to_le_bytes().unwrap(),
            resumed_state.tensors()[name].to_le_bytes().unwrap()
        );
    }
    assert_eq!(
        baseline_optimizer.state_dict().unwrap(),
        resumed_optimizer.state_dict().unwrap()
    );
    assert_eq!(baseline_optimizer.step_count(), 12);
    assert_eq!(resumed_optimizer.step_count(), 12);
    assert_eq!(baseline_optimizer.learning_rates(), &[0.2]);
    assert_eq!(resumed_optimizer.learning_rates(), &[0.2]);
    assert_eq!(baseline_scheduler.epoch(), 12);
    assert_eq!(resumed_scheduler.epoch(), 12);
    assert_eq!(
        baseline_scheduler.state_dict().unwrap(),
        resumed_scheduler.state_dict().unwrap()
    );

    let baseline_inference = infer_mlp(&baseline, &dataset);
    let resumed_inference = infer_mlp(&resumed, &dataset);
    assert_eq!(baseline_inference, resumed_inference);
    assert_eq!(
        baseline_inference.0.to_le_bytes().unwrap(),
        resumed_inference.0.to_le_bytes().unwrap()
    );
    assert_eq!(
        baseline_inference.1.to_le_bytes().unwrap(),
        resumed_inference.1.to_le_bytes().unwrap()
    );
    assert_eq!(
        baseline_inference.2.to_le_bytes().unwrap(),
        resumed_inference.2.to_le_bytes().unwrap()
    );
}

fn parameters_for_test_names<'a>(
    first: &'a Linear,
    second: &'a Linear,
) -> Vec<(String, &'a rustgrad::Parameter)> {
    vec![
        ("first.weight".into(), &first.weight),
        ("first.bias".into(), first.bias.as_ref().unwrap()),
        ("second.weight".into(), &second.weight),
        ("second.bias".into(), second.bias.as_ref().unwrap()),
    ]
}

fn parameters_for_test<'a>(
    first: &'a Linear,
    second: &'a Linear,
    name: &str,
) -> &'a rustgrad::Parameter {
    match name {
        "first.weight" => &first.weight,
        "first.bias" => first.bias.as_ref().unwrap(),
        "second.weight" => &second.weight,
        "second.bias" => second.bias.as_ref().unwrap(),
        _ => unreachable!(),
    }
}
