//! Public cross-process-style checkpoint acceptance.

use rustgrad::nn::{Module, StateKind};
use rustgrad::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    Backend, CpuBackend, DType, Graph, Parameter, PortableTrainingCheckpoint, Result, Scalar,
    Shape, TensorData,
};
use std::collections::BTreeMap;

struct TiedAffine {
    weight: Parameter,
    alias: Parameter,
    bias: Parameter,
    running: Parameter,
}

impl TiedAffine {
    fn new(seed: f64) -> Self {
        let weight = Parameter::new(
            TensorData::from_scalars(
                [2, 2],
                DType::F32,
                [seed, seed + 0.25, seed - 0.5, seed + 0.75]
                    .into_iter()
                    .map(Scalar::F),
            )
            .unwrap(),
            true,
        );
        Self {
            alias: weight.clone(),
            weight,
            bias: Parameter::new(
                TensorData::new([2], vec![seed as f32, -seed as f32]).unwrap(),
                true,
            ),
            running: Parameter::new(TensorData::new([1], vec![seed as f32 + 3.]).unwrap(), false),
        }
    }

    fn optimizer(&self) -> Optimizer {
        Optimizer::sgd(
            vec![
                ("bias".into(), self.bias.clone()),
                ("weight".into(), self.weight.clone()),
            ],
            SgdConfig {
                lr: 0.2,
                momentum: 0.8,
                dampening: 0.,
                nesterov: true,
                weight_decay: 0.01,
            },
        )
        .unwrap()
    }

    fn forward(&self, graph: &mut Graph, input: rustgrad::NodeId) -> Result<rustgrad::NodeId> {
        let weight = self.weight.bind(graph)?;
        let alias = self.alias.bind(graph)?;
        let bias = self.bias.bind(graph)?;
        let left = graph.matmul(input, weight)?;
        let right = graph.matmul(input, alias)?;
        let tied = graph.add(left, right)?;
        graph.add(tied, bias)
    }
}

impl Module for TiedAffine {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        let name = |field: &str| {
            if prefix.is_empty() {
                field.to_string()
            } else {
                format!("{prefix}.{field}")
            }
        };
        visitor(name("weight"), &self.weight, StateKind::Parameter);
        visitor(name("tied_weight"), &self.alias, StateKind::Parameter);
        visitor(name("bias"), &self.bias, StateKind::Parameter);
        visitor(name("running"), &self.running, StateKind::Buffer);
    }
}

fn train_step(
    model: &TiedAffine,
    optimizer: &mut Optimizer,
    scheduler: &mut LearningRateScheduler,
) {
    let gradients = BTreeMap::from([
        (
            "bias".into(),
            Gradient::for_parameter(&model.bias, TensorData::new([2], vec![0.2, -0.1]).unwrap())
                .unwrap(),
        ),
        (
            "weight".into(),
            Gradient::for_parameter(
                &model.weight,
                TensorData::new([2, 2], vec![0.1, -0.2, 0.3, -0.4]).unwrap(),
            )
            .unwrap(),
        ),
    ]);
    optimizer.step(&gradients).unwrap();
    scheduler.step(optimizer).unwrap();
}

fn inference(model: &TiedAffine) -> (TensorData, TensorData, TensorData) {
    let mut graph = Graph::new();
    let input = graph.input("x", Shape::new([2, 2]));
    let logits = model.forward(&mut graph, input).unwrap();
    let column_sum = graph.sum(logits, 0).unwrap();
    let loss = graph.sum(column_sum, 0).unwrap();
    let predictions = graph.argmax(logits, Some(-1), false).unwrap();
    let mut bindings = model.input_bindings(&graph).unwrap();
    bindings.insert(
        "x".into(),
        TensorData::new([2, 2], vec![1., 2., -1., 0.5]).unwrap(),
    );
    let cpu = CpuBackend;
    (
        cpu.execute(&graph, loss, &bindings).unwrap(),
        cpu.execute(&graph, logits, &bindings).unwrap(),
        cpu.execute(&graph, predictions, &bindings).unwrap(),
    )
}

#[test]
fn portable_checkpoint_restores_fresh_identities_and_continues_bit_exactly() {
    let uninterrupted = TiedAffine::new(0.4);
    let mut uninterrupted_optimizer = uninterrupted.optimizer();
    let mut uninterrupted_scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    for _ in 0..3 {
        train_step(
            &uninterrupted,
            &mut uninterrupted_optimizer,
            &mut uninterrupted_scheduler,
        );
    }
    uninterrupted
        .running
        .replace(TensorData::new([1], vec![6.]).unwrap())
        .unwrap();

    let source = TiedAffine::new(0.4);
    let mut source_optimizer = source.optimizer();
    let mut source_scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    for _ in 0..2 {
        train_step(&source, &mut source_optimizer, &mut source_scheduler);
    }
    source
        .running
        .replace(TensorData::new([1], vec![6.]).unwrap())
        .unwrap();
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler).unwrap();
    assert_eq!(
        checkpoint.as_bytes(),
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler)
            .unwrap()
            .as_bytes()
    );
    let decoded = PortableTrainingCheckpoint::from_bytes(checkpoint.clone().into_bytes()).unwrap();

    let restored = TiedAffine::new(9.0);
    assert_ne!(source.weight.id(), restored.weight.id());
    assert_eq!(restored.weight.id(), restored.alias.id());
    let mut restored_optimizer = restored.optimizer();
    let mut restored_scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    decoded
        .restore(&restored, &mut restored_optimizer, &mut restored_scheduler)
        .unwrap();
    assert_eq!(
        source.weight.version().unwrap(),
        restored.weight.version().unwrap()
    );
    assert_eq!(
        source.bias.version().unwrap(),
        restored.bias.version().unwrap()
    );
    assert_eq!(
        source.running.version().unwrap(),
        restored.running.version().unwrap()
    );
    assert_eq!(source.state_dict().unwrap(), restored.state_dict().unwrap());
    assert_eq!(
        source_optimizer.state_dict().unwrap(),
        restored_optimizer.state_dict().unwrap()
    );
    assert_eq!(
        source_scheduler.state_dict().unwrap(),
        restored_scheduler.state_dict().unwrap()
    );

    train_step(&restored, &mut restored_optimizer, &mut restored_scheduler);
    assert_eq!(
        uninterrupted.state_dict().unwrap(),
        restored.state_dict().unwrap()
    );
    for (name, expected) in uninterrupted.state_dict().unwrap().tensors() {
        assert_eq!(
            expected.to_le_bytes().unwrap(),
            restored.state_dict().unwrap().tensors()[name]
                .to_le_bytes()
                .unwrap(),
            "raw bytes for {name}"
        );
    }
    assert_eq!(
        uninterrupted.weight.version().unwrap(),
        restored.weight.version().unwrap()
    );
    assert_eq!(
        uninterrupted.bias.version().unwrap(),
        restored.bias.version().unwrap()
    );
    assert_eq!(
        uninterrupted.running.version().unwrap(),
        restored.running.version().unwrap()
    );
    assert_eq!(
        uninterrupted_optimizer.state_dict().unwrap(),
        restored_optimizer.state_dict().unwrap()
    );
    assert_eq!(
        uninterrupted_scheduler.state_dict().unwrap(),
        restored_scheduler.state_dict().unwrap()
    );
    let expected = inference(&uninterrupted);
    let actual = inference(&restored);
    assert_eq!(
        expected.0.to_le_bytes().unwrap(),
        actual.0.to_le_bytes().unwrap()
    );
    assert_eq!(
        expected.1.to_le_bytes().unwrap(),
        actual.1.to_le_bytes().unwrap()
    );
    assert_eq!(
        expected.2.to_le_bytes().unwrap(),
        actual.2.to_le_bytes().unwrap()
    );
}

#[test]
fn portable_checkpoint_rejects_structural_mismatch_before_mutation() {
    let source = TiedAffine::new(0.2);
    let source_optimizer = source.optimizer();
    let source_scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    let checkpoint =
        PortableTrainingCheckpoint::capture(&source, &source_optimizer, &source_scheduler).unwrap();

    struct Untied(TiedAffine);
    impl Module for Untied {
        fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
            self.0.visit(prefix, visitor);
        }
    }
    let mut untied = TiedAffine::new(8.0);
    untied.alias = Parameter::new(untied.weight.value().unwrap(), true);
    let target = Untied(untied);
    let mut optimizer = target.0.optimizer();
    let mut scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    let before_module = target.state_dict().unwrap();
    let before_optimizer = optimizer.state_dict().unwrap();
    let before_scheduler = scheduler.state_dict().unwrap();
    assert!(
        checkpoint
            .restore(&target, &mut optimizer, &mut scheduler)
            .is_err()
    );
    assert_eq!(target.state_dict().unwrap(), before_module);
    assert_eq!(optimizer.state_dict().unwrap(), before_optimizer);
    assert_eq!(scheduler.state_dict().unwrap(), before_scheduler);

    let compatible = TiedAffine::new(8.0);
    let mut wrong_optimizer = Optimizer::sgd(
        vec![
            ("bias".into(), compatible.bias.clone()),
            ("weight".into(), compatible.weight.clone()),
        ],
        SgdConfig {
            lr: 0.3,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let mut compatible_scheduler = LearningRateScheduler::multi_step(vec![1], 0.5).unwrap();
    let before = compatible.state_dict().unwrap();
    assert!(
        checkpoint
            .restore(&compatible, &mut wrong_optimizer, &mut compatible_scheduler)
            .is_err()
    );
    assert_eq!(compatible.state_dict().unwrap(), before);

    let compatible = TiedAffine::new(8.0);
    let mut compatible_optimizer = compatible.optimizer();
    let mut wrong_scheduler = LearningRateScheduler::multi_step(vec![2], 0.5).unwrap();
    let before_module = compatible.state_dict().unwrap();
    let before_optimizer = compatible_optimizer.state_dict().unwrap();
    let before_scheduler = wrong_scheduler.state_dict().unwrap();
    assert!(
        checkpoint
            .restore(&compatible, &mut compatible_optimizer, &mut wrong_scheduler)
            .is_err()
    );
    assert_eq!(compatible.state_dict().unwrap(), before_module);
    assert_eq!(compatible_optimizer.state_dict().unwrap(), before_optimizer);
    assert_eq!(wrong_scheduler.state_dict().unwrap(), before_scheduler);
}
