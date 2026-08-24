//! Deterministic local dataset readers; network transport is deliberately absent.
use crate::{DType, Error, Result, Shape, TensorData};

fn bad(reason: impl Into<String>) -> Error {
    Error::Dataset {
        reason: reason.into(),
    }
}
fn be32(bytes: &[u8]) -> Result<usize> {
    usize::try_from(u32::from_be_bytes(
        bytes.try_into().map_err(|_| bad("truncated IDX header"))?,
    ))
    .map_err(|_| bad("IDX count overflow"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct MnistIdx {
    pub images: TensorData,
    pub labels: TensorData,
    pub rows: usize,
    pub cols: usize,
}
impl MnistIdx {
    pub fn normalized_f32(&self) -> Result<TensorData> {
        TensorData::from_scalars(
            self.images.shape().clone(),
            DType::F32,
            (0..self.images.len())
                .map(|i| crate::Scalar::F(self.images.scalar_at(i).as_f64() / 255.)),
        )
    }
}
pub fn parse_mnist_idx(images: &[u8], labels: &[u8]) -> Result<MnistIdx> {
    if images.len() < 16 || labels.len() < 8 {
        return Err(bad("truncated IDX header"));
    }
    if be32(&images[..4])? != 2051 || be32(&labels[..4])? != 2049 {
        return Err(bad("invalid IDX magic"));
    }
    let n = be32(&images[4..8])?;
    let rows = be32(&images[8..12])?;
    let cols = be32(&images[12..16])?;
    let ln = be32(&labels[4..8])?;
    if n != ln {
        return Err(bad("IDX image/label counts differ"));
    }
    let pixels = n
        .checked_mul(rows)
        .and_then(|x| x.checked_mul(cols))
        .ok_or_else(|| bad("IDX shape overflow"))?;
    if images.len()
        != 16usize
            .checked_add(pixels)
            .ok_or_else(|| bad("IDX length overflow"))?
        || labels.len()
            != 8usize
                .checked_add(n)
                .ok_or_else(|| bad("IDX trailing or truncated data"))?
    {
        return Err(bad("IDX payload length mismatch"));
    }
    Ok(MnistIdx {
        images: TensorData::from_le_bytes(
            Shape::new([n, 1, rows, cols]),
            DType::U8,
            &images[16..],
        )?,
        labels: TensorData::from_le_bytes([n], DType::U8, &labels[8..])?,
        rows,
        cols,
    })
}

#[derive(Clone, Debug)]
pub struct BatchIter {
    order: Vec<usize>,
    at: usize,
    batch: usize,
    drop_last: bool,
}
impl BatchIter {
    pub fn new(
        len: usize,
        batch: usize,
        seed: u64,
        shuffle: bool,
        drop_last: bool,
    ) -> Result<Self> {
        if batch == 0 {
            return Err(bad("batch size must be nonzero"));
        };
        let mut order: Vec<usize> = (0..len).collect();
        if shuffle {
            for i in (1..len).rev() {
                let mut x = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                order.swap(i, (x as usize) % (i + 1));
            }
        }
        Ok(Self {
            order,
            at: 0,
            batch,
            drop_last,
        })
    }
}
impl Iterator for BatchIter {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.order.len() {
            return None;
        }
        let end = (self.at + self.batch).min(self.order.len());
        if self.drop_last && end - self.at < self.batch {
            self.at = self.order.len();
            return None;
        }
        let out = self.order[self.at..end].to_vec();
        self.at = end;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig, TrainingCheckpoint};
    use crate::{Backend, CpuBackend, Graph, LossOptions, Module, Reduction, cross_entropy};
    use std::collections::BTreeMap;
    #[test]
    fn idx_and_batches_are_exact_and_seeded() {
        let mut i = vec![];
        i.extend_from_slice(&2051u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&2u32.to_be_bytes());
        i.extend_from_slice(&[0, 255, 1, 2, 3, 4, 5, 6]);
        let mut l = vec![];
        l.extend_from_slice(&2049u32.to_be_bytes());
        l.extend_from_slice(&2u32.to_be_bytes());
        l.extend_from_slice(&[1, 9]);
        let d = parse_mnist_idx(&i, &l).unwrap();
        assert_eq!(d.images.shape().dims(), &[2, 1, 2, 2]);
        assert_eq!(d.normalized_f32().unwrap().values()[1], 1.);
        assert_eq!(
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>(),
            BatchIter::new(5, 2, 7, true, false)
                .unwrap()
                .collect::<Vec<_>>()
        );
        assert_eq!(BatchIter::new(5, 2, 0, false, true).unwrap().count(), 2);
        assert!(parse_mnist_idx(&i[..16], &l).is_err());
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
        let parameters: Vec<(String, crate::Parameter)> = vec![
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
                        .map(|i| crate::Scalar::F(dataset.images.scalar_at(i).as_f64() / 255.)),
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

        fn forward(&self, graph: &mut Graph, input: crate::NodeId) -> crate::Result<crate::NodeId> {
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

        fn named_parameters(&self) -> Vec<(String, &crate::Parameter)> {
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
            visitor: &mut dyn FnMut(String, &crate::Parameter, crate::nn::StateKind),
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
                .map(|i| crate::Scalar::F(dataset.images.scalar_at(i).as_f64() / 255.)),
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
                    Gradient::for_parameter(
                        parameter,
                        cpu.execute(&graph, node, &bindings).unwrap(),
                    )
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
                crate::Op::Input { name: input_name }
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
            TrainingCheckpoint::capture(&resumed, &midpoint_optimizer, &midpoint_scheduler)
                .unwrap();
        let (serialized_module, metadata) =
            crate::load_safetensors(checkpoint.module_safetensors()).unwrap();
        assert!(metadata.is_empty());
        assert_eq!(
            crate::nn::StateDict::from(serialized_module),
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
    ) -> Vec<(String, &'a crate::Parameter)> {
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
    ) -> &'a crate::Parameter {
        match name {
            "first.weight" => &first.weight,
            "first.bias" => first.bias.as_ref().unwrap(),
            "second.weight" => &second.weight,
            "second.bias" => second.bias.as_ref().unwrap(),
            _ => unreachable!(),
        }
    }
}
