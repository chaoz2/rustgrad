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
    use crate::optim::{Gradient, LearningRateScheduler, Optimizer, SgdConfig};
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
        let parameters: Vec<(String, crate::Parameter)> = vec![
            ("first.weight".into(), first.weight.clone()),
            ("first.bias".into(), first.bias.clone().unwrap()),
            ("second.weight".into(), second.weight.clone()),
            ("second.bias".into(), second.bias.clone().unwrap()),
        ];
        let grad_nodes = parameters
            .iter()
            .map(|(name, parameter)| {
                (
                    name.clone(),
                    graph.grad(loss, parameter.node(&graph).unwrap()).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
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
            let mut bindings = first.input_bindings().unwrap();
            bindings.extend(second.input_bindings().unwrap());
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
