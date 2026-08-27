//! Embedding lookup module composition.

use super::{Module, Parameter, StateKind, init::uniform, state::join};
use crate::{Error, Graph, NodeId, Result, Scalar, Shape, TensorData};

pub struct Embedding {
    pub weight: Parameter,
    pub padding_idx: Option<usize>,
    embedding_dim: usize,
}
impl Embedding {
    pub fn new(
        _graph: &mut Graph,
        vocab: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        if vocab == 0 || embedding_dim == 0 {
            return Err(Error::InvalidRandom {
                reason: "embedding vocabulary and dimension must be nonzero",
            });
        }
        if padding_idx.is_some_and(|i| i >= vocab) {
            return Err(Error::InvalidIndex);
        }
        let fan_sum = vocab
            .checked_add(embedding_dim)
            .ok_or_else(|| Error::ShapeOverflow(Shape::new([vocab, embedding_dim])))?;
        let bound = (6.0f32 / fan_sum as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(Shape::new([vocab, embedding_dim]), -bound, bound, seed)?,
                true,
            ),
            padding_idx,
            embedding_dim,
        })
    }
    pub fn forward(&self, graph: &mut Graph, index: NodeId) -> Result<NodeId> {
        if !graph.dtype(index)?.is_integer() {
            return Err(Error::InvalidIndexDType {
                op: "embedding",
                actual: graph.dtype(index)?,
            });
        }
        let mut dims = graph.shape(index)?.dims().to_vec();
        dims.push(1);
        let expanded = graph.reshape(index, Shape::new(dims.clone()))?;
        *dims.last_mut().expect("added dimension") = self.embedding_dim;
        let expanded = graph.expand(expanded, Shape::new(dims))?;
        let weight = self.weight.bind(graph)?;
        let output = graph.gather(weight, expanded, 0)?;
        if let Some(padding) = self.padding_idx {
            let pad = graph.constant(TensorData::scalar_with_dtype(
                Scalar::I(padding as i64),
                graph.dtype(index)?,
            ));
            let mask = graph.eq(index, pad)?;
            let mask = graph.reshape(
                mask,
                Shape::new({
                    let mut d = graph.shape(index)?.dims().to_vec();
                    d.push(1);
                    d
                }),
            )?;
            let mask = graph.expand(mask, graph.shape(output)?.clone())?;
            let zero = graph.zeros_like(output, None)?;
            graph.select(mask, zero, output)
        } else {
            Ok(output)
        }
    }
}
impl Module for Embedding {
    fn visit(&self, prefix: &str, v: &mut dyn FnMut(String, &Parameter, StateKind)) {
        v(join(prefix, "weight"), &self.weight, StateKind::Parameter)
    }
}
