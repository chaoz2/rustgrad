//! Embedding lookup module composition.

use super::{Module, ModuleForward, Parameter, StateKind, init::uniform, state::join};
use crate::{DType, Error, Graph, NodeId, Result, Scalar, Shape, TensorData};

pub struct Embedding {
    pub weight: Parameter,
    pub padding_idx: Option<usize>,
    embedding_dim: usize,
}
impl Embedding {
    /// Creates graph-independent embedding state for static module workflows.
    pub fn new_static(
        vocab: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        if padding_idx.is_some_and(|i| i >= vocab) {
            return Err(Error::InvalidIndex);
        }
        let bound = (6.0f32 / (vocab + embedding_dim) as f32).sqrt();
        Ok(Self {
            weight: Parameter::new(
                uniform(Shape::new([vocab, embedding_dim]), -bound, bound, seed)?,
                true,
            ),
            padding_idx,
            embedding_dim,
        })
    }

    /// Source-compatible graph-taking constructor for callers that still own
    /// setup graph construction explicitly.
    pub fn new(
        _graph: &mut Graph,
        vocab: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        seed: u64,
    ) -> Result<Self> {
        Self::new_static(vocab, embedding_dim, padding_idx, seed)
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
impl ModuleForward for Embedding {
    fn accepts_input_dtype(&self, dtype: DType) -> bool {
        dtype.is_integer()
    }

    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}
