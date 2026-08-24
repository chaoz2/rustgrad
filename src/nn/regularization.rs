//! Training-time regularization modules.

use super::{Module, Parameter, StateKind};
use crate::{Error, Graph, NodeId, Result, TensorData};

pub struct Dropout {
    pub probability: f64,
    pub training: bool,
    pub seed: u64,
}
impl Dropout {
    pub fn new(probability: f64, training: bool, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(Error::UnsupportedDropout {
                probability_bits: probability.to_bits(),
            });
        }
        Ok(Self {
            probability,
            training,
            seed,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        if !self.training || self.probability == 0.0 {
            return Ok(input);
        }
        if self.probability == 1.0 {
            return graph.zeros_like(input, None);
        }
        let random = graph.rand_like(input, None, self.seed)?;
        let threshold = graph.constant(TensorData::scalar(self.probability as f32));
        let mask = graph.ge(random, threshold)?;
        let zero = graph.zeros_like(input, None)?;
        let kept = graph.select(mask, input, zero)?;
        let scale = graph.constant(TensorData::scalar((1.0 / (1.0 - self.probability)) as f32));
        graph.mul(kept, scale)
    }
}
impl Module for Dropout {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}
