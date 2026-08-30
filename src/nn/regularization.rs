//! Training-time regularization modules.

use super::{
    Mode, ModeForwardOutput, ModeModuleForward, Module, ModuleForward, Parameter,
    PendingModeEffects, StateKind,
};
use crate::{Error, Graph, NodeId, Result, TensorData};

fn apply_dropout(graph: &mut Graph, input: NodeId, probability: f64, seed: u64) -> Result<NodeId> {
    if probability == 0.0 {
        return Ok(input);
    }
    if probability == 1.0 {
        return graph.zeros_like(input, None);
    }
    let random = graph.rand_like(input, None, seed)?;
    let threshold = graph.constant(TensorData::scalar(probability as f32));
    let mask = graph.ge(random, threshold)?;
    let zero = graph.zeros_like(input, None)?;
    let kept = graph.select(mask, input, zero)?;
    let scale = graph.constant(TensorData::scalar((1.0 / (1.0 - probability)) as f32));
    graph.mul(kept, scale)
}

pub struct Dropout {
    pub probability: f64,
    pub training: bool,
    pub seed: u64,
}
impl Dropout {
    fn validate_probability(probability: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(Error::UnsupportedDropout {
                probability_bits: probability.to_bits(),
            });
        }
        Ok(())
    }

    pub fn new(probability: f64, training: bool, seed: u64) -> Result<Self> {
        Self::validate_probability(probability)?;
        Ok(Self {
            probability,
            training,
            seed,
        })
    }
    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::validate_probability(self.probability)?;
        if !self.training || self.probability == 0.0 {
            return Ok(input);
        }
        apply_dropout(graph, input, self.probability, self.seed)
    }
}
impl Module for Dropout {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}
impl ModuleForward for Dropout {
    fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId> {
        Self::forward(self, graph, input)
    }
}

/// Explicit-mode deterministic dropout for [`super::ModeSequential`].
///
/// Training uses the same fixed-seed inverted-dropout composition as
/// [`Dropout`]; evaluation returns the input node unchanged. This type has no
/// state or pending effects and deliberately does not implement
/// [`ModuleForward`], so callers cannot accidentally hide its selected mode.
pub struct ModeDropout {
    probability: f64,
    seed: u64,
}

impl ModeDropout {
    pub fn new(probability: f64, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(Error::UnsupportedDropout {
                probability_bits: probability.to_bits(),
            });
        }
        Ok(Self { probability, seed })
    }

    pub const fn probability(&self) -> f64 {
        self.probability
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

impl Module for ModeDropout {
    fn visit(&self, _: &str, _: &mut dyn FnMut(String, &Parameter, StateKind)) {}
}

impl ModeModuleForward for ModeDropout {
    fn forward_mode<'a>(
        &'a self,
        graph: &mut Graph,
        input: NodeId,
        mode: Mode,
    ) -> Result<ModeForwardOutput<'a>> {
        let output = match mode {
            Mode::Training => apply_dropout(graph, input, self.probability, self.seed)?,
            Mode::Eval => input,
        };
        Ok(ModeForwardOutput {
            output,
            pending: PendingModeEffects::empty(),
        })
    }
}
