//! Example-owned two-block causal model, shared by training consumers.

use super::{BATCH, EMBEDDING, FEED_FORWARD, HEADS, TIME, VOCAB};
use rustgrad::nn::{Embedding, LayerNorm, Mode, StateKind};
use rustgrad::{
    Graph, Module, NodeId, Parameter, Result, Scalar, TensorData, TrainingDropoutProvider,
    TransformerBlock,
};

pub(super) struct CausalTransformer {
    pub(super) tokens: Embedding,
    pub(super) positions: Embedding,
    first: TransformerBlock,
    second: TransformerBlock,
    norm: LayerNorm,
}

impl CausalTransformer {
    pub(super) fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            tokens: Embedding::new_static(VOCAB, EMBEDDING, None, seed)?,
            positions: Embedding::new_static(TIME, EMBEDDING, None, seed.wrapping_add(1))?,
            first: TransformerBlock::new_static(
                EMBEDDING,
                HEADS,
                FEED_FORWARD,
                true,
                0.1,
                seed.wrapping_add(2),
            )?
            .with_causal_attention(true)
            .with_attention_dropout(0.1)?,
            second: TransformerBlock::new_static(
                EMBEDDING,
                HEADS,
                FEED_FORWARD,
                true,
                0.1,
                seed.wrapping_add(3),
            )?
            .with_causal_attention(true)
            .with_attention_dropout(0.1)?,
            norm: LayerNorm::new_static([EMBEDDING], 1e-5, true)?,
        })
    }

    fn positions(graph: &mut Graph) -> Result<NodeId> {
        let values = (0..BATCH).flat_map(|_| {
            (0..TIME).map(|position| {
                Scalar::I(i64::try_from(position).expect("fixed position fits in i64"))
            })
        });
        Ok(graph.constant(TensorData::from_scalars([BATCH, TIME], DType::I32, values)?))
    }

    pub(super) fn forward_training(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
        dropout: &mut dyn TrainingDropoutProvider,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = Self::positions(graph)?;
        let position_hidden = self.positions.forward(graph, positions)?;
        let hidden = graph.add(token_hidden, position_hidden)?;
        let hidden = self
            .first
            .forward_training_with_dropout_and_attention_mask(
                graph,
                hidden,
                attention_keep_mask,
                dropout,
            )?;
        let hidden = self
            .second
            .forward_training_with_dropout_and_attention_mask(
                graph,
                hidden,
                attention_keep_mask,
                dropout,
            )?;
        self.project(graph, hidden)
    }

    pub(super) fn forward_evaluation(
        &self,
        graph: &mut Graph,
        tokens: NodeId,
        attention_keep_mask: NodeId,
    ) -> Result<NodeId> {
        let token_hidden = self.tokens.forward(graph, tokens)?;
        let positions = Self::positions(graph)?;
        let position_hidden = self.positions.forward(graph, positions)?;
        let hidden = graph.add(token_hidden, position_hidden)?;
        let hidden = self
            .first
            .forward_mode_with_attention_mask(graph, hidden, attention_keep_mask, Mode::Eval)?
            .output;
        let hidden = self
            .second
            .forward_mode_with_attention_mask(graph, hidden, attention_keep_mask, Mode::Eval)?
            .output;
        self.project(graph, hidden)
    }

    fn project(&self, graph: &mut Graph, hidden: NodeId) -> Result<NodeId> {
        let hidden = self.norm.forward(graph, hidden)?;
        let tied_weight = self.tokens.weight.bind(graph)?;
        let tied_weight = graph.permute(tied_weight, [1, 0])?;
        graph.matmul(hidden, tied_weight)
    }
}

impl Module for CausalTransformer {
    fn visit(&self, prefix: &str, visitor: &mut dyn FnMut(String, &Parameter, StateKind)) {
        let child = |name: &str| {
            if prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{prefix}.{name}")
            }
        };
        self.tokens.visit(&child("tokens"), visitor);
        self.positions.visit(&child("positions"), visitor);
        self.first.visit(&child("first"), visitor);
        self.second.visit(&child("second"), visitor);
        self.norm.visit(&child("norm"), visitor);
        visitor(
            child("lm_head.weight"),
            &self.tokens.weight,
            StateKind::Parameter,
        );
    }
}
