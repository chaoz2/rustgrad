use super::training;
use crate::nn::TrainingDropoutProvider;
use crate::{DType, Graph, NodeId, Result, Scalar, Shape, TensorData};

/// Immutable two-word key for compiled Transformer residual dropout.
///
/// The first word occupies the low 32 bits of Threefry's packed U64 key and
/// the second occupies the high 32 bits. This is deliberately an explicit key,
/// not a `u64` seed with an implicit or lossy mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledDropoutKey(pub [u32; 2]);

impl CompiledDropoutKey {
    pub const fn words(self) -> [u32; 2] {
        self.0
    }

    const fn packed(self) -> u64 {
        self.0[0] as u64 | ((self.0[1] as u64) << 32)
    }
}

/// Fixed compiled residual-dropout policy for one AdamW Transformer capture.
///
/// This policy defines a workload-specific U64-block stream with interleaved
/// low/high U32 words and low-mantissa F32 conversion. It is intentionally not
/// sequence-compatible with [`crate::RandomStream`] or its uniform conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledDropoutConfig {
    key: CompiledDropoutKey,
}

impl CompiledDropoutConfig {
    pub const fn new(key: CompiledDropoutKey) -> Self {
        Self { key }
    }

    pub const fn key(self) -> CompiledDropoutKey {
        self.key
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompiledDropoutState {
    pub(super) config: CompiledDropoutConfig,
    pub(super) blocks_per_replay: u64,
}

pub(super) struct CompiledDropoutStream {
    counter: NodeId,
    key: CompiledDropoutKey,
    reserved_blocks: u64,
}

impl CompiledDropoutStream {
    pub(super) fn new(counter: NodeId, config: CompiledDropoutConfig) -> Self {
        Self {
            counter,
            key: config.key,
            reserved_blocks: 0,
        }
    }

    pub(super) fn finish(self, graph: &mut Graph) -> Result<(NodeId, CompiledDropoutState)> {
        if self.reserved_blocks == 0 {
            return Err(training(
                "compiled dropout configuration produced no active F32 draw",
            ));
        }
        let increment =
            graph.full_with_dtype(Shape::from([]), Scalar::U(self.reserved_blocks), DType::U64)?;
        let successor = graph.add(self.counter, increment)?;
        Ok((
            successor,
            CompiledDropoutState {
                config: CompiledDropoutConfig::new(self.key),
                blocks_per_replay: self.reserved_blocks,
            },
        ))
    }
}

pub(super) fn expected_dropout_counter(
    dropout: CompiledDropoutState,
    replay_step: u64,
) -> Result<u64> {
    replay_step
        .checked_mul(dropout.blocks_per_replay)
        .ok_or_else(|| training("compiled dropout block counter would overflow"))
}

impl TrainingDropoutProvider for CompiledDropoutStream {
    fn dropout(&mut self, graph: &mut Graph, input: NodeId, probability: f64) -> Result<NodeId> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(training("compiled dropout probability must be in [0, 1]"));
        }
        if graph.dtype(input)? != DType::F32 {
            return Err(training("compiled dropout requires F32 input"));
        }
        let shape = graph.shape(input)?.clone();
        let elements = shape.numel()?;
        if elements == 0 || probability == 0.0 {
            return Ok(input);
        }
        if probability == 1.0 {
            return graph.zeros_with_dtype(shape, DType::F32);
        }

        let blocks = elements
            .checked_add(1)
            .ok_or_else(|| training("compiled dropout block count overflow"))?
            / 2;
        let blocks =
            u64::try_from(blocks).map_err(|_| training("compiled dropout block count overflow"))?;
        let start = self.reserved_blocks;
        self.reserved_blocks = start
            .checked_add(blocks)
            .ok_or_else(|| training("compiled dropout reservation overflow"))?;
        let offsets = (start..self.reserved_blocks)
            .map(Scalar::U)
            .collect::<Vec<_>>();
        let offsets = graph.constant(TensorData::from_scalars(
            Shape::new([offsets.len()]),
            DType::U64,
            offsets,
        )?);
        let counters = graph.add(self.counter, offsets)?;
        let key =
            graph.full_with_dtype(Shape::from([]), Scalar::U(self.key.packed()), DType::U64)?;
        let blocks = graph.threefry(counters, key)?;
        let words = graph.bitcast(blocks, DType::U32)?;
        let words = graph.shrink(words, vec![(0, elements)])?;
        let words = graph.reshape(words, shape.clone())?;
        let mantissa = graph.bitwise_and_scalar(words, Scalar::U(0x007f_ffff))?;
        let bits = graph.bitwise_or_scalar(mantissa, Scalar::U(0x3f80_0000))?;
        let unit = graph.bitcast(bits, DType::F32)?;
        let one = graph.full_with_dtype(Shape::from([]), Scalar::F(1.0), DType::F32)?;
        let unit = graph.sub(unit, one)?;
        let threshold =
            graph.full_with_dtype(Shape::from([]), Scalar::F(probability), DType::F32)?;
        let keep = graph.ge(unit, threshold)?;
        let keep = graph.contiguous(keep)?;
        let zero = graph.full_with_dtype(Shape::from([]), Scalar::F(0.0), DType::F32)?;
        let masked = graph.select(keep, input, zero)?;
        let denominator =
            graph.full_with_dtype(Shape::from([]), Scalar::F(1.0 - probability), DType::F32)?;
        graph.div(masked, denominator)
    }
}
