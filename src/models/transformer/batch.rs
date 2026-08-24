use super::{
    LlamaModel, LlamaModelConfig, LlamaModelError, OUTPUT_NORM, TOKEN_EMBEDDING,
    layer::{append_dense_batch_layer, batch_embedding, linear, rms_norm},
};
use crate::{Backend, CpuBackend, DType, Graph, NodeId, Scalar, TensorData};
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct BatchLayerCache {
    keys: TensorData,
    values: TensorData,
}

/// Fixed-shape padded batch plan with independent row lengths and positions.
#[derive(Debug)]
pub struct LlamaBatchPlan {
    graph: Graph,
    bindings: HashMap<String, TensorData>,
    logits: NodeId,
    cache_nodes: Vec<(NodeId, NodeId)>,
    chunk_lengths: Vec<usize>,
    next_lengths: Vec<usize>,
}

impl LlamaBatchPlan {
    /// Returns the typed padded batch graph before execution.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Returns the padded `[batch, chunk, vocabulary]` logits node.
    pub const fn logits_node(&self) -> NodeId {
        self.logits
    }

    /// Executes the active logits for each row through the CPU semantic oracle.
    pub fn execute(&self) -> Result<Vec<TensorData>, LlamaModelError> {
        Ok(self.execute_all()?.logits)
    }

    fn execute_all(&self) -> Result<BatchOutput, LlamaModelError> {
        let backend = CpuBackend;
        let padded = backend.execute(&self.graph, self.logits, &self.bindings)?;
        let shape = padded.shape().dims();
        let (batch, padded_sequence, vocab) = (shape[0], shape[1], shape[2]);
        let mut logits = Vec::with_capacity(batch);
        for (row, &length) in self.chunk_lengths.iter().enumerate() {
            let start = row * padded_sequence * vocab;
            logits.push(TensorData::new(
                [length, vocab],
                padded.values()[start..start + length * vocab].to_vec(),
            )?);
        }
        let mut layers = Vec::with_capacity(self.cache_nodes.len());
        for &(keys, values) in &self.cache_nodes {
            layers.push(BatchLayerCache {
                keys: backend.execute(&self.graph, keys, &self.bindings)?,
                values: backend.execute(&self.graph, values, &self.bindings)?,
            });
        }
        Ok(BatchOutput {
            logits,
            layers,
            lengths: self.next_lengths.clone(),
        })
    }
}

struct BatchOutput {
    logits: Vec<TensorData>,
    layers: Vec<BatchLayerCache>,
    lengths: Vec<usize>,
}

/// Transactional fixed-size batch cache with independent row lengths.
#[derive(Clone, Debug)]
pub struct LlamaBatchCache {
    config: LlamaModelConfig,
    batch_size: usize,
    lengths: Vec<usize>,
    layers: Option<Vec<BatchLayerCache>>,
}

impl LlamaBatchCache {
    /// Creates an empty cache for a nonzero fixed batch size.
    pub fn new(config: LlamaModelConfig, batch_size: usize) -> Result<Self, LlamaModelError> {
        if batch_size == 0 {
            return Err(LlamaModelError::EmptyBatch);
        }
        Ok(Self {
            config,
            batch_size,
            lengths: vec![0; batch_size],
            layers: None,
        })
    }

    /// Returns the committed prefix length of every batch row.
    pub fn lengths(&self) -> &[usize] {
        &self.lengths
    }

    /// Drops every layer cache and resets all row lengths to zero.
    pub fn clear(&mut self) {
        self.lengths.fill(0);
        self.layers = None;
    }

    /// Executes one right-padded row chunk and commits all rows/layers only
    /// after every logits/key/value output succeeds.
    pub fn forward(
        &mut self,
        model: &LlamaModel,
        chunks: &[Vec<u32>],
    ) -> Result<Vec<TensorData>, LlamaModelError> {
        if model.config() != &self.config {
            return Err(LlamaModelError::CacheConfigMismatch);
        }
        if chunks.len() != self.batch_size {
            return Err(LlamaModelError::BatchSize {
                expected: self.batch_size,
                actual: chunks.len(),
            });
        }
        let plan = model.plan_batch_with_past(chunks, &self.lengths, self.layers.as_deref())?;
        let output = plan.execute_all()?;
        self.layers = Some(output.layers);
        self.lengths = output.lengths;
        Ok(output.logits)
    }
}

impl LlamaModel {
    /// Builds one real padded batch graph for independent nonempty sequences.
    pub fn plan_batch(&self, sequences: &[Vec<u32>]) -> Result<LlamaBatchPlan, LlamaModelError> {
        let lengths = vec![0; sequences.len()];
        self.plan_batch_with_past(sequences, &lengths, None)
    }

    /// Executes independent sequences together and returns unpadded logits.
    pub fn forward_batch(
        &self,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<TensorData>, LlamaModelError> {
        self.plan_batch(sequences)?.execute()
    }

    fn plan_batch_with_past(
        &self,
        chunks: &[Vec<u32>],
        starts: &[usize],
        past: Option<&[BatchLayerCache]>,
    ) -> Result<LlamaBatchPlan, LlamaModelError> {
        if chunks.is_empty() {
            return Err(LlamaModelError::EmptyBatch);
        }
        if chunks.len() != starts.len() {
            return Err(LlamaModelError::BatchSize {
                expected: chunks.len(),
                actual: starts.len(),
            });
        }
        if chunks.iter().all(Vec::is_empty) {
            return Err(LlamaModelError::EmptyBatchStep);
        }
        let config = self.config();
        let schema = config.schema();
        if config.max_context() > i64::MAX as usize {
            return Err(LlamaModelError::InvalidConfig {
                field: "context_length",
            });
        }
        for (row, (chunk, &start)) in chunks.iter().zip(starts).enumerate() {
            let requested = start
                .checked_add(chunk.len())
                .ok_or(LlamaModelError::ContextOverflow)?;
            if requested > config.max_context() {
                return Err(LlamaModelError::BatchContextLength {
                    row,
                    requested,
                    maximum: config.max_context(),
                });
            }
            for &token in chunk {
                if usize::try_from(token).map_or(true, |token| token >= schema.vocab_size()) {
                    return Err(LlamaModelError::BatchTokenOutOfRange {
                        row,
                        token,
                        vocab_size: schema.vocab_size(),
                    });
                }
            }
        }
        validate_past(self, chunks.len(), starts, past)?;
        let sequence = chunks.iter().map(Vec::len).max().unwrap_or(0);
        let batch = chunks.len();
        let next_lengths = starts
            .iter()
            .zip(chunks)
            .map(|(start, chunk)| start + chunk.len())
            .collect::<Vec<_>>();
        let mut graph = Graph::new();
        let token_node = graph.input_dtype_requires_grad(
            "llama.batch.tokens",
            [batch, sequence],
            DType::I64,
            false,
        );
        let mut bindings = HashMap::from([(
            "llama.batch.tokens".to_owned(),
            TensorData::from_scalars(
                [batch, sequence],
                DType::I64,
                chunks.iter().flat_map(|chunk| {
                    (0..sequence).map(move |column| {
                        Scalar::I(i64::from(chunk.get(column).copied().unwrap_or(0)))
                    })
                }),
            )?,
        )]);
        let embedding_weight = graph.constant(self.state_map()[TOKEN_EMBEDDING].clone());
        let mut x = batch_embedding(
            &mut graph,
            token_node,
            embedding_weight,
            batch,
            sequence,
            schema.vocab_size(),
            schema.embedding_dim(),
        )?;
        let cache_shape = [
            batch,
            schema.kv_heads(),
            config.max_context(),
            schema.head_dim(),
        ];
        let mut cache_nodes = Vec::with_capacity(config.layer_count());
        for layer in 0..config.layer_count() {
            let (past_keys, past_values) = if let Some(past) = past {
                let key_name = format!("llama.batch.cache.{layer}.keys");
                let value_name = format!("llama.batch.cache.{layer}.values");
                let keys =
                    graph.input_dtype_requires_grad(&key_name, cache_shape, DType::F32, false);
                let values =
                    graph.input_dtype_requires_grad(&value_name, cache_shape, DType::F32, false);
                bindings.insert(key_name, past[layer].keys.clone());
                bindings.insert(value_name, past[layer].values.clone());
                (keys, values)
            } else {
                let zeros = TensorData::new(cache_shape, vec![0.0; cache_shape.iter().product()])?;
                (graph.constant(zeros.clone()), graph.constant(zeros))
            };
            let built = append_dense_batch_layer(
                &mut graph,
                x,
                self.state_map(),
                &format!("blk.{layer}"),
                schema,
                batch,
                sequence,
                starts,
                &chunks.iter().map(Vec::len).collect::<Vec<_>>(),
                config.max_context(),
                config.norm_eps(),
                config.rope_theta(),
                past_keys,
                past_values,
            )?;
            x = built.output;
            cache_nodes.push((built.keys, built.values));
        }
        let norm = graph.constant(self.state_map()[OUTPUT_NORM].clone());
        let normalized = rms_norm(
            &mut graph,
            x,
            norm,
            schema.embedding_dim(),
            config.norm_eps(),
        )?;
        let output = graph.constant(self.output_weight().clone());
        let logits = linear(&mut graph, normalized, output)?;
        Ok(LlamaBatchPlan {
            graph,
            bindings,
            logits,
            cache_nodes,
            chunk_lengths: chunks.iter().map(Vec::len).collect(),
            next_lengths,
        })
    }
}

fn validate_past(
    model: &LlamaModel,
    batch: usize,
    starts: &[usize],
    past: Option<&[BatchLayerCache]>,
) -> Result<(), LlamaModelError> {
    let Some(past) = past else {
        if starts.iter().any(|length| *length != 0) {
            return Err(LlamaModelError::CacheLengthMismatch);
        }
        return Ok(());
    };
    if past.len() != model.config().layer_count() {
        return Err(LlamaModelError::CacheLayerCount {
            expected: model.config().layer_count(),
            actual: past.len(),
        });
    }
    let schema = model.config().schema();
    let expected = [
        batch,
        schema.kv_heads(),
        model.config().max_context(),
        schema.head_dim(),
    ];
    for (layer, cache) in past.iter().enumerate() {
        if cache.keys.dtype() != DType::F32 || cache.values.dtype() != DType::F32 {
            return Err(LlamaModelError::CacheDType {
                layer,
                keys: cache.keys.dtype(),
                values: cache.values.dtype(),
            });
        }
        if cache.keys.shape().dims() != expected || cache.values.shape().dims() != expected {
            return Err(LlamaModelError::BatchCacheShape {
                layer,
                expected: expected.to_vec(),
                keys: cache.keys.shape().dims().to_vec(),
                values: cache.values.shape().dims().to_vec(),
            });
        }
    }
    Ok(())
}
