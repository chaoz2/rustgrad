//! Deterministic continuous scheduling over concrete strict-native Llama batches.

use super::{
    LlamaBatchNativeCache, LlamaChatError, LlamaChatMessage, LlamaChatTemplate,
    LlamaGenerationError, LlamaModel, LlamaNativeError, LlamaNativeExecutor,
    generation::select_row, native::LlamaNativePrefixSnapshot,
};
use crate::{TensorData, tokenizer::SimpleTokenizer};
use std::{
    collections::{BTreeMap, VecDeque},
    error, fmt,
    sync::Arc,
};

#[cfg(test)]
mod tests;

/// Stable scheduler-owned request identifier.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LlamaRequestId(u64);

impl LlamaRequestId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Owned deterministic sampling state for one request.
#[derive(Clone, Debug, PartialEq)]
pub enum LlamaServingSampling {
    Greedy,
    /// An explicit row-major `[generation_step, vocabulary]` uniform tape.
    GumbelMax {
        temperature: f32,
        uniforms: Vec<f32>,
    },
}

/// Per-request generation limits and sampling state.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaServingGenerationConfig {
    max_new_tokens: usize,
    sampling: LlamaServingSampling,
}

impl LlamaServingGenerationConfig {
    pub const fn new(max_new_tokens: usize, sampling: LlamaServingSampling) -> Self {
        Self {
            max_new_tokens,
            sampling,
        }
    }
    pub const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }
    pub const fn sampling(&self) -> &LlamaServingSampling {
        &self.sampling
    }
}

/// Fixed native batch width and bounded immutable prefix storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaServingConfig {
    batch_size: usize,
    prefix_max_entries: usize,
    prefix_max_bytes: usize,
}

impl LlamaServingConfig {
    pub fn new(
        batch_size: usize,
        prefix_max_entries: usize,
        prefix_max_bytes: usize,
    ) -> Result<Self, LlamaServingError> {
        if batch_size == 0 {
            return Err(LlamaServingError::EmptyBatch);
        }
        Ok(Self {
            batch_size,
            prefix_max_entries,
            prefix_max_bytes,
        })
    }
    pub const fn batch_size(self) -> usize {
        self.batch_size
    }
}

/// Observable lifecycle of one admitted request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaRequestStatus {
    Queued,
    Running,
    Complete,
}

/// One committed generated token in deterministic scheduler order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaTokenEvent {
    request_id: LlamaRequestId,
    arrival_order: u64,
    generation_step: usize,
    token_id: u32,
    stopped: bool,
    complete: bool,
}

impl LlamaTokenEvent {
    pub const fn request_id(&self) -> LlamaRequestId {
        self.request_id
    }
    pub const fn arrival_order(&self) -> u64 {
        self.arrival_order
    }
    pub const fn generation_step(&self) -> usize {
        self.generation_step
    }
    pub const fn token_id(&self) -> u32 {
        self.token_id
    }
    pub const fn stopped(&self) -> bool {
        self.stopped
    }
    pub const fn complete(&self) -> bool {
        self.complete
    }
}

/// Completed deterministic output retained after cache state is released.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaServingResult {
    request_id: LlamaRequestId,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    decoded: String,
    stopped: bool,
}

impl LlamaServingResult {
    pub const fn request_id(&self) -> LlamaRequestId {
        self.request_id
    }
    pub fn prompt_ids(&self) -> &[u32] {
        &self.prompt_ids
    }
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }
    pub fn decoded(&self) -> &str {
        &self.decoded
    }
    pub const fn stopped(&self) -> bool {
        self.stopped
    }
}

/// Prefix-cache accounting after the last committed scheduler operation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LlamaPrefixCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub stale_rejections: u64,
    pub evictions: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ModelIdentity {
    config: u64,
    state: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PrefixKey {
    model: ModelIdentity,
    tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
struct PrefixEntry {
    generation: u64,
    snapshot: Arc<LlamaNativePrefixSnapshot>,
    logits: Arc<TensorData>,
    bytes: usize,
    last_used: u64,
    references: usize,
}

#[derive(Clone, Debug)]
struct PrefixCache {
    identity: ModelIdentity,
    generation: u64,
    clock: u64,
    max_entries: usize,
    max_bytes: usize,
    bytes: usize,
    hits: u64,
    misses: u64,
    stale_rejections: u64,
    evictions: u64,
    entries: BTreeMap<PrefixKey, PrefixEntry>,
}

impl PrefixCache {
    fn new(identity: ModelIdentity, max_entries: usize, max_bytes: usize) -> Self {
        Self {
            identity,
            generation: 0,
            clock: 0,
            max_entries,
            max_bytes,
            bytes: 0,
            hits: 0,
            misses: 0,
            stale_rejections: 0,
            evictions: 0,
            entries: BTreeMap::new(),
        }
    }

    fn stats(&self) -> LlamaPrefixCacheStats {
        LlamaPrefixCacheStats {
            entries: self.entries.len(),
            bytes: self.bytes,
            hits: self.hits,
            misses: self.misses,
            stale_rejections: self.stale_rejections,
            evictions: self.evictions,
            generation: self.generation,
        }
    }

    fn invalidate(&mut self, identity: ModelIdentity) {
        self.identity = identity;
        self.generation = self.generation.wrapping_add(1);
        self.bytes = 0;
        self.entries.clear();
    }

    fn release(&mut self, key: &PrefixKey) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.references = entry.references.saturating_sub(1);
        }
    }

    fn longest(
        &mut self,
        tokens: &[u32],
    ) -> Option<(PrefixKey, Arc<LlamaNativePrefixSnapshot>, Arc<TensorData>)> {
        let stale = self
            .entries
            .iter()
            .filter(|(key, entry)| {
                key.model != self.identity || entry.generation != self.generation
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        for key in stale {
            if let Some(entry) = self.entries.remove(&key) {
                self.bytes = self.bytes.saturating_sub(entry.bytes);
                self.stale_rejections = self.stale_rejections.saturating_add(1);
            }
        }
        let key = self
            .entries
            .keys()
            .filter(|key| key.model == self.identity && tokens.starts_with(&key.tokens))
            .max_by_key(|key| key.tokens.len())
            .cloned();
        let Some(key) = key else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(&key).expect("selected entry exists");
        entry.last_used = self.clock;
        entry.references = entry.references.saturating_add(1);
        self.hits = self.hits.saturating_add(1);
        Some((key, Arc::clone(&entry.snapshot), Arc::clone(&entry.logits)))
    }

    fn insert(
        &mut self,
        tokens: Vec<u32>,
        snapshot: Arc<LlamaNativePrefixSnapshot>,
        logits: Arc<TensorData>,
    ) -> Result<bool, LlamaServingError> {
        if self.max_entries == 0 || self.max_bytes == 0 {
            return Ok(false);
        }
        let bytes = snapshot
            .byte_len()?
            .checked_add(
                logits
                    .len()
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(LlamaServingError::AccountingOverflow)?,
            )
            .and_then(|value| {
                tokens
                    .len()
                    .checked_mul(std::mem::size_of::<u32>())
                    .and_then(|token_bytes| value.checked_add(token_bytes))
            })
            .ok_or(LlamaServingError::AccountingOverflow)?;
        if bytes > self.max_bytes {
            return Ok(false);
        }
        let key = PrefixKey {
            model: self.identity,
            tokens,
        };
        if let Some(previous) = self.entries.remove(&key) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        while self.entries.len() >= self.max_entries
            || self
                .bytes
                .checked_add(bytes)
                .is_none_or(|value| value > self.max_bytes)
        {
            let victim = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.references == 0)
                .min_by_key(|(key, entry)| (entry.last_used, (*key).clone()))
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else {
                return Ok(false);
            };
            let removed = self.entries.remove(&victim).expect("selected entry exists");
            self.bytes -= removed.bytes;
            self.evictions = self.evictions.saturating_add(1);
        }
        self.clock = self.clock.wrapping_add(1);
        self.bytes += bytes;
        self.entries.insert(
            key,
            PrefixEntry {
                generation: self.generation,
                snapshot,
                logits,
                bytes,
                last_used: self.clock,
                references: 0,
            },
        );
        Ok(true)
    }
}

#[derive(Clone, Debug)]
struct RequestState {
    id: LlamaRequestId,
    arrival: u64,
    prompt: Vec<u32>,
    config: LlamaServingGenerationConfig,
    status: LlamaRequestStatus,
    generated: Vec<u32>,
    stopped: bool,
    snapshot: Option<Arc<LlamaNativePrefixSnapshot>>,
    prefix_reference: Option<PrefixKey>,
    logits: Option<Arc<TensorData>>,
    pending_token: Option<u32>,
    result: Option<LlamaServingResult>,
}

/// Continuous request scheduler over strict-native, concrete fixed batches.
pub struct LlamaServingScheduler<'a> {
    model: &'a LlamaModel,
    tokenizer: &'a SimpleTokenizer,
    config: LlamaServingConfig,
    identity: ModelIdentity,
    next_id: u64,
    next_arrival: u64,
    order: VecDeque<LlamaRequestId>,
    requests: BTreeMap<LlamaRequestId, RequestState>,
    prefixes: PrefixCache,
    executor: LlamaNativeExecutor,
}

impl<'a> LlamaServingScheduler<'a> {
    pub fn new(
        model: &'a LlamaModel,
        tokenizer: &'a SimpleTokenizer,
        config: LlamaServingConfig,
    ) -> Self {
        let identity = model_identity(model);
        Self {
            model,
            tokenizer,
            config,
            identity,
            next_id: 0,
            next_arrival: 0,
            order: VecDeque::new(),
            requests: BTreeMap::new(),
            prefixes: PrefixCache::new(
                identity,
                config.prefix_max_entries,
                config.prefix_max_bytes,
            ),
            executor: LlamaNativeExecutor::new(),
        }
    }

    pub fn submit_ids(
        &mut self,
        prompt: Vec<u32>,
        config: LlamaServingGenerationConfig,
    ) -> Result<LlamaRequestId, LlamaServingError> {
        validate_request(self.model, &prompt, &config)?;
        let id = LlamaRequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(LlamaServingError::RequestIdOverflow)?;
        let arrival = self.next_arrival;
        self.next_arrival = self.next_arrival.wrapping_add(1);
        let complete = config.max_new_tokens == 0;
        let result = complete.then(|| LlamaServingResult {
            request_id: id,
            prompt_ids: prompt.clone(),
            generated_ids: Vec::new(),
            decoded: String::new(),
            stopped: false,
        });
        self.requests.insert(
            id,
            RequestState {
                id,
                arrival,
                prompt,
                config,
                status: if complete {
                    LlamaRequestStatus::Complete
                } else {
                    LlamaRequestStatus::Queued
                },
                generated: Vec::new(),
                stopped: false,
                snapshot: None,
                prefix_reference: None,
                logits: None,
                pending_token: None,
                result,
            },
        );
        self.order.push_back(id);
        Ok(id)
    }

    pub fn submit_text(
        &mut self,
        prompt: &str,
        config: LlamaServingGenerationConfig,
    ) -> Result<LlamaRequestId, LlamaServingError> {
        let mut ids = self.tokenizer.encode(prompt)?;
        if let Some(bos) = self.model.config().token_ids().bos() {
            ids.insert(0, bos);
        }
        self.submit_ids(ids, config)
    }

    pub fn submit_chat(
        &mut self,
        template: LlamaChatTemplate,
        messages: &[LlamaChatMessage],
        config: LlamaServingGenerationConfig,
    ) -> Result<LlamaRequestId, LlamaServingError> {
        let rendered = template.render(self.tokenizer, messages, true)?;
        self.submit_ids(self.tokenizer.encode(&rendered)?, config)
    }

    pub fn status(&self, id: LlamaRequestId) -> Option<LlamaRequestStatus> {
        self.requests.get(&id).map(|request| request.status)
    }

    pub fn result(&self, id: LlamaRequestId) -> Option<&LlamaServingResult> {
        self.requests.get(&id)?.result.as_ref()
    }

    /// Returns completed outputs in stable arrival order.
    pub fn completed_results(&self) -> impl Iterator<Item = &LlamaServingResult> {
        self.order
            .iter()
            .filter_map(|id| self.requests.get(id)?.result.as_ref())
    }

    pub fn pending(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.status != LlamaRequestStatus::Complete)
            .count()
    }

    pub fn prefix_stats(&self) -> LlamaPrefixCacheStats {
        self.prefixes.stats()
    }

    /// Returns the number of native kernels retained across serving steps.
    pub fn native_compile_cache_len(&self) -> usize {
        self.executor.compile_cache_len()
    }

    /// Removes one request only between steps and releases its cache lease.
    pub fn remove(&mut self, id: LlamaRequestId) -> bool {
        let Some(request) = self.requests.remove(&id) else {
            return false;
        };
        if let Some(key) = request.prefix_reference {
            self.prefixes.release(&key);
        }
        self.order.retain(|queued| *queued != id);
        true
    }

    /// Explicitly invalidates all prefix entries and advances their version.
    pub fn invalidate_prefix_cache(&mut self) {
        self.prefixes.invalidate(self.identity);
        for request in self.requests.values_mut() {
            request.prefix_reference = None;
        }
    }

    /// Rebinds an idle scheduler. A model state/config identity change
    /// invalidates every prefix before any new request can be admitted.
    pub fn rebind(
        &mut self,
        model: &'a LlamaModel,
        tokenizer: &'a SimpleTokenizer,
    ) -> Result<bool, LlamaServingError> {
        if self.pending() != 0 {
            return Err(LlamaServingError::ActiveRequests);
        }
        let identity = model_identity(model);
        let changed = identity != self.identity;
        self.model = model;
        self.tokenizer = tokenizer;
        if changed {
            self.identity = identity;
            self.prefixes.invalidate(identity);
        }
        Ok(changed)
    }

    /// Executes at most one native prefill/decode and one token selection per
    /// selected request. All selected states and prefix accounting commit together.
    pub fn step(&mut self) -> Result<Vec<LlamaTokenEvent>, LlamaServingError> {
        let selected = self
            .order
            .iter()
            .filter(|id| {
                self.requests
                    .get(id)
                    .is_some_and(|request| request.status != LlamaRequestStatus::Complete)
            })
            .take(self.config.batch_size)
            .copied()
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Ok(Vec::new());
        }

        let mut staged_requests = selected
            .iter()
            .map(|id| (*id, self.requests[id].clone()))
            .collect::<BTreeMap<_, _>>();
        let mut staged_prefixes = self.prefixes.clone();
        let mut compute_ids = Vec::new();
        let mut chunks = Vec::new();
        let mut snapshots = Vec::new();
        for id in &selected {
            let request = staged_requests
                .get_mut(id)
                .expect("selected request exists");
            request.status = LlamaRequestStatus::Running;
            if request.logits.is_none() {
                let chunk = if let Some(token) = request.pending_token.take() {
                    vec![token]
                } else {
                    if request.snapshot.is_none()
                        && let Some((key, snapshot, logits)) =
                            staged_prefixes.longest(&request.prompt)
                    {
                        request.prefix_reference = Some(key);
                        request.snapshot = Some(snapshot);
                        if request
                            .snapshot
                            .as_ref()
                            .is_some_and(|snapshot| snapshot.len() == request.prompt.len())
                        {
                            request.logits = Some(logits);
                        }
                    }
                    if request.logits.is_some() {
                        Vec::new()
                    } else {
                        let start = request.snapshot.as_ref().map_or(0, |value| value.len());
                        request.prompt[start..].to_vec()
                    }
                };
                if !chunk.is_empty() {
                    compute_ids.push(*id);
                    chunks.push(chunk);
                    snapshots.push(request.snapshot.as_deref().cloned());
                }
            }
        }

        if !compute_ids.is_empty() {
            let mut batch =
                LlamaBatchNativeCache::from_snapshots(self.model.config().clone(), &snapshots)?;
            let execution = batch.forward_with_executor(self.model, &chunks, &self.executor)?;
            let output_snapshots = batch.snapshots()?;
            for ((id, logits), snapshot) in compute_ids
                .iter()
                .zip(execution.rows())
                .zip(output_snapshots)
            {
                let request = staged_requests.get_mut(id).expect("compute request exists");
                if let Some(key) = request.prefix_reference.take() {
                    staged_prefixes.release(&key);
                }
                request.snapshot = Some(Arc::new(snapshot));
                request.logits = Some(Arc::new(logits.clone()));
                let cached_tokens = request.cached_tokens();
                staged_prefixes.insert(
                    cached_tokens,
                    Arc::clone(request.snapshot.as_ref().expect("snapshot assigned")),
                    Arc::clone(request.logits.as_ref().expect("logits assigned")),
                )?;
            }
        }

        let vocab = self.model.config().schema().vocab_size();
        let mut events = Vec::with_capacity(selected.len());
        for id in &selected {
            let request = staged_requests
                .get_mut(id)
                .expect("selected request exists");
            let logits = request
                .logits
                .take()
                .ok_or(LlamaServingError::MissingLogits(*id))?;
            let values = logits.values();
            if values.len() < vocab {
                return Err(LlamaGenerationError::InvalidLogits.into());
            }
            let last = &values[values.len() - vocab..];
            let generation_step = request.generated.len();
            let token = match &request.config.sampling {
                LlamaServingSampling::Greedy => select_row(last, None, None)?,
                LlamaServingSampling::GumbelMax {
                    temperature,
                    uniforms,
                } => {
                    let offset = generation_step * vocab;
                    select_row(
                        last,
                        Some(*temperature),
                        Some(&uniforms[offset..offset + vocab]),
                    )?
                }
            };
            request.generated.push(token);
            request.stopped = self.model.config().token_ids().is_stop(token);
            let complete =
                request.stopped || request.generated.len() == request.config.max_new_tokens;
            if complete {
                if let Some(key) = request.prefix_reference.take() {
                    staged_prefixes.release(&key);
                }
                request.status = LlamaRequestStatus::Complete;
                request.snapshot = None;
                request.pending_token = None;
                request.result = Some(LlamaServingResult {
                    request_id: request.id,
                    prompt_ids: request.prompt.clone(),
                    generated_ids: request.generated.clone(),
                    decoded: self.tokenizer.decode(&request.generated)?,
                    stopped: request.stopped,
                });
            } else {
                request.pending_token = Some(token);
            }
            events.push(LlamaTokenEvent {
                request_id: *id,
                arrival_order: request.arrival,
                generation_step,
                token_id: token,
                stopped: request.stopped,
                complete,
            });
        }

        for (id, request) in staged_requests {
            self.requests.insert(id, request);
        }
        self.prefixes = staged_prefixes;
        Ok(events)
    }

    #[cfg(test)]
    pub(super) fn inject_stage_failure(&mut self, stage: Option<usize>) {
        self.executor.inject_stage_failure(stage);
    }

    #[cfg(test)]
    pub(super) fn make_prefixes_stale(&mut self) {
        self.prefixes.generation = self.prefixes.generation.wrapping_add(1);
    }
}

impl RequestState {
    fn cached_tokens(&self) -> Vec<u32> {
        let length = self.snapshot.as_ref().map_or(0, |snapshot| snapshot.len());
        let mut tokens = self.prompt.clone();
        tokens.extend_from_slice(&self.generated);
        tokens.truncate(length);
        tokens
    }
}

fn validate_request(
    model: &LlamaModel,
    prompt: &[u32],
    config: &LlamaServingGenerationConfig,
) -> Result<(), LlamaServingError> {
    if prompt.is_empty() {
        return Err(LlamaServingError::EmptyPrompt);
    }
    let requested = prompt
        .len()
        .checked_add(config.max_new_tokens)
        .ok_or(LlamaServingError::ContextOverflow)?;
    if requested > model.config().max_context() {
        return Err(LlamaServingError::ContextLength {
            requested,
            maximum: model.config().max_context(),
        });
    }
    let vocab = model.config().schema().vocab_size();
    if let Some(&token) = prompt
        .iter()
        .find(|&&token| usize::try_from(token).map_or(true, |token| token >= vocab))
    {
        return Err(LlamaServingError::TokenOutOfRange { token, vocab });
    }
    if let LlamaServingSampling::GumbelMax {
        temperature,
        uniforms,
    } = &config.sampling
    {
        if !temperature.is_finite() || *temperature <= 0.0 {
            return Err(LlamaGenerationError::InvalidTemperature.into());
        }
        let required = config
            .max_new_tokens
            .checked_mul(vocab)
            .ok_or(LlamaServingError::ContextOverflow)?;
        if uniforms.len() < required {
            return Err(LlamaGenerationError::UniformTapeLength {
                required,
                actual: uniforms.len(),
            }
            .into());
        }
        if let Some(index) = uniforms[..required]
            .iter()
            .position(|value| !value.is_finite() || *value < 0.0 || *value >= 1.0)
        {
            return Err(LlamaGenerationError::InvalidUniform { index }.into());
        }
    }
    Ok(())
}

fn model_identity(model: &LlamaModel) -> ModelIdentity {
    let config = model.config();
    let mut config_hash = Fingerprint::new();
    config_hash.bytes(config.architecture().as_bytes());
    for value in [
        config.layer_count(),
        config.max_context(),
        config.schema().vocab_size(),
        config.schema().embedding_dim(),
        config.schema().hidden_dim(),
        config.schema().query_heads(),
        config.schema().kv_heads(),
        config.schema().head_dim(),
        config.schema().rope_dim(),
    ] {
        config_hash.u64(value as u64);
    }
    config_hash.u64(u64::from(config.norm_eps().to_bits()));
    config_hash.u64(config.rope_theta().to_bits());
    config_hash.u64(match config.qk_norm() {
        super::LlamaQkNorm::None => 0,
        super::LlamaQkNorm::PerHead => 1,
        super::LlamaQkNorm::PerProjection => 2,
    });
    config_hash.u64(u64::from(config.qkv_bias()));
    config_hash.u64(u64::from(config.token_ids().bos().unwrap_or(u32::MAX)));
    config_hash.u64(u64::from(config.token_ids().eos()));
    config_hash.u64(u64::from(config.token_ids().eot().unwrap_or(u32::MAX)));

    let mut state_hash = Fingerprint::new();
    for (name, tensor) in model.dense_state() {
        state_hash.bytes(name.as_bytes());
        for &dimension in tensor.shape().dims() {
            state_hash.u64(dimension as u64);
        }
        for &value in tensor.values() {
            state_hash.u64(u64::from(value.to_bits()));
        }
    }
    for (name, weight) in model.linear_weights() {
        state_hash.bytes(name.as_bytes());
        match weight {
            super::LlamaLinearWeight::Dense(tensor) => {
                state_hash.u64(0);
                for &dimension in tensor.shape().dims() {
                    state_hash.u64(dimension as u64);
                }
                for &value in tensor.values() {
                    state_hash.u64(u64::from(value.to_bits()));
                }
            }
            super::LlamaLinearWeight::Quantized(tensor) => {
                state_hash.u64(1 + u64::from(tensor.descriptor().ggml_type.raw()));
                for &dimension in tensor.descriptor().logical_shape.dims() {
                    state_hash.u64(dimension as u64);
                }
                state_hash.bytes(tensor.bytes());
            }
        }
    }
    ModelIdentity {
        config: config_hash.finish(),
        state: state_hash.finish(),
    }
}

struct Fingerprint(u64);

impl Fingerprint {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }
    const fn finish(self) -> u64 {
        self.0
    }
}

/// Structured admission, native execution, sampling, tokenizer, or accounting failure.
#[derive(Debug)]
pub enum LlamaServingError {
    Native(LlamaNativeError),
    Generation(LlamaGenerationError),
    Tokenizer(crate::tokenizer::TokenizerError),
    Chat(LlamaChatError),
    EmptyBatch,
    EmptyPrompt,
    ContextOverflow,
    ContextLength { requested: usize, maximum: usize },
    TokenOutOfRange { token: u32, vocab: usize },
    RequestIdOverflow,
    MissingLogits(LlamaRequestId),
    AccountingOverflow,
    ActiveRequests,
}

impl fmt::Display for LlamaServingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llama serving error: {self:?}")
    }
}
impl error::Error for LlamaServingError {}
impl From<LlamaNativeError> for LlamaServingError {
    fn from(value: LlamaNativeError) -> Self {
        Self::Native(value)
    }
}
impl From<LlamaGenerationError> for LlamaServingError {
    fn from(value: LlamaGenerationError) -> Self {
        Self::Generation(value)
    }
}
impl From<crate::tokenizer::TokenizerError> for LlamaServingError {
    fn from(value: crate::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
impl From<LlamaChatError> for LlamaServingError {
    fn from(value: LlamaChatError) -> Self {
        Self::Chat(value)
    }
}
