//! Manual, resource-bounded evidence for a larger fixed-shape compiled
//! Transformer training run on strict-native CPU.
//!
//! This example is intentionally excluded from ordinary CI execution. The
//! `native-transformer-scale-evidence` workflow runs it at an exact reviewed
//! commit and uploads its authenticated scoreboard, objective facts, and host
//! provenance. Timings are observations, never pass/fail thresholds.

use rustgrad::nn::{Embedding, LayerNorm, Mode, StateKind};
use rustgrad::{
    CapturedReplayExecutor, CompiledAdamWConfig, CompiledAdamWGraph,
    CompiledAdamWIgnoreIndexContext, CompiledCheckpointRestoreRuntime, CompiledCheckpointRuntime,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledEvaluationRuntime, CompiledInputBatch,
    CompiledInputSpec, CompiledModuleAdamWPlan, CompiledMultiStepLr, DType, Graph, LossOptions,
    Module, NativeCpuCompiledAdamW, NativeCpuCompiledAdamWStepResult, NativeCpuSessionTarget,
    NativeTrainingScoreboard, NodeId, Parameter, Reduction, Result, Scalar, TensorData,
    TrainingDropoutProvider, TransformerBlock, sparse_categorical_cross_entropy,
};
use serde_json::json;
use std::{collections::BTreeMap, env, error::Error, fs, path::PathBuf, time::Instant};

const BATCH: usize = 4;
const TIME: usize = 8;
const VOCAB: usize = 16;
const EMBEDDING: usize = 8;
const HEADS: usize = 2;
const FEED_FORWARD: usize = 32;
const ACCUMULATION_STEPS: u64 = 2;
const REPLAYS: u64 = 6;
const CHECKPOINT_REPLAY: u64 = 3;
const IGNORE_INDEX: i32 = -100;
const TARGETS: &str = "targets";
const TOKENS: &str = "tokens";
const FROZEN_POSITION_WEIGHT: &str = "positions.weight";
const ATTENTION_KEEP_MASK_SHAPE: [usize; 4] = [BATCH, 1, 1, TIME];

struct ScaleTransformer {
    tokens: Embedding,
    positions: Embedding,
    first: TransformerBlock,
    second: TransformerBlock,
    norm: LayerNorm,
}

impl ScaleTransformer {
    fn new(seed: u64) -> Result<Self> {
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

    fn forward_training(
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

    fn forward_evaluation(
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

impl Module for ScaleTransformer {
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

struct ScaleBatch {
    tokens: TensorData,
    targets: TensorData,
}

impl ScaleBatch {
    const SCHEMA: [CompiledInputSpec; 2] = [
        CompiledInputSpec::new(TARGETS, &[BATCH, TIME], DType::I32),
        CompiledInputSpec::host_token(TOKENS, &[BATCH, TIME]),
    ];

    fn new(replay: u64) -> Result<Self> {
        let phase = (replay - 1) % ACCUMULATION_STEPS + 1;
        let valid_lengths = match phase {
            1 => [8, 7, 5, 2],
            2 => [8, 5, 3, 0],
            _ => unreachable!(),
        };
        let mut tokens = Vec::with_capacity(BATCH * TIME);
        let mut targets = Vec::with_capacity(BATCH * TIME);
        for (row, valid) in valid_lengths.into_iter().enumerate() {
            for column in 0..TIME {
                if column < valid {
                    let token = i64::try_from(
                        (usize::try_from(phase).expect("fixed replay phase fits in usize")
                            + row * 3
                            + column)
                            % VOCAB,
                    )
                    .expect("fixed token fits in i64");
                    tokens.push(Scalar::I(token));
                    targets.push(Scalar::I(
                        (token + 1) % i64::try_from(VOCAB).expect("fixed vocabulary fits in i64"),
                    ));
                } else {
                    tokens.push(Scalar::I(0));
                    targets.push(Scalar::I(i64::from(IGNORE_INDEX)));
                }
            }
        }
        Ok(Self {
            tokens: TensorData::from_scalars([BATCH, TIME], DType::I32, tokens)?,
            targets: TensorData::from_scalars([BATCH, TIME], DType::I32, targets)?,
        })
    }
}

impl CompiledInputBatch for ScaleBatch {
    fn schema() -> &'static [CompiledInputSpec] {
        &Self::SCHEMA
    }

    fn into_compiled_inputs(self) -> Result<BTreeMap<String, TensorData>> {
        Ok(BTreeMap::from([
            (TARGETS.into(), self.targets),
            (TOKENS.into(), self.tokens),
        ]))
    }
}

fn losses(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<NodeId> {
    sparse_categorical_cross_entropy(
        graph,
        logits,
        targets,
        LossOptions {
            reduction: Reduction::None,
            class_axis: 2,
            ignore_index: Some(i64::from(IGNORE_INDEX)),
            label_smoothing: 0.0,
        },
    )
}

fn build_training(
    model: &ScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_training(graph, inputs[TOKENS], attention_keep_mask, dropout)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn build_evaluation(
    model: &ScaleTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    ignore_index: CompiledAdamWIgnoreIndexContext,
) -> Result<CompiledAdamWGraph> {
    let attention_keep_mask = graph.reshape(ignore_index.validity(), ATTENTION_KEEP_MASK_SHAPE)?;
    let logits = model.forward_evaluation(graph, inputs[TOKENS], attention_keep_mask)?;
    Ok(CompiledAdamWGraph::token_mean(
        losses(graph, logits, inputs[TARGETS])?,
        BTreeMap::new(),
    ))
}

fn mean_evaluation_loss(
    runtime: &mut rustgrad::CompiledModuleAdamWSession<
        ScaleTransformer,
        NativeCpuCompiledAdamW<'_>,
    >,
) -> Result<f64> {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0_u64;
    for replay in 1..=ACCUMULATION_STEPS {
        let evaluation = runtime.evaluate_batch(ScaleBatch::new(replay)?)?;
        assert_eq!(evaluation.report().fallback_count(), 0);
        assert!(evaluation.loss_weight() > 0);
        weighted_sum += evaluation.loss().scalar_at(0).as_f64() * evaluation.loss_weight() as f64;
        weight_sum = weight_sum
            .checked_add(evaluation.loss_weight())
            .expect("the fixed evaluation weight cannot overflow");
    }
    Ok(weighted_sum / weight_sum as f64)
}

fn assert_strict_native(step: &NativeCpuCompiledAdamWStepResult) {
    assert_eq!(step.report().fallback_count(), 0);
    assert!(step.report().executed_native_item_count() > 0);
    assert!(step.report().module_dispatch_count() > 0);
}

fn required_path(name: &str) -> std::result::Result<PathBuf, Box<dyn Error>> {
    let value = env::var_os(name).ok_or_else(|| format!("{name} must name an evidence file"))?;
    Ok(PathBuf::from(value))
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let git_sha = env::var("RUSTGRAD_EVIDENCE_GIT_SHA")?;
    if git_sha.len() != 40
        || !git_sha
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("RUSTGRAD_EVIDENCE_GIT_SHA must be a lowercase full Git SHA".into());
    }
    let scoreboard_path = required_path("RUSTGRAD_LARGER_SCOREBOARD_PATH")?;
    let objective_path = required_path("RUSTGRAD_LARGER_OBJECTIVE_PATH")?;

    let schedule = CompiledMultiStepLr::new(2e-3, 0.5, [1])?;
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_loss_scale(128.0)?
        .with_gradient_accumulation(ACCUMULATION_STEPS)?
        .with_max_gradient_norm(0.25)?
        .with_input_batch::<ScaleBatch>()?
        .with_token_weighted_ignore_index(TARGETS, IGNORE_INDEX)?
        .with_frozen_parameters([FROZEN_POSITION_WEIGHT])?
        .with_captured_multi_step_lr(schedule)
        .with_clip_report()
        .with_window_loss_report();

    let compile_started = Instant::now();
    let plan = CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
        config,
        CompiledDropoutConfig::new(CompiledDropoutKey([0x5ca1_e001, 0x5ca1_e002])),
        ScaleTransformer::new(0x5ca1_e000)?,
        build_training,
    )
    .map_err(|error| error.into_parts().1)?
    .with_evaluation_graph_and_ignore_index(build_evaluation)
    .map_err(|error| error.into_parts().1)?;
    let compile_wall_time = compile_started.elapsed();
    let inspection = plan.inspection()?;
    assert!(inspection.recurrent_state_count() > 0);
    assert!(inspection.recurrent_state_bytes() > 0);

    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let prepare_started = Instant::now();
    let mut runtime = plan
        .prepare(&target)
        .map_err(|error| error.into_parts().1)?;
    let prepare_wall_time = prepare_started.elapsed();
    let preparation = runtime.native_cpu_preparation_report();
    for program in [
        Some(preparation.main()),
        preparation.accumulation(),
        preparation.partial_flush(),
        preparation.zero_grad(),
        preparation.evaluation(),
    ]
    .into_iter()
    .flatten()
    {
        assert!(program.native_item_count() > 0);
        assert_eq!(program.fallback_count(), 0);
    }
    let mut scoreboard = NativeTrainingScoreboard::new(
        inspection.clone(),
        preparation,
        compile_wall_time,
        prepare_wall_time,
    )?;

    let initial_checkpoint = runtime.checkpoint()?;
    let initial_loss = mean_evaluation_loss(&mut runtime)?;
    assert_eq!(runtime.checkpoint()?, initial_checkpoint);

    let mut pending_checkpoint = None;
    let mut continuation = Vec::new();
    for replay in 1..=REPLAYS {
        let step = runtime.step_batch_commit_only_scheduled(ScaleBatch::new(replay)?)?;
        assert_strict_native(&step);
        assert_eq!(step.did_update(), replay % ACCUMULATION_STEPS == 0);
        assert_eq!(step.optimizer_step(), replay / ACCUMULATION_STEPS);
        assert_eq!(step.accumulation_index(), replay % ACCUMULATION_STEPS);
        assert_eq!(step.clip_report().is_some(), step.did_update());
        assert_eq!(step.window_loss_report().is_some(), step.did_update());
        scoreboard.record_step(&step)?;
        if replay == CHECKPOINT_REPLAY {
            let started = Instant::now();
            let checkpoint = runtime.checkpoint()?;
            scoreboard.observe_checkpoint(&checkpoint, started.elapsed())?;
            assert_eq!(checkpoint.info().replay_step(), CHECKPOINT_REPLAY);
            assert_eq!(checkpoint.info().optimizer_step(), 1);
            assert_eq!(checkpoint.info().accumulation_index(), 1);
            pending_checkpoint = Some(checkpoint);
        } else if replay > CHECKPOINT_REPLAY {
            continuation.push((step.loss().clone(), runtime.checkpoint()?));
        }
    }
    let uninterrupted_final = runtime.checkpoint()?;
    assert_eq!(uninterrupted_final.info().replay_step(), REPLAYS);
    assert_eq!(uninterrupted_final.info().optimizer_step(), 3);
    assert_eq!(uninterrupted_final.info().accumulation_index(), 0);

    let pending_checkpoint = pending_checkpoint.expect("replay three checkpoints a pending window");
    runtime.restore_checkpoint_in_place(&pending_checkpoint)?;
    assert_eq!(runtime.checkpoint()?, pending_checkpoint);
    for (index, replay) in ((CHECKPOINT_REPLAY + 1)..=REPLAYS).enumerate() {
        let step = runtime.step_batch_commit_only_scheduled(ScaleBatch::new(replay)?)?;
        assert_strict_native(&step);
        assert_eq!(step.loss(), &continuation[index].0);
        assert_eq!(runtime.checkpoint()?, continuation[index].1);
    }
    assert_eq!(runtime.checkpoint()?, uninterrupted_final);
    let before_final_evaluation = runtime.checkpoint()?;
    let final_loss = mean_evaluation_loss(&mut runtime)?;
    assert_eq!(runtime.checkpoint()?, before_final_evaluation);
    assert!(
        final_loss < initial_loss,
        "larger compiled Transformer loss did not decrease: {initial_loss} -> {final_loss}"
    );

    let report = scoreboard.report()?;
    assert_eq!(report.fallback_count(), 0);
    assert_eq!(report.successful_replay_count(), REPLAYS);
    assert_eq!(
        report.recurrent_state_count(),
        u64::try_from(inspection.recurrent_state_count())?
    );
    assert_eq!(
        report.recurrent_state_bytes(),
        u64::try_from(inspection.recurrent_state_bytes())?
    );
    assert!(report.checkpoint_byte_count().is_some());
    fs::write(&scoreboard_path, report.to_json_bytes()?)?;

    let objective = json!({
        "schema_version": 1,
        "git_sha": git_sha,
        "workload": {
            "batch": BATCH,
            "time": TIME,
            "vocabulary": VOCAB,
            "embedding": EMBEDDING,
            "heads": HEADS,
            "feed_forward": FEED_FORWARD,
            "blocks": 2,
            "compile_count": 1,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "replays": REPLAYS
        },
        "objective": {
            "initial_token_mean_loss": initial_loss,
            "final_token_mean_loss": final_loss,
            "decreased": true
        },
        "progress": {
            "checkpoint_replay": CHECKPOINT_REPLAY,
            "checkpoint_optimizer_step": 1,
            "checkpoint_accumulation_index": 1,
            "checkpoint_bytes": report.checkpoint_byte_count(),
            "final_replay": REPLAYS,
            "final_optimizer_step": 3,
            "final_accumulation_index": 0,
            "exact_resume": true
        },
        "native": {
            "fallback_count": report.fallback_count(),
            "successful_replay_count": report.successful_replay_count(),
            "recurrent_state_count": report.recurrent_state_count(),
            "recurrent_state_bytes": report.recurrent_state_bytes()
        }
    });
    let mut objective_bytes = serde_json::to_vec_pretty(&objective)?;
    objective_bytes.push(b'\n');
    fs::write(&objective_path, objective_bytes)?;
    Ok(())
}
