//! Train a fixed two-block byte Transformer from a caller-owned local corpus.
//!
//! `train CORPUS OUTPUT STEPS` starts from random initialization;
//! `resume CORPUS CHECKPOINT OUTPUT STEPS` restores the executable and all state.
//! STEPS counts microbatches, not optimizer updates. Outputs are explicit new
//! paths. Checkpoints include the exact corpus bytes; treat them as private data.

#[path = "compiled_corpus_train/data.rs"]
mod data;
#[path = "support/causal_transformer.rs"]
mod model;

use data::{Corpus, CorpusBatch};
use model::CausalTransformer;
use rustgrad::{
    CapturedReplayExecutor, CompiledAdamWConfig, CompiledAdamWGraph,
    CompiledAdamWIgnoreIndexContext, CompiledAdamWResumeBundle, CompiledCheckpointRuntime,
    CompiledDropoutConfig, CompiledDropoutKey, CompiledEvaluationRuntime, CompiledModuleAdamWPlan,
    CompiledModuleAdamWSession, CompiledMultiStepLr, Graph, LossOptions, NativeCpuCompiledAdamW,
    NativeCpuSessionTarget, NodeId, Reduction, Result, TrainingDropoutProvider,
    sparse_categorical_cross_entropy,
};
use std::{collections::BTreeMap, env, error::Error, fs, path::PathBuf};

const BATCH: usize = 2;
const TIME: usize = 16;
const VOCAB: usize = 256;
const EMBEDDING: usize = 16;
const HEADS: usize = 2;
const FEED_FORWARD: usize = 64;
const TOKENS: &str = "tokens";
const TARGETS: &str = "targets";

fn objective(graph: &mut Graph, logits: NodeId, targets: NodeId) -> Result<CompiledAdamWGraph> {
    let losses = sparse_categorical_cross_entropy(
        graph,
        logits,
        targets,
        LossOptions {
            reduction: Reduction::None,
            class_axis: 2,
            ignore_index: Some(-100),
            label_smoothing: 0.0,
        },
    )?;
    Ok(CompiledAdamWGraph::token_mean(losses, BTreeMap::new()))
}

fn training(
    model: &CausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    context: CompiledAdamWIgnoreIndexContext,
    dropout: &mut dyn TrainingDropoutProvider,
) -> Result<CompiledAdamWGraph> {
    let mask = graph.reshape(context.validity(), [BATCH, 1, 1, TIME])?;
    let logits = model.forward_training(graph, inputs[TOKENS], mask, dropout)?;
    objective(graph, logits, inputs[TARGETS])
}

fn evaluation(
    model: &CausalTransformer,
    graph: &mut Graph,
    inputs: &BTreeMap<String, NodeId>,
    context: CompiledAdamWIgnoreIndexContext,
) -> Result<CompiledAdamWGraph> {
    let mask = graph.reshape(context.validity(), [BATCH, 1, 1, TIME])?;
    let logits = model.forward_evaluation(graph, inputs[TOKENS], mask)?;
    objective(graph, logits, inputs[TARGETS])
}

fn compile() -> std::result::Result<CompiledModuleAdamWPlan<CausalTransformer>, Box<dyn Error>> {
    let config = CompiledAdamWConfig::new(0.9, 0.999, 1e-8, 0.01)?
        .with_input_batch::<CorpusBatch>()?
        .with_token_weighted_ignore_index(TARGETS, -100)?
        .with_gradient_accumulation(2)?
        .with_max_gradient_norm(0.25)?
        .with_frozen_parameters(["positions.weight"])?
        .with_captured_multi_step_lr(CompiledMultiStepLr::new(2e-3, 0.5, [100])?)
        .with_window_loss_report();
    Ok(
        CompiledModuleAdamWPlan::compile_graph_with_dropout_and_ignore_index(
            config,
            CompiledDropoutConfig::new(CompiledDropoutKey([0xc07_001, 0xc07_002])),
            CausalTransformer::new(0xc07_000)?,
            training,
        )
        .map_err(|e| e.into_parts().1)?
        .with_evaluation_graph_and_ignore_index(evaluation)
        .map_err(|e| e.into_parts().1)?,
    )
}

fn evaluate(
    runtime: &mut CompiledModuleAdamWSession<CausalTransformer, NativeCpuCompiledAdamW<'_>>,
    corpus: &Corpus,
) -> Result<f64> {
    let mut numerator = 0.0;
    let mut count = 0_u64;
    // A fixed training-data probe, deliberately not a held-out generalization
    // claim. Weight by actual tokens rather than averaging unequal batches.
    for batch in 0..corpus.batch_count().min(4) {
        let result = runtime.evaluate_batch(corpus.batch(batch as u64)?)?;
        assert_eq!(result.report().fallback_count(), 0);
        numerator += result.loss().scalar_at(0).as_f64() * result.loss_weight() as f64;
        count += result.loss_weight();
    }
    Ok(numerator / count as f64)
}

enum Start {
    Fresh,
    Resume(PathBuf),
}

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let (start, corpus_path, output, steps) = match args.as_slice() {
        [mode, corpus, output, steps] if mode == "train" => (Start::Fresh, corpus, output, steps),
        [mode, corpus, checkpoint, output, steps] if mode == "resume" => (Start::Resume(PathBuf::from(checkpoint)), corpus, output, steps),
        _ => return Err("usage: compiled_corpus_train train CORPUS OUTPUT STEPS | resume CORPUS CHECKPOINT OUTPUT STEPS".into()),
    };
    let steps: u64 = steps.to_str().ok_or("STEPS must be UTF-8")?.parse()?;
    if !(1..=10_000).contains(&steps) {
        return Err("STEPS must be in 1..=10000".into());
    }
    let output = PathBuf::from(output);
    match fs::symlink_metadata(&output) {
        Ok(_) => return Err("output already exists; choose a new checkpoint path".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    let corpus = Corpus::load(&PathBuf::from(corpus_path))?;
    let (plan, resumed) = match start {
        Start::Fresh => (compile()?, false),
        Start::Resume(path) => {
            // Data comparison precedes executable decoding/preparation. The
            // different seed also exercises publication into a fresh owner.
            let bundle = corpus.restore(&path)?;
            (
                CompiledModuleAdamWPlan::restore_from_resume_bundle(
                    CausalTransformer::new(0xc07_999)?,
                    &bundle,
                )
                .map_err(|e| e.into_parts().1)?,
                true,
            )
        }
    };
    let artifact = plan.program_artifact()?;
    let executor = CapturedReplayExecutor::default();
    let target = NativeCpuSessionTarget::new(&executor).vectorized(true);
    let mut runtime = plan.prepare(&target).map_err(|e| e.into_parts().1)?;
    let first = runtime.checkpoint()?.info().replay_step();
    let end = first.checked_add(steps).ok_or("replay counter overflow")?;
    let initial_loss = evaluate(&mut runtime, &corpus)?;
    let mut weights = Vec::new();
    for replay in first..end {
        // `end` was checked above, so this committed replay count cannot wrap.
        let completed_replays = replay + 1;
        let step = runtime.step_batch_commit_only_scheduled(corpus.batch(replay)?)?;
        assert_eq!(step.report().fallback_count(), 0);
        assert!(step.report().executed_native_item_count() > 0);
        assert_eq!(step.optimizer_step(), completed_replays / 2);
        assert_eq!(step.accumulation_index(), completed_replays % 2);
        weights.push(step.loss_weight());
    }
    let checkpoint = runtime.module_checkpoint()?;
    let final_loss = evaluate(&mut runtime, &corpus)?;
    assert_eq!(
        runtime.module_checkpoint()?,
        checkpoint,
        "evaluation must not advance training state"
    );
    let (_, published_checkpoint) = runtime
        .finish_with_module_checkpoint()
        .map_err(|e| e.into_parts().1)?;
    assert_eq!(published_checkpoint, checkpoint);
    let bundle = CompiledAdamWResumeBundle::new(artifact, checkpoint)?;
    corpus.save(&output, &bundle)?;
    println!(
        "{}",
        serde_json::json!({
            "format_version": 1, "resumed_without_graph_rebuild": resumed,
            "first_replay": first, "completed_replays": end,
            "optimizer_steps": end / 2, "pending_microbatches": end % 2,
            "initial_training_probe_loss": initial_loss, "final_training_probe_loss": final_loss,
            "valid_tokens_per_microbatch": weights, "fallback_count": 0,
        })
    );
    Ok(())
}
