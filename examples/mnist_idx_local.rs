//! Train and evaluate a small static CPU classifier from a local MNIST IDX pair.
//!
//! `cargo run --example mnist_idx_local -- images.idx3-ubyte labels.idx1-ubyte`

use rustgrad::nn::Linear;
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    BatchIter, ClassificationFeatureLayout, CpuModuleTrainer, ModuleCrossEntropy,
    load_mnist_idx_files, materialize_classification_batch,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let images = args
        .next()
        .ok_or("usage: mnist_idx_local <images> <labels>")?;
    let labels = args
        .next()
        .ok_or("usage: mnist_idx_local <images> <labels>")?;
    if args.next().is_some() {
        return Err("usage: mnist_idx_local <images> <labels>".into());
    }
    let dataset = load_mnist_idx_files(images, labels)?;
    let features = dataset.normalized_f32()?;
    let feature_count = dataset
        .rows
        .checked_mul(dataset.cols)
        .ok_or("MNIST feature shape overflows")?;
    let model = Linear::new_static(feature_count, 10, true, 7)?;
    let mut optimizer = Optimizer::sgd_for_module(
        &model,
        SgdConfig {
            lr: 0.25,
            ..SgdConfig::default()
        },
    )?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let batches = BatchIter::new(dataset.labels.len(), 32, 0, true, false)?.collect::<Vec<_>>();
    if batches.is_empty() {
        return Err("MNIST IDX pair contains no samples to train".into());
    }
    for indices in batches.iter().cycle().take(8) {
        let batch = materialize_classification_batch(
            &features,
            &dataset.labels,
            indices,
            ClassificationFeatureLayout::Flatten,
        )?;
        let result = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )?
        .train_step(batch.features, batch.targets)?;
        println!("step {} loss {:.6}", result.optimizer_step(), result.loss());
    }
    let evaluation_indices = (0..dataset.labels.len()).collect::<Vec<_>>();
    let evaluation = materialize_classification_batch(
        &features,
        &dataset.labels,
        &evaluation_indices,
        ClassificationFeatureLayout::Flatten,
    )?;
    let result = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(evaluation.features, evaluation.targets)?;
    println!("evaluation loss {:.6}", result.loss());
    Ok(())
}
