//! Train and evaluate a tiny configured CIFAR classifier without network access.
//!
//! `cargo run --example cifar10_local -- data_batch_1.bin data_batch_2.bin`

use rustgrad::nn::{AdaptiveAvgPool2d, Conv2d, Flatten, Linear, ReLU, Sequential};
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    BatchIter, ClassificationFeatureLayout, Conv2dOptions, CpuModuleTrainer, ModuleCrossEntropy,
    load_cifar10_files, materialize_classification_batch, summarize_classification,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = std::env::args().skip(1).collect::<Vec<_>>();
    if paths.is_empty() {
        return Err("usage: cifar10_local <batch.bin> [batch.bin ...]".into());
    }
    let dataset = load_cifar10_files(&paths)?;
    let features = dataset.normalized_f32([0.; 3], [255.; 3])?;
    let mut model = Sequential::default();
    model.push(Conv2d::new_static(
        3,
        2,
        [1, 1],
        Conv2dOptions::default(),
        true,
        31,
    )?);
    model.push(ReLU::new());
    model.push(AdaptiveAvgPool2d::new([Some(1), Some(1)]));
    model.push(Flatten::new(1));
    model.push(Linear::new_static(2, 10, true, 32)?);
    let mut optimizer = Optimizer::sgd_for_module(
        &model,
        SgdConfig {
            lr: 0.2,
            ..SgdConfig::default()
        },
    )?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let batches = BatchIter::new(dataset.labels.len(), 32, 0, true, false)?.collect::<Vec<_>>();
    for indices in &batches {
        let batch = materialize_classification_batch(
            &features,
            &dataset.labels,
            indices,
            ClassificationFeatureLayout::Preserve,
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
    let indices = (0..dataset.labels.len()).collect::<Vec<_>>();
    let batch = materialize_classification_batch(
        &features,
        &dataset.labels,
        &indices,
        ClassificationFeatureLayout::Preserve,
    )?;
    let result = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(batch.features, batch.targets.clone())?;
    let summary = summarize_classification(result.logits(), &batch.targets)?;
    println!(
        "evaluated {} CIFAR-10 images: loss {:.6}, accuracy {:?}",
        summary.total_count(),
        result.loss(),
        summary.accuracy()
    );
    Ok(())
}
