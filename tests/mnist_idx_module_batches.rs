//! Public local IDX batch materialization and module-training acceptance.

use rustgrad::nn::Linear;
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{
    BatchIter, ClassificationFeatureLayout, CpuModuleTrainer, Module, ModuleCrossEntropy, Result,
    load_mnist_idx_files, materialize_classification_batch,
};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn fixture() -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "rustgrad-mnist-module-batch-{}",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let mut images = Vec::new();
    images.extend_from_slice(&2051u32.to_be_bytes());
    images.extend_from_slice(&4u32.to_be_bytes());
    images.extend_from_slice(&2u32.to_be_bytes());
    images.extend_from_slice(&2u32.to_be_bytes());
    images.extend_from_slice(&[
        0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255,
    ]);
    let mut labels = Vec::new();
    labels.extend_from_slice(&2049u32.to_be_bytes());
    labels.extend_from_slice(&4u32.to_be_bytes());
    labels.extend_from_slice(&[0, 0, 1, 1]);
    let image_path = root.join("images.idx3-ubyte");
    let label_path = root.join("labels.idx1-ubyte");
    std::fs::write(&image_path, images).unwrap();
    std::fs::write(&label_path, labels).unwrap();
    (image_path, label_path)
}

fn materialize(
    features: &rustgrad::TensorData,
    labels: &rustgrad::TensorData,
    indices: &[usize],
) -> Result<rustgrad::ClassificationBatch> {
    materialize_classification_batch(
        features,
        labels,
        indices,
        ClassificationFeatureLayout::Flatten,
    )
}

#[test]
fn local_idx_batches_train_and_evaluate_without_graph_plumbing() -> Result<()> {
    let (image_path, label_path) = fixture();
    let dataset = load_mnist_idx_files(&image_path, &label_path).map_err(|error| {
        rustgrad::Error::Dataset {
            reason: error.to_string(),
        }
    })?;
    let features = dataset.normalized_f32()?;
    let batches = BatchIter::new(4, 3, 29, true, false)?.collect::<Vec<_>>();
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 1]);
    assert_eq!(
        batches,
        BatchIter::new(4, 3, 29, true, false)?.collect::<Vec<_>>()
    );

    let model = Linear::new_static(4, 2, true, 17)?;
    let mut optimizer = Optimizer::sgd_for_module(
        &model,
        SgdConfig {
            lr: 0.5,
            ..SgdConfig::default()
        },
    )?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.)?;
    let all = materialize(&features, &dataset.labels, &[0, 1, 2, 3])?;
    let initial = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(all.features, all.targets)?;
    let mut losses = Vec::new();
    for indices in batches.iter().cycle().take(8) {
        let batch = materialize(&features, &dataset.labels, indices)?;
        let result = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )?
        .train_step(batch.features, batch.targets)?;
        assert_eq!(
            result.parameter_versions()["weight"],
            losses.len() as u64 + 1
        );
        losses.push(result.loss());
    }
    let all = materialize(&features, &dataset.labels, &[0, 1, 2, 3])?;
    let expected = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(all.features, all.targets)?;
    assert!(losses.last() < losses.first(), "losses: {losses:?}");
    assert!(expected.loss() < initial.loss());
    let before = (
        model.state_dict()?,
        optimizer.state_dict()?,
        scheduler.state_dict()?,
    );
    let all = materialize(&features, &dataset.labels, &[0, 1, 2, 3])?;
    let repeated = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(all.features, all.targets)?;
    assert_eq!(expected.logits(), repeated.logits());
    assert_eq!(before.0, model.state_dict()?);
    assert_eq!(before.1, optimizer.state_dict()?);
    assert_eq!(before.2, scheduler.state_dict()?);
    std::fs::remove_dir_all(image_path.parent().unwrap()).unwrap();
    Ok(())
}
