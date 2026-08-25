//! Train and evaluate a small classifier without exposing Graph or gradient maps.
//!
//! `cargo run --example cpu_module_train`

use rustgrad::nn::Linear;
use rustgrad::optim::{LearningRateScheduler, Optimizer, SgdConfig};
use rustgrad::{CpuModuleTrainer, DType, Graph, ModuleCrossEntropy, Scalar, TensorData};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut construction = Graph::new();
    let model = Linear::new(&mut construction, 2, 2, true, 7)?;
    let mut optimizer = Optimizer::sgd(
        vec![
            ("weight".into(), model.weight.clone()),
            (
                "bias".into(),
                model.bias.clone().ok_or("linear bias missing")?,
            ),
        ],
        SgdConfig {
            lr: 0.25,
            ..SgdConfig::default()
        },
    )?;
    let mut scheduler = LearningRateScheduler::multi_step(vec![3], 0.5)?;
    let input = TensorData::new([4, 2], vec![-1., -1., -1., 1., 1., -1., 1., 1.])?;
    let target = TensorData::from_scalars([4], DType::U8, [0, 0, 1, 1].map(Scalar::U))?;
    for _ in 0..6 {
        let result = CpuModuleTrainer::new(
            &model,
            &mut optimizer,
            &mut scheduler,
            ModuleCrossEntropy::default(),
        )?
        .train_step(input.clone(), target.clone())?;
        println!("step {} loss {:.6}", result.optimizer_step(), result.loss());
    }
    let result = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )?
    .evaluate(input, target)?;
    println!("evaluation loss {:.6}", result.loss());
    Ok(())
}
