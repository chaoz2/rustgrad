use rustgrad::nn::{Embedding, Linear, Sequential};
use rustgrad::optim::LearningRateScheduler;
use rustgrad::{
    CapturedReplayExecutor, CpuModuleTrainer, DType, Error, Flatten, Graph, Module,
    ModuleCrossEntropy, Optimizer, Scalar, SgdConfig, Shape, TensorData, infer_module_cpu,
    infer_module_native_cpu,
};

fn tokens(values: [i64; 2]) -> TensorData {
    TensorData::from_scalars(Shape::from([2]), DType::I32, values.map(Scalar::I)).unwrap()
}

fn targets() -> TensorData {
    TensorData::from_scalars(Shape::from([2]), DType::I64, [Scalar::I(0), Scalar::I(1)]).unwrap()
}

fn model(seed: u64) -> (Sequential, rustgrad::Parameter) {
    let embedding = Embedding::new_static(4, 2, None, seed).unwrap();
    let embedding_weight = embedding.weight.clone();
    let mut model = Sequential::default();
    model.push(embedding);
    model.push(Flatten::new(1));
    model.push(Linear::new_static(4, 2, true, seed + 1).unwrap());
    (model, embedding_weight)
}

#[test]
fn embedding_static_constructor_and_typed_cpu_workflow_are_explicit() {
    let mut setup = Graph::new();
    let legacy = Embedding::new(&mut setup, 4, 2, None, 71).unwrap();
    let static_module = Embedding::new_static(4, 2, None, 71).unwrap();
    assert_eq!(
        legacy.state_dict().unwrap(),
        static_module.state_dict().unwrap()
    );

    let (model, embedding_weight) = model(73);
    assert_eq!(
        model
            .state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["0.weight", "2.bias", "2.weight"]
    );
    let input = tokens([1, 1]);
    let first = infer_module_cpu(&model, input.clone()).unwrap();
    let second = infer_module_cpu(&model, input.clone()).unwrap();
    assert_eq!(first.output(), second.output());
    assert_eq!(first.trace(), second.trace());
    assert_eq!(first.output().shape().dims(), &[2, 2]);

    let before = model.state_dict().unwrap();
    let mut optimizer = Optimizer::sgd_for_module(&model, SgdConfig::default()).unwrap();
    let mut scheduler = LearningRateScheduler::multi_step(vec![], 1.).unwrap();
    let mut trainer = CpuModuleTrainer::new(
        &model,
        &mut optimizer,
        &mut scheduler,
        ModuleCrossEntropy::default(),
    )
    .unwrap();
    let evaluation = trainer.evaluate(input.clone(), targets()).unwrap();
    assert!(evaluation.loss().is_finite());
    assert_eq!(before, model.state_dict().unwrap());
    let old_embedding = embedding_weight.value().unwrap();
    let trained = trainer.train_step(input, targets()).unwrap();
    assert!(trained.loss().is_finite());
    assert_eq!(trained.optimizer_step(), 1);
    assert_eq!(trained.scheduler_epoch(), 1);
    assert_eq!(embedding_weight.version().unwrap(), 1);
    let new_embedding = embedding_weight.value().unwrap();
    assert_ne!(old_embedding.to_vec_f64()[2], new_embedding.to_vec_f64()[2]);
    assert_eq!(old_embedding.to_vec_f64()[6], new_embedding.to_vec_f64()[6]);
}

#[test]
fn embedding_cpu_input_contract_rejects_before_mutation_or_native_cache() {
    let (model, _) = model(79);
    let before = model.state_dict().unwrap();
    assert!(matches!(
        infer_module_cpu(&model, TensorData::new([2], vec![0.; 2]).unwrap()),
        Err(Error::SessionTraining { .. })
    ));
    assert!(
        infer_module_cpu(
            &model,
            TensorData::from_scalars(Shape::from([2]), DType::I32, [Scalar::I(0), Scalar::I(4)],)
                .unwrap(),
        )
        .is_err()
    );
    assert_eq!(before, model.state_dict().unwrap());

    let linear = Linear::new_static(2, 2, true, 83).unwrap();
    assert!(matches!(
        infer_module_cpu(
            &linear,
            TensorData::from_scalars(
                Shape::from([1, 2]),
                DType::F64,
                [Scalar::F(0.), Scalar::F(1.)],
            )
            .unwrap(),
        ),
        Err(Error::SessionTraining { .. })
    ));

    let executor = CapturedReplayExecutor::default();
    assert!(matches!(
        infer_module_native_cpu(&model, tokens([1, 1]), &executor, false),
        Err(Error::SessionTraining { .. })
    ));
    assert_eq!(executor.compile_cache_len(false), 0);
}
