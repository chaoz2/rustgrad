use super::*;
use crate::{Backend, CpuBackend, DType, Graph, NodeId, Scalar, Storage, TensorData};

fn f32s(data: &TensorData) -> Vec<f32> {
    match data.storage() {
        Storage::F32(v) => v.clone(),
        _ => panic!("expected f32"),
    }
}
fn execute(
    graph: &Graph,
    output: NodeId,
    module: &impl Module,
    input: (&str, TensorData),
) -> TensorData {
    let mut bindings = module.input_bindings(graph).unwrap();
    bindings.insert(input.0.into(), input.1);
    CpuBackend.execute(graph, output, &bindings).unwrap()
}

#[test]
fn embedding_norm_and_dropout_have_expected_semantics() {
    let mut graph = Graph::new();
    let embedding = Embedding::new(&mut graph, 3, 2, Some(0), 1).unwrap();
    embedding
        .weight
        .replace(TensorData::new([3, 2], vec![9., 9., 1., 2., 3., 4.]).unwrap())
        .unwrap();
    let indices = graph.input_dtype("i", [2], DType::I32);
    let out = embedding.forward(&mut graph, indices).unwrap();
    assert_eq!(
        f32s(&execute(
            &graph,
            out,
            &embedding,
            (
                "i",
                TensorData::from_scalars([2], DType::I32, [Scalar::I(0), Scalar::I(2)]).unwrap()
            )
        )),
        vec![0., 0., 3., 4.]
    );
    let mut dropout_graph = Graph::new();
    let dropout = Dropout::new(0.5, true, 42).unwrap();
    let x = dropout_graph.input("x", [4]);
    let a = dropout.forward(&mut dropout_graph, x).unwrap();
    let b = dropout.forward(&mut dropout_graph, x).unwrap();
    let data = TensorData::new([4], vec![1.; 4]).unwrap();
    assert_eq!(
        execute(&dropout_graph, a, &dropout, ("x", data.clone())),
        execute(&dropout_graph, b, &dropout, ("x", data))
    );
    let mut norm_graph = Graph::new();
    let norm = RMSNorm::new(&mut norm_graph, 2, 1e-6, false).unwrap();
    let nx = norm_graph.input("nx", [1, 2]);
    let no = norm.forward(&mut norm_graph, nx).unwrap();
    let values = f32s(&execute(
        &norm_graph,
        no,
        &norm,
        ("nx", TensorData::new([1, 2], vec![3., 4.]).unwrap()),
    ));
    assert!((values[0] - 0.848_528_1).abs() < 1e-5 && (values[1] - 1.131_370_9).abs() < 1e-5);
}

#[test]
fn explicit_mode_dropout_is_eval_identity_and_deterministic_training_composition() {
    let dropout = ModeDropout::new(0.5, 97).unwrap();
    assert!(dropout.state_dict().unwrap().tensors().is_empty());
    assert!(ModeDropout::new(-0.1, 1).is_err());
    let mut graph = Graph::new();
    let input = graph.input("x", [4]);
    let eval = dropout.forward_mode(&mut graph, input, Mode::Eval).unwrap();
    assert_eq!(eval.output, input);
    assert!(eval.pending.is_empty());
    let first = dropout
        .forward_mode(&mut graph, input, Mode::Training)
        .unwrap();
    let second = dropout
        .forward_mode(&mut graph, input, Mode::Training)
        .unwrap();
    assert!(first.pending.is_empty());
    assert!(second.pending.is_empty());
    let values = TensorData::new([4], vec![1.; 4]).unwrap();
    let first_values = execute(&graph, first.output, &dropout, ("x", values.clone()));
    let second_values = execute(&graph, second.output, &dropout, ("x", values.clone()));
    assert_eq!(first_values, second_values);
    assert_eq!(execute(&graph, eval.output, &dropout, ("x", values)), TensorData::new([4], vec![1.; 4]).unwrap());

    let mut chain = ModeSequential::default();
    chain.push(ModeDropout::new(0.5, 97).unwrap());
    let mut chain_graph = Graph::new();
    let chain_input = chain_graph.input("x", [4]);
    let chain_eval = chain
        .forward_mode(&mut chain_graph, chain_input, Mode::Eval)
        .unwrap();
    assert_eq!(chain_eval.output, chain_input);
    assert!(chain_eval.pending.is_empty());
}

#[test]
fn convolution_and_pooling_modules_are_stateful_only_at_parameters() {
    let mut graph = Graph::new();
    let conv = Conv2d::new(
        &mut graph,
        1,
        1,
        [2, 2],
        crate::Conv2dOptions::default(),
        true,
        7,
    )
    .unwrap();
    conv.weight
        .replace(TensorData::new([1, 1, 2, 2], vec![1., 0., 0., 1.]).unwrap())
        .unwrap();
    conv.bias
        .as_ref()
        .unwrap()
        .replace(TensorData::new([1], vec![1.]).unwrap())
        .unwrap();
    let x = graph.input("x", [1, 1, 3, 3]);
    let y = conv.forward(&mut graph, x).unwrap();
    assert_eq!(
        f32s(&execute(
            &graph,
            y,
            &conv,
            (
                "x",
                TensorData::new([1, 1, 3, 3], (1..=9).map(|x| x as f32).collect()).unwrap()
            )
        )),
        vec![7., 9., 13., 15.]
    );
    assert_eq!(
        conv.state_dict()
            .unwrap()
            .tensors()
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["bias", "weight"]
    );

    let mut one_d_graph = Graph::new();
    let one_d = Conv1d::new(
        &mut one_d_graph,
        1,
        1,
        2,
        Conv1dOptions::default(),
        false,
        1,
    )
    .unwrap();
    one_d
        .weight
        .replace(TensorData::new([1, 1, 2], vec![2., 1.]).unwrap())
        .unwrap();
    let x = one_d_graph.input("x", [1, 1, 3]);
    let y = one_d.forward(&mut one_d_graph, x).unwrap();
    assert_eq!(
        f32s(&execute(
            &one_d_graph,
            y,
            &one_d,
            ("x", TensorData::new([1, 1, 3], vec![1., 2., 3.]).unwrap())
        )),
        vec![4., 7.]
    );

    let pool = MaxPool2d::new(crate::Pool2dOptions::default());
    let mut pool_graph = Graph::new();
    let px = pool_graph.input("p", [1, 1, 2, 2]);
    let pooled = pool.forward_with_indices(&mut pool_graph, px).unwrap();
    let bindings = std::collections::HashMap::from([(
        "p".into(),
        TensorData::new([1, 1, 2, 2], vec![1., 4., 3., 2.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&pool_graph, pooled.values, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        4.
    );
    assert_eq!(
        CpuBackend
            .execute(&pool_graph, pooled.indices, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_i64(),
        1
    );
    assert!(pool.state_dict().unwrap().tensors().is_empty());
}
