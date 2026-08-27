use super::*;
use crate::{Backend, CpuBackend, DType, Error, Graph, NodeId, Scalar, Storage, TensorData};

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
fn dropout_revalidates_public_probability_before_graph_work() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2]);
    let before = graph.node_count();
    let mut dropout = Dropout::new(0.5, true, 42).unwrap();

    dropout.probability = f64::NAN;
    assert!(matches!(
        dropout.forward(&mut graph, input),
        Err(Error::UnsupportedDropout { .. })
    ));
    assert_eq!(graph.node_count(), before);

    dropout.probability = 1.5;
    assert!(matches!(
        dropout.forward(&mut graph, input),
        Err(Error::UnsupportedDropout { .. })
    ));
    assert_eq!(graph.node_count(), before);

    dropout.probability = 0.5;
    assert!(dropout.forward(&mut graph, input).is_ok());
    assert!(graph.node_count() > before);
}

#[test]
fn embedding_preflights_geometry_and_index_dtype_before_binding_weight() {
    let mut graph = Graph::new();
    assert!(Embedding::new(&mut graph, 0, 2, None, 1).is_err());
    assert!(Embedding::new(&mut graph, 2, 0, None, 1).is_err());
    assert!(Embedding::new(&mut graph, usize::MAX, 1, None, 1).is_err());

    let embedding = Embedding::new(&mut graph, 3, 2, None, 2).unwrap();
    let float_index = graph.input("float_index", [2]);
    assert!(embedding.forward(&mut graph, float_index).is_err());
    assert!(graph.parameter_bindings().is_empty());
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

#[test]
fn conv2d_constructor_rejects_zero_execution_geometry() {
    let mut graph = Graph::new();
    assert!(Conv2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::Conv2dOptions {
            stride: [0, 1],
            ..crate::Conv2dOptions::default()
        },
        true,
        1,
    )
    .is_err());
    assert!(Conv2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::Conv2dOptions {
            dilation: [1, 0],
            ..crate::Conv2dOptions::default()
        },
        true,
        1,
    )
    .is_err());
    assert!(graph.parameter_bindings().is_empty());
    let layer = Conv2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::Conv2dOptions::default(),
        true,
        2,
    )
    .unwrap();
    assert_eq!(layer.weight.shape().unwrap().dims(), &[4, 2, 3, 2]);
}

#[test]
fn transpose_conv2d_preflights_geometry_and_input_before_parameter_binding() {
    let mut graph = Graph::new();
    assert!(ConvTranspose2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::ConvTranspose2dOptions {
            stride: [0, 1],
            ..crate::ConvTranspose2dOptions::default()
        },
        true,
        1,
    )
    .is_err());
    assert!(ConvTranspose2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::ConvTranspose2dOptions {
            stride: [1, 1],
            output_padding: [1, 0],
            ..crate::ConvTranspose2dOptions::default()
        },
        true,
        1,
    )
    .is_err());

    let layer = ConvTranspose2d::new(
        &mut graph,
        2,
        4,
        [3, 2],
        crate::ConvTranspose2dOptions {
            groups: 2,
            stride: [2, 1],
            output_padding: [1, 0],
            ..crate::ConvTranspose2dOptions::default()
        },
        true,
        2,
    )
    .unwrap();
    assert_eq!(layer.weight.shape().unwrap().dims(), &[2, 2, 3, 2]);
    let wrong_rank = graph.input("wrong_rank", [1, 2, 2]);
    assert!(layer.forward(&mut graph, wrong_rank).is_err());
    assert!(graph.parameter_bindings().is_empty());
    let wrong_channels = graph.input("wrong_channels", [1, 1, 2, 2]);
    assert!(layer.forward(&mut graph, wrong_channels).is_err());
    assert!(graph.parameter_bindings().is_empty());
    let input = graph.input("x", [1, 2, 2, 2]);
    let output = layer.forward(&mut graph, input).unwrap();
    assert_eq!(graph.shape(output).unwrap().dims(), &[1, 4, 6, 3]);
}

#[test]
fn transpose_conv1d_preflights_input_before_parameter_binding() {
    let mut graph = Graph::new();
    let layer = ConvTranspose1d::new(
        &mut graph,
        2,
        4,
        3,
        crate::ConvTranspose1dOptions {
            groups: 2,
            stride: 2,
            output_padding: 1,
            ..crate::ConvTranspose1dOptions::default()
        },
        true,
        3,
    )
    .unwrap();
    assert_eq!(layer.weight.shape().unwrap().dims(), &[2, 2, 3]);
    let wrong_rank = graph.input("wrong_rank", [1, 2]);
    assert!(layer.forward(&mut graph, wrong_rank).is_err());
    assert!(graph.parameter_bindings().is_empty());
    let wrong_channels = graph.input("wrong_channels", [1, 1, 3]);
    assert!(layer.forward(&mut graph, wrong_channels).is_err());
    assert!(graph.parameter_bindings().is_empty());
    let input = graph.input("x", [1, 2, 3]);
    let output = layer.forward(&mut graph, input).unwrap();
    assert_eq!(graph.shape(output).unwrap().dims(), &[1, 4, 8]);
}
