use super::*;
use crate::{
    Backend, CpuBackend, DType, Error, Float8Format, Float8Storage, Scalar, Shape, Storage,
    TensorData,
};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

fn bool_data(shape: impl Into<Shape>, values: impl IntoIterator<Item = bool>) -> TensorData {
    TensorData::from_scalars(shape, DType::Bool, values.into_iter().map(Scalar::Bool)).unwrap()
}

#[test]
fn dtype_conveniences_alias_source_cast_and_are_atomic() {
    let mut graph = Graph::new();
    let f32_input = graph.input_dtype_requires_grad("f32", [], DType::F32, true);
    let f16_empty = graph.input_dtype_requires_grad("f16_empty", [0, 2], DType::F16, true);
    let bf16_empty = graph.input_dtype_requires_grad("bf16_empty", [0], DType::BF16, true);
    let f64_scalar = graph.input_dtype_requires_grad("f64_scalar", [], DType::F64, true);
    let i16_scalar = graph.input_dtype("i16_scalar", [], DType::I16);
    let integer = graph.input_dtype("integer", [2], DType::I64);
    let boolean = graph.input_dtype("boolean", [1], DType::Bool);
    let before = graph.node_count();

    assert!(graph.is_floating_point(f32_input).unwrap());
    assert!(graph.is_floating_point(f16_empty).unwrap());
    assert!(!graph.is_floating_point(integer).unwrap());
    assert!(!graph.is_floating_point(boolean).unwrap());
    assert_eq!(graph.to_f32(f32_input).unwrap(), f32_input);
    assert_eq!(graph.to_bf16(bf16_empty).unwrap(), bf16_empty);
    assert_eq!(graph.to_f64(f64_scalar).unwrap(), f64_scalar);
    assert_eq!(graph.to_i64(integer).unwrap(), integer);
    assert_eq!(graph.to_i16(i16_scalar).unwrap(), i16_scalar);

    let widened = graph.to_f32(f16_empty).unwrap();
    assert_eq!(graph.shape(widened).unwrap(), &Shape::new([0, 2]));
    assert_eq!(graph.dtype(widened).unwrap(), DType::F32);
    assert!(
        matches!(graph.op(widened).unwrap(), Op::Cast { input, dtype: DType::F32 } if *input == f16_empty)
    );
    let widened_loss = graph.sum_all(widened).unwrap();
    assert!(graph.grad(widened_loss, f16_empty).is_ok());

    let half = graph.to_f16(integer).unwrap();
    let bfloat = graph.to_bf16(f32_input).unwrap();
    let double = graph.to_f64(f16_empty).unwrap();
    let int = graph.to_i32(f32_input).unwrap();
    let long = graph.to_i64(boolean).unwrap();
    let short = graph.to_i16(integer).unwrap();
    let bool_value = graph.to_bool(integer).unwrap();
    assert_eq!(graph.dtype(half).unwrap(), DType::F16);
    assert_eq!(graph.dtype(bfloat).unwrap(), DType::BF16);
    assert_eq!(graph.dtype(double).unwrap(), DType::F64);
    assert_eq!(graph.dtype(int).unwrap(), DType::I32);
    assert_eq!(graph.dtype(long).unwrap(), DType::I64);
    assert_eq!(graph.dtype(short).unwrap(), DType::I16);
    assert_eq!(graph.dtype(bool_value).unwrap(), DType::Bool);
    assert!(
        matches!(graph.op(half).unwrap(), Op::Cast { input, dtype: DType::F16 } if *input == integer)
    );
    assert!(
        matches!(graph.op(bfloat).unwrap(), Op::Cast { input, dtype: DType::BF16 } if *input == f32_input)
    );
    assert!(
        matches!(graph.op(double).unwrap(), Op::Cast { input, dtype: DType::F64 } if *input == f16_empty)
    );
    assert!(
        matches!(graph.op(int).unwrap(), Op::Cast { input, dtype: DType::I32 } if *input == f32_input)
    );
    assert!(
        matches!(graph.op(long).unwrap(), Op::Cast { input, dtype: DType::I64 } if *input == boolean)
    );
    assert!(
        matches!(graph.op(short).unwrap(), Op::Cast { input, dtype: DType::I16 } if *input == integer)
    );
    assert!(
        matches!(graph.op(bool_value).unwrap(), Op::Cast { input, dtype: DType::Bool } if *input == integer)
    );
    assert!(graph.grad(bfloat, f32_input).is_ok());
    let double_loss = graph.sum_all(double).unwrap();
    assert!(graph.grad(double_loss, f16_empty).is_ok());
    assert!(graph.node_count() > before);

    let unknown = NodeId::from_index(usize::MAX);
    let failed_before = graph.node_count();
    assert!(matches!(
        graph.is_floating_point(unknown),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(graph.to_f32(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(graph.to_bf16(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(graph.to_f64(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(graph.to_i64(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(graph.to_i16(unknown), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), failed_before);

    let overflow = graph.input_dtype("overflow", [usize::MAX, 2], DType::F32);
    let overflow_before = graph.node_count();
    assert!(matches!(
        graph.to_f16(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert!(matches!(
        graph.to_bf16(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert!(matches!(
        graph.to_f64(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert!(matches!(
        graph.to_i64(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert!(matches!(
        graph.to_i16(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), overflow_before);
}

#[test]
fn descriptor_queries_are_read_only_and_source_axis_checked() {
    let mut graph = Graph::new();
    let scalar = graph.input_dtype("scalar", [], DType::Bool);
    let vector = graph.input_dtype("vector", [4], DType::I32);
    let zero_extent = graph.input_dtype("zero_extent", [2, 0, 3], DType::F16);
    let leading_zero = graph.input_dtype("leading_zero", [0, 3], DType::F64);
    let before = graph.node_count();

    assert_eq!(graph.numel(scalar).unwrap(), 1);
    assert_eq!(graph.ndim(scalar).unwrap(), 0);
    assert_eq!(graph.max_shape(scalar).unwrap(), Shape::new([]));
    assert_eq!(graph.max_numel(scalar).unwrap(), 1);
    assert_eq!(graph.size(scalar).unwrap(), Shape::new([]));
    assert_eq!(graph.len_tinygrad(vector).unwrap(), 4);
    assert_eq!(graph.numel(zero_extent).unwrap(), 0);
    assert_eq!(graph.ndim(zero_extent).unwrap(), 3);
    assert_eq!(graph.max_shape(zero_extent).unwrap(), Shape::new([2, 0, 3]));
    assert_eq!(graph.max_numel(zero_extent).unwrap(), 0);
    assert_eq!(graph.size(zero_extent).unwrap(), Shape::new([2, 0, 3]));
    assert_eq!(graph.len_tinygrad(zero_extent).unwrap(), 2);
    assert_eq!(graph.len_tinygrad(leading_zero).unwrap(), 0);
    assert_eq!(graph.size_dim(zero_extent, -3).unwrap(), 2);
    assert_eq!(graph.size_dim(zero_extent, -1).unwrap(), 3);
    assert_eq!(graph.element_size(scalar).unwrap(), DType::Bool.itemsize());

    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let input = graph.input_dtype(format!("dtype_{dtype:?}"), [0], dtype);
        let query_before = graph.node_count();
        assert_eq!(graph.element_size(input).unwrap(), dtype.itemsize());
        assert_eq!(graph.node_count(), query_before);
    }

    assert!(matches!(
        graph.size_dim(scalar, 0),
        Err(Error::InvalidAxis { .. })
    ));
    let scalar_len_error = graph.len_tinygrad(scalar).unwrap_err();
    assert!(matches!(
        &scalar_len_error,
        Error::InvalidTensorLen { node } if *node == scalar
    ));
    assert_eq!(scalar_len_error.to_string(), "len() of a 0-d tensor");
    assert!(matches!(
        graph.size_dim(zero_extent, 3),
        Err(Error::InvalidAxis { .. })
    ));
    let unknown = NodeId::from_index(usize::MAX);
    assert!(matches!(graph.numel(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(graph.ndim(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(
        graph.max_shape(unknown),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(
        graph.max_numel(unknown),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(graph.size(unknown), Err(Error::UnknownNode(_))));
    assert!(matches!(
        graph.size_dim(unknown, 0),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(
        graph.element_size(unknown),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(
        graph.len_tinygrad(unknown),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), before + 13);

    let overflow = graph.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let overflow_before = graph.node_count();
    assert!(matches!(
        graph.numel(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.ndim(overflow).unwrap(), 2);
    assert_eq!(
        graph.max_shape(overflow).unwrap(),
        Shape::new([usize::MAX, 2])
    );
    assert!(matches!(
        graph.max_numel(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert!(matches!(
        graph.nbytes(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.len_tinygrad(overflow).unwrap(), usize::MAX);
    assert_eq!(graph.node_count(), overflow_before);
}

#[test]
fn tensor_bool_tinygrad_is_always_undefined_after_node_validation() {
    let mut graph = Graph::new();
    let scalar = graph.input_dtype("scalar", [], DType::Bool);
    let vector = graph.input_dtype("vector", [2], DType::I32);
    let empty = graph.input_dtype("empty", [0, 3], DType::F16);
    let overflow = graph.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = graph.node_count();

    for id in [scalar, vector, empty, overflow] {
        let error = graph.bool_tinygrad(id).unwrap_err();
        assert_eq!(error, Error::TensorBoolNotDefined);
        assert_eq!(error.to_string(), "__bool__ on Tensor is not defined");
        assert_eq!(graph.node_count(), before);
    }

    let unknown = NodeId::from_index(usize::MAX);
    assert!(matches!(
        graph.bool_tinygrad(unknown),
        Err(Error::UnknownNode(node)) if node == unknown
    ));
    assert_eq!(graph.node_count(), before);
}

#[test]
fn sequential_is_heterogeneous_ordered_and_preserves_prefix_failures() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("x", [], DType::F32, true);
    let identity = graph
        .sequential(input, Vec::<GraphSequentialTransform>::new())
        .unwrap();
    assert_eq!(identity, input);
    let transforms: Vec<GraphSequentialTransform> = vec![
        Box::new(|g: &mut Graph, x: NodeId| g.mul_scalar(x, Scalar::F(2.0))),
        Box::new(|g: &mut Graph, x: NodeId| g.add_scalar(x, Scalar::F(1.0))),
    ];
    let output = graph.sequential(input, transforms).unwrap();
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: crate::BinaryOp::Add,
            ..
        }
    ));
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(graph.grad(output, input).is_ok());

    let invoked_later = Rc::new(Cell::new(false));
    let later = invoked_later.clone();
    let before = graph.node_count();
    let transforms: Vec<GraphSequentialTransform> = vec![
        Box::new(|g, x| g.add_scalar(x, Scalar::F(3.0))),
        Box::new(|_, _| {
            Err(Error::InvalidRandom {
                reason: "sequential stop",
            })
        }),
        Box::new(move |_, _| {
            later.set(true);
            Ok(input)
        }),
    ];
    assert!(graph.sequential(input, transforms).is_err());
    assert_eq!(graph.node_count(), before + 2);
    assert!(!invoked_later.get());
}

#[test]
fn masked_fill_matches_select_broadcasts_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let mask = graph.constant(bool_data([3], [true, false, true]));
    let fill = graph.constant(TensorData::scalar(-4.0));
    let output = graph.masked_fill(input, mask, fill).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);

    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-4., 2., -4., -4., 5., -4.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0., 1., 0., 0., 1., 0.]
    );
}

#[test]
fn masked_fill_rejects_nonboolean_mask_without_allocating_a_node() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2]);
    let nonboolean_mask = graph.input_dtype("mask", [2], DType::I32);
    let fill = graph.constant(TensorData::scalar(0.0));
    let node_count = graph.node_count();

    assert!(matches!(
        graph.masked_fill(input, nonboolean_mask, fill),
        Err(Error::InvalidLogicalDType {
            op: "select",
            actual: DType::I32,
        })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn masked_fill_scalar_commits_the_value_before_literal_mask_where() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let mask = graph.input_dtype("mask", [2], DType::Bool);
        let output = graph.masked_fill_scalar(input, mask, value).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        assert!(matches!(graph.op(NodeId(2)).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        let Op::Select {
            condition,
            on_true,
            on_false,
        } = graph.op(output).unwrap()
        else {
            panic!("masked_fill scalar must lower directly through Select");
        };
        assert_eq!(*condition, mask);
        assert_eq!(*on_true, NodeId(2));
        assert_eq!(*on_false, input);
        if dtype.is_float() {
            let Op::Constant(data) = graph.op(NodeId(2)).unwrap() else {
                panic!("prepared scalar must be a constant");
            };
            assert_eq!(data.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        }
    }

    // Python integer/float values are weak against the input branch. Bool
    // therefore lifts to I32/F32; a live I64/U64 value remains the distinct
    // F32 source bridge in the backward-compatible tensor form.
    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [2, 1], DType::Bool);
    let mask = mixed.input_dtype("mask", [1, 2], DType::Bool);
    let integer = mixed
        .masked_fill_scalar(boolean, mask, Scalar::I(1))
        .unwrap();
    let floating = mixed
        .masked_fill_scalar(boolean, mask, Scalar::F(-0.0))
        .unwrap();
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let narrow_mask = mixed.input_dtype("narrow_mask", [], DType::Bool);
    let narrow_integer = mixed
        .masked_fill_scalar(narrow, narrow_mask, Scalar::I(1))
        .unwrap();
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let integral_mask = mixed.input_dtype("integral_mask", [], DType::Bool);
    let integral_float = mixed
        .masked_fill_scalar(integral, integral_mask, Scalar::F(-0.0))
        .unwrap();
    assert_eq!(mixed.shape(integer).unwrap(), &Shape::new([2, 2]));
    assert_eq!(mixed.dtype(integer).unwrap(), DType::I32);
    assert_eq!(mixed.dtype(floating).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(narrow_integer).unwrap(), DType::F16);
    assert_eq!(mixed.dtype(integral_float).unwrap(), DType::F32);

    let input = mixed.input_dtype("i64", [2], DType::I64);
    let bridge_mask = mixed.input_dtype("bridge_mask", [2], DType::Bool);
    let value = mixed.input_dtype("u64", [2], DType::U64);
    let bridged = mixed.masked_fill(input, bridge_mask, value).unwrap();
    assert_eq!(mixed.dtype(bridged).unwrap(), DType::F32);
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input: source, dtype: DType::F32 } if *source == input
    )));
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input: source, dtype: DType::F32 } if *source == value
    )));

    // The Select branch ordering is the observable payload rule: a scalar
    // signed zero or NaN occupies only true lanes, while matching-dtype input
    // payloads stay untouched on false lanes.
    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [], DType::F64);
    let mask = specials.input_dtype("mask", [], DType::Bool);
    let negative_zero = specials
        .masked_fill_scalar(input, mask, Scalar::F(-0.0))
        .unwrap();
    let nan = specials
        .masked_fill_scalar(input, mask, Scalar::F(f64::NAN))
        .unwrap();
    let Op::Select {
        on_true, on_false, ..
    } = specials.op(negative_zero).unwrap()
    else {
        unreachable!()
    };
    assert!(matches!(specials.op(*on_true).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));
    assert_eq!(*on_false, input);
    assert!(matches!(specials.op(nan).unwrap(), Op::Select { .. }));

    let mut vjp = Graph::new();
    let input = vjp.input_dtype("input", [2, 1], DType::F32);
    let mask = vjp.input_dtype("mask", [1, 3], DType::Bool);
    let output = vjp
        .masked_fill_scalar(input, mask, Scalar::F(-0.0))
        .unwrap();
    let loss = vjp.sum_all(output).unwrap();
    let gradient = vjp.grad(loss, input).unwrap();
    assert_eq!(vjp.shape(gradient).unwrap(), &Shape::new([2, 1]));
    assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let mask = empty.input_dtype("mask", [1, 2], DType::Bool);
    let output = empty
        .masked_fill_scalar(input, mask, Scalar::F(-0.0))
        .unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.masked_fill_scalar(NodeId(usize::MAX), NodeId(0), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let input = malformed.input_dtype("input", [2], DType::F32);
    let nonboolean = malformed.input_dtype("mask", [2], DType::I32);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.masked_fill_scalar(input, nonboolean, Scalar::F(0.0)),
        Err(Error::InvalidLogicalDType {
            op: "select",
            actual: DType::I32
        })
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let mask = malformed.input_dtype("overflow_mask", [1, 2], DType::Bool);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.masked_fill_scalar(overflow, mask, Scalar::F(0.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn allclose_matches_tinygrad_isclose_then_all_for_broadcast_special_and_empty_domains() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2, 1]);
    let rhs = graph.input("rhs", [1, 3]);
    let output = graph.allclose(lhs, rhs, 1e-5, 1e-8, false).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!(
        CpuBackend
            .execute(
                &graph,
                output,
                &HashMap::from([
                    (
                        "lhs".into(),
                        TensorData::new([2, 1], vec![1.0, 1.0]).unwrap()
                    ),
                    (
                        "rhs".into(),
                        TensorData::new([1, 3], vec![1.0, 1.000_005, 1.0]).unwrap(),
                    ),
                ]),
            )
            .unwrap()
            .scalar_at(0)
            .as_bool()
    );

    let mut specials = Graph::new();
    let lhs = specials.input("lhs", [3]);
    let rhs = specials.input("rhs", [3]);
    let unequal_nan = specials.allclose(lhs, rhs, 0.0, 0.0, false).unwrap();
    let equal_nan = specials.allclose(lhs, rhs, 0.0, 0.0, true).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(-0.0),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(0.0),
                ],
            )
            .unwrap(),
        ),
    ]);
    assert!(
        !CpuBackend
            .execute(&specials, unequal_nan, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_bool()
    );
    assert!(
        CpuBackend
            .execute(&specials, equal_nan, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_bool()
    );

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 3], DType::BF16);
    let rhs = empty.input_dtype("rhs", [1, 3], DType::BF16);
    let output = empty.allclose(lhs, rhs, 1e-5, 1e-8, false).unwrap();
    assert!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([
                    (
                        "lhs".into(),
                        TensorData::from_scalars([0, 3], DType::BF16, []).unwrap(),
                    ),
                    (
                        "rhs".into(),
                        TensorData::from_scalars(
                            [1, 3],
                            DType::BF16,
                            [Scalar::F(1.0), Scalar::F(-0.0), Scalar::F(f64::INFINITY)],
                        )
                        .unwrap(),
                    ),
                ]),
            )
            .unwrap()
            .scalar_at(0)
            .as_bool()
    );
}

#[test]
fn allclose_commits_tolerances_at_rhs_width_and_preflights_before_constants() {
    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [1], DType::F64);
    let rhs = narrow.input_dtype("rhs", [1], DType::BF16);
    let output = narrow.allclose(lhs, rhs, 0.125, 0.25, false).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::Bool);
    assert!(narrow.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == DType::BF16)
    }));

    let mut malformed = Graph::new();
    let lhs = malformed.input("lhs", [2]);
    let rhs = malformed.input("rhs", [3]);
    let before = malformed.node_count();
    assert!(malformed.allclose(lhs, rhs, 1e-5, 1e-8, false).is_err());
    assert_eq!(malformed.node_count(), before);

    let overflow = malformed.input_dtype("overflow", [usize::MAX], DType::F64);
    let scalar = malformed.input_dtype("scalar", [], DType::F64);
    let before = malformed.node_count();
    assert!(
        malformed
            .allclose(overflow, scalar, 1e-5, 1e-8, false)
            .is_err()
    );
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn allclose_default_uses_scalar_isclose_then_bool_all() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 1], DType::F64);
    let rhs = graph.input_dtype("rhs", [1, 3], DType::BF16);
    let output = graph.allclose_default(lhs, rhs).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Reduce {
            kind: ReduceKind::Product,
            ..
        }
    ));
    // Defaults are weak Python floats committed independently at other.abs()
    // storage, and false remains a graph Bool scalar in the literal isclose.
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.op,
        Op::Constant(data) if data.shape() == &Shape::new([]) && data.dtype() == DType::BF16))
            .count(),
        2
    );
    assert!(graph.nodes.iter().any(|node| matches!(&node.op,
        Op::Constant(data) if data.dtype() == DType::Bool && !data.scalar_at(0).as_bool())));
    assert!(matches!(
        graph.grad(output, lhs),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.allclose_default(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::Bool);
    assert!((0..bridge.node_count()).any(|index| matches!(
        bridge.op(NodeId(index)).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    )));

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 2], DType::F16);
    let rhs = empty.input_dtype("rhs", [1, 2], DType::F16);
    let output = empty.allclose_default(lhs, rhs).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(empty.dtype(output).unwrap(), DType::Bool);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.allclose_default(NodeId(usize::MAX), NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let scalar = malformed.input_dtype("scalar", [], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.allclose_default(overflow, scalar),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn isclose_scalar_matches_tinygrad_defaults_and_branch_local_weak_widths() {
    let mut defaults = Graph::new();
    let lhs = defaults.input_dtype("lhs", [2, 1], DType::F64);
    let rhs = defaults.input_dtype("rhs", [1, 3], DType::BF16);
    let output = defaults.isclose_default(lhs, rhs).unwrap();
    assert_eq!(defaults.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(defaults.dtype(output).unwrap(), DType::Bool);
    // Both weak Python floats are committed at other.abs()'s BF16 branch,
    // not at self-other's F64 difference branch.
    assert_eq!(
        defaults
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Op::Constant(data) if data.dtype() == DType::BF16))
            .count(),
        2
    );

    let mut custom = Graph::new();
    let lhs = custom.input_dtype("lhs", [], DType::I64);
    let rhs = custom.input_dtype("rhs", [], DType::U64);
    let output = custom.isclose_scalar(lhs, rhs, 0.125, 0.25, true).unwrap();
    assert_eq!(custom.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(custom.dtype(output).unwrap(), DType::Bool);
    assert_eq!(
        custom
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Op::Constant(data) if data.dtype() == DType::F32))
            .count(),
        2
    );
    assert!(custom.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.dtype() == DType::Bool && data.scalar_at(0).as_bool())
    }));
}

#[test]
fn isclose_scalar_preserves_special_empty_and_atomic_failure_contracts() {
    let mut specials = Graph::new();
    let lhs = specials.input("lhs", [3]);
    let rhs = specials.input("rhs", [3]);
    let unequal_nan = specials.isclose_scalar(lhs, rhs, 0.0, 0.0, false).unwrap();
    let equal_nan = specials.isclose_scalar(lhs, rhs, 0.0, 0.0, true).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(-0.0),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [3],
                DType::F32,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(0.0),
                ],
            )
            .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&specials, unequal_nan, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![false, true, true])
    );
    assert_eq!(
        CpuBackend
            .execute(&specials, equal_nan, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![true, true, true])
    );

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 1], DType::I16);
    let rhs = empty.input_dtype("rhs", [3], DType::I16);
    let output = empty.isclose_default(lhs, rhs).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    assert_eq!(empty.dtype(output).unwrap(), DType::Bool);

    let mut malformed = Graph::new();
    let lhs = malformed.input("lhs", [2]);
    let rhs = malformed.input("rhs", [3]);
    let before = malformed.node_count();
    assert!(malformed.isclose_default(lhs, rhs).is_err());
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX], DType::F64);
    let scalar = malformed.input_dtype("scalar", [], DType::F64);
    let before = malformed.node_count();
    assert!(
        malformed
            .isclose_scalar(overflow, scalar, 1e-5, 1e-8, false)
            .is_err()
    );
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn logaddexp_reuses_tinygrad_lub_operands_and_preserves_stable_composition() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 2], DType::I64);
    let rhs = graph.input_dtype("rhs", [1], DType::U64);
    let output = graph.logaddexp(lhs, rhs).unwrap();

    // Mixed I64/U64 follows tinygrad's F32 bridge once per source operand;
    // the same casted values feed Max and both centered paths.
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 2]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(
        matches!(graph.op(NodeId(2)).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(NodeId(3)).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([1, 2], DType::I64, [Scalar::I(0), Scalar::I(0)]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([1], DType::U64, [Scalar::U(0)]).unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert!((values.scalar_at(0).as_f64() - std::f64::consts::LN_2).abs() < 1e-5);
    assert!((values.scalar_at(1).as_f64() - std::f64::consts::LN_2).abs() < 1e-5);

    let mut differentiable = Graph::new();
    let lhs = differentiable.input("lhs", [1, 2]);
    let rhs = differentiable.input("rhs", [1]);
    let output = differentiable.logaddexp(lhs, rhs).unwrap();
    let loss = differentiable.sum_all(output).unwrap();
    let gradient = differentiable.grad(loss, lhs).unwrap();
    assert_eq!(differentiable.shape(gradient).unwrap(), &Shape::new([1, 2]));

    let mut specials = Graph::new();
    let lhs = specials.input("lhs", [2]);
    let rhs = specials.input("rhs", [2]);
    let output = specials.logaddexp(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &specials,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::F32,
                        [Scalar::F(f64::INFINITY), Scalar::F(f64::NAN)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::F32,
                        [Scalar::F(f64::INFINITY), Scalar::F(-0.0)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_f64().is_nan());
    assert!(values.scalar_at(1).as_f64().is_nan());

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 3], DType::F16);
    let rhs = empty.input_dtype("rhs", [1, 3], DType::F16);
    let output = empty.logaddexp(lhs, rhs).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([
                    (
                        "lhs".into(),
                        TensorData::from_scalars([0, 3], DType::F16, []).unwrap()
                    ),
                    (
                        "rhs".into(),
                        TensorData::from_scalars(
                            [1, 3],
                            DType::F16,
                            [
                                Scalar::F(-0.0),
                                Scalar::F(f64::INFINITY),
                                Scalar::F(f64::NAN)
                            ],
                        )
                        .unwrap(),
                    ),
                ]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty()
    );
}

#[test]
fn logaddexp_preflights_broadcast_and_byte_overflow_before_casts_or_nodes() {
    let mut graph = Graph::new();
    let lhs = graph.input("lhs", [2]);
    let rhs = graph.input("rhs", [3]);
    let before = graph.node_count();
    assert!(graph.logaddexp(lhs, rhs).is_err());
    assert_eq!(graph.node_count(), before);

    let overflow = graph.input_dtype("overflow", [usize::MAX], DType::F64);
    let scalar = graph.input_dtype("scalar", [], DType::F64);
    let before = graph.node_count();
    assert!(graph.logaddexp(overflow, scalar).is_err());
    assert_eq!(graph.node_count(), before);
}

#[test]
fn logaddexp_scalar_commits_rhs_once_and_reuses_the_corrected_stable_plan() {
    for (dtype, scalar, output_dtype) in [
        (DType::Bool, Scalar::Bool(false), DType::F32),
        (DType::I8, Scalar::I(1), DType::F32),
        (DType::I16, Scalar::I(1), DType::F32),
        (DType::I32, Scalar::I(1), DType::F32),
        (DType::I64, Scalar::I(1), DType::F32),
        (DType::U8, Scalar::U(1), DType::F32),
        (DType::U16, Scalar::U(1), DType::F32),
        (DType::U32, Scalar::U(1), DType::F32),
        (DType::U64, Scalar::U(1), DType::F32),
        (DType::F16, Scalar::F(-0.0), DType::F16),
        (DType::BF16, Scalar::F(-0.0), DType::BF16),
        (DType::F32, Scalar::F(-0.0), DType::F32),
        (DType::F64, Scalar::F(-0.0), DType::F64),
    ] {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], dtype);
        let output = graph.logaddexp_scalar(lhs, scalar).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), output_dtype);
        // The weak RHS is the only scalar publication before the shared
        // stable plan. Its scalar storage width is the source LUB width,
        // while Exp lifts an integral/Bool stable graph to F32.
        assert!(matches!(graph.op(NodeId(1)).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        assert!(matches!(
            graph.op(output).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!((0..graph.node_count()).any(|index| matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Binary {
                op: BinaryOp::Maximum,
                ..
            }
        )));
        assert!(
            (0..graph.node_count())
                .filter(|index| matches!(
                    graph.op(NodeId(*index)).unwrap(),
                    Op::Unary {
                        op: UnaryOp::Exp2,
                        ..
                    }
                ))
                .count()
                >= 2
        );
        if dtype.is_float() {
            let Op::Constant(data) = graph.op(NodeId(1)).unwrap() else {
                panic!("prepared weak scalar must be a constant");
            };
            assert_eq!(data.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        }
    }

    // Mixed weak constants lift Bool to tinygrad's default I32/F32 before
    // the shared Exp stage, while a live I64/U64 pair now uses the required
    // F32 bridge once per original operand.
    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integer = mixed.logaddexp_scalar(boolean, Scalar::I(1)).unwrap();
    let floating = mixed.logaddexp_scalar(boolean, Scalar::F(-0.0)).unwrap();
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let narrow_integer = mixed.logaddexp_scalar(narrow, Scalar::I(1)).unwrap();
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let integral_float = mixed.logaddexp_scalar(integral, Scalar::F(-0.0)).unwrap();
    let lhs = mixed.input_dtype("i64", [2], DType::I64);
    let rhs = mixed.input_dtype("u64", [1], DType::U64);
    let bridged = mixed.logaddexp(lhs, rhs).unwrap();
    assert_eq!(mixed.dtype(integer).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(floating).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(narrow_integer).unwrap(), DType::F16);
    assert_eq!(mixed.dtype(integral_float).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(bridged).unwrap(), DType::F32);
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input, dtype: DType::F32 } if *input == lhs
    )));
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input, dtype: DType::F32 } if *input == rhs
    )));

    let mut specials = Graph::new();
    let lhs = specials.input_dtype("lhs", [], DType::F64);
    let nan = specials.logaddexp_scalar(lhs, Scalar::F(f64::NAN)).unwrap();
    let infinity = specials
        .logaddexp_scalar(lhs, Scalar::F(f64::INFINITY))
        .unwrap();
    assert!(matches!(specials.op(NodeId(1)).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().is_nan()));
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(infinity).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));

    let mut vjp = Graph::new();
    let lhs = vjp.input_dtype("lhs", [2], DType::F32);
    let output = vjp.logaddexp_scalar(lhs, Scalar::F(1.0)).unwrap();
    let loss = vjp.sum_all(output).unwrap();
    let gradient = vjp.grad(loss, lhs).unwrap();
    assert_eq!(vjp.shape(gradient).unwrap(), &Shape::new([2]));
    assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let lhs = empty.input_dtype("lhs", [0, 2], DType::BF16);
    let output = empty.logaddexp_scalar(lhs, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.logaddexp_scalar(NodeId(usize::MAX), Scalar::F(1.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let lhs = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.logaddexp_scalar(lhs, Scalar::F(1.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn select_uses_tinygrad_branch_lub_before_where() {
    let mut graph = Graph::new();
    let condition = graph.input_dtype("condition", [2, 1], DType::Bool);
    let on_true = graph.input_dtype("on_true", [1, 3], DType::I64);
    let on_false = graph.input_dtype("on_false", [2, 3], DType::U64);

    let output = graph.select(condition, on_true, on_false).unwrap();

    // tinygrad's least-upper lattice bridges I64/U64 through default float,
    // then WHERE receives two F32 branches.
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Select {
        condition: selected_condition,
        on_true: selected_true,
        on_false: selected_false,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Select");
    };
    assert_eq!(*selected_condition, condition);
    assert!(
        matches!(graph.op(*selected_true).unwrap(), Op::Cast { input, dtype }
        if *input == on_true && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*selected_false).unwrap(), Op::Cast { input, dtype }
        if *input == on_false && *dtype == DType::F32)
    );
}

#[test]
fn public_where_scalar_branches_match_tinygrad_reference_order() {
    // A live payload is tinygrad's reference for weak scalar commitment,
    // regardless of whether it occupies the true or false branch.
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut graph = Graph::new();
        let condition = graph.input_dtype("condition", [2, 1], DType::Bool);
        let payload = graph.input_dtype("payload", [1, 3], dtype);
        let true_scalar = graph.where_true_scalar(condition, value, payload).unwrap();
        let false_scalar = graph.where_false_scalar(condition, payload, value).unwrap();
        for output in [true_scalar, false_scalar] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            let Op::Select {
                condition: selected,
                ..
            } = graph.op(output).unwrap()
            else {
                panic!("public where scalar form must lower through Select");
            };
            assert_eq!(*selected, condition);
        }
        if dtype.is_float() {
            assert!((0..graph.node_count()).any(|index| matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Constant(data) if data.dtype() == dtype
                    && data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()
            )));
        }
    }

    // With no live payload, `where` materializes the true scalar from its
    // Bool condition before using it as the second scalar's weak reference.
    let mut both_scalars = Graph::new();
    let condition = both_scalars.input_dtype("condition", [2], DType::Bool);
    let integer = both_scalars
        .where_scalars(condition, Scalar::Bool(true), Scalar::I(3))
        .unwrap();
    let floating = both_scalars
        .where_scalars(condition, Scalar::I(3), Scalar::F(-0.0))
        .unwrap();
    assert_eq!(both_scalars.dtype(integer).unwrap(), DType::I32);
    assert_eq!(both_scalars.dtype(floating).unwrap(), DType::F32);
    assert!(matches!(
        both_scalars.op(integer).unwrap(),
        Op::Select { .. }
    ));
    assert!(matches!(
        both_scalars.op(floating).unwrap(),
        Op::Select { .. }
    ));

    let mut weak = Graph::new();
    let condition = weak.input_dtype("condition", [], DType::Bool);
    let boolean = weak.input_dtype("boolean", [], DType::Bool);
    let integral = weak.input_dtype("integral", [], DType::I16);
    let narrow = weak.input_dtype("narrow", [], DType::F16);
    let boolean_output = weak
        .where_true_scalar(condition, Scalar::I(1), boolean)
        .unwrap();
    assert_eq!(weak.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = weak
        .where_false_scalar(condition, integral, Scalar::F(-0.0))
        .unwrap();
    assert_eq!(weak.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = weak
        .where_true_scalar(condition, Scalar::I(1), narrow)
        .unwrap();
    assert_eq!(weak.dtype(narrow_output).unwrap(), DType::F16);

    // The live alias preserves the existing source F32 bridge for I64/U64.
    let mut bridge = Graph::new();
    let condition = bridge.input_dtype("condition", [2], DType::Bool);
    let on_true = bridge.input_dtype("on_true", [2], DType::I64);
    let on_false = bridge.input_dtype("on_false", [2], DType::U64);
    let output = bridge.r#where(condition, on_true, on_false).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
    assert!(
        matches!(bridge.op(output).unwrap(), Op::Select { condition: selected, .. } if *selected == condition)
    );

    // Scalar payload bits remain on their literal branch; the other branch
    // remains the supplied live tensor, which is also the only VJP payload.
    let mut specials = Graph::new();
    let condition = specials.input_dtype("condition", [2, 1], DType::Bool);
    let on_false = specials.input_dtype("on_false", [1, 3], DType::F64);
    let output = specials
        .where_true_scalar(condition, Scalar::F(-0.0), on_false)
        .unwrap();
    let Op::Select {
        on_true,
        on_false: selected_false,
        ..
    } = specials.op(output).unwrap()
    else {
        unreachable!()
    };
    assert!(matches!(specials.op(*on_true).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));
    assert_eq!(*selected_false, on_false);
    let nan = specials
        .where_false_scalar(condition, on_false, Scalar::F(f64::NAN))
        .unwrap();
    assert!(matches!(specials.op(nan).unwrap(), Op::Select { .. }));
    let loss = specials.sum_all(output).unwrap();
    let gradient = specials.grad(loss, on_false).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([1, 3]));
    let reverse = specials
        .where_false_scalar(condition, on_false, Scalar::F(-0.0))
        .unwrap();
    let reverse_loss = specials.sum_all(reverse).unwrap();
    let reverse_gradient = specials.grad(reverse_loss, on_false).unwrap();
    assert_eq!(
        specials.shape(reverse_gradient).unwrap(),
        &Shape::new([1, 3])
    );

    let mut empty = Graph::new();
    let condition = empty.input_dtype("condition", [0, 2], DType::Bool);
    let payload = empty.input_dtype("payload", [1, 2], DType::BF16);
    let output = empty
        .where_true_scalar(condition, Scalar::F(-0.0), payload)
        .unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.where_scalars(NodeId(usize::MAX), Scalar::I(1), Scalar::I(2)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let nonboolean = malformed.input_dtype("condition", [2], DType::I32);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.where_scalars(nonboolean, Scalar::I(1), Scalar::I(2)),
        Err(Error::InvalidLogicalDType {
            op: "select",
            actual: DType::I32
        })
    ));
    assert_eq!(malformed.node_count(), before);
    let condition = malformed.input_dtype("valid_condition", [2], DType::Bool);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.where_true_scalar(condition, Scalar::F(0.0), NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let condition = malformed.input_dtype("overflow_condition", [1, 2], DType::Bool);
    let payload = malformed.input_dtype("overflow_payload", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.where_false_scalar(condition, payload, Scalar::F(0.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn select_preserves_selected_float_payloads_and_routes_broadcast_vjps() {
    let mut graph = Graph::new();
    let condition = graph.constant(bool_data([2, 3], [true, false, true, false, true, false]));
    let on_true = graph.input_dtype("on_true", [1, 3], DType::F64);
    let on_false = graph.input_dtype("on_false", [2, 1], DType::F64);
    let output = graph.select(condition, on_true, on_false).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let true_gradient = graph.grad(loss, on_true).unwrap();
    let false_gradient = graph.grad(loss, on_false).unwrap();
    let bindings = HashMap::from([
        (
            "on_true".into(),
            TensorData::from_scalars(
                [1, 3],
                DType::F64,
                [
                    Scalar::F(-0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "on_false".into(),
            TensorData::from_scalars(
                [2, 1],
                DType::F64,
                [Scalar::F(1.0), Scalar::F(-f64::INFINITY)],
            )
            .unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(1).as_f64(), 1.0);
    assert_eq!(values.scalar_at(2).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(3).as_f64(), -f64::INFINITY);
    assert_eq!(values.scalar_at(4).as_f64().to_bits(), f64::NAN.to_bits());
    assert_eq!(values.scalar_at(5).as_f64(), -f64::INFINITY);
    assert_eq!(
        CpuBackend
            .execute(&graph, true_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, false_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 2.0]
    );
}

#[test]
fn select_preflights_all_branch_casts_before_mutation() {
    let mut graph = Graph::new();
    let condition = graph.input_dtype("condition", [2], DType::Bool);
    let on_true = graph.input_dtype("on_true", [2], DType::I64);
    let on_false = graph.input_dtype("on_false", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.select(condition, on_true, on_false),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn relu_uses_tinygrad_strict_typed_zero_select_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("x", [7], DType::F64, true);
    let output = graph.relu(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([7]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);

    let Op::Select {
        condition,
        on_true,
        on_false,
    } = graph.op(output).unwrap()
    else {
        panic!("public ReLU must lower to the source WHERE");
    };
    assert_eq!(*on_true, input);
    let Op::Constant(zero) = graph.op(*on_false).unwrap() else {
        panic!("public ReLU false branch must be its typed scalar zero");
    };
    assert_eq!(zero.shape(), &Shape::new([]));
    assert_eq!(zero.dtype(), DType::F64);
    assert_eq!(zero.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    let Op::Compare { op, lhs, rhs } = graph.op(*condition).unwrap() else {
        panic!("public ReLU condition must be an ordered comparison");
    };
    assert_eq!(*op, CompareOp::Lt);
    // Graph::gt keeps tinygrad's reverse-CMPLT lowering: typed zero < input.
    assert_eq!(*lhs, *on_false);
    assert_eq!(*rhs, input);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [7],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-1.0),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(f64::INFINITY),
                Scalar::F(3.0),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    for index in 0..5 {
        assert_eq!(values.scalar_at(index).as_f64().to_bits(), 0.0f64.to_bits());
    }
    assert_eq!(values.scalar_at(5).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(6).as_f64(), 3.0);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );

    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut typed = Graph::new();
        let input = typed.input_dtype("x", [], dtype);
        let output = typed.relu(input).unwrap();
        let Op::Select {
            condition,
            on_true,
            on_false,
        } = typed.op(output).unwrap()
        else {
            panic!("typed ReLU must remain a Select");
        };
        assert_eq!(*on_true, input);
        let Op::Constant(zero) = typed.op(*on_false).unwrap() else {
            panic!("typed ReLU zero must remain a graph constant");
        };
        assert_eq!(zero.dtype(), dtype);
        assert_eq!(zero.shape(), &Shape::new([]));
        assert_eq!(zero.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
        assert_eq!(typed.dtype(*condition).unwrap(), DType::Bool);
        assert_eq!(typed.shape(*condition).unwrap(), &Shape::new([]));
        assert_eq!(typed.dtype(output).unwrap(), dtype);
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0, 2], DType::F16);
    let output = empty.relu(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);

    let mut raw = Graph::new();
    let input = raw.input("x", [1]);
    let raw_relu = raw.unary(UnaryOp::Relu, input).unwrap();
    assert!(matches!(
        raw.op(raw_relu).unwrap(),
        Op::Unary {
            op: UnaryOp::Relu,
            input: source
        } if *source == input
    ));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(malformed.relu(crate::NodeId(usize::MAX)).is_err());
    assert_eq!(malformed.node_count(), before);

    let mut overflow = Graph::new();
    let input = overflow.input("x", [usize::MAX, 2]);
    let before = overflow.node_count();
    assert!(overflow.relu(input).is_err());
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn eq_uses_tinygrad_branch_lub_before_the_bool_predicate() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.eq(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Ne,
        lhs: boolean,
        rhs: truth,
    } = graph.op(output).unwrap()
    else {
        panic!("expected source logical-not comparison");
    };
    assert!(matches!(graph.op(*truth).unwrap(), Op::Constant(data)
        if data.dtype() == DType::Bool && data.scalar_at(0).as_bool()));
    let Op::Cast {
        input: inner,
        dtype: DType::Bool,
    } = graph.op(*boolean).unwrap()
    else {
        panic!("expected source logical-not Bool cast");
    };
    let Op::Compare {
        op: CompareOp::Ne,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(*inner).unwrap()
    else {
        panic!("expected source Eq inner Ne comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn eq_keeps_typed_wide_integer_equality_and_float_special_values() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::I64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::I64,
            [Scalar::I((1_i64 << 53) + 1), Scalar::I(i64::MIN)],
        )
        .unwrap(),
    );
    let integer_eq = graph.eq(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [2], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars([2], DType::F64, [Scalar::F(0.0), Scalar::F(f64::NAN)]).unwrap(),
    );
    let float_eq = graph.eq(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::I64,
                [Scalar::I(1_i64 << 53), Scalar::I(i64::MIN)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars([2], DType::F64, [Scalar::F(-0.0), Scalar::F(f64::NAN)])
                .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_eq, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_eq, &bindings).unwrap();
    assert!(!integers.scalar_at(0).as_bool());
    assert!(integers.scalar_at(1).as_bool());
    assert!(floats.scalar_at(0).as_bool());
    assert!(!floats.scalar_at(1).as_bool());

    // Unlike same-kind I64, tinygrad promotes this mixed pair to F32 before
    // comparison.  The adjacent wide values consequently compare equal at
    // the source-selected storage width.
    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [2], DType::I64);
    let rhs = mixed.input_dtype("rhs", [2], DType::U64);
    let output = mixed.eq(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::I64,
                        [Scalar::I(1_i64 << 53), Scalar::I(-1)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [2],
                        DType::U64,
                        [Scalar::U((1_u64 << 53) + 1), Scalar::U(0)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());
    assert!(!values.scalar_at(1).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.eq(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn eq_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.eq(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn equality_scalar_forms_preserve_weak_lub_and_eq_not_ne_structure() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(f64::NAN)),
        (DType::F64, Scalar::F(f64::INFINITY)),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 1], dtype);
        let equal = graph.eq_scalar(input, value).unwrap();
        let unequal = graph.ne_scalar(input, value).unwrap();
        assert_eq!(graph.shape(equal).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(equal).unwrap(), DType::Bool);
        assert_eq!(graph.shape(unequal).unwrap(), &Shape::new([2, 1]));
        assert_eq!(graph.dtype(unequal).unwrap(), DType::Bool);
        let Op::Compare {
            op: CompareOp::Ne,
            lhs: boolean,
            rhs: truth,
        } = graph.op(equal).unwrap()
        else {
            panic!("eq scalar must be source logical_not(ne)");
        };
        assert!(matches!(graph.op(*truth).unwrap(), Op::Constant(data)
            if data.dtype() == DType::Bool && data.scalar_at(0).as_bool()));
        let Op::Cast {
            input: inner,
            dtype: DType::Bool,
        } = graph.op(*boolean).unwrap()
        else {
            panic!("eq scalar logical_not must cast the inner Bool predicate");
        };
        assert!(matches!(
            graph.op(*inner).unwrap(),
            Op::Compare {
                op: CompareOp::Ne,
                ..
            }
        ));
        assert!(matches!(
            graph.op(unequal).unwrap(),
            Op::Compare {
                op: CompareOp::Ne,
                ..
            }
        ));
        assert!(matches!(
            graph.grad(equal, input),
            Err(Error::NonDifferentiableTarget(node)) if node == equal
        ));
        assert!(matches!(
            graph.grad(unequal, input),
            Err(Error::NonDifferentiableTarget(node)) if node == unequal
        ));
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integer = mixed.input_dtype("integer", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let bool_integer = mixed.ne_scalar(boolean, Scalar::I(1)).unwrap();
    let integer_float = mixed.eq_scalar(integer, Scalar::F(-0.0)).unwrap();
    let narrow_integer = mixed.ne_scalar(narrow, Scalar::I(1)).unwrap();
    for output in [bool_integer, integer_float, narrow_integer] {
        assert_eq!(mixed.dtype(output).unwrap(), DType::Bool);
    }
    assert!(
        (0..mixed.node_count()).any(|index| matches!(mixed.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I32 && data.scalar_at(0).as_i64() == 1))
    );
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    )));

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F64);
    let output = scalar.eq_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.ne_scalar(input, Scalar::F(f64::NAN)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::Bool);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.ne(lhs, rhs).unwrap();
    let Op::Compare {
        lhs: promoted_lhs,
        rhs: promoted_rhs,
        ..
    } = bridge.op(output).unwrap()
    else {
        panic!("expected live bridge comparison");
    };
    assert!(matches!(
        bridge.op(*promoted_lhs).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    ));
    assert!(matches!(
        bridge.op(*promoted_rhs).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    ));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.eq_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.ne_scalar(overflow, Scalar::F(0.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn ne_uses_tinygrad_branch_lub_before_the_bool_predicate() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.ne(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Ne,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Ne comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn ne_keeps_typed_wide_comparison_and_source_float_special_values() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::U64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::U64,
            [Scalar::U((1_u64 << 53) + 1), Scalar::U(u64::MAX)],
        )
        .unwrap(),
    );
    let integer_ne = graph.ne(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [2], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars([2], DType::F64, [Scalar::F(0.0), Scalar::F(f64::NAN)]).unwrap(),
    );
    let float_ne = graph.ne(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::U64,
                [Scalar::U(1_u64 << 53), Scalar::U(u64::MAX)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars([2], DType::F64, [Scalar::F(-0.0), Scalar::F(f64::NAN)])
                .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_ne, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_ne, &bindings).unwrap();
    assert!(integers.scalar_at(0).as_bool());
    assert!(!integers.scalar_at(1).as_bool());
    assert!(!floats.scalar_at(0).as_bool());
    assert!(floats.scalar_at(1).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.ne(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn ne_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.ne(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn lt_uses_tinygrad_branch_lub_before_the_bool_predicate() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.lt(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Lt comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn lt_keeps_typed_ordering_and_float_special_values() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::I64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::I64,
            [Scalar::I((1_i64 << 53) + 1), Scalar::I(i64::MIN)],
        )
        .unwrap(),
    );
    let integer_lt = graph.lt(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [3], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars(
            [3],
            DType::F64,
            [Scalar::F(0.0), Scalar::F(0.0), Scalar::F(f64::INFINITY)],
        )
        .unwrap(),
    );
    let float_lt = graph.lt(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::I64,
                [Scalar::I(1_i64 << 53), Scalar::I(i64::MIN)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars(
                [3],
                DType::F64,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(-0.0),
                    Scalar::F(f64::NEG_INFINITY),
                ],
            )
            .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_lt, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_lt, &bindings).unwrap();
    assert!(integers.scalar_at(0).as_bool());
    assert!(!integers.scalar_at(1).as_bool());
    assert!(!floats.scalar_at(0).as_bool());
    assert!(!floats.scalar_at(1).as_bool());
    assert!(floats.scalar_at(2).as_bool());

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.lt(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(0), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.lt(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn lt_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.lt(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn gt_uses_tinygrad_reversed_lt_with_source_branch_lub() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.gt(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_rhs,
        rhs: compared_lhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected reversed Lt comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn gt_keeps_typed_ordering_and_float_special_values() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::I64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::I64,
            [Scalar::I(1_i64 << 53), Scalar::I(i64::MIN)],
        )
        .unwrap(),
    );
    let integer_gt = graph.gt(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [3], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars(
            [3],
            DType::F64,
            [Scalar::F(0.0), Scalar::F(0.0), Scalar::F(f64::NEG_INFINITY)],
        )
        .unwrap(),
    );
    let float_gt = graph.gt(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::I64,
                [Scalar::I((1_i64 << 53) + 1), Scalar::I(i64::MIN)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars(
                [3],
                DType::F64,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(-0.0),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_gt, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_gt, &bindings).unwrap();
    assert!(integers.scalar_at(0).as_bool());
    assert!(!integers.scalar_at(1).as_bool());
    assert!(!floats.scalar_at(0).as_bool());
    assert!(!floats.scalar_at(1).as_bool());
    assert!(floats.scalar_at(2).as_bool());

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.gt(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(0), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.gt(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn gt_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.gt(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn le_uses_tinygrad_not_of_reversed_lt_with_source_branch_lub() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.le(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Ne,
        lhs: boolean,
        rhs: truth,
    } = graph.op(output).unwrap()
    else {
        panic!("expected source logical-not comparison");
    };
    assert!(matches!(graph.op(*truth).unwrap(), Op::Constant(data)
        if data.dtype() == DType::Bool && data.scalar_at(0).as_bool()));
    let Op::Cast {
        input: greater,
        dtype: DType::Bool,
    } = graph.op(*boolean).unwrap()
    else {
        panic!("expected logical-not Bool cast");
    };
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_rhs,
        rhs: compared_lhs,
    } = graph.op(*greater).unwrap()
    else {
        panic!("expected reversed Lt comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn le_keeps_tinygrad_nan_and_typed_ordering_behavior() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::I64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::I64,
            [Scalar::I((1_i64 << 53) + 1), Scalar::I(i64::MIN)],
        )
        .unwrap(),
    );
    let integer_le = graph.le(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [3], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars(
            [3],
            DType::F64,
            [Scalar::F(0.0), Scalar::F(0.0), Scalar::F(f64::NEG_INFINITY)],
        )
        .unwrap(),
    );
    let float_le = graph.le(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::I64,
                [Scalar::I(1_i64 << 53), Scalar::I(i64::MIN)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars(
                [3],
                DType::F64,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(-0.0),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_le, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_le, &bindings).unwrap();
    assert!(integers.scalar_at(0).as_bool());
    assert!(integers.scalar_at(1).as_bool());
    assert!(floats.scalar_at(0).as_bool());
    assert!(floats.scalar_at(1).as_bool());
    assert!(!floats.scalar_at(2).as_bool());

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.le(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(0), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.le(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn le_preflights_source_composition_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.le(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn ge_uses_tinygrad_not_of_lt_with_source_branch_lub() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.ge(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Compare {
        op: CompareOp::Ne,
        lhs: boolean,
        rhs: truth,
    } = graph.op(output).unwrap()
    else {
        panic!("expected source logical-not comparison");
    };
    assert!(matches!(graph.op(*truth).unwrap(), Op::Constant(data)
        if data.dtype() == DType::Bool && data.scalar_at(0).as_bool()));
    let Op::Cast {
        input: less,
        dtype: DType::Bool,
    } = graph.op(*boolean).unwrap()
    else {
        panic!("expected logical-not Bool cast");
    };
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(*less).unwrap()
    else {
        panic!("expected Lt comparison");
    };
    assert!(
        matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn ge_keeps_tinygrad_nan_and_typed_ordering_behavior() {
    let mut graph = Graph::new();
    let integers = graph.input_dtype("integers", [2], DType::I64);
    let integer_rhs = graph.constant(
        TensorData::from_scalars(
            [2],
            DType::I64,
            [Scalar::I(1_i64 << 53), Scalar::I(i64::MIN)],
        )
        .unwrap(),
    );
    let integer_ge = graph.ge(integers, integer_rhs).unwrap();
    let floats = graph.input_dtype("floats", [4], DType::F64);
    let float_rhs = graph.constant(
        TensorData::from_scalars(
            [4],
            DType::F64,
            [
                Scalar::F(0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(f64::INFINITY),
            ],
        )
        .unwrap(),
    );
    let float_ge = graph.ge(floats, float_rhs).unwrap();
    let bindings = HashMap::from([
        (
            "integers".into(),
            TensorData::from_scalars(
                [2],
                DType::I64,
                [Scalar::I((1_i64 << 53) + 1), Scalar::I(i64::MIN)],
            )
            .unwrap(),
        ),
        (
            "floats".into(),
            TensorData::from_scalars(
                [4],
                DType::F64,
                [
                    Scalar::F(f64::NAN),
                    Scalar::F(-0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::NEG_INFINITY),
                ],
            )
            .unwrap(),
        ),
    ]);
    let integers = CpuBackend.execute(&graph, integer_ge, &bindings).unwrap();
    let floats = CpuBackend.execute(&graph, float_ge, &bindings).unwrap();
    assert!(integers.scalar_at(0).as_bool());
    assert!(integers.scalar_at(1).as_bool());
    assert!(floats.scalar_at(0).as_bool());
    assert!(floats.scalar_at(1).as_bool());
    assert!(floats.scalar_at(2).as_bool());
    assert!(!floats.scalar_at(3).as_bool());

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.ge(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(0), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.ge(input, rhs).unwrap();
    assert!(matches!(
        predicate.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));
}

#[test]
fn ge_preflights_source_composition_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.ge(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn ordered_comparison_scalar_forms_preserve_tensor_and_reflected_orientations() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(f64::NAN)),
        (DType::F64, Scalar::F(f64::INFINITY)),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 1], dtype);
        let less = graph.lt_scalar(input, value).unwrap();
        let greater = graph.gt_scalar(input, value).unwrap();
        let less_equal = graph.le_scalar(input, value).unwrap();
        let greater_equal = graph.ge_scalar(input, value).unwrap();
        for output in [less, greater, less_equal, greater_equal] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
            assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
            assert!(matches!(
                graph.grad(output, input),
                Err(Error::NonDifferentiableTarget(node)) if node == output
            ));
        }
        assert!(matches!(
            graph.op(less).unwrap(),
            Op::Compare {
                op: CompareOp::Lt,
                ..
            }
        ));
        assert!(matches!(
            graph.op(greater).unwrap(),
            Op::Compare {
                op: CompareOp::Lt,
                ..
            }
        ));
        for output in [less_equal, greater_equal] {
            let Op::Compare {
                op: CompareOp::Ne,
                lhs: boolean,
                rhs: truth,
            } = graph.op(output).unwrap()
            else {
                panic!("inclusive scalar comparison must be source logical_not");
            };
            assert!(matches!(graph.op(*truth).unwrap(), Op::Constant(data)
                if data.dtype() == DType::Bool && data.scalar_at(0).as_bool()));
            let Op::Cast {
                input: inner,
                dtype: DType::Bool,
            } = graph.op(*boolean).unwrap()
            else {
                panic!("inclusive scalar comparison must cast its Bool predicate");
            };
            assert!(matches!(
                graph.op(*inner).unwrap(),
                Op::Compare {
                    op: CompareOp::Lt,
                    ..
                }
            ));
        }
    }

    let mut reflected = Graph::new();
    let input = reflected.input_dtype("input", [2], DType::F64);
    let scalar_less = reflected.scalar_lt(Scalar::F(-0.0), input).unwrap();
    let scalar_greater = reflected.scalar_gt(Scalar::F(f64::NAN), input).unwrap();
    let scalar_less_equal = reflected
        .scalar_le(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let scalar_greater_equal = reflected.scalar_ge(Scalar::F(-0.0), input).unwrap();
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: scalar,
        rhs,
    } = reflected.op(scalar_less).unwrap()
    else {
        panic!("scalar < Tensor must use Tensor.__gt__ reversed LT");
    };
    assert!(matches!(reflected.op(*scalar).unwrap(), Op::Constant(data)
        if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));
    assert_eq!(*rhs, input);
    let Op::Compare {
        op: CompareOp::Lt,
        lhs,
        rhs: scalar,
    } = reflected.op(scalar_greater).unwrap()
    else {
        panic!("scalar > Tensor must use Tensor.__lt__");
    };
    assert_eq!(*lhs, input);
    assert!(
        matches!(reflected.op(*scalar).unwrap(), Op::Constant(data) if data.dtype() == DType::F64)
    );
    for output in [scalar_less_equal, scalar_greater_equal] {
        assert!(matches!(
            reflected.op(output).unwrap(),
            Op::Compare {
                op: CompareOp::Ne,
                ..
            }
        ));
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integer = mixed.input_dtype("integer", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.lt_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::Bool);
    let integer_output = mixed.scalar_ge(Scalar::F(-0.0), integer).unwrap();
    assert_eq!(mixed.dtype(integer_output).unwrap(), DType::Bool);
    let narrow_output = mixed.gt_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::Bool);
    assert!(
        (0..mixed.node_count()).any(|index| matches!(mixed.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I32 && data.scalar_at(0).as_i64() == 1))
    );
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    )));

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F64);
    let output = scalar.ge_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.scalar_lt(Scalar::F(f64::NAN), input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::Bool);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.lt(lhs, rhs).unwrap();
    let Op::Compare {
        lhs: promoted_lhs,
        rhs: promoted_rhs,
        ..
    } = bridge.op(output).unwrap()
    else {
        panic!("expected live bridge comparison");
    };
    assert!(matches!(
        bridge.op(*promoted_lhs).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    ));
    assert!(matches!(
        bridge.op(*promoted_rhs).unwrap(),
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    ));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.lt_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_ge(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn add_uses_tinygrad_branch_lub_before_storage_width_addition() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.add(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Binary {
        op: BinaryOp::Add,
        lhs: added_lhs,
        rhs: added_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Add");
    };
    assert!(
        matches!(graph.op(*added_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*added_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn add_scalar_commits_weak_values_and_preserves_reflected_root_order() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.add_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } = forward.op(output).unwrap()
        else {
            panic!("add_scalar must lower to Add");
        };
        assert_eq!(*lhs, input);
        assert!(matches!(forward.op(*rhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_add(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } = reflected.op(output).unwrap()
        else {
            panic!("scalar_add must lower to Add");
        };
        assert!(matches!(reflected.op(*lhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        assert_eq!(*rhs, input);
    }

    // Python weak crossings follow the source tensor reference; live I64/U64
    // remains the distinct F32 bridge already covered by Graph::add.
    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.add_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_add(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.add_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.add(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.add_scalar(input, Scalar::F(-0.0)).unwrap();
    let nan = specials.scalar_add(Scalar::F(f64::NAN), input).unwrap();
    let Op::Binary { lhs, rhs, .. } = specials.op(negative_zero).unwrap() else {
        unreachable!()
    };
    assert_eq!(*lhs, input);
    assert!(matches!(specials.op(*rhs).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));
    let Op::Binary { lhs, rhs, .. } = specials.op(nan).unwrap() else {
        unreachable!()
    };
    assert!(
        matches!(specials.op(*lhs).unwrap(), Op::Constant(data) if data.scalar_at(0).as_f64().is_nan())
    );
    assert_eq!(*rhs, input);
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));
    let reverse_loss = specials.sum_all(nan).unwrap();
    let reverse_gradient = specials.grad(reverse_loss, input).unwrap();
    assert_eq!(
        specials.shape(reverse_gradient).unwrap(),
        &Shape::new([2, 1])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.add_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.add_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_add(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn add_keeps_source_width_special_values_and_broadcast_vjp() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::F64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::F64);
    let output = graph.add(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [1, 3],
                DType::F64,
                [
                    Scalar::F(-0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [2, 1],
                DType::F64,
                [Scalar::F(-0.0), Scalar::F(f64::NEG_INFINITY)],
            )
            .unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert_eq!(values.scalar_at(2).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(3).as_f64(), f64::NEG_INFINITY);
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![3.0, 3.0]
    );

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.add(lhs, rhs).unwrap();
    assert_eq!(mixed.dtype(output).unwrap(), DType::I32);
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(2), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), 1);

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.add(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
    assert!(matches!(
        narrow.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
}

#[test]
fn add_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.add(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn sub_uses_tinygrad_branch_lub_before_ordered_subtraction() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.sub(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Binary {
        op: BinaryOp::Add,
        lhs: added_lhs,
        rhs: negated_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected source Add");
    };
    assert!(
        matches!(graph.op(*added_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*negated_rhs).unwrap(), Op::Unary { op: UnaryOp::Neg, input }
        if matches!(graph.op(*input).unwrap(), Op::Cast { input: source, dtype: DType::F32 } if *source == rhs))
    );
}

#[test]
fn sub_scalar_preserves_tinygrad_neg_then_add_and_reflected_order() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.sub_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } = forward.op(output).unwrap()
        else {
            panic!("sub_scalar must root at source Add");
        };
        assert_eq!(*lhs, input);
        if dtype == DType::Bool {
            assert!(matches!(
                forward.op(*rhs).unwrap(),
                Op::Compare {
                    op: CompareOp::Ne,
                    ..
                }
            ));
        } else {
            assert!(
                matches!(forward.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Neg, input: scalar }
                if matches!(forward.op(*scalar).unwrap(), Op::Constant(_)))
            );
        }

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_sub(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } = reflected.op(output).unwrap()
        else {
            panic!("scalar_sub must root at source Add");
        };
        assert!(matches!(reflected.op(*lhs).unwrap(), Op::Constant(_)));
        if dtype == DType::Bool {
            assert!(matches!(
                reflected.op(*rhs).unwrap(),
                Op::Compare {
                    op: CompareOp::Ne,
                    lhs,
                    ..
                } if matches!(reflected.op(*lhs).unwrap(), Op::Cast {
                    input: source,
                    dtype: DType::Bool,
                } if *source == input)
            ));
        } else {
            assert!(
                matches!(reflected.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Neg, input: source }
                if *source == input)
            );
        }
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.sub_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_sub(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.sub_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.sub(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.sub_scalar(input, Scalar::F(-0.0)).unwrap();
    let nan = specials.scalar_sub(Scalar::F(f64::NAN), input).unwrap();
    let Op::Binary { lhs, rhs, .. } = specials.op(negative_zero).unwrap() else {
        unreachable!()
    };
    assert_eq!(*lhs, input);
    assert!(
        matches!(specials.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Neg, input: scalar }
        if matches!(specials.op(*scalar).unwrap(), Op::Constant(data)
            if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()))
    );
    let Op::Binary { lhs, rhs, .. } = specials.op(nan).unwrap() else {
        unreachable!()
    };
    assert!(
        matches!(specials.op(*lhs).unwrap(), Op::Constant(data) if data.scalar_at(0).as_f64().is_nan())
    );
    assert!(matches!(
        specials.op(*rhs).unwrap(),
        Op::Unary {
            op: UnaryOp::Neg,
            input: source,
        } if *source == input
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));
    let reverse_loss = specials.sum_all(nan).unwrap();
    let reverse_gradient = specials.grad(reverse_loss, input).unwrap();
    assert_eq!(
        specials.shape(reverse_gradient).unwrap(),
        &Shape::new([2, 1])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.sub_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.sub_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_sub(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn sub_matches_tinygrad_bool_negation_and_float_broadcast_vjp() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [4], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [4], DType::Bool);
    let output = booleans.sub(lhs, rhs).unwrap();
    let Op::Binary {
        op: BinaryOp::Add,
        lhs: added_lhs,
        rhs: added_rhs,
    } = booleans.op(output).unwrap()
    else {
        panic!("expected Bool Add");
    };
    assert_eq!(*added_lhs, lhs);
    assert!(matches!(
        booleans.op(*added_rhs).unwrap(),
        Op::Compare {
            op: CompareOp::Ne,
            lhs: input,
            ..
        } if matches!(booleans.op(*input).unwrap(), Op::Cast {
            input: source,
            dtype: DType::Bool,
        } if *source == rhs)
    ));
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([4], [false, false, true, true])),
                ("rhs".into(), bool_data([4], [false, true, false, true])),
            ]),
        )
        .unwrap();
    assert_eq!(
        (0..4)
            .map(|index| values.scalar_at(index).as_bool())
            .collect::<Vec<_>>(),
        vec![true, false, true, true]
    );

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::F64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::F64);
    let output = graph.sub(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [1, 3],
                DType::F64,
                [
                    Scalar::F(-0.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [2, 1],
                DType::F64,
                [Scalar::F(0.0), Scalar::F(f64::INFINITY)],
            )
            .unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert_eq!(values.scalar_at(2).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(3).as_f64(), f64::NEG_INFINITY);
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-3.0, -3.0]
    );

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.sub(lhs, rhs).unwrap();
    assert_eq!(mixed.dtype(output).unwrap(), DType::I32);
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-1), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(2), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), -3);
}

#[test]
fn sub_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.sub(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn mul_uses_tinygrad_branch_lub_before_storage_width_multiplication() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.mul(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs: multiplied_lhs,
        rhs: multiplied_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Mul");
    };
    assert!(
        matches!(graph.op(*multiplied_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(
        matches!(graph.op(*multiplied_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)
    );
}

#[test]
fn mul_scalar_commits_weak_values_and_preserves_reflected_root_order() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.mul_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } = forward.op(output).unwrap()
        else {
            panic!("mul_scalar must lower to Mul");
        };
        assert_eq!(*lhs, input);
        assert!(matches!(forward.op(*rhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_mul(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), dtype);
        let Op::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } = reflected.op(output).unwrap()
        else {
            panic!("scalar_mul must lower to Mul");
        };
        assert!(matches!(reflected.op(*lhs).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        assert_eq!(*rhs, input);
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.mul_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_mul(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.mul_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.mul(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.mul_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .scalar_mul(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let nan = specials.scalar_mul(Scalar::F(f64::NAN), input).unwrap();
    let Op::Binary { lhs, rhs, .. } = specials.op(negative_zero).unwrap() else {
        unreachable!()
    };
    assert_eq!(*lhs, input);
    assert!(matches!(specials.op(*rhs).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));
    let Op::Binary { lhs, rhs, .. } = specials.op(infinity).unwrap() else {
        unreachable!()
    };
    assert!(
        matches!(specials.op(*lhs).unwrap(), Op::Constant(data) if data.scalar_at(0).as_f64() == f64::INFINITY)
    );
    assert_eq!(*rhs, input);
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.mul_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.mul_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_mul(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn mul_matches_tinygrad_bool_special_values_and_broadcast_vjp() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [4], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [4], DType::Bool);
    let output = booleans.mul(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([4], [false, false, true, true])),
                ("rhs".into(), bool_data([4], [false, true, false, true])),
            ]),
        )
        .unwrap();
    assert_eq!(
        (0..4)
            .map(|index| values.scalar_at(index).as_bool())
            .collect::<Vec<_>>(),
        vec![false, false, false, true]
    );

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [3], DType::F64);
    let rhs = special.input_dtype("rhs", [3], DType::F64);
    let output = special.mul(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [3],
                        DType::F64,
                        [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(f64::INFINITY)],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [3],
                        DType::F64,
                        [Scalar::F(2.0), Scalar::F(f64::INFINITY), Scalar::F(0.0)],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::F64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::F64);
    let output = graph.mul(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [1, 3],
                DType::F64,
                [Scalar::F(1.0), Scalar::F(2.0), Scalar::F(3.0)],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([2, 1], DType::F64, [Scalar::F(2.0), Scalar::F(3.0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![5.0, 5.0, 5.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![6.0, 6.0]
    );

    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [], DType::I16);
    let rhs = mixed.input_dtype("rhs", [], DType::U16);
    let output = mixed.mul(lhs, rhs).unwrap();
    assert_eq!(mixed.dtype(output).unwrap(), DType::I32);
    let values = CpuBackend
        .execute(
            &mixed,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-2), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(3), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), -6);

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.mul(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
}

#[test]
fn mul_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.mul(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn div_uses_tinygrad_true_division_lub_and_reciprocal_composition() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::U64);

    let output = graph.div(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs: dividend,
        rhs: reciprocal,
    } = graph.op(output).unwrap()
    else {
        panic!("expected true-division Mul");
    };
    assert!(
        matches!(graph.op(*dividend).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32)
    );
    assert!(matches!(graph.op(*reciprocal).unwrap(), Op::Unary {
        op: UnaryOp::Reciprocal,
        input,
    } if matches!(graph.op(*input).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32)));

    let mut integer = Graph::new();
    let lhs = integer.input_dtype("lhs", [], DType::I16);
    let rhs = integer.input_dtype("rhs", [], DType::U16);
    let output = integer.div(lhs, rhs).unwrap();
    assert_eq!(integer.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &integer,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-3), DType::I16),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::U(2), DType::U16),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -1.5);
}

#[test]
fn div_scalar_preserves_true_division_roles_and_storage_boundaries() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let expected_dtype = if dtype.is_float() { dtype } else { DType::F32 };
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.div_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), expected_dtype);
        let Op::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } = forward.op(output).unwrap()
        else {
            panic!("div_scalar must root at true-division Mul");
        };
        assert!(matches!(
            forward.op(*rhs).unwrap(),
            Op::Unary {
                op: UnaryOp::Reciprocal,
                ..
            }
        ));
        if dtype.is_float() {
            assert_eq!(*lhs, input);
        } else {
            assert!(
                matches!(forward.op(*lhs).unwrap(), Op::Cast { input: source, dtype: DType::F32 } if *source == input)
            );
        }

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_div(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), expected_dtype);
        let Op::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } = reflected.op(output).unwrap()
        else {
            panic!("scalar_div must root at true-division Mul");
        };
        assert!(matches!(
            reflected.op(*rhs).unwrap(),
            Op::Unary {
                op: UnaryOp::Reciprocal,
                ..
            }
        ));
        if dtype.is_float() {
            assert!(
                matches!(reflected.op(*lhs).unwrap(), Op::Constant(data) if data.dtype() == dtype)
            );
        } else {
            assert!(matches!(
                reflected.op(*lhs).unwrap(),
                Op::Cast {
                    dtype: DType::F32,
                    ..
                }
            ));
        }
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.div_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::F32);
    let integral_output = mixed.scalar_div(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.div_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.div(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.div_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .scalar_div(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let nan = specials.scalar_div(Scalar::F(f64::NAN), input).unwrap();
    let Op::Binary { lhs, rhs, .. } = specials.op(negative_zero).unwrap() else {
        unreachable!()
    };
    assert_eq!(*lhs, input);
    assert!(
        matches!(specials.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input: scalar }
        if matches!(specials.op(*scalar).unwrap(), Op::Constant(data)
            if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()))
    );
    let Op::Binary { lhs, rhs, .. } = specials.op(infinity).unwrap() else {
        unreachable!()
    };
    assert!(
        matches!(specials.op(*lhs).unwrap(), Op::Constant(data) if data.scalar_at(0).as_f64() == f64::INFINITY)
    );
    assert!(
        matches!(specials.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input: source } if *source == input)
    );
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));
    let reverse_loss = specials.sum_all(infinity).unwrap();
    let reverse_gradient = specials.grad(reverse_loss, input).unwrap();
    assert_eq!(
        specials.shape(reverse_gradient).unwrap(),
        &Shape::new([2, 1])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.div_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.div_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_div(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn div_matches_tinygrad_bool_special_values_and_broadcast_vjp() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [3], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [3], DType::Bool);
    let output = booleans.div(lhs, rhs).unwrap();
    assert_eq!(booleans.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([3], [false, true, false])),
                ("rhs".into(), bool_data([3], [true, false, false])),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert!(values.scalar_at(1).as_f64().is_infinite());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [5], DType::F64);
    let rhs = special.input_dtype("rhs", [5], DType::F64);
    let output = special.div(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(1.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::NAN),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(0.0),
                            Scalar::F(3.0),
                        ],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert_eq!(values.scalar_at(2).as_f64(), 0.0);
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [1, 3], DType::F64);
    let rhs = graph.input_dtype("rhs", [2, 1], DType::F64);
    let output = graph.div(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [1, 3],
                DType::F64,
                [Scalar::F(2.0), Scalar::F(4.0), Scalar::F(8.0)],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([2, 1], DType::F64, [Scalar::F(2.0), Scalar::F(4.0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.75, 0.75, 0.75]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-3.5, -0.875]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.div(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
}

#[test]
fn div_preflights_true_division_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.div(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn trunc_div_uses_tinygrad_integer_cdiv_lub_and_zero_sentinel() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I16);
    let rhs = graph.input_dtype("rhs", [2], DType::U16);
    let output = graph.trunc_div(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    let Op::Select {
        condition,
        on_true,
        on_false,
    } = graph.op(output).unwrap()
    else {
        panic!("expected zero-sentinel Select");
    };
    assert!(matches!(
        graph.op(*condition).unwrap(),
        Op::Compare {
            op: CompareOp::Ne,
            ..
        }
    ));
    assert!(matches!(graph.op(*on_true).unwrap(), Op::Constant(_)));
    assert!(matches!(
        graph.op(*on_false).unwrap(),
        Op::Binary {
            op: BinaryOp::TruncDiv,
            ..
        }
    ));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([2], DType::I16, [Scalar::I(-3), Scalar::I(5)]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([2], DType::U16, [Scalar::U(2), Scalar::U(0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0, 0.0]
    );
    let loss = graph.sum_all(output).unwrap();
    assert!(matches!(
        graph.grad(loss, lhs),
        Err(Error::NonDifferentiableTarget(node)) if node == loss
    ));

    let mut signed_edge = Graph::new();
    let lhs = signed_edge.input_dtype("lhs", [], DType::I64);
    let rhs = signed_edge.input_dtype("rhs", [], DType::I64);
    let output = signed_edge.trunc_div(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &signed_edge,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(i64::MIN), DType::I64),
                ),
                (
                    "rhs".into(),
                    TensorData::scalar_with_dtype(Scalar::I(-1), DType::I64),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), i64::MIN);

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = wide.input_dtype("rhs", [2, 1], DType::U64);
    let output = wide.trunc_div(lhs, rhs).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);
    assert_eq!(wide.shape(output).unwrap(), &Shape::new([2, 3]));
    assert!(
        matches!(wide.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input }
        if matches!(wide.op(*input).unwrap(), Op::Binary { op: BinaryOp::Mul, .. }))
    );
}

#[test]
fn trunc_div_matches_tinygrad_bool_float_special_values_and_narrow_dtype() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [3], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [3], DType::Bool);
    let output = booleans.trunc_div(lhs, rhs).unwrap();
    assert_eq!(booleans.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([3], [false, true, false])),
                ("rhs".into(), bool_data([3], [true, false, false])),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert!(values.scalar_at(1).as_f64().is_infinite());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [5], DType::F64);
    let rhs = special.input_dtype("rhs", [5], DType::F64);
    let output = special.trunc_div(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(1.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::NAN),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(0.0),
                            Scalar::F(3.0),
                        ],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert_eq!(values.scalar_at(2).as_f64(), 0.0);
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.trunc_div(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
}

#[test]
fn trunc_div_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.trunc_div(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn trunc_div_scalar_preserves_source_integer_and_float_branches() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let expected_dtype = if dtype.is_integer() || dtype.is_float() {
            dtype
        } else {
            DType::F32
        };
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.trunc_div_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), expected_dtype);
        if dtype.is_integer() {
            assert!(matches!(forward.op(output).unwrap(), Op::Select { .. }));
        } else {
            assert!(
                matches!(forward.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input }
                if matches!(forward.op(*input).unwrap(), Op::Binary { op: BinaryOp::Mul, .. }))
            );
        }

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_trunc_div(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), expected_dtype);
        if dtype.is_integer() {
            assert!(matches!(reflected.op(output).unwrap(), Op::Select { .. }));
        } else {
            assert!(
                matches!(reflected.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input }
                if matches!(reflected.op(*input).unwrap(), Op::Binary { op: BinaryOp::Mul, .. }))
            );
        }
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.trunc_div_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_trunc_div(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.trunc_div_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2, 1], DType::U64);
    let output = bridge.trunc_div(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
    assert_eq!(bridge.shape(output).unwrap(), &Shape::new([2, 3]));

    // Integer scalar zero follows CDIV's typed-zero sentinel, while floating
    // scalar special values retain the literal reciprocal-Mul-Trunc branch.
    let mut sentinel = Graph::new();
    let input = sentinel.input_dtype("input", [2], DType::I16);
    let output = sentinel.trunc_div_scalar(input, Scalar::I(0)).unwrap();
    assert!(matches!(sentinel.op(output).unwrap(), Op::Select { .. }));
    assert!((0..sentinel.node_count()).any(|index| matches!(
        sentinel.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I16 && data.scalar_at(0).as_i64() == 0
    )));
    assert!(matches!(
        sentinel.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.trunc_div_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .scalar_trunc_div(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let nan = specials
        .scalar_trunc_div(Scalar::F(f64::NAN), input)
        .unwrap();
    assert!(matches!(
        specials.op(negative_zero).unwrap(),
        Op::Unary {
            op: UnaryOp::Trunc,
            ..
        }
    ));
    assert!(matches!(
        specials.op(infinity).unwrap(),
        Op::Unary {
            op: UnaryOp::Trunc,
            ..
        }
    ));
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Unary {
            op: UnaryOp::Trunc,
            ..
        }
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.dtype(gradient).unwrap(), DType::F64);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.trunc_div_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.trunc_div_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_trunc_div(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn floor_div_uses_tinygrad_python_floor_correction_and_zero_sentinel() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [5], DType::I16);
    let rhs = graph.input_dtype("rhs", [5], DType::U16);
    let output = graph.floor_div(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    let Op::Select {
        condition,
        on_true,
        on_false,
    } = graph.op(output).unwrap()
    else {
        panic!("expected zero-sentinel Select");
    };
    assert!(matches!(
        graph.op(*condition).unwrap(),
        Op::Compare {
            op: CompareOp::Ne,
            ..
        }
    ));
    assert!(matches!(graph.op(*on_true).unwrap(), Op::Constant(_)));
    assert!(matches!(graph.op(*on_false).unwrap(), Op::Select { .. }));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [5],
                DType::I16,
                [
                    Scalar::I(-3),
                    Scalar::I(3),
                    Scalar::I(-3),
                    Scalar::I(3),
                    Scalar::I(5),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [5],
                DType::U16,
                [
                    Scalar::U(2),
                    Scalar::U(2),
                    Scalar::U(2),
                    Scalar::U(2),
                    Scalar::U(0),
                ],
            )
            .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-2.0, 1.0, -2.0, 1.0, 0.0]
    );
    let loss = graph.sum_all(output).unwrap();
    assert!(matches!(
        graph.grad(loss, lhs),
        Err(Error::NonDifferentiableTarget(node)) if node == loss
    ));

    let mut negative_divisor = Graph::new();
    let lhs = negative_divisor.input_dtype("lhs", [2], DType::I64);
    let rhs = negative_divisor.input_dtype("rhs", [2], DType::I64);
    let output = negative_divisor.floor_div(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &negative_divisor,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(1), Scalar::I(i64::MIN)])
                        .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(-2), Scalar::I(-1)])
                        .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), -1);
    assert_eq!(values.scalar_at(1).as_i64(), i64::MIN);

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = wide.input_dtype("rhs", [2, 1], DType::U64);
    let output = wide.floor_div(lhs, rhs).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);
    assert_eq!(wide.shape(output).unwrap(), &Shape::new([2, 3]));
    assert!(matches!(wide.op(output).unwrap(), Op::Select { .. }));
    assert!(wide.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Unary {
            op: UnaryOp::Trunc,
            ..
        }
    )));
}

#[test]
fn floor_div_scalar_preserves_source_integer_and_float_branches() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let expected_dtype = if dtype.is_integer() || dtype.is_float() {
            dtype
        } else {
            DType::F32
        };
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.floor_div_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), expected_dtype);
        if dtype.is_integer() {
            assert!(matches!(forward.op(output).unwrap(), Op::Select { .. }));
        } else {
            assert!(matches!(forward.op(output).unwrap(), Op::Select { .. }));
            assert!(forward.nodes.iter().any(|node| matches!(
                &node.op,
                Op::Unary {
                    op: UnaryOp::Trunc,
                    ..
                }
            )));
        }

        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_floor_div(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), expected_dtype);
        if dtype.is_integer() {
            assert!(matches!(reflected.op(output).unwrap(), Op::Select { .. }));
        } else {
            assert!(matches!(reflected.op(output).unwrap(), Op::Select { .. }));
            assert!(reflected.nodes.iter().any(|node| matches!(
                &node.op,
                Op::Unary {
                    op: UnaryOp::Trunc,
                    ..
                }
            )));
        }
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.floor_div_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_floor_div(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.floor_div_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.floor_div(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    // Integer zero divisors retain the typed-zero sentinel and the full
    // correction Select tree; no host division or partial scalar publication.
    let mut sentinel = Graph::new();
    let input = sentinel.input_dtype("input", [2], DType::I16);
    let output = sentinel.floor_div_scalar(input, Scalar::I(0)).unwrap();
    assert!(matches!(sentinel.op(output).unwrap(), Op::Select { .. }));
    assert!((0..sentinel.node_count()).any(|index| matches!(
        sentinel.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I16 && data.scalar_at(0).as_i64() == 0
    )));
    assert!(matches!(
        sentinel.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.floor_div_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .scalar_floor_div(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let nan = specials
        .scalar_floor_div(Scalar::F(f64::NAN), input)
        .unwrap();
    for output in [negative_zero, infinity, nan] {
        assert!(matches!(specials.op(output).unwrap(), Op::Select { .. }));
    }
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.dtype(gradient).unwrap(), DType::F64);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.floor_div_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.floor_div_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_floor_div(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn floor_div_matches_tinygrad_bool_float_special_values_and_narrow_dtype() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [3], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [3], DType::Bool);
    let output = booleans.floor_div(lhs, rhs).unwrap();
    assert_eq!(booleans.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([3], [false, true, false])),
                ("rhs".into(), bool_data([3], [true, false, false])),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert!(values.scalar_at(1).as_f64().is_infinite());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [5], DType::F64);
    let rhs = special.input_dtype("rhs", [5], DType::F64);
    let output = special.floor_div(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(1.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::NAN),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(0.0),
                            Scalar::F(3.0),
                        ],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert_eq!(values.scalar_at(2).as_f64(), 0.0);
    assert!(values.scalar_at(3).as_f64().is_infinite());
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.floor_div(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    let output = narrow.floor_div(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::BF16);
}

#[test]
fn floor_div_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.floor_div(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn modulo_uses_tinygrad_floor_composition_lub_and_zero_sentinel() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [3], DType::I16);
    let rhs = graph.input_dtype("rhs", [3], DType::U16);
    let output = graph.modulo(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([3], DType::I16, [Scalar::I(-3), Scalar::I(3), Scalar::I(5)])
                .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([3], DType::U16, [Scalar::U(2), Scalar::U(2), Scalar::U(0)])
                .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 1.0, 5.0]
    );

    let mut negative_divisor = Graph::new();
    let lhs = negative_divisor.input_dtype("lhs", [2], DType::I64);
    let rhs = negative_divisor.input_dtype("rhs", [2], DType::I64);
    let output = negative_divisor.modulo(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &negative_divisor,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(i64::MIN)])
                        .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(-2), Scalar::I(-1)])
                        .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), -1);
    assert_eq!(values.scalar_at(1).as_i64(), 0);

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = wide.input_dtype("rhs", [2, 1], DType::U64);
    let output = wide.modulo(lhs, rhs).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);
    assert_eq!(wide.shape(output).unwrap(), &Shape::new([2, 3]));
}

#[test]
fn modulo_scalar_preserves_floor_composition_and_reflected_roles() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let expected = if dtype.is_integer() || dtype.is_float() {
            dtype
        } else {
            DType::F32
        };
        let mut forward = Graph::new();
        let input = forward.input_dtype("input", [2], dtype);
        let output = forward.modulo_scalar(input, value).unwrap();
        assert_eq!(forward.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(forward.dtype(output).unwrap(), expected);
        assert!(matches!(
            forward.op(output).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        let mut reflected = Graph::new();
        let input = reflected.input_dtype("input", [2], dtype);
        let output = reflected.scalar_modulo(value, input).unwrap();
        assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(reflected.dtype(output).unwrap(), expected);
        assert!(matches!(
            reflected.op(output).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.modulo_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.scalar_modulo(Scalar::F(-0.0), integral).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.modulo_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.modulo(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut sentinel = Graph::new();
    let input = sentinel.input_dtype("input", [2], DType::I16);
    let output = sentinel.modulo_scalar(input, Scalar::I(0)).unwrap();
    assert!(matches!(
        sentinel.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(
        (0..sentinel.node_count()).any(|index| matches!(sentinel.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I16 && data.scalar_at(0).as_i64() == 0))
    );
    assert!(matches!(
        sentinel.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.modulo_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .scalar_modulo(Scalar::F(f64::INFINITY), input)
        .unwrap();
    let nan = specials.scalar_modulo(Scalar::F(f64::NAN), input).unwrap();
    assert!(matches!(
        specials.op(negative_zero).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(infinity).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.modulo_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.modulo_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.scalar_modulo(Scalar::F(0.0), overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn fmod_scalar_preserves_non_reflected_trunc_composition() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let expected = if dtype.is_integer() || dtype.is_float() {
            dtype
        } else {
            DType::F32
        };
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let output = graph.fmod_scalar(input, value).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), expected);
        assert!(matches!(
            graph.op(output).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integral = mixed.input_dtype("integral", [], DType::I16);
    let narrow = mixed.input_dtype("narrow", [], DType::F16);
    let boolean_output = mixed.fmod_scalar(boolean, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(boolean_output).unwrap(), DType::I32);
    let integral_output = mixed.fmod_scalar(integral, Scalar::F(-0.0)).unwrap();
    assert_eq!(mixed.dtype(integral_output).unwrap(), DType::F32);
    let narrow_output = mixed.fmod_scalar(narrow, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(narrow_output).unwrap(), DType::F16);

    let mut bridge = Graph::new();
    let lhs = bridge.input_dtype("lhs", [2], DType::I64);
    let rhs = bridge.input_dtype("rhs", [2], DType::U64);
    let output = bridge.fmod(lhs, rhs).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);

    let mut sentinel = Graph::new();
    let input = sentinel.input_dtype("input", [2], DType::I16);
    let output = sentinel.fmod_scalar(input, Scalar::I(0)).unwrap();
    assert!(matches!(
        sentinel.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(
        (0..sentinel.node_count()).any(|index| matches!(sentinel.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I16 && data.scalar_at(0).as_i64() == 0))
    );
    assert!(matches!(
        sentinel.grad(output, input),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let mut specials = Graph::new();
    let input = specials.input_dtype("input", [2, 1], DType::F64);
    let negative_zero = specials.fmod_scalar(input, Scalar::F(-0.0)).unwrap();
    let infinity = specials
        .fmod_scalar(input, Scalar::F(f64::INFINITY))
        .unwrap();
    let nan = specials.fmod_scalar(input, Scalar::F(f64::NAN)).unwrap();
    assert!(matches!(
        specials.op(negative_zero).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(infinity).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let loss = specials.sum_all(negative_zero).unwrap();
    let gradient = specials.grad(loss, input).unwrap();
    assert_eq!(specials.shape(gradient).unwrap(), &Shape::new([2, 1]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.fmod_scalar(input, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.fmod_scalar(NodeId(usize::MAX), Scalar::F(0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.fmod_scalar(overflow, Scalar::F(0.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn modulo_matches_tinygrad_float_bool_special_values_and_vjp() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [3], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [3], DType::Bool);
    let output = booleans.modulo(lhs, rhs).unwrap();
    assert_eq!(booleans.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([3], [false, true, false])),
                ("rhs".into(), bool_data([3], [true, false, false])),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [5], DType::F64);
    let rhs = special.input_dtype("rhs", [5], DType::F64);
    let output = special.modulo(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(1.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::NAN),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(0.0),
                            Scalar::F(3.0),
                        ],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::F64);
    let rhs = graph.input_dtype("rhs", [1], DType::F64);
    let output = graph.modulo(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([2], DType::F64, [Scalar::F(1.0), Scalar::F(3.0)]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([1], DType::F64, [Scalar::F(2.0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.modulo(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    let output = narrow.modulo(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::BF16);
}

#[test]
fn modulo_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.modulo(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn fmod_uses_tinygrad_trunc_composition_lub_and_zero_sentinel() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [3], DType::I16);
    let rhs = graph.input_dtype("rhs", [3], DType::U16);
    let output = graph.fmod(lhs, rhs).unwrap();

    assert_eq!(graph.dtype(output).unwrap(), DType::I32);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([3], DType::I16, [Scalar::I(-3), Scalar::I(3), Scalar::I(5)])
                .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([3], DType::U16, [Scalar::U(2), Scalar::U(2), Scalar::U(0)])
                .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0, 1.0, 5.0]
    );

    let mut negative_divisor = Graph::new();
    let lhs = negative_divisor.input_dtype("lhs", [2], DType::I64);
    let rhs = negative_divisor.input_dtype("rhs", [2], DType::I64);
    let output = negative_divisor.fmod(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &negative_divisor,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(3), Scalar::I(i64::MIN)])
                        .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars([2], DType::I64, [Scalar::I(-2), Scalar::I(-1)])
                        .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_i64(), 1);
    assert_eq!(values.scalar_at(1).as_i64(), 0);

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1, 3], DType::I64);
    let rhs = wide.input_dtype("rhs", [2, 1], DType::U64);
    let output = wide.fmod(lhs, rhs).unwrap();
    assert_eq!(wide.dtype(output).unwrap(), DType::F32);
    assert_eq!(wide.shape(output).unwrap(), &Shape::new([2, 3]));
}

#[test]
fn fmod_matches_tinygrad_float_bool_special_values_and_vjp() {
    let mut booleans = Graph::new();
    let lhs = booleans.input_dtype("lhs", [3], DType::Bool);
    let rhs = booleans.input_dtype("rhs", [3], DType::Bool);
    let output = booleans.fmod(lhs, rhs).unwrap();
    assert_eq!(booleans.dtype(output).unwrap(), DType::F32);
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                ("lhs".into(), bool_data([3], [false, true, false])),
                ("rhs".into(), bool_data([3], [true, false, false])),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert!(values.scalar_at(2).as_f64().is_nan());

    let mut special = Graph::new();
    let lhs = special.input_dtype("lhs", [5], DType::F64);
    let rhs = special.input_dtype("rhs", [5], DType::F64);
    let output = special.fmod(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(-0.0),
                            Scalar::F(0.0),
                            Scalar::F(1.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(f64::NAN),
                        ],
                    )
                    .unwrap(),
                ),
                (
                    "rhs".into(),
                    TensorData::from_scalars(
                        [5],
                        DType::F64,
                        [
                            Scalar::F(2.0),
                            Scalar::F(0.0),
                            Scalar::F(f64::INFINITY),
                            Scalar::F(0.0),
                            Scalar::F(3.0),
                        ],
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(1).as_f64().is_nan());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::F64);
    let rhs = graph.input_dtype("rhs", [1], DType::F64);
    let output = graph.fmod(lhs, rhs).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let lhs_gradient = graph.grad(loss, lhs).unwrap();
    let rhs_gradient = graph.grad(loss, rhs).unwrap();
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([2], DType::F64, [Scalar::F(1.0), Scalar::F(3.0)]).unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars([1], DType::F64, [Scalar::F(2.0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, lhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, rhs_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    let output = narrow.fmod(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::F16);
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    let output = narrow.fmod(lhs, rhs).unwrap();
    assert_eq!(narrow.dtype(output).unwrap(), DType::BF16);
}

#[test]
fn fmod_preflights_source_casts_before_mutation() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [3], DType::U64);
    let node_count = graph.node_count();

    assert!(matches!(
        graph.fmod(lhs, rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn clip_is_a_clamp_alias_with_the_existing_vjp() {
    let mut graph = Graph::new();
    let input = graph.input("x", [3]);
    let min = graph.constant(TensorData::scalar(-1.0));
    let max = graph.constant(TensorData::scalar(1.0));
    let output = graph.clip(input, Some(min), Some(max)).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([3], vec![-2., 0.5, 3.]).unwrap(),
    )]);

    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1., 0.5, 1.]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0., 1., 0.]
    );
}

#[test]
fn clip_uses_tinygrad_strict_selects_for_ties_nans_and_gradients() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [4], DType::F64);
    let lower = graph.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F64));
    let upper = graph.constant(TensorData::scalar_with_dtype(Scalar::F(0.0), DType::F64));
    let output = graph.clip(input, Some(lower), Some(upper)).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::from_scalars(
            [4],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::NAN),
                Scalar::F(1.0),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    // Both strict predicates retain the data lane on equal signed-zero values
    // and on unordered NaN. The final positive lane selects the upper bound.
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert_eq!(
        CpuBackend
            .execute(&graph, input_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 1.0, 1.0, 0.0]
    );

    let mut inverted = Graph::new();
    let x = inverted.input("x", [1]);
    let min = inverted.constant(TensorData::scalar(2.0));
    let max = inverted.constant(TensorData::scalar(1.0));
    let output = inverted.clamp(x, Some(min), Some(max)).unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &inverted,
                output,
                &HashMap::from([("x".into(), TensorData::new([1], vec![0.0]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![1.0]
    );
}

#[test]
fn clip_preflights_and_applies_the_i64_u64_f32_bridge_per_stage() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2], DType::I64);
    let min = graph.input_dtype("min", [], DType::U64);
    let output = graph.clip(input, Some(min), None).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    let bindings = HashMap::from([
        (
            "x".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(-1), Scalar::I(1_i64 << 53)])
                .unwrap(),
        ),
        (
            "min".into(),
            TensorData::from_scalars([], DType::U64, [Scalar::U(0)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.0, (1_i64 << 53) as f32 as f64]
    );
}

#[test]
fn clip_preflights_both_bounds_before_graph_growth() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let valid_min = graph.constant(TensorData::scalar(-1.0));
    let incompatible_max = graph.constant(TensorData::new([2, 2], vec![1.; 4]).unwrap());
    let node_count = graph.node_count();

    assert!(matches!(
        graph.clip(input, Some(valid_min), Some(incompatible_max)),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn clip_rejects_bounds_that_only_conflict_with_each_other_without_graph_growth() {
    let mut graph = Graph::new();
    let input = graph.input("x", [1]);
    let min = graph.constant(TensorData::new([2], vec![-1., -2.]).unwrap());
    let max = graph.constant(TensorData::new([3], vec![1., 2., 3.]).unwrap());
    let node_count = graph.node_count();

    assert!(matches!(
        graph.clip(input, Some(min), Some(max)),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn clamp_scalar_bounds_preflight_and_keep_tinygrad_stage_order() {
    let cases = [
        Scalar::Bool(true),
        Scalar::I(-2),
        Scalar::U(3),
        Scalar::F(0.25),
    ];
    for bound in cases {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 1], DType::F16);
        let output = graph
            .clamp_with_scalars(input, Some(bound), Some(Scalar::F(1.0)))
            .unwrap();
        // The outer root is the upper strict Select; its false branch is the
        // lower strict Select, preserving tinygrad's lower-then-upper graph.
        let Op::Select { on_false, .. } = graph.op(output).unwrap() else {
            unreachable!()
        };
        assert!(matches!(graph.op(*on_false).unwrap(), Op::Select { .. }));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 1]));
    }

    for (dtype, lower, upper) in [
        (DType::Bool, Scalar::Bool(false), Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1), Scalar::I(1)),
        (DType::U8, Scalar::U(0), Scalar::U(1)),
        (DType::I16, Scalar::I(-1), Scalar::I(1)),
        (DType::U16, Scalar::U(0), Scalar::U(1)),
        (DType::I32, Scalar::I(-1), Scalar::I(1)),
        (DType::U32, Scalar::U(0), Scalar::U(1)),
        (DType::I64, Scalar::I(-1), Scalar::I(1)),
        (DType::U64, Scalar::U(0), Scalar::U(1)),
        (DType::BF16, Scalar::F(-1.0), Scalar::F(1.0)),
        (DType::F32, Scalar::F(-1.0), Scalar::F(1.0)),
        (DType::F64, Scalar::F(-1.0), Scalar::F(1.0)),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [], dtype);
        let output = graph.hardtanh_with_scalars(input, lower, upper).unwrap();
        assert_eq!(graph.dtype(output).unwrap(), dtype);
    }

    let mut lower_only = Graph::new();
    let input = lower_only.input_dtype("x", [], DType::Bool);
    let output = lower_only
        .clip_with_scalars(input, Some(Scalar::I(1)), None)
        .unwrap();
    // Bool plus a weak Python integer commits at tinygrad's default I32.
    assert_eq!(lower_only.dtype(output).unwrap(), DType::I32);
    assert!(matches!(lower_only.op(output).unwrap(), Op::Select { .. }));

    let mut bridge = Graph::new();
    let input = bridge.input_dtype("x", [2], DType::I64);
    let upper = bridge.input_dtype("upper", [], DType::U64);
    let output = bridge.clamp(input, None, Some(upper)).unwrap();
    assert_eq!(bridge.dtype(output).unwrap(), DType::F32);
}

#[test]
fn hardtanh_scalar_defaults_and_clamp_scalar_failures_are_atomic() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [], DType::BF16);
    let output = graph.hardtanh_default(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::BF16);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, input).is_ok());

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0, 3], DType::F32);
    let output = empty
        .hardtanh_with_scalars(input, Scalar::F(-0.0), Scalar::F(f64::NAN))
        .unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));

    let mut missing = Graph::new();
    let input = missing.input("x", [1]);
    let nodes = missing.node_count();
    assert!(missing.clamp_with_scalars(input, None, None).is_err());
    assert_eq!(missing.node_count(), nodes);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("x", [usize::MAX, 2], DType::F32);
    let nodes = overflow.node_count();
    assert!(
        overflow
            .clamp_with_scalars(input, Some(Scalar::F(-1.0)), None)
            .is_err()
    );
    assert_eq!(overflow.node_count(), nodes);
}

#[test]
fn extrema_keep_ordered_forward_selection_and_split_equal_tie_gradients() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [6], DType::F64);
    let rhs = graph.input_dtype("rhs", [6], DType::F64);
    let maximum = graph.maximum(lhs, rhs).unwrap();
    let minimum = graph.minimum(lhs, rhs).unwrap();
    let lhs_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [6],
                DType::F64,
                [
                    Scalar::F(lhs_nan),
                    Scalar::F(5.0),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [6],
                DType::F64,
                [
                    Scalar::F(3.0),
                    Scalar::F(f64::NAN),
                    Scalar::F(0.0),
                    Scalar::F(-0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::INFINITY),
                ],
            )
            .unwrap(),
        ),
    ]);

    let maximum = CpuBackend.execute(&graph, maximum, &bindings).unwrap();
    let minimum = CpuBackend.execute(&graph, minimum, &bindings).unwrap();
    // Ordered comparisons are false for NaN and equality, retaining the left
    // payload; minimum only selects the right lane when it is strictly lower.
    assert_eq!(maximum.scalar_at(0).as_f64().to_bits(), lhs_nan.to_bits());
    assert_eq!(maximum.scalar_at(1).as_f64(), 5.0);
    assert_eq!(maximum.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(maximum.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(maximum.scalar_at(4).as_f64(), f64::INFINITY);
    assert_eq!(minimum.scalar_at(0).as_f64().to_bits(), lhs_nan.to_bits());
    assert_eq!(minimum.scalar_at(1).as_f64(), 5.0);
    assert_eq!(minimum.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(minimum.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(minimum.scalar_at(4).as_f64(), f64::NEG_INFINITY);
    assert_eq!(minimum.scalar_at(5).as_f64(), f64::INFINITY);

    let mut ties = Graph::new();
    let lhs = ties.input("lhs", [2]);
    let rhs = ties.input("rhs", [2]);
    let output = ties.maximum(lhs, rhs).unwrap();
    let loss = ties.sum_all(output).unwrap();
    let lhs_gradient = ties.grad(loss, lhs).unwrap();
    let rhs_gradient = ties.grad(loss, rhs).unwrap();
    let equal = HashMap::from([
        ("lhs".into(), TensorData::new([2], vec![-0.0, 3.0]).unwrap()),
        ("rhs".into(), TensorData::new([2], vec![0.0, 3.0]).unwrap()),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&ties, lhs_gradient, &equal)
            .unwrap()
            .to_vec_f64(),
        vec![0.5; 2]
    );
    assert_eq!(
        CpuBackend
            .execute(&ties, rhs_gradient, &equal)
            .unwrap()
            .to_vec_f64(),
        vec![0.5; 2]
    );
}

#[test]
fn extrema_i64_u64_uses_the_source_f32_bridge_before_ordered_comparison() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2], DType::I64);
    let rhs = graph.input_dtype("rhs", [2], DType::U64);
    let output = graph.maximum(lhs, rhs).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Maximum,
            ..
        }
    ));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(1_i64 << 53), Scalar::I(-1)])
                .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [2],
                DType::U64,
                [Scalar::U((1_u64 << 53) + 1), Scalar::U(0)],
            )
            .unwrap(),
        ),
    ]);
    let output = CpuBackend.execute(&graph, output, &bindings).unwrap();
    // The first converted pair is equal at F32 precision, so ordered Max
    // preserves the left converted operand; the second selects zero.
    assert_eq!(output.scalar_at(0).as_f64(), (1_u64 << 53) as f32 as f64);
    assert_eq!(output.scalar_at(1).as_f64(), 0.0);
}

#[test]
fn extrema_cover_every_storage_family_without_changing_result_dtype() {
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let (lhs_values, rhs_values, maximum_values, minimum_values) = if dtype == DType::Bool {
            (
                vec![Scalar::Bool(false), Scalar::Bool(true)],
                vec![Scalar::Bool(true), Scalar::Bool(false)],
                vec![Scalar::Bool(true), Scalar::Bool(true)],
                vec![Scalar::Bool(false), Scalar::Bool(false)],
            )
        } else if dtype.is_float() {
            (
                vec![Scalar::F(1.0), Scalar::F(3.0)],
                vec![Scalar::F(2.0), Scalar::F(2.0)],
                vec![Scalar::F(2.0), Scalar::F(3.0)],
                vec![Scalar::F(1.0), Scalar::F(2.0)],
            )
        } else if dtype.category() == crate::DTypeCategory::Unsigned {
            (
                vec![Scalar::U(1), Scalar::U(3)],
                vec![Scalar::U(2), Scalar::U(2)],
                vec![Scalar::U(2), Scalar::U(3)],
                vec![Scalar::U(1), Scalar::U(2)],
            )
        } else {
            (
                vec![Scalar::I(1), Scalar::I(3)],
                vec![Scalar::I(2), Scalar::I(2)],
                vec![Scalar::I(2), Scalar::I(3)],
                vec![Scalar::I(1), Scalar::I(2)],
            )
        };
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [2], dtype);
        let rhs = graph.input_dtype("rhs", [2], dtype);
        let maximum = graph.maximum(lhs, rhs).unwrap();
        let minimum = graph.minimum(lhs, rhs).unwrap();
        assert_eq!(graph.dtype(maximum).unwrap(), dtype);
        assert_eq!(graph.dtype(minimum).unwrap(), dtype);
        let bindings = HashMap::from([
            (
                "lhs".into(),
                TensorData::from_scalars([2], dtype, lhs_values).unwrap(),
            ),
            (
                "rhs".into(),
                TensorData::from_scalars([2], dtype, rhs_values).unwrap(),
            ),
        ]);
        assert_eq!(
            CpuBackend.execute(&graph, maximum, &bindings).unwrap(),
            TensorData::from_scalars([2], dtype, maximum_values).unwrap(),
            "maximum {dtype:?}",
        );
        assert_eq!(
            CpuBackend.execute(&graph, minimum, &bindings).unwrap(),
            TensorData::from_scalars([2], dtype, minimum_values).unwrap(),
            "minimum {dtype:?}",
        );
    }
}

#[test]
fn extrema_scalar_commits_weak_rhs_before_the_existing_ordered_extrema_root() {
    for (dtype, value) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(1)),
        (DType::I16, Scalar::I(1)),
        (DType::I32, Scalar::I(1)),
        (DType::I64, Scalar::I(1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(1.0)),
        (DType::BF16, Scalar::F(1.0)),
        (DType::F32, Scalar::F(1.0)),
        (DType::F64, Scalar::F(1.0)),
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let maximum = graph.maximum_scalar(input, value).unwrap();
        let minimum = graph.minimum_scalar(input, value).unwrap();
        for (output, op) in [(maximum, BinaryOp::Maximum), (minimum, BinaryOp::Minimum)] {
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            let rhs = match graph.op(output).unwrap() {
                Op::Binary {
                    op: actual,
                    lhs,
                    rhs,
                } if *actual == op => {
                    assert_eq!(*lhs, input);
                    *rhs
                }
                actual => panic!("expected ordered extrema root, got {actual:?}"),
            };
            assert!(matches!(graph.op(rhs).unwrap(), Op::Constant(data)
                if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        }
    }

    // Weak integers and weak floats lift a Bool tensor to tinygrad's default
    // I32/F32 widths. A strong I64 tensor instead commits the same Python
    // integer at I64 (there is no live I64/U64 bridge in a scalar-right API).
    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integer = mixed.maximum_scalar(boolean, Scalar::I(1)).unwrap();
    let floating = mixed.minimum_scalar(boolean, Scalar::F(-0.0)).unwrap();
    let signed = mixed.input_dtype("signed", [], DType::I64);
    let wrapped = mixed.maximum_scalar(signed, Scalar::U(u64::MAX)).unwrap();
    assert_eq!(mixed.dtype(integer).unwrap(), DType::I32);
    assert_eq!(mixed.dtype(floating).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(wrapped).unwrap(), DType::I64);
    let integer_lhs = match mixed.op(integer).unwrap() {
        Op::Binary { lhs, .. } => *lhs,
        _ => unreachable!(),
    };
    let floating_rhs = match mixed.op(floating).unwrap() {
        Op::Binary { rhs, .. } => *rhs,
        _ => unreachable!(),
    };
    assert!(
        matches!(mixed.op(integer_lhs).unwrap(), Op::Cast { input, dtype: DType::I32 } if *input == boolean)
    );
    assert!(matches!(mixed.op(floating_rhs).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits()));

    // The Binary extrema root is the source ordered comparison/select
    // abstraction, so its existing VJP retains the left signed-zero/NaN
    // payload on equality or unordered lanes and splits equal gradients.
    let mut vjp = Graph::new();
    let input = vjp.input_dtype("input", [2], DType::F32);
    let output = vjp.maximum_scalar(input, Scalar::F(f64::NAN)).unwrap();
    let loss = vjp.sum_all(output).unwrap();
    let gradient = vjp.grad(loss, input).unwrap();
    assert_eq!(vjp.shape(gradient).unwrap(), &Shape::new([2]));
    assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let input = empty.input_dtype("empty", [0, 2], DType::F16);
    let output = empty.minimum_scalar(input, Scalar::F(0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.maximum_scalar(NodeId(usize::MAX), Scalar::I(1)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.minimum_scalar(overflow, Scalar::F(1.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn copysign_scalar_commits_the_weak_rhs_before_its_literal_predicate_graph() {
    for (dtype, sign) in [
        (DType::Bool, Scalar::Bool(true)),
        (DType::I8, Scalar::I(-1)),
        (DType::I16, Scalar::I(-1)),
        (DType::I32, Scalar::I(-1)),
        (DType::I64, Scalar::I(-1)),
        (DType::U8, Scalar::U(1)),
        (DType::U16, Scalar::U(1)),
        (DType::U32, Scalar::U(1)),
        (DType::U64, Scalar::U(1)),
        (DType::F16, Scalar::F(-0.0)),
        (DType::BF16, Scalar::F(-0.0)),
        (DType::F32, Scalar::F(-0.0)),
        (DType::F64, Scalar::F(-0.0)),
    ] {
        let mut graph = Graph::new();
        let magnitude = graph.input_dtype("magnitude", [2], dtype);
        let output = graph.copysign_scalar(magnitude, sign).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        // The first published node is the prepared RHS scalar. The public
        // lowerer must then retain its literal reciprocal/ordered-compare/OR
        // predicate and final Select rather than raw BinaryOp::Copysign.
        assert!(matches!(graph.op(NodeId(1)).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
        assert!((0..graph.node_count()).any(|index| matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Reciprocal,
                ..
            }
        )));
        assert!((0..graph.node_count()).any(|index| matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Logical {
                op: LogicalOp::Or,
                ..
            }
        )));
        if dtype.is_float() {
            let Op::Constant(data) = graph.op(NodeId(1)).unwrap() else {
                panic!("prepared scalar must be a constant");
            };
            assert_eq!(data.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        }
    }

    // A scalar never becomes a live U64 operand: weak integers commit at the
    // tensor's width. The live I64/U64 form remains the separately observable
    // source F32 bridge and both operands are cast before the literal graph.
    let mut mixed = Graph::new();
    let boolean = mixed.input_dtype("boolean", [], DType::Bool);
    let integer = mixed.copysign_scalar(boolean, Scalar::I(1)).unwrap();
    let floating = mixed.copysign_scalar(boolean, Scalar::F(-0.0)).unwrap();
    let signed = mixed.input_dtype("signed", [], DType::I64);
    let wrapped = mixed.copysign_scalar(signed, Scalar::U(u64::MAX)).unwrap();
    let unsigned = mixed.input_dtype("unsigned", [], DType::U64);
    let bridged = mixed.copysign(signed, unsigned).unwrap();
    assert_eq!(mixed.dtype(integer).unwrap(), DType::I32);
    assert_eq!(mixed.dtype(floating).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(wrapped).unwrap(), DType::I64);
    assert_eq!(mixed.dtype(bridged).unwrap(), DType::F32);

    let mut specials = Graph::new();
    let magnitude = specials.input_dtype("magnitude", [], DType::F64);
    let nan = specials
        .copysign_scalar(magnitude, Scalar::F(f64::NAN))
        .unwrap();
    let infinity = specials
        .copysign_scalar(magnitude, Scalar::F(f64::NEG_INFINITY))
        .unwrap();
    assert!(matches!(specials.op(NodeId(1)).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().is_nan()));
    assert!(matches!(specials.op(infinity).unwrap(), Op::Select { .. }));
    assert!(matches!(specials.op(nan).unwrap(), Op::Select { .. }));

    // The shared literal lowerer retains select-based VJP routing for a live
    // floating magnitude while the prepared scalar is non-differentiable.
    let mut vjp = Graph::new();
    let magnitude = vjp.input_dtype("magnitude", [2], DType::F32);
    let output = vjp.copysign_scalar(magnitude, Scalar::F(-0.0)).unwrap();
    let loss = vjp.sum_all(output).unwrap();
    let gradient = vjp.grad(loss, magnitude).unwrap();
    assert_eq!(vjp.shape(gradient).unwrap(), &Shape::new([2]));
    assert_eq!(vjp.dtype(gradient).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let magnitude = empty.input_dtype("magnitude", [0, 2], DType::BF16);
    let output = empty.copysign_scalar(magnitude, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.copysign_scalar(NodeId(usize::MAX), Scalar::F(-0.0)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let magnitude = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.copysign_scalar(magnitude, Scalar::F(-0.0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn squeeze_of_a_nonunit_axis_is_a_tinygrad_style_noop() {
    let mut graph = Graph::new();
    let input = graph.input("x", [2, 3]);
    let node_count = graph.node_count();

    let output = graph.squeeze(input, Some(-1)).unwrap();
    assert_eq!(output, input);
    assert_eq!(graph.node_count(), node_count);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "x".into(),
        TensorData::new([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    )]);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.; 6]
    );
}

#[test]
fn isinf_sign_selection_preserves_tinygrad_predicate_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let positive = graph.isinf_with_signs(input, true, false).unwrap();
    let negative = graph.isinf_with_signs(input, false, true).unwrap();
    let neither = graph.isinf_with_signs(input, false, false).unwrap();
    let both = graph.isinf_with_signs(input, true, true).unwrap();
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let scalar_positive = graph.isinf_with_signs(scalar, true, false).unwrap();
    let integers = graph.input_dtype("integers", [2], DType::I32);
    let integer_positive = graph.isinf_with_signs(integers, true, false).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::NAN),
                ],
            )
            .unwrap(),
        ),
        ("scalar".into(), TensorData::scalar(f32::INFINITY)),
        (
            "integers".into(),
            TensorData::from_scalars([2], DType::I32, [Scalar::I(-1), Scalar::I(0)]).unwrap(),
        ),
    ]);
    for (node, expected) in [
        (positive, vec![false, false, false, true, false]),
        (negative, vec![true, false, false, false, false]),
        (neither, vec![false; 5]),
        (both, vec![true, false, false, true, false]),
        (integer_positive, vec![false; 2]),
    ] {
        let output = CpuBackend.execute(&graph, node, &bindings).unwrap();
        assert_eq!(output.dtype(), DType::Bool);
        assert_eq!(output.storage(), &crate::Storage::Bool(expected));
    }
    assert_eq!(
        CpuBackend
            .execute(&graph, scalar_positive, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![true])
    );

    assert!(matches!(
        graph.grad(positive, input),
        Err(Error::NonDifferentiableTarget(node)) if node == positive
    ));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.isinf_with_signs(input, false, true).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F32, []).unwrap(),
                )]),
            )
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![])
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.isinf_with_signs(NodeId(usize::MAX), true, false),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn sign_uses_tinygrad_ordered_nan_and_signed_zero_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.sign(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sign, input: signed }
        if *signed == input)
    );
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    let scalar_output = graph.sign(scalar).unwrap();
    let integers = graph.input_dtype("integers", [3], DType::I32);
    let integer_output = graph.sign(integers).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::from_scalars(
                [5],
                DType::F64,
                [
                    Scalar::F(f64::NEG_INFINITY),
                    Scalar::F(-0.0),
                    Scalar::F(0.0),
                    Scalar::F(f64::INFINITY),
                    Scalar::F(f64::NAN),
                ],
            )
            .unwrap(),
        ),
        ("scalar".into(), TensorData::scalar(-0.0)),
        (
            "integers".into(),
            TensorData::from_scalars([3], DType::I32, [Scalar::I(-3), Scalar::I(0), Scalar::I(4)])
                .unwrap(),
        ),
    ]);
    let values = CpuBackend
        .execute(&graph, output, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(values, vec![-1.0, 0.0, 0.0, 1.0, 1.0]);
    assert!(values[1].is_sign_positive());
    assert!(values[2].is_sign_positive());
    let scalar_value = CpuBackend
        .execute(&graph, scalar_output, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    assert_eq!(scalar_value, 0.0);
    assert!(scalar_value.is_sign_positive());
    assert_eq!(
        CpuBackend
            .execute(&graph, integer_output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0, 0.0, 1.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.0; 5]
    );

    let mut discrete = Graph::new();
    let boolean = discrete.input_dtype("boolean", [2], DType::Bool);
    let signed = discrete.input_dtype("signed", [2], DType::I64);
    let unsigned = discrete.input_dtype("unsigned", [2], DType::U64);
    let boolean_output = discrete.sign(boolean).unwrap();
    let signed_output = discrete.sign(signed).unwrap();
    let unsigned_output = discrete.sign(unsigned).unwrap();
    let f16 = discrete.input_dtype("f16", [], DType::F16);
    let bf16 = discrete.input_dtype("bf16", [], DType::BF16);
    let f16_output = discrete.sign(f16).unwrap();
    assert_eq!(discrete.dtype(f16_output).unwrap(), DType::F16);
    let bf16_output = discrete.sign(bf16).unwrap();
    assert_eq!(discrete.dtype(bf16_output).unwrap(), DType::BF16);
    let bindings = HashMap::from([
        ("boolean".into(), bool_data([2], [false, true])),
        (
            "signed".into(),
            TensorData::from_scalars([2], DType::I64, [Scalar::I(i64::MIN), Scalar::I(4)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([2], DType::U64, [Scalar::U(0), Scalar::U(u64::MAX)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&discrete, boolean_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![false, true])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, signed_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::I64(vec![-1, 1])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, unsigned_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::U64(vec![0, 1])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.sign(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F32, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.sign(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn reciprocal_preserves_tinygrad_alu_dtype_special_and_vjp_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [6], DType::F64);
    let output = graph.reciprocal(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input: reciprocal }
        if *reciprocal == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::from_scalars(
            [6],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(f64::NAN),
                Scalar::F(2.0),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(1).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert_eq!(values.scalar_at(5).as_f64(), 0.5);

    let mut differentiable = Graph::new();
    let input = differentiable.input_dtype("input", [2], DType::F64);
    let output = differentiable.reciprocal(input).unwrap();
    let loss = differentiable.sum_all(output).unwrap();
    let gradient = differentiable.grad(loss, input).unwrap();
    assert_eq!(
        CpuBackend
            .execute(
                &differentiable,
                gradient,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([2], DType::F64, [Scalar::F(2.0), Scalar::F(-4.0)])
                        .unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![-0.25, -0.0625]
    );

    let mut nonfloat = Graph::new();
    let boolean = nonfloat.input_dtype("boolean", [2], DType::Bool);
    let signed = nonfloat.input_dtype("signed", [1], DType::I64);
    let unsigned = nonfloat.input_dtype("unsigned", [1], DType::U64);
    let boolean_output = nonfloat.reciprocal(boolean).unwrap();
    let signed_output = nonfloat.reciprocal(signed).unwrap();
    let unsigned_output = nonfloat.reciprocal(unsigned).unwrap();
    assert_eq!(nonfloat.dtype(boolean_output).unwrap(), DType::F32);
    assert_eq!(nonfloat.dtype(signed_output).unwrap(), DType::F32);
    assert_eq!(nonfloat.dtype(unsigned_output).unwrap(), DType::F32);
    macro_rules! assert_nonfloat_reciprocal {
        ($source:expr, $output:expr) => {{
            let Op::Unary {
                op: UnaryOp::Reciprocal,
                input: reciprocal_input,
            } = nonfloat.op($output).unwrap()
            else {
                panic!("nonfloat reciprocal must remain a raw reciprocal after promotion");
            };
            assert_eq!(nonfloat.dtype(*reciprocal_input).unwrap(), DType::F32);
            assert!(matches!(nonfloat.op(*reciprocal_input).unwrap(), Op::Cast { input, dtype }
                if *input == $source && *dtype == DType::F32));
        }};
    }
    assert_nonfloat_reciprocal!(boolean, boolean_output);
    assert_nonfloat_reciprocal!(signed, signed_output);
    assert_nonfloat_reciprocal!(unsigned, unsigned_output);
    for (name, dtype) in [
        ("i8", DType::I8),
        ("u8", DType::U8),
        ("i16", DType::I16),
        ("u16", DType::U16),
        ("i32", DType::I32),
        ("u32", DType::U32),
    ] {
        let source = nonfloat.input_dtype(name, [1], dtype);
        let output = nonfloat.reciprocal(source).unwrap();
        assert_nonfloat_reciprocal!(source, output);
    }
    let f16 = nonfloat.input_dtype("f16", [], DType::F16);
    let bf16 = nonfloat.input_dtype("bf16", [], DType::BF16);
    let f16_output = nonfloat.reciprocal(f16).unwrap();
    let bf16_output = nonfloat.reciprocal(bf16).unwrap();
    assert_eq!(nonfloat.dtype(f16_output).unwrap(), DType::F16);
    assert_eq!(nonfloat.dtype(bf16_output).unwrap(), DType::BF16);
    assert!(
        matches!(nonfloat.op(f16_output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input }
        if *input == f16)
    );
    assert!(
        matches!(nonfloat.op(bf16_output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input }
        if *input == bf16)
    );
    let bindings = HashMap::from([
        ("boolean".into(), bool_data([2], [false, true])),
        (
            "signed".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(-2)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([1], DType::U64, [Scalar::U(4)]).unwrap(),
        ),
    ]);
    let boolean_values = CpuBackend
        .execute(&nonfloat, boolean_output, &bindings)
        .unwrap();
    assert_eq!(boolean_values.scalar_at(0).as_f64(), f64::INFINITY);
    assert_eq!(boolean_values.scalar_at(1).as_f64(), 1.0);
    assert_eq!(
        CpuBackend
            .execute(&nonfloat, signed_output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-0.5]
    );
    assert_eq!(
        CpuBackend
            .execute(&nonfloat, unsigned_output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.25]
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F16);
    let output = empty.reciprocal(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.reciprocal(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn exp_uses_tinygrad_exp2_promotion_special_values_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.exp(input).unwrap();
    let Op::Unary {
        op: UnaryOp::Exp2,
        input: exponent,
    } = graph.op(output).unwrap()
    else {
        panic!("tinygrad F64 exp must end in Exp2");
    };
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs,
        rhs,
    } = graph.op(*exponent).unwrap()
    else {
        panic!("tinygrad exp must scale before Exp2");
    };
    assert_eq!(*lhs, input);
    assert_eq!(graph.dtype(*rhs).unwrap(), DType::F64);
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);

    let bindings = HashMap::from([(
        "input".into(),
        TensorData::from_scalars(
            [5],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(f64::NAN),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 1.0);
    assert_eq!(values.scalar_at(1).as_f64(), 1.0);
    assert_eq!(values.scalar_at(2).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert!(values.scalar_at(4).as_f64().is_nan());

    let mut differentiable = Graph::new();
    let input = differentiable.input_dtype("input", [2], DType::F64);
    let output = differentiable.exp(input).unwrap();
    let loss = differentiable.sum_all(output).unwrap();
    let gradient = differentiable.grad(loss, input).unwrap();
    let gradient_values = CpuBackend
        .execute(
            &differentiable,
            gradient,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([2], DType::F64, [Scalar::F(0.0), Scalar::F(1.0)])
                    .unwrap(),
            )]),
        )
        .unwrap()
        .to_vec_f64();
    assert!((gradient_values[0] - 1.0).abs() < 1e-12);
    assert!((gradient_values[1] - std::f64::consts::E).abs() < 1e-12);

    let mut promoted = Graph::new();
    let f16 = promoted.input_dtype("f16", [1], DType::F16);
    let bf16 = promoted.input_dtype("bf16", [1], DType::BF16);
    let boolean = promoted.input_dtype("boolean", [1], DType::Bool);
    let signed = promoted.input_dtype("signed", [1], DType::I64);
    let unsigned = promoted.input_dtype("unsigned", [1], DType::U64);
    let f16_output = promoted.exp(f16).unwrap();
    let bf16_output = promoted.exp(bf16).unwrap();
    let boolean_output = promoted.exp(boolean).unwrap();
    let signed_output = promoted.exp(signed).unwrap();
    let unsigned_output = promoted.exp(unsigned).unwrap();
    assert_eq!(promoted.dtype(f16_output).unwrap(), DType::F16);
    assert_eq!(promoted.dtype(bf16_output).unwrap(), DType::BF16);
    assert_eq!(promoted.dtype(boolean_output).unwrap(), DType::F32);
    assert_eq!(promoted.dtype(signed_output).unwrap(), DType::F32);
    assert_eq!(promoted.dtype(unsigned_output).unwrap(), DType::F32);
    let Op::Cast {
        input: f16_exp2,
        dtype: DType::F16,
    } = promoted.op(f16_output).unwrap()
    else {
        panic!("tinygrad F16 exp must narrow after F32 Exp2");
    };
    assert!(matches!(
        promoted.op(*f16_exp2).unwrap(),
        Op::Unary {
            op: UnaryOp::Exp2,
            ..
        }
    ));
    let bindings = HashMap::from([
        ("boolean".into(), bool_data([1], [true])),
        (
            "signed".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([1], DType::U64, [Scalar::U(0)]).unwrap(),
        ),
    ]);
    assert!(
        (CpuBackend
            .execute(&promoted, boolean_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64()
            - std::f64::consts::E)
            .abs()
            < 1e-5
    );
    assert!(
        (CpuBackend
            .execute(&promoted, signed_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64()
            - (-1.0f64).exp())
        .abs()
            < 1e-5
    );
    assert_eq!(
        CpuBackend
            .execute(&promoted, unsigned_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        1.0
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F16);
    let output = empty.exp(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.exp(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    assert!(differentiable.node(gradient).is_ok());
}

#[test]
fn exp2_preserves_tinygrad_storage_width_special_values_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.exp2(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Exp2, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(3.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 1.0);
    assert_eq!(values.scalar_at(1).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(2).as_f64(), 0.0);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), 8.0);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let gradient_values = CpuBackend
        .execute(
            &graph,
            gradient,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([5], DType::F64, [Scalar::F(0.0); 5]).unwrap(),
            )]),
        )
        .unwrap()
        .to_vec_f64();
    assert!(
        gradient_values
            .iter()
            .all(|value| (*value - std::f64::consts::LN_2).abs() < 1e-12)
    );

    let mut dtypes = Graph::new();
    let f16 = dtypes.input_dtype("f16", [1], DType::F16);
    let bf16 = dtypes.input_dtype("bf16", [1], DType::BF16);
    let boolean = dtypes.input_dtype("boolean", [1], DType::Bool);
    let signed = dtypes.input_dtype("signed", [1], DType::I64);
    let unsigned = dtypes.input_dtype("unsigned", [1], DType::U64);
    let f16_output = dtypes.exp2(f16).unwrap();
    let bf16_output = dtypes.exp2(bf16).unwrap();
    assert_eq!(dtypes.dtype(f16_output).unwrap(), DType::F16);
    assert_eq!(dtypes.dtype(bf16_output).unwrap(), DType::BF16);
    let boolean_output = dtypes.exp2(boolean).unwrap();
    let signed_output = dtypes.exp2(signed).unwrap();
    let unsigned_output = dtypes.exp2(unsigned).unwrap();
    assert_eq!(dtypes.dtype(boolean_output).unwrap(), DType::F32);
    assert_eq!(dtypes.dtype(signed_output).unwrap(), DType::F32);
    assert_eq!(dtypes.dtype(unsigned_output).unwrap(), DType::F32);
    for (name, dtype) in [
        ("i8", DType::I8),
        ("u8", DType::U8),
        ("i16", DType::I16),
        ("u16", DType::U16),
        ("i32", DType::I32),
        ("u32", DType::U32),
    ] {
        let input = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.exp2(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), DType::F32);
    }
    let bindings = HashMap::from([
        ("boolean".into(), bool_data([1], [true])),
        (
            "signed".into(),
            TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([1], DType::U64, [Scalar::U(3)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&dtypes, boolean_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        2.0
    );
    assert_eq!(
        CpuBackend
            .execute(&dtypes, signed_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        0.5
    );
    assert_eq!(
        CpuBackend
            .execute(&dtypes, unsigned_output, &bindings)
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        8.0
    );

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("input", [1], DType::F16);
    let output = narrow.exp2(input).unwrap();
    let loss = narrow.sum_all(output).unwrap();
    let gradient = narrow.grad(loss, input).unwrap();
    assert_eq!(narrow.dtype(gradient).unwrap(), DType::F16);
    assert!(
        (CpuBackend
            .execute(
                &narrow,
                gradient,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([1], DType::F16, [Scalar::F(0.0)]).unwrap(),
                )]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64()
            - std::f64::consts::LN_2)
            .abs()
            < 1e-3
    );
    let input = narrow.input_dtype("bf16", [1], DType::BF16);
    let output = narrow.exp2(input).unwrap();
    let loss = narrow.sum_all(output).unwrap();
    let gradient = narrow.grad(loss, input).unwrap();
    assert_eq!(narrow.dtype(gradient).unwrap(), DType::BF16);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.exp2(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.exp2(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.exp2(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn sqrt_preserves_direct_storage_width_special_values_and_typed_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.sqrt(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sqrt, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [8],
                    DType::F64,
                    [
                        Scalar::F(4.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(9.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 2.0);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    assert_eq!(values.scalar_at(7).as_f64(), 3.0);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let has_f64_two = (0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Constant(data)
            if data.dtype() == DType::F64 && data.shape().rank() == 0 && data.scalar_at(0).as_f64() == 2.0)
    });
    assert!(has_f64_two);
    assert_eq!(
        CpuBackend
            .execute(
                &graph,
                gradient,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([8], DType::F64, [Scalar::F(4.0); 8]).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![0.25; 8]
    );

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("f64", DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32),
        ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32),
        ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32),
        ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.sqrt(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
        let Op::Unary {
            op: UnaryOp::Sqrt,
            input: sqrt_input,
        } = dtypes.op(output).unwrap()
        else {
            panic!("public sqrt must end in its raw SQRT ALU");
        };
        if dtype.is_float() {
            assert_eq!(*sqrt_input, source, "{dtype:?} stays a homogeneous SQRT");
        } else {
            assert!(
                matches!(dtypes.op(*sqrt_input).unwrap(), Op::Cast { input, dtype: DType::F32 }
                if *input == source),
                "{dtype:?} must use Cast(F32) before SQRT"
            );
        }
        // The public cast makes the nonfloat unary UOp homogeneous, while
        // retaining the raw UnaryOp::Sqrt node for downstream backends.
        assert!(
            crate::lower_graph_elementwise(&dtypes, output).is_ok(),
            "{dtype:?} lowers"
        );
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.sqrt(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut narrow = Graph::new();
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        let source = narrow.input_dtype(name, [1], dtype);
        let output = narrow.sqrt(source).unwrap();
        let loss = narrow.sum_all(output).unwrap();
        let gradient = narrow.grad(loss, source).unwrap();
        assert_eq!(narrow.dtype(gradient).unwrap(), dtype);
        assert!((0..narrow.node_count()).any(|index| {
            matches!(narrow.op(NodeId(index)).unwrap(), Op::Constant(data)
                if data.dtype() == dtype && data.shape().rank() == 0 && data.scalar_at(0).as_f64() == 2.0)
        }));
    }
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.sqrt(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.sqrt(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.sqrt(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn rsqrt_uses_tinygrad_sqrt_then_reciprocal_structure_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.rsqrt(input).unwrap();
    let Op::Unary {
        op: UnaryOp::Reciprocal,
        input: root,
    } = graph.op(output).unwrap()
    else {
        panic!("rsqrt must end in reciprocal");
    };
    assert!(
        matches!(graph.op(*root).unwrap(), Op::Unary { op: UnaryOp::Sqrt, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(4.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.5);
    assert_eq!(values.scalar_at(1).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(2).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), 0.0);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Rsqrt,
                    ..
                }
            )
        }),
        "the public VJP must follow the source composition, not raw Rsqrt"
    );

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("f64", DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32),
        ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32),
        ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32),
        ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.rsqrt(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
        let Op::Unary {
            op: UnaryOp::Reciprocal,
            input: root,
        } = dtypes.op(output).unwrap()
        else {
            panic!("rsqrt must remain compositional");
        };
        assert_eq!(dtypes.dtype(*root).unwrap(), output_dtype);
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.rsqrt(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.rsqrt(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.rsqrt(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.rsqrt(input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn square_uses_tinygrad_self_multiplication_structure_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.square(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Mul, lhs, rhs }
        if *lhs == input && *rhs == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-2.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(3.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 4.0);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(values.scalar_at(6).as_f64(), 9.0);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Square,
                    ..
                }
            )
        }),
        "the public VJP must retain self-multiplication rather than raw Square"
    );

    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i8", DType::I8),
        ("u8", DType::U8),
        ("i16", DType::I16),
        ("u16", DType::U16),
        ("i32", DType::I32),
        ("u32", DType::U32),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
        ("f32", DType::F32),
        ("f64", DType::F64),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.square(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), dtype);
        assert!(
            matches!(dtypes.op(output).unwrap(), Op::Binary { op: BinaryOp::Mul, lhs, rhs }
            if *lhs == source && *rhs == source)
        );
    }
    let signed_min = dtypes.input_dtype("signed_min", [1], DType::I64);
    let unsigned = dtypes.input_dtype("unsigned_wrap", [1], DType::U64);
    let signed_output = dtypes.square(signed_min).unwrap();
    let unsigned_output = dtypes.square(unsigned).unwrap();
    let integer_values = CpuBackend
        .execute(
            &dtypes,
            signed_output,
            &HashMap::from([
                (
                    "signed_min".into(),
                    TensorData::from_scalars([1], DType::I64, [Scalar::I(i64::MIN)]).unwrap(),
                ),
                (
                    "unsigned_wrap".into(),
                    TensorData::from_scalars([1], DType::U64, [Scalar::U(u64::MAX)]).unwrap(),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(integer_values.scalar_at(0).as_i64(), 0);
    assert_eq!(
        CpuBackend
            .execute(
                &dtypes,
                unsigned_output,
                &HashMap::from([
                    (
                        "signed_min".into(),
                        TensorData::from_scalars([1], DType::I64, [Scalar::I(i64::MIN)]).unwrap(),
                    ),
                    (
                        "unsigned_wrap".into(),
                        TensorData::from_scalars([1], DType::U64, [Scalar::U(u64::MAX)]).unwrap(),
                    ),
                ]),
            )
            .unwrap()
            .scalar_at(0)
            .as_u64(),
        1
    );

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.square(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.square(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.square(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.square(input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn sin_preserves_direct_storage_and_tinygrad_phase_shift_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.sin(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sin, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(std::f64::consts::FRAC_PI_2),
                        Scalar::F(1.0e20),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(2).as_f64() - 1.0).abs() < 1e-12);
    assert_eq!(values.scalar_at(3).as_f64(), (1.0e20f64).sin());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let has_f64_half_pi = (0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Constant(data)
            if data.dtype() == DType::F64
                && data.shape().rank() == 0
                && data.scalar_at(0).as_f64() == std::f64::consts::FRAC_PI_2)
    });
    assert!(has_f64_half_pi);
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Cos,
                    ..
                }
            )
        }),
        "tinygrad differentiates sin through a phase-shifted Sin, not Cos"
    );
    let expected = (std::f64::consts::FRAC_PI_2 - 0.0).sin();
    assert!(
        CpuBackend
            .execute(
                &graph,
                gradient,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([7], DType::F64, [Scalar::F(0.0); 7]).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .iter()
            .all(|value| (*value - expected).abs() < 1e-12)
    );

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("f64", DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32),
        ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32),
        ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32),
        ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.sin(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
    }
    let mut narrow = Graph::new();
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        let source = narrow.input_dtype(name, [1], dtype);
        let output = narrow.sin(source).unwrap();
        let loss = narrow.sum_all(output).unwrap();
        let gradient = narrow.grad(loss, source).unwrap();
        assert_eq!(narrow.dtype(gradient).unwrap(), dtype);
        assert!((0..narrow.node_count()).any(|index| {
            matches!(narrow.op(NodeId(index)).unwrap(), Op::Constant(data)
                if data.dtype() == dtype
                    && data.shape().rank() == 0
                    && (data.scalar_at(0).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-2)
        }));
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.sin(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.sin(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.sin(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.sin(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn cos_uses_tinygrad_widened_phase_shift_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.cos(input).unwrap();
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Unary {
            op: UnaryOp::Sin,
            ..
        }
    ));
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Cos,
                    ..
                }
            )
        }),
        "public cosine must use the source phase-shifted Sin"
    );
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(std::f64::consts::FRAC_PI_2),
                        Scalar::F(1.0e20),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert!((values.scalar_at(0).as_f64() - 1.0).abs() < 1e-12);
    assert!((values.scalar_at(1).as_f64() - 1.0).abs() < 1e-12);
    assert!(values.scalar_at(2).as_f64().abs() < 1e-12);
    assert_eq!(
        values.scalar_at(3).as_f64(),
        (std::f64::consts::FRAC_PI_2 - 1.0e20).sin()
    );
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Cos,
                    ..
                }
            )
        }),
        "the public VJP must inherit the phase-shifted Sin graph"
    );

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype, work_dtype) in [
        ("f16", DType::F16, DType::F16, DType::F32),
        ("bf16", DType::BF16, DType::BF16, DType::F32),
        ("f32", DType::F32, DType::F32, DType::F32),
        ("f64", DType::F64, DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32, DType::F32),
        ("i8", DType::I8, DType::F32, DType::F32),
        ("u8", DType::U8, DType::F32, DType::F32),
        ("i16", DType::I16, DType::F32, DType::F32),
        ("u16", DType::U16, DType::F32, DType::F32),
        ("i32", DType::I32, DType::F32, DType::F32),
        ("u32", DType::U32, DType::F32, DType::F32),
        ("i64", DType::I64, DType::F32, DType::F32),
        ("u64", DType::U64, DType::F32, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.cos(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
        assert!((0..dtypes.node_count()).any(|index| {
            matches!(dtypes.op(NodeId(index)).unwrap(), Op::Constant(data)
                if data.dtype() == work_dtype
                    && data.shape().rank() == 0
                    && (data.scalar_at(0).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-2)
        }));
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.cos(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.cos(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.cos(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.cos(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn tan_uses_tinygrad_sin_cos_true_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.tan(input).unwrap();
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(
                graph.op(NodeId(index)).unwrap(),
                Op::Unary {
                    op: UnaryOp::Tan,
                    ..
                }
            )
        }),
        "public tan must use source Sin/Cos division"
    );
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(std::f64::consts::FRAC_PI_2),
                        Scalar::F(1.0e20),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(
        values.scalar_at(3).as_f64(),
        (1.0e20f64).sin() / (std::f64::consts::FRAC_PI_2 - 1.0e20).sin()
    );
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Tan,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("f64", DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32),
        ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32),
        ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32),
        ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.tan(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F16);
    let output = scalar.tan(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.tan(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.tan(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.tan(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn asin_uses_tinygrad_polynomial_structure_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.asin(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Asin,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Sqrt,
                ..
            }
        )
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(2.0),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() + std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.asin(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
    }
    let node_count = graph.node_count();
    assert!(matches!(
        graph.asin(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.asin(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn acos_uses_tinygrad_half_pi_minus_asin_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.acos(input).unwrap();
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            rhs,
            ..
        } if matches!(graph.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Neg, .. })
    ));
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Acos,
            ..
        }
    )));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(2.0),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert!((values.scalar_at(0).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!(values.scalar_at(1).as_f64().abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() - std::f64::consts::PI).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.acos(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.acos(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn atan_uses_tinygrad_sqrt_div_asin_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.atan(input).unwrap();
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Atan,
            ..
        }
    )));
    assert!((0..graph.node_count()).any(|index| matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Sqrt,
            ..
        }
    )));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [5],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() + std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.atan(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.atan(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn sinh_uses_tinygrad_exp_difference_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.sinh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Sinh,
                ..
            }
        )
    }));
    assert!(
        (0..graph.node_count())
            .filter(|index| {
                matches!(
                    graph.op(NodeId(*index)).unwrap(),
                    Op::Unary {
                        op: UnaryOp::Exp2,
                        ..
                    }
                )
            })
            .count()
            >= 2
    );
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Neg, input: source }
            if *source == input)
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!(
        (values.scalar_at(2).as_f64() - ((1.0f64.exp() - (-1.0f64).exp()) / 2.0)).abs() < 1e-12
    );
    assert!(
        (values.scalar_at(3).as_f64() - (((-1.0f64).exp() - 1.0f64.exp()) / 2.0)).abs() < 1e-12
    );
    assert!(
        values.scalar_at(4).as_f64().is_infinite()
            && values.scalar_at(4).as_f64().is_sign_positive()
    );
    assert!(
        values.scalar_at(5).as_f64().is_infinite()
            && values.scalar_at(5).as_f64().is_sign_negative()
    );
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Sinh,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.sinh(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.sinh(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(result).unwrap(), DType::F16);
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.sinh(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(result).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.sinh(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.sinh(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn cosh_uses_tinygrad_exp_sum_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.cosh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Cosh,
                ..
            }
        )
    }));
    assert!(
        (0..graph.node_count())
            .filter(|index| {
                matches!(
                    graph.op(NodeId(*index)).unwrap(),
                    Op::Unary {
                        op: UnaryOp::Exp2,
                        ..
                    }
                )
            })
            .count()
            >= 2
    );
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Neg, input: source }
            if *source == input)
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 1.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 1.0f64.to_bits());
    assert!(
        (values.scalar_at(2).as_f64() - ((1.0f64.exp() + (-1.0f64).exp()) / 2.0)).abs() < 1e-12
    );
    assert!(
        (values.scalar_at(3).as_f64() - (((-1.0f64).exp() + 1.0f64.exp()) / 2.0)).abs() < 1e-12
    );
    assert!(
        values.scalar_at(4).as_f64().is_infinite()
            && values.scalar_at(4).as_f64().is_sign_positive()
    );
    assert!(
        values.scalar_at(5).as_f64().is_infinite()
            && values.scalar_at(5).as_f64().is_sign_positive()
    );
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Cosh,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.cosh(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.cosh(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(result).unwrap(), DType::F16);
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.cosh(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(result).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.cosh(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.cosh(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn asinh_uses_tinygrad_square_sqrt_log_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.asinh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Asinh,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Sqrt,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Log2,
                ..
            }
        )
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(1.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(2).as_f64() - (1.0f64 + 2.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!((values.scalar_at(3).as_f64() - (-1.0f64 + 2.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!(
        values.scalar_at(4).as_f64().is_infinite()
            && values.scalar_at(4).as_f64().is_sign_positive()
    );
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Asinh,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.asinh(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.asinh(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(result).unwrap(), DType::F16);
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.asinh(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(result).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.asinh(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.asinh(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn acosh_uses_tinygrad_square_sub_sqrt_log_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.acosh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Acosh,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Sqrt,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Log2,
                ..
            }
        )
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(1.0),
                        Scalar::F(2.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - (2.0f64 + 3.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(
        values.scalar_at(4).as_f64().is_infinite()
            && values.scalar_at(4).as_f64().is_sign_positive()
    );
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Acosh,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.acosh(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.acosh(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(result).unwrap(), DType::F16);
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.acosh(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(result).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.acosh(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.acosh(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn atanh_uses_tinygrad_ratio_log_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [9], DType::F64);
    let output = graph.atanh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Atanh,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Log2,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Reciprocal,
                ..
            }
        )
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [9],
                    DType::F64,
                    [
                        Scalar::F(-1.0),
                        Scalar::F(1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(0.5),
                        Scalar::F(2.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert!(
        values.scalar_at(0).as_f64().is_infinite()
            && values.scalar_at(0).as_f64().is_sign_negative()
    );
    assert!(
        values.scalar_at(1).as_f64().is_infinite()
            && values.scalar_at(1).as_f64().is_sign_positive()
    );
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(4).as_f64() - 0.5f64 * 3.0f64.ln()).abs() < 1e-12);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    assert!(values.scalar_at(7).as_f64().is_nan());
    assert!(values.scalar_at(8).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Atanh,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.atanh(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.atanh(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    assert_eq!(scalar.dtype(result).unwrap(), DType::F16);
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.atanh(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    assert_eq!(empty.dtype(result).unwrap(), DType::BF16);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.atanh(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.atanh(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn erf_uses_tinygrad_aands_polynomial_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.erf(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Erf,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Exp2,
                ..
            }
        )
    }));
    assert!(
        (0..graph.node_count())
            .filter(|index| {
                matches!(
                    graph.op(NodeId(*index)).unwrap(),
                    Op::Unary {
                        op: UnaryOp::Sign,
                        ..
                    }
                )
            })
            .count()
            >= 2
    );
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert!((values.scalar_at(0).as_f64() + 0.84270079).abs() < 1e-6);
    assert!(values.scalar_at(1).as_f64().abs() < 1e-12);
    assert!(values.scalar_at(2).as_f64().abs() < 1e-12);
    assert!((values.scalar_at(3).as_f64() - 0.84270079).abs() < 1e-6);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert_eq!(values.scalar_at(5).as_f64(), -1.0);
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Erf,
                ..
            }
        )
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.erf(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), output_dtype);
    }
    let mut scalar = Graph::new();
    let source = scalar.input_dtype("input", [], DType::F16);
    let result = scalar.erf(source).unwrap();
    assert_eq!(scalar.shape(result).unwrap(), &Shape::new([]));
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::BF16);
    let result = empty.erf(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(
        graph.erf(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.erf(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn floor_uses_tinygrad_trunc_compare_select_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.floor(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Floor,
                ..
            }
        )
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Trunc, input: source }
            if *source == input)
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [8],
                    DType::F64,
                    [
                        Scalar::F(-1.5),
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.5),
                        Scalar::F(1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-2.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(
        values.scalar_at(5).as_f64().is_infinite()
            && values.scalar_at(5).as_f64().is_sign_positive()
    );
    assert!(
        values.scalar_at(6).as_f64().is_infinite()
            && values.scalar_at(6).as_f64().is_sign_negative()
    );
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
        ("f32", DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.floor(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.floor(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(
        graph.floor(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.floor(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn ceil_uses_tinygrad_trunc_compare_select_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.ceil(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    assert!((0..graph.node_count()).all(|index| {
        !matches!(
            graph.op(NodeId(index)).unwrap(),
            Op::Unary {
                op: UnaryOp::Ceil,
                ..
            }
        )
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [8],
                    DType::F64,
                    [
                        Scalar::F(-1.5),
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.5),
                        Scalar::F(1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-1.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 1.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(
        values.scalar_at(5).as_f64().is_infinite()
            && values.scalar_at(5).as_f64().is_sign_positive()
    );
    assert!(
        values.scalar_at(6).as_f64().is_infinite()
            && values.scalar_at(6).as_f64().is_sign_negative()
    );
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
        ("f32", DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.ceil(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.ceil(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(
        graph.ceil(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.ceil(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn trunc_preserves_tinygrad_direct_alu_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.trunc(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input: source } if *source == input)
    );
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [8],
                    DType::F64,
                    [
                        Scalar::F(-1.5),
                        Scalar::F(-1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.5),
                        Scalar::F(1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-1.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(
        values.scalar_at(5).as_f64().is_infinite()
            && values.scalar_at(5).as_f64().is_sign_positive()
    );
    assert!(
        values.scalar_at(6).as_f64().is_infinite()
            && values.scalar_at(6).as_f64().is_sign_negative()
    );
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
        ("f32", DType::F32),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.trunc(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.trunc(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(
        graph.trunc(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.trunc(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn round_uses_tinygrad_ties_even_composition_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.round(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Round,
            ..
        }
    )));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [8],
                    DType::F64,
                    [
                        Scalar::F(-2.5),
                        Scalar::F(-1.5),
                        Scalar::F(-0.5),
                        Scalar::F(0.5),
                        Scalar::F(1.5),
                        Scalar::F(2.5),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.to_vec_f64()[0..6], [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]);
    assert!(values.scalar_at(6).as_f64().is_infinite());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.round(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.round(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn logical_not_uses_tinygrad_bool_cast_ne_true_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F64);
    let output = graph.logical_not(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Compare {
            op: CompareOp::Ne,
            ..
        }
    ));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [4],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(2.0),
                        Scalar::F(f64::NAN),
                        Scalar::F(f64::INFINITY),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());
    assert!(!values.scalar_at(1).as_bool());
    assert!(!values.scalar_at(2).as_bool());
    assert!(!values.scalar_at(3).as_bool());
    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
    ] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.logical_not(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), DType::Bool);
    }
    let node_count = graph.node_count();
    assert!(matches!(
        graph.logical_not(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn bitwise_not_uses_tinygrad_logical_not_or_storage_typed_xor_and_preflights() {
    // Bool takes the public logical_not spelling; each integer width retains
    // its storage dtype and uses exactly the source scalar mask. BitXor has
    // RustGrad's existing zero-VJP/nondifferentiable treatment, so this adds
    // no new differentiable raw operation.
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let output = graph.bitwise_not(input).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        if dtype == DType::Bool {
            assert!(matches!(
                graph.op(output).unwrap(),
                Op::Compare {
                    op: CompareOp::Ne,
                    ..
                }
            ));
        } else {
            let Op::Binary {
                op: BinaryOp::BitXor,
                lhs,
                rhs,
            } = graph.op(output).unwrap()
            else {
                panic!("integer bitwise_not must lower to typed XOR");
            };
            assert_eq!(*lhs, input);
            let Op::Constant(mask) = graph.op(*rhs).unwrap() else {
                panic!("integer bitwise_not mask must be a scalar constant");
            };
            assert_eq!(mask.dtype(), dtype);
            assert_eq!(mask.shape(), &Shape::new([]));
            match dtype {
                DType::U8 => assert_eq!(mask.scalar_at(0).as_u64(), u64::from(u8::MAX)),
                DType::U16 => assert_eq!(mask.scalar_at(0).as_u64(), u64::from(u16::MAX)),
                DType::U32 => assert_eq!(mask.scalar_at(0).as_u64(), u64::from(u32::MAX)),
                DType::U64 => assert_eq!(mask.scalar_at(0).as_u64(), u64::MAX),
                _ => assert_eq!(mask.scalar_at(0).as_i64(), -1),
            }
        }
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::I32);
    let output = scalar.bitwise_not(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::U16);
    let output = empty.bitwise_not(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut invalid = Graph::new();
        let input = invalid.input_dtype("input", [1], dtype);
        let node_count = invalid.node_count();
        assert!(matches!(
            invalid.bitwise_not(input),
            Err(Error::InvalidElementwiseDType { op: "bitwise_not", actual }) if actual == dtype
        ));
        assert_eq!(invalid.node_count(), node_count);
    }

    let mut unknown = Graph::new();
    let node_count = unknown.node_count();
    assert!(matches!(
        unknown.bitwise_not(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(unknown.node_count(), node_count);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX / 8 + 1], DType::I64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.bitwise_not(input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn bitwise_binary_public_and_scalar_forms_use_tinygrad_lub_before_publication() {
    // Every local Bool/integer storage family remains admitted through the
    // public names and retains a raw bitwise Binary root. These operations
    // use the existing zero-VJP/nondifferentiable Binary treatment.
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
    ] {
        for (op, lower) in [
            (
                BinaryOp::BitAnd,
                Graph::bitwise_and as fn(&mut Graph, NodeId, NodeId) -> crate::Result<NodeId>,
            ),
            (
                BinaryOp::BitOr,
                Graph::bitwise_or as fn(&mut Graph, NodeId, NodeId) -> crate::Result<NodeId>,
            ),
            (
                BinaryOp::BitXor,
                Graph::bitwise_xor as fn(&mut Graph, NodeId, NodeId) -> crate::Result<NodeId>,
            ),
        ] {
            let mut graph = Graph::new();
            let lhs = graph.input_dtype("lhs", [2, 1], dtype);
            let rhs = graph.input_dtype("rhs", [3], dtype);
            let output = lower(&mut graph, lhs, rhs).unwrap();
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert!(
                matches!(graph.op(output).unwrap(), Op::Binary { op: actual, lhs: actual_lhs, rhs: actual_rhs } if *actual == op && *actual_lhs == lhs && *actual_rhs == rhs)
            );
        }
    }

    // Mixed signed/unsigned operands are explicitly cast to the source LUB
    // before the root operation, including Bool's promotion into I32.
    let mut mixed = Graph::new();
    let lhs = mixed.input_dtype("lhs", [2, 1], DType::I8);
    let rhs = mixed.input_dtype("rhs", [3], DType::U8);
    let output = mixed.bitwise_xor(lhs, rhs).unwrap();
    assert_eq!(mixed.dtype(output).unwrap(), DType::I16);
    let Op::Binary {
        op: BinaryOp::BitXor,
        lhs: cast_lhs,
        rhs: cast_rhs,
    } = mixed.op(output).unwrap()
    else {
        panic!("mixed bitwise_xor must retain its Binary root");
    };
    assert!(
        matches!(mixed.op(*cast_lhs).unwrap(), Op::Cast { input, dtype: DType::I16 } if *input == lhs)
    );
    assert!(
        matches!(mixed.op(*cast_rhs).unwrap(), Op::Cast { input, dtype: DType::I16 } if *input == rhs)
    );

    let bool_input = mixed.input_dtype("bool", [2], DType::Bool);
    let bool_scalar = mixed.bitwise_or_scalar(bool_input, Scalar::I(2)).unwrap();
    assert_eq!(mixed.dtype(bool_scalar).unwrap(), DType::I32);
    let Op::Binary {
        op: BinaryOp::BitOr,
        lhs: bool_cast,
        rhs: bool_constant,
    } = mixed.op(bool_scalar).unwrap()
    else {
        panic!("Bool/int scalar form must retain its Binary root");
    };
    assert!(
        matches!(mixed.op(*bool_cast).unwrap(), Op::Cast { input, dtype: DType::I32 } if *input == bool_input)
    );
    assert!(
        matches!(mixed.op(*bool_constant).unwrap(), Op::Constant(data) if data.dtype() == DType::I32 && data.scalar_at(0).as_i64() == 2)
    );

    let scalar_input = mixed.input_dtype("scalar", [], DType::U8);
    let scalar_output = mixed
        .bitwise_xor_scalar(scalar_input, Scalar::I(-1))
        .unwrap();
    assert_eq!(mixed.shape(scalar_output).unwrap(), &Shape::new([]));
    assert!(
        matches!(mixed.op(scalar_output).unwrap(), Op::Binary { op: BinaryOp::BitXor, lhs, rhs }
        if *lhs == scalar_input && matches!(mixed.op(*rhs).unwrap(), Op::Constant(data) if data.dtype() == DType::U8 && data.scalar_at(0).as_u64() == u64::from(u8::MAX)))
    );
    let reflected = mixed
        .scalar_bitwise_and(Scalar::U(3), scalar_input)
        .unwrap();
    assert!(
        matches!(mixed.op(reflected).unwrap(), Op::Binary { op: BinaryOp::BitAnd, lhs, rhs }
        if matches!(mixed.op(*lhs).unwrap(), Op::Constant(data) if data.dtype() == DType::U8 && data.scalar_at(0).as_u64() == 3) && *rhs == scalar_input)
    );

    let empty_input = mixed.input_dtype("empty", [0, 2], DType::U16);
    let empty_output = mixed
        .bitwise_and_scalar(empty_input, Scalar::U(u16::MAX.into()))
        .unwrap();
    assert_eq!(mixed.shape(empty_output).unwrap(), &Shape::new([0, 2]));

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut invalid = Graph::new();
        let input = invalid.input_dtype("input", [1], dtype);
        let rhs = invalid.input_dtype("rhs", [1], DType::I32);
        let node_count = invalid.node_count();
        assert!(matches!(
            invalid.bitwise_and(input, rhs),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(invalid.node_count(), node_count);
    }

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1], DType::I64);
    let rhs = wide.input_dtype("rhs", [1], DType::U64);
    let node_count = wide.node_count();
    assert!(matches!(
        wide.bitwise_or(lhs, rhs),
        Err(Error::InvalidElementwiseDType {
            actual: DType::F32,
            ..
        })
    ));
    assert_eq!(wide.node_count(), node_count);

    let mut malformed = Graph::new();
    let input = malformed.input_dtype("input", [1], DType::I32);
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.bitwise_xor_scalar(input, Scalar::F(1.0)),
        Err(Error::InvalidElementwiseDType { .. })
    ));
    assert_eq!(malformed.node_count(), node_count);
    assert!(matches!(
        malformed.bitwise_and(NodeId(usize::MAX), input),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), node_count);

    let mut overflow = Graph::new();
    let lhs = overflow.input_dtype("lhs", [usize::MAX / 8 + 1], DType::I64);
    let rhs = overflow.input_dtype("rhs", [], DType::I64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.bitwise_and(lhs, rhs),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);

    let mut scalar_overflow = Graph::new();
    let input = scalar_overflow.input_dtype("input", [usize::MAX / 4 + 1], DType::Bool);
    let node_count = scalar_overflow.node_count();
    assert!(matches!(
        scalar_overflow.bitwise_or_scalar(input, Scalar::I(1)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(scalar_overflow.node_count(), node_count);
}

#[test]
fn shift_public_and_scalar_forms_use_tinygrad_lub_before_publication() {
    // Live operands are both committed to `_broadcasted`'s source LUB before
    // the shift. Every concrete integer family remains storage-typed.
    for dtype in DType::INTS {
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let mut graph = Graph::new();
            let value = graph.input_dtype("value", [2, 1], dtype);
            let count = graph.input_dtype("count", [3], dtype);
            let output = if op == BinaryOp::Shl {
                graph.lshift(value, count).unwrap()
            } else {
                graph.rshift(value, count).unwrap()
            };
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
            assert_eq!(graph.dtype(output).unwrap(), dtype);
            assert!(matches!(
                graph.op(output).unwrap(),
                Op::Binary { op: actual, lhs, rhs }
                    if *actual == op && *lhs == value && *rhs == count
            ));
        }
    }

    let mut mixed = Graph::new();
    let value = mixed.input_dtype("value", [2, 1], DType::I8);
    let count = mixed.input_dtype("count", [3], DType::U8);
    let output = mixed.lshift(value, count).unwrap();
    assert_eq!(mixed.dtype(output).unwrap(), DType::I16);
    let Op::Binary {
        op: BinaryOp::Shl,
        lhs,
        rhs,
    } = mixed.op(output).unwrap()
    else {
        panic!("mixed lshift must retain its Binary root");
    };
    assert!(matches!(
        mixed.op(*lhs).unwrap(),
        Op::Cast { input, dtype: DType::I16 } if *input == value
    ));
    assert!(matches!(
        mixed.op(*rhs).unwrap(),
        Op::Cast { input, dtype: DType::I16 } if *input == count
    ));

    // A Python integer is weak: it commits at the live integer width, while
    // Bool plus weakint lifts to the configured default I32 width. Reflected
    // forms reverse only the final operands.
    let u8_value = mixed.input_dtype("u8", [], DType::U8);
    let shifted = mixed.lshift_scalar(u8_value, Scalar::I(2)).unwrap();
    assert!(matches!(
        mixed.op(shifted).unwrap(),
        Op::Binary { op: BinaryOp::Shl, lhs, rhs }
            if *lhs == u8_value
                && matches!(mixed.op(*rhs).unwrap(), Op::Constant(data)
                    if data.dtype() == DType::U8 && data.scalar_at(0).as_u64() == 2)
    ));
    let reflected = mixed.scalar_rshift(Scalar::U(128), u8_value).unwrap();
    assert!(matches!(
        mixed.op(reflected).unwrap(),
        Op::Binary { op: BinaryOp::Shr, lhs, rhs }
            if *rhs == u8_value
                && matches!(mixed.op(*lhs).unwrap(), Op::Constant(data)
                    if data.dtype() == DType::U8 && data.scalar_at(0).as_u64() == 128)
    ));

    let bool_count = mixed.input_dtype("bool", [2], DType::Bool);
    let bool_lift = mixed.rshift_scalar(bool_count, Scalar::I(1)).unwrap();
    assert_eq!(mixed.dtype(bool_lift).unwrap(), DType::I32);
    let Op::Binary {
        op: BinaryOp::Shr,
        lhs,
        rhs,
    } = mixed.op(bool_lift).unwrap()
    else {
        panic!("Bool/weakint rshift must retain its Binary root");
    };
    assert!(matches!(
        mixed.op(*lhs).unwrap(),
        Op::Cast { input, dtype: DType::I32 } if *input == bool_count
    ));
    assert!(matches!(
        mixed.op(*rhs).unwrap(),
        Op::Constant(data) if data.dtype() == DType::I32 && data.scalar_at(0).as_i64() == 1
    ));

    let empty = mixed.input_dtype("empty", [0, 2], DType::U16);
    let empty_output = mixed.rshift_scalar(empty, Scalar::I(3)).unwrap();
    assert_eq!(mixed.shape(empty_output).unwrap(), &Shape::new([0, 2]));

    // Bool/Bool and every floating family are rejected before casts or roots.
    let mut invalid = Graph::new();
    let bool_lhs = invalid.input_dtype("bool_lhs", [1], DType::Bool);
    let bool_rhs = invalid.input_dtype("bool_rhs", [1], DType::Bool);
    let node_count = invalid.node_count();
    assert!(matches!(
        invalid.lshift(bool_lhs, bool_rhs),
        Err(Error::InvalidElementwiseDType {
            actual: DType::Bool,
            ..
        })
    ));
    assert_eq!(invalid.node_count(), node_count);
    assert!(matches!(
        invalid.rshift_scalar(bool_lhs, Scalar::Bool(true)),
        Err(Error::InvalidElementwiseDType {
            actual: DType::Bool,
            ..
        })
    ));
    assert_eq!(invalid.node_count(), node_count);

    for dtype in DType::FLOATS {
        let mut graph = Graph::new();
        let lhs = graph.input_dtype("lhs", [1], dtype);
        let rhs = graph.input_dtype("rhs", [1], DType::I32);
        let node_count = graph.node_count();
        assert!(matches!(
            graph.rshift(lhs, rhs),
            Err(Error::InvalidElementwiseDType { .. })
        ));
        assert_eq!(graph.node_count(), node_count);
    }

    let mut wide = Graph::new();
    let lhs = wide.input_dtype("lhs", [1], DType::I64);
    let rhs = wide.input_dtype("rhs", [1], DType::U64);
    let node_count = wide.node_count();
    assert!(matches!(
        wide.lshift(lhs, rhs),
        Err(Error::InvalidElementwiseDType {
            actual: DType::F32,
            ..
        })
    ));
    assert_eq!(wide.node_count(), node_count);

    // Known scalar counts are validated before their Constant is published.
    let mut counts = Graph::new();
    let value = counts.input_dtype("value", [1], DType::U8);
    let node_count = counts.node_count();
    assert!(matches!(
        counts.lshift_scalar(value, Scalar::I(-1)),
        Err(Error::InvalidShiftCount { bits: 8, .. })
    ));
    assert_eq!(counts.node_count(), node_count);
    assert!(matches!(
        counts.rshift_scalar(value, Scalar::I(8)),
        Err(Error::InvalidShiftCount { count: 8, bits: 8 })
    ));
    assert_eq!(counts.node_count(), node_count);
    assert!(matches!(
        counts.lshift_scalar(value, Scalar::F(1.0)),
        Err(Error::InvalidElementwiseDType { .. })
    ));
    assert_eq!(counts.node_count(), node_count);
    assert!(matches!(
        counts.rshift(NodeId(usize::MAX), value),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(counts.node_count(), node_count);

    let mut overflow = Graph::new();
    let lhs = overflow.input_dtype("lhs", [usize::MAX / 8 + 1], DType::I64);
    let rhs = overflow.input_dtype("rhs", [], DType::I64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.lshift(lhs, rhs),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn isnan_uses_tinygrad_self_inequality_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F64);
    let output = graph.isnan(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Compare { op: CompareOp::Ne, lhs, rhs } if *lhs == input && *rhs == input)
    );
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::IsNan,
            ..
        }
    )));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [4],
                    DType::F64,
                    [
                        Scalar::F(-0.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(
        (0..4)
            .map(|index| values.scalar_at(index).as_bool())
            .collect::<Vec<_>>(),
        vec![false, false, true, true]
    );
    let mut dtypes = Graph::new();
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i64", DType::I64),
        ("u64", DType::U64),
        ("f16", DType::F16),
        ("bf16", DType::BF16),
    ] {
        let source = dtypes.input_dtype(name, [0], dtype);
        let result = dtypes.isnan(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), DType::Bool);
        assert_eq!(dtypes.shape(result).unwrap(), &Shape::new([0]));
    }
    let node_count = graph.node_count();
    assert!(matches!(
        graph.isnan(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn isinf_preserves_tinygrad_default_both_signs_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.isinf(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::IsInf, input: source } if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    let node_count = graph.node_count();
    assert!(matches!(
        graph.isinf(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.isinf(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn isfinite_uses_tinygrad_isinf_isnan_logical_not_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.isfinite(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::IsFinite,
            ..
        }
    )));
    let node_count = graph.node_count();
    assert!(matches!(
        graph.isfinite(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(
        overflow.isfinite(source),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn log10_commits_weak_scale_at_log2_storage_width_and_preflights() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.log10(input).unwrap();
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs,
        rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("tinygrad log10 must scale Log2 by log10(2)");
    };
    assert!(
        matches!(graph.op(*lhs).unwrap(), Op::Unary { op: UnaryOp::Log2, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(*rhs).unwrap(), DType::F64);
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(8.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert_eq!(values.scalar_at(1).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(2).as_f64(), f64::NEG_INFINITY);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(
        values.scalar_at(6).as_f64(),
        std::f64::consts::LOG10_2 * 3.0
    );

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);

    let mut dtypes = Graph::new();
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        let input = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.log10(input).unwrap();
        let Op::Binary { rhs, .. } = dtypes.op(output).unwrap() else {
            panic!("tinygrad log10 must end in a scale multiply");
        };
        assert_eq!(dtypes.dtype(*rhs).unwrap(), dtype);
        assert_eq!(dtypes.dtype(output).unwrap(), dtype);
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
    ] {
        let input = dtypes.input_dtype(format!("{dtype:?}"), [1], dtype);
        let output = dtypes.log10(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), DType::F32);
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.log10(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap()
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty()
    );

    let before = graph.node_count();
    assert!(matches!(
        graph.log10(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), before);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.log10(input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn logsigmoid_uses_tinygrad_neg_softplus_neg_with_typed_default_beta() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F64);
    let output = graph.logsigmoid(input).unwrap();
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Unary {
            op: UnaryOp::Neg,
            ..
        }
    ));
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == DType::F64 && data.scalar_at(0).as_f64() == 1.0)
    }));
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [4],
                    DType::F64,
                    [
                        Scalar::F(-1000.0),
                        Scalar::F(1000.0),
                        Scalar::F(f64::NEG_INFINITY),
                        Scalar::F(f64::NAN),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), -1000.0);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([4]));
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);

    let mut dtypes = Graph::new();
    for (name, dtype) in [("f16", DType::F16), ("bf16", DType::BF16)] {
        let input = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.logsigmoid(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), dtype);
        assert!(dtypes.nodes.iter().any(|node| {
            matches!(&node.op, Op::Constant(data)
                if data.shape() == &Shape::new([]) && data.dtype() == dtype && data.scalar_at(0).as_f64() == 1.0)
        }));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
    ] {
        let input = dtypes.input_dtype(format!("{dtype:?}"), [1], dtype);
        let output = dtypes.logsigmoid(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), DType::F32);
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.logsigmoid(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap()
                )]),
            )
            .unwrap()
            .to_vec_f64()
            .is_empty()
    );
}

#[test]
fn logsigmoid_preflights_unknown_and_overflow_inputs_before_constants_or_nodes() {
    let mut graph = Graph::new();
    let before = graph.node_count();
    assert!(matches!(
        graph.logsigmoid(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), before);

    let input = graph.input_dtype("input", [usize::MAX, 2], DType::F64);
    let before = graph.node_count();
    assert!(matches!(
        graph.logsigmoid(input),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), before);
}

#[test]
fn log_uses_tinygrad_log2_scale_promotion_special_values_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.log(input).unwrap();
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs,
        rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("tinygrad log must scale Log2 by ln(2)");
    };
    assert!(
        matches!(graph.op(*lhs).unwrap(), Op::Unary { op: UnaryOp::Log2, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(*rhs).unwrap(), DType::F64);
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(4.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert_eq!(values.scalar_at(1).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(2).as_f64(), f64::NEG_INFINITY);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(values.scalar_at(6).as_f64(), std::f64::consts::LN_2 * 2.0);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let gradient_values = CpuBackend
        .execute(
            &graph,
            gradient,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([7], DType::F64, [Scalar::F(2.0); 7]).unwrap(),
            )]),
        )
        .unwrap()
        .to_vec_f64();
    assert!(
        gradient_values
            .iter()
            .all(|value| (*value - 0.5).abs() < 1e-12)
    );

    let mut dtypes = Graph::new();
    let f16 = dtypes.input_dtype("f16", [1], DType::F16);
    let bf16 = dtypes.input_dtype("bf16", [1], DType::BF16);
    let f16_output = dtypes.log(f16).unwrap();
    let bf16_output = dtypes.log(bf16).unwrap();
    assert_eq!(dtypes.dtype(f16_output).unwrap(), DType::F16);
    assert_eq!(dtypes.dtype(bf16_output).unwrap(), DType::BF16);
    let Op::Binary { lhs, .. } = dtypes.op(f16_output).unwrap() else {
        panic!("tinygrad F16 log must end in a scale multiply");
    };
    assert!(matches!(
        dtypes.op(*lhs).unwrap(),
        Op::Unary {
            op: UnaryOp::Log2,
            ..
        }
    ));
    for (name, dtype) in [
        ("bool", DType::Bool),
        ("i8", DType::I8),
        ("u8", DType::U8),
        ("i16", DType::I16),
        ("u16", DType::U16),
        ("i32", DType::I32),
        ("u32", DType::U32),
        ("i64", DType::I64),
        ("u64", DType::U64),
    ] {
        let input = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.log(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), DType::F32);
    }

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("input", [1], DType::F16);
    let output = narrow.log2(input).unwrap();
    let loss = narrow.sum_all(output).unwrap();
    let gradient = narrow.grad(loss, input).unwrap();
    assert_eq!(narrow.dtype(gradient).unwrap(), DType::F16);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.log(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::BF16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.log(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.log(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn log2_preserves_tinygrad_storage_width_special_values_and_typed_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.log2(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Log2, input: source }
        if *source == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let values = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars(
                    [7],
                    DType::F64,
                    [
                        Scalar::F(1.0),
                        Scalar::F(-0.0),
                        Scalar::F(0.0),
                        Scalar::F(-1.0),
                        Scalar::F(f64::INFINITY),
                        Scalar::F(f64::NAN),
                        Scalar::F(8.0),
                    ],
                )
                .unwrap(),
            )]),
        )
        .unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), 0.0);
    assert_eq!(values.scalar_at(1).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(2).as_f64(), f64::NEG_INFINITY);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert_eq!(values.scalar_at(4).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert_eq!(values.scalar_at(6).as_f64(), 3.0);

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let gradient_values = CpuBackend
        .execute(
            &graph,
            gradient,
            &HashMap::from([(
                "input".into(),
                TensorData::from_scalars([7], DType::F64, [Scalar::F(2.0); 7]).unwrap(),
            )]),
        )
        .unwrap()
        .to_vec_f64();
    let expected = 1.0 / (2.0 * std::f64::consts::LN_2);
    assert!(
        gradient_values
            .iter()
            .all(|value| (*value - expected).abs() < 1e-12)
    );

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16),
        ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32),
        ("bool", DType::Bool, DType::F32),
        ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32),
        ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32),
        ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32),
        ("i64", DType::I64, DType::F32),
        ("u64", DType::U64, DType::F32),
    ] {
        let input = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.log2(input).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
    }

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("input", [1], DType::F16);
    let output = narrow.log2(input).unwrap();
    let loss = narrow.sum_all(output).unwrap();
    let gradient = narrow.grad(loss, input).unwrap();
    assert_eq!(narrow.dtype(gradient).unwrap(), DType::F16);
    let input = narrow.input_dtype("bf16", [1], DType::BF16);
    let output = narrow.log2(input).unwrap();
    let loss = narrow.sum_all(output).unwrap();
    let gradient = narrow.grad(loss, input).unwrap();
    assert_eq!(narrow.dtype(gradient).unwrap(), DType::BF16);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.log2(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.log2(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.log2(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn abs_uses_tinygrad_sign_multiply_structure_special_values_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.abs(input).unwrap();
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs,
        rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("tinygrad abs must end in Mul");
    };
    assert_eq!(*lhs, input);
    assert!(
        matches!(graph.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Sign, input: signed }
        if *signed == input)
    );
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::from_scalars(
            [5],
            DType::F64,
            [
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NAN),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64(), f64::INFINITY);
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), f64::INFINITY);
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0, 0.0, 0.0, 1.0, 1.0]
    );

    let mut discrete = Graph::new();
    let signed = discrete.input_dtype("signed", [1], DType::I8);
    let unsigned = discrete.input_dtype("unsigned", [2], DType::U64);
    let boolean = discrete.input_dtype("boolean", [2], DType::Bool);
    let signed_output = discrete.abs(signed).unwrap();
    let unsigned_output = discrete.abs(unsigned).unwrap();
    let boolean_output = discrete.abs(boolean).unwrap();
    let bindings = HashMap::from([
        (
            "signed".into(),
            TensorData::from_scalars([1], DType::I8, [Scalar::I(i8::MIN as i64)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([2], DType::U64, [Scalar::U(0), Scalar::U(u64::MAX)]).unwrap(),
        ),
        ("boolean".into(), bool_data([2], [false, true])),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&discrete, signed_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::I8(vec![i8::MIN])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, unsigned_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::U64(vec![0, u64::MAX])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, boolean_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![false, true])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F16);
    let output = empty.abs(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.abs(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn neg_uses_tinygrad_bool_logical_not_and_preflighted_numeric_unary() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [6], DType::F64);
    let output = graph.neg(input).unwrap();
    assert!(
        matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Neg, input: negated }
        if *negated == input)
    );
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let nan_bits = f64::NAN.to_bits();
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::from_scalars(
            [6],
            DType::F64,
            [
                Scalar::F(-0.0),
                Scalar::F(0.0),
                Scalar::F(f64::INFINITY),
                Scalar::F(f64::NEG_INFINITY),
                Scalar::F(f64::from_bits(nan_bits)),
                Scalar::F(3.0),
            ],
        )
        .unwrap(),
    )]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(2).as_f64(), f64::NEG_INFINITY);
    assert_eq!(values.scalar_at(3).as_f64(), f64::INFINITY);
    assert_eq!(
        values.scalar_at(4).as_f64().to_bits(),
        nan_bits ^ (1_u64 << 63)
    );
    assert_eq!(values.scalar_at(5).as_f64(), -3.0);
    assert_eq!(
        CpuBackend
            .execute(&graph, gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![-1.0; 6]
    );

    let mut discrete = Graph::new();
    let boolean = discrete.input_dtype("boolean", [2], DType::Bool);
    let signed = discrete.input_dtype("signed", [1], DType::I8);
    let unsigned = discrete.input_dtype("unsigned", [1], DType::U64);
    let boolean_output = discrete.neg(boolean).unwrap();
    let signed_output = discrete.neg(signed).unwrap();
    let unsigned_output = discrete.neg(unsigned).unwrap();
    assert!(matches!(
        discrete.op(boolean_output).unwrap(),
        Op::Compare {
            op: CompareOp::Ne,
            lhs,
            ..
        } if matches!(discrete.op(*lhs).unwrap(), Op::Cast {
            input,
            dtype: DType::Bool,
        } if *input == boolean)
    ));
    assert!(matches!(
        discrete.op(signed_output).unwrap(),
        Op::Unary {
            op: UnaryOp::Neg,
            ..
        }
    ));
    let bindings = HashMap::from([
        ("boolean".into(), bool_data([2], [false, true])),
        (
            "signed".into(),
            TensorData::from_scalars([1], DType::I8, [Scalar::I(i8::MIN as i64)]).unwrap(),
        ),
        (
            "unsigned".into(),
            TensorData::from_scalars([1], DType::U64, [Scalar::U(1)]).unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend
            .execute(&discrete, boolean_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![true, false])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, signed_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::I8(vec![i8::MIN])
    );
    assert_eq!(
        CpuBackend
            .execute(&discrete, unsigned_output, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::U64(vec![u64::MAX])
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F16);
    let output = empty.neg(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([(
                    "input".into(),
                    TensorData::from_scalars([0], DType::F16, []).unwrap(),
                )]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.neg(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn lerp_preflights_all_operands_and_preserves_broadcast_vjps() {
    let mut graph = Graph::new();
    let start = graph.input("start", [2, 1]);
    let end = graph.input("end", [3]);
    let weight = graph.input("weight", [2, 3]);
    let output = graph.lerp(start, end, weight).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let start_gradient = graph.grad(loss, start).unwrap();
    let end_gradient = graph.grad(loss, end).unwrap();
    let weight_gradient = graph.grad(loss, weight).unwrap();
    let bindings = HashMap::from([
        (
            "start".into(),
            TensorData::new([2, 1], vec![1.0, 4.0]).unwrap(),
        ),
        (
            "end".into(),
            TensorData::new([3], vec![3.0, 5.0, 7.0]).unwrap(),
        ),
        (
            "weight".into(),
            TensorData::new([2, 3], vec![0.0, 0.5, 1.0, 0.25, 0.5, 0.75]).unwrap(),
        ),
    ]);
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.0, 3.0, 7.0, 3.75, 4.5, 6.25]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, start_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![1.5, 1.5]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, end_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![0.25, 1.0, 1.75]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, weight_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![2.0, 4.0, 6.0, -1.0, 1.0, 3.0]
    );

    let mut scalar = Graph::new();
    let start = scalar.input("start", []);
    let end = scalar.input("end", []);
    let weight = scalar.input("weight", []);
    let output = scalar.lerp(start, end, weight).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert!(
        CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([
                    ("start".into(), TensorData::scalar(1.0)),
                    ("end".into(), TensorData::scalar(f32::INFINITY)),
                    ("weight".into(), TensorData::scalar(0.0)),
                ]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64()
            .is_nan()
    );

    let mut empty = Graph::new();
    let start = empty.input("start", [0]);
    let end = empty.input("end", [0]);
    let weight = empty.input("weight", []);
    let output = empty.lerp(start, end, weight).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([
                    ("start".into(), TensorData::new([0], vec![]).unwrap()),
                    ("end".into(), TensorData::new([0], vec![]).unwrap()),
                    ("weight".into(), TensorData::scalar(0.5)),
                ]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let mut malformed = Graph::new();
    let start = malformed.input("start", [2, 1]);
    let end = malformed.input("end", [3]);
    let incompatible_weight = malformed.input("weight", [2, 2]);
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.lerp(start, end, incompatible_weight),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(malformed.node_count(), node_count);
}

#[test]
fn lerp_u8_live_weight_uses_tinygrad_fixed_point_path() {
    // The checked-in source has a non-generic U8/tensor-weight path. It must
    // retain the I8 delta, I16 quantized weight, U16 rounding/shift, and final
    // U8 cast rather than silently taking the ordinary lerp graph.
    let mut graph = Graph::new();
    let start = graph.input_dtype("start", [2, 1], DType::U8);
    let end = graph.input_dtype("end", [3], DType::U8);
    let weight = graph.input_dtype("weight", [2, 3], DType::I16);
    let output = graph.lerp(start, end, weight).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.dtype(output).unwrap(), DType::U8);

    let Op::Cast {
        input: final_add,
        dtype,
    } = graph.op(output).unwrap()
    else {
        panic!("U8 lerp must finish with its source cast")
    };
    assert_eq!(*dtype, DType::U8);
    let Op::Binary {
        op: BinaryOp::Add,
        rhs: shifted,
        ..
    } = graph.op(*final_add).unwrap()
    else {
        panic!("U8 lerp must add the shifted fixed-point delta")
    };
    let Op::Binary {
        op: BinaryOp::Shr,
        rhs: shift,
        ..
    } = graph.op(*shifted).unwrap()
    else {
        panic!("U8 lerp must use its fixed seven-bit shift")
    };
    assert!(matches!(
        graph.op(*shift).unwrap(),
        Op::Constant(data) if data.dtype() == DType::U16 && data.scalar_at(0).as_u64() == 7
    ));
    let operations: Vec<_> = graph
        .trace(output)
        .unwrap()
        .steps
        .into_iter()
        .map(|step| step.operation)
        .collect();
    assert!(operations.iter().any(|operation| operation.contains("I8")));
    assert!(operations.iter().any(|operation| operation.contains("I16")));
    assert!(operations.iter().any(|operation| operation.contains("F32")));

    let mut empty = Graph::new();
    let start = empty.input_dtype("start", [0, 1], DType::U8);
    let end = empty.input_dtype("end", [3], DType::U8);
    let weight = empty.input_dtype("weight", [0, 3], DType::F16);
    let output = empty.lerp(start, end, weight).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    assert_eq!(empty.dtype(output).unwrap(), DType::U8);
}

#[test]
fn lerp_scalar_uses_the_ordinary_source_composition_even_for_u8_start() {
    let mut u8 = Graph::new();
    let start = u8.input_dtype("start", [2, 1], DType::U8);
    let end = u8.input_dtype("end", [3], DType::U8);
    let output = u8.lerp_scalar(start, end, Scalar::F(0.5)).unwrap();
    assert_eq!(u8.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(u8.dtype(output).unwrap(), DType::F32);
    assert!(matches!(u8.op(NodeId(2)).unwrap(), Op::Constant(data)
        if data.dtype() == DType::F32 && data.scalar_at(0).as_f64() == 0.5));
    assert!(matches!(
        u8.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!((0..u8.node_count()).any(|index| matches!(
        u8.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Neg,
            ..
        }
    )));
    assert!((0..u8.node_count()).any(|index| matches!(
        u8.op(NodeId(index)).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    )));
    assert!((0..u8.node_count()).all(|index| !matches!(
        u8.op(NodeId(index)).unwrap(),
        Op::Binary {
            op: BinaryOp::Shl | BinaryOp::Shr,
            ..
        } | Op::Cast {
            dtype: DType::I8 | DType::I16 | DType::U16,
            ..
        }
    )));

    // With an integer scalar the same U8 source path stays at U8 storage;
    // it is still ordinary Sub/Mul/Add rather than the live-weight path.
    let integer = u8.lerp_scalar(start, end, Scalar::I(1)).unwrap();
    assert_eq!(u8.dtype(integer).unwrap(), DType::U8);
    assert!(matches!(
        u8.op(integer).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));

    for (dtype, weight, output_dtype) in [
        (DType::Bool, Scalar::Bool(true), DType::Bool),
        (DType::I8, Scalar::I(1), DType::I8),
        (DType::I16, Scalar::I(1), DType::I16),
        (DType::I32, Scalar::I(1), DType::I32),
        (DType::I64, Scalar::I(1), DType::I64),
        (DType::U8, Scalar::U(1), DType::U8),
        (DType::U16, Scalar::U(1), DType::U16),
        (DType::U32, Scalar::U(1), DType::U32),
        (DType::U64, Scalar::U(1), DType::U64),
        (DType::F16, Scalar::F(-0.0), DType::F16),
        (DType::BF16, Scalar::F(-0.0), DType::BF16),
        (DType::F32, Scalar::F(-0.0), DType::F32),
        (DType::F64, Scalar::F(-0.0), DType::F64),
    ] {
        let mut graph = Graph::new();
        let start = graph.input_dtype("start", [], dtype);
        let end = graph.input_dtype("end", [], dtype);
        let output = graph.lerp_scalar(start, end, weight).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
        assert_eq!(graph.dtype(output).unwrap(), output_dtype);
        assert!(matches!(graph.op(NodeId(2)).unwrap(), Op::Constant(data)
            if data.shape() == &Shape::new([]) && data.dtype() == dtype));
        if dtype.is_float() {
            let Op::Constant(data) = graph.op(NodeId(2)).unwrap() else {
                panic!("prepared scalar must be a constant");
            };
            assert_eq!(data.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());
        }
    }

    // Weak scalar staging happens at the multiplication consumer: an integer
    // weight retains F16 storage, while a float weight lifts an integer delta
    // and the final add to F32. Live I64/U64 still bridges at the initial Sub.
    let mut mixed = Graph::new();
    let start = mixed.input_dtype("narrow_start", [], DType::F16);
    let end = mixed.input_dtype("narrow_end", [], DType::F16);
    let narrow = mixed.lerp_scalar(start, end, Scalar::I(1)).unwrap();
    let start_i = mixed.input_dtype("integer_start", [], DType::I16);
    let end_i = mixed.input_dtype("integer_end", [], DType::I16);
    let lifted = mixed.lerp_scalar(start_i, end_i, Scalar::F(-0.0)).unwrap();
    let start_wide = mixed.input_dtype("i64", [2], DType::I64);
    let end_wide = mixed.input_dtype("u64", [1], DType::U64);
    let bridged = mixed
        .lerp_scalar(start_wide, end_wide, Scalar::F(0.5))
        .unwrap();
    assert_eq!(mixed.dtype(narrow).unwrap(), DType::F16);
    assert_eq!(mixed.dtype(lifted).unwrap(), DType::F32);
    assert_eq!(mixed.dtype(bridged).unwrap(), DType::F32);
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input, dtype: DType::F32 } if *input == start_wide
    )));
    assert!((0..mixed.node_count()).any(|index| matches!(
        mixed.op(NodeId(index)).unwrap(),
        Op::Cast { input, dtype: DType::F32 } if *input == end_wide
    )));

    let mut specials = Graph::new();
    let start = specials.input_dtype("start", [], DType::F64);
    let end = specials.input_dtype("end", [], DType::F64);
    let nan = specials
        .lerp_scalar(start, end, Scalar::F(f64::NAN))
        .unwrap();
    let infinity = specials
        .lerp_scalar(start, end, Scalar::F(f64::INFINITY))
        .unwrap();
    assert!(matches!(specials.op(NodeId(2)).unwrap(), Op::Constant(data)
        if data.scalar_at(0).as_f64().is_nan()));
    assert!(matches!(
        specials.op(nan).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!(matches!(
        specials.op(infinity).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));

    let mut vjp = Graph::new();
    let start = vjp.input_dtype("start", [2, 1], DType::F32);
    let end = vjp.input_dtype("end", [3], DType::F32);
    let output = vjp.lerp_scalar(start, end, Scalar::F(0.5)).unwrap();
    let loss = vjp.sum_all(output).unwrap();
    let start_gradient = vjp.grad(loss, start).unwrap();
    let end_gradient = vjp.grad(loss, end).unwrap();
    assert_eq!(vjp.shape(start_gradient).unwrap(), &Shape::new([2, 1]));
    assert_eq!(vjp.shape(end_gradient).unwrap(), &Shape::new([3]));

    let mut empty = Graph::new();
    let start = empty.input_dtype("start", [0, 2], DType::BF16);
    let end = empty.input_dtype("end", [1, 2], DType::BF16);
    let output = empty.lerp_scalar(start, end, Scalar::F(-0.0)).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.lerp_scalar(NodeId(usize::MAX), NodeId(0), Scalar::F(0.5)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let start = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let end = malformed.input_dtype("end", [], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.lerp_scalar(start, end, Scalar::F(0.5)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn linear_matches_tinygrad_weight_layout_dtype_and_vjps() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2, 2]);
    let weight = graph.input("weight", [2, 2]);
    let bias = graph.input("bias", [2]);
    let output = graph
        .linear(input, weight, Some(bias), Some(DType::F64))
        .unwrap();
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let weight_gradient = graph.grad(loss, weight).unwrap();
    let bias_gradient = graph.grad(loss, bias).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ),
        (
            "weight".into(),
            TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ),
        (
            "bias".into(),
            TensorData::new([2], vec![0.5, -0.5]).unwrap(),
        ),
    ]);
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert_eq!(graph.dtype(input_gradient).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend
            .execute(&graph, output, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![7.5, 9.5, 15.5, 21.5]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, input_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![3.0, 7.0, 3.0, 7.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, weight_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![4.0, 4.0, 6.0, 6.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, bias_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![2.0, 2.0]
    );

    let mut vector_weight = Graph::new();
    let input = vector_weight.input("input", [2, 2]);
    let weight = vector_weight.input("weight", [2]);
    let output = vector_weight.linear(input, weight, None, None).unwrap();
    assert_eq!(vector_weight.shape(output).unwrap(), &Shape::new([2, 2]));
    assert_eq!(
        CpuBackend
            .execute(
                &vector_weight,
                output,
                &HashMap::from([
                    (
                        "input".into(),
                        TensorData::new([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                    ),
                    (
                        "weight".into(),
                        TensorData::new([2], vec![10.0, 100.0]).unwrap()
                    ),
                ]),
            )
            .unwrap()
            .to_vec_f64(),
        vec![10.0, 200.0, 30.0, 400.0]
    );

    let mut empty = Graph::new();
    let input = empty.input("input", [2, 0]);
    let weight = empty.input("weight", [0]);
    let output = empty.linear(input, weight, None, None).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([2, 0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([
                    ("input".into(), TensorData::new([2, 0], vec![]).unwrap()),
                    ("weight".into(), TensorData::new([0], vec![]).unwrap()),
                ]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let mut malformed = Graph::new();
    let input = malformed.input("input", [2, 2]);
    let weight = malformed.input("weight", [2, 2]);
    let bias = malformed.input("bias", [3]);
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.linear(input, weight, Some(bias), None),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(malformed.node_count(), node_count);

    let scalar_weight = malformed.input("scalar_weight", []);
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.linear(input, scalar_weight, None, None),
        Err(Error::InvalidMatmul { .. })
    ));
    assert_eq!(malformed.node_count(), node_count);
}

#[test]
fn linear_is_source_dot_not_raw_matmul_and_is_atomic() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2, 3], DType::F16);
    // tinygrad passes this `[contract, output]` descriptor directly to dot.
    let weight = graph.input_dtype("weight", [3, 4], DType::BF16);
    let bias = graph.input_dtype("bias", [4], DType::I64);
    let output = graph
        .linear(input, weight, Some(bias), Some(DType::F32))
        .unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 4]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(
        (0..graph.node_count())
            .all(|node| !matches!(graph.op(NodeId(node)).unwrap(), Op::Matmul { .. }))
    );
    assert!((0..graph.node_count()).any(|node| matches!(
        graph.op(NodeId(node)).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    )));
    assert!((0..graph.node_count()).any(|node| matches!(
        graph.op(NodeId(node)).unwrap(),
        Op::Reduce {
            kind: ReduceKind::Sum,
            ..
        }
    )));
    // Dot's sole transpose applies to its own reshaped rhs, never directly
    // to the caller's weight as the stale conventional layout did.
    assert!(
        (0..graph.node_count())
            .filter_map(|node| match graph.op(NodeId(node)).unwrap() {
                Op::Permute { input, .. } => Some(*input),
                _ => None,
            })
            .all(|input| matches!(graph.op(input).unwrap(), Op::Reshape { .. }))
    );
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.shape(input_gradient).unwrap(), &Shape::new([2, 3]));
    let weight_gradient = graph.grad(loss, weight).unwrap();
    assert_eq!(graph.shape(weight_gradient).unwrap(), &Shape::new([3, 4]));
    assert!(matches!(
        graph.grad(loss, bias),
        Err(Error::NonDifferentiableTarget(node)) if node == bias
    ));

    let mut rank_one = Graph::new();
    let input = rank_one.input_dtype("input", [2, 3], DType::U8);
    let weight = rank_one.input_dtype("weight", [3], DType::I16);
    let output = rank_one.linear(input, weight, None, None).unwrap();
    assert_eq!(rank_one.shape(output).unwrap(), &Shape::new([2, 3]));
    assert!((0..rank_one.node_count()).all(|node| !matches!(
        rank_one.op(NodeId(node)).unwrap(),
        Op::Reduce { .. } | Op::Matmul { .. }
    )));

    let mut zero = Graph::new();
    let input = zero.input_dtype("input", [2, 0, 3], DType::F16);
    let weight = zero.input_dtype("weight", [3, 4], DType::F16);
    let output = zero.linear(input, weight, None, None).unwrap();
    assert_eq!(zero.shape(output).unwrap(), &Shape::new([2, 0, 4]));

    let mut malformed = Graph::new();
    let input = malformed.input_dtype("input", [usize::MAX / 8, 1], DType::F32);
    let weight = malformed.input_dtype("weight", [1, 3], DType::F32);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.linear(input, weight, None, None),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
    assert!(matches!(
        malformed.linear(NodeId(usize::MAX), weight, None, None),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn public_pad_modes_are_literal_composites_and_atomic() {
    for mode in [
        PadMode::Constant,
        PadMode::Circular,
        PadMode::Reflect,
        PadMode::Replicate,
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("x", [2, 3], DType::F16);
        let output = graph
            .pad_with_mode(input, [(1, 0), (1, 1)], mode, Scalar::F(0.0))
            .unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([3, 5]));
        assert_eq!(graph.dtype(output).unwrap(), DType::F16);
    }

    let mut constant = Graph::new();
    let input = constant.input_dtype("x", [2], DType::I16);
    let output = constant
        .pad_with_mode(input, [(1, 1)], PadMode::Constant, Scalar::F(f64::NAN))
        .unwrap();
    // Nonzero source fill is Bool-mask Where, so its weak scalar can widen
    // the raw-pad payload instead of being forcibly truncated by Op::Pad.
    assert_eq!(constant.dtype(output).unwrap(), DType::F32);
    assert!(
        (0..constant.node_count())
            .any(|node| matches!(constant.op(NodeId(node)).unwrap(), Op::Select { .. }))
    );
    let signed_zero = constant
        .pad_with_mode(input, [(0, 0)], PadMode::Constant, Scalar::F(-0.0))
        .unwrap();
    assert_eq!(constant.dtype(signed_zero).unwrap(), DType::I16);

    let mut circular = Graph::new();
    let input = circular.input("x", [3]);
    let output = circular
        .pad_with_mode(input, [(-1, 1)], PadMode::Circular, Scalar::I(0))
        .unwrap();
    assert_eq!(circular.shape(output).unwrap(), &Shape::new([3]));
    let nodes = circular.node_count();
    assert!(
        circular
            .pad_with_mode(input, [(4, 0)], PadMode::Circular, Scalar::I(0))
            .is_err()
    );
    assert_eq!(circular.node_count(), nodes);

    let mut reflected = Graph::new();
    let input = reflected.input("x", [3]);
    let output = reflected
        .pad_with_mode(input, [(1, -1)], PadMode::Reflect, Scalar::I(0))
        .unwrap();
    assert_eq!(reflected.shape(output).unwrap(), &Shape::new([3]));
    assert!(
        (0..reflected.node_count())
            .any(|node| matches!(reflected.op(NodeId(node)).unwrap(), Op::Stride { .. }))
    );
    let loss = reflected.sum_all(output).unwrap();
    let gradient = reflected.grad(loss, input).unwrap();
    assert_eq!(reflected.shape(gradient).unwrap(), &Shape::new([3]));

    let mut replicate = Graph::new();
    let input = replicate.input("x", [2, 1]);
    let output = replicate
        .pad_with_mode(input, [(0, 0), (2, 1)], PadMode::Replicate, Scalar::I(0))
        .unwrap();
    assert_eq!(replicate.shape(output).unwrap(), &Shape::new([2, 4]));
    assert!(
        (0..replicate.node_count())
            .any(|node| matches!(replicate.op(NodeId(node)).unwrap(), Op::Expand { .. }))
    );

    let mut scalar = Graph::new();
    let input = scalar.input("x", []);
    assert_eq!(
        scalar
            .pad_with_mode(input, [], PadMode::Circular, Scalar::I(0))
            .unwrap(),
        input
    );

    let mut malformed = Graph::new();
    let input = malformed.input("x", [usize::MAX / 4, 1]);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.pad_with_mode(input, [(0, 1), (0, 0)], PadMode::Constant, Scalar::I(0)),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
    assert!(matches!(
        malformed.pad_with_mode(
            NodeId(usize::MAX),
            [(0, 0), (0, 0)],
            PadMode::Constant,
            Scalar::I(0)
        ),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn public_pad_to_is_strict_target_shape_and_source_mask_fill() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [1, 2], DType::I16);
    let default = graph.pad_to(input, [Some(3), None]).unwrap();
    let cropped = graph.pad_to(input, [Some(0), Some(1)]).unwrap();
    assert_eq!(graph.shape(default).unwrap(), &Shape::new([3, 2]));
    assert_eq!(graph.dtype(default).unwrap(), DType::I16);
    assert_eq!(graph.shape(cropped).unwrap(), &Shape::new([0, 1]));

    // A changed target with a concrete nonzero Python fill follows OpMixin's
    // Bool pad then `where(base, value)` shell, allowing its weak scalar to
    // lift storage instead of raw Pad truncating it to I16.
    let filled = graph
        .pad_to_with_value(input, [Some(2), Some(2)], Scalar::F(f64::NAN))
        .unwrap();
    assert_eq!(graph.shape(filled).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.dtype(filled).unwrap(), DType::F32);
    assert!(
        (0..graph.node_count())
            .any(|node| matches!(graph.op(NodeId(node)).unwrap(), Op::Select { .. }))
    );
    let mut differentiable = Graph::new();
    let input = differentiable.input("x", [1, 2]);
    let output = differentiable.pad_to(input, [Some(3), None]).unwrap();
    let loss = differentiable.sum_all(output).unwrap();
    let gradient = differentiable.grad(loss, input).unwrap();
    assert_eq!(differentiable.shape(gradient).unwrap(), &Shape::new([1, 2]));

    // Source returns self before considering `value` when no target extent
    // changes, including a signed-zero/nonrepresentable scalar.
    assert_eq!(
        graph
            .pad_to_with_value(input, [Some(1), Some(2)], Scalar::F(-0.0))
            .unwrap(),
        input
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0, 2], DType::U8);
    let output = empty.pad_to(input, [Some(1), None]).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([1, 2]));

    let mut malformed = Graph::new();
    let input = malformed.input("x", [2, 3]);
    let before = malformed.node_count();
    assert!(malformed.pad_to(input, [Some(2)]).is_err());
    assert_eq!(malformed.node_count(), before);
    assert!(
        malformed
            .pad_to(NodeId(usize::MAX), [Some(2), Some(3)])
            .is_err()
    );
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype(
        "overflow",
        [usize::MAX / DType::F64.itemsize() + 1],
        DType::F64,
    );
    let before = malformed.node_count();
    assert!(malformed.pad_to(overflow, [None]).is_err());
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn softplus_uses_tinygrad_logaddexp_and_preflights_beta() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let beta = graph.input("beta", []);
    let output = graph.softplus(input, beta).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let beta_gradient = graph.grad(loss, beta).unwrap();
    let bindings = HashMap::from([
        (
            "input".into(),
            TensorData::new([2], vec![0.0, 1.0]).unwrap(),
        ),
        ("beta".into(), TensorData::scalar(2.0)),
    ]);
    let output_values = CpuBackend
        .execute(&graph, output, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!((output_values[0] - (2.0f64.ln() / 2.0)).abs() < 1e-6);
    assert!((output_values[1] - ((1.0 + 2.0f64.exp()).ln() / 2.0)).abs() < 1e-6);
    let input_values = CpuBackend
        .execute(&graph, input_gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert!((input_values[0] - 0.5).abs() < 1e-6);
    assert!((input_values[1] - (2.0f64.exp() / (1.0 + 2.0f64.exp()))).abs() < 1e-6);
    let beta_value = CpuBackend
        .execute(&graph, beta_gradient, &bindings)
        .unwrap()
        .scalar_at(0)
        .as_f64();
    let expected_beta = -2.0f64.ln() / 4.0 + 2.0f64.exp() / (2.0 * (1.0 + 2.0f64.exp()))
        - (1.0 + 2.0f64.exp()).ln() / 4.0;
    assert!((beta_value - expected_beta).abs() < 1e-6);

    let mut special = Graph::new();
    let input = special.input("input", [4]);
    let beta = special.input("beta", []);
    let output = special.softplus(input, beta).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([
                (
                    "input".into(),
                    TensorData::new([4], vec![1000.0, -1000.0, f32::INFINITY, f32::NAN]).unwrap(),
                ),
                ("beta".into(), TensorData::scalar(1.0)),
            ]),
        )
        .unwrap()
        .to_vec_f64();
    assert_eq!(values[0], 1000.0);
    assert_eq!(values[1], 0.0);
    // Source logaddexp subtracts its ordered maximum. At +inf that literal
    // composition evaluates `inf - inf`, so the lane remains NaN.
    assert!(values[2].is_nan());
    assert!(values[3].is_nan());

    let mut scalar = Graph::new();
    let input = scalar.input("input", []);
    let beta = scalar.input("beta", []);
    let output = scalar.softplus(input, beta).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert!(
        CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([
                    ("input".into(), TensorData::scalar(0.0)),
                    ("beta".into(), TensorData::scalar(1.0)),
                ]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64()
            .is_finite()
    );

    let mut empty = Graph::new();
    let input = empty.input("input", [0]);
    let beta = empty.input("beta", []);
    let output = empty.softplus(input, beta).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([
                    ("input".into(), TensorData::new([0], vec![]).unwrap()),
                    ("beta".into(), TensorData::scalar(1.0)),
                ]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let mut malformed = Graph::new();
    let input = malformed.input("input", [2, 3]);
    let beta = malformed.input("beta", [2, 2]);
    let node_count = malformed.node_count();
    assert!(matches!(
        malformed.softplus(input, beta),
        Err(Error::BroadcastMismatch { .. })
    ));
    assert_eq!(malformed.node_count(), node_count);
}

#[test]
fn mish_reuses_the_stable_tinygrad_softplus_composition() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let output = graph.mish(input).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let bindings = HashMap::from([(
        "input".into(),
        TensorData::new([2], vec![0.0, 1.0]).unwrap(),
    )]);
    let values = CpuBackend
        .execute(&graph, output, &bindings)
        .unwrap()
        .to_vec_f64();
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(values[0], 0.0);
    assert!((values[1] - (1.0 + 1.0f64.exp()).ln().tanh()).abs() < 1e-6);
    let gradient_values = CpuBackend
        .execute(&graph, gradient, &bindings)
        .unwrap()
        .to_vec_f64();
    assert!((gradient_values[0] - 0.6).abs() < 1e-6);

    let mut special = Graph::new();
    let input = special.input("input", [4]);
    let output = special.mish(input).unwrap();
    let values = CpuBackend
        .execute(
            &special,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::new([4], vec![1000.0, -1000.0, f32::INFINITY, f32::NAN]).unwrap(),
            )]),
        )
        .unwrap()
        .to_vec_f64();
    assert_eq!(values[0], 1000.0);
    assert_eq!(values[1], 0.0);
    assert!(values[1].is_sign_negative());
    assert!(values[2].is_nan());
    assert!(values[3].is_nan());

    let mut signed_zero = Graph::new();
    let input = signed_zero.input_dtype("input", [], DType::F64);
    let output = signed_zero.mish(input).unwrap();
    let value = CpuBackend
        .execute(
            &signed_zero,
            output,
            &HashMap::from([(
                "input".into(),
                TensorData::scalar_with_dtype(Scalar::F(-0.0), DType::F64),
            )]),
        )
        .unwrap();
    assert_eq!(value.scalar_at(0).as_f64().to_bits(), (-0.0f64).to_bits());

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let input = narrow.input_dtype("input", [], dtype);
        let output = narrow.mish(input).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut promoted = Graph::new();
        let input = promoted.input_dtype("input", [], dtype);
        let output = promoted.mish(input).unwrap();
        assert_eq!(promoted.dtype(output).unwrap(), DType::F32);
    }

    let mut scalar = Graph::new();
    let input = scalar.input("input", []);
    let output = scalar.mish(input).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(
        CpuBackend
            .execute(
                &scalar,
                output,
                &HashMap::from([("input".into(), TensorData::scalar(0.0))]),
            )
            .unwrap()
            .scalar_at(0)
            .as_f64(),
        0.0
    );

    let mut empty = Graph::new();
    let input = empty.input("input", [0]);
    let output = empty.mish(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(
        graph.mish(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn hardsigmoid_scalar_preserves_source_left_alpha_and_staged_relu_difference() {
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2], dtype);
        let output = graph.hardsigmoid_scalar(input, 0.25, -0.0).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
        assert!(matches!(
            graph.op(output).unwrap(),
            Op::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        assert!((0..graph.node_count()).any(|index| matches!(graph.op(NodeId(index)).unwrap(),
            Op::Binary { op: BinaryOp::Mul, lhs, rhs } if matches!(graph.op(*lhs).unwrap(), Op::Constant(_)) && *rhs == input)));
        let loss = graph.sum_all(output).unwrap();
        assert!(graph.grad(loss, input).is_ok());
    }

    let mut default = Graph::new();
    let input = default.input_dtype("input", [], DType::F64);
    let output = default.hardsigmoid(input).unwrap();
    assert!(matches!(
        default.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    assert!((0..default.node_count()).any(|index| matches!(default.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().to_bits() == (1.0f64 / 6.0).to_bits())));
    assert!((0..default.node_count()).any(|index| matches!(default.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().to_bits() == 0.5f64.to_bits())));

    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [], dtype);
        let output = graph
            .hardsigmoid_scalar(input, f64::NAN, f64::INFINITY)
            .unwrap();
        assert_eq!(graph.dtype(output).unwrap(), DType::F32);
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([]));
    }

    let mut scalar = Graph::new();
    let input = scalar.input_dtype("input", [], DType::F64);
    let output = scalar.hardsigmoid_scalar(input, -0.0, f64::NAN).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert!((0..scalar.node_count()).any(|index| matches!(scalar.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().to_bits() == (-0.0f64).to_bits())));
    assert!(
        (0..scalar.node_count()).any(|index| matches!(scalar.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().is_nan()))
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::BF16);
    let output = empty.hardsigmoid_scalar(input, 0.25, 0.5).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.hardsigmoid_scalar(NodeId(usize::MAX), 0.25, 0.5),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.hardsigmoid_scalar(overflow, 0.25, 0.5),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn gelu_default_delegates_to_tinygrad_tanh_without_affecting_onnx_mode() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [2], DType::F64);
    let output = graph.gelu_default(input).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Mul,
            ..
        }
    ));
    assert!((0..graph.node_count()).any(|index| matches!(graph.op(NodeId(index)).unwrap(),
        Op::Constant(data) if data.dtype() == DType::F64 && data.scalar_at(0).as_f64().to_bits() == 0.044_715f64.to_bits())));
    assert!((0..graph.node_count()).any(|index| matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Binary {
            op: BinaryOp::Pow,
            ..
        }
    )));
    assert!((0..graph.node_count()).all(|index| !matches!(
        graph.op(NodeId(index)).unwrap(),
        Op::Unary {
            op: UnaryOp::Erf,
            ..
        }
    )));
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, input).is_ok());

    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let mut narrow = Graph::new();
        let input = narrow.input_dtype("input", [], dtype);
        let output = narrow.gelu_default(input).unwrap();
        assert_eq!(narrow.dtype(output).unwrap(), dtype);
        assert_eq!(narrow.shape(output).unwrap(), &Shape::new([]));
    }
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
    ] {
        let mut promoted = Graph::new();
        let input = promoted.input_dtype("input", [], dtype);
        let output = promoted.gelu_default(input).unwrap();
        assert_eq!(promoted.dtype(output).unwrap(), DType::F32);
    }

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0, 2], DType::F16);
    let output = empty.gelu_default(input).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 2]));

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.gelu_default(NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.gelu_default(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}

#[test]
fn batchnorm_is_source_literal_affine_with_raw_axis_membership() {
    // Source reshapes mean/weight/bias to the membership-derived shape, then
    // reshapes invstd only when its rank equals `len(argfix(axis))`.
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3, 4], DType::F16);
    let weight = graph.input_dtype("weight", [3], DType::U64);
    let bias = graph.input_dtype("bias", [3], DType::BF16);
    let mean = graph.input_dtype("mean", [3], DType::I64);
    let invstd = graph.input_dtype("invstd", [3], DType::F32);
    let output = graph
        .batchnorm(input, Some(weight), Some(bias), mean, invstd, 1)
        .unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3, 4]));
    // The I64/U64 intermediate crosses the checked-in source's F32 bridge.
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Binary {
            op: BinaryOp::Add,
            ..
        }
    ));
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, input).is_ok());

    let mut default_axis = Graph::new();
    let input = default_axis.input("x", [2, 3]);
    let mean = default_axis.input("mean", [3]);
    let invstd = default_axis.input("invstd", [3]);
    assert_eq!(
        default_axis
            .batchnorm_default(input, None, None, mean, invstd)
            .and_then(|node| default_axis.shape(node).cloned())
            .unwrap(),
        Shape::new([2, 3])
    );

    // `argfix` does not normalize these axes. The duplicate changes only the
    // invstd rank test; the negative axis never matches enumerate().
    let mut unusual = Graph::new();
    let input = unusual.input_dtype("x", [2, 3], DType::F32);
    let mean = unusual.input_dtype("mean", [3], DType::F32);
    let invstd = unusual.input_dtype("invstd", [1, 3], DType::F32);
    let output = unusual
        .batchnorm_with_axes(input, None, None, mean, invstd, vec![1, 1])
        .unwrap();
    assert_eq!(unusual.shape(output).unwrap(), &Shape::new([2, 3]));
    let input = unusual.input_dtype("negative_axis_x", [2, 3], DType::F32);
    let mean = unusual.input_dtype("negative_axis_mean", [], DType::F32);
    let invstd = unusual.input_dtype("negative_axis_invstd", [1], DType::F32);
    assert_eq!(
        unusual
            .batchnorm(input, None, None, mean, invstd, -1)
            .and_then(|node| unusual.shape(node).cloned())
            .unwrap(),
        Shape::new([2, 3])
    );
}

#[test]
fn batchnorm_preflights_optional_reshapes_and_late_broadcast_overflow() {
    let mut malformed = Graph::new();
    let input = malformed.input_dtype("x", [2, 3], DType::F32);
    let mean = malformed.input_dtype("mean", [2], DType::F32);
    let invstd = malformed.input_dtype("invstd", [3], DType::F32);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.batchnorm(input, None, None, mean, invstd, 1),
        Err(Error::InvalidReshape { .. })
    ));
    assert_eq!(malformed.node_count(), before);

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0, 3], DType::I16);
    let mean = empty.input_dtype("mean", [3], DType::I16);
    let invstd = empty.input_dtype("invstd", [3], DType::I16);
    let output = empty.batchnorm(input, None, None, mean, invstd, 1).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    assert_eq!(empty.dtype(output).unwrap(), DType::I16);

    // Input and centered descriptors fit individually. Only invstd's raw
    // branch expands the later multiply, so this proves whole-operation
    // planning rather than relying on the first published reshape/sub node.
    let mut overflow = Graph::new();
    let extent = usize::MAX / 4;
    let input = overflow.input_dtype("x", [extent, 1], DType::F32);
    let mean = overflow.input_dtype("mean", [extent, 1], DType::F32);
    let invstd = overflow.input_dtype("invstd", [1, 2], DType::F32);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.batchnorm(input, None, None, mean, invstd, 0),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn source_dot_uses_tinygrad_typed_sum_and_source_layout() {
    let mut graph = Graph::new();
    let lhs = graph.input_dtype("lhs", [2, 3], DType::F16);
    let rhs = graph.input_dtype("rhs", [3, 4], DType::F16);
    let output = graph.dot_default(lhs, rhs).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 4]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F16);
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Reduce { kind: ReduceKind::Sum, axes, keepdim: false, .. } if axes == &vec![2])
            && node.dtype == DType::F32
    }));
    // Narrow products are accumulated in F32, then source-cast back.
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    )));
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Cast {
            dtype: DType::F16,
            ..
        }
    ));

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, lhs).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([2, 3]));
    let gradient_loss = graph.sum_all(gradient).unwrap();
    assert!(matches!(
        graph.grad(gradient_loss, lhs),
        Err(Error::NoGradient(node)) if node == lhs
    ));

    // The literal Mul then Sum sequence owns IEEE special propagation rather
    // than inheriting a raw Matmul implementation.
    let mut specials = Graph::new();
    let lhs = specials.input("lhs", [2]);
    let rhs = specials.input("rhs", [2]);
    let output = specials.dot_default(lhs, rhs).unwrap();
    let values = CpuBackend
        .execute(
            &specials,
            output,
            &HashMap::from([
                ("lhs".into(), TensorData::new([2], vec![1.0, -1.0]).unwrap()),
                (
                    "rhs".into(),
                    TensorData::new([2], vec![f32::INFINITY, f32::INFINITY]).unwrap(),
                ),
            ]),
        )
        .unwrap()
        .to_vec_f64();
    assert!(values[0].is_nan());
}

#[test]
fn source_dot_covers_vectors_batches_storage_and_empty_contractions() {
    let mut vector = Graph::new();
    let lhs = vector.input_dtype("lhs", [3], DType::I8);
    let rhs = vector.input_dtype("rhs", [3], DType::I8);
    let output = vector.dot_default(lhs, rhs).unwrap();
    assert_eq!(vector.shape(output).unwrap(), &Shape::new([]));
    assert_eq!(vector.dtype(output).unwrap(), DType::I8);
    // I8 products reduce in I32 before the source-required final narrowing.
    assert!(vector.nodes.iter().any(|node| {
        matches!(
            &node.op,
            Op::Reduce {
                kind: ReduceKind::Sum,
                ..
            }
        ) && node.dtype == DType::I32
    }));
    assert!(matches!(
        vector.op(output).unwrap(),
        Op::Cast {
            dtype: DType::I8,
            ..
        }
    ));

    let mut bridged = Graph::new();
    let lhs = bridged.input_dtype("lhs", [2, 3, 4], DType::I64);
    let rhs = bridged.input_dtype("rhs", [1, 4, 5], DType::U64);
    let output = bridged.dot_default(lhs, rhs).unwrap();
    assert_eq!(bridged.shape(output).unwrap(), &Shape::new([2, 3, 5]));
    assert_eq!(bridged.dtype(output).unwrap(), DType::F32);
    assert!(
        bridged
            .nodes
            .iter()
            .filter(|node| matches!(
                &node.op,
                Op::Cast {
                    dtype: DType::F32,
                    ..
                }
            ))
            .count()
            >= 2
    );

    let mut explicit = Graph::new();
    let lhs = explicit.input_dtype("lhs", [2, 0], DType::BF16);
    let rhs = explicit.input_dtype("rhs", [0, 3], DType::BF16);
    let output = explicit.dot(lhs, rhs, Some(DType::F64)).unwrap();
    assert_eq!(explicit.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(explicit.dtype(output).unwrap(), DType::F64);
    assert!(explicit.nodes.iter().any(|node| {
        matches!(
            &node.op,
            Op::Reduce {
                kind: ReduceKind::Sum,
                ..
            }
        ) && node.dtype == DType::F64
    }));
}

#[test]
fn source_dot_preflights_invalid_and_overflow_contracts_atomically() {
    let mut scalar = Graph::new();
    let lhs = scalar.input_dtype("lhs", [], DType::F32);
    let rhs = scalar.input_dtype("rhs", [1], DType::F32);
    let before = scalar.node_count();
    assert!(matches!(
        scalar.dot_default(lhs, rhs),
        Err(Error::InvalidMatmul { .. })
    ));
    assert_eq!(scalar.node_count(), before);

    let mut mismatch = Graph::new();
    let lhs = mismatch.input("lhs", [2, 3]);
    let rhs = mismatch.input("rhs", [4, 2]);
    let before = mismatch.node_count();
    assert!(matches!(
        mismatch.dot_default(lhs, rhs),
        Err(Error::InvalidMatmul { .. })
    ));
    assert_eq!(mismatch.node_count(), before);

    let mut overflow = Graph::new();
    let lhs = overflow.input_dtype("lhs", [usize::MAX / 2, 2], DType::F64);
    let rhs = overflow.input_dtype("rhs", [2, 1], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.dot_default(lhs, rhs),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn qr_is_full_householder_composition_with_typed_dot_updates() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 3], DType::F16);
    let (q, r) = graph.qr(input).unwrap();
    assert_eq!(graph.shape(q).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.shape(r).unwrap(), &Shape::new([2, 3]));
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Unary {
            op: UnaryOp::Sign,
            ..
        }
    )));
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Select { .. }))
    );
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Reduce { kind: ReduceKind::Sum, axes, .. } if axes.len() == 1)
            && node.dtype == DType::F32
    }));
    assert!(
        !graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Matmul { .. }))
    );
    // Eye and the Householder row index use scalar-backed lazy ranges only.
    assert!(
        graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|len| len == 1)
    );
    let q_loss = graph.sum_all(q).unwrap();
    assert!(graph.grad(q_loss, input).is_ok());
    let r_loss = graph.sum_all(r).unwrap();
    assert!(graph.grad(r_loss, input).is_ok());
}

#[test]
fn qr_covers_tall_wide_batched_zero_and_storage_descriptors() {
    let mut tall = Graph::new();
    let input = tall.input_dtype("x", [2, 4, 2], DType::I16);
    let (q, r) = tall.qr(input).unwrap();
    assert_eq!(tall.shape(q).unwrap(), &Shape::new([2, 4, 4]));
    assert_eq!(tall.shape(r).unwrap(), &Shape::new([2, 4, 2]));
    // Nonfloating reflectors promote through sqrt/div while retaining full Q/R.
    assert_eq!(tall.dtype(q).unwrap(), DType::F32);
    assert_eq!(tall.dtype(r).unwrap(), DType::F32);

    let mut wide = Graph::new();
    let input = wide.input_dtype("x", [2, 4], DType::F64);
    let (q, r) = wide.qr(input).unwrap();
    assert_eq!(wide.shape(q).unwrap(), &Shape::new([2, 2]));
    assert_eq!(wide.shape(r).unwrap(), &Shape::new([2, 4]));
    assert_eq!(wide.dtype(q).unwrap(), DType::F64);
    assert_eq!(wide.dtype(r).unwrap(), DType::F64);

    let mut zero_rows = Graph::new();
    let input = zero_rows.input_dtype("x", [0, 3], DType::BF16);
    let (q, r) = zero_rows.qr(input).unwrap();
    assert_eq!(zero_rows.shape(q).unwrap(), &Shape::new([0, 0]));
    assert_eq!(r, input);

    let mut zero_columns = Graph::new();
    let input = zero_columns.input_dtype("x", [3, 0], DType::Bool);
    let (q, r) = zero_columns.qr(input).unwrap();
    assert_eq!(zero_columns.shape(q).unwrap(), &Shape::new([3, 3]));
    assert_eq!(zero_columns.dtype(q).unwrap(), DType::Bool);
    assert_eq!(r, input);
}

#[test]
fn qr_preflights_rank_and_extent_failures_without_publication() {
    let mut scalar = Graph::new();
    let input = scalar.input_dtype("x", [], DType::F32);
    let before = scalar.node_count();
    assert!(matches!(scalar.qr(input), Err(Error::InvalidMatmul { .. })));
    assert_eq!(scalar.node_count(), before);

    let mut vector = Graph::new();
    let input = vector.input_dtype("x", [3], DType::F32);
    let before = vector.node_count();
    assert!(matches!(vector.qr(input), Err(Error::InvalidMatmul { .. })));
    assert_eq!(vector.node_count(), before);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("x", [usize::MAX, 2], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(overflow.qr(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn newton_schulz_is_source_literal_typed_dot_polynomial() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("x", [2, 2], DType::F16);
    let output = graph
        .newton_schulz_default_eps(input, 2, &[2, -1, 1])
        .unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F16);
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Unary {
            op: UnaryOp::Sqrt,
            ..
        }
    )));
    assert!(graph.nodes.iter().any(|node| {
        matches!(&node.op, Op::Reduce { kind: ReduceKind::Sum, axes, keepdim: true, .. }
            if axes == &vec![0, 1])
    }));
    // Every polynomial Gram/update product is the typed Dot composite, not
    // raw Matmul, and lazy/scalar construction never carries a dense payload.
    assert!(
        !graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Matmul { .. }))
    );
    assert!(
        graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|len| len == 1)
    );
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, input).is_ok());
}

#[test]
fn newton_schulz_covers_rectangular_batches_steps_and_empty_shapes() {
    let mut tall = Graph::new();
    let input = tall.input_dtype("x", [3, 2], DType::F32);
    let output = tall
        .newton_schulz(input, 1, &[1, -1], f64::INFINITY)
        .unwrap();
    assert_eq!(tall.shape(output).unwrap(), &Shape::new([3, 2]));
    assert!(
        tall.nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Permute { .. }))
    );

    let mut batched = Graph::new();
    let input = batched.input_dtype("x", [2, 2, 4], DType::I16);
    let output = batched.newton_schulz(input, 1, &[2, -1], f64::NAN).unwrap();
    assert_eq!(batched.shape(output).unwrap(), &Shape::new([2, 2, 4]));
    assert_eq!(batched.dtype(output).unwrap(), DType::F32);

    let mut empty = Graph::new();
    let input = empty.input_dtype("x", [0, 3], DType::BF16);
    let output = empty.newton_schulz_default_eps(input, 1, &[1]).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));

    // Python `range(-1)` skips the reduce entirely, so empty params are not
    // observed and the normalized G remains the result.
    let mut negative = Graph::new();
    let input = negative.input_dtype("x", [2, 3], DType::F32);
    let output = negative.newton_schulz_default_eps(input, -1, &[]).unwrap();
    assert_eq!(negative.shape(output).unwrap(), &Shape::new([2, 3]));
}

#[test]
fn newton_schulz_preflights_rank_params_and_overflow_atomically() {
    let mut scalar = Graph::new();
    let input = scalar.input_dtype("x", [], DType::F32);
    let before = scalar.node_count();
    assert!(matches!(
        scalar.newton_schulz_default_eps(input, 1, &[1]),
        Err(Error::InvalidMatmul { .. })
    ));
    assert_eq!(scalar.node_count(), before);

    let mut empty_params = Graph::new();
    let input = empty_params.input("x", [2, 2]);
    let before = empty_params.node_count();
    assert!(matches!(
        empty_params.newton_schulz_default_eps(input, 1, &[]),
        Err(Error::InvalidRandom {
            reason: "newton_schulz requires nonempty params for positive steps"
        })
    ));
    assert_eq!(empty_params.node_count(), before);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("x", [usize::MAX, 2], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.newton_schulz_default_eps(input, 1, &[1]),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn scatter_reduce_is_source_one_hot_select_composition_for_every_kind() {
    for kind in [
        ScatterReduceKind::Sum,
        ScatterReduceKind::Prod,
        ScatterReduceKind::Mean,
        ScatterReduceKind::Amax,
        ScatterReduceKind::Amin,
    ] {
        for include_self in [true, false] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [2, 3], DType::F16);
            let index = graph.input_dtype("index", [2, 2], DType::I32);
            // The source crops this wider update tensor to `index.shape`.
            let src = graph.input_dtype("src", [2, 4], DType::F16);
            let output = graph
                .scatter_reduce(base, -1, index, src, kind, include_self)
                .unwrap();
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
            assert_eq!(graph.dtype(output).unwrap(), DType::F16);
            assert!(
                graph
                    .nodes
                    .iter()
                    .any(|node| matches!(&node.op, Op::Select { .. }))
            );
            assert!(
                graph
                    .nodes
                    .iter()
                    .any(|node| matches!(&node.op, Op::Reduce { axes, .. } if axes.len() == 1))
            );
            // Invalid negative/out-of-range labels stay false through Eq and
            // Select; raw Scatter would instead expose an indexing contract.
            assert!(
                !graph
                    .nodes
                    .iter()
                    .any(|node| matches!(&node.op, Op::Scatter { .. }))
            );
            assert!(
                graph
                    .nodes
                    .iter()
                    .filter_map(|node| match &node.op {
                        Op::Constant(data) => Some(data.len()),
                        _ => None,
                    })
                    .all(|length| length == 1)
            );
            let loss = graph.sum_all(output).unwrap();
            assert!(graph.grad(loss, base).is_ok());
            assert!(graph.grad(loss, src).is_ok());
            assert!(graph.grad(loss, index).is_err());
        }
    }
}

#[test]
fn scatter_reduce_covers_signed_dims_zero_domains_and_dtype_boundaries() {
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut family = Graph::new();
        let base = family.input_dtype("base", [1, 2], dtype);
        let index = family.input_dtype("index", [1, 1], DType::I32);
        let src = family.input_dtype("src", [1, 1], dtype);
        let output = family
            .scatter_reduce_default(base, 1, index, src, ScatterReduceKind::Amax)
            .unwrap();
        assert_eq!(family.shape(output).unwrap(), &Shape::new([1, 2]));
    }

    let mut graph = Graph::new();
    let base = graph.input_dtype("base", [2, 0], DType::U8);
    let index = graph.input_dtype("index", [1, 0], DType::U64);
    let src = graph.input_dtype("src", [1, 0], DType::U8);
    let output = graph
        .scatter_reduce_default(base, -1, index, src, ScatterReduceKind::Mean)
        .unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 0]));
    // The Bool one-hot range is source-default I32 unless endpoint planning
    // requires I64; it is never materialized as a dense class constant.
    assert!(
        graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|length| length == 1)
    );

    let mut scalar = Graph::new();
    let base = scalar.input_dtype("base", [], DType::F32);
    let index = scalar.input_dtype("index", [], DType::I32);
    let src = scalar.input_dtype("src", [], DType::F32);
    let before = scalar.node_count();
    assert!(
        scalar
            .scatter_reduce_default(base, 0, index, src, ScatterReduceKind::Sum)
            .is_err()
    );
    assert_eq!(scalar.node_count(), before);
}

#[test]
fn scatter_reduce_preflights_malformed_and_late_overflow_atomically() {
    let mut malformed = Graph::new();
    let base = malformed.input_dtype("base", [2, 3], DType::F32);
    let index = malformed.input_dtype("index", [2, 2], DType::F32);
    let src = malformed.input_dtype("src", [2, 2], DType::F32);
    let before = malformed.node_count();
    assert!(
        malformed
            .scatter_reduce_default(base, 1, index, src, ScatterReduceKind::Sum)
            .is_err()
    );
    assert_eq!(malformed.node_count(), before);

    let mut overflow = Graph::new();
    let base = overflow.input_dtype("base", [usize::MAX / 8, 3], DType::F16);
    let index = overflow.input_dtype("index", [usize::MAX / 8, 2], DType::I32);
    let src = overflow.input_dtype("src", [usize::MAX / 8, 2], DType::F16);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.scatter_reduce_default(base, 1, index, src, ScatterReduceKind::Sum),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn tinygrad_scatter_replacement_is_ordered_mask_fold_not_raw_scatter() {
    let mut graph = Graph::new();
    let base = graph.input_dtype("base", [2, 3], DType::F32);
    let index = graph.input_dtype("index", [2, 2], DType::I32);
    let src = graph.input_dtype("src", [2, 4], DType::F32);
    let output = graph
        .scatter_tinygrad_default(base, -1, index, ScatterSource::Tensor(src))
        .unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    // `_masked_merge` splits the synthetic lane into unit Shrinks, ORs the
    // masks, and lets the right-hand (later row-major) Select win.
    assert!(
        graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Op::Shrink { .. }))
            .count()
            >= 4
    );
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Logical {
            op: LogicalOp::Or,
            ..
        }
    )));
    assert!(
        graph
            .nodes
            .iter()
            .filter(|node| matches!(&node.op, Op::Select { .. }))
            .count()
            >= 2
    );
    assert!(
        !graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Scatter { .. }))
    );
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, base).is_ok());
    assert!(graph.grad(loss, src).is_ok());
    assert!(graph.grad(loss, index).is_err());
}

#[test]
fn tinygrad_scatter_scalar_modes_reuse_literal_scatter_reduce() {
    for mode in [
        ScatterMode::Replace,
        ScatterMode::Add,
        ScatterMode::Multiply,
    ] {
        for dtype in [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
        ] {
            let mut graph = Graph::new();
            let base = graph.input_dtype("base", [2, 3], dtype);
            let index = graph.input_dtype("index", [1, 2], DType::I64);
            let output = graph
                .scatter_tinygrad(base, 1, index, ScatterSource::Scalar(Scalar::F(-0.0)), mode)
                .unwrap();
            assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
            assert!(
                !graph
                    .nodes
                    .iter()
                    .any(|node| matches!(&node.op, Op::Scatter { .. }))
            );
            assert!(
                graph
                    .nodes
                    .iter()
                    .filter_map(|node| match &node.op {
                        Op::Constant(data) => Some(data.len()),
                        _ => None,
                    })
                    .all(|length| length == 1)
            );
            if mode != ScatterMode::Replace {
                assert!(
                    graph
                        .nodes
                        .iter()
                        .any(|node| matches!(&node.op, Op::Reduce { .. }))
                );
            }
        }
    }
}

#[test]
fn tinygrad_scatter_preflights_live_reduce_shapes_and_late_fold_overflow() {
    let mut live_reduce = Graph::new();
    let base = live_reduce.input_dtype("base", [2, 3], DType::F32);
    let index = live_reduce.input_dtype("index", [2, 2], DType::I32);
    let src = live_reduce.input_dtype("src", [2, 2], DType::F32);
    let before = live_reduce.node_count();
    assert!(
        live_reduce
            .scatter_tinygrad(base, 1, index, ScatterSource::Tensor(src), ScatterMode::Add)
            .is_err()
    );
    assert_eq!(live_reduce.node_count(), before);

    let mut mismatch = Graph::new();
    let base = mismatch.input_dtype("base", [2, 3], DType::F32);
    let index = mismatch.input_dtype("index", [2, 2], DType::F32);
    let before = mismatch.node_count();
    assert!(
        mismatch
            .scatter_tinygrad_default(base, 1, index, ScatterSource::Scalar(Scalar::I(1)))
            .is_err()
    );
    assert_eq!(mismatch.node_count(), before);

    let mut overflow = Graph::new();
    let base = overflow.input_dtype("base", [usize::MAX / 8, 3], DType::F16);
    let index = overflow.input_dtype("index", [usize::MAX / 8, 2], DType::I32);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.scatter_tinygrad_default(base, 1, index, ScatterSource::Scalar(Scalar::I(1))),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn tinygrad_gather_is_source_one_hot_select_not_raw_gather() {
    let mut graph = Graph::new();
    let value = graph.input_dtype("value", [3, 4], DType::F16);
    // A smaller non-axis extent is cropped before the synthetic class axis.
    let index = graph.input_dtype("index", [2, 2], DType::I64);
    let output = graph.gather_tinygrad(value, -1, index).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 2]));
    assert_eq!(graph.dtype(output).unwrap(), DType::F16);
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Shrink { .. }))
    );
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Select { .. }))
    );
    assert!(graph.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Reduce {
            kind: ReduceKind::Sum,
            ..
        }
    )));
    assert!(
        !graph
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Gather { .. }))
    );
    assert!(
        graph
            .nodes
            .iter()
            .filter_map(|node| match &node.op {
                Op::Constant(data) => Some(data.len()),
                _ => None,
            })
            .all(|length| length == 1)
    );
    let loss = graph.sum_all(output).unwrap();
    assert!(graph.grad(loss, value).is_ok());
    assert!(graph.grad(loss, index).is_err());
}

#[test]
fn tinygrad_gather_admits_every_value_family_and_zero_domains() {
    for dtype in [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ] {
        let mut graph = Graph::new();
        let value = graph.input_dtype("value", [1, 2], dtype);
        let index = graph.input_dtype("index", [1, 1], DType::I32);
        let output = graph.gather_tinygrad(value, 1, index).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([1, 1]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let value = empty.input_dtype("value", [2, 0], DType::F32);
    let index = empty.input_dtype("index", [1, 0], DType::I32);
    let output = empty.gather_tinygrad(value, 1, index).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([1, 0]));
}

#[test]
fn tinygrad_gather_preflights_invalid_descriptors_atomically() {
    let mut noninteger = Graph::new();
    let value = noninteger.input_dtype("value", [2, 3], DType::F32);
    let index = noninteger.input_dtype("index", [2, 2], DType::F32);
    let before = noninteger.node_count();
    assert!(noninteger.gather_tinygrad(value, 1, index).is_err());
    assert_eq!(noninteger.node_count(), before);

    let mut extent = Graph::new();
    let value = extent.input_dtype("value", [2, 3], DType::F32);
    let index = extent.input_dtype("index", [3, 2], DType::I32);
    let before = extent.node_count();
    assert!(extent.gather_tinygrad(value, 1, index).is_err());
    assert_eq!(extent.node_count(), before);

    let mut overflow = Graph::new();
    let value = overflow.input_dtype("value", [usize::MAX / 8, 3], DType::F16);
    let index = overflow.input_dtype("index", [usize::MAX / 8, 2], DType::I32);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.gather_tinygrad(value, 1, index),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn tinygrad_matmul_wrappers_are_exact_typed_dot_shells() {
    let mut forward = Graph::new();
    let lhs = forward.input_dtype("lhs", [2, 3], DType::F16);
    let rhs = forward.input_dtype("rhs", [3, 4], DType::F16);
    let output = forward.matmul_tinygrad_default(lhs, rhs).unwrap();
    assert_eq!(forward.shape(output).unwrap(), &Shape::new([2, 4]));
    assert_eq!(forward.dtype(output).unwrap(), DType::F16);
    assert!(forward.nodes.iter().any(|node| matches!(&node.op,
        Op::Reduce { kind: ReduceKind::Sum, .. } if node.dtype == DType::F32)));
    assert!(
        !forward
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Matmul { .. }))
    );

    let mut reflected = Graph::new();
    let rhs = reflected.input_dtype("rhs", [3, 4], DType::I64);
    let lhs = reflected.input_dtype("lhs", [2, 3], DType::U64);
    let output = reflected.rmatmul_tinygrad_default(rhs, lhs).unwrap();
    assert_eq!(reflected.shape(output).unwrap(), &Shape::new([2, 4]));
    assert_eq!(reflected.dtype(output).unwrap(), DType::F32);
    assert!(
        !reflected
            .nodes
            .iter()
            .any(|node| matches!(&node.op, Op::Matmul { .. }))
    );
}

#[test]
fn tinygrad_matmul_wrappers_cover_rank_families_dtype_and_atomic_errors() {
    let mut vector = Graph::new();
    let lhs = vector.input_dtype("lhs", [0], DType::I8);
    let rhs = vector.input_dtype("rhs", [0], DType::I8);
    let output = vector.matmul_tinygrad_default(lhs, rhs).unwrap();
    assert_eq!(vector.shape(output).unwrap(), &Shape::new([]));

    let mut batch = Graph::new();
    let lhs = batch.input_dtype("lhs", [2, 3, 4], DType::BF16);
    let rhs = batch.input_dtype("rhs", [1, 4, 5], DType::BF16);
    let output = batch
        .matmul_tinygrad(lhs, rhs, false, Some(DType::F64))
        .unwrap();
    assert_eq!(batch.shape(output).unwrap(), &Shape::new([2, 3, 5]));
    assert_eq!(batch.dtype(output).unwrap(), DType::F64);
    let loss = batch.sum_all(output).unwrap();
    assert!(batch.grad(loss, lhs).is_ok());

    let mut scalar = Graph::new();
    let lhs = scalar.input_dtype("lhs", [], DType::F32);
    let rhs = scalar.input_dtype("rhs", [1], DType::F32);
    let before = scalar.node_count();
    assert!(scalar.matmul_tinygrad_default(lhs, rhs).is_err());
    assert_eq!(scalar.node_count(), before);

    let mut overflow = Graph::new();
    let lhs = overflow.input_dtype("lhs", [usize::MAX / 2, 2], DType::F64);
    let rhs = overflow.input_dtype("rhs", [2, 1], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.matmul_tinygrad_default(lhs, rhs),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn tinygrad_usum_and_uprod_are_ordered_receiver_selected_folds() {
    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [2], DType::F32);
    let before = empty.node_count();
    assert_eq!(empty.usum(input, &[]).unwrap(), input);
    assert_eq!(empty.uprod(input, &[]).unwrap(), input);
    assert_eq!(empty.node_count(), before);

    let mut numeric = Graph::new();
    let first = numeric.input_dtype_requires_grad("first", [2, 1], DType::F32, true);
    let second = numeric.input_dtype("second", [1, 3], DType::F32);
    let third = numeric.input_dtype("third", [2, 3], DType::F32);
    let single = numeric.usum(first, &[second]).unwrap();
    assert!(
        matches!(numeric.op(single).unwrap(), Op::Binary { op: BinaryOp::Add, lhs, rhs }
        if *lhs == first && *rhs == second)
    );
    let sum = numeric.usum(first, &[second, third]).unwrap();
    let Op::Binary {
        op: BinaryOp::Add,
        lhs: sum_prefix,
        rhs,
    } = numeric.op(sum).unwrap()
    else {
        panic!("usum must finish with its final source-order Add");
    };
    assert_eq!(*rhs, third);
    assert!(
        matches!(numeric.op(*sum_prefix).unwrap(), Op::Binary { op: BinaryOp::Add, lhs, rhs }
        if *lhs == first && *rhs == second)
    );
    assert_eq!(numeric.shape(sum).unwrap(), &Shape::new([2, 3]));
    let sum_loss = numeric.sum_all(sum).unwrap();
    assert!(numeric.grad(sum_loss, first).is_ok());

    let product = numeric.uprod(first, &[second, third]).unwrap();
    let Op::Binary {
        op: BinaryOp::Mul,
        lhs: product_prefix,
        rhs,
    } = numeric.op(product).unwrap()
    else {
        panic!("uprod must finish with its final source-order Mul");
    };
    assert_eq!(*rhs, third);
    assert!(
        matches!(numeric.op(*product_prefix).unwrap(), Op::Binary { op: BinaryOp::Mul, lhs, rhs }
        if *lhs == first && *rhs == second)
    );
    let product_loss = numeric.sum_all(product).unwrap();
    assert!(numeric.grad(product_loss, first).is_ok());

    let mut boolean = Graph::new();
    let first = boolean.input_dtype("first", [2], DType::Bool);
    let second = boolean.input_dtype("second", [1], DType::Bool);
    let third = boolean.input_dtype("third", [2], DType::Bool);
    let sum = boolean.usum(first, &[second, third]).unwrap();
    assert!(
        matches!(boolean.op(sum).unwrap(), Op::Binary { op: BinaryOp::BitOr, rhs, .. } if *rhs == third)
    );
    let product = boolean.uprod(first, &[second, third]).unwrap();
    assert!(
        matches!(boolean.op(product).unwrap(), Op::Binary { op: BinaryOp::BitAnd, rhs, .. } if *rhs == third)
    );
}

#[test]
fn tinygrad_usum_and_uprod_preserve_source_lub_and_are_atomic() {
    let mut promoted = Graph::new();
    let signed = promoted.input_dtype("signed", [], DType::I64);
    let unsigned = promoted.input_dtype("unsigned", [2], DType::U64);
    let sum = promoted.usum(signed, &[unsigned]).unwrap();
    let product = promoted.uprod(signed, &[unsigned]).unwrap();
    assert_eq!(promoted.dtype(sum).unwrap(), DType::F32);
    assert_eq!(promoted.dtype(product).unwrap(), DType::F32);
    assert_eq!(promoted.shape(sum).unwrap(), &Shape::new([2]));
    assert!(promoted.nodes.iter().any(|node| matches!(
        &node.op,
        Op::Cast {
            dtype: DType::F32,
            ..
        }
    )));

    let mut unknown = Graph::new();
    let input = unknown.input_dtype("input", [2], DType::F32);
    let valid = unknown.input_dtype("valid", [2], DType::F32);
    let before = unknown.node_count();
    assert!(matches!(
        unknown.usum(input, &[valid, NodeId::from_index(usize::MAX)]),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(unknown.node_count(), before);

    let mut mismatch = Graph::new();
    let input = mismatch.input_dtype("input", [2, 2], DType::F32);
    let valid = mismatch.input_dtype("valid", [2, 2], DType::F32);
    let late = mismatch.input_dtype("late", [3], DType::F32);
    let before = mismatch.node_count();
    assert!(mismatch.uprod(input, &[valid, late]).is_err());
    assert_eq!(mismatch.node_count(), before);

    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX / 8, 1], DType::F64);
    let valid = overflow.input_dtype("valid", [1], DType::F64);
    let late = overflow.input_dtype("late", [1, 2], DType::F64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.usum(input, &[valid, late]),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);
}

#[test]
fn live_threefry_matches_source_vectors_and_captured_graph_structure() {
    let mut graph = Graph::new();
    let counter = graph.input_dtype("counter", [10], DType::U64);
    let key = graph.input_dtype("key", [], DType::U64);
    let output = graph.threefry(counter, key).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([10]));
    assert_eq!(graph.dtype(output).unwrap(), DType::U64);
    assert!(!graph.requires_grad(output).unwrap());
    assert!(matches!(
        graph.op(output).unwrap(),
        Op::Threefry { counter: got_counter, key: got_key }
            if *got_counter == counter && *got_key == key
    ));
    assert_eq!(graph.node_count(), 3);
    assert!(matches!(
        graph.grad(output, counter),
        Err(Error::NonDifferentiableTarget(node)) if node == output
    ));

    let packed_counters = (0_u32..10)
        .map(|index| (u64::from(index + 10) << 32) | u64::from(index))
        .collect::<Vec<_>>();
    let packed_key = u64::from(1337_u32) << 32;
    let counter_data = TensorData::from_storage([10], Storage::U64(packed_counters)).unwrap();
    let key_data = TensorData::from_storage([], Storage::U64(vec![packed_key])).unwrap();
    let result = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                ("counter".into(), counter_data.clone()),
                ("key".into(), key_data.clone()),
            ]),
        )
        .unwrap();
    let Storage::U64(packed) = result.storage() else {
        panic!("threefry output storage")
    };
    assert_eq!(
        packed,
        &[
            0xd342_182a_846d_667f,
            0x559e_be96_686f_0b31,
            0x8155_69fc_26f7_5b74,
            0x9930_336f_7546_32c9,
            0x8e49_076a_5329_2542,
            0xdb3f_7af6_e4e8_37a8,
            0xad8c_faff_80b5_0445,
            0x1809_0571_23f8_ce0b,
            0x98a3_a5c6_c5db_260e,
            0x64df_5db2_c880_8773,
        ]
    );
    let lows = packed.iter().map(|value| *value as u32);
    let highs = packed.iter().map(|value| (*value >> 32) as u32);
    assert_eq!(
        lows.chain(highs).collect::<Vec<_>>(),
        [
            2221762175, 1752107825, 653745012, 1967534793, 1395205442, 3840423848, 2159346757,
            603508235, 3319473678, 3363866483, 3544324138, 1436466838, 2169858556, 2570072943,
            2387150698, 3678370550, 2911697663, 403244401, 2560861638, 1692360114,
        ]
    );

    let scheduled = crate::schedule(&graph, output).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let item = &scheduled.items[0];
    assert!(item.boundary.is_none());
    let crate::Operation::Threefry(plan) = item.kernel.operation() else {
        panic!("live Threefry must retain its typed schedule plan")
    };
    assert_eq!(
        (plan.counter, plan.key, plan.output),
        (counter, key, output)
    );
    assert_eq!(
        item.input_bindings
            .iter()
            .map(|binding| (binding.input_node, binding.abi_index))
            .collect::<Vec<_>>(),
        vec![(counter, 0), (key, 1)]
    );
    let mut tampered = scheduled.clone();
    tampered.items[0].input_bindings[0].desc.shape = Shape::from([5, 2]);
    tampered.items[0].inputs[0].shape = Shape::from([5, 2]);
    assert!(tampered.validate().is_err());
    let kernel = &item.kernel;
    let c11 = crate::CpuJit::render(kernel).unwrap();
    assert!(c11.source.contains("rustgrad-c11-live-threefry-v1"));
    assert!(c11.source.contains("rg_x1<<13"));
    assert_eq!(c11.abi.buffers.len(), 3);
    let malformed_dtype = crate::UOp::from_operation(
        crate::Operation::Threefry(plan.clone()),
        Some(crate::UType::scalar(DType::F32)),
        vec![],
    );
    let stray_source = crate::UOp::scalar_constant(DType::U64, 0, crate::UType::scalar(DType::U64));
    let malformed_sources = crate::UOp::from_operation(
        crate::Operation::Threefry(plan.clone()),
        Some(crate::UType::scalar(DType::U64)),
        vec![stray_source],
    );
    for malformed in [&malformed_dtype, &malformed_sources] {
        assert!(crate::CpuJit::render(malformed).is_err());
        assert!(crate::CpuJit::render_vectorized(malformed).is_err());
    }
    let ptx = crate::ptx::PtxRenderer::new(80)
        .unwrap()
        .render(kernel)
        .unwrap();
    assert!(
        ptx.source
            .contains(crate::ptx::PTX_THREEFRY_RENDERER_VERSION)
    );
    assert!(ptx.source.contains("xor.b32 %r22"));
    assert!(ptx.source.contains("ld.param.u64 %rd3, [p2]"));
    assert!(ptx.source.contains("add.u64 %rd25, %rd3, %rd25"));
    assert!(!ptx.source.contains("cvt.u64.u32 %rd3,"));
    assert_eq!(ptx.buffers.len(), 3);
    let metal = crate::runtime::metal::MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 20,
            unified_memory: true,
            family: "ThreefryBoundary".into(),
        },
    )
    .unwrap();
    let metal = metal.render(kernel).unwrap();
    assert_eq!(metal.buffers.len(), 3);
    let opencl = crate::runtime::opencl::OpenClRenderer::with_capabilities(
        8,
        crate::runtime::opencl::OpenClCapabilities::FULL,
    )
    .unwrap()
    .render(kernel)
    .unwrap();
    assert_eq!(opencl.buffers.len(), 3);
    let wgsl = crate::runtime::webgpu::WgslRenderer::new(
        8,
        crate::runtime::webgpu::WebGpuCapabilities {
            max_buffer_size: 1 << 20,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 65_535,
            timestamp_query: false,
            shader_f16: false,
        },
    )
    .unwrap();
    let wgsl = wgsl.render(kernel).unwrap();
    assert_eq!(wgsl.buffers.len(), 3);
    let captured = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    let encoded = captured.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&encoded).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), encoded);
    let replay_inputs =
        BTreeMap::from([("counter".into(), counter_data), ("key".into(), key_data)]);
    let executor = crate::CapturedReplayExecutor::default();
    let interpreted = executor
        .replay(
            &decoded,
            &replay_inputs,
            crate::CapturedReplayOptions::default(),
        )
        .unwrap();
    let native = executor
        .replay(
            &decoded,
            &replay_inputs,
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(interpreted.outputs[0].storage(), result.storage());
    assert_eq!(native.outputs[0].storage(), result.storage());
    assert!(!native.trace.items.is_empty());
    assert!(
        native
            .trace
            .items
            .iter()
            .all(|item| item.backend == crate::ItemBackend::NativeJit)
    );
}

#[test]
fn live_threefry_broadcasts_scalar_empty_and_rectangular_u64_operands() {
    let mut graph = Graph::new();
    let counter = graph.input_dtype("counter", [2, 1], DType::U64);
    let key = graph.input_dtype("key", [1, 3], DType::U64);
    let output = graph.threefry(counter, key).unwrap();
    assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 3]));
    let counters = [[1_u32, 7_u32], [u32::MAX, 13_u32]];
    let keys = [[0_u32, 1337_u32], [5_u32, 9_u32], [u32::MAX, 1_u32]];
    let pack = |words: [u32; 2]| (u64::from(words[1]) << 32) | u64::from(words[0]);
    let realized = CpuBackend
        .execute(
            &graph,
            output,
            &HashMap::from([
                (
                    "counter".into(),
                    TensorData::from_storage(
                        [2, 1],
                        Storage::U64(counters.into_iter().map(pack).collect()),
                    )
                    .unwrap(),
                ),
                (
                    "key".into(),
                    TensorData::from_storage(
                        [1, 3],
                        Storage::U64(keys.into_iter().map(pack).collect()),
                    )
                    .unwrap(),
                ),
            ]),
        )
        .unwrap();
    let expected = counters
        .into_iter()
        .flat_map(|counter| {
            keys.into_iter()
                .map(move |key| pack(crate::random::threefry2x32(key, counter)))
        })
        .collect::<Vec<_>>();
    assert_eq!(realized.storage(), &Storage::U64(expected));

    let mut scalar = Graph::new();
    let counter = scalar.input_dtype("counter", [], DType::U64);
    let key = scalar.input_dtype("key", [], DType::U64);
    let output = scalar.threefry(counter, key).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));

    let mut shared = Graph::new();
    let words = shared.input_dtype("words", [4], DType::U64);
    let output = shared.threefry(words, words).unwrap();
    let scheduled = crate::schedule(&shared, output).unwrap();
    assert_eq!(scheduled.items[0].input_bindings.len(), 1);
    assert_eq!(scheduled.items[0].input_bindings[0].input_node, words);
    assert_eq!(
        crate::CpuJit::render(&scheduled.items[0].kernel)
            .unwrap()
            .abi
            .buffers
            .len(),
        2
    );
    let aliased_ptx = crate::ptx::PtxRenderer::new(80)
        .unwrap()
        .render(&scheduled.items[0].kernel)
        .unwrap();
    assert_eq!(aliased_ptx.buffers.len(), 2);
    assert!(aliased_ptx.source.contains("ld.param.u64 %rd2, [p1]"));
    assert!(aliased_ptx.source.contains("add.u64 %rd25, %rd2, %rd25"));

    let mut empty = Graph::new();
    let counter = empty.input_dtype("counter", [0, 3], DType::U64);
    let key = empty.input_dtype("key", [1, 3], DType::U64);
    let output = empty.threefry(counter, key).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0, 3]));
    let scheduled = crate::schedule(&empty, output).unwrap();
    let c11 = crate::CpuJit::render(&scheduled.items[0].kernel).unwrap();
    let ptx = crate::ptx::PtxRenderer::new(80)
        .unwrap()
        .render(&scheduled.items[0].kernel)
        .unwrap();
    assert!(c11.source.contains("rg_i<0u"));
    assert_eq!(ptx.extent, 0);
}

#[test]
fn live_threefry_captures_computed_dependencies_and_deduplicates_shared_producers() {
    let mut graph = Graph::new();
    let counter_left = graph.input_dtype("counter_left", [2, 1], DType::U64);
    let counter_right = graph.input_dtype("counter_right", [2, 1], DType::U64);
    let key_left = graph.input_dtype("key_left", [1, 3], DType::U64);
    let key_right = graph.input_dtype("key_right", [1, 3], DType::U64);
    let counter = graph.add(counter_left, counter_right).unwrap();
    let key = graph.add(key_left, key_right).unwrap();
    let output = graph.threefry(counter, key).unwrap();
    let scheduled = crate::schedule(&graph, output).unwrap();
    let producer_ids = [counter, key].map(|node| {
        scheduled
            .items
            .iter()
            .find(|item| item.primary_output().id == node.index() as u64)
            .unwrap()
            .id
    });
    let threefry = scheduled
        .items
        .iter()
        .find(|item| item.primary_output().id == output.index() as u64)
        .unwrap();
    assert_eq!(threefry.dependencies, producer_ids);
    assert!(producer_ids.into_iter().all(|id| id < threefry.id));

    let inputs = HashMap::from([
        (
            "counter_left".into(),
            TensorData::from_storage(
                [2, 1],
                Storage::U64(vec![0x0000_000a_0000_0000, 0x0000_000b_0000_0001]),
            )
            .unwrap(),
        ),
        (
            "counter_right".into(),
            TensorData::from_storage([2, 1], Storage::U64(vec![1, 2])).unwrap(),
        ),
        (
            "key_left".into(),
            TensorData::from_storage([1, 3], Storage::U64(vec![0, 1, 2])).unwrap(),
        ),
        (
            "key_right".into(),
            TensorData::from_storage([1, 3], Storage::U64(vec![0x0000_0539_0000_0000; 3])).unwrap(),
        ),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    let captured = crate::CapturedSchedule::capture(&graph, &scheduled, &[output]).unwrap();
    assert_eq!(captured.items.last().unwrap().dependencies, producer_ids);
    let replay_inputs = inputs.into_iter().collect::<BTreeMap<_, _>>();
    let executor = crate::CapturedReplayExecutor::default();
    let interpreted = executor
        .replay(
            &captured,
            &replay_inputs,
            crate::CapturedReplayOptions::default(),
        )
        .unwrap();
    let native = executor
        .replay(
            &captured,
            &replay_inputs,
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    assert_eq!(interpreted.outputs[0].storage(), expected.storage());
    assert_eq!(native.outputs[0].storage(), expected.storage());

    let mut shared = Graph::new();
    let left = shared.input_dtype("left", [2], DType::U64);
    let right = shared.input_dtype("right", [2], DType::U64);
    let computed = shared.add(left, right).unwrap();
    let shared_output = shared.threefry(computed, computed).unwrap();
    let shared_schedule = crate::schedule(&shared, shared_output).unwrap();
    let shared_item = shared_schedule.items.last().unwrap();
    assert_eq!(shared_item.dependencies.len(), 1);
    assert_eq!(shared_item.input_bindings.len(), 1);
    let crate::Operation::Threefry(plan) = shared_item.kernel.operation() else {
        unreachable!()
    };
    assert_eq!(plan.buffer_operands().count(), 2);
}

#[test]
fn live_random_bits_captures_and_replays_source_composition_exactly() {
    let mut graph = Graph::new();
    let key = graph.input_dtype("key", [2], DType::U32);
    let counter = graph.input_dtype("counter", [2], DType::U32);
    let empty = graph.random_bits(key, counter, 0).unwrap();
    let negative = graph.random_bits(key, counter, -11).unwrap();
    let output = graph.random_bits(key, counter, 7).unwrap();
    let key_words = [0x0123_4567, 0x89ab_cdef];
    let counter_words = [u32::MAX - 2, 9];
    let inputs = HashMap::from([
        (
            "key".into(),
            TensorData::from_storage([2], Storage::U32(key_words.to_vec())).unwrap(),
        ),
        (
            "counter".into(),
            TensorData::from_storage([2], Storage::U32(counter_words.to_vec())).unwrap(),
        ),
    ]);
    let expected = CpuBackend.execute(&graph, output, &inputs).unwrap();
    assert_eq!(
        expected.storage(),
        &Storage::U32(crate::random::words(key_words, counter_words, 7))
    );

    let requested = [output, empty, negative, output];
    let scheduled = crate::schedule_many(&graph, &requested).unwrap();
    assert!(!scheduled.items.is_empty());
    for requested in [empty, negative] {
        let passthrough = scheduled
            .requested_passthroughs
            .iter()
            .find(|passthrough| passthrough.requested == requested)
            .expect("nonpositive random_bits is a source-owned empty view");
        assert_eq!(passthrough.source, counter);
        // The physical descriptor remains the exact U32 [2] counter. Only
        // its authenticated affine view has the requested empty geometry.
        assert_eq!(passthrough.desc.id, counter.index() as u64);
        assert_eq!(passthrough.desc.shape, Shape::new([2]));
        assert!(passthrough.desc.read_only);
        let view = passthrough.desc.view.as_ref().expect("empty counter view");
        assert_eq!(view.source_shape, Shape::new([2]));
        assert_eq!(view.logical_shape, Shape::new([0]));
        let normalized = view.normalized_read().unwrap();
        assert_eq!(normalized.offset, 0);
        assert!(
            normalized
                .axes
                .iter()
                .all(|axis| axis.stride == 0 && !axis.reversed)
        );
    }
    assert!(
        scheduled
            .items
            .iter()
            .any(|item| matches!(item.kernel.operation(), crate::Operation::Threefry(_)))
    );
    for item in &scheduled.items {
        crate::CpuJit::render(&item.kernel).unwrap();
        crate::ptx::PtxRenderer::new(80)
            .unwrap()
            .render(&item.kernel)
            .unwrap();
    }
    let metal = crate::runtime::metal::MetalRenderer::new(
        8,
        crate::runtime::metal::MetalCapabilities {
            max_buffer_length: 1 << 20,
            unified_memory: true,
            family: "RandomBitsBoundary".into(),
        },
    )
    .unwrap();
    assert!(
        scheduled
            .items
            .iter()
            .any(|item| metal.render(&item.kernel).is_err())
    );
    let wgsl = crate::runtime::webgpu::WgslRenderer::new(
        8,
        crate::runtime::webgpu::WebGpuCapabilities {
            max_buffer_size: 1 << 20,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroups_per_dimension: 65_535,
            timestamp_query: false,
            shader_f16: false,
        },
    )
    .unwrap();
    assert!(
        scheduled
            .items
            .iter()
            .any(|item| wgsl.render(&item.kernel).is_err())
    );

    let captured = crate::CapturedSchedule::capture(&graph, &scheduled, &requested).unwrap();
    let encoded = captured.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&encoded).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), encoded);
    let replay_inputs = inputs.into_iter().collect::<BTreeMap<_, _>>();
    let executor = crate::CapturedReplayExecutor::default();
    let interpreted = executor
        .replay(
            &decoded,
            &replay_inputs,
            crate::CapturedReplayOptions::default(),
        )
        .unwrap();
    let native = executor
        .replay(
            &decoded,
            &replay_inputs,
            crate::CapturedReplayOptions {
                backend: crate::CapturedBackendPolicy::NativeJit { vectorized: false },
            },
        )
        .unwrap();
    for result in [&interpreted, &native] {
        assert_eq!(result.outputs.len(), 4);
        assert_eq!(result.outputs[0].storage(), expected.storage());
        assert_eq!(result.outputs[1].shape(), &Shape::new([0]));
        assert_eq!(result.outputs[1].storage(), &Storage::U32(Vec::new()));
        assert_eq!(result.outputs[2].shape(), &Shape::new([0]));
        assert_eq!(result.outputs[2].storage(), &Storage::U32(Vec::new()));
        assert_eq!(result.outputs[3].storage(), expected.storage());
    }
    assert!(
        native
            .trace
            .items
            .iter()
            .all(|item| item.backend == crate::ItemBackend::NativeJit)
    );
}

#[test]
fn live_threefry_rejects_invalid_descriptors_without_publication() {
    let mut unknown = Graph::new();
    let counter = unknown.input_dtype("counter", [2], DType::U64);
    let before = unknown.node_count();
    assert!(matches!(
        unknown.threefry(counter, NodeId(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(unknown.node_count(), before);

    let mut dtype = Graph::new();
    let counter = dtype.input_dtype("counter", [2], DType::U64);
    let key = dtype.input_dtype("key", [2], DType::I64);
    let before = dtype.node_count();
    assert!(matches!(
        dtype.threefry(counter, key),
        Err(Error::InvalidElementwiseDType {
            op: "threefry",
            actual: DType::I64
        })
    ));
    assert_eq!(dtype.node_count(), before);

    let mut mismatch = Graph::new();
    let counter = mismatch.input_dtype("counter", [2], DType::U64);
    let key = mismatch.input_dtype("key", [3], DType::U64);
    let before = mismatch.node_count();
    assert!(mismatch.threefry(counter, key).is_err());
    assert_eq!(mismatch.node_count(), before);

    let mut overflow = Graph::new();
    let counter = overflow.input_dtype("counter", [usize::MAX, 2], DType::U64);
    let key = overflow.input_dtype("key", [], DType::U64);
    let before = overflow.node_count();
    assert!(matches!(
        overflow.threefry(counter, key),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(overflow.node_count(), before);

    let mut output_overflow = Graph::new();
    let counter = output_overflow.input_dtype("counter", [usize::MAX / 16, 1], DType::U64);
    let key = output_overflow.input_dtype("key", [1, 32], DType::U64);
    let before = output_overflow.node_count();
    assert!(matches!(
        output_overflow.threefry(counter, key),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(output_overflow.node_count(), before);
}

#[test]
fn tinygrad_bitcast_preserves_raw_storage_and_rescales_only_the_final_axis() {
    let mut graph = Graph::new();
    let float = graph.input_dtype_requires_grad("float", [4], DType::F32, true);
    let bits = graph.bitcast(float, DType::U32).unwrap();
    assert_eq!(graph.shape(bits).unwrap(), &Shape::new([4]));
    assert_eq!(graph.dtype(bits).unwrap(), DType::U32);
    assert!(!graph.requires_grad(bits).unwrap());
    assert!(
        matches!(graph.op(bits).unwrap(), Op::Bitcast { input, dtype: DType::U32 } if *input == float)
    );
    let values = TensorData::from_storage(
        [4],
        Storage::F32(vec![
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7fc0_0123),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]),
    )
    .unwrap();
    let result = CpuBackend
        .execute(&graph, bits, &HashMap::from([("float".into(), values)]))
        .unwrap();
    assert_eq!(
        result.storage(),
        &Storage::U32(vec![0x8000_0000, 0x7fc0_0123, 0x7f80_0000, 0xff80_0000,])
    );

    let mut widening = Graph::new();
    let bytes = widening.input_dtype("bytes", [2, 8], DType::U8);
    let words = widening.bitcast(bytes, DType::U32).unwrap();
    assert_eq!(widening.shape(words).unwrap(), &Shape::new([2, 2]));
    let source = TensorData::from_storage([2, 8], Storage::U8((1..=16).collect())).unwrap();
    let result = CpuBackend
        .execute(&widening, words, &HashMap::from([("bytes".into(), source)]))
        .unwrap();
    assert_eq!(
        result.storage(),
        &Storage::U32(vec![0x0403_0201, 0x0807_0605, 0x0c0b_0a09, 0x100f_0e0d])
    );
    let scheduled = crate::schedule(&widening, words).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
        scheduled.items[0].kernel.operation()
    else {
        panic!("bitcast must schedule as a materializing movement kernel")
    };
    assert!(matches!(
        &plan.kind,
        crate::MovementKernelKind::Bitcast { .. }
    ));
    let captured = crate::CapturedSchedule::capture(&widening, &scheduled, &[words]).unwrap();
    let bytes = captured.to_bytes().unwrap();
    assert_eq!(
        crate::CapturedSchedule::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );

    let mut narrowing = Graph::new();
    let halves = narrowing.input_dtype("halves", [1, 2], DType::U16);
    let bytes = narrowing.bitcast(halves, DType::U8).unwrap();
    assert_eq!(narrowing.shape(bytes).unwrap(), &Shape::new([1, 4]));
    let source = TensorData::from_storage([1, 2], Storage::U16(vec![0x0201, 0x0403])).unwrap();
    let result = CpuBackend
        .execute(
            &narrowing,
            bytes,
            &HashMap::from([("halves".into(), source)]),
        )
        .unwrap();
    assert_eq!(result.storage(), &Storage::U8(vec![1, 2, 3, 4]));

    let mut booleans = Graph::new();
    let raw = booleans.input_dtype("raw", [4], DType::U8);
    let truth = booleans.bitcast(raw, DType::Bool).unwrap();
    let source = TensorData::from_storage([4], Storage::U8(vec![0, 2, 255, 1])).unwrap();
    let result = CpuBackend
        .execute(&booleans, truth, &HashMap::from([("raw".into(), source)]))
        .unwrap();
    assert_eq!(
        result.storage(),
        &Storage::Bool(vec![false, true, true, true])
    );

    let mut float8 = Graph::new();
    let raw = float8.input_dtype("raw", [4], DType::U8);
    let narrow = float8.bitcast(raw, DType::F8E4M3).unwrap();
    let source = TensorData::from_storage([4], Storage::U8(vec![0x00, 0x80, 0x7f, 0xff])).unwrap();
    let result = CpuBackend
        .execute(&float8, narrow, &HashMap::from([("raw".into(), source)]))
        .unwrap();
    let Storage::Float8(values) = result.storage() else {
        panic!("float8 bitcast must retain float8 storage")
    };
    assert_eq!(values.as_raw(), [0x00, 0x80, 0x7f, 0xff]);
    assert_eq!(values.format().dtype(), DType::F8E4M3);
}

#[test]
fn tinygrad_bitcast_covers_concrete_dtype_shapes_and_is_atomic() {
    for source_dtype in DType::ALL {
        for target_dtype in DType::ALL {
            let mut graph = Graph::new();
            let input = graph.input_dtype("input", [2, 8], source_dtype);
            let before = graph.node_count();
            let output = graph.bitcast(input, target_dtype).unwrap();
            if source_dtype == target_dtype {
                assert_eq!(output, input);
                assert_eq!(graph.node_count(), before);
            } else {
                assert_eq!(
                    graph.shape(output).unwrap(),
                    &Shape::new([2, 8 * source_dtype.itemsize() / target_dtype.itemsize()])
                );
                assert_eq!(graph.dtype(output).unwrap(), target_dtype);
                assert!(
                    matches!(graph.op(output).unwrap(), Op::Bitcast { input: source, dtype }
                    if *source == input && *dtype == target_dtype)
                );
            }
        }
    }

    let mut graph = Graph::new();
    let scalar = graph.input_dtype("scalar", [], DType::F32);
    assert_eq!(graph.bitcast(scalar, DType::F32).unwrap(), scalar);
    let empty = graph.input_dtype("empty", [3, 0], DType::F16);
    let empty_bits = graph.bitcast(empty, DType::U8).unwrap();
    assert_eq!(graph.shape(empty_bits).unwrap(), &Shape::new([3, 0]));

    let before = graph.node_count();
    assert!(matches!(
        graph.bitcast(NodeId::from_index(usize::MAX), DType::U8),
        Err(Error::UnknownNode(_))
    ));
    assert!(matches!(
        graph.bitcast(scalar, DType::U8),
        Err(Error::InvalidBitcast { .. })
    ));
    let odd = graph.input_dtype("odd", [3], DType::U8);
    let odd_before = graph.node_count();
    assert!(matches!(
        graph.bitcast(odd, DType::U16),
        Err(Error::InvalidBitcast { .. })
    ));
    assert_eq!(graph.node_count(), odd_before);

    let overflow = graph.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let overflow_before = graph.node_count();
    assert!(matches!(
        graph.bitcast(overflow, DType::U64),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(graph.node_count(), overflow_before);
    assert!(graph.node_count() >= before);
}

#[test]
fn tinygrad_contiguous_materializes_exact_values_and_preserves_buffer_identities() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("input", [2, 3], DType::F32, true);
    let before = graph.node_count();
    assert_eq!(graph.contiguous(input).unwrap(), input);
    assert_eq!(graph.node_count(), before);

    let reshaped = graph.reshape(input, [1, 2, 3]).unwrap();
    let before = graph.node_count();
    assert_eq!(graph.contiguous(reshaped).unwrap(), reshaped);
    assert_eq!(graph.node_count(), before);

    let transposed = graph.permute(input, [1, 0]).unwrap();
    let contiguous = graph.contiguous(transposed).unwrap();
    assert_eq!(graph.shape(contiguous).unwrap(), &Shape::new([3, 2]));
    assert_eq!(graph.dtype(contiguous).unwrap(), DType::F32);
    assert!(graph.requires_grad(contiguous).unwrap());
    assert!(
        matches!(graph.op(contiguous).unwrap(), Op::Contiguous { input: source } if *source == transposed)
    );
    let before = graph.node_count();
    assert_eq!(graph.contiguous(contiguous).unwrap(), contiguous);
    assert_eq!(graph.node_count(), before);

    let data = TensorData::from_storage(
        [2, 3],
        Storage::F32(vec![
            f32::from_bits(0x8000_0000),
            f32::from_bits(0x7fc0_0123),
            f32::INFINITY,
            f32::NEG_INFINITY,
            5.0,
            6.0,
        ]),
    )
    .unwrap();
    let result = CpuBackend
        .execute(&graph, contiguous, &HashMap::from([("input".into(), data)]))
        .unwrap();
    let Storage::F32(result) = result.storage() else {
        panic!("contiguous output storage")
    };
    assert_eq!(
        result
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            0x8000_0000,
            0xff80_0000,
            0x7fc0_0123,
            0x40a0_0000,
            0x7f80_0000,
            0x40c0_0000
        ]
    );

    let scheduled = crate::schedule(&graph, contiguous).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert!(scheduled.items[0].dependencies.is_empty());
    let crate::Operation::Movement(crate::MovementValue::Plan(plan)) =
        scheduled.items[0].kernel.operation()
    else {
        panic!("contiguous must schedule as a movement copy")
    };
    let crate::MovementKernelKind::AffineCopy {
        input: operand,
        view,
    } = &plan.kind
    else {
        panic!("contiguous view must schedule as one affine copy")
    };
    assert_eq!(operand.node, input);
    assert_eq!(operand.shape, Shape::new([2, 3]));
    assert_eq!(view.logical_shape, Shape::new([3, 2]));
    assert_eq!(view.strides, [1, 3]);
    let captured = crate::CapturedSchedule::capture(&graph, &scheduled, &[contiguous]).unwrap();
    let encoded = captured.to_bytes().unwrap();
    assert_eq!(
        crate::CapturedSchedule::from_bytes(&encoded)
            .unwrap()
            .to_bytes()
            .unwrap(),
        encoded
    );

    let mut float8 = Graph::new();
    let input = float8.input_dtype("input", [4], DType::F8E4M3);
    let detached = float8.detach(input).unwrap();
    let contiguous = float8.contiguous(detached).unwrap();
    let data = TensorData::from_storage(
        [4],
        Storage::Float8(Float8Storage::from_raw(
            Float8Format::E4M3,
            vec![0x00, 0x80, 0x7f, 0xff],
        )),
    )
    .unwrap();
    let result = CpuBackend
        .execute(
            &float8,
            contiguous,
            &HashMap::from([("input".into(), data)]),
        )
        .unwrap();
    let Storage::Float8(result) = result.storage() else {
        panic!("contiguous float8 storage")
    };
    assert_eq!(result.as_raw(), [0x00, 0x80, 0x7f, 0xff]);
}

#[test]
fn tinygrad_contiguous_backward_has_distinct_reverse_rule_and_atomic_admission() {
    let mut graph = Graph::new();
    let input = graph.input_dtype_requires_grad("input", [2, 1], DType::F32, true);
    let squared = graph.square(input).unwrap();
    let boundary = graph.contiguous_backward(squared).unwrap();
    assert!(
        matches!(graph.op(boundary).unwrap(), Op::ContiguousBackward { input: source } if *source == squared)
    );
    assert_eq!(graph.shape(boundary).unwrap(), &Shape::new([2, 1]));
    assert!(graph.requires_grad(boundary).unwrap());

    let seed_input = graph.input_dtype_requires_grad("seed", [1, 2], DType::F32, false);
    let seed = graph.permute(seed_input, [1, 0]).unwrap();
    let before = graph.node_count();
    let gradient = graph.grad_with(boundary, input, Some(seed), true).unwrap();
    assert_eq!(graph.shape(gradient).unwrap(), &Shape::new([2, 1]));
    let cotangent_copy = graph.nodes[before..]
        .iter()
        .position(|node| matches!(node.op, Op::Contiguous { input: source } if source == seed))
        .map(|position| NodeId::from_index(before + position))
        .expect("contiguous-backward cotangent copy");
    let scheduled = crate::schedule(&graph, cotangent_copy).unwrap();
    assert_eq!(scheduled.items.len(), 1);
    assert!(matches!(
        scheduled.items[0].kernel.operation(),
        crate::Operation::Movement(crate::MovementValue::Plan(plan))
            if matches!(
                &plan.kind,
                crate::MovementKernelKind::AffineCopy { input, .. }
                    if input.node == seed_input
            )
    ));

    let mut ordinary = Graph::new();
    let input = ordinary.input_dtype_requires_grad("input", [2], DType::F32, true);
    let squared = ordinary.square(input).unwrap();
    let contiguous = ordinary.contiguous(squared).unwrap();
    let seed = ordinary.input_dtype_requires_grad("seed", [2], DType::F32, false);
    let before = ordinary.node_count();
    ordinary
        .grad_with(contiguous, input, Some(seed), true)
        .unwrap();
    assert!(
        !ordinary.nodes[before..]
            .iter()
            .any(|node| matches!(node.op, Op::Contiguous { .. }))
    );

    for dtype in DType::ALL {
        let mut graph = Graph::new();
        let input = graph.input_dtype("input", [2, 0, 3], dtype);
        let computed = graph.cast(input, dtype).unwrap();
        let output = graph.contiguous(computed).unwrap();
        assert_eq!(graph.shape(output).unwrap(), &Shape::new([2, 0, 3]));
        assert_eq!(graph.dtype(output).unwrap(), dtype);
    }

    let mut malformed = Graph::new();
    let before = malformed.node_count();
    assert!(matches!(
        malformed.contiguous(NodeId::from_index(usize::MAX)),
        Err(Error::UnknownNode(_))
    ));
    assert_eq!(malformed.node_count(), before);
    let overflow = malformed.input_dtype("overflow", [usize::MAX, 2], DType::F64);
    let before = malformed.node_count();
    assert!(matches!(
        malformed.contiguous(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
    assert!(matches!(
        malformed.contiguous_backward(overflow),
        Err(Error::ShapeOverflow(_))
    ));
    assert_eq!(malformed.node_count(), before);
}
