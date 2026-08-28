use super::*;
use crate::{Backend, CpuBackend, Error, Scalar};
use std::collections::HashMap;

fn bool_data(shape: impl Into<Shape>, values: impl IntoIterator<Item = bool>) -> TensorData {
    TensorData::from_scalars(shape, DType::Bool, values.into_iter().map(Scalar::Bool)).unwrap()
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
    assert!(matches!(graph.op(*selected_true).unwrap(), Op::Cast { input, dtype }
        if *input == on_true && *dtype == DType::F32));
    assert!(matches!(graph.op(*selected_false).unwrap(), Op::Cast { input, dtype }
        if *input == on_false && *dtype == DType::F32));
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
                [Scalar::F(-0.0), Scalar::F(f64::NAN), Scalar::F(f64::INFINITY)],
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
        CpuBackend.execute(&graph, true_gradient, &bindings).unwrap().to_vec_f64(),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, false_gradient, &bindings).unwrap().to_vec_f64(),
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
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
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
        op: CompareOp::Eq,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Eq comparison");
    };
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
            ]),
        )
        .unwrap();
    assert!(values.scalar_at(0).as_bool());
    assert!(!values.scalar_at(1).as_bool());

    let mut predicate = Graph::new();
    let input = predicate.input_dtype("input", [], DType::F32);
    let rhs = predicate.constant(TensorData::scalar_with_dtype(Scalar::F(1.0), DType::F32));
    let output = predicate.eq(input, rhs).unwrap();
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
                [Scalar::F(f64::NAN), Scalar::F(-0.0), Scalar::F(f64::NEG_INFINITY)],
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
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
                [Scalar::F(f64::NAN), Scalar::F(-0.0), Scalar::F(f64::INFINITY)],
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
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    let Op::Logical {
        op: LogicalOp::Not,
        lhs: greater,
        rhs: None,
    } = graph.op(output).unwrap()
    else {
        panic!("expected logical-not");
    };
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_rhs,
        rhs: compared_lhs,
    } = graph.op(*greater).unwrap()
    else {
        panic!("expected reversed Lt comparison");
    };
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
                [Scalar::F(f64::NAN), Scalar::F(-0.0), Scalar::F(f64::INFINITY)],
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
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    let Op::Logical {
        op: LogicalOp::Not,
        lhs: less,
        rhs: None,
    } = graph.op(output).unwrap()
    else {
        panic!("expected logical-not");
    };
    let Op::Compare {
        op: CompareOp::Lt,
        lhs: compared_lhs,
        rhs: compared_rhs,
    } = graph.op(*less).unwrap()
    else {
        panic!("expected Lt comparison");
    };
    assert!(matches!(graph.op(*compared_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*compared_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
    assert!(matches!(predicate.grad(output, input), Err(Error::NoGradient(_))));
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
    assert!(matches!(graph.op(*added_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*added_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
                [Scalar::F(-0.0), Scalar::F(f64::NAN), Scalar::F(f64::INFINITY)],
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
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
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
    assert!(matches!(narrow.op(output).unwrap(), Op::Binary { op: BinaryOp::Add, .. }));
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
        op: BinaryOp::Sub,
        lhs: subtracted_lhs,
        rhs: subtracted_rhs,
    } = graph.op(output).unwrap()
    else {
        panic!("expected Sub");
    };
    assert!(matches!(graph.op(*subtracted_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*subtracted_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
    assert!(matches!(booleans.op(*added_rhs).unwrap(), Op::Logical { op: LogicalOp::Not, lhs: input, rhs: None }
        if *input == rhs));
    let values = CpuBackend
        .execute(
            &booleans,
            output,
            &HashMap::from([
                (
                    "lhs".into(),
                    bool_data([4], [false, false, true, true]),
                ),
                (
                    "rhs".into(),
                    bool_data([4], [false, true, false, true]),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(
        (0..4).map(|index| values.scalar_at(index).as_bool()).collect::<Vec<_>>(),
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
                [Scalar::F(-0.0), Scalar::F(f64::NAN), Scalar::F(f64::INFINITY)],
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
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
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
    assert!(matches!(graph.op(*multiplied_lhs).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
    assert!(matches!(graph.op(*multiplied_rhs).unwrap(), Op::Cast { input, dtype }
        if *input == rhs && *dtype == DType::F32));
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
                (
                    "lhs".into(),
                    bool_data([4], [false, false, true, true]),
                ),
                (
                    "rhs".into(),
                    bool_data([4], [false, true, false, true]),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(
        (0..4).map(|index| values.scalar_at(index).as_bool()).collect::<Vec<_>>(),
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
            TensorData::from_scalars([2, 1], DType::F64, [Scalar::F(2.0), Scalar::F(3.0)])
                .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![5.0, 5.0, 5.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
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
    assert!(matches!(graph.op(*dividend).unwrap(), Op::Cast { input, dtype }
        if *input == lhs && *dtype == DType::F32));
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
            TensorData::from_scalars([2, 1], DType::F64, [Scalar::F(2.0), Scalar::F(4.0)])
                .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![0.75, 0.75, 0.75]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![-3.5, -0.875]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    assert_eq!(narrow.dtype(narrow.div(lhs, rhs).unwrap()).unwrap(), DType::F16);
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
    assert!(matches!(graph.op(*condition).unwrap(), Op::Compare { op: CompareOp::Eq, .. }));
    assert!(matches!(graph.op(*on_true).unwrap(), Op::Constant(_)));
    assert!(matches!(graph.op(*on_false).unwrap(), Op::Binary { op: BinaryOp::TruncDiv, .. }));
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
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
        vec![-1.0, 0.0]
    );
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, lhs).unwrap();
    assert_eq!(
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
        vec![0.0, 0.0]
    );

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
    assert!(matches!(wide.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input }
        if matches!(wide.op(*input).unwrap(), Op::Binary { op: BinaryOp::Mul, .. })));
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
    assert_eq!(
        narrow.dtype(narrow.trunc_div(lhs, rhs).unwrap()).unwrap(),
        DType::F16
    );
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
    assert!(matches!(graph.op(*condition).unwrap(), Op::Compare { op: CompareOp::Eq, .. }));
    assert!(matches!(graph.op(*on_true).unwrap(), Op::Constant(_)));
    assert!(matches!(graph.op(*on_false).unwrap(), Op::Select { .. }));
    let bindings = HashMap::from([
        (
            "lhs".into(),
            TensorData::from_scalars(
                [5],
                DType::I16,
                [Scalar::I(-3), Scalar::I(3), Scalar::I(-3), Scalar::I(3), Scalar::I(5)],
            )
            .unwrap(),
        ),
        (
            "rhs".into(),
            TensorData::from_scalars(
                [5],
                DType::U16,
                [Scalar::U(2), Scalar::U(2), Scalar::U(2), Scalar::U(2), Scalar::U(0)],
            )
            .unwrap(),
        ),
    ]);
    assert_eq!(
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
        vec![-2.0, 1.0, -2.0, 1.0, 0.0]
    );
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, lhs).unwrap();
    assert_eq!(
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
        vec![0.0; 5]
    );

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
    assert!(matches!(wide.op(output).unwrap(), Op::Unary { op: UnaryOp::Floor, input }
        if matches!(wide.op(*input).unwrap(), Op::Binary { op: BinaryOp::Mul, .. })));
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
    assert_eq!(
        narrow.dtype(narrow.floor_div(lhs, rhs).unwrap()).unwrap(),
        DType::F16
    );
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    assert_eq!(
        narrow.dtype(narrow.floor_div(lhs, rhs).unwrap()).unwrap(),
        DType::BF16
    );
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
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Sub, .. }));
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
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
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
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![-1.0]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    assert_eq!(narrow.dtype(narrow.modulo(lhs, rhs).unwrap()).unwrap(), DType::F16);
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    assert_eq!(narrow.dtype(narrow.modulo(lhs, rhs).unwrap()).unwrap(), DType::BF16);
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
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Sub, .. }));
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
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
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
        CpuBackend.execute(&graph, lhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![1.0, 1.0]
    );
    assert_eq!(
        CpuBackend.execute(&graph, rhs_gradient, &bindings).unwrap().to_vec_f64(),
        vec![-1.0]
    );

    let mut narrow = Graph::new();
    let lhs = narrow.input_dtype("lhs", [], DType::F16);
    let rhs = narrow.input_dtype("rhs", [], DType::F16);
    assert_eq!(narrow.dtype(narrow.fmod(lhs, rhs).unwrap()).unwrap(), DType::F16);
    let lhs = narrow.input_dtype("bf16_lhs", [], DType::BF16);
    let rhs = narrow.input_dtype("bf16_rhs", [], DType::BF16);
    assert_eq!(narrow.dtype(narrow.fmod(lhs, rhs).unwrap()).unwrap(), DType::BF16);
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
            [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(f64::NAN), Scalar::F(1.0)],
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
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
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
        CpuBackend.execute(&ties, lhs_gradient, &equal).unwrap().to_vec_f64(),
        vec![0.5; 2]
    );
    assert_eq!(
        CpuBackend.execute(&ties, rhs_gradient, &equal).unwrap().to_vec_f64(),
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
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Maximum, .. }));
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
        CpuBackend.execute(&graph, scalar_positive, &bindings)
            .unwrap()
            .storage(),
        &crate::Storage::Bool(vec![true])
    );

    assert!(matches!(graph.grad(positive, input), Err(Error::NoGradient(_))));

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F32);
    let output = empty.isinf_with_signs(input, false, true).unwrap();
    assert_eq!(empty.shape(output).unwrap(), &Shape::new([0]));
    assert_eq!(
        CpuBackend
            .execute(
                &empty,
                output,
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
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
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sign, input: signed }
        if *signed == input));
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
            TensorData::from_scalars(
                [3],
                DType::I32,
                [Scalar::I(-3), Scalar::I(0), Scalar::I(4)],
            )
            .unwrap(),
        ),
    ]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64();
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
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
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
    assert_eq!(discrete.dtype(discrete.sign(f16).unwrap()).unwrap(), DType::F16);
    assert_eq!(discrete.dtype(discrete.sign(bf16).unwrap()).unwrap(), DType::BF16);
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
        CpuBackend.execute(&discrete, boolean_output, &bindings).unwrap().storage(),
        &crate::Storage::Bool(vec![false, true])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, signed_output, &bindings).unwrap().storage(),
        &crate::Storage::I64(vec![-1, 1])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, unsigned_output, &bindings).unwrap().storage(),
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.sign(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn reciprocal_preserves_tinygrad_alu_dtype_special_and_vjp_contract() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [6], DType::F64);
    let output = graph.reciprocal(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input: reciprocal }
        if *reciprocal == input));
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
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
    let gradient = differentiable.grad(differentiable.sum_all(output).unwrap(), input).unwrap();
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
    assert!(matches!(nonfloat.op(f16_output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input }
        if *input == f16));
    assert!(matches!(nonfloat.op(bf16_output).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, input }
        if *input == bf16));
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
    let boolean_values = CpuBackend.execute(&nonfloat, boolean_output, &bindings).unwrap();
    assert_eq!(boolean_values.scalar_at(0).as_f64(), f64::INFINITY);
    assert_eq!(boolean_values.scalar_at(1).as_f64(), 1.0);
    assert_eq!(
        CpuBackend.execute(&nonfloat, signed_output, &bindings).unwrap().to_vec_f64(),
        vec![-0.5]
    );
    assert_eq!(
        CpuBackend.execute(&nonfloat, unsigned_output, &bindings).unwrap().to_vec_f64(),
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.reciprocal(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    let mut differentiable = Graph::new();
    let input = differentiable.input_dtype("input", [2], DType::F64);
    let output = differentiable.exp(input).unwrap();
    let gradient = differentiable
        .grad(differentiable.sum_all(output).unwrap(), input)
        .unwrap();
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
    assert!(matches!(promoted.op(*f16_exp2).unwrap(), Op::Unary { op: UnaryOp::Exp2, .. }));
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
    assert!((CpuBackend.execute(&promoted, boolean_output, &bindings).unwrap().scalar_at(0).as_f64()
        - std::f64::consts::E)
        .abs()
        < 1e-5);
    assert!((CpuBackend.execute(&promoted, signed_output, &bindings).unwrap().scalar_at(0).as_f64()
        - (-1.0f64).exp())
        .abs()
        < 1e-5);
    assert_eq!(CpuBackend.execute(&promoted, unsigned_output, &bindings).unwrap().scalar_at(0).as_f64(), 1.0);

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::F16);
    let output = empty.exp(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::F16);
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
    assert!(matches!(graph.exp(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    assert!(graph.node(gradient).is_ok());
}

#[test]
fn exp2_preserves_tinygrad_storage_width_special_values_and_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.exp2(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Exp2, input: source }
        if *source == input));
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
    assert!(gradient_values.iter().all(|value| (*value - std::f64::consts::LN_2).abs() < 1e-12));

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
        ("signed".into(), TensorData::from_scalars([1], DType::I64, [Scalar::I(-1)]).unwrap()),
        ("unsigned".into(), TensorData::from_scalars([1], DType::U64, [Scalar::U(3)]).unwrap()),
    ]);
    assert_eq!(CpuBackend.execute(&dtypes, boolean_output, &bindings).unwrap().scalar_at(0).as_f64(), 2.0);
    assert_eq!(CpuBackend.execute(&dtypes, signed_output, &bindings).unwrap().scalar_at(0).as_f64(), 0.5);
    assert_eq!(CpuBackend.execute(&dtypes, unsigned_output, &bindings).unwrap().scalar_at(0).as_f64(), 8.0);

    let mut narrow = Graph::new();
    let input = narrow.input_dtype("input", [1], DType::F16);
    let output = narrow.exp2(input).unwrap();
    let gradient = narrow.grad(narrow.sum_all(output).unwrap(), input).unwrap();
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
    assert_eq!(
        narrow.dtype(gradient).unwrap(),
        DType::BF16
    );

    let mut empty = Graph::new();
    let input = empty.input_dtype("input", [0], DType::BF16);
    let output = empty.exp2(input).unwrap();
    assert_eq!(empty.dtype(output).unwrap(), DType::BF16);
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
    assert!(matches!(graph.exp2(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sqrt, input: source }
        if *source == input));
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
        let Op::Unary { op: UnaryOp::Sqrt, input: sqrt_input } = dtypes.op(output).unwrap() else {
            panic!("public sqrt must end in its raw SQRT ALU");
        };
        if dtype.is_float() {
            assert_eq!(*sqrt_input, source, "{dtype:?} stays a homogeneous SQRT");
        } else {
            assert!(matches!(dtypes.op(*sqrt_input).unwrap(), Op::Cast { input, dtype: DType::F32 }
                if *input == source), "{dtype:?} must use Cast(F32) before SQRT");
        }
        // The public cast makes the nonfloat unary UOp homogeneous, while
        // retaining the raw UnaryOp::Sqrt node for downstream backends.
        assert!(crate::lower_graph_elementwise(&dtypes, output).is_ok(), "{dtype:?} lowers");
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.sqrt(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(*root).unwrap(), Op::Unary { op: UnaryOp::Sqrt, input: source }
        if *source == input));
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
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Rsqrt, .. })
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
        let Op::Unary { op: UnaryOp::Reciprocal, input: root } = dtypes.op(output).unwrap() else {
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.rsqrt(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.rsqrt(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn square_uses_tinygrad_self_multiplication_structure_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.square(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Mul, lhs, rhs }
        if *lhs == input && *rhs == input));
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
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Square, .. })
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
        assert!(matches!(dtypes.op(output).unwrap(), Op::Binary { op: BinaryOp::Mul, lhs, rhs }
            if *lhs == source && *rhs == source));
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.square(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let input = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.square(input), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn sin_preserves_direct_storage_and_tinygrad_phase_shift_vjp() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.sin(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sin, input: source }
        if *source == input));
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
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Cos, .. })
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.sin(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Sin, .. }));
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Cos, .. })
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
    assert_eq!(values.scalar_at(3).as_f64(), (std::f64::consts::FRAC_PI_2 - 1.0e20).sin());
    assert!(values.scalar_at(4).as_f64().is_nan());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());

    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Cos, .. })
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.cos(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Mul, .. }));
    assert!(
        (0..graph.node_count()).all(|index| {
            !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Tan, .. })
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Tan, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [
        ("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16),
        ("f32", DType::F32, DType::F32), ("f64", DType::F64, DType::F64),
        ("bool", DType::Bool, DType::F32), ("i8", DType::I8, DType::F32),
        ("u8", DType::U8, DType::F32), ("i16", DType::I16, DType::F32),
        ("u16", DType::U16, DType::F32), ("i32", DType::I32, DType::F32),
        ("u32", DType::U32, DType::F32), ("i64", DType::I64, DType::F32),
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
    assert!(matches!(graph.tan(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Asin, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sqrt, .. })
    }));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([5], DType::F64,
            [Scalar::F(-0.0), Scalar::F(1.0), Scalar::F(-1.0), Scalar::F(2.0), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() + std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16",DType::F16,DType::F16),("bf16",DType::BF16,DType::BF16),("f32",DType::F32,DType::F32),("bool",DType::Bool,DType::F32),("i64",DType::I64,DType::F32),("u64",DType::U64,DType::F32)] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let output = dtypes.asin(source).unwrap();
        assert_eq!(dtypes.dtype(output).unwrap(), output_dtype);
    }
    let node_count = graph.node_count();
    assert!(matches!(graph.asin(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(output).unwrap(), Op::Binary { op: BinaryOp::Sub, .. }));
    assert!((0..graph.node_count()).all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Acos, .. })));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([5], DType::F64,
            [Scalar::F(-0.0), Scalar::F(1.0), Scalar::F(-1.0), Scalar::F(2.0), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert!((values.scalar_at(0).as_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
    assert!(values.scalar_at(1).as_f64().abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() - std::f64::consts::PI).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(graph.acos(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!((0..graph.node_count()).all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Atan, .. })));
    assert!((0..graph.node_count()).any(|index| matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sqrt, .. })));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([5], DType::F64,
            [Scalar::F(-0.0), Scalar::F(1.0), Scalar::F(-1.0), Scalar::F(f64::INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((values.scalar_at(2).as_f64() + std::f64::consts::FRAC_PI_4).abs() < 1e-6);
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(graph.atan(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sinh, .. })
    }));
    assert!((0..graph.node_count()).filter(|index| {
        matches!(graph.op(NodeId(*index)).unwrap(), Op::Unary { op: UnaryOp::Exp2, .. })
    }).count() >= 2);
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Neg, input: source }
            if *source == input)
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([7], DType::F64,
            [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(1.0), Scalar::F(-1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(2).as_f64() - ((1.0f64.exp() - (-1.0f64).exp()) / 2.0)).abs() < 1e-12);
    assert!((values.scalar_at(3).as_f64() - (((-1.0f64).exp() - 1.0f64.exp()) / 2.0)).abs() < 1e-12);
    assert!(values.scalar_at(4).as_f64().is_infinite() && values.scalar_at(4).as_f64().is_sign_positive());
    assert!(values.scalar_at(5).as_f64().is_infinite() && values.scalar_at(5).as_f64().is_sign_negative());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sinh, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.sinh(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.sinh(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn cosh_uses_tinygrad_exp_sum_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.cosh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Cosh, .. })
    }));
    assert!((0..graph.node_count()).filter(|index| {
        matches!(graph.op(NodeId(*index)).unwrap(), Op::Unary { op: UnaryOp::Exp2, .. })
    }).count() >= 2);
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Neg, input: source }
            if *source == input)
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([7], DType::F64,
            [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(1.0), Scalar::F(-1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 1.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 1.0f64.to_bits());
    assert!((values.scalar_at(2).as_f64() - ((1.0f64.exp() + (-1.0f64).exp()) / 2.0)).abs() < 1e-12);
    assert!((values.scalar_at(3).as_f64() - (((-1.0f64).exp() + 1.0f64.exp()) / 2.0)).abs() < 1e-12);
    assert!(values.scalar_at(4).as_f64().is_infinite() && values.scalar_at(4).as_f64().is_sign_positive());
    assert!(values.scalar_at(5).as_f64().is_infinite() && values.scalar_at(5).as_f64().is_sign_positive());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Cosh, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.cosh(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.cosh(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn asinh_uses_tinygrad_square_sqrt_log_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.asinh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Asinh, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sqrt, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Log2, .. })
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([7], DType::F64,
            [Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(1.0), Scalar::F(-1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert_eq!(values.scalar_at(1).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(2).as_f64() - (1.0f64 + 2.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!((values.scalar_at(3).as_f64() - (-1.0f64 + 2.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!(values.scalar_at(4).as_f64().is_infinite() && values.scalar_at(4).as_f64().is_sign_positive());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Asinh, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.asinh(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.asinh(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn acosh_uses_tinygrad_square_sub_sqrt_log_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.acosh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Acosh, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Sqrt, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Log2, .. })
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([7], DType::F64,
            [Scalar::F(1.0), Scalar::F(2.0), Scalar::F(-0.0), Scalar::F(0.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_f64().to_bits(), 0.0f64.to_bits());
    assert!((values.scalar_at(1).as_f64() - (2.0f64 + 3.0f64.sqrt()).ln()).abs() < 1e-12);
    assert!(values.scalar_at(2).as_f64().is_nan());
    assert!(values.scalar_at(3).as_f64().is_nan());
    assert!(values.scalar_at(4).as_f64().is_infinite() && values.scalar_at(4).as_f64().is_sign_positive());
    assert!(values.scalar_at(5).as_f64().is_nan());
    assert!(values.scalar_at(6).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Acosh, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.acosh(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.acosh(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn atanh_uses_tinygrad_ratio_log_division_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [9], DType::F64);
    let output = graph.atanh(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Atanh, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Log2, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Reciprocal, .. })
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([9], DType::F64,
            [Scalar::F(-1.0), Scalar::F(1.0), Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(0.5),
             Scalar::F(2.0), Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert!(values.scalar_at(0).as_f64().is_infinite() && values.scalar_at(0).as_f64().is_sign_negative());
    assert!(values.scalar_at(1).as_f64().is_infinite() && values.scalar_at(1).as_f64().is_sign_positive());
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Atanh, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.atanh(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.atanh(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn erf_uses_tinygrad_aands_polynomial_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [7], DType::F64);
    let output = graph.erf(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert!((0..graph.node_count()).all(|index| {
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Erf, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Exp2, .. })
    }));
    assert!((0..graph.node_count()).filter(|index| {
        matches!(graph.op(NodeId(*index)).unwrap(), Op::Unary { op: UnaryOp::Sign, .. })
    }).count() >= 2);
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([7], DType::F64,
            [Scalar::F(-1.0), Scalar::F(-0.0), Scalar::F(0.0), Scalar::F(1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Erf, .. })
    }));

    let mut dtypes = Graph::new();
    for (name, dtype, output_dtype) in [("f16", DType::F16, DType::F16), ("bf16", DType::BF16, DType::BF16), ("f32", DType::F32, DType::F32), ("bool", DType::Bool, DType::F32), ("i64", DType::I64, DType::F32), ("u64", DType::U64, DType::F32)] {
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
    assert!(matches!(graph.erf(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Floor, .. })
    }));
    assert!((0..graph.node_count()).any(|index| {
        matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Trunc, input: source }
            if *source == input)
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([8], DType::F64,
            [Scalar::F(-1.5), Scalar::F(-1.0), Scalar::F(-0.0), Scalar::F(0.5), Scalar::F(1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-2.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(values.scalar_at(5).as_f64().is_infinite() && values.scalar_at(5).as_f64().is_sign_positive());
    assert!(values.scalar_at(6).as_f64().is_infinite() && values.scalar_at(6).as_f64().is_sign_negative());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [("bool", DType::Bool), ("i64", DType::I64), ("u64", DType::U64), ("f16", DType::F16), ("bf16", DType::BF16), ("f32", DType::F32)] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.floor(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.floor(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(graph.floor(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.floor(source), Err(Error::ShapeOverflow(_))));
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
        !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Ceil, .. })
    }));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([8], DType::F64,
            [Scalar::F(-1.5), Scalar::F(-1.0), Scalar::F(-0.0), Scalar::F(0.5), Scalar::F(1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-1.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 1.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(values.scalar_at(5).as_f64().is_infinite() && values.scalar_at(5).as_f64().is_sign_positive());
    assert!(values.scalar_at(6).as_f64().is_infinite() && values.scalar_at(6).as_f64().is_sign_negative());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [("bool", DType::Bool), ("i64", DType::I64), ("u64", DType::U64), ("f16", DType::F16), ("bf16", DType::BF16), ("f32", DType::F32)] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.ceil(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.ceil(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(graph.ceil(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.ceil(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn trunc_preserves_tinygrad_direct_alu_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.trunc(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Trunc, input: source } if *source == input));
    let values = CpuBackend::execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([8], DType::F64,
            [Scalar::F(-1.5), Scalar::F(-1.0), Scalar::F(-0.0), Scalar::F(0.5), Scalar::F(1.0),
             Scalar::F(f64::INFINITY), Scalar::F(f64::NEG_INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.to_vec_f64()[0..2], [-1.0, -1.0]);
    assert_eq!(values.scalar_at(2).as_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values.scalar_at(3).as_f64(), 0.0);
    assert_eq!(values.scalar_at(4).as_f64(), 1.0);
    assert!(values.scalar_at(5).as_f64().is_infinite() && values.scalar_at(5).as_f64().is_sign_positive());
    assert!(values.scalar_at(6).as_f64().is_infinite() && values.scalar_at(6).as_f64().is_sign_negative());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let mut dtypes = Graph::new();
    for (name, dtype) in [("bool", DType::Bool), ("i64", DType::I64), ("u64", DType::U64), ("f16", DType::F16), ("bf16", DType::BF16), ("f32", DType::F32)] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.trunc(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), dtype);
    }
    let mut empty = Graph::new();
    let source = empty.input_dtype("input", [0], DType::F16);
    let result = empty.trunc(source).unwrap();
    assert_eq!(empty.shape(result).unwrap(), &Shape::new([0]));
    let node_count = graph.node_count();
    assert!(matches!(graph.trunc(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.trunc(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn round_uses_tinygrad_ties_even_composition_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [8], DType::F64);
    let output = graph.round(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Select { .. }));
    assert!((0..graph.node_count()).all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::Round, .. })));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([8], DType::F64,
            [Scalar::F(-2.5), Scalar::F(-1.5), Scalar::F(-0.5), Scalar::F(0.5), Scalar::F(1.5), Scalar::F(2.5), Scalar::F(f64::INFINITY), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.to_vec_f64()[0..6], [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0]);
    assert!(values.scalar_at(6).as_f64().is_infinite());
    assert!(values.scalar_at(7).as_f64().is_nan());
    let loss = graph.sum_all(output).unwrap();
    let gradient = graph.grad(loss, input).unwrap();
    assert_eq!(graph.dtype(gradient).unwrap(), DType::F64);
    let node_count = graph.node_count();
    assert!(matches!(graph.round(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.round(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn logical_not_uses_tinygrad_bool_cast_ne_true_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F64);
    let output = graph.logical_not(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!(matches!(graph.op(output).unwrap(), Op::Compare { op: CompareOp::Ne, .. }));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([4], DType::F64, [Scalar::F(-0.0), Scalar::F(2.0), Scalar::F(f64::NAN), Scalar::F(f64::INFINITY)]).unwrap(),
    )])).unwrap();
    assert_eq!(values.scalar_at(0).as_bool(), true);
    assert_eq!(values.scalar_at(1).as_bool(), false);
    assert_eq!(values.scalar_at(2).as_bool(), false);
    assert_eq!(values.scalar_at(3).as_bool(), false);
    let mut dtypes = Graph::new();
    for (name, dtype) in [("bool", DType::Bool), ("i64", DType::I64), ("u64", DType::U64), ("f16", DType::F16), ("bf16", DType::BF16)] {
        let source = dtypes.input_dtype(name, [1], dtype);
        let result = dtypes.logical_not(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), DType::Bool);
    }
    let node_count = graph.node_count();
    assert!(matches!(graph.logical_not(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn isnan_uses_tinygrad_self_inequality_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [4], DType::F64);
    let output = graph.isnan(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Compare { op: CompareOp::Ne, lhs, rhs } if *lhs == input && *rhs == input));
    assert!((0..graph.node_count()).all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::IsNan, .. })));
    let values = CpuBackend.execute(&graph, output, &HashMap::from([(
        "input".into(), TensorData::from_scalars([4], DType::F64, [Scalar::F(-0.0), Scalar::F(f64::INFINITY), Scalar::F(f64::NAN), Scalar::F(f64::NAN)]).unwrap(),
    )])).unwrap();
    assert_eq!((0..4).map(|index| values.scalar_at(index).as_bool()).collect::<Vec<_>>(), vec![false, false, true, true]);
    let mut dtypes = Graph::new();
    for (name, dtype) in [("bool", DType::Bool), ("i64", DType::I64), ("u64", DType::U64), ("f16", DType::F16), ("bf16", DType::BF16)] {
        let source = dtypes.input_dtype(name, [0], dtype);
        let result = dtypes.isnan(source).unwrap();
        assert_eq!(dtypes.dtype(result).unwrap(), DType::Bool);
        assert_eq!(dtypes.shape(result).unwrap(), &Shape::new([0]));
    }
    let node_count = graph.node_count();
    assert!(matches!(graph.isnan(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn isinf_preserves_tinygrad_default_both_signs_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.isinf(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::IsInf, input: source } if *source == input));
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    let node_count = graph.node_count();
    assert!(matches!(graph.isinf(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.isinf(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
}

#[test]
fn isfinite_uses_tinygrad_isinf_isnan_logical_not_and_preflight() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [5], DType::F64);
    let output = graph.isfinite(input).unwrap();
    assert_eq!(graph.dtype(output).unwrap(), DType::Bool);
    assert!((0..graph.node_count()).all(|index| !matches!(graph.op(NodeId(index)).unwrap(), Op::Unary { op: UnaryOp::IsFinite, .. })));
    let node_count = graph.node_count();
    assert!(matches!(graph.isfinite(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
    let mut overflow = Graph::new();
    let source = overflow.input_dtype("input", [usize::MAX, 2], DType::F64);
    let node_count = overflow.node_count();
    assert!(matches!(overflow.isfinite(source), Err(Error::ShapeOverflow(_))));
    assert_eq!(overflow.node_count(), node_count);
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
    assert!(matches!(graph.op(*lhs).unwrap(), Op::Unary { op: UnaryOp::Log2, input: source }
        if *source == input));
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
    assert!(gradient_values.iter().all(|value| (*value - 0.5).abs() < 1e-12));

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
    assert!(matches!(dtypes.op(*lhs).unwrap(), Op::Unary { op: UnaryOp::Log2, .. }));
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.log(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Log2, input: source }
        if *source == input));
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
    assert!(gradient_values.iter().all(|value| (*value - expected).abs() < 1e-12));

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
    assert!(matches!(graph.log2(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
    assert!(matches!(graph.op(*rhs).unwrap(), Op::Unary { op: UnaryOp::Sign, input: signed }
        if *signed == input));
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
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
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
        CpuBackend.execute(&discrete, signed_output, &bindings).unwrap().storage(),
        &crate::Storage::I8(vec![i8::MIN])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, unsigned_output, &bindings).unwrap().storage(),
        &crate::Storage::U64(vec![0, u64::MAX])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, boolean_output, &bindings).unwrap().storage(),
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.abs(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}

#[test]
fn neg_uses_tinygrad_bool_logical_not_and_preflighted_numeric_unary() {
    let mut graph = Graph::new();
    let input = graph.input_dtype("input", [6], DType::F64);
    let output = graph.neg(input).unwrap();
    assert!(matches!(graph.op(output).unwrap(), Op::Unary { op: UnaryOp::Neg, input: negated }
        if *negated == input));
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
    assert_eq!(values.scalar_at(4).as_f64().to_bits(), nan_bits ^ (1_u64 << 63));
    assert_eq!(values.scalar_at(5).as_f64(), -3.0);
    assert_eq!(
        CpuBackend.execute(&graph, gradient, &bindings).unwrap().to_vec_f64(),
        vec![-1.0; 6]
    );

    let mut discrete = Graph::new();
    let boolean = discrete.input_dtype("boolean", [2], DType::Bool);
    let signed = discrete.input_dtype("signed", [1], DType::I8);
    let unsigned = discrete.input_dtype("unsigned", [1], DType::U64);
    let boolean_output = discrete.neg(boolean).unwrap();
    let signed_output = discrete.neg(signed).unwrap();
    let unsigned_output = discrete.neg(unsigned).unwrap();
    assert!(matches!(discrete.op(boolean_output).unwrap(), Op::Logical { op: LogicalOp::Not, lhs, rhs: None }
        if *lhs == boolean));
    assert!(matches!(discrete.op(signed_output).unwrap(), Op::Unary { op: UnaryOp::Neg, .. }));
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
        CpuBackend.execute(&discrete, boolean_output, &bindings).unwrap().storage(),
        &crate::Storage::Bool(vec![true, false])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, signed_output, &bindings).unwrap().storage(),
        &crate::Storage::I8(vec![i8::MIN])
    );
    assert_eq!(
        CpuBackend.execute(&discrete, unsigned_output, &bindings).unwrap().storage(),
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
                &HashMap::from([("input".into(), TensorData::new([0], vec![]).unwrap())]),
            )
            .unwrap()
            .to_vec_f64(),
        Vec::<f64>::new()
    );

    let node_count = graph.node_count();
    assert!(matches!(graph.neg(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
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
        ("start".into(), TensorData::new([2, 1], vec![1.0, 4.0]).unwrap()),
        ("end".into(), TensorData::new([3], vec![3.0, 5.0, 7.0]).unwrap()),
        (
            "weight".into(),
            TensorData::new([2, 3], vec![0.0, 0.5, 1.0, 0.25, 0.5, 0.75]).unwrap(),
        ),
    ]);
    assert_eq!(graph.dtype(output).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
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
    assert!(CpuBackend
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
        .is_nan());

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
        ("bias".into(), TensorData::new([2], vec![0.5, -0.5]).unwrap()),
    ]);
    assert_eq!(graph.dtype(output).unwrap(), DType::F64);
    assert_eq!(graph.dtype(input_gradient).unwrap(), DType::F32);
    assert_eq!(
        CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64(),
        vec![5.5, 10.5, 11.5, 24.5]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, input_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![4.0, 6.0, 4.0, 6.0]
    );
    assert_eq!(
        CpuBackend
            .execute(&graph, weight_gradient, &bindings)
            .unwrap()
            .to_vec_f64(),
        vec![4.0, 6.0, 4.0, 6.0]
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
                    ("weight".into(), TensorData::new([2], vec![10.0, 100.0]).unwrap()),
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
fn softplus_uses_tinygrad_logaddexp_and_preflights_beta() {
    let mut graph = Graph::new();
    let input = graph.input("input", [2]);
    let beta = graph.input("beta", []);
    let output = graph.softplus(input, beta).unwrap();
    let loss = graph.sum_all(output).unwrap();
    let input_gradient = graph.grad(loss, input).unwrap();
    let beta_gradient = graph.grad(loss, beta).unwrap();
    let bindings = HashMap::from([
        ("input".into(), TensorData::new([2], vec![0.0, 1.0]).unwrap()),
        ("beta".into(), TensorData::scalar(2.0)),
    ]);
    let output_values = CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64();
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
    let expected_beta = -2.0f64.ln() / 4.0
        + 2.0f64.exp() / (2.0 * (1.0 + 2.0f64.exp()))
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
                    TensorData::new([4], vec![1000.0, -1000.0, f32::INFINITY, f32::NAN])
                        .unwrap(),
                ),
                ("beta".into(), TensorData::scalar(1.0)),
            ]),
        )
        .unwrap()
        .to_vec_f64();
    assert_eq!(values[0], 1000.0);
    assert_eq!(values[1], 0.0);
    assert!(values[2].is_infinite() && values[2].is_sign_positive());
    assert!(values[3].is_nan());

    let mut scalar = Graph::new();
    let input = scalar.input("input", []);
    let beta = scalar.input("beta", []);
    let output = scalar.softplus(input, beta).unwrap();
    assert_eq!(scalar.shape(output).unwrap(), &Shape::new([]));
    assert!(CpuBackend
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
        .is_finite());

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
    let bindings = HashMap::from([("input".into(), TensorData::new([2], vec![0.0, 1.0]).unwrap())]);
    let values = CpuBackend.execute(&graph, output, &bindings).unwrap().to_vec_f64();
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
    assert!(values[2].is_infinite() && values[2].is_sign_positive());
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
    assert!(matches!(graph.mish(NodeId(usize::MAX)), Err(Error::UnknownNode(_))));
    assert_eq!(graph.node_count(), node_count);
}
