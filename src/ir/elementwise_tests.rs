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
