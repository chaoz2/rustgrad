use crate::uop::{self, Binary, Operation, Ternary, UType, Walk};
use crate::{AddressValue, DType, Graph, IndexValue, LiteralValue, Shape, TensorData, UOp, UPat};

fn i64t() -> UType {
    UType::scalar(DType::I64)
}

fn f32t() -> UType {
    UType::scalar(DType::F32)
}

fn i32t() -> UType {
    UType::scalar(DType::I32)
}

#[test]
fn uop_spec_and_dag_order_are_deterministic() {
    let x = UOp::constant(4, i64t());
    let y = UOp::constant(2, i64t());
    let add = UOp::binary(Binary::Add, x.clone(), y.clone());
    let root = UOp::sink(vec![add.clone(), add]);
    root.validate().unwrap();
    let order = root.topological().unwrap();
    assert_eq!(order.len(), 4);
    assert_eq!(order.last(), Some(&root));
    let bad = UOp::from_operation(Operation::Binary(Binary::Add), Some(i64t()), vec![x]);
    assert!(bad.validate().is_err());
}

#[test]
fn typed_operations_make_payload_mismatches_unrepresentable_and_validate_arity() {
    let logical_source = UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool));
    UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::Not),
        Some(UType::scalar(DType::Bool)),
        vec![logical_source.clone()],
    )
    .validate()
    .unwrap();
    UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::And),
        Some(UType::scalar(DType::Bool)),
        vec![logical_source.clone(), logical_source],
    )
    .validate()
    .unwrap();
    let arithmetic_source = UOp::constant(1, i64t());
    let wrong_arity = UOp::from_operation(
        Operation::Binary(Binary::Add),
        Some(i64t()),
        vec![arithmetic_source],
    );
    assert!(matches!(
        wrong_arity.validate(),
        Err(crate::UOpError::InvalidArity { actual: 1, .. })
    ));
}

#[test]
fn graph_unary_predicates_have_bool_outputs_and_retain_typed_inputs() {
    let input = UOp::scalar_constant(DType::F32, 1.0_f32.to_bits() as u64, f32t());
    for op in [
        crate::UnaryOp::IsNan,
        crate::UnaryOp::IsInf,
        crate::UnaryOp::IsFinite,
    ] {
        UOp::from_operation(
            Operation::GraphUnary(op),
            Some(UType::scalar(DType::Bool)),
            vec![input.clone()],
        )
        .validate()
        .unwrap();
    }

    let wrong_predicate_output = UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::IsNan),
        Some(f32t()),
        vec![input.clone()],
    );
    assert!(wrong_predicate_output.validate().is_err());
    let wrong_value_output = UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::Relu),
        Some(UType::scalar(DType::Bool)),
        vec![input],
    );
    assert!(wrong_value_output.validate().is_err());
}

#[test]
fn upat_rewrites_are_prioritized_shared_and_pure() {
    let x = UOp::scalar_constant(DType::I32, 7, i32t());
    let zero = UOp::scalar_constant(DType::I32, 0, i32t());
    let shared = UOp::binary(Binary::Add, x.clone(), zero);
    let root = UOp::sink(vec![shared.clone(), shared]);
    let pattern = UPat::op(Operation::Binary(Binary::Add))
        .sources(vec![UPat::any().named("left"), UPat::any().named("right")]);
    assert_eq!(
        pattern.matches(&root.sources()[0]).unwrap().get("left"),
        Some(&x)
    );
    let (rewritten, trace) =
        uop::rewrite(&root, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(trace.rules, vec!["add-zero"]);
    assert_eq!(rewritten.sources()[0], rewritten.sources()[1]);
}

#[test]
fn raw_scalar_identity_rewrite_is_type_checked_and_preserves_signed_zero() {
    let x = UOp::scalar_constant(DType::I32, 3, i32t());
    let positive_zero = UOp::scalar_constant(DType::I32, 0, i32t());
    let lhs = UOp::binary(Binary::Add, positive_zero.clone(), x.clone());
    let rhs = UOp::binary(Binary::Add, x.clone(), positive_zero);
    let root = UOp::sink(vec![lhs, rhs]);
    let (rewritten, trace) =
        uop::rewrite(&root, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(trace.rules, vec!["add-zero-left", "add-zero"]);
    assert_eq!(rewritten.sources(), &[x.clone(), x]);

    let negative_zero = UOp::scalar_constant(DType::F32, 0x8000_0000, f32t());
    let preserved = UOp::binary(
        Binary::Add,
        UOp::scalar_constant(DType::F32, 3.0_f32.to_bits() as u64, f32t()),
        negative_zero,
    );
    let (unchanged, trace) =
        uop::rewrite(&preserved, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert!(trace.rules.is_empty());
    assert_eq!(unchanged, preserved);
}

#[test]
fn scalar_literals_fail_closed_on_type_or_raw_bit_mismatch() {
    let malformed = [
        UOp::from_operation(
            Operation::Const(LiteralValue::Scalar {
                dtype: DType::F64,
                bits: 0,
            }),
            Some(f32t()),
            vec![],
        ),
        UOp::from_operation(
            Operation::Const(LiteralValue::Scalar {
                dtype: DType::U8,
                bits: 0x100,
            }),
            Some(UType::scalar(DType::U8)),
            vec![],
        ),
        UOp::from_operation(
            Operation::VConst(LiteralValue::Scalar {
                dtype: DType::Bool,
                bits: 2,
            }),
            Some(UType::scalar(DType::Bool)),
            vec![],
        ),
        UOp::from_operation(Operation::Const(LiteralValue::Int(0)), None, vec![]),
    ];
    for literal in malformed {
        assert!(literal.validate().is_err());
        assert!(uop::artifact::encode(&literal).is_err());
    }

    for (dtype, bits) in [
        (DType::Bool, 1),
        (DType::U64, u64::MAX),
        (DType::F16, 0x8001),
        (DType::F32, 0x7fc0_1234),
        (DType::F64, 0x8000_0000_0000_0000),
    ] {
        let literal = UOp::scalar_constant(dtype, bits, UType::scalar(dtype));
        literal.validate().unwrap();
        let artifact = uop::artifact::encode(&literal).unwrap();
        assert_eq!(uop::artifact::encode(&literal).unwrap(), artifact);
        assert_eq!(uop::artifact::decode(&artifact).unwrap(), literal);
    }
}

#[test]
fn add_zero_preserves_floating_signed_zero_and_keeps_integer_identity() {
    let float_types = [DType::F16, DType::BF16, DType::F32, DType::F64];
    for dtype in float_types {
        let ty = UType::scalar(dtype);
        // `LiteralValue::Int(0)` is deliberately used here: it is the only literal
        // shape matched by add-zero, and is valid with a floating UType.
        let negative_zero = UOp::scalar_constant(
            dtype,
            match dtype {
                DType::F16 => 0x8000,
                DType::BF16 => 0x8000,
                DType::F32 => 0x8000_0000,
                DType::F64 => 0x8000_0000_0000_0000,
                _ => unreachable!(),
            },
            ty,
        );
        let positive_zero = UOp::constant(0, ty);
        let add = UOp::binary(Binary::Add, negative_zero.clone(), positive_zero);
        let (rewritten, trace) =
            uop::rewrite(&add, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
        // The unfurled node is retained, so its CPU evaluation keeps IEEE
        // `-0 + +0 == +0` behavior rather than exposing the input's -0 bits.
        assert_eq!(trace.rules, Vec::<&str>::new());
        assert_eq!(rewritten, add);

        let reverse = UOp::binary(Binary::Add, UOp::constant(0, ty), negative_zero);
        let (rewritten, trace) =
            uop::rewrite(&reverse, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
        assert_eq!(trace.rules, Vec::<&str>::new());
        assert_eq!(rewritten, reverse);
    }

    let value = UOp::constant(-7, UType::scalar(DType::I32));
    let zero = UOp::constant(0, UType::scalar(DType::I32));
    let add = UOp::binary(Binary::Add, value.clone(), zero);
    let (rewritten, trace) = uop::rewrite(&add, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(trace.rules, vec!["add-zero"]);
    assert_eq!(rewritten, value);
}

#[test]
fn where_same_keeps_fallible_condition_but_folds_constant_condition() {
    let i32t = UType::scalar(DType::I32);
    let boolt = UType::scalar(DType::Bool);
    let one = UOp::constant(1, i32t);
    let zero = UOp::constant(0, i32t);
    let div = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Div),
        Some(i32t),
        vec![one.clone(), zero],
    );
    let condition = UOp::from_operation(
        Operation::GraphCompare(crate::CompareOp::Eq),
        Some(boolt),
        vec![div.clone(), one],
    );
    let guarded = UOp::from_operation(
        Operation::Ternary(Ternary::Where),
        Some(i32t),
        vec![condition.clone(), div.clone(), div.clone()],
    );
    guarded.validate().unwrap();
    let (rewritten, trace) =
        uop::rewrite(&guarded, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    // Evaluation visits Where's condition first; retaining this dependency
    // preserves the division-by-zero status rather than exposing either arm.
    assert_eq!(trace.rules, Vec::<&str>::new());
    assert_eq!(rewritten, guarded);
    assert!(rewritten.topological().unwrap().contains(&condition));

    let nan = UOp::scalar_constant(DType::F32, 0x7fc0_1234, UType::scalar(DType::F32));
    let constant_condition = UOp::constant(0, boolt);
    let safe = UOp::from_operation(
        Operation::Ternary(Ternary::Where),
        Some(UType::scalar(DType::F32)),
        vec![constant_condition, nan.clone(), nan.clone()],
    );
    safe.validate().unwrap();
    let (rewritten, trace) =
        uop::rewrite(&safe, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(trace.rules, vec!["where-same"]);
    // Returning the original branch preserves exact dtype and NaN payload.
    assert_eq!(rewritten, nan);
}

#[test]
fn uop_graph_scalar_pilot_is_inspectable() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::new([]));
    let one = graph.constant(TensorData::scalar(1.0));
    let y = graph.add(x, one).unwrap();
    let uop = uop::lower_graph_scalar(&graph, y).unwrap();
    uop.validate().unwrap();
    assert!(matches!(uop.operation(), Operation::Binary(Binary::Add)));
    let condition = UOp::constant(1, UType::scalar(DType::Bool));
    let where_ = UOp::from_operation(
        Operation::Ternary(Ternary::Where),
        uop.ty(),
        vec![condition, uop.clone(), uop],
    );
    where_.validate().unwrap();
}

#[test]
fn scalar_literals_retain_raw_storage_bits() {
    let cases = [
        (DType::Bool, crate::Storage::Bool(vec![true]), 1),
        (
            DType::U64,
            crate::Storage::U64(vec![0xfedc_ba98_7654_3210]),
            0xfedc_ba98_7654_3210,
        ),
        (DType::F16, crate::Storage::F16(vec![0x8001]), 0x8001),
        (DType::BF16, crate::Storage::BF16(vec![0x7fc1]), 0x7fc1),
        (
            DType::F32,
            crate::Storage::F32(vec![f32::from_bits(0x7fc0_1234)]),
            0x7fc0_1234,
        ),
        (
            DType::F64,
            crate::Storage::F64(vec![f64::from_bits(0x8000_0000_0000_0000)]),
            0x8000_0000_0000_0000,
        ),
    ];
    for (dtype, storage, bits) in cases {
        let mut graph = Graph::new();
        let node = graph.constant(TensorData::from_storage(Shape::new([]), storage).unwrap());
        let uop = uop::lower_graph_scalar(&graph, node).unwrap();
        assert_eq!(uop.ty(), Some(UType::scalar(dtype)));
        assert!(
            matches!(uop.operation(), Operation::Const(LiteralValue::Scalar { dtype: got, bits: raw }) if *got == dtype && *raw == bits)
        );
    }
}

#[test]
fn typed_scalar_rewrite_leaves_fixed_schedule_cache_identity_stable() {
    let mut graph = Graph::new();
    let x = graph.input_dtype("x", Shape::new([]), DType::I32);
    let zero = graph.constant(TensorData::scalar_with_dtype(
        crate::Scalar::I(0),
        DType::I32,
    ));
    let output = graph.add(x, zero).unwrap();
    let first = crate::schedule(&graph, output).unwrap();
    let second = crate::schedule(&graph, output).unwrap();
    assert_eq!(first.items[0].cache_key, second.items[0].cache_key);

    let lowered = uop::lower_graph_scalar(&graph, output).unwrap();
    let (rewritten, trace) =
        uop::rewrite(&lowered, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(trace.rules, vec!["add-zero"]);
    assert_eq!(rewritten.operation(), lowered.sources()[0].operation());
}

#[test]
fn typed_buffer_index_rejects_malformed_rank_and_element_metadata() {
    let base = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: crate::AddressSpace::Global,
            name: "b0".into(),
            element: i64t(),
        }),
        Some(i64t()),
        vec![],
    );
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(i64t()),
        vec![UOp::constant(6, i64t())],
    );
    let invalid = UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer: 0,
            elements: 5,
            input_shape: Shape::from([2, 3]),
            output_shape: Shape::from([3]),
        }),
        Some(i64t()),
        vec![base, range.clone()],
    );
    let root = UOp::sink(vec![
        invalid,
        UOp::from_operation(Operation::EndRange, None, vec![range]),
    ]);
    assert!(root.validate().is_err());
}

#[test]
fn static_view_map_composes_shrinks_and_rejects_invalid_bounds() {
    let first = crate::ViewMap::identity(Shape::from([3, 4]))
        .shrink(&[(1, 3), (0, 4)])
        .unwrap();
    let nested = first.shrink(&[(0, 2), (1, 3)]).unwrap();
    assert_eq!(nested.logical_shape, Shape::from([2, 2]));
    assert_eq!(nested.element_offset(0).unwrap(), 5);
    assert_eq!(nested.element_offset(3).unwrap(), 10);
    assert!(nested.shrink(&[(0, 3), (0, 1)]).is_err());
}

#[test]
fn affine_reads_allow_broadcast_aliases_but_writes_require_injectivity() {
    let broadcast = crate::AffineView {
        source_shape: Shape::from([1]),
        logical_shape: Shape::from([3]),
        strides: vec![0],
        offset: 0,
    };
    assert!(broadcast.validate_read().is_ok());
    assert!(broadcast.validate_write().is_err());
    let flipped = crate::AffineView::identity(Shape::from([3]))
        .flip(0)
        .unwrap();
    assert_eq!(flipped.element_offset(0).unwrap(), 2);
    assert!(flipped.validate_write().is_ok());
}
