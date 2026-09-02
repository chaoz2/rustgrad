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
fn reduction_uops_require_one_exact_scalar_init_accumulate_finalize_chain() {
    let ty = UType::scalar(DType::F32);
    let init = UOp::from_operation(
        Operation::ReduceInit(crate::ReductionValue {
            input_shape: Shape::from([2]),
            output_shape: Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind: crate::ReduceKind::Max,
        }),
        Some(ty),
        vec![],
    );
    let value = UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, ty);
    let update = UOp::from_operation(
        Operation::ReduceAccumulate,
        Some(ty),
        vec![init.clone(), value.clone()],
    );
    let finalize = UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update.clone()]);
    finalize.validate().unwrap();
    assert!(init.validate().is_err());
    assert!(update.validate().is_err());
    assert!(
        UOp::sink(vec![finalize.clone(), finalize.clone()])
            .validate()
            .is_err()
    );

    let second_update =
        UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, value]);
    let second_finalize =
        UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![second_update]);
    assert!(
        UOp::sink(vec![finalize, second_finalize])
            .validate()
            .is_err()
    );

    let separate_chain = || {
        let init = UOp::from_operation(
            Operation::ReduceInit(crate::ReductionValue {
                input_shape: Shape::from([2]),
                output_shape: Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Max,
            }),
            Some(ty),
            vec![],
        );
        let value = UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, ty);
        let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, value]);
        UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update])
    };
    let first = separate_chain();
    let second = separate_chain();
    let bytes = crate::uop::artifact::encode(&first).unwrap();
    crate::uop::artifact::decode(&bytes)
        .unwrap()
        .validate()
        .unwrap();
    let structurally_duplicated = UOp::sink(vec![first, second]);
    assert!(structurally_duplicated.validate().is_err());
    assert!(crate::uop::artifact::encode(&structurally_duplicated).is_err());

    let vector_ty = UType {
        scalar: DType::F32,
        lanes: 2,
    };
    let vector_init = UOp::from_operation(
        Operation::ReduceInit(crate::ReductionValue {
            input_shape: Shape::from([2]),
            output_shape: Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind: crate::ReduceKind::Sum,
        }),
        Some(vector_ty),
        vec![],
    );
    let vector_value = UOp::from_operation(
        Operation::VConst(crate::LiteralValue::Int(1)),
        Some(vector_ty),
        vec![],
    );
    let vector_update = UOp::from_operation(
        Operation::ReduceAccumulate,
        Some(vector_ty),
        vec![vector_init, vector_value],
    );
    let vector_finalize = UOp::from_operation(
        Operation::ReduceFinalize,
        Some(vector_ty),
        vec![vector_update],
    );
    assert!(vector_finalize.validate().is_err());
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
fn graph_unary_transcendentals_follow_source_dtype_lifting_and_lane_width() {
    let scalar_input = UOp::scalar_constant(DType::I64, 1, i64t());
    for op in [
        crate::UnaryOp::Exp2,
        crate::UnaryOp::Log2,
        crate::UnaryOp::Sin,
    ] {
        UOp::from_operation(
            Operation::GraphUnary(op),
            Some(f32t()),
            vec![scalar_input.clone()],
        )
        .validate()
        .unwrap();

        let wrong_storage = UOp::from_operation(
            Operation::GraphUnary(op),
            Some(i64t()),
            vec![scalar_input.clone()],
        );
        assert!(matches!(
            wrong_storage.validate(),
            Err(crate::UOpError::InvalidDType)
        ));
    }

    let four_i64 = UType::vector(DType::I64, 4).unwrap();
    let four_f32 = UType::vector(DType::F32, 4).unwrap();
    let vector_input = UOp::from_operation(
        Operation::Special("transcendental_input".into()),
        Some(four_i64),
        vec![],
    );
    UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::Sin),
        Some(four_f32),
        vec![vector_input.clone()],
    )
    .validate()
    .unwrap();

    let wrong_lanes = UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::Sin),
        Some(UType::vector(DType::F32, 2).unwrap()),
        vec![vector_input],
    );
    assert!(matches!(
        wrong_lanes.validate(),
        Err(crate::UOpError::InvalidDType)
    ));
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
    assert!(rewritten.sources()[0].shares_node_with(&rewritten.sources()[1]));
}

#[test]
fn upat_varargs_and_typed_payload_predicates_drive_rewrites() {
    fn scalar_i32(ty: Option<UType>) -> bool {
        ty == Some(i32t())
    }
    fn positive_int(operation: &Operation) -> bool {
        matches!(operation, Operation::Const(LiteralValue::Int(value)) if *value > 0)
    }
    fn select_first(captures: &uop::Captures, _: &UOp) -> Option<UOp> {
        if captures.get("vector")?.sources().len() < 2 {
            return None;
        }
        captures.get("first").cloned()
    }

    let one = UOp::constant(1, i32t());
    let two = UOp::constant(2, i32t());
    let three = UOp::constant(3, i32t());
    let vector = UOp::from_operation(
        Operation::Vectorize,
        Some(i32t()),
        vec![one.clone(), two.clone(), three.clone()],
    );
    vector.validate().unwrap();

    // Exact sources retain their old arity contract. Prefix varargs match the
    // declared ordered prefix, while repeated varargs validate every source.
    assert!(
        UPat::op(Operation::Vectorize)
            .sources(vec![UPat::any()])
            .matches(&vector)
            .is_none()
    );
    let prefix = UPat::op(Operation::Vectorize)
        .type_predicate(scalar_i32)
        .sources_prefix(vec![
            UPat::operation_predicate(positive_int)
                .type_predicate(scalar_i32)
                .named("first"),
        ])
        .named("vector");
    let captures = prefix.matches(&vector).unwrap();
    assert_eq!(captures.get("first"), Some(&one));
    assert_eq!(captures.get("vector"), Some(&vector));
    assert!(
        UPat::op(Operation::Vectorize)
            .sources_prefix(vec![UPat::any(), UPat::any(), UPat::any(), UPat::any()])
            .matches(&vector)
            .is_none()
    );
    assert!(
        UPat::op(Operation::Vectorize)
            .sources_varargs(UPat::any().type_predicate(scalar_i32))
            .matches(&vector)
            .is_some()
    );

    let mixed = UOp::from_operation(
        Operation::Vectorize,
        Some(i32t()),
        vec![one.clone(), UOp::constant(4, UType::scalar(DType::I64))],
    );
    assert!(
        UPat::op(Operation::Vectorize)
            .sources_varargs(UPat::any().type_predicate(scalar_i32))
            .matches(&mixed)
            .is_none()
    );

    // A named repeated child retains the existing structural-equality rule.
    let duplicated = UOp::from_operation(
        Operation::Vectorize,
        Some(i32t()),
        vec![one.clone(), one.clone()],
    );
    let all_same = UPat::op(Operation::Vectorize).sources_varargs(UPat::any().named("same"));
    assert!(all_same.matches(&duplicated).is_some());
    assert!(all_same.matches(&vector).is_none());

    let encoded = uop::artifact::encode(&vector).unwrap();
    let mut rules = vec![uop::RewriteRule {
        name: "select-first-vararg",
        priority: 0,
        pattern: prefix,
        apply: select_first,
    }];
    let (rewritten, trace) = uop::rewrite(&vector, &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(rewritten, one);
    assert_eq!(trace.rules, vec!["select-first-vararg"]);
    assert_eq!(uop::artifact::encode(&vector).unwrap(), encoded);

    // Repeated-source varargs admit an empty list, matching tinygrad's
    // repeated-source pattern contract without weakening UOp validation.
    let source_free = UOp::from_operation(Operation::Special("source-free".into()), None, vec![]);
    source_free.validate().unwrap();
    assert!(
        UPat::op(Operation::Special("source-free".into()))
            .sources_varargs(UPat::any())
            .matches(&source_free)
            .is_some()
    );
}

#[test]
fn upat_alternatives_and_permuted_sources_preserve_candidate_isolation() {
    fn select_named(captures: &uop::Captures, _: &UOp) -> Option<UOp> {
        captures.get("selected").cloned()
    }
    fn keep_first_or_select(captures: &uop::Captures, node: &UOp) -> Option<UOp> {
        captures
            .get("first")
            .map(|_| node.clone())
            .or_else(|| captures.get("selected").cloned())
    }
    fn remove_zero(captures: &uop::Captures, _: &UOp) -> Option<UOp> {
        captures.get("value").cloned()
    }

    assert!(matches!(
        UPat::any_of(Vec::<UPat>::new()),
        Err(crate::UOpError::InvalidArgument)
    ));

    let one = UOp::constant(1, i32t());
    let two = UOp::constant(2, i32t());
    let add = UOp::binary(Binary::Add, one.clone(), two.clone());
    let alternatives = UPat::any_of([
        UPat::op(Operation::Binary(Binary::Add))
            .sources(vec![UPat::any().named("first"), UPat::any()]),
        UPat::op(Operation::Binary(Binary::Add))
            .sources(vec![UPat::any(), UPat::any().named("selected")]),
    ])
    .unwrap()
    .named("root");
    let first = alternatives.matches(&add).unwrap();
    assert_eq!(first.get("first"), Some(&one));
    assert!(first.get("selected").is_none());
    assert_eq!(first.get("root"), Some(&add));

    // The first structural match is allowed to decline the rewrite. Candidate
    // captures are isolated, so the second alternative can then apply without
    // inheriting `first`.
    let encoded = uop::artifact::encode(&add).unwrap();
    let mut rules = vec![uop::RewriteRule {
        name: "select-second-alternative",
        priority: 0,
        pattern: alternatives.clone(),
        apply: select_named,
    }];
    let (rewritten, trace) = uop::rewrite(&add, &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(rewritten, two);
    assert_eq!(trace.rules, vec!["select-second-alternative"]);
    assert_eq!(uop::artifact::encode(&add).unwrap(), encoded);

    // Returning the original node accepts the first candidate but performs no
    // rewrite; only `None` asks the matcher to try the next alternative.
    let mut rules = vec![uop::RewriteRule {
        name: "keep-first-alternative",
        priority: 0,
        pattern: alternatives,
        apply: keep_first_or_select,
    }];
    let (unchanged, trace) = uop::rewrite(&add, &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(unchanged, add);
    assert!(trace.rules.is_empty());

    let partial_failure = UPat::any_of([
        UPat::op(Operation::Binary(Binary::Add)).sources(vec![
            UPat::any().named("leaked"),
            UPat::op(Operation::Const(LiteralValue::Int(99))),
        ]),
        UPat::op(Operation::Binary(Binary::Add)).sources(vec![
            UPat::op(Operation::Const(LiteralValue::Int(1))),
            UPat::any().named("selected"),
        ]),
    ])
    .unwrap();
    let captures = partial_failure.matches(&add).unwrap();
    assert!(captures.get("leaked").is_none());
    assert_eq!(captures.get("selected"), Some(&two));

    let zero = UOp::constant(0, i32t());
    let left = UOp::binary(Binary::Add, zero.clone(), one.clone());
    let right = UOp::binary(Binary::Add, one.clone(), zero);
    let commutative = UPat::op(Operation::Binary(Binary::Add)).sources_permuted(vec![
        UPat::op(Operation::Const(LiteralValue::Int(0))),
        UPat::any().named("value"),
    ]);
    assert!(
        commutative
            .clone()
            .matches(&UOp::sink(vec![one.clone()]))
            .is_none()
    );
    let mut rules = vec![uop::RewriteRule {
        name: "commutative-add-zero",
        priority: 0,
        pattern: commutative,
        apply: remove_zero,
    }];
    let (rewritten, trace) =
        uop::rewrite(&UOp::sink(vec![left, right]), &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(rewritten.sources(), &[one.clone(), one.clone()]);
    assert_eq!(
        trace.rules,
        vec!["commutative-add-zero", "commutative-add-zero"]
    );

    let same_role = UPat::op(Operation::Binary(Binary::Add))
        .sources_permuted(vec![UPat::any().named("same"), UPat::any().named("same")]);
    assert!(same_role.matches(&add).is_none());
    assert!(
        same_role
            .matches(&UOp::binary(Binary::Add, one.clone(), one.clone()))
            .is_some()
    );

    fn choose_three(captures: &uop::Captures, _: &UOp) -> Option<UOp> {
        captures.get("three").cloned()
    }
    let three = UOp::constant(3, i32t());
    let vector = UOp::from_operation(
        Operation::Vectorize,
        Some(i32t()),
        vec![one.clone(), two.clone(), three.clone()],
    );
    let arbitrary_arity = UPat::op(Operation::Vectorize).sources_permuted(vec![
        UPat::op(Operation::Const(LiteralValue::Int(3))).named("three"),
        UPat::op(Operation::Const(LiteralValue::Int(1))),
        UPat::op(Operation::Const(LiteralValue::Int(2))),
    ]);
    let mut rules = vec![uop::RewriteRule {
        name: "permuted-vector-sources",
        priority: 0,
        pattern: arbitrary_arity,
        apply: choose_three,
    }];
    let (rewritten, trace) = uop::rewrite(&vector, &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(rewritten, three);
    assert_eq!(trace.rules, vec!["permuted-vector-sources"]);

    let permutation_rollback = UPat::op(Operation::Vectorize).sources_permuted(vec![
        UPat::any().named("first"),
        UPat::op(Operation::Const(LiteralValue::Int(2))),
        UPat::op(Operation::Const(LiteralValue::Int(1))),
    ]);
    let captures = permutation_rollback.matches(&vector).unwrap();
    assert_eq!(captures.get("first"), Some(&three));

    let declared_first = UPat::op(Operation::Binary(Binary::Add)).sources_permuted(vec![
        UPat::any().named("declared_first"),
        UPat::any().named("declared_second"),
    ]);
    let captures = declared_first.matches(&add).unwrap();
    assert_eq!(captures.get("declared_first"), Some(&one));
    assert_eq!(captures.get("declared_second"), Some(&two));

    static ALL_SAME_ATTEMPTS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    fn count_and_decline(_: &uop::Captures, _: &UOp) -> Option<UOp> {
        ALL_SAME_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }
    ALL_SAME_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
    let all_same = UPat::op(Operation::Vectorize).sources_permuted(vec![
        UPat::any(),
        UPat::any(),
        UPat::any(),
    ]);
    let mut rules = vec![uop::RewriteRule {
        name: "all-same-permutation",
        priority: 0,
        pattern: all_same,
        apply: count_and_decline,
    }];
    let (unchanged, trace) = uop::rewrite(&vector, &mut rules, Walk::BottomUp).unwrap();
    assert_eq!(unchanged, vector);
    assert!(trace.rules.is_empty());
    assert_eq!(
        ALL_SAME_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Outer constraints intentionally intersect every alternative rather than
    // being copied into or applied after only one branch.
    let outer_type = UPat::any_of([
        UPat::op(Operation::Binary(Binary::Add)),
        UPat::op(Operation::Binary(Binary::Mul)),
    ])
    .unwrap()
    .dtype(i32t());
    assert!(outer_type.matches(&add).is_some());
    let float_add = UOp::binary(
        Binary::Add,
        UOp::constant(1, f32t()),
        UOp::constant(2, f32t()),
    );
    assert!(outer_type.matches(&float_add).is_none());
}

#[test]
fn scheduled_normalization_converges_across_replacement_chains() {
    let two = UOp::scalar_constant(DType::I32, 2, i32t());
    let three = UOp::scalar_constant(DType::I32, 3, i32t());
    let sum = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Add),
        Some(i32t()),
        vec![two, three],
    );
    let negated = UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::Neg),
        Some(i32t()),
        vec![sum],
    );
    let one = UOp::scalar_constant(DType::I32, 1, i32t());
    let root = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Mul),
        Some(i32t()),
        vec![negated, one],
    );
    root.validate().unwrap();

    let (normalized, trace) =
        uop::rewrite(&root, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(
        trace.rules,
        vec![
            "fold-integral-binary",
            "fold-integral-unary",
            "typed-mul-one"
        ]
    );
    assert!(matches!(
        normalized.operation(),
        Operation::Const(LiteralValue::Scalar {
            dtype: DType::I32,
            bits: 0xffff_fffb,
        })
    ));
    let sink = UOp::sink(vec![root]);
    assert_eq!(
        uop::normalize_kernel(&sink).unwrap().sources()[0],
        normalized
    );
}

#[test]
fn comparison_and_where_normalization_revisits_the_replacement() {
    let four = UOp::scalar_constant(DType::U64, 4, UType::scalar(DType::U64));
    let condition = UOp::from_operation(
        Operation::GraphCompare(crate::CompareOp::Eq),
        Some(UType::scalar(DType::Bool)),
        vec![four.clone(), four],
    );
    let on_true = UOp::scalar_constant(DType::I32, 7, i32t());
    let on_false = UOp::scalar_constant(DType::I32, 9, i32t());
    let root = UOp::from_operation(
        Operation::Ternary(Ternary::Where),
        Some(i32t()),
        vec![condition, on_true.clone(), on_false],
    );
    root.validate().unwrap();

    let (normalized, trace) =
        uop::rewrite(&root, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert_eq!(
        trace.rules,
        vec!["fold-integral-compare", "typed-where-const"]
    );
    assert_eq!(normalized, on_true);
}

#[test]
fn bool_logical_and_same_cast_folds_preserve_typed_storage() {
    let false_ = UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool));
    let not = UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::Not),
        Some(UType::scalar(DType::Bool)),
        vec![false_],
    );
    let true_ = UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool));
    let and = UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::And),
        Some(UType::scalar(DType::Bool)),
        vec![not, true_],
    );
    let nan = UOp::scalar_constant(DType::F32, 0x7fc0_1234, f32t());
    let cast = UOp::from_operation(Operation::Cast, Some(f32t()), vec![nan.clone()]);
    let sink = UOp::sink(vec![and, cast]);
    let normalized = uop::normalize_kernel(&sink).unwrap();
    assert!(matches!(
        normalized.sources()[0].operation(),
        Operation::Const(LiteralValue::Scalar {
            dtype: DType::Bool,
            bits: 1,
        })
    ));
    assert_eq!(normalized.sources()[1], nan);

    let malformed = UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::Not),
        Some(UType::scalar(DType::Bool)),
        vec![UOp::scalar_constant(DType::I32, 1, i32t())],
    );
    let (unchanged, trace) =
        uop::rewrite(&malformed, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert!(trace.rules.is_empty());
    assert_eq!(unchanged, malformed);

    let malformed_rhs = UOp::from_operation(
        Operation::GraphLogical(crate::LogicalOp::And),
        Some(UType::scalar(DType::Bool)),
        vec![
            UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool)),
            UOp::scalar_constant(DType::I32, 1, i32t()),
        ],
    );
    let (unchanged, trace) =
        uop::rewrite(&malformed_rhs, &mut uop::builtin_rules(), Walk::BottomUp).unwrap();
    assert!(trace.rules.is_empty());
    assert_eq!(unchanged, malformed_rhs);
}

#[test]
fn rewrite_rejects_rule_cycles_and_step_exhaustion_without_mutating_the_root() {
    fn int_literal(operation: &Operation) -> bool {
        matches!(operation, Operation::Const(LiteralValue::Int(_)))
    }
    fn increment(_: &uop::Captures, node: &UOp) -> Option<UOp> {
        let Operation::Const(LiteralValue::Int(value)) = node.operation() else {
            return None;
        };
        Some(UOp::constant(value.checked_add(1)?, node.ty()?))
    }

    let root = UOp::constant(0, i32t());
    let mut cycle = vec![
        uop::RewriteRule {
            name: "zero-to-one",
            priority: 0,
            pattern: UPat::op(Operation::Const(LiteralValue::Int(0))),
            apply: |_, node| Some(UOp::constant(1, node.ty()?)),
        },
        uop::RewriteRule {
            name: "one-to-zero",
            priority: 1,
            pattern: UPat::op(Operation::Const(LiteralValue::Int(1))),
            apply: |_, node| Some(UOp::constant(0, node.ty()?)),
        },
    ];
    assert!(matches!(
        uop::rewrite(&root, &mut cycle, Walk::BottomUp),
        Err(crate::UOpError::RewriteCycle)
    ));
    assert!(matches!(
        root.operation(),
        Operation::Const(LiteralValue::Int(0))
    ));

    let mut unbounded = vec![uop::RewriteRule {
        name: "increment",
        priority: 0,
        pattern: UPat::operation_predicate(int_literal),
        apply: increment,
    }];
    assert!(matches!(
        uop::rewrite(&root, &mut unbounded, Walk::BottomUp),
        Err(crate::UOpError::RewriteStepLimit)
    ));
    assert!(matches!(
        root.operation(),
        Operation::Const(LiteralValue::Int(0))
    ));
}

#[test]
fn artifact_decode_preserves_historical_unnormalized_uops() {
    let root = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Add),
        Some(i32t()),
        vec![
            UOp::scalar_constant(DType::I32, 7, i32t()),
            UOp::scalar_constant(DType::I32, 0, i32t()),
        ],
    );
    root.validate().unwrap();
    let encoded = uop::artifact::encode(&root).unwrap();
    let decoded = uop::artifact::decode(&encoded).unwrap();
    assert_eq!(decoded, root);
    assert!(matches!(decoded.operation(), Operation::GraphBinary(_)));
    let normalized = uop::normalize_kernel(&UOp::sink(vec![decoded.clone()])).unwrap();
    assert_ne!(normalized.sources()[0], decoded);
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

    for bits in [0x8000_0000_u64, 0x7fc0_1234] {
        let value = UOp::scalar_constant(DType::F32, bits, f32t());
        let zero = UOp::scalar_constant(DType::F32, 0, f32t());
        let graph_add = UOp::from_operation(
            Operation::GraphBinary(crate::BinaryOp::Add),
            Some(f32t()),
            vec![value, zero],
        );
        graph_add.validate().unwrap();
        let normalized = uop::normalize_kernel(&UOp::sink(vec![graph_add.clone()])).unwrap();
        assert_eq!(normalized.sources()[0], graph_add);
    }
}

#[test]
fn scheduled_normalization_rejects_effect_and_conditional_roots_without_rewriting() {
    let value = UOp::scalar_constant(DType::I32, 7, i32t());
    let zero = UOp::scalar_constant(DType::I32, 0, i32t());
    let foldable = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Add),
        Some(i32t()),
        vec![value, zero],
    );
    let condition = UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool));
    let if_ = UOp::from_operation(Operation::If, None, vec![condition]);
    let end_if = UOp::from_operation(Operation::EndIf, None, vec![if_]);
    let root = UOp::sink(vec![foldable.clone(), end_if.clone()]);
    assert!(matches!(
        uop::normalize_kernel(&root),
        Err(crate::UOpError::EffectRewrite)
    ));
    assert_eq!(root.sources(), &[foldable, end_if]);
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
    assert_eq!(first.items[0].kernel, second.items[0].kernel);
    assert_eq!(
        first.items[0]
            .ordered_inputs()
            .iter()
            .map(|binding| binding.input_node)
            .collect::<Vec<_>>(),
        vec![x]
    );
    assert!(first.items[0].dependencies.is_empty());
    assert!(
        first.items[0]
            .kernel
            .topological()
            .unwrap()
            .iter()
            .all(|node| !matches!(node.operation(), Operation::GraphBinary(_)))
    );

    let raw = crate::kernel::lower_graph_elementwise(&graph, output).unwrap();
    assert!(raw.topological().unwrap().len() > first.items[0].kernel.topological().unwrap().len());
    let capture = crate::CapturedSchedule::capture(&graph, &first, &[output]).unwrap();
    let encoded = capture.to_bytes().unwrap();
    let decoded = crate::CapturedSchedule::from_bytes(&encoded).unwrap();
    assert_eq!(decoded.items[0].kernel, first.items[0].kernel);
    assert_eq!(decoded.identity, capture.identity);

    let bindings = std::collections::HashMap::from([(
        "x".into(),
        TensorData::scalar_with_dtype(crate::Scalar::I(-17), DType::I32),
    )]);
    let interpreted = crate::realize_with_options(
        &graph,
        &first,
        &[output],
        &bindings,
        crate::RealizationOptions {
            backend: crate::RealizationPolicy::Interpreter,
            memory_reuse: crate::MemoryReuse::Disabled,
        },
    )
    .unwrap();
    let native = crate::realize_with_options(
        &graph,
        &first,
        &[output],
        &bindings,
        crate::RealizationOptions {
            backend: crate::RealizationPolicy::CpuJit {
                fallback_to_interpreter: false,
            },
            memory_reuse: crate::MemoryReuse::Disabled,
        },
    )
    .unwrap();
    assert_eq!(interpreted.outputs[0].to_vec_f64(), vec![-17.0]);
    assert_eq!(
        native.outputs[0].storage(),
        interpreted.outputs[0].storage()
    );

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
            addressing: crate::IndexAddressing::Broadcast,
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
fn typed_address_definitions_require_their_declared_memory_space() {
    let address = |space| AddressValue {
        space,
        name: "buffer".into(),
        element: i64t(),
    };
    for operation in [
        Operation::DefineGlobal(address(crate::AddressSpace::Global)),
        Operation::DefineLocal(address(crate::AddressSpace::Local)),
        Operation::DefineRegister(address(crate::AddressSpace::Register)),
    ] {
        UOp::from_operation(operation, Some(i64t()), vec![])
            .validate()
            .unwrap();
    }
    for operation in [
        Operation::DefineGlobal(address(crate::AddressSpace::Local)),
        Operation::DefineLocal(address(crate::AddressSpace::Register)),
        Operation::DefineRegister(address(crate::AddressSpace::Global)),
    ] {
        assert_eq!(
            UOp::from_operation(operation, Some(i64t()), vec![]).validate(),
            Err(crate::UOpError::InvalidArgument)
        );
    }
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

#[test]
fn empty_affine_reads_still_validate_products_and_address_representability() {
    let boundary = crate::AffineView {
        source_shape: Shape::from([4]),
        logical_shape: Shape::from([4]),
        strides: vec![-1],
        offset: 3,
    };
    let normalized = boundary.normalized_read().unwrap();
    assert_eq!(normalized.offset, 0);
    assert_eq!(normalized.axes[0].stride, 1);
    assert!(normalized.axes[0].reversed);

    let negative_minimum = crate::AffineView {
        offset: 2,
        ..boundary.clone()
    };
    assert!(negative_minimum.validate_read().is_err());
    let upper_oob = crate::AffineView {
        offset: 4,
        strides: vec![1],
        ..boundary.clone()
    };
    assert!(upper_oob.validate_read().is_err());

    let overflowing_source = crate::AffineView {
        source_shape: Shape::from([usize::MAX, 2]),
        logical_shape: Shape::from([0]),
        strides: vec![1],
        offset: 0,
    };
    assert!(overflowing_source.validate_read().is_err());

    let overflowing_logical = crate::AffineView {
        source_shape: Shape::from([0]),
        logical_shape: Shape::from([usize::MAX, 2, 0]),
        strides: vec![0, 0, 0],
        offset: 0,
    };
    assert!(overflowing_logical.validate_read().is_err());

    let extreme_stride = crate::AffineView {
        source_shape: Shape::from([0, 2]),
        logical_shape: Shape::from([0, 2]),
        strides: vec![0, i64::MIN],
        offset: 0,
    };
    assert!(extreme_stride.validate_read().is_err());

    let skipped_singleton = crate::AffineView {
        source_shape: Shape::from([0, 1]),
        logical_shape: Shape::from([0, 1]),
        strides: vec![0, i64::MIN],
        offset: 0,
    };
    let normalized = skipped_singleton.normalized_read().unwrap();
    assert_eq!(normalized.axes[1].stride, 0);
    assert!(!normalized.axes[1].reversed);

    let addressless_reverse = crate::AffineView {
        source_shape: Shape::from([0, 4]),
        logical_shape: Shape::from([1, 0, 4]),
        strides: vec![0, 4, -1],
        offset: 0,
    };
    let normalized = addressless_reverse.normalized_read().unwrap();
    assert_eq!(normalized.offset, 0);
    assert!(
        normalized
            .axes
            .iter()
            .all(|axis| axis.stride == 0 && !axis.reversed)
    );
}
