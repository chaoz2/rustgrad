use super::*;

// Preserve the previous independent sequence as an oracle for admission and
// canonical bytes; a retained traversal must not change either contract.
fn independent_normalization(root: &UOp) -> Result<UOp, UOpError> {
    root.validate()?;
    let (root, _) = rewrite(root, &mut projected_index_rules(), Walk::BottomUp)?;
    root.validate()?;
    let nodes = root.topological()?;
    if !matches!(root.operation(), Operation::Sink)
        || nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::If
                    | Operation::EndIf
                    | Operation::EffectStore(_)
                    | Operation::After(_)
                    | Operation::Barrier
            )
        })
    {
        return Err(UOpError::EffectRewrite);
    }
    if nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::ReduceInit(_)
                | Operation::ReduceAccumulate
                | Operation::ReduceFinalize
                | Operation::TensorGuard(_)
        )
    }) {
        return Ok(root);
    }
    let (normalized, _) = rewrite(&root, &mut builtin_rules(), Walk::BottomUp)?;
    normalized.validate()?;
    Ok(normalized)
}

// Exact released table-building sequence, independent of root-allocation reuse.
fn released_table_normalization(root: &UOp) -> Result<CanonicalNodeTable, UOpError> {
    root.validate()?;
    let (root, _) = rewrite(root, &mut projected_index_rules(), Walk::BottomUp)?;
    let order = CanonicalNodeTable::new(&root)?;
    order.order().validate_for(&root)?;
    let nodes = order.order().nodes();
    if !matches!(root.operation(), Operation::Sink)
        || nodes.iter().any(|node| {
            matches!(
                node.operation(),
                Operation::If
                    | Operation::EndIf
                    | Operation::EffectStore(_)
                    | Operation::After(_)
                    | Operation::Barrier
            )
        })
    {
        return Err(UOpError::EffectRewrite);
    }
    if nodes.iter().any(|node| {
        matches!(
            node.operation(),
            Operation::ReduceInit(_)
                | Operation::ReduceAccumulate
                | Operation::ReduceFinalize
                | Operation::TensorGuard(_)
        )
    }) {
        return Ok(order);
    }
    let (normalized, _) = rewrite(&root, &mut builtin_rules(), Walk::BottomUp)?;
    drop(order);
    let order = CanonicalNodeTable::new(&normalized)?;
    order.order().validate_for(&normalized)?;
    Ok(order)
}

fn integer_fold() -> UOp {
    let ty = UType::scalar(DType::I32);
    UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Add),
        Some(ty),
        vec![
            UOp::scalar_constant(DType::I32, 7, ty),
            UOp::scalar_constant(DType::I32, 0, ty),
        ],
    )
}

fn reduction_kernel() -> UOp {
    let ty = UType::scalar(DType::F32);
    let init = UOp::from_operation(
        Operation::ReduceInit(crate::ReductionValue {
            input_shape: crate::Shape::from([2]),
            output_shape: crate::Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind: crate::ReduceKind::Sum,
        }),
        Some(ty),
        vec![],
    );
    let value = UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, ty);
    let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, value]);
    UOp::sink(vec![UOp::from_operation(
        Operation::ReduceFinalize,
        Some(ty),
        vec![update],
    )])
}

#[test]
fn unchanged_normalization_reuses_tables_but_repeats_every_validation_gate() {
    let ty = UType::scalar(DType::F32);
    let ordinary = UOp::from_operation(
        Operation::GraphUnary(crate::UnaryOp::Neg),
        Some(ty),
        vec![UOp::scalar_constant(DType::F32, 0x8000_0000, ty)],
    );
    let projected = projected_kernel();
    let mut both = projected_kernel().sources().to_vec();
    both.push(integer_fold());
    for (name, root, saved_walks, expected_validations, unchanged) in [
        ("empty", UOp::sink(vec![]), 2, 3, true),
        ("float no-op", UOp::sink(vec![ordinary]), 2, 3, true),
        ("reduction", reduction_kernel(), 1, 2, true),
        (
            "builtin change",
            UOp::sink(vec![integer_fold()]),
            1,
            3,
            false,
        ),
        ("projected change", projected, 1, 3, false),
        ("both changes", UOp::sink(both), 0, 3, false),
    ] {
        let walks = canonical_traversal_count();
        let validations = ordered_validation_count();
        let actual = normalize_kernel_with_table(&root).unwrap();
        let actual_walks = canonical_traversal_count() - walks;
        let actual_validations = ordered_validation_count() - validations;
        let walks = canonical_traversal_count();
        let validations = ordered_validation_count();
        let reference = released_table_normalization(&root).unwrap();
        assert_eq!(
            canonical_traversal_count() - walks,
            actual_walks + saved_walks,
            "{name}"
        );
        assert_eq!(actual_validations, expected_validations, "{name}");
        assert_eq!(
            ordered_validation_count() - validations,
            expected_validations,
            "{name}"
        );
        assert_eq!(
            actual.order().root().shares_node_with(&root),
            unchanged,
            "{name}"
        );
        assert_eq!(actual.order().nodes(), reference.order().nodes(), "{name}");
        for node in 0..actual.order().nodes().len() {
            assert_eq!(
                actual.sources(node),
                reference.sources(node),
                "{name}: {node}"
            );
        }
        assert_eq!(
            artifact::encode_schedule_identity(actual.order().root()).unwrap(),
            artifact::encode_schedule_identity(reference.order().root()).unwrap(),
            "{name}"
        );
    }
}

#[test]
fn normalization_tables_preserve_failure_gates_and_release_owners() {
    let ty = UType::scalar(DType::I32);
    let malformed = UOp::from_operation(Operation::Unary(Unary::Neg), Some(ty), vec![]);
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(UType::scalar(DType::I64)),
        vec![UOp::constant(2, UType::scalar(DType::I64))],
    );
    for (name, root, expected) in [
        (
            "malformed",
            UOp::sink(vec![malformed]),
            UOpError::InvalidArity {
                expected: "one source",
                actual: 0,
            },
        ),
        (
            "open control",
            UOp::sink(vec![range]),
            UOpError::UnclosedControl,
        ),
        (
            "not a kernel",
            UOp::constant(7, ty),
            UOpError::EffectRewrite,
        ),
    ] {
        let validations = ordered_validation_count();
        let actual = normalize_kernel_with_table(&root).err().unwrap();
        let actual_validations = ordered_validation_count() - validations;
        let validations = ordered_validation_count();
        let reference = released_table_normalization(&root).err().unwrap();
        assert_eq!(actual, expected, "{name}");
        assert_eq!(actual, reference, "{name}");
        assert_eq!(
            ordered_validation_count() - validations,
            actual_validations,
            "{name}"
        );
    }
    for root in [UOp::sink(vec![]), UOp::sink(vec![integer_fold()])] {
        let order = normalize_kernel_with_table(&root).unwrap();
        let probes = order
            .order()
            .nodes()
            .iter()
            .map(UOp::allocation_liveness_probe)
            .collect::<Vec<_>>();
        drop(root);
        assert!(probes.iter().all(|probe| probe()));
        drop(order);
        assert!(probes.iter().all(|probe| !probe()));
    }
}

#[test]
fn retained_normalization_preserves_bytes_and_rejections() {
    let i32_type = UType::scalar(DType::I32);
    let f32_type = UType::scalar(DType::F32);
    let seven = UOp::scalar_constant(DType::I32, 7, i32_type);
    let zero = UOp::scalar_constant(DType::I32, 0, i32_type);
    let folded = UOp::from_operation(
        Operation::GraphBinary(crate::BinaryOp::Add),
        Some(i32_type),
        vec![seven.clone(), zero],
    );
    let nan = UOp::scalar_constant(DType::F32, 0x7fc0_1234, f32_type);
    let cast = UOp::cast(nan, f32_type);
    let malformed = UOp::from_operation(
        Operation::Binary(Binary::Add),
        Some(i32_type),
        vec![seven.clone()],
    );
    let condition = UOp::scalar_constant(DType::Bool, 1, UType::scalar(DType::Bool));
    let if_ = UOp::from_operation(Operation::If, None, vec![condition]);
    let end_if = UOp::from_operation(Operation::EndIf, None, vec![if_]);

    let reduction = reduction_kernel().sources()[0].clone();

    for (name, root) in [
        ("empty", UOp::sink(vec![])),
        ("integer fold", UOp::sink(vec![folded.clone()])),
        ("raw NaN cast", UOp::sink(vec![cast])),
        ("reduction", UOp::sink(vec![reduction.clone()])),
        (
            "duplicate reduction",
            UOp::sink(vec![reduction.clone(), reduction]),
        ),
        ("malformed ALU", UOp::sink(vec![malformed])),
        ("effect", UOp::sink(vec![folded, end_if])),
        ("non-Sink", seven),
    ] {
        let expected = independent_normalization(&root);
        match normalize_kernel_with_order(&root) {
            Ok(order) => {
                let expected = expected.unwrap();
                assert_eq!(order.root(), &expected, "{name}");
                assert_eq!(order.nodes(), expected.topological().unwrap(), "{name}");
                assert_eq!(order.validate_for(order.root()), Ok(()), "{name}");
                assert_eq!(
                    artifact::encode_schedule_identity(order.root()).unwrap(),
                    artifact::encode_schedule_identity(&expected).unwrap(),
                    "{name}"
                );
                assert_eq!(
                    normalize_kernel(&root).unwrap(),
                    order.into_root(),
                    "{name}"
                );
            }
            Err(error) => assert_eq!(expected.unwrap_err(), error, "{name}"),
        }
    }
}

fn projected_kernel() -> UOp {
    let i64_type = UType::scalar(DType::I64);
    let f32_type = UType::scalar(DType::F32);
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(i64_type),
        vec![UOp::constant(8, i64_type)],
    );
    let four = UOp::constant(4, i64_type);
    let quotient = UOp::binary(Binary::FloorDiv, range.clone(), four.clone());
    let remainder = UOp::binary(Binary::Mod, range.clone(), four.clone());
    let expression = UOp::binary(
        Binary::Add,
        remainder,
        UOp::binary(Binary::Mul, four, quotient),
    );
    let address = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: crate::AddressSpace::Global,
            name: "b7".into(),
            element: f32_type,
        }),
        Some(f32_type),
        vec![],
    );
    let index = UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer: 7,
            elements: 8,
            input_shape: crate::Shape::from([8]),
            output_shape: crate::Shape::from([8]),
            addressing: crate::IndexAddressing::Projected,
        }),
        Some(f32_type),
        vec![address, expression],
    );
    UOp::sink(vec![
        UOp::from_operation(Operation::Load, Some(f32_type), vec![index]),
        UOp::from_operation(Operation::EndRange, None, vec![range]),
    ])
}

#[test]
fn retained_order_tracks_the_rewritten_projected_index() {
    let root = projected_kernel();
    let original_bytes = artifact::encode_schedule_identity(&root).unwrap();
    let expected = independent_normalization(&root).unwrap();
    let order = normalize_kernel_with_order(&root).unwrap();
    assert_ne!(order.root(), &root);
    assert_eq!(order.root(), &expected);
    assert_eq!(order.nodes(), expected.topological().unwrap());
    assert_eq!(order.validate_for(&root), Err(UOpError::InvalidArgument));
    let index = order
        .nodes()
        .iter()
        .find(|node| crate::projected_index::ProjectedIndexPlan::is_projected(node))
        .unwrap();
    assert_eq!(
        crate::projected_index::ProjectedIndexPlan::from_index(index)
            .unwrap()
            .expression,
        crate::projected_index::ProjectedExpr::Linear
    );
    assert_eq!(
        artifact::encode_schedule_identity(order.root()).unwrap(),
        artifact::encode_schedule_identity(&expected).unwrap()
    );
    assert_eq!(
        artifact::encode_schedule_identity(&root).unwrap(),
        original_bytes
    );
}
