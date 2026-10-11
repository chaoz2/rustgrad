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

    let init = UOp::from_operation(
        Operation::ReduceInit(crate::ReductionValue {
            input_shape: crate::Shape::from([2]),
            output_shape: crate::Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind: crate::ReduceKind::Sum,
        }),
        Some(f32_type),
        vec![],
    );
    let value = UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, f32_type);
    let update = UOp::from_operation(
        Operation::ReduceAccumulate,
        Some(f32_type),
        vec![init, value],
    );
    let reduction = UOp::from_operation(Operation::ReduceFinalize, Some(f32_type), vec![update]);

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

#[test]
fn retained_order_tracks_the_rewritten_projected_index() {
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
    let root = UOp::sink(vec![
        UOp::from_operation(Operation::Load, Some(f32_type), vec![index]),
        UOp::from_operation(Operation::EndRange, None, vec![range]),
    ]);
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
