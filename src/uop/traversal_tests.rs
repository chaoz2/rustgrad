use super::*;
use crate::uop::{Binary, LiteralValue, Operation, ReductionValue, UOpTag, UType, Unary, artifact};
use crate::{DType, Shape};
use std::hash::{BuildHasherDefault, DefaultHasher};

#[derive(Default)]
struct CollidingHasher;

impl Hasher for CollidingHasher {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, _bytes: &[u8]) {}
}

fn recursive_reference(root: &UOp) -> Result<Vec<UOp>, UOpError> {
    fn visit(
        node: &UOp,
        seen: &mut BTreeSet<UOp>,
        active: &mut BTreeSet<UOp>,
        out: &mut Vec<UOp>,
    ) -> Result<(), UOpError> {
        if seen.contains(node) {
            return Ok(());
        }
        if !active.insert(node.clone()) {
            return Err(UOpError::Cycle);
        }
        for source in node.sources() {
            visit(source, seen, active, out)?;
        }
        active.remove(node);
        seen.insert(node.clone());
        out.push(node.clone());
        Ok(())
    }
    let mut out = Vec::new();
    visit(root, &mut BTreeSet::new(), &mut BTreeSet::new(), &mut out)?;
    Ok(out)
}

fn assert_reference_order(root: &UOp) {
    let expected = recursive_reference(root).unwrap();
    let reference_order = crate::uop::TopologicalOrder {
        root: root.clone(),
        nodes: expected.clone(),
    };
    let reference_bytes = artifact::encode_schedule_identity_with_order(root, &reference_order);
    let assert_order = |actual: Vec<UOp>| {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.shares_node_with(expected), "representative {index}");
        }
        let order = crate::uop::TopologicalOrder {
            root: root.clone(),
            nodes: actual,
        };
        assert_eq!(
            artifact::encode_schedule_identity_with_order(root, &order),
            reference_bytes
        );
    };
    for _ in 0..4 {
        assert_order(topological_with_hasher(root, RandomState::new()).unwrap());
    }
    assert_order(
        topological_with_hasher(root, BuildHasherDefault::<CollidingHasher>::default()).unwrap(),
    );
    assert_eq!(reference_bytes, artifact::encode_schedule_identity(root));
    assert_eq!(root.topological().unwrap(), expected);
}

#[test]
fn canonical_order_preserves_sharing_tags_types_and_operand_order() {
    let ty = UType::scalar(DType::I32);
    let first = UOp::constant(1, ty);
    let equivalent = UOp::constant(1, ty);
    assert!(!first.shares_node_with(&equivalent));
    let second = UOp::constant(2, ty);
    let lhs = UOp::binary(Binary::Add, first.clone(), second.clone());
    let same_lhs = UOp::binary(Binary::Add, equivalent.clone(), second.clone());
    let reversed = UOp::binary(Binary::Add, second, first.clone());
    let tagged = equivalent.retag(Some(UOpTag::Tuple(vec![
        UOpTag::Int(7),
        UOpTag::Text("optimizer".into()),
    ])));
    let different_type = UOp::constant(1, UType::scalar(DType::I64));
    let floats = [0, 0x8000_0000, 0x7fc0_1234, 0x7fc0_5678]
        .map(|bits| UOp::scalar_constant(DType::F32, bits, UType::scalar(DType::F32)));
    let mut sources = vec![
        lhs.clone(),
        same_lhs,
        reversed,
        first.clone(),
        tagged,
        different_type,
    ];
    sources.extend(floats);
    let root = UOp::sink(sources);
    let before = artifact::encode(&root).unwrap();
    assert_reference_order(&root);
    let untagged = UOp::sink(
        root.sources()
            .iter()
            .filter(|node| node.tag().is_none())
            .cloned()
            .collect(),
    );
    assert_reference_order(&untagged);
    let order = topological(&root).unwrap();
    assert!(order[0].shares_node_with(&first));
    assert!(order[2].shares_node_with(&lhs));
    assert_eq!(order.iter().filter(|node| **node == lhs).count(), 1);
    assert_eq!(artifact::encode(&root).unwrap(), before);
    let decoded = artifact::decode(&before).unwrap();
    assert_reference_order(&decoded);
    assert_eq!(artifact::encode(&decoded).unwrap(), before);

    // Traversal describes even malformed nodes; semantic admission remains
    // the validator's responsibility and must keep its existing typed error.
    let malformed = UOp::from_operation(Operation::Unary(Unary::Neg), Some(ty), vec![]);
    assert_reference_order(&malformed);
    assert!(matches!(
        malformed.validate(),
        Err(UOpError::InvalidArity { .. })
    ));
    assert_reference_order(&UOp::sink(vec![]));
}

#[test]
fn generated_dags_preserve_first_representatives() {
    let ty = UType::scalar(DType::I32);
    for seed in 1u64..=32 {
        let mut state = seed;
        let mut nodes = vec![UOp::constant(1, ty), UOp::constant(2, ty)];
        for step in 0..16 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let left = nodes[(state as usize) % nodes.len()].clone();
            let right = nodes[((state >> 32) as usize) % nodes.len()].clone();
            let node = match step % 4 {
                0 => UOp::binary(Binary::Add, left, right),
                1 => UOp::unary(Unary::Neg, left),
                2 => UOp::from_operation(
                    left.operation().clone(),
                    left.ty(),
                    left.sources().to_vec(),
                )
                .retag(left.tag().cloned()),
                _ => left.retag(Some(UOpTag::Int(step))),
            };
            nodes.push(node);
        }
        let root = UOp::sink(nodes);
        assert_reference_order(&root);
        let bytes = artifact::encode(&root).unwrap();
        assert_eq!(
            artifact::encode(&artifact::decode(&bytes).unwrap()).unwrap(),
            bytes
        );
    }
}

#[test]
fn shallow_key_hash_matches_every_equality_field() {
    let ty = Some(UType::scalar(DType::F32));
    let operation = Operation::Const(LiteralValue::Scalar {
        dtype: DType::F32,
        bits: 0x7fc0_1234,
    });
    let key = |operation, ty, sources, tag| NodeKey {
        representative: UOp::from_operation(operation, ty, vec![]).retag(tag),
        sources,
    };
    let hash = |key: &NodeKey| {
        let mut state = DefaultHasher::new();
        key.hash(&mut state);
        state.finish()
    };
    let original = key(operation.clone(), ty, vec![1, 2], None);
    let cases = [
        ("same", operation.clone(), ty, vec![1, 2], None, true),
        (
            "payload bits",
            Operation::Const(LiteralValue::Scalar {
                dtype: DType::F32,
                bits: 0x7fc0_5678,
            }),
            ty,
            vec![1, 2],
            None,
            false,
        ),
        (
            "absent type",
            operation.clone(),
            None,
            vec![1, 2],
            None,
            false,
        ),
        (
            "lane width",
            operation.clone(),
            Some(UType::vector(DType::F32, 2).unwrap()),
            vec![1, 2],
            None,
            false,
        ),
        (
            "source order",
            operation.clone(),
            ty,
            vec![2, 1],
            None,
            false,
        ),
        (
            "nested tag",
            operation,
            ty,
            vec![1, 2],
            Some(UOpTag::Tuple(vec![
                UOpTag::DType(DType::F32),
                UOpTag::Int(7),
            ])),
            false,
        ),
    ];
    for (name, operation, ty, sources, tag, equal) in cases {
        let first = key(operation.clone(), ty, sources.clone(), tag.clone());
        let independent = key(operation, ty, sources, tag);
        assert!(
            !first
                .representative
                .shares_node_with(&independent.representative)
        );
        assert!(first == independent, "independent {name}");
        assert_eq!(hash(&first), hash(&independent), "hash {name}");
        assert_eq!(original == first, equal, "equality {name}");
    }
}

#[test]
fn interner_releases_owners_after_returned_order_is_dropped() {
    let ty = UType::scalar(DType::I32);
    let first = UOp::constant(1, ty);
    let equal = UOp::constant(1, ty);
    let root = UOp::sink(vec![first.clone(), equal.clone()]);
    let weak = [&first, &equal, &root].map(|node| std::sync::Arc::downgrade(&node.0));
    let order = topological(&root).unwrap();
    assert_eq!(order.len(), 2);
    assert!(order[0].shares_node_with(&first));
    drop(first);
    drop(equal);
    drop(root);
    assert!(weak.iter().all(|node| node.upgrade().is_some()));
    drop(order);
    assert!(weak.iter().all(|node| node.upgrade().is_none()));
}

#[test]
fn hashed_order_preserves_control_and_reduction_admission() {
    let index_ty = UType::scalar(DType::I64);
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(index_ty),
        vec![UOp::constant(0, index_ty)],
    );
    let end = UOp::from_operation(Operation::EndRange, None, vec![range.clone()]);
    let closed = UOp::sink(vec![range.clone(), end]);
    let ty = UType::scalar(DType::F32);
    let reduction = || {
        let init = UOp::from_operation(
            Operation::ReduceInit(ReductionValue {
                input_shape: Shape::from([2]),
                output_shape: Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Max,
            }),
            Some(ty),
            vec![],
        );
        let source = UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, ty);
        let update = UOp::from_operation(Operation::ReduceAccumulate, Some(ty), vec![init, source]);
        UOp::from_operation(Operation::ReduceFinalize, Some(ty), vec![update])
    };
    for (name, root, expected) in [
        ("closed empty range", closed, Ok(())),
        ("unclosed range", range, Err(UOpError::UnclosedControl)),
        ("reduction", reduction(), Ok(())),
        (
            "distinct equal reduction owners",
            UOp::sink(vec![reduction(), reduction()]),
            Err(UOpError::InvalidArgument),
        ),
    ] {
        assert_reference_order(&root);
        assert_eq!(root.validate(), expected, "{name}");
        if expected.is_ok() {
            let bytes = artifact::encode(&root).unwrap();
            assert_eq!(
                artifact::encode(&artifact::decode(&bytes).unwrap()).unwrap(),
                bytes
            );
        }
    }
}

#[test]
fn deep_chains_and_shared_diamonds_use_bounded_work() {
    let ty = UType::scalar(DType::I32);
    let mut owners = vec![UOp::constant(1, ty)];
    for _ in 0..4096 {
        owners.push(UOp::unary(Unary::Neg, owners.last().unwrap().clone()));
    }
    let root = owners.last().unwrap().clone();
    let order = topological(&root).unwrap();
    assert_eq!(order.len(), owners.len());
    for (node, owner) in order.iter().zip(&owners) {
        assert!(node.shares_node_with(owner));
    }
    // Keep all nodes owned while releasing deepest parents first. This test
    // targets traversal depth, independently of Arc's recursive destruction.
    drop(root);
    owners.into_iter().rev().for_each(drop);
    order.into_iter().rev().for_each(drop);

    let mut owners = vec![UOp::constant(1, ty)];
    for _ in 0..128 {
        let source = owners.last().unwrap().clone();
        owners.push(UOp::binary(Binary::Add, source.clone(), source));
    }
    let root = owners.last().unwrap();
    let order = topological(root).unwrap();
    assert_eq!(order.len(), 129);
    for (node, owner) in order.iter().zip(&owners) {
        assert!(node.shares_node_with(owner));
    }
    assert!(matches!(
        order[0].operation(),
        Operation::Const(LiteralValue::Int(1))
    ));
}
