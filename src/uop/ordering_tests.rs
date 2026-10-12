use super::*;
use std::cell::Cell;
use std::hash::{DefaultHasher, Hash, Hasher};

// This oracle deliberately bypasses UOp::Ord, Vec<UOp>::cmp and pointer
// equality. It follows the released UOpNode declaration's field order.
pub(super) fn previous_cmp(lhs: &UOp, rhs: &UOp) -> Ordering {
    lhs.operation()
        .cmp(rhs.operation())
        .then_with(|| lhs.ty().cmp(&rhs.ty()))
        .then_with(|| {
            for (lhs, rhs) in lhs.sources().iter().zip(rhs.sources()) {
                let order = previous_cmp(lhs, rhs);
                if order != Ordering::Equal {
                    return order;
                }
            }
            lhs.sources().len().cmp(&rhs.sources().len())
        })
        .then_with(|| lhs.tag().cmp(&rhs.tag()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ReferenceKey(pub(super) UOp);

impl Ord for ReferenceKey {
    fn cmp(&self, rhs: &Self) -> Ordering {
        previous_cmp(&self.0, &rhs.0)
    }
}

impl PartialOrd for ReferenceKey {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        Some(self.cmp(rhs))
    }
}

thread_local! {
    static FALLBACKS: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn record_fallback() {
    if let Some(count) = FALLBACKS.get() {
        FALLBACKS.set(Some(count + 1));
    }
}

fn count_fallbacks<T>(run: impl FnOnce() -> T) -> (T, usize) {
    struct Reset(Option<usize>);
    impl Drop for Reset {
        fn drop(&mut self) {
            FALLBACKS.set(self.0);
        }
    }
    let _reset = Reset(FALLBACKS.replace(Some(0)));
    let result = run();
    (result, FALLBACKS.get().unwrap())
}

fn corpus() -> Vec<UOp> {
    let ty = UType::scalar(DType::I32);
    let one = UOp::constant(1, ty);
    let two = UOp::constant(2, ty);
    let mut nodes = vec![
        one.clone(),
        one.clone(),
        UOp::constant(1, ty),
        two.clone(),
        UOp::constant(1, UType::scalar(DType::I64)),
        UOp::constant(1, UType::vector(DType::I32, 4).unwrap()),
        UOp::from_operation(Operation::Const(LiteralValue::Int(1)), None, vec![]),
        one.retag(Some(UOpTag::Int(7))),
        one.retag(Some(UOpTag::Tuple(vec![
            UOpTag::Text("training".into()),
            UOpTag::DType(DType::F32),
        ]))),
        UOp::binary(Binary::Add, one.clone(), two.clone()),
        UOp::binary(Binary::Add, two, one),
        UOp::sink(vec![]),
        UOp::from_operation(
            Operation::DefineVar(VariableValue {
                name: "guard".into(),
                bounds: SymbolicExpr::constant(8),
            }),
            Some(UType::scalar(DType::I64)),
            vec![],
        ),
        UOp::from_operation(
            Operation::DefineVar(VariableValue {
                name: "guard".into(),
                bounds: SymbolicExpr::constant(8),
            }),
            Some(UType::scalar(DType::I64)),
            vec![],
        ),
        UOp::from_operation(
            Operation::ReduceInit(ReductionValue {
                input_shape: Shape::from([2]),
                output_shape: Shape::new([]),
                axes: vec![0],
                keepdim: false,
                kind: crate::ReduceKind::Sum,
            }),
            Some(UType::scalar(DType::F32)),
            vec![],
        ),
    ];
    nodes.extend(
        [0, 0x8000_0000, 0x7f80_0000, 0x7fc0_1234, 0x7fc0_5678]
            .map(|bits| UOp::scalar_constant(DType::F32, bits, UType::scalar(DType::F32))),
    );
    for index in 0..24 {
        let node = UOp::from_operation(
            Operation::Binary(Binary::Sub),
            Some(ty),
            vec![
                nodes[index % nodes.len()].clone(),
                nodes[(index * 7 + 1) % nodes.len()].clone(),
            ],
        );
        let equal =
            UOp::from_operation(node.operation().clone(), node.ty(), node.sources().to_vec());
        nodes.extend([node, equal]);
    }
    nodes
}

fn hash(node: &UOp) -> u64 {
    let mut state = DefaultHasher::new();
    node.hash(&mut state);
    state.finish()
}

#[test]
fn shared_ordering_matches_independent_field_order_and_container_laws() {
    let nodes = corpus();
    for lhs in &nodes {
        for rhs in &nodes {
            let expected = previous_cmp(lhs, rhs);
            assert_eq!(lhs.cmp(rhs), expected);
            assert_eq!(lhs.partial_cmp(rhs), Some(expected));
            assert_eq!(expected == Ordering::Equal, lhs == rhs);
            assert_eq!(lhs.cmp(rhs), rhs.cmp(lhs).reverse());
            if lhs == rhs {
                assert_eq!(hash(lhs), hash(rhs));
            }
        }
    }
    // Explicit conflicts prove priority rather than only independent equality.
    let ty = UType::scalar(DType::I32);
    let parent =
        |value, tag| UOp::sink(vec![UOp::constant(value, ty)]).retag(Some(UOpTag::Int(tag)));
    assert_eq!(parent(1, 99).cmp(&parent(2, -99)), Ordering::Less);
    let short = UOp::sink(vec![nodes[0].clone()]).retag(Some(UOpTag::Int(99)));
    let long = UOp::sink(vec![nodes[0].clone(), nodes[0].clone()]).retag(Some(UOpTag::Int(-99)));
    assert_eq!(short.cmp(&long), Ordering::Less);
    let conflict = |operation, ty, value| {
        UOp::from_operation(
            operation,
            ty,
            vec![UOp::constant(value, UType::scalar(DType::I32))],
        )
    };
    assert_eq!(
        conflict(
            Operation::Binary(Binary::Add),
            Some(UType::scalar(DType::I64)),
            99
        )
        .cmp(&conflict(Operation::Binary(Binary::Sub), None, -99)),
        Ordering::Less,
    );
    assert_eq!(
        conflict(Operation::Binary(Binary::Add), None, 99).cmp(&conflict(
            Operation::Binary(Binary::Add),
            Some(ty),
            -99
        )),
        Ordering::Less,
    );
    for lhs in &nodes[..12] {
        for middle in &nodes[..12] {
            for rhs in &nodes[..12] {
                if lhs <= middle && middle <= rhs {
                    assert!(lhs <= rhs);
                }
            }
        }
    }
    let mut actual_set = BTreeSet::new();
    let mut reference_set = BTreeSet::new();
    let mut actual_map = BTreeMap::new();
    let mut reference_map = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        actual_set.insert(node.clone());
        reference_set.insert(ReferenceKey(node.clone()));
        actual_map.entry(node.clone()).or_insert(index);
        reference_map
            .entry(ReferenceKey(node.clone()))
            .or_insert(index);
    }
    assert_eq!(actual_set.len(), reference_set.len());
    assert_eq!(actual_map.len(), reference_map.len());
    for (actual, reference) in actual_set.iter().zip(&reference_set) {
        assert!(actual.shares_node_with(&reference.0));
    }
    for node in &nodes {
        let copy =
            UOp::from_operation(node.operation().clone(), node.ty(), node.sources().to_vec())
                .retag(node.tag().cloned());
        assert_eq!(
            actual_map.get(&copy),
            reference_map.get(&ReferenceKey(copy))
        );
    }
}

fn chain(depth: usize, value: i64) -> Vec<UOp> {
    let mut owners = vec![UOp::constant(value, UType::scalar(DType::I32))];
    for _ in 0..depth {
        owners.push(UOp::unary(Unary::Neg, owners.last().unwrap().clone()));
    }
    owners
}

#[test]
fn shared_ordering_shortcuts_only_owned_aliases_without_retention() {
    let mut owners = chain(2048, 1);
    let root = owners.last().unwrap().clone();
    let weak = Arc::downgrade(&root.0);
    let count = Arc::strong_count(&root.0);
    let ((order, partial), fallbacks) =
        count_fallbacks(|| (root.cmp(&root), root.partial_cmp(&root)));
    assert_eq!((order, partial), (Ordering::Equal, Some(Ordering::Equal)));
    assert_eq!(fallbacks, 0);
    assert_eq!(Arc::strong_count(&root.0), count);
    let copy = UOp::from_operation(root.operation().clone(), root.ty(), root.sources().to_vec());
    assert!(!root.shares_node_with(&copy));
    let (order, fallbacks) = count_fallbacks(|| root.cmp(&copy));
    assert_eq!(order, Ordering::Equal);
    assert_eq!(fallbacks, 1, "independent parent still compares its fields");
    // Independent deep chains retain the released recursive boundary; no
    // unbounded-depth guarantee follows from a shared-allocation shortcut.
    let mut left = chain(192, 1);
    let mut equal = chain(192, 1);
    let mut different = chain(192, 2);
    for rhs in [equal.last().unwrap(), different.last().unwrap()] {
        let lhs = left.last().unwrap();
        let (order, fallbacks) = count_fallbacks(|| lhs.cmp(rhs));
        assert_eq!(order, previous_cmp(lhs, rhs));
        assert!(fallbacks > 1);
    }
    drop(copy);
    drop(root);
    while owners.pop().is_some() {}
    while left.pop().is_some() {}
    while equal.pop().is_some() {}
    while different.pop().is_some() {}
    assert!(weak.upgrade().is_none());
}
