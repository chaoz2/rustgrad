use super::*;
use crate::uop::{
    AddressSpace, AddressValue, Binary, IndexAddressing, IndexValue, UOpTag, UType, Unary,
};
use crate::{DType, Shape};
use std::collections::BTreeSet;

// The previous all-edge summary is an independent admission oracle for small
// graphs. Do not use it for the shared-diamond bounded-work regression below.
fn previous_summary(nodes: &[UOp]) -> Result<(), UOpError> {
    let mut consumers = HashMap::<UOp, bool>::new();
    for consumer in nodes {
        let is_load = matches!(consumer.operation(), Operation::Load);
        for source in consumer.sources() {
            consumers
                .entry(source.clone())
                .and_modify(|only_loads| *only_loads &= is_load)
                .or_insert(is_load);
        }
    }
    for index in nodes
        .iter()
        .filter(|node| ProjectedIndexPlan::is_projected(node))
    {
        if consumers.get(index) != Some(&true) {
            return Err(UOpError::InvalidIndex);
        }
    }
    Ok(())
}

fn previous_validation(root: &UOp) -> Result<(), UOpError> {
    let nodes = root.topological()?;
    let mut ranges = BTreeSet::new();
    let mut ifs = Vec::new();
    for node in &nodes {
        crate::uop::validate_one(node, &mut ranges, &mut ifs)?;
    }
    previous_summary(&nodes)?;
    if !ifs.is_empty() || !ranges.is_empty() {
        return Err(UOpError::UnclosedControl);
    }
    crate::uop::validate_reduction_topology(root)
}

fn index(buffer: u64, addressing: IndexAddressing, extent: usize) -> UOp {
    let ty = UType::scalar(DType::F32);
    let address = UOp::from_operation(
        Operation::DefineGlobal(AddressValue {
            space: AddressSpace::Global,
            name: format!("b{buffer}"),
            element: ty,
        }),
        Some(ty),
        vec![],
    );
    let coordinate_type = UType::scalar(DType::I64);
    let zero = UOp::constant(0, coordinate_type);
    let coordinate = if extent == 0 && addressing != IndexAddressing::Broadcast {
        let range = UOp::from_operation(
            Operation::Range(0),
            Some(coordinate_type),
            vec![zero.clone()],
        );
        UOp::binary(Binary::Mul, range, zero)
    } else {
        zero
    };
    let mut sources = vec![address, coordinate];
    if addressing == IndexAddressing::Predicated {
        sources.push(UOp::scalar_constant(
            DType::Bool,
            1,
            UType::scalar(DType::Bool),
        ));
    }
    UOp::from_operation(
        Operation::Index(IndexValue::Buffer {
            buffer,
            elements: 1,
            input_shape: Shape::from([1]),
            output_shape: Shape::from([extent]),
            addressing,
        }),
        Some(ty),
        sources,
    )
}

fn load(index: UOp) -> UOp {
    UOp::from_operation(Operation::Load, index.ty(), vec![index])
}

fn store(index: UOp) -> UOp {
    let value = UOp::scalar_constant(DType::F32, 0, UType::scalar(DType::F32));
    UOp::from_operation(Operation::Store, None, vec![index, value])
}

#[test]
fn projected_admission_matches_previous_summary_and_validation() {
    for addressing in [IndexAddressing::Projected, IndexAddressing::Predicated] {
        for extent in [0, 1] {
            let first = index(0, addressing, extent);
            let equal = index(0, addressing, extent);
            assert_eq!(first, equal);
            assert!(!first.shares_node_with(&equal));
            let cases = [
                ("empty", UOp::sink(vec![]), Ok(())),
                ("load", load(first.clone()), Ok(())),
                ("unused", first.clone(), Err(UOpError::InvalidIndex)),
                ("store", store(first.clone()), Err(UOpError::InvalidIndex)),
                (
                    "arithmetic",
                    UOp::unary(Unary::Neg, first.clone()),
                    Err(UOpError::InvalidIndex),
                ),
                (
                    "equal nonload",
                    UOp::sink(vec![load(first.clone()), store(equal)]),
                    Err(UOpError::InvalidIndex),
                ),
                (
                    "two loads",
                    UOp::sink(vec![
                        load(first.clone()),
                        load(index(1, addressing, extent)),
                    ]),
                    Ok(()),
                ),
                (
                    "ordinary write",
                    UOp::sink(vec![
                        load(first),
                        store(index(1, IndexAddressing::Broadcast, extent)),
                    ]),
                    Ok(()),
                ),
            ];
            for (name, root, expected) in cases {
                // A zero-domain projected coordinate needs its real empty
                // range, closed only after the index's consumers. Constant
                // zero alone has a nonempty coordinate bound and is invalid.
                let ends = root
                    .topological()
                    .unwrap()
                    .into_iter()
                    .filter(|node| matches!(node.operation(), Operation::Range(_)))
                    .map(|range| UOp::from_operation(Operation::EndRange, None, vec![range]))
                    .collect::<Vec<_>>();
                let root = if ends.is_empty() {
                    root
                } else {
                    UOp::sink(std::iter::once(root).chain(ends).collect())
                };
                let nodes = root.topological().unwrap();
                assert_eq!(
                    validate(&nodes),
                    expected,
                    "{name}, {addressing:?}, {extent}"
                );
                assert_eq!(validate(&nodes), previous_summary(&nodes), "{name}");
                assert_eq!(root.validate(), previous_validation(&root), "{name}");
                assert_eq!(root.validate(), expected, "full {name}");
            }
        }
    }

    // Structural admission must not collapse different tag metadata or claim
    // missing-source semantic errors are projected-consumer failures.
    let plain = index(0, IndexAddressing::Projected, 1);
    let tagged = plain.retag(Some(UOpTag::Int(7)));
    let nodes = [plain.clone(), load(tagged)];
    assert_eq!(validate(&nodes), previous_summary(&nodes));
    assert_eq!(validate(&nodes), Err(UOpError::InvalidIndex));
    let active = index(0, IndexAddressing::Predicated, 1);
    let mut sources = active.sources().to_vec();
    sources[2] = UOp::scalar_constant(DType::Bool, 0, UType::scalar(DType::Bool));
    let inactive = UOp::from_operation(active.operation().clone(), active.ty(), sources);
    for (root, expected) in [
        (load(inactive.clone()), Ok(())),
        (store(inactive), Err(UOpError::InvalidIndex)),
    ] {
        assert_eq!(root.validate(), previous_validation(&root));
        assert_eq!(root.validate(), expected);
    }
    let malformed = UOp::from_operation(
        Operation::Unary(Unary::Neg),
        Some(UType::scalar(DType::F32)),
        vec![],
    );
    let root = UOp::sink(vec![plain, malformed]);
    assert_eq!(root.validate(), previous_validation(&root));
    assert!(matches!(
        root.validate(),
        Err(UOpError::InvalidArity { .. })
    ));
}

#[test]
fn ordinary_shared_diamond_avoids_recursive_structural_hashing() {
    let mut owners = vec![UOp::constant(1, UType::scalar(DType::I32))];
    for _ in 0..128 {
        let source = owners.last().unwrap().clone();
        owners.push(UOp::binary(Binary::Add, source.clone(), source));
    }
    let root = owners.last().unwrap().clone();
    let nodes = root.topological().unwrap();
    assert_eq!(nodes.len(), 129);
    assert_eq!(validate(&nodes), Ok(()));
    assert_eq!(root.validate(), Ok(()));
    // Isolate consumer admission from Arc's independent destruction depth.
    drop(root);
    owners.into_iter().rev().for_each(drop);
    nodes.into_iter().rev().for_each(drop);
}

#[test]
fn large_projected_consumer_summary_uses_one_edge_pass() {
    let mut nodes = Vec::with_capacity(8193);
    let mut first = None;
    for buffer in 0..4096_u64 {
        let index = index(buffer, IndexAddressing::Projected, 1);
        if first.is_none() {
            first = Some(index.clone());
        }
        nodes.push(index.clone());
        nodes.push(load(index));
    }
    assert_eq!(validate(&nodes), Ok(()));
    nodes.push(store(first.unwrap()));
    assert_eq!(validate(&nodes), Err(UOpError::InvalidIndex));
}
