use super::*;
use crate::{Backend, Operation, Scalar, TensorData, UType};
use std::cell::Cell;

thread_local! {
    static LEGACY_FINALIZATION: Cell<bool> = const { Cell::new(false) };
    static FINALIZATIONS: Cell<usize> = const { Cell::new(0) };
}

// Independent previous finalization: normalize the accepted root again rather
// than trust the retained first-pass table. Only focused tests select it.
pub(super) fn finalize(table: crate::uop::CanonicalNodeTable) -> crate::uop::CanonicalNodeTable {
    FINALIZATIONS.set(FINALIZATIONS.get() + 1);
    if LEGACY_FINALIZATION.get() {
        crate::uop::normalize_kernel_with_table(table.order().root()).unwrap()
    } else {
        table
    }
}

fn with_legacy_finalization<T>(run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            LEGACY_FINALIZATION.set(self.0);
        }
    }
    let _reset = Reset(LEGACY_FINALIZATION.replace(true));
    run()
}

fn contiguous(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [1, 0]).unwrap();
    graph.contiguous(view).unwrap()
}

fn aliases(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let shared = graph.permute(producer, [1, 0]).unwrap();
    let first = graph.reshape(shared, [1, 3, 2]).unwrap();
    let second = graph.reshape(shared, [3, 2, 1]).unwrap();
    let second = graph.permute(second, [2, 0, 1]).unwrap();
    graph.add(first, second).unwrap()
}

fn projected(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph.permute(producer, [0, 2, 1, 3]).unwrap();
    let view = graph.reshape(view, [1, 2, 4]).unwrap();
    let lhs = graph.relu(view).unwrap();
    let rhs = graph.square(view).unwrap();
    graph.add(lhs, rhs).unwrap()
}

fn padded(graph: &mut Graph, input: NodeId) -> NodeId {
    let pad = graph.pad(input, [(1, 2)], Scalar::F(0.0)).unwrap();
    let view = graph
        .reshape(pad, [1, graph.shape(pad).unwrap().dims()[0]])
        .unwrap();
    graph.square(view).unwrap()
}

fn reversed(graph: &mut Graph, input: NodeId) -> NodeId {
    let producer = graph.square(input).unwrap();
    let view = graph
        .stride(
            producer,
            [
                crate::Slice {
                    start: None,
                    stop: None,
                    step: -1,
                },
                crate::Slice {
                    start: None,
                    stop: None,
                    step: 1,
                },
            ],
        )
        .unwrap();
    graph.contiguous(view).unwrap()
}

fn epilogue(graph: &mut Graph, input: NodeId) -> NodeId {
    let sum = graph.sum(input, 1).unwrap();
    let bias = graph.constant(TensorData::scalar(0.5));
    graph.add(sum, bias).unwrap()
}

#[test]
fn accepted_rehearsals_match_independent_final_normalization() {
    type Builder = fn(&mut Graph, NodeId) -> NodeId;
    let cases: &[(&str, &[usize], Builder)] = &[
        ("contiguous", &[2, 3], contiguous),
        ("scalar aliases", &[2, 3], aliases),
        ("projected alias", &[1, 2, 2, 2], projected),
        ("predicated pad", &[2], padded),
        ("empty source pad", &[0], padded),
        ("signed stride", &[2, 3], reversed),
        ("empty output", &[0, 3], contiguous),
        ("reduction epilogue", &[2, 3], epilogue),
    ];
    for &(name, shape, build) in cases {
        let mut graph = Graph::new();
        let input = graph.input("input", Shape::new(shape.to_vec()));
        let output = build(&mut graph, input);
        let before = FINALIZATIONS.get();
        let retained = schedule(&graph, output).unwrap();
        let reused = FINALIZATIONS.get() - before;
        assert!(reused > 0, "{name}: no accepted rehearsal");
        let legacy = with_legacy_finalization(|| schedule(&graph, output).unwrap());
        assert_eq!(retained.items.len(), legacy.items.len(), "{name}");
        for (actual, expected) in retained.items.iter().zip(&legacy.items) {
            assert_eq!(actual.kernel, expected.kernel, "{name}");
            assert_eq!(
                actual.kernel.topological().unwrap(),
                expected.kernel.topological().unwrap(),
                "{name}"
            );
            assert_eq!(actual.cache_key, expected.cache_key, "{name}");
            assert_eq!(actual.input_bindings, expected.input_bindings, "{name}");
            assert_eq!(actual.inputs, expected.inputs, "{name}");
            assert_eq!(actual.dependencies, expected.dependencies, "{name}");
        }
        retained.validate().unwrap();
        legacy.validate().unwrap();
        let actual = crate::CapturedSchedule::capture(&graph, &retained, &[output]).unwrap();
        let expected = crate::CapturedSchedule::capture(&graph, &legacy, &[output]).unwrap();
        assert_eq!(
            actual.to_bytes().unwrap(),
            expected.to_bytes().unwrap(),
            "{name}"
        );
        let data = TensorData::new(
            Shape::new(shape.to_vec()),
            (0..graph.shape(input).unwrap().numel().unwrap())
                .map(|i| (i as f32 - 2.0) * 0.5)
                .collect(),
        )
        .unwrap();
        let oracle = crate::CpuBackend
            .execute(
                &graph,
                output,
                &std::collections::HashMap::from([("input".into(), data.clone())]),
            )
            .unwrap();
        let inputs = BTreeMap::from([("input".into(), data)]);
        let replayed = actual.replay(&inputs).unwrap();
        let legacy_replayed = expected.replay(&inputs).unwrap();
        assert_eq!(replayed[0].shape(), oracle.shape(), "{name}");
        assert_eq!(replayed[0].storage(), oracle.storage(), "{name}");
        assert_eq!(legacy_replayed[0].storage(), oracle.storage(), "{name}");
    }
}

fn projected_kernel(tagged: bool, out_of_bounds: bool) -> UOp {
    let integer = UType::scalar(DType::I64);
    let value = UType::scalar(DType::F32);
    let range = UOp::from_operation(
        Operation::Range(0),
        Some(integer),
        vec![UOp::constant(8, integer)],
    );
    let four = UOp::constant(4, integer);
    let quotient = UOp::binary(crate::uop::Binary::FloorDiv, range.clone(), four.clone());
    let remainder = UOp::binary(crate::uop::Binary::Mod, range.clone(), four.clone());
    let expression = UOp::binary(
        crate::uop::Binary::Add,
        remainder,
        UOp::binary(crate::uop::Binary::Mul, four, quotient),
    );
    let expression = if tagged {
        UOp::binary(
            crate::uop::Binary::Add,
            expression,
            UOp::constant(0, integer),
        )
        .retag(Some(crate::uop::UOpTag::Text("wrapper".into())))
    } else if out_of_bounds {
        UOp::binary(
            crate::uop::Binary::Add,
            expression,
            UOp::constant(8, integer),
        )
    } else {
        expression
    };
    let address = UOp::from_operation(
        Operation::DefineGlobal(crate::uop::AddressValue {
            space: crate::AddressSpace::Global,
            name: "b7".into(),
            element: value,
        }),
        Some(value),
        vec![],
    );
    let index = UOp::from_operation(
        Operation::Index(crate::uop::IndexValue::Buffer {
            buffer: 7,
            elements: 8,
            input_shape: Shape::from([8]),
            output_shape: Shape::from([8]),
            addressing: crate::IndexAddressing::Projected,
        }),
        Some(value),
        vec![address, expression],
    );
    UOp::sink(vec![
        UOp::from_operation(Operation::Load, Some(value), vec![index]),
        UOp::from_operation(Operation::EndRange, None, vec![range]),
    ])
}

#[test]
fn generic_tagged_normalization_is_not_a_rehearsal_fixed_point() {
    let root = projected_kernel(true, false);
    let first = crate::uop::normalize_kernel_with_table(&root).unwrap();
    let second = crate::uop::normalize_kernel_with_table(first.order().root()).unwrap();
    assert_ne!(first.order().root(), second.order().root());
    assert!(matches!(
        PreparedRehearsalKernel::from_graph_lowering(&root),
        Err(UOpError::InvalidArgument)
    ));
    let foreign = projected_kernel(false, false);
    let prepared = PreparedRehearsalKernel::from_graph_lowering(&foreign).unwrap();
    let equal_root = crate::uop::artifact::decode(
        &crate::uop::artifact::encode(prepared.order().root()).unwrap(),
    )
    .unwrap();
    assert_eq!(&equal_root, prepared.order().root());
    assert!(prepared.order().for_root(&equal_root).is_none());
}

#[test]
fn rehearsal_metadata_does_not_admit_invalid_kernels() {
    let value = UType::scalar(DType::F32);
    let init = UOp::from_operation(
        Operation::ReduceInit(crate::ReductionValue {
            input_shape: Shape::from([2]),
            output_shape: Shape::new([]),
            axes: vec![0],
            keepdim: false,
            kind: crate::ReduceKind::Sum,
        }),
        Some(value),
        vec![],
    );
    let update = UOp::from_operation(
        Operation::ReduceAccumulate,
        Some(value),
        vec![
            init,
            UOp::scalar_constant(DType::F32, 1.0f32.to_bits() as u64, value),
        ],
    );
    let reduced = UOp::from_operation(Operation::ReduceFinalize, Some(value), vec![update]);
    let integer = UType::scalar(DType::I64);
    let unclosed = UOp::from_operation(
        Operation::Range(0),
        Some(integer),
        vec![UOp::constant(2, integer)],
    );
    for (name, root, expected) in [
        (
            "projected bounds",
            projected_kernel(false, true),
            UOpError::InvalidIndex,
        ),
        (
            "duplicate reduction owner",
            UOp::sink(vec![reduced.clone(), reduced]),
            UOpError::InvalidArgument,
        ),
        (
            "unclosed range",
            UOp::sink(vec![unclosed]),
            UOpError::UnclosedControl,
        ),
    ] {
        assert!(
            matches!(PreparedRehearsalKernel::from_graph_lowering(&root), Err(error) if error == expected),
            "{name}"
        );
    }
}
