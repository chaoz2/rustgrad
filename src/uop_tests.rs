use crate::uop::{self, Binary, Ternary, UArg, UOpKind, UType, Walk};
use crate::{DType, Graph, Shape, TensorData, UOp, UPat};

fn i64t() -> UType {
    UType::scalar(DType::I64)
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
    let bad = UOp::new(
        UOpKind::Binary(Binary::Add),
        Some(i64t()),
        vec![x],
        UArg::None,
    );
    assert!(bad.validate().is_err());
}

#[test]
fn upat_rewrites_are_prioritized_shared_and_pure() {
    let x = UOp::constant(7, i64t());
    let zero = UOp::constant(0, i64t());
    let shared = UOp::binary(Binary::Add, x.clone(), zero);
    let root = UOp::sink(vec![shared.clone(), shared]);
    let pattern = UPat::op(UOpKind::Binary(Binary::Add))
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
fn uop_graph_scalar_pilot_is_inspectable() {
    let mut graph = Graph::new();
    let x = graph.input("x", Shape::new([]));
    let one = graph.constant(TensorData::scalar(1.0));
    let y = graph.add(x, one).unwrap();
    let uop = uop::lower_graph_scalar(&graph, y).unwrap();
    uop.validate().unwrap();
    assert!(matches!(uop.kind(), UOpKind::Binary(Binary::Add)));
    let condition = UOp::constant(1, UType::scalar(DType::Bool));
    let where_ = UOp::new(
        UOpKind::Ternary(Ternary::Where),
        uop.ty(),
        vec![condition, uop.clone(), uop],
        UArg::None,
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
            matches!(uop.arg(), UArg::Scalar { dtype: got, bits: raw } if *got == dtype && *raw == bits)
        );
    }
}

#[test]
fn typed_buffer_index_rejects_malformed_rank_and_element_metadata() {
    let base = UOp::new(
        UOpKind::DefineGlobal,
        Some(i64t()),
        vec![],
        UArg::Address {
            space: crate::AddressSpace::Global,
            name: "b0".into(),
            element: i64t(),
        },
    );
    let range = UOp::new(
        UOpKind::Range,
        Some(i64t()),
        vec![UOp::constant(6, i64t())],
        UArg::RangeAxis(0),
    );
    let invalid = UOp::new(
        UOpKind::Index,
        Some(i64t()),
        vec![base, range.clone()],
        UArg::BufferIndex {
            buffer: 0,
            elements: 5,
            input_shape: Shape::from([2, 3]),
            output_shape: Shape::from([3]),
        },
    );
    let root = UOp::sink(vec![
        invalid,
        UOp::new(UOpKind::EndRange, None, vec![range], UArg::None),
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
