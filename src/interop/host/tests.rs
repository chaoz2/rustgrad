use super::*;
use crate::{DType, Shape, TensorData};
use std::sync::Arc;

#[test]
fn dtype_widths_and_contiguous_offsets_are_exact() {
    let cases = [
        (DType::Bool, 1),
        (DType::I8, 1),
        (DType::U8, 1),
        (DType::I16, 2),
        (DType::U16, 2),
        (DType::I32, 4),
        (DType::U32, 4),
        (DType::I64, 8),
        (DType::U64, 8),
        (DType::F16, 2),
        (DType::BF16, 2),
        (DType::F32, 4),
        (DType::F64, 8),
    ];
    for (dtype, width) in cases {
        let layout = HostTensorLayout::contiguous(dtype, [2, 3]).unwrap();
        assert_eq!(layout.element_width(), width);
        assert_eq!(layout.validate_read(6 * width).unwrap(), 0..6 * width);
        assert_eq!(
            layout.logical_byte_range(6 * width, 4).unwrap().as_range(),
            4 * width..5 * width
        );
    }
    let offset = HostTensorLayout::new(DType::I16, [2], 2, vec![2]).unwrap();
    assert_eq!(offset.validate_read(6).unwrap(), 2..6);
    assert_eq!(offset.logical_byte_range(6, 1).unwrap().as_range(), 4..6);
}

#[test]
fn signed_permuted_broadcast_scalar_and_empty_reads_are_checked() {
    let flip = HostTensorLayout::new(DType::I16, [3], 4, vec![-2]).unwrap();
    assert_eq!(flip.validate_read(6).unwrap(), 0..6);
    assert_eq!(flip.logical_byte_range(6, 0).unwrap().as_range(), 4..6);
    assert_eq!(flip.logical_byte_range(6, 2).unwrap().as_range(), 0..2);
    let permuted = HostTensorLayout::new(DType::I32, [2, 3], 0, vec![4, 8]).unwrap();
    assert_eq!(permuted.validate_read(24).unwrap(), 0..24);
    assert_eq!(
        permuted.logical_byte_range(24, 1).unwrap().as_range(),
        8..12
    );
    let broadcast = HostTensorLayout::new(DType::U8, [2, 3], 1, vec![1, 0]).unwrap();
    assert_eq!(broadcast.validate_read(3).unwrap(), 1..3);
    assert_eq!(broadcast.logical_byte_range(3, 5).unwrap().as_range(), 2..3);
    let scalar = HostTensorLayout::new(DType::F64, [], 0, vec![]).unwrap();
    assert_eq!(scalar.validate_read(8).unwrap(), 0..8);
    let empty = HostTensorLayout::new(DType::F32, [0, 3], 4, vec![12, 4]).unwrap();
    assert_eq!(empty.validate_read(4).unwrap(), 4..4);
}

#[test]
fn write_validation_rejects_aliases_and_invalid_layouts() {
    let broadcast = HostTensorLayout::new(DType::U8, [2, 3], 0, vec![1, 0]).unwrap();
    assert_eq!(
        broadcast.validate_write(2),
        Err(HostInteropError::NonInjectiveWrite)
    );
    let overlap = HostTensorLayout::new(DType::I16, [2], 0, vec![0]).unwrap();
    assert_eq!(
        overlap.validate_write(2),
        Err(HostInteropError::NonInjectiveWrite)
    );
    assert!(matches!(
        HostTensorLayout::new(DType::I32, [2], 2, vec![4]),
        Err(HostInteropError::Misaligned { .. })
    ));
    assert!(matches!(
        HostTensorLayout::new(DType::I16, [2], 0, vec![]),
        Err(HostInteropError::Rank { .. })
    ));
    let overflow = HostTensorLayout::new(DType::I64, [2], isize::MAX - 7, vec![8]).unwrap();
    assert_eq!(
        overflow.validate_read(usize::MAX),
        Err(HostInteropError::Overflow)
    );
    let out = HostTensorLayout::new(DType::I32, [2], 4, vec![4]).unwrap();
    assert!(matches!(
        out.validate_read(8),
        Err(HostInteropError::Bounds { .. })
    ));
}

#[test]
fn borrowed_and_owned_views_retain_layout_without_copying() {
    let layout = HostTensorLayout::contiguous(DType::U16, [2]).unwrap();
    let bytes: Arc<[u8]> = Arc::from([1u8, 0, 2, 0]);
    let owned = OwnedHostTensor::new(bytes.clone(), layout.clone()).unwrap();
    let cloned = owned.clone();
    drop(owned);
    assert_eq!(Arc::strong_count(&bytes), 2);
    assert_eq!(cloned.logical_byte_range(1).unwrap().as_range(), 2..4);
    let source = [1u8, 0, 2, 0];
    let borrowed = BorrowedHostTensor::new(&source, layout).unwrap();
    assert_eq!(borrowed.layout().dtype(), DType::U16);
    assert_eq!(borrowed.logical_byte_range(0).unwrap().as_range(), 0..2);
}

#[test]
fn identities_are_stable_and_descriptor_only() {
    let a = HostTensorLayout::new(DType::F32, Shape::from([2]), 0, vec![4]).unwrap();
    let b = a.clone();
    let changed = HostTensorLayout::new(DType::F32, [2], 4, vec![4]).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert_ne!(a.identity(), changed.identity());
}

fn raw_pair(dtype: DType) -> Vec<u8> {
    match dtype.itemsize() {
        1 if dtype == DType::Bool => vec![0, 1],
        1 => vec![0x80, 0x7f],
        2 => vec![0x01, 0x80, 0xff, 0x7f],
        4 => vec![0x01, 0x00, 0x80, 0x80, 0xff, 0xff, 0xc0, 0x7f],
        8 => vec![
            0x01, 0, 0, 0, 0, 0, 0xf0, 0xff, 0, 0, 0, 0, 0, 0, 0xf8, 0x7f,
        ],
        _ => unreachable!(),
    }
}

#[test]
fn all_dtypes_copy_raw_bits_through_a_different_signed_layout() {
    let dtypes = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::F8E4M3FNUZ,
        DType::F8E5M2FNUZ,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::BF16,
        DType::F32,
        DType::F64,
    ];
    for dtype in dtypes {
        let source_bytes = raw_pair(dtype);
        let source_layout = HostTensorLayout::contiguous(dtype, [2]).unwrap();
        let source = BorrowedHostTensor::new(&source_bytes, source_layout).unwrap();
        let tensor = source.to_tensor_data().unwrap();
        assert_eq!(tensor.to_le_bytes().unwrap(), source_bytes);

        let width = dtype.itemsize();
        let mut destination_bytes = vec![0xa5; 3 * width];
        let destination_layout =
            HostTensorLayout::new(dtype, [2], (2 * width) as isize, vec![-(width as isize)])
                .unwrap();
        let mut destination =
            MutableBorrowedHostTensor::new(&mut destination_bytes, destination_layout.clone())
                .unwrap();
        tensor.copy_to_host(&mut destination).unwrap();
        drop(destination);
        let copied = BorrowedHostTensor::new(&destination_bytes, destination_layout).unwrap();
        assert_eq!(
            copied.to_tensor_data().unwrap().to_le_bytes().unwrap(),
            source_bytes
        );
    }
}

#[test]
fn materialization_visits_offset_permuted_and_broadcasted_logical_order() {
    let offset = HostTensorLayout::new(DType::U16, [2], 2, vec![2]).unwrap();
    let bytes = [0xa5, 0xa5, 1, 0, 2, 0];
    assert_eq!(
        BorrowedHostTensor::new(&bytes, offset)
            .unwrap()
            .to_tensor_data()
            .unwrap()
            .to_le_bytes()
            .unwrap(),
        vec![1, 0, 2, 0]
    );

    let permuted = HostTensorLayout::new(DType::U16, [2, 3], 0, vec![2, 4]).unwrap();
    let physical = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0];
    assert_eq!(
        BorrowedHostTensor::new(&physical, permuted)
            .unwrap()
            .to_tensor_data()
            .unwrap()
            .to_le_bytes()
            .unwrap(),
        vec![1, 0, 3, 0, 5, 0, 2, 0, 4, 0, 6, 0]
    );

    let broadcast = HostTensorLayout::new(DType::F16, [2, 3], 0, vec![2, 0]).unwrap();
    let float_bits = [0x00, 0x80, 0x01, 0x7e]; // -0 and a half NaN payload
    assert_eq!(
        BorrowedHostTensor::new(&float_bits, broadcast)
            .unwrap()
            .to_tensor_data()
            .unwrap()
            .to_le_bytes()
            .unwrap(),
        vec![0, 0x80, 0, 0x80, 0, 0x80, 1, 0x7e, 1, 0x7e, 1, 0x7e]
    );
}

#[test]
fn writes_support_permuted_scalar_and_empty_and_are_transactional() {
    let tensor =
        TensorData::from_le_bytes([2, 3], DType::U16, &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0])
            .unwrap();
    let mut bytes = vec![0xa5; 12];
    let layout = HostTensorLayout::new(DType::U16, [2, 3], 0, vec![2, 4]).unwrap();
    let mut destination = MutableBorrowedHostTensor::new(&mut bytes, layout.clone()).unwrap();
    tensor.copy_to_host(&mut destination).unwrap();
    drop(destination);
    assert_eq!(
        BorrowedHostTensor::new(&bytes, layout)
            .unwrap()
            .to_tensor_data()
            .unwrap(),
        tensor
    );

    let scalar = TensorData::from_le_bytes([], DType::F32, &[0, 0, 0, 0x80]).unwrap();
    let mut scalar_bytes = [0u8; 4];
    scalar
        .copy_to_host(
            &mut MutableBorrowedHostTensor::new(
                &mut scalar_bytes,
                HostTensorLayout::contiguous(DType::F32, []).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(scalar_bytes, [0, 0, 0, 0x80]);
    let empty = TensorData::from_le_bytes([0, 3], DType::U64, &[]).unwrap();
    let mut empty_bytes = [];
    empty
        .copy_to_host(
            &mut MutableBorrowedHostTensor::new(
                &mut empty_bytes,
                HostTensorLayout::new(DType::U64, [0, 3], 0, vec![24, 8]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let mut unchanged = [0xa5; 4];
    let before = unchanged;
    let mut wrong_dtype = MutableBorrowedHostTensor::new(
        &mut unchanged,
        HostTensorLayout::contiguous(DType::U32, [1]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        scalar.copy_to_host(&mut wrong_dtype),
        Err(HostInteropError::DTypeMismatch { .. })
    ));
    drop(wrong_dtype);
    assert_eq!(unchanged, before);

    let mut wrong_shape = MutableBorrowedHostTensor::new(
        &mut unchanged,
        HostTensorLayout::contiguous(DType::U16, [1]).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        tensor.copy_to_host(&mut wrong_shape),
        Err(HostInteropError::ShapeMismatch)
    ));
    drop(wrong_shape);
    assert_eq!(unchanged, before);
}

#[test]
fn destination_construction_rejects_noninjective_writes() {
    let mut bytes = [0u8; 2];
    let before = bytes;
    let broadcast = HostTensorLayout::new(DType::U8, [2], 0, vec![0]).unwrap();
    assert!(matches!(
        MutableBorrowedHostTensor::new(&mut bytes, broadcast),
        Err(HostInteropError::NonInjectiveWrite)
    ));
    let out_of_bounds = HostTensorLayout::new(DType::U8, [2], 1, vec![1]).unwrap();
    assert!(matches!(
        MutableBorrowedHostTensor::new(&mut bytes, out_of_bounds),
        Err(HostInteropError::Bounds { .. })
    ));
    assert_eq!(bytes, before);
}

#[test]
fn materialized_tensor_does_not_alias_its_host_source() {
    let mut bytes = [0, 0x80, 1, 0x7e];
    let tensor = {
        let source = BorrowedHostTensor::new(
            &bytes,
            HostTensorLayout::contiguous(DType::F16, [2]).unwrap(),
        )
        .unwrap();
        source.to_tensor_data().unwrap()
    };
    bytes.fill(0);
    assert_eq!(tensor.to_le_bytes().unwrap(), vec![0, 0x80, 1, 0x7e]);
}
