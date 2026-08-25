use super::*;
use crate::{DType, Shape};
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
