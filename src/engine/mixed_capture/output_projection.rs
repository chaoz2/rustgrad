//! Immutable output selection for one prepared recurrent native program.

use super::validate_requested_selection;
use crate::ReplayError;
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug)]
pub(crate) struct PreparedRecurrentOutputProjectionHandle {
    capture_identity: u64,
    full_requested: Arc<[u64]>,
    requested: Arc<[u64]>,
}

impl PreparedRecurrentOutputProjectionHandle {
    pub(super) fn new(
        capture_identity: u64,
        full_requested: Arc<[u64]>,
        requested: Arc<[u64]>,
    ) -> Self {
        Self {
            capture_identity,
            full_requested,
            requested,
        }
    }

    pub(crate) fn authenticates_requested(&self, requested: &[u64]) -> bool {
        self.full_requested.as_ref() == requested
    }

    pub(super) const fn capture_identity(&self) -> u64 {
        self.capture_identity
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedRecurrentOutputDisposition {
    Take,
    CloneSuccessor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedRecurrentOutput {
    id: u64,
    disposition: PreparedRecurrentOutputDisposition,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedRecurrentOutputProjection {
    requested: Arc<[u64]>,
    outputs: Box<[PreparedRecurrentOutput]>,
    public: BTreeSet<u64>,
}

impl PreparedRecurrentOutputProjection {
    pub(super) fn prepare(
        requested: Arc<[u64]>,
        full_requested: &[u64],
        successor_outputs: &BTreeSet<u64>,
    ) -> Result<Self, ReplayError> {
        validate_requested_selection(full_requested, requested.as_ref())?;
        let outputs = requested
            .iter()
            .copied()
            .map(|id| PreparedRecurrentOutput {
                id,
                disposition: if successor_outputs.contains(&id) {
                    PreparedRecurrentOutputDisposition::CloneSuccessor
                } else {
                    PreparedRecurrentOutputDisposition::Take
                },
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let public = requested.iter().copied().collect();
        Ok(Self {
            requested,
            outputs,
            public,
        })
    }

    pub(super) fn is_owned_by(&self, handle: &PreparedRecurrentOutputProjectionHandle) -> bool {
        Arc::ptr_eq(&self.requested, &handle.requested)
    }

    pub(super) fn public(&self) -> &BTreeSet<u64> {
        &self.public
    }

    pub(super) fn extract(
        &self,
        values: &mut super::super::captured_replay::ReplayValues,
    ) -> Result<Vec<crate::TensorData>, ReplayError> {
        self.outputs
            .iter()
            .map(|output| match output.disposition {
                PreparedRecurrentOutputDisposition::CloneSuccessor => {
                    values.tensor(output.id, "requested mixed output").cloned()
                }
                PreparedRecurrentOutputDisposition::Take => {
                    values.take_tensor(output.id, "requested mixed output")
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bits(value: &crate::TensorData) -> Vec<u32> {
        let crate::Storage::F32(values) = value.storage() else {
            panic!("output projection fixture must remain F32")
        };
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn projection_preserves_order_duplicates_and_successor_disposition() {
        let requested = [5, 7, 5, 9];
        let selected = Arc::<[u64]>::from([5, 5, 9]);
        let projection = PreparedRecurrentOutputProjection::prepare(
            Arc::clone(&selected),
            &requested,
            &BTreeSet::from([5]),
        )
        .unwrap();
        assert_eq!(
            projection.outputs.as_ref(),
            [
                PreparedRecurrentOutput {
                    id: 5,
                    disposition: PreparedRecurrentOutputDisposition::CloneSuccessor,
                },
                PreparedRecurrentOutput {
                    id: 5,
                    disposition: PreparedRecurrentOutputDisposition::CloneSuccessor,
                },
                PreparedRecurrentOutput {
                    id: 9,
                    disposition: PreparedRecurrentOutputDisposition::Take,
                },
            ]
        );
        assert_eq!(projection.public, BTreeSet::from([5, 9]));
        let successor = crate::TensorData::new([1], vec![-0.0]).unwrap();
        let ordinary = crate::TensorData::new([1], vec![f32::from_bits(0x7fc0_0011)]).unwrap();
        let mut values = crate::engine::captured_replay::ReplayValues::from_materialized(
            [(5, successor.clone()), (9, ordinary.clone())]
                .into_iter()
                .collect(),
        );
        let extracted = projection.extract(&mut values).unwrap();
        assert_eq!(extracted.len(), 3);
        assert_eq!(f32_bits(&extracted[0]), f32_bits(&successor));
        assert_eq!(f32_bits(&extracted[1]), f32_bits(&successor));
        assert_eq!(f32_bits(&extracted[2]), f32_bits(&ordinary));
        assert_eq!(
            f32_bits(values.tensor(5, "retained successor").unwrap()),
            f32_bits(&successor)
        );
        assert!(values.tensor(9, "taken ordinary output").is_err());
        let foreign = PreparedRecurrentOutputProjectionHandle::new(
            1,
            Arc::from(requested),
            Arc::from([5, 5, 9]),
        );
        assert!(!projection.is_owned_by(&foreign));
        let local = PreparedRecurrentOutputProjectionHandle::new(1, Arc::from(requested), selected);
        assert!(projection.is_owned_by(&local));
    }

    #[test]
    fn projection_rejects_out_of_order_or_excess_duplicates() {
        assert!(
            PreparedRecurrentOutputProjection::prepare(
                Arc::from([7, 5]),
                &[5, 7],
                &BTreeSet::new(),
            )
            .is_err()
        );
        assert!(
            PreparedRecurrentOutputProjection::prepare(
                Arc::from([5, 5]),
                &[5, 7],
                &BTreeSet::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn projection_extracts_the_logical_requested_passthrough_id() {
        let projection =
            PreparedRecurrentOutputProjection::prepare(Arc::from([41]), &[41], &BTreeSet::new())
                .unwrap();
        let source = crate::TensorData::new(
            [2],
            vec![f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_0021)],
        )
        .unwrap();
        let mut values = crate::engine::captured_replay::ReplayValues::from_materialized(
            [(17, source.clone())].into_iter().collect(),
        );
        values
            .project_requested_aliases(&[crate::RequestedPassthrough {
                requested: crate::NodeId::from_index(41),
                source: crate::NodeId::from_index(17),
                desc: crate::BufferDesc {
                    id: 17,
                    shape: crate::Shape::from([2]),
                    dtype: crate::DType::F32,
                    bytes: 2 * crate::DType::F32.itemsize(),
                    alignment: crate::DType::F32.itemsize(),
                    read_only: true,
                    view: Some(crate::AffineView::identity(crate::Shape::from([2]))),
                },
            }])
            .unwrap();
        let extracted = projection.extract(&mut values).unwrap();
        assert_eq!(extracted.len(), 1);
        assert_eq!(f32_bits(&extracted[0]), f32_bits(&source));
        assert_eq!(
            f32_bits(values.tensor(17, "physical alias source").unwrap()),
            f32_bits(&source)
        );
        assert!(values.tensor(41, "taken logical passthrough").is_err());
    }
}
