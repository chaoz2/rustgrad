//! Capture-bound immutable schema for prepared recurrent host transactions.

use super::{BufferState, runtime::RecurrentBankMode, validate_buffer_state};
use std::sync::Arc;

/// Allocation identity independently retained by one prepared bank layout and
/// its immutable transaction schema. It authenticates their association only:
/// live runtime slots, leases, generations, pointers, and current versions are
/// deliberately not retained here.
#[derive(Clone, Debug)]
pub(crate) struct PreparedRecurrentTransactionOwner(Arc<()>);

impl PreparedRecurrentTransactionOwner {
    pub(crate) fn new() -> Self {
        Self(Arc::new(()))
    }
}

/// Immutable descriptor and bank-mode schema for one exact sealed recurrent
/// preparation. Current versions remain call-scoped so an older valid
/// checkpoint can restore into a newly constructed effect runtime without
/// rebuilding native programs.
#[derive(Debug)]
pub(crate) struct PreparedRecurrentTransactionSchema {
    owner: PreparedRecurrentTransactionOwner,
    initial: Box<[BufferState]>,
    modes: Box<[RecurrentBankMode]>,
}

pub(crate) struct PreparedRecurrentTransaction<'a> {
    schema: &'a PreparedRecurrentTransactionSchema,
    current: &'a mut [BufferState],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PreparedRecurrentTransactionError {
    Owner,
    Cardinality,
    Descriptor,
    VersionOverflow,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedRecurrentTransactionTestCounts {
    pub(crate) schema_builds: usize,
    pub(crate) cursor_descriptor_admissions: usize,
    pub(crate) ordered_transactions: usize,
    pub(crate) fallback_frontier_reconstructions: usize,
}

#[cfg(test)]
std::thread_local! {
    static PREPARED_RECURRENT_TRANSACTION_TEST_COUNTS:
        std::cell::Cell<PreparedRecurrentTransactionTestCounts> = const {
            std::cell::Cell::new(PreparedRecurrentTransactionTestCounts {
                schema_builds: 0,
                cursor_descriptor_admissions: 0,
                ordered_transactions: 0,
                fallback_frontier_reconstructions: 0,
            })
        };
}

#[cfg(test)]
pub(super) fn record_prepared_recurrent_transaction(
    update: impl FnOnce(&mut PreparedRecurrentTransactionTestCounts),
) {
    PREPARED_RECURRENT_TRANSACTION_TEST_COUNTS.with(|counts| {
        let mut next = counts.get();
        update(&mut next);
        counts.set(next);
    });
}

#[cfg(test)]
pub(crate) fn reset_prepared_recurrent_transaction_test_counts() {
    PREPARED_RECURRENT_TRANSACTION_TEST_COUNTS.with(|counts| counts.set(Default::default()));
}

#[cfg(test)]
pub(crate) fn prepared_recurrent_transaction_test_counts() -> PreparedRecurrentTransactionTestCounts
{
    PREPARED_RECURRENT_TRANSACTION_TEST_COUNTS.with(std::cell::Cell::get)
}

impl PreparedRecurrentTransactionSchema {
    pub(crate) fn new(
        owner: PreparedRecurrentTransactionOwner,
        initial: Vec<BufferState>,
        modes: Vec<RecurrentBankMode>,
    ) -> Result<Self, &'static str> {
        if initial.is_empty() || initial.len() != modes.len() {
            return Err("recurrent replacement frontier cardinality mismatch");
        }
        let mut previous = None;
        for state in &initial {
            validate_buffer_state(state).map_err(|_| "recurrent replacement state mismatch")?;
            if previous.is_some_and(|buffer| buffer >= state.buffer) {
                return Err("recurrent replacement state mismatch");
            }
            previous = Some(state.buffer);
        }
        #[cfg(test)]
        record_prepared_recurrent_transaction(|counts| {
            counts.schema_builds = counts.schema_builds.saturating_add(1);
        });
        Ok(Self {
            owner,
            initial: initial.into_boxed_slice(),
            modes: modes.into_boxed_slice(),
        })
    }

    pub(crate) fn prepare<'a>(
        &'a self,
        owner: &PreparedRecurrentTransactionOwner,
        current: &'a mut [BufferState],
    ) -> Result<PreparedRecurrentTransaction<'a>, PreparedRecurrentTransactionError> {
        if !Arc::ptr_eq(&self.owner.0, &owner.0) {
            return Err(PreparedRecurrentTransactionError::Owner);
        }
        if current.len() != self.initial.len() {
            return Err(PreparedRecurrentTransactionError::Cardinality);
        }
        #[cfg(test)]
        record_prepared_recurrent_transaction(|counts| {
            counts.cursor_descriptor_admissions =
                counts.cursor_descriptor_admissions.saturating_add(1);
        });
        for (actual, initial) in current.iter().zip(self.initial.iter()) {
            if actual.buffer != initial.buffer
                || actual.version < initial.version
                || actual.shape != initial.shape
                || actual.dtype != initial.dtype
                || actual.bytes != initial.bytes
            {
                return Err(PreparedRecurrentTransactionError::Descriptor);
            }
        }
        // Keep overflow admission as a distinct second pass. In particular,
        // every descriptor error must win over every version overflow.
        for state in current.iter() {
            state
                .version
                .checked_add(1)
                .ok_or(PreparedRecurrentTransactionError::VersionOverflow)?;
        }
        Ok(PreparedRecurrentTransaction {
            schema: self,
            current,
        })
    }

    pub(crate) fn initial(&self) -> &[BufferState] {
        &self.initial
    }

    pub(crate) fn modes(&self) -> &[RecurrentBankMode] {
        &self.modes
    }

    #[cfg(test)]
    pub(crate) fn allocation_identity(&self) -> usize {
        Arc::as_ptr(&self.owner.0) as usize
    }
}

impl PreparedRecurrentTransaction<'_> {
    pub(super) fn current(&self) -> &[BufferState] {
        &*self.current
    }

    pub(super) fn initial(&self) -> &[BufferState] {
        &self.schema.initial
    }

    pub(super) fn modes(&self) -> &[RecurrentBankMode] {
        &self.schema.modes
    }

    pub(super) fn successor_version(&self, ordinal: usize) -> u64 {
        admitted_successor_version(self.current[ordinal].version)
    }

    /// Consumes the transaction and advances only the exact mutable frontier
    /// admitted by `prepare`. Callers retain this transaction until runtime
    /// staging succeeds, so failures cannot publish cursor progress.
    pub(crate) fn advance(self) {
        for state in self.current {
            state.version = admitted_successor_version(state.version);
        }
    }
}

fn admitted_successor_version(version: u64) -> u64 {
    version
        .checked_add(1)
        .expect("prepared recurrent version overflow was admitted")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    fn state(buffer: u64, version: u64) -> BufferState {
        BufferState {
            buffer,
            version,
            shape: Shape::from([2]),
            dtype: DType::F32,
            bytes: 2 * DType::F32.itemsize(),
        }
    }

    fn legacy_successor_versions(frontier: &[BufferState]) -> Vec<u64> {
        frontier
            .iter()
            .map(|state| state.version.checked_add(1).unwrap())
            .collect()
    }

    #[test]
    fn admission_preserves_error_order_and_derives_legacy_successors_without_a_vector() {
        reset_prepared_recurrent_transaction_test_counts();
        let owner = PreparedRecurrentTransactionOwner::new();
        let foreign = PreparedRecurrentTransactionOwner::new();
        let initial = vec![state(7, 0), state(9, 4)];
        let schema = PreparedRecurrentTransactionSchema::new(
            owner.clone(),
            initial.clone(),
            vec![RecurrentBankMode::Replace, RecurrentBankMode::Retain],
        )
        .unwrap();

        let mut incomplete = initial[..1].to_vec();
        assert!(matches!(
            schema.prepare(&foreign, &mut incomplete),
            Err(PreparedRecurrentTransactionError::Owner)
        ));
        assert!(matches!(
            schema.prepare(&owner, &mut incomplete),
            Err(PreparedRecurrentTransactionError::Cardinality)
        ));

        let mut descriptor_before_overflow = initial.clone();
        descriptor_before_overflow[0].version = u64::MAX;
        descriptor_before_overflow[1].shape = Shape::from([1]);
        descriptor_before_overflow[1].bytes = DType::F32.itemsize();
        assert!(matches!(
            schema.prepare(&owner, &mut descriptor_before_overflow),
            Err(PreparedRecurrentTransactionError::Descriptor)
        ));
        descriptor_before_overflow[1] = initial[1].clone();
        assert!(matches!(
            schema.prepare(&owner, &mut descriptor_before_overflow),
            Err(PreparedRecurrentTransactionError::VersionOverflow)
        ));

        let mut admitted = initial;
        admitted[1].version = u64::MAX - 1;
        let expected = legacy_successor_versions(&admitted);
        let transaction = schema.prepare(&owner, &mut admitted).unwrap();
        transaction.advance();
        assert_eq!(
            admitted
                .iter()
                .map(|state| state.version)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
