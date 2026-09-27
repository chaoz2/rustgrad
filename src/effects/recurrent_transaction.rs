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
    pub(super) initial: Box<[BufferState]>,
    pub(super) modes: Box<[RecurrentBankMode]>,
}

pub(crate) struct PreparedRecurrentTransaction<'a> {
    pub(super) schema: &'a PreparedRecurrentTransactionSchema,
    pub(super) current: &'a [BufferState],
    pub(super) next_versions: Box<[u64]>,
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
        current: &'a [BufferState],
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
        let next_versions = current
            .iter()
            .map(|state| {
                state
                    .version
                    .checked_add(1)
                    .ok_or(PreparedRecurrentTransactionError::VersionOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_boxed_slice();
        Ok(PreparedRecurrentTransaction {
            schema: self,
            current,
            next_versions,
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
    pub(crate) fn into_next_versions(self) -> Box<[u64]> {
        self.next_versions
    }
}
