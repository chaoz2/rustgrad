//! Optimizer-neutral ownership and recoverable errors for compiled modules.

use super::*;

struct CompiledModuleErrorDescriptor {
    debug_name: &'static str,
    message: &'static str,
}

const ADAMW_COMPILE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleAdamWCompileError",
    message: "owned compiled AdamW compilation failed",
};
const MOMENTUM_SGD_COMPILE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleMomentumSgdCompileError",
    message: "owned compiled momentum-SGD compilation failed",
};
const ADAMW_RESTORE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleAdamWRestoreError",
    message: "owned compiled AdamW checkpoint restore failed",
};
const ADAMW_EVALUATION_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleAdamWEvaluationError",
    message: "owned compiled AdamW evaluation capture failed",
};
const ADAMW_PREPARE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleAdamWPrepareError",
    message: "owned compiled AdamW preparation failed",
};
const MOMENTUM_SGD_RESTORE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleMomentumSgdRestoreError",
    message: "owned compiled momentum-SGD checkpoint restore failed",
};
const MOMENTUM_SGD_PREPARE_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleMomentumSgdPrepareError",
    message: "owned compiled momentum-SGD preparation failed",
};
const TRAINING_FINISH_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleTrainingFinishError",
    message: "owned compiled training finalization failed",
};
const ADAMW_FINISH_ERROR: CompiledModuleErrorDescriptor = CompiledModuleErrorDescriptor {
    debug_name: "CompiledModuleAdamWFinishError",
    message: "owned compiled AdamW finalization failed",
};

/// Resource-free compiled-training plan paired with the exact module value
/// used to build it.
///
/// `P` retains optimizer-specific compilation and checkpoint policy while this
/// owner provides the optimizer-neutral module lifecycle. `A` retains any
/// typed optimizer-specific admission state that must survive until target
/// preparation. The module is not exposed while the plan or its prepared
/// session exists, preventing callers from treating stale host parameters as
/// the active training frontier.
pub struct CompiledModuleTrainingPlan<M, P, A = ()> {
    pub(super) module: M,
    pub(super) plan: P,
    pub(super) seal: CompiledModuleSeal,
    pub(super) attachment: A,
}

/// Owned-module AdamW plan using the shared compiled-training lifecycle.
pub type CompiledModuleAdamWPlan<M> = CompiledModuleTrainingPlan<M, CompiledAdamWPlan, Option<u64>>;

/// Owned-module momentum-SGD plan using the shared compiled-training lifecycle.
pub type CompiledModuleMomentumSgdPlan<M> = CompiledModuleTrainingPlan<M, CompiledMomentumSgdPlan>;

/// Prepared compiled training session that owns its source module for the
/// complete replay lifecycle.
///
/// Optimizer policy remains on `R`. This owner only seals the source module,
/// forwards the runtime's supported capabilities, and publishes one exact
/// detached parameter frontier when the session finishes.
pub struct CompiledModuleTrainingSession<M, R> {
    pub(super) module: M,
    pub(super) runtime: R,
    pub(super) seal: CompiledModuleSeal,
    pub(super) evaluation_capture_identity: Option<u64>,
    descriptor: &'static CompiledModuleErrorDescriptor,
}

/// AdamW session compatibility alias over the optimizer-neutral owner.
pub type CompiledModuleAdamWSession<M, R> = CompiledModuleTrainingSession<M, R>;

impl<M, P, A> CompiledModuleTrainingPlan<M, P, A> {
    /// Consumes this owner into a target-specific training session.
    ///
    /// Optimizer and backend policy remain expressed by the concrete `P` and
    /// [`SessionTarget`] implementation; this lifecycle does not dispatch on
    /// either at runtime.
    pub fn prepare<T>(
        self,
        target: &T,
    ) -> std::result::Result<<T as SessionTarget<Self>>::Session, <T as SessionTarget<Self>>::Error>
    where
        T: SessionTarget<Self>,
    {
        target.prepare(self)
    }

    pub(super) fn into_module(self) -> M {
        self.module
    }
}

impl<M, R> CompiledModuleTrainingSession<M, R> {
    pub(super) fn training(
        module: M,
        runtime: R,
        seal: CompiledModuleSeal,
        evaluation_capture_identity: Option<u64>,
    ) -> Self {
        Self {
            module,
            runtime,
            seal,
            evaluation_capture_identity,
            descriptor: &TRAINING_FINISH_ERROR,
        }
    }

    pub(super) fn adamw(
        module: M,
        runtime: R,
        seal: CompiledModuleSeal,
        evaluation_capture_identity: Option<u64>,
    ) -> Self {
        Self {
            module,
            runtime,
            seal,
            evaluation_capture_identity,
            descriptor: &ADAMW_FINISH_ERROR,
        }
    }

    #[cfg(test)]
    pub(super) fn map_runtime<S>(
        self,
        map: impl FnOnce(R) -> S,
    ) -> CompiledModuleTrainingSession<M, S> {
        CompiledModuleTrainingSession {
            module: self.module,
            runtime: map(self.runtime),
            seal: self.seal,
            evaluation_capture_identity: self.evaluation_capture_identity,
            descriptor: self.descriptor,
        }
    }
}

/// Recoverable owned-module compilation failure.
///
/// The exact module is retained without publication or mutation. Public
/// optimizer-specific aliases preserve their source-facing names while future
/// optimizers reuse this one lifecycle.
pub struct CompiledModuleCompileError<M> {
    module: M,
    source: Error,
    descriptor: &'static CompiledModuleErrorDescriptor,
}

impl<M> CompiledModuleCompileError<M> {
    fn new(module: M, source: Error, descriptor: &'static CompiledModuleErrorDescriptor) -> Self {
        Self {
            module,
            source,
            descriptor,
        }
    }

    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn into_module(self) -> M {
        self.module
    }

    pub fn into_parts(self) -> (M, Error) {
        (self.module, self.source)
    }
}

impl<M> fmt::Debug for CompiledModuleCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct(self.descriptor.debug_name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M> fmt::Display for CompiledModuleCompileError<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.descriptor.message, self.source)
    }
}

impl<M> std::error::Error for CompiledModuleCompileError<M> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Recoverable owned-plan operation failure.
///
/// The unconsumed plan is boxed once so restore, evaluation attachment, and
/// target preparation share one error ABI without inflating successful result
/// paths. Public aliases retain optimizer- and operation-specific names.
pub struct CompiledModulePlanError<P, E> {
    plan: Box<P>,
    source: E,
    descriptor: &'static CompiledModuleErrorDescriptor,
}

/// Finalization failure retaining the intact owned module/session pair.
///
/// The retained session remains available for inspection, retry, or recovery
/// without publication. AdamW and optimizer-neutral aliases preserve their
/// existing diagnostics while sharing this lifecycle.
pub struct CompiledModuleTrainingFinishError<M, R> {
    session: Box<CompiledModuleTrainingSession<M, R>>,
    source: Error,
    descriptor: &'static CompiledModuleErrorDescriptor,
}

/// AdamW compatibility alias for a recoverable owned-session failure.
pub type CompiledModuleAdamWFinishError<M, R> = CompiledModuleTrainingFinishError<M, R>;

/// Recoverable result of publishing one owned module together with its
/// complete optimizer-neutral checkpoint.
pub type CompiledModuleCheckpointFinishResult<M, R, C> =
    std::result::Result<(M, CompiledModuleCheckpoint<C>), CompiledModuleTrainingFinishError<M, R>>;

impl<M, R> CompiledModuleTrainingFinishError<M, R> {
    pub(super) fn new(session: CompiledModuleTrainingSession<M, R>, source: Error) -> Self {
        let descriptor = session.descriptor;
        Self {
            session: Box::new(session),
            source,
            descriptor,
        }
    }

    pub fn source_error(&self) -> &Error {
        &self.source
    }

    pub fn session(&self) -> &CompiledModuleTrainingSession<M, R> {
        &self.session
    }

    pub fn into_session(self) -> CompiledModuleTrainingSession<M, R> {
        *self.session
    }

    pub fn into_parts(self) -> (CompiledModuleTrainingSession<M, R>, Error) {
        (*self.session, self.source)
    }

    /// Discards the failed runtime frontier and returns the sealed host module
    /// exactly as it currently exists, without attempting publication again.
    pub fn into_module_without_publication(self) -> M {
        (*self.session).into_module_without_publication()
    }
}

impl<M, R> fmt::Debug for CompiledModuleTrainingFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct(self.descriptor.debug_name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<M, R> fmt::Display for CompiledModuleTrainingFinishError<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.descriptor.message, self.source)
    }
}

impl<M, R> std::error::Error for CompiledModuleTrainingFinishError<M, R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<P, E> CompiledModulePlanError<P, E> {
    fn new(plan: P, source: E, descriptor: &'static CompiledModuleErrorDescriptor) -> Self {
        Self {
            plan: Box::new(plan),
            source,
            descriptor,
        }
    }

    pub fn source_error(&self) -> &E {
        &self.source
    }

    pub fn plan(&self) -> &P {
        &self.plan
    }

    pub fn into_plan(self) -> P {
        *self.plan
    }

    pub fn into_parts(self) -> (P, E) {
        (*self.plan, self.source)
    }
}

impl<P, E: fmt::Debug> fmt::Debug for CompiledModulePlanError<P, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct(self.descriptor.debug_name)
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl<P, E: fmt::Display> fmt::Display for CompiledModulePlanError<P, E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.descriptor.message, self.source)
    }
}

impl<P, E: std::error::Error + 'static> std::error::Error for CompiledModulePlanError<P, E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub type CompiledModuleAdamWCompileError<M> = CompiledModuleCompileError<M>;
pub type CompiledModuleMomentumSgdCompileError<M> = CompiledModuleCompileError<M>;

pub type CompiledModuleAdamWRestoreError<M> =
    CompiledModulePlanError<CompiledModuleAdamWPlan<M>, Error>;
pub type CompiledModuleAdamWEvaluationError<M> =
    CompiledModulePlanError<CompiledModuleAdamWPlan<M>, Error>;
pub type CompiledModuleAdamWPrepareError<M, E> =
    CompiledModulePlanError<CompiledModuleAdamWPlan<M>, E>;
pub type CompiledModuleMomentumSgdRestoreError<M> =
    CompiledModulePlanError<CompiledModuleMomentumSgdPlan<M>, Error>;
pub type CompiledModuleMomentumSgdPrepareError<M, E> =
    CompiledModulePlanError<CompiledModuleMomentumSgdPlan<M>, E>;

pub(super) fn adamw_compile_error<M>(
    module: M,
    source: Error,
) -> CompiledModuleAdamWCompileError<M> {
    CompiledModuleCompileError::new(module, source, &ADAMW_COMPILE_ERROR)
}

pub(super) fn momentum_sgd_compile_error<M>(
    module: M,
    source: Error,
) -> CompiledModuleMomentumSgdCompileError<M> {
    CompiledModuleCompileError::new(module, source, &MOMENTUM_SGD_COMPILE_ERROR)
}

pub(super) fn adamw_restore_error<M>(
    plan: CompiledModuleAdamWPlan<M>,
    source: Error,
) -> CompiledModuleAdamWRestoreError<M> {
    CompiledModulePlanError::new(plan, source, &ADAMW_RESTORE_ERROR)
}

pub(super) fn adamw_evaluation_error<M>(
    plan: CompiledModuleAdamWPlan<M>,
    source: Error,
) -> CompiledModuleAdamWEvaluationError<M> {
    CompiledModulePlanError::new(plan, source, &ADAMW_EVALUATION_ERROR)
}

pub(super) fn adamw_prepare_error<M, E>(
    plan: CompiledModuleAdamWPlan<M>,
    source: E,
) -> CompiledModuleAdamWPrepareError<M, E> {
    CompiledModulePlanError::new(plan, source, &ADAMW_PREPARE_ERROR)
}

pub(super) fn momentum_sgd_restore_error<M>(
    plan: CompiledModuleMomentumSgdPlan<M>,
    source: Error,
) -> CompiledModuleMomentumSgdRestoreError<M> {
    CompiledModulePlanError::new(plan, source, &MOMENTUM_SGD_RESTORE_ERROR)
}

pub(super) fn momentum_sgd_prepare_error<M, E>(
    plan: CompiledModuleMomentumSgdPlan<M>,
    source: E,
) -> CompiledModuleMomentumSgdPrepareError<M, E> {
    CompiledModulePlanError::new(plan, source, &MOMENTUM_SGD_PREPARE_ERROR)
}
