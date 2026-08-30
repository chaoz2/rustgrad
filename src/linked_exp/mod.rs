//! Opt-in CUDA linked-resource execution for one source-exact F32 Exp kernel.
//!
//! This subsystem is intentionally separate from generic captured replay. Its
//! artifacts contain only deterministic descriptors and require caller-owned
//! immutable link payloads and device-buffer leases to be rebound explicitly.

mod artifact;
mod batch;
mod capture;
mod resource;

pub use artifact::{
    BoundLinkedF32ExpResources, LINKED_F32_EXP_RESOURCE_ARTIFACT_SCHEMA_VERSION,
    LinkedF32ExpResourceArtifact, LinkedF32ExpResourceSlot,
};
pub use batch::{
    BoundLinkedF32ExpBatchResources, BoundPreparedLinkedF32ExpBatchCapture,
    LINKED_F32_EXP_BATCH_SCHEMA_VERSION, LinkedF32ExpBatchArtifact, LinkedF32ExpBatchSlot,
    PreparedLinkedF32ExpBatchCapture, PreparedLinkedF32ExpBatchSlot,
    execute_prepared_linked_f32_exp_batch,
};
pub use capture::{
    BoundPreparedLinkedF32ExpCapture, PreparedLinkedF32ExpBindingTable,
    PreparedLinkedF32ExpCapture, PreparedLinkedF32ExpExternalRole, execute_prepared_linked_f32_exp,
};
pub use resource::{
    LINKED_F32_EXP_RESOURCE_SCHEMA_VERSION, LinkedF32ExpResourceBinding,
    LinkedF32ExpResourceDescriptor,
};
