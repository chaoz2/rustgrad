//! Pure, validated movement-to-index planning shared by scheduling and kernels.
mod view;
pub(crate) use view::{computed_view, static_view};
