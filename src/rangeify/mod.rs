//! Pure, validated movement-to-index planning shared by scheduling and kernels.
mod view;
pub(crate) use view::{
    RangeifyError, computed_broadcast_view, computed_view, projected_source, projected_view,
    static_view,
};
