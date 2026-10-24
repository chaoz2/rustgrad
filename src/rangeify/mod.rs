//! Pure, validated movement-to-index planning shared by scheduling and kernels.
mod projection;
mod view;
pub(crate) use projection::ProjectedForestCache;
#[cfg(test)]
pub(crate) use projection::{ProjectedForestCounts, with_projected_forest_mode};
#[cfg(test)]
pub(crate) use view::{ProjectionSource, projected_view};
pub(crate) use view::{
    RangeifyError, computed_broadcast_view, computed_view, is_constant_zero_pad, static_view,
};
