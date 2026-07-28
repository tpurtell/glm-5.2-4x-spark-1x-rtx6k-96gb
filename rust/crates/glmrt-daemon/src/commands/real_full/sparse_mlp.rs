pub(in crate::commands::real_full) mod math;
pub(in crate::commands::real_full) mod route;
pub(in crate::commands::real_full) mod router;
mod shared;

pub(in crate::commands::real_full) use router::cache_router_correction_bias_host_values;
pub(in crate::commands::real_full) use shared::{
    real_sparse_mlp_shared_layer_full_output_hidden_from_initial,
    real_sparse_mlp_shared_layer_full_output_hidden_from_initial_device_input,
    real_sparse_mlp_shared_layer_hidden_from_initial, RealFullSparseMlpSharedLayerHidden,
};
