mod real_full;
mod real_slice;
mod synthetic_glm;
mod tiny;

pub(crate) use real_full::{real_glm_full_completion, try_real_glm_full_streaming_response};
pub(crate) use real_slice::real_glm_slice_completion;
pub(crate) use synthetic_glm::synthetic_glm_layer_completion;
pub(crate) use tiny::tiny_backend_completion;
