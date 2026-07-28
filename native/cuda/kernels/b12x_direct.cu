#include "common.h"

#include "down_m1.h"
#include "down_m128.h"
#include "down_m16.h"
#include "down_m2.h"
#include "down_m256.h"
#include "down_m32.h"
#include "down_m4.h"
#include "down_m64.h"
#include "down_m8.h"
#include "down_tp4_m1.h"
#include "down_tp4_m128.h"
#include "down_tp4_m16.h"
#include "down_tp4_m2.h"
#include "down_tp4_m256.h"
#include "down_tp4_m32.h"
#include "down_tp4_m4.h"
#include "down_tp4_m64.h"
#include "down_tp4_m8.h"
#include "gate_m1.h"
#include "gate_m128.h"
#include "gate_m16.h"
#include "gate_m2.h"
#include "gate_m256.h"
#include "gate_m32.h"
#include "gate_m4.h"
#include "gate_m64.h"
#include "gate_m8.h"
#include "gate_tp4_m1.h"
#include "gate_tp4_m128.h"
#include "gate_tp4_m16.h"
#include "gate_tp4_m2.h"
#include "gate_tp4_m256.h"
#include "gate_tp4_m32.h"
#include "gate_tp4_m4.h"
#include "gate_tp4_m64.h"
#include "gate_tp4_m8.h"
#include "moe_tp4_m1.h"
#include "moe_tp4_w4a4_prefill_m256_topk8.h"
#include "moe_tp4_w4a16_decode_m1.h"
#include "moe_tp4_w4a16_decode_m1_fused_sum.h"
#include "moe_tp4_w4a16_m1_parity_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_m8_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_m8_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m2_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m3_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m4_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m5_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m6_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m7_topk8.h"
#include "moe_tp4_w4a16_m1_parity_grouped_wide_m8_topk8.h"
#include "moe_tp4_w4a16_modelopt_decode_m1.h"
#include "moe_tp4_w4a16_prefill_m2_topk8.h"
#include "moe_tp4_w4a16_prefill_m4_topk8.h"
#include "moe_tp4_w4a16_prefill_m8_topk8.h"
#include "moe_tp4_w4a16_prefill_m16_topk8.h"
#include "moe_tp4_w4a16_prefill_m32_topk8.h"
#include "moe_tp4_w4a16_prefill_m64_topk8.h"
#include "moe_tp4_w4a16_prefill_m128_topk8.h"
#include "moe_tp4_w4a16_prefill_m256_topk8.h"
#include "moe_tp4_w4a16_prefill_m1024_topk8.h"
#include "moe_tp4_w4a16_prefill_m2048_topk8.h"
#include "moe_tp4_w4a16_prefill_m512_topk8.h"
#include "moe_tp4_w4a16_top1_m1.h"
#include "moe_tp4_w4a16_top1_m128.h"
#include "moe_tp4_w4a16_top1_m16.h"
#include "moe_tp4_w4a16_top1_m2.h"
#include "moe_tp4_w4a16_top1_m256.h"
#include "moe_tp4_w4a16_top1_m32.h"
#include "moe_tp4_w4a16_top1_m4.h"
#include "moe_tp4_w4a16_top1_m64.h"
#include "moe_tp4_w4a16_top1_m8.h"
#include "b12x_spark_moe_aot_config.h"
#include "b12x_spark_w4a16_m1_parity_aot_config.h"
#include "b12x_spark_mixed_w4a4_aot_config.h"
#include "mixed_w4a16_activation_m1.h"
#include "mixed_w4a16_activation_m2.h"
#include "mixed_w4a16_activation_m4.h"
#include "mixed_w4a16_activation_m8.h"
#include "mixed_w4a16_activation_m16.h"
#include "mixed_w4a16_activation_m32.h"
#include "mixed_w4a16_activation_m64.h"
#include "mixed_w4a16_activation_m128.h"
#include "mixed_w4a16_activation_m256.h"
#include "mixed_w4a16_fc2_m1.h"
#include "mixed_w4a16_fc2_m2.h"
#include "mixed_w4a16_fc2_m4.h"
#include "mixed_w4a16_fc2_m8.h"
#include "mixed_w4a16_fc2_m16.h"
#include "mixed_w4a16_fc2_m32.h"
#include "mixed_w4a16_fc2_m64.h"
#include "mixed_w4a16_fc2_m128.h"
#include "mixed_w4a16_fc2_m256.h"
#include "mixed_w4a4_fc1_m1.h"
#include "mixed_w4a4_fc1_m2.h"
#include "mixed_w4a4_fc1_m4.h"
#include "mixed_w4a4_fc1_m8.h"
#include "mixed_w4a4_fc1_m16.h"
#include "mixed_w4a4_fc1_m32.h"
#include "mixed_w4a4_fc1_m64.h"
#include "mixed_w4a4_fc1_m128.h"
#include "mixed_w4a4_fc1_m256.h"

#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
#include "sparkinfer_source_w4a16_aot_config.h"
#include "sparkinfer_source_w4a16_direct_m1_topk8.h"
#include "sparkinfer_source_w4a16_direct_m2_topk8.h"
#include "sparkinfer_source_w4a16_direct_m3_topk8.h"
#include "sparkinfer_source_w4a16_direct_m4_topk8.h"
#include "sparkinfer_source_w4a16_direct_m5_topk8.h"
#include "sparkinfer_source_w4a16_direct_m6_topk8.h"
#include "sparkinfer_source_w4a16_direct_m7_topk8.h"
#include "sparkinfer_source_w4a16_direct_m8_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m16_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m32_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m64_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m128_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m256_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m512_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m1024_topk8.h"
#include "sparkinfer_source_w4a16_prefill_m2048_topk8.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdlib>
#include <mutex>

struct glmrt_b12x_spark_mixed_w4a4_buffers_t {
  glmrt_device_buffer_t input_packed;
  glmrt_device_buffer_t input_scale;
  glmrt_device_buffer_t w13_weight_source;
  glmrt_device_buffer_t w13_scale_source;
  glmrt_device_buffer_t w13_global_scale;
  glmrt_device_buffer_t fc1_output;
  glmrt_device_buffer_t fc1_reordered;
  glmrt_device_buffer_t activated;
  glmrt_device_buffer_t w2_weight_packed;
  glmrt_device_buffer_t w2_scale_packed;
  glmrt_device_buffer_t w2_global_scale;
  glmrt_device_buffer_t output;
  glmrt_device_buffer_t packed_route_indices;
  glmrt_device_buffer_t block_expert_ids;
  glmrt_device_buffer_t packed_route_count;
  glmrt_device_buffer_t topk_weights;
  glmrt_device_buffer_t fc2_scratch;
  glmrt_device_buffer_t locks;
};

namespace {

constexpr size_t kB12xMaxRows = 256;
constexpr size_t kB12xW4a16MaxRows = 2048;
constexpr size_t kB12xHidden = 6144;
constexpr size_t kB12xIntermediate = 2048;
constexpr size_t kB12xTp4Intermediate = 512;
constexpr size_t kB12xOutput = 6144;
constexpr size_t kB12xExperts = 256;
constexpr size_t kB12xTopK = 8;
constexpr int kQuantThreads = 32;
constexpr size_t kMixedW4a4LockElements = 1024;

glmrt_b12x_gate_m1_Kernel_Module_t gate_m1_module;
glmrt_b12x_gate_m2_Kernel_Module_t gate_m2_module;
glmrt_b12x_gate_m4_Kernel_Module_t gate_m4_module;
glmrt_b12x_gate_m8_Kernel_Module_t gate_m8_module;
glmrt_b12x_gate_m16_Kernel_Module_t gate_m16_module;
glmrt_b12x_gate_m32_Kernel_Module_t gate_m32_module;
glmrt_b12x_gate_m64_Kernel_Module_t gate_m64_module;
glmrt_b12x_gate_m128_Kernel_Module_t gate_m128_module;
glmrt_b12x_gate_m256_Kernel_Module_t gate_m256_module;
glmrt_b12x_down_m1_Kernel_Module_t down_m1_module;
glmrt_b12x_down_m2_Kernel_Module_t down_m2_module;
glmrt_b12x_down_m4_Kernel_Module_t down_m4_module;
glmrt_b12x_down_m8_Kernel_Module_t down_m8_module;
glmrt_b12x_down_m16_Kernel_Module_t down_m16_module;
glmrt_b12x_down_m32_Kernel_Module_t down_m32_module;
glmrt_b12x_down_m64_Kernel_Module_t down_m64_module;
glmrt_b12x_down_m128_Kernel_Module_t down_m128_module;
glmrt_b12x_down_m256_Kernel_Module_t down_m256_module;
glmrt_b12x_gate_tp4_m1_Kernel_Module_t gate_tp4_m1_module;
glmrt_b12x_gate_tp4_m2_Kernel_Module_t gate_tp4_m2_module;
glmrt_b12x_gate_tp4_m4_Kernel_Module_t gate_tp4_m4_module;
glmrt_b12x_gate_tp4_m8_Kernel_Module_t gate_tp4_m8_module;
glmrt_b12x_gate_tp4_m16_Kernel_Module_t gate_tp4_m16_module;
glmrt_b12x_gate_tp4_m32_Kernel_Module_t gate_tp4_m32_module;
glmrt_b12x_gate_tp4_m64_Kernel_Module_t gate_tp4_m64_module;
glmrt_b12x_gate_tp4_m128_Kernel_Module_t gate_tp4_m128_module;
glmrt_b12x_gate_tp4_m256_Kernel_Module_t gate_tp4_m256_module;
glmrt_b12x_down_tp4_m1_Kernel_Module_t down_tp4_m1_module;
glmrt_b12x_down_tp4_m2_Kernel_Module_t down_tp4_m2_module;
glmrt_b12x_down_tp4_m4_Kernel_Module_t down_tp4_m4_module;
glmrt_b12x_down_tp4_m8_Kernel_Module_t down_tp4_m8_module;
glmrt_b12x_down_tp4_m16_Kernel_Module_t down_tp4_m16_module;
glmrt_b12x_down_tp4_m32_Kernel_Module_t down_tp4_m32_module;
glmrt_b12x_down_tp4_m64_Kernel_Module_t down_tp4_m64_module;
glmrt_b12x_down_tp4_m128_Kernel_Module_t down_tp4_m128_module;
glmrt_b12x_down_tp4_m256_Kernel_Module_t down_tp4_m256_module;
glmrt_b12x_moe_tp4_m1_Kernel_Module_t moe_tp4_m1_module;
glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Kernel_Module_t
    moe_tp4_w4a4_prefill_m256_topk8_module;
glmrt_b12x_moe_tp4_w4a16_decode_m1_Kernel_Module_t moe_tp4_w4a16_decode_m1_module;
glmrt_b12x_moe_tp4_w4a16_decode_m1_fused_sum_Kernel_Module_t
    moe_tp4_w4a16_decode_m1_fused_sum_module;
#define GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(M)                                  \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8_Kernel_Module_t              \
      moe_tp4_w4a16_m1_parity_m##M##_topk8_module;
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(2)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(3)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(4)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(5)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(6)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(7)
GLMRT_DEFINE_W4A16_M1_PARITY_MODULE(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_MODULE
#define GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(M)                          \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_Kernel_Module_t      \
      moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module;
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(2)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(3)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(4)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(5)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(6)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(7)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_MODULE
#define GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(M)                    \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_Kernel_Module_t \
      moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module;
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(2)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(3)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(4)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(5)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(6)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(7)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_MODULE
glmrt_b12x_moe_tp4_w4a16_modelopt_decode_m1_Kernel_Module_t
    moe_tp4_w4a16_modelopt_decode_m1_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m2_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m2_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m4_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m4_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m8_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m8_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m16_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m16_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m32_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m32_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m64_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m64_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m128_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m128_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m256_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m256_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m1024_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m1024_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m2048_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m2048_topk8_module;
glmrt_b12x_moe_tp4_w4a16_prefill_m512_topk8_Kernel_Module_t
    moe_tp4_w4a16_prefill_m512_topk8_module;
glmrt_b12x_moe_tp4_w4a16_top1_m1_Kernel_Module_t moe_tp4_w4a16_top1_m1_module;
glmrt_b12x_moe_tp4_w4a16_top1_m2_Kernel_Module_t moe_tp4_w4a16_top1_m2_module;
glmrt_b12x_moe_tp4_w4a16_top1_m4_Kernel_Module_t moe_tp4_w4a16_top1_m4_module;
glmrt_b12x_moe_tp4_w4a16_top1_m8_Kernel_Module_t moe_tp4_w4a16_top1_m8_module;
glmrt_b12x_moe_tp4_w4a16_top1_m16_Kernel_Module_t moe_tp4_w4a16_top1_m16_module;
glmrt_b12x_moe_tp4_w4a16_top1_m32_Kernel_Module_t moe_tp4_w4a16_top1_m32_module;
glmrt_b12x_moe_tp4_w4a16_top1_m64_Kernel_Module_t moe_tp4_w4a16_top1_m64_module;
glmrt_b12x_moe_tp4_w4a16_top1_m128_Kernel_Module_t moe_tp4_w4a16_top1_m128_module;
glmrt_b12x_moe_tp4_w4a16_top1_m256_Kernel_Module_t moe_tp4_w4a16_top1_m256_module;
#define GLMRT_DEFINE_MIXED_W4A4_MODULES(M)                                      \
  glmrt_b12x_mixed_w4a4_fc1_m##M##_Kernel_Module_t                             \
      mixed_w4a4_fc1_m##M##_module;                                             \
  glmrt_b12x_mixed_w4a16_activation_m##M##_Kernel_Module_t                     \
      mixed_w4a16_activation_m##M##_module;                                     \
  glmrt_b12x_mixed_w4a16_fc2_m##M##_Kernel_Module_t                            \
      mixed_w4a16_fc2_m##M##_module;
GLMRT_DEFINE_MIXED_W4A4_MODULES(1)
GLMRT_DEFINE_MIXED_W4A4_MODULES(2)
GLMRT_DEFINE_MIXED_W4A4_MODULES(4)
GLMRT_DEFINE_MIXED_W4A4_MODULES(8)
GLMRT_DEFINE_MIXED_W4A4_MODULES(16)
GLMRT_DEFINE_MIXED_W4A4_MODULES(32)
GLMRT_DEFINE_MIXED_W4A4_MODULES(64)
GLMRT_DEFINE_MIXED_W4A4_MODULES(128)
GLMRT_DEFINE_MIXED_W4A4_MODULES(256)
#undef GLMRT_DEFINE_MIXED_W4A4_MODULES
#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
#define GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(M)                   \
  glmrt_sparkinfer_source_w4a16_direct_m##M##_topk8_Kernel_Module_t            \
      sparkinfer_source_w4a16_direct_m##M##_topk8_module;
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(1)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(2)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(3)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(4)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(5)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(6)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(7)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(8)
#undef GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE
#define GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(M)                  \
  glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Kernel_Module_t           \
      sparkinfer_source_w4a16_prefill_m##M##_topk8_module;
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(16)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(32)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(64)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(128)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(256)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(512)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(1024)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(2048)
#undef GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE
std::once_flag sparkinfer_source_w4a16_module_init_once;
glmrt_status_t sparkinfer_source_w4a16_module_init_status = GLMRT_STATUS_OK;
#endif
std::once_flag b12x_module_init_once;
glmrt_status_t b12x_module_init_status = GLMRT_STATUS_OK;
std::once_flag mixed_w4a4_module_init_once;
glmrt_status_t mixed_w4a4_module_init_status = GLMRT_STATUS_OK;
constexpr size_t kB12xW4a16LockElements = 48 * 4 + 2;
constexpr int kB12xW4a16DecodeMaxGridX =
    static_cast<int>((kB12xW4a16LockElements - 2) / 2);
constexpr int kB12xW4a16DecodeResidentGridX = 48;
constexpr int kB12xW4a16Top1M1GridX = 32;
constexpr int kB12xW4a16Top1GridX = 48;

size_t align_up_size(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

int w4a16_decode_grid_x() {
  static const int grid_x = [] {
    const char* raw = std::getenv("GLMRT_B12X_SPARK_W4A16_DECODE_GRID_X");
    if (raw == nullptr || *raw == '\0') {
      return GLMRT_B12X_W4A16_DECODE_M1_GRID_X;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed <= 0 ||
        parsed > kB12xW4a16DecodeMaxGridX) {
      return GLMRT_B12X_W4A16_DECODE_M1_GRID_X;
    }
    return static_cast<int>(parsed);
  }();
  return grid_x;
}

bool w4a16_m1_fused_sum_enabled() {
  static const bool enabled = [] {
    const char* raw =
        std::getenv("GLMRT_B12X_SPARK_W4A16_M1_FUSED_SUM");
    return raw != nullptr && *raw != '\0' &&
           !(raw[0] == '0' && raw[1] == '\0');
  }();
  return enabled;
}

#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
size_t sparkinfer_source_w4a16_direct_max_rows() {
  static const size_t max_rows = [] {
    const char* raw =
        std::getenv("GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_MAX_ROWS");
    if (raw == nullptr || *raw == '\0') {
      return size_t{2};
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0 || parsed > 8) {
      return size_t{2};
    }
    return static_cast<size_t>(parsed);
  }();
  return max_rows;
}
#endif

bool buffer_has_bytes(glmrt_device_buffer_t buffer, size_t required) {
  return buffer.ptr != nullptr && buffer.bytes >= required;
}

__device__ size_t swizzled_scale_offset_device(size_t row, size_t col, size_t cols) {
  const size_t row_block = row / 128;
  const size_t row_quarter = (row % 128) / 32;
  const size_t row_inner = row % 32;
  const size_t col_block = col / 4;
  const size_t col_inner = col % 4;
  const size_t col_blocks = cols / 4;
  return (((((row_block * col_blocks + col_block) * 32 + row_inner) * 4 + row_quarter) * 4) +
          col_inner);
}

__device__ uint8_t fp4_e2m1_code(float value) {
  const bool negative = signbit(value);
  const float magnitude = fabsf(value);
  uint8_t code = 0;
  if (magnitude <= 0.25f) {
    code = 0;
  } else if (magnitude < 0.75f) {
    code = 1;
  } else if (magnitude <= 1.25f) {
    code = 2;
  } else if (magnitude < 1.75f) {
    code = 3;
  } else if (magnitude <= 2.5f) {
    code = 4;
  } else if (magnitude < 3.5f) {
    code = 5;
  } else if (magnitude <= 5.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return static_cast<uint8_t>(code | (negative ? 8 : 0));
}

__global__ void swizzle_modelopt_scale_kernel(const uint8_t* input, uint8_t* output,
                                               size_t rows, size_t cols) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * cols;
  if (index >= values) {
    return;
  }
  const size_t row = index / cols;
  const size_t col = index % cols;
  output[swizzled_scale_offset_device(row, col, cols)] = input[index];
}

__global__ void write_alpha_kernel(float* alphas, float gate, float up, float down) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    alphas[0] = gate;
    alphas[1] = up;
    alphas[2] = down;
  }
}

__global__ void quantize_bf16_nvfp4_b12x_kernel(const uint16_t* input, uint8_t* packed,
                                                 uint8_t* scale, size_t rows,
                                                 size_t cols) {
  __shared__ float values[16];
  __shared__ uint8_t codes[16];
  __shared__ float inverse_scale;
  const size_t row = blockIdx.y;
  const size_t col_block = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane < 16) {
    values[lane] = bf16_to_f32(input[row * cols + col_block * 16 + lane]);
  }
  __syncthreads();

  float maximum = lane < 16 ? fabsf(values[lane]) : 0.0f;
  for (int offset = 8; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffu, maximum, offset));
  }
  if (lane == 0) {
    const uint8_t scale_byte =
        static_cast<uint8_t>(__nv_cvt_float_to_fp8(maximum / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float decoded_scale = f8e4m3_to_f32(scale_byte);
    inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    const size_t scale_cols = cols / 16;
    scale[swizzled_scale_offset_device(row, col_block, scale_cols)] = scale_byte;
  }
  __syncthreads();
  if (lane < 16) {
    const float quantized = fminf(fmaxf(values[lane] * inverse_scale, -6.0f), 6.0f);
    codes[lane] = fp4_e2m1_code(quantized);
  }
  __syncthreads();
  if (lane < 8) {
    packed[row * (cols / 2) + col_block * 8 + lane] =
        static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
  }
}

__global__ void prepare_nvfp4_row_payload_b12x_kernel(
    const uint8_t* payload, size_t source_rows, size_t source_row_stride_bytes,
    const uint32_t* row_indices, uint8_t* packed, uint8_t* scale, size_t rows,
    size_t cols) {
  const size_t row = blockIdx.y;
  const size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t source_row = row_indices[row];
  if (source_row >= source_rows) {
    return;
  }
  const size_t packed_row_bytes = cols / 2;
  const size_t scale_cols = cols / 16;
  const uint8_t* source = payload + source_row * source_row_stride_bytes;
  if (byte < packed_row_bytes) {
    packed[row * packed_row_bytes + byte] = source[byte];
  }
  if (byte < scale_cols) {
    scale[swizzled_scale_offset_device(row, byte, scale_cols)] =
        source[packed_row_bytes + byte];
  }
}

__global__ void dequantize_nvfp4_row_payload_bf16_kernel(const uint8_t* payload,
                                                          uint16_t* output,
                                                          size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= hidden_dim) {
    return;
  }
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = payload[index / 2];
  const uint8_t code = index % 2 == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(payload[packed_bytes + index / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void dequantize_nvfp4_row_payloads_bf16_kernel(
    const uint8_t* payload, size_t row_stride_bytes, uint16_t* output,
    size_t rows, size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  const uint8_t* source = payload + row * row_stride_bytes;
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = source[col / 2];
  const uint8_t code = (col & 1) == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(source[packed_bytes + col / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void sum_w4a16_topk_bf16_kernel(const uint16_t* routed,
                                             uint16_t* output, size_t rows,
                                             size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  float sum = 0.0f;
#pragma unroll
  for (size_t route = 0; route < kB12xTopK; ++route) {
    sum += bf16_to_f32(routed[(row * kB12xTopK + route) * hidden_dim + col]);
  }
  output[index] = f32_to_bf16(sum);
}

__global__ void sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel(
    const uint16_t* routed, uint8_t* output, size_t rows,
    size_t output_row_stride_bytes) {
  constexpr int kBlock = 256;
  __shared__ uint16_t rounded_values[kB12xHidden];
  __shared__ float maxima[kBlock];
  __shared__ float row_scale;
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  float maximum = 0.0f;
  for (size_t col = threadIdx.x; col < kB12xHidden; col += blockDim.x) {
    float sum = 0.0f;
#pragma unroll
    for (size_t route = 0; route < kB12xTopK; ++route) {
      sum += bf16_to_f32(
          routed[(row * kB12xTopK + route) * kB12xHidden + col]);
    }
    const uint16_t rounded = f32_to_bf16(sum);
    const float value = bf16_to_f32(rounded);
    rounded_values[col] = rounded;
    maximum = fmaxf(maximum, isfinite(value) ? fabsf(value) : 0.0f);
  }
  maxima[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      maxima[threadIdx.x] =
          fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  uint8_t* row_output = output + row * output_row_stride_bytes;
  if (threadIdx.x == 0) {
    row_scale = maxima[0] > 0.0f ? maxima[0] / 448.0f : 1.0f;
    *reinterpret_cast<float*>(row_output + kB12xHidden) = row_scale;
  }
  __syncthreads();
  for (size_t col = threadIdx.x; col < kB12xHidden; col += blockDim.x) {
    const float value = bf16_to_f32(rounded_values[col]);
    row_output[col] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
        isfinite(value) ? value / row_scale : 0.0f, __NV_SATFINITE,
        __NV_E4M3));
  }
}

__global__ void gather_nvfp4_rows_bf16_kernel(
    const uint8_t* payload, size_t source_rows, size_t source_row_stride_bytes,
    const uint32_t* row_indices, uint16_t* output, size_t rows, size_t hidden_dim) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = rows * hidden_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / hidden_dim;
  const size_t col = index % hidden_dim;
  const size_t source_row = row_indices[row];
  if (source_row >= source_rows) {
    output[index] = 0;
    return;
  }
  const uint8_t* source = payload + source_row * source_row_stride_bytes;
  const size_t packed_bytes = hidden_dim / 2;
  const uint8_t packed = source[col / 2];
  const uint8_t code = (col & 1) == 0 ? packed & 0x0f : packed >> 4;
  const float scale = f8e4m3_to_f32(source[packed_bytes + col / 16]);
  output[index] = f32_to_bf16(nvfp4_e2m1_code_value(code) * scale);
}

__global__ void pack_w4a16_weight_kernel(const uint8_t* source, uint32_t* destination,
                                          size_t size_k, size_t size_n,
                                          size_t row_rotation) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t k_tiles = size_k / 16;
  const size_t n_tiles = size_n / 64;
  const size_t output_words = k_tiles * n_tiles * 128;
  if (output_index >= output_words) {
    return;
  }
  const size_t packed_position = output_index % 128;
  const size_t tile_index = output_index / 128;
  const size_t n_tile = tile_index % n_tiles;
  const size_t k_tile = tile_index / n_tiles;
  const size_t thread_group = packed_position / 4;
  const size_t warp_column = packed_position % 4;
  const size_t tensor_column = thread_group / 4;
  const size_t tensor_row = (thread_group % 4) * 2;
  constexpr int element_offsets[4] = {0, 1, 8, 9};
  constexpr int pack_order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  uint32_t result = 0;
  for (int slot = 0; slot < 8; ++slot) {
    const int source_slot = pack_order[slot];
    const int element_slot = source_slot & 3;
    const size_t element = tensor_row + element_offsets[element_slot];
    const size_t k_half = element / 8;
    const size_t nibble = element % 8;
    const size_t column_base = warp_column * 16 + tensor_column;
    const size_t packed_row = n_tile * 64 + column_base + (source_slot >= 4 ? 8 : 0);
    const size_t source_row = (packed_row + row_rotation) % size_n;
    const size_t source_word = k_tile * 2 + k_half;
    const uint32_t word = reinterpret_cast<const uint32_t*>(source)[
        source_row * (size_k / 8) + source_word];
    result |= ((word >> (nibble * 4)) & 0x0fU) << (slot * 4);
  }
  destination[output_index] = result;
}

__global__ void pack_w4a16_scale_kernel(const uint8_t* source, uint8_t* destination,
                                         size_t size_k, size_t size_n,
                                         size_t row_rotation, float scale_factor) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = (size_k / 16) * size_n;
  if (output_index >= values) {
    return;
  }
  const size_t k_block = output_index / size_n;
  const size_t output_row = output_index % size_n;
  constexpr int swap_four[4] = {0, 2, 1, 3};
  const size_t swapped = (output_row & ~size_t{3}) + swap_four[output_row & 3];
  const size_t group_base = (swapped / 64) * 64;
  const size_t group_offset = swapped % 64;
  const size_t permuted_row = group_base + group_offset / 8 + 8 * (group_offset % 8);
  const size_t source_row = (permuted_row + row_rotation) % size_n;
  const float source_scale = f8e4m3_to_f32(source[source_row * (size_k / 16) + k_block]);
  const float adjusted = source_scale * scale_factor * 128.0f;
  if (adjusted < 2.0f) {
    destination[output_index] = 0;
    return;
  }
  const __half_raw encoded = __float2half_rn(adjusted);
  destination[output_index] = static_cast<uint8_t>((encoded.x >> 7) & 0xffU);
}

__global__ void initialize_w4a16_top1_routes_kernel(
    int32_t* packed_route_indices, int32_t* block_expert_ids,
    int32_t* packed_route_count, float* topk_weights, size_t rows,
    size_t capacity_rows, uint32_t expert_id, bool direct_topk) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < capacity_rows) {
    topk_weights[index] = 1.0f;
  }
  if (direct_topk) {
    if (index < rows) {
      packed_route_indices[index] = static_cast<int32_t>(expert_id);
    }
    return;
  }
  const size_t padded_rows = ((rows + 7) / 8) * 8;
  if (index < padded_rows) {
    packed_route_indices[index] =
        index < rows ? static_cast<int32_t>(index) : static_cast<int32_t>(rows);
  }
  if (index < padded_rows / 8) {
    block_expert_ids[index] = static_cast<int32_t>(expert_id);
  }
  if (index == 0) {
    packed_route_count[0] = static_cast<int32_t>(padded_rows);
  }
}

__global__ void initialize_w4a16_modelopt_decode_routes_kernel(
    const int32_t* topk_ids, int32_t* packed_route_indices,
    int32_t* block_expert_ids, int32_t* packed_route_count) {
  constexpr int32_t route_slots =
      GLMRT_B12X_W4A16_MODELOPT_DECODE_M1_PACKED_ROUTE_SLOTS;
  constexpr int32_t block_size =
      GLMRT_B12X_W4A16_MODELOPT_DECODE_M1_BLOCK_SIZE;
  const int32_t slot = static_cast<int32_t>(threadIdx.x);
  if (slot >= route_slots) {
    return;
  }
  const int32_t route = slot / block_size;
  const int32_t block_lane = slot % block_size;
  packed_route_indices[slot] = block_lane == 0 ? route : static_cast<int32_t>(kB12xTopK);
  if (block_lane == 0) {
    block_expert_ids[route] = topk_ids[route];
  }
  if (slot == 0) {
    packed_route_count[0] = route_slots;
  }
}

__global__ void initialize_mixed_w4a4_routes_kernel(
    int32_t* packed_route_indices, int32_t* block_expert_ids,
    int32_t* packed_route_count, float* topk_weights, size_t rows,
    size_t route_slots, size_t route_blocks, size_t block_size) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < route_slots) {
    packed_route_indices[index] =
        index < rows ? static_cast<int32_t>(index) : static_cast<int32_t>(rows);
  }
  if (index < route_blocks) {
    block_expert_ids[index] = 0;
  }
  if (index < rows) {
    topk_weights[index] = 1.0f;
  }
  if (index == 0) {
    packed_route_count[0] = static_cast<int32_t>(
        ((rows + block_size - 1) / block_size) * block_size);
  }
}

__global__ void silu_product_quantize_nvfp4_b12x_kernel(const uint16_t* gate,
                                                         const uint16_t* up, uint8_t* packed,
                                                         uint8_t* scale, size_t rows,
                                                         size_t cols) {
  __shared__ float values[16];
  __shared__ uint8_t codes[16];
  __shared__ float inverse_scale;
  const size_t row = blockIdx.y;
  const size_t col_block = blockIdx.x;
  const int lane = threadIdx.x;
  if (lane < 16) {
    const size_t index = row * cols + col_block * 16 + lane;
    const float gate_value = bf16_to_f32(gate[index]);
    const float up_value = bf16_to_f32(up[index]);
    const float activation = sigmoid_f32(gate_value) * gate_value * up_value;
    values[lane] = __bfloat162float(__float2bfloat16_rn(activation));
  }
  __syncthreads();

  float maximum = lane < 16 ? fabsf(values[lane]) : 0.0f;
  for (int offset = 8; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffu, maximum, offset));
  }
  if (lane == 0) {
    const uint8_t scale_byte =
        static_cast<uint8_t>(__nv_cvt_float_to_fp8(maximum / 6.0f, __NV_SATFINITE, __NV_E4M3));
    const float decoded_scale = f8e4m3_to_f32(scale_byte);
    inverse_scale = decoded_scale == 0.0f ? 0.0f : 1.0f / decoded_scale;
    const size_t scale_cols = cols / 16;
    scale[swizzled_scale_offset_device(row, col_block, scale_cols)] = scale_byte;
  }
  __syncthreads();
  if (lane < 16) {
    const float quantized = fminf(fmaxf(values[lane] * inverse_scale, -6.0f), 6.0f);
    codes[lane] = fp4_e2m1_code(quantized);
  }
  __syncthreads();
  if (lane < 8) {
    packed[row * (cols / 2) + col_block * 8 + lane] =
        static_cast<uint8_t>(codes[lane * 2] | (codes[lane * 2 + 1] << 4));
  }
}

__global__ void reorder_w13_fc1_bf16_kernel(const uint16_t* source,
                                             uint16_t* destination, size_t rows) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t row_width = 2 * kB12xTp4Intermediate;
  const size_t values = rows * row_width;
  if (index >= values) {
    return;
  }
  const size_t row_base = (index / row_width) * row_width;
  const size_t col = index % row_width;
  const size_t source_col = col < kB12xTp4Intermediate
                                ? col + kB12xTp4Intermediate
                                : col - kB12xTp4Intermediate;
  destination[index] = source[row_base + source_col];
}

void initialize_b12x_modules() {
  glmrt_b12x_gate_m1_Kernel_Module_Load(&gate_m1_module);
  glmrt_b12x_gate_m2_Kernel_Module_Load(&gate_m2_module);
  glmrt_b12x_gate_m4_Kernel_Module_Load(&gate_m4_module);
  glmrt_b12x_gate_m8_Kernel_Module_Load(&gate_m8_module);
  glmrt_b12x_gate_m16_Kernel_Module_Load(&gate_m16_module);
  glmrt_b12x_gate_m32_Kernel_Module_Load(&gate_m32_module);
  glmrt_b12x_gate_m64_Kernel_Module_Load(&gate_m64_module);
  glmrt_b12x_gate_m128_Kernel_Module_Load(&gate_m128_module);
  glmrt_b12x_gate_m256_Kernel_Module_Load(&gate_m256_module);
  glmrt_b12x_down_m1_Kernel_Module_Load(&down_m1_module);
  glmrt_b12x_down_m2_Kernel_Module_Load(&down_m2_module);
  glmrt_b12x_down_m4_Kernel_Module_Load(&down_m4_module);
  glmrt_b12x_down_m8_Kernel_Module_Load(&down_m8_module);
  glmrt_b12x_down_m16_Kernel_Module_Load(&down_m16_module);
  glmrt_b12x_down_m32_Kernel_Module_Load(&down_m32_module);
  glmrt_b12x_down_m64_Kernel_Module_Load(&down_m64_module);
  glmrt_b12x_down_m128_Kernel_Module_Load(&down_m128_module);
  glmrt_b12x_down_m256_Kernel_Module_Load(&down_m256_module);
  glmrt_b12x_gate_tp4_m1_Kernel_Module_Load(&gate_tp4_m1_module);
  glmrt_b12x_gate_tp4_m2_Kernel_Module_Load(&gate_tp4_m2_module);
  glmrt_b12x_gate_tp4_m4_Kernel_Module_Load(&gate_tp4_m4_module);
  glmrt_b12x_gate_tp4_m8_Kernel_Module_Load(&gate_tp4_m8_module);
  glmrt_b12x_gate_tp4_m16_Kernel_Module_Load(&gate_tp4_m16_module);
  glmrt_b12x_gate_tp4_m32_Kernel_Module_Load(&gate_tp4_m32_module);
  glmrt_b12x_gate_tp4_m64_Kernel_Module_Load(&gate_tp4_m64_module);
  glmrt_b12x_gate_tp4_m128_Kernel_Module_Load(&gate_tp4_m128_module);
  glmrt_b12x_gate_tp4_m256_Kernel_Module_Load(&gate_tp4_m256_module);
  glmrt_b12x_down_tp4_m1_Kernel_Module_Load(&down_tp4_m1_module);
  glmrt_b12x_down_tp4_m2_Kernel_Module_Load(&down_tp4_m2_module);
  glmrt_b12x_down_tp4_m4_Kernel_Module_Load(&down_tp4_m4_module);
  glmrt_b12x_down_tp4_m8_Kernel_Module_Load(&down_tp4_m8_module);
  glmrt_b12x_down_tp4_m16_Kernel_Module_Load(&down_tp4_m16_module);
  glmrt_b12x_down_tp4_m32_Kernel_Module_Load(&down_tp4_m32_module);
  glmrt_b12x_down_tp4_m64_Kernel_Module_Load(&down_tp4_m64_module);
  glmrt_b12x_down_tp4_m128_Kernel_Module_Load(&down_tp4_m128_module);
  glmrt_b12x_down_tp4_m256_Kernel_Module_Load(&down_tp4_m256_module);
  glmrt_b12x_moe_tp4_m1_Kernel_Module_Load(&moe_tp4_m1_module);
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Kernel_Module_Load(
      &moe_tp4_w4a4_prefill_m256_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_decode_m1_Kernel_Module_Load(
      &moe_tp4_w4a16_decode_m1_module);
  glmrt_b12x_moe_tp4_w4a16_decode_m1_fused_sum_Kernel_Module_Load(
      &moe_tp4_w4a16_decode_m1_fused_sum_module);
#define GLMRT_LOAD_W4A16_M1_PARITY_MODULE(M)                                    \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8_Kernel_Module_Load(           \
      &moe_tp4_w4a16_m1_parity_m##M##_topk8_module);
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(2)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(3)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(4)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(5)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(6)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(7)
  GLMRT_LOAD_W4A16_M1_PARITY_MODULE(8)
#undef GLMRT_LOAD_W4A16_M1_PARITY_MODULE
#define GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(M)                           \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_Kernel_Module_Load(  \
      &moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module);
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(2)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(3)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(4)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(5)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(6)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(7)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE(8)
#undef GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_MODULE
#define GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(M)                      \
  glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_Kernel_Module_Load( \
      &moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module);
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(2)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(3)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(4)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(5)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(6)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(7)
  GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE(8)
#undef GLMRT_LOAD_W4A16_M1_PARITY_GROUPED_WIDE_MODULE
  glmrt_b12x_moe_tp4_w4a16_modelopt_decode_m1_Kernel_Module_Load(
      &moe_tp4_w4a16_modelopt_decode_m1_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m2_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m2_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m4_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m4_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m8_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m8_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m16_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m16_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m32_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m32_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m64_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m64_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m128_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m128_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m256_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m256_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m1024_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m1024_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m2048_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m2048_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_prefill_m512_topk8_Kernel_Module_Load(
      &moe_tp4_w4a16_prefill_m512_topk8_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m1_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m1_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m2_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m2_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m4_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m4_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m8_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m8_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m16_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m16_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m32_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m32_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m64_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m64_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m128_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m128_module);
  glmrt_b12x_moe_tp4_w4a16_top1_m256_Kernel_Module_Load(
      &moe_tp4_w4a16_top1_m256_module);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    b12x_module_init_status = GLMRT_STATUS_INTERNAL_ERROR;
  }
}

void initialize_mixed_w4a4_modules() {
#define GLMRT_LOAD_MIXED_W4A4_MODULES(M)                                        \
  glmrt_b12x_mixed_w4a4_fc1_m##M##_Kernel_Module_Load(                          \
      &mixed_w4a4_fc1_m##M##_module);                                           \
  glmrt_b12x_mixed_w4a16_activation_m##M##_Kernel_Module_Load(                  \
      &mixed_w4a16_activation_m##M##_module);                                   \
  glmrt_b12x_mixed_w4a16_fc2_m##M##_Kernel_Module_Load(                         \
      &mixed_w4a16_fc2_m##M##_module);
  GLMRT_LOAD_MIXED_W4A4_MODULES(1)
  GLMRT_LOAD_MIXED_W4A4_MODULES(2)
  GLMRT_LOAD_MIXED_W4A4_MODULES(4)
  GLMRT_LOAD_MIXED_W4A4_MODULES(8)
  GLMRT_LOAD_MIXED_W4A4_MODULES(16)
  GLMRT_LOAD_MIXED_W4A4_MODULES(32)
  GLMRT_LOAD_MIXED_W4A4_MODULES(64)
  GLMRT_LOAD_MIXED_W4A4_MODULES(128)
  GLMRT_LOAD_MIXED_W4A4_MODULES(256)
#undef GLMRT_LOAD_MIXED_W4A4_MODULES
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    mixed_w4a4_module_init_status = GLMRT_STATUS_INTERNAL_ERROR;
  }
}

#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
void initialize_sparkinfer_source_w4a16_modules() {
#define GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(M)                     \
  glmrt_sparkinfer_source_w4a16_direct_m##M##_topk8_Kernel_Module_Load(         \
      &sparkinfer_source_w4a16_direct_m##M##_topk8_module);
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(1)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(2)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(3)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(4)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(5)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(6)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(7)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE(8)
#undef GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_DIRECT_MODULE
#define GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(M)                    \
  glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Kernel_Module_Load(        \
      &sparkinfer_source_w4a16_prefill_m##M##_topk8_module);
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(16)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(32)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(64)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(128)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(256)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(512)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(1024)
  GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE(2048)
#undef GLMRT_LOAD_SPARKINFER_SOURCE_W4A16_PREFILL_MODULE
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    glmrt_set_last_error_message(cudaGetErrorString(error));
    sparkinfer_source_w4a16_module_init_status =
        GLMRT_STATUS_INTERNAL_ERROR;
  }
}
#endif

int launch_gate(void* a, void* b, void* sfa, void* sfb, void* c, void* alpha, size_t rows,
                cudaStream_t stream) {
  const int m = static_cast<int>(rows);
  if (rows == 1) {
    return cute_dsl_glmrt_b12x_gate_m1_wrapper(&gate_m1_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 2) {
    return cute_dsl_glmrt_b12x_gate_m2_wrapper(&gate_m2_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 4) {
    return cute_dsl_glmrt_b12x_gate_m4_wrapper(&gate_m4_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 8) {
    return cute_dsl_glmrt_b12x_gate_m8_wrapper(&gate_m8_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 16) {
    return cute_dsl_glmrt_b12x_gate_m16_wrapper(&gate_m16_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 32) {
    return cute_dsl_glmrt_b12x_gate_m32_wrapper(&gate_m32_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 64) {
    return cute_dsl_glmrt_b12x_gate_m64_wrapper(&gate_m64_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 128) {
    return cute_dsl_glmrt_b12x_gate_m128_wrapper(&gate_m128_module, a, b, sfa, sfb, c, alpha, m,
                                                 stream);
  }
  return cute_dsl_glmrt_b12x_gate_m256_wrapper(&gate_m256_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
}

int launch_down(void* a, void* b, void* sfa, void* sfb, void* c, void* alpha, size_t rows,
                cudaStream_t stream) {
  const int m = static_cast<int>(rows);
  if (rows == 1) {
    return cute_dsl_glmrt_b12x_down_m1_wrapper(&down_m1_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 2) {
    return cute_dsl_glmrt_b12x_down_m2_wrapper(&down_m2_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 4) {
    return cute_dsl_glmrt_b12x_down_m4_wrapper(&down_m4_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 8) {
    return cute_dsl_glmrt_b12x_down_m8_wrapper(&down_m8_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
  }
  if (rows <= 16) {
    return cute_dsl_glmrt_b12x_down_m16_wrapper(&down_m16_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 32) {
    return cute_dsl_glmrt_b12x_down_m32_wrapper(&down_m32_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 64) {
    return cute_dsl_glmrt_b12x_down_m64_wrapper(&down_m64_module, a, b, sfa, sfb, c, alpha, m,
                                                stream);
  }
  if (rows <= 128) {
    return cute_dsl_glmrt_b12x_down_m128_wrapper(&down_m128_module, a, b, sfa, sfb, c, alpha, m,
                                                 stream);
  }
  return cute_dsl_glmrt_b12x_down_m256_wrapper(&down_m256_module, a, b, sfa, sfb, c, alpha, m,
                                               stream);
}

#define GLMRT_DEFINE_B12X_TP4_LAUNCHER(function_name, prefix)                                  \
  int function_name(void* a, void* b, void* sfa, void* sfb, void* c, void* alpha, size_t rows, \
                    cudaStream_t stream) {                                                     \
    const int m = static_cast<int>(rows);                                                       \
    if (rows == 1) {                                                                            \
      return cute_dsl_glmrt_b12x_##prefix##_m1_wrapper(                                        \
          &prefix##_m1_module, a, b, sfa, sfb, c, alpha, m, stream);                            \
    }                                                                                            \
    if (rows <= 2) {                                                                             \
      return cute_dsl_glmrt_b12x_##prefix##_m2_wrapper(                                         \
          &prefix##_m2_module, a, b, sfa, sfb, c, alpha, m, stream);                            \
    }                                                                                            \
    if (rows <= 4) {                                                                             \
      return cute_dsl_glmrt_b12x_##prefix##_m4_wrapper(                                         \
          &prefix##_m4_module, a, b, sfa, sfb, c, alpha, m, stream);                            \
    }                                                                                            \
    if (rows <= 8) {                                                                             \
      return cute_dsl_glmrt_b12x_##prefix##_m8_wrapper(                                         \
          &prefix##_m8_module, a, b, sfa, sfb, c, alpha, m, stream);                            \
    }                                                                                            \
    if (rows <= 16) {                                                                            \
      return cute_dsl_glmrt_b12x_##prefix##_m16_wrapper(                                        \
          &prefix##_m16_module, a, b, sfa, sfb, c, alpha, m, stream);                           \
    }                                                                                            \
    if (rows <= 32) {                                                                            \
      return cute_dsl_glmrt_b12x_##prefix##_m32_wrapper(                                        \
          &prefix##_m32_module, a, b, sfa, sfb, c, alpha, m, stream);                           \
    }                                                                                            \
    if (rows <= 64) {                                                                            \
      return cute_dsl_glmrt_b12x_##prefix##_m64_wrapper(                                        \
          &prefix##_m64_module, a, b, sfa, sfb, c, alpha, m, stream);                           \
    }                                                                                            \
    if (rows <= 128) {                                                                           \
      return cute_dsl_glmrt_b12x_##prefix##_m128_wrapper(                                       \
          &prefix##_m128_module, a, b, sfa, sfb, c, alpha, m, stream);                          \
    }                                                                                            \
    return cute_dsl_glmrt_b12x_##prefix##_m256_wrapper(                                         \
        &prefix##_m256_module, a, b, sfa, sfb, c, alpha, m, stream);                            \
  }

GLMRT_DEFINE_B12X_TP4_LAUNCHER(launch_gate_tp4, gate_tp4)
GLMRT_DEFINE_B12X_TP4_LAUNCHER(launch_down_tp4, down_tp4)

#undef GLMRT_DEFINE_B12X_TP4_LAUNCHER

struct MixedW4a4Config {
  size_t block_size;
  size_t route_slots;
  size_t route_blocks;
  size_t scratch_elements;
  size_t grid_x;
};

bool mixed_w4a4_config(size_t rows, MixedW4a4Config* config) {
  if (config == nullptr) {
    return false;
  }
#define GLMRT_MIXED_W4A4_CONFIG_CASE(M)                                         \
  case M:                                                                       \
    *config = MixedW4a4Config{                                                   \
        GLMRT_B12X_MIXED_W4A4_M##M##_BLOCK_SIZE,                                \
        GLMRT_B12X_MIXED_W4A4_M##M##_ROUTE_SLOTS,                               \
        GLMRT_B12X_MIXED_W4A4_M##M##_ROUTE_BLOCKS,                              \
        GLMRT_B12X_MIXED_W4A4_M##M##_SCRATCH_ELEMENTS,                          \
        GLMRT_B12X_MIXED_W4A4_M##M##_GRID_X};                                   \
    return true;
  switch (rows) {
    GLMRT_MIXED_W4A4_CONFIG_CASE(1)
    GLMRT_MIXED_W4A4_CONFIG_CASE(2)
    GLMRT_MIXED_W4A4_CONFIG_CASE(4)
    GLMRT_MIXED_W4A4_CONFIG_CASE(8)
    GLMRT_MIXED_W4A4_CONFIG_CASE(16)
    GLMRT_MIXED_W4A4_CONFIG_CASE(32)
    GLMRT_MIXED_W4A4_CONFIG_CASE(64)
    GLMRT_MIXED_W4A4_CONFIG_CASE(128)
    GLMRT_MIXED_W4A4_CONFIG_CASE(256)
    default:
      return false;
  }
#undef GLMRT_MIXED_W4A4_CONFIG_CASE
}

#define GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(M)                                   \
  int launch_mixed_w4a4_fc1_m##M(                                               \
      const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers,                     \
      cudaStream_t stream) {                                                     \
    return cute_dsl_glmrt_b12x_mixed_w4a4_fc1_m##M##_wrapper(                   \
        &mixed_w4a4_fc1_m##M##_module, buffers->input_packed.ptr,               \
        buffers->w13_weight_source.ptr, buffers->input_scale.ptr,               \
        buffers->w13_scale_source.ptr, buffers->fc1_output.ptr,                 \
        buffers->w13_global_scale.ptr, static_cast<int32_t>(M), stream);         \
  }                                                                              \
  int launch_mixed_w4a16_activation_m##M(                                       \
      const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers,                     \
      cudaStream_t stream) {                                                     \
    glmrt_b12x_mixed_w4a16_activation_m##M##_Tensor_fc1_flat_t fc1{             \
        buffers->fc1_reordered.ptr};                                             \
    glmrt_b12x_mixed_w4a16_activation_m##M##_Tensor_activated_flat_t activated{ \
        buffers->activated.ptr};                                                 \
    return cute_dsl_glmrt_b12x_mixed_w4a16_activation_m##M##_wrapper(           \
        &mixed_w4a16_activation_m##M##_module, &fc1, &activated,                \
        static_cast<int32_t>(M), stream);                                        \
  }                                                                              \
  int launch_mixed_w4a16_fc2_m##M(                                              \
      const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers,                     \
      size_t grid_x, cudaStream_t stream) {                                      \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_a_bf16_flat_t input{               \
        buffers->activated.ptr};                                                 \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_b_i32_flat_t weight{               \
        buffers->w2_weight_packed.ptr};                                          \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_c_bf16_flat_t output{              \
        buffers->output.ptr};                                                    \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_scales_i32_flat_t scale{           \
        buffers->w2_scale_packed.ptr};                                           \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_global_scale_t global_scale{       \
        buffers->w2_global_scale.ptr};                                           \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_packed_route_indices_t routes{     \
        buffers->packed_route_indices.ptr};                                      \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_block_expert_ids_t block_experts{  \
        buffers->block_expert_ids.ptr};                                          \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_packed_route_count_t route_count{  \
        buffers->packed_route_count.ptr};                                        \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_topk_weights_flat_t topk_weights{  \
        buffers->topk_weights.ptr};                                              \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_c_tmp_f32_flat_t scratch{          \
        buffers->fc2_scratch.ptr};                                               \
    glmrt_b12x_mixed_w4a16_fc2_m##M##_Tensor_locks_i32_flat_t locks{            \
        buffers->locks.ptr};                                                     \
    return cute_dsl_glmrt_b12x_mixed_w4a16_fc2_m##M##_wrapper(                  \
        &mixed_w4a16_fc2_m##M##_module, &input, &weight, &output, &scale,       \
        &global_scale, &routes, &block_experts, &route_count, &topk_weights,    \
        &scratch, &locks, static_cast<int32_t>(M),                              \
        static_cast<int32_t>(grid_x), stream);                                  \
  }
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(1)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(2)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(4)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(8)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(16)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(32)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(64)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(128)
GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS(256)
#undef GLMRT_DEFINE_MIXED_W4A4_LAUNCHERS

#define GLMRT_DEFINE_MIXED_W4A4_DISPATCH(FUNCTION)                              \
  int FUNCTION(const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers,           \
               size_t rows, cudaStream_t stream) {                              \
    switch (rows) {                                                              \
      case 1: return FUNCTION##_m1(buffers, stream);                            \
      case 2: return FUNCTION##_m2(buffers, stream);                            \
      case 4: return FUNCTION##_m4(buffers, stream);                            \
      case 8: return FUNCTION##_m8(buffers, stream);                            \
      case 16: return FUNCTION##_m16(buffers, stream);                          \
      case 32: return FUNCTION##_m32(buffers, stream);                          \
      case 64: return FUNCTION##_m64(buffers, stream);                          \
      case 128: return FUNCTION##_m128(buffers, stream);                        \
      case 256: return FUNCTION##_m256(buffers, stream);                        \
      default: return -1;                                                        \
    }                                                                            \
  }
GLMRT_DEFINE_MIXED_W4A4_DISPATCH(launch_mixed_w4a4_fc1)
GLMRT_DEFINE_MIXED_W4A4_DISPATCH(launch_mixed_w4a16_activation)
#undef GLMRT_DEFINE_MIXED_W4A4_DISPATCH

int launch_mixed_w4a16_fc2(
    const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers, size_t rows,
    size_t grid_x, cudaStream_t stream) {
  switch (rows) {
    case 1: return launch_mixed_w4a16_fc2_m1(buffers, grid_x, stream);
    case 2: return launch_mixed_w4a16_fc2_m2(buffers, grid_x, stream);
    case 4: return launch_mixed_w4a16_fc2_m4(buffers, grid_x, stream);
    case 8: return launch_mixed_w4a16_fc2_m8(buffers, grid_x, stream);
    case 16: return launch_mixed_w4a16_fc2_m16(buffers, grid_x, stream);
    case 32: return launch_mixed_w4a16_fc2_m32(buffers, grid_x, stream);
    case 64: return launch_mixed_w4a16_fc2_m64(buffers, grid_x, stream);
    case 128: return launch_mixed_w4a16_fc2_m128(buffers, grid_x, stream);
    case 256: return launch_mixed_w4a16_fc2_m256(buffers, grid_x, stream);
    default: return -1;
  }
}

glmrt_status_t validate_mixed_w4a4_buffers(
    const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers, size_t rows,
    const MixedW4a4Config& config) {
  const size_t input_scale_bytes =
      align_up_size(rows, 128) * align_up_size(kB12xHidden / 16, 4);
  constexpr size_t w13_weight_bytes =
      2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xOutput * kB12xTp4Intermediate / 16;
  if (buffers == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      buffer_has_bytes(buffers->input_packed,
                       rows * kB12xHidden / 2) &&
      buffer_has_bytes(buffers->input_scale, input_scale_bytes) &&
      buffer_has_bytes(buffers->w13_weight_source, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w13_scale_source, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w13_global_scale, sizeof(float)) &&
      buffer_has_bytes(buffers->fc1_output,
                       rows * 2 * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->fc1_reordered,
                       rows * 2 * kB12xTp4Intermediate * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->activated,
                       rows * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w2_weight_packed, w2_weight_bytes) &&
      buffer_has_bytes(buffers->w2_scale_packed, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w2_global_scale, sizeof(float)) &&
      buffer_has_bytes(buffers->output,
                       rows * kB12xOutput * sizeof(uint16_t)) &&
      buffer_has_bytes(
          buffers->packed_route_indices,
          config.route_slots * sizeof(int32_t)) &&
      buffer_has_bytes(
          buffers->block_expert_ids,
          config.route_blocks * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, rows * sizeof(float)) &&
      buffer_has_bytes(
          buffers->fc2_scratch,
          config.scratch_elements * sizeof(float)) &&
      buffer_has_bytes(buffers->locks,
                       kMixedW4a4LockElements * sizeof(int32_t));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t validate_b12x_buffers(const glmrt_b12x_spark_mlp_buffers_t* buffers, size_t rows,
                                     size_t hidden_dim, size_t intermediate_dim,
                                     size_t output_dim, bool require_bf16_input) {
  if (buffers == nullptr || rows == 0 || rows > kB12xMaxRows || hidden_dim != kB12xHidden ||
      (intermediate_dim != kB12xIntermediate && intermediate_dim != kB12xTp4Intermediate) ||
      output_dim != kB12xOutput) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t input_scale_bytes = align_up_size(rows, 128) * align_up_size(hidden_dim / 16, 4);
  const size_t activation_scale_bytes =
      align_up_size(rows, 128) * align_up_size(intermediate_dim / 16, 4);
  const bool valid =
      (!require_bf16_input ||
       buffer_has_bytes(buffers->input, rows * hidden_dim * sizeof(uint16_t))) &&
      buffer_has_bytes(buffers->gate_weight, intermediate_dim * hidden_dim / 2) &&
      buffer_has_bytes(buffers->gate_scale, intermediate_dim * hidden_dim / 16) &&
      buffer_has_bytes(buffers->up_weight, intermediate_dim * hidden_dim / 2) &&
      buffer_has_bytes(buffers->up_scale, intermediate_dim * hidden_dim / 16) &&
      buffer_has_bytes(buffers->down_weight, output_dim * intermediate_dim / 2) &&
      buffer_has_bytes(buffers->down_scale, output_dim * intermediate_dim / 16) &&
      buffer_has_bytes(buffers->output, rows * output_dim * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->input_packed, rows * hidden_dim / 2) &&
      buffer_has_bytes(buffers->input_scale, input_scale_bytes) &&
      buffer_has_bytes(buffers->gate_output, rows * intermediate_dim * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->up_output, rows * intermediate_dim * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->activation_packed, rows * intermediate_dim / 2) &&
      buffer_has_bytes(buffers->activation_scale, activation_scale_bytes) &&
      buffer_has_bytes(buffers->gate_scale_swizzled, intermediate_dim * hidden_dim / 16) &&
      buffer_has_bytes(buffers->up_scale_swizzled, intermediate_dim * hidden_dim / 16) &&
      buffer_has_bytes(buffers->down_scale_swizzled, output_dim * intermediate_dim / 16) &&
      buffer_has_bytes(buffers->alphas, 3 * sizeof(float));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t check_aot_launch(int result, const char* label) {
  if (result == 0) {
    return GLMRT_STATUS_OK;
  }
  glmrt_set_last_error_message(label);
  return GLMRT_STATUS_INTERNAL_ERROR;
}

glmrt_status_t validate_b12x_moe_tp4_m1_buffers(
    const glmrt_b12x_spark_moe_tp4_m1_buffers_t* buffers,
    size_t input_payload_stride_bytes) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  constexpr size_t w13_weight_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 16;
  constexpr size_t expert_scalars_bytes = kB12xExperts * sizeof(float);
  constexpr size_t intermediate_bytes =
      kB12xTopK * kB12xTp4Intermediate * sizeof(uint16_t);
  if (buffers == nullptr || input_payload_stride_bytes < input_payload_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      buffer_has_bytes(buffers->input_payload, input_payload_stride_bytes) &&
      buffer_has_bytes(buffers->input_bf16, kB12xHidden * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_weight, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w13_scale, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w1_alphas, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->a1_gscale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->a2_gscale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->intermediate, intermediate_bytes) &&
      buffer_has_bytes(buffers->w2_weight, w2_weight_bytes) &&
      buffer_has_bytes(buffers->w2_scale, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w2_alphas, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->topk_ids, kB12xTopK * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, kB12xTopK * sizeof(float)) &&
      buffer_has_bytes(buffers->output, kB12xOutput * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->barrier_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->barrier_epoch, sizeof(int32_t));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t validate_b12x_w4a4_prefill_buffers(
    const glmrt_b12x_spark_w4a4_moe_buffers_t* buffers, size_t rows) {
  constexpr size_t w13_weight_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 16;
  constexpr size_t expert_scalars_bytes = kB12xExperts * sizeof(float);
  if (buffers == nullptr || rows == 0 || rows > kB12xMaxRows) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const bool valid =
      buffer_has_bytes(buffers->input, rows * kB12xHidden * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->topk_ids, rows * kB12xTopK * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, rows * kB12xTopK * sizeof(float)) &&
      buffer_has_bytes(buffers->w13_weight, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w13_scale, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w1_alphas, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->a1_gscale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->w2_weight, w2_weight_bytes) &&
      buffer_has_bytes(buffers->w2_scale, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w2_alphas, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->a2_gscale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->output, rows * kB12xOutput * sizeof(uint16_t)) &&
      buffer_has_bytes(
          buffers->scratch,
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_SCRATCH_BYTES);
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

int launch_w4a4_prefill_m256_topk8(
    const glmrt_b12x_spark_w4a4_moe_buffers_t* buffers, size_t rows,
    cudaStream_t stream) {
  auto* scratch = static_cast<uint8_t*>(buffers->scratch.ptr);
#define GLMRT_W4A4_SCRATCH_PTR(name) \
  (scratch + GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_##name##_OFFSET)
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_barrier_count_t
      barrier_count{GLMRT_W4A4_SCRATCH_PTR(BARRIER_COUNT)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_barrier_epoch_t
      barrier_epoch{GLMRT_W4A4_SCRATCH_PTR(BARRIER_EPOCH)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_pair_head_t pair_head{
      GLMRT_W4A4_SCRATCH_PTR(PAIR_HEAD)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_producers_done_count_t
      producers_done_count{GLMRT_W4A4_SCRATCH_PTR(PRODUCERS_DONE_COUNT)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_all_work_published_t
      all_work_published{GLMRT_W4A4_SCRATCH_PTR(ALL_WORK_PUBLISHED)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_task_head_t task_head{
      GLMRT_W4A4_SCRATCH_PTR(TASK_HEAD)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_task_tail_t task_tail{
      GLMRT_W4A4_SCRATCH_PTR(TASK_TAIL)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_b_w13_t w13{
      buffers->w13_weight.ptr};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_b_down_t w2{
      buffers->w2_weight.ptr};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_row_counts_t row_counts{
      GLMRT_W4A4_SCRATCH_PTR(ROW_COUNTS)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_expert_write_rows_t
      expert_write_rows{GLMRT_W4A4_SCRATCH_PTR(EXPERT_WRITE_ROWS)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_expert_tile_base_t
      expert_tile_base{GLMRT_W4A4_SCRATCH_PTR(EXPERT_TILE_BASE)};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_input_global_scale_t
      input_global_scale{buffers->a1_gscale.ptr};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_alpha_t alpha{
      buffers->w1_alphas.ptr};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_down_alpha_t down_alpha{
      buffers->w2_alphas.ptr};
  glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_Tensor_global_scale_t global_scale{
      buffers->a2_gscale.ptr};

  const int result =
      cute_dsl_glmrt_b12x_moe_tp4_w4a4_prefill_m256_topk8_wrapper(
          &moe_tp4_w4a4_prefill_m256_topk8_module, buffers->input.ptr,
          buffers->topk_ids.ptr, buffers->topk_weights.ptr,
          GLMRT_W4A4_SCRATCH_PTR(PACKED_INPUT),
          GLMRT_W4A4_SCRATCH_PTR(PACKED_INPUT_SCALE),
          GLMRT_W4A4_SCRATCH_PTR(PACKED_INPUT),
          GLMRT_W4A4_SCRATCH_PTR(PACKED_INPUT_SCALE),
          GLMRT_W4A4_SCRATCH_PTR(ROUTE_OUTPUT), &barrier_count, &barrier_epoch,
          &pair_head, &producers_done_count, &all_work_published, &task_head,
          &task_tail, GLMRT_W4A4_SCRATCH_PTR(TASK_READY),
          GLMRT_W4A4_SCRATCH_PTR(TASK_EXPERT),
          GLMRT_W4A4_SCRATCH_PTR(TASK_M_TILE),
          GLMRT_W4A4_SCRATCH_PTR(TASK_SLICE_BEGIN),
          GLMRT_W4A4_SCRATCH_PTR(TASK_SLICE_COUNT),
          GLMRT_W4A4_SCRATCH_PTR(TASK_VALID_ROWS),
          GLMRT_W4A4_SCRATCH_PTR(TILE_WRITE_COUNT), &w13,
          buffers->w13_scale.ptr, &w2, buffers->w2_scale.ptr, &row_counts,
          &expert_write_rows, &expert_tile_base, &input_global_scale, &alpha,
          &down_alpha, &global_scale, buffers->output.ptr,
          GLMRT_W4A4_SCRATCH_PTR(TOKEN_MAP),
          GLMRT_W4A4_SCRATCH_PTR(TOKEN_WEIGHTS), static_cast<int32_t>(rows),
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_MAX_ROUTED_ROWS,
          static_cast<int32_t>(rows),
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_ROWS_PADDED,
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_TASK_CAPACITY,
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_PHYSICAL_TILES,
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_GRID_X, stream);
#undef GLMRT_W4A4_SCRATCH_PTR
  return result;
}

glmrt_status_t launch_b12x_mlp_from_quantized(
    const glmrt_b12x_spark_mlp_buffers_t* buffers, size_t rows, size_t intermediate_dim,
    float gate_scale_2, float up_scale_2, float down_scale_2, cudaStream_t stream) {
  auto* gate_launcher =
      intermediate_dim == kB12xTp4Intermediate ? &launch_gate_tp4 : &launch_gate;
  auto* down_launcher =
      intermediate_dim == kB12xTp4Intermediate ? &launch_down_tp4 : &launch_down;
  write_alpha_kernel<<<1, 1, 0, stream>>>(static_cast<float*>(buffers->alphas.ptr), gate_scale_2,
                                          up_scale_2, down_scale_2);
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  status = check_aot_launch(
      gate_launcher(buffers->input_packed.ptr, buffers->gate_weight.ptr,
                    buffers->input_scale.ptr, buffers->gate_scale_swizzled.ptr,
                    buffers->gate_output.ptr, buffers->alphas.ptr, rows, stream),
      "B12X Spark gate AOT launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = check_aot_launch(
      gate_launcher(buffers->input_packed.ptr, buffers->up_weight.ptr,
                    buffers->input_scale.ptr, buffers->up_scale_swizzled.ptr,
                    buffers->up_output.ptr, static_cast<float*>(buffers->alphas.ptr) + 1, rows,
                    stream),
      "B12X Spark up AOT launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  const size_t activation_scale_bytes =
      align_up_size(rows, 128) * align_up_size(intermediate_dim / 16, 4);
  cudaError_t error = cudaMemsetAsync(buffers->activation_scale.ptr, 0, activation_scale_bytes,
                                      stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  const dim3 activation_grid(static_cast<unsigned int>(intermediate_dim / 16),
                             static_cast<unsigned int>(rows));
  silu_product_quantize_nvfp4_b12x_kernel<<<activation_grid, kQuantThreads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->gate_output.ptr),
      static_cast<const uint16_t*>(buffers->up_output.ptr),
      static_cast<uint8_t*>(buffers->activation_packed.ptr),
      static_cast<uint8_t*>(buffers->activation_scale.ptr), rows, intermediate_dim);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      down_launcher(buffers->activation_packed.ptr, buffers->down_weight.ptr,
                    buffers->activation_scale.ptr, buffers->down_scale_swizzled.ptr,
                    buffers->output.ptr, static_cast<float*>(buffers->alphas.ptr) + 2, rows,
                    stream),
      "B12X Spark down AOT launch failed");
}

#define GLMRT_DEFINE_W4A16_LAUNCH(function_name, prefix, module_name, default_grid_x)          \
  int function_name##_grid(const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,                \
                           size_t active_m, int grid_x, cudaStream_t stream) {                  \
    prefix##_Tensor_w13_i32_flat_t w13{buffers->w13_weight.ptr};                               \
    prefix##_Tensor_w2_i32_flat_t w2{buffers->w2_weight.ptr};                                  \
    prefix##_Tensor_fc1_bf16_flat_t fc1{buffers->fc1_output.ptr};                              \
    prefix##_Tensor_activated_bf16_flat_t activated{buffers->activated.ptr};                   \
    prefix##_Tensor_fc2_bf16_flat_t fc2{buffers->output.ptr};                                  \
    prefix##_Tensor_w13_scales_i32_flat_t w13_scale{buffers->w13_scale.ptr};                   \
    prefix##_Tensor_w2_scales_i32_flat_t w2_scale{buffers->w2_scale.ptr};                      \
    prefix##_Tensor_w13_global_scale_t w13_global{buffers->w13_global_scale.ptr};              \
    prefix##_Tensor_w2_global_scale_t w2_global{buffers->w2_global_scale.ptr};                 \
    prefix##_Tensor_packed_route_indices_t packed_routes{buffers->packed_route_indices.ptr};   \
    prefix##_Tensor_block_expert_ids_t block_experts{buffers->block_expert_ids.ptr};           \
    prefix##_Tensor_packed_route_count_t route_count{buffers->packed_route_count.ptr};          \
    prefix##_Tensor_activation_amax_flat_t activation_amax{buffers->w13_global_scale.ptr};      \
    prefix##_Tensor_fc1_c_tmp_f32_flat_t fc1_scratch{buffers->fc1_scratch.ptr};                \
    prefix##_Tensor_fc2_c_tmp_f32_flat_t fc2_scratch{buffers->fc2_scratch.ptr};                \
    prefix##_Tensor_locks_i32_flat_t locks{buffers->locks.ptr};                                \
    return cute_dsl_##prefix##_wrapper(                                                         \
        &module_name, buffers->input.ptr, &w13, &w2, &fc1, &activated, &fc2, &w13_scale,        \
        &w2_scale, &w13_global, &w2_global, &packed_routes, &block_experts, &route_count,        \
        &activation_amax, 0, buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &locks,     \
        static_cast<int32_t>(active_m), static_cast<int32_t>(grid_x), stream);                  \
  }                                                                                            \
  int function_name(const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t active_m,      \
                    cudaStream_t stream) {                                                     \
    return function_name##_grid(buffers, active_m, static_cast<int>(default_grid_x), stream);  \
  }

GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_decode_m1, glmrt_b12x_moe_tp4_w4a16_decode_m1,
    moe_tp4_w4a16_decode_m1_module, w4a16_decode_grid_x())
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_decode_m1_fused_sum,
    glmrt_b12x_moe_tp4_w4a16_decode_m1_fused_sum,
    moe_tp4_w4a16_decode_m1_fused_sum_module,
    GLMRT_B12X_W4A16_DECODE_M1_FUSED_SUM_GRID_X)
#define GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(M)                                  \
  GLMRT_DEFINE_W4A16_LAUNCH(                                                    \
      launch_w4a16_m1_parity_m##M##_topk8,                                     \
      glmrt_b12x_moe_tp4_w4a16_m1_parity_m##M##_topk8,                         \
      moe_tp4_w4a16_m1_parity_m##M##_topk8_module,                             \
      GLMRT_B12X_W4A16_M1_PARITY_M##M##_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(2)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(3)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(4)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(5)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(6)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(7)
GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_LAUNCH
#define GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(M)                         \
  GLMRT_DEFINE_W4A16_LAUNCH(                                                   \
      launch_w4a16_m1_parity_grouped_m##M##_topk8,                            \
      glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8,                \
      moe_tp4_w4a16_m1_parity_grouped_m##M##_topk8_module,                    \
      GLMRT_B12X_W4A16_M1_PARITY_GROUPED_M##M##_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(2)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(3)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(4)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(5)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(6)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(7)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_LAUNCH
#define GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(M)                    \
  GLMRT_DEFINE_W4A16_LAUNCH(                                                   \
      launch_w4a16_m1_parity_grouped_wide_m##M##_topk8,                       \
      glmrt_b12x_moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8,           \
      moe_tp4_w4a16_m1_parity_grouped_wide_m##M##_topk8_module,               \
      GLMRT_B12X_W4A16_M1_PARITY_GROUPED_WIDE_M##M##_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(2)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(3)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(4)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(5)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(6)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(7)
GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH(8)
#undef GLMRT_DEFINE_W4A16_M1_PARITY_GROUPED_WIDE_LAUNCH
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_modelopt_decode_m1,
    glmrt_b12x_moe_tp4_w4a16_modelopt_decode_m1,
    moe_tp4_w4a16_modelopt_decode_m1_module,
    GLMRT_B12X_W4A16_MODELOPT_DECODE_M1_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m2_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m2_topk8,
    moe_tp4_w4a16_prefill_m2_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M2_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m4_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m4_topk8,
    moe_tp4_w4a16_prefill_m4_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M4_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m8_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m8_topk8,
    moe_tp4_w4a16_prefill_m8_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M8_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m16_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m16_topk8,
    moe_tp4_w4a16_prefill_m16_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M16_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m32_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m32_topk8,
    moe_tp4_w4a16_prefill_m32_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M32_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m64_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m64_topk8,
    moe_tp4_w4a16_prefill_m64_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M64_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m128_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m128_topk8,
    moe_tp4_w4a16_prefill_m128_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M128_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m256_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m256_topk8,
    moe_tp4_w4a16_prefill_m256_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M256_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m512_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m512_topk8,
    moe_tp4_w4a16_prefill_m512_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M512_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m1024_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m1024_topk8,
    moe_tp4_w4a16_prefill_m1024_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M1024_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_prefill_m2048_topk8,
    glmrt_b12x_moe_tp4_w4a16_prefill_m2048_topk8,
    moe_tp4_w4a16_prefill_m2048_topk8_module,
    GLMRT_B12X_W4A16_PREFILL_M2048_TOPK8_GRID_X)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m1, glmrt_b12x_moe_tp4_w4a16_top1_m1,
    moe_tp4_w4a16_top1_m1_module, kB12xW4a16Top1M1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m2, glmrt_b12x_moe_tp4_w4a16_top1_m2,
    moe_tp4_w4a16_top1_m2_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m4, glmrt_b12x_moe_tp4_w4a16_top1_m4,
    moe_tp4_w4a16_top1_m4_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m8, glmrt_b12x_moe_tp4_w4a16_top1_m8,
    moe_tp4_w4a16_top1_m8_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m16, glmrt_b12x_moe_tp4_w4a16_top1_m16,
    moe_tp4_w4a16_top1_m16_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m32, glmrt_b12x_moe_tp4_w4a16_top1_m32,
    moe_tp4_w4a16_top1_m32_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m64, glmrt_b12x_moe_tp4_w4a16_top1_m64,
    moe_tp4_w4a16_top1_m64_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m128, glmrt_b12x_moe_tp4_w4a16_top1_m128,
    moe_tp4_w4a16_top1_m128_module, kB12xW4a16Top1GridX)
GLMRT_DEFINE_W4A16_LAUNCH(
    launch_w4a16_top1_m256, glmrt_b12x_moe_tp4_w4a16_top1_m256,
    moe_tp4_w4a16_top1_m256_module, kB12xW4a16Top1GridX)

#undef GLMRT_DEFINE_W4A16_LAUNCH

#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
#define GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(M)                   \
  int launch_sparkinfer_source_w4a16_direct_m##M(                              \
      const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,                     \
      glmrt_device_buffer_t topk_ids, size_t active_m,                         \
      cudaStream_t stream) {                                                    \
    glmrt_sparkinfer_source_w4a16_direct_m##M##_topk8_Tensor_barrier_count_t   \
        barrier_count{buffers->barrier_count.ptr};                              \
    glmrt_sparkinfer_source_w4a16_direct_m##M##_topk8_Tensor_barrier_epoch_t   \
        barrier_epoch{buffers->barrier_epoch.ptr};                              \
    return cute_dsl_glmrt_sparkinfer_source_w4a16_direct_m##M##_topk8_wrapper( \
        &sparkinfer_source_w4a16_direct_m##M##_topk8_module,                   \
        buffers->input.ptr, buffers->w13_weight.ptr,                           \
        buffers->w13_scale.ptr, buffers->micro_w13_global_scale.ptr,           \
        buffers->micro_w13_global_scale.ptr,                                   \
        buffers->micro_w2_global_scale.ptr, buffers->activated.ptr,            \
        buffers->w2_weight.ptr, buffers->w2_scale.ptr,                         \
        buffers->micro_w2_global_scale.ptr, topk_ids.ptr,                      \
        buffers->topk_weights.ptr, buffers->output.ptr, &barrier_count,        \
        &barrier_epoch, static_cast<int32_t>(active_m),                         \
        GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_M##M##_TOPK8_GRID_X, stream);      \
  }
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(1)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(2)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(3)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(4)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(5)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(6)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(7)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH(8)
#undef GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_DIRECT_LAUNCH

#define GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(M)                  \
  int launch_sparkinfer_source_w4a16_prefill_m##M(                             \
      const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,                     \
      size_t active_m, cudaStream_t stream) {                                   \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w13_i32_flat_t  \
        w13{buffers->w13_weight.ptr};                                           \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w2_i32_flat_t   \
        w2{buffers->w2_weight.ptr};                                             \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_fc1_bf16_flat_t \
        fc1{buffers->fc1_output.ptr};                                           \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_activated_bf16_flat_t \
        activated{buffers->activated.ptr};                                      \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_fc2_bf16_flat_t \
        fc2{buffers->output.ptr};                                               \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w13_scales_i32_flat_t \
        w13_scale{buffers->w13_scale.ptr};                                      \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w2_scales_i32_flat_t \
        w2_scale{buffers->w2_scale.ptr};                                        \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w13_global_scale_t \
        w13_global{buffers->w13_global_scale.ptr};                              \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_w2_global_scale_t \
        w2_global{buffers->w2_global_scale.ptr};                                \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_packed_route_indices_t \
        routes{buffers->packed_route_indices.ptr};                              \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_block_expert_ids_t \
        block_experts{buffers->block_expert_ids.ptr};                           \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_packed_route_count_t \
        route_count{buffers->packed_route_count.ptr};                           \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_activation_amax_flat_t \
        activation_amax{buffers->w13_global_scale.ptr};                         \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_fc1_c_tmp_f32_flat_t \
        fc1_scratch{buffers->fc1_scratch.ptr};                                  \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_fc2_c_tmp_f32_flat_t \
        fc2_scratch{buffers->fc2_scratch.ptr};                                  \
    glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_Tensor_locks_i32_flat_t \
        locks{buffers->locks.ptr};                                              \
    return cute_dsl_glmrt_sparkinfer_source_w4a16_prefill_m##M##_topk8_wrapper( \
        &sparkinfer_source_w4a16_prefill_m##M##_topk8_module,                  \
        buffers->input.ptr, &w13, &w2, &fc1, &activated, &fc2,                 \
        &w13_scale, &w2_scale, &w13_global, &w2_global, &routes,               \
        &block_experts, &route_count, &activation_amax, 0,                     \
        buffers->topk_weights.ptr, &fc1_scratch, &fc2_scratch, &locks,         \
        static_cast<int32_t>(active_m),                                         \
        GLMRT_SPARKINFER_SOURCE_W4A16_PREFILL_M##M##_TOPK8_GRID_X, stream);     \
  }
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(16)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(32)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(64)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(128)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(256)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(512)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(1024)
GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH(2048)
#undef GLMRT_DEFINE_SPARKINFER_SOURCE_W4A16_PREFILL_LAUNCH

using SparkInferSourceW4A16DirectLaunchFn = int (*)(
    const glmrt_b12x_spark_w4a16_moe_buffers_t*, glmrt_device_buffer_t,
    size_t, cudaStream_t);
using SparkInferSourceW4A16PrefillLaunchFn = int (*)(
    const glmrt_b12x_spark_w4a16_moe_buffers_t*, size_t, cudaStream_t);

SparkInferSourceW4A16DirectLaunchFn
sparkinfer_source_w4a16_direct_launcher(size_t rows) {
  switch (rows) {
    case 1: return &launch_sparkinfer_source_w4a16_direct_m1;
    case 2: return &launch_sparkinfer_source_w4a16_direct_m2;
    case 3: return &launch_sparkinfer_source_w4a16_direct_m3;
    case 4: return &launch_sparkinfer_source_w4a16_direct_m4;
    case 5: return &launch_sparkinfer_source_w4a16_direct_m5;
    case 6: return &launch_sparkinfer_source_w4a16_direct_m6;
    case 7: return &launch_sparkinfer_source_w4a16_direct_m7;
    case 8: return &launch_sparkinfer_source_w4a16_direct_m8;
    default: return nullptr;
  }
}

SparkInferSourceW4A16PrefillLaunchFn
sparkinfer_source_w4a16_prefill_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 16: return &launch_sparkinfer_source_w4a16_prefill_m16;
    case 32: return &launch_sparkinfer_source_w4a16_prefill_m32;
    case 64: return &launch_sparkinfer_source_w4a16_prefill_m64;
    case 128: return &launch_sparkinfer_source_w4a16_prefill_m128;
    case 256: return &launch_sparkinfer_source_w4a16_prefill_m256;
    case 512: return &launch_sparkinfer_source_w4a16_prefill_m512;
    case 1024: return &launch_sparkinfer_source_w4a16_prefill_m1024;
    case 2048: return &launch_sparkinfer_source_w4a16_prefill_m2048;
    default: return nullptr;
  }
}
#endif

glmrt_status_t validate_w4a16_moe_buffers(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t capacity_rows,
    size_t top_k) {
  constexpr size_t w13_weight_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 16;
  constexpr size_t expert_scalars_bytes = kB12xExperts * sizeof(float);
  constexpr size_t max_packed_route_slots =
      GLMRT_B12X_W4A16_PREFILL_M2048_TOPK8_PACKED_ROUTE_SLOTS;
  constexpr size_t max_route_blocks =
      GLMRT_B12X_W4A16_PREFILL_M2048_TOPK8_MAX_M_BLOCKS;
  constexpr size_t max_scratch_elements = 1572864;
  if (buffers == nullptr || capacity_rows == 0 || capacity_rows > kB12xW4a16MaxRows ||
      (top_k != 1 && top_k != kB12xTopK)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t output_rows =
      top_k == kB12xTopK ? capacity_rows * top_k : capacity_rows;
  const bool valid =
      buffer_has_bytes(buffers->input, capacity_rows * kB12xHidden * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_weight, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w2_weight, w2_weight_bytes) &&
      buffer_has_bytes(buffers->fc1_output,
                       capacity_rows * top_k * 2 * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->activated,
                       capacity_rows * top_k * kB12xTp4Intermediate *
                           sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->output, output_rows * kB12xOutput * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_scale, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w2_scale, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w13_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->w2_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->packed_route_indices,
                       max_packed_route_slots * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->block_expert_ids, max_route_blocks * sizeof(int32_t)) &&
      buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
      buffer_has_bytes(buffers->topk_weights, capacity_rows * top_k * sizeof(float)) &&
      buffer_has_bytes(buffers->fc1_scratch, max_scratch_elements * sizeof(float)) &&
      buffer_has_bytes(buffers->fc2_scratch, max_scratch_elements * sizeof(float)) &&
      buffer_has_bytes(buffers->locks, kB12xW4a16LockElements * sizeof(int32_t));
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

glmrt_status_t reset_w4a16_locks_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, cudaStream_t stream) {
  return status_from_cuda(
      cudaMemsetAsync(buffers->locks.ptr, 0, kB12xW4a16LockElements * sizeof(int32_t), stream));
}

using W4A16LaunchFn = int (*)(const glmrt_b12x_spark_w4a16_moe_buffers_t*, size_t,
                              cudaStream_t);
using W4A16GridLaunchFn = int (*)(const glmrt_b12x_spark_w4a16_moe_buffers_t*, size_t,
                                  int, cudaStream_t);

W4A16LaunchFn w4a16_m1_parity_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_m1_parity_grouped_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_grouped_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_grouped_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_grouped_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_grouped_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_grouped_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_grouped_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_grouped_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_m1_parity_grouped_wide_launcher(size_t rows) {
  switch (rows) {
    case 2:
      return &launch_w4a16_m1_parity_grouped_wide_m2_topk8;
    case 3:
      return &launch_w4a16_m1_parity_grouped_wide_m3_topk8;
    case 4:
      return &launch_w4a16_m1_parity_grouped_wide_m4_topk8;
    case 5:
      return &launch_w4a16_m1_parity_grouped_wide_m5_topk8;
    case 6:
      return &launch_w4a16_m1_parity_grouped_wide_m6_topk8;
    case 7:
      return &launch_w4a16_m1_parity_grouped_wide_m7_topk8;
    case 8:
      return &launch_w4a16_m1_parity_grouped_wide_m8_topk8;
    default:
      return nullptr;
  }
}

W4A16LaunchFn w4a16_top1_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 1:
      return &launch_w4a16_top1_m1;
    case 2:
      return &launch_w4a16_top1_m2;
    case 4:
      return &launch_w4a16_top1_m4;
    case 8:
      return &launch_w4a16_top1_m8;
    case 16:
      return &launch_w4a16_top1_m16;
    case 32:
      return &launch_w4a16_top1_m32;
    case 64:
      return &launch_w4a16_top1_m64;
    case 128:
      return &launch_w4a16_top1_m128;
    case 256:
      return &launch_w4a16_top1_m256;
    default:
      return nullptr;
  }
}

W4A16GridLaunchFn w4a16_top1_grid_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 1:
      return &launch_w4a16_top1_m1_grid;
    case 2:
      return &launch_w4a16_top1_m2_grid;
    case 4:
      return &launch_w4a16_top1_m4_grid;
    case 8:
      return &launch_w4a16_top1_m8_grid;
    case 16:
      return &launch_w4a16_top1_m16_grid;
    case 32:
      return &launch_w4a16_top1_m32_grid;
    case 64:
      return &launch_w4a16_top1_m64_grid;
    case 128:
      return &launch_w4a16_top1_m128_grid;
    case 256:
      return &launch_w4a16_top1_m256_grid;
    default:
      return nullptr;
  }
}

W4A16GridLaunchFn w4a16_prefill_grid_launcher(size_t capacity_rows) {
  switch (capacity_rows) {
    case 2:
      return &launch_w4a16_prefill_m2_topk8_grid;
    case 4:
      return &launch_w4a16_prefill_m4_topk8_grid;
    case 8:
      return &launch_w4a16_prefill_m8_topk8_grid;
    case 16:
      return &launch_w4a16_prefill_m16_topk8_grid;
    case 32:
      return &launch_w4a16_prefill_m32_topk8_grid;
    case 64:
      return &launch_w4a16_prefill_m64_topk8_grid;
    case 128:
      return &launch_w4a16_prefill_m128_topk8_grid;
    case 256:
      return &launch_w4a16_prefill_m256_topk8_grid;
    case 512:
      return &launch_w4a16_prefill_m512_topk8_grid;
    case 1024:
      return &launch_w4a16_prefill_m1024_topk8_grid;
    case 2048:
      return &launch_w4a16_prefill_m2048_topk8_grid;
    default:
      return nullptr;
  }
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_b12x_swizzle_scale_async(
    glmrt_device_buffer_t input, glmrt_device_buffer_t output, size_t rows, size_t scale_cols,
    void* cuda_stream) {
  if (rows == 0 || rows % 128 != 0 || scale_cols == 0 || scale_cols % 4 != 0 ||
      rows > std::numeric_limits<size_t>::max() / scale_cols) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t values = rows * scale_cols;
  if (!buffer_has_bytes(input, values) || !buffer_has_bytes(output, values)) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((values + threads - 1) / threads);
  swizzle_modelopt_scale_kernel<<<blocks, threads, 0,
                                  reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(input.ptr), static_cast<uint8_t*>(output.ptr), rows,
      scale_cols);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_aot_available(int* out_available) {
  if (out_available == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_available = 1;
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_aot_init(void) {
  std::call_once(b12x_module_init_once, initialize_b12x_modules);
  return b12x_module_init_status;
}

extern "C" glmrt_status_t
glmrt_cuda_sparkinfer_source_w4a16_aot_available(int* out_available) {
  if (out_available == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
  *out_available = 1;
#else
  *out_available = 0;
#endif
  return GLMRT_STATUS_OK;
}

#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
static glmrt_status_t validate_sparkinfer_source_w4a16_buffers(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    size_t capacity_rows, bool direct) {
  constexpr size_t w13_weight_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 2;
  constexpr size_t w13_scale_bytes =
      kB12xExperts * 2 * kB12xTp4Intermediate * kB12xHidden / 16;
  constexpr size_t w2_weight_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 2;
  constexpr size_t w2_scale_bytes =
      kB12xExperts * kB12xOutput * kB12xTp4Intermediate / 16;
  constexpr size_t expert_scalars_bytes = kB12xExperts * sizeof(float);
  if (buffers == nullptr || capacity_rows == 0 ||
      capacity_rows > kB12xW4a16MaxRows) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t routed_rows = capacity_rows * kB12xTopK;
  bool valid =
      buffer_has_bytes(
          buffers->input,
          capacity_rows * kB12xHidden * sizeof(uint16_t)) &&
      buffer_has_bytes(buffers->w13_weight, w13_weight_bytes) &&
      buffer_has_bytes(buffers->w2_weight, w2_weight_bytes) &&
      buffer_has_bytes(buffers->w13_scale, w13_scale_bytes) &&
      buffer_has_bytes(buffers->w2_scale, w2_scale_bytes) &&
      buffer_has_bytes(buffers->w13_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(buffers->w2_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(
          buffers->micro_w13_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(
          buffers->micro_w2_global_scale, expert_scalars_bytes) &&
      buffer_has_bytes(
          buffers->topk_weights,
          capacity_rows * kB12xTopK * sizeof(float));
  if (direct) {
    valid =
        valid &&
        buffer_has_bytes(
            buffers->activated,
            capacity_rows * 2048 * sizeof(uint32_t)) &&
        buffer_has_bytes(
            buffers->output,
            capacity_rows * kB12xHidden * sizeof(uint16_t)) &&
        buffer_has_bytes(buffers->barrier_count, sizeof(int32_t)) &&
        buffer_has_bytes(buffers->barrier_epoch, sizeof(int32_t));
  } else {
    valid =
        valid &&
        buffer_has_bytes(
            buffers->fc1_output,
            routed_rows * 2 * kB12xTp4Intermediate * sizeof(uint16_t)) &&
        buffer_has_bytes(
            buffers->activated,
            routed_rows * kB12xTp4Intermediate * sizeof(uint16_t)) &&
        buffer_has_bytes(
            buffers->output,
            routed_rows * kB12xHidden * sizeof(uint16_t)) &&
        buffer_has_bytes(
            buffers->packed_route_indices,
            GLMRT_SPARKINFER_SOURCE_W4A16_MAX_ROUTE_SLOTS *
                sizeof(int32_t)) &&
        buffer_has_bytes(
            buffers->block_expert_ids,
            GLMRT_SPARKINFER_SOURCE_W4A16_MAX_ROUTE_BLOCKS *
                sizeof(int32_t)) &&
        buffer_has_bytes(buffers->packed_route_count, sizeof(int32_t)) &&
        buffer_has_bytes(
            buffers->fc1_scratch,
            GLMRT_SPARKINFER_SOURCE_W4A16_MAX_SCRATCH_ELEMENTS *
                sizeof(float)) &&
        buffer_has_bytes(
            buffers->fc2_scratch,
            GLMRT_SPARKINFER_SOURCE_W4A16_MAX_SCRATCH_ELEMENTS *
                sizeof(float)) &&
        buffer_has_bytes(
            buffers->locks,
            GLMRT_SPARKINFER_SOURCE_W4A16_LOCK_ELEMENTS *
                sizeof(int32_t));
  }
  return valid ? GLMRT_STATUS_OK : GLMRT_STATUS_BUFFER_TOO_SMALL;
}

static glmrt_status_t launch_sparkinfer_source_w4a16_topk8_nvfp4(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, size_t rows,
    glmrt_device_buffer_t output_fp8, size_t output_fp8_row_stride_bytes,
    bool fuse_fp8_response, void* cuda_stream) {
  constexpr size_t input_payload_bytes =
      kB12xHidden / 2 + kB12xHidden / 16;
  const bool direct =
      rows <= sparkinfer_source_w4a16_direct_max_rows();
  if (rows == 0 || rows > kB12xW4a16MaxRows ||
      input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(
          input_payload, rows * input_payload_stride_bytes) ||
      (direct &&
       !buffer_has_bytes(
           topk_ids, rows * kB12xTopK * sizeof(int32_t))) ||
      (fuse_fp8_response &&
       (output_fp8_row_stride_bytes < kB12xHidden + sizeof(float) ||
        !buffer_has_bytes(
            output_fp8, rows * output_fp8_row_stride_bytes)))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  size_t capacity_rows = direct ? rows : 16;
  while (capacity_rows < rows) {
    capacity_rows *= 2;
  }
  const glmrt_status_t valid =
      validate_sparkinfer_source_w4a16_buffers(
          buffers, capacity_rows, direct);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  std::call_once(
      sparkinfer_source_w4a16_module_init_once,
      initialize_sparkinfer_source_w4a16_modules);
  if (sparkinfer_source_w4a16_module_init_status != GLMRT_STATUS_OK) {
    return sparkinfer_source_w4a16_module_init_status;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<
      static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  int launch_status = -1;
  if (direct) {
    status = status_from_cuda(cudaMemsetAsync(
        buffers->barrier_count.ptr, 0, sizeof(int32_t), stream));
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
    status = status_from_cuda(cudaMemsetAsync(
        buffers->barrier_epoch.ptr, 0, sizeof(int32_t), stream));
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
    const SparkInferSourceW4A16DirectLaunchFn launcher =
        sparkinfer_source_w4a16_direct_launcher(rows);
    if (launcher == nullptr) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    launch_status = launcher(buffers, topk_ids, rows, stream);
  } else {
    status = status_from_cuda(cudaMemsetAsync(
        buffers->locks.ptr, 0,
        GLMRT_SPARKINFER_SOURCE_W4A16_LOCK_ELEMENTS * sizeof(int32_t),
        stream));
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
    const SparkInferSourceW4A16PrefillLaunchFn launcher =
        sparkinfer_source_w4a16_prefill_launcher(capacity_rows);
    if (launcher == nullptr) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    launch_status = launcher(buffers, rows, stream);
  }
  status = check_aot_launch(
      launch_status,
      direct
          ? "SparkInfer source W4A16 direct top-k=8 launch failed"
          : "SparkInfer source W4A16 prefill top-k=8 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  if (direct) {
    if (fuse_fp8_response) {
      return glmrt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
          static_cast<const uint16_t*>(buffers->output.ptr),
          static_cast<uint8_t*>(output_fp8.ptr), rows, kB12xHidden,
          output_fp8_row_stride_bytes, cuda_stream);
    }
    return status_from_cuda(cudaMemcpyAsync(
        buffers->input.ptr, buffers->output.ptr,
        rows * kB12xHidden * sizeof(uint16_t),
        cudaMemcpyDeviceToDevice, stream));
  }
  if (fuse_fp8_response) {
    sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel<<<
        static_cast<unsigned int>(rows), threads, 0, stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint8_t*>(output_fp8.ptr), rows,
        output_fp8_row_stride_bytes);
  } else {
    sum_w4a16_topk_bf16_kernel<<<
        static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  }
  return status_from_cuda(cudaGetLastError());
}
#endif

extern "C" glmrt_status_t
glmrt_cuda_sparkinfer_source_w4a16_topk8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, size_t rows, void* cuda_stream) {
#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
  return launch_sparkinfer_source_w4a16_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, topk_ids, rows,
      glmrt_device_buffer_t{}, 0, false, cuda_stream);
#else
  (void)buffers;
  (void)input_payload;
  (void)input_payload_stride_bytes;
  (void)topk_ids;
  (void)rows;
  (void)cuda_stream;
  glmrt_set_last_error_message(
      "SparkInfer source W4A16 AOT support is unavailable");
  return GLMRT_STATUS_CUDA_UNAVAILABLE;
#endif
}

extern "C" glmrt_status_t
glmrt_cuda_sparkinfer_source_w4a16_topk8_nvfp4_fp8_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, size_t rows,
    glmrt_device_buffer_t output_fp8, size_t output_fp8_row_stride_bytes,
    void* cuda_stream) {
#if GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
  return launch_sparkinfer_source_w4a16_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, topk_ids, rows,
      output_fp8, output_fp8_row_stride_bytes, true, cuda_stream);
#else
  (void)buffers;
  (void)input_payload;
  (void)input_payload_stride_bytes;
  (void)topk_ids;
  (void)rows;
  (void)output_fp8;
  (void)output_fp8_row_stride_bytes;
  (void)cuda_stream;
  glmrt_set_last_error_message(
      "SparkInfer source W4A16 AOT support is unavailable");
  return GLMRT_STATUS_CUDA_UNAVAILABLE;
#endif
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_mixed_w4a4_candidate_requirements(
    size_t rows, size_t* block_size, size_t* route_slots,
    size_t* route_blocks, size_t* scratch_elements, size_t* grid_x) {
  if (block_size == nullptr || route_slots == nullptr ||
      route_blocks == nullptr || scratch_elements == nullptr ||
      grid_x == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  MixedW4a4Config config{};
  if (!mixed_w4a4_config(rows, &config)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *block_size = config.block_size;
  *route_slots = config.route_slots;
  *route_blocks = config.route_blocks;
  *scratch_elements = config.scratch_elements;
  *grid_x = config.grid_x;
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_mixed_w4a4_grid_candidate_async(
    const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers, size_t rows,
    size_t grid_x, void* cuda_stream) {
  MixedW4a4Config config{};
  if (!mixed_w4a4_config(rows, &config)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (grid_x == 0) {
    grid_x = config.grid_x;
  }
  if (grid_x > 96) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_mixed_w4a4_buffers(buffers, rows, config);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  std::call_once(mixed_w4a4_module_init_once, initialize_mixed_w4a4_modules);
  if (mixed_w4a4_module_init_status != GLMRT_STATUS_OK) {
    return mixed_w4a4_module_init_status;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(
      buffers->locks.ptr, 0, kMixedW4a4LockElements * sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  size_t route_init_values = config.route_slots;
  if (route_init_values < config.route_blocks) {
    route_init_values = config.route_blocks;
  }
  if (route_init_values < rows) {
    route_init_values = rows;
  }
  constexpr size_t threads = 256;
  const size_t route_init_blocks =
      (route_init_values + threads - 1) / threads;
  initialize_mixed_w4a4_routes_kernel<<<route_init_blocks, threads, 0, stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), rows, config.route_slots,
      config.route_blocks, config.block_size);
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = check_aot_launch(
      launch_mixed_w4a4_fc1(buffers, rows, stream),
      "B12X Spark mixed W4A4 FC1 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  const size_t activation_values = rows * 2 * kB12xTp4Intermediate;
  const size_t blocks = (activation_values + threads - 1) / threads;
  reorder_w13_fc1_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->fc1_output.ptr),
      static_cast<uint16_t*>(buffers->fc1_reordered.ptr), rows);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = check_aot_launch(
      launch_mixed_w4a16_activation(buffers, rows, stream),
      "B12X Spark mixed W4A4 activation launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      launch_mixed_w4a16_fc2(buffers, rows, grid_x, stream),
      "B12X Spark mixed W4A4 FC2 launch failed");
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_mixed_w4a4_candidate_async(
    const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers, size_t rows,
    void* cuda_stream) {
  return glmrt_cuda_b12x_spark_mixed_w4a4_grid_candidate_async(
      buffers, rows, 0, cuda_stream);
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_mixed_w4a4_m8_candidate_async(
    const glmrt_b12x_spark_mixed_w4a4_buffers_t* buffers,
    void* cuda_stream) {
  return glmrt_cuda_b12x_spark_mixed_w4a4_candidate_async(
      buffers, 8, cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_b12x_prepare_nvfp4_row_payload_async(
    glmrt_device_buffer_t payload, size_t source_rows, size_t source_row_stride_bytes,
    glmrt_device_buffer_t row_indices, glmrt_device_buffer_t input_packed,
    glmrt_device_buffer_t input_scale, size_t rows, size_t hidden_dim, void* cuda_stream) {
  if (source_rows == 0 || rows == 0 || rows > source_rows || rows > kB12xMaxRows ||
      hidden_dim == 0 || hidden_dim % 16 != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t packed_row_bytes = hidden_dim / 2;
  const size_t scale_cols = hidden_dim / 16;
  const size_t logical_row_bytes = packed_row_bytes + scale_cols;
  if (source_row_stride_bytes < logical_row_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t scale_bytes = align_up_size(rows, 128) * align_up_size(scale_cols, 4);
  if (!buffer_has_bytes(payload, source_rows * source_row_stride_bytes) ||
      !buffer_has_bytes(row_indices, rows * sizeof(uint32_t)) ||
      !buffer_has_bytes(input_packed, rows * packed_row_bytes) ||
      !buffer_has_bytes(input_scale, scale_bytes)) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(input_scale.ptr, 0, scale_bytes, stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  constexpr size_t threads = 256;
  const dim3 grid(static_cast<unsigned int>((packed_row_bytes + threads - 1) / threads),
                  static_cast<unsigned int>(rows));
  prepare_nvfp4_row_payload_b12x_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const uint8_t*>(payload.ptr), source_rows, source_row_stride_bytes,
      static_cast<const uint32_t*>(row_indices.ptr), static_cast<uint8_t*>(input_packed.ptr),
      static_cast<uint8_t*>(input_scale.ptr), rows, hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_mlp_async(
    const glmrt_b12x_spark_mlp_buffers_t* buffers, size_t rows, size_t hidden_dim,
    size_t intermediate_dim, size_t output_dim, float gate_scale_2, float up_scale_2,
    float down_scale_2, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_b12x_buffers(buffers, rows, hidden_dim, intermediate_dim, output_dim, true);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const size_t input_scale_bytes = align_up_size(rows, 128) * align_up_size(hidden_dim / 16, 4);
  cudaError_t error = cudaMemsetAsync(buffers->input_scale.ptr, 0, input_scale_bytes, stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  const dim3 input_grid(static_cast<unsigned int>(hidden_dim / 16),
                        static_cast<unsigned int>(rows));
  quantize_bf16_nvfp4_b12x_kernel<<<input_grid, kQuantThreads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->input.ptr),
      static_cast<uint8_t*>(buffers->input_packed.ptr),
      static_cast<uint8_t*>(buffers->input_scale.ptr), rows, hidden_dim);
  glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return launch_b12x_mlp_from_quantized(buffers, rows, intermediate_dim, gate_scale_2, up_scale_2,
                                        down_scale_2, stream);
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_mlp_prequantized_async(
    const glmrt_b12x_spark_mlp_buffers_t* buffers, size_t rows, size_t hidden_dim,
    size_t intermediate_dim, size_t output_dim, float gate_scale_2, float up_scale_2,
    float down_scale_2, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_b12x_buffers(buffers, rows, hidden_dim, intermediate_dim, output_dim, false);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  return launch_b12x_mlp_from_quantized(
      buffers, rows, intermediate_dim, gate_scale_2, up_scale_2, down_scale_2,
      reinterpret_cast<cudaStream_t>(cuda_stream));
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_moe_tp4_m1_nvfp4_async(
    const glmrt_b12x_spark_moe_tp4_m1_buffers_t* buffers,
    size_t input_payload_stride_bytes, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_b12x_moe_tp4_m1_buffers(buffers, input_payload_stride_bytes);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr int threads = 256;
  constexpr int blocks = static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(buffers->input_payload.ptr),
      static_cast<uint16_t*>(buffers->input_bf16.ptr), kB12xHidden);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  error = cudaMemsetAsync(buffers->barrier_count.ptr, 0, sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  error = cudaMemsetAsync(buffers->barrier_epoch.ptr, 0, sizeof(int32_t), stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  glmrt_b12x_moe_tp4_m1_Tensor_barrier_count_t barrier_count{
      buffers->barrier_count.ptr};
  glmrt_b12x_moe_tp4_m1_Tensor_barrier_epoch_t barrier_epoch{
      buffers->barrier_epoch.ptr};
  return check_aot_launch(
      cute_dsl_glmrt_b12x_moe_tp4_m1_wrapper(
          &moe_tp4_m1_module, buffers->input_bf16.ptr, buffers->w13_weight.ptr,
          buffers->w13_scale.ptr, buffers->w1_alphas.ptr, buffers->a1_gscale.ptr,
          buffers->a2_gscale.ptr, buffers->intermediate.ptr, buffers->w2_weight.ptr,
          buffers->w2_scale.ptr, buffers->w2_alphas.ptr, buffers->topk_ids.ptr,
          buffers->topk_weights.ptr, buffers->output.ptr, &barrier_count, &barrier_epoch, 1,
          GLMRT_B12X_MOE_TP4_M1_GRID_X, stream),
      "B12X Spark grouped TP4 M1 MoE AOT launch failed");
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a4_prefill_topk8_bf16_async(
    const glmrt_b12x_spark_w4a4_moe_buffers_t* buffers, size_t rows,
    void* cuda_stream) {
  const glmrt_status_t valid = validate_b12x_w4a4_prefill_buffers(buffers, rows);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t error = cudaMemsetAsync(
      static_cast<uint8_t*>(buffers->scratch.ptr) +
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_BARRIER_COUNT_OFFSET,
      0, GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_BARRIER_COUNT_BYTES, stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  error = cudaMemsetAsync(
      static_cast<uint8_t*>(buffers->scratch.ptr) +
          GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_BARRIER_EPOCH_OFFSET,
      0, GLMRT_B12X_W4A4_PREFILL_M256_TOPK8_BARRIER_EPOCH_BYTES, stream);
  if (error != cudaSuccess) {
    return status_from_cuda(error);
  }
  return check_aot_launch(
      launch_w4a4_prefill_m256_topk8(buffers, rows, stream),
      "B12X Spark dynamic W4A4 prefill top-k=8 launch failed");
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a4_prefill_topk8_nvfp4_async(
    const glmrt_b12x_spark_w4a4_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  const glmrt_status_t valid = validate_b12x_w4a4_prefill_buffers(buffers, rows);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes)) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                                stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr), input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  const glmrt_status_t status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return glmrt_cuda_b12x_spark_w4a4_prefill_topk8_bf16_async(buffers, rows,
                                                              cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_b12x_w4a16_pack_weight_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, void* cuda_stream) {
  if (size_k == 0 || size_n == 0 || size_k % 16 != 0 || size_n % 64 != 0 ||
      row_rotation >= size_n || size_n > std::numeric_limits<size_t>::max() / size_k) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t bytes = size_n * size_k / 2;
  if (!buffer_has_bytes(source, bytes) || !buffer_has_bytes(destination, bytes)) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t words = bytes / sizeof(uint32_t);
  constexpr size_t threads = 256;
  const size_t blocks = (words + threads - 1) / threads;
  pack_w4a16_weight_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                               reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint32_t*>(destination.ptr),
      size_k, size_n, row_rotation);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_w4a16_pack_scale_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, float scale_factor, void* cuda_stream) {
  if (size_k == 0 || size_n == 0 || size_k % 16 != 0 || size_n % 64 != 0 ||
      row_rotation >= size_n || !isfinite(scale_factor) || scale_factor <= 0.0f ||
      size_n > std::numeric_limits<size_t>::max() / (size_k / 16)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t bytes = size_n * (size_k / 16);
  if (!buffer_has_bytes(source, bytes) || !buffer_has_bytes(destination, bytes)) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (bytes + threads - 1) / threads;
  pack_w4a16_scale_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                              reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(source.ptr), static_cast<uint8_t*>(destination.ptr),
      size_k, size_n, row_rotation, scale_factor);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_gather_nvfp4_rows_bf16_async(
    glmrt_device_buffer_t payload, size_t source_rows, size_t source_row_stride_bytes,
    glmrt_device_buffer_t row_indices, glmrt_device_buffer_t output, size_t rows,
    size_t hidden_dim, void* cuda_stream) {
  const size_t logical_row_bytes = hidden_dim / 2 + hidden_dim / 16;
  if (source_rows == 0 || rows == 0 || rows > source_rows || hidden_dim == 0 ||
      hidden_dim % 16 != 0 || source_row_stride_bytes < logical_row_bytes) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (!buffer_has_bytes(payload, source_rows * source_row_stride_bytes) ||
      !buffer_has_bytes(row_indices, rows * sizeof(uint32_t)) ||
      !buffer_has_bytes(output, rows * hidden_dim * sizeof(uint16_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const size_t values = rows * hidden_dim;
  constexpr size_t threads = 256;
  const size_t blocks = (values + threads - 1) / threads;
  gather_nvfp4_rows_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                    reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
      static_cast<const uint8_t*>(payload.ptr), source_rows, source_row_stride_bytes,
      static_cast<const uint32_t*>(row_indices.ptr), static_cast<uint16_t*>(output.ptr), rows,
      hidden_dim);
  return status_from_cuda(cudaGetLastError());
}

glmrt_status_t launch_w4a16_decode_m1_nvfp4_grid(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, int grid_x, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  if (grid_x < 0 || grid_x > kB12xW4a16DecodeResidentGridX) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_w4a16_moe_buffers(buffers, 1, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, kB12xTopK * sizeof(int32_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr int threads = 256;
  constexpr int blocks = static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  glmrt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  const int launch_status =
      grid_x == 0
          ? launch_w4a16_decode_m1(&launch_buffers, 1, stream)
          : launch_w4a16_decode_m1_grid(&launch_buffers, 1, grid_x, stream);
  status = check_aot_launch(
      launch_status, "B12X Spark packed W4A16 decode M1 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->output.ptr), 1, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream) {
  return launch_w4a16_decode_m1_nvfp4_grid(
      buffers, input_payload, input_payload_stride_bytes, topk_ids,
      0, cuda_stream);
}

glmrt_status_t launch_w4a16_decode_m1_fused_sum_nvfp4(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  const glmrt_status_t valid =
      validate_w4a16_moe_buffers(buffers, 1, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, kB12xTopK * sizeof(int32_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr int threads = 256;
  constexpr int blocks =
      static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  glmrt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  return check_aot_launch(
      launch_w4a16_decode_m1_fused_sum(&launch_buffers, 1, stream),
      "B12X Spark packed W4A16 decode M1 fused-sum launch failed");
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream) {
  return launch_w4a16_decode_m1_fused_sum_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, topk_ids,
      cuda_stream);
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, rows * kB12xTopK * sizeof(int32_t))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr), input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  glmrt_b12x_spark_w4a16_moe_buffers_t launch_buffers = *buffers;
  launch_buffers.packed_route_indices = topk_ids;
  status = check_aot_launch(
      launcher(&launch_buffers, rows, stream),
      "B12X Spark ordered direct-top-k W4A16 M=2..8 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_grouped_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  status = check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark grouped block-8 W4A16 M=2..8 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  W4A16LaunchFn launcher = w4a16_m1_parity_grouped_wide_launcher(rows);
  if (launcher == nullptr || input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_w4a16_moe_buffers(buffers, rows, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks),
                                                threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }

  status = check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark grouped-wide W4A16 M=2..8 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, int grid_x, void* cuda_stream) {
  if (grid_x <= 0 || grid_x > kB12xW4a16DecodeResidentGridX) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return launch_w4a16_decode_m1_nvfp4_grid(
      buffers, input_payload, input_payload_stride_bytes, topk_ids, grid_x,
      cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a16_modelopt_decode_m1_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream) {
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  const glmrt_status_t valid = validate_w4a16_moe_buffers(buffers, 1, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, input_payload_stride_bytes) ||
      !buffer_has_bytes(topk_ids, kB12xTopK * sizeof(int32_t)) ||
      !buffer_has_bytes(buffers->output,
                        kB12xTopK * kB12xHidden * sizeof(uint16_t))) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr int threads = 256;
  constexpr int blocks = static_cast<int>((kB12xHidden + threads - 1) / threads);
  dequantize_nvfp4_row_payload_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr),
      static_cast<uint16_t*>(buffers->input.ptr), kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  initialize_w4a16_modelopt_decode_routes_kernel<<<1,
      GLMRT_B12X_W4A16_MODELOPT_DECODE_M1_PACKED_ROUTE_SLOTS, 0, stream>>>(
      static_cast<const int32_t*>(topk_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr));
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  status = check_aot_launch(
      launch_w4a16_modelopt_decode_m1(buffers, 1, stream),
      "B12X Spark ModelOpt W4A16 decode M1 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  sum_w4a16_topk_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint16_t*>(buffers->output.ptr),
      static_cast<uint16_t*>(buffers->output.ptr), 1, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

static glmrt_status_t launch_w4a16_prefill_topk8_nvfp4(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, glmrt_device_buffer_t output_fp8,
    size_t output_fp8_row_stride_bytes, bool fuse_fp8_response,
    int grid_x, void* cuda_stream) {
  size_t capacity_rows = 2;
  while (capacity_rows < rows && capacity_rows < kB12xW4a16MaxRows) {
    capacity_rows *= 2;
  }
  constexpr size_t input_payload_bytes = kB12xHidden / 2 + kB12xHidden / 16;
  if (rows == 0 || rows > capacity_rows ||
      input_payload_stride_bytes < input_payload_bytes ||
      !buffer_has_bytes(input_payload, rows * input_payload_stride_bytes) ||
      (fuse_fp8_response &&
       (output_fp8_row_stride_bytes < kB12xHidden + sizeof(float) ||
        !buffer_has_bytes(output_fp8,
                          rows * output_fp8_row_stride_bytes)))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 1 && w4a16_m1_fused_sum_enabled()) {
    const glmrt_status_t fused_status =
        launch_w4a16_decode_m1_fused_sum_nvfp4(
            buffers, input_payload, input_payload_stride_bytes,
            buffers->packed_route_indices, cuda_stream);
    if (fused_status != GLMRT_STATUS_OK) {
      return fused_status;
    }
    if (fuse_fp8_response) {
      return glmrt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
          static_cast<const uint16_t*>(buffers->output.ptr),
          static_cast<uint8_t*>(output_fp8.ptr), 1, kB12xHidden,
          output_fp8_row_stride_bytes, cuda_stream);
    }
    return status_from_cuda(cudaMemcpyAsync(
        buffers->input.ptr, buffers->output.ptr,
        kB12xHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice,
        reinterpret_cast<cudaStream_t>(cuda_stream)));
  }
  const glmrt_status_t valid =
      validate_w4a16_moe_buffers(buffers, capacity_rows, kB12xTopK);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t values = rows * kB12xHidden;
  const size_t blocks = (values + threads - 1) / threads;
  dequantize_nvfp4_row_payloads_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                                stream>>>(
      static_cast<const uint8_t*>(input_payload.ptr), input_payload_stride_bytes,
      static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  W4A16LaunchFn launcher = &launch_w4a16_prefill_m2048_topk8;
  if (capacity_rows == 2) {
    launcher = &launch_w4a16_prefill_m2_topk8;
  } else if (capacity_rows == 4) {
    launcher = &launch_w4a16_prefill_m4_topk8;
  } else if (capacity_rows == 8) {
    launcher = &launch_w4a16_prefill_m8_topk8;
  } else if (capacity_rows == 16) {
    launcher = &launch_w4a16_prefill_m16_topk8;
  } else if (capacity_rows == 32) {
    launcher = &launch_w4a16_prefill_m32_topk8;
  } else if (capacity_rows == 64) {
    launcher = &launch_w4a16_prefill_m64_topk8;
  } else if (capacity_rows == 128) {
    launcher = &launch_w4a16_prefill_m128_topk8;
  } else if (capacity_rows == 256) {
    launcher = &launch_w4a16_prefill_m256_topk8;
  } else if (capacity_rows == 512) {
    launcher = &launch_w4a16_prefill_m512_topk8;
  } else if (capacity_rows == 1024) {
    launcher = &launch_w4a16_prefill_m1024_topk8;
  }
  const W4A16GridLaunchFn grid_launcher =
      grid_x > 0 ? w4a16_prefill_grid_launcher(capacity_rows) : nullptr;
  if (grid_x > 0 && grid_launcher == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int launch_status = grid_launcher != nullptr
                                ? grid_launcher(buffers, rows, grid_x, stream)
                                : launcher(buffers, rows, stream);
  status = check_aot_launch(
      launch_status, "B12X Spark packed W4A16 prefill top-k=8 launch failed");
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  if (fuse_fp8_response) {
    sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel<<<
        static_cast<unsigned int>(rows), threads, 0, stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint8_t*>(output_fp8.ptr), rows,
        output_fp8_row_stride_bytes);
  } else {
    sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                  stream>>>(
        static_cast<const uint16_t*>(buffers->output.ptr),
        static_cast<uint16_t*>(buffers->input.ptr), rows, kB12xHidden);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream) {
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows,
      glmrt_device_buffer_t{}, 0, false, 0, cuda_stream);
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, int grid_x, void* cuda_stream) {
  if (grid_x <= 0 || grid_x > kB12xW4a16DecodeMaxGridX) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows,
      glmrt_device_buffer_t{}, 0, false, grid_x, cuda_stream);
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, glmrt_device_buffer_t output_fp8,
    size_t output_fp8_row_stride_bytes, void* cuda_stream) {
  return launch_w4a16_prefill_topk8_nvfp4(
      buffers, input_payload, input_payload_stride_bytes, rows, output_fp8,
      output_fp8_row_stride_bytes, true, 0, cuda_stream);
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_sum_topk8_bf16_async(
    glmrt_device_buffer_t routed_bf16, glmrt_device_buffer_t output_bf16,
    size_t rows, void* cuda_stream) {
  const size_t routed_values = rows * kB12xTopK * kB12xHidden;
  const size_t output_values = rows * kB12xHidden;
  if (rows == 0 || !buffer_has_bytes(routed_bf16, routed_values * sizeof(uint16_t)) ||
      !buffer_has_bytes(output_bf16, output_values * sizeof(uint16_t))) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr int threads = 256;
  const size_t blocks = (output_values + threads - 1) / threads;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sum_w4a16_topk_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                stream>>>(
      static_cast<const uint16_t*>(routed_bf16.ptr),
      static_cast<uint16_t*>(output_bf16.ptr), rows, kB12xHidden);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_b12x_spark_sum_topk8_bf16_to_fp8_async(
    glmrt_device_buffer_t routed_bf16, glmrt_device_buffer_t output_fp8,
    size_t rows, size_t output_row_stride_bytes, void* cuda_stream) {
  const size_t routed_values = rows * kB12xTopK * kB12xHidden;
  const size_t minimum_output_row_bytes = kB12xHidden + sizeof(float);
  if (rows == 0 || output_row_stride_bytes < minimum_output_row_bytes ||
      !buffer_has_bytes(routed_bf16, routed_values * sizeof(uint16_t)) ||
      !buffer_has_bytes(output_fp8, rows * output_row_stride_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr int threads = 256;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sum_w4a16_topk_bf16_to_fp8_row_scaled_kernel<<<
      static_cast<unsigned int>(rows), threads, 0, stream>>>(
      static_cast<const uint16_t*>(routed_bf16.ptr),
      static_cast<uint8_t*>(output_fp8.ptr), rows,
      output_row_stride_bytes);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a16_top1_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, void* cuda_stream) {
  if (rows == 0 || rows > capacity_rows || expert_id >= kB12xExperts) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const W4A16LaunchFn launcher = w4a16_top1_launcher(capacity_rows);
  if (launcher == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_w4a16_moe_buffers(buffers, capacity_rows, 1);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t init_values = capacity_rows;
  const size_t blocks = (init_values + threads - 1) / threads;
  initialize_w4a16_top1_routes_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                          stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), rows, capacity_rows, expert_id,
      capacity_rows <= 8);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      launcher(buffers, rows, stream),
      "B12X Spark packed W4A16 top-k=1 launch failed");
}

extern "C" glmrt_status_t glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, int grid_x, void* cuda_stream) {
  if (rows == 0 || rows > capacity_rows || expert_id >= kB12xExperts || grid_x <= 0 ||
      grid_x > kB12xW4a16DecodeMaxGridX) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const W4A16GridLaunchFn launcher = w4a16_top1_grid_launcher(capacity_rows);
  if (launcher == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid = validate_w4a16_moe_buffers(buffers, capacity_rows, 1);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  const glmrt_status_t initialized = glmrt_cuda_b12x_spark_aot_init();
  if (initialized != GLMRT_STATUS_OK) {
    return initialized;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  glmrt_status_t status = reset_w4a16_locks_async(buffers, stream);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  constexpr size_t threads = 256;
  const size_t blocks = (capacity_rows + threads - 1) / threads;
  initialize_w4a16_top1_routes_kernel<<<static_cast<unsigned int>(blocks), threads, 0,
                                          stream>>>(
      static_cast<int32_t*>(buffers->packed_route_indices.ptr),
      static_cast<int32_t*>(buffers->block_expert_ids.ptr),
      static_cast<int32_t*>(buffers->packed_route_count.ptr),
      static_cast<float*>(buffers->topk_weights.ptr), rows, capacity_rows, expert_id,
      capacity_rows <= 8);
  status = status_from_cuda(cudaGetLastError());
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return check_aot_launch(
      launcher(buffers, rows, grid_x, stream),
      "B12X Spark packed W4A16 top-k=1 grid candidate launch failed");
}
