cmake_minimum_required(VERSION 3.24)
project(glmrt_native LANGUAGES C CXX)

option(GLMRT_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(GLMRT_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(GLMRT_ENABLE_B12X_AOT "Build Spark B12X AOT dense kernels" OFF)
option(GLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT
  "Build latest SparkInfer source-layout W4A16 MoE kernels" OFF)
option(GLMRT_ENABLE_B12X_COORDINATOR_AOT "Build coordinator B12X AOT attention kernels" OFF)
option(GLMRT_ENABLE_W8A16_AOT "Build coordinator row-major W8A16 Triton AOT kernels" OFF)
option(GLMRT_ENABLE_FLASHINFER_PACKED_MLA
  "Build the exact grouped-split SM120 packed-FP8 MLA kernel" OFF)
option(GLMRT_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
set(GLMRT_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(GLMRT_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../.glmrt-cache/external/sparkinfer"
  CACHE PATH "SparkInfer source checkout used to export source-layout W4A16 kernels")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GLMRT_NATIVE_SOURCES src/glmrt_native.cc)
if(GLMRT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/norm.cu
    cuda/kernels/mlp.cu
    cuda/kernels/residual.cu
    cuda/kernels/kv_cache.cu
    cuda/kernels/dsa_indexer.cu
    cuda/kernels/router.cu
    cuda/kernels/linear.cu
    cuda/kernels/attention.cu
    cuda/kernels/embedding.cu
    cuda/kernels/sampling.cu
    cuda/kernels/nibbles.cu
    cuda/kernels/nvfp4_payload.cu
  )
endif()

set(GLMRT_BUILD_FLASHINFER_PACKED_MLA OFF)
if(GLMRT_ENABLE_FLASHINFER_PACKED_MLA OR GLMRT_ENABLE_B12X_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  set(GLMRT_BUILD_FLASHINFER_PACKED_MLA ON)
endif()

if(GLMRT_BUILD_FLASHINFER_PACKED_MLA)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_FLASHINFER_PACKED_MLA=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent / 'data' / 'include')"
    OUTPUT_VARIABLE GLMRT_FLASHINFER_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT EXISTS "${GLMRT_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh")
    message(FATAL_ERROR
      "FlashInfer SM120 sparse-MLA headers were not found under ${GLMRT_FLASHINFER_INCLUDE_DIR}")
  endif()
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/packed_fp8_mla_exact.cu
  )
  set_source_files_properties(cuda/kernels/packed_fp8_mla_exact.cu PROPERTIES
    COMPILE_OPTIONS "-use_fast_math;-O3"
  )
  if(GLMRT_CUDA_ARCHITECTURES STREQUAL "120")
    set(GLMRT_CUDA_ARCHITECTURES "120f")
  endif()
endif()

if(GLMRT_ENABLE_B12X_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_B12X_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_B12X_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/b12x_aot")
  set(GLMRT_B12X_AOT_REGIMES 1 2 4 8 16 32 64 128 256)
  set(GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048)
  set(GLMRT_B12X_AOT_OBJECTS)
  set(GLMRT_B12X_AOT_HEADERS)
  foreach(projection gate down gate_tp4 down_tp4)
    foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
      list(APPEND GLMRT_B12X_AOT_OBJECTS
        "${GLMRT_B12X_AOT_DIR}/${projection}_m${regime}.o"
      )
      list(APPEND GLMRT_B12X_AOT_HEADERS
        "${GLMRT_B12X_AOT_DIR}/${projection}_m${regime}.h"
      )
    endforeach()
  endforeach()
  list(APPEND GLMRT_B12X_AOT_OBJECTS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_m1.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a4_prefill_m256_topk8.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_modelopt_decode_m1.o"
  )
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a4_fc1_m${regime}.o"
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a16_activation_m${regime}.o"
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a16_fc2_m${regime}.o"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.o"
    )
  endforeach()
  list(APPEND GLMRT_B12X_AOT_HEADERS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_m1.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a4_prefill_m256_topk8.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_modelopt_decode_m1.h"
    "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot_config.h"
    "${GLMRT_B12X_AOT_DIR}/b12x_spark_mixed_w4a4_aot_config.h"
  )
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a4_fc1_m${regime}.h"
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a16_activation_m${regime}.h"
      "${GLMRT_B12X_AOT_DIR}/mixed_w4a16_fc2_m${regime}.h"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_mixed_w4a4_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_mixed_w4a4_aot.meta"
      ${GLMRT_B12X_AOT_OBJECTS}
      ${GLMRT_B12X_AOT_HEADERS}
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
      --target-sms 48
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_mixed_w4a4_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_mixed_w4a4_aot.py"
    COMMENT "Exporting Spark B12X AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_aot_export
    DEPENDS
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_mixed_w4a4_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_mixed_w4a4_aot.meta"
      ${GLMRT_B12X_AOT_OBJECTS}
      ${GLMRT_B12X_AOT_HEADERS}
  )
  set_source_files_properties(${GLMRT_B12X_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/b12x_direct.cu
    ${GLMRT_B12X_AOT_OBJECTS}
  )

  set(GLMRT_B12X_M1_PARITY_AOT_REGIMES 2 3 4 5 6 7 8)
  set(GLMRT_B12X_M1_PARITY_AOT_OBJECTS)
  set(GLMRT_B12X_M1_PARITY_AOT_HEADERS
    "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot_config.h"
  )
  foreach(regime IN LISTS GLMRT_B12X_M1_PARITY_AOT_REGIMES)
    list(APPEND GLMRT_B12X_M1_PARITY_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.o"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.o"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.o"
    )
    list(APPEND GLMRT_B12X_M1_PARITY_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.h"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.h"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS}
      ${GLMRT_B12X_M1_PARITY_AOT_HEADERS}
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
      --target-sms 48
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
    COMMENT "Exporting ordered Spark W4A16 M=2..8 parity candidates"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_m1_parity_aot_export
    DEPENDS
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS}
      ${GLMRT_B12X_M1_PARITY_AOT_HEADERS}
  )
  set_source_files_properties(${GLMRT_B12X_M1_PARITY_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS})

  if(GLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT)
    if(NOT EXISTS "${GLMRT_SPARKINFER_SOURCE_DIR}/sparkinfer")
      message(FATAL_ERROR
        "SparkInfer package was not found under GLMRT_SPARKINFER_SOURCE_DIR="
        "${GLMRT_SPARKINFER_SOURCE_DIR}")
    endif()
    file(GLOB_RECURSE GLMRT_SPARKINFER_SOURCE_W4A16_EXPORT_INPUTS
      CONFIGURE_DEPENDS
      "${GLMRT_SPARKINFER_SOURCE_DIR}/sparkinfer/moe/_shared/kernels/*.py"
    )
    set(GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_REGIMES 1 2 3 4 5 6 7 8)
    set(GLMRT_SPARKINFER_SOURCE_W4A16_PREFILL_REGIMES
      16 32 64 128 256 512 1024 2048)
    set(GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS)
    set(GLMRT_SPARKINFER_SOURCE_W4A16_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_aot_config.h"
    )
    foreach(regime IN LISTS GLMRT_SPARKINFER_SOURCE_W4A16_DIRECT_REGIMES)
      list(APPEND GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_direct_m${regime}_topk8.o"
      )
      list(APPEND GLMRT_SPARKINFER_SOURCE_W4A16_AOT_HEADERS
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_direct_m${regime}_topk8.h"
      )
    endforeach()
    foreach(regime IN LISTS GLMRT_SPARKINFER_SOURCE_W4A16_PREFILL_REGIMES)
      list(APPEND GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_prefill_m${regime}_topk8.o"
      )
      list(APPEND GLMRT_SPARKINFER_SOURCE_W4A16_AOT_HEADERS
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_prefill_m${regime}_topk8.h"
      )
    endforeach()
    add_custom_command(
      OUTPUT
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_aot.stamp"
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_aot.meta"
        ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS}
        ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_HEADERS}
      COMMAND "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${GLMRT_SPARKINFER_SOURCE_DIR}"
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_sparkinfer_source_w4a16_aot.py"
        --output-dir "${GLMRT_B12X_AOT_DIR}"
        --target-sms 48
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_sparkinfer_source_w4a16_aot.py"
        ${GLMRT_SPARKINFER_SOURCE_W4A16_EXPORT_INPUTS}
      COMMENT "Exporting latest SparkInfer source-layout W4A16 AOT kernels"
      VERBATIM
    )
    add_custom_target(glmrt_sparkinfer_source_w4a16_aot_export
      DEPENDS
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_aot.stamp"
        "${GLMRT_B12X_AOT_DIR}/sparkinfer_source_w4a16_aot.meta"
        ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS}
        ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_HEADERS}
    )
    set_source_files_properties(
      ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS}
      PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE
    )
    list(APPEND GLMRT_NATIVE_SOURCES
      ${GLMRT_SPARKINFER_SOURCE_W4A16_AOT_OBJECTS}
    )
  endif()
endif()

if(GLMRT_ENABLE_B12X_COORDINATOR_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_B12X_COORDINATOR_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_B12X_COORDINATOR_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/b12x_coordinator_aot"
  )
  set(GLMRT_B12X_COORDINATOR_AOT_OBJECTS
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.o"
  )
  set(GLMRT_B12X_COORDINATOR_AOT_HEADERS
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
      ${GLMRT_B12X_COORDINATOR_AOT_HEADERS}
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      --output-dir "${GLMRT_B12X_COORDINATOR_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
    COMMENT "Exporting coordinator B12X AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_coordinator_aot_export
    DEPENDS
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
      ${GLMRT_B12X_COORDINATOR_AOT_HEADERS}
  )
  set_source_files_properties(${GLMRT_B12X_COORDINATOR_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/b12x_coordinator.cu
    ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
  )
endif()

if(GLMRT_ENABLE_W8A16_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_W8A16_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_W8A16_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/w8a16_aot")
  set(GLMRT_W8A16_AOT_HEADER
    "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.h"
  )
  set(GLMRT_W8A16_AOT_SOURCE
    "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.cc"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/w8a16_packed_o_aot"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_OBJECTS
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.o"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_HEADERS
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${GLMRT_W8A16_AOT_HEADER}"
      "${GLMRT_W8A16_AOT_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      --output-dir "${GLMRT_W8A16_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_triton_prefill.py"
    COMMENT "Exporting coordinator row-major W8A16 AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_w8a16_aot_export
    DEPENDS
      "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${GLMRT_W8A16_AOT_HEADER}"
      "${GLMRT_W8A16_AOT_SOURCE}"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
      ${GLMRT_W8A16_PACKED_O_AOT_HEADERS}
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      --output-dir "${GLMRT_W8A16_PACKED_O_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_cute_packed_prefill.py"
    COMMENT "Exporting coordinator packed W8A16 O-projection AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_w8a16_packed_o_aot_export
    DEPENDS
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
      ${GLMRT_W8A16_PACKED_O_AOT_HEADERS}
  )
  set_source_files_properties("${GLMRT_W8A16_AOT_SOURCE}" PROPERTIES
    GENERATED TRUE
  )
  set_source_files_properties(${GLMRT_W8A16_PACKED_O_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    "${GLMRT_W8A16_AOT_SOURCE}"
    cuda/kernels/w8a16_packed.cu
    ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
  )
endif()

if(GLMRT_ENABLE_B12X_AOT OR GLMRT_ENABLE_B12X_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE GLMRT_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(GLMRT_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${GLMRT_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

add_library(glmrt_native SHARED ${GLMRT_NATIVE_SOURCES})
target_include_directories(glmrt_native PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(GLMRT_ENABLE_CUDA)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_CUDA=1)
  target_link_libraries(glmrt_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(glmrt_native PROPERTIES
    CUDA_ARCHITECTURES "${GLMRT_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_CUDA=0)
endif()
if(GLMRT_BUILD_FLASHINFER_PACKED_MLA)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_FLASHINFER_INCLUDE_DIR}")
endif()
if(GLMRT_ENABLE_B12X_AOT)
  add_dependencies(
    glmrt_native
    glmrt_b12x_aot_export
    glmrt_b12x_m1_parity_aot_export
  )
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_B12X_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_AOT=0)
endif()
if(GLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT)
  if(NOT GLMRT_ENABLE_B12X_AOT)
    message(FATAL_ERROR
      "GLMRT_ENABLE_SPARKINFER_SOURCE_W4A16_AOT=ON requires GLMRT_ENABLE_B12X_AOT=ON")
  endif()
  add_dependencies(glmrt_native glmrt_sparkinfer_source_w4a16_aot_export)
  target_compile_definitions(
    glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT=1
  )
else()
  target_compile_definitions(
    glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_SPARKINFER_SOURCE_W4A16_AOT=0
  )
endif()
if(GLMRT_ENABLE_B12X_COORDINATOR_AOT)
  add_dependencies(glmrt_native glmrt_b12x_coordinator_aot_export)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_B12X_COORDINATOR_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=0)
endif()
if(GLMRT_ENABLE_W8A16_AOT)
  add_dependencies(glmrt_native glmrt_w8a16_aot_export)
  add_dependencies(glmrt_native glmrt_w8a16_packed_o_aot_export)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_W8A16_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_W8A16_AOT_DIR}")
  target_include_directories(glmrt_native PRIVATE "${GLMRT_W8A16_PACKED_O_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_W8A16_AOT=0)
endif()
if(GLMRT_ENABLE_B12X_AOT OR GLMRT_ENABLE_B12X_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  target_link_libraries(glmrt_native PUBLIC "${GLMRT_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET glmrt_native APPEND PROPERTY BUILD_RPATH "${GLMRT_B12X_AOT_RUNTIME_DIR}")
endif()
if(GLMRT_ENABLE_RDMA)
  find_path(GLMRT_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(GLMRT_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT GLMRT_IBVERBS_INCLUDE_DIR OR NOT GLMRT_IBVERBS_LIBRARY)
    message(FATAL_ERROR "GLMRT_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_RDMA=1)
  target_include_directories(glmrt_native PRIVATE ${GLMRT_IBVERBS_INCLUDE_DIR})
  target_link_libraries(glmrt_native PUBLIC ${GLMRT_IBVERBS_LIBRARY})
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_RDMA=0)
endif()
if(GLMRT_ENABLE_NCCL)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_NCCL=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_path(GLMRT_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(GLMRT_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_NCCL=1)
  target_include_directories(glmrt_native PRIVATE ${GLMRT_NCCL_INCLUDE_DIR})
  target_link_libraries(glmrt_native PUBLIC ${GLMRT_NCCL_LIBRARY})
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(glmrt_native_selftest tests/native_selftest.cc)
target_link_libraries(glmrt_native_selftest PRIVATE glmrt_native)

enable_testing()
add_test(NAME glmrt_native_selftest COMMAND glmrt_native_selftest)

if(GLMRT_ENABLE_CUDA)
  add_executable(glmrt_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(glmrt_cuda_selftest PRIVATE glmrt_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME glmrt_cuda_selftest COMMAND glmrt_cuda_selftest)
endif()
