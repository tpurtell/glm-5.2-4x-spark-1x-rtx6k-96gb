#include "glmrt_native.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr float kGlm52RoutedScalingFactor = 2.5f;

void require_status(glmrt_status_t status, const char* action) {
  if (status == GLMRT_STATUS_OK) {
    return;
  }
  char error[256] = {};
  glmrt_last_error(error, sizeof(error));
  std::cerr << action << " failed with status " << status << " error=" << error << "\n";
  std::abort();
}

void require_cuda(cudaError_t status, const char* action) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << action << " failed: " << cudaGetErrorString(status) << "\n";
  std::abort();
}

void assert_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  float tolerance = 1.0e-5f) {
  assert(actual.size() == expected.size());
  for (size_t idx = 0; idx < actual.size(); ++idx) {
    const float diff = std::fabs(actual[idx] - expected[idx]);
    if (diff > tolerance) {
      std::cerr << "mismatch at " << idx << " actual=" << actual[idx]
                << " expected=" << expected[idx] << " diff=" << diff << "\n";
      std::abort();
    }
  }
}

std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
                               int rows, int hidden, float eps) {
  std::vector<float> out(x.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < hidden; ++col) {
      const float value = x[row * hidden + col];
      sum += value * value;
    }
    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(hidden) + eps);
    for (int col = 0; col < hidden; ++col) {
      out[row * hidden + col] = x[row * hidden + col] * inv * weight[col];
    }
  }
  return out;
}

std::vector<float> cpu_mlp(const std::vector<float>& x, const std::vector<float>& gate_weight,
                           const std::vector<float>& up_weight,
                           const std::vector<float>& down_weight, int hidden, int intermediate) {
  std::vector<float> out(hidden, 0.0f);
  for (int out_col = 0; out_col < hidden; ++out_col) {
    float acc = 0.0f;
    for (int mid = 0; mid < intermediate; ++mid) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int col = 0; col < hidden; ++col) {
        gate += x[col] * gate_weight[mid * hidden + col];
        up += x[col] * up_weight[mid * hidden + col];
      }
      const float silu = gate / (1.0f + std::exp(-gate));
      acc += silu * up * down_weight[out_col * intermediate + mid];
    }
    out[out_col] = acc;
  }
  return out;
}

std::vector<float> cpu_mlp_rows(const std::vector<float>& x, const std::vector<float>& gate_weight,
                                const std::vector<float>& up_weight,
                                const std::vector<float>& down_weight, size_t rows,
                                size_t hidden, size_t intermediate) {
  std::vector<float> out(rows * hidden, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    const float* row_x = x.data() + row * hidden;
    for (size_t out_col = 0; out_col < hidden; ++out_col) {
      float acc = 0.0f;
      for (size_t mid = 0; mid < intermediate; ++mid) {
        float gate = 0.0f;
        float up = 0.0f;
        for (size_t col = 0; col < hidden; ++col) {
          gate += row_x[col] * gate_weight[mid * hidden + col];
          up += row_x[col] * up_weight[mid * hidden + col];
        }
        const float silu = gate / (1.0f + std::exp(-gate));
        acc += silu * up * down_weight[out_col * intermediate + mid];
      }
      out[row * hidden + out_col] = acc;
    }
  }
  return out;
}

float nvfp4_code_value(uint8_t code) {
  constexpr float kCodebook[16] = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      0.0f,  -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return kCodebook[code & 0x0f];
}

float f8e4m3_to_f32(uint8_t byte) {
  if (byte == 0 || byte == 0x80) {
    return 0.0f;
  }
  const float sign = (byte & 0x80) == 0 ? 1.0f : -1.0f;
  const int exponent = static_cast<int>((byte >> 3) & 0x0f);
  const float mantissa = static_cast<float>(byte & 0x07);
  const float significand = exponent == 0 ? mantissa / 8.0f : 1.0f + mantissa / 8.0f;
  const int exponent_power = exponent == 0 ? -6 : exponent - 7;
  return sign * std::ldexp(significand, exponent_power);
}

uint16_t f32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

uint16_t f32_to_bf16_rn(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t round_to_even = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + round_to_even) >> 16);
}

float bf16_to_f32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::vector<uint16_t> bf16_values(const std::vector<float>& values) {
  std::vector<uint16_t> out(values.size(), 0);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    out[idx] = f32_to_bf16(values[idx]);
  }
  return out;
}

std::vector<float> bf16_to_f32_values(const std::vector<uint16_t>& values) {
  std::vector<float> out(values.size(), 0.0f);
  for (size_t idx = 0; idx < values.size(); ++idx) {
    out[idx] = bf16_to_f32(values[idx]);
  }
  return out;
}

float packed_nvfp4_value(const std::vector<uint8_t>& packed_row,
                         const std::vector<uint8_t>& scale_row, size_t value_idx,
                         float scale_2) {
  const uint8_t packed = packed_row[value_idx / 2];
  const uint8_t code = value_idx % 2 == 0 ? (packed & 0x0f) : (packed >> 4);
  return nvfp4_code_value(code) * f8e4m3_to_f32(scale_row[value_idx / 16]) * scale_2;
}

float dot_packed_nvfp4(const std::vector<float>& input, const std::vector<uint8_t>& packed_row,
                       const std::vector<uint8_t>& scale_row, float scale_2) {
  float sum = 0.0f;
  for (size_t idx = 0; idx < input.size(); ++idx) {
    sum += input[idx] * packed_nvfp4_value(packed_row, scale_row, idx, scale_2);
  }
  return sum;
}

std::vector<float> cpu_nvfp4_route(const std::vector<float>& hidden,
                                   const std::vector<uint8_t>& gate_weight,
                                   const std::vector<uint8_t>& gate_scale,
                                   const std::vector<uint8_t>& up_weight,
                                   const std::vector<uint8_t>& up_scale,
                                   const std::vector<uint8_t>& down_weight,
                                   const std::vector<uint8_t>& down_scale, size_t hidden_dim,
                                   size_t intermediate, size_t output_dim, float gate_scale_2,
                                   float up_scale_2, float down_scale_2, float route_weight) {
  const size_t packed_hidden_bytes = (hidden_dim + 1) / 2;
  const size_t hidden_scale_bytes = (hidden_dim + 15) / 16;
  const size_t packed_intermediate_bytes = (intermediate + 1) / 2;
  const size_t intermediate_scale_bytes = (intermediate + 15) / 16;
  std::vector<float> activations(intermediate, 0.0f);
  for (size_t mid = 0; mid < intermediate; ++mid) {
    const std::vector<uint8_t> gate_row(gate_weight.begin() + mid * packed_hidden_bytes,
                                        gate_weight.begin() + (mid + 1) * packed_hidden_bytes);
    const std::vector<uint8_t> gate_scale_row(gate_scale.begin() + mid * hidden_scale_bytes,
                                              gate_scale.begin() + (mid + 1) * hidden_scale_bytes);
    const std::vector<uint8_t> up_row(up_weight.begin() + mid * packed_hidden_bytes,
                                      up_weight.begin() + (mid + 1) * packed_hidden_bytes);
    const std::vector<uint8_t> up_scale_row(up_scale.begin() + mid * hidden_scale_bytes,
                                            up_scale.begin() + (mid + 1) * hidden_scale_bytes);
    const float gate = dot_packed_nvfp4(hidden, gate_row, gate_scale_row, gate_scale_2);
    const float up = dot_packed_nvfp4(hidden, up_row, up_scale_row, up_scale_2);
    activations[mid] = gate / (1.0f + std::exp(-gate)) * up;
  }

  std::vector<float> out(output_dim, 0.0f);
  for (size_t out_col = 0; out_col < output_dim; ++out_col) {
    const std::vector<uint8_t> down_row(
        down_weight.begin() + out_col * packed_intermediate_bytes,
        down_weight.begin() + (out_col + 1) * packed_intermediate_bytes);
    const std::vector<uint8_t> down_scale_row(
        down_scale.begin() + out_col * intermediate_scale_bytes,
        down_scale.begin() + (out_col + 1) * intermediate_scale_bytes);
    float acc = 0.0f;
    for (size_t mid = 0; mid < intermediate; ++mid) {
      acc += activations[mid] *
             packed_nvfp4_value(down_row, down_scale_row, mid, down_scale_2);
    }
    out[out_col] = route_weight * acc;
  }
  return out;
}

std::vector<float> cpu_residual_add(const std::vector<float>& residual,
                                    const std::vector<float>& delta) {
  assert(residual.size() == delta.size());
  std::vector<float> out(residual.size(), 0.0f);
  for (size_t idx = 0; idx < residual.size(); ++idx) {
    out[idx] = residual[idx] + delta[idx];
  }
  return out;
}

std::vector<float> cpu_residual_add_bf16(const std::vector<uint16_t>& residual,
                                         const std::vector<uint16_t>& delta) {
  assert(residual.size() == delta.size());
  std::vector<float> out(residual.size(), 0.0f);
  for (size_t idx = 0; idx < residual.size(); ++idx) {
    out[idx] =
        bf16_to_f32(f32_to_bf16(bf16_to_f32(residual[idx]) + bf16_to_f32(delta[idx])));
  }
  return out;
}

std::vector<float> cpu_gather_rows(const std::vector<float>& src,
                                   const std::vector<uint32_t>& row_indices, size_t row_width) {
  std::vector<float> out(row_indices.size() * row_width, 0.0f);
  for (size_t compact_row = 0; compact_row < row_indices.size(); ++compact_row) {
    const size_t source_row = row_indices[compact_row];
    for (size_t col = 0; col < row_width; ++col) {
      out[compact_row * row_width + col] = src[source_row * row_width + col];
    }
  }
  return out;
}

std::vector<float> cpu_scatter_add_rows(const std::vector<float>& src,
                                        const std::vector<uint32_t>& row_indices,
                                        size_t output_rows, size_t row_width) {
  std::vector<float> out(output_rows * row_width, 0.0f);
  for (size_t compact_row = 0; compact_row < row_indices.size(); ++compact_row) {
    const size_t dest_row = row_indices[compact_row];
    for (size_t col = 0; col < row_width; ++col) {
      out[dest_row * row_width + col] += src[compact_row * row_width + col];
    }
  }
  return out;
}

struct RouterTopKRef {
  std::vector<uint32_t> indices;
  std::vector<float> scores;
  std::vector<float> weights;
};

float cpu_sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

RouterTopKRef cpu_router_topk(const std::vector<float>& hidden,
                              const std::vector<float>& router_weight,
                              const std::vector<float>& correction_bias, size_t rows,
                              size_t hidden_dim, size_t experts, size_t top_k) {
  RouterTopKRef out = {};
  out.indices.resize(rows * top_k, 0);
  out.scores.resize(rows * top_k, 0.0f);
  out.weights.resize(rows * top_k, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    std::vector<float> best_scores(top_k, 0.0f);
    std::vector<float> best_corrected(top_k, -INFINITY);
    std::vector<uint32_t> best_indices(top_k, 0);
    for (size_t expert = 0; expert < experts; ++expert) {
      float logit = 0.0f;
      for (size_t col = 0; col < hidden_dim; ++col) {
        logit += hidden[row * hidden_dim + col] * router_weight[expert * hidden_dim + col];
      }
      const float score = cpu_sigmoid(logit);
      const float corrected = score + correction_bias[expert];
      for (size_t rank = 0; rank < top_k; ++rank) {
        if (corrected > best_corrected[rank]) {
          for (size_t shift = top_k - 1; shift > rank; --shift) {
            best_corrected[shift] = best_corrected[shift - 1];
            best_scores[shift] = best_scores[shift - 1];
            best_indices[shift] = best_indices[shift - 1];
          }
          best_corrected[rank] = corrected;
          best_scores[rank] = score;
          best_indices[rank] = static_cast<uint32_t>(expert);
          break;
        }
      }
    }
    float score_sum = 0.0f;
    for (float score : best_scores) {
      score_sum += score;
    }
    score_sum = std::max(score_sum, 1.0e-12f);
    for (size_t rank = 0; rank < top_k; ++rank) {
      out.indices[row * top_k + rank] = best_indices[rank];
      out.scores[row * top_k + rank] = best_scores[rank];
      out.weights[row * top_k + rank] = best_scores[rank] / score_sum * kGlm52RoutedScalingFactor;
    }
  }
  return out;
}

std::vector<float> cpu_linear(const std::vector<float>& input, const std::vector<float>& weight,
                              const float* bias, size_t rows, size_t input_dim,
                              size_t output_dim) {
  std::vector<float> output(rows * output_dim, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t out_col = 0; out_col < output_dim; ++out_col) {
      float acc = bias == nullptr ? 0.0f : bias[out_col];
      for (size_t col = 0; col < input_dim; ++col) {
        acc += input[row * input_dim + col] * weight[out_col * input_dim + col];
      }
      output[row * output_dim + out_col] = acc;
    }
  }
  return output;
}

std::vector<float> cpu_causal_attention(const std::vector<float>& q, const std::vector<float>& k,
                                        const std::vector<float>& v, size_t rows, size_t heads,
                                        size_t qk_dim, size_t v_dim, float scale) {
  std::vector<float> out(rows * heads * v_dim, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t head = 0; head < heads; ++head) {
      const float* q_vec = q.data() + (row * heads + head) * qk_dim;
      float max_score = -INFINITY;
      for (size_t key_row = 0; key_row <= row; ++key_row) {
        const float* k_vec = k.data() + (key_row * heads + head) * qk_dim;
        float dot = 0.0f;
        for (size_t col = 0; col < qk_dim; ++col) {
          dot += q_vec[col] * k_vec[col];
        }
        max_score = std::max(max_score, dot * scale);
      }
      for (size_t v_col = 0; v_col < v_dim; ++v_col) {
        float denom = 0.0f;
        float acc = 0.0f;
        for (size_t key_row = 0; key_row <= row; ++key_row) {
          const float* k_vec = k.data() + (key_row * heads + head) * qk_dim;
          float dot = 0.0f;
          for (size_t col = 0; col < qk_dim; ++col) {
            dot += q_vec[col] * k_vec[col];
          }
          const float weight = std::exp(dot * scale - max_score);
          denom += weight;
          acc += weight * v[(key_row * heads + head) * v_dim + v_col];
        }
        out[(row * heads + head) * v_dim + v_col] = acc / std::max(denom, 1.0e-12f);
      }
    }
  }
  return out;
}

std::vector<float> cpu_rope(const std::vector<float>& input, const std::vector<uint32_t>& positions,
                            size_t rows, size_t heads, size_t rotary_dim, float theta) {
  std::vector<float> out(input.size(), 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t head = 0; head < heads; ++head) {
      for (size_t pair = 0; pair < rotary_dim / 2; ++pair) {
        const size_t offset = (row * heads + head) * rotary_dim + pair * 2;
        const float angle = static_cast<float>(positions[row]) *
                            std::pow(theta, -2.0f * static_cast<float>(pair) /
                                                static_cast<float>(rotary_dim));
        const float cos_value = std::cos(angle);
        const float sin_value = std::sin(angle);
        const float even = input[offset];
        const float odd = input[offset + 1];
        out[offset] = even * cos_value - odd * sin_value;
        out[offset + 1] = even * sin_value + odd * cos_value;
      }
    }
  }
  return out;
}

std::vector<float> cpu_mla_rope_attention(const std::vector<float>& q_nope,
                                          const std::vector<float>& q_rope,
                                          const std::vector<float>& k_nope,
                                          const std::vector<float>& k_rope,
                                          const std::vector<float>& v, size_t rows,
                                          size_t heads, size_t nope_dim, size_t rope_dim,
                                          size_t v_dim, float scale) {
  std::vector<float> out(rows * heads * v_dim, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t head = 0; head < heads; ++head) {
      const float* q_nope_vec = q_nope.data() + (row * heads + head) * nope_dim;
      const float* q_rope_vec = q_rope.data() + (row * heads + head) * rope_dim;
      float max_score = -INFINITY;
      for (size_t key_row = 0; key_row <= row; ++key_row) {
        const float* k_nope_vec = k_nope.data() + (key_row * heads + head) * nope_dim;
        const float* k_rope_vec = k_rope.data() + key_row * rope_dim;
        float nope_dot = 0.0f;
        for (size_t col = 0; col < nope_dim; ++col) {
          nope_dot += q_nope_vec[col] * k_nope_vec[col];
        }
        float rope_dot = 0.0f;
        for (size_t col = 0; col < rope_dim; ++col) {
          rope_dot += q_rope_vec[col] * k_rope_vec[col];
        }
        max_score = std::max(max_score, (nope_dot + rope_dot) * scale);
      }
      for (size_t v_col = 0; v_col < v_dim; ++v_col) {
        float denom = 0.0f;
        float acc = 0.0f;
        for (size_t key_row = 0; key_row <= row; ++key_row) {
          const float* k_nope_vec = k_nope.data() + (key_row * heads + head) * nope_dim;
          const float* k_rope_vec = k_rope.data() + key_row * rope_dim;
          float nope_dot = 0.0f;
          for (size_t col = 0; col < nope_dim; ++col) {
            nope_dot += q_nope_vec[col] * k_nope_vec[col];
          }
          float rope_dot = 0.0f;
          for (size_t col = 0; col < rope_dim; ++col) {
            rope_dot += q_rope_vec[col] * k_rope_vec[col];
          }
          const float weight = std::exp((nope_dot + rope_dot) * scale - max_score);
          denom += weight;
          acc += weight * v[(key_row * heads + head) * v_dim + v_col];
        }
        out[(row * heads + head) * v_dim + v_col] = acc / std::max(denom, 1.0e-12f);
      }
    }
  }
  return out;
}

std::vector<float> cpu_embedding_lookup(const std::vector<float>& embedding,
                                        const std::vector<uint32_t>& token_ids, size_t vocab,
                                        size_t hidden) {
  std::vector<float> out(token_ids.size() * hidden, 0.0f);
  for (size_t row = 0; row < token_ids.size(); ++row) {
    const size_t token_id = token_ids[row];
    assert(token_id < vocab);
    for (size_t col = 0; col < hidden; ++col) {
      out[row * hidden + col] = embedding[token_id * hidden + col];
    }
  }
  return out;
}

struct CpuArgmaxRows {
  std::vector<uint32_t> indices;
  std::vector<float> scores;
};

CpuArgmaxRows cpu_logits_argmax(const std::vector<float>& logits, size_t rows, size_t vocab) {
  CpuArgmaxRows out;
  out.indices.resize(rows, 0);
  out.scores.resize(rows, -INFINITY);
  for (size_t row = 0; row < rows; ++row) {
    float best_score = -INFINITY;
    uint32_t best_index = 0;
    for (size_t col = 0; col < vocab; ++col) {
      const float score = logits[row * vocab + col];
      const uint32_t token_id = static_cast<uint32_t>(col);
      if (score > best_score || (score == best_score && token_id < best_index)) {
        best_score = score;
        best_index = token_id;
      }
    }
    out.indices[row] = best_index;
    out.scores[row] = best_score;
  }
  return out;
}

CpuArgmaxRows cpu_lm_head_argmax_bf16(const std::vector<uint16_t>& hidden,
                                      const std::vector<uint16_t>& lm_head, size_t rows,
                                      size_t hidden_dim, size_t vocab) {
  CpuArgmaxRows out;
  out.indices.resize(rows, 0);
  out.scores.resize(rows, -INFINITY);
  for (size_t row = 0; row < rows; ++row) {
    float best_score = -INFINITY;
    uint32_t best_index = 0;
    for (size_t token = 0; token < vocab; ++token) {
      float score = 0.0f;
      for (size_t col = 0; col < hidden_dim; ++col) {
        score += bf16_to_f32(hidden[row * hidden_dim + col]) *
                 bf16_to_f32(lm_head[token * hidden_dim + col]);
      }
      const uint32_t token_id = static_cast<uint32_t>(token);
      if (score > best_score || (score == best_score && token_id < best_index)) {
        best_score = score;
        best_index = token_id;
      }
    }
    out.indices[row] = best_index;
    out.scores[row] = best_score;
  }
  return out;
}

struct CpuSampleRows {
  std::vector<uint32_t> indices;
  std::vector<float> scores;
};

CpuSampleRows cpu_logits_sample_topk_topp(const std::vector<float>& logits,
                                          const std::vector<float>& random_uniforms, size_t rows,
                                          size_t vocab, float temperature, size_t top_k,
                                          float top_p) {
  CpuSampleRows out;
  out.indices.resize(rows, 0);
  out.scores.resize(rows, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    std::vector<float> best_logits(top_k, -INFINITY);
    std::vector<uint32_t> best_indices(top_k, 0);
    for (size_t col = 0; col < vocab; ++col) {
      const float logit = logits[row * vocab + col];
      const uint32_t token_id = static_cast<uint32_t>(col);
      for (size_t rank = 0; rank < top_k; ++rank) {
        if (logit > best_logits[rank] ||
            (logit == best_logits[rank] && token_id < best_indices[rank])) {
          for (size_t shift = top_k - 1; shift > rank; --shift) {
            best_logits[shift] = best_logits[shift - 1];
            best_indices[shift] = best_indices[shift - 1];
          }
          best_logits[rank] = logit;
          best_indices[rank] = token_id;
          break;
        }
      }
    }

    std::vector<float> scaled(top_k, 0.0f);
    float max_scaled = -INFINITY;
    for (size_t rank = 0; rank < top_k; ++rank) {
      scaled[rank] = best_logits[rank] / temperature;
      max_scaled = std::max(max_scaled, scaled[rank]);
    }
    std::vector<float> probs(top_k, 0.0f);
    float total = 0.0f;
    for (size_t rank = 0; rank < top_k; ++rank) {
      probs[rank] = std::exp(scaled[rank] - max_scaled);
      total += probs[rank];
    }
    total = std::max(total, 1.0e-20f);
    for (float& prob : probs) {
      prob /= total;
    }

    const float top_p_clamped = std::min(std::max(top_p, 1.0e-6f), 1.0f);
    float nucleus_mass = 0.0f;
    size_t nucleus_count = 0;
    for (size_t rank = 0; rank < top_k; ++rank) {
      nucleus_mass += probs[rank];
      nucleus_count = rank + 1;
      if (nucleus_mass >= top_p_clamped) {
        break;
      }
    }
    nucleus_mass = std::max(nucleus_mass, 1.0e-20f);

    const float target =
        std::min(std::max(random_uniforms[row], 0.0f), 0.99999994f) * nucleus_mass;
    float cumulative = 0.0f;
    size_t selected_rank = nucleus_count - 1;
    for (size_t rank = 0; rank < nucleus_count; ++rank) {
      cumulative += probs[rank];
      if (target <= cumulative) {
        selected_rank = rank;
        break;
      }
    }
    out.indices[row] = best_indices[selected_rank];
    out.scores[row] = probs[selected_rank] / nucleus_mass;
  }
  return out;
}

CpuSampleRows cpu_lm_head_sample_topk_topp_bf16(const std::vector<uint16_t>& hidden,
                                                const std::vector<uint16_t>& lm_head, size_t rows,
                                                size_t hidden_dim, size_t vocab,
                                                const std::vector<float>& random_uniforms,
                                                float temperature, size_t top_k, float top_p) {
  std::vector<float> logits(rows * vocab, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t token = 0; token < vocab; ++token) {
      float score = 0.0f;
      for (size_t col = 0; col < hidden_dim; ++col) {
        score += bf16_to_f32(hidden[row * hidden_dim + col]) *
                 bf16_to_f32(lm_head[token * hidden_dim + col]);
      }
      logits[row * vocab + token] = score;
    }
  }
  return cpu_logits_sample_topk_topp(logits, random_uniforms, rows, vocab, temperature, top_k,
                                     top_p);
}

glmrt_device_buffer_t device_buffer(size_t bytes) {
  glmrt_device_buffer_t buffer = {};
  require_status(glmrt_alloc_device_buffer(bytes, &buffer), "glmrt_alloc_device_buffer");
  return buffer;
}

template <typename T>
void copy_h2d(glmrt_device_buffer_t buffer, const std::vector<T>& values) {
  require_status(glmrt_copy_h2d(buffer, values.data(), values.size() * sizeof(T)),
                 "glmrt_copy_h2d");
}

template <typename T>
std::vector<T> copy_d2h(glmrt_device_buffer_t buffer, size_t count) {
  std::vector<T> values(count);
  require_status(glmrt_copy_d2h(values.data(), buffer, values.size() * sizeof(T)),
                 "glmrt_copy_d2h");
  return values;
}

void free_buffer(glmrt_device_buffer_t* buffer) {
  require_status(glmrt_free_device_buffer(buffer), "glmrt_free_device_buffer");
}

void test_cuda_copy_d2d_2d_async_copies_active_row_prefixes() {
  const std::vector<uint8_t> source = {
      1, 2, 3, 90, 91,
      4, 5, 6, 92, 93,
      7, 8, 9, 94, 95,
  };
  const std::vector<uint8_t> initial(12, 0);
  const std::vector<uint8_t> expected = {
      1, 2, 3, 0,
      4, 5, 6, 0,
      7, 8, 9, 0,
  };
  auto source_device = device_buffer(source.size());
  auto destination_device = device_buffer(initial.size());
  copy_h2d(source_device, source);
  copy_h2d(destination_device, initial);
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate 2D D2D copy");
  require_status(glmrt_copy_d2d_2d_async(destination_device, 4, source_device, 5, 3, 3, stream),
                 "glmrt_copy_d2d_2d_async");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize 2D D2D copy");
  assert(copy_d2h<uint8_t>(destination_device, initial.size()) == expected);
  require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy 2D D2D copy");
  free_buffer(&source_device);
  free_buffer(&destination_device);
}

void test_cuda_mla_merge_state_bf16_matches_weighted_reference() {
  constexpr size_t heads = 2;
  constexpr size_t rank = 512;
  std::vector<uint16_t> accumulator(heads * rank);
  std::vector<uint16_t> partial(heads * rank);
  std::fill(accumulator.begin(), accumulator.begin() + rank, f32_to_bf16(1.0f));
  std::fill(accumulator.begin() + rank, accumulator.end(), f32_to_bf16(2.0f));
  std::fill(partial.begin(), partial.begin() + rank, f32_to_bf16(4.0f));
  std::fill(partial.begin() + rank, partial.end(), f32_to_bf16(8.0f));
  const std::vector<float> accumulator_lse = {std::log(2.0f), 0.0f};
  const std::vector<float> partial_lse = {0.0f, std::log(3.0f)};

  auto accumulator_device = device_buffer(accumulator.size() * sizeof(uint16_t));
  auto accumulator_lse_device = device_buffer(accumulator_lse.size() * sizeof(float));
  auto partial_device = device_buffer(partial.size() * sizeof(uint16_t));
  auto partial_lse_device = device_buffer(partial_lse.size() * sizeof(float));
  copy_h2d(accumulator_device, accumulator);
  copy_h2d(accumulator_lse_device, accumulator_lse);
  copy_h2d(partial_device, partial);
  copy_h2d(partial_lse_device, partial_lse);
  require_status(glmrt_cuda_mla_merge_state_bf16(
                     static_cast<uint16_t*>(accumulator_device.ptr),
                     static_cast<float*>(accumulator_lse_device.ptr),
                     static_cast<const uint16_t*>(partial_device.ptr),
                     static_cast<const float*>(partial_lse_device.ptr), heads, rank),
                 "glmrt_cuda_mla_merge_state_bf16");

  const auto merged = bf16_to_f32_values(
      copy_d2h<uint16_t>(accumulator_device, accumulator.size()));
  for (size_t col = 0; col < rank; ++col) {
    assert(std::fabs(merged[col] - 2.0f) <= 1.0e-2f);
    assert(std::fabs(merged[rank + col] - 6.5f) <= 1.0e-2f);
  }
  const auto merged_lse = copy_d2h<float>(accumulator_lse_device, heads);
  assert(std::fabs(merged_lse[0] - std::log(3.0f)) <= 1.0e-5f);
  assert(std::fabs(merged_lse[1] - std::log(4.0f)) <= 1.0e-5f);

  free_buffer(&accumulator_device);
  free_buffer(&accumulator_lse_device);
  free_buffer(&partial_device);
  free_buffer(&partial_lse_device);
}

void test_cuda_device_info() {
  glmrt_cuda_device_info_t info = {};
  require_status(glmrt_cuda_device_info(0, &info), "glmrt_cuda_device_info");
  assert(info.cuda_available == 1);
  assert(info.compute_capability_major >= 12);
}

void test_cuda_rmsnorm_matches_ref() {
  const int rows = 2;
  const int hidden = 8;
  const float eps = 1.0e-5f;
  std::vector<float> x(rows * hidden);
  std::iota(x.begin(), x.end(), 1.0f);
  for (float& value : x) {
    value = value * 0.125f - 0.5f;
  }
  std::vector<float> weight(hidden);
  for (int idx = 0; idx < hidden; ++idx) {
    weight[idx] = 1.0f + idx * 0.05f;
  }

  auto dx = device_buffer(x.size() * sizeof(float));
  auto dw = device_buffer(weight.size() * sizeof(float));
  auto dy = device_buffer(x.size() * sizeof(float));
  copy_h2d(dx, x);
  copy_h2d(dw, weight);
  require_status(glmrt_cuda_rmsnorm_f32(static_cast<float*>(dx.ptr), static_cast<float*>(dw.ptr),
                                       static_cast<float*>(dy.ptr), rows, hidden, eps),
                 "glmrt_cuda_rmsnorm_f32");
  assert_close(copy_d2h<float>(dy, x.size()), cpu_rmsnorm(x, weight, rows, hidden, eps));
  free_buffer(&dx);
  free_buffer(&dw);
  free_buffer(&dy);
}

void test_cuda_rmsnorm_bf16_matches_ref() {
  const int rows = 2;
  const int hidden = 8;
  const float eps = 1.0e-5f;
  std::vector<float> x_f32(rows * hidden);
  std::iota(x_f32.begin(), x_f32.end(), 1.0f);
  for (float& value : x_f32) {
    value = value * 0.125f - 0.5f;
  }
  std::vector<float> weight_f32(hidden);
  for (int idx = 0; idx < hidden; ++idx) {
    weight_f32[idx] = 1.0f + idx * 0.05f;
  }
  const std::vector<uint16_t> x = bf16_values(x_f32);
  const std::vector<uint16_t> weight = bf16_values(weight_f32);
  std::vector<float> expected =
      cpu_rmsnorm(bf16_to_f32_values(x), bf16_to_f32_values(weight), rows, hidden, eps);
  for (float& value : expected) {
    value = bf16_to_f32(f32_to_bf16(value));
  }

  auto dx = device_buffer(x.size() * sizeof(uint16_t));
  auto dw = device_buffer(weight.size() * sizeof(uint16_t));
  auto dy = device_buffer(x.size() * sizeof(uint16_t));
  copy_h2d(dx, x);
  copy_h2d(dw, weight);
  require_status(glmrt_cuda_rmsnorm_bf16(static_cast<uint16_t*>(dx.ptr),
                                        static_cast<uint16_t*>(dw.ptr),
                                        static_cast<uint16_t*>(dy.ptr), rows, hidden, eps),
                 "glmrt_cuda_rmsnorm_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dy, x.size())), expected);
  free_buffer(&dx);
  free_buffer(&dw);
  free_buffer(&dy);
}

void test_cuda_mlp_matches_ref_small() {
  const int hidden = 4;
  const int intermediate = 3;
  std::vector<float> x = {0.25f, -0.5f, 0.75f, 1.0f};
  std::vector<float> gate_weight = {
      0.1f, 0.2f, -0.1f, 0.3f,
      -0.4f, 0.5f, 0.2f, -0.2f,
      0.3f, -0.1f, 0.4f, 0.2f,
  };
  std::vector<float> up_weight = {
      -0.2f, 0.1f, 0.3f, 0.4f,
      0.5f, -0.3f, 0.2f, 0.1f,
      0.2f, 0.4f, -0.5f, 0.3f,
  };
  std::vector<float> down_weight = {
      0.2f, -0.1f, 0.3f,
      0.4f, 0.1f, -0.2f,
      -0.3f, 0.5f, 0.2f,
      0.1f, 0.2f, 0.4f,
  };

  auto dx = device_buffer(x.size() * sizeof(float));
  auto dgate = device_buffer(gate_weight.size() * sizeof(float));
  auto dup = device_buffer(up_weight.size() * sizeof(float));
  auto ddown = device_buffer(down_weight.size() * sizeof(float));
  auto dy = device_buffer(x.size() * sizeof(float));
  copy_h2d(dx, x);
  copy_h2d(dgate, gate_weight);
  copy_h2d(dup, up_weight);
  copy_h2d(ddown, down_weight);
  require_status(glmrt_cuda_silu_gated_mlp_f32(
                     static_cast<float*>(dx.ptr), static_cast<float*>(dgate.ptr),
                     static_cast<float*>(dup.ptr), static_cast<float*>(ddown.ptr),
                     static_cast<float*>(dy.ptr), hidden, intermediate),
                 "glmrt_cuda_silu_gated_mlp_f32");
  assert_close(copy_d2h<float>(dy, x.size()),
               cpu_mlp(x, gate_weight, up_weight, down_weight, hidden, intermediate));
  free_buffer(&dx);
  free_buffer(&dgate);
  free_buffer(&dup);
  free_buffer(&ddown);
  free_buffer(&dy);
}

void test_cuda_mlp_rows_matches_ref_small() {
  const size_t rows = 2;
  const size_t hidden = 3;
  const size_t intermediate = 2;
  std::vector<float> x = {
      0.25f, -0.5f, 0.75f,
      -1.0f, 0.5f, 0.125f,
  };
  std::vector<float> gate_weight = {
      0.1f, 0.2f, -0.1f,
      -0.4f, 0.5f, 0.2f,
  };
  std::vector<float> up_weight = {
      -0.2f, 0.1f, 0.3f,
      0.5f, -0.3f, 0.2f,
  };
  std::vector<float> down_weight = {
      0.2f, -0.1f,
      0.4f, 0.1f,
      -0.3f, 0.5f,
  };

  auto dx = device_buffer(x.size() * sizeof(float));
  auto dgate = device_buffer(gate_weight.size() * sizeof(float));
  auto dup = device_buffer(up_weight.size() * sizeof(float));
  auto ddown = device_buffer(down_weight.size() * sizeof(float));
  auto dy = device_buffer(x.size() * sizeof(float));
  copy_h2d(dx, x);
  copy_h2d(dgate, gate_weight);
  copy_h2d(dup, up_weight);
  copy_h2d(ddown, down_weight);
  require_status(glmrt_cuda_silu_gated_mlp_rows_f32(
                     static_cast<float*>(dx.ptr), static_cast<float*>(dgate.ptr),
                     static_cast<float*>(dup.ptr), static_cast<float*>(ddown.ptr),
                     static_cast<float*>(dy.ptr), rows, hidden, intermediate),
                 "glmrt_cuda_silu_gated_mlp_rows_f32");
  assert_close(copy_d2h<float>(dy, x.size()),
               cpu_mlp_rows(x, gate_weight, up_weight, down_weight, rows, hidden, intermediate));
  free_buffer(&dx);
  free_buffer(&dgate);
  free_buffer(&dup);
  free_buffer(&ddown);
  free_buffer(&dy);
}

void test_cuda_nvfp4_route_bf16_staged_reduces_wide_dims() {
  const size_t hidden_dim = 300;
  const size_t intermediate = 320;
  const size_t output_dim = 5;
  const size_t packed_hidden_bytes = (hidden_dim + 1) / 2;
  const size_t hidden_scale_bytes = (hidden_dim + 15) / 16;
  const size_t packed_intermediate_bytes = (intermediate + 1) / 2;
  const size_t intermediate_scale_bytes = (intermediate + 15) / 16;
  std::vector<float> hidden(hidden_dim, 0.0f);
  for (size_t idx = 0; idx < hidden.size(); ++idx) {
    hidden[idx] = static_cast<float>(static_cast<int>(idx % 17) - 8) * 0.03125f;
  }
  const std::vector<uint16_t> hidden_bf16 = bf16_values(hidden);
  const std::vector<float> hidden_expected = bf16_to_f32_values(hidden_bf16);
  std::vector<uint8_t> gate_weight(intermediate * packed_hidden_bytes, 0);
  std::vector<uint8_t> up_weight(intermediate * packed_hidden_bytes, 0);
  std::vector<uint8_t> down_weight(output_dim * packed_intermediate_bytes, 0);
  std::vector<uint8_t> gate_scale(intermediate * hidden_scale_bytes, 0x38);
  std::vector<uint8_t> up_scale(intermediate * hidden_scale_bytes, 0x38);
  std::vector<uint8_t> down_scale(output_dim * intermediate_scale_bytes, 0x38);
  for (size_t idx = 0; idx < gate_weight.size(); ++idx) {
    gate_weight[idx] = static_cast<uint8_t>(((idx * 7) & 0x0f) | (((idx * 11 + 3) & 0x0f) << 4));
    up_weight[idx] = static_cast<uint8_t>(((idx * 5 + 1) & 0x0f) | (((idx * 13 + 2) & 0x0f) << 4));
  }
  for (size_t idx = 0; idx < down_weight.size(); ++idx) {
    down_weight[idx] =
        static_cast<uint8_t>(((idx * 3 + 4) & 0x0f) | (((idx * 9 + 1) & 0x0f) << 4));
  }
  const float route_weight = 0.625f;
  std::vector<float> expected =
      cpu_nvfp4_route(hidden_expected, gate_weight, gate_scale, up_weight, up_scale, down_weight,
                      down_scale, hidden_dim, intermediate, output_dim, 1.0f, 1.0f, 1.0f,
                      route_weight);

  std::vector<uint32_t> row_indices = {0};
  std::vector<float> route_weights = {route_weight};
  std::vector<float> accumulator(output_dim, 0.0f);
  std::vector<glmrt_nvfp4_route_batched_metadata_t> route_metadata(1);
  auto dhidden = device_buffer(hidden_bf16.size() * sizeof(uint16_t));
  auto drow_indices = device_buffer(row_indices.size() * sizeof(uint32_t));
  auto droute_weights = device_buffer(route_weights.size() * sizeof(float));
  auto droute_metadata = device_buffer(route_metadata.size() *
                                       sizeof(glmrt_nvfp4_route_batched_metadata_t));
  auto dgate_weight = device_buffer(gate_weight.size());
  auto dgate_scale = device_buffer(gate_scale.size());
  auto dup_weight = device_buffer(up_weight.size());
  auto dup_scale = device_buffer(up_scale.size());
  auto ddown_weight = device_buffer(down_weight.size());
  auto ddown_scale = device_buffer(down_scale.size());
  auto dactivations = device_buffer(intermediate * sizeof(float));
  auto daccumulator = device_buffer(accumulator.size() * sizeof(float));
  auto dout_bf16 = device_buffer(output_dim * sizeof(uint16_t));
  copy_h2d(dhidden, hidden_bf16);
  copy_h2d(drow_indices, row_indices);
  copy_h2d(droute_weights, route_weights);
  copy_h2d(dgate_weight, gate_weight);
  copy_h2d(dgate_scale, gate_scale);
  copy_h2d(dup_weight, up_weight);
  copy_h2d(dup_scale, up_scale);
  copy_h2d(ddown_weight, down_weight);
  copy_h2d(ddown_scale, down_scale);
  copy_h2d(daccumulator, accumulator);

  require_status(
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
          static_cast<uint16_t*>(dhidden.ptr), static_cast<uint32_t*>(drow_indices.ptr),
          static_cast<float*>(droute_weights.ptr), static_cast<uint8_t*>(dgate_weight.ptr),
          static_cast<uint8_t*>(dgate_scale.ptr), static_cast<uint8_t*>(dup_weight.ptr),
          static_cast<uint8_t*>(dup_scale.ptr), static_cast<uint8_t*>(ddown_weight.ptr),
          static_cast<uint8_t*>(ddown_scale.ptr), static_cast<float*>(dactivations.ptr),
          static_cast<float*>(daccumulator.ptr), 1, row_indices.size(), hidden_dim, hidden_dim,
          intermediate, output_dim, packed_intermediate_bytes, intermediate_scale_bytes, 1.0f,
          1.0f, 1.0f),
      "glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32 wide");
  assert_close(copy_d2h<float>(daccumulator, output_dim), expected, 1.0e-3f);

  route_metadata[0].gate_weight = reinterpret_cast<uintptr_t>(dgate_weight.ptr);
  route_metadata[0].gate_scale = reinterpret_cast<uintptr_t>(dgate_scale.ptr);
  route_metadata[0].up_weight = reinterpret_cast<uintptr_t>(dup_weight.ptr);
  route_metadata[0].up_scale = reinterpret_cast<uintptr_t>(dup_scale.ptr);
  route_metadata[0].down_weight = reinterpret_cast<uintptr_t>(ddown_weight.ptr);
  route_metadata[0].down_scale = reinterpret_cast<uintptr_t>(ddown_scale.ptr);
  route_metadata[0].intermediate = intermediate;
  route_metadata[0].down_weight_row_stride_bytes = packed_intermediate_bytes;
  route_metadata[0].down_scale_row_stride_bytes = intermediate_scale_bytes;
  route_metadata[0].gate_scale_2 = 1.0f;
  route_metadata[0].up_scale_2 = 1.0f;
  route_metadata[0].down_scale_2 = 1.0f;
  copy_h2d(droute_metadata, route_metadata);
  copy_h2d(daccumulator, accumulator);
  require_status(
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
          static_cast<uint16_t*>(dhidden.ptr), static_cast<uint32_t*>(drow_indices.ptr),
          static_cast<float*>(droute_weights.ptr),
          static_cast<glmrt_nvfp4_route_batched_metadata_t*>(droute_metadata.ptr),
          static_cast<float*>(dactivations.ptr), static_cast<float*>(daccumulator.ptr), 1,
          row_indices.size(), hidden_dim, hidden_dim, intermediate, output_dim),
      "glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32 wide");
  assert_close(copy_d2h<float>(daccumulator, output_dim), expected, 1.0e-3f);

  std::vector<float> expected_bf16 = expected;
  for (float& value : expected_bf16) {
    value = bf16_to_f32(f32_to_bf16(value));
  }
  require_status(
      glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
          static_cast<uint16_t*>(dhidden.ptr), static_cast<uint32_t*>(drow_indices.ptr),
          static_cast<float*>(droute_weights.ptr),
          static_cast<glmrt_nvfp4_route_batched_metadata_t*>(droute_metadata.ptr),
          static_cast<float*>(dactivations.ptr), static_cast<uint16_t*>(dout_bf16.ptr), 1,
          row_indices.size(), hidden_dim, hidden_dim, intermediate, output_dim),
      "glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16 wide");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout_bf16, output_dim)), expected_bf16,
               1.0e-3f);

  free_buffer(&dhidden);
  free_buffer(&drow_indices);
  free_buffer(&droute_weights);
  free_buffer(&droute_metadata);
  free_buffer(&dgate_weight);
  free_buffer(&dgate_scale);
  free_buffer(&dup_weight);
  free_buffer(&dup_scale);
  free_buffer(&ddown_weight);
  free_buffer(&ddown_scale);
  free_buffer(&dactivations);
  free_buffer(&daccumulator);
  free_buffer(&dout_bf16);
}

void test_cuda_residual_add_matches_ref() {
  std::vector<float> residual = {0.25f, -0.5f, 1.5f, 2.0f, -3.0f, 4.5f, 0.0f, -0.125f};
  std::vector<float> delta = {-0.5f, 0.25f, 0.5f, -1.0f, 3.5f, -2.0f, 0.125f, 0.25f};

  auto dresidual = device_buffer(residual.size() * sizeof(float));
  auto ddelta = device_buffer(delta.size() * sizeof(float));
  auto dout = device_buffer(residual.size() * sizeof(float));
  copy_h2d(dresidual, residual);
  copy_h2d(ddelta, delta);
  require_status(glmrt_cuda_residual_add_f32(static_cast<float*>(dresidual.ptr),
                                            static_cast<float*>(ddelta.ptr),
                                            static_cast<float*>(dout.ptr), residual.size()),
                 "glmrt_cuda_residual_add_f32");
  assert_close(copy_d2h<float>(dout, residual.size()), cpu_residual_add(residual, delta));
  free_buffer(&dresidual);
  free_buffer(&ddelta);
  free_buffer(&dout);
}

void test_cuda_residual_add_bf16_matches_ref() {
  std::vector<uint16_t> residual =
      bf16_values({0.25f, -0.5f, 1.5f, 2.0f, -3.0f, 4.5f, 0.0f, -0.125f});
  std::vector<uint16_t> delta =
      bf16_values({-0.5f, 0.25f, 0.5f, -1.0f, 3.5f, -2.0f, 0.125f, 0.25f});

  auto dresidual = device_buffer(residual.size() * sizeof(uint16_t));
  auto ddelta = device_buffer(delta.size() * sizeof(uint16_t));
  auto dout = device_buffer(residual.size() * sizeof(uint16_t));
  copy_h2d(dresidual, residual);
  copy_h2d(ddelta, delta);
  require_status(glmrt_cuda_residual_add_bf16(static_cast<uint16_t*>(dresidual.ptr),
                                             static_cast<uint16_t*>(ddelta.ptr),
                                             static_cast<uint16_t*>(dout.ptr), residual.size()),
                 "glmrt_cuda_residual_add_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, residual.size())),
               cpu_residual_add_bf16(residual, delta));
  free_buffer(&dresidual);
  free_buffer(&ddelta);
  free_buffer(&dout);
}

void test_cuda_residual_add_f32_delta_bf16_matches_ref() {
  std::vector<uint16_t> residual =
      bf16_values({0.25f, -0.5f, 1.5f, 2.0f, -3.0f, 4.5f, 0.0f, -0.125f});
  std::vector<float> delta = {-0.51f, 0.26f, 0.501f, -1.001f, 3.51f, -2.01f, 0.126f, 0.251f};
  std::vector<float> expected;
  expected.reserve(residual.size());
  for (size_t i = 0; i < residual.size(); ++i) {
    const float rounded_delta = bf16_to_f32(f32_to_bf16(delta[i]));
    expected.push_back(bf16_to_f32(f32_to_bf16(bf16_to_f32(residual[i]) + rounded_delta)));
  }

  auto dresidual = device_buffer(residual.size() * sizeof(uint16_t));
  auto ddelta = device_buffer(delta.size() * sizeof(float));
  auto dout = device_buffer(residual.size() * sizeof(uint16_t));
  copy_h2d(dresidual, residual);
  copy_h2d(ddelta, delta);
  require_status(glmrt_cuda_residual_add_f32_delta_bf16(
                     static_cast<uint16_t*>(dresidual.ptr), static_cast<float*>(ddelta.ptr),
                     static_cast<uint16_t*>(dout.ptr), residual.size()),
                 "glmrt_cuda_residual_add_f32_delta_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, residual.size())), expected);
  free_buffer(&dresidual);
  free_buffer(&ddelta);
  free_buffer(&dout);
}

void test_cuda_residual_add_shared_f32_delta_bf16_matches_ref() {
  std::vector<uint16_t> residual =
      bf16_values({0.25f, -0.5f, 1.5f, 2.0f, -3.0f, 4.5f, 0.0f, -0.125f});
  std::vector<uint16_t> shared_delta =
      bf16_values({-0.25f, 0.75f, 0.125f, -1.5f, 2.0f, -0.5f, 0.25f, 0.5f});
  std::vector<float> routed_delta = {-0.51f, 0.26f, 0.501f, -1.001f,
                                     3.51f, -2.01f, 0.126f, 0.251f};
  std::vector<float> expected;
  expected.reserve(residual.size());
  for (size_t i = 0; i < residual.size(); ++i) {
    const float routed = bf16_to_f32(f32_to_bf16(routed_delta[i]));
    const float mlp_delta = bf16_to_f32(f32_to_bf16(bf16_to_f32(shared_delta[i]) + routed));
    expected.push_back(bf16_to_f32(f32_to_bf16(bf16_to_f32(residual[i]) + mlp_delta)));
  }

  auto dresidual = device_buffer(residual.size() * sizeof(uint16_t));
  auto dshared = device_buffer(shared_delta.size() * sizeof(uint16_t));
  auto drouted = device_buffer(routed_delta.size() * sizeof(float));
  auto dout = device_buffer(residual.size() * sizeof(uint16_t));
  copy_h2d(dresidual, residual);
  copy_h2d(dshared, shared_delta);
  copy_h2d(drouted, routed_delta);
  require_status(glmrt_cuda_residual_add_shared_f32_delta_bf16(
                     static_cast<uint16_t*>(dresidual.ptr), static_cast<uint16_t*>(dshared.ptr),
                     static_cast<float*>(drouted.ptr), static_cast<uint16_t*>(dout.ptr),
                     residual.size()),
                 "glmrt_cuda_residual_add_shared_f32_delta_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, residual.size())), expected);
  free_buffer(&dresidual);
  free_buffer(&dshared);
  free_buffer(&drouted);
  free_buffer(&dout);
}

void test_cuda_row_gather_scatter_add_matches_ref() {
  const size_t row_width = 3;
  const size_t source_rows = 4;
  std::vector<float> src = {
      0.0f,  0.1f,  0.2f,
      1.0f,  1.1f,  1.2f,
      2.0f,  2.1f,  2.2f,
      -3.0f, -3.1f, -3.2f,
  };
  assert(src.size() == source_rows * row_width);
  std::vector<uint32_t> gather_indices = {2, 0, 3};
  std::vector<float> partials = {
      0.25f, 0.50f, 0.75f,
      1.00f, 1.25f, 1.50f,
      -0.25f, -0.50f, -0.75f,
  };
  std::vector<uint32_t> scatter_indices = {1, 3, 1};

  auto dsrc = device_buffer(src.size() * sizeof(float));
  auto dgather_indices = device_buffer(gather_indices.size() * sizeof(uint32_t));
  auto dgathered = device_buffer(gather_indices.size() * row_width * sizeof(float));
  auto dpartials = device_buffer(partials.size() * sizeof(float));
  auto dscatter_indices = device_buffer(scatter_indices.size() * sizeof(uint32_t));
  auto dscattered = device_buffer(source_rows * row_width * sizeof(float));
  copy_h2d(dsrc, src);
  copy_h2d(dgather_indices, gather_indices);
  copy_h2d(dpartials, partials);
  copy_h2d(dscatter_indices, scatter_indices);
  copy_h2d(dscattered, std::vector<float>(source_rows * row_width, 0.0f));

  require_status(glmrt_cuda_gather_rows_f32(static_cast<float*>(dsrc.ptr),
                                           static_cast<uint32_t*>(dgather_indices.ptr),
                                           static_cast<float*>(dgathered.ptr),
                                           gather_indices.size(), row_width),
                 "glmrt_cuda_gather_rows_f32");
  assert_close(copy_d2h<float>(dgathered, gather_indices.size() * row_width),
               cpu_gather_rows(src, gather_indices, row_width));

  require_status(glmrt_cuda_scatter_add_rows_f32(static_cast<float*>(dpartials.ptr),
                                                static_cast<uint32_t*>(dscatter_indices.ptr),
                                                static_cast<float*>(dscattered.ptr),
                                                scatter_indices.size(), row_width),
                 "glmrt_cuda_scatter_add_rows_f32");
  assert_close(copy_d2h<float>(dscattered, source_rows * row_width),
               cpu_scatter_add_rows(partials, scatter_indices, source_rows, row_width));

  free_buffer(&dsrc);
  free_buffer(&dgather_indices);
  free_buffer(&dgathered);
  free_buffer(&dpartials);
  free_buffer(&dscatter_indices);
  free_buffer(&dscattered);
}

void test_cuda_row_gather_scatter_add_bf16_matches_ref() {
  const size_t row_width = 3;
  const size_t source_rows = 4;
  const std::vector<uint16_t> src = bf16_values({
      0.0f,  0.1f,  0.2f,
      1.0f,  1.1f,  1.2f,
      2.0f,  2.1f,  2.2f,
      -3.0f, -3.1f, -3.2f,
  });
  assert(src.size() == source_rows * row_width);
  std::vector<uint32_t> gather_indices = {2, 0, 3};
  const std::vector<uint16_t> partials = bf16_values({
      0.25f, 0.50f, 0.75f,
      1.00f, 1.25f, 1.50f,
      -0.25f, -0.50f, -0.75f,
  });
  std::vector<uint32_t> scatter_indices = {1, 3, 1};

  std::vector<uint16_t> expected_gathered;
  expected_gathered.reserve(gather_indices.size() * row_width);
  for (uint32_t row : gather_indices) {
    expected_gathered.insert(expected_gathered.end(), src.begin() + row * row_width,
                             src.begin() + (row + 1) * row_width);
  }
  const auto expected_scattered = cpu_scatter_add_rows(
      bf16_to_f32_values(partials), scatter_indices, source_rows, row_width);

  auto dsrc = device_buffer(src.size() * sizeof(uint16_t));
  auto dgather_indices = device_buffer(gather_indices.size() * sizeof(uint32_t));
  auto dgathered = device_buffer(gather_indices.size() * row_width * sizeof(uint16_t));
  auto dpartials = device_buffer(partials.size() * sizeof(uint16_t));
  auto dscatter_indices = device_buffer(scatter_indices.size() * sizeof(uint32_t));
  auto dscattered = device_buffer(source_rows * row_width * sizeof(float));
  copy_h2d(dsrc, src);
  copy_h2d(dgather_indices, gather_indices);
  copy_h2d(dpartials, partials);
  copy_h2d(dscatter_indices, scatter_indices);
  copy_h2d(dscattered, std::vector<float>(source_rows * row_width, 0.0f));

  require_status(glmrt_cuda_gather_rows_bf16(static_cast<uint16_t*>(dsrc.ptr),
                                            static_cast<uint32_t*>(dgather_indices.ptr),
                                            static_cast<uint16_t*>(dgathered.ptr),
                                            gather_indices.size(), row_width),
                 "glmrt_cuda_gather_rows_bf16");
  assert(copy_d2h<uint16_t>(dgathered, gather_indices.size() * row_width) == expected_gathered);

  require_status(glmrt_cuda_scatter_add_rows_bf16_to_f32(
                     static_cast<uint16_t*>(dpartials.ptr),
                     static_cast<uint32_t*>(dscatter_indices.ptr),
                     static_cast<float*>(dscattered.ptr), scatter_indices.size(), row_width),
                 "glmrt_cuda_scatter_add_rows_bf16_to_f32");
  assert_close(copy_d2h<float>(dscattered, source_rows * row_width), expected_scattered);

  free_buffer(&dsrc);
  free_buffer(&dgather_indices);
  free_buffer(&dgathered);
  free_buffer(&dpartials);
  free_buffer(&dscatter_indices);
  free_buffer(&dscattered);
}

void test_cuda_router_topk_matches_ref() {
  const size_t rows = 2;
  const size_t hidden_dim = 3;
  const size_t experts = 4;
  const size_t top_k = 2;
  std::vector<float> hidden = {
      1.0f, -0.5f, 0.25f,
      -0.25f, 0.75f, 1.0f,
  };
  std::vector<float> router_weight = {
      0.2f, -0.1f, 0.5f,
      0.0f, 0.3f, -0.4f,
      0.6f, -0.2f, 0.1f,
      -0.3f, 0.4f, 0.2f,
  };
  std::vector<float> correction_bias = {0.01f, -0.02f, 0.03f, 0.0f};
  const RouterTopKRef expected =
      cpu_router_topk(hidden, router_weight, correction_bias, rows, hidden_dim, experts, top_k);

  auto dhidden = device_buffer(hidden.size() * sizeof(float));
  auto dweight = device_buffer(router_weight.size() * sizeof(float));
  auto dbias = device_buffer(correction_bias.size() * sizeof(float));
  auto dindices = device_buffer(expected.indices.size() * sizeof(uint32_t));
  auto dscores = device_buffer(expected.scores.size() * sizeof(float));
  auto dweights = device_buffer(expected.weights.size() * sizeof(float));
  copy_h2d(dhidden, hidden);
  copy_h2d(dweight, router_weight);
  copy_h2d(dbias, correction_bias);

  require_status(glmrt_cuda_router_topk_f32(
                     static_cast<float*>(dhidden.ptr), static_cast<float*>(dweight.ptr),
                     static_cast<float*>(dbias.ptr), static_cast<uint32_t*>(dindices.ptr),
                     static_cast<float*>(dscores.ptr), static_cast<float*>(dweights.ptr), rows,
                     hidden_dim, experts, top_k),
                 "glmrt_cuda_router_topk_f32");
  assert(copy_d2h<uint32_t>(dindices, expected.indices.size()) == expected.indices);
  assert_close(copy_d2h<float>(dscores, expected.scores.size()), expected.scores);
  assert_close(copy_d2h<float>(dweights, expected.weights.size()), expected.weights);

  free_buffer(&dhidden);
  free_buffer(&dweight);
  free_buffer(&dbias);
  free_buffer(&dindices);
  free_buffer(&dscores);
  free_buffer(&dweights);
}

void test_cuda_router_topk_bf16_matches_ref() {
  const size_t rows = 2;
  const size_t hidden_dim = 3;
  const size_t experts = 4;
  const size_t top_k = 2;
  const std::vector<uint16_t> hidden = bf16_values({
      1.0f, -0.5f, 0.25f,
      -0.25f, 0.75f, 1.0f,
  });
  const std::vector<uint16_t> router_weight = bf16_values({
      0.2f, -0.1f, 0.5f,
      0.0f, 0.3f, -0.4f,
      0.6f, -0.2f, 0.1f,
      -0.3f, 0.4f, 0.2f,
  });
  std::vector<float> correction_bias = {0.01f, -0.02f, 0.03f, 0.0f};
  const RouterTopKRef expected = cpu_router_topk(bf16_to_f32_values(hidden),
                                                bf16_to_f32_values(router_weight),
                                                correction_bias, rows, hidden_dim, experts, top_k);

  auto dhidden = device_buffer(hidden.size() * sizeof(uint16_t));
  auto dweight = device_buffer(router_weight.size() * sizeof(uint16_t));
  auto dbias = device_buffer(correction_bias.size() * sizeof(float));
  auto dindices = device_buffer(expected.indices.size() * sizeof(uint32_t));
  auto dscores = device_buffer(expected.scores.size() * sizeof(float));
  auto dweights = device_buffer(expected.weights.size() * sizeof(float));
  copy_h2d(dhidden, hidden);
  copy_h2d(dweight, router_weight);
  copy_h2d(dbias, correction_bias);

  require_status(glmrt_cuda_router_topk_bf16(
                     static_cast<uint16_t*>(dhidden.ptr), static_cast<uint16_t*>(dweight.ptr),
                     static_cast<float*>(dbias.ptr), static_cast<uint32_t*>(dindices.ptr),
                     static_cast<float*>(dscores.ptr), static_cast<float*>(dweights.ptr), rows,
                     hidden_dim, experts, top_k),
                 "glmrt_cuda_router_topk_bf16");
  assert(copy_d2h<uint32_t>(dindices, expected.indices.size()) == expected.indices);
  assert_close(copy_d2h<float>(dscores, expected.scores.size()), expected.scores);
  assert_close(copy_d2h<float>(dweights, expected.weights.size()), expected.weights);

  free_buffer(&dhidden);
  free_buffer(&dweight);
  free_buffer(&dbias);
  free_buffer(&dindices);
  free_buffer(&dscores);
  free_buffer(&dweights);
}

void test_cuda_linear_matches_ref() {
  const size_t rows = 2;
  const size_t input_dim = 3;
  const size_t output_dim = 4;
  std::vector<float> input = {
      0.5f, -1.0f, 2.0f,
      -0.25f, 0.75f, 1.5f,
  };
  std::vector<float> weight = {
      0.2f, -0.1f, 0.5f,
      0.0f, 0.3f, -0.4f,
      0.6f, -0.2f, 0.1f,
      -0.3f, 0.4f, 0.2f,
  };
  std::vector<float> bias = {0.05f, -0.10f, 0.15f, 0.20f};

  auto dinput = device_buffer(input.size() * sizeof(float));
  auto dweight = device_buffer(weight.size() * sizeof(float));
  auto dbias = device_buffer(bias.size() * sizeof(float));
  auto doutput = device_buffer(rows * output_dim * sizeof(float));
  copy_h2d(dinput, input);
  copy_h2d(dweight, weight);
  copy_h2d(dbias, bias);

  require_status(glmrt_cuda_linear_f32(static_cast<float*>(dinput.ptr),
                                      static_cast<float*>(dweight.ptr),
                                      static_cast<float*>(dbias.ptr),
                                      static_cast<float*>(doutput.ptr), rows, input_dim,
                                      output_dim),
                 "glmrt_cuda_linear_f32 bias");
  assert_close(copy_d2h<float>(doutput, rows * output_dim),
               cpu_linear(input, weight, bias.data(), rows, input_dim, output_dim));

  require_status(glmrt_cuda_linear_f32(static_cast<float*>(dinput.ptr),
                                      static_cast<float*>(dweight.ptr), nullptr,
                                      static_cast<float*>(doutput.ptr), rows, input_dim,
                                      output_dim),
                 "glmrt_cuda_linear_f32 no_bias");
  assert_close(copy_d2h<float>(doutput, rows * output_dim),
               cpu_linear(input, weight, nullptr, rows, input_dim, output_dim));

  free_buffer(&dinput);
  free_buffer(&dweight);
  free_buffer(&dbias);
  free_buffer(&doutput);
}

void test_cuda_linear_bf16_matches_ref() {
  const size_t rows = 2;
  const size_t input_dim = 3;
  const size_t output_dim = 4;
  const std::vector<uint16_t> input = bf16_values({
      0.5f, -1.0f, 2.0f,
      -0.25f, 0.75f, 1.5f,
  });
  const std::vector<uint16_t> weight = bf16_values({
      0.2f, -0.1f, 0.5f,
      0.0f, 0.3f, -0.4f,
      0.6f, -0.2f, 0.1f,
      -0.3f, 0.4f, 0.2f,
  });
  const std::vector<uint16_t> bias = bf16_values({0.05f, -0.10f, 0.15f, 0.20f});
  std::vector<float> expected_bias = cpu_linear(
      bf16_to_f32_values(input), bf16_to_f32_values(weight), bf16_to_f32_values(bias).data(),
      rows, input_dim, output_dim);
  std::vector<float> expected_no_bias =
      cpu_linear(bf16_to_f32_values(input), bf16_to_f32_values(weight), nullptr, rows, input_dim,
                 output_dim);
  for (float& value : expected_bias) {
    value = bf16_to_f32(f32_to_bf16(value));
  }
  for (float& value : expected_no_bias) {
    value = bf16_to_f32(f32_to_bf16(value));
  }

  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  auto dweight = device_buffer(weight.size() * sizeof(uint16_t));
  auto dbias = device_buffer(bias.size() * sizeof(uint16_t));
  auto doutput = device_buffer(rows * output_dim * sizeof(uint16_t));
  copy_h2d(dinput, input);
  copy_h2d(dweight, weight);
  copy_h2d(dbias, bias);

  require_status(glmrt_cuda_linear_bf16(static_cast<uint16_t*>(dinput.ptr),
                                       static_cast<uint16_t*>(dweight.ptr),
                                       static_cast<uint16_t*>(dbias.ptr),
                                       static_cast<uint16_t*>(doutput.ptr), rows, input_dim,
                                       output_dim),
                 "glmrt_cuda_linear_bf16 bias");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(doutput, rows * output_dim)),
               expected_bias);

  require_status(glmrt_cuda_linear_bf16(static_cast<uint16_t*>(dinput.ptr),
                                       static_cast<uint16_t*>(dweight.ptr), nullptr,
                                       static_cast<uint16_t*>(doutput.ptr), rows, input_dim,
                                       output_dim),
                 "glmrt_cuda_linear_bf16 no_bias");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(doutput, rows * output_dim)),
               expected_no_bias);

  free_buffer(&dinput);
  free_buffer(&dweight);
  free_buffer(&dbias);
  free_buffer(&doutput);
}

void test_cuda_linear_bf16_m1_parity_batched_matches_recurrent_m1() {
  constexpr size_t max_rows = 8;
  constexpr size_t input_dim = 6144;
  std::vector<uint16_t> input(max_rows * input_dim, 0);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = f32_to_bf16(
        (static_cast<float>((index * 17 + index / input_dim) % 53) - 26.0f) /
        13.0f);
  }
  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  copy_h2d(dinput, input);
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream),
               "cudaStreamCreate BF16 M1-parity batch");
  for (const size_t output_dim : {size_t{128}, size_t{2048}, size_t{2112}}) {
    std::vector<uint16_t> weight(output_dim * input_dim, 0);
    for (size_t index = 0; index < weight.size(); ++index) {
      weight[index] = f32_to_bf16(
          (static_cast<float>((index * 29 + index / input_dim) % 67) - 33.0f) /
          21.0f);
    }
    auto dweight = device_buffer(weight.size() * sizeof(uint16_t));
    auto dbatched = device_buffer(max_rows * output_dim * sizeof(uint16_t));
    auto drecurrent = device_buffer(max_rows * output_dim * sizeof(uint16_t));
    copy_h2d(dweight, weight);
    for (size_t rows = 2; rows <= max_rows; ++rows) {
      require_status(
          glmrt_cuda_linear_bf16_m1_parity_batched_cublaslt_async(
              static_cast<uint16_t*>(dinput.ptr),
              static_cast<uint16_t*>(dweight.ptr),
              static_cast<uint16_t*>(dbatched.ptr), rows, input_dim, output_dim,
              stream),
          "glmrt_cuda_linear_bf16_m1_parity_batched_cublaslt_async");
      for (size_t row = 0; row < rows; ++row) {
        require_status(
            glmrt_cuda_linear_bf16_cublas_async(
                static_cast<uint16_t*>(dinput.ptr) + row * input_dim,
                static_cast<uint16_t*>(dweight.ptr), nullptr,
                static_cast<uint16_t*>(drecurrent.ptr) + row * output_dim, 1,
                input_dim, output_dim, stream),
            "glmrt_cuda_linear_bf16_cublas_async parity reference");
      }
      require_cuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize BF16 M1-parity batch");
      assert(copy_d2h<uint16_t>(dbatched, rows * output_dim) ==
             copy_d2h<uint16_t>(drecurrent, rows * output_dim));
    }
    free_buffer(&dweight);
    free_buffer(&dbatched);
    free_buffer(&drecurrent);
  }

  require_cuda(cudaStreamDestroy(stream),
               "cudaStreamDestroy BF16 M1-parity batch");
  free_buffer(&dinput);
}

void test_cuda_glm_dsa_sort_selected_indices_orders_each_row() {
  constexpr size_t rows = 4;
  constexpr size_t width = 2048;
  std::vector<int32_t> selected(rows * width, 0);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < width; ++column) {
      selected[row * width + column] =
          static_cast<int32_t>(row * 4096 + ((column * 811) % width)) - 1024;
    }
  }
  std::vector<int32_t> expected = selected;
  for (size_t row = 0; row < rows; ++row) {
    std::sort(expected.begin() + row * width,
              expected.begin() + (row + 1) * width);
  }

  auto dselected = device_buffer(selected.size() * sizeof(int32_t));
  copy_h2d(dselected, selected);
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream),
               "cudaStreamCreate DSA selected-index sort");
  require_status(glmrt_cuda_glm_dsa_sort_selected_indices_async(
                     static_cast<int32_t*>(dselected.ptr), rows, width, stream),
                 "glmrt_cuda_glm_dsa_sort_selected_indices_async");
  require_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize DSA selected-index sort");
  assert(copy_d2h<int32_t>(dselected, selected.size()) == expected);

  require_cuda(cudaStreamDestroy(stream),
               "cudaStreamDestroy DSA selected-index sort");
  free_buffer(&dselected);
}

void test_cuda_w8a16_parity_batched_matches_recurrent_m1() {
  constexpr size_t rows = 7;
  constexpr size_t input_dim = 256;
  constexpr size_t output_dim = 64;
  std::vector<uint16_t> input(rows * input_dim, 0);
  std::vector<uint16_t> source_weight(output_dim * input_dim, 0);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = f32_to_bf16(
        (static_cast<float>((index * 17 + index / input_dim) % 53) - 26.0f) / 13.0f);
  }
  for (size_t index = 0; index < source_weight.size(); ++index) {
    source_weight[index] = f32_to_bf16(
        (static_cast<float>((index * 29 + index / input_dim) % 67) - 33.0f) / 21.0f);
  }

  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  auto dsource_weight = device_buffer(source_weight.size() * sizeof(uint16_t));
  auto dweight = device_buffer(output_dim * input_dim);
  auto dscales = device_buffer(output_dim * sizeof(float));
  auto dbatched = device_buffer(rows * output_dim * sizeof(uint16_t));
  auto drecurrent = device_buffer(rows * output_dim * sizeof(uint16_t));
  copy_h2d(dinput, input);
  copy_h2d(dsource_weight, source_weight);
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate W8A16 parity batch");
  require_status(glmrt_cuda_quantize_bf16_w8a16_group256_async(
                     static_cast<uint16_t*>(dsource_weight.ptr),
                     static_cast<int8_t*>(dweight.ptr), static_cast<float*>(dscales.ptr),
                     input_dim, output_dim, 0, stream),
                 "glmrt_cuda_quantize_bf16_w8a16_group256_async parity batch");
  require_status(glmrt_cuda_linear_w8a16_group256_m1_parity_batched_async(
                     static_cast<uint16_t*>(dinput.ptr), static_cast<int8_t*>(dweight.ptr),
                     static_cast<float*>(dscales.ptr), static_cast<uint16_t*>(dbatched.ptr), rows,
                     input_dim, output_dim, stream),
                 "glmrt_cuda_linear_w8a16_group256_m1_parity_batched_async");
  for (size_t row = 0; row < rows; ++row) {
    require_status(glmrt_cuda_linear_w8a16_group256_m1_simt_async(
                       static_cast<uint16_t*>(dinput.ptr) + row * input_dim,
                       static_cast<int8_t*>(dweight.ptr), static_cast<float*>(dscales.ptr),
                       static_cast<uint16_t*>(drecurrent.ptr) + row * output_dim, input_dim,
                       output_dim, 3, stream),
                   "glmrt_cuda_linear_w8a16_group256_m1_simt_async parity reference");
  }
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize W8A16 parity batch");
  assert(copy_d2h<uint16_t>(dbatched, rows * output_dim) ==
         copy_d2h<uint16_t>(drecurrent, rows * output_dim));

  require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy W8A16 parity batch");
  free_buffer(&dinput);
  free_buffer(&dsource_weight);
  free_buffer(&dweight);
  free_buffer(&dscales);
  free_buffer(&dbatched);
  free_buffer(&drecurrent);
}

void test_cuda_w8a16_packed_parity_batched_matches_recurrent_m1() {
  constexpr size_t max_rows = 8;
  constexpr size_t input_dim = 8192;
  constexpr size_t output_dim = 64;
  std::vector<uint16_t> input(max_rows * input_dim, 0);
  std::vector<uint16_t> source_weight(output_dim * input_dim, 0);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = f32_to_bf16(
        (static_cast<float>((index * 17 + index / input_dim) % 53) - 26.0f) / 13.0f);
  }
  for (size_t index = 0; index < source_weight.size(); ++index) {
    source_weight[index] = f32_to_bf16(
        (static_cast<float>((index * 29 + index / input_dim) % 67) - 33.0f) / 21.0f);
  }

  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  auto dsource_weight = device_buffer(source_weight.size() * sizeof(uint16_t));
  auto dweight = device_buffer(output_dim * input_dim);
  auto dscales = device_buffer(
      output_dim * (input_dim / 256) * sizeof(float));
  auto dbatched = device_buffer(max_rows * output_dim * sizeof(uint16_t));
  auto drecurrent = device_buffer(max_rows * output_dim * sizeof(uint16_t));
  copy_h2d(dinput, input);
  copy_h2d(dsource_weight, source_weight);
  cudaStream_t stream = nullptr;
  require_cuda(cudaStreamCreate(&stream),
               "cudaStreamCreate packed W8A16 parity batch");
  require_status(glmrt_cuda_quantize_bf16_w8a16_group256_packed_async(
                     static_cast<uint16_t*>(dsource_weight.ptr),
                     static_cast<int8_t*>(dweight.ptr),
                     static_cast<float*>(dscales.ptr), input_dim, output_dim,
                     stream),
                 "glmrt_cuda_quantize_bf16_w8a16_group256_packed_async parity batch");
  for (size_t rows = 2; rows <= max_rows; ++rows) {
    require_status(
        glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
            static_cast<uint16_t*>(dinput.ptr),
            static_cast<int8_t*>(dweight.ptr),
            static_cast<float*>(dscales.ptr),
            static_cast<uint16_t*>(dbatched.ptr), rows, input_dim, output_dim,
            stream),
        "glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async");
    for (size_t row = 0; row < rows; ++row) {
      require_status(glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
                         static_cast<uint16_t*>(dinput.ptr) + row * input_dim,
                         static_cast<int8_t*>(dweight.ptr),
                         static_cast<float*>(dscales.ptr),
                         static_cast<uint16_t*>(drecurrent.ptr) +
                             row * output_dim,
                         input_dim, output_dim, stream),
                     "glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async parity reference");
    }
    require_cuda(cudaStreamSynchronize(stream),
                 "cudaStreamSynchronize packed W8A16 parity batch");
    assert(copy_d2h<uint16_t>(dbatched, rows * output_dim) ==
           copy_d2h<uint16_t>(drecurrent, rows * output_dim));
  }

  require_cuda(cudaStreamDestroy(stream),
               "cudaStreamDestroy packed W8A16 parity batch");
  free_buffer(&dinput);
  free_buffer(&dsource_weight);
  free_buffer(&dweight);
  free_buffer(&dscales);
  free_buffer(&dbatched);
  free_buffer(&drecurrent);
}

void test_cuda_matmul_bf16_strided_batched_matches_ref() {
  const size_t batch_count = 2;
  const size_t rows = 2;
  const size_t input_dim = 3;
  const size_t output_dim = 4;
  const std::vector<uint16_t> input = bf16_values({
      0.5f, -1.0f, 2.0f,
      -0.25f, 0.75f, 1.5f,
      1.0f, 0.5f, -0.5f,
      -1.0f, 2.0f, 0.25f,
  });
  const std::vector<uint16_t> right = bf16_values({
      0.2f, -0.1f, 0.5f, 0.3f,
      0.0f, 0.3f, -0.4f, 0.2f,
      0.6f, -0.2f, 0.1f, -0.3f,
      -0.1f, 0.4f, 0.2f, 0.5f,
      0.3f, -0.2f, 0.6f, 0.1f,
      0.7f, 0.0f, -0.5f, 0.2f,
  });
  const std::vector<float> input_f32 = bf16_to_f32_values(input);
  const std::vector<float> right_f32 = bf16_to_f32_values(right);
  std::vector<float> expected(batch_count * rows * output_dim, 0.0f);
  for (size_t batch = 0; batch < batch_count; ++batch) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < output_dim; ++col) {
        float value = 0.0f;
        for (size_t inner = 0; inner < input_dim; ++inner) {
          value += input_f32[(batch * rows + row) * input_dim + inner] *
                   right_f32[(batch * input_dim + inner) * output_dim + col];
        }
        expected[(batch * rows + row) * output_dim + col] =
            bf16_to_f32(f32_to_bf16_rn(value));
      }
    }
  }

  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  auto dright = device_buffer(right.size() * sizeof(uint16_t));
  auto doutput = device_buffer(expected.size() * sizeof(uint16_t));
  copy_h2d(dinput, input);
  copy_h2d(dright, right);
  require_status(glmrt_cuda_matmul_bf16_strided_batched_cublas_async(
                     static_cast<uint16_t*>(dinput.ptr),
                     static_cast<uint16_t*>(dright.ptr),
                     static_cast<uint16_t*>(doutput.ptr), batch_count, rows,
                     input_dim, output_dim, rows * input_dim,
                     input_dim * output_dim, rows * output_dim, nullptr),
                 "glmrt_cuda_matmul_bf16_strided_batched_cublas_async");
  assert_close(
      bf16_to_f32_values(copy_d2h<uint16_t>(doutput, expected.size())), expected);

  free_buffer(&dinput);
  free_buffer(&dright);
  free_buffer(&doutput);
}

void test_cuda_causal_attention_matches_ref() {
  const size_t rows = 3;
  const size_t heads = 2;
  const size_t qk_dim = 2;
  const size_t v_dim = 3;
  const float scale = 0.5f;
  std::vector<float> q = {
      0.1f, 0.2f,
      -0.3f, 0.4f,
      0.5f, -0.6f,
      0.7f, 0.8f,
      -0.2f, 0.9f,
      0.3f, -0.4f,
  };
  std::vector<float> k = {
      0.2f, -0.1f,
      0.4f, 0.3f,
      -0.5f, 0.6f,
      0.7f, -0.8f,
      0.1f, 0.5f,
      -0.2f, 0.9f,
  };
  std::vector<float> v = {
      0.1f, 0.2f, 0.3f,
      -0.4f, -0.5f, -0.6f,
      0.7f, 0.8f, 0.9f,
      1.0f, -1.1f, 1.2f,
      -0.2f, 0.4f, -0.6f,
      0.3f, -0.7f, 0.5f,
  };

  auto dq = device_buffer(q.size() * sizeof(float));
  auto dk = device_buffer(k.size() * sizeof(float));
  auto dv = device_buffer(v.size() * sizeof(float));
  auto dout = device_buffer(rows * heads * v_dim * sizeof(float));
  copy_h2d(dq, q);
  copy_h2d(dk, k);
  copy_h2d(dv, v);

  require_status(glmrt_cuda_causal_attention_f32(
                     static_cast<float*>(dq.ptr), static_cast<float*>(dk.ptr),
                     static_cast<float*>(dv.ptr), static_cast<float*>(dout.ptr), rows, heads,
                     qk_dim, v_dim, scale),
                 "glmrt_cuda_causal_attention_f32");
  assert_close(copy_d2h<float>(dout, rows * heads * v_dim),
               cpu_causal_attention(q, k, v, rows, heads, qk_dim, v_dim, scale));

  free_buffer(&dq);
  free_buffer(&dk);
  free_buffer(&dv);
  free_buffer(&dout);
}

void test_cuda_causal_attention_bf16_matches_ref() {
  const size_t rows = 3;
  const size_t heads = 2;
  const size_t qk_dim = 2;
  const size_t v_dim = 3;
  const float scale = 0.5f;
  const std::vector<uint16_t> q = bf16_values({
      0.1f, 0.2f,
      -0.3f, 0.4f,
      0.5f, -0.6f,
      0.7f, 0.8f,
      -0.2f, 0.9f,
      0.3f, -0.4f,
  });
  const std::vector<uint16_t> k = bf16_values({
      0.2f, -0.1f,
      0.4f, 0.3f,
      -0.5f, 0.6f,
      0.7f, -0.8f,
      0.1f, 0.5f,
      -0.2f, 0.9f,
  });
  const std::vector<uint16_t> v = bf16_values({
      0.1f, 0.2f, 0.3f,
      -0.4f, -0.5f, -0.6f,
      0.7f, 0.8f, 0.9f,
      1.0f, -1.1f, 1.2f,
      -0.2f, 0.4f, -0.6f,
      0.3f, -0.7f, 0.5f,
  });
  std::vector<float> expected =
      cpu_causal_attention(bf16_to_f32_values(q), bf16_to_f32_values(k), bf16_to_f32_values(v),
                           rows, heads, qk_dim, v_dim, scale);
  for (float& value : expected) {
    value = bf16_to_f32(f32_to_bf16(value));
  }

  auto dq = device_buffer(q.size() * sizeof(uint16_t));
  auto dk = device_buffer(k.size() * sizeof(uint16_t));
  auto dv = device_buffer(v.size() * sizeof(uint16_t));
  auto dout = device_buffer(rows * heads * v_dim * sizeof(uint16_t));
  copy_h2d(dq, q);
  copy_h2d(dk, k);
  copy_h2d(dv, v);

  require_status(glmrt_cuda_causal_attention_bf16(
                     static_cast<uint16_t*>(dq.ptr), static_cast<uint16_t*>(dk.ptr),
                     static_cast<uint16_t*>(dv.ptr), static_cast<uint16_t*>(dout.ptr), rows, heads,
                     qk_dim, v_dim, scale),
                 "glmrt_cuda_causal_attention_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, rows * heads * v_dim)), expected,
               1.0e-5f);

  free_buffer(&dq);
  free_buffer(&dk);
  free_buffer(&dv);
  free_buffer(&dout);
}

void test_cuda_rope_matches_ref() {
  const size_t rows = 3;
  const size_t heads = 2;
  const size_t rotary_dim = 4;
  const float theta = 10'000.0f;
  std::vector<float> input = {
      -0.20f, 0.45f, 0.70f, -0.10f,
      0.15f, -0.35f, 0.25f, 0.05f,
      0.30f, -0.15f, 0.40f, 0.25f,
      -0.55f, 0.20f, -0.10f, 0.65f,
      0.55f, 0.20f, -0.35f, 0.60f,
      -0.40f, 0.30f, 0.20f, -0.55f,
  };
  std::vector<uint32_t> positions = {0, 1, 2};
  const auto expected = cpu_rope(input, positions, rows, heads, rotary_dim, theta);

  auto dinput = device_buffer(input.size() * sizeof(float));
  auto dpositions = device_buffer(positions.size() * sizeof(uint32_t));
  auto dout = device_buffer(input.size() * sizeof(float));
  copy_h2d(dinput, input);
  copy_h2d(dpositions, positions);

  require_status(glmrt_cuda_rope_f32(static_cast<float*>(dinput.ptr),
                                    static_cast<uint32_t*>(dpositions.ptr),
                                    static_cast<float*>(dout.ptr), rows, heads, rotary_dim, theta),
                 "glmrt_cuda_rope_f32");
  assert_close(copy_d2h<float>(dout, input.size()), expected, 1.0e-5f);

  free_buffer(&dinput);
  free_buffer(&dpositions);
  free_buffer(&dout);
}

void test_cuda_rope_bf16_matches_ref() {
  const size_t rows = 3;
  const size_t heads = 2;
  const size_t rotary_dim = 4;
  const float theta = 10'000.0f;
  const std::vector<uint16_t> input = bf16_values({
      -0.20f, 0.45f, 0.70f, -0.10f,
      0.15f, -0.35f, 0.25f, 0.05f,
      0.30f, -0.15f, 0.40f, 0.25f,
      -0.55f, 0.20f, -0.10f, 0.65f,
      0.55f, 0.20f, -0.35f, 0.60f,
      -0.40f, 0.30f, 0.20f, -0.55f,
  });
  std::vector<uint32_t> positions = {0, 1, 2};
  std::vector<float> expected =
      cpu_rope(bf16_to_f32_values(input), positions, rows, heads, rotary_dim, theta);
  for (float& value : expected) {
    value = bf16_to_f32(f32_to_bf16(value));
  }

  auto dinput = device_buffer(input.size() * sizeof(uint16_t));
  auto dpositions = device_buffer(positions.size() * sizeof(uint32_t));
  auto dout = device_buffer(input.size() * sizeof(uint16_t));
  copy_h2d(dinput, input);
  copy_h2d(dpositions, positions);

  require_status(glmrt_cuda_rope_bf16(static_cast<uint16_t*>(dinput.ptr),
                                     static_cast<uint32_t*>(dpositions.ptr),
                                     static_cast<uint16_t*>(dout.ptr), rows, heads, rotary_dim,
                                     theta),
                 "glmrt_cuda_rope_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, input.size())), expected, 1.0e-5f);

  free_buffer(&dinput);
  free_buffer(&dpositions);
  free_buffer(&dout);
}

void test_cuda_mla_rope_attention_bf16_matches_ref() {
  const size_t rows = 3;
  const size_t heads = 2;
  const size_t nope_dim = 2;
  const size_t rope_dim = 4;
  const size_t v_dim = 3;
  const float theta = 10'000.0f;
  const float scale = 1.0f / std::sqrt(static_cast<float>(nope_dim + rope_dim));
  std::vector<uint32_t> positions = {0, 1, 2};
  const std::vector<uint16_t> q_nope = bf16_values({
      0.10f, -0.20f,
      0.30f, 0.05f,
      -0.40f, 0.25f,
      0.15f, -0.35f,
      0.50f, 0.10f,
      -0.25f, 0.45f,
  });
  const std::vector<uint16_t> q_rope_unrotated = bf16_values({
      -0.20f, 0.45f, 0.70f, -0.10f,
      0.15f, -0.35f, 0.25f, 0.05f,
      0.30f, -0.15f, 0.40f, 0.25f,
      -0.55f, 0.20f, -0.10f, 0.65f,
      0.55f, 0.20f, -0.35f, 0.60f,
      -0.40f, 0.30f, 0.20f, -0.55f,
  });
  const std::vector<uint16_t> k_nope = bf16_values({
      0.25f, 0.15f,
      -0.10f, 0.40f,
      0.35f, -0.45f,
      0.60f, 0.20f,
      -0.30f, 0.50f,
      0.45f, -0.15f,
  });
  const std::vector<uint16_t> k_rope_unrotated = bf16_values({
      0.10f, 0.50f, -0.20f, 0.30f,
      0.35f, -0.25f, 0.45f, 0.15f,
      -0.40f, 0.30f, 0.20f, -0.55f,
  });
  const std::vector<uint16_t> v = bf16_values({
      0.10f, 0.20f, 0.30f,
      -0.40f, -0.50f, -0.60f,
      0.70f, 0.80f, 0.90f,
      1.00f, -1.10f, 1.20f,
      -0.20f, 0.40f, -0.60f,
      0.30f, -0.70f, 0.50f,
  });
  const auto q_rope_rotated_f32 =
      cpu_rope(bf16_to_f32_values(q_rope_unrotated), positions, rows, heads, rope_dim, theta);
  const auto k_rope_rotated_f32 =
      cpu_rope(bf16_to_f32_values(k_rope_unrotated), positions, rows, 1, rope_dim, theta);
  const std::vector<uint16_t> q_rope = bf16_values(q_rope_rotated_f32);
  const std::vector<uint16_t> k_rope = bf16_values(k_rope_rotated_f32);
  auto expected = cpu_mla_rope_attention(
      bf16_to_f32_values(q_nope), bf16_to_f32_values(q_rope), bf16_to_f32_values(k_nope),
      bf16_to_f32_values(k_rope), bf16_to_f32_values(v), rows, heads, nope_dim, rope_dim, v_dim,
      scale);
  for (float& value : expected) {
    value = bf16_to_f32(f32_to_bf16(value));
  }

  auto dq_nope = device_buffer(q_nope.size() * sizeof(uint16_t));
  auto dq_rope = device_buffer(q_rope.size() * sizeof(uint16_t));
  auto dk_nope = device_buffer(k_nope.size() * sizeof(uint16_t));
  auto dk_rope = device_buffer(k_rope.size() * sizeof(uint16_t));
  auto dv = device_buffer(v.size() * sizeof(uint16_t));
  auto dout = device_buffer(rows * heads * v_dim * sizeof(uint16_t));
  copy_h2d(dq_nope, q_nope);
  copy_h2d(dq_rope, q_rope);
  copy_h2d(dk_nope, k_nope);
  copy_h2d(dk_rope, k_rope);
  copy_h2d(dv, v);

  require_status(glmrt_cuda_mla_rope_attention_bf16(
                     static_cast<uint16_t*>(dq_nope.ptr), static_cast<uint16_t*>(dq_rope.ptr),
                     static_cast<uint16_t*>(dk_nope.ptr), static_cast<uint16_t*>(dk_rope.ptr),
                     static_cast<uint16_t*>(dv.ptr), static_cast<uint16_t*>(dout.ptr), rows, heads,
                     nope_dim, rope_dim, v_dim, scale),
                 "glmrt_cuda_mla_rope_attention_bf16");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dout, rows * heads * v_dim)), expected,
               1.0e-5f);

  free_buffer(&dq_nope);
  free_buffer(&dq_rope);
  free_buffer(&dk_nope);
  free_buffer(&dk_rope);
  free_buffer(&dv);
  free_buffer(&dout);
}

void test_cuda_mla_rope_attention_bf16_suffix_graph_captures_glm52_shape() {
  const size_t rows = 84;
  const size_t query_row_offset = rows - 1;
  const size_t query_rows = 1;
  const size_t heads = 64;
  const size_t nope_dim = 192;
  const size_t rope_dim = 64;
  const size_t v_dim = 256;
  const float scale = 1.0f / std::sqrt(static_cast<float>(nope_dim + rope_dim));
  auto patterned = [](size_t count, float step) {
    std::vector<uint16_t> values(count, 0);
    for (size_t idx = 0; idx < count; ++idx) {
      const float value = (static_cast<float>(idx % 31) - 15.0f) * step;
      values[idx] = f32_to_bf16(value);
    }
    return values;
  };

  const auto q_nope = patterned(rows * heads * nope_dim, 0.001f);
  const auto q_rope = patterned(rows * heads * rope_dim, 0.0015f);
  const auto k_nope = patterned(rows * heads * nope_dim, 0.00075f);
  const auto k_rope = patterned(rows * rope_dim, 0.002f);
  const auto v = patterned(rows * heads * v_dim, 0.0005f);
  auto dq_nope = device_buffer(q_nope.size() * sizeof(uint16_t));
  auto dq_rope = device_buffer(q_rope.size() * sizeof(uint16_t));
  auto dk_nope = device_buffer(k_nope.size() * sizeof(uint16_t));
  auto dk_rope = device_buffer(k_rope.size() * sizeof(uint16_t));
  auto dv = device_buffer(v.size() * sizeof(uint16_t));
  auto dout = device_buffer(query_rows * heads * v_dim * sizeof(uint16_t));
  copy_h2d(dq_nope, q_nope);
  copy_h2d(dq_rope, q_rope);
  copy_h2d(dk_nope, k_nope);
  copy_h2d(dk_rope, k_rope);
  copy_h2d(dv, v);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate suffix MLA graph");
  require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "cudaStreamBeginCapture suffix MLA graph");
  require_status(glmrt_cuda_mla_rope_attention_bf16_suffix_async(
                     static_cast<uint16_t*>(dq_nope.ptr), static_cast<uint16_t*>(dq_rope.ptr),
                     static_cast<uint16_t*>(dk_nope.ptr), static_cast<uint16_t*>(dk_rope.ptr),
                     static_cast<uint16_t*>(dv.ptr), static_cast<uint16_t*>(dout.ptr), rows,
                     query_row_offset, query_rows, heads, nope_dim, rope_dim, v_dim, scale,
                     stream),
                 "capture glmrt_cuda_mla_rope_attention_bf16_suffix_async");
  require_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture suffix MLA graph");
  require_cuda(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate suffix MLA graph");
  require_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch suffix MLA graph");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize suffix MLA graph");

  const auto out = copy_d2h<uint16_t>(dout, query_rows * heads * v_dim);
  bool suffix_nonzero = false;
  for (size_t idx = 0; idx < out.size(); ++idx) {
    suffix_nonzero = suffix_nonzero || out[idx] != 0;
  }
  assert(suffix_nonzero);

  require_cuda(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy suffix MLA graph");
  require_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy suffix MLA graph");
  require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy suffix MLA graph");
  free_buffer(&dq_nope);
  free_buffer(&dq_rope);
  free_buffer(&dk_nope);
  free_buffer(&dk_rope);
  free_buffer(&dv);
  free_buffer(&dout);
}

void test_cuda_embedding_lookup_matches_ref() {
  const size_t vocab = 5;
  const size_t hidden = 4;
  std::vector<float> embedding = {
      0.0f, 0.1f, 0.2f, 0.3f,
      1.0f, 1.1f, 1.2f, 1.3f,
      -2.0f, -2.1f, -2.2f, -2.3f,
      3.0f, 3.1f, 3.2f, 3.3f,
      -4.0f, -4.1f, -4.2f, -4.3f,
  };
  std::vector<uint32_t> token_ids = {3, 0, 4, 2};

  auto dembedding = device_buffer(embedding.size() * sizeof(float));
  auto dtokens = device_buffer(token_ids.size() * sizeof(uint32_t));
  auto dout = device_buffer(token_ids.size() * hidden * sizeof(float));
  copy_h2d(dembedding, embedding);
  copy_h2d(dtokens, token_ids);

  require_status(glmrt_cuda_embedding_lookup_f32(static_cast<float*>(dembedding.ptr),
                                                static_cast<uint32_t*>(dtokens.ptr),
                                                static_cast<float*>(dout.ptr), token_ids.size(),
                                                vocab, hidden),
                 "glmrt_cuda_embedding_lookup_f32");
  assert_close(copy_d2h<float>(dout, token_ids.size() * hidden),
               cpu_embedding_lookup(embedding, token_ids, vocab, hidden));

  free_buffer(&dembedding);
  free_buffer(&dtokens);
  free_buffer(&dout);
}

void test_cuda_embedding_lookup_bf16_matches_ref() {
  const size_t vocab = 5;
  const size_t hidden = 4;
  const std::vector<uint16_t> embedding = bf16_values({
      0.0f, 0.1f, 0.2f, 0.3f,
      1.0f, 1.1f, 1.2f, 1.3f,
      -2.0f, -2.1f, -2.2f, -2.3f,
      3.0f, 3.1f, 3.2f, 3.3f,
      -4.0f, -4.1f, -4.2f, -4.3f,
  });
  std::vector<uint32_t> token_ids = {3, 0, 4, 2};
  std::vector<uint16_t> expected;
  expected.reserve(token_ids.size() * hidden);
  for (uint32_t token_id : token_ids) {
    expected.insert(expected.end(), embedding.begin() + token_id * hidden,
                    embedding.begin() + (token_id + 1) * hidden);
  }

  auto dembedding = device_buffer(embedding.size() * sizeof(uint16_t));
  auto dtokens = device_buffer(token_ids.size() * sizeof(uint32_t));
  auto dout = device_buffer(token_ids.size() * hidden * sizeof(uint16_t));
  copy_h2d(dembedding, embedding);
  copy_h2d(dtokens, token_ids);

  require_status(glmrt_cuda_embedding_lookup_bf16(static_cast<uint16_t*>(dembedding.ptr),
                                                 static_cast<uint32_t*>(dtokens.ptr),
                                                 static_cast<uint16_t*>(dout.ptr),
                                                 token_ids.size(), vocab, hidden),
                 "glmrt_cuda_embedding_lookup_bf16");
  assert(copy_d2h<uint16_t>(dout, token_ids.size() * hidden) == expected);

  free_buffer(&dembedding);
  free_buffer(&dtokens);
  free_buffer(&dout);
}

void test_cuda_logits_argmax_matches_ref() {
  const size_t rows = 3;
  const size_t vocab = 6;
  std::vector<float> logits = {
      -0.5f, 0.1f, 0.8f, 0.0f, 0.8f, -0.2f,
      -1.0f, -0.7f, -0.9f, -0.8f, -0.6f, -0.4f,
      1.25f, 1.0f, 0.5f, 1.25f, -2.0f, 0.0f,
  };
  const CpuArgmaxRows expected = cpu_logits_argmax(logits, rows, vocab);

  auto dlogits = device_buffer(logits.size() * sizeof(float));
  auto dindices = device_buffer(rows * sizeof(uint32_t));
  auto dscores = device_buffer(rows * sizeof(float));
  copy_h2d(dlogits, logits);

  require_status(glmrt_cuda_logits_argmax_f32(static_cast<float*>(dlogits.ptr),
                                             static_cast<uint32_t*>(dindices.ptr),
                                             static_cast<float*>(dscores.ptr), rows, vocab),
                 "glmrt_cuda_logits_argmax_f32");
  assert(copy_d2h<uint32_t>(dindices, rows) == expected.indices);
  assert_close(copy_d2h<float>(dscores, rows), expected.scores);

  free_buffer(&dlogits);
  free_buffer(&dindices);
  free_buffer(&dscores);
}

void test_cuda_lm_head_argmax_bf16_matches_ref() {
  const size_t rows = 2;
  const size_t hidden_dim = 3;
  const size_t vocab = 5;
  const std::vector<uint16_t> hidden = bf16_values({
      0.5f, -1.0f, 0.25f,
      -0.5f, 0.75f, 1.25f,
  });
  const std::vector<uint16_t> lm_head = bf16_values({
      0.25f, -0.5f, 0.75f,
      -0.25f, 0.5f, 0.125f,
      1.0f, 0.0f, -0.5f,
      0.25f, -0.5f, 0.75f,
      -1.0f, 0.25f, 0.5f,
  });
  const CpuArgmaxRows expected =
      cpu_lm_head_argmax_bf16(hidden, lm_head, rows, hidden_dim, vocab);

  auto dhidden = device_buffer(hidden.size() * sizeof(uint16_t));
  auto dlm_head = device_buffer(lm_head.size() * sizeof(uint16_t));
  auto dindices = device_buffer(rows * sizeof(uint32_t));
  auto dscores = device_buffer(rows * sizeof(float));
  copy_h2d(dhidden, hidden);
  copy_h2d(dlm_head, lm_head);

  require_status(glmrt_cuda_lm_head_argmax_bf16(
                     static_cast<uint16_t*>(dhidden.ptr), static_cast<uint16_t*>(dlm_head.ptr),
                     static_cast<uint32_t*>(dindices.ptr), static_cast<float*>(dscores.ptr), rows,
                     hidden_dim, vocab),
                 "glmrt_cuda_lm_head_argmax_bf16");
  assert(copy_d2h<uint32_t>(dindices, rows) == expected.indices);
  assert_close(copy_d2h<float>(dscores, rows), expected.scores);

  free_buffer(&dhidden);
  free_buffer(&dlm_head);
  free_buffer(&dindices);
  free_buffer(&dscores);
}

void test_cuda_lm_head_sample_topk_topp_bf16_matches_ref() {
  const size_t rows = 2;
  const size_t hidden_dim = 3;
  const size_t vocab = 5;
  const float temperature = 0.7f;
  const size_t top_k = 4;
  const float top_p = 0.82f;
  const std::vector<uint16_t> hidden = bf16_values({
      0.5f, -1.0f, 0.25f,
      -0.5f, 0.75f, 1.25f,
  });
  const std::vector<uint16_t> lm_head = bf16_values({
      0.25f, -0.5f, 0.75f,
      -0.25f, 0.5f, 0.125f,
      1.0f, 0.0f, -0.5f,
      0.25f, -0.5f, 0.75f,
      -1.0f, 0.25f, 0.5f,
  });
  const std::vector<float> random_uniforms = {0.0f, 0.99f};
  const CpuSampleRows expected = cpu_lm_head_sample_topk_topp_bf16(
      hidden, lm_head, rows, hidden_dim, vocab, random_uniforms, temperature, top_k, top_p);

  auto dhidden = device_buffer(hidden.size() * sizeof(uint16_t));
  auto dlm_head = device_buffer(lm_head.size() * sizeof(uint16_t));
  auto drandom = device_buffer(random_uniforms.size() * sizeof(float));
  auto dindices = device_buffer(rows * sizeof(uint32_t));
  auto dscores = device_buffer(rows * sizeof(float));
  copy_h2d(dhidden, hidden);
  copy_h2d(dlm_head, lm_head);
  copy_h2d(drandom, random_uniforms);

  require_status(glmrt_cuda_lm_head_sample_topk_topp_bf16(
                     static_cast<uint16_t*>(dhidden.ptr), static_cast<uint16_t*>(dlm_head.ptr),
                     static_cast<float*>(drandom.ptr), static_cast<uint32_t*>(dindices.ptr),
                     static_cast<float*>(dscores.ptr), rows, hidden_dim, vocab, temperature,
                     top_k, top_p),
                 "glmrt_cuda_lm_head_sample_topk_topp_bf16");
  assert(copy_d2h<uint32_t>(dindices, rows) == expected.indices);
  assert_close(copy_d2h<float>(dscores, rows), expected.scores);

  free_buffer(&dhidden);
  free_buffer(&dlm_head);
  free_buffer(&drandom);
  free_buffer(&dindices);
  free_buffer(&dscores);
}

void test_cuda_logits_sample_topk_topp_matches_ref() {
  const size_t rows = 3;
  const size_t vocab = 6;
  const float temperature = 0.7f;
  const size_t top_k = 4;
  const float top_p = 0.82f;
  std::vector<float> logits = {
      -0.5f, 0.1f, 0.8f, 0.0f, 0.8f, -0.2f,
      -1.0f, -0.7f, -0.9f, -0.8f, -0.6f, -0.4f,
      1.25f, 1.0f, 0.5f, 1.25f, -2.0f, 0.0f,
  };
  std::vector<float> random_uniforms = {0.0f, 0.42f, 0.99f};
  const CpuSampleRows expected = cpu_logits_sample_topk_topp(
      logits, random_uniforms, rows, vocab, temperature, top_k, top_p);

  auto dlogits = device_buffer(logits.size() * sizeof(float));
  auto drandom = device_buffer(random_uniforms.size() * sizeof(float));
  auto dindices = device_buffer(rows * sizeof(uint32_t));
  auto dscores = device_buffer(rows * sizeof(float));
  copy_h2d(dlogits, logits);
  copy_h2d(drandom, random_uniforms);

  require_status(glmrt_cuda_logits_sample_topk_topp_f32(
                     static_cast<float*>(dlogits.ptr), static_cast<float*>(drandom.ptr),
                     static_cast<uint32_t*>(dindices.ptr), static_cast<float*>(dscores.ptr), rows,
                     vocab, temperature, top_k, top_p),
                 "glmrt_cuda_logits_sample_topk_topp_f32");
  assert(copy_d2h<uint32_t>(dindices, rows) == expected.indices);
  assert_close(copy_d2h<float>(dscores, rows), expected.scores);

  free_buffer(&dlogits);
  free_buffer(&drandom);
  free_buffer(&dindices);
  free_buffer(&dscores);
}

void test_nvfp4_pack_unpack_or_skip_with_reason() {
  std::vector<uint8_t> codes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3};
  const size_t packed_count = (codes.size() + 1) / 2;
  auto dcodes = device_buffer(codes.size());
  auto dpacked = device_buffer(packed_count);
  auto dunpacked = device_buffer(codes.size());
  copy_h2d(dcodes, codes);
  require_status(glmrt_cuda_pack_nibbles(static_cast<uint8_t*>(dcodes.ptr),
                                         static_cast<uint8_t*>(dpacked.ptr), codes.size()),
                 "glmrt_cuda_pack_nibbles");
  require_status(glmrt_cuda_unpack_nibbles(static_cast<uint8_t*>(dpacked.ptr),
                                           static_cast<uint8_t*>(dunpacked.ptr), codes.size()),
                 "glmrt_cuda_unpack_nibbles");
  assert(copy_d2h<uint8_t>(dunpacked, codes.size()) == codes);
  free_buffer(&dcodes);
  free_buffer(&dpacked);
  free_buffer(&dunpacked);
}

void test_cuda_mla_kv_prepare_bf16_matches_normalized_rotated_reference() {
  constexpr size_t rows = 2;
  constexpr size_t kv_lora_rank = 512;
  constexpr size_t rope_dim = 64;
  constexpr size_t row_values = kv_lora_rank + rope_dim;
  constexpr size_t row_stride_bytes = row_values * sizeof(uint16_t);
  constexpr float eps = 1.0e-6f;
  constexpr float theta = 10000.0f;
  std::vector<float> projected_f32(rows * row_values, 0.0f);
  std::vector<float> weight_f32(kv_lora_rank, 0.0f);
  for (size_t col = 0; col < kv_lora_rank; ++col) {
    weight_f32[col] = 0.75f + static_cast<float>(col % 17) / 32.0f;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < kv_lora_rank; ++col) {
      projected_f32[row * row_values + col] =
          (static_cast<float>((row * 19 + col) % 41) - 20.0f) / 13.0f;
    }
    for (size_t col = 0; col < rope_dim; ++col) {
      projected_f32[row * row_values + kv_lora_rank + col] =
          (static_cast<float>((row * 11 + col) % 23) - 11.0f) / 17.0f;
    }
  }
  const std::vector<uint16_t> projected = bf16_values(projected_f32);
  const std::vector<uint16_t> weight = bf16_values(weight_f32);
  const std::vector<uint32_t> positions = {0, 7};
  const std::vector<float> projected_rounded = bf16_to_f32_values(projected);
  const std::vector<float> weight_rounded = bf16_to_f32_values(weight);
  std::vector<float> expected(rows * row_values, 0.0f);
  for (size_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (size_t col = 0; col < kv_lora_rank; ++col) {
      const float value = projected_rounded[row * row_values + col];
      sum += value * value;
    }
    const float inverse_rms =
        1.0f / std::sqrt(sum / static_cast<float>(kv_lora_rank) + eps);
    for (size_t col = 0; col < kv_lora_rank; ++col) {
      expected[row * row_values + col] = bf16_to_f32(f32_to_bf16(
          projected_rounded[row * row_values + col] * inverse_rms * weight_rounded[col]));
    }
    for (size_t pair = 0; pair < rope_dim / 2; ++pair) {
      const size_t col = pair * 2;
      const float angle = static_cast<float>(positions[row]) *
                          std::pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(rope_dim));
      const float even = projected_rounded[row * row_values + kv_lora_rank + col];
      const float odd = projected_rounded[row * row_values + kv_lora_rank + col + 1];
      expected[row * row_values + kv_lora_rank + col] =
          bf16_to_f32(f32_to_bf16(even * std::cos(angle) - odd * std::sin(angle)));
      expected[row * row_values + kv_lora_rank + col + 1] =
          bf16_to_f32(f32_to_bf16(even * std::sin(angle) + odd * std::cos(angle)));
    }
  }

  auto dprojected = device_buffer(projected.size() * sizeof(uint16_t));
  auto dpositions = device_buffer(positions.size() * sizeof(uint32_t));
  auto dweight = device_buffer(weight.size() * sizeof(uint16_t));
  auto dprepared = device_buffer(projected.size() * sizeof(uint16_t));
  copy_h2d(dprojected, projected);
  copy_h2d(dpositions, positions);
  copy_h2d(dweight, weight);
  require_status(glmrt_cuda_mla_kv_prepare_bf16(
                     static_cast<uint16_t*>(dprojected.ptr),
                     static_cast<uint32_t*>(dpositions.ptr),
                     static_cast<uint16_t*>(dweight.ptr), static_cast<uint16_t*>(dprepared.ptr),
                     rows, row_stride_bytes, row_stride_bytes, eps, theta),
                 "glmrt_cuda_mla_kv_prepare_bf16");
  assert_close(
      bf16_to_f32_values(copy_d2h<uint16_t>(dprepared, projected.size())), expected, 2.0e-2f);
  free_buffer(&dprojected);
  free_buffer(&dpositions);
  free_buffer(&dweight);
  free_buffer(&dprepared);
}

void test_cuda_mla_compressed_attention_reads_interleaved_cache_formats() {
  constexpr size_t rows = 3;
  constexpr size_t heads = 2;
  constexpr size_t rank = 512;
  constexpr size_t rope_dim = 64;
  constexpr size_t projected_values = rank + rope_dim;
  constexpr size_t projected_stride_bytes = projected_values * sizeof(uint16_t);
  constexpr size_t dsa_bytes = 128 * sizeof(uint16_t);
  constexpr float scale = 0.125f;

  std::vector<float> q_absorbed_f32(heads * rank);
  std::vector<float> q_rope_f32(heads * rope_dim);
  std::vector<float> projected_f32(rows * projected_values);
  for (size_t idx = 0; idx < q_absorbed_f32.size(); ++idx) {
    q_absorbed_f32[idx] = (static_cast<float>(idx % 29) - 14.0f) / 64.0f;
  }
  for (size_t idx = 0; idx < q_rope_f32.size(); ++idx) {
    q_rope_f32[idx] = (static_cast<float>(idx % 17) - 8.0f) / 32.0f;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < rank; ++col) {
      projected_f32[row * projected_values + col] =
          (static_cast<float>((row * 31 + col) % 47) - 23.0f) / 24.0f;
    }
    for (size_t col = 0; col < rope_dim; ++col) {
      projected_f32[row * projected_values + rank + col] =
          (static_cast<float>((row * 13 + col) % 23) - 11.0f) / 20.0f;
    }
  }
  const std::vector<uint16_t> q_absorbed = bf16_values(q_absorbed_f32);
  const std::vector<uint16_t> q_rope = bf16_values(q_rope_f32);
  const std::vector<uint16_t> projected = bf16_values(projected_f32);
  auto dq_absorbed = device_buffer(q_absorbed.size() * sizeof(uint16_t));
  auto dq_rope = device_buffer(q_rope.size() * sizeof(uint16_t));
  copy_h2d(dq_absorbed, q_absorbed);
  copy_h2d(dq_rope, q_rope);

  auto run_split = [&](const std::vector<uint16_t>& source) {
    std::vector<uint16_t> latent(rows * rank);
    std::vector<uint16_t> rope(rows * rope_dim);
    for (size_t row = 0; row < rows; ++row) {
      std::copy_n(source.data() + row * projected_values, rank,
                  latent.data() + row * rank);
      std::copy_n(source.data() + row * projected_values + rank, rope_dim,
                  rope.data() + row * rope_dim);
    }
    auto dlatent = device_buffer(latent.size() * sizeof(uint16_t));
    auto drope = device_buffer(rope.size() * sizeof(uint16_t));
    auto dout = device_buffer(heads * rank * sizeof(uint16_t));
    copy_h2d(dlatent, latent);
    copy_h2d(drope, rope);
    require_status(glmrt_cuda_mla_compressed_attention_bf16(
                       static_cast<uint16_t*>(dq_absorbed.ptr),
                       static_cast<uint16_t*>(dq_rope.ptr),
                       static_cast<uint16_t*>(dlatent.ptr), static_cast<uint16_t*>(drope.ptr),
                       static_cast<uint16_t*>(dout.ptr), rows, heads, rope_dim, rank, scale),
                   "glmrt_cuda_mla_compressed_attention_bf16");
    const auto output = copy_d2h<uint16_t>(dout, heads * rank);
    free_buffer(&dlatent);
    free_buffer(&drope);
    free_buffer(&dout);
    return output;
  };

  const auto bf16_reference = run_split(projected);
  constexpr size_t bf16_stride_bytes = projected_stride_bytes + dsa_bytes;
  std::vector<uint8_t> interleaved_bf16(rows * bf16_stride_bytes, 0);
  for (size_t row = 0; row < rows; ++row) {
    std::memcpy(interleaved_bf16.data() + row * bf16_stride_bytes,
                projected.data() + row * projected_values, projected_stride_bytes);
  }
  auto dbf16 = device_buffer(interleaved_bf16.size());
  auto dbf16_out = device_buffer(heads * rank * sizeof(uint16_t));
  copy_h2d(dbf16, interleaved_bf16);
  require_status(glmrt_cuda_mla_compressed_attention_interleaved_bf16(
                     static_cast<uint16_t*>(dq_absorbed.ptr),
                     static_cast<uint16_t*>(dq_rope.ptr), static_cast<uint16_t*>(dbf16.ptr),
                     static_cast<uint16_t*>(dbf16_out.ptr), rows, heads, rope_dim, rank,
                     bf16_stride_bytes, rank * sizeof(uint16_t), scale),
                 "glmrt_cuda_mla_compressed_attention_interleaved_bf16");
  assert(copy_d2h<uint16_t>(dbf16_out, heads * rank) == bf16_reference);

  auto dprojected = device_buffer(projected.size() * sizeof(uint16_t));
  copy_h2d(dprojected, projected);
  constexpr size_t fp8_base_stride = rank + 4 * sizeof(float) + rope_dim * sizeof(uint16_t);
  constexpr size_t fp8_stride = fp8_base_stride + dsa_bytes;
  auto dfp8 = device_buffer(rows * fp8_stride);
  auto dfp8_unpacked = device_buffer(projected.size() * sizeof(uint16_t));
  auto dfp8_out = device_buffer(heads * rank * sizeof(uint16_t));
  require_status(glmrt_cuda_mla_kv_pack_fp8_ds_mla(
                     static_cast<uint16_t*>(dprojected.ptr), static_cast<uint8_t*>(dfp8.ptr),
                     rows, projected_stride_bytes, fp8_stride),
                 "glmrt_cuda_mla_kv_pack_fp8_ds_mla");
  require_status(glmrt_cuda_mla_kv_unpack_fp8_ds_mla(
                     static_cast<uint8_t*>(dfp8.ptr), static_cast<uint16_t*>(dfp8_unpacked.ptr),
                     rows, fp8_stride, projected_stride_bytes),
                 "glmrt_cuda_mla_kv_unpack_fp8_ds_mla");
  const auto fp8_unpacked = copy_d2h<uint16_t>(dfp8_unpacked, projected.size());
  const auto fp8_reference = run_split(fp8_unpacked);
  require_status(glmrt_cuda_mla_compressed_attention_interleaved_fp8(
                     static_cast<uint16_t*>(dq_absorbed.ptr),
                     static_cast<uint16_t*>(dq_rope.ptr), static_cast<uint8_t*>(dfp8.ptr),
                     static_cast<uint16_t*>(dfp8_out.ptr), rows, heads, rope_dim, rank,
                     fp8_stride, scale),
                 "glmrt_cuda_mla_compressed_attention_interleaved_fp8");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dfp8_out, heads * rank)),
               bf16_to_f32_values(fp8_reference), 2.0e-2f);

  constexpr size_t mxfp4_base_stride =
      rank / 2 + rank / 16 + 16 + rope_dim * sizeof(uint16_t);
  constexpr size_t mxfp4_stride = mxfp4_base_stride + dsa_bytes;
  auto dmxfp4 = device_buffer(rows * mxfp4_stride);
  auto dmxfp4_unpacked = device_buffer(projected.size() * sizeof(uint16_t));
  auto dmxfp4_out = device_buffer(heads * rank * sizeof(uint16_t));
  require_status(glmrt_cuda_mla_kv_pack_mxfp4_ds_mla(
                     static_cast<uint16_t*>(dprojected.ptr), static_cast<uint8_t*>(dmxfp4.ptr),
                     rows, projected_stride_bytes, mxfp4_stride),
                 "glmrt_cuda_mla_kv_pack_mxfp4_ds_mla");
  require_status(glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla(
                     static_cast<uint8_t*>(dmxfp4.ptr),
                     static_cast<uint16_t*>(dmxfp4_unpacked.ptr), rows, mxfp4_stride,
                     projected_stride_bytes),
                 "glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla");
  const auto mxfp4_unpacked = copy_d2h<uint16_t>(dmxfp4_unpacked, projected.size());
  const auto mxfp4_reference = run_split(mxfp4_unpacked);
  require_status(glmrt_cuda_mla_compressed_attention_interleaved_mxfp4(
                     static_cast<uint16_t*>(dq_absorbed.ptr),
                     static_cast<uint16_t*>(dq_rope.ptr), static_cast<uint8_t*>(dmxfp4.ptr),
                     static_cast<uint16_t*>(dmxfp4_out.ptr), rows, heads, rope_dim, rank,
                     mxfp4_stride, scale),
                 "glmrt_cuda_mla_compressed_attention_interleaved_mxfp4");
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dmxfp4_out, heads * rank)),
               bf16_to_f32_values(mxfp4_reference), 2.0e-2f);

  free_buffer(&dq_absorbed);
  free_buffer(&dq_rope);
  free_buffer(&dbf16);
  free_buffer(&dbf16_out);
  free_buffer(&dprojected);
  free_buffer(&dfp8);
  free_buffer(&dfp8_unpacked);
  free_buffer(&dfp8_out);
  free_buffer(&dmxfp4);
  free_buffer(&dmxfp4_unpacked);
  free_buffer(&dmxfp4_out);
}

void test_cuda_mla_kv_mxfp4_pack_roundtrip_matches_representable_values() {
  constexpr size_t rows = 2;
  constexpr size_t nope_values = 512;
  constexpr size_t rope_values = 64;
  constexpr size_t projected_values = nope_values + rope_values;
  constexpr size_t projected_stride_bytes = projected_values * sizeof(uint16_t);
  constexpr size_t code_bytes = nope_values / 2;
  constexpr size_t scale_bytes = nope_values / 16;
  constexpr size_t padding_bytes = 16;
  constexpr size_t rope_offset = code_bytes + scale_bytes + padding_bytes;
  constexpr size_t packed_stride_bytes = rope_offset + rope_values * sizeof(uint16_t);
  const std::vector<float> codebook = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      0.0f,  -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  std::vector<float> projected_f32;
  projected_f32.reserve(rows * projected_values);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t idx = 0; idx < nope_values; ++idx) {
      projected_f32.push_back(codebook[(row + idx) % codebook.size()]);
    }
    for (size_t idx = 0; idx < rope_values; ++idx) {
      projected_f32.push_back((static_cast<float>(row * 13 + idx % 29) - 14.0f) / 64.0f);
    }
  }
  const std::vector<uint16_t> projected = bf16_values(projected_f32);

  auto dprojected = device_buffer(projected.size() * sizeof(uint16_t));
  auto dpacked = device_buffer(rows * packed_stride_bytes);
  auto dunpacked = device_buffer(projected.size() * sizeof(uint16_t));
  copy_h2d(dprojected, projected);
  require_status(glmrt_cuda_mla_kv_pack_mxfp4_ds_mla(
                     static_cast<uint16_t*>(dprojected.ptr), static_cast<uint8_t*>(dpacked.ptr),
                     rows, projected_stride_bytes, packed_stride_bytes),
                 "glmrt_cuda_mla_kv_pack_mxfp4_ds_mla");
  require_status(glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla(
                     static_cast<uint8_t*>(dpacked.ptr), static_cast<uint16_t*>(dunpacked.ptr),
                     rows, packed_stride_bytes, projected_stride_bytes),
                 "glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla");

  const std::vector<uint8_t> packed = copy_d2h<uint8_t>(dpacked, rows * packed_stride_bytes);
  const std::vector<uint16_t> unpacked = copy_d2h<uint16_t>(dunpacked, projected.size());
  for (size_t row = 0; row < rows; ++row) {
    const size_t projected_row = row * projected_values;
    const size_t packed_row = row * packed_stride_bytes;
    bool any_code = false;
    for (size_t idx = 0; idx < code_bytes; ++idx) {
      any_code = any_code || packed[packed_row + idx] != 0;
    }
    assert(any_code);
    for (size_t idx = 0; idx < scale_bytes; ++idx) {
      assert(packed[packed_row + code_bytes + idx] == 0x38);
    }
    for (size_t idx = 0; idx < padding_bytes; ++idx) {
      assert(packed[packed_row + code_bytes + scale_bytes + idx] == 0);
    }
    for (size_t idx = 0; idx < rope_values; ++idx) {
      assert(unpacked[projected_row + nope_values + idx] ==
             projected[projected_row + nope_values + idx]);
      const uint8_t* rope_bytes =
          reinterpret_cast<const uint8_t*>(&projected[projected_row + nope_values + idx]);
      assert(packed[packed_row + rope_offset + idx * sizeof(uint16_t)] ==
             rope_bytes[0]);
      assert(packed[packed_row + rope_offset + idx * sizeof(uint16_t) + 1] ==
             rope_bytes[1]);
    }
    for (size_t idx = 0; idx < nope_values; ++idx) {
      assert(unpacked[projected_row + idx] == projected[projected_row + idx]);
    }
  }

  free_buffer(&dprojected);
  free_buffer(&dpacked);
  free_buffer(&dunpacked);
}

void test_cuda_graph_replay_matches_uncaptured() {
  const int rows = 1;
  const int hidden = 8;
  const float eps = 1.0e-5f;
  std::vector<float> x0 = {0.1f, 0.2f, 0.3f, 0.4f, -0.5f, -0.4f, -0.3f, -0.2f};
  std::vector<float> x1 = {0.8f, 0.7f, -0.6f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f};
  std::vector<float> weight(hidden, 1.25f);
  std::vector<float> delta = {0.05f, -0.10f, 0.15f, -0.20f, 0.25f, -0.30f, 0.35f, -0.40f};

  auto dx = device_buffer(x0.size() * sizeof(float));
  auto dw = device_buffer(weight.size() * sizeof(float));
  auto ddelta = device_buffer(delta.size() * sizeof(float));
  auto dy_graph = device_buffer(x0.size() * sizeof(float));
  auto dy_direct = device_buffer(x0.size() * sizeof(float));
  copy_h2d(dx, x0);
  copy_h2d(dw, weight);
  copy_h2d(ddelta, delta);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
  require_status(glmrt_cuda_rmsnorm_f32_async(static_cast<float*>(dx.ptr),
                                             static_cast<float*>(dw.ptr),
                                             static_cast<float*>(dy_graph.ptr), rows, hidden, eps,
                                             stream),
                 "warmup glmrt_cuda_rmsnorm_f32_async");
  require_status(glmrt_cuda_residual_add_f32_async(static_cast<float*>(dy_graph.ptr),
                                                  static_cast<float*>(ddelta.ptr),
                                                  static_cast<float*>(dy_graph.ptr), x0.size(),
                                                  stream),
                 "warmup glmrt_cuda_residual_add_f32_async");
  require_cuda(cudaStreamSynchronize(stream), "warmup cudaStreamSynchronize");
  require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture");
  require_status(glmrt_cuda_rmsnorm_f32_async(static_cast<float*>(dx.ptr),
                                             static_cast<float*>(dw.ptr),
                                             static_cast<float*>(dy_graph.ptr), rows, hidden, eps,
                                             stream),
                 "capture glmrt_cuda_rmsnorm_f32_async");
  require_status(glmrt_cuda_residual_add_f32_async(static_cast<float*>(dy_graph.ptr),
                                                  static_cast<float*>(ddelta.ptr),
                                                  static_cast<float*>(dy_graph.ptr), x0.size(),
                                                  stream),
                 "capture glmrt_cuda_residual_add_f32_async");
  require_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  require_cuda(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");

  require_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch x0");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize x0");
  assert_close(copy_d2h<float>(dy_graph, x0.size()),
               cpu_residual_add(cpu_rmsnorm(x0, weight, rows, hidden, eps), delta));

  copy_h2d(dx, x1);
  require_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch x1");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize x1");
  const auto graph_out = copy_d2h<float>(dy_graph, x1.size());

  require_status(glmrt_cuda_rmsnorm_f32(static_cast<float*>(dx.ptr), static_cast<float*>(dw.ptr),
                                       static_cast<float*>(dy_direct.ptr), rows, hidden, eps),
                 "direct glmrt_cuda_rmsnorm_f32");
  require_status(glmrt_cuda_residual_add_f32(static_cast<float*>(dy_direct.ptr),
                                            static_cast<float*>(ddelta.ptr),
                                            static_cast<float*>(dy_direct.ptr), x1.size()),
                 "direct glmrt_cuda_residual_add_f32");
  assert_close(graph_out, copy_d2h<float>(dy_direct, x1.size()));

  require_cuda(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
  require_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
  require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  free_buffer(&dx);
  free_buffer(&dw);
  free_buffer(&ddelta);
  free_buffer(&dy_graph);
  free_buffer(&dy_direct);
}

void test_cuda_graph_bf16_replay_matches_uncaptured() {
  const int rows = 1;
  const int hidden = 8;
  const float eps = 1.0e-5f;
  const std::vector<uint16_t> x0 =
      bf16_values({0.1f, 0.2f, 0.3f, 0.4f, -0.5f, -0.4f, -0.3f, -0.2f});
  const std::vector<uint16_t> x1 =
      bf16_values({0.8f, 0.7f, -0.6f, -0.5f, 0.4f, 0.3f, -0.2f, 0.1f});
  const std::vector<uint16_t> weight = bf16_values(std::vector<float>(hidden, 1.25f));
  const std::vector<uint16_t> delta =
      bf16_values({0.05f, -0.10f, 0.15f, -0.20f, 0.25f, -0.30f, 0.35f, -0.40f});

  auto dx = device_buffer(x0.size() * sizeof(uint16_t));
  auto dw = device_buffer(weight.size() * sizeof(uint16_t));
  auto ddelta = device_buffer(delta.size() * sizeof(uint16_t));
  auto dy_graph = device_buffer(x0.size() * sizeof(uint16_t));
  auto dy_direct = device_buffer(x0.size() * sizeof(uint16_t));
  copy_h2d(dx, x0);
  copy_h2d(dw, weight);
  copy_h2d(ddelta, delta);

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate bf16 graph");
  require_status(glmrt_cuda_rmsnorm_bf16_async(static_cast<uint16_t*>(dx.ptr),
                                              static_cast<uint16_t*>(dw.ptr),
                                              static_cast<uint16_t*>(dy_graph.ptr), rows, hidden,
                                              eps, stream),
                 "warmup glmrt_cuda_rmsnorm_bf16_async");
  require_status(glmrt_cuda_residual_add_bf16_async(static_cast<uint16_t*>(dy_graph.ptr),
                                                   static_cast<uint16_t*>(ddelta.ptr),
                                                   static_cast<uint16_t*>(dy_graph.ptr),
                                                   x0.size(), stream),
                 "warmup glmrt_cuda_residual_add_bf16_async");
  require_cuda(cudaStreamSynchronize(stream), "warmup bf16 cudaStreamSynchronize");
  require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture bf16");
  require_status(glmrt_cuda_rmsnorm_bf16_async(static_cast<uint16_t*>(dx.ptr),
                                              static_cast<uint16_t*>(dw.ptr),
                                              static_cast<uint16_t*>(dy_graph.ptr), rows, hidden,
                                              eps, stream),
                 "capture glmrt_cuda_rmsnorm_bf16_async");
  require_status(glmrt_cuda_residual_add_bf16_async(static_cast<uint16_t*>(dy_graph.ptr),
                                                   static_cast<uint16_t*>(ddelta.ptr),
                                                   static_cast<uint16_t*>(dy_graph.ptr),
                                                   x0.size(), stream),
                 "capture glmrt_cuda_residual_add_bf16_async");
  require_cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture bf16");
  require_cuda(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate bf16");

  require_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch bf16 x0");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize bf16 x0");
  const auto x0_rmsnorm =
      cpu_rmsnorm(bf16_to_f32_values(x0), bf16_to_f32_values(weight), rows, hidden, eps);
  assert_close(bf16_to_f32_values(copy_d2h<uint16_t>(dy_graph, x0.size())),
               cpu_residual_add_bf16(bf16_values(x0_rmsnorm), delta));

  copy_h2d(dx, x1);
  require_cuda(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch bf16 x1");
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize bf16 x1");
  const auto graph_out = bf16_to_f32_values(copy_d2h<uint16_t>(dy_graph, x1.size()));

  require_status(glmrt_cuda_rmsnorm_bf16(static_cast<uint16_t*>(dx.ptr),
                                        static_cast<uint16_t*>(dw.ptr),
                                        static_cast<uint16_t*>(dy_direct.ptr), rows, hidden, eps),
                 "direct glmrt_cuda_rmsnorm_bf16");
  require_status(glmrt_cuda_residual_add_bf16(static_cast<uint16_t*>(dy_direct.ptr),
                                             static_cast<uint16_t*>(ddelta.ptr),
                                             static_cast<uint16_t*>(dy_direct.ptr), x1.size()),
                 "direct glmrt_cuda_residual_add_bf16");
  assert_close(graph_out, bf16_to_f32_values(copy_d2h<uint16_t>(dy_direct, x1.size())));

  require_cuda(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy bf16");
  require_cuda(cudaGraphDestroy(graph), "cudaGraphDestroy bf16");
  require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy bf16");
  free_buffer(&dx);
  free_buffer(&dw);
  free_buffer(&ddelta);
  free_buffer(&dy_graph);
  free_buffer(&dy_direct);
}

void test_bf16_weight_nvfp4_quantization_matches_representable_block() {
  constexpr size_t rows = 1;
  constexpr size_t cols = 16;
  const std::vector<float> values = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  std::vector<uint16_t> input(values.size());
  std::transform(values.begin(), values.end(), input.begin(), f32_to_bf16);
  auto input_device = device_buffer(input.size() * sizeof(uint16_t));
  auto packed_device = device_buffer(values.size() / 2);
  auto scale_device = device_buffer(values.size() / 16);
  copy_h2d(input_device, input);
  require_status(glmrt_cuda_quantize_bf16_weight_nvfp4_async(
                     input_device, packed_device, scale_device, rows, cols,
                     448.0f, nullptr),
                 "glmrt_cuda_quantize_bf16_weight_nvfp4_async");
  require_cuda(cudaStreamSynchronize(nullptr),
               "cudaStreamSynchronize BF16 weight NVFP4 quantization");
  const auto packed = copy_d2h<uint8_t>(packed_device, values.size() / 2);
  const auto scales = copy_d2h<uint8_t>(scale_device, values.size() / 16);
  const std::vector<uint8_t> expected = {
      0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
  };
  assert(packed == expected);
  assert(scales == std::vector<uint8_t>{0x7e});
  free_buffer(&input_device);
  free_buffer(&packed_device);
  free_buffer(&scale_device);
}

}  // namespace

int main() {
  test_cuda_device_info();
  test_cuda_copy_d2d_2d_async_copies_active_row_prefixes();
  test_cuda_mla_merge_state_bf16_matches_weighted_reference();
  test_cuda_rmsnorm_matches_ref();
  test_cuda_rmsnorm_bf16_matches_ref();
  test_cuda_mlp_matches_ref_small();
  test_cuda_mlp_rows_matches_ref_small();
  test_bf16_weight_nvfp4_quantization_matches_representable_block();
  test_cuda_nvfp4_route_bf16_staged_reduces_wide_dims();
  test_cuda_residual_add_matches_ref();
  test_cuda_residual_add_bf16_matches_ref();
  test_cuda_residual_add_f32_delta_bf16_matches_ref();
  test_cuda_residual_add_shared_f32_delta_bf16_matches_ref();
  test_cuda_row_gather_scatter_add_matches_ref();
  test_cuda_row_gather_scatter_add_bf16_matches_ref();
  test_cuda_router_topk_matches_ref();
  test_cuda_router_topk_bf16_matches_ref();
  test_cuda_linear_matches_ref();
  test_cuda_linear_bf16_matches_ref();
  test_cuda_linear_bf16_m1_parity_batched_matches_recurrent_m1();
  test_cuda_glm_dsa_sort_selected_indices_orders_each_row();
  test_cuda_w8a16_parity_batched_matches_recurrent_m1();
  test_cuda_w8a16_packed_parity_batched_matches_recurrent_m1();
  test_cuda_matmul_bf16_strided_batched_matches_ref();
  test_cuda_causal_attention_matches_ref();
  test_cuda_causal_attention_bf16_matches_ref();
  test_cuda_rope_matches_ref();
  test_cuda_rope_bf16_matches_ref();
  test_cuda_mla_rope_attention_bf16_matches_ref();
  test_cuda_mla_rope_attention_bf16_suffix_graph_captures_glm52_shape();
  test_cuda_embedding_lookup_matches_ref();
  test_cuda_embedding_lookup_bf16_matches_ref();
  test_cuda_logits_argmax_matches_ref();
  test_cuda_lm_head_argmax_bf16_matches_ref();
  test_cuda_lm_head_sample_topk_topp_bf16_matches_ref();
  test_cuda_logits_sample_topk_topp_matches_ref();
  test_cuda_mla_kv_prepare_bf16_matches_normalized_rotated_reference();
  test_cuda_mla_compressed_attention_reads_interleaved_cache_formats();
  test_cuda_mla_kv_mxfp4_pack_roundtrip_matches_representable_values();
  test_cuda_graph_replay_matches_uncaptured();
  test_cuda_graph_bf16_replay_matches_uncaptured();
  test_nvfp4_pack_unpack_or_skip_with_reason();
  std::cout << "glmrt_cuda_selftest passed\n";
  return 0;
}
