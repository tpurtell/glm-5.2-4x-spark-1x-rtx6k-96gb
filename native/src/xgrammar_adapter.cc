#include "glmrt_native.h"

#include <dlpack/dlpack.h>
#include <picojson.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void CopyError(const std::string& message, char* error, size_t error_bytes) {
  if (error == nullptr || error_bytes == 0) {
    return;
  }
  const size_t bytes = std::min(message.size(), error_bytes - 1);
  std::memcpy(error, message.data(), bytes);
  error[bytes] = '\0';
}

std::string ReadFile(const char* path) {
  if (path == nullptr || path[0] == '\0') {
    throw std::invalid_argument("tokenizer JSON path is empty");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument(std::string("cannot open tokenizer JSON ") + path);
  }
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

picojson::value ParseJSON(const std::string& text, const char* label) {
  picojson::value value;
  const std::string parse_error = picojson::parse(value, text);
  if (!parse_error.empty()) {
    throw std::invalid_argument(std::string("cannot parse ") + label + ": " + parse_error);
  }
  return value;
}

const picojson::object& RequireObject(const picojson::value& value, const char* label) {
  if (!value.is<picojson::object>()) {
    throw std::invalid_argument(std::string(label) + " must be an object");
  }
  return value.get<picojson::object>();
}

const picojson::array& RequireArray(const picojson::value& value, const char* label) {
  if (!value.is<picojson::array>()) {
    throw std::invalid_argument(std::string(label) + " must be an array");
  }
  return value.get<picojson::array>();
}

const picojson::value& RequireField(
    const picojson::object& object, const char* key, const char* label) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::invalid_argument(std::string(label) + " is missing " + key);
  }
  return found->second;
}

int RequireInteger(const picojson::value& value, const char* label) {
  int64_t integer = 0;
  if (value.is<int64_t>()) {
    integer = value.get<int64_t>();
  } else if (value.is<double>()) {
    const double number = value.get<double>();
    integer = static_cast<int64_t>(number);
    if (static_cast<double>(integer) != number) {
      throw std::invalid_argument(std::string(label) + " must be an integer");
    }
  } else {
    throw std::invalid_argument(std::string(label) + " must be an integer");
  }
  if (integer < 0 || integer > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(label) + " is outside the supported range");
  }
  return static_cast<int>(integer);
}

xgrammar::TokenizerInfo LoadTokenizer(
    const char* tokenizer_json_path, size_t vocab_size, const int32_t* stop_token_ids,
    size_t stop_token_count) {
  if (vocab_size == 0 || vocab_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("XGrammar vocabulary size is outside the supported range");
  }
  if (stop_token_count > 0 && stop_token_ids == nullptr) {
    throw std::invalid_argument("XGrammar stop-token array is null");
  }
  const std::string tokenizer_json = ReadFile(tokenizer_json_path);
  const picojson::value root_value = ParseJSON(tokenizer_json, "tokenizer JSON");
  const auto& root = RequireObject(root_value, "tokenizer JSON");
  const auto& model =
      RequireObject(RequireField(root, "model", "tokenizer JSON"), "tokenizer JSON.model");
  const auto& model_vocab = RequireObject(
      RequireField(model, "vocab", "tokenizer JSON.model"), "tokenizer JSON.model.vocab");
  std::vector<std::string> vocab(vocab_size);
  for (const auto& [token, id_value] : model_vocab) {
    const int id = RequireInteger(id_value, "tokenizer vocabulary ID");
    if (static_cast<size_t>(id) >= vocab_size) {
      throw std::invalid_argument("tokenizer vocabulary ID exceeds model vocabulary size");
    }
    vocab[static_cast<size_t>(id)] = token;
  }
  const auto added = root.find("added_tokens");
  if (added != root.end()) {
    for (const auto& added_value : RequireArray(added->second, "tokenizer JSON.added_tokens")) {
      const auto& token = RequireObject(added_value, "tokenizer JSON.added_tokens[]");
      const int id = RequireInteger(
          RequireField(token, "id", "tokenizer JSON.added_tokens[]"), "added-token ID");
      if (static_cast<size_t>(id) >= vocab_size) {
        throw std::invalid_argument("added-token ID exceeds model vocabulary size");
      }
      const auto& content = RequireField(token, "content", "tokenizer JSON.added_tokens[]");
      if (!content.is<std::string>()) {
        throw std::invalid_argument("added-token content must be a string");
      }
      vocab[static_cast<size_t>(id)] = content.get<std::string>();
    }
  }

  const picojson::value metadata_value = ParseJSON(
      xgrammar::TokenizerInfo::DetectMetadataFromHF(tokenizer_json), "tokenizer metadata");
  const auto& metadata = RequireObject(metadata_value, "tokenizer metadata");
  const int vocab_type = RequireInteger(
      RequireField(metadata, "vocab_type", "tokenizer metadata"), "tokenizer vocab_type");
  const auto& add_prefix_space =
      RequireField(metadata, "add_prefix_space", "tokenizer metadata");
  if (!add_prefix_space.is<bool>()) {
    throw std::invalid_argument("tokenizer add_prefix_space metadata must be boolean");
  }
  std::vector<int32_t> stops;
  if (stop_token_count > 0) {
    stops.assign(stop_token_ids, stop_token_ids + stop_token_count);
  }
  return xgrammar::TokenizerInfo(
      vocab, static_cast<xgrammar::VocabType>(vocab_type), static_cast<int>(vocab_size),
      std::move(stops), add_prefix_space.get<bool>());
}

struct CompilerHandle {
  explicit CompilerHandle(xgrammar::TokenizerInfo tokenizer_info)
      // glmrt owns a bounded compiled-grammar cache. Keeping XGrammar's
      // independent cache enabled would retain evicted client schemas and
      // make unique-schema traffic grow memory without a bound.
      : tokenizer(std::move(tokenizer_info)), compiler(tokenizer, 4, false) {}

  xgrammar::TokenizerInfo tokenizer;
  xgrammar::GrammarCompiler compiler;
  std::mutex mutex;
};

struct GrammarHandle {
  explicit GrammarHandle(xgrammar::CompiledGrammar compiled) : grammar(std::move(compiled)) {}
  xgrammar::CompiledGrammar grammar;
};

struct MatcherHandle {
  MatcherHandle(xgrammar::GrammarMatcher value, int vocab_size)
      : matcher(std::move(value)), vocab_size(vocab_size) {}
  xgrammar::GrammarMatcher matcher;
  int vocab_size;
};

template <typename Function>
glmrt_status_t Guard(char* error, size_t error_bytes, glmrt_status_t exception_status,
                     Function&& function) {
  try {
    function();
    CopyError("", error, error_bytes);
    return GLMRT_STATUS_OK;
  } catch (const std::invalid_argument& exception) {
    CopyError(exception.what(), error, error_bytes);
    return GLMRT_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& exception) {
    CopyError(exception.what(), error, error_bytes);
    return exception_status;
  } catch (...) {
    CopyError("unknown native XGrammar failure", error, error_bytes);
    return exception_status;
  }
}

}  // namespace

extern "C" glmrt_status_t glmrt_xgrammar_compiler_create(
    const char* tokenizer_json_path, size_t vocab_size, const int32_t* stop_token_ids,
    size_t stop_token_count, void** out_compiler, char* error, size_t error_bytes) {
  if (out_compiler == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_compiler = nullptr;
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    auto compiler = std::make_unique<CompilerHandle>(
        LoadTokenizer(tokenizer_json_path, vocab_size, stop_token_ids, stop_token_count));
    *out_compiler = compiler.release();
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_compiler_destroy(void* compiler) {
  delete static_cast<CompilerHandle*>(compiler);
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_xgrammar_compile(
    void* compiler, glmrt_xgrammar_kind_t kind, const char* grammar_json, int strict,
    void** out_grammar, char* error, size_t error_bytes) {
  if (compiler == nullptr || out_grammar == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_grammar = nullptr;
  return Guard(error, error_bytes, GLMRT_STATUS_INVALID_ARGUMENT, [&] {
    auto* handle = static_cast<CompilerHandle*>(compiler);
    std::lock_guard<std::mutex> lock(handle->mutex);
    xgrammar::CompiledGrammar compiled = [&] {
      switch (kind) {
        case GLMRT_XGRAMMAR_JSON_OBJECT:
          return handle->compiler.CompileJSONSchema(R"({"type":"object"})");
        case GLMRT_XGRAMMAR_JSON_SCHEMA:
          if (grammar_json == nullptr) {
            throw std::invalid_argument("JSON Schema text is null");
          }
          return handle->compiler.CompileJSONSchema(
              grammar_json, true, std::nullopt, std::nullopt, strict != 0);
        case GLMRT_XGRAMMAR_STRUCTURAL_TAG:
          if (grammar_json == nullptr) {
            throw std::invalid_argument("structural-tag JSON text is null");
          }
          return handle->compiler.CompileStructuralTag(grammar_json);
      }
      throw std::invalid_argument("unknown XGrammar grammar kind");
    }();
    *out_grammar = new GrammarHandle(std::move(compiled));
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_grammar_destroy(void* grammar) {
  delete static_cast<GrammarHandle*>(grammar);
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_create(
    const void* grammar, void** out_matcher, char* error, size_t error_bytes) {
  if (grammar == nullptr || out_matcher == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_matcher = nullptr;
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    const auto* handle = static_cast<const GrammarHandle*>(grammar);
    *out_matcher = new MatcherHandle(
        xgrammar::GrammarMatcher(handle->grammar),
        handle->grammar.GetTokenizerInfo().GetVocabSize());
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_fork(
    const void* matcher, void** out_matcher, char* error, size_t error_bytes) {
  if (matcher == nullptr || out_matcher == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  *out_matcher = nullptr;
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    const auto* handle = static_cast<const MatcherHandle*>(matcher);
    *out_matcher = new MatcherHandle(handle->matcher.Fork(), handle->vocab_size);
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_destroy(void* matcher) {
  delete static_cast<MatcherHandle*>(matcher);
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_fill_bitmask(
    void* matcher, uint32_t* bitmask, size_t bitmask_words, int* out_needs_mask,
    char* error, size_t error_bytes) {
  if (matcher == nullptr || bitmask == nullptr || bitmask_words == 0 ||
      out_needs_mask == nullptr ||
      bitmask_words > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    auto* handle = static_cast<MatcherHandle*>(matcher);
    const size_t expected_words =
        static_cast<size_t>(xgrammar::GetBitmaskSize(handle->vocab_size));
    if (bitmask_words != expected_words) {
      throw std::invalid_argument(
          "XGrammar bitmask word count does not match the tokenizer vocabulary");
    }
    int64_t shape = static_cast<int64_t>(bitmask_words);
    DLTensor tensor{};
    tensor.data = bitmask;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = &shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    *out_needs_mask = handle->matcher.FillNextTokenBitmask(&tensor) ? 1 : 0;
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_accept_token(
    void* matcher, uint32_t token_id, int* out_accepted, char* error, size_t error_bytes) {
  if (matcher == nullptr || out_accepted == nullptr ||
      token_id > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    auto* handle = static_cast<MatcherHandle*>(matcher);
    *out_accepted = handle->matcher.AcceptToken(static_cast<int32_t>(token_id)) ? 1 : 0;
  });
}

extern "C" glmrt_status_t glmrt_xgrammar_matcher_is_completed(
    const void* matcher, int* out_completed, char* error, size_t error_bytes) {
  if (matcher == nullptr || out_completed == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return Guard(error, error_bytes, GLMRT_STATUS_INTERNAL_ERROR, [&] {
    const auto* handle = static_cast<const MatcherHandle*>(matcher);
    *out_completed = handle->matcher.IsCompleted() ? 1 : 0;
  });
}
