#include <dlpack/dlpack.h>
#include <picojson.h>
#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string tokenizer_path;
  std::string fixture_path;
  std::string xgrammar_commit = "unknown";
  int warmup = 2;
  int iterations = 100;
  int mtp_width = 6;
};

struct Sample {
  std::string name;
  std::string text;
  std::vector<int32_t> token_ids;
};

struct Fixture {
  std::string name;
  int model_vocab_size = 0;
  std::vector<int32_t> stop_token_ids;
  std::string structural_tag_json;
  std::vector<Sample> samples;
};

struct CpuI32Tensor {
  explicit CpuI32Tensor(int64_t rows, int64_t columns)
      : data(static_cast<size_t>(rows * columns)), shape{rows, columns} {
    tensor.data = data.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = rows == 1 ? 1 : 2;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = rows == 1 ? &shape[1] : shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
  }

  std::vector<int32_t> data;
  int64_t shape[2];
  DLTensor tensor{};
};

struct CpuI64Tensor {
  explicit CpuI64Tensor(std::vector<int64_t> values)
      : data(std::move(values)), shape{static_cast<int64_t>(data.size())} {
    tensor.data = data.data();
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 64, 1};
    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
  }

  std::vector<int64_t> data;
  int64_t shape[1];
  DLTensor tensor{};
};

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
}

std::string ReadFile(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    Fail("cannot open " + path);
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

picojson::value ParseJSON(const std::string &text, const std::string &label) {
  picojson::value value;
  const std::string error = picojson::parse(value, text);
  if (!error.empty()) {
    Fail("cannot parse " + label + ": " + error);
  }
  return value;
}

const picojson::object &RequireObject(const picojson::value &value,
                                      std::string_view label) {
  if (!value.is<picojson::object>()) {
    Fail(std::string(label) + " must be an object");
  }
  return value.get<picojson::object>();
}

const picojson::array &RequireArray(const picojson::value &value,
                                    std::string_view label) {
  if (!value.is<picojson::array>()) {
    Fail(std::string(label) + " must be an array");
  }
  return value.get<picojson::array>();
}

const picojson::value &RequireField(const picojson::object &object,
                                    std::string_view key,
                                    std::string_view label) {
  const auto it = object.find(std::string(key));
  if (it == object.end()) {
    Fail(std::string(label) + " is missing " + std::string(key));
  }
  return it->second;
}

std::string RequireString(const picojson::value &value,
                          std::string_view label) {
  if (!value.is<std::string>()) {
    Fail(std::string(label) + " must be a string");
  }
  return value.get<std::string>();
}

int64_t RequireInteger(const picojson::value &value, std::string_view label) {
  if (value.is<int64_t>()) {
    return value.get<int64_t>();
  }
  if (value.is<double>()) {
    const double number = value.get<double>();
    const int64_t integer = static_cast<int64_t>(number);
    if (static_cast<double>(integer) == number) {
      return integer;
    }
  }
  Fail(std::string(label) + " must be an integer");
}

int CheckedInt(int64_t value, std::string_view label) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    Fail(std::string(label) + " is out of range");
  }
  return static_cast<int>(value);
}

std::vector<int32_t> ParseIds(const picojson::value &value,
                              std::string_view label) {
  std::vector<int32_t> ids;
  for (const auto &item : RequireArray(value, label)) {
    ids.push_back(
        static_cast<int32_t>(CheckedInt(RequireInteger(item, label), label)));
  }
  return ids;
}

Fixture LoadFixture(const std::string &path) {
  const picojson::value root_value = ParseJSON(ReadFile(path), path);
  const picojson::object &root = RequireObject(root_value, path);
  Fixture fixture;
  fixture.name =
      RequireString(RequireField(root, "name", path), path + ".name");
  fixture.model_vocab_size =
      CheckedInt(RequireInteger(RequireField(root, "model_vocab_size", path),
                                path + ".model_vocab_size"),
                 path + ".model_vocab_size");
  fixture.stop_token_ids = ParseIds(RequireField(root, "stop_token_ids", path),
                                    path + ".stop_token_ids");
  fixture.structural_tag_json =
      RequireField(root, "structural_tag", path).serialize(false);
  for (const auto &sample_value :
       RequireArray(RequireField(root, "samples", path), path + ".samples")) {
    const auto &sample_object =
        RequireObject(sample_value, path + ".samples[]");
    Sample sample;
    sample.name =
        RequireString(RequireField(sample_object, "name", path + ".samples[]"),
                      path + ".samples[].name");
    sample.text =
        RequireString(RequireField(sample_object, "text", path + ".samples[]"),
                      path + ".samples[].text");
    sample.token_ids =
        ParseIds(RequireField(sample_object, "token_ids", path + ".samples[]"),
                 path + ".samples[].token_ids");
    if (sample.token_ids.empty()) {
      Fail("fixture sample " + sample.name + " has no token ids");
    }
    fixture.samples.push_back(std::move(sample));
  }
  if (fixture.samples.empty()) {
    Fail("fixture has no samples");
  }
  return fixture;
}

struct TokenizerLoadResult {
  xgrammar::TokenizerInfo tokenizer_info;
  double elapsed_ms;
};

TokenizerLoadResult LoadTokenizer(const Options &options,
                                  const Fixture &fixture) {
  const auto started = Clock::now();
  const std::string tokenizer_json = ReadFile(options.tokenizer_path);
  const picojson::value root_value =
      ParseJSON(tokenizer_json, options.tokenizer_path);
  const picojson::object &root =
      RequireObject(root_value, options.tokenizer_path);
  std::vector<std::string> vocab(static_cast<size_t>(fixture.model_vocab_size));

  const auto &model =
      RequireObject(RequireField(root, "model", options.tokenizer_path),
                    options.tokenizer_path + ".model");
  const auto &model_vocab = RequireObject(
      RequireField(model, "vocab", options.tokenizer_path + ".model"),
      options.tokenizer_path + ".model.vocab");
  for (const auto &[token, id_value] : model_vocab) {
    const int id = CheckedInt(RequireInteger(id_value, "model vocab id"),
                              "model vocab id");
    if (id >= fixture.model_vocab_size) {
      Fail("model vocab id exceeds model_vocab_size");
    }
    vocab[static_cast<size_t>(id)] = token;
  }

  const auto added_it = root.find("added_tokens");
  if (added_it != root.end()) {
    for (const auto &added_value :
         RequireArray(added_it->second, "added_tokens")) {
      const auto &added = RequireObject(added_value, "added_tokens[]");
      const int id =
          CheckedInt(RequireInteger(RequireField(added, "id", "added_tokens[]"),
                                    "added token id"),
                     "added token id");
      if (id >= fixture.model_vocab_size) {
        Fail("added token id exceeds model_vocab_size");
      }
      vocab[static_cast<size_t>(id)] =
          RequireString(RequireField(added, "content", "added_tokens[]"),
                        "added token content");
    }
  }

  const picojson::value metadata_value =
      ParseJSON(xgrammar::TokenizerInfo::DetectMetadataFromHF(tokenizer_json),
                "HF metadata");
  const auto &metadata = RequireObject(metadata_value, "HF metadata");
  const int vocab_type = CheckedInt(
      RequireInteger(RequireField(metadata, "vocab_type", "HF metadata"),
                     "vocab_type"),
      "vocab_type");
  const auto add_prefix_it = metadata.find("add_prefix_space");
  if (add_prefix_it == metadata.end() || !add_prefix_it->second.is<bool>()) {
    Fail("HF metadata add_prefix_space must be a bool");
  }

  xgrammar::TokenizerInfo tokenizer_info(
      vocab, static_cast<xgrammar::VocabType>(vocab_type),
      fixture.model_vocab_size, fixture.stop_token_ids,
      add_prefix_it->second.get<bool>());
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  return {std::move(tokenizer_info), elapsed_ms};
}

bool IsAllowed(const CpuI32Tensor &mask, int32_t token_id, int row = 0) {
  const int64_t columns = mask.shape[1];
  const size_t word = static_cast<size_t>(row * columns + token_id / 32);
  const uint32_t bit = uint32_t{1} << static_cast<uint32_t>(token_id % 32);
  return (static_cast<uint32_t>(mask.data.at(word)) & bit) != 0;
}

uint64_t CountAllowed(const CpuI32Tensor &mask, int vocab_size, int row = 0) {
  const int64_t columns = mask.shape[1];
  uint64_t count = 0;
  for (int64_t word = 0; word < columns; ++word) {
    uint32_t bits = static_cast<uint32_t>(
        mask.data.at(static_cast<size_t>(row * columns + word)));
    if (word + 1 == columns && vocab_size % 32 != 0) {
      bits &= (uint32_t{1} << static_cast<uint32_t>(vocab_size % 32)) - 1;
    }
    count += static_cast<uint64_t>(__builtin_popcount(bits));
  }
  return count;
}

size_t ExtractAllowedIds(const CpuI32Tensor &mask, int vocab_size,
                         volatile int32_t *output, int row = 0) {
  const int64_t columns = mask.shape[1];
  const int32_t *row_data =
      mask.data.data() + static_cast<size_t>(row * columns);
  size_t count = 0;
  for (int64_t word = 0; word < columns; ++word) {
    uint32_t bits = static_cast<uint32_t>(row_data[word]);
    if (word + 1 == columns && vocab_size % 32 != 0) {
      bits &= (uint32_t{1} << static_cast<uint32_t>(vocab_size % 32)) - 1;
    }
    while (bits != 0) {
      const uint32_t bit = static_cast<uint32_t>(__builtin_ctz(bits));
      output[count++] = static_cast<int32_t>(word * 32 + bit);
      bits &= bits - 1;
    }
  }
  return count;
}

std::vector<xgrammar::GrammarMatcher>
ValidateSample(const xgrammar::CompiledGrammar &grammar,
               const xgrammar::TokenizerInfo &tokenizer_info,
               const Sample &sample, uint64_t *allowed_total) {
  std::string decoded;
  const auto &vocab = tokenizer_info.GetDecodedVocab();
  for (const int32_t id : sample.token_ids) {
    if (id < 0 || id >= tokenizer_info.GetVocabSize()) {
      Fail("sample " + sample.name + " contains an out-of-range token id");
    }
    decoded += vocab.at(static_cast<size_t>(id));
  }
  if (decoded != sample.text) {
    Fail("sample " + sample.name + " token ids do not decode to fixture text");
  }

  const int mask_words =
      xgrammar::GetBitmaskSize(tokenizer_info.GetVocabSize());
  CpuI32Tensor mask(1, mask_words);
  xgrammar::GrammarMatcher matcher(grammar, std::nullopt, true, 64);
  std::vector<xgrammar::GrammarMatcher> states;
  states.reserve(sample.token_ids.size());
  for (const int32_t id : sample.token_ids) {
    states.push_back(matcher.Fork());
    matcher.FillNextTokenBitmask(&mask.tensor);
    if (!IsAllowed(mask, id)) {
      Fail("grammar rejects token " + std::to_string(id) + " in sample " +
           sample.name);
    }
    *allowed_total += CountAllowed(mask, tokenizer_info.GetVocabSize());
    if (!matcher.AcceptToken(id)) {
      Fail("matcher failed to accept validated token in sample " + sample.name);
    }
  }
  if (!matcher.IsCompleted()) {
    Fail("sample " + sample.name + " did not complete the grammar");
  }
  return states;
}

template <typename Function>
double MeasureUs(int warmup, int iterations, Function &&function) {
  for (int i = 0; i < warmup; ++i) {
    function();
  }
  const auto started = Clock::now();
  for (int i = 0; i < iterations; ++i) {
    function();
  }
  return std::chrono::duration<double, std::micro>(Clock::now() - started)
             .count() /
         static_cast<double>(iterations);
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) * 0.5;
  }
  return values[middle];
}

struct ExtractionDensityResult {
  int allowed_tokens;
  double elapsed_us;
};

std::vector<ExtractionDensityResult>
BenchmarkExtractionDensity(int vocab_size, int warmup, int iterations) {
  const std::vector<int> counts = {8,     256,   1024,  4096,
                                   8192,  16384, 24576, 38720,
                                   77440, 116160, 154880};
  CpuI32Tensor masks(static_cast<int64_t>(counts.size()),
                     xgrammar::GetBitmaskSize(vocab_size));
  for (size_t row = 0; row < counts.size(); ++row) {
    for (int64_t index = 0; index < counts[row]; ++index) {
      const int token = static_cast<int>(
          (index * int64_t{7919} + static_cast<int64_t>(row) * 1009 + 17) %
          vocab_size);
      const size_t mask_index = static_cast<size_t>(
          static_cast<int64_t>(row) * masks.shape[1] + token / 32);
      const uint32_t word = static_cast<uint32_t>(masks.data[mask_index]);
      masks.data[mask_index] = static_cast<int32_t>(
          word | (uint32_t{1} << static_cast<uint32_t>(token % 32)));
    }
  }

  std::vector<int32_t> extracted_ids(static_cast<size_t>(vocab_size));
  std::vector<ExtractionDensityResult> results;
  results.reserve(counts.size());
  for (size_t row = 0; row < counts.size(); ++row) {
    const size_t extracted =
        ExtractAllowedIds(masks, vocab_size, extracted_ids.data(),
                          static_cast<int>(row));
    if (extracted != static_cast<size_t>(counts[row])) {
      Fail("synthetic allowed-ID extraction count differs from packed mask");
    }
    const double elapsed = MeasureUs(warmup, iterations, [&] {
      ExtractAllowedIds(masks, vocab_size, extracted_ids.data(),
                        static_cast<int>(row));
    });
    results.push_back({counts[row], elapsed});
  }
  return results;
}

double BenchmarkBatchFill(const xgrammar::GrammarMatcher &state,
                          int concurrency, int vocab_size, int warmup,
                          int iterations) {
  std::vector<xgrammar::GrammarMatcher> matchers;
  matchers.reserve(static_cast<size_t>(concurrency));
  for (int i = 0; i < concurrency; ++i) {
    matchers.push_back(state.Fork());
  }
  CpuI32Tensor masks(concurrency, xgrammar::GetBitmaskSize(vocab_size));
  xgrammar::BatchGrammarMatcher batch_matcher(
      std::variant<std::string, int32_t>(
          static_cast<int32_t>(std::min(concurrency, 4))));
  return MeasureUs(warmup, iterations, [&] {
    batch_matcher.BatchFillNextTokenBitmask(&matchers, &masks.tensor);
  });
}

double BenchmarkScalarFill(const xgrammar::GrammarMatcher &state,
                           int concurrency, int vocab_size, int warmup,
                           int iterations) {
  std::vector<xgrammar::GrammarMatcher> matchers;
  matchers.reserve(static_cast<size_t>(concurrency));
  for (int i = 0; i < concurrency; ++i) {
    matchers.push_back(state.Fork());
  }
  CpuI32Tensor masks(concurrency, xgrammar::GetBitmaskSize(vocab_size));
  return MeasureUs(warmup, iterations, [&] {
    for (int index = 0; index < concurrency; ++index) {
      matchers[static_cast<size_t>(index)].FillNextTokenBitmask(&masks.tensor,
                                                                index);
    }
  });
}

double BenchmarkDraftTraversal(const xgrammar::GrammarMatcher &state,
                               const std::vector<int32_t> &remaining_tokens,
                               int width, int vocab_size, int warmup,
                               int iterations) {
  width = std::min(width, static_cast<int>(remaining_tokens.size()));
  if (width < 2) {
    Fail("MTP traversal needs at least two remaining tokens");
  }
  std::vector<int64_t> next(static_cast<size_t>(width), -1);
  std::vector<int64_t> sibling(static_cast<size_t>(width), -1);
  std::vector<int64_t> draft(static_cast<size_t>(width), 0);
  for (int index = 0; index < width; ++index) {
    if (index + 1 < width) {
      next[static_cast<size_t>(index)] = index + 1;
    }
    draft[static_cast<size_t>(index)] =
        remaining_tokens[static_cast<size_t>(index)];
  }
  CpuI64Tensor next_tensor(std::move(next));
  CpuI64Tensor sibling_tensor(std::move(sibling));
  CpuI64Tensor draft_tensor(std::move(draft));
  CpuI32Tensor masks(width, xgrammar::GetBitmaskSize(vocab_size));
  return MeasureUs(warmup, iterations, [&] {
    auto branch = state.Fork();
    if (!branch.TraverseDraftTree(&next_tensor.tensor, &sibling_tensor.tensor,
                                  &draft_tensor.tensor, &masks.tensor)) {
      Fail("draft traversal timed out unexpectedly");
    }
  });
}

Options ParseOptions(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        Fail("missing value for " + argument);
      }
      return argv[index];
    };
    if (argument == "--tokenizer") {
      options.tokenizer_path = value();
    } else if (argument == "--fixture") {
      options.fixture_path = value();
    } else if (argument == "--xgrammar-commit") {
      options.xgrammar_commit = value();
    } else if (argument == "--warmup") {
      options.warmup = std::stoi(value());
    } else if (argument == "--iterations") {
      options.iterations = std::stoi(value());
    } else if (argument == "--mtp-width") {
      options.mtp_width = std::stoi(value());
    } else {
      Fail("unknown argument " + argument);
    }
  }
  if (options.tokenizer_path.empty() || options.fixture_path.empty()) {
    Fail("--tokenizer and --fixture are required");
  }
  if (options.warmup < 0 || options.iterations < 1 || options.mtp_width < 2) {
    Fail("warmup must be nonnegative; iterations and mtp-width must be "
         "positive");
  }
  return options;
}

picojson::value Number(double value) { return picojson::value(value); }

int Run(const Options &options) {
  const Fixture fixture = LoadFixture(options.fixture_path);
  TokenizerLoadResult tokenizer = LoadTokenizer(options, fixture);
  xgrammar::GrammarCompiler compiler(tokenizer.tokenizer_info, 4, true);

  const auto compile_started = Clock::now();
  const xgrammar::CompiledGrammar grammar =
      compiler.CompileStructuralTag(fixture.structural_tag_json);
  const double compile_cold_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - compile_started)
          .count();
  std::vector<double> cache_compile_us;
  for (int iteration = 0; iteration < 5; ++iteration) {
    const auto started = Clock::now();
    const auto cached =
        compiler.CompileStructuralTag(fixture.structural_tag_json);
    cache_compile_us.push_back(
        std::chrono::duration<double, std::micro>(Clock::now() - started)
            .count());
    if (cached.MemorySizeBytes() != grammar.MemorySizeBytes()) {
      Fail("cached grammar differs from cold grammar");
    }
  }

  uint64_t allowed_total = 0;
  size_t token_count = 0;
  std::vector<xgrammar::GrammarMatcher> benchmark_states;
  for (const auto &sample : fixture.samples) {
    auto states = ValidateSample(grammar, tokenizer.tokenizer_info, sample,
                                 &allowed_total);
    token_count += sample.token_ids.size();
    benchmark_states.insert(benchmark_states.end(),
                            std::make_move_iterator(states.begin()),
                            std::make_move_iterator(states.end()));
  }

  CpuI32Tensor scalar_mask(1,
                           xgrammar::GetBitmaskSize(fixture.model_vocab_size));
  const double scalar_fill_us =
      MeasureUs(options.warmup, options.iterations, [&] {
        for (auto &state : benchmark_states) {
          state.FillNextTokenBitmask(&scalar_mask.tensor);
        }
      });

  CpuI32Tensor extraction_masks(
      static_cast<int64_t>(benchmark_states.size()),
      xgrammar::GetBitmaskSize(fixture.model_vocab_size));
  for (size_t index = 0; index < benchmark_states.size(); ++index) {
    benchmark_states[index].FillNextTokenBitmask(
        &extraction_masks.tensor, static_cast<int>(index));
  }
  std::vector<int32_t> extracted_ids(
      static_cast<size_t>(fixture.model_vocab_size));
  uint64_t extraction_sink = 0;
  for (size_t index = 0; index < benchmark_states.size(); ++index) {
    const size_t count = ExtractAllowedIds(
        extraction_masks, fixture.model_vocab_size, extracted_ids.data(),
        static_cast<int>(index));
    if (count != CountAllowed(extraction_masks, fixture.model_vocab_size,
                              static_cast<int>(index))) {
      Fail("allowed-ID extraction count differs from packed mask");
    }
    for (size_t id_index = 1; id_index < count; ++id_index) {
      if (extracted_ids[id_index - 1] >= extracted_ids[id_index]) {
        Fail("allowed-ID extraction is not strictly sorted");
      }
    }
  }
  const double extract_sequence_us =
      MeasureUs(options.warmup, options.iterations, [&] {
        uint64_t iteration_total = 0;
        for (size_t index = 0; index < benchmark_states.size(); ++index) {
          iteration_total += ExtractAllowedIds(
              extraction_masks, fixture.model_vocab_size,
              extracted_ids.data(), static_cast<int>(index));
        }
        extraction_sink += iteration_total;
      });
  const double fill_extract_sequence_us =
      MeasureUs(options.warmup, options.iterations, [&] {
        uint64_t iteration_total = 0;
        for (auto &state : benchmark_states) {
          state.FillNextTokenBitmask(&scalar_mask.tensor);
          iteration_total += ExtractAllowedIds(
              scalar_mask, fixture.model_vocab_size, extracted_ids.data());
        }
        extraction_sink += iteration_total;
      });
  const auto extraction_density = BenchmarkExtractionDensity(
      fixture.model_vocab_size, options.warmup, options.iterations);

  const Sample &mtp_sample = fixture.samples.front();
  const int mtp_offset =
      std::min<int>(2, static_cast<int>(mtp_sample.token_ids.size()) - 2);
  xgrammar::GrammarMatcher mtp_state(grammar, std::nullopt, true, 64);
  for (int index = 0; index < mtp_offset; ++index) {
    if (!mtp_state.AcceptToken(
            mtp_sample.token_ids[static_cast<size_t>(index)])) {
      Fail("cannot construct MTP prefix state");
    }
  }
  const std::vector<int32_t> remaining(
      mtp_sample.token_ids.begin() + mtp_offset, mtp_sample.token_ids.end());
  const int effective_mtp_width =
      std::min(options.mtp_width, static_cast<int>(remaining.size()));
  const double draft_traverse_us = BenchmarkDraftTraversal(
      mtp_state, remaining, effective_mtp_width, fixture.model_vocab_size,
      options.warmup, options.iterations);

  const auto &batch_state = benchmark_states.at(benchmark_states.size() / 2);
  picojson::object batch_results;
  for (const int concurrency : {1, 2, 4}) {
    const double batch_elapsed =
        BenchmarkBatchFill(batch_state, concurrency, fixture.model_vocab_size,
                           options.warmup, options.iterations);
    const double scalar_elapsed =
        BenchmarkScalarFill(batch_state, concurrency, fixture.model_vocab_size,
                            options.warmup, options.iterations);
    picojson::object result;
    result["batch_total_us"] = Number(batch_elapsed);
    result["batch_us_per_request"] = Number(batch_elapsed / concurrency);
    result["scalar_loop_total_us"] = Number(scalar_elapsed);
    result["scalar_loop_us_per_request"] = Number(scalar_elapsed / concurrency);
    batch_results["c" + std::to_string(concurrency)] =
        picojson::value(std::move(result));
  }

  const double fork_us = MeasureUs(options.warmup, options.iterations, [&] {
    auto branch = batch_state.Fork();
    (void)branch.IsCompleted();
  });

  xgrammar::GrammarMatcher rollback_state(grammar, std::nullopt, true, 64);
  const int32_t rollback_token = fixture.samples.front().token_ids.front();
  const double accept_rollback_us =
      MeasureUs(options.warmup, options.iterations, [&] {
        if (!rollback_state.AcceptToken(rollback_token)) {
          Fail("rollback benchmark token rejected");
        }
        rollback_state.Rollback(1);
      });

  picojson::object output;
  output["benchmark"] = picojson::value(std::string("xgrammar_glm_native"));
  output["status"] = picojson::value(std::string("ok"));
  output["fixture"] = picojson::value(fixture.name);
  output["xgrammar_commit"] = picojson::value(options.xgrammar_commit);
  output["model_vocab_size"] =
      picojson::value(static_cast<int64_t>(fixture.model_vocab_size));
  output["sample_count"] =
      picojson::value(static_cast<int64_t>(fixture.samples.size()));
  output["sample_tokens"] = picojson::value(static_cast<int64_t>(token_count));
  output["tokenizer_load_ms"] = Number(tokenizer.elapsed_ms);
  output["compile_cold_ms"] = Number(compile_cold_ms);
  output["compile_cached_p50_us"] = Number(Median(cache_compile_us));
  output["compiled_bytes"] =
      picojson::value(static_cast<int64_t>(grammar.MemorySizeBytes()));
  output["mask_fill_sequence_us"] = Number(scalar_fill_us);
  output["mask_fill_us_per_token"] = Number(scalar_fill_us / token_count);
  output["mask_extract_sequence_us"] = Number(extract_sequence_us);
  output["mask_extract_us_per_token"] =
      Number(extract_sequence_us / token_count);
  output["mask_fill_extract_sequence_us"] =
      Number(fill_extract_sequence_us);
  output["mask_fill_extract_us_per_token"] =
      Number(fill_extract_sequence_us / token_count);
  output["mask_extract_checksum"] =
      picojson::value(static_cast<int64_t>(extraction_sink));
  picojson::array extraction_density_json;
  for (const auto &result : extraction_density) {
    picojson::object item;
    item["allowed_tokens"] =
        picojson::value(static_cast<int64_t>(result.allowed_tokens));
    item["extract_us"] = Number(result.elapsed_us);
    extraction_density_json.emplace_back(std::move(item));
  }
  output["mask_extract_density"] =
      picojson::value(std::move(extraction_density_json));
  output["mean_allowed_tokens"] = Number(static_cast<double>(allowed_total) /
                                         static_cast<double>(token_count));
  output["batch_mask_fill"] = picojson::value(std::move(batch_results));
  output["matcher_fork_us"] = Number(fork_us);
  output["accept_rollback_us"] = Number(accept_rollback_us);
  output["mtp_width"] =
      picojson::value(static_cast<int64_t>(effective_mtp_width));
  output["mtp_linear_traverse_us"] = Number(draft_traverse_us);
  output["iterations"] =
      picojson::value(static_cast<int64_t>(options.iterations));
  output["hardware_threads"] = picojson::value(
      static_cast<int64_t>(std::thread::hardware_concurrency()));
  std::cout << picojson::value(std::move(output)).serialize(true) << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return Run(ParseOptions(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "xgrammar_glm_bench: " << error.what() << '\n';
    return 1;
  }
}
