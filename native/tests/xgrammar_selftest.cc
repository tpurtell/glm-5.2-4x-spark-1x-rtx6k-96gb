#include "glmrt_native.h"

#include <cassert>
#include <cstdint>
#include <iostream>

#ifndef GLMRT_XGRAMMAR_TEST_TOKENIZER
#error "GLMRT_XGRAMMAR_TEST_TOKENIZER is required"
#endif

namespace {

bool Allows(const uint32_t* mask, uint32_t token) {
  return (mask[token / 32] & (uint32_t{1} << (token % 32))) != 0;
}

}  // namespace

int main() {
  char error[2048] = {};
  const int32_t stops[] = {6};
  void* compiler = nullptr;
  assert(glmrt_xgrammar_compiler_create(
             GLMRT_XGRAMMAR_TEST_TOKENIZER, 7, stops, 1, &compiler, error, sizeof(error)) ==
         GLMRT_STATUS_OK);
  assert(compiler != nullptr);

  void* grammar = nullptr;
  assert(glmrt_xgrammar_compile(
             compiler, GLMRT_XGRAMMAR_JSON_SCHEMA,
             R"({"type":"object","properties":{"x":{"type":"string"}},"required":["x"],"additionalProperties":false})",
             1, &grammar, error, sizeof(error)) == GLMRT_STATUS_OK);
  assert(grammar != nullptr);

  void* matcher = nullptr;
  assert(glmrt_xgrammar_matcher_create(grammar, &matcher, error, sizeof(error)) ==
         GLMRT_STATUS_OK);
  uint32_t mask[1] = {};
  int needs_mask = 0;
  assert(glmrt_xgrammar_matcher_fill_bitmask(
             matcher, mask, 1, &needs_mask, error, sizeof(error)) == GLMRT_STATUS_OK);
  assert(needs_mask != 0);
  assert(Allows(mask, 1));
  assert(!Allows(mask, 4));

  for (const uint32_t token : {1U, 2U, 3U, 4U, 5U}) {
    int accepted = 0;
    assert(glmrt_xgrammar_matcher_accept_token(
               matcher, token, &accepted, error, sizeof(error)) == GLMRT_STATUS_OK);
    assert(accepted != 0);
  }
  int completed = 0;
  assert(glmrt_xgrammar_matcher_is_completed(
             matcher, &completed, error, sizeof(error)) == GLMRT_STATUS_OK);
  assert(completed != 0);
  mask[0] = 0;
  assert(glmrt_xgrammar_matcher_fill_bitmask(
             matcher, mask, 1, &needs_mask, error, sizeof(error)) == GLMRT_STATUS_OK);
  assert(Allows(mask, 6));

  void* forked = nullptr;
  assert(glmrt_xgrammar_matcher_fork(matcher, &forked, error, sizeof(error)) ==
         GLMRT_STATUS_OK);
  assert(glmrt_xgrammar_matcher_destroy(forked) == GLMRT_STATUS_OK);
  assert(glmrt_xgrammar_matcher_destroy(matcher) == GLMRT_STATUS_OK);
  assert(glmrt_xgrammar_grammar_destroy(grammar) == GLMRT_STATUS_OK);
  assert(glmrt_xgrammar_compiler_destroy(compiler) == GLMRT_STATUS_OK);
  std::cout << "glmrt_xgrammar_selftest: ok\n";
  return 0;
}
