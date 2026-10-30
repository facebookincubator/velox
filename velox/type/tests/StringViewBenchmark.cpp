/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <folly/Random.h>
#include <folly/init/Init.h>
#include "folly/Benchmark.h"
#include "velox/type/Type.h"

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox {
namespace {

void runStringViewCreate(uint32_t iterations, uint32_t len) {
  folly::BenchmarkSuspender suspender;

  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  if (len > chars.size()) {
    len = chars.size();
  }
  size_t sum = 0;
  suspender.dismiss();

  for (auto i = 0; i < iterations; i++) {
    StringView str(chars.data() + i % (chars.size() - len), len);
    sum += str.data()[0];
  }

  folly::doNotOptimizeAway(sum);
}

constexpr auto INLINE_SIZE = StringView::kInlineSize;
constexpr auto NON_INLINE_SIZE = StringView::kInlineSize + 10;

// Short strings which can be inlined.
BENCHMARK_PARAM(runStringViewCreate, INLINE_SIZE);

// Larger strings which won't be inlined.
BENCHMARK_PARAM(runStringViewCreate, NON_INLINE_SIZE);

constexpr size_t kNumPairs = 1'000'000;

using StringPairs = std::vector<std::pair<std::string, std::string>>;

// Returns pairs of strings of 'a's with sizes in [minSize, maxSize]. The
// second string of a pair has a 'b' at a random position up to the smaller
// size. At the smaller size there is no 'b', so one string is a prefix of the
// other.
StringPairs makeStrings(size_t minSize, size_t maxSize) {
  std::mt19937 gen;
  std::uniform_int_distribution<size_t> sizes(minSize, maxSize);
  std::bernoulli_distribution swaps;
  StringPairs result(kNumPairs);
  for (auto& [left, right] : result) {
    left.assign(sizes(gen), 'a');
    right.assign(sizes(gen), 'a');
    const auto smallerSize = std::min(left.size(), right.size());
    std::uniform_int_distribution<size_t> mismatches(0, smallerSize);
    const auto mismatch = mismatches(gen);
    if (mismatch < smallerSize) {
      right[mismatch] = 'b';
    }
    if (swaps(gen)) {
      std::swap(left, right);
    }
  }
  return result;
}

template <size_t kMinSize, size_t kMaxSize>
const std::vector<std::pair<StringView, StringView>>& compareInputs() {
  static const StringPairs strings = makeStrings(kMinSize, kMaxSize);
  static const std::vector<std::pair<StringView, StringView>> result(
      strings.begin(), strings.end());
  return result;
}

template <size_t kMinSize, size_t kMaxSize>
void runStringViewCompare(size_t n) {
  const auto& inputs = compareInputs<kMinSize, kMaxSize>();
  while (n--) {
    for (const auto& [left, right] : inputs) {
      folly::doNotOptimizeAway(left.compare(right));
    }
  }
}

void makeCompareInputs() {
  compareInputs<1, 4>();
  compareInputs<5, 12>();
  compareInputs<13, 64>();
  compareInputs<1, 64>();
}

// clang-format off
BENCHMARK(StringViewCompare_Prefix, n) { runStringViewCompare<1, 4>(n); }
BENCHMARK(StringViewCompare_Inline, n) { runStringViewCompare<5, 12>(n); }
BENCHMARK(StringViewCompare_Large,  n) { runStringViewCompare<13, 64>(n); }
BENCHMARK(StringViewCompare_Mixed,  n) { runStringViewCompare<1, 64>(n); }
// clang-format on
} // namespace
} // namespace facebook::velox

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::makeCompareInputs();
  folly::runBenchmarks();
  return 0;
}
