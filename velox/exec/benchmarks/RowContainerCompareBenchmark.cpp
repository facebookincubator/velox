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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/exec/RowContainer.h"
#include "velox/type/StringView.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {
namespace {

constexpr size_t kNumPairs = 1'000'000;

using StringPairs = std::vector<std::pair<std::string, std::string>>;

// Returns pairs of strings of 'a's with sizes in [minSize, maxSize]. The second
// string of a pair has a 'b' at a random position up to the smaller size. At
// the smaller size there is no 'b', so one string is a prefix of the other.
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
const StringPairs& stringInputs() {
  static const StringPairs result = makeStrings(kMinSize, kMaxSize);
  return result;
}

memory::MemoryPool& pool() {
  static const auto result = memory::memoryManager()->addLeafPool();
  return *result;
}

RowContainer& rowContainer() {
  static RowContainer result({VARCHAR()}, &pool());
  return result;
}

template <size_t kMinSize, size_t kMaxSize>
const std::vector<std::pair<char*, char*>>& rowContainerInputs() {
  static const auto result = [] {
    const auto& strings = stringInputs<kMinSize, kMaxSize>();
    const auto numPairs = static_cast<vector_size_t>(strings.size());
    auto vector = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), 2 * numPairs, &pool());
    for (vector_size_t i = 0; i < numPairs; ++i) {
      vector->set(2 * i, StringView(strings[i].first));
      vector->set(2 * i + 1, StringView(strings[i].second));
    }
    const DecodedVector decoded(*vector);

    auto& container = rowContainer();
    auto store = [&](vector_size_t index) {
      auto* row = container.newRow();
      container.store(decoded, index, row, 0);
      return row;
    };

    std::vector<std::pair<char*, char*>> rows;
    for (vector_size_t i = 0; i < numPairs; ++i) {
      auto* left = store(2 * i);
      rows.emplace_back(left, store(2 * i + 1));
    }
    return rows;
  }();
  return result;
}

template <size_t kMinSize, size_t kMaxSize>
void rowContainerCompare(size_t n) {
  const auto& inputs = rowContainerInputs<kMinSize, kMaxSize>();
  const auto& container = rowContainer();
  while (n--) {
    for (const auto& [left, right] : inputs) {
      folly::doNotOptimizeAway(container.compare(left, right, 0));
    }
  }
}

void makeInputs() {
  rowContainerInputs<1, 4>();
  rowContainerInputs<5, 12>();
  rowContainerInputs<13, 64>();
  rowContainerInputs<1, 64>();
}

// clang-format off
BENCHMARK(RowContainerCompare_Prefix, n) { rowContainerCompare<1, 4>(n); }
BENCHMARK(RowContainerCompare_Inline, n) { rowContainerCompare<5, 12>(n); }
BENCHMARK(RowContainerCompare_Large,  n) { rowContainerCompare<13, 64>(n); }
BENCHMARK(RowContainerCompare_Mixed,  n) { rowContainerCompare<1, 64>(n); }
// clang-format on

} // namespace
} // namespace facebook::velox::exec

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::velox::exec::makeInputs();
  folly::runBenchmarks();
  return 0;
}
