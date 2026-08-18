/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook::nimble;

namespace {

constexpr uint32_t kValueCount{1 << 20};
// Keep dense cases at the same limit used by Statistics.cpp.
constexpr uint32_t kDenseRangeSize{4096};

std::vector<int64_t> int64DenseRange{};
std::vector<int64_t> int64SingleValue{};
std::vector<int64_t> int64DenseRangeAtMin{};
std::vector<int64_t> int64SparseRange{};
std::vector<int64_t> int64FullSpan{};
std::vector<uint64_t> uint64DenseRangeAtMax{};

template <typename T>
void updateUniqueStats(uint32_t iters, const std::vector<T>& data) {
  while (iters-- > 0) {
    auto statistics =
        Statistics<T>::create(std::span<const T>{data.data(), data.size()});
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    folly::doNotOptimizeAway(uniqueCounts.size());
    folly::doNotOptimizeAway(uniqueCounts.at(data[data.size() / 2]));
  }
}

void prepareData() {
  int64DenseRange.resize(kValueCount);
  int64SingleValue.resize(kValueCount);
  int64DenseRangeAtMin.resize(kValueCount);
  int64SparseRange.resize(kValueCount);
  int64FullSpan.resize(kValueCount);
  uint64DenseRangeAtMax.resize(kValueCount);

  constexpr int64_t int64Min{std::numeric_limits<int64_t>::lowest()};
  constexpr int64_t int64Max{std::numeric_limits<int64_t>::max()};
  constexpr uint64_t uint64HighBase{
      std::numeric_limits<uint64_t>::max() - kDenseRangeSize + 1};

  for (uint32_t i = 0; i < kValueCount; ++i) {
    const uint32_t denseOffset{i % kDenseRangeSize};
    int64DenseRange[i] = static_cast<int64_t>(denseOffset);
    int64SingleValue[i] = 42;
    int64DenseRangeAtMin[i] = int64Min + static_cast<int64_t>(denseOffset);
    int64SparseRange[i] = static_cast<int64_t>(denseOffset * 2);
    uint64DenseRangeAtMax[i] = uint64HighBase + denseOffset;

    const uint32_t fullSpanOffset{i % 3};
    int64FullSpan[i] = fullSpanOffset == 0
        ? int64Min
        : (fullSpanOffset == 1 ? int64_t{0} : int64Max);
  }
}

} // namespace

BENCHMARK(UniqueStats_Int64DenseRange, iters) {
  updateUniqueStats(iters, int64DenseRange);
}

BENCHMARK_RELATIVE(UniqueStats_Int64SingleValue, iters) {
  updateUniqueStats(iters, int64SingleValue);
}

BENCHMARK_RELATIVE(UniqueStats_Int64DenseRangeAtMin, iters) {
  updateUniqueStats(iters, int64DenseRangeAtMin);
}

BENCHMARK_RELATIVE(UniqueStats_Uint64DenseRangeAtMax, iters) {
  updateUniqueStats(iters, uint64DenseRangeAtMax);
}

BENCHMARK_RELATIVE(UniqueStats_Int64SparseRangeHash, iters) {
  updateUniqueStats(iters, int64SparseRange);
}

BENCHMARK_RELATIVE(UniqueStats_Int64FullSpanHash, iters) {
  updateUniqueStats(iters, int64FullSpan);
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  prepareData();
  folly::runBenchmarks();
}
