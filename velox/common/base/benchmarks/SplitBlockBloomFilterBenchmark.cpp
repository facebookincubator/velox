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

#include "velox/common/base/SplitBlockBloomFilter.h"

#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#define VELOX_BENCHMARK(_make, _name, ...) \
  [[maybe_unused]] auto _name = _make(FOLLY_PP_STRINGIZE(_name), __VA_ARGS__)

namespace facebook::velox {
namespace {

template <typename T, typename Hasher>
class SplitBlockBloomFilterBenchmark {
 public:
  SplitBlockBloomFilterBenchmark(
      const char* name,
      Hasher hasher,
      double falsePositive,
      int numInserts,
      int numTests)
      : hasher_(std::move(hasher)),
        numTests_(numTests),
        blocks_(SplitBlockBloomFilter::numBlocks(numInserts, falsePositive)),
        filter_(blocks_) {
    for (int i = 0; i < numInserts; ++i) {
      filter_.insert(hasher_(generateValue()));
    }
    folly::addBenchmark(__FILE__, name, [this] { return run(); });
  }

 private:
  static T generateValue() {
    if constexpr (sizeof(T) == 8) {
      return folly::Random::rand64();
    } else {
      static_assert(sizeof(T) == 4);
      return folly::Random::rand32();
    }
  }

  unsigned run() const {
    int numHits = 0;
    for (int i = 0; i < numTests_; ++i) {
      numHits += filter_.mayContain(hasher_(generateValue()));
    }
    folly::doNotOptimizeAway(numHits);
    return numTests_;
  }

  const Hasher hasher_;
  const double numTests_;
  std::vector<SplitBlockBloomFilter::Block> blocks_;
  SplitBlockBloomFilter filter_;
};

template <typename T, typename Hasher>
SplitBlockBloomFilterBenchmark<T, Hasher> makeBenchmark(
    const char* name,
    Hasher hasher,
    double falsePositive,
    int numInserts,
    int numTests) {
  return SplitBlockBloomFilterBenchmark<T, Hasher>(
      name, std::move(hasher), falsePositive, numInserts, numTests);
}

} // namespace
} // namespace facebook::velox

int main(int argc, char* argv[]) {
  using namespace facebook::velox;
  folly::Init follyInit(&argc, &argv);
  VELOX_BENCHMARK(
      makeBenchmark<int32_t>,
      int32,
      folly::hasher<int64_t>(),
      0.01,
      5'000'000,
      10'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64,
      folly::hasher<int64_t>(),
      0.01,
      5'000'000,
      10'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64_nohash,
      folly::identity,
      0.01,
      5'000'000,
      10'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int32_t>,
      int32_small,
      folly::hasher<int64_t>(),
      0.01,
      500'000,
      1'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64_small,
      folly::hasher<int64_t>(),
      0.01,
      500'000,
      1'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64_nohash_small,
      folly::identity,
      0.01,
      500'000,
      1'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int32_t>,
      int32_large,
      folly::hasher<int64_t>(),
      0.01,
      50'000'000,
      100'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64_large,
      folly::hasher<int64_t>(),
      0.01,
      50'000'000,
      100'000'000);
  VELOX_BENCHMARK(
      makeBenchmark<int64_t>,
      int64_nohash_large,
      folly::identity,
      0.01,
      50'000'000,
      100'000'000);
  folly::runBenchmarks();
  return 0;
}
