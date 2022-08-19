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
#include <folly/Random.h>
#include <folly/init/Init.h>

#include <optional>

#include "velox/common/base/SimdUtil.h"

namespace facebook {
namespace velox {
namespace test {
class SimdBenchmark {
 public:
  static constexpr uint64_t kMaxNumEntries = 1l << 24;

  explicit SimdBenchmark() {
    init();
  }

  static SimdBenchmark* getInstance() {
    static std::unique_ptr<SimdBenchmark> instance;
    if (instance == nullptr) {
      instance = std::make_unique<SimdBenchmark>();
    }
    return instance.get();
  }

  template <typename T>
  void initBitsMap(std::unordered_map<uint64_t, std::vector<T>>& map) {
    // for (uint64_t i = 10; i < kMaxNumEntries; i *= 10) {
    for (uint64_t i = 10; i < 100; i += 10) {
      std::vector<T> vec;
      vec.reserve(i);
      for (uint64_t j = 0; j < i; j++) {
        vec.emplace_back(j < 2 ? j : folly::Random::rand32(rng) % 8 + 2);
      }
      map.emplace(i, std::move(vec));
    }
  }

  void init() {
    initBitsMap(vecs8Bits);
    initBitsMap(vecs16Bits);
    initBitsMap(vecs32Bits);
    initBitsMap(vecs64Bits);
  }

  void BM_lowerBound_8bits(uint64_t numEntries) {
    simd::lowerBound(
        reinterpret_cast<uint8_t*>(vecs8Bits[numEntries].data()),
        numEntries,
        (uint8_t)1);
  }

  void BM_stdLowerBound_8bits(uint64_t numEntries) {
    auto& vec = vecs8Bits[numEntries];
    std::lower_bound(vec.begin(), vec.end(), (uint8_t)1);
  }

  void BM_lowerBound_16bits(uint64_t numEntries) {
    simd::lowerBound(
        reinterpret_cast<uint16_t*>(vecs16Bits[numEntries].data()),
        numEntries,
        (uint16_t)1);
  }

  void BM_stdLowerBound_16bits(uint64_t numEntries) {
    auto& vec = vecs16Bits[numEntries];
    std::lower_bound(vec.begin(), vec.end(), (uint16_t)1);
  }

  void BM_lowerBound_32bits(uint64_t numEntries) {
    simd::lowerBound(
        reinterpret_cast<uint32_t*>(vecs32Bits[numEntries].data()),
        numEntries,
        (uint32_t)1);
  }

  void BM_stdLowerBound_32bits(uint64_t numEntries) {
    auto& vec = vecs32Bits[numEntries];
    std::lower_bound(vec.begin(), vec.end(), (uint32_t)1);
  }

  void BM_lowerBound_64bits(uint64_t numEntries) {
    simd::lowerBound(
        reinterpret_cast<uint64_t*>(vecs64Bits[numEntries].data()),
        numEntries,
        (uint64_t)1);
  }

  void BM_stdLowerBound_64bits(uint64_t numEntries) {
    auto& vec = vecs64Bits[numEntries];
    std::lower_bound(vec.begin(), vec.end(), (uint64_t)1);
  }

 private:
  std::unordered_map<uint64_t, std::vector<uint8_t>> vecs8Bits;
  std::unordered_map<uint64_t, std::vector<uint16_t>> vecs16Bits;
  std::unordered_map<uint64_t, std::vector<uint32_t>> vecs32Bits;
  std::unordered_map<uint64_t, std::vector<uint64_t>> vecs64Bits;
  uint64_t result;
  folly::Random::DefaultGenerator rng;
};

constexpr uint64_t kNumEntries = 40;
constexpr uint64_t kNumRuns = 10000;

BENCHMARK(lowerBound_8bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    auto g = 0;
    benchmark->BM_lowerBound_8bits(kNumEntries);
  }
}
BENCHMARK_RELATIVE(stdLowerBound_8bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_stdLowerBound_8bits(kNumEntries);
  }
}

BENCHMARK(lowerBound_16bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_lowerBound_16bits(kNumEntries);
  }
}
BENCHMARK_RELATIVE(stdLowerBound_16bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_stdLowerBound_16bits(kNumEntries);
  }
}

BENCHMARK(lowerBound_32bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_lowerBound_32bits(kNumEntries);
  }
}
BENCHMARK_RELATIVE(stdLowerBound_32bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_stdLowerBound_32bits(kNumEntries);
  }
}

BENCHMARK(lowerBound_64bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_lowerBound_64bits(kNumEntries);
  }
}
BENCHMARK_RELATIVE(stdLowerBound_64bits) {
  auto benchmark = SimdBenchmark::getInstance();
  for (int i = 0; i < kNumRuns; i++) {
    benchmark->BM_stdLowerBound_64bits(kNumEntries);
  }
}

} // namespace test
} // namespace velox
} // namespace facebook

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  // Init upfront
  auto instance = facebook::velox::test::SimdBenchmark::getInstance();
  folly::runBenchmarks();
  return 0;
}
