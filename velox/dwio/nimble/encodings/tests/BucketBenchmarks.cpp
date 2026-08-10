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
#include <tsl/robin_map.h>
#include <bit>

#include "common/init/light.h"
#include "folly/Benchmark.h"
#include "folly/Random.h"

constexpr size_t kDataSize = 20 * 1024 * 1024; // 20M
constexpr size_t kRepeatSize = 5 * 1024; // 5K

std::vector<int64_t> repeatingData;
std::vector<int64_t> uniqueData;

BENCHMARK(NaiveBranching, n) {
  for (int i = 0; i < n; ++i) {
    std::vector<uint64_t> bucketCounts(10, 0);
    for (auto value : uniqueData) {
      if (value < (1 << 7)) {
        ++bucketCounts[0];
      } else if (value < (1ULL << 14)) {
        ++bucketCounts[1];
      } else if (value < (1ULL << 21)) {
        ++bucketCounts[2];
      } else if (value < (1ULL << 28)) {
        ++bucketCounts[3];
      } else if (value < (1ULL << 35)) {
        ++bucketCounts[4];
      } else if (value < (1ULL << 42)) {
        ++bucketCounts[5];
      } else if (value < (1ULL << 49)) {
        ++bucketCounts[6];
      } else if (value < (1ULL << 56)) {
        ++bucketCounts[7];
      } else if (value < (1ULL << 63)) {
        ++bucketCounts[8];
      } else {
        ++bucketCounts[9];
      }
    }
  }
}

BENCHMARK_RELATIVE(CountLZero, n) {
  using T = int64_t;
  constexpr int bitSize = sizeof(T) * 8;
  for (int i = 0; i < n; ++i) {
    std::array<uint64_t, bitSize> bitCounts{};
    for (auto value : uniqueData) {
      ++(bitCounts[bitSize - std::countl_zero((uint64_t)value)]);
    }

    std::vector<uint64_t> bucketCounts(sizeof(uint64_t) * 8 / 7 + 1, 0);
    uint8_t start = 0;
    uint8_t end = 8;
    uint8_t iteration = 0;
    while (start < bitCounts.size()) {
      if (bitCounts.size() < end) {
        end = bitCounts.size();
      }

      for (auto j = start; j < end; ++j) {
        bucketCounts[iteration] += bitCounts[j];
      }
      ++iteration;
      start = end;
      end += 7;
    }
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(NaiveBranchingRepeat, n) {
  for (int i = 0; i < n; ++i) {
    std::vector<uint64_t> bucketCounts(10, 0);
    for (auto value : repeatingData) {
      if (value < (1 << 7)) {
        ++bucketCounts[0];
      } else if (value < (1ULL << 14)) {
        ++bucketCounts[1];

      } else if (value < (1ULL << 21)) {
        ++bucketCounts[2];
      } else if (value < (1ULL << 28)) {
        ++bucketCounts[3];
      } else if (value < (1ULL << 35)) {
        ++bucketCounts[4];
      } else if (value < (1ULL << 42)) {
        ++bucketCounts[5];
      } else if (value < (1ULL << 49)) {
        ++bucketCounts[6];
      } else if (value < (1ULL << 56)) {
        ++bucketCounts[7];
      } else if (value < (1ULL << 63)) {
        ++bucketCounts[8];
      } else {
        ++bucketCounts[9];
      }
    }
  }
}

BENCHMARK_RELATIVE(CountLZeroRepeat, n) {
  using T = int64_t;
  constexpr int bitSize = sizeof(T) * 8;
  for (int i = 0; i < n; ++i) {
    std::array<uint64_t, bitSize> bitCounts{};
    for (auto value : repeatingData) {
      ++(bitCounts[bitSize - std::countl_zero((uint64_t)value)]);
    }

    std::vector<uint64_t> bucketCounts(sizeof(uint64_t) * 8 / 7 + 1, 0);
    uint8_t start = 0;
    uint8_t end = 8;
    uint8_t iteration = 0;
    while (start < bitCounts.size()) {
      if (bitCounts.size() < end) {
        end = bitCounts.size();
      }

      for (auto j = start; j < end; ++j) {
        bucketCounts[iteration] += bitCounts[j];
      }
      ++iteration;
      start = end;
      end += 7;
    }
  }
}

int main(int argc, char** argv) {
  facebook::init::initFacebookLight(&argc, &argv);
  repeatingData.resize(kDataSize);
  uniqueData.resize(kDataSize);
  for (auto i = 0; i < kDataSize; ++i) {
    repeatingData[i] = i % kRepeatSize;
    uniqueData[i] = i;
  }

  std::random_shuffle(repeatingData.begin(), repeatingData.end());
  std::random_shuffle(uniqueData.begin(), uniqueData.end());

  folly::runBenchmarks();
  return 0;
}
