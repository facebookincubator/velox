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
#include <folly/hash/Hash.h>
#include <folly/init/Init.h>

#include <random>
#include <vector>

#include "velox/exec/AdaptivePrefetch.h"

namespace {

using namespace facebook::velox::exec;
using NormalizedKeyT = uint64_t;

struct FakeRowContainer {
  explicit FakeRowContainer(int64_t numRows) {
    constexpr int32_t kRowSize = 32;
    constexpr int32_t kKeySize = sizeof(NormalizedKeyT);
    data_.resize(static_cast<size_t>(numRows) * kRowSize);
    rows_.resize(static_cast<size_t>(numRows));

    std::mt19937_64 rng(42);
    for (int64_t i = 0; i < numRows; ++i) {
      char* base = data_.data() + i * kRowSize;
      auto* key = reinterpret_cast<NormalizedKeyT*>(base);
      *key = rng();
      rows_[i] = base + kKeySize;
    }
    std::shuffle(rows_.begin(), rows_.end(), std::mt19937(123));
  }

  char** rows() {
    return rows_.data();
  }
  int64_t numRows() const {
    return static_cast<int64_t>(rows_.size());
  }

 private:
  std::vector<char> data_;
  std::vector<char*> rows_;
};

inline NormalizedKeyT& normalizedKey(char* row) {
  return reinterpret_cast<NormalizedKeyT*>(row)[-1];
}

constexpr int32_t kBatchSize = 1024;

std::unique_ptr<FakeRowContainer> g4K;
std::unique_ptr<FakeRowContainer> g40K;
std::unique_ptr<FakeRowContainer> g400K;
std::unique_ptr<FakeRowContainer> g4M;

void hashRowsNoPrefetch(FakeRowContainer& container, int32_t iters) {
  std::vector<uint64_t> hashes(kBatchSize);
  auto** allRows = container.rows();
  const int64_t numRows = container.numRows();

  for (int32_t iter = 0; iter < iters; ++iter) {
    for (int64_t start = 0; start + kBatchSize <= numRows;
         start += kBatchSize) {
      char** rows = allRows + start;
      for (int32_t i = 0; i < kBatchSize; ++i) {
        hashes[i] = folly::hasher<uint64_t>()(normalizedKey(rows[i]));
      }
    }
  }
  folly::doNotOptimizeAway(hashes.data());
}

void hashRowsWithPrefetch(FakeRowContainer& container, int32_t iters) {
  std::vector<uint64_t> hashes(kBatchSize);
  auto** allRows = container.rows();
  const int64_t numRows = container.numRows();

  for (int32_t iter = 0; iter < iters; ++iter) {
    for (int64_t start = 0; start + kBatchSize <= numRows;
         start += kBatchSize) {
      char** rows = allRows + start;
      AdaptivePrefetch prefetch(kBatchSize);
      for (int32_t i = 0; i < kBatchSize; ++i) {
        if (auto ahead = prefetch.lookAhead()) {
          __builtin_prefetch(rows[i + ahead] - sizeof(NormalizedKeyT));
        }
        hashes[i] = folly::hasher<uint64_t>()(normalizedKey(rows[i]));
      }
    }
  }
  folly::doNotOptimizeAway(hashes.data());
}

BENCHMARK(noPrefetch_4K) {
  hashRowsNoPrefetch(*g4K, 5);
}
BENCHMARK_RELATIVE(withPrefetch_4K) {
  hashRowsWithPrefetch(*g4K, 5);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(noPrefetch_40K) {
  hashRowsNoPrefetch(*g40K, 5);
}
BENCHMARK_RELATIVE(withPrefetch_40K) {
  hashRowsWithPrefetch(*g40K, 5);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(noPrefetch_400K) {
  hashRowsNoPrefetch(*g400K, 5);
}
BENCHMARK_RELATIVE(withPrefetch_400K) {
  hashRowsWithPrefetch(*g400K, 5);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(noPrefetch_4M) {
  hashRowsNoPrefetch(*g4M, 5);
}
BENCHMARK_RELATIVE(withPrefetch_4M) {
  hashRowsWithPrefetch(*g4M, 5);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  g4K = std::make_unique<FakeRowContainer>(4'000);
  g40K = std::make_unique<FakeRowContainer>(40'000);
  g400K = std::make_unique<FakeRowContainer>(400'000);
  g4M = std::make_unique<FakeRowContainer>(4'000'000);
  folly::runBenchmarks();
  return 0;
}
