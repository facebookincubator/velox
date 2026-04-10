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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace {

constexpr vector_size_t kBatchSize = 10'000;
constexpr int kIterations = 100;

class BenchmarkHelper {
 public:
  memory::MemoryPool* pool() {
    return pool_.get();
  }

  test::VectorMaker& vectorMaker() {
    return vectorMaker_;
  }

  void runMarkSorted(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders) {
    auto plan = PlanBuilder()
                    .values(input)
                    .markSorted("is_sorted", sortingKeys, sortingOrders)
                    .planNode();
    auto result = AssertQueryBuilder(plan).copyResults(pool());
    folly::doNotOptimizeAway(result);
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  test::VectorMaker vectorMaker_{pool_.get()};
};

// --- Sorted INTEGER (SIMD path) ---
BENCHMARK(sortedInteger) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto data = helper.vectorMaker().rowVector({
      helper.vectorMaker().flatVector<int32_t>(
          kBatchSize, [](vector_size_t i) { return i; }),
  });
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted({data}, {"c0"}, {core::kAscNullsLast});
  }
}

// --- Sorted BIGINT (SIMD path) ---
BENCHMARK_RELATIVE(sortedBigint) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto data = helper.vectorMaker().rowVector({
      helper.vectorMaker().flatVector<int64_t>(
          kBatchSize, [](vector_size_t i) { return static_cast<int64_t>(i); }),
  });
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted({data}, {"c0"}, {core::kAscNullsLast});
  }
}

// --- Sorted VARCHAR (generic path, no SIMD) ---
// U8 fix: strings stored in vector with lifetime spanning the benchmark.
BENCHMARK_RELATIVE(sortedVarchar) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  std::vector<std::string> storage(kBatchSize);
  for (vector_size_t i = 0; i < kBatchSize; ++i) {
    storage[i] = fmt::format("str_{:08d}", i);
  }
  auto data = helper.vectorMaker().rowVector({
      helper.vectorMaker().flatVector<StringView>(
          kBatchSize,
          [&storage](vector_size_t i) { return StringView(storage[i]); }),
  });
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted({data}, {"c0"}, {core::kAscNullsLast});
  }
}

// --- Unsorted INTEGER (SIMD path, many false bits) ---
BENCHMARK(unsortedInteger) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto data = helper.vectorMaker().rowVector({
      helper.vectorMaker().flatVector<int32_t>(
          kBatchSize,
          [](vector_size_t i) {
            // Alternating pattern creates many unsorted pairs.
            return (i % 2 == 0) ? i : kBatchSize - i;
          }),
  });
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted({data}, {"c0"}, {core::kAscNullsLast});
  }
}

// --- ConstantVector (O(1) fast path) ---
BENCHMARK_RELATIVE(constantVector) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  auto data = helper.vectorMaker().rowVector({
      BaseVector::createConstant(INTEGER(), 42, kBatchSize, helper.pool()),
  });
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted({data}, {"c0"}, {core::kAscNullsLast});
  }
}

// --- Cross-batch comparison (measures overhead) ---
BENCHMARK(crossBatch) {
  folly::BenchmarkSuspender suspender;
  BenchmarkHelper helper;
  constexpr vector_size_t kSmallBatch = 100;
  constexpr int kNumBatches = 100;
  std::vector<RowVectorPtr> batches;
  batches.reserve(kNumBatches);
  for (int b = 0; b < kNumBatches; ++b) {
    batches.push_back(helper.vectorMaker().rowVector({
        helper.vectorMaker().flatVector<int32_t>(
            kSmallBatch, [b](vector_size_t i) { return b * kSmallBatch + i; }),
    }));
  }
  suspender.dismiss();

  for (int i = 0; i < kIterations; ++i) {
    helper.runMarkSorted(batches, {"c0"}, {core::kAscNullsLast});
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
