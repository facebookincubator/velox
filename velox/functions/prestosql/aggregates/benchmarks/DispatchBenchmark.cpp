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
#define XXH_INLINE_ALL

#include <folly/Benchmark.h>
#include "velox/expression/tests/VectorFuzzer.h"
#include "velox/external/xxhash.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/aggregates/PrestoHasher.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::aggregate;

namespace {

template <typename T>
FOLLY_ALWAYS_INLINE int64_t hashInteger(const T& value) {
  return XXH64_round(0, value);
}

int64_t computeHashesBulkDispatch(VectorPtr vector, memory::MemoryPool* pool) {
  SelectivityVector rows(vector->size());
  PrestoHasher hasher(*vector.get(), rows);

  BufferPtr bufferPtr = AlignedBuffer::allocate<int64_t>(vector->size(), pool);
  hasher.hash(rows, bufferPtr);
  auto rawBuffer = bufferPtr->as<int64_t>();
  int64_t hash = 0;
  for (int i = 0; i < bufferPtr->size(); i++) {
    hash += rawBuffer[i];
  }
  return hash;
}

int64_t computeHashesOptimizedArray(
    VectorPtr arrayVector,
    memory::MemoryPool* pool) {
  // compute Hash for Array of bigints.
  SelectivityVector rows(arrayVector->size());
  DecodedVector decodedVector(*arrayVector, rows);

  auto baseArray = decodedVector.base()->as<ArrayVector>();
  auto rawOffsets = baseArray->rawOffsets();
  auto rawSizes = baseArray->rawSizes();
  auto rawElements =
      baseArray->elements()->as<FlatVector<int64_t>>()->rawValues();
  BufferPtr hashes = AlignedBuffer::allocate<int64_t>(rows.end(), pool);
  auto rawHashes = hashes->asMutable<int64_t>();

  rows.applyToSelected([&](auto row) {
    const auto start = rawOffsets[row];
    const auto end = start + rawSizes[row];
    int64_t hash = 0;
    for (int i = start; i < end; i++) {
      hash = 31 * hash + hashInteger(rawElements[i]);
    }
    rawHashes[row] = hash;
  });

  int64_t hash = 0;
  for (int i = 0; i < rows.end(); i++) {
    hash += rawHashes[i];
  }

  return hash;
}

int64_t computeHashesOptimizedInts(VectorPtr vector, memory::MemoryPool* pool) {
  // compute Hash for Array of bigints.
  SelectivityVector rows(vector->size());
  DecodedVector decodedVector(*vector, rows);

  BufferPtr hashes = AlignedBuffer::allocate<int64_t>(rows.end(), pool);
  auto rawHashes = hashes->asMutable<int64_t>();

  rows.applyToSelected([&](auto row) {
    if (!decodedVector.isNullAt(row)) {
      rawHashes[row] = hashInteger(decodedVector.valueAt<int64_t>(row));
    }
  });

  int64_t hash = 0;
  for (int i = 0; i < rows.end(); i++) {
    hash += rawHashes[i];
  }

  return hash;
}

class DispatchBenchmark : public functions::test::FunctionBenchmarkBase {
  VectorPtr createVector(
      bool hasNulls = true,
      const TypePtr& type = ARRAY(BIGINT())) {
    VectorFuzzer::Options opts;
    opts.vectorSize = 10'000;
    if (!hasNulls) {
      opts.nullChance = 0;
    }
    VectorFuzzer fuzzer(opts, execCtx_.pool());

    if (type->isPrimitiveType()) {
      return fuzzer.fuzz(type);
    }
    return fuzzer.fuzzComplex(type);
  }

 public:
  void runOptimizedArray() {
    folly::BenchmarkSuspender suspender;
    auto arrayVector = createVector();
    suspender.dismiss();

    int64_t hash = 0;
    for (auto i = 0; i < 100; i++) {
      hash += computeHashesOptimizedArray(arrayVector, execCtx_.pool());
    }
    folly::doNotOptimizeAway(hash);
  }

  void runOptimizedInts() {
    folly::BenchmarkSuspender suspender;
    auto vector = createVector(true, BIGINT());
    suspender.dismiss();

    int64_t hash = 0;
    for (auto i = 0; i < 100; i++) {
      hash += computeHashesOptimizedInts(vector, execCtx_.pool());
    }
    folly::doNotOptimizeAway(hash);
  }

  void runBulkDispatch(
      bool hasNulls = true,
      const TypePtr& type = ARRAY(BIGINT())) {
    folly::BenchmarkSuspender suspender;
    auto vector = createVector(hasNulls, type);
    suspender.dismiss();

    int64_t hash = 0;
    for (auto i = 0; i < 100; i++) {
      hash += computeHashesBulkDispatch(vector, execCtx_.pool());
    }
    folly::doNotOptimizeAway(hash);
  }
};

BENCHMARK(presto_hasher) {
  DispatchBenchmark benchmark;
  benchmark.runBulkDispatch(true, BIGINT());
}

BENCHMARK_RELATIVE(presto_hasher_no_nulls) {
  DispatchBenchmark benchmark;
  benchmark.runBulkDispatch(false, BIGINT());
}

BENCHMARK_RELATIVE(optimized_ints) {
  DispatchBenchmark benchmark;
  benchmark.runOptimizedInts();
}

BENCHMARK(presto_hasher_array) {
  DispatchBenchmark benchmark;
  benchmark.runBulkDispatch();
}

BENCHMARK_RELATIVE(presto_hasher_array_no_nulls) {
  DispatchBenchmark benchmark;
  benchmark.runBulkDispatch(false);
}

BENCHMARK_RELATIVE(optimized_array) {
  DispatchBenchmark benchmark;
  benchmark.runOptimizedArray();
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}