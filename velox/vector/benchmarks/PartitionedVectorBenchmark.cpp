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

#include <absl/random/uniform_int_distribution.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include <algorithm>

#include "dwio/common/tests/utils/BatchMaker.h"
#include "vector/PartitionedVector.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::test {

namespace {

thread_local auto gen = std::mt19937(42);

auto noNulls = [](vector_size_t) { return false; };

auto allNulls = [](vector_size_t) { return true; };

auto halfNulls = [](vector_size_t row) { return row % 2 == 0; };

template <TypeKind T>
RowTypePtr scalarTypeGenerator(int32_t numColumns) {
  return ROW(std::vector<TypePtr>(numColumns, createScalarType<T>()));
}

RowTypePtr dateTypeGenerator(int32_t numColumns) {
  return ROW(std::vector<TypePtr>(numColumns, DATE()));
}

RowTypePtr shortDecimalTypeGenerator(int32_t numColumns) {
  return ROW(std::vector<TypePtr>(numColumns, DECIMAL(10, 2)));
}

RowTypePtr longDecimalTypeGenerator(int32_t numColumns) {
  return ROW(std::vector<TypePtr>(numColumns, DECIMAL(20, 3)));
}

RowTypePtr mixedFlatTypeGenerator(int32_t numColumns) {
  const std::vector<TypePtr> typeSelection = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      HUGEINT(),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      DATE(),
      DECIMAL(10, 2),
      DECIMAL(20, 3),
  };

  std::vector<TypePtr> types;
  types.reserve(numColumns);

  for (int i = 0; i < numColumns; ++i) {
    types.push_back(typeSelection[i % typeSelection.size()]);
  }

  std::ranges::shuffle(types, gen);

  return ROW(std::move(types));
}

auto randomPartitionFunction = [](const RowVectorPtr& vector,
                                  uint32_t numPartitions,
                                  std::vector<uint32_t>& partitions) {
  partitions.resize(vector->size());
  for (int i = 0; i < vector->size(); ++i) {
    partitions[i] = gen() % numPartitions;
  }
};

std::shared_ptr<memory::MemoryPool> pool;
std::vector<uint32_t> partitions;

RowVectorPtr createTestVector(
    const std::function<RowTypePtr(int32_t)>& rowTypeGenerator,
    vector_size_t numRows,
    int32_t numColumns,
    const std::function<bool(vector_size_t)>& isNullAt) {
  auto rowType = rowTypeGenerator(numColumns);
  const auto batch = BatchMaker::createBatch(rowType, numRows, *pool, isNullAt);
  return std::static_pointer_cast<RowVector>(batch);
}

} // namespace

void runBM(
    uint32_t iterations,
    const std::function<RowTypePtr(int32_t)>& rowTypeGenerator,
    int32_t numColumns,
    uint32_t numPartitions,
    const std::function<bool(vector_size_t)>& isNullAt = noNulls,
    vector_size_t numRows = 10000) {
  folly::BenchmarkSuspender suspender;
  PartitionBuildContext ctx;
  auto vector =
      createTestVector(rowTypeGenerator, numRows, numColumns, isNullAt);
  randomPartitionFunction(vector, numPartitions, partitions);
  for (uint32_t i = 0; i < iterations; ++i) {
    // PartitionedVector::create mutates its input, so each iteration needs a
    // fresh copy to keep inputs consistent.
    const auto vectorCopy = std::static_pointer_cast<RowVector>(
        BaseVector::copy(*vector, pool.get()));
    suspender.dismiss();
    PartitionedVector::create(
        vectorCopy, partitions, numPartitions, ctx, pool.get());
    suspender.rehire();
  }
}

#define BENCHMARK_CONFIG(name, generator, numCols, nulls, numParts) \
  BENCHMARK_NAMED_PARAM(                                            \
      runBM,                                                        \
      name##_##numCols##Cols_##nulls##_P##numParts,                 \
      generator,                                                    \
      numCols,                                                      \
      numParts,                                                     \
      nulls);

#define BENCHMARK_PARTITIONS(name, generator, numCols, nulls) \
  BENCHMARK_CONFIG(name, generator, numCols, nulls, 4)        \
  BENCHMARK_CONFIG(name, generator, numCols, nulls, 16)       \
  BENCHMARK_CONFIG(name, generator, numCols, nulls, 64)       \
  BENCHMARK_CONFIG(name, generator, numCols, nulls, 256)      \
  BENCHMARK_CONFIG(name, generator, numCols, nulls, 1024)

#define BENCHMARK_SIZES(name, generator, nulls)     \
  BENCHMARK_PARTITIONS(name, generator, 1, nulls)   \
  BENCHMARK_PARTITIONS(name, generator, 10, nulls)  \
  BENCHMARK_PARTITIONS(name, generator, 100, nulls) \
  BENCHMARK_PARTITIONS(name, generator, 1000, nulls)

#define BENCHMARK_TYPE(name, generator)      \
  BENCHMARK_SIZES(name, generator, noNulls)  \
  BENCHMARK_SIZES(name, generator, allNulls) \
  BENCHMARK_SIZES(name, generator, halfNulls)

BENCHMARK_TYPE(BOOLEAN, scalarTypeGenerator<TypeKind::BOOLEAN>);
BENCHMARK_TYPE(SMALLINT, scalarTypeGenerator<TypeKind::SMALLINT>);
BENCHMARK_TYPE(INTEGER, scalarTypeGenerator<TypeKind::INTEGER>);
BENCHMARK_TYPE(BIGINT, scalarTypeGenerator<TypeKind::BIGINT>);
BENCHMARK_TYPE(HUGEINT, scalarTypeGenerator<TypeKind::HUGEINT>);
BENCHMARK_TYPE(REAL, scalarTypeGenerator<TypeKind::REAL>);
BENCHMARK_TYPE(DOUBLE, scalarTypeGenerator<TypeKind::DOUBLE>);
BENCHMARK_TYPE(TIMESTAMP, scalarTypeGenerator<TypeKind::TIMESTAMP>);
BENCHMARK_TYPE(VARCHAR, scalarTypeGenerator<TypeKind::VARCHAR>);
BENCHMARK_TYPE(VARBINARY, scalarTypeGenerator<TypeKind::VARBINARY>);
BENCHMARK_TYPE(DATE, dateTypeGenerator);
BENCHMARK_TYPE(ShortDecimal, shortDecimalTypeGenerator);
BENCHMARK_TYPE(LongDecimal, longDecimalTypeGenerator);
BENCHMARK_TYPE(Mixed, mixedFlatTypeGenerator);

} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  pool = memory::memoryManager()->addLeafPool();
  folly::runBenchmarks();
  return 0;
}
