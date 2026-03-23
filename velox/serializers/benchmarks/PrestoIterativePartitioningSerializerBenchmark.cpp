/*
 * Copyright (c) International Business Machines Corporation
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

#include "velox/serializers/PrestoIterativePartitioningSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::serializer::presto;

constexpr int64_t kBufferSize = 2 * 1024 * 1024;

namespace {

class PrestoIterativePartitioningSerializerBenchmark
    : public test::VectorTestBase {
 public:
  /// Creates a flat vector of type T with deterministic null pattern.
  /// Rows where (row % 100) < nullPct are null.
  template <typename T>
  VectorPtr makeColumnOfType(vector_size_t size, int32_t nullPct) {
    if (nullPct == 0) {
      return makeFlatVector<T>(
          size, [](auto row) { return static_cast<T>(row); });
    }
    return makeFlatVector<T>(
        size,
        [](auto row) { return static_cast<T>(row); },
        [nullPct](auto row) { return (row % 100) < nullPct; });
  }

  /// Creates a flat vector of the given TypeKind with the given null ratio.
  VectorPtr makeColumn(vector_size_t size, TypeKind colKind, int32_t nullPct) {
    switch (colKind) {
      case TypeKind::BOOLEAN:
        return makeColumnOfType<bool>(size, nullPct);
      case TypeKind::INTEGER:
        return makeColumnOfType<int32_t>(size, nullPct);
      case TypeKind::BIGINT:
        return makeColumnOfType<int64_t>(size, nullPct);
      case TypeKind::HUGEINT:
        return makeColumnOfType<int128_t>(size, nullPct);
      default:
        VELOX_UNSUPPORTED(
            "Unsupported TypeKind: {}", TypeKindName::toName(colKind));
    }
  }

  /// Creates a RowVector with numCols columns of the given TypeKind.
  RowVectorPtr makeInput(
      vector_size_t size,
      TypeKind colKind,
      uint32_t numCols,
      int32_t nullPct) {
    std::vector<std::string> names;
    std::vector<VectorPtr> children;
    names.reserve(numCols);
    children.reserve(numCols);
    for (uint32_t i = 0; i < numCols; ++i) {
      names.push_back(fmt::format("c{}", i));
      children.push_back(makeColumn(size, colKind, nullPct));
    }
    return makeRowVector(names, children);
  }

  std::vector<uint32_t> makePartitions(
      vector_size_t size,
      uint32_t numPartitions) {
    std::vector<uint32_t> partitions(size);
    for (vector_size_t i = 0; i < size; ++i) {
      partitions[i] = i % numPartitions;
    }
    return partitions;
  }

  std::unique_ptr<PrestoIterativePartitioningSerializer> makeSerializer(
      const RowTypePtr& type,
      uint32_t numPartitions) {
    SerdeOpts opts;
    return std::make_unique<PrestoIterativePartitioningSerializer>(
        type, numPartitions, opts, pool_.get());
  }
};

} // namespace

/// Single benchmark function parameterized by (colKind, numCols, nullPct,
/// numPartitions). Registered via BENCHMARK_NAMED_PARAM below.
///
/// All runs use 10'000 rows. Setup (input creation, serializer construction,
/// append) is excluded from the measured time.
void benchmarkFlush(
    uint32_t /* iters */,
    TypeKind colKind,
    uint32_t numCols,
    int32_t nullPct,
    uint32_t numPartitions) {
  folly::BenchmarkSuspender suspender;
  PrestoIterativePartitioningSerializerBenchmark benchmark;
  auto input = benchmark.makeInput(10'000, colKind, numCols, nullPct);
  auto parts = benchmark.makePartitions(10'000, numPartitions);
  auto serializer = benchmark.makeSerializer(
      std::static_pointer_cast<const RowType>(input->type()), numPartitions);

  while (serializer->bytesBuffered() < kBufferSize) {
    serializer->append(input, parts);
  }

  suspender.dismiss();

  auto result = serializer->flush();
  folly::doNotOptimizeAway(result);
}

// clang-format off
// Dimensions:
//   col type:       {bool, int, bigint, hugeint}
//   num cols:       {1, 4, 16, 64}
//   null pct:       {0, 25, 50, 75, 100}
//   num partitions: {1, 4, 16, 64, 256, 1024}
//
// Naming: flush_<type>_<N>cols_<P>pct_<K>parts

#define FLUSH_PARAM(type_name, kind, num_cols, null_pct, num_parts) \
  BENCHMARK_NAMED_PARAM(                                            \
      benchmarkFlush,                                               \
      type_name## _## num_cols## cols_## null_pct## pct_## num_parts## parts, \
      TypeKind::kind, num_cols, null_pct, num_parts)

#define FLUSH_FOR_PARTS(type_name, kind, num_cols, null_pct) \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 1)        \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 4)        \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 16)       \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 64)       \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 256)      \
  FLUSH_PARAM(type_name, kind, num_cols, null_pct, 1024)

#define FLUSH_FOR_NULLS(type_name, kind, num_cols) \
  FLUSH_FOR_PARTS(type_name, kind, num_cols, 0)    \
  FLUSH_FOR_PARTS(type_name, kind, num_cols, 25)   \
  FLUSH_FOR_PARTS(type_name, kind, num_cols, 50)   \
  FLUSH_FOR_PARTS(type_name, kind, num_cols, 75)   \
  FLUSH_FOR_PARTS(type_name, kind, num_cols, 100)

#define FLUSH_FOR_COLS(type_name, kind) \
  FLUSH_FOR_NULLS(type_name, kind, 1)   \
  FLUSH_FOR_NULLS(type_name, kind, 4)   \
  FLUSH_FOR_NULLS(type_name, kind, 16)  \
  FLUSH_FOR_NULLS(type_name, kind, 64)

FLUSH_FOR_COLS(bool, BOOLEAN)
FLUSH_FOR_COLS(int, INTEGER)
FLUSH_FOR_COLS(bigint, BIGINT)
FLUSH_FOR_COLS(ldec, HUGEINT)
// clang-format on

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  PrestoVectorSerde::registerVectorSerde();
  folly::runBenchmarks();
  return 0;
}
