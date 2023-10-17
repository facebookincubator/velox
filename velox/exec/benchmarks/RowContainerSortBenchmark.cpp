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

#include "velox/exec/RowContainer.h"
#include "velox/external/timsort/TimSort.hpp"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/tests/VectorTestUtils.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::test {
namespace {

template <typename T>
VectorGeneratedData<T>
testData(size_t length, size_t cardinality, bool sequences, size_t iteration) {
  int32_t seqCount =
      sequences ? std::max(2, (int32_t)(length / cardinality)) : 0;
  int32_t seqLength = sequences ? std::max(2, (int32_t)(length / seqCount)) : 0;
  return genTestDataWithSequences<T>(
      length,
      cardinality,
      false /* isSorted */,
      true /* includeNulls */,
      seqCount,
      seqLength,
      false /* useFullTypeRange */,
      length + iteration /* seed */);
}

std::vector<char*> store(
    velox::exec::RowContainer& rowContainer,
    DecodedVector& decodedVector,
    vector_size_t size) {
  std::vector<char*> rows(size);
  for (size_t row = 0; row < size; ++row) {
    rows[row] = rowContainer.newRow();
    rowContainer.store(decodedVector, row, rows[row], 0);
  }
  return rows;
}

template <typename T>
void rowContainerStdSortBenchmark(
    uint32_t iterations,
    size_t length,
    size_t cardinality,
    bool sequences) {
  folly::BenchmarkSuspender suspender;
  auto pool = memory::addDefaultLeafMemoryPool();
  VectorMaker vectorMaker(pool.get());

  for (size_t k = 0; k < iterations; ++k) {
    auto data = testData<T>(length, cardinality, sequences, k);
    auto vector =
        vectorMaker.encodedVector<T>(VectorEncoding::Simple::FLAT, data.data());
    DecodedVector decoded(*vector);
    // Create row container.
    std::vector<TypePtr> types{vector->type()};

    // Store the vector in the rowContainer.
    auto rowContainer =
        std::make_unique<velox::exec::RowContainer>(types, pool.get());
    int size = data.data().size();
    auto rows = store(*rowContainer, decoded, size);
    suspender.dismiss();
    std::sort(
        rows.begin(), rows.end(), [&](const char* left, const char* right) {
          return rowContainer->compareRows(left, right) < 0;
        });
    suspender.rehire();
  }
}

template <typename T>
void rowContainerTimSortBenchmark(
    uint32_t iterations,
    size_t length,
    size_t cardinality,
    bool sequences) {
  folly::BenchmarkSuspender suspender;
  auto pool = memory::addDefaultLeafMemoryPool();
  VectorMaker vectorMaker(pool.get());

  for (size_t k = 0; k < iterations; ++k) {
    auto data = testData<T>(length, cardinality, sequences, k);
    auto vector =
        vectorMaker.encodedVector<T>(VectorEncoding::Simple::FLAT, data.data());
    DecodedVector decoded(*vector);
    // Create row container.
    std::vector<TypePtr> types{vector->type()};

    // Store the vector in the rowContainer.
    auto rowContainer =
        std::make_unique<velox::exec::RowContainer>(types, pool.get());
    int size = vector->size();
    auto rows = store(*rowContainer, decoded, size);
    suspender.dismiss();
    gfx::timsort(
        rows.begin(), rows.end(), [&](const char* left, const char* right) {
          return rowContainer->compareRows(left, right) < 0;
        });
    suspender.rehire();
  }
}

void BM_Int64_stdSort(
    uint32_t iterations,
    size_t numRows,
    size_t cardinality,
    bool sequences) {
  rowContainerStdSortBenchmark<int64_t>(
      iterations, numRows, cardinality, sequences);
}

void BM_Int64_timSort(
    uint32_t iterations,
    size_t numRows,
    size_t cardinality,
    bool sequences) {
  rowContainerTimSortBenchmark<int64_t>(
      iterations, numRows, cardinality, sequences);
}
} // namespace

// 100k rows===============
BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    100k_rows_100k_uni_noseq,
    100000,
    100000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    100k_rows_100k_uni_noseq,
    100000,
    100000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    100k_rows_10k_uni_noseq,
    100000,
    10000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    100k_rows_10k_uni_noseq,
    100000,
    10000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    100k_rows_1k_uni_noseq,
    100000,
    1000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    100k_rows_1k_uni_noseq,
    100000,
    1000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    100k_rows_10k_uni_seq,
    100000,
    10000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    100k_rows_10k_uni_seq,
    100000,
    10000,
    true);

BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    100k_rows_1k_uni_seq,
    100000,
    1000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    100k_rows_1k_uni_seq,
    100000,
    1000,
    true);

BENCHMARK_DRAW_LINE();

// 1M rows===============
BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    1M_rows_10k_uni_noseq,
    1000000,
    10000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    1M_rows_10k_uni_noseq,
    1000000,
    10000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    1M_rows_1k_uni_noseq,
    1000000,
    1000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    1M_rows_1k_uni_noseq,
    1000000,
    1000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    1M_rows_10k_uni_seq,
    1000000,
    10000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    1M_rows_10k_uni_seq,
    1000000,
    10000,
    true);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    1M_rows_1k_uni_seq,
    1000000,
    1000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    1M_rows_1k_uni_seq,
    1000000,
    1000,
    true);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    1M_rows_100_uni_seq,
    1000000,
    100,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    1M_rows_100_uni_seq,
    1000000,
    100,
    true);
BENCHMARK_DRAW_LINE();

// 2M rows===============
BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    2M_rows_10k_uni_noseq,
    2000000,
    10000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    2M_rows_10k_uni_noseq,
    2000000,
    10000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    2M_rows_1k_uni_noseq,
    2000000,
    1000,
    false);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    2M_rows_1k_uni_noseq,
    2000000,
    1000,
    false);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    2M_rows_10k_uni_seq,
    2000000,
    10000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    2M_rows_10k_uni_seq,
    2000000,
    10000,
    true);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    2M_rows_1k_uni_seq,
    2000000,
    1000,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    2M_rows_1k_uni_seq,
    2000000,
    1000,
    true);
BENCHMARK_DRAW_LINE();

BENCHMARK_NAMED_PARAM(
    BM_Int64_stdSort,
    2M_rows_100_uni_seq,
    2000000,
    100,
    true);
BENCHMARK_RELATIVE_NAMED_PARAM(
    BM_Int64_timSort,
    2M_rows_100_uni_seq,
    2000000,
    100,
    true);
BENCHMARK_DRAW_LINE();

} // namespace facebook::velox::test

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  folly::runBenchmarks();
  return 0;
}
