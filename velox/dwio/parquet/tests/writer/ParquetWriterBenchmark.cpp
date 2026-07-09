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

#include "folly/Benchmark.h"
#include "folly/init/Init.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

namespace {

constexpr vector_size_t kNumRows = 100'000;
constexpr int32_t kNumIterations = 50;
constexpr int32_t kSinkSize = 200 * 1024 * 1024;

// Writes a RowVector to a Parquet in-memory sink kNumIterations times.
// Vector creation is excluded from timing by the caller's BenchmarkSuspender.
// Sink and writer option allocation are excluded via a per-iteration suspender.
void writeParquet(const RowVectorPtr& data, memory::MemoryPool* rootPool) {
  auto leafPool = rootPool->addLeafChild("sink");
  for (int32_t i = 0; i < kNumIterations; ++i) {
    folly::BenchmarkSuspender suspender;
    auto sink = std::make_unique<MemorySink>(
        kSinkSize, FileSink::Options{.pool = leafPool.get()});
    WriterOptions options;
    options.memoryPool = rootPool;
    options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
    suspender.dismiss();
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), options, asRowType(data->type()));
    writer->write(data);
    writer->close();
    suspender.rehire();
  }
}

// Builds a dictionary-encoded VARCHAR column with the given cardinality.
VectorPtr makeDictVarchar(
    vector_size_t numRows,
    int32_t dictionarySize,
    memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  auto dictionary = maker.flatVector<std::string>(
      dictionarySize,
      [](vector_size_t i) { return fmt::format("value_{:06d}", i); });
  auto indices = test::makeIndices(
      numRows,
      [dictionarySize](vector_size_t i) { return i % dictionarySize; },
      pool);
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, numRows, dictionary);
}

// Builds a dictionary-encoded INTEGER column with the given cardinality.
VectorPtr makeDictInteger(
    vector_size_t numRows,
    int32_t dictionarySize,
    memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  auto dictionary = maker.flatVector<int32_t>(
      dictionarySize, [](vector_size_t i) { return i * 7; });
  auto indices = test::makeIndices(
      numRows,
      [dictionarySize](vector_size_t i) { return i % dictionarySize; },
      pool);
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, numRows, dictionary);
}

// Builds a flat VARCHAR column (control case, no dictionary).
VectorPtr makeFlatVarchar(vector_size_t numRows, memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  return maker.flatVector<std::string>(numRows, [](vector_size_t i) {
    return fmt::format("value_{:06d}", i % 10);
  });
}

// Builds a flat INTEGER column (control case).
VectorPtr makeFlatInteger(vector_size_t numRows, memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  return maker.flatVector<int32_t>(
      numRows, [](vector_size_t i) { return static_cast<int32_t>(i); });
}

std::shared_ptr<memory::MemoryPool> rootPool;

// -- Dictionary VARCHAR benchmarks at various cardinalities --

void benchDictVarchar(int32_t dictionarySize) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());
  auto column = makeDictVarchar(kNumRows, dictionarySize, leafPool.get());
  auto data = maker.rowVector({"c0"}, {column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(DictVarchar_Card10) {
  benchDictVarchar(10);
}
BENCHMARK(DictVarchar_Card100) {
  benchDictVarchar(100);
}
BENCHMARK(DictVarchar_Card1000) {
  benchDictVarchar(1'000);
}
BENCHMARK(DictVarchar_Card10000) {
  benchDictVarchar(10'000);
}

BENCHMARK_DRAW_LINE();

// -- Dictionary INTEGER benchmarks --

void benchDictInteger(int32_t dictionarySize) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());
  auto column = makeDictInteger(kNumRows, dictionarySize, leafPool.get());
  auto data = maker.rowVector({"c0"}, {column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(DictInteger_Card10) {
  benchDictInteger(10);
}
BENCHMARK(DictInteger_Card100) {
  benchDictInteger(100);
}
BENCHMARK(DictInteger_Card1000) {
  benchDictInteger(1'000);
}

BENCHMARK_DRAW_LINE();

// -- Flat baseline benchmarks (no dictionary, control case) --

BENCHMARK(FlatVarchar) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());
  auto column = makeFlatVarchar(kNumRows, leafPool.get());
  auto data = maker.rowVector({"c0"}, {column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(FlatInteger) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());
  auto column = makeFlatInteger(kNumRows, leafPool.get());
  auto data = maker.rowVector({"c0"}, {column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK_DRAW_LINE();

// -- Multi-column benchmarks for selective flattening --
// These test the case where one column forces flattening (dict-of-dict) while
// other columns are passthrough dictionaries.  With blanket flattening, all
// columns get materialized.  With selective flattening, only the one that
// needs it is flattened.

// Builds a dict-of-dict INTEGER column (forces flattening in needFlatten).
VectorPtr makeDictOfDictInteger(
    vector_size_t numRows,
    int32_t dictionarySize,
    memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  auto dictionary = maker.flatVector<int32_t>(
      dictionarySize, [](vector_size_t i) { return i * 13; });
  auto innerIndices = test::makeIndices(
      numRows,
      [dictionarySize](vector_size_t i) { return i % dictionarySize; },
      pool);
  auto innerDict = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), innerIndices, numRows, dictionary);
  auto outerIndices =
      test::makeIndices(numRows, [](vector_size_t i) { return i; }, pool);
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), outerIndices, numRows, innerDict);
}

// N passthrough dict VARCHAR columns + 1 dict-of-dict column that forces
// flattening.  With blanket flattening all N+1 columns are flattened; with
// selective flattening only the dict-of-dict column is.
void benchMixedColumns(int32_t numPassthroughColumns) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;

  // Passthrough dict VARCHAR columns (low cardinality).
  for (int32_t i = 0; i < numPassthroughColumns; ++i) {
    columns.push_back(makeDictVarchar(kNumRows, 10, leafPool.get()));
    names.push_back(fmt::format("dict_{}", i));
  }

  // One dict-of-dict INTEGER column that forces flattening.
  columns.push_back(makeDictOfDictInteger(kNumRows, 50, leafPool.get()));
  names.push_back("nested");

  auto data = maker.rowVector(std::move(names), columns);

  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(Mixed_1DictPassthrough_1Nested) {
  benchMixedColumns(1);
}
BENCHMARK(Mixed_5DictPassthrough_1Nested) {
  benchMixedColumns(5);
}
BENCHMARK(Mixed_10DictPassthrough_1Nested) {
  benchMixedColumns(10);
}
BENCHMARK(Mixed_20DictPassthrough_1Nested) {
  benchMixedColumns(20);
}

BENCHMARK_DRAW_LINE();

// Control: N passthrough dict VARCHAR columns with NO column that forces
// flattening.  Should show no difference between blanket and selective.
void benchAllPassthroughColumns(int32_t numColumns) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;

  for (int32_t i = 0; i < numColumns; ++i) {
    columns.push_back(makeDictVarchar(kNumRows, 10, leafPool.get()));
    names.push_back(fmt::format("c{}", i));
  }

  auto data = maker.rowVector(std::move(names), columns);

  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(AllPassthrough_5Cols) {
  benchAllPassthroughColumns(5);
}
BENCHMARK(AllPassthrough_10Cols) {
  benchAllPassthroughColumns(10);
}

BENCHMARK_DRAW_LINE();

// -- Multi-batch benchmarks for schema caching --
// These write many small batches to the same file.  Without schema caching,
// each batch re-exports, re-imports, and re-walks the Arrow schema.  With
// caching, only the first batch pays that cost.

// Writes numBatches batches of batchSize rows each to a single file.
void writeParquetMultiBatch(
    const RowVectorPtr& batch,
    int32_t numBatches,
    memory::MemoryPool* rootPool) {
  auto leafPool = rootPool->addLeafChild("sink");
  folly::BenchmarkSuspender suspender;
  auto sink = std::make_unique<MemorySink>(
      kSinkSize, FileSink::Options{.pool = leafPool.get()});
  WriterOptions options;
  options.memoryPool = rootPool;
  options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
  suspender.dismiss();
  auto writer = std::make_unique<parquet::Writer>(
      std::move(sink), options, asRowType(batch->type()));
  for (int32_t i = 0; i < numBatches; ++i) {
    writer->write(batch);
  }
  writer->close();
  suspender.rehire();
}

void benchMultiBatch(int32_t numColumns, int32_t numBatches) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;
  constexpr vector_size_t kBatchSize = 10'000;

  for (int32_t i = 0; i < numColumns; ++i) {
    columns.push_back(makeDictVarchar(kBatchSize, 10, leafPool.get()));
    names.push_back(fmt::format("c{}", i));
  }

  auto batch = maker.rowVector(std::move(names), columns);

  suspender.dismiss();
  writeParquetMultiBatch(batch, numBatches, rootPool.get());
}

BENCHMARK(MultiBatch_5Cols_50Batches) {
  benchMultiBatch(5, 50);
}
BENCHMARK(MultiBatch_10Cols_50Batches) {
  benchMultiBatch(10, 50);
}
BENCHMARK(MultiBatch_20Cols_50Batches) {
  benchMultiBatch(20, 50);
}
BENCHMARK(MultiBatch_10Cols_200Batches) {
  benchMultiBatch(10, 200);
}

BENCHMARK_DRAW_LINE();

// -- MAP(VARCHAR, INTEGER) column benchmarks --
// Tests the encoding overhead of nested map structures.

// Builds a MAP(VARCHAR, INTEGER) column with the given number of entries per
// map row.
VectorPtr makeMapVarcharInteger(
    vector_size_t numRows,
    int32_t entriesPerRow,
    memory::MemoryPool* pool) {
  test::VectorMaker maker(pool);
  return maker.mapVector<std::string, int32_t>(
      numRows,
      [entriesPerRow](vector_size_t /*mapRow*/) { return entriesPerRow; },
      [](vector_size_t /*mapRow*/, vector_size_t row) {
        return fmt::format("key_{}", row);
      },
      [](vector_size_t mapRow, vector_size_t row) {
        return static_cast<int32_t>(mapRow * 10 + row);
      });
}

void benchMapColumn(int32_t entriesPerRow) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  test::VectorMaker maker(leafPool.get());
  auto column = makeMapVarcharInteger(kNumRows, entriesPerRow, leafPool.get());
  auto data = maker.rowVector({"c0"}, {column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(MapVarcharInt_3Entries) {
  benchMapColumn(3);
}
BENCHMARK(MapVarcharInt_5Entries) {
  benchMapColumn(5);
}
BENCHMARK(MapVarcharInt_10Entries) {
  benchMapColumn(10);
}

} // namespace

int32_t main(int32_t argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  rootPool = memory::memoryManager()->addRootPool("ParquetWriterBenchmark");
  folly::runBenchmarks();
  return 0;
}
