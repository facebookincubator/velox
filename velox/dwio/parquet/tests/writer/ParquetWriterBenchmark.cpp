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
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::parquet;

namespace {

constexpr vector_size_t kNumRows = 100'000;
constexpr int kNumIterations = 50;
constexpr int kSinkSize = 200 * 1024 * 1024;

/// Writes a RowVector to a Parquet in-memory sink kNumIterations times.
/// Setup (vector creation, sink allocation) is excluded from timing.
void writeParquet(const RowVectorPtr& data, memory::MemoryPool* rootPool) {
  auto leafPool = rootPool->addLeafChild("sink");
  for (int i = 0; i < kNumIterations; ++i) {
    auto sink = std::make_unique<MemorySink>(
        kSinkSize, FileSink::Options{.pool = leafPool.get()});
    WriterOptions options;
    options.memoryPool = rootPool;
    options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
    auto writer = std::make_unique<parquet::Writer>(
        std::move(sink), options, asRowType(data->type()));
    writer->write(data);
    writer->close();
  }
}

/// Builds a dictionary-encoded VARCHAR column with the given cardinality.
VectorPtr
makeDictVarchar(vector_size_t numRows, int dictSize, memory::MemoryPool* pool) {
  // Build stable string storage for the dictionary values.
  auto strings = std::make_shared<std::vector<std::string>>(dictSize);
  for (int i = 0; i < dictSize; ++i) {
    (*strings)[i] = fmt::format("value_{:06d}", i);
  }

  auto dictionary = BaseVector::create(VARCHAR(), dictSize, pool);
  auto* flat = dictionary->asFlatVector<StringView>();
  for (int i = 0; i < dictSize; ++i) {
    flat->set(i, StringView((*strings)[i]));
  }

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawIndices[i] = i % dictSize;
  }

  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, numRows, dictionary);
}

/// Builds a dictionary-encoded INTEGER column with the given cardinality.
VectorPtr
makeDictInteger(vector_size_t numRows, int dictSize, memory::MemoryPool* pool) {
  auto dictionary = BaseVector::create(INTEGER(), dictSize, pool);
  auto* flat = dictionary->asFlatVector<int32_t>();
  for (int i = 0; i < dictSize; ++i) {
    flat->set(i, i * 7);
  }

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawIndices[i] = i % dictSize;
  }

  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), indices, numRows, dictionary);
}

/// Builds a flat VARCHAR column (control case, no dictionary).
VectorPtr makeFlatVarchar(vector_size_t numRows, memory::MemoryPool* pool) {
  auto vector = BaseVector::create(VARCHAR(), numRows, pool);
  auto* flat = vector->asFlatVector<StringView>();
  auto strings = std::make_shared<std::vector<std::string>>(numRows);
  for (vector_size_t i = 0; i < numRows; ++i) {
    (*strings)[i] = fmt::format("value_{:06d}", i % 10);
  }
  for (vector_size_t i = 0; i < numRows; ++i) {
    flat->set(i, StringView((*strings)[i]));
  }
  return vector;
}

/// Builds a flat INTEGER column (control case).
VectorPtr makeFlatInteger(vector_size_t numRows, memory::MemoryPool* pool) {
  auto vector = BaseVector::create(INTEGER(), numRows, pool);
  auto* flat = vector->asFlatVector<int32_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    flat->set(i, static_cast<int32_t>(i));
  }
  return vector;
}

std::shared_ptr<memory::MemoryPool> rootPool;

// -- Dictionary VARCHAR benchmarks at various cardinalities --

void benchDictVarchar(int dictSize) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  auto column = makeDictVarchar(kNumRows, dictSize, leafPool.get());
  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW({"c0"}, {VARCHAR()}),
      BufferPtr(nullptr),
      kNumRows,
      std::vector<VectorPtr>{column});
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

void benchDictInteger(int dictSize) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  auto column = makeDictInteger(kNumRows, dictSize, leafPool.get());
  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW({"c0"}, {INTEGER()}),
      BufferPtr(nullptr),
      kNumRows,
      std::vector<VectorPtr>{column});
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
  auto column = makeFlatVarchar(kNumRows, leafPool.get());
  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW({"c0"}, {VARCHAR()}),
      BufferPtr(nullptr),
      kNumRows,
      std::vector<VectorPtr>{column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK(FlatInteger) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");
  auto column = makeFlatInteger(kNumRows, leafPool.get());
  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW({"c0"}, {INTEGER()}),
      BufferPtr(nullptr),
      kNumRows,
      std::vector<VectorPtr>{column});
  suspender.dismiss();
  writeParquet(data, rootPool.get());
}

BENCHMARK_DRAW_LINE();

// -- Multi-column benchmarks for selective flattening --
// These test the case where one column forces flattening (dict-of-dict) while
// other columns are passthrough dictionaries.  With blanket flattening, all
// columns get materialized.  With selective flattening, only the one that
// needs it is flattened.

/// Builds a dict-of-dict INTEGER column (forces flattening in needFlatten).
VectorPtr makeDictOfDictInteger(
    vector_size_t numRows,
    int dictSize,
    memory::MemoryPool* pool) {
  auto dictionary = BaseVector::create(INTEGER(), dictSize, pool);
  auto* flat = dictionary->asFlatVector<int32_t>();
  for (int i = 0; i < dictSize; ++i) {
    flat->set(i, i * 13);
  }

  BufferPtr innerIdx = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto rawInner = innerIdx->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawInner[i] = i % dictSize;
  }
  auto innerDict = BaseVector::wrapInDictionary(
      BufferPtr(nullptr), innerIdx, numRows, dictionary);

  BufferPtr outerIdx = AlignedBuffer::allocate<vector_size_t>(numRows, pool);
  auto rawOuter = outerIdx->asMutable<vector_size_t>();
  for (vector_size_t i = 0; i < numRows; ++i) {
    rawOuter[i] = i;
  }
  return BaseVector::wrapInDictionary(
      BufferPtr(nullptr), outerIdx, numRows, innerDict);
}

/// N passthrough dict VARCHAR columns + 1 dict-of-dict column that forces
/// flattening.  With blanket flattening all N+1 columns are flattened; with
/// selective flattening only the dict-of-dict column is.
void benchMixedColumns(int numPassthroughCols) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  // Passthrough dict VARCHAR columns (low cardinality).
  for (int i = 0; i < numPassthroughCols; ++i) {
    columns.push_back(makeDictVarchar(kNumRows, 10, leafPool.get()));
    names.push_back(fmt::format("dict_{}", i));
    types.push_back(VARCHAR());
  }

  // One dict-of-dict INTEGER column that forces flattening.
  columns.push_back(makeDictOfDictInteger(kNumRows, 50, leafPool.get()));
  names.push_back("nested");
  types.push_back(INTEGER());

  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW(std::move(names), std::move(types)),
      BufferPtr(nullptr),
      kNumRows,
      std::move(columns));

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

/// Control: N passthrough dict VARCHAR columns with NO column that forces
/// flattening.  Should show no difference between blanket and selective.
void benchAllPassthroughColumns(int numCols) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;
  std::vector<TypePtr> types;

  for (int i = 0; i < numCols; ++i) {
    columns.push_back(makeDictVarchar(kNumRows, 10, leafPool.get()));
    names.push_back(fmt::format("c{}", i));
    types.push_back(VARCHAR());
  }

  auto data = std::make_shared<RowVector>(
      leafPool.get(),
      ROW(std::move(names), std::move(types)),
      BufferPtr(nullptr),
      kNumRows,
      std::move(columns));

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

/// Writes numBatches batches of batchSize rows each to a single file.
void writeParquetMultiBatch(
    const RowVectorPtr& batch,
    int numBatches,
    memory::MemoryPool* rootPool) {
  auto leafPool = rootPool->addLeafChild("sink");
  auto sink = std::make_unique<MemorySink>(
      kSinkSize, FileSink::Options{.pool = leafPool.get()});
  WriterOptions options;
  options.memoryPool = rootPool;
  options.formatSpecificOptions = std::make_shared<ParquetWriterOptions>();
  auto writer = std::make_unique<parquet::Writer>(
      std::move(sink), options, asRowType(batch->type()));
  for (int i = 0; i < numBatches; ++i) {
    writer->write(batch);
  }
  writer->close();
}

void benchMultiBatch(int numCols, int numBatches) {
  folly::BenchmarkSuspender suspender;
  auto leafPool = rootPool->addLeafChild("bench");

  std::vector<VectorPtr> columns;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  constexpr vector_size_t kBatchSize = 10'000;

  for (int i = 0; i < numCols; ++i) {
    columns.push_back(makeDictVarchar(kBatchSize, 10, leafPool.get()));
    names.push_back(fmt::format("c{}", i));
    types.push_back(VARCHAR());
  }

  auto batch = std::make_shared<RowVector>(
      leafPool.get(),
      ROW(std::move(names), std::move(types)),
      BufferPtr(nullptr),
      kBatchSize,
      std::move(columns));

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

} // namespace

int32_t main(int32_t argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  rootPool = memory::memoryManager()->addRootPool("ParquetWriterBenchmark");
  folly::runBenchmarks();
  return 0;
}
