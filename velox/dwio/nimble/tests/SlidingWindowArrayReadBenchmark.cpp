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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

DEFINE_int32(num_rows, 1'000'000, "Rows in each benchmark file.");
DEFINE_int32(
    window_size,
    32,
    "Values in each row. Each row slides the window of the previous row by "
    "one value.");
DEFINE_int32(batch_size, 10'000, "Rows in each write and read batch.");

using namespace facebook;

namespace {

// Holds the files the benchmarks read and the pools that read them.
struct Fixture {
  std::shared_ptr<velox::memory::MemoryPool> rootPool;
  std::shared_ptr<velox::memory::MemoryPool> leafPool;
  // Indexed by [withNulls][slidingWindow].
  std::string files[2][2];
};

std::unique_ptr<Fixture> fixture;

// Builds rows of BIGINT arrays. Row i holds the values [i, i + window_size), so
// each row slides the previous one by one value. With 'withNulls', every tenth
// row is null.
velox::RowVectorPtr makeRows(bool withNulls, velox::memory::MemoryPool* pool) {
  const velox::vector_size_t numRows = FLAGS_num_rows;
  const velox::vector_size_t windowSize = FLAGS_window_size;
  auto elements = velox::BaseVector::create<velox::FlatVector<int64_t>>(
      velox::BIGINT(), numRows + windowSize, pool);
  for (velox::vector_size_t i = 0; i < numRows + windowSize; ++i) {
    elements->set(i, i);
  }
  auto offsets = velox::allocateOffsets(numRows, pool);
  auto sizes = velox::allocateSizes(numRows, pool);
  auto* rawOffsets = offsets->asMutable<velox::vector_size_t>();
  auto* rawSizes = sizes->asMutable<velox::vector_size_t>();
  for (velox::vector_size_t i = 0; i < numRows; ++i) {
    rawOffsets[i] = i;
    rawSizes[i] = windowSize;
  }
  velox::BufferPtr nulls;
  if (withNulls) {
    nulls = velox::allocateNulls(numRows, pool);
    auto* rawNulls = nulls->asMutable<uint64_t>();
    for (velox::vector_size_t i = 0; i < numRows; i += 10) {
      velox::bits::setNull(rawNulls, i);
    }
  }
  auto arrays = std::make_shared<velox::ArrayVector>(
      pool,
      velox::ARRAY(velox::BIGINT()),
      std::move(nulls),
      numRows,
      std::move(offsets),
      std::move(sizes),
      std::move(elements));
  return std::make_shared<velox::RowVector>(
      pool,
      velox::ROW({{"events", velox::ARRAY(velox::BIGINT())}}),
      /*nulls=*/nullptr,
      numRows,
      std::vector<velox::VectorPtr>{std::move(arrays)});
}

// Writes 'rows' to an in-memory Nimble file, as sliding window arrays when
// 'slidingWindow' is set and as plain arrays otherwise.
std::string writeFile(
    const velox::RowVectorPtr& rows,
    bool slidingWindow,
    velox::memory::MemoryPool& rootPool) {
  nimble::WriterOptions options;
  if (slidingWindow) {
    options.slidingWindowArrayColumns.insert("events");
  }
  std::string file;
  nimble::Writer writer(
      rows->type(),
      std::make_unique<velox::InMemoryWriteFile>(&file),
      rootPool,
      std::move(options));
  for (velox::vector_size_t offset = 0; offset < rows->size();
       offset += FLAGS_batch_size) {
    writer.write(rows->slice(
        offset,
        std::min<velox::vector_size_t>(
            FLAGS_batch_size, rows->size() - offset)));
  }
  writer.close();
  return file;
}

// Reads every row of a benchmark file in batches of 'batch_size' rows.
void readFile(bool withNulls, bool slidingWindow) {
  folly::BenchmarkSuspender suspender;
  auto file = std::make_shared<velox::InMemoryReadFile>(
      fixture->files[withNulls][slidingWindow]);
  nimble::BatchReader reader{file, *fixture->leafPool};
  velox::VectorPtr result;
  suspender.dismiss();
  while (reader.next(FLAGS_batch_size, result)) {
    folly::doNotOptimizeAway(result);
  }
}

} // namespace

BENCHMARK(arrayRead) {
  readFile(/*withNulls=*/false, /*slidingWindow=*/false);
}

BENCHMARK_RELATIVE(slidingWindowArrayRead) {
  readFile(/*withNulls=*/false, /*slidingWindow=*/true);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(arrayReadWithNulls) {
  readFile(/*withNulls=*/true, /*slidingWindow=*/false);
}

BENCHMARK_RELATIVE(slidingWindowArrayReadWithNulls) {
  readFile(/*withNulls=*/true, /*slidingWindow=*/true);
}

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  velox::memory::MemoryManager::initialize({});
  fixture = std::make_unique<Fixture>();
  fixture->rootPool = velox::memory::memoryManager()->addRootPool(
      "sliding_window_array_read_benchmark");
  fixture->leafPool = fixture->rootPool->addLeafChild("reads");
  for (const bool withNulls : {false, true}) {
    const auto rows = makeRows(withNulls, fixture->leafPool.get());
    for (const bool slidingWindow : {false, true}) {
      fixture->files[withNulls][slidingWindow] =
          writeFile(rows, slidingWindow, *fixture->rootPool);
    }
  }
  folly::runBenchmarks();
  fixture.reset();
  return 0;
}
