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

// Regression coverage for the flat-map passthrough schema-creation path.
//
// FlatMapFieldWriter's passthrough new-field path
// (createPassthroughValueFieldWriter) mints a value field for a new key, which
// mutates shared FieldWriterContext state: FieldWriter::create appends to the
// shared streams_ and schemaBuilder_, and handleFlatmapFieldAddEvent updates
// shared context. The passthrough and injected-key paths use the same critical
// section for these mutations.
//
// The input is many flat-map columns with disjoint key sets, all keys present
// in the first batch, so the first write() mints every passthrough value field.
// Reading back must return every row; a corrupt stripe/stream index throws or
// short-counts.

#include <folly/executors/GlobalExecutor.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/velox/VeloxReader.h"
#include "velox/dwio/nimble/writer/Writer.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;

namespace {

constexpr int kNumColumns = 16;
constexpr int kNumRows = 100;
constexpr int kKeysPerColumn = 8;
constexpr int kBatches = 4;
constexpr int kIterations = 50;

class NimbleWriterPassthroughFlatmapRaceTest
    : public ::testing::Test,
      public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::Options options;
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  // Builds one column's input as a velox FlatMapVector (so the writer takes the
  // passthrough path). Keys are disjoint per column but present in every row of
  // every batch, so the flat map is dense and reads back cleanly; all new
  // passthrough keys are minted during the first batch's parallel column write.
  velox::VectorPtr makeFlatMapColumn(int column) {
    std::vector<std::vector<std::pair<int32_t, std::optional<int64_t>>>> maps;
    maps.reserve(kNumRows);
    const int keyBase = column * 1'000;
    for (int row = 0; row < kNumRows; ++row) {
      std::vector<std::pair<int32_t, std::optional<int64_t>>> mapRow;
      mapRow.reserve(kKeysPerColumn);
      for (int k = 0; k < kKeysPerColumn; ++k) {
        mapRow.emplace_back(keyBase + k, static_cast<int64_t>(row + k));
      }
      maps.push_back(std::move(mapRow));
    }
    return makeFlatMapVector<int32_t, int64_t>(maps);
  }

  velox::RowVectorPtr makeBatch() {
    std::vector<velox::VectorPtr> columns;
    columns.reserve(kNumColumns);
    for (int column = 0; column < kNumColumns; ++column) {
      columns.push_back(makeFlatMapColumn(column));
    }
    return makeRowVector(columns);
  }

  nimble::WriterOptions makeOptions() {
    nimble::WriterOptions options;
    for (int column = 0; column < kNumColumns; ++column) {
      options.flatMapColumns["c" + folly::to<std::string>(column)];
    }
    // Encode streams on the shared process-global executor.
    options.encodingExecutor = folly::getGlobalCPUExecutor();
    options.maxEncodeParallelism = 8;
    options.minStreamsPerEncodingTask = 1;
    return options;
  }
};

TEST_F(NimbleWriterPassthroughFlatmapRaceTest, passthroughNewKeysRoundTrip) {
  const auto schema = makeBatch()->type();
  uint32_t corrupt = 0;

  for (int iter = 0; iter < kIterations; ++iter) {
    auto writerRoot = velox::memory::memoryManager()->addRootPool("");
    std::string file;
    uint64_t expectedRows = 0;
    {
      nimble::Writer writer(
          schema,
          std::make_unique<velox::InMemoryWriteFile>(&file),
          *writerRoot,
          makeOptions());
      for (int batch = 0; batch < kBatches; ++batch) {
        auto row = makeBatch();
        expectedRows += row->size();
        writer.write(row);
      }
      writer.close();
    }

    // Read back: a corrupt stripe/stream index throws; a short/over count means
    // silent corruption.
    try {
      auto readerRoot = velox::memory::memoryManager()->addRootPool("");
      auto readerLeaf = readerRoot->addLeafChild("reader");
      auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
      nimble::VeloxReader reader(readFile.get(), *readerLeaf);
      uint64_t rows = 0;
      velox::VectorPtr result;
      while (reader.next(1'000, result)) {
        rows += result->size();
      }
      if (rows != expectedRows) {
        ++corrupt;
        ADD_FAILURE() << "iter " << iter << " expected " << expectedRows
                      << " rows, read " << rows;
      }
    } catch (const std::exception& e) {
      ++corrupt;
      ADD_FAILURE() << "iter " << iter << " read failed: " << e.what();
    }
  }
  EXPECT_EQ(corrupt, 0u);
}

} // namespace
