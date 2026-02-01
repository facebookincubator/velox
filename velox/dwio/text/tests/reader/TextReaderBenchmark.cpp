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

#include <fstream>
#include <iomanip>

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/text/reader/TextReader.h"

// Hardcoded production file path
constexpr const char* kProductionFile = "/home/user/large_text";

// Number of rows for synthetic data generation
constexpr int64_t kNumRows = 10000000;

using namespace facebook::velox;
using namespace facebook::velox::dwio::common;

namespace {

// Simple schema for opensource benchmark
RowTypePtr getSimpleSchema() {
  return ROW({
      {"bool_col", BOOLEAN()},
      {"int_col", INTEGER()},
      {"bigint_col", BIGINT()},
      {"real_col", REAL()},
      {"double_col", DOUBLE()},
      {"timestamp_col", TIMESTAMP()},
      {"string_col", VARCHAR()},
      {"binary_col", VARBINARY()},
  });
}

class TextReaderBenchmark {
 public:
  TextReaderBenchmark() {
    memory::MemoryManager::initialize({});
    rootPool_ = memory::memoryManager()->addRootPool("TextReaderBenchmark");
    leafPool_ = rootPool_->addLeafChild("leaf");
    filesystems::registerLocalFileSystem();
  }

  // Read entire file and return total rows read
  uint64_t readFile(
      const std::string& filePath,
      const RowTypePtr& schema,
      uint32_t batchSize = 10000) {
    dwio::common::ReaderOptions readerOptions{leafPool_.get()};
    readerOptions.setFileSchema(schema);
    auto fs = filesystems::getFileSystem(filePath, nullptr);
    auto input = std::make_unique<dwio::common::BufferedInput>(
        fs->openFileForRead(filePath), *leafPool_);

    auto reader = std::make_unique<facebook::velox::text::TextReader>(
        readerOptions, std::move(input));

    RowReaderOptions rowReaderOptions;
    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*schema);
    rowReaderOptions.setScanSpec(scanSpec);
    rowReaderOptions.select(
        std::make_shared<ColumnSelector>(schema, schema->names()));

    auto rowReader = reader->createRowReader(rowReaderOptions);
    auto result = BaseVector::create(schema, batchSize, leafPool_.get());

    uint64_t totalRows = 0;
    while (true) {
      auto rowsRead = rowReader->next(batchSize, result);
      if (rowsRead == 0) {
        break;
      }
      totalRows += rowsRead;
      // Ensure vectors are materialized
      for (auto i = 0; i < result->as<RowVector>()->childrenSize(); ++i) {
        result->as<RowVector>()->childAt(i)->loadedVector();
      }
      folly::doNotOptimizeAway(result->size());
    }
    return totalRows;
  }

  // Generate synthetic data file for opensource benchmark
  void generateSyntheticFile(
      const std::string& filePath,
      const RowTypePtr& /*schema*/,
      uint64_t numRows) {
    std::ofstream out(filePath);
    VELOX_CHECK(out.is_open(), "Failed to create file: {}", filePath);

    for (uint64_t row = 0; row < numRows; ++row) {
      // bool_col
      out << (row % 2 == 0 ? "true" : "false") << "\x01";
      // int_col
      out << (row % 1000) << "\x01";
      // bigint_col
      out << row << "\x01";
      // real_col
      out << (row * 1.5f) << "\x01";
      // double_col
      out << (row * 2.5) << "\x01";
      // timestamp_col
      out << "1970-01-01 00:00:" << std::setw(2) << std::setfill('0')
          << (row % 60) << ".000"
          << "\x01";
      // string_col
      out << "row_" << row << "_data"
          << "\x01";
      // binary_col (base64)
      out << "YmluYXJ5"
          << "\n";
    }
    out.close();
  }

  memory::MemoryPool* pool() {
    return leafPool_.get();
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

TextReaderBenchmark* getBenchmark() {
  static TextReaderBenchmark benchmark;
  return &benchmark;
}

std::string getSyntheticFilePath() {
  static std::string path;
  if (path.empty()) {
    path = "/tmp/text_reader_benchmark_synthetic.txt";
    getBenchmark()->generateSyntheticFile(path, getSimpleSchema(), kNumRows);
    LOG(INFO) << "Generated synthetic file with " << kNumRows
              << " rows at: " << path;
  }
  return path;
}

} // namespace

BENCHMARK_DRAW_LINE();

BENCHMARK(SyntheticFile_BatchSize_1k) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  auto schema = getSimpleSchema();
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 1000);
  folly::doNotOptimizeAway(rows);
}

BENCHMARK_RELATIVE(SyntheticFile_BatchSize_10k) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  auto schema = getSimpleSchema();
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 10000);
  folly::doNotOptimizeAway(rows);
}

BENCHMARK_RELATIVE(SyntheticFile_BatchSize_50k) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  auto schema = getSimpleSchema();
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 50000);
  folly::doNotOptimizeAway(rows);
}

BENCHMARK_DRAW_LINE();

// ============================================================================
// Column Projection Benchmark
// ============================================================================
// Tests the impact of reading fewer columns.
// ============================================================================

BENCHMARK(SyntheticFile_AllColumns) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  auto schema = getSimpleSchema();
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 10000);
  folly::doNotOptimizeAway(rows);
}

BENCHMARK_RELATIVE(SyntheticFile_HalfColumns) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  // Only select first 4 columns
  auto schema = ROW({
      {"bool_col", BOOLEAN()},
      {"int_col", INTEGER()},
      {"bigint_col", BIGINT()},
      {"real_col", REAL()},
  });
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 10000);
  folly::doNotOptimizeAway(rows);
}

BENCHMARK_RELATIVE(SyntheticFile_SingleColumn) {
  folly::BenchmarkSuspender suspender;
  auto* benchmark = getBenchmark();
  auto schema = ROW({{"bigint_col", BIGINT()}});
  auto filePath = getSyntheticFilePath();
  suspender.dismiss();

  auto rows = benchmark->readFile(filePath, schema, 10000);
  folly::doNotOptimizeAway(rows);
}

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  folly::runBenchmarks();
  return 0;
}