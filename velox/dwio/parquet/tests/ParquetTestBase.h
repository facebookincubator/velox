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

#pragma once

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include "velox/common/base/Fs.h"
#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/tests/utils/DataFiles.h"
#include "velox/dwio/parquet/reader/PageReader.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::parquet {
using TempDirectoryPath = common::testutil::TempDirectoryPath;

class ParquetTestBase;

/// Owns a ParquetReader and RowReader created for a single read test.
/// Declares reader before rowReader so C++ destroys rowReader first.
struct ParquetReaderBundle {
  /// Owns file metadata and input state required by rowReader.
  std::unique_ptr<ParquetReader> reader;

  /// Reads rows from reader.
  std::unique_ptr<dwio::common::RowReader> rowReader;
};

/// Builder for creating ParquetReader + RowReader bundles in unit tests.
///
/// Construct via ParquetTestBase::readerBuilder with a data source (example
/// file or in-memory buffer) and output projection, then apply optional
/// overrides.
class ParquetReaderBuilder {
 public:
  /// Reads Parquet from an on-disk file at filePath (a resolved path, e.g. from
  /// getExampleFilePath()) and projects outputType.
  ParquetReaderBuilder(
      memory::MemoryPool* pool,
      const std::shared_ptr<velox::io::IoStatistics>& dataIoStats,
      const std::shared_ptr<velox::io::IoStatistics>& metadataIoStats,
      const std::string& filePath,
      const RowTypePtr& outputType);

  /// Reads Parquet from bytes held in buffer (e.g. a MemorySink from write())
  /// and projects outputType.
  ParquetReaderBuilder(
      memory::MemoryPool* pool,
      const std::shared_ptr<velox::io::IoStatistics>& dataIoStats,
      const std::shared_ptr<velox::io::IoStatistics>& metadataIoStats,
      const dwio::common::MemorySink& buffer,
      const RowTypePtr& outputType);

  /// Limits the read to byte range [offset, offset + length) in the file.
  /// Use for split-level / slice read tests.
  ParquetReaderBuilder& byteRange(uint64_t offset, uint64_t length);

  /// Configures RowReaderOptions with only a ScanSpec and no ColumnSelector.
  /// By default, build() adds a ColumnSelector derived from the output type
  /// passed to readerBuilder(). Use
  /// this when tests should not select output columns explicitly—for example
  /// empty projection (count(*)), metadata-only reads, or reading via
  /// ParquetReader::rowType() without a predetermined projection.
  ParquetReaderBuilder& withScanSpecOnly();

  /// Supplies ReaderOptions instead of the default (leaf pool + IoStatistics).
  ParquetReaderBuilder& options(
      const dwio::common::ReaderOptions& readerOptions);

  /// Creates reader handles from the configured inputs.
  ParquetReaderBundle build();

 private:
  memory::MemoryPool* pool_;
  std::shared_ptr<velox::io::IoStatistics> dataIoStats_;
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_;
  std::optional<std::string> filePath_;
  const dwio::common::MemorySink* buffer_{nullptr};
  RowTypePtr outputType_;
  std::optional<std::pair<uint64_t, uint64_t>> byteRange_;
  bool withScanSpecOnly_{false};
  std::optional<dwio::common::ReaderOptions> readerOptions_;
};

class ParquetTestBase : public testing::Test,
                        public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    dwio::common::LocalFileSink::registerFactory();
    rootPool_ = memory::memoryManager()->addRootPool("ParquetTests");
    leafPool_ = rootPool_->addLeafChild("ParquetTests");
    tempPath_ = TempDirectoryPath::create();
  }

  static RowTypePtr sampleSchema() {
    return ROW({"a", "b"}, {BIGINT(), DOUBLE()});
  }

  static RowTypePtr dateSchema() {
    return ROW("date", DATE());
  }

  static RowTypePtr intSchema() {
    return ROW({"int", "bigint"}, {INTEGER(), BIGINT()});
  }

  static RowTypePtr upperSchemaToLowerCase() {
    return ROW({"a", "b"}, {BIGINT(), BIGINT()});
  }

  std::unique_ptr<facebook::velox::parquet::ParquetReader> createReader(
      const std::string& fileName,
      const dwio::common::ReaderOptions& opts);

  std::unique_ptr<facebook::velox::parquet::ParquetReader> createReader(
      const std::string& fileName) {
    return createReader(fileName, makeDefaultReaderOptions());
  }

  // Creates a ReaderOptions pre-populated with the test's shared IoStatistics
  // and the leaf memory pool. Callers needing extra settings should obtain a
  // copy via this method and then apply their overrides.
  dwio::common::ReaderOptions makeDefaultReaderOptions() const;

  // Returns RowReaderOptions with a ColumnSelector for the given schema.
  dwio::common::RowReaderOptions makeRowReaderOpts(
      const RowTypePtr& rowType,
      bool fileColumnNamesReadAsLowerCase = false) {
    dwio::common::RowReaderOptions rowReaderOpts;
    rowReaderOpts.select(
        std::make_shared<facebook::velox::dwio::common::ColumnSelector>(
            rowType,
            rowType->names(),
            nullptr,
            fileColumnNamesReadAsLowerCase));
    return rowReaderOpts;
  }

  std::shared_ptr<velox::common::ScanSpec> makeScanSpec(
      const RowTypePtr& rowType) {
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
    scanSpec->addAllChildFields(*rowType);
    return scanSpec;
  }

  using FilterMap =
      std::unordered_map<std::string, std::unique_ptr<velox::common::Filter>>;

  void assertEqualVectorPart(
      const VectorPtr& expected,
      const VectorPtr& actual,
      vector_size_t offset);

  void assertReadWithReaderAndExpected(
      std::shared_ptr<const RowType> outputType,
      dwio::common::RowReader& reader,
      RowVectorPtr expected,
      memory::MemoryPool& memoryPool);

  void assertReadWithReaderAndFilters(
      dwio::common::Reader& reader,
      const RowTypePtr& fileSchema,
      FilterMap filters,
      const RowVectorPtr& expected);

  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath);

  std::unique_ptr<facebook::velox::parquet::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      std::function<
          std::unique_ptr<facebook::velox::parquet::DefaultFlushPolicy>()>
          flushPolicy,
      const RowTypePtr& rowType,
      facebook::velox::common::CompressionKind compressionKind =
          facebook::velox::common::CompressionKind_NONE);

  std::vector<RowVectorPtr> createBatches(
      const RowTypePtr& rowType,
      uint64_t numBatches,
      uint64_t vectorSize);

  std::string getExampleFilePath(const std::string& fileName) {
    return test::getDataFilePath(
        "velox/dwio/parquet/tests/reader", "../examples/" + fileName);
  }

  dwio::common::MemorySink* write(
      const RowVectorPtr& data,
      const ParquetWriterOptions& writerOptions,
      const RowTypePtr& rowType = nullptr);

  dwio::common::MemorySink* write(
      const RowVectorPtr& data,
      const dwio::common::WriterOptions& options,
      const ParquetWriterOptions& writerOptions,
      const RowTypePtr& rowType = nullptr);

  /// Writes batches data into Parquet file.
  /// Uses batches[0]->rowType() as the file schema. Configure
  /// WriterOptions (e.g. flushPolicyFactory) so batch sizes are not split
  /// further by the writer.
  dwio::common::MemorySink* write(
      const std::vector<RowVectorPtr>& batches,
      const dwio::common::WriterOptions& options,
      const ParquetWriterOptions& writerOptions);

  dwio::common::MemorySink* write(
      const RowVectorPtr& data,
      std::unordered_map<std::string, std::string> configFromFile = {},
      std::unordered_map<std::string, std::string> sessionProperties = {});

  std::unique_ptr<ParquetReader> createReaderInMemory(
      const dwio::common::MemorySink& sink,
      const dwio::common::ReaderOptions& opts);

  std::unique_ptr<ParquetReader> createReaderInMemory(
      const dwio::common::MemorySink& sink) {
    return createReaderInMemory(sink, makeDefaultReaderOptions());
  }

  /// Creates a RowReader for reader. When useColumnSelector is true (default),
  /// RowReaderOptions include a ColumnSelector for rowType;
  /// otherwise only a ScanSpec is set (see
  /// ParquetReaderBuilder::withScanSpecOnly).
  std::unique_ptr<dwio::common::RowReader> createRowReaderFromReader(
      dwio::common::Reader& reader,
      const RowTypePtr& rowType,
      bool useColumnSelector = true);

  /// exampleFileName is a file under the parquet reader examples directory.
  ParquetReaderBuilder readerBuilder(
      const std::string& exampleFileName,
      const RowTypePtr& outputType) {
    return ParquetReaderBuilder(
        leafPool_.get(),
        dataIoStats_,
        metadataIoStats_,
        getExampleFilePath(exampleFileName),
        outputType);
  }

  ParquetReaderBuilder readerBuilder(
      const dwio::common::MemorySink& buffer,
      const RowTypePtr& outputType) {
    return ParquetReaderBuilder(
        leafPool_.get(), dataIoStats_, metadataIoStats_, buffer, outputType);
  }

  static constexpr uint64_t kRowsInRowGroup = 10'000;
  static constexpr uint64_t kBytesInRowGroup = 128 * 1'024 * 1'024;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
  std::shared_ptr<velox::io::IoStatistics> dataIoStats_ =
      std::make_shared<velox::io::IoStatistics>();
  std::shared_ptr<velox::io::IoStatistics> metadataIoStats_ =
      std::make_shared<velox::io::IoStatistics>();
  std::shared_ptr<TempDirectoryPath> tempPath_;
  // Stores writers created by write() helper to keep sinks alive for reading.
  std::vector<std::unique_ptr<Writer>> writers_;
};
} // namespace facebook::velox::parquet
