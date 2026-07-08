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

#include "velox/dwio/parquet/tests/ParquetTestBase.h"

#include "velox/common/Casts.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::parquet {
namespace {

dwio::common::ReaderOptions makeReaderOptions(
    memory::MemoryPool* pool,
    const std::shared_ptr<velox::io::IoStatistics>& dataIoStats,
    const std::shared_ptr<velox::io::IoStatistics>& metadataIoStats) {
  dwio::common::ReaderOptions opts(pool);
  opts.setDataIoStats(dataIoStats);
  opts.setMetadataIoStats(metadataIoStats);
  return opts;
}

dwio::common::RowReaderOptions makeRowReaderOptsWithSelector(
    const RowTypePtr& rowType) {
  dwio::common::RowReaderOptions rowReaderOpts;
  rowReaderOpts.select(
      std::make_shared<dwio::common::ColumnSelector>(
          rowType, rowType->names(), nullptr, false));
  return rowReaderOpts;
}

std::shared_ptr<velox::common::ScanSpec> makeScanSpec(
    const RowTypePtr& rowType) {
  auto scanSpec = std::make_shared<velox::common::ScanSpec>("");
  scanSpec->addAllChildFields(*rowType);
  return scanSpec;
}

} // namespace

ParquetReaderBuilder::ParquetReaderBuilder(
    memory::MemoryPool* pool,
    const std::shared_ptr<velox::io::IoStatistics>& dataIoStats,
    const std::shared_ptr<velox::io::IoStatistics>& metadataIoStats,
    const std::string& filePath,
    const RowTypePtr& outputType)
    : pool_(pool),
      dataIoStats_(dataIoStats),
      metadataIoStats_(metadataIoStats),
      filePath_(filePath),
      outputType_(outputType) {}

ParquetReaderBuilder::ParquetReaderBuilder(
    memory::MemoryPool* pool,
    const std::shared_ptr<velox::io::IoStatistics>& dataIoStats,
    const std::shared_ptr<velox::io::IoStatistics>& metadataIoStats,
    const dwio::common::MemorySink& buffer,
    const RowTypePtr& outputType)
    : pool_(pool),
      dataIoStats_(dataIoStats),
      metadataIoStats_(metadataIoStats),
      buffer_(&buffer),
      outputType_(outputType) {}

ParquetReaderBuilder& ParquetReaderBuilder::byteRange(
    uint64_t offset,
    uint64_t length) {
  byteRange_ = {offset, length};
  return *this;
}

ParquetReaderBuilder& ParquetReaderBuilder::withScanSpecOnly() {
  withScanSpecOnly_ = true;
  return *this;
}

ParquetReaderBuilder& ParquetReaderBuilder::options(
    const dwio::common::ReaderOptions& readerOptions) {
  readerOptions_ = readerOptions;
  return *this;
}

ParquetReaderBundle ParquetReaderBuilder::build() {
  VELOX_CHECK(
      filePath_.has_value() ^ (buffer_ != nullptr),
      "file path or in-memory buffer must be set");

  auto readerOpts = readerOptions_.value_or(
      makeReaderOptions(pool_, dataIoStats_, metadataIoStats_));
  std::unique_ptr<ParquetReader> reader;
  if (buffer_ != nullptr) {
    std::string data(buffer_->data(), buffer_->size());
    reader = std::make_unique<ParquetReader>(
        std::make_unique<dwio::common::BufferedInput>(
            std::make_shared<InMemoryReadFile>(std::move(data)),
            readerOpts.memoryPool()),
        readerOpts);
  } else {
    auto input = std::make_unique<dwio::common::BufferedInput>(
        std::make_shared<LocalReadFile>(*filePath_), readerOpts.memoryPool());
    reader = std::make_unique<ParquetReader>(std::move(input), readerOpts);
  }

  dwio::common::RowReaderOptions rowReaderOpts;
  if (!withScanSpecOnly_) {
    rowReaderOpts = makeRowReaderOptsWithSelector(outputType_);
  }
  rowReaderOpts.setScanSpec(makeScanSpec(outputType_));
  if (byteRange_.has_value()) {
    rowReaderOpts.range(byteRange_->first, byteRange_->second);
  }
  auto rowReader = reader->createRowReader(rowReaderOpts);
  return {std::move(reader), std::move(rowReader)};
}

dwio::common::ReaderOptions ParquetTestBase::makeDefaultReaderOptions() const {
  return makeReaderOptions(leafPool_.get(), dataIoStats_, metadataIoStats_);
}

std::unique_ptr<facebook::velox::parquet::ParquetReader>
ParquetTestBase::createReader(
    const std::string& fileName,
    const dwio::common::ReaderOptions& opts) {
  const auto path = getExampleFilePath(fileName);
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(path), opts.memoryPool());
  return std::make_unique<facebook::velox::parquet::ParquetReader>(
      std::move(input), opts);
}

void ParquetTestBase::assertEqualVectorPart(
    const VectorPtr& expected,
    const VectorPtr& actual,
    vector_size_t offset) {
  ASSERT_GE(expected->size(), actual->size() + offset);
  ASSERT_EQ(expected->typeKind(), actual->typeKind());
  for (vector_size_t i = 0; i < actual->size(); i++) {
    ASSERT_TRUE(expected->equalValueAt(actual.get(), i + offset, i))
        << "at " << (i + offset) << ": expected "
        << expected->toString(i + offset) << ", but got "
        << actual->toString(i);
  }
}

void ParquetTestBase::assertReadWithReaderAndExpected(
    std::shared_ptr<const RowType> outputType,
    dwio::common::RowReader& reader,
    RowVectorPtr expected,
    memory::MemoryPool& memoryPool) {
  uint64_t total = 0;
  VectorPtr result = BaseVector::create(outputType, 0, &memoryPool);
  while (total < expected->size()) {
    auto part = reader.next(1000, result);
    if (part > 0) {
      assertEqualVectorPart(expected, result, total);
      total += result->size();
    } else {
      break;
    }
  }
  EXPECT_EQ(total, expected->size());
  EXPECT_EQ(reader.next(1000, result), 0);
}

void ParquetTestBase::assertReadWithReaderAndFilters(
    dwio::common::Reader& reader,
    const RowTypePtr& fileSchema,
    FilterMap filters,
    const RowVectorPtr& expected) {
  auto scanSpec = makeScanSpec(fileSchema);
  for (auto&& [column, filter] : filters) {
    scanSpec->getOrCreateChild(velox::common::Subfield(column))
        ->setFilter(std::move(filter));
  }

  auto rowReaderOpts = makeRowReaderOpts(fileSchema);
  rowReaderOpts.setScanSpec(scanSpec);
  auto rowReader = reader.createRowReader(rowReaderOpts);
  assertReadWithReaderAndExpected(fileSchema, *rowReader, expected, *leafPool_);
}

std::unique_ptr<dwio::common::FileSink> ParquetTestBase::createSink(
    const std::string& filePath) {
  auto sink = dwio::common::FileSink::create(
      fmt::format("file:{}", filePath), {.pool = rootPool_.get()});
  EXPECT_TRUE(sink->isBuffered());
  EXPECT_TRUE(fs::exists(filePath));
  EXPECT_FALSE(sink->isClosed());
  return sink;
}

std::unique_ptr<facebook::velox::parquet::Writer> ParquetTestBase::createWriter(
    std::unique_ptr<dwio::common::FileSink> sink,
    std::function<
        std::unique_ptr<facebook::velox::parquet::DefaultFlushPolicy>()>
        flushPolicy,
    const RowTypePtr& rowType,
    facebook::velox::common::CompressionKind compressionKind) {
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  options.flushPolicyFactory = flushPolicy;
  options.compressionKind = compressionKind;
  return std::make_unique<facebook::velox::parquet::Writer>(
      std::move(sink), options, rowType);
}

std::vector<RowVectorPtr> ParquetTestBase::createBatches(
    const RowTypePtr& rowType,
    uint64_t numBatches,
    uint64_t vectorSize) {
  std::vector<RowVectorPtr> batches;
  batches.reserve(numBatches);
  VectorFuzzer fuzzer({.vectorSize = vectorSize}, leafPool_.get());
  for (auto i = 0; i < numBatches; ++i) {
    batches.emplace_back(fuzzer.fuzzInputFlatRow(rowType));
  }
  return batches;
}

dwio::common::MemorySink* ParquetTestBase::write(
    const RowVectorPtr& data,
    const ParquetWriterOptions& writerOptions,
    const RowTypePtr& rowType) {
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  return write(data, options, writerOptions, rowType);
}

dwio::common::MemorySink* ParquetTestBase::write(
    const RowVectorPtr& data,
    const dwio::common::WriterOptions& options,
    const ParquetWriterOptions& writerOptions,
    const RowTypePtr& rowType) {
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  auto writerOptionsBase = options;
  if (writerOptionsBase.memoryPool == nullptr) {
    writerOptionsBase.memoryPool = rootPool_.get();
  }
  writerOptionsBase.formatSpecificOptions =
      std::make_shared<ParquetWriterOptions>(writerOptions);
  auto writer = std::make_unique<Writer>(
      std::move(sink),
      writerOptionsBase,
      rowType != nullptr ? rowType : data->rowType());
  writer->write(data);
  writer->close();
  writers_.push_back(std::move(writer));
  return sinkPtr;
}

dwio::common::MemorySink* ParquetTestBase::write(
    const std::vector<RowVectorPtr>& batches,
    const dwio::common::WriterOptions& options,
    const ParquetWriterOptions& writerOptions) {
  VELOX_CHECK(!batches.empty());
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  auto writerOptionsBase = options;
  writerOptionsBase.formatSpecificOptions =
      std::make_shared<ParquetWriterOptions>(writerOptions);
  auto writer = std::make_unique<Writer>(
      std::move(sink), writerOptionsBase, batches[0]->rowType());
  for (size_t i = 0; i < batches.size(); ++i) {
    writer->write(batches[i]);
  }
  writer->close();
  writers_.push_back(std::move(writer));
  return sinkPtr;
}

dwio::common::MemorySink* ParquetTestBase::write(
    const RowVectorPtr& data,
    std::unordered_map<std::string, std::string> configFromFile,
    std::unordered_map<std::string, std::string> sessionProperties) {
  dwio::common::WriterOptions options;
  options.memoryPool = rootPool_.get();
  ParquetWriterFactory factory;
  auto writerOptions =
      checkedPointerCast<ParquetWriterOptions>(factory.createFormatOptions(
          config::ConfigBase(std::move(configFromFile)),
          config::ConfigBase(std::move(sessionProperties))));
  return write(data, options, *writerOptions);
}

std::unique_ptr<ParquetReader> ParquetTestBase::createReaderInMemory(
    const dwio::common::MemorySink& sink,
    const dwio::common::ReaderOptions& opts) {
  std::string data(sink.data(), sink.size());
  return std::make_unique<ParquetReader>(
      std::make_unique<dwio::common::BufferedInput>(
          std::make_shared<InMemoryReadFile>(std::move(data)),
          opts.memoryPool()),
      opts);
}

std::unique_ptr<dwio::common::RowReader>
ParquetTestBase::createRowReaderFromReader(
    dwio::common::Reader& reader,
    const RowTypePtr& rowType,
    bool useColumnSelector) {
  dwio::common::RowReaderOptions rowReaderOpts;
  if (useColumnSelector) {
    rowReaderOpts = makeRowReaderOpts(rowType);
  }
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  return reader.createRowReader(rowReaderOpts);
}

} // namespace facebook::velox::parquet
