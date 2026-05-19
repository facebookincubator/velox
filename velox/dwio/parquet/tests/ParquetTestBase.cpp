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

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::parquet {

ParquetTestReaderBuilder::ParquetTestReaderBuilder(ParquetTestBase* testBase)
    : testBase_(testBase) {}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::file(
    const std::string& fileName) {
  fileName_ = fileName;
  sink_ = nullptr;
  return *this;
}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::sink(
    const dwio::common::MemorySink& sink) {
  sink_ = &sink;
  fileName_.reset();
  return *this;
}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::schema(
    const RowTypePtr& rowType) {
  schema_ = rowType;
  return *this;
}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::range(
    uint64_t offset,
    uint64_t length) {
  range_ = {offset, length};
  return *this;
}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::noColumnSelector() {
  noColumnSelector_ = true;
  return *this;
}

ParquetTestReaderBuilder& ParquetTestReaderBuilder::readerOptions(
    const dwio::common::ReaderOptions& readerOptions) {
  readerOptions_ = readerOptions;
  return *this;
}

std::pair<
    std::unique_ptr<ParquetReader>,
    std::unique_ptr<dwio::common::RowReader>>
ParquetTestReaderBuilder::build() {
  VELOX_CHECK(schema_ != nullptr, "schema is required");
  VELOX_CHECK(
      fileName_.has_value() ^ (sink_ != nullptr),
      "either file or sink must be set");

  auto readerOpts =
      readerOptions_.value_or(testBase_->makeDefaultReaderOptions());
  std::unique_ptr<ParquetReader> reader;
  if (sink_ != nullptr) {
    reader = testBase_->createReaderInMemory(*sink_, readerOpts);
  } else {
    reader = testBase_->createReader(*fileName_, readerOpts);
  }

  dwio::common::RowReaderOptions rowReaderOpts;
  if (!noColumnSelector_) {
    rowReaderOpts = testBase_->makeRowReaderOpts(schema_);
  }
  rowReaderOpts.setScanSpec(testBase_->makeScanSpec(schema_));
  if (range_.has_value()) {
    rowReaderOpts.range(range_->first, range_->second);
  }
  auto rowReader = reader->createRowReader(rowReaderOpts);
  return {std::move(reader), std::move(rowReader)};
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
  facebook::velox::parquet::WriterOptions options;
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
    const WriterOptions& writerOptions,
    const RowTypePtr& rowType) {
  auto sink = std::make_unique<dwio::common::MemorySink>(
      200 * 1024 * 1024,
      dwio::common::FileSink::Options{.pool = leafPool_.get()});
  auto* sinkPtr = sink.get();
  auto writer = std::make_unique<Writer>(
      std::move(sink),
      writerOptions,
      rowType != nullptr ? rowType : data->rowType());
  writer->write(data);
  writer->close();
  writers_.push_back(std::move(writer));
  return sinkPtr;
}

dwio::common::MemorySink* ParquetTestBase::write(
    const RowVectorPtr& data,
    std::unordered_map<std::string, std::string> configFromFile,
    std::unordered_map<std::string, std::string> sessionProperties) {
  parquet::WriterOptions writerOptions;
  writerOptions.memoryPool = rootPool_.get();
  auto connectorConfig = config::ConfigBase(std::move(configFromFile));
  auto connectorSessionProperties =
      config::ConfigBase(std::move(sessionProperties));
  writerOptions.processConfigs(connectorConfig, connectorSessionProperties);
  return write(data, writerOptions);
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
    const RowTypePtr& rowType) {
  auto rowReaderOpts = makeRowReaderOpts(rowType);
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  return reader.createRowReader(rowReaderOpts);
}

std::unique_ptr<dwio::common::RowReader>
ParquetTestBase::createRowReaderFromReaderNoSelect(
    dwio::common::Reader& reader,
    const RowTypePtr& rowType) {
  dwio::common::RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  return reader.createRowReader(rowReaderOpts);
}

} // namespace facebook::velox::parquet
