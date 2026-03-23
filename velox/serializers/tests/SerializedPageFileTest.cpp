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
#include "velox/serializers/SerializedPageFile.h"
#include <boost/random/uniform_int_distribution.hpp>
#include <folly/Random.h>
#include <gtest/gtest.h>
#include <vector>
#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/expression/fuzzer/FuzzerToolkit.h"
#include "velox/serializers/CompactRowSerializer.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/UnsafeRowSerializer.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::serializer;
using namespace facebook::velox::common::testutil;

enum class SerdeType { kPresto, kCompactRow, kUnsafeRow };

struct TestParams {
  SerdeType serdeType;
  common::CompressionKind compressionKind;
};

class SerializedPageFileTest : public ::testing::TestWithParam<TestParams>,
                               public VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    const auto& params = GetParam();

    deregisterVectorSerde();
    deregisterNamedVectorSerde("Presto");
    deregisterNamedVectorSerde("CompactRow");
    deregisterNamedVectorSerde("UnsafeRow");

    setupSerde(params.serdeType);
    filesystems::registerLocalFileSystem();
    tempDirPath_ = TempDirectoryPath::create();
    compressionKind_ = params.compressionKind;
  }

  void TearDown() override {
    deregisterVectorSerde();
    deregisterNamedVectorSerde("Presto");
    deregisterNamedVectorSerde("CompactRow");
    deregisterNamedVectorSerde("UnsafeRow");
  }

  void setupSerde(SerdeType serdeType) {
    switch (serdeType) {
      case SerdeType::kPresto:
        serializer::presto::PrestoVectorSerde::registerVectorSerde();
        serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
        serde_ = getNamedVectorSerde("Presto");
        break;
      case SerdeType::kCompactRow:
        serializer::CompactRowVectorSerde::registerVectorSerde();
        serializer::CompactRowVectorSerde::registerNamedVectorSerde();
        serde_ = getNamedVectorSerde("CompactRow");
        break;
      case SerdeType::kUnsafeRow:
        serializer::spark::UnsafeRowVectorSerde::registerVectorSerde();
        serializer::spark::UnsafeRowVectorSerde::registerNamedVectorSerde();
        serde_ = getNamedVectorSerde("UnsafeRow");
        break;
    }
  }

  std::unique_ptr<VectorSerde::Options> createSerdeOptions() {
    const auto& params = GetParam();
    switch (params.serdeType) {
      case SerdeType::kPresto:
        return std::make_unique<
            serializer::presto::PrestoVectorSerde::PrestoOptions>(
            false, // useLosslessTimestamp
            compressionKind_, // compressionKind
            0.8, // minCompressionRatio
            false, // nullsFirst
            false); // preserveEncodings
      case SerdeType::kCompactRow:
      case SerdeType::kUnsafeRow:
        return std::make_unique<VectorSerde::Options>(compressionKind_, 0.8);
    }
    return nullptr;
  }

  RowVectorPtr createTestVector(int32_t numRows) {
    VectorFuzzer fuzzer({.vectorSize = static_cast<size_t>(numRows)}, pool());
    auto rowType = ROW({"a", "b", "c"}, {BIGINT(), VARCHAR(), DOUBLE()});
    return fuzzer.fuzzRow(rowType);
  }

  std::string getTestFilePath(const std::string& name = "test") {
    return fmt::format("{}/{}", tempDirPath_->getPath(), name);
  }

  std::unique_ptr<folly::IOBuf> createTestIOBuf(
      const std::string& data = "test_data") {
    return folly::IOBuf::copyBuffer(data);
  }

  VectorSerde* serde_{nullptr};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;
  common::CompressionKind compressionKind_;
};

TEST_P(SerializedPageFileTest, serializedPageFileBasic) {
  const std::string pathPrefix = getTestFilePath();
  const uint32_t kFileId = 123;

  auto pageFile =
      SerializedPageFile::create(kFileId, pathPrefix, "", /*ioStats=*/nullptr);

  EXPECT_EQ(pageFile->id(), kFileId);
  EXPECT_TRUE(pageFile->path().find(pathPrefix) == 0);
  EXPECT_EQ(pageFile->size(), 0);
  EXPECT_NE(pageFile->file(), nullptr);

  auto iobuf = createTestIOBuf("test data for file");
  const uint64_t dataSize = iobuf->computeChainDataLength();
  const uint64_t written = pageFile->write(std::move(iobuf));

  EXPECT_EQ(written, dataSize);
  EXPECT_EQ(pageFile->size(), dataSize);

  auto fileInfo = pageFile->fileInfo();
  EXPECT_EQ(fileInfo.id, kFileId);
  EXPECT_EQ(fileInfo.path, pageFile->path());
  EXPECT_EQ(fileInfo.size, dataSize);

  pageFile->finish();
  EXPECT_EQ(pageFile->size(), dataSize);
  EXPECT_EQ(pageFile->file(), nullptr);
}

TEST_P(SerializedPageFileTest, serializedPageFileMultipleWrites) {
  const uint32_t kFileId = 1;
  const uint32_t kNumWrites = 10;

  const std::string pathPrefix = getTestFilePath();
  auto pageFile =
      SerializedPageFile::create(kFileId, pathPrefix, "", /*ioStats=*/nullptr);

  uint64_t totalSize = 0;
  for (int i = 0; i < kNumWrites; ++i) {
    auto data = fmt::format("chunk_{}", i);
    auto iobuf = createTestIOBuf(data);
    const uint64_t chunkSize = iobuf->computeChainDataLength();
    const uint64_t written = pageFile->write(std::move(iobuf));
    totalSize += chunkSize;

    EXPECT_EQ(written, chunkSize);
    EXPECT_EQ(pageFile->size(), totalSize);
  }

  pageFile->finish();
  EXPECT_EQ(pageFile->size(), totalSize);
}

TEST_P(SerializedPageFileTest, serializedPageFileWriterBasicOperations) {
  const std::string pathPrefix = getTestFilePath();
  const uint64_t kTargetFileSize = 1024;
  const uint64_t kWriteBufferSize = 512;
  const uint32_t kRowCount = 100;

  auto serdeOptions = createSerdeOptions();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      std::move(serdeOptions),
      serde_,
      pool());

  EXPECT_EQ(writer.numFinishedFiles(), 0);

  auto rowVector = createTestVector(kRowCount);
  IndexRange range{0, kRowCount};
  folly::Range<IndexRange*> ranges(&range, 1);

  writer.write(rowVector, ranges);

  writer.finishFile();
  const auto numFinishedFiles = writer.numFinishedFiles();
  const auto fileInfos = writer.finish();

  EXPECT_EQ(numFinishedFiles, 1);
  EXPECT_EQ(fileInfos.size(), numFinishedFiles);

  for (const auto& fileInfo : fileInfos) {
    EXPECT_FALSE(fileInfo.path.empty());
    EXPECT_GT(fileInfo.size, 0);
  }
}

TEST_P(SerializedPageFileTest, serializedPageFileWriterMultipleBatches) {
  const uint64_t kTargetFileSize = 1024; // Increased target size
  const uint64_t kWriteBufferSize = 512; // Smaller buffer to force more flushes
  const uint32_t kNumWrites = 30; // Fewer writes but larger data
  const uint32_t kRowsPerBatch = 100; // More rows to generate more data

  const std::string pathPrefix = getTestFilePath();
  auto serdeOptions = createSerdeOptions();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      std::move(serdeOptions),
      serde_,
      pool());

  for (uint32_t i = 0; i < kNumWrites; ++i) {
    auto rowVector = createTestVector(kRowsPerBatch);
    IndexRange range{0, kRowsPerBatch};
    folly::Range<IndexRange*> ranges(&range, 1);
    writer.write(rowVector, ranges);
  }

  auto fileInfos = writer.finish();
  EXPECT_GT(fileInfos.size(), 1);

  uint64_t totalFileSize = 0;
  for (const auto& fileInfo : fileInfos) {
    totalFileSize += fileInfo.size;
  }
  EXPECT_GT(totalFileSize, 0);
}

TEST_P(SerializedPageFileTest, serializedPageFileWriterFinishMultipleTimes) {
  const std::string pathPrefix = getTestFilePath();
  const uint64_t kTargetFileSize = 1024;
  const uint64_t kWriteBufferSize = 512;
  const uint32_t kRowCount = 10;

  auto serdeOptions = createSerdeOptions();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      std::move(serdeOptions),
      serde_,
      pool());

  auto rowVector = createTestVector(kRowCount);
  IndexRange range{0, kRowCount};
  folly::Range<IndexRange*> ranges(&range, 1);
  writer.write(rowVector, ranges);

  auto fileInfos = writer.finish();
  EXPECT_EQ(fileInfos.size(), 1);

  VELOX_ASSERT_THROW(
      writer.write(rowVector, ranges), "SerializedPageFileWriter has finished");
}

TEST_P(SerializedPageFileTest, serializedPageRoundTripBasic) {
  constexpr uint64_t kTargetFileSize = 2048;
  constexpr uint64_t kWriteBufferSize = 512;
  constexpr uint64_t kReadBufferSize = 1024;

  const std::string pathPrefix = getTestFilePath();
  auto writeOptions = createSerdeOptions();
  auto readOptions = createSerdeOptions();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      std::move(writeOptions),
      serde_,
      pool());

  auto originalVector = createTestVector(100);
  IndexRange range{0, 100};
  folly::Range<IndexRange*> ranges(&range, 1);
  writer.write(originalVector, ranges);

  auto fileInfos = writer.finish();

  // Since we only write once with a small vector (100 rows) and large target
  // file size (2048 bytes), there should be exactly one file created
  ASSERT_EQ(fileInfos.size(), 1);

  const auto& filePath = fileInfos[0].path;
  SerializedPageFileReader reader(
      filePath,
      kReadBufferSize,
      asRowType(originalVector->type()),
      serde_,
      createSerdeOptions(),
      pool(),
      /*ioStats=*/nullptr);

  RowVectorPtr readVector;
  bool hasData = reader.nextBatch(readVector);
  EXPECT_TRUE(hasData);

  // Verify that the data content is identical
  test::assertEqualVectors(originalVector, readVector);

  // Verify there's no more data after the first batch
  hasData = reader.nextBatch(readVector);
  EXPECT_FALSE(hasData);
}

TEST_P(SerializedPageFileTest, serializedPageRoundTripMultipleBatches) {
  const uint64_t kTargetFileSize = 256;
  const uint64_t kWriteBufferSize = 100;
  const uint64_t kReadBufferSize = 256;
  const uint32_t kNumWrites = 30;
  const uint32_t kRowsPerBatch = 50;
  const auto pathPrefix = getTestFilePath();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      createSerdeOptions(),
      serde_,
      pool());

  std::vector<RowVectorPtr> originalVectors;
  IndexRange range{0, kRowsPerBatch};
  folly::Range<IndexRange*> ranges(&range, 1);
  for (int i = 0; i < kNumWrites; ++i) {
    auto vector = createTestVector(kRowsPerBatch);
    originalVectors.push_back(vector);
    writer.write(vector, ranges);
  }

  auto fileInfos = writer.finish();
  ASSERT_GT(fileInfos.size(), 1);

  // Read from ALL files that were created and collect all the data
  std::vector<RowVectorPtr> readVectors;

  for (const auto& fileInfo : fileInfos) {
    SerializedPageFileReader reader(
        fileInfo.path,
        kReadBufferSize,
        asRowType(originalVectors[0]->type()),
        serde_,
        createSerdeOptions(),
        pool(),
        /*ioStats=*/nullptr);

    RowVectorPtr readVector;
    while (reader.nextBatch(readVector)) {
      EXPECT_NE(readVector, nullptr);
      EXPECT_EQ(
          readVector->type()->toString(),
          originalVectors[0]->type()->toString());
      readVectors.push_back(readVector);
    }
  }

  test::assertEqualVectors(
      fuzzer::mergeRowVectors(originalVectors, pool()),
      fuzzer::mergeRowVectors(readVectors, pool()));
}

TEST_P(SerializedPageFileTest, roundTripWithComplexTypes) {
  // Skip UnsafeRow for complex types as it has limitations with nested
  // structures
  if (GetParam().serdeType == SerdeType::kUnsafeRow) {
    GTEST_SKIP() << "UnsafeRow does not support complex nested types";
  }

  const uint64_t kTargetFileSize = 1024; // Small to encourage multiple files
  const uint64_t kWriteBufferSize = 256;
  const uint64_t kReadBufferSize = 512;
  const uint64_t kNumWrites = 20;
  const uint64_t kBatchSize = 30;
  const std::string pathPrefix = getTestFilePath();

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      createSerdeOptions(),
      serde_,
      pool());

  VectorFuzzer fuzzer({.vectorSize = kBatchSize}, pool());
  auto complexType =
      ROW({"int_field", "string_field", "array_field", "map_field"},
          {BIGINT(), VARCHAR(), ARRAY(INTEGER()), MAP(VARCHAR(), DOUBLE())});

  // Write multiple batches with complex data
  std::vector<RowVectorPtr> originalVectors;
  for (int i = 0; i < kNumWrites; ++i) {
    auto vector = fuzzer.fuzzRow(complexType);
    originalVectors.push_back(vector);
    IndexRange range{0, kBatchSize};
    folly::Range<IndexRange*> ranges(&range, 1);
    writer.write(vector, ranges);
  }

  auto fileInfos = writer.finish();
  ASSERT_GT(fileInfos.size(), 1);

  // Read from ALL files that were created and collect all the data
  std::vector<RowVectorPtr> readVectors;

  for (const auto& fileInfo : fileInfos) {
    SerializedPageFileReader reader(
        fileInfo.path,
        kReadBufferSize,
        asRowType(originalVectors[0]->type()),
        serde_,
        createSerdeOptions(),
        pool(),
        /*ioStats=*/nullptr);

    RowVectorPtr readVector;
    while (reader.nextBatch(readVector)) {
      EXPECT_NE(readVector, nullptr);
      EXPECT_EQ(
          readVector->type()->toString(),
          originalVectors[0]->type()->toString());
      readVectors.push_back(readVector);
    }
  }

  // Compare the concatenated vectors for exact data equality
  test::assertEqualVectors(
      fuzzer::mergeRowVectors(originalVectors, pool()),
      fuzzer::mergeRowVectors(readVectors, pool()));
}
TEST_P(SerializedPageFileTest, emptyBatches) {
  const std::string pathPrefix = getTestFilePath();
  const uint64_t kTargetFileSize = 1024;
  const uint64_t kWriteBufferSize = 512;

  SerializedPageFileWriter writer(
      pathPrefix,
      kTargetFileSize,
      kWriteBufferSize,
      "",
      createSerdeOptions(),
      serde_,
      pool());

  auto rowType = ROW({"a", "b", "c"}, {BIGINT(), VARCHAR(), DOUBLE()});
  auto emptyVector = BaseVector::create<RowVector>(rowType, 0, pool());

  IndexRange range{0, 0};
  folly::Range<IndexRange*> ranges(&range, 1);
  writer.write(emptyVector, ranges);

  auto fileInfos = writer.finish();
  EXPECT_EQ(fileInfos.size(), 0);
}

// Generate test parameters for different serde types and compression kinds
std::vector<TestParams> generateTestParams() {
  std::vector<TestParams> params;

  std::vector<SerdeType> serdeTypes = {
      SerdeType::kPresto,
  };

  std::vector<common::CompressionKind> compressionKinds = {
      common::CompressionKind::CompressionKind_NONE,
      common::CompressionKind::CompressionKind_ZLIB,
      common::CompressionKind::CompressionKind_SNAPPY,
      common::CompressionKind::CompressionKind_ZSTD,
      common::CompressionKind::CompressionKind_LZ4,
      common::CompressionKind::CompressionKind_GZIP};

  for (const auto& serdeType : serdeTypes) {
    for (const auto& compressionKind : compressionKinds) {
      params.push_back({serdeType, compressionKind});
    }
  }

  return params;
}

// Custom test name generator
std::string testNameGenerator(
    const ::testing::TestParamInfo<TestParams>& info) {
  std::string serdeName;
  switch (info.param.serdeType) {
    case SerdeType::kPresto:
      serdeName = "Presto";
      break;
    case SerdeType::kCompactRow:
      serdeName = "CompactRow";
      break;
    case SerdeType::kUnsafeRow:
      serdeName = "UnsafeRow";
      break;
    default:
      VELOX_UNSUPPORTED();
  }

  std::string compressionName;
  switch (info.param.compressionKind) {
    case common::CompressionKind::CompressionKind_NONE:
      compressionName = "NoCompression";
      break;
    case common::CompressionKind::CompressionKind_ZLIB:
      compressionName = "ZLIB";
      break;
    case common::CompressionKind::CompressionKind_SNAPPY:
      compressionName = "Snappy";
      break;
    case common::CompressionKind::CompressionKind_ZSTD:
      compressionName = "ZSTD";
      break;
    case common::CompressionKind::CompressionKind_LZ4:
      compressionName = "LZ4";
      break;
    case common::CompressionKind::CompressionKind_GZIP:
      compressionName = "GZIP";
      break;
    default:
      VELOX_UNSUPPORTED();
  }

  return serdeName + "_" + compressionName;
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    SerializedPageFileTest,
    SerializedPageFileTest,
    ::testing::ValuesIn(generateTestParams()),
    testNameGenerator);
