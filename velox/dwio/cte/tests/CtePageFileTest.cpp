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
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>

#include "velox/common/base/Fs.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/Writer.h"
#include "velox/dwio/cte/reader/CtePageReader.h"
#include "velox/dwio/cte/writer/CtePageWriter.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::filesystems;
using facebook::velox::exec::test::TempDirectoryPath;

namespace {
static const int64_t kGB = 1'000'000'000;
}
struct TestParam {
  const common::CompressionKind compressionKind;

  TestParam(common::CompressionKind _compressionKind)
      : compressionKind(_compressionKind) {}

  TestParam(uint32_t value)
      : compressionKind(static_cast<common::CompressionKind>(value)) {}

  uint32_t value() const {
    return static_cast<uint32_t>(compressionKind);
  }

  std::string toString() const {
    return fmt::format("compressionKind: {}", compressionKind);
  }
};

class CtePageFileTest : public ::testing::TestWithParam<uint32_t>,
                        public facebook::velox::test::VectorTestBase {
 public:
  static std::vector<uint32_t> getTestParams() {
    std::vector<uint32_t> testParams;
    testParams.emplace_back(
        TestParam{common::CompressionKind::CompressionKind_NONE}.value());
    testParams.emplace_back(
        TestParam{common::CompressionKind::CompressionKind_ZLIB}.value());
    testParams.emplace_back(
        TestParam{common::CompressionKind::CompressionKind_SNAPPY}.value());
    testParams.emplace_back(
        TestParam{common::CompressionKind::CompressionKind_ZSTD}.value());
    testParams.emplace_back(
        TestParam{common::CompressionKind::CompressionKind_LZ4}.value());
    return testParams;
  }

  void assertEqualVectorPart(
      const VectorPtr& expected,
      const VectorPtr& actual,
      vector_size_t offset) {
    ASSERT_GE(expected->size(), actual->size() + offset);
    for (vector_size_t i = 0; i < actual->size(); i++) {
      ASSERT_TRUE(expected->equalValueAt(actual.get(), i + offset, i))
          << "at " << (i + offset) << ": expected "
          << expected->toString(i + offset) << ", but got "
          << actual->toString(i);
    }
  }

 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
    facebook::velox::filesystems::registerLocalFileSystem();
    memory::MemoryManager::testingSetInstance({});
    if (!isRegisteredVectorSerde()) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerVectorSerde();
    }
    if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
      facebook::velox::serializer::presto::PrestoVectorSerde::
          registerNamedVectorSerde();
    }
  }

  void SetUp() override {
    allocator_ = memory::memoryManager()->allocator();
    tempDir_ = exec::test::TempDirectoryPath::create();
    filesystems::registerLocalFileSystem();
    compressionKind_ = TestParam{GetParam()}.compressionKind;
    dwio::common::LocalFileSink::registerFactory();
    rootPool_ = memory::memoryManager()->addRootPool("PagefileTestsRoot");
    leafPool_ = rootPool_->addLeafChild("PagefileTestsLeaf");
  }

  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath) {
    auto sink = dwio::common::FileSink::create(
        fmt::format("file:{}", filePath), {.pool = rootPool_.get()});
    EXPECT_TRUE(sink->isBuffered());
    EXPECT_TRUE(fs::exists(filePath));
    EXPECT_FALSE(sink->isClosed());
    return sink;
  }

  std::shared_ptr<TempDirectoryPath> tempDir_;
  memory::MemoryAllocator* allocator_;
  common::CompressionKind compressionKind_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> leafPool_;
};

TEST_P(CtePageFileTest, teste2eFlow) {
  int numBatches = 50;
  int numDuplicates = 1;
  std::vector<RowVectorPtr> batches;
  std::vector<std::optional<int64_t>> values;
  batches.reserve(numBatches);

  const int numRowsPerBatch = 50;
  values.resize(numBatches * numRowsPerBatch);
  // Create a sequence of sorted 'values' in ascending order starting at -10.
  // Each distinct value occurs 'numDuplicates' times. The sequence total has
  // numBatches * kNumRowsPerBatch item. Each batch created in the test below,
  // contains a subsequence with index mod being equal to its batch number.
  const int kNumNulls = numBatches;
  for (int i = 0, value = -10; i < numRowsPerBatch * numBatches;) {
    while (i < kNumNulls) {
      values[i++] = std::nullopt;
    }
    for (int j = 0; j < numDuplicates; ++j) {
      values[i++] = value++;
    }
  }

  auto schema = ROW({"c0"}, {BIGINT()});

  dwio::common::WriterOptions writerOptions;
  writerOptions.memoryPool = leafPool_.get();
  writerOptions.schema = schema;
  writerOptions.compressionKind = compressionKind_;

  // Create a pagefile writer.
  auto filePath = fs::path(fmt::format("{}/test.txt", tempDir_->getPath()));
  auto sink = createSink(filePath.string());
  auto sinkPtr = sink.get();
  auto writer = std::make_unique<velox::pagefile::CtePageWriter>(
      std::move(sink), writerOptions, leafPool_);

  for (auto iter = 0; iter < numBatches / 2; ++iter) {
    auto data = makeRowVector({makeFlatVector<int64_t>(
        numRowsPerBatch,
        [&](auto row) {
          return values[row * numBatches / 2 + iter].has_value()
              ? values[row * numBatches / 2 + iter].value()
              : 0;
        },
        [&](auto row) {
          return !values[row * numBatches / 2 + iter].has_value();
        })});
    batches.push_back(data);
    writer->write(data);

    // batch 2
    data = makeRowVector({makeFlatVector<int64_t>(
        numRowsPerBatch,
        [&](auto row) {
          return values[(numRowsPerBatch + row) * numBatches / 2 + iter]
                     .has_value()
              ? values[(numRowsPerBatch + row) * numBatches / 2 + iter].value()
              : 0;
        },
        [&](auto row) {
          return !values[(numRowsPerBatch + row) * numBatches / 2 + iter]
                      .has_value();
        })});
    batches.push_back(data);
    writer->write(data);
  }
  writer->close();

  // create a pagefile reader
  dwio::common::ReaderOptions readerOptions{leafPool_.get()};
  readerOptions.setFileSchema(schema);
  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::make_shared<LocalReadFile>(filePath.string()),
      readerOptions.memoryPool());
  auto reader = std::make_unique<velox::pagefile::CtePageReader>(
      readerOptions, std::move(input));

  // create a row reader
  dwio::common::RowReaderOptions rowReaderOpts;
  TypePtr outputType = createType(TypeKind::BIGINT, {});
  rowReaderOpts.select(
      std::make_shared<facebook::velox::dwio::common::ColumnSelector>(
          schema, schema->names(), nullptr, false));
  auto rowReader = reader->createRowReader(rowReaderOpts);

  // Read verification.
  int batchIdx = 0;
  VectorPtr result = BaseVector::create(outputType, 0, leafPool_.get());
  while (rowReader->next(numRowsPerBatch, result)) {
    assertEqualVectorPart(batches[batchIdx], result, 0);
    ++batchIdx;
  }
  ASSERT_EQ(batchIdx, numBatches);
}

VELOX_INSTANTIATE_TEST_SUITE_P(
    CtePageFileTestSuite,
    CtePageFileTest,
    ::testing::ValuesIn(CtePageFileTest::getTestParams()));
