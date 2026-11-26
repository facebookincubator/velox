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

#include "velox/connectors/common/BufferedInputBuilder.h"

#include <folly/CancellationToken.h>
#include <gtest/gtest.h>

#include <typeinfo>

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/file/File.h"
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::connector {
namespace {

class BufferedInputBuilderTest : public testing::Test,
                                 public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void TearDown() override {
    cache_->shutdown();
  }

  // Checks that 'createDefault' builds 'ExpectedInput' for 'format' and
  // 'cache', and that canceling the query reaches the input's file reads.
  template <typename ExpectedInput>
  void testFileReadCancellation(
      dwio::common::FileFormat format,
      cache::AsyncDataCache* cache) {
    folly::CancellationSource cancellation;
    const config::ConfigBase session{
        std::unordered_map<std::string, std::string>{}};
    const auto ctx = ConnectorQueryCtx::Builder()
                         .operatorPool(pool())
                         .connectorPool(pool())
                         .sessionProperties(&session)
                         .queryId("query.BufferedInputBuilderTest")
                         .taskId("task.BufferedInputBuilderTest")
                         .planNodeId("planNodeId.BufferedInputBuilderTest")
                         .cancellationToken(cancellation.getToken())
                         .asyncDataCache(cache)
                         .build();
    FileHandle fileHandle;
    fileHandle.file = std::make_shared<InMemoryReadFile>(std::string("data"));
    dwio::common::ReaderOptions readerOptions(pool());
    readerOptions.setFileFormat(format);

    const auto input = BufferedInputBuilder::createDefault(
        fileHandle,
        readerOptions,
        ctx.get(),
        std::make_shared<io::IoStatistics>(),
        std::make_shared<IoStats>(),
        /*executor=*/nullptr,
        /*fileReadOps=*/{});
    const auto& inputRef = *input;
    ASSERT_EQ(typeid(inputRef), typeid(ExpectedInput));
    const auto& readCancellation =
        input->getInputStream()->fileIoContext().cancellationToken;

    EXPECT_FALSE(readCancellation.isCancellationRequested());
    cancellation.requestCancellation();
    EXPECT_TRUE(readCancellation.isCancellationRequested());
  }

  const std::shared_ptr<cache::AsyncDataCache> cache_ =
      cache::AsyncDataCache::create(memory::memoryManager()->allocator());
};

TEST_F(BufferedInputBuilderTest, fileReadCancellation) {
  testFileReadCancellation<dwio::common::CachedBufferedInput>(
      dwio::common::FileFormat::DWRF, cache_.get());
  testFileReadCancellation<dwio::common::BufferedInput>(
      dwio::common::FileFormat::NIMBLE, /*cache=*/nullptr);
  testFileReadCancellation<dwio::common::DirectBufferedInput>(
      dwio::common::FileFormat::DWRF, /*cache=*/nullptr);
}

} // namespace
} // namespace facebook::velox::connector
