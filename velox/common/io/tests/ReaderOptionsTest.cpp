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

#include "velox/common/io/Options.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::io {

class ReaderOptionsTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  const std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool("ReaderOptionsTest")};
  std::shared_ptr<IoStatistics> dataIoStats_{std::make_shared<IoStatistics>()};
  std::shared_ptr<IoStatistics> metadataIoStats_{
      std::make_shared<IoStatistics>()};
  std::shared_ptr<IoStatistics> indexIoStats_{std::make_shared<IoStatistics>()};
};

TEST_F(ReaderOptionsTest, constructor) {
  ReaderOptions options(pool_.get());

  EXPECT_EQ(&options.memoryPool(), pool_.get());
  EXPECT_EQ(options.dataIoStats(), nullptr);
  EXPECT_EQ(options.metadataIoStats(), nullptr);
  EXPECT_EQ(options.indexIoStats(), nullptr);
  EXPECT_EQ(options.ioExecutor(), nullptr);
  EXPECT_EQ(options.autoPreloadLength(), DEFAULT_AUTO_PRELOAD_SIZE);
  EXPECT_EQ(options.prefetchMode(), PrefetchMode::PREFETCH);
  EXPECT_EQ(options.loadQuantum(), ReaderOptions::kDefaultLoadQuantum);
  EXPECT_EQ(
      options.maxCoalesceDistance(), ReaderOptions::kDefaultCoalesceDistance);
  EXPECT_EQ(options.maxCoalesceBytes(), ReaderOptions::kDefaultCoalesceBytes);
  EXPECT_EQ(
      options.prefetchRowGroups(), ReaderOptions::kDefaultPrefetchRowGroups);
  EXPECT_TRUE(options.cacheable());
}

TEST_F(ReaderOptionsTest, setters) {
  ReaderOptions options(pool_.get());

  options.setAutoPreloadLength(1'024);
  EXPECT_EQ(options.autoPreloadLength(), 1'024);

  options.setPrefetchMode(PrefetchMode::NOT_SET);
  EXPECT_EQ(options.prefetchMode(), PrefetchMode::NOT_SET);

  options.setLoadQuantum(4 << 20);
  EXPECT_EQ(options.loadQuantum(), 4 << 20);

  options.setMaxCoalesceDistance(1 << 20);
  EXPECT_EQ(options.maxCoalesceDistance(), 1 << 20);

  options.setMaxCoalesceBytes(64 << 20);
  EXPECT_EQ(options.maxCoalesceBytes(), 64 << 20);

  options.setPrefetchRowGroups(4);
  EXPECT_EQ(options.prefetchRowGroups(), 4);

  options.setCacheable(false);
  EXPECT_FALSE(options.cacheable());
}

TEST_F(ReaderOptionsTest, ioExecutor) {
  ReaderOptions options(pool_.get());
  EXPECT_EQ(options.ioExecutor(), nullptr);

  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(2);
  options.setIOExecutor(executor);
  EXPECT_EQ(options.ioExecutor(), executor);

  options.setIOExecutor(nullptr);
  EXPECT_EQ(options.ioExecutor(), nullptr);
}

TEST_F(ReaderOptionsTest, ioStats) {
  ReaderOptions options(pool_.get());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setIndexIoStats(indexIoStats_);

  EXPECT_EQ(options.dataIoStats(), dataIoStats_);
  EXPECT_EQ(options.metadataIoStats(), metadataIoStats_);
  EXPECT_EQ(options.indexIoStats(), indexIoStats_);

  options.dataIoStats()->read().increment(100);
  EXPECT_EQ(dataIoStats_->read().count(), 1);
  EXPECT_EQ(dataIoStats_->read().sum(), 100);

  options.metadataIoStats()->read().increment(50);
  EXPECT_EQ(metadataIoStats_->read().count(), 1);
  EXPECT_EQ(metadataIoStats_->read().sum(), 50);

  options.indexIoStats()->read().increment(25);
  EXPECT_EQ(indexIoStats_->read().count(), 1);
  EXPECT_EQ(indexIoStats_->read().sum(), 25);
}

TEST_F(ReaderOptionsTest, chainingSetters) {
  ReaderOptions options(pool_.get());
  auto& result = options.setDataIoStats(dataIoStats_)
                     .setMetadataIoStats(metadataIoStats_)
                     .setIndexIoStats(indexIoStats_)
                     .setAutoPreloadLength(1'024)
                     .setPrefetchMode(PrefetchMode::PRELOAD)
                     .setLoadQuantum(4 << 20)
                     .setMaxCoalesceDistance(1 << 20)
                     .setMaxCoalesceBytes(64 << 20)
                     .setPrefetchRowGroups(4);

  EXPECT_EQ(&result, &options);
  EXPECT_EQ(options.dataIoStats(), dataIoStats_);
  EXPECT_EQ(options.metadataIoStats(), metadataIoStats_);
  EXPECT_EQ(options.indexIoStats(), indexIoStats_);
  EXPECT_EQ(options.autoPreloadLength(), 1'024);
  EXPECT_EQ(options.prefetchMode(), PrefetchMode::PRELOAD);
  EXPECT_EQ(options.loadQuantum(), 4 << 20);
  EXPECT_EQ(options.maxCoalesceDistance(), 1 << 20);
  EXPECT_EQ(options.maxCoalesceBytes(), 64 << 20);
  EXPECT_EQ(options.prefetchRowGroups(), 4);
}

TEST_F(ReaderOptionsTest, doubleSetIoStatsThrows) {
  ReaderOptions options(pool_.get());
  options.setDataIoStats(dataIoStats_);
  VELOX_ASSERT_THROW(
      options.setDataIoStats(dataIoStats_), "dataIoStats already set");

  ReaderOptions options2(pool_.get());
  options2.setMetadataIoStats(metadataIoStats_);
  VELOX_ASSERT_THROW(
      options2.setMetadataIoStats(metadataIoStats_),
      "metadataIoStats already set");

  ReaderOptions options3(pool_.get());
  options3.setIndexIoStats(indexIoStats_);
  VELOX_ASSERT_THROW(
      options3.setIndexIoStats(indexIoStats_), "indexIoStats already set");
}

TEST_F(ReaderOptionsTest, copyConstruct) {
  ReaderOptions options(pool_.get());
  options.setDataIoStats(dataIoStats_);
  options.setMetadataIoStats(metadataIoStats_);
  options.setLoadQuantum(4 << 20);

  ReaderOptions copy(options);
  EXPECT_EQ(&copy.memoryPool(), pool_.get());
  EXPECT_EQ(copy.dataIoStats(), dataIoStats_);
  EXPECT_EQ(copy.metadataIoStats(), metadataIoStats_);
  EXPECT_EQ(copy.indexIoStats(), nullptr);
  EXPECT_EQ(copy.loadQuantum(), 4 << 20);
}

} // namespace facebook::velox::io
