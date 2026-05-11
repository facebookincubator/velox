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

namespace facebook::velox::io {

class ReaderOptionsTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  const std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool("ReaderOptionsTest")};
  IoStatistics dataIoStats_;
  IoStatistics metadataIoStats_;
};

TEST_F(ReaderOptionsTest, constructor) {
  ReaderOptions options(pool_.get(), &dataIoStats_, &metadataIoStats_);

  EXPECT_EQ(&options.memoryPool(), pool_.get());
  EXPECT_EQ(options.dataIoStats(), &dataIoStats_);
  EXPECT_EQ(options.metadataIoStats(), &metadataIoStats_);
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
  ReaderOptions options(pool_.get(), &dataIoStats_, &metadataIoStats_);

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
  ReaderOptions options(pool_.get(), &dataIoStats_, &metadataIoStats_);
  EXPECT_EQ(options.ioExecutor(), nullptr);

  auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(2);
  options.setIOExecutor(executor);
  EXPECT_EQ(options.ioExecutor(), executor);

  options.setIOExecutor(nullptr);
  EXPECT_EQ(options.ioExecutor(), nullptr);
}

TEST_F(ReaderOptionsTest, ioStats) {
  ReaderOptions options(pool_.get(), &dataIoStats_, &metadataIoStats_);

  EXPECT_EQ(options.dataIoStats(), &dataIoStats_);
  EXPECT_EQ(options.metadataIoStats(), &metadataIoStats_);

  options.dataIoStats()->read().increment(100);
  EXPECT_EQ(dataIoStats_.read().count(), 1);
  EXPECT_EQ(dataIoStats_.read().sum(), 100);

  options.metadataIoStats()->read().increment(50);
  EXPECT_EQ(metadataIoStats_.read().count(), 1);
  EXPECT_EQ(metadataIoStats_.read().sum(), 50);
}

TEST_F(ReaderOptionsTest, chainingSetters) {
  ReaderOptions options(pool_.get(), &dataIoStats_, &metadataIoStats_);
  auto& result = options.setAutoPreloadLength(1'024)
                     .setPrefetchMode(PrefetchMode::PRELOAD)
                     .setLoadQuantum(4 << 20)
                     .setMaxCoalesceDistance(1 << 20)
                     .setMaxCoalesceBytes(64 << 20)
                     .setPrefetchRowGroups(4);

  EXPECT_EQ(&result, &options);
  EXPECT_EQ(options.autoPreloadLength(), 1'024);
  EXPECT_EQ(options.prefetchMode(), PrefetchMode::PRELOAD);
  EXPECT_EQ(options.loadQuantum(), 4 << 20);
  EXPECT_EQ(options.maxCoalesceDistance(), 1 << 20);
  EXPECT_EQ(options.maxCoalesceBytes(), 64 << 20);
  EXPECT_EQ(options.prefetchRowGroups(), 4);
}

} // namespace facebook::velox::io
