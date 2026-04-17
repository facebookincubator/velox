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
#include "velox/exec/IndexLookupJoinBridge.h"

#include <folly/executors/InlineExecutor.h>
#include <gtest/gtest.h>
#include <thread>

#include "velox/common/base/tests/GTestUtils.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::connector;

namespace {

std::shared_ptr<ConnectorSplit> makeFakeSplit(const std::string& id) {
  return std::make_shared<ConnectorSplit>(id);
}

std::vector<std::shared_ptr<ConnectorSplit>> makeFakeSplits(int count) {
  std::vector<std::shared_ptr<ConnectorSplit>> splits;
  splits.reserve(count);
  for (int i = 0; i < count; ++i) {
    splits.push_back(makeFakeSplit("connector_" + std::to_string(i)));
  }
  return splits;
}

class IndexLookupJoinBridgeTest : public testing::Test {
 protected:
  std::shared_ptr<IndexLookupJoinBridge> createBridge() {
    auto bridge = std::make_shared<IndexLookupJoinBridge>();
    bridge->start();
    return bridge;
  }
};

TEST_F(IndexLookupJoinBridgeTest, setAndGetSplits) {
  auto bridge = createBridge();
  auto splits = makeFakeSplits(3);
  const auto expectedSize = splits.size();

  bridge->setIndexSplits(std::move(splits));

  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = bridge->splitsOrFuture(&future);
  ASSERT_FALSE(result.empty());
  ASSERT_EQ(result.size(), expectedSize);
  ASSERT_FALSE(future.valid());
}

TEST_F(IndexLookupJoinBridgeTest, splitsOrFutureBlocks) {
  auto bridge = createBridge();

  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = bridge->splitsOrFuture(&future);
  ASSERT_TRUE(result.empty());
  ASSERT_TRUE(future.valid());

  // Set splits and verify the future is fulfilled.
  auto splits = makeFakeSplits(2);
  bridge->setIndexSplits(std::move(splits));

  std::move(future).via(&folly::InlineExecutor::instance()).get();

  // Now splitsOrFuture should return immediately on repeated calls.
  for (int i = 0; i < 3; ++i) {
    ContinueFuture future2 = ContinueFuture::makeEmpty();
    auto result2 = bridge->splitsOrFuture(&future2);
    ASSERT_FALSE(result2.empty());
    ASSERT_EQ(result2.size(), 2);
    ASSERT_FALSE(future2.valid());
  }
}

TEST_F(IndexLookupJoinBridgeTest, multipleFollowers) {
  auto bridge = createBridge();

  // Multiple followers wait for splits.
  const int numFollowers = 4;
  std::vector<ContinueFuture> futures;
  for (int i = 0; i < numFollowers; ++i) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = bridge->splitsOrFuture(&future);
    ASSERT_TRUE(result.empty());
    ASSERT_TRUE(future.valid());
    futures.push_back(std::move(future));
  }

  // Leader sets splits.
  bridge->setIndexSplits(makeFakeSplits(5));

  // All futures should be fulfilled.
  for (auto& future : futures) {
    std::move(future).via(&folly::InlineExecutor::instance()).get();
  }

  // All followers can now get splits.
  for (int i = 0; i < numFollowers; ++i) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = bridge->splitsOrFuture(&future);
    ASSERT_FALSE(result.empty());
    ASSERT_EQ(result.size(), 5);
    ASSERT_FALSE(future.valid());
  }
}

TEST_F(IndexLookupJoinBridgeTest, setBeforeGet) {
  auto bridge = createBridge();

  // Leader sets splits before any follower calls splitsOrFuture.
  bridge->setIndexSplits(makeFakeSplits(3));

  // Followers get splits immediately without blocking.
  for (int i = 0; i < 3; ++i) {
    ContinueFuture future = ContinueFuture::makeEmpty();
    auto result = bridge->splitsOrFuture(&future);
    ASSERT_FALSE(result.empty());
    ASSERT_EQ(result.size(), 3);
    ASSERT_FALSE(future.valid());
  }
}

TEST_F(IndexLookupJoinBridgeTest, setTwiceThrows) {
  auto bridge = createBridge();
  bridge->setIndexSplits(makeFakeSplits(1));

  VELOX_ASSERT_THROW(
      bridge->setIndexSplits(makeFakeSplits(1)),
      "setIndexSplits must be called only once");
}

TEST_F(IndexLookupJoinBridgeTest, setEmptySplitsThrows) {
  auto bridge = createBridge();

  VELOX_ASSERT_THROW(
      bridge->setIndexSplits({}), "Index splits must not be empty");
}

TEST_F(IndexLookupJoinBridgeTest, setBeforeStartThrows) {
  auto bridge = std::make_shared<IndexLookupJoinBridge>();
  // Do not call start().

  VELOX_ASSERT_THROW(
      bridge->setIndexSplits(makeFakeSplits(1)),
      "Bridge must be started before setting index splits");
}

TEST_F(IndexLookupJoinBridgeTest, cancelUnblocksFollowers) {
  auto bridge = createBridge();

  ContinueFuture future = ContinueFuture::makeEmpty();
  auto result = bridge->splitsOrFuture(&future);
  ASSERT_TRUE(result.empty());
  ASSERT_TRUE(future.valid());

  bridge->cancel();

  // Future should be fulfilled by cancel.
  std::move(future).via(&folly::InlineExecutor::instance()).get();
}

TEST_F(IndexLookupJoinBridgeTest, getAfterCancelThrows) {
  auto bridge = createBridge();
  bridge->cancel();

  ContinueFuture future = ContinueFuture::makeEmpty();
  VELOX_ASSERT_THROW(
      bridge->splitsOrFuture(&future),
      "Getting index splits after the bridge is cancelled");
}

TEST_F(IndexLookupJoinBridgeTest, setAfterCancelThrows) {
  auto bridge = createBridge();
  bridge->cancel();

  VELOX_ASSERT_THROW(
      bridge->setIndexSplits(makeFakeSplits(1)),
      "Setting index splits after the bridge is cancelled");
}

TEST_F(IndexLookupJoinBridgeTest, concurrentSetAndGet) {
  auto bridge = createBridge();
  const int numFollowers = 8;
  const int numSplits = 5;

  std::vector<std::thread> threads;
  std::atomic_int successCount{0};
  std::atomic_bool stop{false};

  // Start followers that keep getting splits until stopped.
  threads.reserve(numFollowers);
  for (int i = 0; i < numFollowers; ++i) {
    threads.emplace_back([&bridge, &successCount, &stop, numSplits]() {
      ContinueFuture future = ContinueFuture::makeEmpty();
      auto result = bridge->splitsOrFuture(&future);
      if (result.empty()) {
        std::move(future).via(&folly::InlineExecutor::instance()).get();
      }
      while (!stop.load()) {
        ContinueFuture f = ContinueFuture::makeEmpty();
        auto r = bridge->splitsOrFuture(&f);
        ASSERT_FALSE(r.empty());
        ASSERT_EQ(r.size(), numSplits);
        ASSERT_FALSE(f.valid());
        ++successCount;
      }
    });
  }

  // Give followers time to register.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Leader sets splits.
  bridge->setIndexSplits(makeFakeSplits(numSplits));

  // Let followers run for a while.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stop.store(true);

  for (auto& t : threads) {
    t.join();
  }
  ASSERT_GE(successCount.load(), numFollowers);
}

} // namespace
