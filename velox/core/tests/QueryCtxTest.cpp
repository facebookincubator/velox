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

#include <thread>
#include <vector>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::core::test {

class QueryCtxTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(QueryCtxTest, withSysRootPool) {
  auto queryCtx = QueryCtx::create(
      nullptr,
      QueryConfig{{}},
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      nullptr,
      memory::deprecatedRootPool().shared_from_this());
  auto* queryPool = queryCtx->pool();
  ASSERT_EQ(&memory::deprecatedRootPool(), queryPool);
  ASSERT_NE(queryPool->reclaimer(), nullptr);
  try {
    VELOX_FAIL("Trigger Error");
  } catch (const velox::VeloxRuntimeError&) {
    VELOX_ASSERT_THROW(
        queryPool->reclaimer()->abort(queryPool, std::current_exception()),
        "SysMemoryReclaimer::abort is not supported");
  }
  ASSERT_EQ(queryPool->reclaimer()->priority(), 0);
  memory::MemoryReclaimer::Stats stats;
  ASSERT_EQ(queryPool->reclaimer()->reclaim(queryPool, 1'000, 1'000, stats), 0);
  uint64_t reclaimableBytes{0};
  ASSERT_FALSE(
      queryPool->reclaimer()->reclaimableBytes(*queryPool, reclaimableBytes));
}

TEST_F(QueryCtxTest, releaseCallbacks) {
  int callbackCount = 0;
  std::string capturedQueryId;

  {
    auto queryCtx = QueryCtx::create(
        nullptr,
        QueryConfig{{}},
        std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
        nullptr,
        nullptr,
        nullptr,
        "test_query_id");

    // Add multiple callbacks.
    queryCtx->addReleaseCallback([&callbackCount]() { ++callbackCount; });

    queryCtx->addReleaseCallback(
        [&callbackCount, &capturedQueryId, id = queryCtx->queryId()]() {
          ++callbackCount;
          capturedQueryId = id;
        });

    // Callbacks should not be invoked yet.
    ASSERT_EQ(callbackCount, 0);
  }

  // After QueryCtx destruction, all callbacks should have been invoked.
  ASSERT_EQ(callbackCount, 2);
  ASSERT_EQ(capturedQueryId, "test_query_id");
}

TEST_F(QueryCtxTest, releaseCallbackException) {
  int callbackCount = 0;

  {
    auto queryCtx = QueryCtx::create(
        nullptr,
        QueryConfig{{}},
        std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
        nullptr,
        nullptr,
        nullptr,
        "test_query_id");

    // First callback succeeds.
    queryCtx->addReleaseCallback([&callbackCount]() { ++callbackCount; });

    // Second callback throws an exception.
    queryCtx->addReleaseCallback(
        []() { throw std::runtime_error("Test exception"); });

    // Third callback should still execute despite the previous exception.
    queryCtx->addReleaseCallback([&callbackCount]() { ++callbackCount; });
  }

  // All callbacks should have been attempted, with exception caught and logged.
  // First and third callbacks should have incremented the counter.
  ASSERT_EQ(callbackCount, 2);
}

TEST_F(QueryCtxTest, builderReleaseCallbacks) {
  int callbackCount = 0;
  std::string capturedQueryId;

  {
    // Use builder to add release callbacks during construction.
    auto queryCtx =
        QueryCtx::Builder()
            .queryId("builder_test_query_id")
            .releaseCallback([&callbackCount]() { ++callbackCount; })
            .releaseCallback([&callbackCount, &capturedQueryId]() {
              ++callbackCount;
              capturedQueryId = "builder_test_query_id";
            })
            .build();

    // Callbacks should not be invoked yet.
    ASSERT_EQ(callbackCount, 0);
  }

  // After QueryCtx destruction, all callbacks should have been invoked.
  ASSERT_EQ(callbackCount, 2);
  ASSERT_EQ(capturedQueryId, "builder_test_query_id");
}
TEST_F(QueryCtxTest, transportTypeDefaultsToHttp) {
  auto queryCtx = QueryCtx::Builder().build();
  EXPECT_EQ(
      queryCtx->inputTransportType("node1"), ExchangeTransportType::kHttp);
  EXPECT_EQ(
      queryCtx->outputTransportType("node1"), ExchangeTransportType::kHttp);
}

TEST_F(QueryCtxTest, transportTypeViaBuilder) {
  auto queryCtx =
      QueryCtx::Builder()
          .inputTransportType("exchange1", ExchangeTransportType::kUcx)
          .inputTransportType("exchange2", ExchangeTransportType::kHttp)
          .outputTransportType("output1", ExchangeTransportType::kUcx)
          .build();

  EXPECT_EQ(
      queryCtx->inputTransportType("exchange1"), ExchangeTransportType::kUcx);
  EXPECT_EQ(
      queryCtx->inputTransportType("exchange2"), ExchangeTransportType::kHttp);
  EXPECT_EQ(
      queryCtx->outputTransportType("output1"), ExchangeTransportType::kUcx);
  // Unset node defaults to kHttp.
  EXPECT_EQ(
      queryCtx->outputTransportType("output2"), ExchangeTransportType::kHttp);
}

TEST_F(QueryCtxTest, transportTypeMutableSetters) {
  auto queryCtx = QueryCtx::Builder().build();

  queryCtx->setInputTransportType("exchange1", ExchangeTransportType::kUcx);
  queryCtx->setOutputTransportType("output1", ExchangeTransportType::kUcx);

  EXPECT_EQ(
      queryCtx->inputTransportType("exchange1"), ExchangeTransportType::kUcx);
  EXPECT_EQ(
      queryCtx->outputTransportType("output1"), ExchangeTransportType::kUcx);

  // Overwrite with a different value.
  queryCtx->setInputTransportType("exchange1", ExchangeTransportType::kHttp);
  EXPECT_EQ(
      queryCtx->inputTransportType("exchange1"), ExchangeTransportType::kHttp);
}

TEST_F(QueryCtxTest, transportTypeParallelSetters) {
  auto queryCtx = QueryCtx::Builder().build();

  constexpr int kNumThreads = 8;
  constexpr int kNodesPerThread = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads * 2);

  // Parallel writes to inputTransportTypes_ from multiple threads, each writing
  // to its own set of node IDs.
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&queryCtx, t]() {
      for (int i = 0; i < kNodesPerThread; ++i) {
        auto nodeId = fmt::format("input_t{}_n{}", t, i);
        queryCtx->setInputTransportType(nodeId, ExchangeTransportType::kUcx);
      }
    });
  }

  // Parallel writes to outputTransportTypes_ from multiple threads.
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&queryCtx, t]() {
      for (int i = 0; i < kNodesPerThread; ++i) {
        auto nodeId = fmt::format("output_t{}_n{}", t, i);
        queryCtx->setOutputTransportType(nodeId, ExchangeTransportType::kUcx);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Verify all entries were written correctly.
  for (int t = 0; t < kNumThreads; ++t) {
    for (int i = 0; i < kNodesPerThread; ++i) {
      EXPECT_EQ(
          queryCtx->inputTransportType(fmt::format("input_t{}_n{}", t, i)),
          ExchangeTransportType::kUcx);
      EXPECT_EQ(
          queryCtx->outputTransportType(fmt::format("output_t{}_n{}", t, i)),
          ExchangeTransportType::kUcx);
    }
  }
}

TEST_F(QueryCtxTest, transportTypeParallelSettersSameKey) {
  auto queryCtx = QueryCtx::Builder().build();

  constexpr int kNumThreads = 8;
  constexpr int kIterations = 1'000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  // Multiple threads race to set the same key. The final value depends on
  // scheduling, but no crash or data corruption should occur.
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&queryCtx, t]() {
      auto type = (t % 2 == 0) ? ExchangeTransportType::kUcx
                               : ExchangeTransportType::kHttp;
      for (int i = 0; i < kIterations; ++i) {
        queryCtx->setInputTransportType("contested_node", type);
        queryCtx->setOutputTransportType("contested_node", type);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // The final value must be one of the two valid enum values.
  auto inputType = queryCtx->inputTransportType("contested_node");
  EXPECT_TRUE(
      inputType == ExchangeTransportType::kHttp ||
      inputType == ExchangeTransportType::kUcx);

  auto outputType = queryCtx->outputTransportType("contested_node");
  EXPECT_TRUE(
      outputType == ExchangeTransportType::kHttp ||
      outputType == ExchangeTransportType::kUcx);
}

} // namespace facebook::velox::core::test
