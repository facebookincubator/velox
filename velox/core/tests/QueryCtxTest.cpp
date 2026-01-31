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
      ConnectorConfigs{},
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
        ConnectorConfigs{},
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
        ConnectorConfigs{},
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
} // namespace facebook::velox::core::test
