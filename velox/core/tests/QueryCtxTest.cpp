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

#include <atomic>
#include <thread>

#include <folly/synchronization/Baton.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::core::test {

class QueryCtxTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(QueryCtxTest, credentialKeysWrite) {
  const std::string key{"token"};
  const std::string credential{"secret"};

  // Absent from the config: the credential becomes the stored value.
  {
    CredentialKeys keys;
    std::unordered_map<std::string, std::string> config;
    keys.write(config, "", key, credential, CredentialKeys::OnConflict::kKeep);
    EXPECT_EQ(config.at(key), credential);
    EXPECT_TRUE(keys.isQueryConfigCredential(key));
    EXPECT_FALSE(keys.isConnectorCredential("hive", key));
  }

  // kReplace overwrites whatever was there.
  {
    CredentialKeys keys;
    std::unordered_map<std::string, std::string> config{{key, "other"}};
    keys.write(
        config, "", key, credential, CredentialKeys::OnConflict::kReplace);
    EXPECT_EQ(config.at(key), credential);
    EXPECT_TRUE(keys.isQueryConfigCredential(key));
  }

  // kKeep leaves a different value in place. Recording the name anyway would
  // redact a property that replay has to parse back.
  {
    CredentialKeys keys;
    std::unordered_map<std::string, std::string> config{{key, "other"}};
    keys.write(config, "", key, credential, CredentialKeys::OnConflict::kKeep);
    EXPECT_EQ(config.at(key), "other");
    EXPECT_FALSE(keys.isQueryConfigCredential(key));
  }

  // kKeep over a value that happens to equal the credential records nothing
  // either. The stored value is the config's own, and redacting a typed
  // property because a credential collided with it would break replay.
  {
    CredentialKeys keys;
    std::unordered_map<std::string, std::string> config{{key, credential}};
    keys.write(config, "", key, credential, CredentialKeys::OnConflict::kKeep);
    EXPECT_EQ(config.at(key), credential);
    EXPECT_FALSE(keys.isQueryConfigCredential(key));
  }

  // A connector id routes the record away from the query config.
  {
    CredentialKeys keys;
    std::unordered_map<std::string, std::string> config;
    keys.write(
        config, "hive", key, credential, CredentialKeys::OnConflict::kKeep);
    EXPECT_EQ(config.at(key), credential);
    EXPECT_FALSE(keys.isQueryConfigCredential(key));
    EXPECT_TRUE(keys.isConnectorCredential("hive", key));
    EXPECT_FALSE(keys.isConnectorCredential("iceberg", key));
  }
}

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

namespace {
// Reports reclaimable memory and blocks in reclaim() until 'reclaimRelease' is
// posted, which keeps the query that owns the memory pool under memory
// arbitration.
class BlockingMemoryReclaimer : public memory::MemoryReclaimer {
 public:
  BlockingMemoryReclaimer(
      folly::Baton<>& reclaimStarted,
      folly::Baton<>& reclaimRelease)
      : memory::MemoryReclaimer(0),
        reclaimStarted_(reclaimStarted),
        reclaimRelease_(reclaimRelease) {}

  bool reclaimableBytes(
      const memory::MemoryPool& /*pool*/,
      uint64_t& reclaimableBytes) const override {
    reclaimableBytes = 1;
    return true;
  }

  uint64_t reclaim(
      memory::MemoryPool* /*pool*/,
      uint64_t /*targetBytes*/,
      uint64_t /*maxWaitMs*/,
      memory::MemoryReclaimer::Stats& /*stats*/) override {
    reclaimStarted_.post();
    reclaimRelease_.wait();
    return 0;
  }

 private:
  folly::Baton<>& reclaimStarted_;
  folly::Baton<>& reclaimRelease_;
};
} // namespace

TEST_F(QueryCtxTest, arbitrationWaitLimit) {
  auto queryCtx = QueryCtx::create();
  folly::Baton<> reclaimStarted;
  folly::Baton<> reclaimRelease;
  auto blockerPool = queryCtx->pool()->addLeafChild(
      "blocker",
      true,
      std::make_unique<BlockingMemoryReclaimer>(
          reclaimStarted, reclaimRelease));
  const auto startArbitration = [&]() {
    return std::thread([&]() {
      memory::MemoryReclaimer::Stats stats;
      queryCtx->pool()->reclaim(1, 0, stats);
    });
  };

  auto arbitrationThread = startArbitration();
  reclaimStarted.wait();
  ContinueFuture unlimitedWait = ContinueFuture::makeEmpty();
  const bool unlimitedWaitUnderArbitration =
      queryCtx->checkUnderArbitration(&unlimitedWait, 0);
  ContinueFuture limitedWait = ContinueFuture::makeEmpty();
  const bool limitedWaitUnderArbitration =
      queryCtx->checkUnderArbitration(&limitedWait, 500);

  // The limited wait ends with a value while the arbitration still runs. Once
  // the arbitration has lasted the limit, callers stop waiting for it. The
  // arbitration ends either way so that the test cannot hang.
  bool limitedWaitHasValue{false};
  if (limitedWaitUnderArbitration) {
    try {
      limitedWaitHasValue =
          std::move(limitedWait).getTry(std::chrono::seconds(30)).hasValue();
    } catch (const folly::FutureTimeout&) {
    }
  }
  const bool unlimitedWaitDone = unlimitedWait.isReady();
  ContinueFuture lateWait = ContinueFuture::makeEmpty();
  const bool lateWaitUnderArbitration =
      queryCtx->checkUnderArbitration(&lateWait, 1);
  reclaimRelease.post();
  arbitrationThread.join();
  ASSERT_TRUE(unlimitedWaitUnderArbitration);
  ASSERT_TRUE(limitedWaitUnderArbitration);
  ASSERT_TRUE(limitedWaitHasValue);
  ASSERT_FALSE(unlimitedWaitDone);
  ASSERT_FALSE(lateWaitUnderArbitration);
  ASSERT_FALSE(lateWait.valid());
  ASSERT_TRUE(unlimitedWait.isReady());
  ASSERT_FALSE(queryCtx->testingUnderArbitration());

  // The limit counts from the start of each arbitration: the first one has
  // lasted more than 400ms, the second one has just started.
  reclaimStarted.reset();
  reclaimRelease.reset();
  arbitrationThread = startArbitration();
  reclaimStarted.wait();
  ContinueFuture nextWait = ContinueFuture::makeEmpty();
  const bool nextWaitUnderArbitration =
      queryCtx->checkUnderArbitration(&nextWait, 400);
  reclaimRelease.post();
  arbitrationThread.join();
  ASSERT_TRUE(nextWaitUnderArbitration);
  ASSERT_TRUE(std::move(nextWait).wait(std::chrono::seconds(30)));
}

// Registry overrides follow a derived QueryCtx, and destination-only entries
// remain in place.
TEST_F(QueryCtxTest, copyRegistries) {
  struct RegistryA {};
  struct RegistryB {};
  struct RegistryC {};

  auto source = QueryCtx::create();
  auto destination = QueryCtx::create();
  auto registryA = std::make_shared<RegistryA>();
  auto registryB = std::make_shared<RegistryB>();
  auto registryC = std::make_shared<RegistryC>();
  source->setRegistry("a", registryA);
  source->setRegistry("b", registryB);
  destination->setRegistry("c", registryC);

  EXPECT_EQ(destination->registry<RegistryA>("a"), nullptr);
  EXPECT_EQ(destination->registry<RegistryB>("b"), nullptr);
  EXPECT_EQ(destination->registry<RegistryC>("c"), registryC);

  destination->copyRegistriesFrom(*source);

  EXPECT_EQ(destination->registry<RegistryA>("a"), registryA);
  EXPECT_EQ(destination->registry<RegistryB>("b"), registryB);
  EXPECT_EQ(destination->registry<RegistryC>("c"), registryC);
}

// Registry copies reject conflicts unless replacement is requested, and a
// rejected copy leaves the destination unchanged.
TEST_F(QueryCtxTest, copyRegistryConflicts) {
  struct RegistryA {};
  struct RegistryB {};
  struct RegistryC {};

  auto source = QueryCtx::create();
  auto destination = QueryCtx::create();
  auto registryA = std::make_shared<RegistryA>();
  auto sourceRegistryB = std::make_shared<RegistryB>();
  auto destinationRegistryB = std::make_shared<RegistryB>();
  auto registryC = std::make_shared<RegistryC>();
  source->setRegistry("a", registryA);
  source->setRegistry("b", sourceRegistryB);
  destination->setRegistry("b", destinationRegistryB);
  destination->setRegistry("c", registryC);

  VELOX_ASSERT_THROW(
      destination->copyRegistriesFrom(*source), "Registry already set: b");
  EXPECT_EQ(destination->registry<RegistryA>("a"), nullptr);
  EXPECT_EQ(destination->registry<RegistryB>("b"), destinationRegistryB);
  EXPECT_EQ(destination->registry<RegistryC>("c"), registryC);

  destination->copyRegistriesFrom(*source, /*overwrite=*/true);
  EXPECT_EQ(destination->registry<RegistryA>("a"), registryA);
  EXPECT_EQ(destination->registry<RegistryB>("b"), sourceRegistryB);
  EXPECT_EQ(destination->registry<RegistryC>("c"), registryC);
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

TEST_F(QueryCtxTest, concurrentReleaseCallbackRegistration) {
  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kCallbacksPerThread = 500;
  std::atomic<int32_t> callbackCount{0};

  {
    auto queryCtx = QueryCtx::create(
        nullptr,
        QueryConfig{{}},
        std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
        nullptr,
        nullptr,
        nullptr,
        "test_query_id");

    // Tasks of one query register release callbacks from their own threads.
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int32_t i = 0; i < kNumThreads; ++i) {
      threads.emplace_back([&]() {
        for (int32_t j = 0; j < kCallbacksPerThread; ++j) {
          queryCtx->addReleaseCallback([&callbackCount]() { ++callbackCount; });
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  ASSERT_EQ(callbackCount, kNumThreads * kCallbacksPerThread);
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
} // namespace facebook::velox::core::test
