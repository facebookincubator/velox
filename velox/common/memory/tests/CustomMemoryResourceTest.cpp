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

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::memory::test {
namespace {

CustomMemoryResource makeResource(const std::string& tag) {
  MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = 1L << 30;
  CustomMemoryResource resource;
  resource.tag = tag;
  resource.allocator = std::make_shared<MallocAllocator>(allocatorOptions);
  resource.arbitrator = MemoryArbitrator::create({});
  resource.reclaimerFactory = [](core::QueryCtx*) {
    return MemoryReclaimer::create(0);
  };
  return resource;
}

} // namespace

// MemoryManager-level tests use a local MemoryManager so they don't touch
// the process-wide singleton.
class CustomMemoryResourceManagerTest : public testing::Test {};

TEST_F(CustomMemoryResourceManagerTest, registrationValidation) {
  MemoryManager manager{};
  ASSERT_TRUE(manager.customResources().empty());

  auto resource = makeResource("test-resource");
  resource.maxCapacity = 1L << 28;
  manager.registerCustomResource(std::move(resource));

  ASSERT_EQ(manager.customResources().size(), 1);
  EXPECT_EQ(manager.customResources()[0]->tag, "test-resource");
  EXPECT_EQ(manager.customResources()[0]->maxCapacity, 1L << 28);

  // Try to register another resource with the same tag, which should be rejected.
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(makeResource("test-resource")),
      "CustomMemoryResource already registered for tag: test-resource");
  ASSERT_EQ(manager.customResources().size(), 1);

  // Try to register resources with missing tag, which should be rejected.
  CustomMemoryResource emptyTag;
  emptyTag.allocator = makeResource("ignored").allocator;
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(emptyTag)),
      "CustomMemoryResource tag is empty");

  // Try to register resources with null allocator, which should be rejected.
  CustomMemoryResource nullAllocator;
  nullAllocator.tag = "another";
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullAllocator)),
      "CustomMemoryResource allocator is null for tag: another");

  // Try to register resources with null arbitrator, which should be rejected.
  auto nullArbitrator = makeResource("another");
  nullArbitrator.arbitrator.reset();
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullArbitrator)),
      "CustomMemoryResource arbitrator is null for tag: another");

  // Try to register resources with null reclaimerFactory, which should be
  // rejected.
  auto nullFactory = makeResource("another");
  nullFactory.reclaimerFactory = nullptr;
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullFactory)),
      "CustomMemoryResource reclaimerFactory is null for tag: another");
}

TEST_F(CustomMemoryResourceManagerTest, registrationOrderPreserved) {
  MemoryManager manager{};
  for (const auto* tag : {"a", "b", "c"}) {
    manager.registerCustomResource(makeResource(tag));
  }
  ASSERT_EQ(manager.customResources().size(), 3);
  EXPECT_EQ(manager.customResources()[0]->tag, "a");
  EXPECT_EQ(manager.customResources()[1]->tag, "b");
  EXPECT_EQ(manager.customResources()[2]->tag, "c");
}

TEST_F(CustomMemoryResourceManagerTest, addRootPoolRejectsUnregisteredTag) {
  MemoryManager manager{};
  manager.registerCustomResource(makeResource("gpu"));

  // Adding a root pool with a registered tag should succeed.
  auto tagged = manager.addRootPool(
      "tagged-root", kMaxMemory, nullptr, std::nullopt, "gpu");
  ASSERT_NE(tagged, nullptr);

  // Adding a root pool without a tag should succeed.
  auto untagged = manager.addRootPool("untagged-root");
  ASSERT_NE(untagged, nullptr);

  // Adding a root pool with an unregistered tag should be rejected.
  VELOX_ASSERT_THROW(
      manager.addRootPool(
          "bad-root", kMaxMemory, nullptr, std::nullopt, "not-registered"),
      "No CustomMemoryResource registered for tag: not-registered");
}

// QueryCtx-level tests run against the process-wide MemoryManager because
// QueryCtx::Builder reads it directly. Each test cleans up after itself so
// resources don't leak into siblings.
class CustomMemoryResourceQueryCtxTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }

  void TearDown() override {
    memoryManager()->testingClearCustomResources();
  }
};

TEST_F(CustomMemoryResourceQueryCtxTest, customPoolCreation) {
  auto resource = makeResource("gpu");
  resource.maxCapacity = 1L << 28;

  bool factoryInvoked = false;
  core::QueryCtx* factorySawCtx = nullptr;
  resource.reclaimerFactory = [&](core::QueryCtx* ctx) {
    factoryInvoked = true;
    factorySawCtx = ctx;
    return MemoryReclaimer::create(0);
  };

  memoryManager()->registerCustomResource(std::move(resource));

  auto queryCtx = core::QueryCtx::Builder().queryId("q0").build();
  EXPECT_TRUE(factoryInvoked);
  EXPECT_EQ(factorySawCtx, queryCtx.get());

  auto pool = queryCtx->customPool("gpu");
  ASSERT_NE(pool, nullptr);
  EXPECT_TRUE(pool->name().starts_with("query.q0."));
  EXPECT_TRUE(pool->name().ends_with(".gpu"));
  EXPECT_EQ(pool->maxCapacity(), 1L << 28);

  EXPECT_EQ(queryCtx->customPool("missing"), nullptr);
  EXPECT_EQ(queryCtx->customPools().size(), 1);
}

TEST_F(CustomMemoryResourceQueryCtxTest, customPoolsKeyedByTag) {
  for (const auto* tag : {"a", "b", "c"}) {
    memoryManager()->registerCustomResource(makeResource(tag));
  }

  auto queryCtx = core::QueryCtx::Builder().queryId("q-keyed").build();
  ASSERT_EQ(queryCtx->customPools().size(), 3);
  EXPECT_NE(queryCtx->customPool("a"), nullptr);
  EXPECT_NE(queryCtx->customPool("b"), nullptr);
  EXPECT_NE(queryCtx->customPool("c"), nullptr);
}

// Locks down the core dispatch invariant: a custom-resource root pool and its
// children must allocate through the resource's allocator, not the
// MemoryManager default.
TEST_F(
    CustomMemoryResourceQueryCtxTest,
    customPoolDispatchesToResourceAllocator) {
  auto resource = makeResource("gpu");
  auto* expectedAllocator = resource.allocator.get();
  memoryManager()->registerCustomResource(std::move(resource));

  auto queryCtx = core::QueryCtx::Builder().queryId("q-dispatch").build();
  auto root = queryCtx->customPool("gpu");
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(
      static_cast<MemoryPoolImpl*>(root.get())->testingAllocator(),
      expectedAllocator);

  auto aggregate = root->addAggregateChild("agg");
  auto leaf = aggregate->addLeafChild("leaf");
  EXPECT_EQ(
      static_cast<MemoryPoolImpl*>(aggregate.get())->testingAllocator(),
      expectedAllocator);
  EXPECT_EQ(
      static_cast<MemoryPoolImpl*>(leaf.get())->testingAllocator(),
      expectedAllocator);
}

// Test reclaimer that "spills" by allocating from a sibling resource's pool.
// Models the chained-reclaimer pattern (e.g. device-memory pool spills into a
// pinned-host pool).
class SpillToSiblingReclaimer : public MemoryReclaimer {
 public:
  static std::unique_ptr<MemoryReclaimer> create(
      std::shared_ptr<MemoryPool> sibling) {
    return std::unique_ptr<MemoryReclaimer>(
        new SpillToSiblingReclaimer(std::move(sibling)));
  }

  ~SpillToSiblingReclaimer() override {
    // Release any buffers we acquired during reclaim before the leaf pools
    // they came from get destroyed; otherwise the leaf destructor aborts on
    // outstanding usage.
    for (auto& spill : spills_) {
      spill.leaf->free(spill.buffer, static_cast<int64_t>(spill.bytes));
    }
  }

  uint64_t reclaim(
      MemoryPool* /*pool*/,
      uint64_t targetBytes,
      uint64_t /*maxWaitMs*/,
      Stats& /*stats*/) override {
    ++numReclaimCalls_;
    auto leaf = sibling_->addLeafChild(
        fmt::format("spill-{}", numReclaimCalls_));
    void* buffer = leaf->allocate(static_cast<int64_t>(targetBytes));
    spills_.push_back({std::move(leaf), buffer, targetBytes});
    return targetBytes;
  }

  int numReclaimCalls() const {
    return numReclaimCalls_;
  }

 private:
  explicit SpillToSiblingReclaimer(std::shared_ptr<MemoryPool> sibling)
      : MemoryReclaimer(0), sibling_(std::move(sibling)) {}

  struct Spill {
    std::shared_ptr<MemoryPool> leaf;
    void* buffer;
    uint64_t bytes;
  };

  std::shared_ptr<MemoryPool> sibling_;
  int numReclaimCalls_{0};
  std::vector<Spill> spills_;
};

// Locks down the cross-resource spill flow: when reclaim is triggered on a
// custom pool, the resource's reclaimer can allocate into a sibling resource's
// pool, modeling device -> pinned-host spill.
TEST_F(CustomMemoryResourceQueryCtxTest, deviceReclaimerSpillsToHostSibling) {
  memoryManager()->registerCustomResource(makeResource("host"));

  SpillToSiblingReclaimer* mockReclaimer = nullptr;
  auto device = makeResource("device");
  device.reclaimerFactory = [&](core::QueryCtx* ctx) {
    auto reclaimer = SpillToSiblingReclaimer::create(ctx->customPool("host"));
    mockReclaimer = static_cast<SpillToSiblingReclaimer*>(reclaimer.get());
    return reclaimer;
  };
  memoryManager()->registerCustomResource(std::move(device));

  auto queryCtx = core::QueryCtx::Builder().queryId("q-spill").build();
  auto devicePool = queryCtx->customPool("device");
  auto hostPool = queryCtx->customPool("host");
  ASSERT_NE(devicePool, nullptr);
  ASSERT_NE(hostPool, nullptr);
  ASSERT_NE(mockReclaimer, nullptr);

  const uint64_t target = 4 * 1024;
  MemoryReclaimer::Stats stats;
  const auto reclaimed = devicePool->reclaim(target, 0, stats);

  EXPECT_EQ(mockReclaimer->numReclaimCalls(), 1);
  EXPECT_EQ(reclaimed, target);
  EXPECT_GE(hostPool->usedBytes(), static_cast<int64_t>(target));
}
} // namespace facebook::velox::memory::test
