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
  resource.reclaimerFactory = []() { return MemoryReclaimer::create(0); };
  return resource;
}

// Builds a root pool from a CustomMemoryResource that the caller has already
// registered on 'manager'. The reclaimer is whatever the caller chooses; for
// tests that don't need a QueryCtx-aware reclaimer this is just the resource's
// default factory.
std::shared_ptr<MemoryPool> buildPool(
    MemoryManager* manager,
    const CustomMemoryResource& resource,
    const std::string& poolName,
    int64_t maxCapacity = kMaxMemory,
    std::unique_ptr<MemoryReclaimer> reclaimer = nullptr) {
  if (reclaimer == nullptr && resource.reclaimerFactory) {
    reclaimer = resource.reclaimerFactory();
  }
  return manager->addRootPool(
      poolName,
      maxCapacity,
      std::move(reclaimer),
      std::nullopt,
      resource.tag);
}

} // namespace

// Each test gets its own MemoryManager so per-test resource registrations
// cannot collide on tag.
class CustomMemoryPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
  }
};

TEST_F(CustomMemoryPoolTest, customPoolCreation) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("gpu"));
  const auto& registered = *manager->customResources().at("gpu");

  auto pool = buildPool(
      manager, registered, "query.q0.gpu", /*maxCapacity=*/1L << 28);
  auto queryCtx = core::QueryCtx::Builder()
                      .customPool("gpu", pool)
                      .queryId("q0")
                      .build();

  auto looked = queryCtx->customPool("gpu");
  ASSERT_NE(looked, nullptr);
  EXPECT_EQ(looked.get(), pool.get());
  EXPECT_EQ(looked->name(), "query.q0.gpu");
  EXPECT_EQ(looked->maxCapacity(), 1L << 28);

  EXPECT_EQ(queryCtx->customPool("missing"), nullptr);
  EXPECT_EQ(queryCtx->customPools().size(), 1);
}

TEST_F(CustomMemoryPoolTest, customPoolsKeyedByTag) {
  auto* manager = memoryManager();
  for (const auto* tag : {"a", "b", "c"}) {
    manager->registerCustomResource(makeResource(tag));
  }

  auto builder = core::QueryCtx::Builder().queryId("q-keyed");
  for (const auto* tag : {"a", "b", "c"}) {
    const auto& registered = *manager->customResources().at(tag);
    builder.customPool(
        tag, buildPool(manager, registered, fmt::format("q-keyed.{}", tag)));
  }
  auto queryCtx = builder.build();

  ASSERT_EQ(queryCtx->customPools().size(), 3);
  EXPECT_NE(queryCtx->customPool("a"), nullptr);
  EXPECT_NE(queryCtx->customPool("b"), nullptr);
  EXPECT_NE(queryCtx->customPool("c"), nullptr);
}

// Locks down the core dispatch invariant: a custom-resource root pool and its
// children must allocate through the resource's allocator, not the
// MemoryManager default.
TEST_F(CustomMemoryPoolTest, customPoolDispatchesToResourceAllocator) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("gpu"));
  const auto& registered = *manager->customResources().at("gpu");
  auto* expectedAllocator = registered.allocator.get();

  auto pool = buildPool(manager, registered, "q-dispatch.gpu");
  auto queryCtx = core::QueryCtx::Builder()
                      .customPool("gpu", pool)
                      .queryId("q-dispatch")
                      .build();
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

namespace {

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
    auto leaf =
        sibling_->addLeafChild(fmt::format("spill-{}", numReclaimCalls_));
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

} // namespace

// Locks down the cross-resource spill flow: when reclaim is triggered on a
// custom pool, the resource's reclaimer can allocate into a sibling resource's
// pool, modeling device -> pinned-host spill. The caller builds the host pool
// first, captures it inside the device reclaimer, then builds the device pool
// with that reclaimer — no QueryCtx pointer is needed during reclaimer
// construction because the sibling pool itself is the link.
TEST_F(CustomMemoryPoolTest, deviceReclaimerSpillsToHostSibling) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("host"));
  manager->registerCustomResource(makeResource("device"));
  const auto& hostResource = *manager->customResources().at("host");
  const auto& deviceResource = *manager->customResources().at("device");

  auto hostPool = buildPool(manager, hostResource, "q-spill.host");
  auto deviceReclaimer = SpillToSiblingReclaimer::create(hostPool);
  auto* mockReclaimer =
      static_cast<SpillToSiblingReclaimer*>(deviceReclaimer.get());
  auto devicePool = buildPool(
      manager,
      deviceResource,
      "q-spill.device",
      kMaxMemory,
      std::move(deviceReclaimer));

  auto queryCtx = core::QueryCtx::Builder()
                      .customPool("host", hostPool)
                      .customPool("device", devicePool)
                      .queryId("q-spill")
                      .build();
  ASSERT_EQ(queryCtx->customPool("host").get(), hostPool.get());
  ASSERT_EQ(queryCtx->customPool("device").get(), devicePool.get());

  const uint64_t target = 4 * 1024;
  MemoryReclaimer::Stats stats;
  const auto reclaimed = devicePool->reclaim(target, 0, stats);

  EXPECT_EQ(mockReclaimer->numReclaimCalls(), 1);
  EXPECT_EQ(reclaimed, target);
  EXPECT_GE(hostPool->usedBytes(), static_cast<int64_t>(target));
}

// Builder rejects duplicate tags and null pools.
TEST_F(CustomMemoryPoolTest, builderRejectsBadInputs) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("gpu"));
  const auto& registered = *manager->customResources().at("gpu");
  auto pool = buildPool(manager, registered, "q-bad.gpu");

  auto builder = core::QueryCtx::Builder().queryId("q-bad");
  builder.customPool("gpu", pool);
  EXPECT_THROW(builder.customPool("gpu", pool), VeloxRuntimeError);
  EXPECT_THROW(builder.customPool("other", nullptr), VeloxRuntimeError);
  EXPECT_THROW(builder.customPool("", pool), VeloxRuntimeError);
}

// QueryCtx::addCustomPool is part of the public surface so a query can
// attach a custom pool after the Builder has already produced the QueryCtx.
// Verifies the same validation as Builder::customPool applies.
TEST_F(CustomMemoryPoolTest, addCustomPoolDirectly) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("gpu"));
  const auto& registered = *manager->customResources().at("gpu");

  auto queryCtx = core::QueryCtx::Builder().queryId("q-direct").build();
  ASSERT_EQ(queryCtx->customPools().size(), 0);

  auto pool = buildPool(manager, registered, "q-direct.gpu");
  queryCtx->addCustomPool("gpu", pool);
  EXPECT_EQ(queryCtx->customPool("gpu").get(), pool.get());
  EXPECT_EQ(queryCtx->customPools().size(), 1);

  EXPECT_THROW(queryCtx->addCustomPool("gpu", pool), VeloxRuntimeError);
  EXPECT_THROW(queryCtx->addCustomPool("other", nullptr), VeloxRuntimeError);
  EXPECT_THROW(queryCtx->addCustomPool("", pool), VeloxRuntimeError);
}

// Models the QueryCtx-aware reclaimer flow: build the pool without a
// reclaimer (skipping the resource's factory), hand it to the Builder, then
// attach a reclaimer that references the QueryCtx via
// MemoryPool::setReclaimer. The reclaimer reaches a sibling pool through
// queryCtx->customPool().
TEST_F(CustomMemoryPoolTest, postBuildSetReclaimerWithQueryCtxAware) {
  auto* manager = memoryManager();
  manager->registerCustomResource(makeResource("host"));
  manager->registerCustomResource(makeResource("device"));

  auto hostPool = manager->addRootPool(
      "q-postbuild.host",
      kMaxMemory,
      /*reclaimer=*/nullptr,
      std::nullopt,
      "host");
  auto devicePool = manager->addRootPool(
      "q-postbuild.device",
      kMaxMemory,
      /*reclaimer=*/nullptr,
      std::nullopt,
      "device");

  auto queryCtx = core::QueryCtx::Builder()
                      .customPool("host", hostPool)
                      .customPool("device", devicePool)
                      .queryId("q-postbuild")
                      .build();

  // Attach a sibling-aware reclaimer to the device pool after the QueryCtx
  // exists, reaching the host pool through customPool().
  auto reclaimer =
      SpillToSiblingReclaimer::create(queryCtx->customPool("host"));
  auto* mockReclaimer = static_cast<SpillToSiblingReclaimer*>(reclaimer.get());
  devicePool->setReclaimer(std::move(reclaimer));

  const uint64_t target = 4 * 1024;
  MemoryReclaimer::Stats stats;
  const auto reclaimed = devicePool->reclaim(target, 0, stats);

  EXPECT_EQ(mockReclaimer->numReclaimCalls(), 1);
  EXPECT_EQ(reclaimed, target);
  EXPECT_GE(hostPool->usedBytes(), static_cast<int64_t>(target));
}

} // namespace facebook::velox::memory::test
