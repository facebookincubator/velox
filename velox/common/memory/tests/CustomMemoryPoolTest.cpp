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
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/core/QueryCtx.h"

namespace facebook::velox::memory::test {
namespace {

std::shared_ptr<CustomMemoryResource> makeResource(
    const std::string& tag,
    int64_t maxCapacity = 1L << 30) {
  MemoryAllocator::Options allocatorOptions;
  allocatorOptions.capacity = maxCapacity;
  return std::make_shared<CustomMemoryResource>(
      tag,
      std::make_shared<MallocAllocator>(allocatorOptions),
      MemoryArbitrator::create({}),
      []() { return MemoryReclaimer::create(0); },
      maxCapacity);
}

} // namespace

// Each test gets a fresh MemoryManager and its own isolated registry so
// concurrent tests cannot collide on tag registration in the global scope.
class CustomMemoryPoolTest : public testing::Test {
 protected:
  void SetUp() override {
    MemoryManager::testingSetInstance(MemoryManager::Options{});
    registry_ = CustomMemoryResourceRegistry::createRegistry(nullptr);
  }

  std::shared_ptr<CustomMemoryResourceRegistry::Registry> registry_;
};

TEST_F(CustomMemoryPoolTest, customPoolCreation) {
  auto* manager = memoryManager();
  registry_->insert("gpu", makeResource("gpu", /*maxCapacity=*/1L << 28));
  auto resource = registry_->find("gpu");
  ASSERT_NE(resource, nullptr);

  auto pool = manager->addCustomRootPool("query.q0.gpu", resource);
  auto queryCtx =
      core::QueryCtx::Builder().customPool("gpu", pool).queryId("q0").build();

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
    registry_->insert(tag, makeResource(tag));
  }

  auto builder = core::QueryCtx::Builder().queryId("q-keyed");
  for (const auto* tag : {"a", "b", "c"}) {
    auto resource = registry_->find(tag);
    ASSERT_NE(resource, nullptr);
    builder.customPool(
        tag,
        manager->addCustomRootPool(fmt::format("q-keyed.{}", tag), resource));
  }
  auto queryCtx = builder.build();

  ASSERT_EQ(queryCtx->customPools().size(), 3);
  EXPECT_NE(queryCtx->customPool("a"), nullptr);
  EXPECT_NE(queryCtx->customPool("b"), nullptr);
  EXPECT_NE(queryCtx->customPool("c"), nullptr);
}

// A custom-resource root pool and its children must allocate through the
// resource's allocator, not the MemoryManager default.
TEST_F(CustomMemoryPoolTest, customPoolDispatchesToResourceAllocator) {
  auto* manager = memoryManager();
  registry_->insert("gpu", makeResource("gpu"));
  auto resource = registry_->find("gpu");
  ASSERT_NE(resource, nullptr);
  auto* expectedAllocator = resource->allocator();

  auto pool = manager->addCustomRootPool("q-dispatch.gpu", resource);
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

TEST_F(CustomMemoryPoolTest, builderRejectsBadInputs) {
  auto* manager = memoryManager();
  registry_->insert("gpu", makeResource("gpu"));
  auto resource = registry_->find("gpu");
  ASSERT_NE(resource, nullptr);
  auto pool = manager->addCustomRootPool("q-bad.gpu", resource);

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
  registry_->insert("gpu", makeResource("gpu"));
  auto resource = registry_->find("gpu");
  ASSERT_NE(resource, nullptr);

  auto queryCtx = core::QueryCtx::Builder().queryId("q-direct").build();
  ASSERT_EQ(queryCtx->customPools().size(), 0);

  auto pool = manager->addCustomRootPool("q-direct.gpu", resource);
  queryCtx->addCustomPool("gpu", pool);
  EXPECT_EQ(queryCtx->customPool("gpu").get(), pool.get());
  EXPECT_EQ(queryCtx->customPools().size(), 1);

  EXPECT_THROW(queryCtx->addCustomPool("gpu", pool), VeloxRuntimeError);
  EXPECT_THROW(queryCtx->addCustomPool("other", nullptr), VeloxRuntimeError);
  EXPECT_THROW(queryCtx->addCustomPool("", pool), VeloxRuntimeError);
}

TEST_F(CustomMemoryPoolTest, addCustomRootPoolRejectsNullResource) {
  auto* manager = memoryManager();
  EXPECT_THROW(manager->addCustomRootPool("q.null", nullptr), VeloxUserError);
}

namespace {

// Test reclaimer that "spills" by allocating from a sibling resource's pool.
// Models the chained-reclaimer pattern (e.g. device-memory pool spills into a
// pinned-host pool). The sibling pool is captured via shared_ptr.
class SpillToSiblingReclaimer : public MemoryReclaimer {
 public:
  static std::unique_ptr<MemoryReclaimer> create(
      std::shared_ptr<MemoryPool> sibling) {
    return std::unique_ptr<MemoryReclaimer>(
        new SpillToSiblingReclaimer(std::move(sibling)));
  }

  ~SpillToSiblingReclaimer() override {
    // Release any buffers acquired during reclaim before the leaf pools
    // they came from get destroyed; otherwise the leaf destructor aborts
    // on outstanding usage.
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

// The device resource's reclaimerFactory closes over the previously-built
// host pool, so reclaim on the device pool routes allocations into the host.
TEST_F(CustomMemoryPoolTest, deviceReclaimerSpillsToHostSibling) {
  auto* manager = memoryManager();

  registry_->insert("host", makeResource("host"));
  auto hostResource = registry_->find("host");
  ASSERT_NE(hostResource, nullptr);
  auto hostPool = manager->addCustomRootPool("q-spill.host", hostResource);

  MemoryAllocator::Options deviceAllocatorOptions;
  deviceAllocatorOptions.capacity = 1L << 30;
  auto deviceResource = std::make_shared<CustomMemoryResource>(
      "device",
      std::make_shared<MallocAllocator>(deviceAllocatorOptions),
      MemoryArbitrator::create({}),
      [hostPool]() { return SpillToSiblingReclaimer::create(hostPool); });

  auto devicePool =
      manager->addCustomRootPool("q-spill.device", deviceResource);
  auto* mockReclaimer =
      static_cast<SpillToSiblingReclaimer*>(devicePool->reclaimer());
  ASSERT_NE(mockReclaimer, nullptr);

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

} // namespace facebook::velox::memory::test
