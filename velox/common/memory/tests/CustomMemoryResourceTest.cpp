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

  CustomMemoryResource emptyTag;
  emptyTag.allocator = makeResource("ignored").allocator;
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(emptyTag)),
      "CustomMemoryResource tag is empty");

  CustomMemoryResource nullAllocator;
  nullAllocator.tag = "another";
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullAllocator)),
      "CustomMemoryResource allocator is null for tag: another");

  auto nullArbitrator = makeResource("another");
  nullArbitrator.arbitrator.reset();
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullArbitrator)),
      "CustomMemoryResource arbitrator is null for tag: another");

  auto nullFactory = makeResource("another");
  nullFactory.reclaimerFactory = nullptr;
  VELOX_ASSERT_THROW(
      manager.registerCustomResource(std::move(nullFactory)),
      "CustomMemoryResource reclaimerFactory is null for tag: another");

  VELOX_ASSERT_THROW(
      manager.registerCustomResource(makeResource("test-resource")),
      "CustomMemoryResource already registered for tag: test-resource");

  ASSERT_EQ(manager.customResources().size(), 1);
}

TEST_F(CustomMemoryResourceManagerTest, customResourcesEmptyByDefault) {
  MemoryManager manager{};
  EXPECT_TRUE(manager.customResources().empty());
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

TEST_F(CustomMemoryResourceManagerTest, addRootPoolPropagatesResourceTag) {
  MemoryManager manager{};
  manager.registerCustomResource(makeResource("gpu"));

  auto tagged = manager.addRootPool(
      "tagged-root", kMaxMemory, nullptr, std::nullopt, "gpu");
  ASSERT_NE(tagged, nullptr);
  ASSERT_TRUE(tagged->resourceTag().has_value());
  EXPECT_EQ(*tagged->resourceTag(), "gpu");

  auto untagged = manager.addRootPool("untagged-root");
  EXPECT_FALSE(untagged->resourceTag().has_value());

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

TEST_F(CustomMemoryResourceQueryCtxTest, queryCtxWithoutResourcesHasNoCustomPools) {
  auto queryCtx = core::QueryCtx::Builder().queryId("q-empty").build();
  EXPECT_TRUE(queryCtx->customPools().empty());
  EXPECT_EQ(queryCtx->customPool("anything"), nullptr);
}

TEST_F(CustomMemoryResourceQueryCtxTest, addCustomPoolRejectsNull) {
  auto queryCtx = core::QueryCtx::Builder().queryId("q-null").build();
  VELOX_ASSERT_THROW(queryCtx->addCustomPool(nullptr), "");
}

TEST_F(CustomMemoryResourceQueryCtxTest, reclaimerFactoryExceptionPropagates) {
  auto resource = makeResource("boom");
  resource.reclaimerFactory =
      [](core::QueryCtx*) -> std::unique_ptr<MemoryReclaimer> {
    throw std::runtime_error("factory boom");
  };
  memoryManager()->registerCustomResource(std::move(resource));

  EXPECT_THROW(
      core::QueryCtx::Builder().queryId("q-boom").build(),
      std::runtime_error);
}

TEST_F(CustomMemoryResourceQueryCtxTest, multipleQueryCtxsGetDistinctCustomPools) {
  memoryManager()->registerCustomResource(makeResource("gpu"));

  auto q1 = core::QueryCtx::Builder().queryId("q-shared").build();
  auto q2 = core::QueryCtx::Builder().queryId("q-shared").build();
  auto p1 = q1->customPool("gpu");
  auto p2 = q2->customPool("gpu");
  ASSERT_NE(p1, nullptr);
  ASSERT_NE(p2, nullptr);
  EXPECT_NE(p1, p2);
  EXPECT_NE(p1->name(), p2->name());
}

TEST_F(CustomMemoryResourceQueryCtxTest, perQueryRootPoolCreated) {
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
  ASSERT_TRUE(pool->resourceTag().has_value());
  EXPECT_EQ(*pool->resourceTag(), "gpu");
  EXPECT_EQ(pool->maxCapacity(), 1L << 28);

  EXPECT_EQ(queryCtx->customPool("missing"), nullptr);
  EXPECT_EQ(queryCtx->customPools().size(), 1);
}

TEST_F(CustomMemoryResourceQueryCtxTest, customPoolsMatchRegistrationOrder) {
  for (const auto* tag : {"a", "b", "c"}) {
    memoryManager()->registerCustomResource(makeResource(tag));
  }

  auto queryCtx = core::QueryCtx::Builder().queryId("q-order").build();
  ASSERT_EQ(queryCtx->customPools().size(), 3);
  EXPECT_EQ(*queryCtx->customPools()[0]->resourceTag(), "a");
  EXPECT_EQ(*queryCtx->customPools()[1]->resourceTag(), "b");
  EXPECT_EQ(*queryCtx->customPools()[2]->resourceTag(), "c");
}

// Locks down the chained-reclaimer ordering contract: a later-registered
// resource's reclaimerFactory must see earlier resources' per-query pools
// already attached to the QueryCtx. A factory cannot see its own pool because
// the factory runs before that pool is added.
TEST_F(CustomMemoryResourceQueryCtxTest, reclaimerFactorySeesPriorSiblingPool) {
  memoryManager()->registerCustomResource(makeResource("primary"));

  std::shared_ptr<MemoryPool> capturedPrimary;
  std::shared_ptr<MemoryPool> capturedSelf;
  auto secondary = makeResource("secondary");
  secondary.reclaimerFactory = [&](core::QueryCtx* ctx) {
    capturedPrimary = ctx->customPool("primary");
    capturedSelf = ctx->customPool("secondary");
    return MemoryReclaimer::create(0);
  };
  memoryManager()->registerCustomResource(std::move(secondary));

  auto queryCtx = core::QueryCtx::Builder().queryId("q-chain").build();
  ASSERT_NE(capturedPrimary, nullptr);
  EXPECT_EQ(capturedPrimary, queryCtx->customPool("primary"));
  EXPECT_EQ(capturedSelf, nullptr);
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

// Gap-coverage tests below. These pin the contracts that allocator dispatch
// and pointer-stability must satisfy. They are expected to fail until the
// dispatch and storage gaps are implemented.

// Gap 1: a pool created against a custom resource must allocate through that
// resource's allocator, not the MemoryManager default.
TEST_F(
    CustomMemoryResourceQueryCtxTest,
    rootPoolDispatchesToCustomResourceAllocator) {
  auto resource = makeResource("gpu");
  auto* expectedAllocator = resource.allocator.get();
  memoryManager()->registerCustomResource(std::move(resource));

  auto queryCtx = core::QueryCtx::Builder().queryId("q-dispatch").build();
  auto pool = queryCtx->customPool("gpu");
  ASSERT_NE(pool, nullptr);
  auto* impl = static_cast<MemoryPoolImpl*>(pool.get());
  EXPECT_EQ(impl->testingAllocator(), expectedAllocator)
      << "Root pool must resolve its allocator from the custom resource, "
         "not the MemoryManager default.";
}

// Gap 3: a child of a custom-resource root pool must inherit the resource's
// allocator (and, transitively, its arbitrator once dispatch is wired).
TEST_F(
    CustomMemoryResourceQueryCtxTest,
    childPoolInheritsCustomResourceAllocator) {
  auto resource = makeResource("gpu");
  auto* expectedAllocator = resource.allocator.get();
  memoryManager()->registerCustomResource(std::move(resource));

  auto queryCtx = core::QueryCtx::Builder().queryId("q-child").build();
  auto root = queryCtx->customPool("gpu");
  ASSERT_NE(root, nullptr);
  auto aggregate = root->addAggregateChild("agg");
  auto leaf = aggregate->addLeafChild("leaf");

  EXPECT_EQ(
      static_cast<MemoryPoolImpl*>(aggregate.get())->testingAllocator(),
      expectedAllocator);
  EXPECT_EQ(
      static_cast<MemoryPoolImpl*>(leaf.get())->testingAllocator(),
      expectedAllocator);
  ASSERT_TRUE(leaf->resourceTag().has_value());
  EXPECT_EQ(*leaf->resourceTag(), "gpu");
}

// Gap 4: the registration list must offer stable references. Once dispatch is
// wired, pools will store CustomMemoryResource* pointers obtained at
// construction time; later registrations must not invalidate them.
//
// Compares addresses only — does not dereference the captured pointer, since
// dereferencing a dangling pointer on contract violation would crash gtest
// before it can report the failure.
TEST_F(CustomMemoryResourceManagerTest, registrationKeepsExistingPointersStable) {
  MemoryManager manager{};
  manager.registerCustomResource(makeResource("first"));
  const CustomMemoryResource* firstAddress =
      manager.customResources()[0].get();

  for (int i = 0; i < 128; ++i) {
    manager.registerCustomResource(makeResource(fmt::format("r{}", i)));
  }

  EXPECT_EQ(manager.customResources()[0].get(), firstAddress)
      << "Address of an already-registered resource must not change when "
         "new resources are added.";
  EXPECT_EQ(firstAddress->tag, "first");
}

} // namespace facebook::velox::memory::test
