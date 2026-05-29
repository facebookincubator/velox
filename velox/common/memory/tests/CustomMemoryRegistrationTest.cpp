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
#include "velox/common/memory/CustomMemoryResource.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"

namespace facebook::velox::memory::test {
namespace {

std::shared_ptr<MemoryAllocator> makeAllocator() {
  MemoryAllocator::Options options;
  options.capacity = 1L << 30;
  return std::make_shared<MallocAllocator>(options);
}

CustomMemoryResource::ReclaimerFactory makeReclaimerFactory() {
  return []() { return MemoryReclaimer::create(0); };
}

std::shared_ptr<CustomMemoryResource> makeResource(const std::string& tag) {
  return std::make_shared<CustomMemoryResource>(
      tag,
      makeAllocator(),
      MemoryArbitrator::create({}),
      makeReclaimerFactory());
}

} // namespace

TEST(CustomMemoryRegistration, constructorRejectsInvalidFields) {
  auto allocator = makeAllocator();
  std::shared_ptr<MemoryArbitrator> arbitrator = MemoryArbitrator::create({});
  auto factory = makeReclaimerFactory();

  // Valid construction succeeds.
  CustomMemoryResource("ok", allocator, arbitrator, factory);

  VELOX_ASSERT_THROW(
      CustomMemoryResource("", allocator, arbitrator, factory),
      "CustomMemoryResource tag is empty");
  VELOX_ASSERT_THROW(
      CustomMemoryResource("tag", nullptr, arbitrator, factory),
      "CustomMemoryResource allocator is null for tag: tag");
  VELOX_ASSERT_THROW(
      CustomMemoryResource("tag", allocator, nullptr, factory),
      "CustomMemoryResource arbitrator is null for tag: tag");
  VELOX_ASSERT_THROW(
      CustomMemoryResource("tag", allocator, arbitrator, nullptr),
      "CustomMemoryResource reclaimerFactory is null for tag: tag");
}

TEST(CustomMemoryRegistration, insertAndFindOnIsolatedRegistry) {
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  for (const auto* tag : {"a", "b", "c"}) {
    registry->insert(tag, makeResource(tag));
  }
  EXPECT_NE(registry->find("a"), nullptr);
  EXPECT_NE(registry->find("b"), nullptr);
  EXPECT_NE(registry->find("c"), nullptr);
  EXPECT_EQ(registry->find("missing"), nullptr);
}

TEST(CustomMemoryRegistration, insertRejectsDuplicateTag) {
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert("device", makeResource("device"));
  VELOX_ASSERT_THROW(
      registry->insert("device", makeResource("device")),
      "Key already registered: device");
}

TEST(CustomMemoryRegistration, childRegistryShadowsParent) {
  auto parent = CustomMemoryResourceRegistry::createRegistry(nullptr);
  auto defaultResource = makeResource("device");
  auto* defaultAllocator = defaultResource->allocator();
  parent->insert("device", std::move(defaultResource));

  auto child = CustomMemoryResourceRegistry::createRegistry(parent.get());
  auto tenantResource = makeResource("device");
  auto* tenantAllocator = tenantResource->allocator();
  child->insert("device", std::move(tenantResource));

  EXPECT_EQ(child->find("device")->allocator(), tenantAllocator);
  EXPECT_EQ(parent->find("device")->allocator(), defaultAllocator)
      << "Parent must remain untouched by scoped registration.";
}

TEST(CustomMemoryRegistration, childRegistryFallsBackToParent) {
  auto parent = CustomMemoryResourceRegistry::createRegistry(nullptr);
  parent->insert("only-parent", makeResource("only-parent"));
  auto child = CustomMemoryResourceRegistry::createRegistry(parent.get());
  EXPECT_NE(child->find("only-parent"), nullptr);
}

TEST(CustomMemoryRegistration, isolationRegistryHasNoFallback) {
  auto parent = CustomMemoryResourceRegistry::createRegistry(nullptr);
  parent->insert("only-parent", makeResource("only-parent"));

  auto isolated = CustomMemoryResourceRegistry::createRegistry(nullptr);
  EXPECT_EQ(isolated->find("only-parent"), nullptr)
      << "Isolation mode must not fall back to any parent.";

  isolated->insert("only-local", makeResource("only-local"));
  EXPECT_NE(isolated->find("only-local"), nullptr);
  EXPECT_EQ(parent->find("only-local"), nullptr);
}

TEST(CustomMemoryRegistration, addCustomRootPoolWithResource) {
  MemoryManager manager{};
  auto registry = CustomMemoryResourceRegistry::createRegistry(nullptr);
  registry->insert("gpu", makeResource("gpu"));

  auto resource = registry->find("gpu");
  ASSERT_NE(resource, nullptr);

  auto tagged = manager.addCustomRootPool("tagged-root", resource);
  ASSERT_NE(tagged, nullptr);
}

} // namespace facebook::velox::memory::test
