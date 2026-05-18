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
#include "velox/common/memory/MallocAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/MemoryArbitrator.h"

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

} // namespace

TEST(CustomMemoryRegistration, registrationValidation) {
  MemoryManager manager{};
  ASSERT_TRUE(manager.customResources().empty());

  auto resource = makeResource("test-resource");
  resource.maxCapacity = 1L << 28;
  manager.registerCustomResource(std::move(resource));

  ASSERT_EQ(manager.customResources().size(), 1);
  const auto& stored = manager.customResources().at("test-resource");
  EXPECT_EQ(stored->tag, "test-resource");
  EXPECT_EQ(stored->maxCapacity, 1L << 28);

  // Try to register another resource with the same tag, which should be
  // rejected.
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

TEST(CustomMemoryRegistration, registrationsAreReachableByTag) {
  MemoryManager manager{};
  for (const auto* tag : {"a", "b", "c"}) {
    manager.registerCustomResource(makeResource(tag));
  }
  ASSERT_EQ(manager.customResources().size(), 3);
  EXPECT_NE(manager.customResources().find("a"), manager.customResources().end());
  EXPECT_NE(manager.customResources().find("b"), manager.customResources().end());
  EXPECT_NE(manager.customResources().find("c"), manager.customResources().end());
}

TEST(CustomMemoryRegistration, addRootPoolRejectsUnregisteredTag) {
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

} // namespace facebook::velox::memory::test
