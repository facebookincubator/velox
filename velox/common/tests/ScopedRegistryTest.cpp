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

#include "velox/common/ScopedRegistry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox {
namespace {

// Minimal value type for testing.
class TestEntry {
 public:
  explicit TestEntry(const std::string& name) : name_{name} {}

  const std::string& name() const {
    return name_;
  }

 private:
  std::string name_;
};

TEST(ScopedRegistryTest, insertAndFind) {
  ScopedRegistry<std::string, TestEntry> registry;
  auto entry = std::make_shared<TestEntry>("test-1");
  registry.insert("test-1", entry);

  EXPECT_EQ(registry.find("test-1"), entry);
  EXPECT_EQ(registry.find("nonexistent"), nullptr);
}

TEST(ScopedRegistryTest, insertDuplicateThrows) {
  ScopedRegistry<std::string, TestEntry> registry;
  registry.insert("key", std::make_shared<TestEntry>("key"));

  EXPECT_THROW(
      registry.insert("key", std::make_shared<TestEntry>("key")),
      VeloxRuntimeError);
}

TEST(ScopedRegistryTest, insertOverwrite) {
  ScopedRegistry<std::string, TestEntry> registry;
  auto first = std::make_shared<TestEntry>("first");
  auto second = std::make_shared<TestEntry>("second");
  registry.insert("key", first);
  registry.insert("key", second, /*overwrite=*/true);

  EXPECT_EQ(registry.find("key"), second);
}

TEST(ScopedRegistryTest, erase) {
  ScopedRegistry<std::string, TestEntry> registry;
  registry.insert("key", std::make_shared<TestEntry>("key"));

  EXPECT_TRUE(registry.erase("key"));
  EXPECT_EQ(registry.find("key"), nullptr);
  EXPECT_FALSE(registry.erase("key"));
}

TEST(ScopedRegistryTest, clear) {
  ScopedRegistry<std::string, TestEntry> registry;
  registry.insert("a", std::make_shared<TestEntry>("a"));
  registry.insert("b", std::make_shared<TestEntry>("b"));
  registry.clear();

  EXPECT_EQ(registry.find("a"), nullptr);
  EXPECT_EQ(registry.find("b"), nullptr);
}

TEST(ScopedRegistryTest, snapshot) {
  ScopedRegistry<std::string, TestEntry> registry;
  registry.insert("a", std::make_shared<TestEntry>("a"));
  registry.insert("b", std::make_shared<TestEntry>("b"));

  auto entries = registry.snapshot();
  EXPECT_EQ(entries.size(), 2);

  std::set<std::string> keys;
  for (const auto& [key, _] : entries) {
    keys.insert(key);
  }
  EXPECT_THAT(keys, testing::UnorderedElementsAre("a", "b"));
}

TEST(ScopedRegistryTest, parentFallback) {
  ScopedRegistry<std::string, TestEntry> parent;
  auto entry = std::make_shared<TestEntry>("from-parent");
  parent.insert("key", entry);

  ScopedRegistry<std::string, TestEntry> child(&parent);
  EXPECT_EQ(child.find("key"), entry);
}

TEST(ScopedRegistryTest, childOverridesParent) {
  ScopedRegistry<std::string, TestEntry> parent;
  auto parentEntry = std::make_shared<TestEntry>("parent");
  parent.insert("key", parentEntry);

  ScopedRegistry<std::string, TestEntry> child(&parent);
  auto childEntry = std::make_shared<TestEntry>("child");
  child.insert("key", childEntry);

  EXPECT_EQ(child.find("key"), childEntry);
  EXPECT_EQ(parent.find("key"), parentEntry);
}

TEST(ScopedRegistryTest, childEraseDoesNotAffectParent) {
  ScopedRegistry<std::string, TestEntry> parent;
  auto entry = std::make_shared<TestEntry>("parent");
  parent.insert("key", entry);

  ScopedRegistry<std::string, TestEntry> child(&parent);
  child.insert("key", std::make_shared<TestEntry>("child"));
  child.erase("key");

  // Child erased its own override; parent entry is still visible via fallback.
  EXPECT_EQ(child.find("key"), entry);
  EXPECT_EQ(parent.find("key"), entry);
}

TEST(ScopedRegistryTest, snapshotMergesParent) {
  ScopedRegistry<std::string, TestEntry> parent;
  parent.insert("a", std::make_shared<TestEntry>("a"));
  parent.insert("b", std::make_shared<TestEntry>("b-parent"));

  ScopedRegistry<std::string, TestEntry> child(&parent);
  child.insert("b", std::make_shared<TestEntry>("b-child"));
  child.insert("c", std::make_shared<TestEntry>("c"));

  auto entries = child.snapshot();
  EXPECT_EQ(entries.size(), 3);

  std::map<std::string, std::string> snapshot;
  for (const auto& [key, value] : entries) {
    snapshot[key] = value->name();
  }
  EXPECT_EQ(snapshot["a"], "a");
  EXPECT_EQ(snapshot["b"], "b-child");
  EXPECT_EQ(snapshot["c"], "c");
}

} // namespace
} // namespace facebook::velox
