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
#include "velox/exec/OutputBufferManagerRegistry.h"

#include <gtest/gtest.h>

#include "velox/common/base/VeloxException.h"

namespace facebook::velox::exec {
namespace {

class MockOutputBufferManager : public IOutputBufferManager {
 public:
  void initializeTask(
      std::shared_ptr<Task> /*task*/,
      core::PartitionedOutputNode::Kind /*kind*/,
      int /*numDestinations*/,
      int /*numDrivers*/) override {}

  bool updateOutputBuffers(
      const std::string& /*taskId*/,
      int /*numBuffers*/,
      bool /*noMoreBuffers*/) override {
    return true;
  }

  void removeTask(const std::string& /*taskId*/) override {}

  std::optional<OutputBuffer::Stats> stats(
      const std::string& /*taskId*/) override {
    return std::nullopt;
  }
};

class OutputBufferManagerRegistryTest : public testing::Test {
 protected:
  void SetUp() override {
    OutputBufferManagerRegistry::clear();
  }

  void TearDown() override {
    OutputBufferManagerRegistry::clear();
  }
};

TEST_F(OutputBufferManagerRegistryTest, registerAndGet) {
  auto manager = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::registerManager("test", manager);
  ASSERT_EQ(OutputBufferManagerRegistry::getManager("test"), manager);
}

TEST_F(OutputBufferManagerRegistryTest, getMissing) {
  ASSERT_EQ(OutputBufferManagerRegistry::getManager("nonexistent"), nullptr);
}

TEST_F(OutputBufferManagerRegistryTest, hasManager) {
  ASSERT_FALSE(OutputBufferManagerRegistry::hasManager("test"));
  OutputBufferManagerRegistry::registerManager(
      "test", std::make_shared<MockOutputBufferManager>());
  ASSERT_TRUE(OutputBufferManagerRegistry::hasManager("test"));
}

TEST_F(OutputBufferManagerRegistryTest, doubleRegisterFails) {
  OutputBufferManagerRegistry::registerManager(
      "test", std::make_shared<MockOutputBufferManager>());
  ASSERT_THROW(
      OutputBufferManagerRegistry::registerManager(
          "test", std::make_shared<MockOutputBufferManager>()),
      ::facebook::velox::VeloxRuntimeError);
}

TEST_F(OutputBufferManagerRegistryTest, unregister) {
  OutputBufferManagerRegistry::registerManager(
      "test", std::make_shared<MockOutputBufferManager>());
  ASSERT_TRUE(OutputBufferManagerRegistry::unregisterManager("test"));
  ASSERT_FALSE(OutputBufferManagerRegistry::hasManager("test"));
  ASSERT_FALSE(OutputBufferManagerRegistry::unregisterManager("test"));
}

TEST_F(OutputBufferManagerRegistryTest, getAllManagers) {
  auto m1 = std::make_shared<MockOutputBufferManager>();
  auto m2 = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::registerManager("a", m1);
  OutputBufferManagerRegistry::registerManager("b", m2);
  auto all = OutputBufferManagerRegistry::getAllManagers();
  ASSERT_EQ(all.size(), 2);
}

TEST_F(OutputBufferManagerRegistryTest, getManagerAs) {
  auto manager = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::registerManager("test", manager);
  auto typed =
      OutputBufferManagerRegistry::getManagerAs<MockOutputBufferManager>(
          "test");
  ASSERT_NE(typed, nullptr);
  ASSERT_EQ(typed, manager);
}

TEST_F(OutputBufferManagerRegistryTest, getManagerAsWrongType) {
  auto manager = std::make_shared<MockOutputBufferManager>();
  OutputBufferManagerRegistry::registerManager("test", manager);
  auto base =
      OutputBufferManagerRegistry::getManagerAs<IOutputBufferManager>("test");
  ASSERT_NE(base, nullptr);
}

TEST_F(OutputBufferManagerRegistryTest, clear) {
  OutputBufferManagerRegistry::registerManager(
      "a", std::make_shared<MockOutputBufferManager>());
  OutputBufferManagerRegistry::registerManager(
      "b", std::make_shared<MockOutputBufferManager>());
  OutputBufferManagerRegistry::clear();
  ASSERT_EQ(OutputBufferManagerRegistry::getAllManagers().size(), 0);
}

} // namespace
} // namespace facebook::velox::exec
