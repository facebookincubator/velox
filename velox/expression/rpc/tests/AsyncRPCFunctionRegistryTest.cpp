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

#include "velox/expression/rpc/AsyncRPCFunctionRegistry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::exec::rpc {
namespace {

class AsyncRPCFunctionRegistryTest : public testing::Test {
 protected:
  void SetUp() override {
    AsyncRPCFunctionRegistry::testingClear();
  }

  void TearDown() override {
    AsyncRPCFunctionRegistry::testingClear();
  }

  static AsyncRPCFunctionRegistry::Factory factory() {
    return []() -> std::shared_ptr<AsyncRPCFunction> { return nullptr; };
  }

  static AsyncRPCFunctionRegistry::Signatures signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("varchar")
                .argumentType("varchar")
                .build()};
  }
};

TEST_F(AsyncRPCFunctionRegistryTest, find) {
  AsyncRPCFunctionRegistry::Metadata metadata;
  metadata.defaultNullBehavior = false;
  AsyncRPCFunctionRegistry::registerFunction(
      "echo", factory(), signatures(), metadata);

  const auto entry = AsyncRPCFunctionRegistry::find("echo");
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->name, "echo");
  ASSERT_EQ(entry->signatures.size(), 1);
  EXPECT_EQ(entry->signatures[0]->toString(), signatures()[0]->toString());
  EXPECT_FALSE(entry->metadata.defaultNullBehavior);

  EXPECT_FALSE(AsyncRPCFunctionRegistry::find("missing").has_value());
}

// A function nothing can resolve a call to is not registrable.
TEST_F(AsyncRPCFunctionRegistryTest, registerWithoutSignatures) {
  VELOX_ASSERT_THROW(
      AsyncRPCFunctionRegistry::registerFunction("bare", factory(), {}),
      "RPC function must be registered with at least one signature: bare");
  EXPECT_FALSE(AsyncRPCFunctionRegistry::find("bare").has_value());
}

TEST_F(AsyncRPCFunctionRegistryTest, functions) {
  AsyncRPCFunctionRegistry::registerFunction("echo", factory(), signatures());
  AsyncRPCFunctionRegistry::registerFunction(
      "reverse", factory(), signatures());

  std::vector<std::string> names;
  for (const auto& entry : AsyncRPCFunctionRegistry::functions()) {
    names.push_back(entry.name);
  }
  EXPECT_THAT(names, testing::UnorderedElementsAre("echo", "reverse"));
}

} // namespace
} // namespace facebook::velox::exec::rpc
