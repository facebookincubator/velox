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
#include "velox/common/base/Exceptions.h"
#include "velox/common/serialization/Registry.h"

using namespace ::facebook::velox;

namespace {
TEST(Registry, smartPointerFactoryWithNoArgument) {
  Registry<size_t, std::unique_ptr<size_t>()> registry;

  const size_t key = 0;
  const size_t value = 1;

  EXPECT_FALSE(registry.Has(key));
  EXPECT_EQ(registry.Create(key), nullptr);

  registry.Register(0, [value]() -> std::unique_ptr<size_t> {
    return std::make_unique<size_t>(value);
  });

  EXPECT_TRUE(registry.Has(key));
  EXPECT_EQ(*registry.Create(key), value);
}

TEST(Registry, valueFactoryWithArguments) {
  Registry<size_t, size_t(size_t, size_t)> registry;

  const size_t key = 0;

  EXPECT_FALSE(registry.Has(key));
  EXPECT_THROW(registry.Create(key, 1, 1), facebook::velox::VeloxUserError);

  registry.Register(0, [](size_t l, size_t r) -> size_t { return l + r; });

  EXPECT_TRUE(registry.Has(key));
  EXPECT_EQ(registry.Create(key, 1, 1), 2);
}

TEST(FunctionRegistry, registerDifferentFunctions) {
  Registry<ssize_t, std::unique_ptr<size_t>()> registry;
  const size_t key1 = 0;
  const size_t key2 = 1;

  auto dummyFunction = []() -> std::unique_ptr<size_t> { return nullptr; };

  // Test registering 2 functions with different keys (should succeed)
  registry.Register(key1, dummyFunction);
  registry.Register(key2, dummyFunction);

  ASSERT_EQ(registry.Keys().size(), 2);
  ASSERT_TRUE(registry.Has(key1));
  ASSERT_TRUE(registry.Has(key2));
}

TEST(FunctionRegistry, registerSameFunction) {
  Registry<ssize_t, std::unique_ptr<size_t>()> registry;
  const size_t key = 0;

  auto dummyFunction = []() -> std::unique_ptr<size_t> { return nullptr; };

  // Test registering the same key twice, the second attempt should fail
  registry.Register(key, dummyFunction);
  ASSERT_THROW(registry.Register(key, dummyFunction), VeloxUserError);
}
} // namespace
