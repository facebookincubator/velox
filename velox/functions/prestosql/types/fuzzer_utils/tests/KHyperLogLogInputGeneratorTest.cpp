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

#include "velox/functions/prestosql/types/fuzzer_utils/KHyperLogLogInputGenerator.h"

#include <gtest/gtest.h>
#include "velox/common/hyperloglog/KHyperLogLog.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

using namespace facebook::velox;
using namespace facebook::velox::fuzzer;

class KHyperLogLogInputGeneratorTest : public testing::Test {
 protected:
  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
};

TEST_F(KHyperLogLogInputGeneratorTest, generate) {
  KHyperLogLogInputGenerator generator(
      /*seed=*/1, /*nullRatio=*/0.0, pool_.get());

  for (int i = 0; i < 10; ++i) {
    auto result = generator.generate();
    ASSERT_FALSE(result.isNull());
    ASSERT_EQ(result.kind(), TypeKind::VARBINARY);

    // Verify deserialization works
    auto serialized = result.value<StringView>();
    auto khll = common::hll::KHyperLogLog<memory::MemoryPool>::deserialize(
        serialized.data(), serialized.size(), pool_.get());
    ASSERT_NE(khll, nullptr);

    // Verify cardinality is non-negative
    ASSERT_GE(khll->cardinality(), 0);
  }
}

TEST_F(KHyperLogLogInputGeneratorTest, generateWithNulls) {
  KHyperLogLogInputGenerator generator(
      /*seed=*/1, /*nullRatio=*/0.5, pool_.get());

  int nullCount = 0;
  int nonNullCount = 0;

  for (int i = 0; i < 100; ++i) {
    auto result = generator.generate();
    if (result.isNull()) {
      nullCount++;
    } else {
      nonNullCount++;
      ASSERT_EQ(result.kind(), TypeKind::VARBINARY);

      // Verify deserialization works
      auto serialized = result.value<StringView>();
      auto khll = common::hll::KHyperLogLog<memory::MemoryPool>::deserialize(
          serialized.data(), serialized.size(), pool_.get());
      ASSERT_NE(khll, nullptr);
    }
  }

  // With null ratio 0.5, we should have roughly half nulls
  ASSERT_GT(nullCount, 0);
  ASSERT_GT(nonNullCount, 0);
}

TEST_F(KHyperLogLogInputGeneratorTest, generateDifferentSeeds) {
  KHyperLogLogInputGenerator generator1(
      /*seed=*/1, /*nullRatio=*/0.0, pool_.get());
  KHyperLogLogInputGenerator generator2(
      /*seed=*/2, /*nullRatio=*/0.0, pool_.get());

  auto result1 = generator1.generate();
  auto result2 = generator2.generate();

  ASSERT_FALSE(result1.isNull());
  ASSERT_FALSE(result2.isNull());

  // Different seeds should produce different results
  auto serialized1 = result1.value<StringView>();
  auto serialized2 = result2.value<StringView>();
  ASSERT_NE(serialized1, serialized2);
}
