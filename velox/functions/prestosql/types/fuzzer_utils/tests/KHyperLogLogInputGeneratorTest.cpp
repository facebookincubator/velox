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
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

namespace facebook::velox::fuzzer::test {

class KHyperLogLogInputGeneratorTest
    : public functions::test::FunctionBaseTest {};

TEST_F(KHyperLogLogInputGeneratorTest, generate) {
  KHyperLogLogInputGenerator generator(1, 0.0, pool());

  for (int i = 0; i < 10; ++i) {
    auto result = generator.generate();
    ASSERT_FALSE(result.isNull());
    ASSERT_EQ(result.kind(), TypeKind::VARBINARY);

    // Verify deserialization works
    const auto& serialized = result.value<TypeKind::VARBINARY>();
    HashStringAllocator allocator{pool()};
    auto khll =
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
            serialized.data(), serialized.size(), &allocator);
    ASSERT_TRUE(khll.hasValue())
        << "Deserialization failed: " << khll.error().message();
    ASSERT_NE(*khll, nullptr);
    ASSERT_GE((*khll)->cardinality(), 0);
  }
}

TEST_F(KHyperLogLogInputGeneratorTest, generateWithNulls) {
  KHyperLogLogInputGenerator generator(1, 0.5, pool());

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
      const auto& serialized = result.value<TypeKind::VARBINARY>();
      HashStringAllocator allocator{pool()};
      auto khll =
          common::hll::KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
              serialized.data(), serialized.size(), &allocator);
      ASSERT_TRUE(khll.hasValue())
          << "Deserialization failed: " << khll.error().message();
    }
  }

  // With null ratio 0.5, we should have roughly half nulls
  ASSERT_GT(nullCount, 0);
  ASSERT_GT(nonNullCount, 0);
}

TEST_F(KHyperLogLogInputGeneratorTest, generateDifferentSeeds) {
  KHyperLogLogInputGenerator generator1(1, 0.0, pool());
  KHyperLogLogInputGenerator generator2(2, 0.0, pool());

  auto result1 = generator1.generate();
  auto result2 = generator2.generate();

  ASSERT_FALSE(result1.isNull());
  ASSERT_FALSE(result2.isNull());
  ASSERT_EQ(result1.kind(), TypeKind::VARBINARY);
  ASSERT_EQ(result2.kind(), TypeKind::VARBINARY);

  // Different seeds should produce different results
  const auto& serialized1 = result1.value<TypeKind::VARBINARY>();
  const auto& serialized2 = result2.value<TypeKind::VARBINARY>();
  ASSERT_NE(serialized1, serialized2);
}

} // namespace facebook::velox::fuzzer::test
