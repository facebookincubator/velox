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

#include "velox/functions/sparksql/MightContain.h"
#include "velox/common/base/BloomFilter.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using namespace facebook::velox::test;

class MightContainTest : public SparkFunctionBaseTest {
 protected:
  void testMightContain(
      const VectorPtr& bloom,
      const VectorPtr& value,
      const VectorPtr& expected) {
    auto result = evaluate(
        "might_contain(cast(c0 as varbinary), c1)",
        makeRowVector({bloom, value}));
    assertEqualVectors(expected, result);
  }

  std::string getSerializedBloomFilter() {
    constexpr int64_t kSize = 10;
    BloomFilter bloom;
    bloom.reset(kSize);
    for (auto i = 0; i < kSize; ++i) {
      bloom.insert(folly::hasher<int64_t>()(i));
    }
    std::string data;
    data.resize(bloom.serializedSize());
    bloom.serialize(data.data());
    return data;
  }
};

TEST_F(MightContainTest, basic) {
  auto serialized = getSerializedBloomFilter();
  auto bloom = makeConstant<StringView>(StringView(serialized), 10);
  auto value =
      makeFlatVector<int64_t>(10, [](vector_size_t row) { return row; });
  auto expected = makeConstant(true, 10);
  testMightContain(bloom, value, expected);

  auto valueNotContain = makeFlatVector<int64_t>(
      10, [](vector_size_t row) { return row + 123451; });
  auto expectedNotContain = makeConstant(false, 10);
  testMightContain(bloom, valueNotContain, expectedNotContain);

  auto values = makeNullableFlatVector<int64_t>(
      {1, 2, 3, 4, 5, std::nullopt, 123451, 23456, 4, 5});
  auto expects = makeNullableFlatVector<bool>(
      {true, true, true, true, true, std::nullopt, false, false, true, true});
  testMightContain(bloom, values, expects);
}

TEST_F(MightContainTest, nullBloom) {
  auto bloom = makeConstant<StringView>(std::nullopt, 2);
  auto value = makeFlatVector<int64_t>({2, 4});
  auto expected = makeNullConstant(TypeKind::BOOLEAN, 2);
  testMightContain(bloom, value, expected);
}

TEST_F(MightContainTest, nullValue) {
  auto serializedBloom = getSerializedBloomFilter();
  auto bloom = makeConstant<StringView>(StringView(serializedBloom), 2);
  auto value = makeNullableFlatVector<int64_t>({std::nullopt, 2});
  auto expected = makeNullableFlatVector<bool>({std::nullopt, true});
  testMightContain(bloom, value, expected);
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
