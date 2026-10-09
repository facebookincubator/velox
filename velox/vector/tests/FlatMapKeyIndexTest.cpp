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

#include "velox/vector/FlatMapKeyIndex.h"

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/type/tests/utils/CustomTypesForTesting.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::detail {
namespace {

class FlatMapKeyIndexTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(FlatMapKeyIndexTest, hashedFind) {
  auto keys = makeFlatVector<int64_t>({10, 20, 30});
  FlatMapHashedKeyIndex index(*keys);
  auto findKey = [&](int64_t key) {
    return index.find(
        folly::hasher<int64_t>{}(key),
        [&](column_index_t channel) { return keys->valueAt(channel) == key; });
  };

  EXPECT_EQ(findKey(10), 0);
  EXPECT_EQ(findKey(20), 1);
  EXPECT_EQ(findKey(30), 2);
  EXPECT_EQ(findKey(40), std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, hashedHashMatchesButKeyDoesNot) {
  auto keys = makeFlatVector<int64_t>({10});
  FlatMapHashedKeyIndex index(*keys);
  EXPECT_EQ(
      index.find(keys->hashValueAt(0), [](column_index_t) { return false; }),
      std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, hashedCollidingHashes) {
  FlatMapHashedKeyIndex index(*makeFlatVector<int64_t>({}));
  index.add(7);
  index.add(7);
  index.add(8);
  auto findChannel = [&](uint64_t hash, column_index_t wanted) {
    return index.find(
        hash, [&](column_index_t channel) { return channel == wanted; });
  };

  EXPECT_EQ(findChannel(7, 0), 0);
  EXPECT_EQ(findChannel(7, 1), 1);
  EXPECT_EQ(findChannel(8, 2), 2);
  EXPECT_EQ(findChannel(7, 2), std::nullopt);
  EXPECT_EQ(findChannel(9, 0), std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, hashedAddAfterConstruction) {
  auto keys = makeFlatVector<int64_t>({10, 20});
  FlatMapHashedKeyIndex index(*keys);
  index.add(folly::hasher<int64_t>{}(30));
  EXPECT_EQ(
      index.find(
          folly::hasher<int64_t>{}(30),
          [](column_index_t channel) { return channel == 2; }),
      2);
}

TEST_F(FlatMapKeyIndexTest, integerKeys) {
  auto testKeys = [&](const VectorPtr& keys, const auto& missing) {
    SCOPED_TRACE(keys->type()->toString());
    FlatMapKeyIndex index(*keys);
    for (vector_size_t i = 0; i < keys->size(); ++i) {
      EXPECT_EQ(index.find(*keys, *keys, i), i);
    }
    EXPECT_EQ(index.find(*keys, *missing, 0), std::nullopt);
  };
  testKeys(makeFlatVector<int8_t>({1, 2, 3}), makeFlatVector<int8_t>({4}));
  testKeys(makeFlatVector<int16_t>({1, 2, 3}), makeFlatVector<int16_t>({4}));
  testKeys(makeFlatVector<int32_t>({1, 2, 3}), makeFlatVector<int32_t>({4}));
  testKeys(makeFlatVector<int64_t>({1, 2, 3}), makeFlatVector<int64_t>({4}));
  testKeys(makeFlatVector<bool>({true}), makeFlatVector<bool>({false}));
}

TEST_F(FlatMapKeyIndexTest, typedFind) {
  auto keys = makeFlatVector<int64_t>({10, 20, 30});
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(index.find(*keys, int64_t{20}), 1);
  EXPECT_EQ(index.find(*keys, int64_t{40}), std::nullopt);
  VELOX_ASSERT_THROW(
      index.find(*keys, int32_t{20}),
      "Incompatible vector type for flat map vector keys");
}

TEST_F(FlatMapKeyIndexTest, repeatedIntegerKeyResolvesToLastChannel) {
  auto keys = makeFlatVector<int64_t>({10, 20, 10});
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(index.find(*keys, int64_t{10}), 2);
}

TEST_F(FlatMapKeyIndexTest, appendIntegerKey) {
  auto keys = makeFlatVector<int64_t>({10, 20});
  FlatMapKeyIndex index(*keys);
  keys->resize(3);
  keys->set(2, 30);
  index.appendLast(*keys);
  EXPECT_EQ(index.find(*keys, int64_t{30}), 2);
  EXPECT_EQ(index.find(*keys, int64_t{10}), 0);
}

TEST_F(FlatMapKeyIndexTest, nullProbe) {
  auto keys = makeFlatVector<int64_t>({10});
  FlatMapKeyIndex index(*keys);
  auto probe = makeNullableFlatVector<int64_t>({std::nullopt});
  EXPECT_EQ(index.find(*keys, *probe, 0), std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, nullDistinctKey) {
  auto keys = makeNullableFlatVector<int64_t>({1, std::nullopt, 0});
  FlatMapKeyIndex index(*keys);
  auto nullProbe = makeNullableFlatVector<int64_t>({std::nullopt});
  EXPECT_EQ(index.find(*keys, *nullProbe, 0), 1);
  EXPECT_EQ(index.find(*keys, int64_t{0}), 2);
  EXPECT_EQ(index.find(*keys, int64_t{1}), 0);
}

TEST_F(FlatMapKeyIndexTest, appendNullKey) {
  auto keys = makeFlatVector<int64_t>({0, 1});
  FlatMapKeyIndex index(*keys);
  keys->resize(3);
  keys->setNull(2, true);
  index.appendLast(*keys);
  auto nullProbe = makeNullableFlatVector<int64_t>({std::nullopt});
  EXPECT_EQ(index.find(*keys, *nullProbe, 0), 2);
  EXPECT_EQ(index.find(*keys, int64_t{0}), 0);
  EXPECT_EQ(index.find(*keys, int64_t{1}), 1);
}

TEST_F(FlatMapKeyIndexTest, lazyVectors) {
  auto keys = makeFlatVector<int64_t>({10, 20});
  FlatMapKeyIndex index(*keys);
  auto lazyProbe = makeLazyFlatVector<int64_t>(1, [](auto) { return 20; });
  EXPECT_EQ(index.find(*keys, *lazyProbe, 0), 1);

  auto lazyKeys =
      makeLazyFlatVector<int64_t>(2, [](auto row) { return 10 * (row + 1); });
  FlatMapKeyIndex lazyIndex(*lazyKeys);
  EXPECT_EQ(lazyIndex.find(*lazyKeys, int64_t{20}), 1);
}

TEST_F(FlatMapKeyIndexTest, probeOfDifferentType) {
  auto keys = makeFlatVector<int64_t>({1});
  FlatMapKeyIndex index(*keys);
  VELOX_ASSERT_THROW(
      index.find(*keys, *makeFlatVector<int32_t>({1}), 0),
      "Incompatible vector type for flat map vector keys");
}

TEST_F(FlatMapKeyIndexTest, emptyKeys) {
  auto keys = makeFlatVector<int64_t>(std::vector<int64_t>{});
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(index.find(*keys, int64_t{10}), std::nullopt);

  keys->resize(1);
  keys->set(0, 10);
  index.appendLast(*keys);
  EXPECT_EQ(index.find(*keys, int64_t{10}), 0);
}

TEST_F(FlatMapKeyIndexTest, encodedProbes) {
  auto keys = makeFlatVector<int64_t>({10, 20, 30});
  FlatMapKeyIndex index(*keys);

  auto constantProbe = makeConstant<int64_t>(20, 5);
  EXPECT_EQ(index.find(*keys, *constantProbe, 3), 1);

  auto dictionaryProbe = BaseVector::wrapInDictionary(
      nullptr, makeIndices({1, 0}), 2, makeFlatVector<int64_t>({10, 30}));
  EXPECT_EQ(index.find(*keys, *dictionaryProbe, 0), 2);
  EXPECT_EQ(index.find(*keys, *dictionaryProbe, 1), 0);
}

TEST_F(FlatMapKeyIndexTest, nullFromDictionaryWrapper) {
  // The base has no nulls; the dictionary wrapper makes row 1 null.
  BufferPtr nulls = allocateNulls(3, pool());
  bits::setNull(nulls->asMutable<uint64_t>(), 1);
  auto keys = BaseVector::wrapInDictionary(
      nulls, makeIndices({0, 0, 1}), 3, makeFlatVector<int64_t>({10, 20}));
  FlatMapKeyIndex index(*keys);
  auto nullProbe = makeNullableFlatVector<int64_t>({std::nullopt});
  EXPECT_EQ(index.find(*keys, *nullProbe, 0), 1);
  EXPECT_EQ(index.find(*keys, int64_t{10}), 0);
  EXPECT_EQ(index.find(*keys, int64_t{20}), 2);
}

TEST_F(FlatMapKeyIndexTest, customComparisonKeysUseHashedIndex) {
  // This type compares only the bottom 8 bits, so 257 equals 1.
  const auto type = test::BIGINT_TYPE_WITH_CUSTOM_COMPARISON();
  auto keys = makeFlatVector<int64_t>({1, 2}, type);
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(index.find(*keys, *makeFlatVector<int64_t>({257}, type), 0), 0);
  EXPECT_EQ(
      index.find(*keys, *makeFlatVector<int64_t>({3}, type), 0), std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, appendNonIntegerKey) {
  auto keys = makeFlatVector<StringView>({"a", "b"});
  FlatMapKeyIndex index(*keys);
  keys->resize(3);
  keys->set(2, StringView("c"));
  index.appendLast(*keys);
  EXPECT_EQ(index.find(*keys, StringView("c")), 2);
  EXPECT_EQ(index.find(*keys, StringView("a")), 0);
}

TEST_F(FlatMapKeyIndexTest, nonIntegerKeysUseHashedIndex) {
  auto keys = makeFlatVector<StringView>({"a", "b"});
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(index.find(*keys, StringView("b")), 1);
  EXPECT_EQ(index.find(*keys, *keys, 0), 0);
  EXPECT_EQ(index.find(*keys, StringView("c")), std::nullopt);
}

} // namespace
} // namespace facebook::velox::detail
