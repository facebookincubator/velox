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

#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::detail {
namespace {

class FlatMapKeyIndexTest : public testing::Test, public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(FlatMapKeyIndexTest, find) {
  auto keys = makeFlatVector<int64_t>({10, 20, 30});
  FlatMapKeyIndex index(*keys);
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

TEST_F(FlatMapKeyIndexTest, hashMatchesButKeyDoesNot) {
  auto keys = makeFlatVector<int64_t>({10});
  FlatMapKeyIndex index(*keys);
  EXPECT_EQ(
      index.find(keys->hashValueAt(0), [](column_index_t) { return false; }),
      std::nullopt);
}

TEST_F(FlatMapKeyIndexTest, collidingHashes) {
  FlatMapKeyIndex index(*makeFlatVector<int64_t>({}));
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

TEST_F(FlatMapKeyIndexTest, addAfterConstruction) {
  auto keys = makeFlatVector<int64_t>({10, 20});
  FlatMapKeyIndex index(*keys);
  index.add(folly::hasher<int64_t>{}(30));
  EXPECT_EQ(
      index.find(
          folly::hasher<int64_t>{}(30),
          [](column_index_t channel) { return channel == 2; }),
      2);
}

} // namespace
} // namespace facebook::velox::detail
