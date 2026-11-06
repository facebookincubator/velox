/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <algorithm>
#include <string_view>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "velox/dwio/nimble/index/KeyComparator.h"

namespace facebook::nimble::index {
namespace {

using namespace std::literals;

// Orders keys in reverse bytewise order.
class ReverseKeyComparator final : public KeyComparator {
 public:
  int compare(std::string_view lhs, std::string_view rhs) const override {
    return rhs.compare(lhs);
  }
};

// Returns keys in strictly ascending bytewise order: a key sorts before the
// keys it prefixes, including a zero byte suffix, and bytes compare unsigned.
std::vector<std::string_view> bytewiseKeys() {
  return {""sv, "a"sv, "a\0"sv, "ab"sv, "b"sv, "\x80"sv};
}

// Expects 'keyOrder' to order 'keys' strictly ascending through both
// compare() and operator().
template <typename KeyOrder>
void expectStrictlyAscending(
    const KeyOrder& keyOrder,
    const std::vector<std::string_view>& keys) {
  for (size_t i = 1; i < keys.size(); ++i) {
    SCOPED_TRACE(fmt::format("i={}", i));
    EXPECT_LT(keyOrder.compare(keys[i - 1], keys[i]), 0);
    EXPECT_GT(keyOrder.compare(keys[i], keys[i - 1]), 0);
    EXPECT_EQ(keyOrder.compare(keys[i], keys[i]), 0);
    EXPECT_TRUE(keyOrder(keys[i - 1], keys[i]));
    EXPECT_FALSE(keyOrder(keys[i], keys[i - 1]));
    EXPECT_FALSE(keyOrder(keys[i], keys[i]));
  }
}

TEST(KeyComparatorTest, bytewiseKeyOrder) {
  expectStrictlyAscending(BytewiseKeyOrder{}, bytewiseKeys());
}

TEST(KeyComparatorTest, comparatorKeyOrder) {
  const ReverseKeyComparator comparator;
  const auto keys = bytewiseKeys();
  expectStrictlyAscending(
      ComparatorKeyOrder{&comparator},
      std::vector<std::string_view>{keys.rbegin(), keys.rend()});
}

TEST(KeyComparatorTest, compareKeys) {
  const ReverseKeyComparator comparator;
  EXPECT_LT(compareKeys("a", "b", /*comparator=*/nullptr), 0);
  EXPECT_EQ(compareKeys("a", "a", /*comparator=*/nullptr), 0);
  EXPECT_GT(compareKeys("a", "b", &comparator), 0);
  EXPECT_EQ(compareKeys("a", "a", &comparator), 0);
}

TEST(KeyComparatorTest, withKeyOrder) {
  const ReverseKeyComparator comparator;
  const auto isBytewise = [](auto keyOrder) {
    return std::is_same_v<decltype(keyOrder), BytewiseKeyOrder>;
  };
  EXPECT_TRUE(withKeyOrder(/*comparator=*/nullptr, isBytewise));
  EXPECT_FALSE(withKeyOrder(&comparator, isBytewise));

  // Returns the result of the function, here the position of the first key
  // that does not sort before "b".
  const auto lowerBound = [](const std::vector<std::string_view>& keys,
                             const KeyComparator* keyComparator) {
    return withKeyOrder(keyComparator, [&](auto keyOrder) {
      return std::lower_bound(keys.begin(), keys.end(), "b"sv, keyOrder) -
          keys.begin();
    });
  };
  EXPECT_EQ(lowerBound({"a", "b", "c"}, /*keyComparator=*/nullptr), 1);
  EXPECT_EQ(lowerBound({"c", "b", "a"}, &comparator), 1);
  EXPECT_EQ(lowerBound({"c", "a"}, &comparator), 1);
}

} // namespace
} // namespace facebook::nimble::index
