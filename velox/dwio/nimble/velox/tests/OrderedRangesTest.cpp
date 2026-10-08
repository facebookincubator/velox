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
#include "velox/dwio/nimble/velox/OrderedRanges.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>

namespace facebook::nimble::tests {
namespace {

using namespace testing;
using range_helper::OrderedRanges;

template <typename T>
std::vector<std::tuple<int32_t, int32_t>> collectRange(const T& t) {
  std::vector<std::tuple<int32_t, int32_t>> ret;
  t.apply([&](auto begin, auto size) { ret.emplace_back(begin, size); });
  return ret;
}

template <typename T>
std::vector<int32_t> collectEach(const T& t) {
  std::vector<int32_t> ret;
  t.applyEach([&](auto val) { ret.emplace_back(val); });
  return ret;
}

TEST(OrderedRanges, apply) {
  OrderedRanges ranges;
  ranges.add(0, 1);
  ranges.add(1, 1);
  ranges.add(100, 1);
  ranges.add(50, 50);
  EXPECT_THAT(
      collectRange(ranges),
      ElementsAre(
          std::make_tuple(0, 2),
          std::make_tuple(100, 1),
          std::make_tuple(50, 50)));
}

TEST(OrderedRanges, applyEach) {
  OrderedRanges ranges;
  ranges.add(0, 5);
  ranges.add(5, 30);
  ranges.add(35, 15);

  std::vector<int32_t> expected(50);
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_THAT(collectEach(ranges), ElementsAreArray(expected));
}

} // namespace
} // namespace facebook::nimble::tests
