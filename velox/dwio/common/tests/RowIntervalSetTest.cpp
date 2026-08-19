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

#include "velox/dwio/common/RowIntervalSet.h"

#include <gtest/gtest.h>

using namespace facebook::velox::dwio::common;

TEST(RowIntervalSetTest, normalizesIntervals) {
  RowIntervalSet intervals;
  intervals.add({10, 20});
  intervals.add({0, 5});
  intervals.add({5, 10});
  intervals.add({19, 30});

  ASSERT_EQ(intervals.intervals().size(), 1);
  EXPECT_EQ(intervals.intervals()[0].begin, 0);
  EXPECT_EQ(intervals.intervals()[0].end, 30);
}

TEST(RowIntervalSetTest, setOperations) {
  RowIntervalSet left;
  left.add({0, 10});
  left.add({20, 30});
  RowIntervalSet right;
  right.add({5, 25});

  EXPECT_EQ(RowIntervalSet::setUnion(left, right).toString(), "[[0, 30)]");
  EXPECT_EQ(
      RowIntervalSet::intersection(left, right).toString(),
      "[[5, 10), [20, 25)]");
  EXPECT_EQ(
      RowIntervalSet::difference(left, right).toString(),
      "[[0, 5), [25, 30)]");
}

TEST(RowIntervalSetTest, setOperationsDoNotMutateInputs) {
  RowIntervalSet left;
  left.add({0, 10});
  RowIntervalSet right;
  right.add({5, 15});

  EXPECT_EQ(RowIntervalSet::setUnion(left, right).toString(), "[[0, 15)]");
  EXPECT_EQ(left.toString(), "[[0, 10)]");
  EXPECT_EQ(right.toString(), "[[5, 15)]");

  EXPECT_EQ(
      RowIntervalSet::intersection(left, right).toString(), "[[5, 10)]");
  EXPECT_EQ(left.toString(), "[[0, 10)]");
  EXPECT_EQ(right.toString(), "[[5, 15)]");
}

TEST(RowIntervalSetTest, splitUsesCursor) {
  RowIntervalSet retained;
  retained.add({10, 20});
  retained.add({30, 40});

  size_t cursor = 0;
  auto split = retained.firstSplit({0, 50}, cursor);
  EXPECT_FALSE(split.second);
  EXPECT_EQ(split.first.begin, 0);
  EXPECT_EQ(split.first.end, 10);

  split = retained.firstSplit({10, 50}, cursor);
  EXPECT_TRUE(split.second);
  EXPECT_EQ(split.first.begin, 10);
  EXPECT_EQ(split.first.end, 20);

  split = retained.firstSplit({20, 50}, cursor);
  EXPECT_FALSE(split.second);
  EXPECT_EQ(split.first.begin, 20);
  EXPECT_EQ(split.first.end, 30);

  split = retained.firstSplit({30, 50}, cursor);
  EXPECT_TRUE(split.second);
  EXPECT_EQ(split.first.begin, 30);
  EXPECT_EQ(split.first.end, 40);
}

TEST(RowIntervalSetTest, handlesLargeBounds) {
  RowIntervalSet intervals;
  intervals.add({UINT64_MAX - 10, UINT64_MAX});
  intervals.add({UINT64_MAX - 20, UINT64_MAX - 10});
  EXPECT_EQ(
      intervals.toString(),
      "[[18446744073709551595, 18446744073709551615)]");
  EXPECT_TRUE(intervals.overlaps({UINT64_MAX - 1, UINT64_MAX}));
  EXPECT_FALSE(intervals.overlaps({0, 1}));
}

TEST(RowIntervalSetTest, full) {
  EXPECT_TRUE(RowIntervalSet::full(0).intervals().empty());
  EXPECT_EQ(RowIntervalSet::full(4).toString(), "[[0, 4)]");
}

TEST(RowIntervalSetTest, reversedIntervalIsEmpty) {
  const RowInterval interval{5, 3};
  EXPECT_TRUE(interval.empty());
  EXPECT_EQ(interval.size(), 0);
}