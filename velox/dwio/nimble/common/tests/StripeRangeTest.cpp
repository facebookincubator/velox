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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/common/StripeRange.h"

namespace facebook::nimble::test {

TEST(StripeRangeTest, defaultIsEmpty) {
  const StripeRange range;
  EXPECT_EQ(range.startStripe, 0);
  EXPECT_EQ(range.endStripe, 0);
  EXPECT_TRUE(range.empty());
  EXPECT_FALSE(range.contains(0));
}

TEST(StripeRangeTest, empty) {
  EXPECT_FALSE(StripeRange(2, 5).empty());
  EXPECT_FALSE(StripeRange(2, 3).empty());
  EXPECT_TRUE(StripeRange(3, 3).empty());
  // A range that ends before it starts holds no stripe either.
  EXPECT_TRUE(StripeRange(5, 2).empty());
}

TEST(StripeRangeTest, numStripes) {
  EXPECT_EQ(StripeRange(2, 5).numStripes(), 3);
  EXPECT_EQ(StripeRange(2, 3).numStripes(), 1);
  EXPECT_EQ(StripeRange(3, 3).numStripes(), 0);
  EXPECT_EQ(StripeRange().numStripes(), 0);
  // A range that ends before it starts holds no stripe either.
  EXPECT_EQ(StripeRange(5, 2).numStripes(), 0);
}

TEST(StripeRangeTest, contains) {
  // Holds its start stripe, and ends before its end stripe.
  const StripeRange range(2, 5);
  EXPECT_FALSE(range.contains(1));
  EXPECT_TRUE(range.contains(2));
  EXPECT_TRUE(range.contains(4));
  EXPECT_FALSE(range.contains(5));
  EXPECT_FALSE(StripeRange(3, 3).contains(3));
}

TEST(StripeRangeTest, equality) {
  EXPECT_TRUE(StripeRange(2, 5) == StripeRange(2, 5));
  EXPECT_FALSE(StripeRange(2, 5) == StripeRange(1, 5));
  EXPECT_FALSE(StripeRange(2, 5) == StripeRange(2, 4));
}

TEST(StripeRangeTest, toString) {
  EXPECT_EQ(StripeRange(2, 5).toString(), "[2, 5)");
  EXPECT_EQ(StripeRange().toString(), "[0, 0)");
}

} // namespace facebook::nimble::test
