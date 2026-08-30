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

#include "velox/connectors/hive/HiveIndexSourceUtils.h"
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace facebook::velox::connector::hive {
namespace {

class ExtractRangeBoundsTest : public ::testing::Test {};

// --- BigintRange tests ---

TEST_F(ExtractRangeBoundsTest, bigintRangeInclusive) {
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/10, /*upper=*/20, /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->first.value<int64_t>(), 10);
  EXPECT_EQ(result->second.value<int64_t>(), 20);
}

TEST_F(ExtractRangeBoundsTest, bigintRangeSingleValue) {
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/42, /*upper=*/42, /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->first.value<int64_t>(), 42);
  EXPECT_EQ(result->second.value<int64_t>(), 42);
}

// --- DoubleRange tests ---

TEST_F(ExtractRangeBoundsTest, doubleRangeBothInclusive) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->first.value<double>(), 1.0);
  EXPECT_EQ(result->second.value<double>(), 5.0);
}

TEST_F(ExtractRangeBoundsTest, doubleRangeBothExclusive) {
  // score > 1.0 AND score < 5.0
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(
      result->first.value<double>(),
      std::nextafter(1.0, std::numeric_limits<double>::infinity()));
  EXPECT_EQ(
      result->second.value<double>(),
      std::nextafter(5.0, -std::numeric_limits<double>::infinity()));
}

TEST_F(ExtractRangeBoundsTest, doubleRangeLowerExclusive) {
  // score > 1.0 AND score <= 5.0
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(
      result->first.value<double>(),
      std::nextafter(1.0, std::numeric_limits<double>::infinity()));
  EXPECT_EQ(result->second.value<double>(), 5.0);
}

TEST_F(ExtractRangeBoundsTest, doubleRangeUpperExclusive) {
  // score >= 1.0 AND score < 5.0
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->first.value<double>(), 1.0);
  EXPECT_EQ(
      result->second.value<double>(),
      std::nextafter(5.0, -std::numeric_limits<double>::infinity()));
}

TEST_F(ExtractRangeBoundsTest, doubleRangeUnboundedReturnsNullopt) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/0.0,
      /*upperUnbounded=*/true,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  EXPECT_FALSE(result.has_value());
}

// --- FloatRange tests ---
// FloatingPointRange::lower()/upper() return double regardless of T,
// so the variant is always DOUBLE kind.

TEST_F(ExtractRangeBoundsTest, floatRangeBothInclusive) {
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result->first.value<double>(), 1.0);
  EXPECT_DOUBLE_EQ(result->second.value<double>(), 5.0);
}

TEST_F(ExtractRangeBoundsTest, floatRangeBothExclusive) {
  // score > 1.0f AND score < 5.0f
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  double lower = result->first.value<double>();
  double upper = result->second.value<double>();
  EXPECT_GT(lower, 1.0);
  EXPECT_LT(upper, 5.0);
}

TEST_F(ExtractRangeBoundsTest, floatRangeLowerExclusive) {
  // score > 1.0f AND score <= 5.0f
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_GT(result->first.value<double>(), 1.0);
  EXPECT_DOUBLE_EQ(result->second.value<double>(), 5.0);
}

TEST_F(ExtractRangeBoundsTest, floatRangeUpperExclusive) {
  // score >= 1.0f AND score < 5.0f
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result->first.value<double>(), 1.0);
  EXPECT_LT(result->second.value<double>(), 5.0);
}

TEST_F(ExtractRangeBoundsTest, floatRangeUnboundedReturnsNullopt) {
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/0.0f,
      /*lowerUnbounded=*/true,
      /*lowerExclusive=*/true,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractRangeBounds(filter.get());
  EXPECT_FALSE(result.has_value());
}

// --- nullAllowed tests ---
// Filters with nullAllowed=true must not be converted, because the
// caller erases the original filter after conversion and the simplified
// condition cannot represent IS NULL semantics.

TEST_F(ExtractRangeBoundsTest, bigintRangeNullAllowedReturnsNullopt) {
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/10, /*upper=*/20, /*nullAllowed=*/true);
  auto result = extractRangeBounds(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractRangeBoundsTest, doubleRangeNullAllowedReturnsNullopt) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/true);
  auto result = extractRangeBounds(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractRangeBoundsTest, floatRangeNullAllowedReturnsNullopt) {
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/1.0f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/true);
  auto result = extractRangeBounds(filter.get());
  EXPECT_FALSE(result.has_value());
}

// --- extractPointLookupValue tests ---

class ExtractPointLookupValueTest : public ::testing::Test {};

TEST_F(ExtractPointLookupValueTest, bigintSingleValue) {
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/42, /*upper=*/42, /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value<int64_t>(), 42);
}

TEST_F(ExtractPointLookupValueTest, bigintRangeNotPoint) {
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/10, /*upper=*/20, /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, doubleExactPoint) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/3.14,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/3.14,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->value<double>(), 3.14);
}

TEST_F(ExtractPointLookupValueTest, doubleExclusiveNotPoint) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/3.14,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/true,
      /*upper=*/3.14,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, doubleRangeNotPoint) {
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/1.0,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/5.0,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, floatExactPoint) {
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/2.5f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/2.5f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(result->value<double>(), 2.5);
}

TEST_F(ExtractPointLookupValueTest, floatExclusiveNotPoint) {
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/2.5f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/2.5f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/true,
      /*nullAllowed=*/false);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, bigintNullAllowedReturnsNullopt) {
  // key = 42 OR key IS NULL
  auto filter = std::make_unique<common::BigintRange>(
      /*lower=*/42, /*upper=*/42, /*nullAllowed=*/true);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, doubleNullAllowedReturnsNullopt) {
  // score = 3.14 OR score IS NULL
  auto filter = std::make_unique<common::DoubleRange>(
      /*lower=*/3.14,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/3.14,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/true);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

TEST_F(ExtractPointLookupValueTest, floatNullAllowedReturnsNullopt) {
  // score = 2.5f OR score IS NULL
  auto filter = std::make_unique<common::FloatRange>(
      /*lower=*/2.5f,
      /*lowerUnbounded=*/false,
      /*lowerExclusive=*/false,
      /*upper=*/2.5f,
      /*upperUnbounded=*/false,
      /*upperExclusive=*/false,
      /*nullAllowed=*/true);
  auto result = extractPointLookupValue(filter.get());
  EXPECT_FALSE(result.has_value());
}

// --- createEqualConditionWithConstant tests ---

class CreateEqualConditionTest : public ::testing::Test {};

TEST_F(CreateEqualConditionTest, bigintConstant) {
  auto condition = createEqualConditionWithConstant(
      "id", BIGINT(), variant(static_cast<int64_t>(42)));
  ASSERT_NE(condition, nullptr);
  auto equal =
      std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(condition);
  ASSERT_NE(equal, nullptr);
  EXPECT_EQ(equal->key->name(), "id");
  EXPECT_EQ(*equal->key->type(), *BIGINT());
  auto constant =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(equal->value);
  ASSERT_NE(constant, nullptr);
  EXPECT_EQ(constant->value().value<int64_t>(), 42);
}

TEST_F(CreateEqualConditionTest, doubleConstant) {
  auto condition =
      createEqualConditionWithConstant("score", DOUBLE(), variant(3.14));
  auto equal =
      std::dynamic_pointer_cast<core::EqualIndexLookupCondition>(condition);
  ASSERT_NE(equal, nullptr);
  EXPECT_EQ(equal->key->name(), "score");
  auto constant =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(equal->value);
  ASSERT_NE(constant, nullptr);
  EXPECT_EQ(constant->value().value<double>(), 3.14);
}

// --- createBetweenConditionWithConstants tests ---

class CreateBetweenConditionTest : public ::testing::Test {};

TEST_F(CreateBetweenConditionTest, bigintRange) {
  auto condition = createBetweenConditionWithConstants(
      "id",
      BIGINT(),
      variant(static_cast<int64_t>(10)),
      variant(static_cast<int64_t>(20)));
  ASSERT_NE(condition, nullptr);
  auto between =
      std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(condition);
  ASSERT_NE(between, nullptr);
  EXPECT_EQ(between->key->name(), "id");
  EXPECT_EQ(*between->key->type(), *BIGINT());
  auto lower =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(between->lower);
  auto upper =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(between->upper);
  ASSERT_NE(lower, nullptr);
  ASSERT_NE(upper, nullptr);
  EXPECT_EQ(lower->value().value<int64_t>(), 10);
  EXPECT_EQ(upper->value().value<int64_t>(), 20);
}

TEST_F(CreateBetweenConditionTest, doubleRange) {
  auto condition = createBetweenConditionWithConstants(
      "score", DOUBLE(), variant(1.0), variant(5.0));
  auto between =
      std::dynamic_pointer_cast<core::BetweenIndexLookupCondition>(condition);
  ASSERT_NE(between, nullptr);
  EXPECT_EQ(between->key->name(), "score");
  auto lower =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(between->lower);
  auto upper =
      std::dynamic_pointer_cast<const core::ConstantTypedExpr>(between->upper);
  ASSERT_NE(lower, nullptr);
  ASSERT_NE(upper, nullptr);
  EXPECT_EQ(lower->value().value<double>(), 1.0);
  EXPECT_EQ(upper->value().value<double>(), 5.0);
}

} // namespace
} // namespace facebook::velox::connector::hive
