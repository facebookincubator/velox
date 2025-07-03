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
#include "velox/core/FilterToExpression.h"
#include "velox/core/Expressions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

namespace facebook::velox::core::test {

class FilterToExpressionTest : public testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Helper method to create a row type for testing
  RowTypePtr createTestRowType() {
    return ROW(
        {"a", "b", "c", "d", "e", "f"},
        {BIGINT(), DOUBLE(), VARCHAR(), BOOLEAN(), REAL(), TIMESTAMP()});
  }

  // Helper method to verify expression type and structure
  void verifyExpr(
      const TypedExprPtr& expr,
      const std::string& expectedType,
      const std::string& expectedName) {
    ASSERT_TRUE(expr != nullptr);
    ASSERT_EQ(expr->type()->toString(), expectedType);

    auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
    ASSERT_TRUE(callExpr != nullptr);
    ASSERT_EQ(callExpr->name(), expectedName);
  }
};

TEST_F(FilterToExpressionTest, AlwaysTrue) {
  auto filter = std::make_unique<common::AlwaysTrue>();
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "true");
  ASSERT_TRUE(
      std::dynamic_pointer_cast<const CallTypedExpr>(expr)->inputs().empty());
}

TEST_F(FilterToExpressionTest, AlwaysFalse) {
  auto filter = std::make_unique<common::AlwaysFalse>();
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "false");
  ASSERT_TRUE(
      std::dynamic_pointer_cast<const CallTypedExpr>(expr)->inputs().empty());
}

TEST_F(FilterToExpressionTest, IsNull) {
  auto filter = std::make_unique<common::IsNull>();
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "is_null");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);
}

TEST_F(FilterToExpressionTest, IsNotNull) {
  auto filter = std::make_unique<common::IsNotNull>();
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "is_not_null");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);
}

TEST_F(FilterToExpressionTest, BoolValue) {
  auto filter = std::make_unique<common::BoolValue>(true, false);
  common::Subfield subfield("d");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "equals");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto constantExpr =
      std::dynamic_pointer_cast<const ConstantTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_EQ(constantExpr->value().value<TypeKind::BOOLEAN>(), true);
}

TEST_F(FilterToExpressionTest, BigintRangeSingleValue) {
  auto filter = std::make_unique<common::BigintRange>(42, 42, false);
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "equals");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto constantExpr =
      std::dynamic_pointer_cast<const ConstantTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_EQ(constantExpr->value().value<TypeKind::BIGINT>(), 42);
}

TEST_F(FilterToExpressionTest, BigintRangeWithRange) {
  auto filter = std::make_unique<common::BigintRange>(10, 20, false);
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "greater_than_or_equal");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "less_than_or_equal");
}

TEST_F(FilterToExpressionTest, NegatedBigintRangeSingleValue) {
  auto filter = std::make_unique<common::NegatedBigintRange>(42, 42, false);
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "not_equals");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto constantExpr =
      std::dynamic_pointer_cast<const ConstantTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_EQ(constantExpr->value().value<TypeKind::BIGINT>(), 42);
}

TEST_F(FilterToExpressionTest, DoubleRange) {
  auto filter = std::make_unique<common::DoubleRange>(
      1.5, false, false, 3.5, false, false, false);
  common::Subfield subfield("b");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "greater_than_or_equal");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "less_than_or_equal");
}

TEST_F(FilterToExpressionTest, FloatRange) {
  auto filter = std::make_unique<common::FloatRange>(
      1.5f, false, true, 3.5f, false, true, false);
  common::Subfield subfield("e");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterThan =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterThan != nullptr);
  ASSERT_EQ(greaterThan->name(), "greater_than");

  auto lessThan =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessThan != nullptr);
  ASSERT_EQ(lessThan->name(), "less_than");
}

// Test BytesRange filter
TEST_F(FilterToExpressionTest, BytesRange) {
  auto filter = std::make_unique<common::BytesRange>(
      "apple", false, false, "orange", false, false, false);
  common::Subfield subfield("c");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "greater_than_or_equal");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "less_than_or_equal");
}

TEST_F(FilterToExpressionTest, BigintValuesUsingHashTable) {
  std::vector<int64_t> values = {10, 20, 30};
  auto filter = common::createBigintValues(values, false);
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "contains");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto arrayExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(arrayExpr != nullptr);
  ASSERT_EQ(arrayExpr->name(), "array_constructor");
  ASSERT_EQ(arrayExpr->inputs().size(), 3);
}

TEST_F(FilterToExpressionTest, BytesValues) {
  std::vector<std::string> values = {"apple", "banana", "orange"};
  auto filter = std::make_unique<common::BytesValues>(values, false);
  common::Subfield subfield("c");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "contains");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto arrayExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(arrayExpr != nullptr);
  ASSERT_EQ(arrayExpr->name(), "array_constructor");
  ASSERT_EQ(arrayExpr->inputs().size(), 3);
}

TEST_F(FilterToExpressionTest, NegatedBytesValues) {
  std::vector<std::string> values = {"apple", "banana", "orange"};
  auto filter = std::make_unique<common::NegatedBytesValues>(values, false);
  common::Subfield subfield("c");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "not");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);

  auto containsExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(containsExpr != nullptr);
  ASSERT_EQ(containsExpr->name(), "contains");
}

TEST_F(FilterToExpressionTest, TimestampRange) {
  auto timestamp1 = Timestamp::fromMillis(1609459200000); // 2021-01-01
  auto timestamp2 = Timestamp::fromMillis(1640995200000); // 2022-01-01
  auto filter =
      std::make_unique<common::TimestampRange>(timestamp1, timestamp2, false);
  common::Subfield subfield("f");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "greater_than_or_equal");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "less_than_or_equal");
}

TEST_F(FilterToExpressionTest, BigintMultiRange) {
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(std::make_unique<common::BigintRange>(10, 20, false));
  ranges.push_back(std::make_unique<common::BigintRange>(30, 40, false));
  auto filter =
      std::make_unique<common::BigintMultiRange>(std::move(ranges), false);
  common::Subfield subfield("a");
  auto rowType = createTestRowType();

  auto expr = filterToExpr(subfield, filter.get(), rowType, pool());

  verifyExpr(expr, "BOOLEAN", "or");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);
}

} // namespace facebook::velox::core::test
