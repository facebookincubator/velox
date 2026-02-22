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
#include "velox/exec/tests/utils/FilterToExpression.h"
#include <gtest/gtest.h>
#include "velox/core/Expressions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::core::test {

class FilterToExpressionTest : public testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

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

  TypedExprPtr toExpr(const common::Filter* filter, const TypePtr& type) {
    common::Subfield subfield("a");
    return filterToExpr(subfield, filter, ROW({"a"}, {type}), pool());
  }
};

TEST_F(FilterToExpressionTest, alwaysTrue) {
  auto filter = std::make_unique<common::AlwaysTrue>();
  auto expr = toExpr(filter.get(), BIGINT());

  ASSERT_TRUE(expr != nullptr);
  ASSERT_EQ(expr->type()->toString(), "BOOLEAN");

  auto constantExpr = std::dynamic_pointer_cast<const ConstantTypedExpr>(expr);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_TRUE(constantExpr->value().value<TypeKind::BOOLEAN>());
}

TEST_F(FilterToExpressionTest, alwaysFalse) {
  auto filter = std::make_unique<common::AlwaysFalse>();
  auto expr = toExpr(filter.get(), BIGINT());

  ASSERT_TRUE(expr != nullptr);
  ASSERT_EQ(expr->type()->toString(), "BOOLEAN");

  auto constantExpr = std::dynamic_pointer_cast<const ConstantTypedExpr>(expr);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_FALSE(constantExpr->value().value<TypeKind::BOOLEAN>());
}

TEST_F(FilterToExpressionTest, isNull) {
  auto filter = std::make_unique<common::IsNull>();
  auto expr = toExpr(filter.get(), BIGINT());

  verifyExpr(expr, "BOOLEAN", "is_null");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);
}

TEST_F(FilterToExpressionTest, isNotNull) {
  auto filter = std::make_unique<common::IsNotNull>();
  auto expr = toExpr(filter.get(), BIGINT());

  verifyExpr(expr, "BOOLEAN", "not");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);

  // Verify the inner expression is an IS_NULL operation
  auto isNullExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(isNullExpr != nullptr);
  ASSERT_EQ(isNullExpr->name(), "is_null");
}

TEST_F(FilterToExpressionTest, boolValue) {
  auto filter = std::make_unique<common::BoolValue>(true, false);
  auto expr = toExpr(filter.get(), BOOLEAN());

  verifyExpr(expr, "BOOLEAN", "eq");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  // First input should be the field access expression
  auto fieldExpr = callExpr->inputs()[0];
  ASSERT_TRUE(fieldExpr != nullptr);

  // Second input should be the boolean constant
  auto constantExpr =
      std::dynamic_pointer_cast<const ConstantTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_EQ(constantExpr->value().value<TypeKind::BOOLEAN>(), true);
}

TEST_F(FilterToExpressionTest, bigintRangeSingleValue) {
  auto filter = std::make_unique<common::BigintRange>(42, 42, false);
  auto expr = toExpr(filter.get(), BIGINT());

  verifyExpr(expr, "BOOLEAN", "eq");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto constantExpr =
      std::dynamic_pointer_cast<const ConstantTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(constantExpr != nullptr);
  ASSERT_EQ(constantExpr->value().value<TypeKind::BIGINT>(), 42);
}

TEST_F(FilterToExpressionTest, bigintRangeWithRange) {
  auto filter = std::make_unique<common::BigintRange>(10, 20, false);
  auto expr = toExpr(filter.get(), BIGINT());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "gte");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "lte");
}

TEST_F(FilterToExpressionTest, negatedBigintRangeSingleValue) {
  auto filter = std::make_unique<common::NegatedBigintRange>(42, 42, false);
  auto expr = toExpr(filter.get(), BIGINT());

  // The implementation now uses getNonNegated() which creates a NOT expression
  // even for single values, so we expect "not" instead of "neq"
  verifyExpr(expr, "BOOLEAN", "not");
  auto notExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(notExpr->inputs().size(), 1);

  // The inner expression might be an OR expression due to handleNullAllowed
  auto innerExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(notExpr->inputs()[0]);
  ASSERT_TRUE(innerExpr != nullptr);

  if (innerExpr->name() == "or") {
    // If it's an OR expression, the first input should be the EQ operation
    ASSERT_EQ(innerExpr->inputs().size(), 2);
    auto eqExpr =
        std::dynamic_pointer_cast<const CallTypedExpr>(innerExpr->inputs()[0]);
    ASSERT_TRUE(eqExpr != nullptr);
    ASSERT_EQ(eqExpr->name(), "eq");
    ASSERT_EQ(eqExpr->inputs().size(), 2);

    // Verify the constant value is 42
    auto constantExpr =
        std::dynamic_pointer_cast<const ConstantTypedExpr>(eqExpr->inputs()[1]);
    ASSERT_TRUE(constantExpr != nullptr);
    ASSERT_EQ(constantExpr->value().value<TypeKind::BIGINT>(), 42);
  } else if (innerExpr->name() == "eq") {
    // If it's directly an EQ expression
    ASSERT_EQ(innerExpr->inputs().size(), 2);

    // Verify the constant value is 42
    auto constantExpr = std::dynamic_pointer_cast<const ConstantTypedExpr>(
        innerExpr->inputs()[1]);
    ASSERT_TRUE(constantExpr != nullptr);
    ASSERT_EQ(constantExpr->value().value<TypeKind::BIGINT>(), 42);
  } else {
    FAIL() << "Expected either 'or' or 'eq' expression, got: "
           << innerExpr->name();
  }
}

TEST_F(FilterToExpressionTest, doubleRange) {
  auto filter = std::make_unique<common::DoubleRange>(
      1.5, false, false, 3.5, false, false, false);
  auto expr = toExpr(filter.get(), DOUBLE());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "gte");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "lte");
}

TEST_F(FilterToExpressionTest, floatRange) {
  auto filter = std::make_unique<common::FloatRange>(
      1.5f, false, true, 3.5f, false, true, false);
  auto expr = toExpr(filter.get(), REAL());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterThan =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterThan != nullptr);
  ASSERT_EQ(greaterThan->name(), "gt");

  auto lessThan =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessThan != nullptr);
  ASSERT_EQ(lessThan->name(), "lt");
}

TEST_F(FilterToExpressionTest, bytesRange) {
  auto filter = std::make_unique<common::BytesRange>(
      "apple", false, false, "orange", false, false, false);
  auto expr = toExpr(filter.get(), VARCHAR());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "gte");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "lte");
}

TEST_F(FilterToExpressionTest, bigintValuesUsingHashTable) {
  std::vector<int64_t> values = {10, 20, 30};
  auto filter = common::createBigintValues(values, false);
  auto expr = toExpr(filter.get(), BIGINT());

  // The implementation creates an optimized expression: (range check) AND (in
  // check)
  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  // First input should be the range check (field >= min AND field <= max)
  auto rangeCheckExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(rangeCheckExpr != nullptr);
  ASSERT_EQ(rangeCheckExpr->name(), "and");

  // Second input should be the IN expression
  auto inExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(inExpr != nullptr);
  ASSERT_EQ(inExpr->name(), "in");
  ASSERT_EQ(inExpr->inputs().size(), 2);

  auto arrayExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(inExpr->inputs()[1]);
  ASSERT_TRUE(arrayExpr != nullptr);
  ASSERT_EQ(arrayExpr->name(), "array_constructor");
  ASSERT_EQ(arrayExpr->inputs().size(), 3);
}

TEST_F(FilterToExpressionTest, bytesValues) {
  std::vector<std::string> values = {"apple", "banana", "orange"};
  auto filter = std::make_unique<common::BytesValues>(values, false);
  auto expr = toExpr(filter.get(), VARCHAR());

  verifyExpr(expr, "BOOLEAN", "in");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto arrayExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(arrayExpr != nullptr);
  ASSERT_EQ(arrayExpr->name(), "array_constructor");
  ASSERT_EQ(arrayExpr->inputs().size(), 3);
}

TEST_F(FilterToExpressionTest, negatedBytesValues) {
  std::vector<std::string> values = {"apple", "banana", "orange"};
  auto filter = std::make_unique<common::NegatedBytesValues>(values, false);
  auto expr = toExpr(filter.get(), VARCHAR());

  verifyExpr(expr, "BOOLEAN", "not");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 1);

  auto containsExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(containsExpr != nullptr);

  ASSERT_TRUE(containsExpr->name() == "in" || containsExpr->name() == "or");
}

TEST_F(FilterToExpressionTest, negatedBigintValuesUsingHashTable) {
  std::vector<int64_t> values = {10, 20, 30};
  auto filter = std::make_unique<common::NegatedBigintValuesUsingHashTable>(
      10, 30, values, false);
  auto expr = toExpr(filter.get(), BIGINT());

  // The implementation creates a NOT expression for the optimized IN check
  verifyExpr(expr, "BOOLEAN", "not");
  auto notExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(notExpr->inputs().size(), 1);

  // The input should be an OR expression
  auto orExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(notExpr->inputs()[0]);
  ASSERT_TRUE(orExpr != nullptr);
  ASSERT_EQ(orExpr->name(), "or");
  ASSERT_EQ(orExpr->inputs().size(), 2);

  // First input of OR should be range check
  auto rangeCheckExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(orExpr->inputs()[0]);
  ASSERT_TRUE(rangeCheckExpr != nullptr);
  ASSERT_EQ(rangeCheckExpr->name(), "and");

  // Second input of OR should be IS_NULL expression
  auto isNullExpr =
      std::dynamic_pointer_cast<const CallTypedExpr>(orExpr->inputs()[1]);
  ASSERT_TRUE(isNullExpr != nullptr);
  ASSERT_EQ(isNullExpr->name(), "is_null");
}

TEST_F(FilterToExpressionTest, timestampRange) {
  auto timestamp1 = Timestamp::fromMillis(1609459200000); // 2021-01-01
  auto timestamp2 = Timestamp::fromMillis(1640995200000); // 2022-01-01
  auto filter =
      std::make_unique<common::TimestampRange>(timestamp1, timestamp2, false);
  auto expr = toExpr(filter.get(), TIMESTAMP());

  verifyExpr(expr, "BOOLEAN", "and");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);

  auto greaterOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(greaterOrEqual != nullptr);
  ASSERT_EQ(greaterOrEqual->name(), "gte");

  auto lessOrEqual =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(lessOrEqual != nullptr);
  ASSERT_EQ(lessOrEqual->name(), "lte");
}

TEST_F(FilterToExpressionTest, bigintMultiRange) {
  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  ranges.push_back(std::make_unique<common::BigintRange>(10, 20, false));
  ranges.push_back(std::make_unique<common::BigintRange>(30, 40, false));
  auto filter =
      std::make_unique<common::BigintMultiRange>(std::move(ranges), false);
  auto expr = toExpr(filter.get(), BIGINT());

  verifyExpr(expr, "BOOLEAN", "or");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 2);
}

TEST_F(FilterToExpressionTest, multiRange) {
  // Create a MultiRange filter with compatible filters for BIGINT field
  std::vector<std::unique_ptr<common::Filter>> filters;

  // Add a BigintRange filter
  filters.push_back(std::make_unique<common::BigintRange>(10, 20, false));

  // Add an IsNull filter
  filters.push_back(std::make_unique<common::IsNull>());

  // Add another BigintRange filter instead of BytesRange to avoid type mismatch
  filters.push_back(std::make_unique<common::BigintRange>(30, 40, false));

  auto filter = std::make_unique<common::MultiRange>(std::move(filters), false);
  auto expr = toExpr(filter.get(), BIGINT());

  // Verify the top-level expression is an OR
  verifyExpr(expr, "BOOLEAN", "or");
  auto callExpr = std::dynamic_pointer_cast<const CallTypedExpr>(expr);
  ASSERT_EQ(callExpr->inputs().size(), 3);

  // Verify the first input is a BigintRange expression (AND of
  // greater_than_or_equal and less_than_or_equal)
  auto firstInput =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[0]);
  ASSERT_TRUE(firstInput != nullptr);
  ASSERT_EQ(firstInput->name(), "and");
  ASSERT_EQ(firstInput->inputs().size(), 2);

  // Verify the second input is an IsNull expression
  auto secondInput =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[1]);
  ASSERT_TRUE(secondInput != nullptr);
  ASSERT_EQ(secondInput->name(), "is_null");

  // Verify the third input is another BigintRange expression (AND of
  // greater_than_or_equal and less_than_or_equal)
  auto thirdInput =
      std::dynamic_pointer_cast<const CallTypedExpr>(callExpr->inputs()[2]);
  ASSERT_TRUE(thirdInput != nullptr);
  ASSERT_EQ(thirdInput->name(), "and");
  ASSERT_EQ(thirdInput->inputs().size(), 2);
}

} // namespace facebook::velox::core::test
