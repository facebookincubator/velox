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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::test;

namespace facebook::velox::exec::test {
namespace {

class NullIfExprTest : public testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
  }

  // Parses a SQL expression string into a TypedExprPtr.
  core::TypedExprPtr parseExpr(
      const std::string& sql,
      const RowTypePtr& rowType) {
    parse::ParseOptions options;
    auto iExpr = parse::DuckSqlExpressionsParser(options).parseExpr(sql);
    return core::Expressions::inferTypes(iExpr, rowType, pool());
  }

  // Evaluates NULLIF(a, b). Common type defaults to a's type.
  VectorPtr evaluateNullIf(
      const core::TypedExprPtr& aExpr,
      const core::TypedExprPtr& bExpr,
      const RowVectorPtr& input,
      const TypePtr& commonType = nullptr) {
    auto nullIfExpr = std::make_shared<core::NullIfTypedExpr>(
        aExpr, bExpr, commonType ? commonType : aExpr->type());

    ExprSet exprSet({nullIfExpr}, execCtx_.get());
    EvalCtx context(execCtx_.get(), &exprSet, input.get());

    std::vector<VectorPtr> result(1);
    SelectivityVector rows(input->size());
    exprSet.eval(rows, context, result);
    return result[0];
  }

  // Evaluates NULLIF(a, b) using column references.
  VectorPtr evaluateNullIf(
      const VectorPtr& a,
      const VectorPtr& b,
      const TypePtr& commonType = nullptr) {
    auto aField = std::make_shared<core::FieldAccessTypedExpr>(a->type(), "a");
    auto bField = std::make_shared<core::FieldAccessTypedExpr>(b->type(), "b");
    return evaluateNullIf(
        aField, bField, makeRowVector({"a", "b"}, {a, b}), commonType);
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

// Same-type inputs, some equal, some not.
TEST_F(NullIfExprTest, sameType) {
  auto a = makeFlatVector<int32_t>({1, 2, 3, 4, 5});
  auto b = makeFlatVector<int32_t>({1, 0, 3, 0, 5});

  auto result = evaluateNullIf(a, b);

  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, 2, std::nullopt, 4, std::nullopt});
  assertEqualVectors(expected, result);
}

// All rows equal — should return constant null.
TEST_F(NullIfExprTest, allEqual) {
  auto a = makeFlatVector<int32_t>({1, 2, 3});
  auto b = makeFlatVector<int32_t>({1, 2, 3});

  auto result = evaluateNullIf(a, b);

  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  assertEqualVectors(expected, result);
}

// No rows equal — should return a as is.
TEST_F(NullIfExprTest, noneEqual) {
  auto a = makeFlatVector<int32_t>({1, 2, 3});
  auto b = makeFlatVector<int32_t>({4, 5, 6});

  auto result = evaluateNullIf(a, b);
  assertEqualVectors(a, result);
}

// Null in first argument.
TEST_F(NullIfExprTest, nullInFirst) {
  auto a = makeNullableFlatVector<int32_t>({std::nullopt, 2, std::nullopt});
  auto b = makeFlatVector<int32_t>({1, 2, 3});

  auto result = evaluateNullIf(a, b);

  // NULLIF(NULL, x) returns NULL (the first argument).
  auto expected = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  assertEqualVectors(expected, result);
}

// Null in second argument.
TEST_F(NullIfExprTest, nullInSecond) {
  auto a = makeFlatVector<int32_t>({1, 2, 3});
  auto b = makeNullableFlatVector<int32_t>(
      {std::nullopt, std::nullopt, std::nullopt});

  auto result = evaluateNullIf(a, b);

  // NULLIF(x, NULL) returns x (comparison is indeterminate).
  assertEqualVectors(a, result);
}

// Both arguments null.
TEST_F(NullIfExprTest, bothNull) {
  auto a = makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt});
  auto b = makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt});

  auto result = evaluateNullIf(a, b);

  // NULLIF(NULL, NULL) returns NULL (first argument).
  auto expected = makeNullableFlatVector<int32_t>({std::nullopt, std::nullopt});
  assertEqualVectors(expected, result);
}

// Different types with common supertype.
TEST_F(NullIfExprTest, differentTypes) {
  auto a = makeFlatVector<int16_t>({1, 2, 3, 4});
  auto b = makeFlatVector<int64_t>({1, 0, 3, 0});

  auto result = evaluateNullIf(a, b, BIGINT());

  // Return type is SMALLINT (first argument's type).
  EXPECT_EQ(result->type()->kind(), TypeKind::SMALLINT);
  auto expected =
      makeNullableFlatVector<int16_t>({std::nullopt, 2, std::nullopt, 4});
  assertEqualVectors(expected, result);
}

// Arrays with nested nulls. With kNullAsIndeterminate, ARRAY[1, NULL] is not
// equal to ARRAY[1, NULL] (comparison is indeterminate), so NULLIF returns the
// first array unchanged.
TEST_F(NullIfExprTest, arrayWithNestedNulls) {
  auto a = makeArrayVectorFromJson<int32_t>({
      "[1, null, 3]",
      "[4, 5]",
      "[null]",
      "[1, 2, null]",
  });
  auto b = makeArrayVectorFromJson<int32_t>({
      "[1, null, 3]",
      "[4, 5]",
      "[null]",
      "[1, 1, null]",
  });

  auto result = evaluateNullIf(a, b);

  // Row 0: [1, null, 3] vs [1, null, 3] — indeterminate, returns a.
  // Row 1: [4, 5] vs [4, 5] — equal, returns NULL.
  // Row 2: [null] vs [null] — indeterminate, returns a ([null]).
  // Row 3: [1, 2, null] vs [1, 1, null] — not equal, returns a.
  auto expected = makeArrayVectorFromJson<int32_t>({
      "[1, null, 3]",
      "null",
      "[null]",
      "[1, 2, null]",
  });
  assertEqualVectors(expected, result);
}

// Asserts the result has at least one null and returns the set of unique
// non-null values.
template <typename T>
std::unordered_set<T> uniqueNonNullValues(const VectorPtr& result) {
  auto* values = result->as<SimpleVector<T>>();
  VELOX_CHECK_NOT_NULL(values);

  std::unordered_set<T> unique;
  bool hasNull = false;
  for (vector_size_t row = 0; row < result->size(); ++row) {
    if (result->isNullAt(row)) {
      hasNull = true;
    } else {
      unique.insert(values->valueAt(row));
    }
  }

  EXPECT_TRUE(hasNull) << "Expected some null rows";
  EXPECT_FALSE(unique.empty()) << "Expected some non-null rows";
  return unique;
}

// Non-deterministic expression evaluated once. NULLIF(rand() < 0.5, true)
// should produce only NULL and false, never true. If the expression is
// evaluated twice, the condition and return value can diverge, producing true.
TEST_F(NullIfExprTest, nonDeterministic) {
  auto input = makeRowVector(ROW({}), 1'000);

  auto aExpr = parseExpr("rand() < 0.5", input->rowType());
  auto bExpr = parseExpr("true", input->rowType());

  auto result = evaluateNullIf(aExpr, bExpr, input);

  // Only 'false' should appear as a non-null value.
  EXPECT_THAT(uniqueNonNullValues<bool>(result), testing::ElementsAre(false));
}

// Same as nonDeterministic but with different input types to exercise the cast
// path. NULLIF(cast(rand() * 10 as smallint), 5::bigint) returns NULL when the
// cast result is 5, else returns the smallint value. Value 5 should never
// appear in the output. The return type must be SMALLINT.
TEST_F(NullIfExprTest, nonDeterministicWithCast) {
  auto input = makeRowVector(ROW({}), 1'000);

  auto aExpr = parseExpr("cast(rand() * 5.0 as smallint)", input->rowType());
  auto bExpr = parseExpr("3::bigint", input->rowType());

  auto result = evaluateNullIf(aExpr, bExpr, input, BIGINT());

  EXPECT_EQ(result->type()->kind(), TypeKind::SMALLINT);

  // All values 0-5 except 3 (which is nulled out).
  EXPECT_THAT(
      uniqueNonNullValues<int16_t>(result),
      testing::UnorderedElementsAre(0, 1, 2, 4, 5));
}

} // namespace
} // namespace facebook::velox::exec::test
