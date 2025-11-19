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

#include "velox/expression/ExprToSubfieldFilter.h"
#include <gtest/gtest.h>
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

using facebook::velox::common::Subfield;

#define VELOX_ASSERT_FILTER(expected, actual)   \
  ASSERT_TRUE(expected->testingEquals(*actual)) \
      << expected->toString() << " vs " << actual->toString();

namespace facebook::velox::exec {
namespace {

void validateSubfield(
    const Subfield& subfield,
    const std::vector<std::string>& expectedPath) {
  ASSERT_EQ(subfield.path().size(), expectedPath.size());
  for (int i = 0; i < expectedPath.size(); ++i) {
    ASSERT_TRUE(subfield.path()[i]);
    ASSERT_EQ(*subfield.path()[i], Subfield::NestedField(expectedPath[i]));
  }
}

class ExprToSubfieldFilterTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  core::TypedExprPtr parseExpr(
      const std::string& expr,
      const RowTypePtr& type) {
    return core::Expressions::inferTypes(
        parse::parseExpr(expr, {}), type, pool_.get());
  }

  core::CallTypedExprPtr parseCallExpr(
      const std::string& expr,
      const RowTypePtr& type) {
    auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(
        parseExpr(expr, type));
    VELOX_CHECK_NOT_NULL(call);
    return call;
  }

  core::ExpressionEvaluator* evaluator() {
    return &evaluator_;
  }

  std::pair<common::Subfield, std::unique_ptr<common::Filter>>
  leafCallToSubfieldFilter(const core::CallTypedExprPtr& call) {
    if (auto result =
            ExprToSubfieldFilterParser::getInstance()->leafCallToSubfieldFilter(
                *call, evaluator())) {
      return std::move(result.value());
    }

    return std::make_pair(common::Subfield(), nullptr);
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  SimpleExpressionEvaluator evaluator_{queryCtx_.get(), pool_.get()};
};

TEST_F(ExprToSubfieldFilterTest, eq) {
  auto call = parseCallExpr("a = 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(equal(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, eqExpr) {
  auto call = parseCallExpr("a = 21 * 2", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(equal(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, eqSubfield) {
  auto call = parseCallExpr("a.b = 42", ROW("a", ROW("b", BIGINT())));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a", "b"});

  VELOX_ASSERT_FILTER(equal(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, neq) {
  auto call = parseCallExpr("a <> 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  // TODO Optimize to notEqual(42).
  VELOX_ASSERT_FILTER(bigintOr(lessThan(42), greaterThan(42)), filter);
}

TEST_F(ExprToSubfieldFilterTest, lte) {
  auto call = parseCallExpr("a <= 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(lessThanOrEqual(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, lt) {
  auto call = parseCallExpr("a < 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(lessThan(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, gte) {
  auto call = parseCallExpr("a >= 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(greaterThanOrEqual(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, gt) {
  auto call = parseCallExpr("a > 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(greaterThan(42), filter);
}

TEST_F(ExprToSubfieldFilterTest, between) {
  auto call = parseCallExpr("a between 40 and 42", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(between(40, 42), filter);
}

TEST_F(ExprToSubfieldFilterTest, in) {
  auto call = parseCallExpr("a in (40, 42)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(in({40, 42}), filter);
}

TEST_F(ExprToSubfieldFilterTest, isNull) {
  auto call = parseCallExpr("a is null", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  VELOX_ASSERT_FILTER(isNull(), filter);
}

TEST_F(ExprToSubfieldFilterTest, like) {
  auto call = parseCallExpr("a like 'foo%'", ROW("a", VARCHAR()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, nonConstant) {
  auto call = parseCallExpr("a = b + 1", ROW({"a", "b"}, BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, userError) {
  auto call = parseCallExpr("a = 1 / 0", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

TEST_F(ExprToSubfieldFilterTest, dereferenceWithEmptyField) {
  auto call = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      "is_null",
      std::make_shared<core::DereferenceTypedExpr>(
          REAL(),
          std::make_shared<core::FieldAccessTypedExpr>(
              ROW({DOUBLE(), REAL(), BIGINT()}),
              std::make_shared<core::InputTypedExpr>(
                  ROW("c0", ROW({DOUBLE(), REAL(), BIGINT()}))),
              "c0"),
          1));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_FALSE(filter);
}

} // namespace
} // namespace facebook::velox::exec

#undef VELOX_ASSERT_FILTER
