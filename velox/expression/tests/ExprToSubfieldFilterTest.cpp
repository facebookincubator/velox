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
        parse::DuckSqlExpressionsParser().parseExpr(expr), type, pool_.get());
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
  leafCallToSubfieldFilter(
      const core::CallTypedExprPtr& call,
      bool negated = false) {
    if (auto result =
            ExprToSubfieldFilterParser::getInstance()->leafCallToSubfieldFilter(
                *call, evaluator(), negated)) {
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
  {
    auto call = parseCallExpr("a in (40, 42)", ROW("a", BIGINT()));
    auto [subfield, filter] = leafCallToSubfieldFilter(call);

    ASSERT_TRUE(filter);
    validateSubfield(subfield, {"a"});

    VELOX_ASSERT_FILTER(in({40, 42}), filter);
  }

  {
    auto call = parseCallExpr("a in (40, 42, null)", ROW("a", BIGINT()));
    auto [subfield, filter] = leafCallToSubfieldFilter(call);

    ASSERT_TRUE(filter);
    validateSubfield(subfield, {"a"});

    VELOX_ASSERT_FILTER(in({40, 42}), filter);
  }
}

TEST_F(ExprToSubfieldFilterTest, notIn) {
  {
    auto call = parseCallExpr("a in (40, 42)", ROW("a", BIGINT()));
    auto [subfield, filter] = leafCallToSubfieldFilter(call, true);

    ASSERT_TRUE(filter);
    validateSubfield(subfield, {"a"});

    VELOX_ASSERT_FILTER(notIn({40, 42}), filter);
  }

  {
    auto call = parseCallExpr("a in (40, 42, null)", ROW("a", BIGINT()));
    auto [subfield, filter] = leafCallToSubfieldFilter(call, true);

    ASSERT_TRUE(filter);
    validateSubfield(subfield, {"a"});

    auto expected = std::make_unique<common::AlwaysFalse>();
    VELOX_ASSERT_FILTER(expected, filter);
  }
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

template <typename... Disjuncts>
static std::unique_ptr<common::Filter> makeOr(Disjuncts&&... disjuncts) {
  return ExprToSubfieldFilterParser::makeOrFilter(
      std::forward<Disjuncts>(disjuncts)...);
}

TEST_F(ExprToSubfieldFilterTest, makeOrFilterBigint) {
  // a = 1 or a = 5
  {
    auto expected = in({1, 5});
    VELOX_ASSERT_FILTER(expected, makeOr(equal(1), equal(5)));
    VELOX_ASSERT_FILTER(expected, makeOr(equal(5), equal(1)));
  }

  // a = 1 or a = 2 ==> a between 1 and 2
  {
    auto expected = between(1, 2);
    VELOX_ASSERT_FILTER(expected, makeOr(equal(1), equal(2)));
    VELOX_ASSERT_FILTER(expected, makeOr(equal(2), equal(1)));
    VELOX_ASSERT_FILTER(expected, makeOr(between(1, 2), equal(1)));
    VELOX_ASSERT_FILTER(expected, makeOr(equal(2), between(1, 2)));
  }

  // a = 1 or a between 5 and 10 or a = 11 ==> a = 1 or a between 5 and 11
  {
    auto expected = bigintOr(equal(1), between(5, 11));

    VELOX_ASSERT_FILTER(expected, makeOr(equal(1), between(5, 10), equal(11)));
    VELOX_ASSERT_FILTER(expected, makeOr(equal(1), equal(11), between(5, 10)));
    VELOX_ASSERT_FILTER(expected, makeOr(between(5, 10), equal(11), equal(1)));
  }

  // a < 10 or a > 0
  {
    VELOX_ASSERT_FILTER(isNotNull(), makeOr(lessThan(10), greaterThan(0)));
    VELOX_ASSERT_FILTER(
        alwaysTrue(),
        makeOr(lessThan(10), greaterThan(0, /*nullAllowed=*/true)));
  }

  // a > 10 or a between 12 and 100
  {
    auto expected = greaterThanOrEqual(11);
    VELOX_ASSERT_FILTER(expected, makeOr(greaterThan(10), between(12, 100)));
    VELOX_ASSERT_FILTER(expected, makeOr(between(12, 100), greaterThan(10)));
  }
}

TEST_F(ExprToSubfieldFilterTest, makeOrFilterDouble) {
  // a < 1.5 or a > 1.0 ==> not null
  {
    VELOX_ASSERT_FILTER(
        isNotNull(), makeOr(lessThanDouble(1.5), greaterThanDouble(1.0)));

    VELOX_ASSERT_FILTER(
        alwaysTrue(),
        makeOr(
            lessThanDouble(1.5, /*nullAllowed=*/true), greaterThanDouble(1.0)));
  }

  // a < 10.1 or a between 9.0 and 12.0 or a between 11.0 and 15.0 ==> a <= 15.0
  {
    auto expected = lessThanOrEqualDouble(15.0);
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(
            lessThanDouble(10.1),
            betweenDouble(9.0, 12.0),
            betweenDouble(11.0, 15.0)));
  }

  // a < 0.0 or a > 4.5
  {
    auto expected = orFilter(lessThanDouble(0.0), greaterThanDouble(4.5));
    VELOX_ASSERT_FILTER(
        expected, makeOr(lessThanDouble(0.0), greaterThanDouble(4.5)));
  }
}

TEST_F(ExprToSubfieldFilterTest, makeOrFilterFloat) {
  // a < 1.5 or a > 1.0 ==> not null
  {
    VELOX_ASSERT_FILTER(
        isNotNull(), makeOr(lessThanFloat(1.5), greaterThanFloat(1.0)));

    VELOX_ASSERT_FILTER(
        alwaysTrue(),
        makeOr(
            lessThanFloat(1.5, /*nullAllowed=*/true), greaterThanFloat(1.0)));
  }

  // a < 10.1 or a between 9.0 and 12.0 or a between 11.0 and 15.0 ==> a <= 15.0
  {
    auto expected = lessThanOrEqualFloat(15.0);
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(
            lessThanFloat(10.1),
            betweenFloat(9.0, 12.0),
            betweenFloat(11.0, 15.0)));
  }

  // a < 0.0 or a > 4.5
  {
    auto expected = orFilter(lessThanFloat(0.0), greaterThanFloat(4.5));
    VELOX_ASSERT_FILTER(
        expected, makeOr(lessThanFloat(0.0), greaterThanFloat(4.5)));
  }

  // a < 1 or a > 1
  {
    auto expected = orFilter(lessThanFloat(1.0), greaterThanFloat(1.0));
    VELOX_ASSERT_FILTER(
        expected, makeOr(lessThanFloat(1.0), greaterThanFloat(1.0)));
    VELOX_ASSERT_FILTER(
        expected, makeOr(greaterThanFloat(1.0), lessThanFloat(1.0)));
  }
}

TEST_F(ExprToSubfieldFilterTest, makeOrFilterBytesValues) {
  // a = 'FRANCE' or a = 'JAPAN' ==> a in ('FRANCE', 'JAPAN')
  // Note: equal(string) generates BytesValues filter
  {
    auto expected = in({"FRANCE", "JAPAN"});
    VELOX_ASSERT_FILTER(expected, makeOr(equal("FRANCE"), equal("JAPAN")));
    VELOX_ASSERT_FILTER(expected, makeOr(equal("JAPAN"), equal("FRANCE")));
  }

  // a = 'FRANCE' or a = 'GERMANY' or a = 'JAPAN'
  // ==> a in ('FRANCE', 'GERMANY', 'JAPAN')
  {
    auto expected = in({"FRANCE", "GERMANY", "JAPAN"});
    VELOX_ASSERT_FILTER(
        expected, makeOr(equal("FRANCE"), equal("GERMANY"), equal("JAPAN")));
  }

  // a in ('FRANCE', 'JAPAN') or a = 'GERMANY'
  // ==> a in ('FRANCE', 'JAPAN', 'GERMANY')
  {
    auto expected = in({"FRANCE", "JAPAN", "GERMANY"});
    VELOX_ASSERT_FILTER(
        expected, makeOr(in({"FRANCE", "JAPAN"}), equal("GERMANY")));
  }

  // a in ('FRANCE', 'JAPAN') or a in ('GERMANY', 'ITALY')
  // ==> a in ('FRANCE', 'JAPAN', 'GERMANY', 'ITALY')
  {
    auto expected = in({"FRANCE", "JAPAN", "GERMANY", "ITALY"});
    VELOX_ASSERT_FILTER(
        expected, makeOr(in({"FRANCE", "JAPAN"}), in({"GERMANY", "ITALY"})));
  }

  // Test with nullAllowed
  {
    auto expected = std::make_unique<common::BytesValues>(
        std::vector<std::string>{"FRANCE", "JAPAN"}, /*nullAllowed=*/true);
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(
            equal("FRANCE"),
            std::make_unique<common::BytesValues>(
                std::vector<std::string>{"JAPAN"}, /*nullAllowed=*/true)));
  }
}

TEST_F(ExprToSubfieldFilterTest, makeOrFilterBytesRange) {
  // Test BytesRange with single value (singleValue_ = true)
  // between("FRANCE", "FRANCE") creates a BytesRange with singleValue_ = true
  {
    auto expected = in({"FRANCE", "JAPAN"});
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(between("FRANCE", "FRANCE"), between("JAPAN", "JAPAN")));
  }

  // Test mixing BytesRange (single value) with BytesValues
  {
    auto expected = in({"FRANCE", "JAPAN", "GERMANY"});
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(between("FRANCE", "FRANCE"), in({"JAPAN", "GERMANY"})));
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(in({"JAPAN", "GERMANY"}), between("FRANCE", "FRANCE")));
  }

  // Test BytesRange with single value combined with equal() (BytesValues)
  {
    auto expected = in({"FRANCE", "JAPAN"});
    VELOX_ASSERT_FILTER(
        expected, makeOr(between("FRANCE", "FRANCE"), equal("JAPAN")));
    VELOX_ASSERT_FILTER(
        expected, makeOr(equal("JAPAN"), between("FRANCE", "FRANCE")));
  }

  // Test multiple BytesRange with single values
  {
    auto expected = in({"FRANCE", "GERMANY", "JAPAN"});
    VELOX_ASSERT_FILTER(
        expected,
        makeOr(
            between("FRANCE", "FRANCE"),
            between("GERMANY", "GERMANY"),
            between("JAPAN", "JAPAN")));
  }
}

// Test NULL comparison handling - comparisons with NULL should return
// AlwaysFalse as per SQL three-valued logic
TEST_F(ExprToSubfieldFilterTest, eqNull) {
  auto call = parseCallExpr("a = cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, eqNullVarchar) {
  auto call = parseCallExpr("a = cast(null as varchar)", ROW("a", VARCHAR()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, eqNullTimestamp) {
  auto call =
      parseCallExpr("a = cast(null as timestamp)", ROW("a", TIMESTAMP()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, neqNull) {
  auto call = parseCallExpr("a <> cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, ltNull) {
  auto call = parseCallExpr("a < cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, ltNullDouble) {
  auto call = parseCallExpr("a < cast(null as double)", ROW("a", DOUBLE()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, ltNullVarchar) {
  auto call = parseCallExpr("a < cast(null as varchar)", ROW("a", VARCHAR()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, lteNull) {
  auto call = parseCallExpr("a <= cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, lteNullReal) {
  auto call = parseCallExpr("a <= cast(null as real)", ROW("a", REAL()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, gtNull) {
  auto call = parseCallExpr("a > cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, gtNullInteger) {
  auto call = parseCallExpr("a > cast(null as integer)", ROW("a", INTEGER()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, gteNull) {
  auto call = parseCallExpr("a >= cast(null as bigint)", ROW("a", BIGINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, gteNullSmallint) {
  auto call =
      parseCallExpr("a >= cast(null as smallint)", ROW("a", SMALLINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

TEST_F(ExprToSubfieldFilterTest, gteNullTinyint) {
  auto call = parseCallExpr("a >= cast(null as tinyint)", ROW("a", TINYINT()));
  auto [subfield, filter] = leafCallToSubfieldFilter(call);

  ASSERT_TRUE(filter);
  validateSubfield(subfield, {"a"});

  auto expected = std::make_unique<common::AlwaysFalse>();
  VELOX_ASSERT_FILTER(expected, filter);
}

} // namespace
} // namespace facebook::velox::exec

#undef VELOX_ASSERT_FILTER
