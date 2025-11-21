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

#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/ExprOptimizer.h"
#include "velox/expression/ExprRewriteRegistry.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/JsonType.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"

namespace facebook::velox::expression {
namespace {

class ExprOptimizerTest : public functions::test::FunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions("");
    parse::registerTypeResolver();
    options_.parseDecimalAsDouble = false;
  }

  void TearDown() override {
    expression::ExprRewriteRegistry::instance().clear();
  }

  /// Validates expression::optimize API. The input expression is optimized and
  /// then evaluated with a randomly generated input vector.
  /// @param input String representation of input SQL expression to be
  /// optimized.
  /// @param expected String representation of expected SQL expression after
  /// optimization of `input`.
  /// @param type Type of `input` and `expected`.
  /// @param makeFailExpr Callback that returns an expression to replace any
  /// failing subexpression(s) during optimization of `input`.
  /// @param evalError Expected error message upon evaluating optimized `input`.
  /// Should be used iff `makeFailExpr` returns an expression that throws upon
  /// evaluation (eg: Presto `fail` function).
  void testExpression(
      const std::string& input,
      const std::string& expected,
      const RowTypePtr& type = ROW({}),
      const MakeFailExpr& makeFailExpr = nullptr,
      const std::string& evalError = "") {
    const auto typedExpr = makeTypedExpr(input, type);
    auto optimized =
        expression::optimize(typedExpr, queryCtx_.get(), pool(), makeFailExpr);
    const auto expectedTypedExpr = makeTypedExpr(expected, type);
    SCOPED_TRACE(
        fmt::format(
            "Input: {}\nOptimized: {}\nExpected: {}",
            typedExpr->toString(),
            optimized->toString(),
            expectedTypedExpr->toString()));
    ASSERT_TRUE(*optimized == *expectedTypedExpr);

    // Generate random values with fuzzer to validate evaluation of optimized
    // expression.
    const auto data = fuzzFlat(type);
    if (makeFailExpr != nullptr && !evalError.empty()) {
      VELOX_ASSERT_THROW(evaluate(optimized, data), evalError);
    } else {
      const auto optimizedResult = evaluate(optimized, data);
      const auto expectedResult = evaluate(expectedTypedExpr, data);
      test::assertEqualVectors(optimizedResult, expectedResult);
    }
  }

  void setQueryTimeZone(const std::string& timeZone) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kSessionTimezone, timeZone},
         {core::QueryConfig::kAdjustTimestampToTimezone, "true"}});
  }

 private:
  RowVectorPtr fuzzFlat(const RowTypePtr& rowType) {
    VectorFuzzer::Options options;
    options.vectorSize = 100;
    VectorFuzzer fuzzer(options, pool());
    return fuzzer.fuzzInputFlatRow(rowType);
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
  parse::ParseOptions options_;
};

TEST_F(ExprOptimizerTest, constantFolding) {
  // comparison.
  testExpression("'a' = substr('abc', 1, 1)", "true");
  testExpression("'a' = cast(null as varchar)", "cast(null AS BOOLEAN)");
  testExpression(
      "cast((12345678901234567890.000 + 0.123) AS DECIMAL(23, 3)) = cast(12345678901234567890.123 AS DECIMAL(23, 3))",
      "true");
  testExpression("'0ab01c' LIKE '%ab%c%'", "true");
  testExpression("3 between (4 - 2) and 2 * 2", "true");
  testExpression(
      "3 between cast(null AS BIGINT) and 4", "cast(null AS BOOLEAN)");
  testExpression("IF(2 = 2, 3, 4)", "3");
  testExpression("IF(cast(null AS BOOLEAN), 3, 4)", "4");
  testExpression("IF('hello' = 'foo', cast(null AS BIGINT), 4)", "4");
  testExpression(
      "IF('hello' = 'hello', cast(null AS BIGINT), 4)", "cast(null AS BIGINT)");
  testExpression("random() = random()", "random() = random()");
  testExpression("current_date() = current_date()", "true");

  // cast.
  testExpression("cast('-123' as BIGINT)", "-123");
  testExpression("cast(DECIMAL '1234567890.123' as BIGINT)", "1234567890");
  testExpression("cast('t' as BOOLEAN)", "true");
  testExpression("cast('f' as BOOLEAN)", "false");
  testExpression("cast('1' as BOOLEAN)", "true");
  testExpression("cast('0' as BOOLEAN)", "false");
  testExpression("cast(-123.456E0 as VARCHAR)", "'-123.456'");
  testExpression("cast(cast('abcxyz' as VARCHAR) as VARCHAR)", "'abcxyz'");

  // Partial constant folding in expression tree.
  testExpression(
      "1234 between a and 2000 + 1",
      "1234 between a and 2001",
      ROW({"a"}, {BIGINT()}));
  testExpression(
      "concat(substr(a, 1 + 5 - 2), substr(b, 2 * 3))",
      "concat(substr(a, 4), substr(b, 6))",
      ROW({"a", "b"}, {VARCHAR(), VARCHAR()}));
  testExpression(
      "a = 'z' and b = (1 + 1) or c = (5 - 1) * 3",
      "a = 'z' and b = 2 or c = 12",
      ROW({"a", "b", "c"}, {VARCHAR(), BIGINT(), INTEGER()}));
}

TEST_F(ExprOptimizerTest, specialFormConstantFolding) {
  // AND, OR.
  testExpression("true and true", "true");
  testExpression("false or false", "false");
  testExpression("null::boolean and false", "false");
  testExpression("true or null::boolean", "true");
  testExpression(
      "null::boolean and null::boolean and null::boolean", "null::boolean");
  testExpression(
      "null::boolean or null::boolean or null::boolean", "null::boolean");
  testExpression("null::boolean and true", "null::boolean");
  testExpression("false or null::boolean", "null::boolean");

  // IF, SWITCH.
  testExpression("if(true, 'hello', 'world')", "'hello'");
  testExpression("case when false then 1 when true then 3 end", "3");
  testExpression(
      "case when false then 1 when false then 3 end", "null::bigint");
  testExpression("case when false then 1 when false then 3 else 2 end", "2");
  testExpression(
      "case when false then 'hello' when false then 'world' when true then 'foo' else 'bar' end",
      "'foo'");
}

TEST_F(ExprOptimizerTest, lambdas) {
  auto type = ROW({"a"}, {ARRAY(BIGINT())});
  testExpression(
      "filter(a, (x) -> abs(x + (1 + 2)) = (6 / 2))",
      "filter(a, (x) -> abs(x + 3) = 3)",
      type);
  testExpression(
      "filter(a, (x) -> 1 * (4 / 2) = abs(-4 * 3))",
      "filter(a, (x) -> false)",
      type);
  type = ROW({"a"}, {ARRAY(VARCHAR())});
  testExpression(
      "filter(a, (x) -> length(x) + (3 - 2) = 4 / 2)",
      "filter(a, (x) -> length(x) + 1 = 2)",
      type);
  testExpression(
      "array_sort(a, x -> length(x) * abs(-1 * 3 * 2))",
      "array_sort(a, x -> length(x) * 6)",
      type);
  testExpression(
      "reduce(c0, (89.0E0 + 11.0E0), (s, x) -> s + x * (0.2E0 - 0.1E0), s -> (s < (100.0E0 + 1.0E0)))",
      "reduce(c0, 100.0E0, (s, x) -> s + x * 0.1E0, s -> (s < 101.0E0))",
      ROW({"c0"}, {ARRAY(DOUBLE())}));
  testExpression(
      "reduce(c0, 1 - 1, (s, x) -> s + x, s -> coalesce(s, abs(2 - 10)) * (5 * 2))",
      "reduce(c0, 0, (s, x) -> s + x, s -> coalesce(s, 8) * 10)",
      ROW({"c0"}, {ARRAY(SMALLINT())}));
}

TEST_F(ExprOptimizerTest, rewritesWithConstantFolding) {
  auto type = ROW({"c0"}, {ARRAY(VARCHAR())});
  testExpression(
      "array_sort(c0, (x, y) -> if(length(x) < length(y), (-1 * 1), if(length(x) = length(y), abs(0), (2 / 2))))",
      "array_sort(c0, x -> length(x))",
      type);
  testExpression(
      "array_sort(c0, (x, y) -> if(length(x) < length(y), abs(-1), if(length(x) > length(y), -4 / 3, 0 / 3)))",
      "array_sort_desc(c0, x -> length(x))",
      type);

  testExpression(
      "reduce(c0, 8 / 2, (s, x) -> if(x % 2 = 0, s + 1, s), s -> s)",
      "4 + cast(array_sum_propagate_element_null(transform(c0, x -> if(x % 2 = 0, 1, 0))) AS BIGINT)",
      ROW({"c0"}, {ARRAY(INTEGER())}));
  testExpression(
      "reduce(c0, length(concat('abc', 'xyz')) * 3, (s, x) -> s + (abs(-2) + (5 - 4)) - x, s -> s)",
      "18 + cast(array_sum_propagate_element_null(transform(c0, x -> 3 - x)) AS BIGINT)",
      ROW({"c0"}, {ARRAY(TINYINT())}));

  // Conjunct rewrites with constant folding.
  type = ROW({"a"}, {ARRAY(BIGINT())});
  testExpression("if(array_position(a, 1) > 10 and false, 0, 1)", "1", type);
  testExpression("if(cardinality(a) > 10 or true, 0, 1)", "0", type);
  testExpression(
      "filter(a, x -> (2 * 3) = x and abs(-4) = (3 - 1))",
      "filter(a, x -> false)",
      type);
  testExpression(
      "filter(a, x -> (2 * 3) = x or (8 / 2) = (3 + 1))",
      "filter(a, x -> true)",
      type);

  // Switch rewrite with constant folding.
  testExpression(
      "case when ARRAY[abs(-1)] = ARRAY[2] then 'not_matched' when ARRAY[1] = ARRAY[2 - 1] then 'matched' else 'default' end",
      "'matched'");
  testExpression(
      "case when 10 + a = 100 / 2 then 10 - 2 when a / (2 + 1) = abs(-5) then 10 * 10 when (123 * 10) + 4 = abs(-1234) then 11 * 3 else a end",
      "case when 10 + a = 50 then 8 when a / 3 = 5 then 100 else 33 end",
      ROW({"a"}, {BIGINT()}));

  // Coalesce rewrite with constant folding.
  type = ROW({"a", "b"}, {BIGINT(), BIGINT()});
  testExpression(
      "coalesce(a - (4 / 2), a - (1 * 2), null::bigint, 1 - 1, b, null::bigint)",
      "coalesce(a - 2, 0)",
      type);
  testExpression(
      "coalesce(null::bigint, a / abs(-2 * 3), coalesce(coalesce(b, 8 / 2), a * b), b)",
      "coalesce(a / 6, b, 4)",
      type);
}

/// Test to ensure session queryCtx is used during expression optimization.
TEST_F(ExprOptimizerTest, queryCtx) {
  auto testFromUnixtime = [&](const std::string& function,
                              const std::string& timezone,
                              const std::string& expected) {
    setQueryTimeZone(timezone);
    testExpression(
        fmt::format("{}(from_unixtime(9.98489045321E8))", function), expected);
  };

  testFromUnixtime("hour", "Pacific/Apia", "3");
  testFromUnixtime("hour", "America/Los_Angeles", "7");
  testFromUnixtime("minute", "Pacific/Apia", "4");
  testFromUnixtime("minute", "America/Los_Angeles", "4");
}

/// Test cast optimization that avoids expression evaluation when input to
/// cast is same as the type of cast expression.
TEST_F(ExprOptimizerTest, castExpr) {
  auto testCast = [&](const TypePtr& type, const std::string& typeString = "") {
    const auto rowType = ROW({"a"}, {type});
    if (type->isPrimitiveType()) {
      testExpression(
          fmt::format("cast(a as {})", type->toString()), "a", rowType);
    } else {
      testExpression(fmt::format("cast(a as {})", typeString), "a", rowType);
    }
  };

  // Primitive types.
  std::vector<TypePtr> primitiveTypes = {
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      DOUBLE(),
      REAL(),
      VARBINARY(),
      VARCHAR(),
      DECIMAL(5, 2),
      DECIMAL(25, 10),
      DATE(),
      TIMESTAMP_WITH_TIME_ZONE()};
  for (const auto& type : primitiveTypes) {
    testCast(type);
  }

  // Complex types.
  testCast(ARRAY(DATE()), "DATE[]");
  testCast(MAP(DOUBLE(), VARCHAR()), "map(DOUBLE, VARCHAR)");
  testCast(
      ROW({"c0", "c1", "c2"},
          {BIGINT(), DECIMAL(5, 2), TIMESTAMP_WITH_TIME_ZONE()}),
      "struct(c0 BIGINT, c1 DECIMAL(5, 2), c2 TIMESTAMP WITH TIME ZONE)");

  // Functions.
  auto type = ROW({"a"}, {VARCHAR()});
  testExpression(
      "cast(concat(a, 'test') as VARCHAR)", "concat(a, 'test')", type);
  testExpression(
      "cast(substr(a, abs(-2) * (3 / 1)) as VARCHAR)", "substr(a, 6)", type);
  testExpression(
      "cast((a = (5 / 1) * abs(-3)) as BOOLEAN) and cast((b = concat('hello', 'world')) as BOOLEAN)",
      "a = 15 and b = 'helloworld'",
      ROW({"a", "b"}, {BIGINT(), VARCHAR()}));
}

/// Test to validate `makeFailExpr` returns an expression that replaces all
/// failing subexpressions in the expression tree.
TEST_F(ExprOptimizerTest, makeFailExpr) {
  const auto divideByZeroError = "division by zero";
  auto assertPrestoFailExpr = [&](const std::string& expression,
                                  const std::string& expected,
                                  const RowTypePtr& type) {
    const static MakeFailExpr prestoFailExpr =
        [&](const std::string& error,
            const TypePtr& type) -> core::TypedExprPtr {
      return makeTypedExpr(
          fmt::format("cast(fail('{}') AS {})", error, type->toString()),
          ROW({}));
    };
    testExpression(
        expression, expected, type, prestoFailExpr, divideByZeroError);
  };
  auto prestoFailCall = [&]() {
    return fmt::format("cast(fail('{}') AS BIGINT)", divideByZeroError);
  };

  // Primitive types.
  assertPrestoFailExpr("0 / 0", prestoFailCall(), ROW({}));
  assertPrestoFailExpr(
      "if(a = 2 * 2, a / abs(-1 * 3), 0 / 0)",
      fmt::format("if(a = 4, a / 3, {})", prestoFailCall()),
      ROW({"a"}, {BIGINT()}));
  assertPrestoFailExpr(
      "json_extract(a, substr(b, 1 / 0))",
      fmt::format("json_extract(a, substr(b, {}))", prestoFailCall()),
      ROW({"a", "b"}, {JSON(), VARCHAR()}));

  // Complex types.
  assertPrestoFailExpr(
      "filter(a, x -> (1 / 0) > 1)",
      fmt::format("filter(a, x -> {} > 1)", prestoFailCall()),
      ROW({"a"}, {ARRAY(BIGINT())}));
  assertPrestoFailExpr(
      "map_top_n(a, 1 / 0)",
      fmt::format("map_top_n(a, {})", prestoFailCall()),
      ROW({"a"}, {MAP(VARCHAR(), BIGINT())}));

  // Special form expressions.
  assertPrestoFailExpr(
      "coalesce(0 / 0 > 1, a, 0 / 0 = 0)",
      fmt::format(
          "coalesce({} > 1, a, {} = 0)", prestoFailCall(), prestoFailCall()),
      ROW({"a"}, {BOOLEAN()}));
  assertPrestoFailExpr(
      "CASE abs(-1234) WHEN a + (2 + 3) THEN 2 / 1 WHEN 0 / 0 THEN abs(-2) ELSE length('abc') END",
      fmt::format(
          "CASE 1234 WHEN a + 5 THEN 2 WHEN {} THEN 2 ELSE 3 END",
          prestoFailCall()),
      ROW({"a"}, {BIGINT()}));
}

} // namespace
} // namespace facebook::velox::expression
