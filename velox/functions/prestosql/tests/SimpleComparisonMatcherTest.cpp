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

#include <limits>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/SimpleComparisonMatcher.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions::prestosql {
namespace {

class SimpleComparisonMatcherTest : public testing::Test,
                                    public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions(prefix_);
    parse::registerTypeResolver();
  }

  core::TypedExprPtr parseExpression(
      const std::string& text,
      const RowTypePtr& rowType) {
    parse::ParseOptions options;
    options.functionPrefix = prefix_;
    options.parseIntegerAsBigint = false;
    auto untyped = parse::DuckSqlExpressionsParser(options).parseExpr(text);
    return core::Expressions::inferTypes(untyped, rowType, execCtx_->pool());
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
  const std::string prefix_ = "tp.";
};

class TestFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_UNSUPPORTED();
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(T)")
                .argumentType("array(T)")
                .argumentType("function(T,T,integer)")
                .build()};
  }
};

class AlwaysMatchingMatcher : public Matcher {
 public:
  bool match(const core::TypedExprPtr& /*expr*/) override {
    return true;
  }
};

TEST_F(SimpleComparisonMatcherTest, arityMismatch) {
  std::vector<core::TypedExprPtr> expressions(2);
  std::vector<std::shared_ptr<Matcher>> matchers{
      std::make_shared<AlwaysMatchingMatcher>(),
      std::make_shared<AlwaysMatchingMatcher>(),
      std::make_shared<AlwaysMatchingMatcher>()};

  ASSERT_FALSE(Matcher::allMatch(expressions, matchers));
}

TEST_F(SimpleComparisonMatcherTest, integerVectorConstants) {
  for (const auto expected :
       {std::numeric_limits<int32_t>::min(),
        -10,
        0,
        37,
        std::numeric_limits<int32_t>::max()}) {
    int64_t actual{0};
    ComparisonConstantMatcher matcher(&actual);
    const auto expression = std::make_shared<core::ConstantTypedExpr>(
        makeFlatVector<int32_t>({expected}));

    ASSERT_TRUE(matcher.match(expression));
    ASSERT_EQ(expected, actual);
  }

  int64_t actual{123};
  ComparisonConstantMatcher matcher(&actual);
  const auto nullExpression = std::make_shared<core::ConstantTypedExpr>(
      makeNullableFlatVector<int32_t>({std::nullopt}));
  ASSERT_FALSE(matcher.match(nullExpression));
  ASSERT_EQ(123, actual);
}

TEST_F(SimpleComparisonMatcherTest, basic) {
  exec::registerVectorFunction(
      prefix_ + "test_array_sort",
      TestFunction::signatures(),
      std::make_unique<TestFunction>());

  const auto inputType =
      ROW({"a", "captured", "captured_array"},
          {ARRAY(ROW({"f", "g"}, {BIGINT(), BIGINT()})),
           BIGINT(),
           ARRAY(BIGINT())});

  auto checker = std::make_unique<SimpleComparisonChecker>();

  auto testMatcher = [&](const std::string& expr,
                         std::optional<bool> lessThan) {
    SCOPED_TRACE(expr);
    auto parsedExpr = parseExpression(
        fmt::format("test_array_sort(a, (x, y) -> {})", expr), inputType);

    auto lambdaExpr = std::dynamic_pointer_cast<const core::LambdaTypedExpr>(
        parsedExpr->inputs()[1]);

    auto comparison = checker->isSimpleComparison(prefix_, *lambdaExpr);

    ASSERT_EQ(lessThan.has_value(), comparison.has_value());
    if (lessThan.has_value()) {
      ASSERT_EQ(lessThan.value(), comparison->isLessThen);

      if (expr.find("captured") == std::string::npos) {
        auto field = dynamic_cast<const core::DereferenceTypedExpr*>(
            comparison->expr.get());
        ASSERT_TRUE(field != nullptr);
        ASSERT_EQ(0, field->index());
      }
    }
  };

  // Different ways to define x < y (asc) sort order.
  testMatcher("if(x.f > y.f, 1, if(x.f < y.f, -1, 0))", true);
  testMatcher("if(x.f > y.f, 1, if(y.f > x.f, -1, 0))", true);
  testMatcher("if(x.f > y.f, 1, if(x.f = y.f, 0, -1))", true);

  testMatcher("if(x.f < y.f, -1, if(x.f > y.f, 1, 0))", true);
  testMatcher("if(x.f < y.f, -1, if(y.f < x.f, 1, 0))", true);
  testMatcher("if(x.f < y.f, -1, if(x.f = y.f, 0, 1))", true);

  testMatcher("if(x.f = y.f, 0, if(x.f < y.f, -1, 1))", true);
  testMatcher("if(x.f = y.f, 0, if(y.f > x.f, -1, 1))", true);
  testMatcher("if(x.f = y.f, 0, if(x.f > y.f, 1, -1))", true);
  testMatcher("if(x.f = y.f, 0, if(y.f < x.f, 1, -1))", true);

  // Different ways to define x > y (desc) sort order.
  testMatcher("if (x.f < y.f, 1, if (x.f > y.f, -1, 0))", false);
  testMatcher("if (x.f < y.f, 1, if (y.f < x.f, -1, 0))", false);
  testMatcher("if (x.f < y.f, 1, if (y.f = x.f, 0, -1))", false);

  testMatcher("if (x.f > y.f, -1, if (x.f < y.f, 1, 0))", false);
  testMatcher("if (x.f > y.f, -1, if (y.f > x.f, 1, 0))", false);
  testMatcher("if (x.f > y.f, -1, if (y.f = x.f, 0, 1))", false);

  testMatcher("if(x.f = y.f, 0, if(x.f < y.f, 1, -1))", false);
  testMatcher("if(x.f = y.f, 0, if(y.f > x.f, 1, -1))", false);
  testMatcher("if(x.f = y.f, 0, if(x.f > y.f, -1, 1))", false);
  testMatcher("if(x.f = y.f, 0, if(y.f < x.f, -1, 1))", false);

  // Non-unit comparator values are not supported by Presto.
  testMatcher("if(x.f = y.f, 0, if(x.f < y.f, -10, 37))", std::nullopt);
  testMatcher("if(x.f < y.f, -10, if(x.f = y.f, 0, 37))", std::nullopt);
  testMatcher("if(x.f = y.f, 0, if(x.f < y.f, 37, -10))", std::nullopt);
  testMatcher("if(x.f < y.f, 37, if(x.f = y.f, 0, -10))", std::nullopt);

  // Captures shared by the left and right transforms.
  testMatcher(
      "if(coalesce(x.f, captured) < coalesce(y.f, captured), -1, "
      "if(coalesce(x.f, captured) > coalesce(y.f, captured), 1, 0))",
      true);
  testMatcher(
      "if(x.f + cardinality(filter(captured_array, z -> z > captured)) < "
      "y.f + cardinality(filter(captured_array, z -> z > captured)), -1, "
      "if(x.f + cardinality(filter(captured_array, z -> z > captured)) > "
      "y.f + cardinality(filter(captured_array, z -> z > captured)), 1, 0))",
      true);

  // Non-matching expressions.
  testMatcher("if(x.f + y.f > 0, 1, -1)", std::nullopt);
  testMatcher("if(x.f < y.f, 1, -1)", std::nullopt);
  testMatcher("if(x.f = y.f, 0, if(x.f > y.f, -1, 0))", std::nullopt);
  testMatcher("if(x.f = y.f, 1, if(x.f > y.f, -1, 0))", std::nullopt);
  testMatcher("if(x.f = y.f, 5, if(x.f < y.f, -10, 37))", std::nullopt);
  testMatcher("if(x.f < y.f, -10, if(x.f = y.f, 5, 37))", std::nullopt);
  testMatcher("if(x.f = y.f, 0, if(x.f < y.f, -10, -20))", std::nullopt);
  testMatcher("if(x.f < y.f, 10, if(x.f = y.f, 0, 20))", std::nullopt);
  testMatcher("if(x.f < x.f, -10, if(x.f > x.f, 37, 0))", std::nullopt);
  testMatcher(
      "if(x.f < captured, -10, if(x.f > captured, 37, 0))", std::nullopt);
  testMatcher(
      "if(coalesce(x.f, captured) < y.f, -10, "
      "if(coalesce(x.f, captured) > y.f, 37, 0))",
      std::nullopt);
  testMatcher(
      "if(x.f + cardinality(filter(captured_array, z -> z > y.f)) < "
      "y.f + cardinality(filter(captured_array, z -> z > x.f)), -1, "
      "if(x.f + cardinality(filter(captured_array, z -> z > y.f)) > "
      "y.f + cardinality(filter(captured_array, z -> z > x.f)), 1, 0))",
      std::nullopt);
  testMatcher(
      "if(x.f + cardinality(filter(captured_array, y -> y > captured)) < "
      "y.f + cardinality(filter(captured_array, x -> x > captured)), -1, "
      "if(x.f + cardinality(filter(captured_array, y -> y > captured)) > "
      "y.f + cardinality(filter(captured_array, x -> x > captured)), 1, 0))",
      std::nullopt);
  testMatcher(
      "if(x.f + cardinality(filter(captured_array, "
      "z -> z > cast(random() as bigint))) < "
      "y.f + cardinality(filter(captured_array, "
      "z -> z > cast(random() as bigint))), -1, "
      "if(x.f + cardinality(filter(captured_array, "
      "z -> z > cast(random() as bigint))) > "
      "y.f + cardinality(filter(captured_array, "
      "z -> z > cast(random() as bigint))), 1, 0))",
      std::nullopt);
  testMatcher("if(x.f > (y.f + 5), 1, if(x.f < y.f, -1, 0))", std::nullopt);
  testMatcher("x.f + y.f", std::nullopt);
}

} // namespace
} // namespace facebook::velox::functions::prestosql
