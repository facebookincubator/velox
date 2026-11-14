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
#include <optional>

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/lib/Sequence.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::functions::test;

namespace facebook::velox::functions::sparksql::test {
using namespace facebook::velox::test;

class SequenceTest : public SparkFunctionBaseTest {};

TEST_F(SequenceTest, readsMaxElementsOncePerApply) {
  class CountingConfig final : public config::IConfig {
   public:
    std::string maxElements = "3";
    mutable int32_t reads = 0;
    bool failWithSystemError = false;
    bool failWithUserError = false;

    std::unordered_map<std::string, std::string> rawConfigsCopy()
        const override {
      return {
          {core::QueryConfig::kMaxElementsSizeInRepeatAndSequence,
           maxElements}};
    }

   private:
    std::optional<std::string> access(const std::string& key) const override {
      if (key == core::QueryConfig::kMaxElementsSizeInRepeatAndSequence) {
        ++reads;
        if (failWithSystemError) {
          VELOX_FAIL("Config provider failed");
        }
        if (failWithUserError) {
          VELOX_USER_FAIL("Config provider user failure");
        }
        return maxElements;
      }
      return std::nullopt;
    }
  };

  auto config = std::make_shared<CountingConfig>();
  auto queryCtx = core::QueryCtx::create(
      nullptr, core::QueryConfig(core::QueryConfig::ConfigTag{}, config));
  core::ExecCtx execCtx(pool_.get(), queryCtx.get());
  SequenceFunction<int64_t, int64_t> function;
  std::vector<VectorPtr> args{
      makeFlatVector<int64_t>({1, 10, -2}),
      makeFlatVector<int64_t>({3, 11, 0})};
  VectorPtr result;

  exec::EvalCtx firstContext(&execCtx);
  function.apply(
      SelectivityVector(3), args, ARRAY(BIGINT()), firstContext, result);
  EXPECT_EQ(config->reads, 1);
  assertEqualVectors(
      makeArrayVector<int64_t>({{1, 2, 3}, {10, 11}, {-2, -1, 0}}), result);

  config->maxElements = "4";
  args[1] = makeFlatVector<int64_t>({4, 12, 1});
  result.reset();
  exec::EvalCtx nextContext(&execCtx);
  function.apply(
      SelectivityVector(3), args, ARRAY(BIGINT()), nextContext, result);
  EXPECT_EQ(config->reads, 2);
  assertEqualVectors(
      makeArrayVector<int64_t>({{1, 2, 3, 4}, {10, 11, 12}, {-2, -1, 0, 1}}),
      result);

  auto input = makeRowVector(args);
  auto evaluateWithConfig = [&](const std::string& expression,
                                const SelectivityVector& rows =
                                    SelectivityVector(3)) {
    exec::ExprSet expressions(
        {makeTypedExpr(expression, asRowType(input->type()))}, &execCtx);
    exec::EvalCtx context(&execCtx, &expressions, input.get());
    std::vector<VectorPtr> results(1);
    expressions.eval(rows, context, results);
    return results[0];
  };

  SelectivityVector selected(3, false);
  selected.setValid(0, true);
  selected.setValid(2, true);
  selected.updateBounds();
  for (bool failWithUserError : {false, true}) {
    SCOPED_TRACE(failWithUserError);
    config->failWithUserError = failWithUserError;
    config->maxElements = failWithUserError ? "4" : "invalid";
    config->reads = 0;
    assertEqualVectors(
        makeNullableArrayVector<int64_t>(
            {std::nullopt, std::nullopt, std::nullopt}),
        evaluateWithConfig("try(sequence(c0, c1))"));
    EXPECT_EQ(config->reads, 1);

    config->reads = 0;
    if (failWithUserError) {
      VELOX_ASSERT_USER_THROW(
          evaluateWithConfig("sequence(c0, c1)"),
          "Config provider user failure");
    } else {
      EXPECT_THROW(evaluateWithConfig("sequence(c0, c1)"), VeloxException);
    }
    EXPECT_EQ(config->reads, 1);

    config->reads = 0;
    result.reset();
    evaluateWithConfig("sequence(c0, c1)", SelectivityVector(3, false));
    EXPECT_EQ(config->reads, 0);

    exec::EvalCtx errorContext(&execCtx);
    *errorContext.mutableThrowOnError() = false;
    function.apply(selected, args, ARRAY(BIGINT()), errorContext, result);
    EXPECT_EQ(config->reads, 1);
    ASSERT_NE(errorContext.errors(), nullptr);
    EXPECT_TRUE(errorContext.errors()->hasErrorAt(0));
    EXPECT_FALSE(errorContext.errors()->hasErrorAt(1));
    EXPECT_TRUE(errorContext.errors()->hasErrorAt(2));
  }

  config->failWithUserError = false;
  config->reads = 0;
  config->failWithSystemError = true;
  EXPECT_THROW(evaluateWithConfig("try(sequence(c0, c1))"), VeloxRuntimeError);
  EXPECT_EQ(config->reads, 1);
}

TEST_F(SequenceTest, maxElementsErrorsRemainPerRow) {
  auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 10, -2}),
      makeFlatVector<int64_t>({4, 11, 0}),
  });
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxElementsSizeInRepeatAndSequence, "3"}});
  assertEqualVectors(
      makeNullableArrayVector<int64_t>(
          {std::nullopt, {{10, 11}}, {{-2, -1, 0}}}),
      evaluate("try(sequence(c0, c1))", input));
  VELOX_ASSERT_THROW(
      evaluate("sequence(c0, c1)", input),
      "result of sequence function must not have more than 3 entries");

  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxElementsSizeInRepeatAndSequence, "invalid"}});
  assertEqualVectors(
      makeNullableArrayVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt}),
      evaluate("try(sequence(c0, c1))", input));
}

TEST_F(SequenceTest, configErrorWithReusedResult) {
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxElementsSizeInRepeatAndSequence, "invalid"}});
  core::ExecCtx execCtx(pool_.get(), queryCtx_.get());
  exec::ExprSet expressions(
      {makeTypedExpr(
          "try(sequence(c0, c1))", ROW({"c0", "c1"}, {BIGINT(), BIGINT()}))},
      &execCtx);
  std::vector<VectorPtr> results(1);
  for (vector_size_t size : {1, 3}) {
    SCOPED_TRACE(size);
    auto input = makeRowVector({
        makeFlatVector<int64_t>(size, [](auto row) { return row; }),
        makeFlatVector<int64_t>(size, [](auto row) { return row + 3; }),
    });
    exec::EvalCtx context(&execCtx, &expressions, input.get());
    expressions.eval(SelectivityVector(size), context, results);
    ASSERT_NE(results[0], nullptr);
    ASSERT_EQ(results[0]->size(), size);
    for (vector_size_t row = 0; row < size; ++row) {
      EXPECT_TRUE(results[0]->isNullAt(row));
    }
  }
}

TEST_F(SequenceTest, configErrorInConditional) {
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxElementsSizeInRepeatAndSequence, "invalid"}});
  auto expected = makeNullableArrayVector<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  for (vector_size_t nullRow = 0; nullRow < 3; ++nullRow) {
    SCOPED_TRACE(nullRow);
    auto input = makeRowVector({
        makeFlatVector<bool>(3, [nullRow](auto row) { return row == nullRow; }),
        makeFlatVector<int64_t>({1, 10, -2}),
        makeFlatVector<int64_t>({4, 13, 1}),
    });
    for (const auto& expression :
         {"if(c0, cast(null as bigint[]), try(sequence(c1, c2)))",
          "case when c0 then cast(null as bigint[]) "
          "else try(sequence(c1, c2)) end"}) {
      SCOPED_TRACE(expression);
      assertEqualVectors(expected, evaluate(expression, input));
    }
  }
}

TEST_F(SequenceTest, configErrorPreservesUnselectedRows) {
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxElementsSizeInRepeatAndSequence, "invalid"}});
  auto arrays = makeArrayVector<int64_t>({{91}, {92}, {93}});
  auto input = makeRowVector({
      makeFlatVector<bool>({false, true, false}),
      makeFlatVector<int64_t>({1, 10, -2}),
      makeFlatVector<int64_t>({4, 13, 1}),
      arrays,
  });
  assertEqualVectors(
      makeNullableArrayVector<int64_t>({std::nullopt, {{92}}, std::nullopt}),
      evaluate("if(c0, c3, try(sequence(c1, c2)))", input));
  assertEqualVectors(makeArrayVector<int64_t>({{91}, {92}, {93}}), arrays);
}

TEST_F(SequenceTest, ascending) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({1, 3}),
          makeFlatVector<int64_t>({5, 6}),
      }));
  auto expected = makeArrayVector<int64_t>({{1, 2, 3, 4, 5}, {3, 4, 5, 6}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, descending) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({5}),
          makeFlatVector<int64_t>({1}),
      }));
  auto expected = makeArrayVector<int64_t>({{5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, withStep) {
  auto result = evaluate(
      "sequence(c0, c1, c2)",
      makeRowVector({
          makeFlatVector<int64_t>({1, 5}),
          makeFlatVector<int64_t>({10, 1}),
          makeFlatVector<int64_t>({3, -2}),
      }));
  auto expected = makeArrayVector<int64_t>({{1, 4, 7, 10}, {5, 3, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, singleElement) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int64_t>({5}),
          makeFlatVector<int64_t>({5}),
      }));
  auto expected = makeArrayVector<int64_t>({{5}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, integerType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int32_t>({1, 5}),
          makeFlatVector<int32_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int32_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, integerWithStep) {
  auto result = evaluate(
      "sequence(c0, c1, c2)",
      makeRowVector({
          makeFlatVector<int32_t>({1}),
          makeFlatVector<int32_t>({9}),
          makeFlatVector<int32_t>({2}),
      }));
  auto expected = makeArrayVector<int32_t>({{1, 3, 5, 7, 9}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, smallintType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int16_t>({1, 5}),
          makeFlatVector<int16_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int16_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, tinyintType) {
  auto result = evaluate(
      "sequence(c0, c1)",
      makeRowVector({
          makeFlatVector<int8_t>({1, 5}),
          makeFlatVector<int8_t>({5, 1}),
      }));
  auto expected = makeArrayVector<int8_t>({{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, nullInputs) {
  auto start = makeNullableFlatVector<int64_t>({std::nullopt, 1, 1});
  auto stop = makeNullableFlatVector<int64_t>({5, std::nullopt, 3});
  auto result = evaluate("sequence(c0, c1)", makeRowVector({start, stop}));

  auto expected = makeNullableArrayVector<int64_t>(
      {std::nullopt,
       std::nullopt,
       std::optional<std::vector<std::optional<int64_t>>>({{1, 2, 3}})});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, nullStep) {
  auto start = makeNullableFlatVector<int64_t>({1});
  auto stop = makeNullableFlatVector<int64_t>({5});
  auto step = makeNullableFlatVector<int64_t>({std::nullopt});
  auto result =
      evaluate("sequence(c0, c1, c2)", makeRowVector({start, stop, step}));

  auto expected = makeNullableArrayVector<int64_t>(
      {std::optional<std::vector<std::optional<int64_t>>>(std::nullopt)});
  assertEqualVectors(expected, result);
}

TEST_F(SequenceTest, stepZeroError) {
  VELOX_ASSERT_THROW(
      evaluate(
          "sequence(c0, c1, c2)",
          makeRowVector({
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({5}),
              makeFlatVector<int64_t>({0}),
          })),
      "step must not be zero");
}

TEST_F(SequenceTest, wrongDirectionError) {
  VELOX_ASSERT_THROW(
      evaluate(
          "sequence(c0, c1, c2)",
          makeRowVector({
              makeFlatVector<int64_t>({1}),
              makeFlatVector<int64_t>({5}),
              makeFlatVector<int64_t>({-1}),
          })),
      "sequence stop value should be greater than or equal to start value if "
      "step is greater than zero otherwise stop should be less than or equal "
      "to start");
}

} // namespace facebook::velox::functions::sparksql::test
