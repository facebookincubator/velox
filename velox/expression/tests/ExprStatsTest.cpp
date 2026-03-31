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
#include "velox/core/Expressions.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

/// A deliberately expensive scalar function used to test adaptive CPU sampling.
/// The loop makes the per-row cost high enough that clock_gettime overhead is
/// negligible. volatile prevents the compiler from optimizing away the loop.
template <typename T>
struct SlowAddFunction {
  template <typename TInput>
  void call(TInput& result, const TInput& a, const TInput& b) {
    result = a + b;
    volatile TInput sink = result;
    for (int i = 0; i < 50'000; ++i) {
      sink = sink + 1;
    }
  }
};

class ExprStatsTest : public functions::test::FunctionBaseTest {
 protected:
  void SetUp() override {
    functions::prestosql::registerAllScalarFunctions();
    parse::registerTypeResolver();
    // Enable CPU usage tracking.
    queryCtx_->testingOverrideConfigUnsafe({
        {core::QueryConfig::kExprTrackCpuUsage, "true"},
    });
  }
};

TEST_F(ExprStatsTest, printWithStats) {
  vector_size_t size = 1'024;

  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());
  {
    auto exprSet =
        compileExpressions({"(c0 + 3) * c1", "(c0 + c1) % 2 = 0"}, rowType);

    // Check stats before evaluation.
    ASSERT_EQ(
        exec::printExprWithStats(*exprSet),
        "multiply [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#1]\n"
        "   plus [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#2]\n"
        "      cast(c0 as BIGINT) [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#3]\n"
        "         c0 [cpu time: 0ns, rows: 0, batches: 0] -> INTEGER [#4]\n"
        "      3:BIGINT [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#5]\n"
        "   cast(c1 as BIGINT) [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#6]\n"
        "      c1 [cpu time: 0ns, rows: 0, batches: 0] -> INTEGER [#7]\n"
        "\n"
        "eq [cpu time: 0ns, rows: 0, batches: 0] -> BOOLEAN [#8]\n"
        "   mod [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#9]\n"
        "      cast(plus as BIGINT) [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#10]\n"
        "         plus [cpu time: 0ns, rows: 0, batches: 0] -> INTEGER [#11]\n"
        "            c0 -> INTEGER [CSE #4]\n"
        "            c1 -> INTEGER [CSE #7]\n"
        "      2:BIGINT [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#12]\n"
        "   0:BIGINT [cpu time: 0ns, rows: 0, batches: 0] -> BIGINT [#13]\n");

    evaluate(*exprSet, data);

    // Check stats after evaluation.
    ASSERT_THAT(
        exec::printExprWithStats(*exprSet),
        ::testing::MatchesRegex(
            "multiply .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#1.\n"
            "   plus .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#2.\n"
            "      cast.c0 as BIGINT. .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#3.\n"
            "         c0 .cpu time: 0ns, rows: 2048, batches: 2. -> INTEGER .#4.\n"
            "      3:BIGINT .cpu time: 0ns, rows: 1024, batches: 1. -> BIGINT .#5.\n"
            "   cast.c1 as BIGINT. .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#6.\n"
            "      c1 .cpu time: 0ns, rows: 2048, batches: 2. -> INTEGER .#7.\n"
            "\n"
            "eq .cpu time: .+, rows: 1024, batches: 1. -> BOOLEAN .#8.\n"
            "   mod .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#9.\n"
            "      cast.plus as BIGINT. .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#10.\n"
            "         plus .cpu time: .+, rows: 1024, batches: 1. -> INTEGER .#11.\n"
            "            c0 -> INTEGER .CSE #4.\n"
            "            c1 -> INTEGER .CSE #7.\n"
            "      2:BIGINT .cpu time: 0ns, rows: 1024, batches: 1. -> BIGINT .#12.\n"
            "   0:BIGINT .cpu time: 0ns, rows: 1024, batches: 1. -> BIGINT .#13.\n"));
  }

  // Verify that common sub-expressions are identified properly.
  {
    auto exprSet =
        compileExpressions({"(c0 + c1) % 5", "(c0 + c1) % 3"}, rowType);
    evaluate(*exprSet, data);
    ASSERT_THAT(
        exec::printExprWithStats(*exprSet),
        ::testing::MatchesRegex(
            "mod .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#1.\n"
            "   cast.plus as BIGINT. .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#2.\n"
            "      plus .cpu time: .+, rows: 1024, batches: 1. -> INTEGER .#3.\n"
            "         c0 .cpu time: 0ns, rows: 1024, batches: 1. -> INTEGER .#4.\n"
            "         c1 .cpu time: 0ns, rows: 1024, batches: 1. -> INTEGER .#5.\n"
            "   5:BIGINT .cpu time: 0ns, rows: 1024, batches: 1. -> BIGINT .#6.\n"
            "\n"
            "mod .cpu time: .+, rows: 1024, batches: 1. -> BIGINT .#7.\n"
            "   cast..plus.c0, c1.. as BIGINT. -> BIGINT .CSE #2.\n"
            "   3:BIGINT .cpu time: 0ns, rows: 1024, batches: 1. -> BIGINT .#8.\n"));
  }

  // Use dictionary encoding to repeat each row 5 times.
  auto indices = makeIndices(size, [](auto row) { return row / 5; });
  data = makeRowVector({
      wrapInDictionary(indices, size, data->childAt(0)),
      wrapInDictionary(indices, size, data->childAt(1)),
  });

  {
    auto exprSet =
        compileExpressions({"(c0 + 3) * c1", "(c0 + c1) % 2 = 0"}, rowType);
    evaluate(*exprSet, data);

    ASSERT_THAT(
        exec::printExprWithStats(*exprSet),
        ::testing::MatchesRegex(
            "multiply .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#1.\n"
            "   plus .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#2.\n"
            "      cast.c0 as BIGINT. .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#3.\n"
            "         c0 .cpu time: 0ns, rows: 410, batches: 2. -> INTEGER .#4.\n"
            "      3:BIGINT .cpu time: 0ns, rows: 205, batches: 1. -> BIGINT .#5.\n"
            "   cast.c1 as BIGINT. .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#6.\n"
            "      c1 .cpu time: 0ns, rows: 410, batches: 2. -> INTEGER .#7.\n"
            "\n"
            "eq .cpu time: .+, rows: 205, batches: 1. -> BOOLEAN .#8.\n"
            "   mod .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#9.\n"
            "      cast.plus as BIGINT. .cpu time: .+, rows: 205, batches: 1. -> BIGINT .#10.\n"
            "         plus .cpu time: .+, rows: 205, batches: 1. -> INTEGER .#11.\n"
            "            c0 -> INTEGER .CSE #4.\n"
            "            c1 -> INTEGER .CSE #7.\n"
            "      2:BIGINT .cpu time: 0ns, rows: 205, batches: 1. -> BIGINT .#12.\n"
            "   0:BIGINT .cpu time: 0ns, rows: 205, batches: 1. -> BIGINT .#13.\n"));
  }
}

struct Event {
  std::string uuid;
  std::unordered_map<std::string, exec::ExprStats> stats;
  std::vector<std::string> sqls;
};

class TestListener : public exec::ExprSetListener {
 public:
  explicit TestListener(std::vector<Event>& events)
      : events_{events}, exceptionCount_{0} {}

  void onCompletion(
      const std::string& uuid,
      const exec::ExprSetCompletionEvent& event) override {
    events_.push_back({uuid, event.stats, event.sqls});
  }

  void onError(vector_size_t numRows, const std::string& /*queryId*/) override {
    exceptionCount_ += numRows;
  }

  int exceptionCount() const {
    return exceptionCount_;
  }

  void reset() {
    exceptionCount_ = 0;
    events_.clear();
  }

 private:
  std::vector<Event>& events_;
  int exceptionCount_;
};

TEST_F(ExprStatsTest, listener) {
  vector_size_t size = 1'024;

  // Register a listener to receive stats on ExprSet destruction.
  std::vector<Event> events;
  auto listener = std::make_shared<TestListener>(events);
  ASSERT_TRUE(exec::registerExprSetListener(listener));
  ASSERT_FALSE(exec::registerExprSetListener(listener));

  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
  });

  // Evaluate a couple of expressions and sanity check the stats received by the
  // listener.
  auto rowType = asRowType(data->type());
  {
    auto exprSet =
        compileExpressions({"(c0 + 3) * c1", "(c0 + c1) % 2 = 0"}, rowType);
    evaluate(*exprSet, data);
  }
  ASSERT_EQ(1, events.size());
  auto stats = events.back().stats;
  auto sqls = events.back().sqls;

  ASSERT_EQ(2, sqls.size());
  ASSERT_EQ(
      "\"multiply\"(\"plus\"(cast((\"c0\") as BIGINT), '3'::BIGINT), cast((\"c1\") as BIGINT))",
      sqls.front());
  ASSERT_EQ(
      "\"eq\"(\"mod\"(cast((\"plus\"(\"c0\", \"c1\")) as BIGINT), '2'::BIGINT), '0'::BIGINT)",
      sqls.back());

  ASSERT_EQ(2, stats.at("plus").numProcessedVectors);
  ASSERT_EQ(1024 * 2, stats.at("plus").numProcessedRows);

  ASSERT_EQ(1, stats.at("multiply").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("multiply").numProcessedRows);

  ASSERT_EQ(1, stats.at("mod").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("mod").numProcessedRows);

  for (const auto& name : {"plus", "multiply", "mod"}) {
    ASSERT_GT(stats.at(name).timing.cpuNanos, 0);
  }

  // Evaluate the same expressions twice and verify that stats received by the
  // listener are "doubled".
  {
    auto exprSet =
        compileExpressions({"(c0 + 3) * c1", "(c0 + c1) % 2 = 0"}, rowType);
    evaluate(*exprSet, data);
    evaluate(*exprSet, data);
  }
  ASSERT_EQ(2, events.size());
  stats = events.back().stats;

  ASSERT_EQ(4, stats.at("plus").numProcessedVectors);
  ASSERT_EQ(1024 * 2 * 2, stats.at("plus").numProcessedRows);

  ASSERT_EQ(2, stats.at("multiply").numProcessedVectors);
  ASSERT_EQ(1024 * 2, stats.at("multiply").numProcessedRows);

  ASSERT_EQ(2, stats.at("mod").numProcessedVectors);
  ASSERT_EQ(1024 * 2, stats.at("mod").numProcessedRows);
  for (const auto& name : {"plus", "multiply", "mod"}) {
    ASSERT_GT(stats.at(name).timing.cpuNanos, 0);
  }

  ASSERT_NE(events[0].uuid, events[1].uuid);

  // Evaluate an expression with CTE and verify no double accounting.
  {
    auto exprSet =
        compileExpressions({"(c0 + c1) % 5", "pow(c0 + c1, 2)"}, rowType);
    evaluate(*exprSet, data);
  }
  ASSERT_EQ(3, events.size());
  stats = events.back().stats;
  ASSERT_EQ(1024, stats.at("plus").numProcessedRows);
  ASSERT_EQ(1024, stats.at("mod").numProcessedRows);
  ASSERT_EQ(1024, stats.at("pow").numProcessedRows);
  for (const auto& name : {"plus", "mod", "pow"}) {
    ASSERT_EQ(1, stats.at(name).numProcessedVectors);
  }

  // Unregister the listener, evaluate expressions again and verify the listener
  // wasn't invoked.
  ASSERT_TRUE(exec::unregisterExprSetListener(listener));
  ASSERT_FALSE(exec::unregisterExprSetListener(listener));

  {
    auto exprSet =
        compileExpressions({"(c0 + 3) * c1", "(c0 + c1) % 2 = 0"}, rowType);
    evaluate(*exprSet, data);
  }
  ASSERT_EQ(3, events.size());
}

TEST_F(ExprStatsTest, specialForms) {
  vector_size_t size = 1'024;

  // Register a listener to receive stats on ExprSet destruction.
  std::vector<Event> events;
  auto listener = std::make_shared<TestListener>(events);
  ASSERT_TRUE(exec::registerExprSetListener(listener));

  auto data = makeRowVector({
      makeFlatVector<int32_t>(size, [](auto row) { return row; }),
      makeFlatVector<int32_t>(size, [](auto row) { return row % 7; }),
  });

  // AND.
  evaluate("c0 > 5 and c1 > 5", data);
  ASSERT_EQ(1, events.size());
  auto stats = events.back().stats;

  ASSERT_EQ(1, stats.at("and").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("and").numProcessedRows);

  // OR.
  events.clear();
  evaluate("c0 > 5 or c1 > 5", data);
  ASSERT_EQ(1, events.size());
  stats = events.back().stats;

  ASSERT_EQ(1, stats.at("or").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("or").numProcessedRows);

  // TRY.
  events.clear();
  evaluate("try(c0 / c1)", data);
  ASSERT_EQ(1, events.size());
  stats = events.back().stats;

  ASSERT_EQ(1, stats.at("try").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("try").numProcessedRows);

  // COALESCE.
  events.clear();
  evaluate("coalesce(c0, c1)", data);
  ASSERT_EQ(1, events.size());
  stats = events.back().stats;

  ASSERT_EQ(1, stats.at("coalesce").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("coalesce").numProcessedRows);

  // SWITCH.
  events.clear();
  evaluate("case c0 when 7 then 1 when 11 then 2 else 0 end", data);
  ASSERT_EQ(1, events.size());
  stats = events.back().stats;

  ASSERT_EQ(1, stats.at("switch").numProcessedVectors);
  ASSERT_EQ(1024, stats.at("switch").numProcessedRows);

  ASSERT_TRUE(exec::unregisterExprSetListener(listener));
}

TEST_F(ExprStatsTest, errorLog) {
  // Register a listener to log exceptions.
  std::vector<Event> events;
  auto listener = std::make_shared<TestListener>(events);
  ASSERT_TRUE(exec::registerExprSetListener(listener));

  auto data = makeRowVector(
      {makeNullableFlatVector<StringView>(
           {"12"_sv, "1a"_sv, "34"_sv, ""_sv, std::nullopt, " 1"_sv}),
       makeNullableFlatVector<StringView>(
           {"12.3a"_sv, "ab"_sv, "3.4"_sv, "5.6"_sv, std::nullopt, "1.1a"_sv}),
       makeNullableFlatVector<StringView>(
           {"12"_sv, "34"_sv, "0"_sv, "78"_sv, "0"_sv, "0"_sv})});

  auto rowType = asRowType(data->type());
  auto exprSet = compileExpressions({"try(cast(c0 as integer))"}, rowType);

  evaluate(*exprSet, data);

  // Expect errors at rows 2 and 4.
  ASSERT_EQ(2, listener->exceptionCount());

  // Test with multiple try expressions. Expect errors at rows 1, 2, 4, and 6.
  // The second row in c1 does not cause an additional error because the
  // corresponding row in c0 already triggered an error first that caused this
  // row to be nulled out.
  listener->reset();
  exprSet = compileExpressions(
      {"try(cast(c0 as integer)) + try(cast(c1 as double))"}, rowType);

  evaluate(*exprSet, data);
  ASSERT_EQ(4, listener->exceptionCount());

  // Test with nested try expressions. Expect errors at rows 2, 3, 4, and 6. Row
  // 5 in c2 does not cause an error because the corresponding row in c0 is
  // null, so this row is not evaluated for the division.
  listener->reset();
  exprSet = compileExpressions(
      {"try(try(cast(c0 as integer)) / cast(c2 as integer))"}, rowType);

  evaluate(*exprSet, data);
  ASSERT_EQ(4, listener->exceptionCount());

  // Test with no error.
  listener->reset();
  exprSet = compileExpressions({"try(cast(c2 as integer))"}, rowType);

  evaluate(*exprSet, data);
  ASSERT_EQ(0, listener->exceptionCount());

  ASSERT_TRUE(exec::unregisterExprSetListener(listener));
}

TEST_F(ExprStatsTest, complexConstants) {
  // Expressions with constants of types not expressible in SQL should use
  // '__complex_constant(c#)' pseudo functions.

  std::vector<Event> events;
  auto listener = std::make_shared<TestListener>(events);
  ASSERT_TRUE(exec::registerExprSetListener(listener));

  std::vector<core::TypedExprPtr> expressions = {
      std::make_shared<const core::ConstantTypedExpr>(
          makeConstant("12"_sv, 1, VARBINARY()))};
  {
    auto exprSet =
        std::make_unique<exec::ExprSet>(std::move(expressions), &execCtx_);
    evaluate(*exprSet, makeRowVector(ROW({}), 10));
  }

  ASSERT_EQ(1, events.size());
  ASSERT_EQ(0, listener->exceptionCount());

  ASSERT_EQ(1, events[0].sqls.size());
  ASSERT_EQ("__complex_constant(c0)", events[0].sqls[0]);

  ASSERT_TRUE(exec::unregisterExprSetListener(listener));
}
TEST_F(ExprStatsTest, selectiveCpuTrackingForUdfs) {
  // Test selective CPU tracking for specific UDFs using
  // kExprTrackCpuUsageUdfs config.
  vector_size_t inputSize = 10;

  auto data = makeRowVector({
      makeFlatVector<int32_t>(inputSize, [](auto row) { return row; }),
      makeFlatVector<int32_t>(inputSize, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());

  // Helper to test selective CPU tracking with specified configuration.
  auto testCpuTracking =
      [&](bool enableGlobalTracking,
          const std::string& funcList,
          const std::vector<std::string>& expressions,
          const std::unordered_set<std::string>& expectedFuncsWithNonZeroCpu) {
        queryCtx_->testingOverrideConfigUnsafe({
            {core::QueryConfig::kExprTrackCpuUsage,
             enableGlobalTracking ? "true" : "false"},
            {core::QueryConfig::kExprTrackCpuUsageForFunctions, funcList},
        });

        std::vector<Event> events;
        auto listener = std::make_shared<TestListener>(events);
        ASSERT_TRUE(exec::registerExprSetListener(listener));

        {
          auto exprSet = compileExpressions(expressions, rowType);
          evaluate(*exprSet, data);
        }

        // Ensure listener is unregistered before accessing events.
        ASSERT_TRUE(exec::unregisterExprSetListener(listener));

        ASSERT_EQ(1, events.size());
        auto stats = events.back().stats;

        // Verify selective CPU tracking for the specified functions.
        for (const auto& [funcName, funcStats] : stats) {
          if (expectedFuncsWithNonZeroCpu.count(funcName)) {
            ASSERT_GT(funcStats.timing.cpuNanos, 0) << funcName;
          } else {
            ASSERT_EQ(funcStats.timing.cpuNanos, 0) << funcName;
          }
        }
      };

  // Test with empty config - should work same as before.
  testCpuTracking(false, "", {"c0 + c1", "c0 * c1"}, {});

  // Test with single function name.
  testCpuTracking(false, "plus", {"c0 + c1", "c0 * c1"}, {"plus"});

  // Test with multiple function names.
  testCpuTracking(
      false, "plus,mod", {"c0 + c1", "c0 * c1", "c0 % 5"}, {"plus", "mod"});

  // Test with mixed case function names (should be normalized to lowercase).
  testCpuTracking(
      false,
      "PLUS,Multiply",
      {"c0 + c1", "c0 * c1", "c0 % 5"},
      {"plus", "multiply"});

  // Test with extra commas and empty entries (should be handled gracefully).
  testCpuTracking(
      false, "plus,,multiply,", {"c0 + c1", "c0 * c1"}, {"plus", "multiply"});

  // Test with larger expression trees.
  testCpuTracking(
      false,
      "plus,pow",
      {"(c0 + c1) * c1", "pow(c0 + c1, 2)"},
      {"plus", "pow"});

  // Test with global CPU tracking enabled.
  testCpuTracking(
      true,
      "plus,pow",
      {"(c0 + c1) * c1", "pow(c0 + c1, 2)"},
      {"plus", "pow", "multiply", "cast"});
}

TEST_F(ExprStatsTest, adaptiveCpuSampling) {
  // Test that adaptive CPU sampling calibrates per-function and produces
  // timing data after calibration completes.
  constexpr vector_size_t kInputSize{100};
  // Need 1 warmup + 5 calibration + enough steady-state batches.
  constexpr int kNumBatches{50};

  auto data = makeRowVector({
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row; }),
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());

  // Enable adaptive CPU sampling with 5% max overhead threshold.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kExprTrackCpuUsage, "false"},
      {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
      {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "5.0"},
  });

  auto exprSet = compileExpressions({"c0 + c1", "c0 * c1"}, rowType);
  SelectivityVector rows(data->size());

  for (int i = 0; i < kNumBatches; ++i) {
    exec::EvalCtx evalCtx(&execCtx_, exprSet.get(), data.get());
    std::vector<VectorPtr> results(exprSet->size());
    exprSet->eval(rows, evalCtx, results);
  }

  auto stats = exprSet->stats();

  // All batches should be counted in numProcessedVectors.
  ASSERT_TRUE(stats.count("plus"));
  ASSERT_EQ(stats["plus"].numProcessedVectors, kNumBatches);

  // After calibration (1 warmup + 5 calibration), the function should have
  // timing data from steady-state batches (either always-track or sampled).
  // With 50 batches and 6 calibration, 44 steady-state batches remain.
  // Even with sampling, stats adjustment extrapolates timing.count.
  ASSERT_GT(stats["plus"].timing.count, 0u);

  // If the function was classified as needing sampling (likely for cheap
  // builtins), timing.count should be less than numProcessedVectors.
  // If classified as always-track, timing.count == numProcessedVectors.
  // Either way, stats should have been adjusted, so cpuNanos > 0.
  ASSERT_GT(stats["plus"].timing.cpuNanos, 0u);
}

TEST_F(ExprStatsTest, adaptiveCpuSamplingDisabledByDefault) {
  constexpr vector_size_t kInputSize{100};
  constexpr int kNumBatches{10};

  auto data = makeRowVector({
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row; }),
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());

  // Disable everything — no tracking, no sampling, no adaptive.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kExprTrackCpuUsage, "false"},
  });

  auto exprSet = compileExpressions({"c0 + c1"}, rowType);
  SelectivityVector rows(data->size());

  for (int i = 0; i < kNumBatches; ++i) {
    exec::EvalCtx evalCtx(&execCtx_, exprSet.get(), data.get());
    std::vector<VectorPtr> results(exprSet->size());
    exprSet->eval(rows, evalCtx, results);
  }

  auto stats = exprSet->stats();

  // All batches processed but no CPU timing recorded (adaptive is off).
  ASSERT_TRUE(stats.count("plus"));
  ASSERT_EQ(stats["plus"].numProcessedVectors, kNumBatches);
  ASSERT_EQ(stats["plus"].timing.count, 0u);
  ASSERT_EQ(stats["plus"].timing.cpuNanos, 0u);
}

TEST_F(ExprStatsTest, adaptiveCpuSamplingStatsAdjustment) {
  // Test that stats are properly adjusted (extrapolated) for functions
  // in sampling mode.
  constexpr vector_size_t kInputSize{100};
  // Need enough batches: 6 calibration + enough steady-state for sampling.
  constexpr int kNumBatches{50};

  auto data = makeRowVector({
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row; }),
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());

  // Use 5% overhead threshold. Built-in plus (~13% overhead in ASAN) will be
  // classified as needing sampling, and stats should be extrapolated.
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kExprTrackCpuUsage, "false"},
      {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
      {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "5.0"},
  });

  std::vector<Event> events;
  auto listener = std::make_shared<TestListener>(events);
  ASSERT_TRUE(exec::registerExprSetListener(listener));

  {
    auto exprSet = compileExpressions({"c0 + c1"}, rowType);
    SelectivityVector rows(data->size());

    for (int i = 0; i < kNumBatches; ++i) {
      exec::EvalCtx evalCtx(&execCtx_, exprSet.get(), data.get());
      std::vector<VectorPtr> results(exprSet->size());
      exprSet->eval(rows, evalCtx, results);
    }

    auto stats = exprSet->stats();
    ASSERT_TRUE(stats.count("plus"));

    // If function is being sampled, the adjusted stats should have
    // timing.count == numProcessedVectors (extrapolated to full count).
    if (stats["plus"].timing.count == stats["plus"].numProcessedVectors) {
      // Stats were adjusted. cpuNanos should be extrapolated.
      ASSERT_GT(stats["plus"].timing.cpuNanos, 0u);
    }
  }

  // Listener should also receive adjusted stats.
  ASSERT_EQ(1, events.size());
  auto listenerStats = events.back().stats;
  ASSERT_TRUE(listenerStats.count("plus"));
  ASSERT_GT(listenerStats["plus"].timing.cpuNanos, 0u);

  ASSERT_TRUE(exec::unregisterExprSetListener(listener));
}

TEST_F(ExprStatsTest, adaptiveCpuSamplingPerFunctionRates) {
  // Test that adaptive sampling assigns different rates to cheap vs expensive
  // functions. A cheap built-in (plus) should get sampling while an expensive
  // UDF (slow_add) should always track.
  //
  // Use a small batch so that clock_gettime overhead is significant relative
  // to the cheap function (plus) but negligible relative to the expensive one
  // (slow_add with 50k volatile iterations per row).
  constexpr vector_size_t kInputSize{10};
  constexpr int kNumBatches{2'000};

  registerFunction<SlowAddFunction, int32_t, int32_t, int32_t>({"slow_add"});

  auto data = makeRowVector({
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row; }),
      makeFlatVector<int32_t>(kInputSize, [](auto row) { return row % 7; }),
  });

  auto rowType = asRowType(data->type());

  // Enable adaptive CPU sampling with 5% max overhead. For cheap functions
  // like plus on 10 rows, clock_gettime overhead is very significant (>>5%).
  // For slow_add (50k iterations/row), the overhead is negligible (<<5%).
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kExprTrackCpuUsage, "false"},
      {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
      {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "5.0"},
  });

  auto exprSet = compileExpressions({"c0 + c1", "slow_add(c0, c1)"}, rowType);
  SelectivityVector rows(data->size());

  for (int i = 0; i < kNumBatches; ++i) {
    exec::EvalCtx evalCtx(&execCtx_, exprSet.get(), data.get());
    std::vector<VectorPtr> results(exprSet->size());
    exprSet->eval(rows, evalCtx, results);
  }

  // Traverse the expression tree to find the plus and slow_add Expr nodes.
  const exec::Expr* plusExpr = nullptr;
  const exec::Expr* slowAddExpr = nullptr;

  std::function<void(const exec::Expr*)> findExprs =
      [&](const exec::Expr* expr) {
        if (expr->name() == "plus") {
          plusExpr = expr;
        } else if (expr->name() == "slow_add") {
          slowAddExpr = expr;
        }
        for (const auto& input : expr->inputs()) {
          findExprs(input.get());
        }
      };

  for (const auto& expr : exprSet->exprs()) {
    findExprs(expr.get());
  }

  ASSERT_NE(plusExpr, nullptr) << "Failed to find 'plus' expression";
  ASSERT_NE(slowAddExpr, nullptr) << "Failed to find 'slow_add' expression";

  // Cheap function (plus) should be in sampling mode with rate > 1.
  ASSERT_TRUE(plusExpr->isAdaptiveSampling())
      << "Expected cheap function 'plus' to be in sampling mode";
  ASSERT_GT(plusExpr->adaptiveSamplingRate(), 1u)
      << "Expected sampling rate > 1 for cheap function";

  // Expensive function (slow_add) should always track (not sampling).
  ASSERT_FALSE(slowAddExpr->isAdaptiveSampling())
      << "Expected expensive function 'slow_add' to always track";

  // Both functions should have timing data.
  auto stats = exprSet->stats();
  ASSERT_GT(stats["plus"].timing.cpuNanos, 0u);
  ASSERT_GT(stats["slow_add"].timing.cpuNanos, 0u);

  // slow_add is always-track after calibration. It won't have timing for the
  // warmup + calibration batches (1 + 5 = 6), but all post-calibration batches
  // should be tracked.
  constexpr uint64_t kCalibrationOverhead = 6; // 1 warmup + 5 calibration
  ASSERT_EQ(
      stats["slow_add"].timing.count,
      stats["slow_add"].numProcessedVectors - kCalibrationOverhead);

  // plus is in sampling mode. Stats should be adjusted (extrapolated) so
  // timing.count matches numProcessedVectors.
  ASSERT_EQ(stats["plus"].timing.count, stats["plus"].numProcessedVectors);
  ASSERT_GT(stats["plus"].timing.cpuNanos, 0u);

  // Verify the sampling rate for plus is reasonable (should be > 1).
  LOG(INFO) << "plus sampling rate: " << plusExpr->adaptiveSamplingRate()
            << ", slow_add sampling rate: "
            << slowAddExpr->adaptiveSamplingRate();
}
