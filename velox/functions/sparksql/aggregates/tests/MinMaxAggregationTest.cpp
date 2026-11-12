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
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

std::string min(const std::string& column) {
  return fmt::format("spark_min({})", column);
}

std::string max(const std::string& column) {
  return fmt::format("spark_max({})", column);
}

class MinMaxAggregationTest
    : public functions::aggregate::test::AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
  }

  std::vector<RowVectorPtr> fuzzData(const RowTypePtr& rowType) {
    VectorFuzzer::Options options;
    options.vectorSize = 1'000;
    options.nullRatio = 0.1;
    VectorFuzzer fuzzer(options, pool());
    std::vector<RowVectorPtr> vectors(10);
    for (auto i = 0; i < 10; ++i) {
      vectors[i] = fuzzer.fuzzInputRow(rowType);
    }
    return vectors;
  }

  void doTest(const TypePtr& inputType, bool testWithTableScan = true) {
    auto rowType = ROW({"c0", "c1", "mask"}, {BIGINT(), inputType, BOOLEAN()});
    auto vectors = fuzzData(rowType);
    createDuckDbTable(vectors);

    static const std::string c0 = "c0";
    static const std::string c1 = "c1";
    static const std::string a0 = "a0";

    // Global aggregation.
    AggregationTester(*this)
        .inputs(vectors)
        .aggregates({min(c1), max(c1)})
        .withCompanion()
        .assertResults("SELECT min(c1), max(c1) FROM tmp");

    // Group by aggregation.
    AggregationTester(*this)
        .inputs(vectors)
        .preAggregation(
            [&](auto& builder) { builder.project({"c0 % 10", "c1"}); })
        .groupBy({"p0"})
        .aggregates({min(c1), max(c1)})
        .assertResults("SELECT c0 % 10, min(c1), max(c1) FROM tmp GROUP BY 1");

    // Masked aggregations.
    auto minMaskedAgg = min(c1) + " filter (where mask)";
    AggregationTester(*this)
        .inputs(vectors)
        .aggregates({minMaskedAgg})
        .assertResults("SELECT min(c1) filter (where mask) FROM tmp");

    AggregationTester(*this)
        .inputs(vectors)
        .preAggregation(
            [&](auto& builder) { builder.project({"c0 % 10", "c1", "mask"}); })
        .groupBy({"p0"})
        .aggregates({minMaskedAgg})
        .assertResults(
            "SELECT c0 % 10, min(c1) filter (where mask) FROM tmp GROUP BY 1");

    auto maxMaskedAgg = max(c1) + " filter (where mask)";
    AggregationTester(*this)
        .inputs(vectors)
        .aggregates({maxMaskedAgg})
        .assertResults("SELECT max(c1) filter (where mask) FROM tmp");

    AggregationTester(*this)
        .inputs(vectors)
        .preAggregation(
            [&](auto& builder) { builder.project({"c0 % 10", "c1", "mask"}); })
        .groupBy({"p0"})
        .aggregates({maxMaskedAgg})
        .assertResults(
            "SELECT c0 % 10, max(c1) filter (where mask) FROM tmp GROUP BY 1");

    // Encodings: use filter to wrap aggregation inputs in a dictionary.
    AggregationTester(*this)
        .inputs(vectors)
        .preAggregation([&](auto& builder) {
          builder.filter("c0 % 2 = 0").project({"c0 % 11", "c1"});
        })
        .groupBy({"p0"})
        .aggregates({min(c1), max(c1)})
        .assertResults(
            "SELECT c0 % 11, min(c1), max(c1) FROM tmp WHERE c0 % 2 = 0 GROUP BY 1");

    AggregationTester(*this)
        .inputs(vectors)
        .preAggregation([&](auto& builder) { builder.filter("c0 % 2 = 0"); })
        .aggregates({min(c1), max(c1)})
        .assertResults("SELECT min(c1), max(c1) FROM tmp WHERE c0 % 2 = 0");
  }
};

TEST_F(MinMaxAggregationTest, tinyint) {
  doTest(TINYINT());
}

TEST_F(MinMaxAggregationTest, smallint) {
  doTest(SMALLINT());
}

TEST_F(MinMaxAggregationTest, integer) {
  doTest(INTEGER());
}

TEST_F(MinMaxAggregationTest, bigint) {
  doTest(BIGINT());
}

TEST_F(MinMaxAggregationTest, real) {
  doTest(REAL());
}

TEST_F(MinMaxAggregationTest, double) {
  doTest(DOUBLE());
}

TEST_F(MinMaxAggregationTest, varchar) {
  doTest(VARCHAR());
}

TEST_F(MinMaxAggregationTest, boolean) {
  doTest(BOOLEAN());
}

TEST_F(MinMaxAggregationTest, interval) {
  doTest(INTERVAL_DAY_TIME());
}

TEST_F(MinMaxAggregationTest, shortDecimal) {
  doTest(DECIMAL(18, 3), false);
}

TEST_F(MinMaxAggregationTest, longDecimal) {
  doTest(DECIMAL(20, 3), false);
}

TEST_F(MinMaxAggregationTest, timestamp) {
  auto rowType = ROW({"c0", "c1"}, {SMALLINT(), TIMESTAMP()});
  auto vectors = makeVectors(rowType, 1'000, 10);
  createDuckDbTable(vectors);

  AggregationTester(*this)
      .inputs(vectors)
      .aggregates({min("c1"), max("c1")})
      .assertResults(
          "SELECT date_trunc('microsecond', min(c1)), "
          "date_trunc('microsecond', max(c1)) FROM tmp");

  AggregationTester(*this)
      .inputs(vectors)
      .preAggregation(
          [&](auto& builder) { builder.project({"c0 % 17 as k", "c1"}); })
      .groupBy({"k"})
      .aggregates({min("c1"), max("c1")})
      .assertResults(
          "SELECT c0 % 17, date_trunc('microsecond', min(c1)), "
          "date_trunc('microsecond', max(c1)) FROM tmp GROUP BY 1");
}

TEST_F(MinMaxAggregationTest, timestampUtc) {
  const auto beforeEpoch = Timestamp::fromMicros(-1);
  const auto afterEpoch = Timestamp::fromMicros(1);
  const Timestamp earlier{1'704'067'200, 123'456'000};
  const Timestamp later{1'704'067'200, 123'457'000};
  // Keep the lower bound on a whole second to avoid overflow during truncation.
  const Timestamp earliest{
      std::numeric_limits<int64_t>::min() / Timestamp::kMicrosecondsInSecond,
      0,
  };
  const auto latest =
      Timestamp::fromMicros(std::numeric_limits<int64_t>::max());
  const std::vector<std::string> aggregates{min("c1"), max("c1")};

  std::vector<RowVectorPtr> inputs;
  RowVectorPtr expected;
  const auto assertResults = [&](auto& builder) {
    std::shared_ptr<exec::Task> task;
    auto actual = builder.copyResults(pool(), task);
    // Compare logical types explicitly, since value comparisons only
    // check the shared TypeKind::TIMESTAMP.
    EXPECT_TRUE(expected->type()->equivalent(*actual->type()))
        << actual->type()->toString();
    assertEqualResults({expected}, {actual});
    return task;
  };

  for (const auto* timeZone : {"UTC", "America/Los_Angeles", "Asia/Kolkata"}) {
    SCOPED_TRACE(timeZone);
    const auto tester = [&]() {
      return AggregationTester(*this)
          .config(core::QueryConfig::kSessionTimezone, timeZone)
          .config(core::QueryConfig::kAdjustTimestampToTimezone, "true");
    };

    {
      SCOPED_TRACE(
          "Microsecond truncation, range boundaries, and all-null groups");
      auto data = makeRowVector({
          makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2, 3, 3}),
          makeNullableFlatVector<Timestamp>(
              {
                  Timestamp{1'704'067'200, 123'457'999},
                  Timestamp{1'704'067'200, 123'456'789},
                  std::nullopt,
                  Timestamp{-1, 999'999'999},
                  Timestamp{0, 1'999},
                  std::nullopt,
                  std::nullopt,
                  earliest,
                  latest,
              },
              TIMESTAMP_UTC()),
          makeFlatVector<bool>(
              {false, true, true, false, true, true, false, false, false}),
      });
      inputs = {data, data};
      expected = makeRowVector({
          makeFlatVector<Timestamp>({earliest}, TIMESTAMP_UTC()),
          makeFlatVector<Timestamp>({latest}, TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .aggregates(aggregates)
          .assertResults(assertResults);

      expected = makeRowVector({
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<Timestamp>(
              {earlier, beforeEpoch, std::nullopt, earliest}, TIMESTAMP_UTC()),
          makeNullableFlatVector<Timestamp>(
              {later, afterEpoch, std::nullopt, latest}, TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .groupBy({"c0"})
          .aggregates(aggregates)
          .assertResults(assertResults);

      const std::vector<std::string> maskedAggregates{
          min("c1") + " FILTER (WHERE c2)",
          max("c1") + " FILTER (WHERE c2)",
      };
      expected = makeRowVector({
          makeFlatVector<Timestamp>({afterEpoch}, TIMESTAMP_UTC()),
          makeFlatVector<Timestamp>({earlier}, TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .aggregates(maskedAggregates)
          .assertResults(assertResults);

      expected = makeRowVector({
          makeFlatVector<int64_t>({0, 1, 2, 3}),
          makeNullableFlatVector<Timestamp>(
              {earlier, afterEpoch, std::nullopt, std::nullopt},
              TIMESTAMP_UTC()),
          makeNullableFlatVector<Timestamp>(
              {earlier, afterEpoch, std::nullopt, std::nullopt},
              TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .groupBy({"c0"})
          .aggregates(maskedAggregates)
          .assertResults(assertResults);
    }

    {
      SCOPED_TRACE("All-null and empty input");
      auto data = makeRowVector({
          makeFlatVector<int64_t>({0, 0, 1}),
          makeNullableFlatVector<Timestamp>(
              {std::nullopt, std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
      });
      inputs = {data};
      expected = makeRowVector({
          makeNullableFlatVector<Timestamp>({std::nullopt}, TIMESTAMP_UTC()),
          makeNullableFlatVector<Timestamp>({std::nullopt}, TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .aggregates(aggregates)
          .assertResults(assertResults);

      auto emptyInput = makeRowVector(asRowType(data->type()), 0);
      inputs = {emptyInput};
      tester()
          .inputs(inputs)
          .aggregates(aggregates)
          .assertResults(assertResults);

      inputs = {data};
      expected = makeRowVector({
          makeFlatVector<int64_t>({0, 1}),
          makeNullableFlatVector<Timestamp>(
              {std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
          makeNullableFlatVector<Timestamp>(
              {std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
      });
      tester()
          .inputs(inputs)
          .groupBy({"c0"})
          .aggregates(aggregates)
          .assertResults(assertResults);

      inputs = {emptyInput};
      expected = makeRowVector(asRowType(expected->type()), 0);
      tester()
          .inputs(inputs)
          .groupBy({"c0"})
          .aggregates(aggregates)
          .assertResults(assertResults);
    }
  }
}

TEST_F(MinMaxAggregationTest, array) {
  auto data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {1, 2, 3},
          {1, std::nullopt},
          {1, 7, 8},
      }),
  });

  auto expected = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {1, std::nullopt},
      }),
      makeArrayVector<int64_t>({
          {1, 7, 8},
      }),
  });

  AggregationTester(*this)
      .inputs(data)
      .aggregates({min("c0"), max("c0")})
      .assertResults(expected);

  data = makeRowVector({
      makeNullableArrayVector<int64_t>({
          {1, 2, 3},
          {2, 3},
          {3, 7, 8},
      }),
  });
  expected = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
      }),
      makeArrayVector<int64_t>({
          {3, 7, 8},
      }),
  });
  AggregationTester(*this)
      .inputs(data)
      .aggregates({min("c0"), max("c0")})
      .assertResults(expected);
}

TEST_F(MinMaxAggregationTest, row) {
  auto data = makeRowVector({
      makeRowVector({
          makeFlatVector<StringView>({
              "a"_sv,
              "b"_sv,
              "c"_sv,
          }),
          makeNullableFlatVector<StringView>({
              std::nullopt,
              "efg"_sv,
              "hij"_sv,
          }),
      }),
  });

  auto expected = makeRowVector({
      makeRowVector(
          {makeFlatVector<StringView>({"a"_sv}),
           makeNullableFlatVector<StringView>({std::nullopt})}),
      makeRowVector(
          {makeFlatVector<StringView>({"c"_sv}),
           makeFlatVector<StringView>({"hij"_sv})}),
  });

  AggregationTester(*this)
      .inputs(data)
      .aggregates({min("c0"), max("c0")})
      .assertResults(expected);

  data = makeRowVector({
      makeRowVector({
          makeFlatVector<StringView>({
              "a"_sv,
              "b"_sv,
              "c"_sv,
          }),
          makeNullableFlatVector<StringView>({
              "abc"_sv,
              "efg"_sv,
              "hij"_sv,
          }),
      }),
  });
  expected = makeRowVector({
      makeRowVector(
          {makeFlatVector<StringView>({"a"_sv}),
           makeFlatVector<StringView>({"abc"_sv})}),
      makeRowVector(
          {makeFlatVector<StringView>({"c"_sv}),
           makeFlatVector<StringView>({"hij"_sv})}),
  });
  AggregationTester(*this)
      .inputs(data)
      .aggregates({min("c0"), max("c0")})
      .assertResults(expected);
}

TEST_F(MinMaxAggregationTest, failOnUnorderableType) {
  auto data = makeRowVector({makeMapVectorFromJson<int32_t, double>({
      "{0: 0.05, 2: 2.05}",
      "{4: 4.05}",
      "{6: 6.05, 8: 8.05}",
  })});

  static const std::string kErrorMessage =
      "Aggregate function signature is not supported";
  for (const auto& expr : {min("c0"), max("c0")}) {
    {
      auto builder = PlanBuilder().values({data});
      VELOX_ASSERT_THROW(builder.singleAggregation({}, {expr}), kErrorMessage);
    }

    {
      auto builder = PlanBuilder().values({data});
      VELOX_ASSERT_THROW(
          builder.singleAggregation({"c1"}, {expr}), kErrorMessage);
    }
  }
}

TEST_F(MinMaxAggregationTest, partialCompanionAbandonPartialAggregation) {
  constexpr vector_size_t kBatchSize = 100;
  std::vector<RowVectorPtr> data;
  for (auto batch = 0; batch < 3; ++batch) {
    data.push_back(makeRowVector(
        {"k", "v"},
        {makeFlatVector<int64_t>(
             kBatchSize, [&](auto row) { return batch * kBatchSize + row; }),
         makeFlatIdentityVector<int64_t>(kBatchSize)}));
  }
  createDuckDbTable(data);

  core::PlanNodeId partialNodeId;
  auto plan = PlanBuilder()
                  .values(data)
                  .partialAggregation({"k"}, {"spark_min_partial(v)"})
                  .capturePlanNodeId(partialNodeId)
                  .finalAggregation()
                  .planNode();
  auto task =
      AssertQueryBuilder(plan, duckDbQueryRunner_)
          .maxDrivers(1)
          .config(core::QueryConfig::kAbandonPartialAggregationMinRows, 1)
          .config(core::QueryConfig::kAbandonPartialAggregationMinPct, 0)
          .assertResults("SELECT k, min(v) FROM tmp GROUP BY k");

  const auto stats = exec::toPlanStats(task->taskStats());
  EXPECT_LT(
      0,
      stats.at(partialNodeId)
          .customStats.at("abandonedPartialAggregationRows")
          .sum);
  EXPECT_GT(
      stats.at(partialNodeId).customStats.at("toIntermediateFastPathCalls").sum,
      0);
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
