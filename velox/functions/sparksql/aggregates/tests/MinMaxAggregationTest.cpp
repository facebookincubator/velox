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

  // Check logical types and values across aggregation stages and time zones.
  void testTimestampUtcAggregations(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates,
      const RowVectorPtr& expected) {
    for (const auto* timeZone :
         {"UTC", "America/Los_Angeles", "Asia/Kolkata"}) {
      SCOPED_TRACE(timeZone);
      testAggregations(
          [&](auto& builder) { builder.values(data); },
          groupingKeys,
          aggregates,
          {},
          [&](auto& builder) {
            std::shared_ptr<exec::Task> task;
            auto actual = builder.copyResults(pool(), task);
            // Compare logical types explicitly, since TIMESTAMP_UTC and
            // TIMESTAMP both use TypeKind::TIMESTAMP.
            EXPECT_TRUE(expected->type()->equivalent(*actual->type()))
                << actual->type()->toString();
            assertEqualResults({expected}, {actual});
            return task;
          },
          {
              {core::QueryConfig::kSessionTimezone, timeZone},
              {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
          });
    }
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
    testAggregations(
        vectors, {}, {min(c1), max(c1)}, "SELECT min(c1), max(c1) FROM tmp");

    // Group by aggregation.
    testAggregations(
        [&](auto& builder) {
          builder.values(vectors).project({"c0 % 10", "c1"});
        },
        {"p0"},
        {min(c1), max(c1)},
        "SELECT c0 % 10, min(c1), max(c1) FROM tmp GROUP BY 1");

    // Masked aggregations.
    auto minMaskedAgg = min(c1) + " filter (where mask)";
    testAggregations(
        vectors,
        {},
        {minMaskedAgg},
        "SELECT min(c1) filter (where mask) FROM tmp");

    testAggregations(
        [&](auto& builder) {
          builder.values(vectors).project({"c0 % 10", "c1", "mask"});
        },
        {"p0"},
        {minMaskedAgg},
        "SELECT c0 % 10, min(c1) filter (where mask) FROM tmp GROUP BY 1");

    auto maxMaskedAgg = max(c1) + " filter (where mask)";
    testAggregations(
        vectors,
        {},
        {maxMaskedAgg},
        "SELECT max(c1) filter (where mask) FROM tmp");

    testAggregations(
        [&](auto& builder) {
          builder.values(vectors).project({"c0 % 10", "c1", "mask"});
        },
        {"p0"},
        {maxMaskedAgg},
        "SELECT c0 % 10, max(c1) filter (where mask) FROM tmp GROUP BY 1");

    // Encodings: use filter to wrap aggregation inputs in a dictionary.
    testAggregations(
        [&](auto& builder) {
          builder.values(vectors)
              .filter("c0 % 2 = 0")
              .project({"c0 % 11", "c1"});
        },
        {"p0"},
        {min(c1), max(c1)},
        "SELECT c0 % 11, min(c1), max(c1) FROM tmp WHERE c0 % 2 = 0 GROUP BY 1");

    testAggregations(
        [&](auto& builder) { builder.values(vectors).filter("c0 % 2 = 0"); },
        {},
        {min(c1), max(c1)},
        "SELECT min(c1), max(c1) FROM tmp WHERE c0 % 2 = 0");
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

  testAggregations(
      vectors,
      {},
      {min("c1"), max("c1")},
      "SELECT date_trunc('microsecond', min(c1)), "
      "date_trunc('microsecond', max(c1)) FROM tmp");

  testAggregations(
      [&](auto& builder) {
        builder.values(vectors).project({"c0 % 17 as k", "c1"});
      },
      {"k"},
      {min("c1"), max("c1")},
      "SELECT c0 % 17, date_trunc('microsecond', min(c1)), "
      "date_trunc('microsecond', max(c1)) FROM tmp GROUP BY 1");
}

TEST_F(MinMaxAggregationTest, timestampUtc) {
  const auto beforeEpoch = Timestamp::fromMicros(-1);
  const auto afterEpoch = Timestamp::fromMicros(1);
  const Timestamp earlier{1'704'067'200, 123'456'000};
  const Timestamp later{1'704'067'200, 123'457'000};
  const std::vector<std::string> aggregates{min("c1"), max("c1")};

  {
    SCOPED_TRACE("Mixed values and all-null groups");
    auto data = makeRowVector({
        makeFlatVector<int64_t>({0, 0, 0, 1, 1, 2, 2}),
        makeNullableFlatVector<Timestamp>(
            {
                later,
                earlier,
                std::nullopt,
                beforeEpoch,
                afterEpoch,
                std::nullopt,
                std::nullopt,
            },
            TIMESTAMP_UTC()),
        makeFlatVector<bool>({false, true, true, false, true, true, false}),
    });
    auto expected = makeRowVector({
        makeFlatVector<Timestamp>({beforeEpoch}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({later}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data, data}, {}, aggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1, 2}),
        makeNullableFlatVector<Timestamp>(
            {earlier, beforeEpoch, std::nullopt}, TIMESTAMP_UTC()),
        makeNullableFlatVector<Timestamp>(
            {later, afterEpoch, std::nullopt}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data, data}, {"c0"}, aggregates, expected);

    const std::vector<std::string> maskedAggregates{
        min("c1") + " FILTER (WHERE c2)",
        max("c1") + " FILTER (WHERE c2)",
    };
    expected = makeRowVector({
        makeFlatVector<Timestamp>({afterEpoch}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({earlier}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data, data}, {}, maskedAggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1, 2}),
        makeNullableFlatVector<Timestamp>(
            {earlier, afterEpoch, std::nullopt}, TIMESTAMP_UTC()),
        makeNullableFlatVector<Timestamp>(
            {earlier, afterEpoch, std::nullopt}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations(
        {data, data}, {"c0"}, maskedAggregates, expected);
  }

  {
    SCOPED_TRACE("All-null and empty input");
    auto data = makeRowVector({
        makeFlatVector<int64_t>({0, 0, 1}),
        makeNullableFlatVector<Timestamp>(
            {std::nullopt, std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
    });
    auto expected = makeRowVector({
        makeNullableFlatVector<Timestamp>({std::nullopt}, TIMESTAMP_UTC()),
        makeNullableFlatVector<Timestamp>({std::nullopt}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {}, aggregates, expected);

    auto emptyInput = makeRowVector(asRowType(data->type()), 0);
    testTimestampUtcAggregations({emptyInput}, {}, aggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1}),
        makeNullableFlatVector<Timestamp>(
            {std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
        makeNullableFlatVector<Timestamp>(
            {std::nullopt, std::nullopt}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {"c0"}, aggregates, expected);
    testTimestampUtcAggregations(
        {emptyInput},
        {"c0"},
        aggregates,
        makeRowVector(asRowType(expected->type()), 0));
  }

  {
    SCOPED_TRACE("Constant encoding");
    auto data = makeRowVector({
        makeFlatVector<int64_t>({0, 0, 1, 1}),
        makeConstant<Timestamp>(earlier, 4, TIMESTAMP_UTC()),
    });
    auto expected = makeRowVector({
        makeFlatVector<Timestamp>({earlier}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({earlier}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {}, aggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1}),
        makeFlatVector<Timestamp>({earlier, earlier}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({earlier, earlier}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {"c0"}, aggregates, expected);
  }

  {
    SCOPED_TRACE("Dictionary encoding");
    auto timestamps = makeNullableFlatVector<Timestamp>(
        {beforeEpoch, earlier, later, std::nullopt}, TIMESTAMP_UTC());
    auto data = makeRowVector({
        makeFlatVector<int64_t>({0, 0, 0, 0, 1, 1, 1, 1}),
        wrapInDictionary(makeIndices({3, 0, 2, 1, 2, 3, 1, 1}), 8, timestamps),
    });
    auto expected = makeRowVector({
        makeFlatVector<Timestamp>({beforeEpoch}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({later}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {}, aggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1}),
        makeFlatVector<Timestamp>({beforeEpoch, earlier}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({later, later}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {"c0"}, aggregates, expected);
  }

  {
    SCOPED_TRACE("Microsecond range boundaries");
    const auto earliest =
        Timestamp::fromMicros(std::numeric_limits<int64_t>::min());
    const auto latest =
        Timestamp::fromMicros(std::numeric_limits<int64_t>::max());
    auto data = makeRowVector({
        makeFlatVector<int64_t>({0, 0, 1, 1}),
        makeFlatVector<Timestamp>(
            {earliest, beforeEpoch, earlier, latest}, TIMESTAMP_UTC()),
    });
    auto expected = makeRowVector({
        makeFlatVector<Timestamp>({earliest}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({latest}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {}, aggregates, expected);

    expected = makeRowVector({
        makeFlatVector<int64_t>({0, 1}),
        makeFlatVector<Timestamp>({earliest, earlier}, TIMESTAMP_UTC()),
        makeFlatVector<Timestamp>({beforeEpoch, latest}, TIMESTAMP_UTC()),
    });
    testTimestampUtcAggregations({data}, {"c0"}, aggregates, expected);
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

  testAggregations({data}, {}, {min("c0"), max("c0")}, {expected});

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
  testAggregations({data}, {}, {min("c0"), max("c0")}, {expected});
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

  testAggregations({data}, {}, {min("c0"), max("c0")}, {expected});

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
  testAggregations({data}, {}, {min("c0"), max("c0")}, {expected});
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
         makeFlatVector<int64_t>(kBatchSize, folly::identity)}));
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
          .config(core::QueryConfig::kAbandonPartialAggregationMinRows, "1")
          .config(core::QueryConfig::kAbandonPartialAggregationMinPct, "0")
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
