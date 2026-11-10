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

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/fuzzer/SparkQueryRunner.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class SparkQueryRunnerTest : public ::testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    parquet::registerParquetWriterFactory();
  }

  void SetUp() override {
    velox::functions::sparksql::registerFunctions("");
    velox::aggregate::prestosql::registerAllAggregateFunctions();
    velox::functions::aggregate::sparksql::registerAggregateFunctions("");
    velox::parse::registerTypeResolver();
  }
};

// This test requires a Spark coordinator running at localhost, so disable it
// by default.
TEST_F(SparkQueryRunnerTest, DISABLED_basic) {
  auto aggregatePool = rootPool_->addAggregateChild("basic");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "localhost:15002", "test", "basic");

  auto input = makeRowVector({
      makeConstant<int64_t>(1, 25),
  });
  auto plan = exec::test::PlanBuilder()
                  .values({input})
                  .singleAggregation({}, {"count(c0)"})
                  .planNode();
  auto sparkResults = queryRunner->executeAndReturnVector(plan);
  auto expected = makeRowVector({
      makeConstant<int64_t>(25, 1),
  });
  ASSERT_EQ(sparkResults.second, exec::test::ReferenceQueryErrorCode::kSuccess);
  exec::test::assertEqualResults(sparkResults.first.value(), {expected});

  input = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2,
                               3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4}),
      makeFlatVector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
  });
  plan = exec::test::PlanBuilder()
             .values({input})
             .singleAggregation({"c0"}, {"count(c1)"})
             .planNode();
  sparkResults = queryRunner->executeAndReturnVector(plan);
  expected = makeRowVector({
      makeFlatVector<int64_t>({0, 1, 2, 3, 4}),
      makeFlatVector<int64_t>({5, 5, 5, 5, 5}),
  });
  ASSERT_EQ(sparkResults.second, exec::test::ReferenceQueryErrorCode::kSuccess);
  exec::test::assertEqualResults(sparkResults.first.value(), {expected});
}

// This test requires a Spark coordinator running at localhost, so disable it
// by default.
TEST_F(SparkQueryRunnerTest, DISABLED_decimal) {
  auto aggregatePool = rootPool_->addAggregateChild("decimal");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "localhost:15002", "test", "decimal");
  auto input = makeRowVector({
      makeConstant<int128_t>(123456789, 25, DECIMAL(34, 2)),
  });
  auto plan =
      exec::test::PlanBuilder().values({input}).project({"abs(c0)"}).planNode();
  auto sparkResults = queryRunner->executeAndReturnVector(plan);
  ASSERT_EQ(sparkResults.second, exec::test::ReferenceQueryErrorCode::kSuccess);
  exec::test::assertEqualResults(sparkResults.first.value(), {input});
}

// This test requires a Spark coordinator running at localhost, so disable it
// by default.
TEST_F(SparkQueryRunnerTest, DISABLED_fuzzer) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
      makeNullableFlatVector<int64_t>({std::nullopt, 1, 2, std::nullopt, 4, 5}),
  });

  auto plan = exec::test::PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"sum(c0)", "collect_list(c1)"})
                  .project({"a0", "array_sort(a1)"})
                  .planNode();

  auto aggregatePool = rootPool_->addAggregateChild("fuzzer");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "localhost:15002", "test", "fuzzer");
  auto sql = queryRunner->toSql(plan);
  ASSERT_TRUE(sql.has_value());

  auto sparkResults = queryRunner->executeAndReturnVector(plan);
  auto veloxResults = exec::test::AssertQueryBuilder(plan).copyResults(pool());
  exec::test::assertEqualResults(sparkResults.first.value(), {veloxResults});
}

TEST_F(SparkQueryRunnerTest, toSql) {
  auto aggregatePool = rootPool_->addAggregateChild("toSql");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "unused", "unused", "unused");

  auto dataType = ROW({"c0", "c1", "c2"}, {DOUBLE(), DOUBLE(), BOOLEAN()});
  auto plan = exec::test::PlanBuilder()
                  .tableScan("tmp", dataType)
                  .singleAggregation({"c1"}, {"avg(c0)"})
                  .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan),
      "SELECT c1, avg(c0) as a0 FROM (tmp) GROUP BY c1");

  plan = exec::test::PlanBuilder()
             .tableScan("tmp", dataType)
             .singleAggregation({"c1"}, {"sum(c0)"})
             .project({"a0 / c1"})
             .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan),
      "SELECT (a0 / c1) as p0 FROM (SELECT c1, sum(c0) as a0 FROM (tmp) GROUP BY c1)");

  plan = exec::test::PlanBuilder()
             .tableScan("tmp", dataType)
             .singleAggregation({}, {"avg(c0)", "avg(c1)"}, {"c2"})
             .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan),
      "SELECT avg(c0) filter (where c2) as a0, avg(c1) as a1 FROM (tmp)");

  auto data =
      makeRowVector({makeFlatVector<int64_t>({}), makeFlatVector<int64_t>({})});
  plan = exec::test::PlanBuilder()
             .values({data})
             .singleAggregation({}, {"sum(distinct c0)"})
             .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan), "SELECT sum(distinct c0) as a0 FROM (t_0)");

  const std::vector<std::pair<VectorPtr, std::string>> cases{
      {makeFlatVector<bool>({true}), "true"},
      {makeFlatVector<bool>({false}), "false"},
      {makeFlatVector<int8_t>({-128}), "-128Y"},
      {makeFlatVector<int16_t>({-32'768}), "-32768S"},
      {makeFlatVector<int32_t>({42}), "42"},
      {makeFlatVector<int32_t>({0}, DATE()), "DATE '1970-01-01'"},
      {makeFlatVector<int64_t>({9'007'199'254'740'993LL}), "9007199254740993L"},
      {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::min()}),
       "-9223372036854775808L"},
      {makeFlatVector<float>({0.125f}), "0.125F"},
      {makeFlatVector<double>({0.10000000000000002}), "0.10000000000000002D"},
      {makeFlatVector<double>({-0.0}), "-0D"},
      {makeFlatVector<float>({std::numeric_limits<float>::infinity()}),
       "cast('Infinity' as FLOAT)"},
      {makeFlatVector<double>({-std::numeric_limits<double>::infinity()}),
       "cast('-Infinity' as DOUBLE)"},
      {makeFlatVector<double>({std::numeric_limits<double>::quiet_NaN()}),
       "cast('NaN' as DOUBLE)"},
      {makeNullableFlatVector<double>({std::nullopt}), "cast(null as DOUBLE)"},
      {makeNullableFlatVector<StringView>({std::nullopt}),
       "cast(null as STRING)"},
      {makeFlatVector<StringView>({StringView("a'b\\c\0", 6)}),
       "cast(X'6127625c6300' as STRING)"},
      {makeFlatVector<StringView>({StringView("\0\x80\xff", 3)}, VARBINARY()),
       "X'0080ff'"},
      {makeNullableFlatVector<StringView>({std::nullopt}, VARBINARY()),
       "cast(null as BINARY)"},
      {makeFlatVector<Timestamp>({Timestamp(0, 0)}), "timestamp_micros(0)"},
      {makeFlatVector<Timestamp>({Timestamp(-1, 999'999'000)}),
       "timestamp_micros(-1)"},
      {makeFlatVector<Timestamp>({Timestamp(1, 234'567'000)}),
       "timestamp_micros(1234567)"},
      {makeNullableFlatVector<Timestamp>({std::nullopt}),
       "cast(null as TIMESTAMP)"},
      {makeFlatVector<Timestamp>({Timestamp::minMicros()}),
       "timestamp_micros(-9223372036854775808)"},
      {makeFlatVector<Timestamp>({Timestamp::maxMicros()}),
       "timestamp_micros(9223372036854775807)"},
      {makeFlatVector<int64_t>({123}, DECIMAL(10, 2)),
       "cast(1.23BD as DECIMAL(10, 2))"},
      {makeFlatVector<int128_t>({123}, DECIMAL(25, 2)),
       "cast(1.23BD as DECIMAL(25, 2))"},
      {makeArrayVector<int64_t>({{1, 2}}),
       "cast(array(1L, 2L) as ARRAY<BIGINT>)"},
      {makeArrayVector<int64_t>({{}}), "cast(array() as ARRAY<BIGINT>)"},
      {makeArrayVectorFromJson<int64_t>({"[1, null]"}),
       "cast(array(1L, cast(null as BIGINT)) as ARRAY<BIGINT>)"},
      {makeMapVector<int64_t, int64_t>({{{1, 2}}}),
       "cast(map(1L, 2L) as MAP<BIGINT, BIGINT>)"},
      {makeMapVector<int64_t, int64_t>({{}}),
       "cast(map() as MAP<BIGINT, BIGINT>)"},
      {makeRowVector({makeFlatVector<int64_t>({3})}),
       "cast(struct(3L) as STRUCT<`c0`: BIGINT>)"},
      {makeRowVector(
           {"a`b"}, {makeNullableFlatVector<StringView>({std::nullopt})}),
       "cast(struct(cast(null as STRING)) as STRUCT<`a``b`: STRING>)"},
      {makeRowVector({makeArrayVector<int64_t>({{1, 2}})}),
       "cast(struct(cast(array(1L, 2L) as ARRAY<BIGINT>)) as STRUCT<`c0`: ARRAY<BIGINT>>)"},
      {BaseVector::createNullConstant(ARRAY(BIGINT()), 1, pool()),
       "cast(null as ARRAY<BIGINT>)"},
  };
  for (const auto& [value, literalSql] : cases) {
    SCOPED_TRACE(literalSql);
    auto constants = makeRowVector({value});
    auto plan =
        exec::test::PlanBuilder(pool())
            .setComplexConstants(constants)
            .tableScan(
                "tmp",
                ROW({{"c0", value->type()}, {"g", BIGINT()}, {"m", BOOLEAN()}}))
            .singleAggregation(
                {"g"}, {"first(distinct __complex_constant(c0))"}, {"m"})
            .planNode();
    EXPECT_EQ(
        queryRunner->toSql(plan),
        fmt::format(
            "SELECT g, first(distinct {}) filter (where m) as a0 FROM (tmp) GROUP BY g",
            literalSql));
  }

  plan = exec::test::PlanBuilder()
             .tableScan("tmp", ROW("c0", BIGINT()))
             .singleAggregation({}, {"first(true)", "first('abc')"})
             .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan),
      "SELECT first(true) as a0, first(cast(X'616263' as STRING)) as a1 FROM (tmp)");

  const auto* aggregation =
      dynamic_cast<const core::AggregationNode*>(plan.get());
  ASSERT_NE(aggregation, nullptr);
  const auto toSqlWithConstant =
      [&](const core::ConstantTypedExprPtr& constant) {
        auto aggregate = aggregation->aggregates()[0];
        aggregate.call = std::make_shared<core::CallTypedExpr>(
            constant->type(), "first", constant);
        aggregate.rawInputTypes = {constant->type()};
        auto literalPlan = core::AggregationNode::Builder(*aggregation)
                               .aggregates({aggregate})
                               .aggregateNames({"a0"})
                               .build();
        return queryRunner->toSql(literalPlan);
      };
  EXPECT_EQ(
      toSqlWithConstant(
          std::make_shared<core::ConstantTypedExpr>(
              VARBINARY(), variant::binary(std::string("\0\x80\xff", 3)))),
      "SELECT first(X'0080ff') as a0 FROM (tmp)");
  EXPECT_EQ(
      toSqlWithConstant(
          std::make_shared<core::ConstantTypedExpr>(
              TIMESTAMP(), Timestamp(-1, 999'999'000))),
      "SELECT first(timestamp_micros(-1)) as a0 FROM (tmp)");

  auto constants = makeRowVector({makeFlatVector<double>({0.125})});
  plan = exec::test::PlanBuilder(pool())
             .setComplexConstants(constants)
             .tableScan("tmp", ROW("c0", BIGINT()))
             .singleAggregation(
                 {}, {"approx_percentile(c0, __complex_constant(c0))"})
             .planNode();
  EXPECT_EQ(
      queryRunner->toSql(plan),
      "SELECT approx_percentile(c0, 0.125D) as a0 FROM (tmp)");
}

TEST_F(SparkQueryRunnerTest, unsupportedConstantArguments) {
  auto aggregatePool = rootPool_->addAggregateChild("unsupportedConstants");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "unused", "unused", "unused");
  const std::vector<VectorPtr> values{
      makeFlatVector<int128_t>({123}),
      makeArrayVector<int128_t>({{123}}),
      BaseVector::createNullConstant(ARRAY(HUGEINT()), 1, pool()),
  };
  for (const auto& value : values) {
    SCOPED_TRACE(value->toString());
    auto constants = makeRowVector({value});
    auto plan = exec::test::PlanBuilder(pool())
                    .setComplexConstants(constants)
                    .tableScan("tmp", ROW("c0", BIGINT()))
                    .singleAggregation(
                        {}, {"count(c0)", "first(__complex_constant(c0))"})
                    .project({"a1"})
                    .planNode();
    ASSERT_FALSE(queryRunner->toSql(plan).has_value());
    auto result = queryRunner->executeAndReturnVector(plan);
    EXPECT_FALSE(result.first.has_value());
    EXPECT_EQ(
        result.second,
        exec::test::ReferenceQueryErrorCode::kReferenceQueryUnsupported);
  }

  auto bloomPlan =
      exec::test::PlanBuilder(pool())
          .tableScan("tmp", ROW("c0", BIGINT()))
          .singleAggregation({}, {"bloom_filter_agg(c0, 128, 1024)"})
          .planNode();
  EXPECT_FALSE(queryRunner->toSql(bloomPlan).has_value());
  EXPECT_EQ(
      queryRunner->executeAndReturnVector(bloomPlan).second,
      exec::test::ReferenceQueryErrorCode::kReferenceQueryUnsupported);
}

// This test requires a Spark coordinator running at localhost, so disable it
// by default.
TEST_F(SparkQueryRunnerTest, DISABLED_literalArguments) {
  auto aggregatePool = rootPool_->addAggregateChild("literalArguments");
  auto queryRunner = std::make_unique<fuzzer::SparkQueryRunner>(
      aggregatePool.get(), "localhost:15002", "test", "literalArguments");
  queryRunner->execute("SET spark.sql.session.timeZone=America/Los_Angeles");
  auto constants = makeRowVector({
      makeFlatVector<double>({0.125}),
      makeFlatVector<double>({0.99}),
      makeFlatVector<int32_t>({42}),
      makeFlatVector<int64_t>({9'007'199'254'740'993LL}),
      makeNullableFlatVector<double>({std::nullopt}),
      makeFlatVector<StringView>({StringView("a'b\\c\0", 6)}),
      makeFlatVector<StringView>({StringView("\0\x80\xff", 3)}, VARBINARY()),
      makeFlatVector<Timestamp>({Timestamp(0, 0)}),
      makeFlatVector<Timestamp>({Timestamp(-1, 999'999'000)}),
      makeFlatVector<Timestamp>({Timestamp(1, 234'567'000)}),
      makeNullableFlatVector<Timestamp>({std::nullopt}),
  });
  std::vector<std::string> aggregates;
  for (auto i = 0; i < constants->childrenSize(); ++i) {
    aggregates.push_back(fmt::format("first(__complex_constant(c{}))", i));
  }
  auto plan = exec::test::PlanBuilder(pool())
                  .setComplexConstants(constants)
                  .values({makeRowVector({makeFlatVector<int64_t>({1, 2, 3})})})
                  .singleAggregation({}, aggregates)
                  .planNode();
  auto sparkResults = queryRunner->executeAndReturnVector(plan);
  ASSERT_EQ(sparkResults.second, exec::test::ReferenceQueryErrorCode::kSuccess);
  ASSERT_TRUE(sparkResults.first.has_value());
  exec::test::assertEqualResults(sparkResults.first.value(), {constants});
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
