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

#include <algorithm>
#include <utility>

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/fuzzer/AggregationFuzzerBase.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/fuzzer/SparkQueryRunner.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook;
using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

// Exercises signature binding and input generation without a Spark server.
class AggregationFuzzerHarness : public exec::test::AggregationFuzzerBase {
 public:
  explicit AggregationFuzzerHarness(
      std::unique_ptr<exec::test::ReferenceQueryRunner> queryRunner)
      : AggregationFuzzerBase(
            19'360,
            {},
            {},
            VectorFuzzer::Options::TimestampPrecision::kMicroSeconds,
            {},
            {},
            true,
            std::move(queryRunner)) {}

  velox::fuzzer::CallableSignature bind(
      const std::string& name,
      const exec::FunctionSignaturePtr& signature) {
    signatures_.clear();
    signatureTemplates_.clear();
    VELOX_CHECK(addSignature(name, signature));
    signatureStats_.resize(1);
    return pickSignature().first;
  }

  using AggregationFuzzerBase::generateInputData;
};

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
}

TEST_F(SparkQueryRunnerTest, histogramFuzzerConstantArguments) {
  gflags::FlagSaver flagSaver;
  FLAGS_batch_size = 4;
  FLAGS_num_batches = 3;
  FLAGS_null_ratio = 0;

  auto aggregatePool = rootPool_->addAggregateChild("histogramFuzzer");
  AggregationFuzzerHarness harness(
      std::make_unique<fuzzer::SparkQueryRunner>(
          aggregatePool.get(), "unused", "unused", "unused"));

  for (const auto* name : {"histogram_numeric", "histogram_numeric_legacy"}) {
    const auto signatures = exec::getAggregateFunctionSignatures(name);
    ASSERT_TRUE(signatures.has_value());
    auto signature = std::find_if(
        signatures->begin(), signatures->end(), [](const auto& candidate) {
          return candidate->argumentTypes()[0].baseName() == "double";
        });
    ASSERT_NE(signature, signatures->end());
    const auto callable = harness.bind(name, *signature);
    EXPECT_THAT(callable.constantArgs, testing::ElementsAre(false, true));

    const auto input = harness.generateInputData(
        {"c0", "c1", "m0"}, {DOUBLE(), INTEGER(), BOOLEAN()}, callable);
    ASSERT_EQ(input.size(), 3);
    const auto& firstConstant = input.front()->childAt(1);
    for (const auto& batch : input) {
      const auto& constant = batch->childAt(1);
      ASSERT_TRUE(constant->isConstantEncoding());
      for (auto row = 0; row < batch->size(); ++row) {
        EXPECT_TRUE(constant->equalValueAt(firstConstant.get(), row, 0));
      }
    }

    auto data = makeRowVector(
        {makeConstant<double>(1, 3), makeConstant<int32_t>(4, 3)});
    for (const auto& [sorted, distinct] :
         {std::pair{false, false},
          std::pair{false, true},
          std::pair{true, false}}) {
      auto bins = makeRowVector(
          {"x", "y"},
          {makeFlatVector<double>({1}),
           makeFlatVector<double>({distinct ? 1.0 : 3.0})});
      auto expected = makeRowVector({makeArrayVector({0}, bins)});
      const auto call =
          exec::test::makeFunctionCall(callable, data, sorted, distinct);
      auto plan = exec::test::PlanBuilder()
                      .values({data})
                      .singleAggregation({}, {call})
                      .planNode();
      const auto aggregation =
          std::dynamic_pointer_cast<const core::AggregationNode>(plan);
      ASSERT_NE(aggregation, nullptr);
      const auto& arguments = aggregation->aggregates()[0].call->inputs();
      EXPECT_TRUE(arguments[0]->isFieldAccessKind());
      ASSERT_TRUE(arguments[1]->isConstantKind());
      EXPECT_TRUE(arguments[1]->type()->isInteger());
      exec::test::AssertQueryBuilder(plan).assertResults(expected);

      if (!sorted && !distinct) {
        auto partialFinal = exec::test::PlanBuilder()
                                .values({data})
                                .partialAggregation({}, {call})
                                .finalAggregation()
                                .planNode();
        exec::test::AssertQueryBuilder(partialFinal).assertResults(expected);
      }
    }
  }
  const auto signature = exec::AggregateFunctionSignatureBuilder()
                             .typeVariable("T")
                             .returnType("bigint")
                             .intermediateType("varbinary")
                             .argumentType("T")
                             .constantArgumentType("integer")
                             .build();
  const auto callable = harness.bind("test_aggregate", signature);
  EXPECT_THAT(callable.constantArgs, testing::ElementsAre(false, true));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
