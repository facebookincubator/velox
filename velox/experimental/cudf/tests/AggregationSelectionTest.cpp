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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfGroupby.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/tests/utils/ExpressionTestUtil.h"

#include "velox/common/memory/Memory.h"

namespace {
// Simple wrapper for test compatibility - checks if an aggregation function
// is supported in either the groupby or reduce registry (kSingle step).
bool canAggregationBeEvaluatedByCudf(
    const facebook::velox::core::CallTypedExpr& call,
    facebook::velox::core::QueryCtx* queryCtx) {
  std::vector<facebook::velox::TypePtr> rawInputTypes;
  for (const auto& input : call.inputs()) {
    rawInputTypes.push_back(input->type());
  }
  return facebook::velox::cudf_velox::canGroupbyAggregationBeEvaluatedByCudf(
             call,
             facebook::velox::core::AggregationNode::Step::kSingle,
             rawInputTypes,
             queryCtx) ||
      facebook::velox::cudf_velox::canReduceAggregationBeEvaluatedByCudf(
             call,
             facebook::velox::core::AggregationNode::Step::kSingle,
             rawInputTypes,
             queryCtx);
}
} // namespace
#include "velox/core/QueryCtx.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test_utils;
using namespace facebook::velox::exec::test;

namespace {

class CudfAggregationSelectionTest : public ::testing::Test,
                                     public test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    queryCtx_ = core::QueryCtx::create();
    execCtx_ = std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get());
    facebook::velox::functions::prestosql::registerAllScalarFunctions();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();

    rowType_ = ROW({
        {"c0", BIGINT()},
        {"c1", BIGINT()},
        {"c2", INTEGER()},
        {"c3", DOUBLE()},
        {"c4", REAL()},
        {"c5", DOUBLE()},
        {"c6", VARCHAR()},
        {"c7", BOOLEAN()},
    });

    parse::registerTypeResolver();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    execCtx_.reset();
    queryCtx_.reset();
    pool_.reset();
  }

  std::shared_ptr<const core::AggregationNode> createAggregationNode(
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregates) {
    auto plan = PlanBuilder()
                    .values({makeRowVector({
                        makeFlatVector<int64_t>({1, 2, 3}),
                        makeFlatVector<int64_t>({10, 20, 30}),
                        makeFlatVector<int32_t>({100, 200, 300}),
                        makeFlatVector<double>({1.1, 2.2, 3.3}),
                        makeFlatVector<float>({1.5f, 2.5f, 3.5f}),
                        makeFlatVector<double>({10.1, 20.2, 30.3}),
                        makeFlatVector<std::string>({"a", "b", "c"}),
                        makeFlatVector<bool>({true, false, true}),
                    })})
                    .aggregation(
                        groupingKeys,
                        aggregates,
                        {},
                        core::AggregationNode::Step::kSingle,
                        false)
                    .planNode();

    return std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  }

  // One aggregate over c1, grouped by 'groupingKeys' (c0 for the groupby
  // path, none for the reduce path), with the boolean column c2 as its FILTER
  // mask.
  std::shared_ptr<const core::AggregationNode> createMaskedAggregationNode(
      const std::string& aggregate,
      const std::vector<std::string>& groupingKeys = {"c0"}) {
    auto plan = PlanBuilder()
                    .values({makeRowVector({
                        makeFlatVector<int64_t>({1, 2, 3}),
                        makeFlatVector<int64_t>({10, 20, 30}),
                        makeFlatVector<bool>({true, false, true}),
                    })})
                    .aggregation(
                        groupingKeys,
                        {aggregate},
                        {},
                        core::AggregationNode::Step::kSingle,
                        false)
                    .planNode();
    auto node = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    auto aggregates = node->aggregates();
    aggregates[0].mask =
        std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c2");
    return core::AggregationNode::Builder(*node)
        .aggregates(std::move(aggregates))
        .build();
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<core::ExecCtx> execCtx_;
  RowTypePtr rowType_;
};

// Test supported aggregation functions
TEST_F(CudfAggregationSelectionTest, supportedAggregationFunctions) {
  auto aggregationNode = createAggregationNode(
      {"c0"}, {"sum(c1)", "count(c2)", "min(c3)", "max(c4)", "avg(c5)"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test stddev_samp is supported
TEST_F(CudfAggregationSelectionTest, stddevSampSupported) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"stddev_samp(c1)", "stddev_samp(c5)"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// array_agg is supported for any element type cuDF can hold in a list.
TEST_F(CudfAggregationSelectionTest, arrayAggSupported) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"array_agg(c1)", "array_agg(c6)"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test unsupported aggregation functions
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationFunctions) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"stddev(c1)", "variance(c2)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test mixed supported and unsupported functions - should reject if any
// function is unsupported
TEST_F(CudfAggregationSelectionTest, mixedSupportedUnsupportedFunctions) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"sum(c1)", "variance(c2)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test supported grouping key expressions (simple field references)
TEST_F(CudfAggregationSelectionTest, supportedGroupingKeyExpressions) {
  auto aggregationNode = createAggregationNode({"c0", "c1"}, {"sum(c2)"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test unsupported aggregation functions (using variance as example)
TEST_F(CudfAggregationSelectionTest, unsupportedGroupingKeyExpressions) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"sum(c1)", "variance(c2)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test supported aggregation input expressions
TEST_F(CudfAggregationSelectionTest, supportedAggregationInputExpressions) {
  auto aggregationNode =
      createAggregationNode({"c0"}, {"sum(c1 + c2)", "max(length(c6))"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test unsupported aggregation input expressions
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationInputExpressions) {
  auto aggregationNode = createAggregationNode({"c0"}, {"variance(c1)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test global aggregation (no group by)
TEST_F(CudfAggregationSelectionTest, globalAggregationSupported) {
  auto aggregationNode =
      createAggregationNode({}, {"sum(c1)", "count(c2)", "max(c3)"});

  ASSERT_TRUE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test global aggregation with unsupported functions
TEST_F(CudfAggregationSelectionTest, globalAggregationUnsupported) {
  auto aggregationNode = createAggregationNode({}, {"stddev(c1)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// array_agg has no reduce implementation, so its global form stays on the CPU.
TEST_F(CudfAggregationSelectionTest, globalArrayAggUnsupported) {
  auto aggregationNode = createAggregationNode({}, {"array_agg(c1)"});

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test complex groupby clause with expressions
TEST_F(CudfAggregationSelectionTest, complexGroupbyClauseExpressions) {
  auto plan =
      PlanBuilder()
          .values({makeRowVector({
              makeFlatVector<int64_t>({1, 2, 3, 4, 5}),
              makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
          })})
          .project(
              {"c0",
               "c1",
               "to_big_endian_64(c0) AS endian_c0"}) // to_big_endian_64 is
                                                     // unsupported by CUDF
          .aggregation(
              {"endian_c0"},
              {"sum(c1)"},
              {},
              core::AggregationNode::Step::kSingle,
              false)
          .planNode();

  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// Test nested aggregation: allowed -> not allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationAllowedToNotAllowed) {
  auto innerPlan = PlanBuilder()
                       .values({makeRowVector({
                           makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
                           makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
                       })})
                       .aggregation(
                           {"c0"},
                           {"sum(c1) AS inner_sum"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerPlan = PlanBuilder()
                       .addNode([&](std::string id, core::PlanNodePtr input) {
                         return innerPlan;
                       })
                       .aggregation(
                           {},
                           {"stddev(inner_sum) AS outer_stddev"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerAggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);

  ASSERT_FALSE(canBeEvaluatedByCudf(
      *outerAggregationNode, queryCtx_.get(), pool_.get()));
}

// Test nested aggregation: allowed -> allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationAllowedToAllowed) {
  auto innerPlan = PlanBuilder()
                       .values({makeRowVector({
                           makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
                           makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
                       })})
                       .aggregation(
                           {"c0"},
                           {"sum(c1) AS inner_sum"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerPlan = PlanBuilder()
                       .addNode([&](std::string id, core::PlanNodePtr input) {
                         return innerPlan;
                       })
                       .aggregation(
                           {},
                           {"sum(inner_sum) AS outer_sum"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerAggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);

  ASSERT_TRUE(canBeEvaluatedByCudf(
      *outerAggregationNode, queryCtx_.get(), pool_.get()));
}

// Test nested aggregation: not allowed -> allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationNotAllowedToAllowed) {
  auto innerPlan = PlanBuilder()
                       .values({makeRowVector({
                           makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
                           makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
                       })})
                       .aggregation(
                           {"c0"},
                           {"stddev(c1) AS inner_stddev"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerPlan = PlanBuilder()
                       .addNode([&](std::string id, core::PlanNodePtr input) {
                         return innerPlan;
                       })
                       .aggregation(
                           {},
                           {"sum(inner_stddev) AS outer_sum"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerAggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);

  // Only validates the current (outer) aggregation node
  ASSERT_TRUE(canBeEvaluatedByCudf(
      *outerAggregationNode, queryCtx_.get(), pool_.get()));
}

// Test nested aggregation: not allowed -> not allowed
TEST_F(CudfAggregationSelectionTest, nestedAggregationNotAllowedToNotAllowed) {
  auto innerPlan = PlanBuilder()
                       .values({makeRowVector({
                           makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
                           makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
                       })})
                       .aggregation(
                           {"c0"},
                           {"stddev(c1) AS inner_stddev"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerPlan = PlanBuilder()
                       .addNode([&](std::string id, core::PlanNodePtr input) {
                         return innerPlan;
                       })
                       .aggregation(
                           {},
                           {"variance(inner_stddev) AS outer_variance"},
                           {},
                           core::AggregationNode::Step::kSingle,
                           false)
                       .planNode();

  auto outerAggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(outerPlan);

  ASSERT_FALSE(canBeEvaluatedByCudf(
      *outerAggregationNode, queryCtx_.get(), pool_.get()));
}

// Test unsupported aggregation function signatures
TEST_F(CudfAggregationSelectionTest, unsupportedAggregationFunctionSignatures) {
  auto varianceExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "variance");

  ASSERT_FALSE(canAggregationBeEvaluatedByCudf(*varianceExpr, queryCtx_.get()));
}

// Test comprehensive type support validation - all registered CUDF aggregation
// signatures
TEST_F(CudfAggregationSelectionTest, comprehensiveTypeSupportValidation) {
  // SUM signatures
  auto sumTinyintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "c0")},
      "sum");
  auto sumSmallintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")},
      "sum");
  auto sumIntegerExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "sum");
  auto sumBigintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "sum");
  auto sumRealExpr = std::make_shared<core::CallTypedExpr>(
      REAL(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(REAL(), "c4")},
      "sum");
  auto sumDoubleExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c3")},
      "sum");

  // COUNT signatures
  auto countTinyintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "c0")},
      "count");
  auto countSmallintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")},
      "count");
  auto countIntegerExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "count");
  auto countBigintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "count");
  auto countRealExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(REAL(), "c4")},
      "count");
  auto countDoubleExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c3")},
      "count");
  auto countVarcharExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")},
      "count");
  auto countBooleanExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c7")},
      "count");
  auto countStarExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(), std::vector<core::TypedExprPtr>{}, "count");

  // MIN signatures
  auto minTinyintExpr = std::make_shared<core::CallTypedExpr>(
      TINYINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "c0")},
      "min");
  auto minSmallintExpr = std::make_shared<core::CallTypedExpr>(
      SMALLINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")},
      "min");
  auto minIntegerExpr = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "min");
  auto minBigintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "min");
  auto minRealExpr = std::make_shared<core::CallTypedExpr>(
      REAL(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(REAL(), "c4")},
      "min");
  auto minDoubleExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c3")},
      "min");

  // MAX signatures
  auto maxTinyintExpr = std::make_shared<core::CallTypedExpr>(
      TINYINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(TINYINT(), "c0")},
      "max");
  auto maxSmallintExpr = std::make_shared<core::CallTypedExpr>(
      SMALLINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")},
      "max");
  auto maxIntegerExpr = std::make_shared<core::CallTypedExpr>(
      INTEGER(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "max");
  auto maxBigintExpr = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "max");
  auto maxRealExpr = std::make_shared<core::CallTypedExpr>(
      REAL(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(REAL(), "c4")},
      "max");
  auto maxDoubleExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c3")},
      "max");

  // MIN/MAX VARCHAR signatures
  auto minVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")},
      "min");
  auto maxVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")},
      "max");

  // AVG signatures
  auto avgSmallintExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(SMALLINT(), "c1")},
      "avg");
  auto avgIntegerExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "avg");
  auto avgBigintExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0")},
      "avg");
  auto avgDoubleExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "c3")},
      "avg");

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*sumTinyintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*sumSmallintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*sumIntegerExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*sumBigintExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*sumRealExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*sumDoubleExpr, queryCtx_.get()));

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countTinyintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countSmallintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countIntegerExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countBigintExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*countRealExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countDoubleExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countVarcharExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*countBooleanExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*countStarExpr, queryCtx_.get()));

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*minTinyintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*minSmallintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*minIntegerExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*minBigintExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*minRealExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*minDoubleExpr, queryCtx_.get()));

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*maxTinyintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*maxSmallintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*maxIntegerExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*maxBigintExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*maxRealExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*maxDoubleExpr, queryCtx_.get()));

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*minVarcharExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*maxVarcharExpr, queryCtx_.get()));

  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*avgSmallintExpr, queryCtx_.get()));
  ASSERT_TRUE(
      canAggregationBeEvaluatedByCudf(*avgIntegerExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*avgBigintExpr, queryCtx_.get()));
  ASSERT_TRUE(canAggregationBeEvaluatedByCudf(*avgDoubleExpr, queryCtx_.get()));
}

// Test invalid aggregation signatures
TEST_F(CudfAggregationSelectionTest, invalidTypeCombinationsRejected) {
  // avg on varchar
  auto avgVarcharExpr = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")},
      "avg");

  // sum on varchar
  auto sumVarcharExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c6")},
      "sum");

  // max on boolean
  auto maxBooleanExpr = std::make_shared<core::CallTypedExpr>(
      BOOLEAN(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BOOLEAN(), "c7")},
      "max");

  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*avgVarcharExpr, queryCtx_.get()));
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*sumVarcharExpr, queryCtx_.get()));
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*maxBooleanExpr, queryCtx_.get()));
}

// Test `distinct` aggregations should be rejected early (otherwise the throw
// NYI)
TEST_F(CudfAggregationSelectionTest, distinctAggregationsRejected) {
  auto plan = PlanBuilder()
                  .values({makeRowVector({
                      makeFlatVector<int64_t>({1, 1, 2, 2, 3, 3}),
                      makeFlatVector<int64_t>({10, 20, 30, 40, 50, 60}),
                  })})
                  .aggregation(
                      {"c0"},
                      {"count(distinct c1)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();

  auto aggregationNode =
      std::dynamic_pointer_cast<const core::AggregationNode>(plan);

  ASSERT_FALSE(
      canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
}

// An aggregate with ORDER BY stays on the CPU when its result depends on the
// input order, which Velox records per function. One whose result does not is
// accepted, as the CPU skips that sort as well.
TEST_F(CudfAggregationSelectionTest, orderByAggregations) {
  for (const auto* sorted :
       {"array_agg(c1 ORDER BY c2)", "stddev_samp(c1 ORDER BY c2)"}) {
    SCOPED_TRACE(sorted);
    auto aggregationNode = createAggregationNode({"c0"}, {sorted});
    ASSERT_FALSE(
        canBeEvaluatedByCudf(*aggregationNode, queryCtx_.get(), pool_.get()));
  }

  auto unordered = createAggregationNode({"c0"}, {"min(c1 ORDER BY c2)"});
  ASSERT_TRUE(canBeEvaluatedByCudf(*unordered, queryCtx_.get(), pool_.get()));
}

// Which aggregates accept a FILTER mask on each path. stddev has an
// aggregator on the groupby path alone and approx_distinct on the reduce path
// alone, and each is mask-capable there alone: an aggregator added later for
// the other path accepts no mask until its registration declares one.
// array_agg collects every row or every non-null row of a group and cannot
// honor a mask (see GroupbyArrayAggAggregator).
TEST_F(CudfAggregationSelectionTest, maskedAggregatesByPath) {
  struct Case {
    std::string aggregate;
    bool groupby;
    bool reduce;
  };
  const std::vector<Case> cases = {
      {"sum(c1)", true, true},
      {"count(c1)", true, true},
      {"min(c1)", true, true},
      {"max(c1)", true, true},
      {"avg(c1)", true, true},
      {"stddev_samp(c1)", true, false},
      {"stddev(c1)", true, false},
      {"approx_distinct(c1)", false, true},
      {"array_agg(c1)", false, false},
  };
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.aggregate);
    EXPECT_EQ(
        canBeEvaluatedByCudf(
            *createMaskedAggregationNode(testCase.aggregate, {"c0"}),
            queryCtx_.get(),
            pool_.get()),
        testCase.groupby);
    EXPECT_EQ(
        canBeEvaluatedByCudf(
            *createMaskedAggregationNode(testCase.aggregate, {}),
            queryCtx_.get(),
            pool_.get()),
        testCase.reduce);
  }
}

// A function that honors a mask on one path does not do so on the other until
// an aggregator there declares it: with only a signature added for the other
// path, the unmasked aggregate is accepted and the masked one is not.
TEST_F(CudfAggregationSelectionTest, maskSupportNotInheritedAcrossPaths) {
  appendReduceAggregationFunctionForStep(
      "stddev_samp",
      core::AggregationNode::Step::kSingle,
      exec::FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("bigint")
          .build());
  ASSERT_TRUE(canBeEvaluatedByCudf(
      *createAggregationNode({}, {"stddev_samp(c1)"}),
      queryCtx_.get(),
      pool_.get()));
  EXPECT_FALSE(canBeEvaluatedByCudf(
      *createMaskedAggregationNode("stddev_samp(c1)", {}),
      queryCtx_.get(),
      pool_.get()));

  appendGroupbyAggregationFunctionForStep(
      "approx_distinct",
      core::AggregationNode::Step::kSingle,
      exec::FunctionSignatureBuilder()
          .returnType("bigint")
          .argumentType("bigint")
          .build());
  ASSERT_TRUE(canBeEvaluatedByCudf(
      *createAggregationNode({"c0"}, {"approx_distinct(c1)"}),
      queryCtx_.get(),
      pool_.get()));
  EXPECT_FALSE(canBeEvaluatedByCudf(
      *createMaskedAggregationNode("approx_distinct(c1)", {"c0"}),
      queryCtx_.get(),
      pool_.get()));
}

// A non-boolean mask falls back to CPU.
TEST_F(CudfAggregationSelectionTest, maskedNonBooleanRejected) {
  auto plan = PlanBuilder()
                  .values({makeRowVector({
                      makeFlatVector<int64_t>({1, 2, 3}),
                      makeFlatVector<int64_t>({10, 20, 30}),
                  })})
                  .aggregation(
                      {"c0"},
                      {"sum(c1)"},
                      {},
                      core::AggregationNode::Step::kSingle,
                      false)
                  .planNode();
  auto node = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  auto aggs = node->aggregates();
  aggs[0].mask = std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c1");
  auto masked =
      core::AggregationNode::Builder(*node).aggregates(std::move(aggs)).build();
  ASSERT_FALSE(canBeEvaluatedByCudf(*masked, queryCtx_.get(), pool_.get()));
}

// Test return type validation
// DISABLED: This test demonstrates expected failure modes when return type
// matching is enabled.
TEST_F(
    CudfAggregationSelectionTest,
    DISABLED_returnTypeMismatchShouldBeRejected) {
  // These should be rejected because the return type doesn't match the
  // registered signature

  // sum(integer) should return BIGINT, not VARCHAR
  auto sumWrongReturnExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(), // Wrong return type - should be BIGINT
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "sum");

  // avg(integer) should return DOUBLE, not INTEGER
  auto avgWrongReturnExpr = std::make_shared<core::CallTypedExpr>(
      INTEGER(), // Wrong return type - should be DOUBLE
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "avg");

  // count(integer) should return BIGINT, not INTEGER
  auto countWrongReturnExpr = std::make_shared<core::CallTypedExpr>(
      INTEGER(), // Wrong return type - should be BIGINT
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "count");

  // min(integer) should return INTEGER, not VARCHAR
  auto minWrongReturnExpr = std::make_shared<core::CallTypedExpr>(
      VARCHAR(), // Wrong return type - should be INTEGER (min preserves input
                 // type)
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(INTEGER(), "c2")},
      "min");

  // Without return type validation, these would incorrectly pass
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*sumWrongReturnExpr, queryCtx_.get()));
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*avgWrongReturnExpr, queryCtx_.get()));
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*countWrongReturnExpr, queryCtx_.get()));
  ASSERT_FALSE(
      canAggregationBeEvaluatedByCudf(*minWrongReturnExpr, queryCtx_.get()));
}

} // namespace
