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

#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/Expressions.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"

#include <gtest/gtest.h>

using namespace facebook::velox;

namespace facebook::velox::cudf_velox {

class StepAwareAggregationRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    memory::MemoryManager::testingSetInstance({});
    registerCudf();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions();
    queryCtx_ = core::QueryCtx::create();
  }

  void TearDown() override {
    unregisterCudf();
  }

  std::shared_ptr<core::QueryCtx> queryCtx_;
};

TEST_F(StepAwareAggregationRegistryTest, allFunctionsAreRegistered) {
  auto& registry = getStepAwareAggregationRegistry();

  // Check that all expected functions are registered
  EXPECT_TRUE(registry.find("avg") != registry.end())
      << "avg function should be registered";
  EXPECT_TRUE(registry.find("sum") != registry.end())
      << "sum function should be registered";
  EXPECT_TRUE(registry.find("count") != registry.end())
      << "count function should be registered";
  EXPECT_TRUE(registry.find("min") != registry.end())
      << "min function should be registered";
  EXPECT_TRUE(registry.find("max") != registry.end())
      << "max function should be registered";
}

TEST_F(StepAwareAggregationRegistryTest, allStepsAreRegisteredForAvg) {
  auto& registry = getStepAwareAggregationRegistry();

  // Check that all steps are registered for avg
  auto& avgSteps = registry["avg"];
  EXPECT_TRUE(
      avgSteps.find(core::AggregationNode::Step::kSingle) != avgSteps.end())
      << "avg single step should be registered";
  EXPECT_TRUE(
      avgSteps.find(core::AggregationNode::Step::kPartial) != avgSteps.end())
      << "avg partial step should be registered";
  EXPECT_TRUE(
      avgSteps.find(core::AggregationNode::Step::kFinal) != avgSteps.end())
      << "avg final step should be registered";
  EXPECT_TRUE(
      avgSteps.find(core::AggregationNode::Step::kIntermediate) !=
      avgSteps.end())
      << "avg intermediate step should be registered";
}

TEST_F(StepAwareAggregationRegistryTest, avgHasSignaturesForAllSteps) {
  auto& registry = getStepAwareAggregationRegistry();
  auto& avgSteps = registry["avg"];

  // Test that all steps have signatures
  auto& singleSigs = avgSteps[core::AggregationNode::Step::kSingle];
  auto& partialSigs = avgSteps[core::AggregationNode::Step::kPartial];
  auto& finalSigs = avgSteps[core::AggregationNode::Step::kFinal];
  auto& intermediateSigs = avgSteps[core::AggregationNode::Step::kIntermediate];

  EXPECT_FALSE(singleSigs.empty()) << "Single step should have signatures";
  EXPECT_FALSE(partialSigs.empty()) << "Partial step should have signatures";
  EXPECT_FALSE(finalSigs.empty()) << "Final step should have signatures";
  EXPECT_FALSE(intermediateSigs.empty())
      << "Intermediate step should have signatures";
}

TEST_F(StepAwareAggregationRegistryTest, avgPartialStepHasRowReturnTypes) {
  auto& registry = getStepAwareAggregationRegistry();
  auto& avgSteps = registry["avg"];
  auto& partialSigs = avgSteps[core::AggregationNode::Step::kPartial];

  // Verify that partial signatures have ROW return types
  bool foundRowReturn = false;
  for (const auto& sig : partialSigs) {
    if (sig->returnType().baseName().find("row") != std::string::npos) {
      foundRowReturn = true;
      break;
    }
  }
  EXPECT_TRUE(foundRowReturn) << "Partial step should have ROW return types";
}

TEST_F(StepAwareAggregationRegistryTest, avgFinalStepHasRowInputTypes) {
  auto& registry = getStepAwareAggregationRegistry();
  auto& avgSteps = registry["avg"];
  auto& finalSigs = avgSteps[core::AggregationNode::Step::kFinal];

  // Verify that final signatures have ROW input types
  bool foundRowInput = false;
  for (const auto& sig : finalSigs) {
    if (!sig->argumentTypes().empty() &&
        sig->argumentTypes()[0].baseName().find("row") != std::string::npos) {
      foundRowInput = true;
      break;
    }
  }
  EXPECT_TRUE(foundRowInput) << "Final step should have ROW input types";
}

TEST_F(StepAwareAggregationRegistryTest, avgSingleStepValidation) {
  auto avgSingleCall = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "avg");

  bool singleStepSupported = canAggregationBeEvaluatedByCudf(
      *avgSingleCall,
      core::AggregationNode::Step::kSingle,
      {DOUBLE()},
      queryCtx_.get());

  EXPECT_TRUE(singleStepSupported)
      << "Single step avg(DOUBLE) should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, avgPartialStepValidation) {
  auto avgPartialCall = std::make_shared<core::CallTypedExpr>(
      ROW({DOUBLE(), BIGINT()}),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "avg");

  bool partialStepSupported = canAggregationBeEvaluatedByCudf(
      *avgPartialCall,
      core::AggregationNode::Step::kPartial,
      {DOUBLE()},
      queryCtx_.get());

  EXPECT_TRUE(partialStepSupported)
      << "Partial step avg(DOUBLE) should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, avgFinalStepValidation) {
  auto rowType = ROW({DOUBLE(), BIGINT()});
  auto avgFinalCall = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(
              rowType, "intermediate")},
      "avg");

  bool finalStepSupported = canAggregationBeEvaluatedByCudf(
      *avgFinalCall,
      core::AggregationNode::Step::kFinal,
      {rowType},
      queryCtx_.get());

  EXPECT_TRUE(finalStepSupported)
      << "Final step avg(ROW(DOUBLE,BIGINT)) should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, sumStepConsistency) {
  // Sum should have the same signatures for all steps
  auto sumCall = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "sum");

  bool sumSingle = canAggregationBeEvaluatedByCudf(
      *sumCall,
      core::AggregationNode::Step::kSingle,
      {DOUBLE()},
      queryCtx_.get());
  bool sumPartial = canAggregationBeEvaluatedByCudf(
      *sumCall,
      core::AggregationNode::Step::kPartial,
      {DOUBLE()},
      queryCtx_.get());
  bool sumFinal = canAggregationBeEvaluatedByCudf(
      *sumCall,
      core::AggregationNode::Step::kFinal,
      {DOUBLE()},
      queryCtx_.get());
  bool sumIntermediate = canAggregationBeEvaluatedByCudf(
      *sumCall,
      core::AggregationNode::Step::kIntermediate,
      {DOUBLE()},
      queryCtx_.get());

  EXPECT_TRUE(sumSingle) << "Sum single step should be supported";
  EXPECT_TRUE(sumPartial) << "Sum partial step should be supported";
  EXPECT_TRUE(sumFinal) << "Sum final step should be supported";
  EXPECT_TRUE(sumIntermediate) << "Sum intermediate step should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, countStepConsistency) {
  // Count has raw input signatures for single/partial and bigint input for
  // final/intermediate.
  auto countRawCall = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "count");
  auto countIntermediateCall = std::make_shared<core::CallTypedExpr>(
      BIGINT(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "input")},
      "count");

  bool countSingle = canAggregationBeEvaluatedByCudf(
      *countRawCall,
      core::AggregationNode::Step::kSingle,
      {DOUBLE()},
      queryCtx_.get());
  bool countPartial = canAggregationBeEvaluatedByCudf(
      *countRawCall,
      core::AggregationNode::Step::kPartial,
      {DOUBLE()},
      queryCtx_.get());
  bool countFinal = canAggregationBeEvaluatedByCudf(
      *countIntermediateCall,
      core::AggregationNode::Step::kFinal,
      {BIGINT()},
      queryCtx_.get());
  bool countIntermediate = canAggregationBeEvaluatedByCudf(
      *countIntermediateCall,
      core::AggregationNode::Step::kIntermediate,
      {BIGINT()},
      queryCtx_.get());

  EXPECT_TRUE(countSingle) << "Count single step should be supported";
  EXPECT_TRUE(countPartial) << "Count partial step should be supported";
  EXPECT_TRUE(countFinal) << "Count final step should be supported";
  EXPECT_TRUE(countIntermediate)
      << "Count intermediate step should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, minStepConsistency) {
  // Min should have the same signatures for all steps
  auto minCall = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "min");

  bool minSingle = canAggregationBeEvaluatedByCudf(
      *minCall,
      core::AggregationNode::Step::kSingle,
      {DOUBLE()},
      queryCtx_.get());
  bool minPartial = canAggregationBeEvaluatedByCudf(
      *minCall,
      core::AggregationNode::Step::kPartial,
      {DOUBLE()},
      queryCtx_.get());
  bool minFinal = canAggregationBeEvaluatedByCudf(
      *minCall,
      core::AggregationNode::Step::kFinal,
      {DOUBLE()},
      queryCtx_.get());
  bool minIntermediate = canAggregationBeEvaluatedByCudf(
      *minCall,
      core::AggregationNode::Step::kIntermediate,
      {DOUBLE()},
      queryCtx_.get());

  EXPECT_TRUE(minSingle) << "Min single step should be supported";
  EXPECT_TRUE(minPartial) << "Min partial step should be supported";
  EXPECT_TRUE(minFinal) << "Min final step should be supported";
  EXPECT_TRUE(minIntermediate) << "Min intermediate step should be supported";
}

TEST_F(StepAwareAggregationRegistryTest, maxStepConsistency) {
  // Max should have the same signatures for all steps
  auto maxCall = std::make_shared<core::CallTypedExpr>(
      DOUBLE(),
      std::vector<core::TypedExprPtr>{
          std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "input")},
      "max");

  bool maxSingle = canAggregationBeEvaluatedByCudf(
      *maxCall,
      core::AggregationNode::Step::kSingle,
      {DOUBLE()},
      queryCtx_.get());
  bool maxPartial = canAggregationBeEvaluatedByCudf(
      *maxCall,
      core::AggregationNode::Step::kPartial,
      {DOUBLE()},
      queryCtx_.get());
  bool maxFinal = canAggregationBeEvaluatedByCudf(
      *maxCall,
      core::AggregationNode::Step::kFinal,
      {DOUBLE()},
      queryCtx_.get());
  bool maxIntermediate = canAggregationBeEvaluatedByCudf(
      *maxCall,
      core::AggregationNode::Step::kIntermediate,
      {DOUBLE()},
      queryCtx_.get());

  EXPECT_TRUE(maxSingle) << "Max single step should be supported";
  EXPECT_TRUE(maxPartial) << "Max partial step should be supported";
  EXPECT_TRUE(maxFinal) << "Max final step should be supported";
  EXPECT_TRUE(maxIntermediate) << "Max intermediate step should be supported";
}

} // namespace facebook::velox::cudf_velox
