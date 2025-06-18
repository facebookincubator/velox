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

#pragma once

#include <cstdint>
#include <string>
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::test {

class NoisySumResultVerifier : public ResultVerifier {
 public:
  bool supportsCompare() override {
    return false;
  }

  bool supportsVerify() override {
    return true;
  }

  void initialize(
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& aggregateName) override {
    VELOX_CHECK(!input.empty());
    // Extract the noise scale from the function call.

    // Extract the column name to aggregate on
    const auto& args = aggregate.call->inputs();
    extractNoiseScaleAndBound(input[0], args);

    auto field = core::TypedExprs::asFieldAccess(args[0]);
    VELOX_CHECK_NOT_NULL(field);
    aggregateColumn_ = field->name();

    groupingKeys_ = groupingKeys;
    name_ = aggregateName;

    std::ostringstream call;
    call << "noisy_sum_gaussian(" << aggregateColumn_ << ", " << "0.0";
    if (lowerBound_.has_value() && upperBound_.has_value()) {
      call << ", " << lowerBound_.value();
      call << ", " << upperBound_.value();
    }
    call << ")";
    auto sumCall = call.str();

    // Add filter mask if exists.
    std::vector<std::string> mask;
    if (aggregate.mask != nullptr) {
      mask.push_back(aggregate.mask->name());
    }

    core::PlanNodePtr plan;
    // Handle distinct case.
    // Aggregation mask should be specified only once
    // (either explicitly or using FILTER clause)
    if (aggregate.distinct) {
      mask.emplace_back("distinct");
      plan = PlanBuilder()
                 .values(input)
                 .markDistinct("distinct", {aggregateColumn_})
                 .singleAggregation(groupingKeys, {sumCall}, mask)
                 .planNode();
    } else {
      plan = PlanBuilder()
                 .values(input)
                 .projectExpressions(projections)
                 .singleAggregation(groupingKeys, {sumCall}, mask)
                 .planNode();
    }

    expectedNoNoise_ = AssertQueryBuilder(plan).copyResults(input[0]->pool());
  }

  bool compare(
      [[maybe_unused]] const RowVectorPtr& result,
      [[maybe_unused]] const RowVectorPtr& otherResult) override {
    VELOX_UNSUPPORTED();
  }

  bool verify(const RowVectorPtr& result) override {
    // The expected result and actual result are grouped by the same keys,
    // but the rows may be in different order. So we need to union the results.
    // Create sources for expected and actual results
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto expectedSource = PlanBuilder(planNodeIdGenerator)
                              .values({expectedNoNoise_})
                              .appendColumns({"'expected' as label"})
                              .planNode();
    auto actualSource = PlanBuilder(planNodeIdGenerator)
                            .values({result})
                            .appendColumns({"'actual' as label"})
                            .planNode();

    // Combine expected and actual results by grouping keys using map_agg
    auto mapAgg = fmt::format("map_agg(label, {}) as m", name_);
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .localPartition({}, {expectedSource, actualSource})
                    .singleAggregation(groupingKeys_, {mapAgg})
                    .project({"m['actual'] as a", "m['expected'] as e"})
                    .planNode();
    auto combined = AssertQueryBuilder(plan).copyResults(result->pool());

    // Extract actual and expected values
    auto* actual = combined->childAt(0)->as<SimpleVector<double>>();
    auto* expected = combined->childAt(1)->as<SimpleVector<double>>();

    const auto numGroups = result->size();
    VELOX_CHECK_EQ(numGroups, combined->size());

    // Calculate allowed difference based on noise scale
    const int64_t deviationMultiple = 50;
    const double allowedFailureRate = 0.001;
    const auto allowedDifference = deviationMultiple * noiseScale_;
    const auto lowerBound = -allowedDifference;
    const auto upperBound = allowedDifference;

    // Check each group's result
    int failures = 0;
    for (auto i = 0; i < numGroups; ++i) {
      // Skip verification for null rows
      if (expected->isNullAt(i) || actual->isNullAt(i)) {
        continue;
      }

      const auto actualValue = actual->valueAt(i);
      const auto expectedValue = expected->valueAt(i);
      const auto difference = actualValue - expectedValue;

      // Check if actual value is within expected +/- allowedDifference
      if (difference < lowerBound || difference > upperBound) {
        LOG(ERROR) << fmt::format(
            "noisy_sum_gaussian result is outside the expected range.\n"
            "  Group: {}\n"
            "  Actual: {}\n"
            "  Expected: {}\n"
            "  Difference: {}\n"
            "  Allowed range: [{}, {}] (noise_scale = {})",
            i,
            actualValue,
            expectedValue,
            difference,
            expectedValue + lowerBound,
            expectedValue + upperBound,
            noiseScale_);
        failures++;
      }
    }

    // Allow a very small percentage of failures for large result sets
    if (numGroups >= 50) {
      const auto maxFailures = static_cast<int>(allowedFailureRate * numGroups);
      if (failures > maxFailures) {
        LOG(ERROR) << fmt::format(
            "Too many failures: {} out of {} groups (max allowed: {})",
            failures,
            numGroups,
            maxFailures);
        return false;
      }
      return true;
    }

    // For small result sets, require all groups to pass
    return failures == 0;
  }

  void reset() override {
    noiseScale_ = 0.0;
    lowerBound_.reset();
    upperBound_.reset();
    name_.clear();
    groupingKeys_.clear();
    aggregateColumn_.clear();
    expectedNoNoise_.reset();
  }

 private:
  void extractNoiseScaleAndBound(
      const RowVectorPtr& input,
      const std::vector<core::TypedExprPtr>& args) {
    auto secondArg = input->childAt(1);
    if (secondArg->type()->isDouble()) {
      noiseScale_ = secondArg->as<SimpleVector<double>>()->valueAt(0);
    } else if (secondArg->type()->isBigint()) {
      noiseScale_ = static_cast<double>(
          secondArg->as<SimpleVector<int64_t>>()->valueAt(0));
    }

    // Extract lower and upper bound if they exist
    if (args.size() > 3) {
      auto thirdArg = input->childAt(2);
      if (thirdArg->type()->isDouble()) {
        lowerBound_ = thirdArg->as<SimpleVector<double>>()->valueAt(0);
      } else if (thirdArg->type()->isBigint()) {
        lowerBound_ = static_cast<double>(
            thirdArg->as<SimpleVector<int64_t>>()->valueAt(0));
      }

      auto fourthArg = input->childAt(3);
      if (fourthArg->type()->isDouble()) {
        upperBound_ = fourthArg->as<SimpleVector<double>>()->valueAt(0);
      } else if (fourthArg->type()->isBigint()) {
        upperBound_ = static_cast<double>(
            fourthArg->as<SimpleVector<int64_t>>()->valueAt(0));
      }
    }
  }

  double noiseScale_{0.0};
  std::optional<double> lowerBound_;
  std::optional<double> upperBound_;
  std::string name_;
  std::vector<std::string> groupingKeys_;
  std::string aggregateColumn_;
  RowVectorPtr expectedNoNoise_;
};

} // namespace facebook::velox::exec::test
