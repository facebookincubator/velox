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

#include <string>

#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/fuzzer/NoisyCountIfResultVerifier.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::test {

class NoisyCountResultVerifier : public NoisyCountIfResultVerifier {
 public:
  void initialize(
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& aggregateName) override {
    VELOX_CHECK(!input.empty());
    // Extract the noise scale from the function call.
    extractNoiseScale(input[0]);

    // Extract the column name to aggregate on
    const auto& args = aggregate.call->inputs();
    auto field = core::TypedExprs::asFieldAccess(args[0]);
    VELOX_CHECK_NOT_NULL(field);
    aggregateColumn_ = field->name();

    groupingKeys_ = groupingKeys;
    name_ = aggregateName;

    // Create a function to get the expected result without noise.
    auto countCall =
        fmt::format("noisy_count_gaussian({}, 0.0)", aggregateColumn_);

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
                 .markDistinct(
                     "distinct", {aggregateColumn_}) // Add a boolean column to
                                                     // mark distinct rows
                 .singleAggregation(groupingKeys, {countCall}, mask)
                 .planNode();
    } else {
      plan = PlanBuilder()
                 .values(input)
                 .projectExpressions(projections)
                 .singleAggregation(groupingKeys, {countCall}, mask)
                 .planNode();
    }

    expectedNoNoise_ = AssertQueryBuilder(plan).copyResults(input[0]->pool());
  }
};

} // namespace facebook::velox::exec::test
