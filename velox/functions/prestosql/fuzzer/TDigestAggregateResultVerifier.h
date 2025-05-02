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

#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/TDigest.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::test {

class TDigestAggregateResultVerifier : public ResultVerifier {
 public:
  TDigestAggregateResultVerifier() = default;

  bool supportsCompare() override {
    return true;
  }

  bool supportsVerify() override {
    return false;
  }

  void initialize(
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& aggregateName) override {
    input_ = input;
    groupingKeys_ = groupingKeys;
    name_ = aggregateName;
    filterMask_.clear();
    orderByField_.clear();

    // Initialize projections with grouping keys and aggregate name
    projections_ = groupingKeys;
    projections_.push_back(name_);

    // Store filter mask if present
    if (aggregate.mask) {
      if (auto field =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  aggregate.mask)) {
        filterMask_ = field->name();
      }
    }

    // Store ordering information if present
    if (aggregate.call && !aggregate.call->inputs().empty()) {
      auto& inputs = aggregate.call->inputs();
      if (inputs.size() > 0) {
        if (auto field =
                std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                    inputs[0])) {
          orderByField_ = field->name();
        }
      }
    }
  }

  void initializeWindow(
      const std::vector<RowVectorPtr>& input,
      const std::vector<core::ExprPtr>& projections,
      const std::vector<std::string>& /*partitionByKeys*/,
      const std::vector<SortingKeyAndOrder>& /*sortingKeysAndOrders*/,
      const core::WindowNode::Function& function,
      const std::string& /*frame*/,
      const std::string& windowName) override {
    reset();
    input_ = input;
    for (const auto& expr : projections) {
      if (auto field =
              std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(
                  expr)) {
        projections_.push_back(field->name());
      }
    }
    if (projections_.empty()) {
      projections_ = asRowType(input[0]->type())->names();
    }
  }

  bool compare(const RowVectorPtr& result, const RowVectorPtr& altResult)
      override {
    return compareSingleTDigest(result, altResult);
  }

  bool compareSingleTDigest(
      const RowVectorPtr& actual,
      const RowVectorPtr& expected) {
    if (!actual && !expected) {
      return true;
    }
    if (!actual || !expected) {
      return false;
    }
    auto actualTdigestVector =
        actual->childAt(0)->as<SimpleVector<StringView>>();
    auto expectedTdigestVector =
        expected->childAt(0)->as<SimpleVector<StringView>>();
    if (!actualTdigestVector && !expectedTdigestVector) {
      return true;
    }
    if (!actualTdigestVector || !expectedTdigestVector) {
      return false;
    }
    if (actualTdigestVector->isNullAt(0) &&
        expectedTdigestVector->isNullAt(0)) {
      return true;
    }
    if (actualTdigestVector->isNullAt(0) ||
        expectedTdigestVector->isNullAt(0)) {
      return false;
    }
    auto actualSerialized = actualTdigestVector->valueAt(0);
    auto expectedSerialized = expectedTdigestVector->valueAt(0);
    try {
      facebook::velox::functions::TDigest<> actualTdigest =
          createDigest(actualSerialized.data());
      facebook::velox::functions::TDigest<> expectedTdigest =
          createDigest(expectedSerialized.data());
      return compareTDigestValues(actualTdigest, expectedTdigest);
    } catch (const std::exception& e) {
      // Consider false, if can't deserialize
      return false;
    }
  }

  bool compareTDigestValues(
      const facebook::velox::functions::TDigest<>& actual,
      const facebook::velox::functions::TDigest<>& expected) {
    // Compare sum given tolerance
    if (std::abs(actual.sum() - expected.sum()) > kError) {
      return false;
    }
    // Compare other TDigest properties
    return actual.totalWeight() == expected.totalWeight() &&
        actual.min() == expected.min() && actual.max() == expected.max() &&
        actual.compression() == expected.compression();
  }

  bool verify(const RowVectorPtr& /*result*/) override {
    VELOX_UNSUPPORTED();
  }

  void reset() override {
    projections_.clear();
    input_.clear();
    groupingKeys_.clear();
    filterMask_.clear();
    orderByField_.clear();
  }

 private:
  static constexpr double kError = 0.0001;

  std::vector<std::string> projections_;
  std::vector<RowVectorPtr> input_;
  std::vector<std::string> groupingKeys_;
  std::string name_;
  std::string filterMask_;
  std::string orderByField_;

  facebook::velox::functions::TDigest<> createDigest(const char* inputData) {
    VELOX_CHECK_NOT_NULL(inputData, "TDigest input data cannot be null");
    facebook::velox::functions::TDigest<> tdigest;
    std::vector<int16_t> positions;
    try {
      tdigest.mergeDeserialized(positions, inputData);
      tdigest.compress(positions);
    } catch (const std::exception& e) {
      VELOX_FAIL("Failed to deserialize TDigest: {}", e.what());
    }
    return tdigest;
  }

  RowVectorPtr transform(const RowVectorPtr& data) {
    VELOX_CHECK(!projections_.empty());
    auto builder = PlanBuilder().values({data});
    if (!filterMask_.empty()) {
      builder.filter(filterMask_);
    }
    if (!orderByField_.empty()) {
      std::vector<std::string> orderBy = {orderByField_};
      builder.orderBy(orderBy, true);
    }
    builder.project(projections_);
    auto plan = builder.planNode();
    return AssertQueryBuilder(plan).copyResults(data->pool());
  }
};

} // namespace facebook::velox::exec::test
