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
  explicit TDigestAggregateResultVerifier() {}

  bool supportsCompare() override {
    return true;
  }

  bool supportsVerify() override {
    return false;
  }

  void initialize(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& aggregate,
      const std::string& aggregateName) override {
    projections_ = groupingKeys;
  }

  void initializeWindow(
      const std::vector<RowVectorPtr>& input,
      const std::vector<std::string>& /*partitionByKeys*/,
      const std::vector<SortingKeyAndOrder>& /*sortingKeysAndOrders*/,
      const core::WindowNode::Function& function,
      const std::string& /*frame*/,
      const std::string& windowName) override {}

  bool compare(const RowVectorPtr& result, const RowVectorPtr& altResult)
      override {
    if (projections_.empty()) {
      return compareTDigests({result}, {altResult});
    } else {
      return compareTDigests({transform(result)}, {transform(altResult)});
    }
  }

  bool compareTDigests(
      const RowVectorPtr& actual,
      const RowVectorPtr& expected) {
    auto actualTdigestVector =
        actual->childAt(0)->as<SimpleVector<StringView>>();
    auto expectedTdigestVector =
        expected->childAt(0)->as<SimpleVector<StringView>>();

    if (actualTdigestVector->size() != expectedTdigestVector->size()) {
      return false;
    }
    for (auto i = 0; i < actualTdigestVector->size(); ++i) {
      if (actualTdigestVector->isNullAt(i) &&
          expectedTdigestVector->isNullAt(i)) {
        continue;
      }
      if (actualTdigestVector->isNullAt(i) ||
          expectedTdigestVector->isNullAt(i)) {
        return false;
      }
      auto actualSerialized = actualTdigestVector->valueAt(i);
      auto expectedSerialized = expectedTdigestVector->valueAt(i);
      facebook::velox::functions::TDigest<> actualTdigest =
          createDigest(actualSerialized.data());
      facebook::velox::functions::TDigest<> expectedTdigest =
          createDigest(expectedSerialized.data());
      if (!compareTDigest(actualTdigest, expectedTdigest)) {
        return false;
      }
    }
    return true;
  }

  bool compareTDigest(
      const facebook::velox::functions::TDigest<>& actual,
      const facebook::velox::functions::TDigest<>& expected) {
    for (double quantile : kQuantiles) {
      double actualQuantile = actual.estimateQuantile(quantile);
      double expectedQuantile = expected.estimateQuantile(quantile);
      if (std::abs(actualQuantile - expectedQuantile) >
          std::max(
              kQuantileTolerance,
              std::abs(actualQuantile * kQuantileRelativeError))) {
        return false;
      }
    }
    if (std::abs(actual.sum() - expected.sum()) > kSumError) {
      return false;
    }
    return actual.min() == expected.min() && actual.max() == expected.max() &&
        actual.compression() == expected.compression();
  }

  bool verify(const RowVectorPtr& /*result*/) override {
    VELOX_UNSUPPORTED();
  }

  void reset() override {
    projections_.clear();
  }

 private:
  RowVectorPtr transform(const RowVectorPtr& data) {
    auto plan = PlanBuilder().values({data}).project(projections_).planNode();
    return AssertQueryBuilder(plan).copyResults(data->pool());
  }

  facebook::velox::functions::TDigest<> createDigest(const char* inputData) {
    facebook::velox::functions::TDigest<> tdigest;
    std::vector<int16_t> positions;
    tdigest.mergeDeserialized(positions, inputData);
    tdigest.compress(positions);
    return tdigest;
  }

  std::vector<std::string> projections_;
  std::vector<facebook::velox::functions::TDigest<>> expectedDigests_;
  std::vector<RowVectorPtr> input_;
  std::vector<std::string> groupingKeys_;
  double value_{};
  double weight_{};
  double compression_{};
  static constexpr double kSumError = 1e-4;
  static constexpr double kQuantileTolerance = 0.5;
  static constexpr double kQuantileRelativeError = 0.01;
  static constexpr double kQuantiles[] = {
      0.01,
      0.05,
      0.1,
      0.25,
      0.50,
      0.75,
      0.9,
      0.95,
      0.99,
  };
};

} // namespace facebook::velox::exec::test
