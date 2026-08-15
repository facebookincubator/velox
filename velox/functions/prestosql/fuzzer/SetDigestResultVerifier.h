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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/SetDigest.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SimpleVector.h"

namespace facebook::velox::exec::test {

class SetDigestResultVerifier : public ResultVerifier {
 public:
  bool supportsCompare() override {
    return true;
  }

  bool supportsVerify() override {
    return false;
  }

  void initialize(
      const std::vector<RowVectorPtr>& /*input*/,
      const std::vector<core::ExprPtr>& /*projections*/,
      const std::vector<std::string>& groupingKeys,
      const core::AggregationNode::Aggregate& /*aggregate*/,
      const std::string& aggregateName) override {
    keys_ = groupingKeys;
    resultName_ = aggregateName;
  }

  void initializeWindow(
      const std::vector<RowVectorPtr>& /*input*/,
      const std::vector<core::ExprPtr>& /*projections*/,
      const std::vector<std::string>& /*partitionByKeys*/,
      const std::vector<SortingKeyAndOrder>& /*sortingKeysAndOrders*/,
      const core::WindowNode::Function& /*function*/,
      const std::string& /*frame*/,
      const std::string& windowName) override {
    keys_ = {"row_number"};
    resultName_ = windowName;
  }

  bool compare(const RowVectorPtr& result, const RowVectorPtr& altResult)
      override {
    VELOX_CHECK_EQ(result->size(), altResult->size());

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto sortByGroupingKeys = [&](const RowVectorPtr& data) {
      auto projection = keys_;
      projection.push_back(resultName_);

      auto builder = PlanBuilder(planNodeIdGenerator).values({data});
      if (!keys_.empty()) {
        builder = builder.orderBy(keys_, false);
      }
      auto sortByKeys = builder.project(projection).planNode();
      return AssertQueryBuilder(sortByKeys).copyResults(data->pool());
    };

    auto sortedResult = sortByGroupingKeys(result);
    auto sortedAltResult = sortByGroupingKeys(altResult);

    VELOX_CHECK_EQ(sortedResult->size(), sortedAltResult->size());
    auto size = sortedResult->size();
    for (auto i = 0; i < size; i++) {
      auto resultIsNull = sortedResult->childAt(resultName_)->isNullAt(i);
      auto altResultIsNull = sortedAltResult->childAt(resultName_)->isNullAt(i);
      if (resultIsNull || altResultIsNull) {
        VELOX_CHECK(
            resultIsNull && altResultIsNull,
            "Null mismatch at row {}: result={}, altResult={}",
            i,
            resultIsNull,
            altResultIsNull);
        continue;
      }

      auto resultValue = sortedResult->childAt(resultName_)
                             ->as<SimpleVector<StringView>>()
                             ->valueAt(i);
      auto altResultValue = sortedAltResult->childAt(resultName_)
                                ->as<SimpleVector<StringView>>()
                                ->valueAt(i);

      if (resultValue == altResultValue) {
        continue;
      }

      checkEquivalentSetDigest(resultValue, altResultValue);
    }
    return true;
  }

  bool verify(const RowVectorPtr& /*result*/) override {
    VELOX_UNSUPPORTED();
  }

  void reset() override {
    keys_.clear();
    resultName_.clear();
  }

 private:
  void checkEquivalentSetDigest(
      const StringView& result,
      const StringView& altResult) {
    // Create a local pool and allocator for these SetDigest objects.
    // SetDigest objects hold references to allocator memory, so the allocator
    // must outlive them to avoid use-after-free.
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    facebook::velox::functions::SetDigest<int64_t> resultDigest(&allocator);
    facebook::velox::functions::SetDigest<int64_t> altResultDigest(&allocator);

    // Deserialize SetDigests
    auto resultStatus = resultDigest.deserialize(
        result.data(), static_cast<int32_t>(result.size()));
    VELOX_CHECK(
        resultStatus.ok(),
        "Failed to deserialize result SetDigest: {}",
        resultStatus.message());

    auto altResultStatus = altResultDigest.deserialize(
        altResult.data(), static_cast<int32_t>(altResult.size()));
    VELOX_CHECK(
        altResultStatus.ok(),
        "Failed to deserialize altResult SetDigest: {}",
        altResultStatus.message());

    // Extract cardinality and exactness mode
    auto resultCardinality = resultDigest.cardinality();
    auto altResultCardinality = altResultDigest.cardinality();
    const bool resultIsExact = resultDigest.isExact();
    const bool altResultIsExact = altResultDigest.isExact();

    // For exact mode, cardinalities must match exactly
    if (resultIsExact && altResultIsExact) {
      VELOX_CHECK_EQ(
          resultCardinality,
          altResultCardinality,
          "SetDigest exact cardinality mismatch: {} vs {}",
          resultCardinality,
          altResultCardinality);
      return;
    }

    // For approximate mode or mixed modes, compare with error tolerance.
    // SetDigest defaults to maxHashes=8192, so most fuzzer-generated data
    // (typically <1000 values) will be exact. When approximate mode is
    // triggered (>8192 unique values), allow 5% error tolerance.
    if (resultCardinality == 0 && altResultCardinality == 0) {
      return;
    }

    // Calculate relative error
    const double maxCardinality = std::max(
        static_cast<double>(resultCardinality),
        static_cast<double>(altResultCardinality));
    const double errorRate = std::abs(
                                 static_cast<double>(resultCardinality) -
                                 static_cast<double>(altResultCardinality)) /
        maxCardinality;

    VELOX_CHECK_LT(
        errorRate,
        kApproximateErrorTolerance,
        "SetDigest cardinality mismatch: {} vs {} (error: {:.2%})",
        resultCardinality,
        altResultCardinality,
        errorRate);
  }

  static constexpr double kApproximateErrorTolerance = 0.05;

  std::vector<std::string> keys_;
  std::string resultName_;
};

} // namespace facebook::velox::exec::test
