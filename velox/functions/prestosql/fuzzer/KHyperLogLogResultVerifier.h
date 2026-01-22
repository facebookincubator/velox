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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/KHyperLogLog.h"

namespace facebook::velox::exec::test {

/// Result verifier for khyperloglog_agg aggregate function.
/// Compares two KHyperLogLog results by deserializing and comparing their
/// cardinality estimates.
class KHyperLogLogResultVerifier : public ResultVerifier {
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

    auto projection = keys_;
    projection.push_back(resultName_);

    auto sortByGroupingKeys = [&](const RowVectorPtr& input) {
      auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      auto builder = PlanBuilder(planNodeIdGenerator).values({input});
      if (!keys_.empty()) {
        builder = builder.orderBy(keys_, false);
      }
      auto sortByKeys = builder.project(projection).planNode();
      return AssertQueryBuilder(sortByKeys).copyResults(input->pool());
    };

    auto sortedResult = sortByGroupingKeys(result);
    auto sortedAltResult = sortByGroupingKeys(altResult);

    VELOX_CHECK_EQ(sortedResult->size(), sortedAltResult->size());
    auto size = sortedResult->size();
    for (auto i = 0; i < size; i++) {
      auto resultIsNull = sortedResult->childAt(resultName_)->isNullAt(i);
      auto altResultIsNull = sortedAltResult->childAt(resultName_)->isNullAt(i);
      if (resultIsNull || altResultIsNull) {
        VELOX_CHECK(resultIsNull && altResultIsNull);
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
      } else {
        checkEquivalentKHyperLogLog(resultValue, altResultValue);
      }
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
  // Relative error tolerance for approximate cardinality estimates.
  static constexpr double kRelativeErrorTolerance = 0.05;

  void checkEquivalentKHyperLogLog(
      const StringView& result,
      const StringView& altResult) {
    auto pool = memory::MemoryManager::getInstance()->addLeafPool(
        "KHyperLogLogResultVerifier");
    HashStringAllocator allocator(pool.get());

    auto resultKhllExpected =
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
            result.data(), result.size(), &allocator);
    VELOX_CHECK(
        resultKhllExpected.hasValue(),
        "Failed to deserialize KHyperLogLog result");
    auto& resultKhll = resultKhllExpected.value();

    auto altResultKhllExpected =
        common::hll::KHyperLogLog<int64_t, HashStringAllocator>::deserialize(
            altResult.data(), altResult.size(), &allocator);
    VELOX_CHECK(
        altResultKhllExpected.hasValue(),
        "Failed to deserialize KHyperLogLog altResult");
    auto& altResultKhll = altResultKhllExpected.value();

    int64_t resultCardinality = resultKhll->cardinality();
    int64_t altResultCardinality = altResultKhll->cardinality();

    // Check that both are in the same mode (exact or approximate).
    VELOX_CHECK_EQ(resultKhll->isExact(), altResultKhll->isExact());

    if (resultKhll->isExact()) {
      // In exact mode, cardinalities must match exactly.
      VELOX_CHECK_EQ(
          resultCardinality,
          altResultCardinality,
          "KHyperLogLog cardinality should be exact but differs: {} vs. {}",
          resultCardinality,
          altResultCardinality);
    } else {
      // In approximate mode, allow tolerance for probabilistic data structure,
      // 5% relative error or absolute difference of 1 for small cardinalities.
      int64_t maxCardinality =
          std::max(resultCardinality, altResultCardinality);
      int64_t allowedDifference = std::max(
          static_cast<int64_t>(1),
          static_cast<int64_t>(maxCardinality * kRelativeErrorTolerance));

      int64_t actualDifference =
          std::abs(resultCardinality - altResultCardinality);

      VELOX_CHECK_LE(
          actualDifference,
          allowedDifference,
          "KHyperLogLog cardinality estimates differ: {} vs. {}",
          resultCardinality,
          altResultCardinality);
    }
  }

  std::vector<std::string> keys_;
  std::string resultName_;
};

} // namespace facebook::velox::exec::test
