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
#include "velox/exec/fuzzer/FuzzerUtil.h"
#include "velox/exec/fuzzer/ResultVerifier.h"
#include "velox/vector/ComplexVector.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

namespace facebook::velox::exec::test {

/// Verifies aggregation results either directly or by comparing with results
/// from a logically equivalent plan or reference DB.
///
/// Can be used to sort results of array_agg before comparing (uses 'compare'
/// API) or verify approx_distinct by comparing its results with results of
/// count(distinct) (uses 'verify' API).
class ScalarResultVerifier : public ResultVerifier {
 public:
  /// Returns true if 'compare' API is supported. The verifier must support
  /// either 'compare' or 'verify' API. If both are supported, 'compare' API is
  /// used and 'verify' API is ignored.
  bool supportsCompare() override {
    return true;
  };

  /// Return true if 'verify' API is support. The verifier must support either
  /// 'compare' or 'verify' API.
  bool supportsVerify() override {
    return false;
  };

  /// Called once before possibly multiple calls to 'compare' or 'verify' APIs
  /// to specify the input data, grouping keys (may be empty), the aggregate
  /// function and the name of the column that will store aggregate function
  /// results.
  ///
  /// Can be used by array_distinct verifier to compute count(distinct) once and
  /// re-use its results for multiple 'verify' calls.
  void initialize(
      const std::vector<RowVectorPtr>& /*input*/,
      const std::vector<core::ExprPtr>& /*projections*/,
      const std::vector<std::string>& /*groupingKeys*/,
      const core::AggregationNode::Aggregate& /*aggregate*/,
      const std::string& /*aggregateName*/) override {
    VELOX_NYI();
  }

  /// Compares results of two logically equivalent Velox plans or a Velox plan
  /// and a reference DB query.
  ///
  /// 'initialize' must be called first. 'compare' may be called multiple times
  /// after single 'initialize' call.
  bool compare(const RowVectorPtr& result, const RowVectorPtr& altResult)
      override {
    // if (projections_.empty()) {
    //   return assertEqualResults({result}, {altResult});
    // } else {
    //   return assertEqualResults({transform(result)}, {transform(altResult)});
    // }
    return assertEqualResults({result}, {altResult});
  }

  /// Verifies results of a Velox plan or reference DB query.
  ///
  /// 'initialize' must be called first. 'verify' may be called multiple times
  /// after single 'initialize' call.
  bool verify(const RowVectorPtr& result) override {
    VELOX_UNSUPPORTED();
  }

  /// Clears internal state after possibly multiple calls to 'compare' and
  /// 'verify'. 'initialize' must be called again after 'reset' to allow calling
  /// 'compare' or 'verify' again.
  void reset() override {
    VELOX_NYI();
  }

  // Compare two arrays of doubles for relative equality within epsilon.
  bool arraysRelativelyEqual(
      const VectorPtr& left,
      const VectorPtr& right,
      vector_size_t size,
      double epsilon) {
    auto leftFlat = left->as<RowVector<double>>();
    auto rightFlat = right->as<RowVector<double>>();
    for (vector_size_t i = 0; i < size; ++i) {
      if (leftFlat->containsNullAt(i) || rightFlat->containsNullAt(i)) {
        // Optionally, handle nulls as needed (skip or require both null).
        if (leftFlat->containsNullAt(i) && rightFlat->containsNullAt(i)) {
          return false;
        }
      }
      double l = leftFlat->valueAt(i);
      double r = rightFlat->valueAt(i);
      double diff = std::abs(l - r);
      double denom = std::max(std::abs(l), std::abs(r));
      if (denom == 0.0) {
        if (diff > epsilon) {
          return false;
        }
      } else if (diff / denom > epsilon) {
        return false;
      }
    }
    return true;
  };
};

} // namespace facebook::velox::exec::test
