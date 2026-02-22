/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/base/Status.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/Macros.h"
#include "velox/functions/lib/KHyperLogLog.h"
#include "velox/functions/prestosql/types/KHyperLogLogType.h"

namespace facebook::velox::functions {

template <typename T>
struct KHyperLogLogCardinalityFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog cardinality requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status
  call(int64_t& result, const arg_type<KHyperLogLog>& khllData) {
    auto khllInstance = KHyperLogLogUtils::deserialize(
        khllData.data(), khllData.size(), allocator_.get());
    if (khllInstance.hasError()) {
      return khllInstance.error();
    }
    result = khllInstance.value()->cardinality();
    return Status::OK();
  }
};

template <typename T>
struct KHyperLogLogIntersectionCardinalityFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg1*/,
      const arg_type<KHyperLogLog>* /*constantArg2*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog intersection_cardinality requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      int64_t& result,
      const arg_type<KHyperLogLog>& khll1Data,
      const arg_type<KHyperLogLog>& khll2Data) {
    auto khll1Instance = KHyperLogLogUtils::deserialize(
        khll1Data.data(), khll1Data.size(), allocator_.get());
    if (khll1Instance.hasError()) {
      return khll1Instance.error();
    }
    auto khll1 = std::move(khll1Instance.value());
    auto khll2Instance = KHyperLogLogUtils::deserialize(
        khll2Data.data(), khll2Data.size(), allocator_.get());
    if (khll2Instance.hasError()) {
      return khll2Instance.error();
    }
    auto khll2 = std::move(khll2Instance.value());

    // If both khlls are exact, return the exact intersection cardinality.
    if (khll1->isExact() && khll2->isExact()) {
      result = KHyperLogLogUtils::exactIntersectionCardinality(*khll1, *khll2);
      return Status::OK();
    }

    // If either of the khlls are not exact, return an approximation of the
    // intersection cardinality using the Jaccard Index like a similarity
    // index between the 2 key sets.
    int64_t lowestCardinality =
        std::min(khll1->cardinality(), khll2->cardinality());
    double jaccard = KHyperLogLogUtils::jaccardIndex(*khll1, *khll2);
    auto setUnion =
        KHyperLogLogUtils::merge(std::move(khll1), std::move(khll2));
    int64_t computedResult =
        static_cast<int64_t>(std::round(jaccard * setUnion->cardinality()));

    // In a special case where one set is much smaller and almost a true
    // subset of the other, return the size of the smaller set. For example:
    // Set1 = {1,2,3,4,5,6}
    // Set2 = {1,2}
    // Jaccard Index = 1
    // Approximated intersection cardinality = 6
    // This result does not make sense as Set2 does not even have 6
    // elements. Thus return 2 instead.
    result = std::min(computedResult, lowestCardinality);
    return Status::OK();
  }
};

template <typename T>
struct KHyperLogLogJaccardIndexFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg1*/,
      const arg_type<KHyperLogLog>* /*constantArg2*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog jaccard_index requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      double& result,
      const arg_type<KHyperLogLog>& khll1Data,
      const arg_type<KHyperLogLog>& khll2Data) {
    auto khll1Instance = KHyperLogLogUtils::deserialize(
        khll1Data.data(), khll1Data.size(), allocator_.get());
    if (khll1Instance.hasError()) {
      return khll1Instance.error();
    }
    auto khll2Instance = KHyperLogLogUtils::deserialize(
        khll2Data.data(), khll2Data.size(), allocator_.get());
    if (khll2Instance.hasError()) {
      return khll2Instance.error();
    }

    result = KHyperLogLogUtils::jaccardIndex(
        *khll1Instance.value(), *khll2Instance.value());
    return Status::OK();
  }
};

template <typename T>
struct KHyperLogLogReidentificationPotentialFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg1*/,
      const int64_t* /*constantArg2*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog reidentification_potential requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      double& result,
      const arg_type<KHyperLogLog>& khllData,
      int64_t threshold) {
    auto khllInstance = KHyperLogLogUtils::deserialize(
        khllData.data(), khllData.size(), allocator_.get());
    if (khllInstance.hasError()) {
      return khllInstance.error();
    }
    result = khllInstance.value()->reidentificationPotential(threshold);
    return Status::OK();
  }
};

template <typename T>
struct KHyperLogLogUniquenessDistributionFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog uniqueness_distribution requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<KHyperLogLog>* /*constantArg1*/,
      const int64_t* /*constantArg2*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog uniqueness_distribution requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Map<int64_t, double>>& result,
      const arg_type<KHyperLogLog>& khllData) {
    // When histogramSize is not provided, use the minhash size of the KHLL
    // instance as the default.
    auto khllInstance = KHyperLogLogUtils::deserialize(
        khllData.data(), khllData.size(), allocator_.get());
    if (khllInstance.hasError()) {
      return khllInstance.error();
    }
    int64_t histogramSize =
        static_cast<int64_t>(khllInstance.value()->minhashSize());
    return uniquenessDistribution(result, *khllInstance.value(), histogramSize);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Map<int64_t, double>>& result,
      const arg_type<KHyperLogLog>& khllData,
      int64_t histogramSize) {
    auto khllInstance = KHyperLogLogUtils::deserialize(
        khllData.data(), khllData.size(), allocator_.get());
    if (khllInstance.hasError()) {
      return khllInstance.error();
    }
    return uniquenessDistribution(result, *khllInstance.value(), histogramSize);
  }

 private:
  FOLLY_ALWAYS_INLINE Status uniquenessDistribution(
      out_type<Map<int64_t, double>>& result,
      KHyperLogLogUtils& khll,
      int64_t histogramSize) {
    auto distribution = khll.uniquenessDistribution(histogramSize);

    // Sort the distribution by key to ensure sorted output
    std::vector<std::pair<int64_t, double>> sortedDistribution(
        distribution.begin(), distribution.end());
    std::sort(
        sortedDistribution.begin(),
        sortedDistribution.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Populate the result map
    for (const auto& [key, value] : sortedDistribution) {
      result.add_item() = std::make_tuple(key, value);
    }

    return Status::OK();
  }
};

template <typename T>
struct MergeKHyperLogLogFunction {
  using KHyperLogLogUtils =
      common::hll::KHyperLogLog<int64_t, HashStringAllocator>;
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Array<KHyperLogLog>>* /*constantArg*/) {
    VELOX_CHECK_NOT_NULL(
        pool,
        "KHyperLogLog merge_khll requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<KHyperLogLog>& result,
      const arg_type<Array<KHyperLogLog>>& hllArray) {
    std::unique_ptr<KHyperLogLogUtils> merged = nullptr;

    for (const auto& khllData : hllArray) {
      // Skip null elements
      if (!khllData.has_value()) {
        continue;
      }

      // Skip empty HLL data
      const auto& khllValue = khllData.value();
      if (khllValue.empty()) {
        continue;
      }

      if (!merged) {
        // Initialize with first non-null element.
        auto deserializeResult = KHyperLogLogUtils::deserialize(
            khllValue.data(), khllValue.size(), allocator_.get());
        VELOX_CHECK(
            deserializeResult.hasValue(),
            "Failed to deserialize KHyperLogLog: {}",
            deserializeResult.error().message());
        merged = std::move(deserializeResult.value());
      } else {
        auto currentKhll = KHyperLogLogUtils::deserialize(
            khllValue.data(), khllValue.size(), allocator_.get());
        VELOX_CHECK(
            currentKhll.hasValue(),
            "Failed to deserialize KHyperLogLog: {}",
            currentKhll.error().message());
        merged->mergeWith(*currentKhll.value());
      }
    }

    // Return null if all elements were null.
    if (!merged) {
      return false;
    }

    size_t serializedSize = merged->estimatedSerializedSize();
    result.resize(serializedSize);
    merged->serialize(result.data());

    return true;
  }
};

} // namespace facebook::velox::functions
