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

#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"

namespace facebook::velox::functions {

template <typename T>
struct CardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<HyperLogLog>& hll) {
    if (common::hll::SparseHlls::canDeserialize(hll.data())) {
      result = common::hll::SparseHlls::cardinality(hll.data());
    } else {
      result = common::hll::DenseHlls::cardinality(hll.data());
    }
    return true;
  }
};

template <typename T>
struct EmptyApproxSetFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(out_type<HyperLogLog>& result) {
    static const std::string kEmpty =
        common::hll::SparseHlls::serializeEmpty(12);

    result.resize(kEmpty.size());
    memcpy(result.data(), kEmpty.data(), kEmpty.size());
    return true;
  }
};

template <typename T>
struct EmptyApproxSetWithMaxErrorFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  std::string serialized_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const double* maxStandardError) {
    VELOX_USER_CHECK_NOT_NULL(
        maxStandardError,
        "empty_approx_set function requires constant value for maxStandardError argument");
    common::hll::checkMaxStandardError(*maxStandardError);
    serialized_ = common::hll::SparseHlls::serializeEmpty(
        common::hll::toIndexBitLength(*maxStandardError));
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<HyperLogLog>& result,
      double /*maxStandardError*/) {
    result.resize(serialized_.size());
    memcpy(result.data(), serialized_.data(), serialized_.size());
    return true;
  }
};

template <typename T>
struct MergeHllFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      memory::MemoryPool* pool,
      const arg_type<Array<HyperLogLog>>* /*constantArg*/) {
    VELOX_USER_CHECK_NOT_NULL(
        pool, "merge_hll requires MemoryPool for HashStringAllocator");
    allocator_ = std::make_unique<HashStringAllocator>(pool);
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<HyperLogLog>& result,
      const arg_type<Array<HyperLogLog>>& hllArray) {
    using SparseHll = common::hll::SparseHll<HashStringAllocator>;
    using DenseHll = common::hll::DenseHll<HashStringAllocator>;

    std::optional<int8_t> indexBitLength;
    std::optional<SparseHll> sparseHll;
    std::optional<DenseHll> denseHll;

    for (const auto& hllData : hllArray) {
      // Skip null elements
      if (!hllData.has_value()) {
        continue;
      }

      // Skip empty HLL data
      const auto& hllValue = hllData.value();
      if (hllValue.empty()) {
        continue;
      }

      const bool isSparseHllData =
          common::hll::SparseHlls::canDeserialize(hllValue.data());
      const bool isDenseHllData = !isSparseHllData &&
          common::hll::DenseHlls::canDeserialize(hllValue.data());

      VELOX_USER_CHECK(
          isSparseHllData || isDenseHllData, "Invalid HLL data format");

      const auto hllIndexBitLength = isSparseHllData
          ? common::hll::SparseHlls::deserializeIndexBitLength(hllValue.data())
          : common::hll::DenseHlls::deserializeIndexBitLength(hllValue.data());

      if (indexBitLength.has_value()) {
        VELOX_USER_CHECK_EQ(
            indexBitLength.value(),
            hllIndexBitLength,
            "Cannot merge HLLs with different indexBitLength");
      } else {
        indexBitLength = hllIndexBitLength;
      }

      if (isSparseHllData) {
        if (denseHll.has_value()) {
          SparseHll temp{hllValue.data(), allocator_.get()};
          temp.toDense(*denseHll);
        } else {
          if (!sparseHll.has_value()) {
            sparseHll.emplace(allocator_.get());
            sparseHll->setSoftMemoryLimit(
                common::hll::DenseHlls::estimateInMemorySize(
                    indexBitLength.value()));
          }

          sparseHll->mergeWith(hllValue.data());
          if (sparseHll->overLimit()) {
            denseHll.emplace(allocator_.get());
            denseHll->initialize(indexBitLength.value());
            sparseHll->toDense(*denseHll);
            sparseHll.reset();
          }
        }
      } else {
        if (sparseHll.has_value()) {
          denseHll.emplace(allocator_.get());
          denseHll->initialize(indexBitLength.value());
          sparseHll->toDense(*denseHll);
          sparseHll.reset();
        } else if (!denseHll.has_value()) {
          denseHll.emplace(allocator_.get());
          denseHll->initialize(indexBitLength.value());
        }
        denseHll->mergeWith(hllValue.data());
      }
    }

    // Return null if no valid HLLs were merged
    if (!sparseHll.has_value() && !denseHll.has_value()) {
      return false;
    }

    // Serialize the result
    if (sparseHll.has_value()) {
      auto size = sparseHll->serializedSize();
      result.resize(size);
      sparseHll->serialize(indexBitLength.value(), result.data());
    } else {
      auto size = denseHll->serializedSize();
      result.resize(size);
      denseHll->serialize(result.data());
    }

    return true;
  }
};

} // namespace facebook::velox::functions
