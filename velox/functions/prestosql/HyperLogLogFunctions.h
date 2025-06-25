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

#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"

namespace facebook::velox::functions {

template <typename T>
struct CardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<HyperLogLog>& hll) {
    using common::hll::DenseHll;
    using common::hll::SparseHll;

    if (SparseHll::canDeserialize(hll.data())) {
      result = SparseHll::cardinality(hll.data());
    } else {
      result = DenseHll::cardinality(hll.data());
    }
    return true;
  }
};

template <typename T>
struct EmptyApproxSetFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(out_type<HyperLogLog>& result) {
    static const std::string kEmpty =
        common::hll::SparseHll::serializeEmpty(12);

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
    serialized_ = common::hll::SparseHll::serializeEmpty(
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

  FOLLY_ALWAYS_INLINE bool call(
      out_type<HyperLogLog>& result,
      const arg_type<Array<HyperLogLog>>& input) {
    using common::hll::DenseHll;
    using common::hll::SparseHll;

    // Create a memory pool and allocator for this call
    auto pool = memory::memoryManager()->addLeafPool();
    HashStringAllocator allocator(pool.get());

    // Create a sparse HLL accumulator to merge into
    SparseHll sparseHll(&allocator);
    DenseHll denseHll(&allocator);
    bool isSparse = true;
    int8_t indexBitLength = -1;
    bool hasValidInput = false;
    // Merge HLLs
    for (auto i = 0; i < input.size(); i++) {
      if (!input[i].has_value()) {
        continue;
      }
      hasValidInput = true;
      const auto& hll = input[i].value();
      auto inputData = hll.data();
      if (SparseHll::canDeserialize(inputData)) {
        if (isSparse) {
          sparseHll.mergeWith(inputData);
          if (indexBitLength < 0) {
            indexBitLength = DenseHll::deserializeIndexBitLength(inputData);
            sparseHll.setSoftMemoryLimit(
                DenseHll::estimateInMemorySize(indexBitLength));
          }
          if (sparseHll.overLimit()) {
            // Convert to dense
            isSparse = false;
            denseHll.initialize(indexBitLength);
            sparseHll.toDense(denseHll);
            sparseHll.reset();
          }
        } else {
          SparseHll other(inputData, &allocator);
          other.toDense(denseHll);
        }
      } else if (DenseHll::canDeserialize(inputData)) {
        if (isSparse) {
          if (indexBitLength < 0) {
            indexBitLength = DenseHll::deserializeIndexBitLength(inputData);
            sparseHll.setSoftMemoryLimit(
                DenseHll::estimateInMemorySize(indexBitLength));
          }
          isSparse = false;
          denseHll.initialize(indexBitLength);
          sparseHll.toDense(denseHll);
          sparseHll.reset();
        }
        denseHll.mergeWith(inputData);
      } else {
        VELOX_USER_FAIL("Unexpected type of HLL");
      }
    }

    if (!hasValidInput) {
      return false;
    }
    int32_t size =
        isSparse ? sparseHll.serializedSize() : denseHll.serializedSize();
    result.resize(size);
    if (isSparse) {
      sparseHll.serialize(indexBitLength, result.data());
    } else {
      denseHll.serialize(result.data());
    }
    return true;
  }
};
} // namespace facebook::velox::functions
