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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/aggregates/HyperLogLogAggregate.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {

template <typename T>
struct CardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<int64_t>& result,
      const arg_type<HyperLogLog>& hll) {
    if (common::hll::SparseHll::canDeserialize(hll.data())) {
      result = common::hll::SparseHll::cardinality(hll.data());
    } else if (common::hll::DenseHll::canDeserialize(hll.data())) {
      result = common::hll::DenseHll::cardinality(hll.data());
    } else {
      return false;
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

class MergeHllVectorFunction : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // Create a memory pool for the entire batch
    auto pool = memory::memoryManager()->addLeafPool();
    auto allocator = std::make_unique<HashStringAllocator>(pool.get());

    // Decode the input array
    DecodedVector decodedArrays;
    decodedArrays.decode(*args[0], rows);

    // Prepare the output vector
    BaseVector::ensureWritable(rows, HYPERLOGLOG(), context.pool(), result);
    auto flatResult = result->asFlatVector<StringView>();

    // Process each row
    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArrays.isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      // Get the array for this row
      auto arrayVector = decodedArrays.base()->asUnchecked<ArrayVector>();
      auto arrayIndex = decodedArrays.index(row);
      auto offset = arrayVector->offsetAt(arrayIndex);
      auto size = arrayVector->sizeAt(arrayIndex);
      auto elements = arrayVector->elements();

      // Create an accumulator for this row
      auto accumulator =
          std::make_unique<aggregate::prestosql::HllAccumulator<int64_t, true>>(
              allocator.get());

      bool hasValidInput = false;

      // Merge all HLLs in the array
      for (auto i = 0; i < size; i++) {
        auto elementIndex = offset + i;
        if (elements->isNullAt(elementIndex)) {
          continue;
        }

        hasValidInput = true;
        auto hll = elements->asUnchecked<SimpleVector<StringView>>()->valueAt(
            elementIndex);
        accumulator->mergeWith(hll, allocator.get());
      }

      if (!hasValidInput) {
        flatResult->setNull(row, true);
        return;
      }

      // Serialize the result
      auto resultSize = accumulator->serializedSize();
      if (StringView::isInline(resultSize)) {
        // For small results, use inline storage
        std::string buffer(resultSize, '\0');
        accumulator->serialize(buffer.data());
        flatResult->set(row, StringView::makeInline(buffer));
      } else {
        // For larger results, allocate from the string buffer
        char* rawBuffer = flatResult->getRawStringBufferWithSpace(resultSize);
        accumulator->serialize(rawBuffer);
        flatResult->setNoCopy(row, StringView(rawBuffer, resultSize));
      }
    });
  }
};

std::unique_ptr<exec::VectorFunction> makeMergeHll(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
    const core::QueryConfig& /*config*/) {
  return std::make_unique<MergeHllVectorFunction>();
}

inline std::vector<std::shared_ptr<exec::FunctionSignature>>
mergeHllSignatures() {
  return {exec::FunctionSignatureBuilder()
              .returnType("hyperloglog")
              .argumentType("array(hyperloglog)")
              .build()};
}

template <typename T>
struct MergeHllFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> allocator_;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& /*config*/,
      const arg_type<HyperLogLog>* /*hll1*/,
      const arg_type<HyperLogLog>* /*hll2*/) {
    pool_ = memory::memoryManager()->addLeafPool();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<HyperLogLog>& result,
      const arg_type<HyperLogLog>& hll1,
      const arg_type<HyperLogLog>& hll2) {
    // Find a valid HLL to extract indexBitLength
    int8_t indexBitLength = -1;

    if (common::hll::SparseHll::canDeserialize(hll1.data())) {
      indexBitLength = 12; // Default for sparse HLL
    } else if (common::hll::DenseHll::canDeserialize(hll1.data())) {
      indexBitLength =
          common::hll::DenseHll::deserializeIndexBitLength(hll1.data());
    } else if (common::hll::SparseHll::canDeserialize(hll2.data())) {
      indexBitLength = 12; // Default for sparse HLL
    } else if (common::hll::DenseHll::canDeserialize(hll2.data())) {
      indexBitLength =
          common::hll::DenseHll::deserializeIndexBitLength(hll2.data());
    } else {
      // No valid HLL found
      return false;
    }

    // Create a sparse HLL initially
    common::hll::SparseHll sparseHll(allocator_.get());
    sparseHll.setSoftMemoryLimit(
        common::hll::DenseHll::estimateInMemorySize(indexBitLength));

    // Merge first HLL
    sparseHll.mergeWith(hll1.data());

    // Check if we need to convert to dense
    bool needDense = sparseHll.overLimit();

    // Merge second HLL
    if (!needDense) {
      sparseHll.mergeWith(hll2.data());
      needDense = sparseHll.overLimit();
    }

    // Serialize the result
    if (needDense) {
      // Convert to dense and merge
      common::hll::DenseHll denseHll(indexBitLength, allocator_.get());
      sparseHll.toDense(denseHll);

      // If we already converted to dense, merge the second HLL directly
      if (sparseHll.overLimit()) {
        denseHll.mergeWith(hll2.data());
      }

      // Serialize the dense HLL
      result.resize(denseHll.serializedSize());
      denseHll.serialize(result.data());
    } else {
      // Serialize the sparse HLL
      result.resize(sparseHll.serializedSize());
      sparseHll.serialize(indexBitLength, result.data());
    }

    return true;
  }
};
} // namespace facebook::velox::functions
