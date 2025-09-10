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
#include "velox/core/QueryConfig.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"

namespace facebook::velox::functions {

template <typename T>
struct CardinalityFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& result,
      const arg_type<HyperLogLog>& hll) {
    if (common::hll::SparseHllUtils::canDeserialize(hll.data())) {
      result = common::hll::SparseHllUtils::cardinality(hll.data());
    } else {
      result = common::hll::DenseHllUtils::cardinality(hll.data());
    }
    return true;
  }
};

template <typename T>
struct EmptyApproxSetFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(out_type<HyperLogLog>& result) {
    static const std::string kEmpty =
        common::hll::SparseHllUtils::serializeEmpty(12);

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
    serialized_ = common::hll::SparseHllUtils::serializeEmpty(
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

class MergeHllFunction : public exec::VectorFunction {
 public:
  static std::vector<exec::FunctionSignaturePtr> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("hyperloglog")
                .argumentType("array(hyperloglog)")
                .build()};
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 1);

    context.ensureWritable(rows, outputType, result);
    auto* flatResult = result->as<FlatVector<StringView>>();

    exec::LocalDecodedVector arrayDecoder(context, *args[0], rows);
    auto decodedArray = arrayDecoder.get();
    auto baseArray = decodedArray->base()->as<ArrayVector>();
    auto rawSizes = baseArray->rawSizes();
    auto rawOffsets = baseArray->rawOffsets();
    auto indices = decodedArray->indices();
    auto arrayElements = baseArray->elements();

    memory::StlAllocator<uint8_t> allocator{*context.pool()};
    using SparseHll = common::hll::SparseHll<memory::StlAllocator<uint8_t>>;
    using DenseHll = common::hll::DenseHll<memory::StlAllocator<uint8_t>>;

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto arraySize = rawSizes[indices[row]];
      auto arrayOffset = rawOffsets[indices[row]];

      int8_t indexBitLength = -1;
      bool isSparse = true;
      std::optional<SparseHll> sparseHll;
      std::optional<DenseHll> denseHll;

      for (vector_size_t i = 0; i < arraySize; ++i) {
        auto elementIndex = arrayOffset + i;
        if (arrayElements->isNullAt(elementIndex)) {
          continue;
        }

        auto hllData =
            arrayElements->as<FlatVector<StringView>>()->valueAt(elementIndex);
        if (hllData.empty()) {
          continue;
        }

        if (indexBitLength < 0) {
          if (common::hll::DenseHllUtils::canDeserialize(hllData.data())) {
            indexBitLength =
                common::hll::DenseHllUtils::deserializeIndexBitLength(
                    hllData.data());
          } else if (common::hll::SparseHllUtils::canDeserialize(
                         hllData.data())) {
            indexBitLength =
                common::hll::SparseHllUtils::deserializeIndexBitLength(
                    hllData.data());
          }
        }

        if (common::hll::SparseHllUtils::canDeserialize(hllData.data())) {
          auto cardinality =
              common::hll::SparseHllUtils::cardinality(hllData.data());
          if (cardinality == 0) {
            continue;
          }

          if (isSparse) {
            if (!sparseHll) {
              sparseHll.emplace(allocator);
              sparseHll->setSoftMemoryLimit(
                  common::hll::DenseHllUtils::estimateInMemorySize(
                      indexBitLength));
            }

            sparseHll->mergeWith(hllData.data());
            if (sparseHll->overLimit()) {
              denseHll.emplace(indexBitLength, allocator);
              sparseHll->toDense(*denseHll);
              sparseHll.reset();
              isSparse = false;
            }
          } else {
            SparseHll temp{hllData.data(), allocator};
            temp.toDense(*denseHll);
          }
        } else if (common::hll::DenseHllUtils::canDeserialize(hllData.data())) {
          auto hllIndexBitLength =
              common::hll::DenseHllUtils::deserializeIndexBitLength(
                  hllData.data());
          VELOX_USER_CHECK_EQ(
              indexBitLength,
              hllIndexBitLength,
              "Cannot merge HLLs with different indexBitLength: {} vs {}",
              indexBitLength,
              hllIndexBitLength);

          if (isSparse) {
            denseHll.emplace(indexBitLength, allocator);
            if (sparseHll) {
              sparseHll->toDense(*denseHll);
              sparseHll.reset();
            }
            isSparse = false;
          }
          denseHll->mergeWith(hllData.data());
        } else {
          VELOX_USER_FAIL("Invalid HLL data format");
        }
      }

      if (!sparseHll && !denseHll) {
        flatResult->setNull(row, true);
      } else if (isSparse) {
        auto size = sparseHll->serializedSize();
        auto stringView =
            StringView(flatResult->getRawStringBufferWithSpace(size), size);
        sparseHll->serialize(
            indexBitLength, const_cast<char*>(stringView.data()));
        flatResult->setNoCopy(row, stringView);
      } else {
        auto size = denseHll->serializedSize();
        auto stringView =
            StringView(flatResult->getRawStringBufferWithSpace(size), size);
        denseHll->serialize(const_cast<char*>(stringView.data()));
        flatResult->setNoCopy(row, stringView);
      }
    });
  }
};

} // namespace facebook::velox::functions
