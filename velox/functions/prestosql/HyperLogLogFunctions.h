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

    exec::LocalSelectivityVector allElementsRows(
        context, arrayElements->size());
    allElementsRows->setAll();
    exec::LocalDecodedVector elementsDecoder(
        context, *arrayElements, *allElementsRows);
    auto decodedElements = elementsDecoder.get();

    memory::MemoryPool* memoryPool = context.pool();
    using SparseHll = common::hll::SparseHll<memory::MemoryPool>;
    using DenseHll = common::hll::DenseHll<memory::MemoryPool>;

    context.applyToSelectedNoThrow(rows, [&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);
        return;
      }

      auto arraySize = rawSizes[indices[row]];
      auto arrayOffset = rawOffsets[indices[row]];

      std::optional<int8_t> indexBitLength;
      std::optional<SparseHll> sparseHll;
      std::optional<DenseHll> denseHll;

      for (vector_size_t i = 0; i < arraySize; ++i) {
        auto elementIndex = arrayOffset + i;
        if (decodedElements->isNullAt(elementIndex)) {
          continue;
        }

        auto hllData = decodedElements->valueAt<StringView>(elementIndex);
        if (hllData.empty()) {
          continue;
        }

        const bool isSparseHllData =
            common::hll::SparseHlls::canDeserialize(hllData.data());
        const bool isDenseHllData = !isSparseHllData &&
            common::hll::DenseHlls::canDeserialize(hllData.data());

        VELOX_USER_CHECK(
            isSparseHllData || isDenseHllData, "Invalid HLL data format");

        const auto hllIndexBitLength = isSparseHllData
            ? common::hll::SparseHlls::deserializeIndexBitLength(hllData.data())
            : common::hll::DenseHlls::deserializeIndexBitLength(hllData.data());

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
            SparseHll temp{hllData.data(), memoryPool};
            temp.toDense(*denseHll);
          } else {
            if (!sparseHll.has_value()) {
              sparseHll.emplace(memoryPool);
              sparseHll->setSoftMemoryLimit(
                  common::hll::DenseHlls::estimateInMemorySize(
                      indexBitLength.value()));
            }

            sparseHll->mergeWith(hllData.data());
            if (sparseHll->overLimit()) {
              denseHll.emplace(indexBitLength.value(), memoryPool);
              sparseHll->toDense(*denseHll);
              sparseHll.reset();
            }
          }
        } else {
          if (sparseHll.has_value()) {
            denseHll.emplace(indexBitLength.value(), memoryPool);
            sparseHll->toDense(*denseHll);
            sparseHll.reset();
          } else if (!denseHll.has_value()) {
            denseHll.emplace(indexBitLength.value(), memoryPool);
          }
          denseHll->mergeWith(hllData.data());
        }
      }

      if (!sparseHll.has_value() && !denseHll.has_value()) {
        flatResult->setNull(row, true);
      } else if (sparseHll.has_value()) {
        auto size = sparseHll->serializedSize();
        char* buffer = flatResult->getRawStringBufferWithSpace(size);
        sparseHll->serialize(indexBitLength.value(), buffer);
        flatResult->setNoCopy(row, StringView(buffer, size));
      } else {
        auto size = denseHll->serializedSize();
        char* buffer = flatResult->getRawStringBufferWithSpace(size);
        denseHll->serialize(buffer);
        flatResult->setNoCopy(row, StringView(buffer, size));
      }
    });
  }
};

} // namespace facebook::velox::functions
