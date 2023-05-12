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
#define XXH_INLINE_ALL
#include <xxhash.h>

#include "velox/common/hyperloglog/DenseHll.h"
#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/types/HyperLogLogType.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

using facebook::velox::common::hll::DenseHll;
using facebook::velox::common::hll::SparseHll;

namespace facebook::velox::aggregate::prestosql {

namespace {

struct HllAccumulator {
  explicit HllAccumulator(HashStringAllocator* allocator)
      : sparseHll_{allocator}, denseHll_{allocator} {}

  void setIndexBitLength(int8_t indexBitLength) {
    indexBitLength_ = indexBitLength;
    sparseHll_.setSoftMemoryLimit(
        DenseHll::estimateInMemorySize(indexBitLength_));
  }

  void append(uint64_t hash) {
    if (isSparse_) {
      if (sparseHll_.insertHash(hash)) {
        toDense();
      }
    } else {
      denseHll_.insertHash(hash);
    }
  }

  int64_t cardinality() const {
    return isSparse_ ? sparseHll_.cardinality() : denseHll_.cardinality();
  }

  void mergeWith(StringView serialized, HashStringAllocator* allocator) {
    auto input = serialized.data();
    if (SparseHll::canDeserialize(input)) {
      if (isSparse_) {
        sparseHll_.mergeWith(input);
      } else {
        SparseHll other{input, allocator};
        other.toDense(denseHll_);
      }
    } else if (DenseHll::canDeserialize(input)) {
      if (isSparse_) {
        if (indexBitLength_ < 0) {
          setIndexBitLength(DenseHll::deserializeIndexBitLength(input));
        }
        toDense();
      }
      denseHll_.mergeWith(input);
    } else {
      VELOX_USER_FAIL("Unexpected type of HLL");
    }
  }

  int32_t serializedSize() {
    return isSparse_ ? sparseHll_.serializedSize() : denseHll_.serializedSize();
  }

  void serialize(int8_t indexBitLength, char* outputBuffer) {
    return isSparse_ ? sparseHll_.serialize(indexBitLength, outputBuffer)
                     : denseHll_.serialize(outputBuffer);
  }

  void toDense() {
    isSparse_ = false;
    denseHll_.initialize(indexBitLength_);
    sparseHll_.toDense(denseHll_);
    sparseHll_.reset();
  }

  bool isSparse_{true};
  int8_t indexBitLength_{-1};
  SparseHll sparseHll_;
  DenseHll denseHll_;
};

template <typename T>
inline uint64_t hashOne(T value) {
  return XXH64(&value, sizeof(T), 0);
}

template <>
inline uint64_t hashOne<StringView>(StringView value) {
  return XXH64(value.data(), value.size(), 0);
}

template <typename T>
class ApproxDistinctAggregate : public exec::Aggregate {
 public:
  explicit ApproxDistinctAggregate(
      const TypePtr& resultType,
      bool hllAsFinalResult,
      bool hllAsRawInput)
      : exec::Aggregate(resultType),
        hllAsFinalResult_{hllAsFinalResult},
        hllAsRawInput_{hllAsRawInput} {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(HllAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_) HllAccumulator(allocator_);
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    if (hllAsFinalResult_) {
      extractAccumulators(groups, numGroups, result);
    } else {
      VELOX_CHECK(result);
      auto flatResult = (*result)->asFlatVector<int64_t>();

      extract<true>(
          groups,
          numGroups,
          flatResult,
          [](HllAccumulator* accumulator,
             FlatVector<int64_t>* result,
             vector_size_t index) {
            result->set(index, accumulator->cardinality());
          });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    auto flatResult = (*result)->asFlatVector<StringView>();

    extract<false>(
        groups,
        numGroups,
        flatResult,
        [&](HllAccumulator* accumulator,
            FlatVector<StringView>* result,
            vector_size_t index) {
          auto size = accumulator->serializedSize();
          StringView serialized;
          if (StringView::isInline(size)) {
            std::string buffer(size, '\0');
            accumulator->serialize(indexBitLength_, buffer.data());
            serialized = StringView::makeInline(buffer);
          } else {
            Buffer* buffer = flatResult->getBufferWithSpace(size);
            char* ptr = buffer->asMutable<char>() + buffer->size();
            accumulator->serialize(indexBitLength_, ptr);
            buffer->setSize(buffer->size() + size);
            serialized = StringView(ptr, size);
          }
          result->setNoCopy(index, serialized);
        });
  }

  void destroy(folly::Range<char**> /*groups*/) override {}

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    if (hllAsRawInput_) {
      addIntermediateResults(groups, rows, args, false /*unused*/);
    } else {
      decodeArguments(rows, args);

      rows.applyToSelected([&](auto row) {
        if (decodedValue_.isNullAt(row)) {
          return;
        }

        auto group = groups[row];
        auto tracker = trackRowSize(group);
        auto accumulator = value<HllAccumulator>(group);
        clearNull(group);
        accumulator->setIndexBitLength(indexBitLength_);

        auto hash = hashOne(decodedValue_.valueAt<T>(row));
        accumulator->append(hash);
      });
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedHll_.decode(*args[0], rows, true);

    rows.applyToSelected([&](auto row) {
      if (decodedHll_.isNullAt(row)) {
        return;
      }

      auto group = groups[row];
      auto tracker = trackRowSize(group);
      clearNull(group);

      auto serialized = decodedHll_.valueAt<StringView>(row);

      auto accumulator = value<HllAccumulator>(group);
      accumulator->mergeWith(serialized, allocator_);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto tracker = trackRowSize(group);
    if (hllAsRawInput_) {
      addSingleGroupIntermediateResults(group, rows, args, false /*unused*/);
    } else {
      decodeArguments(rows, args);

      rows.applyToSelected([&](auto row) {
        if (decodedValue_.isNullAt(row)) {
          return;
        }

        auto accumulator = value<HllAccumulator>(group);
        clearNull(group);
        accumulator->setIndexBitLength(indexBitLength_);

        auto hash = hashOne(decodedValue_.valueAt<T>(row));
        accumulator->append(hash);
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& row,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedHll_.decode(*args[0], row, true);

    auto tracker = trackRowSize(group);
    row.applyToSelected([&](auto row) {
      if (decodedHll_.isNullAt(row)) {
        return;
      }

      clearNull(group);

      auto serialized = decodedHll_.valueAt<StringView>(row);

      auto accumulator = value<HllAccumulator>(group);
      accumulator->mergeWith(serialized, allocator_);
    });
  }

 private:
  template <
      bool convertNullToZero,
      typename ExtractResult,
      typename ExtractFunc>
  void extract(
      char** groups,
      int32_t numGroups,
      FlatVector<ExtractResult>* result,
      ExtractFunc extractFunction) {
    VELOX_CHECK(result);
    result->resize(numGroups);

    uint64_t* rawNulls = nullptr;
    if (result->mayHaveNulls()) {
      BufferPtr nulls = result->mutableNulls(result->size());
      rawNulls = nulls->asMutable<uint64_t>();
    }

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        if constexpr (convertNullToZero) {
          // This condition is for approx_distinct. approx_distinct is an
          // approximation of count(distinct), hence, it makes sense for it to
          // be consistent with count(distinct) which returns 0 for null input.
          result->set(i, 0);
        } else {
          result->setNull(i, true);
        }
      } else {
        if (rawNulls) {
          bits::clearBit(rawNulls, i);
        }

        auto accumulator = value<HllAccumulator>(group);
        extractFunction(accumulator, result, i);
      }
    }
  }

  void decodeArguments(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args) {
    decodedValue_.decode(*args[0], rows, true);
    if (args.size() > 1) {
      decodedMaxStandardError_.decode(*args[1], rows, true);
      checkSetMaxStandardError();
    }
  }

  void checkSetMaxStandardError() {
    VELOX_USER_CHECK(
        decodedMaxStandardError_.isConstantMapping(),
        "Max standard error argument must be constant for all input rows");

    auto maxStandardError = decodedMaxStandardError_.valueAt<double>(0);
    checkSetMaxStandardError(maxStandardError);
  }

  void checkSetMaxStandardError(double error) {
    common::hll::checkMaxStandardError(error);

    if (maxStandardError_ < 0) {
      maxStandardError_ = error;
      indexBitLength_ = common::hll::toIndexBitLength(error);
    } else {
      VELOX_USER_CHECK_EQ(
          error,
          maxStandardError_,
          "Max standard error argument must be constant for all input rows");
    }
  }

  /// Boolean indicating whether final result is approximate cardinality of the
  /// input set or serialized HLL.
  const bool hllAsFinalResult_;

  /// Boolean indicating whether raw input contains elements of the set or
  /// serialized HLLs.
  const bool hllAsRawInput_;

  int8_t indexBitLength_{
      common::hll::toIndexBitLength(common::hll::kDefaultStandardError)};
  double maxStandardError_{-1};
  DecodedVector decodedValue_;
  DecodedVector decodedMaxStandardError_;
  DecodedVector decodedHll_;
};

template <TypeKind kind>
std::unique_ptr<exec::Aggregate> createApproxDistinct(
    const TypePtr& resultType,
    bool hllAsFinalResult,
    bool hllAsRawInput) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_unique<ApproxDistinctAggregate<T>>(
      resultType, hllAsFinalResult, hllAsRawInput);
}

bool registerApproxDistinct(
    const std::string& name,
    bool hllAsFinalResult,
    bool hllAsRawInput) {
  auto returnType = hllAsFinalResult ? "hyperloglog" : "bigint";

  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  if (hllAsRawInput) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType(returnType)
                             .intermediateType("varbinary")
                             .argumentType("hyperloglog")
                             .build());
  } else {
    for (const auto& inputType :
         {"boolean",
          "tinyint",
          "smallint",
          "integer",
          "bigint",
          "real",
          "double",
          "varchar",
          "timestamp",
          "date"}) {
      signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                               .returnType(returnType)
                               .intermediateType("varbinary")
                               .argumentType(inputType)
                               .build());

      signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                               .returnType(returnType)
                               .intermediateType("varbinary")
                               .argumentType(inputType)
                               .argumentType("double")
                               .build());
    }
  }

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name, hllAsFinalResult, hllAsRawInput](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        TypePtr type = argTypes[0]->isVarbinary() ? BIGINT() : argTypes[0];
        return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            createApproxDistinct,
            type->kind(),
            resultType,
            hllAsFinalResult,
            hllAsRawInput);
      });
  return true;
}

} // namespace

void registerApproxDistinctAggregates(const std::string& prefix) {
  registerCustomType(
      prefix + "hyperloglog",
      std::make_unique<const HyperLogLogTypeFactories>());
  registerApproxDistinct(prefix + kApproxDistinct, false, false);
  registerApproxDistinct(prefix + kApproxSet, true, false);
  registerApproxDistinct(prefix + kMerge, true, true);
}

} // namespace facebook::velox::aggregate::prestosql
