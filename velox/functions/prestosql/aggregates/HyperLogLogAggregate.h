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

#define XXH_INLINE_ALL
#include <xxhash.h> // @manual=third-party//xxHash:xxhash

#include "velox/common/hyperloglog/HllUtils.h"
#include "velox/common/hyperloglog/SparseHll.h"
#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/HllAccumulator.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

using facebook::velox::common::hll::SparseHll;

namespace facebook::velox::aggregate::prestosql {

template <typename T, bool HllAsFinalResult>
class HyperLogLogAggregate : public exec::Aggregate {
 public:
  explicit HyperLogLogAggregate(
      const TypePtr& resultType,
      bool hllAsRawInput,
      double defaultError)
      : exec::Aggregate(resultType),
        hllAsFinalResult_{HllAsFinalResult},
        hllAsRawInput_{hllAsRawInput},
        indexBitLength_{common::hll::toIndexBitLength(defaultError)} {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(velox::common::hll::HllAccumulator<T, HllAsFinalResult>);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(velox::common::hll::HllAccumulator<T, HllAsFinalResult>);
  }

  bool isFixedSize() const override {
    return false;
  }

  bool supportsToIntermediate() const final {
    return hllAsRawInput_;
  }

  void toIntermediate(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      VectorPtr& result) const final {
    singleInputAsIntermediate(rows, args, result);
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
          [](velox::common::hll::HllAccumulator<T, HllAsFinalResult>*
                 accumulator,
             FlatVector<int64_t>* result,
             vector_size_t index) {
            result->set(index, accumulator->cardinality());
          });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK(result);
    if constexpr (std::is_same_v<T, bool>) {
      static_assert(!HllAsFinalResult);
      auto* flatResult = (*result)->asFlatVector<int8_t>();

      for (auto i = 0; i < numGroups; ++i) {
        char* group = groups[i];
        auto* accumulator =
            value<velox::common::hll::HllAccumulator<bool, false>>(group);
        flatResult->set(i, accumulator->getState());
      }

    } else {
      auto* flatResult = (*result)->asFlatVector<StringView>();

      extract<false>(
          groups,
          numGroups,
          flatResult,
          [&](velox::common::hll::HllAccumulator<T, HllAsFinalResult>*
                  accumulator,
              FlatVector<StringView>* result,
              vector_size_t index) {
            auto size = accumulator->serializedSize();
            StringView serialized;
            if (StringView::isInline(size)) {
              std::string buffer(size, '\0');
              accumulator->serialize(buffer.data());
              serialized = StringView::makeInline(buffer);
            } else {
              char* rawBuffer = flatResult->getRawStringBufferWithSpace(size);
              accumulator->serialize(rawBuffer);
              serialized = StringView(rawBuffer, size);
            }
            result->setNoCopy(index, serialized);
          });
    }
  }

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
        auto accumulator =
            value<velox::common::hll::HllAccumulator<T, HllAsFinalResult>>(
                group);
        clearNull(group);
        accumulator->setIndexBitLength(indexBitLength_);
        accumulator->append(decodedValue_.valueAt<T>(row));
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

      mergeToAccumulator(group, row);
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

        auto accumulator =
            value<velox::common::hll::HllAccumulator<T, HllAsFinalResult>>(
                group);
        clearNull(group);
        accumulator->setIndexBitLength(indexBitLength_);

        accumulator->append(decodedValue_.valueAt<T>(row));
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedHll_.decode(*args[0], rows, true);

    auto tracker = trackRowSize(group);
    rows.applyToSelected([&](auto row) {
      if (decodedHll_.isNullAt(row)) {
        return;
      }

      clearNull(group);
      mergeToAccumulator(group, row);
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      auto group = groups[i];
      new (group + offset_)
          velox::common::hll::HllAccumulator<T, HllAsFinalResult>(allocator_);
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    destroyAccumulators<
        velox::common::hll::HllAccumulator<T, HllAsFinalResult>>(groups);
  }

 private:
  void mergeToAccumulator(char* group, const vector_size_t row) {
    if constexpr (std::is_same_v<T, bool>) {
      static_assert(!HllAsFinalResult);
      value<velox::common::hll::HllAccumulator<bool, false>>(group)->mergeWith(
          decodedHll_.valueAt<int8_t>(row));
    } else {
      auto serialized = decodedHll_.valueAt<StringView>(row);
      velox::common::hll::HllAccumulator<T, HllAsFinalResult>* accumulator =
          value<velox::common::hll::HllAccumulator<T, HllAsFinalResult>>(group);
      accumulator->mergeWith(serialized, allocator_);
    }
  }

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
      BufferPtr& nulls = result->mutableNulls(result->size());
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

        auto accumulator =
            value<velox::common::hll::HllAccumulator<T, HllAsFinalResult>>(
                group);
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
      checkSetMaxStandardError(rows);
    }
  }

  void checkSetMaxStandardError(const SelectivityVector& rows) {
    if (decodedMaxStandardError_.isConstantMapping()) {
      const auto maxStandardError = decodedMaxStandardError_.valueAt<double>(0);
      checkSetMaxStandardError(maxStandardError);
      return;
    }

    rows.applyToSelected([&](auto row) {
      VELOX_USER_CHECK(
          !decodedMaxStandardError_.isNullAt(row),
          "Max standard error cannot be null");
      const auto maxStandardError =
          decodedMaxStandardError_.valueAt<double>(row);
      if (maxStandardError_ == -1) {
        checkSetMaxStandardError(maxStandardError);
      } else {
        VELOX_USER_CHECK_EQ(
            maxStandardError,
            maxStandardError_,
            "Max standard error argument must be constant for all input rows");
      }
    });
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

  int8_t indexBitLength_;
  double maxStandardError_{-1};
  DecodedVector decodedValue_;
  DecodedVector decodedMaxStandardError_;
  DecodedVector decodedHll_;
};

template <TypeKind kind>
std::unique_ptr<exec::Aggregate> createHyperLogLogAggregate(
    const TypePtr& resultType,
    bool hllAsFinalResult,
    bool hllAsRawInput,
    double defaultError) {
  using T = typename TypeTraits<kind>::NativeType;
  if (hllAsFinalResult) {
    if constexpr (kind == TypeKind::BOOLEAN) {
      VELOX_UNREACHABLE("approx_set(boolean) is not supported.");
    } else {
      return std::make_unique<HyperLogLogAggregate<T, true>>(
          resultType, hllAsRawInput, defaultError);
    }
  } else {
    return std::make_unique<HyperLogLogAggregate<T, false>>(
        resultType, hllAsRawInput, defaultError);
  }
}

} // namespace facebook::velox::aggregate::prestosql
