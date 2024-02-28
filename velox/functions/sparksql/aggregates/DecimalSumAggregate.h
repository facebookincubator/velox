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
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate::sparksql {

struct DecimalSum {
  int128_t sum{0};
  int64_t overflow{0};
  bool isEmpty{true};

  void mergeWith(const DecimalSum& other) {
    this->overflow += other.overflow;
    this->overflow +=
        DecimalUtil::addWithOverflow(this->sum, other.sum, this->sum);
    this->isEmpty &= other.isEmpty;
  }
};

template <typename TInputType, typename TResultType>
class DecimalSumAggregate : public exec::Aggregate {
 public:
  explicit DecimalSumAggregate(TypePtr resultType, TypePtr sumType)
      : exec::Aggregate(resultType), sumType_(sumType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalSum);
  }

  int32_t accumulatorAlignmentSize() const override {
    return alignof(DecimalSum);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) DecimalSum();
    }
  }

  int128_t computeFinalValue(DecimalSum* decimalSum, bool& overflow) {
    int128_t sum = decimalSum->sum;
    if ((decimalSum->overflow == 1 && decimalSum->sum < 0) ||
        (decimalSum->overflow == -1 && decimalSum->sum > 0)) {
      sum = static_cast<int128_t>(
          DecimalUtil::kOverflowMultiplier * decimalSum->overflow +
          decimalSum->sum);
    } else {
      if (decimalSum->overflow != 0) {
        overflow = true;
        return 0;
      }
    }

    auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*sumType_.get());
    overflow = !DecimalUtil::valueInPrecisionRange(sum, resultPrecision);
    return sum;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    VELOX_CHECK_EQ((*result)->encoding(), VectorEncoding::Simple::FLAT);
    auto vector = (*result)->as<FlatVector<TResultType>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    TResultType* rawValues = vector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* decimalSum = accumulator(group);
        if (decimalSum->isEmpty) {
          // If isEmpty is true, we should set null.
          vector->setNull(i, true);
        } else {
          bool overflow = false;
          auto result = (TResultType)computeFinalValue(decimalSum, overflow);
          if (overflow) {
            // Sum should be set to null on overflow.
            vector->setNull(i, true);
          } else {
            rawValues[i] = result;
          }
        }
      }
    }
  }

  void extractAccumulators(
      char** groups,
      int32_t numGroups,
      facebook::velox::VectorPtr* result) override {
    VELOX_CHECK_EQ((*result)->encoding(), VectorEncoding::Simple::ROW);
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TResultType>();
    auto isEmptyVector = rowVector->childAt(1)->asFlatVector<bool>();

    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    isEmptyVector->resize(numGroups);

    TResultType* rawSums = sumVector->mutableRawValues();
    // Bool uses compact representation, use mutableRawValues<uint64_t>
    // and bits::setBit instead.
    auto* rawIsEmpty = isEmptyVector->mutableRawValues<uint64_t>();
    uint64_t* rawNulls = getRawNulls(rowVector);

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      clearNull(rawNulls, i);
      if (isNull(group)) {
        rawSums[i] = 0;
        bits::setBit(rawIsEmpty, i, true);
      } else {
        auto* decimalSum = accumulator(group);
        bool overflow = false;
        auto result = (TResultType)computeFinalValue(decimalSum, overflow);
        if (overflow) {
          // Sum should be set to null on overflow, and
          // isEmpty should be set to false.
          sumVector->setNull(i, true);
          bits::setBit(rawIsEmpty, i, false);
        } else {
          rawSums[i] = result;
          bits::setBit(rawIsEmpty, i, decimalSum->isEmpty);
        }
      }
    }
  }

  void addRawInput(
      char** groups,
      const facebook::velox::SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(groups[i], value, false);
        });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedRaw_.isNullAt(i)) {
          return;
        }
        updateNonNullValue(
            groups[i], decodedRaw_.valueAt<TInputType>(i), false);
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], data[i], false);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(
            groups[i], decodedRaw_.valueAt<TInputType>(i), false);
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.template applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(group, value, false); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(group, decodedRaw_.valueAt<TInputType>(i), false);
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      DecimalSum decimalSum;
      rows.applyToSelected([&](vector_size_t i) {
        decimalSum.overflow += DecimalUtil::addWithOverflow(
            decimalSum.sum, data[i], decimalSum.sum);
        decimalSum.isEmpty = false;
      });
      mergeAccumulators(group, decimalSum);
    } else {
      DecimalSum decimalSum;
      rows.applyToSelected([&](vector_size_t i) {
        decimalSum.overflow += DecimalUtil::addWithOverflow(
            decimalSum.sum, decodedRaw_.valueAt<TInputType>(i), decimalSum.sum);
        decimalSum.isEmpty = false;
      });
      mergeAccumulators(group, decimalSum);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    VELOX_CHECK_EQ(
        decodedPartial_.base()->encoding(), VectorEncoding::Simple::ROW);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TResultType>>();
    auto isEmptyVector = baseRowVector->childAt(1)->as<SimpleVector<bool>>();

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          rows.applyToSelected([&](vector_size_t i) { setNull(groups[i]); });
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          rows.applyToSelected([&](vector_size_t i) {
            clearNull(groups[i]);
            updateNonNullValue(groups[i], sum, isEmpty);
          });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(groups[i]);
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(groups[i], sum, isEmpty);
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(groups[i]);
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(groups[i], sum, isEmpty);
        }
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    VELOX_CHECK_EQ(
        decodedPartial_.base()->encoding(), VectorEncoding::Simple::ROW);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TResultType>>();
    auto isEmptyVector = baseRowVector->childAt(1)->as<SimpleVector<bool>>();
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(group);
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          if (rows.hasSelections()) {
            clearNull(group);
          }
          rows.applyToSelected([&](vector_size_t i) {
            updateNonNullValue(group, sum, isEmpty);
          });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(group);
          return;
        } else {
          clearNull(group);
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(group, sum, isEmpty);
        }
      });
    } else {
      if (rows.hasSelections()) {
        clearNull(group);
      }
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(group);
          return;
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(group, sum, isEmpty);
        }
      });
    }
  }

 private:
  template <bool tableHasNulls = true>
  inline void updateNonNullValue(char* group, TResultType value, bool isEmpty) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto decimalSum = accumulator(group);
    decimalSum->overflow +=
        DecimalUtil::addWithOverflow(decimalSum->sum, value, decimalSum->sum);
    decimalSum->isEmpty &= isEmpty;
  }

  template <bool tableHasNulls = true>
  inline void mergeAccumulators(char* group, DecimalSum other) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto decimalSum = accumulator(group);
    decimalSum->mergeWith(other);
  }

  inline DecimalSum* accumulator(char* group) {
    return exec::Aggregate::value<DecimalSum>(group);
  }

  inline bool isIntermediateResultOverflow(
      const SimpleVector<bool>* isEmptyVector,
      const SimpleVector<TResultType>* sumVector,
      vector_size_t index) {
    // If isEmpty is false and sum is null, it means this intermediate
    // result has an overflow. The final accumulator of this group will
    // be null.
    return !isEmptyVector->valueAt(index) && sumVector->isNullAt(index);
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
  TypePtr sumType_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
