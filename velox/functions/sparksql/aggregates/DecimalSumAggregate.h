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
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate::sparksql {

/// This struct stores the sum of input values, overflow during accumulation,
/// and a bool value isEmpty used to indicate whether all inputs are null. The
/// initial value of sum is 0. We need to keep sum unchanged if the input is
/// null, as sum function ignores null input. If the isEmpty is true, then it
/// means there were no values to begin with or all the values were null, so the
/// result will be null. If the isEmpty is false, then if sum is null that means
/// an overflow has happened, it returns null.
struct DecimalSum {
  int128_t sum{0};
  int64_t overflow{0};
  bool isEmpty{true};

  std::optional<int128_t> computeFinalValue(const TypePtr& sumType) const {
    auto adjustedSum = DecimalUtil::adjustSumForOverflow(sum, overflow);
    auto [rPrecision, rScale] = getDecimalPrecisionScale(*sumType);
    if (adjustedSum.has_value() &&
        DecimalUtil::valueInPrecisionRange(adjustedSum, rPrecision)) {
      return adjustedSum;
    } else {
      // Find overflow during computing adjusted sum.
      return std::nullopt;
    }
  }
};

template <typename TInputType, typename TResultType>
class DecimalSumAggregate : public exec::Aggregate {
 public:
  // resultType refers to the output type of the aggregate function. For partial
  // aggregation, the resultType is ROW(DECIMAL, BOOLEAN), and for final
  // aggregation, the resultType is DECIMAL. sumType refers to the type of
  // DECIMAL in ROW(DECIMAL, BOOLEAN) used to store the accumulated sum in the
  // output of partial aggregation or the input of final aggregation.
  DecimalSumAggregate(TypePtr resultType, TypePtr sumType)
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

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
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
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              groups[i], decodedRaw_.valueAt<TInputType>(i), false);
        }
      });
    } else if (!numNulls_ && decodedRaw_.isIdentityMapping()) {
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
        rows.applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(group, value, false); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(group, decodedRaw_.valueAt<TInputType>(i), false);
        }
      });
    } else if (!numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(group, data[i], false);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(group, decodedRaw_.valueAt<TInputType>(i), false);
      });
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
    auto baseRowVector = decodedPartial_.base()->as<RowVector>();
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
            updateNonNullValue(groups[i], sum, isEmpty);
          });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedPartial_.isNullAt(i)) {
          auto decodedIndex = decodedPartial_.index(i);
          if (isIntermediateResultOverflow(
                  isEmptyVector, sumVector, decodedIndex)) {
            setNull(groups[i]);
          } else {
            auto sum = sumVector->valueAt(decodedIndex);
            auto isEmpty = isEmptyVector->valueAt(decodedIndex);
            updateNonNullValue(groups[i], sum, isEmpty);
          }
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
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
    auto baseRowVector = decodedPartial_.base()->as<RowVector>();
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
          rows.applyToSelected([&](vector_size_t i) {
            updateNonNullValue(group, sum, isEmpty);
          });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedPartial_.isNullAt(i)) {
          auto decodedIndex = decodedPartial_.index(i);
          if (isIntermediateResultOverflow(
                  isEmptyVector, sumVector, decodedIndex)) {
            setNull(group);
          } else {
            auto sum = sumVector->valueAt(decodedIndex);
            auto isEmpty = isEmptyVector->valueAt(decodedIndex);
            updateNonNullValue(group, sum, isEmpty);
          }
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        if (isIntermediateResultOverflow(
                isEmptyVector, sumVector, decodedIndex)) {
          setNull(group);
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(group, sum, isEmpty);
        }
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
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
        auto finalResult = decimalSum->computeFinalValue(sumType_);
        if (!finalResult.has_value()) {
          // Sum should be set to null on overflow, and
          // isEmpty should be set to false.
          sumVector->setNull(i, true);
          bits::setBit(rawIsEmpty, i, false);
        } else {
          rawSums[i] = (TResultType)finalResult.value();
          bits::setBit(rawIsEmpty, i, decimalSum->isEmpty);
        }
      }
    }
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
          auto finalResult = decimalSum->computeFinalValue(sumType_);
          if (!finalResult.has_value()) {
            // Sum should be set to null on overflow.
            vector->setNull(i, true);
          } else {
            rawValues[i] = (TResultType)finalResult.value();
          }
        }
      }
    }
  }

 private:
  template <bool tableHasNulls = true>
  inline void updateNonNullValue(char* group, TResultType value, bool isEmpty) {
    if constexpr (tableHasNulls) {
      clearNull(group);
    }
    auto decimalSum = accumulator(group);
    decimalSum->overflow +=
        DecimalUtil::addWithOverflow(decimalSum->sum, value, decimalSum->sum);
    decimalSum->isEmpty &= isEmpty;
  }

  inline DecimalSum* accumulator(char* group) {
    return value<DecimalSum>(group);
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
