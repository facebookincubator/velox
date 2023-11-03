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
#include "velox/functions/lib/aggregates/DecimalAggregate.h"
#include "velox/type/DecimalUtil.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::aggregate {

namespace {

/// Translate selected rows of decoded to the corresponding rows of its base
/// vector.
SelectivityVector translateToInnerRows(
    const DecodedVector& decoded,
    const SelectivityVector& rows) {
  VELOX_DCHECK(!decoded.isIdentityMapping());
  if (decoded.isConstantMapping()) {
    auto constantIndex = decoded.index(rows.begin());
    SelectivityVector baseRows{constantIndex + 1, false};
    baseRows.setValid(constantIndex, true);
    baseRows.updateBounds();
    return baseRows;
  } else {
    SelectivityVector baseRows{decoded.base()->size(), false};
    rows.applyToSelected(
        [&](auto row) { baseRows.setValid(decoded.index(row), true); });
    baseRows.updateBounds();
    return baseRows;
  }
}

/// Return the selected rows of the base vector of decoded corresponding to
/// rows. If decoded is not identify mapping, baseRowsHolder contains the
/// selected base rows. Otherwise, baseRowsHolder is unset.
const SelectivityVector* getBaseRows(
    const DecodedVector& decoded,
    const SelectivityVector& rows,
    SelectivityVector& baseRowsHolder) {
  const SelectivityVector* baseRows = &rows;
  if (!decoded.isIdentityMapping() && rows.hasSelections()) {
    baseRowsHolder = translateToInnerRows(decoded, rows);
    baseRows = &baseRowsHolder;
  }
  return baseRows;
}

} // namespace

/// Final = SUM(value * weight) / SUM(weight)
/// To avoid overflow, both TAccumulatorSum and TAccumulatorWeight are +8
/// bytes(double). Otherwise, we need to add logic to check plus and multiply
/// overflow.
/// Decimal type is not supported yet.
template <typename TAccumulatorSum, typename TAccumulatorWeight>
struct SumWeight {
  TAccumulatorSum sum{0};
  TAccumulatorWeight weight{0};
};

template <
    typename TInputValue,
    typename TInputWeight,
    typename TAccumulatorSum,
    typename TAccumulatorWeight,
    typename TResult>
class WeightedAverageAggregateBase : public exec::Aggregate {
 public:
  explicit WeightedAverageAggregateBase(TypePtr resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(SumWeight<TAccumulatorSum, TAccumulatorWeight>);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_)
          SumWeight<TAccumulatorSum, TAccumulatorWeight>();
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResult>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    TResult* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* sumWeight = accumulator(group);
        rawValues[i] = TResult(sumWeight->sum) / sumWeight->weight;
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TAccumulatorSum>();
    auto weightVector =
        rowVector->childAt(1)->asFlatVector<TAccumulatorWeight>();

    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    weightVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    TAccumulatorWeight* rawWeights = weightVector->mutableRawValues();
    TAccumulatorSum* rawSums = sumVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* sumWeight = accumulator(group);
        rawWeights[i] = sumWeight->weight;
        rawSums[i] = sumWeight->sum;
      }
    }
  }

  /// If either the value or the weight is NULL, then the row is ignored.
  /// NULL is returned if there are no non-NULL rows with non-zero weight.
  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);
    decodedWeight_.decode(*args[1], rows);

    if (decodedValue_.isConstantMapping() &&
        decodedWeight_.isConstantMapping()) {
      if (!decodedValue_.isNullAt(0) && !decodedWeight_.isNullAt(0)) {
        auto value = decodedValue_.valueAt<TInputValue>(0);
        auto weight = decodedWeight_.valueAt<TInputWeight>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(
              groups[i], TAccumulatorSum(value), TAccumulatorWeight(weight));
        });
      }
    } else if (decodedValue_.mayHaveNulls() || decodedWeight_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedValue_.isNullAt(i) || decodedWeight_.isNullAt(i)) {
          return;
        }
        auto value = decodedValue_.valueAt<TInputValue>(i);
        auto weight = decodedWeight_.valueAt<TInputWeight>(i);
        updateNonNullValue(
            groups[i], TAccumulatorSum(value), TAccumulatorWeight(weight));
      });
    } else if (
        !exec::Aggregate::numNulls_ && decodedValue_.isIdentityMapping() &&
        decodedWeight_.isIdentityMapping()) {
      // No Null exists in : Accumulators, decodedValue_, decodedWeight_
      // flat Vector without wrappings, access with index
      auto values = decodedValue_.data<TInputValue>();
      auto weights = decodedWeight_.data<TInputWeight>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], values[i], weights[i]);
      });
    } else {
      // 1 layer of wrapping
      rows.applyToSelected([&](vector_size_t i) {
        auto value = decodedValue_.valueAt<TInputValue>(i);
        auto weight = decodedWeight_.valueAt<TInputWeight>(i);
        updateNonNullValue(
            groups[i], TAccumulatorSum(value), TAccumulatorWeight(weight));
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedValue_.decode(*args[0], rows);
    decodedWeight_.decode(*args[1], rows);

    if (decodedValue_.isConstantMapping() &&
        decodedWeight_.isConstantMapping()) {
      if (!decodedValue_.isNullAt(0) && !decodedWeight_.isNullAt(0)) {
        auto value = decodedValue_.valueAt<TInputValue>(0);
        auto weight = decodedWeight_.valueAt<TInputWeight>(0);
        const auto numRows = rows.countSelected();
        updateNonNullIntermediateValues(
            group,
            TAccumulatorSum(value) * TAccumulatorWeight(weight) * numRows,
            TAccumulatorWeight(weight) * numRows);
      }
    } else if (decodedValue_.mayHaveNulls() || decodedWeight_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedValue_.isNullAt(i) || decodedWeight_.isNullAt(i)) {
          return;
        }
        auto value = decodedValue_.valueAt<TInputValue>(i);
        auto weight = decodedWeight_.valueAt<TInputWeight>(i);
        updateNonNullValue(
            group, TAccumulatorSum(value), TAccumulatorWeight(weight));
      });
    } else if (
        !exec::Aggregate::numNulls_ && decodedValue_.isIdentityMapping() &&
        decodedWeight_.isIdentityMapping()) {
      auto values = decodedValue_.data<TInputValue>();
      auto weights = decodedWeight_.data<TInputWeight>();
      TAccumulatorSum totalSum(0);
      TAccumulatorWeight totalWeight(0);
      rows.applyToSelected([&](vector_size_t i) {
        totalSum += values[i] * weights[i];
        totalWeight += weights[i];
      });
      updateNonNullIntermediateValues<false>(group, totalSum, totalWeight);
    } else {
      TAccumulatorSum totalSum(0);
      TAccumulatorWeight totalWeight(0);
      rows.applyToSelected([&](vector_size_t i) {
        // cost of an extra stack variable is negligible compared to the cost of
        // an extra virtual function call
        auto weight = decodedWeight_.valueAt<TInputWeight>(i);
        totalSum += decodedValue_.valueAt<TInputValue>(i) * weight;
        totalWeight += weight;
      });
      updateNonNullIntermediateValues(group, totalSum, totalWeight);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = decodedPartial_.base()->template as<RowVector>();

    if (validateIntermediateInputs_ &&
        (baseRowVector->childAt(0)->mayHaveNulls() ||
         baseRowVector->childAt(1)->mayHaveNulls())) {
      addIntermediateResultsImpl<true>(groups, rows);
      return;
    }
    addIntermediateResultsImpl<false>(groups, rows);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = decodedPartial_.base()->template as<RowVector>();

    if (validateIntermediateInputs_ &&
        (baseRowVector->childAt(0)->mayHaveNulls() ||
         baseRowVector->childAt(1)->mayHaveNulls())) {
      addSingleGroupIntermediateResultsImpl<true>(group, rows);
      return;
    }
    addSingleGroupIntermediateResultsImpl<false>(group, rows);
  }

 protected:
  /// Partial.
  template <bool tableHasNulls = true>
  inline void updateNonNullValue(
      char* group,
      TAccumulatorSum value,
      TAccumulatorWeight weight) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    accumulator(group)->sum += value * weight;
    accumulator(group)->weight += weight;
  }

  template <bool tableHasNulls = true>
  inline void updateNonNullIntermediateValues(
      char* group,
      TAccumulatorSum totalSum,
      TAccumulatorWeight totalWeight) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    accumulator(group)->sum += totalSum;
    accumulator(group)->weight += totalWeight;
  }

  inline SumWeight<TAccumulatorSum, TAccumulatorWeight>* accumulator(
      char* group) {
    return exec::Aggregate::value<
        SumWeight<TAccumulatorSum, TAccumulatorWeight>>(group);
  }

  template <bool checkNullFields>
  void addIntermediateResultsImpl(
      char** groups,
      const SelectivityVector& rows) {
    auto baseRowVector = decodedPartial_.base()->template as<RowVector>();

    SelectivityVector baseRowsHolder;
    auto* baseRows = getBaseRows(decodedPartial_, rows, baseRowsHolder);

    DecodedVector baseSumDecoded{*baseRowVector->childAt(0), *baseRows};
    DecodedVector baseWeightDecoded{*baseRowVector->childAt(1), *baseRows};

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        if constexpr (checkNullFields) {
          VELOX_USER_CHECK(
              !baseSumDecoded.isNullAt(decodedIndex) &&
              !baseWeightDecoded.isNullAt(decodedIndex));
        }
        auto totalSum =
            baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex);
        auto totalWeight =
            baseWeightDecoded.template valueAt<TAccumulatorWeight>(
                decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullIntermediateValues(groups[i], totalSum, totalWeight);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        if constexpr (checkNullFields) {
          VELOX_USER_CHECK(
              !baseSumDecoded.isNullAt(decodedIndex) &&
              !baseWeightDecoded.isNullAt(decodedIndex));
        }
        updateNonNullIntermediateValues(
            groups[i],
            baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex),
            baseWeightDecoded.template valueAt<TAccumulatorWeight>(
                decodedIndex));
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        if constexpr (checkNullFields) {
          VELOX_USER_CHECK(
              !baseSumDecoded.isNullAt(decodedIndex) &&
              !baseWeightDecoded.isNullAt(decodedIndex));
        }
        updateNonNullIntermediateValues(
            groups[i],
            baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex),
            baseWeightDecoded.template valueAt<TAccumulatorWeight>(
                decodedIndex));
      });
    }
  }

  template <bool checkNullFields>
  void addSingleGroupIntermediateResultsImpl(
      char* group,
      const SelectivityVector& rows) {
    auto baseRowVector = decodedPartial_.base()->template as<RowVector>();

    SelectivityVector baseRowsHolder;
    auto* baseRows = getBaseRows(decodedPartial_, rows, baseRowsHolder);

    DecodedVector baseSumDecoded{*baseRowVector->childAt(0), *baseRows};
    DecodedVector baseWeightDecoded{*baseRowVector->childAt(1), *baseRows};

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        if constexpr (checkNullFields) {
          VELOX_USER_CHECK(
              !baseSumDecoded.isNullAt(decodedIndex) &&
              !baseWeightDecoded.isNullAt(decodedIndex));
        }
        const auto numRows = rows.countSelected();
        auto totalSum =
            baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex) *
            numRows;
        auto totalWeight =
            baseWeightDecoded.template valueAt<TAccumulatorWeight>(
                decodedIndex) *
            numRows;
        updateNonNullIntermediateValues(group, totalSum, totalWeight);
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedPartial_.isNullAt(i)) {
          auto decodedIndex = decodedPartial_.index(i);
          if constexpr (checkNullFields) {
            VELOX_USER_CHECK(
                !baseSumDecoded.isNullAt(decodedIndex) &&
                !baseWeightDecoded.isNullAt(decodedIndex));
          }
          updateNonNullIntermediateValues(
              group,
              baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex),
              baseWeightDecoded.template valueAt<TAccumulatorWeight>(
                  decodedIndex));
        }
      });
    } else {
      TAccumulatorSum totalSum(0);
      TAccumulatorWeight totalWeight(0);
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        if constexpr (checkNullFields) {
          VELOX_USER_CHECK(
              !baseSumDecoded.isNullAt(decodedIndex) &&
              !baseWeightDecoded.isNullAt(decodedIndex));
        }
        totalSum +=
            baseSumDecoded.template valueAt<TAccumulatorSum>(decodedIndex);
        totalWeight += baseWeightDecoded.template valueAt<TAccumulatorWeight>(
            decodedIndex);
      });
      updateNonNullIntermediateValues(group, totalSum, totalWeight);
    }
  }

  DecodedVector decodedValue_;
  DecodedVector decodedWeight_;
  DecodedVector decodedPartial_;
};

/// Inspired by MinMaxByAggregateBase to handle multiple inputs
template <
    template <typename V, typename W, typename AS, typename AW, typename R>
    class Aggregate,
    typename IV>
std::unique_ptr<exec::Aggregate> create(
    TypePtr resultType,
    TypePtr weightType,
    const std::string& errorMessage) {
  switch (weightType->kind()) {
    case TypeKind::TINYINT:
      return std::make_unique<Aggregate<IV, int8_t, double, double, double>>(
          resultType);
    case TypeKind::SMALLINT:
      return std::make_unique<Aggregate<IV, int16_t, double, double, double>>(
          resultType);
    case TypeKind::INTEGER:
      return std::make_unique<Aggregate<IV, int32_t, double, double, double>>(
          resultType);
    case TypeKind::BIGINT:
      return std::make_unique<Aggregate<IV, int64_t, double, double, double>>(
          resultType);
    case TypeKind::REAL:
      return std::make_unique<Aggregate<IV, float, double, double, double>>(
          resultType);
    case TypeKind::DOUBLE:
      return std::make_unique<Aggregate<IV, double, double, double, double>>(
          resultType);
    default:
      VELOX_FAIL(errorMessage);
      return nullptr;
  }
}

/// TODO, need to handle and test Raw Input and Intermediate Input
template <
    template <typename V, typename W, typename AS, typename AW, typename R>
    class Aggregate>
std::unique_ptr<exec::Aggregate> create(
    TypePtr resultType,
    TypePtr valueType,
    TypePtr weightType,
    const std::string& errorMessage) {
  switch (valueType->kind()) {
    case TypeKind::TINYINT:
      return create<Aggregate, int8_t>(resultType, weightType, errorMessage);
    case TypeKind::SMALLINT:
      return create<Aggregate, int16_t>(resultType, weightType, errorMessage);
    case TypeKind::INTEGER:
      return create<Aggregate, int32_t>(resultType, weightType, errorMessage);
    case TypeKind::BIGINT:
      return create<Aggregate, int64_t>(resultType, weightType, errorMessage);
    case TypeKind::REAL:
      return create<Aggregate, float>(resultType, weightType, errorMessage);
    case TypeKind::DOUBLE:
      return create<Aggregate, double>(resultType, weightType, errorMessage);
    default:
      VELOX_FAIL(errorMessage);
  }
}

/// @brief Checks the input type for final aggregation of average.
/// @param type input type for final aggregation of average.
void checkWeightedAvgIntermediateType(const TypePtr& type);

} // namespace facebook::velox::functions::aggregate
