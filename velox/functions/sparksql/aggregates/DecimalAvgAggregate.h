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

#include "velox/common/base/IOUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/DecimalAggregate.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql::aggregates {

using velox::aggregate::LongDecimalWithOverflowState;

template <typename TInputType, typename TSumResultType, typename TResultType>
class DecimalAverageAggregate : public exec::Aggregate {
 public:
  explicit DecimalAverageAggregate(TypePtr inputType, TypePtr resultType)
      : exec::Aggregate(resultType), inputType_(inputType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalAverageAggregate);
  }

  int32_t accumulatorAlignmentSize() const override {
    return static_cast<int32_t>(sizeof(int128_t));
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_)
          velox::aggregate::LongDecimalWithOverflowState();
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(groups[i], value); });
      } else {
        // Spark expects the result of partial avg to be non-nullable.
        rows.applyToSelected(
            [&](vector_size_t i) { exec::Aggregate::clearNull(groups[i]); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        // Spark expects the result of partial avg to be non-nullable.
        exec::Aggregate::clearNull(groups[i]);
        if (decodedRaw_.isNullAt(i)) {
          return;
        }
        updateNonNullValue(groups[i], decodedRaw_.valueAt<TInputType>(i));
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], data[i]);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(groups[i], decodedRaw_.valueAt<TInputType>(i));
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        const auto numRows = rows.countSelected();
        int64_t overflow = 0;
        int128_t totalSum{0};
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.template applyToSelected(
            [&](vector_size_t i) { updateNonNullValue(group, value); });
      } else {
        // Spark expects the result of partial avg to be non-nullable.
        exec::Aggregate::clearNull(group);
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(group, decodedRaw_.valueAt<TInputType>(i));
        } else {
          // Spark expects the result of partial avg to be non-nullable.
          exec::Aggregate::clearNull(group);
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      const TInputType* data = decodedRaw_.data<TInputType>();
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum, data[i], accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators<false>(group, serialized);
    } else {
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum,
            decodedRaw_.valueAt<TInputType>(i),
            accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators(group, serialized);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumCol = baseRowVector->childAt(0);
    auto countCol = baseRowVector->childAt(1);
    if (sumCol->type()->isShortDecimal()) {
      addIntermediateDecimalResults(
          groups,
          rows,
          sumCol->as<SimpleVector<int64_t>>(),
          countCol->as<SimpleVector<int64_t>>());
      return;
    }
    if (sumCol->type()->isLongDecimal()) {
      addIntermediateDecimalResults(
          groups,
          rows,
          sumCol->as<SimpleVector<int128_t>>(),
          countCol->as<SimpleVector<int64_t>>());
      return;
    }
    switch (sumCol->typeKind()) {
      default:
        VELOX_FAIL(
            "Unsupported sum type for decimal aggregation: {}",
            sumCol->typeKind());
    }
  }

  template <class UnscaledType>
  void addIntermediateDecimalResults(
      char** groups,
      const SelectivityVector& rows,
      SimpleVector<UnscaledType>* sumVector,
      SimpleVector<int64_t>* countVector) {
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          auto accumulator = decimalAccumulator(groups[i]);
          mergeSumCount(accumulator, sum, count);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        mergeSumCount(accumulator, sum, count);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        mergeSumCount(accumulator, sum, count);
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TInputType>>();
    auto countVector = baseRowVector->childAt(1)->as<SimpleVector<int64_t>>();

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected(
            [&](vector_size_t i) { mergeAccumulators(group, sum, count); });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, count);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, count);
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TSumResultType>();
    auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();
    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);

    uint64_t* rawNulls = getRawNulls(rowVector);

    int64_t* rawCounts = countVector->mutableRawValues();
    TSumResultType* rawSums = sumVector->mutableRawValues();

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* accumulator = decimalAccumulator(group);
        rawCounts[i] = accumulator->count;
        if constexpr (std::is_same_v<TSumResultType, int64_t>) {
          rawSums[i] = TSumResultType((int64_t)accumulator->sum);
        } else {
          rawSums[i] = TSumResultType(accumulator->sum);
        }
      }
    }
  }

  TResultType computeFinalValue(LongDecimalWithOverflowState* accumulator) {
    int128_t sum = accumulator->sum;
    if ((accumulator->overflow == 1 && accumulator->sum < 0) ||
        (accumulator->overflow == -1 && accumulator->sum > 0)) {
      sum = static_cast<int128_t>(
          DecimalUtil::kOverflowMultiplier * accumulator->overflow +
          accumulator->sum);
    } else {
      VELOX_CHECK(
          accumulator->overflow == 0,
          "overflow: decimal avg struct overflow not eq 0");
    }

    auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*this->resultType().get());
    auto sumType = this->inputType().get();
    // Spark use DECIMAL(20,0) to represent long value
    int countPrecision = 20;
    int countScale = 0;
    auto [sumPrecision, sumScale] = getDecimalPrecisionScale(*sumType);
    auto [avgPrecision, avgScale] = computeResultPrecisionScale(
        sumPrecision, sumScale, countPrecision, countScale);
    auto sumRescale = computeRescaleFactor(sumScale, countScale, avgScale);
    auto countDecimal = accumulator->count;
    int128_t avg = 0;

    if (sumType->isShortDecimal()) {
      // sumType is SHORT_DECIMAL, we can safely convert sum to int64_t
      auto longSum = (int64_t)sum;
      DecimalUtil::divideWithRoundUp<int128_t, int64_t, int128_t>(
          avg, (int64_t)longSum, countDecimal, false, sumRescale, 0);
    } else {
      DecimalUtil::divideWithRoundUp<int128_t, int128_t, int128_t>(
          avg, (int128_t)sum, countDecimal, false, sumRescale, 0);
    }
    DecimalUtil::valueInRange(avg);
    auto castedAvg = DecimalUtil::rescaleWithRoundUp<int128_t, TResultType>(
        avg, avgPrecision, avgScale, resultPrecision, resultScale, true);
    if (castedAvg.has_value()) {
      return castedAvg.value();
    } else {
      VELOX_FAIL("compute final average value overflow.");
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResultType>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    TResultType* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = decimalAccumulator(group);
      if (isNull(group) || accumulator->count == 0) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        if (accumulator->overflow > 0) {
          // Spark does not support ansi mode yet,
          // and needs to return null when overflow
          vector->setNull(i, true);
        } else {
          try {
            rawValues[i] = computeFinalValue(accumulator);
          } catch (const VeloxException& err) {
            if (err.message().find("overflow") != std::string::npos ||
                err.message().find("is not in the range of Decimal Type") !=
                    std::string::npos) {
              // find overflow or out of long decimal range in computation
              vector->setNull(i, true);
            } else {
              VELOX_FAIL("compute average failed");
            }
          }
        }
      }
    }
  }

  template <bool tableHasNulls = true>
  void mergeAccumulators(char* group, const StringView& serialized) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->mergeWith(serialized);
  }

  template <bool tableHasNulls = true, class UnscaledType>
  void mergeAccumulators(
      char* group,
      const UnscaledType& otherSum,
      const int64_t& otherCount) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    mergeSumCount(accumulator, otherSum, otherCount);
  }

  template <bool tableHasNulls = true>
  void updateNonNullValue(char* group, int128_t value) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->overflow +=
        DecimalUtil::addWithOverflow(accumulator->sum, value, accumulator->sum);
    accumulator->count += 1;
  }

  template <typename UnscaledType>
  inline void mergeSumCount(
      LongDecimalWithOverflowState* accumulator,
      UnscaledType sum,
      int64_t count) {
    accumulator->count += count;
    accumulator->overflow +=
        DecimalUtil::addWithOverflow(accumulator->sum, sum, accumulator->sum);
  }

  TypePtr inputType() const {
    return inputType_;
  }

 private:
  inline LongDecimalWithOverflowState* decimalAccumulator(char* group) {
    return exec::Aggregate::value<LongDecimalWithOverflowState>(group);
  }

  inline static uint8_t
  computeRescaleFactor(uint8_t fromScale, uint8_t toScale, uint8_t rScale) {
    return rScale - fromScale + toScale;
  }

  inline static std::pair<uint8_t, uint8_t> computeResultPrecisionScale(
      const uint8_t aPrecision,
      const uint8_t aScale,
      const uint8_t bPrecision,
      const uint8_t bScale) {
    uint8_t intDig = aPrecision - aScale + bScale;
    uint8_t scale = std::max(6, aScale + bPrecision + 1);
    uint8_t precision = intDig + scale;
    return adjustPrecisionScale(precision, scale);
  }

  inline static std::pair<uint8_t, uint8_t> adjustPrecisionScale(
      const uint8_t precision,
      const uint8_t scale) {
    VELOX_CHECK(scale >= 0);
    VELOX_CHECK(precision >= scale);
    if (precision <= 38) {
      return {precision, scale};
    } else {
      uint8_t intDigits = precision - scale;
      uint8_t minScaleValue = std::min(scale, (uint8_t)6);
      uint8_t adjustedScale =
          std::max((uint8_t)(38 - intDigits), minScaleValue);
      return {38, adjustedScale};
    }
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
  const TypePtr inputType_;
};

bool registerDecimalAvgAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(a_precision, a_scale), BIGINT)")
          .returnType("DECIMAL(a_precision, a_scale)")
          .build());

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        auto& inputType = argTypes[0];
        if (inputType->isShortDecimal()) {
          if (resultType->isShortDecimal()) {
            return std::make_unique<
                DecimalAverageAggregate<int64_t, int128_t, int64_t>>(
                inputType, resultType);
          }
          if (resultType->isLongDecimal()) {
            return std::make_unique<
                DecimalAverageAggregate<int64_t, int128_t, int128_t>>(
                inputType, resultType);
          }
          switch (resultType->kind()) {
            case TypeKind::ROW: { // Partial
              auto sumResultType = resultType->asRow().childAt(0);
              if (sumResultType->isShortDecimal()) {
                return std::make_unique<
                    DecimalAverageAggregate<int64_t, int128_t, int64_t>>(
                    inputType, resultType);
              } else {
                return std::make_unique<
                    DecimalAverageAggregate<int64_t, int128_t, int128_t>>(
                    inputType, resultType);
              }
            }
            default:
              VELOX_FAIL(
                  "Unknown result type for {} aggregation {}",
                  name,
                  resultType->kindName());
          }
        }
        if (inputType->isLongDecimal()) {
          if (resultType->isLongDecimal()) {
            return std::make_unique<
                DecimalAverageAggregate<int128_t, int128_t, int128_t>>(
                inputType, resultType);
          }
          switch (resultType->kind()) {
            case TypeKind::ROW: { // Partial
              auto sumResultType = resultType->asRow().childAt(0);
              if (sumResultType->kind() == TypeKind::HUGEINT) {
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t, int128_t>>(
                    inputType, resultType);
              } else {
                VELOX_FAIL(
                    "Partial Avg Agg result type must greater than input type. result={}",
                    resultType->kind());
              }
            }
            default:
              VELOX_FAIL(
                  "Unknown result type for {} aggregation {}",
                  name,
                  resultType->kindName());
          }
        }
        switch (inputType->kind()) {
          case TypeKind::ROW: { // Final
            VELOX_CHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            if (sumInputType->isLongDecimal()) {
              if (resultType->isShortDecimal()) {
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t, int64_t>>(
                    sumInputType, resultType);
              } else {
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t, int128_t>>(
                    sumInputType, resultType);
              }
            }
            VELOX_FAIL(
                "Unknown sum type for {} aggregation {}",
                name,
                sumInputType->kindName());
          }
          default:
            VELOX_FAIL(
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      },
      true);
}
} // namespace facebook::velox::functions::sparksql::aggregates
