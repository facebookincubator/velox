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

#include "velox/functions/sparksql/aggregates/AverageAggregate.h"
#include "velox/functions/lib/aggregates/AverageAggregateBase.h"
#include "velox/functions/sparksql/DecimalUtil.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {
namespace {

template <typename TInput, typename TAccumulator, typename TResult>
class AverageAggregate
    : public AverageAggregateBase<TInput, TAccumulator, TResult> {
 public:
  explicit AverageAggregate(TypePtr resultType)
      : AverageAggregateBase<TInput, TAccumulator, TResult>(resultType) {}

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TAccumulator>();
    auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();

    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);
    rowVector->clearAllNulls();

    int64_t* rawCounts = countVector->mutableRawValues();
    TAccumulator* rawSums = sumVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      // When all inputs are nulls, the partial result is (0, 0).
      char* group = groups[i];
      auto* sumCount = this->accumulator(group);
      rawCounts[i] = sumCount->count;
      rawSums[i] = sumCount->sum;
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResult>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = this->getRawNulls(vector);

    TResult* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto* sumCount = this->accumulator(group);
      if (sumCount->count == 0) {
        // In Spark, if all inputs are null, count will be 0,
        // and the result of final avg will be null.
        vector->setNull(i, true);
      } else {
        this->clearNull(rawNulls, i);
        rawValues[i] = (TResult)sumCount->sum / sumCount->count;
      }
    }
  }
};

template <typename TInputType, typename TResultType>
class DecimalAverageAggregate : public DecimalAggregate<TInputType> {
 public:
  explicit DecimalAverageAggregate(TypePtr resultType, TypePtr sumType)
      : DecimalAggregate<TInputType>(resultType), sumType_(sumType) {}

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<int128_t>>();
    auto countVector = baseRowVector->childAt(1)->as<SimpleVector<int64_t>>();
    VELOX_USER_CHECK_NOT_NULL(sumVector);

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto count = countVector->valueAt(decodedIndex);
        if (sumVector->isNullAt(decodedIndex) &&
            !countVector->isNullAt(decodedIndex) && count > 0) {
          // Find overflow, set all groups to null.
          rows.applyToSelected(
              [&](vector_size_t i) { this->setNull(groups[i]); });
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          rows.applyToSelected([&](vector_size_t i) {
            this->clearNull(groups[i]);
            auto accumulator = this->decimalAccumulator(groups[i]);
            mergeSumCount(accumulator, sum, count);
          });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        if (sumVector->isNullAt(decodedIndex) &&
            !countVector->isNullAt(decodedIndex) && count > 0) {
          this->setNull(groups[i]);
        } else {
          this->clearNull(groups[i]);
          auto sum = sumVector->valueAt(decodedIndex);
          auto accumulator = this->decimalAccumulator(groups[i]);
          mergeSumCount(accumulator, sum, count);
        }
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        if (sumVector->isNullAt(decodedIndex) &&
            !countVector->isNullAt(decodedIndex) && count > 0) {
          this->setNull(groups[i]);
        } else {
          this->clearNull(groups[i]);
          auto sum = sumVector->valueAt(decodedIndex);
          auto accumulator = this->decimalAccumulator(groups[i]);
          mergeSumCount(accumulator, sum, count);
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
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<int128_t>>();
    auto countVector = baseRowVector->childAt(1)->as<SimpleVector<int64_t>>();

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        if (isPartialSumOverflow(sumVector, countVector, decodedIndex)) {
          // Find overflow, just set group to null and return.
          this->setNull(group);
          return;
        } else {
          if (rows.hasSelections()) {
            this->clearNull(group);
          }
          auto sum = sumVector->valueAt(decodedIndex);
          auto count = countVector->valueAt(decodedIndex);
          rows.applyToSelected(
              [&](vector_size_t i) { mergeAccumulators(group, sum, count); });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedPartial_.isNullAt(i)) {
          this->clearNull(group);
          auto decodedIndex = decodedPartial_.index(i);
          if (isPartialSumOverflow(sumVector, countVector, decodedIndex)) {
            // Find overflow, just set group to null.
            this->setNull(group);
          } else {
            auto sum = sumVector->valueAt(decodedIndex);
            auto count = countVector->valueAt(decodedIndex);
            mergeAccumulators(group, sum, count);
          }
        }
      });
    } else {
      if (rows.hasSelections()) {
        this->clearNull(group);
      }
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        if (isPartialSumOverflow(sumVector, countVector, decodedIndex)) {
          // Find overflow, just set group to null.
          this->setNull(group);
        } else {
          auto sum = sumVector->valueAt(decodedIndex);
          auto count = countVector->valueAt(decodedIndex);
          mergeAccumulators(group, sum, count);
        }
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<int128_t>();
    auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();
    VELOX_USER_CHECK_NOT_NULL(sumVector);

    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);
    rowVector->clearAllNulls();

    int64_t* rawCounts = countVector->mutableRawValues();
    int128_t* rawSums = sumVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto* accumulator = this->decimalAccumulator(group);
      std::optional<int128_t> adjustedSum = DecimalUtil::adjustSumForOverflow(
          accumulator->sum, accumulator->overflow);
      if (adjustedSum.has_value()) {
        rawCounts[i] = accumulator->count;
        rawSums[i] = adjustedSum.value();
      } else {
        // Find overflow.
        sumVector->setNull(i, true);
        rawCounts[i] = accumulator->count;
      }
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResultType>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = this->getRawNulls(vector);

    TResultType* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = this->decimalAccumulator(group);
      if (accumulator->count == 0) {
        // In Spark, if all inputs are null, count will be 0,
        // and the result of final avg will be null.
        vector->setNull(i, true);
      } else {
        this->clearNull(rawNulls, i);
        std::optional<TResultType> avg = computeAvg(accumulator);
        if (avg.has_value()) {
          rawValues[i] = avg.value();
        } else {
          // Find overflow.
          vector->setNull(i, true);
        }
      }
    }
  }

  std::optional<TResultType> computeAvg(
      LongDecimalWithOverflowState* accumulator) {
    std::optional<int128_t> validSum = DecimalUtil::adjustSumForOverflow(
        accumulator->sum, accumulator->overflow);
    if (!validSum.has_value()) {
      return std::nullopt;
    }

    auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*this->resultType().get());
    // Spark use DECIMAL(20,0) to represent long value.
    const uint8_t countPrecision = 20, countScale = 0;
    auto [sumPrecision, sumScale] =
        getDecimalPrecisionScale(*this->sumType_.get());
    auto [avgPrecision, avgScale] = computeResultPrecisionScale(
        sumPrecision, sumScale, countPrecision, countScale);
    auto sumRescale = computeRescaleFactor(sumScale, countScale, avgScale);
    auto countDecimal = accumulator->count;
    int128_t avg = 0;

    bool overflow = false;
    functions::sparksql::DecimalUtil::
        divideWithRoundUp<int128_t, int128_t, int128_t>(
            avg, validSum.value(), countDecimal, sumRescale, overflow);
    if (overflow) {
      return std::nullopt;
    }
    TResultType rescaledValue;
    const auto status = DecimalUtil::rescaleWithRoundUp<int128_t, TResultType>(
        avg,
        avgPrecision,
        avgScale,
        resultPrecision,
        resultScale,
        rescaledValue);
    return status.ok() ? std::optional<TResultType>(rescaledValue)
                       : std::nullopt;
  }

 private:
  template <typename UnscaledType>
  inline void mergeSumCount(
      LongDecimalWithOverflowState* accumulator,
      UnscaledType sum,
      int64_t count) {
    accumulator->count += count;
    accumulator->overflow +=
        DecimalUtil::addWithOverflow(accumulator->sum, sum, accumulator->sum);
  }

  template <bool tableHasNulls = true, class UnscaledType>
  void mergeAccumulators(
      char* group,
      const UnscaledType& otherSum,
      const int64_t& otherCount) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = this->decimalAccumulator(group);
    mergeSumCount(accumulator, otherSum, otherCount);
  }

  inline static bool isPartialSumOverflow(
      SimpleVector<int128_t>* sumVector,
      SimpleVector<int64_t>* countVector,
      int32_t index) {
    return sumVector->isNullAt(index) && !countVector->isNullAt(index) &&
        countVector->valueAt(index) > 0;
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
    return functions::sparksql::DecimalUtil::adjustPrecisionScale(
        precision, scale);
  }

  inline static std::pair<uint8_t, uint8_t> adjustPrecisionScale(
      const uint8_t precision,
      const uint8_t scale) {
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
  TypePtr sumType_;
};

TypePtr getDecimalSumType(
    const uint8_t rawInputPrecision,
    const uint8_t rawInputScale) {
  // This computational logic is derived from the definition of Spark SQL.
  return DECIMAL(std::min(38, rawInputPrecision + 10), rawInputScale);
}

} // namespace

/// Count is BIGINT() while sum and the final aggregates type depends on
/// the input types:
///       INPUT TYPE    |     SUM             |     AVG
///   ------------------|---------------------|------------------
///     DOUBLE          |     DOUBLE          |    DOUBLE
///     REAL            |     DOUBLE          |    DOUBLE
///     ALL INTs        |     DOUBLE          |    DOUBLE
///     DECIMAL         |     DECIMAL         |    DECIMAL
exec::AggregateRegistrationResult registerAverage(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;

  for (const auto& inputType :
       {"smallint", "integer", "bigint", "real", "double"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType("double")
                             .intermediateType("row(double,bigint)")
                             .argumentType(inputType)
                             .build());
  }

  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .integerVariable("r_precision", "min(38, a_precision + 4)")
          .integerVariable("r_scale", "min(38, a_scale + 4)")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(38, a_scale), BIGINT)")
          .returnType("DECIMAL(r_precision, r_scale)")
          .build());

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
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        const auto& inputType = argTypes[0];
        if (exec::isRawInput(step)) {
          switch (inputType->kind()) {
            case TypeKind::SMALLINT:
              return std::make_unique<
                  AverageAggregate<int16_t, double, double>>(resultType);
            case TypeKind::INTEGER:
              return std::make_unique<
                  AverageAggregate<int32_t, double, double>>(resultType);
            case TypeKind::BIGINT: {
              if (inputType->isShortDecimal()) {
                auto inputPrecision = inputType->asShortDecimal().precision();
                auto inputScale = inputType->asShortDecimal().scale();
                auto sumType =
                    DECIMAL(std::min(38, inputPrecision + 10), inputScale);
                if (exec::isPartialOutput(step)) {
                  return std::make_unique<
                      DecimalAverageAggregate<int64_t, int64_t>>(
                      resultType, sumType);
                } else {
                  if (resultType->isShortDecimal()) {
                    return std::make_unique<
                        DecimalAverageAggregate<int64_t, int64_t>>(
                        resultType, sumType);
                  } else if (resultType->isLongDecimal()) {
                    return std::make_unique<
                        DecimalAverageAggregate<int64_t, int128_t>>(
                        resultType, sumType);
                  } else {
                    VELOX_FAIL("Result type must be decimal");
                  }
                }
              }
              return std::make_unique<
                  AverageAggregate<int64_t, double, double>>(resultType);
            }
            case TypeKind::HUGEINT: {
              if (inputType->isLongDecimal()) {
                auto inputPrecision = inputType->asLongDecimal().precision();
                auto inputScale = inputType->asLongDecimal().scale();
                auto sumType = getDecimalSumType(inputPrecision, inputScale);
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t>>(
                    resultType, sumType);
              }
              VELOX_NYI();
            }
            case TypeKind::REAL:
              return std::make_unique<AverageAggregate<float, double, double>>(
                  resultType);
            case TypeKind::DOUBLE:
              return std::make_unique<AverageAggregate<double, double, double>>(
                  resultType);
            default:
              VELOX_FAIL(
                  "Unknown input type for {} aggregation {}",
                  name,
                  inputType->kindName());
          }
        } else {
          checkAvgIntermediateType(inputType);
          switch (resultType->kind()) {
            case TypeKind::REAL:
              return std::make_unique<AverageAggregate<int64_t, double, float>>(
                  resultType);
            case TypeKind::DOUBLE:
            case TypeKind::ROW:
              if (inputType->childAt(0)->isLongDecimal()) {
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t>>(
                    resultType, inputType->childAt(0));
              }
              return std::make_unique<
                  AverageAggregate<int64_t, double, double>>(resultType);
            case TypeKind::BIGINT:
              VELOX_USER_CHECK(resultType->isShortDecimal());
              return std::make_unique<
                  DecimalAverageAggregate<int64_t, int64_t>>(
                  resultType, inputType->childAt(0));
            case TypeKind::HUGEINT:
              VELOX_USER_CHECK(resultType->isLongDecimal());
              return std::make_unique<
                  DecimalAverageAggregate<int128_t, int128_t>>(
                  resultType, inputType->childAt(0));
            case TypeKind::VARBINARY:
              if (inputType->isLongDecimal()) {
                return std::make_unique<
                    DecimalAverageAggregate<int128_t, int128_t>>(
                    resultType, inputType->childAt(0));
              } else if (
                  inputType->isShortDecimal() ||
                  inputType->kind() == TypeKind::VARBINARY) {
                // If the input and out type are VARBINARY, then the
                // LongDecimalWithOverflowState is used and the template type
                // does not matter.
                return std::make_unique<
                    DecimalAverageAggregate<int64_t, int64_t>>(
                    resultType, inputType->childAt(0));
              }
              [[fallthrough]];
            default:
              VELOX_FAIL(
                  "Unsupported result type for final aggregation: {}",
                  resultType->kindName());
          }
        }
      },
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::functions::aggregate::sparksql
