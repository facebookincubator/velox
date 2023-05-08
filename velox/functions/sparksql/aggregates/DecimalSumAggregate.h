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
#include "velox/functions/prestosql/CheckedArithmeticImpl.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql::aggregates {

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
  explicit DecimalSumAggregate(TypePtr resultType)
      : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalSum);
  }

  /// Use UnscaledLongDecimal instead of int128_t because some CPUs don't
  /// support misaligned access to int128_t type.
  int32_t accumulatorAlignmentSize() const override {
    return static_cast<int32_t>(sizeof(UnscaledLongDecimal));
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) DecimalSum();
    }
  }

  UnscaledLongDecimal computeFinalValue(
      DecimalSum* decimalSum,
      const TypePtr sumType) {
    int128_t sum = decimalSum->sum;
    if ((decimalSum->overflow == 1 && decimalSum->sum < 0) ||
        (decimalSum->overflow == -1 && decimalSum->sum > 0)) {
      sum = static_cast<int128_t>(
          DecimalUtil::kOverflowMultiplier * decimalSum->overflow +
          decimalSum->sum);
    } else {
      VELOX_CHECK(
          decimalSum->overflow == 0,
          "overflow: decimal sum struct overflow not eq 0");
    }

    auto [resultPrecision, resultScale] =
        getDecimalPrecisionScale(*sumType.get());
    auto resultMax = DecimalUtil::kPowersOfTen[resultPrecision] - 1;
    auto resultMin = -resultMax;
    VELOX_CHECK(
        (sum >= resultMin) && (sum <= resultMax),
        "overflow: sum value not in result decimal range");

    return UnscaledLongDecimal(sum);
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
          // isEmpty is trun means all values are null
          vector->setNull(i, true);
        } else {
          try {
            rawValues[i] = computeFinalValue(decimalSum, result->get()->type());
          } catch (const VeloxException& err) {
            if (err.message().find("overflow") != std::string::npos) {
              // find overflow in computation
              vector->setNull(i, true);
            } else {
              VELOX_FAIL("compute sum failed");
            }
          }
        }
      }
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
    // Bool uses compact representation, use mutableRawValues<uint64_t> and
    // bits::setBit instead.
    auto* rawIsEmpty = isEmptyVector->mutableRawValues<uint64_t>();
    uint64_t* rawNulls = getRawNulls(rowVector);

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* decimalSum = accumulator(group);
        try {
          rawSums[i] = computeFinalValue(decimalSum, sumVector->type());
          bits::setBit(rawIsEmpty, i, decimalSum->isEmpty);
        } catch (const VeloxException& err) {
          if (err.message().find("overflow") != std::string::npos) {
            // find overflow in computation
            sumVector->setNull(i, true);
            bits::setBit(rawIsEmpty, i, false);
          } else {
            VELOX_FAIL("compute sum failed");
          }
        }
      }
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
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(groups[i], UnscaledLongDecimal(value), false);
        });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedRaw_.isNullAt(i)) {
          return;
        }
        updateNonNullValue(
            groups[i],
            UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)),
            false);
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(
            groups[i], UnscaledLongDecimal(data[i]), false);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(
            groups[i],
            UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)),
            false);
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
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.template applyToSelected([&](vector_size_t i) {
          updateNonNullValue(group, UnscaledLongDecimal(value), false);
        });
      } else {
        clearNull(group);
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              group,
              UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)),
              false);
        } else {
          clearNull(group);
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      DecimalSum decimalSum;
      rows.applyToSelected([&](vector_size_t i) {
        decimalSum.overflow += DecimalUtil::addWithOverflow(
            decimalSum.sum, data[i].unscaledValue(), decimalSum.sum);
        decimalSum.isEmpty = false;
      });
      mergeAccumulators(group, decimalSum);
    } else {
      DecimalSum decimalSum;
      rows.applyToSelected([&](vector_size_t i) {
        decimalSum.overflow += DecimalUtil::addWithOverflow(
            decimalSum.sum,
            decodedRaw_.valueAt<TInputType>(i).unscaledValue(),
            decimalSum.sum);
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
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TInputType>>();
    auto isEmptyVector = baseRowVector->childAt(1)->as<SimpleVector<bool>>();
    DCHECK(isEmptyVector);

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto sum = sumVector->valueAt(decodedIndex);
        auto isEmpty = isEmptyVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          clearNull(groups[i]);
          updateNonNullValue(groups[i], UnscaledLongDecimal(sum), isEmpty);
        });
      } else {
        auto decodedIndex = decodedPartial_.index(0);
        if ((!isEmptyVector->isNullAt(decodedIndex) &&
             !isEmptyVector->valueAt(decodedIndex)) &&
            sumVector->isNullAt(decodedIndex)) {
          rows.applyToSelected(
              [&](vector_size_t i) { setOverflowGroup(groups[i]); });
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          // if isEmpty is false and if sum is null, then it means
          // we have had an overflow
          auto decodedIndex = decodedPartial_.index(i);
          if ((!isEmptyVector->isNullAt(decodedIndex) &&
               !isEmptyVector->valueAt(decodedIndex)) &&
              sumVector->isNullAt(decodedIndex)) {
            setOverflowGroup(groups[i]);
          }
          return;
        }
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        auto isEmpty = isEmptyVector->valueAt(decodedIndex);
        updateNonNullValue(groups[i], UnscaledLongDecimal(sum), isEmpty);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        auto isEmpty = isEmptyVector->valueAt(decodedIndex);
        updateNonNullValue(groups[i], UnscaledLongDecimal(sum), isEmpty);
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
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TInputType>>();
    auto isEmptyVector = baseRowVector->childAt(1)->as<SimpleVector<bool>>();
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto sum = sumVector->valueAt(decodedIndex);
        auto isEmpty = isEmptyVector->valueAt(decodedIndex);
        if (rows.hasSelections()) {
          clearNull(group);
        }
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(group, UnscaledLongDecimal(sum), isEmpty);
        });
      } else {
        auto decodedIndex = decodedPartial_.index(0);
        if ((!isEmptyVector->isNullAt(decodedIndex) &&
             !isEmptyVector->valueAt(decodedIndex)) &&
            sumVector->isNullAt(decodedIndex)) {
          setOverflowGroup(group);
        }
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedPartial_.isNullAt(i)) {
          clearNull(group);
          auto decodedIndex = decodedPartial_.index(i);
          auto sum = sumVector->valueAt(decodedIndex);
          auto isEmpty = isEmptyVector->valueAt(decodedIndex);
          updateNonNullValue(group, UnscaledLongDecimal(sum), isEmpty);
        } else {
          // if isEmpty is false and if sum is null, then it means
          // we have had an overflow
          auto decodedIndex = decodedPartial_.index(i);
          if ((!isEmptyVector->isNullAt(decodedIndex) &&
               !isEmptyVector->valueAt(decodedIndex)) &&
              sumVector->isNullAt(decodedIndex)) {
            setOverflowGroup(group);
          }
        }
      });
    } else {
      if (rows.hasSelections()) {
        clearNull(group);
      }
      rows.applyToSelected([&](vector_size_t i) {
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        auto isEmpty = isEmptyVector->valueAt(decodedIndex);
        updateNonNullValue(group, UnscaledLongDecimal(sum), isEmpty);
      });
    }
  }

 private:
  template <bool tableHasNulls = true>
  inline void
  updateNonNullValue(char* group, UnscaledLongDecimal value, bool isEmpty) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto decimalSum = accumulator(group);
    decimalSum->overflow += DecimalUtil::addWithOverflow(
        decimalSum->sum, value.unscaledValue(), decimalSum->sum);
    decimalSum->isEmpty &= isEmpty;
  }

  inline void setOverflowGroup(char* group) {
    setNull(group);
    auto decimalSum = accumulator(group);
    decimalSum->isEmpty = false;
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

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
};

bool registerDecimalSumAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(a_precision, a_scale), BOOLEAN)")
          .returnType("DECIMAL(a_precision, a_scale)")
          .build(),
  };

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        auto& inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::SHORT_DECIMAL:
            return std::make_unique<
                DecimalSumAggregate<UnscaledShortDecimal, UnscaledLongDecimal>>(
                resultType);
          case TypeKind::LONG_DECIMAL:
            return std::make_unique<
                DecimalSumAggregate<UnscaledLongDecimal, UnscaledLongDecimal>>(
                resultType);
          case TypeKind::ROW: {
            DCHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            switch (sumInputType->kind()) {
              case TypeKind::SHORT_DECIMAL:
                return std::make_unique<DecimalSumAggregate<
                    UnscaledShortDecimal,
                    UnscaledLongDecimal>>(resultType);
              case TypeKind::LONG_DECIMAL:
                return std::make_unique<DecimalSumAggregate<
                    UnscaledLongDecimal,
                    UnscaledLongDecimal>>(resultType);
              default:
                VELOX_FAIL(
                    "Unknown sum type for {} aggregation {}",
                    name,
                    sumInputType->kindName());
            }
          }
          default:
            VELOX_CHECK(
                false,
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      },
      true);
}

} // namespace facebook::velox::functions::sparksql::aggregates