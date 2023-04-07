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

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/lib/SimpleNumericAggregate.h"
#include "velox/functions/prestosql/CheckedArithmeticImpl.h"
#include "velox/functions/prestosql/aggregates/DecimalAggregate.h"

namespace facebook::velox::aggregate::prestosql {

template <typename TInput, typename TAccumulator, typename ResultType>
class SumAggregate
    : public SimpleNumericAggregate<TInput, TAccumulator, ResultType> {
  using BaseAggregate =
      SimpleNumericAggregate<TInput, TAccumulator, ResultType>;

 public:
  explicit SumAggregate(TypePtr resultType) : BaseAggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(TAccumulator);
  }

  int32_t accumulatorAlignmentSize() const override {
    return 1;
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *exec::Aggregate::value<TAccumulator>(groups[i]) = 0;
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<ResultType>(
        groups, numGroups, result, [&](char* group) {
          // 'ResultType' and 'TAccumulator' might not be same such as sum(real)
          // and we do an explicit type conversion here.
          return (ResultType)(*BaseAggregate::Aggregate::template value<
                              TAccumulator>(group));
        });
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::template doExtractValues<TAccumulator>(
        groups, numGroups, result, [&](char* group) {
          return *BaseAggregate::Aggregate::template value<TAccumulator>(group);
        });
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    updateInternal<TAccumulator>(groups, rows, args, mayPushdown);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    updateInternal<TAccumulator, TAccumulator>(groups, rows, args, mayPushdown);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::template updateOneGroup<TAccumulator>(
        group,
        rows,
        args[0],
        &updateSingleValue<TAccumulator>,
        &updateDuplicateValues<TAccumulator>,
        mayPushdown,
        TAccumulator(0));
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::template updateOneGroup<TAccumulator, TAccumulator>(
        group,
        rows,
        args[0],
        &updateSingleValue<TAccumulator>,
        &updateDuplicateValues<TAccumulator>,
        mayPushdown,
        TAccumulator(0));
  }

 protected:
  // TData is used to store the updated sum state. It can be either
  // TAccumulator or TResult, which in most cases are the same, but for
  // sum(real) can differ. TValue is used to decode the sum input 'args'.
  // It can be either TAccumulator or TInput, which is most cases are the same
  // but for sum(real) can differ.
  template <typename TData, typename TValue = TInput>
  void updateInternal(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) {
    const auto& arg = args[0];

    if (mayPushdown && arg->isLazy()) {
      BaseAggregate::template pushdown<SumHook<TValue, TData>>(
          groups, rows, arg);
      return;
    }

    if (exec::Aggregate::numNulls_) {
      BaseAggregate::template updateGroups<true, TData, TValue>(
          groups, rows, arg, &updateSingleValue<TData>, false);
    } else {
      BaseAggregate::template updateGroups<false, TData, TValue>(
          groups, rows, arg, &updateSingleValue<TData>, false);
    }
  }

 private:
  /// Update functions that check for overflows for integer types.
  /// For floating points, an overflow results in +/- infinity which is a
  /// valid output.
  template <typename TData>
  static void updateSingleValue(TData& result, TData value) {
    if constexpr (
        std::is_same_v<TData, double> || std::is_same_v<TData, float>) {
      result += value;
    } else {
      result = functions::checkedPlus<TData>(result, value);
    }
  }

  template <typename TData>
  static void updateDuplicateValues(TData& result, TData value, int n) {
    if constexpr (
        std::is_same_v<TData, double> || std::is_same_v<TData, float>) {
      result += n * value;
    } else {
      result = functions::checkedPlus<TData>(
          result, functions::checkedMultiply<TData>(TData(n), value));
    }
  }
};

template <typename TInputType>
class DecimalSumAggregate
    : public DecimalAggregate<UnscaledLongDecimal, TInputType> {
 public:
  explicit DecimalSumAggregate(TypePtr resultType)
      : DecimalAggregate<UnscaledLongDecimal, TInputType>(resultType) {}

  virtual UnscaledLongDecimal computeFinalValue(
      LongDecimalWithOverflowState* accumulator) final {
    // Value is valid if the conditions below are true.
    int128_t sum = accumulator->sum;
    if ((accumulator->overflow == 1 && accumulator->sum < 0) ||
        (accumulator->overflow == -1 && accumulator->sum > 0)) {
      sum = static_cast<int128_t>(
          DecimalUtil::kOverflowMultiplier * accumulator->overflow +
          accumulator->sum);
    } else {
      VELOX_CHECK(accumulator->overflow == 0, "Decimal overflow");
    }

    VELOX_CHECK(UnscaledLongDecimal::valueInRange(sum), "Decimal overflow");
    return UnscaledLongDecimal(sum);
  }
};

template <template <typename U, typename V, typename W> class T>
bool registerSum(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("real")
          .intermediateType("double")
          .argumentType("real")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .returnType("double")
          .intermediateType("double")
          .argumentType("double")
          .build(),
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("VARBINARY")
          .returnType("DECIMAL(38, a_scale)")
          .build(),
  };

  for (const auto& inputType : {"tinyint", "smallint", "integer", "bigint"}) {
    signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                             .returnType("bigint")
                             .intermediateType("bigint")
                             .argumentType(inputType)
                             .build());
  }

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        auto inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::TINYINT:
            return std::make_unique<T<int8_t, int64_t, int64_t>>(BIGINT());
          case TypeKind::SMALLINT:
            return std::make_unique<T<int16_t, int64_t, int64_t>>(BIGINT());
          case TypeKind::INTEGER:
            return std::make_unique<T<int32_t, int64_t, int64_t>>(BIGINT());
          case TypeKind::BIGINT:
            return std::make_unique<T<int64_t, int64_t, int64_t>>(BIGINT());
          case TypeKind::REAL:
            if (resultType->kind() == TypeKind::REAL) {
              return std::make_unique<T<float, double, float>>(resultType);
            }
            return std::make_unique<T<float, double, double>>(DOUBLE());
          case TypeKind::DOUBLE:
            if (resultType->kind() == TypeKind::REAL) {
              return std::make_unique<T<double, double, float>>(resultType);
            }
            return std::make_unique<T<double, double, double>>(DOUBLE());
          case TypeKind::SHORT_DECIMAL:
            return std::make_unique<DecimalSumAggregate<UnscaledShortDecimal>>(
                resultType);
          case TypeKind::VARBINARY:
          // Always use UnscaledLongDecimal template for Varbinary as the result
          // type is either UnscaledLongDecimal or
          // UnscaledLongDecimalWithOverflowState.
          case TypeKind::LONG_DECIMAL:
            return std::make_unique<DecimalSumAggregate<UnscaledLongDecimal>>(
                resultType);

          default:
            VELOX_CHECK(
                false,
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      });
}

} // namespace facebook::velox::aggregate::prestosql
