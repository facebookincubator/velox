/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/sparksql/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {
namespace detail {

inline TypePtr getSumType(const TypePtr& rawInputType) {
  if (rawInputType->isRow()) {
    return rawInputType->childAt(0);
  }
  auto [p, s] = getDecimalPrecisionScale(*rawInputType.get());
  // In Spark,
  // intermediate sum precision = input precision + 10
  // intermediate sum scale = input scale
  return DECIMAL(std::min<uint8_t>(LongDecimalType::kMaxPrecision, p + 10), s);
}

} // namespace detail

/// @tparam TInputType The raw input data type.
/// @tparam TResultType The type of the output average value.
template <typename TInputType, typename TResultType>
class DecimalAverageAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType = Row</*sum*/ int128_t,
                               /*count*/ int64_t>;

  using OutputType = TResultType;

  /// Spark's decimal sum doesn't have the concept of a null group, each group
  /// is initialized with an initial value, where sum = 0 and count = 0. The
  /// final agg may fallback to being executed in Spark, so the meaning of the
  /// intermediate data should be consistent with Spark. Therefore, we need to
  /// use the parameter nonNullGroup in writeIntermediateResult to output a null
  /// group as sum = 0, count = 0. nonNullGroup is only available when
  /// default-null behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  void initialize(
      core::AggregationNode::Step step,
      const std::vector<TypePtr>& argTypes,
      const TypePtr& resultType) {
    VELOX_CHECK_EQ(argTypes.size(), 1);
    auto inputType = argTypes[0];
    this->sumType_ = detail::getSumType(inputType);
    this->resultType_ = resultType;
  }

  static bool toIntermediate(
      exec::out_type<Row<int128_t, int64_t>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(in.value(), 1));
    } else {
      out.copy_from(std::make_tuple(0, 0));
    }
    return true;
  }

  /// This struct stores the sum of input values, overflow during accumulation,
  /// and the count number of the input values. If the count is not 0, then if
  /// sum is nullopt that means an overflow has happened. For null group, the
  /// variables in accumulator should be meaningless, so the values of sum and
  /// count are ignored in both writeIntermediateResult and writeFinalResult.
  struct AccumulatorType {
    std::optional<int128_t> sum{0};
    int64_t overflow{0};
    int64_t count{0};

    DecimalAverageAggregate* fn;

    static constexpr bool is_aligned_ = true;

    AccumulatorType() = delete;

    AccumulatorType(
        HashStringAllocator* /*allocator*/,
        DecimalAverageAggregate* fn)
        : fn(fn) {}

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data) {
      if (!data.has_value()) {
        return false;
      }
      if (!sum.has_value()) {
        // sum is initialized to 0. When it is nullopt, it implies that the
        // count number of the input values must not be 0.
        VELOX_CHECK(count != 0);
        return true;
      }
      int128_t result;
      overflow +=
          DecimalUtil::addWithOverflow(result, data.value(), sum.value());
      sum = result;
      count += 1;
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<int128_t, int64_t>> other) {
      if (!other.has_value()) {
        return false;
      }
      auto const otherSum = other.value().template at<0>();
      auto const otherCount = other.value().template at<1>();

      // otherCount is never null.
      VELOX_CHECK(otherCount.has_value());
      if (count == 0 && otherCount.value() == 0) {
        // Both accumulators have no input values, no need to do the
        // combination.
        return false;
      }

      bool currentOverflow = count > 0 && !sum.has_value();
      bool otherOverflow = otherCount.value() > 0 && !otherSum.has_value();
      if (currentOverflow || otherOverflow) {
        sum = std::nullopt;
        count += otherCount.value();
      } else {
        int128_t result;
        overflow +=
            DecimalUtil::addWithOverflow(result, otherSum.value(), sum.value());
        sum = result;
        count += otherCount.value();
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup) {
        // If a group is null, all values in this group are null. In Spark, this
        // group will be the initial value, where sum is 0 and count is 0.
        out =
            std::make_tuple(static_cast<int128_t>(0), static_cast<int64_t>(0));
      } else {
        if (!sum.has_value()) {
          // Sum should be set to null on overflow.
          out.template set_null_at<0>();
          out.template get_writer_at<1>() = count;
        } else {
          auto adjustedSum =
              DecimalUtil::adjustSumForOverflow(sum.value(), overflow);
          if (adjustedSum.has_value()) {
            out = std::make_tuple(adjustedSum.value(), count);
          } else {
            out.template set_null_at<0>();
            out.template get_writer_at<1>() = count;
          }
        }
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!nonNullGroup || count == 0) {
        // In Spark, if all inputs are null, count will be 0, and the result of
        // average value will be null.
        return false;
      }
      auto finalResult = computeFinalResult();
      if (finalResult.has_value()) {
        out = static_cast<TResultType>(finalResult.value());
        return true;
      }
      // Sum should be set to null on overflow.
      return false;
    }

   private:
    std::optional<int128_t> computeFinalResult() const {
      if (!sum.has_value()) {
        return std::nullopt;
      }
      const auto adjustedSum =
          DecimalUtil::adjustSumForOverflow(sum.value(), overflow);
      if (!adjustedSum.has_value()) {
        // Found overflow during computing adjusted sum.
        return std::nullopt;
      }

      auto [resultPrecision, resultScale] =
          getDecimalPrecisionScale(*fn->resultType_.get());
      auto [sumPrecision, sumScale] =
          getDecimalPrecisionScale(*fn->sumType_.get());

      // Spark use DECIMAL(20,0) to represent long value.
      static const uint8_t countPrecision = 20;
      static const uint8_t countScale = 0;

      auto [dividePrecision, divideScale] =
          functions::sparksql::DecimalUtil::computeDivideResultPrecisionScale<
              true>(sumPrecision, sumScale, countPrecision, countScale);
      divideScale = std::max<uint8_t>(divideScale, resultScale);
      auto sumRescale = divideScale - sumScale + countScale;
      int128_t avg;
      bool overflow = false;
      functions::sparksql::DecimalUtil::
          divideWithRoundUp<int128_t, int128_t, int128_t>(
              avg, adjustedSum.value(), count, sumRescale, overflow);
      if (overflow) {
        return std::nullopt;
      }
      TResultType rescaledValue;
      const auto status =
          DecimalUtil::rescaleWithRoundUp<int128_t, TResultType>(
              avg,
              dividePrecision,
              divideScale,
              resultPrecision,
              resultScale,
              rescaledValue);
      return status.ok() ? std::optional<int128_t>(rescaledValue)
                         : std::nullopt;
    }
  };

  TypePtr resultType_;
  TypePtr sumType_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
