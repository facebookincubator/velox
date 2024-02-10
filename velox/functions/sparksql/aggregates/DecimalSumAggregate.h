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

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {

/// @tparam TInputType The raw input data type.
/// @tparam TSumType The type of sum in the output of partial aggregation or the
/// final output type of final aggregation.
template <typename TInputType, typename TSumType>
class DecimalSumAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType =
      Row</*sum*/ TSumType,
          /*isEmpty*/ bool>;

  using OutputType = TSumType;

  // Spark's decimal sum doesn't have the concept of a null group, each group is
  // initialized with an initial value, where sum = 0 and isEmpty = true.
  // Therefore, to maintain consistency, we need to use the parameter
  // nonNullGroup in writeIntermediateResult to output a null group as sum =
  // 0, isEmpty = true. nonNullGroup is only available when default-null
  // behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  static bool toIntermediate(
      exec::out_type<Row<TSumType, bool>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(static_cast<TSumType>(in.value()), false));
      return true;
    }
    return false;
  }

  // This struct stores the sum of input values, overflow during accumulation,
  // and a bool value isEmpty used to indicate whether all inputs are null. The
  // initial value of sum is 0. We need to keep sum unchanged if the input is
  // null, as sum function ignores null input. If the isEmpty is true, then it
  // means there were no values to begin with or all the values were null, so
  // the result will be null. If the isEmpty is false, then if sum is nullopt
  // that means an overflow has happened, it returns null.
  struct AccumulatorType {
    std::optional<int128_t> sum_{0};
    int64_t overflow_{0};
    bool isEmpty_{true};

    AccumulatorType() = delete;

    explicit AccumulatorType(HashStringAllocator* /*allocator*/) {}

    std::optional<int128_t> computeFinalResult() const {
      if (!sum_.has_value()) {
        return std::nullopt;
      }
      auto adjustedSum =
          DecimalUtil::adjustSumForOverflow(sum_.value(), overflow_);
      uint8_t maxPrecision = std::is_same_v<TSumType, int128_t>
          ? LongDecimalType::kMaxPrecision
          : ShortDecimalType::kMaxPrecision;
      if (adjustedSum.has_value() &&
          DecimalUtil::valueInPrecisionRange(adjustedSum, maxPrecision)) {
        return adjustedSum;
      } else {
        // Found overflow during computing adjusted sum.
        return std::nullopt;
      }
    }

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data) {
      if (!data.has_value()) {
        return false;
      }
      int128_t result;
      overflow_ +=
          DecimalUtil::addWithOverflow(result, data.value(), sum_.value());
      sum_ = result;
      isEmpty_ = false;
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<TSumType, bool>> other) {
      if (!other.has_value()) {
        return false;
      }
      auto otherSum = other.value().template at<0>();
      auto otherIsEmpty = other.value().template at<1>();

      // IsEmpty should always has value.
      VELOX_CHECK(otherIsEmpty.has_value());

      bool bufferOverflow = !isEmpty_ && !sum_.has_value();
      bool inputOverflow = !otherIsEmpty.value() && !otherSum.has_value();
      if (bufferOverflow || inputOverflow) {
        sum_ = std::nullopt;
        return false;
      } else {
        int128_t result;
        overflow_ += DecimalUtil::addWithOverflow(
            result, otherSum.value(), sum_.value());
        sum_ = result;
        isEmpty_ &= otherIsEmpty.value();
        return true;
      }
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup) {
        // If a group is null, maybe all values in this group are null. In
        // Spark, this group will be the initial value, where sum is 0 and
        // isEmpty is true.
        out = std::make_tuple(static_cast<TSumType>(0), true);
      } else {
        auto finalResult = computeFinalResult();
        if (finalResult.has_value()) {
          out = std::make_tuple(
              static_cast<TSumType>(finalResult.value()), isEmpty_);
        } else {
          // Sum should be set to null on overflow, and
          // isEmpty should be set to false.
          out.template set_null_at<0>();
          out.template get_writer_at<1>() = false;
        }
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!nonNullGroup || isEmpty_) {
        // If isEmpty is true, we should set null.
        return false;
      }
      auto finalResult = computeFinalResult();
      if (finalResult.has_value()) {
        out = static_cast<TSumType>(finalResult.value());
        return true;
      } else {
        // Sum should be set to null on overflow.
        return false;
      }
    }
  };
};

} // namespace facebook::velox::functions::aggregate::sparksql
