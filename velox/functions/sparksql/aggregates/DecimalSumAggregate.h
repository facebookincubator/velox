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

/// TInputType refer to the raw input data type. TSumType refer to the type of
/// sum in the output of partial aggregation or the final output type of final
/// aggregation.
template <typename TInputType, typename TSumType>
class DecimalSumAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType =
      Row</*sum*/ TSumType,
          /*isEmpty*/ bool>;

  using OutputType = TSumType;

  static constexpr bool default_null_behavior_ = true;

  static bool toIntermediate(
      exec::out_type<Row<TSumType, bool>>& out,
      exec::arg_type<TInputType> in) {
    out.copy_from(std::make_tuple(static_cast<TSumType>(in), false));
    return true;
  }

  // This struct stores the sum of input values, overflow during accumulation,
  // and a bool value isEmpty used to indicate whether all inputs are null. The
  // initial value of sum is 0. We need to keep sum unchanged if the input is
  // null, as sum function ignores null input. If the isEmpty is true, then it
  // means there were no values to begin with or all the values were null, so
  // the result will be null. If the isEmpty is false, then if sum is nullopt
  // that means an overflow has happened, it returns null.
  struct AccumulatorType {
    std::optional<int128_t> sum_;
    int64_t overflow_;
    bool isEmpty_;

    AccumulatorType() = delete;

    explicit AccumulatorType(HashStringAllocator* /*allocator*/) {
      sum_ = 0;
      overflow_ = 0;
      isEmpty_ = true;
    }

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
        // Find overflow during computing adjusted sum.
        return std::nullopt;
      }
    }

    static bool isIntermediateResultOverflow(
        std::optional<TSumType> sum,
        std::optional<bool> isEmpty) {
      return !sum.has_value() && isEmpty.has_value();
    }

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<TInputType> data) {
      int128_t result;
      overflow_ += DecimalUtil::addWithOverflow(result, data, sum_.value());
      sum_ = result;
      isEmpty_ = false;
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Row<TSumType, bool>> other) {
      auto otherSum = other.template at<0>();
      auto otherIsEmpty = other.template at<1>();
      bool bufferOverflow = !isEmpty_ && !sum_.has_value();
      bool inputOverflow = isIntermediateResultOverflow(otherSum, otherIsEmpty);
      if (bufferOverflow || inputOverflow) {
        sum_ = std::nullopt;
      } else {
        int128_t result;
        overflow_ += DecimalUtil::addWithOverflow(
            result, otherSum.value(), sum_.value());
        sum_ = result;
        isEmpty_ &= otherIsEmpty.value();
      }
    }

    bool writeIntermediateResult(exec::out_type<IntermediateType>& out) {
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
      return true;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      if (isEmpty_) {
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
