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

#include <optional>

#include "velox/exec/SimpleAggregateAdapter.h"

namespace facebook::velox::functions::aggregate::sparksql {

/// Spark's try_sum for integral input: the sum of the input values, or NULL
/// once the sum overflows BIGINT. Spark's sum wraps around instead (or throws
/// in ANSI mode), so try_sum is a separate function. For DECIMAL, REAL and
/// DOUBLE input Spark's try_sum gives the same result as sum, so only the
/// integral types are registered here.
///
/// The intermediate result is Spark's aggregation buffer, ROW(sum, isEmpty):
/// sum is NULL after an overflow and isEmpty is true while no non-NULL input
/// has been seen. A final aggregation can therefore run in Spark.
///
/// @tparam TInputType The raw input type.
template <typename TInputType>
class TrySumAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType = Row</*sum*/ int64_t,
                               /*isEmpty*/ bool>;

  using OutputType = int64_t;

  /// Like Spark's decimal sum, a null group is written as the initial buffer,
  /// sum = 0 and isEmpty = true, which needs nonNullGroup in
  /// writeIntermediateResult.
  static constexpr bool default_null_behavior_ = false;

  static bool toIntermediate(
      exec::out_type<Row<int64_t, bool>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(static_cast<int64_t>(in.value()), false));
    } else {
      out.copy_from(std::make_tuple(static_cast<int64_t>(0), true));
    }
    return true;
  }

  struct AccumulatorType {
    // The sum so far, or nullopt once it has overflowed. Starts at 0 like
    // Spark's buffer; isEmpty tells whether any non-NULL input was added.
    std::optional<int64_t> sum{0};
    bool isEmpty{true};

    static constexpr bool is_aligned_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        TrySumAggregate<TInputType>* /*fn*/) {}

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data) {
      if (!data.has_value()) {
        return false;
      }
      isEmpty = false;
      add(static_cast<int64_t>(data.value()));
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<int64_t, bool>> other) {
      if (!other.has_value()) {
        return false;
      }
      const auto otherSum = other.value().template at<0>();
      const auto otherIsEmpty = other.value().template at<1>();
      // isEmpty is never null.
      VELOX_CHECK(otherIsEmpty.has_value());
      if (otherIsEmpty.value()) {
        // An empty buffer holds sum = 0 and adds nothing.
        return false;
      }
      isEmpty = false;
      if (!otherSum.has_value()) {
        // The other buffer has overflowed.
        sum = std::nullopt;
      } else {
        add(otherSum.value());
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup) {
        // A null group is Spark's initial buffer.
        out = std::make_tuple(int64_t{0}, true);
      } else if (sum.has_value()) {
        out = std::make_tuple(sum.value(), isEmpty);
      } else {
        // Overflow: sum is NULL and the buffer is not empty.
        out.template set_null_at<0>();
        out.template get_writer_at<1>() = false;
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!nonNullGroup || isEmpty || !sum.has_value()) {
        return false;
      }
      out = sum.value();
      return true;
    }

   private:
    // Adds 'value' to the sum; the sum becomes nullopt on overflow and stays
    // so, as in Spark where the buffer is NULL after the first overflow.
    void add(int64_t value) {
      if (!sum.has_value()) {
        return;
      }
      int64_t result;
      if (__builtin_add_overflow(sum.value(), value, &result)) {
        sum = std::nullopt;
      } else {
        sum = result;
      }
    }
  };
};

/// Registers try_sum for TINYINT, SMALLINT, INTEGER and BIGINT input.
exec::AggregateRegistrationResult registerTrySum(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
