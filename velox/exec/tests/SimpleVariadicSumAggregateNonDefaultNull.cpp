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

#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate {

namespace {

// A variant of VariadicSumAggregate that ignores nulls but processes all rows.
// Unlike the original which skips entire rows when ANY argument is null
// (default null behavior), this version uses non-default null behavior to
// process all rows and simply ignores null values when summing.
//
// This function takes a dummy integer argument followed by a variadic list of
// integers. It returns an array where the i-th element is the sum of all i-th
// variadic arguments across all rows (ignoring nulls).
//
// Example:
//   SELECT variadic_sum_agg_ignore_nulls(3, a, b, c) FROM (
//     VALUES (1, NULL, 3), (4, 5, 6)
//   ) AS t(a, b, c)
//   => [5, 5, 9]  (i.e., [1+4, 0+5, 3+6])
class VariadicSumAggregateNonDefaultNull {
 public:
  using InputType = AggregateInputType<int64_t, Variadic<int64_t>>;

  using IntermediateType = Array<int64_t>;

  using OutputType = Array<int64_t>;

  // Disable default null behavior so we process all rows, not just those
  // where all arguments are non-null.
  static constexpr bool default_null_behavior_ = false;

  struct AccumulatorType {
    std::vector<int64_t> sums_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        VariadicSumAggregateNonDefaultNull* /*fn*/)
        : sums_{} {}

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<int64_t> /*dummy*/,
        exec::optional_arg_type<Variadic<int64_t>> variadicArgs) {
      if (!variadicArgs.has_value()) {
        return false;
      }

      const auto& args = variadicArgs.value();

      if (sums_.empty()) {
        sums_.resize(args.size(), 0);
      }

      for (auto i = 0; i < args.size(); ++i) {
        if (args.at(i).has_value()) {
          sums_[i] += args.at(i).value();
        }
      }
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Array<int64_t>> other) {
      if (!other.has_value()) {
        return false;
      }

      const auto& otherArray = other.value();

      if (sums_.empty()) {
        sums_.resize(otherArray.size(), 0);
      }

      for (auto i = 0; i < otherArray.size(); ++i) {
        if (otherArray.at(i).has_value()) {
          sums_[i] += otherArray.at(i).value();
        }
      }
      return true;
    }

    bool writeFinalResult(
        bool nonNullGroup,
        exec::out_type<Array<int64_t>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<Array<int64_t>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }
  };
};

} // namespace

exec::AggregateRegistrationResult
registerSimpleVariadicSumAggregateNonDefaultNull(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("array(bigint)")
          .intermediateType("array(bigint)")
          .argumentType("bigint")
          .argumentType("bigint")
          .variableArity()
          .build()};

  return exec::registerAggregateFunction(
      name,
      signatures,
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_GE(
            argTypes.size(), 1, "{} requires at least 1 argument", name);
        return std::make_unique<
            SimpleAggregateAdapter<VariadicSumAggregateNonDefaultNull>>(
            step, argTypes, resultType);
      },
      true /*registerCompanionFunctions*/,
      true /*overwrite*/);
}

} // namespace facebook::velox::aggregate
