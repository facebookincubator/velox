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

// An aggregate function that demonstrates variadic argument support in
// SimpleAggregateAdapter. This function takes a dummy integer argument followed
// by a variadic list of integers. It returns an array where the i-th element is
// the sum of all i-th variadic arguments across all rows.
//
// Example:
//   SELECT variadic_sum_agg(3, a, b, c) FROM (
//     VALUES (1, 2, 3), (4, 5, 6)
//   ) AS t(a, b, c)
//   => [5, 7, 9]  (i.e., [1+4, 2+5, 3+6])
class VariadicSumAggregateDefaultNull {
 public:
  // Force the function to take a dummy integer before the variadic arguments to
  // test variadic list not starting from the beginning.
  using InputType = AggregateInputType<int64_t, Variadic<int64_t>>;

  using IntermediateType = Array<int64_t>;

  using OutputType = Array<int64_t>;

  struct AccumulatorType {
    std::vector<int64_t> sums_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        VariadicSumAggregateDefaultNull* /*fn*/)
        : sums_{} {}

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<int64_t> /*dummy*/,
        exec::arg_type<Variadic<int64_t>> variadicArgs) {
      // Initialize sums_ based on the actual number of variadic arguments.
      if (sums_.empty()) {
        sums_.resize(variadicArgs.size(), 0);
      }

      for (auto i = 0; i < variadicArgs.size(); ++i) {
        sums_[i] += variadicArgs.at(i).value();
      }
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Array<int64_t>> other) {
      // Initialize sums_ based on the incoming array size if not yet
      // initialized.
      if (sums_.empty()) {
        sums_.resize(other.size(), 0);
      }

      // Add element-wise.
      for (auto i = 0; i < other.size(); ++i) {
        if (other.at(i).has_value()) {
          sums_[i] += other.at(i).value();
        }
      }
    }

    bool writeFinalResult(exec::out_type<Array<int64_t>>& out) {
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }

    bool writeIntermediateResult(exec::out_type<Array<int64_t>>& out) {
      for (const auto& sum : sums_) {
        out.add_item() = sum;
      }
      return true;
    }
  };
};

} // namespace

exec::AggregateRegistrationResult registerSimpleVariadicSumAggregateDefaultNull(
    const std::string& name) {
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
            SimpleAggregateAdapter<VariadicSumAggregateDefaultNull>>(
            step, argTypes, resultType);
      },
      true /*registerCompanionFunctions*/,
      true /*overwrite*/);
}

} // namespace facebook::velox::aggregate
