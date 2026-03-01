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
#include "velox/functions/lib/aggregates/ValueList.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate {

namespace {

// An aggregate function that demonstrates variadic argument support with
// Generic types and non-default null behavior in SimpleAggregateAdapter. This
// function takes a variadic list of values of the same type and aggregates all
// of them into a single array, including nulls.
//
// Example:
//   SELECT variadic_array_agg(a, b, c) FROM (
//     VALUES (1, 2, 3), (4, 5, 6)
//   ) AS t(a, b, c)
//   => [1, 2, 3, 4, 5, 6]
class VariadicArrayAggAggregate {
 public:
  using InputType = AggregateInputType<Variadic<Generic<T1>>>;

  using IntermediateType = Array<Generic<T1>>;

  using OutputType = Array<Generic<T1>>;

  static constexpr bool default_null_behavior_ = false;

  static bool toIntermediate(
      exec::out_type<Array<Generic<T1>>>& out,
      exec::optional_arg_type<Variadic<Generic<T1>>> variadicArgs) {
    if (!variadicArgs.has_value()) {
      VELOX_UNREACHABLE(
          "simple_variadic_array_agg requires at least one variadic argument.");
    }
    for (auto i = 0; i < variadicArgs.value().size(); ++i) {
      if (variadicArgs->at(i).has_value()) {
        out.add_item().copy_from(variadicArgs->at(i).value());
      } else {
        out.add_null();
      }
    }
    return true;
  }

  struct AccumulatorType {
    ValueList elements_;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        VariadicArrayAggAggregate* /*fn*/)
        : elements_{} {}

    static constexpr bool is_fixed_size_ = false;

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Variadic<Generic<T1>>> variadicArgs) {
      if (!variadicArgs.has_value()) {
        VELOX_UNREACHABLE(
            "simple_variadic_array_agg requires at least one variadic argument.");
      }
      for (auto i = 0; i < variadicArgs.value().size(); ++i) {
        elements_.appendValue(variadicArgs->at(i), allocator);
      }
      return true;
    }

    bool combine(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Array<Generic<T1>>> other) {
      if (!other.has_value()) {
        return false;
      }
      for (auto i = 0; i < other.value().size(); ++i) {
        elements_.appendValue(other->at(i), allocator);
      }
      return true;
    }

    bool writeFinalResult(
        bool nonNullGroup,
        exec::out_type<Array<Generic<T1>>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      copyValueListToArrayWriter(out, elements_);
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<Array<Generic<T1>>>& out) {
      if (!nonNullGroup) {
        return false;
      }
      copyValueListToArrayWriter(out, elements_);
      return true;
    }

    void destroy(HashStringAllocator* allocator) {
      elements_.free(allocator);
    }
  };
};

} // namespace

exec::AggregateRegistrationResult registerSimpleVariadicArrayAggAggregate(
    const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("array(E)")
          .argumentType("E")
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
            SimpleAggregateAdapter<VariadicArrayAggAggregate>>(
            step, argTypes, resultType);
      },
      true /*registerCompanionFunctions*/,
      true /*overwrite*/);
}

} // namespace facebook::velox::aggregate
