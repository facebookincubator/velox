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
#include "velox/vector/ConstantVector.h"

namespace facebook::velox::aggregate {

namespace {

// Sums its first argument plus its second, constant, argument for every row.
// The constant is read in setConstantInputs(), so a result that is off by the
// constant times the number of rows shows that it was not passed on.
class ConstantForwardingAggregate {
 public:
  using InputType = Row<int64_t, int64_t>;
  using IntermediateType = int64_t;
  using OutputType = int64_t;

  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    // Steps that take the intermediate result have a single input, which is
    // not a constant.
    if (constantInputs.size() == 1) {
      return;
    }
    VELOX_CHECK_EQ(constantInputs.size(), 2);
    VELOX_CHECK_NULL(constantInputs[0]);
    VELOX_CHECK_NOT_NULL(constantInputs[1]);
    auto* constant = constantInputs[1]->as<ConstantVector<int64_t>>();
    VELOX_CHECK_NOT_NULL(constant);

    offset_ = constant->valueAt(0);
  }

  struct Accumulator {
    int64_t sum{0};
    ConstantForwardingAggregate* fn;

    explicit Accumulator(
        HashStringAllocator* /*allocator*/,
        ConstantForwardingAggregate* fn)
        : fn(fn) {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<int64_t> value,
        exec::arg_type<int64_t> /*constantValue*/) {
      sum += value + fn->offset_;
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<int64_t> other) {
      sum += other;
    }

    bool writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      out = sum;
      return true;
    }

    bool writeFinalResult(exec::out_type<OutputType>& out) {
      out = sum;
      return true;
    }
  };

  using AccumulatorType = Accumulator;

  // Constant second argument, set in setConstantInputs().
  int64_t offset_{0};
};

} // namespace

exec::AggregateRegistrationResult registerSimpleConstantForwardingAggregate(
    const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("bigint")
          .intermediateType("bigint")
          .argumentType("bigint")
          .constantArgumentType("bigint")
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
        // Companion functions that take the intermediate result have one
        // argument.
        VELOX_CHECK(
            argTypes.size() == 1 || argTypes.size() == 2,
            "Function takes one or two arguments: {}",
            name);
        return std::make_unique<
            exec::SimpleAggregateAdapter<ConstantForwardingAggregate>>(
            step, argTypes, resultType);
      },
      /*registerCompanionFunctions=*/true,
      /*overwrite=*/true);
}

} // namespace facebook::velox::aggregate
