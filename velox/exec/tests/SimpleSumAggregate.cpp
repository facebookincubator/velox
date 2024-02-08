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
#include "velox/exec/SimpleAggregateAdapterExperiment.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/VectorWriters.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate {

namespace {

// Returns negative sum of input values.
class SumAggregate {
 public:
  // Type(s) of input vector(s) wrapped in Row.
  using InputType = Row<int64_t>;

  // Type of intermediate result vector wrapped in Row.
  using IntermediateType = int64_t;

  // Type of output vector.
  using OutputType = int64_t;

  struct FunctionState {
    bool flag_;
  };

  static void initialize(
      FunctionState& state,
      const TypePtr& type,
      const std::vector<VectorPtr>& constantInputs) {
    state.flag_ = false;
  }

  struct AccumulatorType {
    int64_t sum_;

    AccumulatorType() = delete;

    // Constructor used in initializeNewGroups().
    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        const FunctionState& /*state*/) {
      sum_ = 0;
    }

    void addInput(
        HashStringAllocator* /*allocator*/,
        int64_t data,
        const FunctionState& /*state*/) {
      sum_ += data;
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        int64_t other,
        const FunctionState& /*state*/) {
      sum_ += other;
    }

    bool writeFinalResult(int64_t& out, const FunctionState& state) {
      out = state.flag_ ? sum_ : -sum_;
      return true;
    }

    bool writeIntermediateResult(int64_t& out, const FunctionState& /*state*/) {
      out = sum_;
      return true;
    }
  };
};

} // namespace

exec::AggregateRegistrationResult registerSimpleSumAggregate(
    const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .returnType("bigint")
                           .intermediateType("bigint")
                           .argumentType("bigint")
                           .build());

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<SimpleAggregateAdapterExperiment<SumAggregate>>(
            resultType);
      },
      true);
}

} // namespace facebook::velox::aggregate
