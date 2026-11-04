/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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
#include "velox/functions/prestosql/aggregates/VectorCountAggregate.h"

#include "velox/common/base/CheckedArithmetic.h"
#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::aggregate::prestosql {

namespace {

class VectorCountAggregate {
 public:
  using InputType = Row<VectorIndex>;

  using IntermediateType = int64_t;

  using OutputType = int64_t;

  struct AccumulatorType {
    int64_t numEmbeddings{0};

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        VectorCountAggregate* /*fn*/) {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<VectorIndex> index) {
      numEmbeddings = checkedPlus<int64_t>(
          numEmbeddings,
          common::vectorindex::LocalVectorIndex::size(
              std::string_view(index.data(), index.size())));
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<int64_t> other) {
      numEmbeddings = checkedPlus<int64_t>(numEmbeddings, other);
    }

    void writeFinalResult(exec::out_type<OutputType>& out) {
      out = numEmbeddings;
    }

    void writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      out = numEmbeddings;
    }
  };
};

} // namespace

void registerVectorCountAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("bigint")
          .intermediateType("bigint")
          .argumentType("vector_index")
          .build(),
  };

  exec::registerAggregateFunction(
      names,
      signatures,
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<
            exec::SimpleAggregateAdapter<VectorCountAggregate>>(
            step, argTypes, resultType);
      },
      {.orderSensitive = false, .ignoreNullInputs = true},
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
