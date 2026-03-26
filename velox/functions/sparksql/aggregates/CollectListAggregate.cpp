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

#include "velox/functions/sparksql/aggregates/CollectListAggregate.h"

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/vector/ConstantVector.h"

using namespace facebook::velox::aggregate;
using namespace facebook::velox::exec;

namespace facebook::velox::functions::aggregate::sparksql {
namespace {
class CollectListAggregate {
 public:
  using InputType = Row<Generic<T1>>;

  using IntermediateType = Array<Generic<T1>>;

  using OutputType = Array<Generic<T1>>;

  /// In Spark, when all inputs are null, the output is an empty array instead
  /// of null. Therefore, in the writeIntermediateResult and writeFinalResult,
  /// we still need to output the empty element_ when the group is null. This
  /// behavior can only be achieved when the default-null behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  // Whether null input values should be ignored. Defaults to true.
  // NOTE: toIntermediate() was intentionally removed because it is static and
  // cannot access the runtime ignoreNulls_ config. Without it, partial
  // aggregation uses the accumulator path, which correctly respects the config.
  bool ignoreNulls_{true};

  struct AccumulatorType {
    ValueList elements_;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        CollectListAggregate* fn)
        : elements_{}, ignoreNulls_(fn->ignoreNulls_) {}

    static constexpr bool is_fixed_size_ = false;

    bool addInput(
        HashStringAllocator* allocator,
        exec::optional_arg_type<Generic<T1>> data) {
      if (data.has_value()) {
        elements_.appendValue(data, allocator);
        return true;
      }
      if (!ignoreNulls_) {
        elements_.appendValue(data, allocator);
        return true;
      }
      return false;
    }

    bool ignoreNulls_;

    bool combine(
        HashStringAllocator* allocator,
        exec::optional_arg_type<IntermediateType> other) {
      if (!other.has_value()) {
        return false;
      }
      for (auto element : other.value()) {
        elements_.appendValue(element, allocator);
      }
      return true;
    }

    bool writeIntermediateResult(
        bool /*nonNullGroup*/,
        exec::out_type<IntermediateType>& out) {
      // If the group's accumulator is null, the corresponding intermediate
      // result is an empty array.
      copyValueListToArrayWriter(out, elements_);
      return true;
    }

    bool writeFinalResult(
        bool /*nonNullGroup*/,
        exec::out_type<OutputType>& out) {
      // If the group's accumulator is null, the corresponding result is an
      // empty array.
      copyValueListToArrayWriter(out, elements_);
      return true;
    }

    void destroy(HashStringAllocator* allocator) {
      elements_.free(allocator);
    }
  };
};

// Adapter that overrides setConstantInputs to read the ignoreNulls flag.
class CollectListAdapter : public SimpleAggregateAdapter<CollectListAggregate> {
 public:
  using SimpleAggregateAdapter<CollectListAggregate>::SimpleAggregateAdapter;

  void setConstantInputs(
      const std::vector<VectorPtr>& constantInputs) override {
    if (constantInputs.size() >= 2 && constantInputs[1] != nullptr &&
        !constantInputs[1]->isNullAt(0)) {
      fn_->ignoreNulls_ =
          constantInputs[1]->as<ConstantVector<bool>>()->valueAt(0);
    }
  }
};

AggregateRegistrationResult registerCollectList(
    const std::string& name,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      // collect_list(E) -> array(E): default ignoreNulls=true.
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("array(E)")
          .argumentType("E")
          .build(),
      // collect_list(E, ignoreNulls) -> array(E): explicit flag.
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("E")
          .returnType("array(E)")
          .intermediateType("array(E)")
          .argumentType("E")
          .constantArgumentType("boolean")
          .build()};
  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<CollectListAdapter>(
            step, argTypes, resultType, &config);
      },
      withCompanionFunctions,
      overwrite);
}
} // namespace

void registerCollectListAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite) {
  registerCollectList(
      prefix + "collect_list", withCompanionFunctions, overwrite);
}
} // namespace facebook::velox::functions::aggregate::sparksql
