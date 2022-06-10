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
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/MaxSizeForStatsEstimator.h"
#include "velox/functions/prestosql/aggregates/SimpleNumericAggregate.h"
#include "velox/functions/prestosql/aggregates/SingleValueAccumulator.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::aggregate {

namespace {

// The return type of $internal$max_data_size_for_stats(col) expression.
using TMaxDataSize = int64_t;

class MaxSizeForStatsAggregate
    : public SimpleNumericAggregate<TMaxDataSize, TMaxDataSize, TMaxDataSize> {
  using BaseAggregate =
      SimpleNumericAggregate<TMaxDataSize, TMaxDataSize, TMaxDataSize>;

 public:
  explicit MaxSizeForStatsAggregate(TypePtr resultType)
      : BaseAggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(TMaxDataSize);
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    BaseAggregate::doExtractValues(groups, numGroups, result, [&](char* group) {
      return *BaseAggregate::Aggregate::template value<TMaxDataSize>(group);
    });
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    exec::Aggregate::setAllNulls(groups, indices);
    for (auto i : indices) {
      *BaseAggregate ::value<TMaxDataSize>(groups[i]) = 0;
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    // partial and final aggregations are the same
    extractValues(groups, numGroups, result);
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::template updateGroups<true, TMaxDataSize>(
        groups,
        rows,
        args[0],
        [](TMaxDataSize& result, TMaxDataSize value) {
          if (result < value) {
            result = value;
          }
        },
        mayPushdown);
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    BaseAggregate::updateOneGroup(
        group,
        rows,
        args[0],
        [](TMaxDataSize& result, TMaxDataSize value) {
          result = result > value ? result : value;
        },
        [](TMaxDataSize& result, TMaxDataSize value, int /* unused */) {
          result = value;
        },
        mayPushdown,
        (TMaxDataSize)0);
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdateSingleGroup(group, rows, args[0]);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    doUpdate(groups, rows, args[0]);
  }

 protected:
  void upsertOneAccumulator(
      char* const group,
      const BaseVector& inputVector,
      vector_size_t idx) {
    if (inputVector.isNullAt(idx)) {
      return;
    }
    // Recursively calculate size
    size_t size_out = 0;
    exec::MaxSizeForStatsEstimator::instance().estimateSizeOfVectorElements(
        inputVector, idx, 1, size_out);
    // Clear null
    clearNull(group);
    // Set max(current, this)
    TMaxDataSize& cur = *value<TMaxDataSize>(group);
    cur = std::max(cur, (TMaxDataSize)size_out);
  }

  void
  doUpdate(char** groups, const SelectivityVector& rows, const VectorPtr& arg) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    // TODO: Add test to verify that constant mapping encoding works w.r.t idx.
    if (decoded.isConstantMapping() && decoded.isNullAt(0)) {
      // nothing to do; all values are nulls
      return;
    }

    rows.applyToSelected([&](vector_size_t i) {
      upsertOneAccumulator(groups[i], *baseVector, indices[i]);
    });
  }

  void doUpdateSingleGroup(
      char* group,
      const SelectivityVector& rows,
      const VectorPtr& arg) {
    DecodedVector decoded(*arg, rows, true);
    auto indices = decoded.indices();
    auto baseVector = decoded.base();

    if (decoded.isConstantMapping()) {
      if (decoded.isNullAt(0)) {
        // nothing to do; all values are nulls
        return;
      }
      upsertOneAccumulator(group, *baseVector, 0);
      return;
    }

    rows.applyToSelected(
        [&](vector_size_t i) { upsertOneAccumulator(group, *baseVector, i); });
  }
};

bool registerMaxSizeForStatsAggregate(const std::string& name) {
  // Types here are used by PlanBuilder to populate partial and final
  // aggregation input and output types. E.g. intermediate results will be of
  // type vector<intermediateType> and used to serve as the input vector type to
  // final aggregation. argumentType is type of vector to AddRawInput
  // intermediateType is the type of vector that extractAccumulator write to and
  // that addIntermediateResults reads out of
  // returnType is the type of vector that extractValues write to.
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  for (const auto& inputType :
       {"tinyint", "smallint", "integer", "bigint", "double", "real"}) {
    // See implementation overview for the rationale of choosing these types.
    signatures.push_back(
        exec::AggregateFunctionSignatureBuilder()
            .returnType(mapTypeKindToName(TypeKind::BIGINT))
            .intermediateType(mapTypeKindToName(TypeKind::BIGINT))
            .argumentType(inputType)
            .build());
  }

  // For non-fixed width input types.
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .typeVariable("TInput")
          .returnType(mapTypeKindToName(TypeKind::BIGINT))
          .intermediateType(mapTypeKindToName(TypeKind::BIGINT))
          .argumentType("TInput")
          .build());

  // clang-format off
 // Aggr plan node call table for group aggregation:
 //+--------------------------+-------------------+-------------------+-------------------+-------------------+
 //|            \             |      partial      |       final       |   intermediate    |      single       |
 //+--------------------------+-------------------+-------------------+-------------------+-------------------+
 //| AddRaw                   | x                 |                   |                   | x                 |
 //| AddRawForSingle          | x(if global aggr) |                   |                   | x(if global aggr) |
 //| AddIntermediate          |                   | x                 | x                 |                   |
 //| AddIntermediateForSingle |                   | x(if global aggr) | x(if global aggr) |                   |
 //| ExtractAcct              | x                 |                   | x                 |                   |
 //| ExtractVal               |                   | x                 |                   | x                 |
 //+--------------------------+-------------------+-------------------+-------------------+-------------------+
  // clang-format on
  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        auto inputType = argTypes[0];

        return std::make_unique<MaxSizeForStatsAggregate>(resultType);
      });
}

static bool FB_ANONYMOUS_VARIABLE(g_AggregateFunction) =
    registerMaxSizeForStatsAggregate(kMaxSizeForStats);

} // namespace
} // namespace facebook::velox::aggregate
