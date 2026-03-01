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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateUtil.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

using namespace facebook::velox::exec;

namespace facebook::velox::aggregate {

namespace {

// A baseline implementation of VariadicSumAggregate using the Aggregate base
// class directly (without SimpleAggregateAdapter). This serves as a performance
// baseline for comparison with SimpleAggregateAdapter-based implementation.
//
// This function takes a dummy integer argument followed by a variadic list of
// integers. It returns an array where the i-th element is the sum of all i-th
// variadic arguments across all rows.
//
// Example:
//   SELECT variadic_sum_agg(3, a, b, c) FROM (
//     VALUES (1, 2, 3), (4, 5, 6)
//   ) AS t(a, b, c)
//   => [5, 7, 9]  (i.e., [1+4, 2+5, 3+6])
class VariadicSumAggregateDefaultNull : public Aggregate {
 public:
  explicit VariadicSumAggregateDefaultNull(TypePtr resultType)
      : Aggregate(std::move(resultType)) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(Accumulator);
  }

  bool accumulatorUsesExternalMemory() const override {
    return true;
  }

  bool isFixedSize() const override {
    return false;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto arrayVector = (*result)->as<ArrayVector>();
    arrayVector->resize(numGroups);

    auto* rawOffsets =
        arrayVector->mutableOffsets(numGroups)->asMutable<vector_size_t>();
    auto* rawSizes =
        arrayVector->mutableSizes(numGroups)->asMutable<vector_size_t>();

    vector_size_t totalElements = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto* accumulator = value<Accumulator>(groups[i]);
      totalElements += accumulator->sums.size();
    }

    auto elementsVector = arrayVector->elements()->asFlatVector<int64_t>();
    elementsVector->resize(totalElements);
    auto* rawElements = elementsVector->mutableRawValues();

    vector_size_t offset = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto* accumulator = value<Accumulator>(groups[i]);
      rawOffsets[i] = offset;
      rawSizes[i] = accumulator->sums.size();
      for (size_t j = 0; j < accumulator->sums.size(); ++j) {
        rawElements[offset + j] = accumulator->sums[j];
      }
      offset += accumulator->sums.size();
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    // Decode ALL arguments including dummy arg[0] to match
    // SimpleAggregateAdapter which decodes all inputs.
    const auto numArgs = args.size();
    std::vector<DecodedVector> decodedArgs(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      decodedArgs[i].decode(*args[i], rows);
    }

    const auto numVariadicArgs = numArgs - 1;

    rows.applyToSelected([&](vector_size_t row) {
      // Ignore rows with any nulls.
      for (size_t i = 0; i < numVariadicArgs; ++i) {
        if (decodedArgs[i + 1].isNullAt(row)) {
          return;
        }
      }

      // RowSizeTracker to match SimpleAggregateAdapter overhead for
      // non-fixed-size accumulators.
      RowSizeTracker<char, uint32_t> tracker(
          groups[row][rowSizeOffset_], *allocator_);

      auto* group = groups[row];
      auto* accumulator = value<Accumulator>(group);

      if (accumulator->sums.empty()) {
        accumulator->sums.resize(numVariadicArgs, 0);
      }

      for (size_t i = 0; i < numVariadicArgs; ++i) {
        accumulator->sums[i] += decodedArgs[i + 1].valueAt<int64_t>(row);
      }

      // clearNull to match SimpleAggregateAdapter which clears null on every
      // non-null row.
      clearNull(group);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    const auto numArgs = args.size();
    const auto numVariadicArgs = numArgs - 1;
    if (numVariadicArgs == 0) {
      return;
    }

    auto* accumulator = value<Accumulator>(group);
    if (accumulator->sums.empty()) {
      accumulator->sums.resize(numVariadicArgs, 0);
    }

    // Decode ALL arguments including dummy arg[0].
    std::vector<DecodedVector> decodedArgs(numArgs);
    for (size_t i = 0; i < numArgs; ++i) {
      decodedArgs[i].decode(*args[i], rows);
    }

    rows.applyToSelected([&](vector_size_t row) {
      // Ignore rows with any nulls.
      for (size_t i = 0; i < numVariadicArgs; ++i) {
        if (decodedArgs[i + 1].isNullAt(row)) {
          return;
        }
      }

      // RowSizeTracker to match SimpleAggregateAdapter.
      RowSizeTracker<char, uint32_t> tracker(
          group[rowSizeOffset_], *allocator_);

      for (size_t i = 0; i < numVariadicArgs; ++i) {
        accumulator->sums[i] += decodedArgs[i + 1].valueAt<int64_t>(row);
      }

      clearNull(group);
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto arrayVector = args[0]->as<ArrayVector>();
    auto elementsVector = arrayVector->elements()->asFlatVector<int64_t>();

    rows.applyToSelected([&](vector_size_t row) {
      if (arrayVector->isNullAt(row)) {
        return;
      }

      auto* group = groups[row];
      auto* accumulator = value<Accumulator>(group);

      auto offset = arrayVector->offsetAt(row);
      auto size = arrayVector->sizeAt(row);

      if (accumulator->sums.empty()) {
        accumulator->sums.resize(size, 0);
      }

      for (vector_size_t i = 0; i < size; ++i) {
        if (!elementsVector->isNullAt(offset + i)) {
          accumulator->sums[i] += elementsVector->valueAt(offset + i);
        }
      }
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    auto arrayVector = args[0]->as<ArrayVector>();
    auto elementsVector = arrayVector->elements()->asFlatVector<int64_t>();
    auto* accumulator = value<Accumulator>(group);

    rows.applyToSelected([&](vector_size_t row) {
      if (arrayVector->isNullAt(row)) {
        return;
      }

      auto offset = arrayVector->offsetAt(row);
      auto size = arrayVector->sizeAt(row);

      if (accumulator->sums.empty()) {
        accumulator->sums.resize(size, 0);
      }

      for (vector_size_t i = 0; i < size; ++i) {
        if (!elementsVector->isNullAt(offset + i)) {
          accumulator->sums[i] += elementsVector->valueAt(offset + i);
        }
      }
    });
  }

 protected:
  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    // setAllNulls to match SimpleAggregateAdapter which marks new groups as
    // null (default_null_behavior_: groups start null, cleared on first input).
    setAllNulls(groups, indices);
    for (auto index : indices) {
      new (groups[index] + offset_) Accumulator();
    }
  }

  void destroyInternal(folly::Range<char**> groups) override {
    for (auto* group : groups) {
      if (isInitialized(group)) {
        value<Accumulator>(group)->~Accumulator();
      }
    }
  }

 private:
  struct Accumulator {
    std::vector<int64_t> sums;
  };
};

} // namespace

exec::AggregateRegistrationResult registerVariadicSumAggregateDefaultNull(
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
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& /*argTypes*/,
          const TypePtr& resultType,
          const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<VariadicSumAggregateDefaultNull>(resultType);
      },
      true /*registerCompanionFunctions*/,
      true /*overwrite*/);
}

} // namespace facebook::velox::aggregate
