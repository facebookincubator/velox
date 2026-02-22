/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec {

class AggregateCompanionFunctionBase : public Aggregate {
 public:
  explicit AggregateCompanionFunctionBase(
      std::unique_ptr<Aggregate>&& fn,
      const TypePtr& resultType)
      : Aggregate{resultType}, fn_{std::move(fn)} {}

  int32_t accumulatorFixedWidthSize() const override final;

  int32_t accumulatorAlignmentSize() const override final;

  bool accumulatorUsesExternalMemory() const override final;

  bool isFixedSize() const override final;

  void destroy(folly::Range<char**> groups) override final;

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override final;

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override;

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override final;

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override final;

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override final;

 protected:
  void setOffsetsInternal(
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      int32_t initializedByte,
      uint8_t initializedMask,
      int32_t rowSizeOffset) override final;

  void setAllocatorInternal(HashStringAllocator* allocator) override final;

  void clearInternal() override final;

  void initializeNewGroupsInternal(
      char** groups,
      folly::Range<const vector_size_t*> indices) override final;

  void destroyInternal(folly::Range<char**> groups) override final;

  std::unique_ptr<Aggregate> fn_;
};

struct AggregateCompanionAdapter {
  class PartialFunction : public AggregateCompanionFunctionBase {
   public:
    explicit PartialFunction(
        std::unique_ptr<Aggregate> fn,
        const TypePtr& resultType)
        : AggregateCompanionFunctionBase{std::move(fn), resultType} {}

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
        override;
  };

  class MergeFunction : public AggregateCompanionFunctionBase {
   public:
    explicit MergeFunction(
        std::unique_ptr<Aggregate> fn,
        const TypePtr& resultType)
        : AggregateCompanionFunctionBase{std::move(fn), resultType} {}

    void addRawInput(
        char** groups,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override;

    void addSingleGroupRawInput(
        char* group,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override;

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
        override;
  };

  class MergeExtractFunction : public MergeFunction {
   public:
    explicit MergeExtractFunction(
        std::unique_ptr<Aggregate> fn,
        const TypePtr& resultType)
        : MergeFunction{std::move(fn), resultType} {}

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
        override;
  };

  class ExtractFunction : public VectorFunction {
   public:
    explicit ExtractFunction(std::unique_ptr<Aggregate> fn)
        : fn_{std::move(fn)} {}

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& outputType,
        exec::EvalCtx& context,
        VectorPtr& result) const override;

   private:
    int32_t setOffset() const;

    char** allocateGroups(
        memory::AllocationPool& allocationPool,
        const SelectivityVector& rows,
        uint64_t offsetInGroup) const;

    // Compact `groups` into a contiguous array of groups for selected rows.
    // Return the number of groups after compaction and a mapping from original
    // indices in `groups` to new indices after compaction.
    std::tuple<vector_size_t, BufferPtr> compactGroups(
        memory::MemoryPool* pool,
        const SelectivityVector& rows,
        char** groups) const;

    std::unique_ptr<Aggregate> fn_;
  };
};

/// In Velox, "Step" is a property of the aggregate operator, whereas in
/// Spark, it is tied to individual aggregate functions. Spark executes
/// aggregates using a mix of partial, intermediate, and final aggregate
/// functions. To bridge the two systems, the planner translates Spark's
/// aggregate modes into corresponding Velox companion functions and assigns
/// the "single" step to Velox’s AggregationNode. These companion functions
/// are intended for internal use within the aggregate operator and are not
/// designed to be used as standalone functions, and their result types
/// may not always be inferable from intermediate types. More details can be
/// found in
/// https://github.com/facebookincubator/velox/pull/11999#issuecomment-3274577979
/// and https://github.com/facebookincubator/velox/issues/12830.
class CompanionFunctionsRegistrar {
 public:
  /// Register the partial companion function for an aggregate function of
  /// `name` and `signatures`. When there is already a function of the same
  /// name, if `overwrite` is true, the registration is replaced. Otherwise,
  /// return false without overwriting the registry. This function supports
  /// generating Spark compatible companion functions.
  static bool registerPartialFunction(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      const AggregateFunctionMetadata& metadata,
      bool overwrite = false);

  /// When there is already a function of the same name as the merge companion
  /// function, if `overwrite` is true, the registration is replaced. Otherwise,
  /// return false without overwriting the registry. This function supports
  /// generating Spark compatible companion functions.
  static bool registerMergeFunction(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      const AggregateFunctionMetadata& metadata,
      bool overwrite = false);

  // If there are multiple signatures of the original aggregation function
  // with the same intermediate type, register extract functions with suffix
  // of their result types in the function names for each of them. Otherwise,
  // register one extract function of all supported signatures. The result
  // type of the original aggregation function is required to be resolvable
  // given its intermediate type. When there is already a function of the same
  // name as the extract companion function, if `overwrite` is true, the
  // registration is replaced. Otherwise, return false without overwriting the
  // registry.
  static bool registerExtractFunction(
      const std::string& originalName,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      bool overwrite = false);

  /// If there are multiple signatures of the original aggregate function
  /// with the same intermediate type, register merge-extract functions with
  // suffix of their result types in the function names for each of them. When
  /// there is already a function of the same name as the merge-extract
  /// companion function, if `overwrite` is true, the registration is replaced.
  /// Otherwise, return false without overwriting the registry. This function
  /// supports generating Spark compatible companion functions only when the
  /// return types are explicitly specified (typically in "single" or "final"
  /// steps). It will throw an exception if return types are not provided and
  /// cannot be resolved from the intermediate types.
  static bool registerMergeExtractFunction(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      const AggregateFunctionMetadata& metadata,
      bool overwrite = false);

 private:
  // Register a vector function {originalName}_extract_{suffixOfResultType}
  // that takes input of the intermediate type and returns the result type of
  // the original aggregate function.
  static bool registerExtractFunctionWithSuffix(
      const std::string& originalName,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      bool overwrite);

  static bool registerMergeExtractFunctionWithSuffix(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& signatures,
      const AggregateFunctionMetadata& metadata,
      bool overwrite);
};

} // namespace facebook::velox::exec
