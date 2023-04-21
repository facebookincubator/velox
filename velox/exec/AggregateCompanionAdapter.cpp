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

#include "velox/exec/AggregateCompanionAdapter.h"

#include "velox/exec/AggregateCompanionSignatures.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/RowContainer.h"
#include "velox/expression/SignatureBinder.h"

namespace facebook::velox::exec {

void AggregateCompanionFunctionBase::setOffsetsInternal(
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t rowSizeOffset) {
  fn_->setOffsets(offset, nullByte, nullMask, rowSizeOffset);
}

int32_t AggregateCompanionFunctionBase::accumulatorFixedWidthSize() const {
  return fn_->accumulatorFixedWidthSize();
}

int32_t AggregateCompanionFunctionBase::accumulatorAlignmentSize() const {
  return fn_->accumulatorAlignmentSize();
}

int32_t AggregateCompanionFunctionBase::combineAlignmentInternal(
    int32_t otherAlignment) const {
  return fn_->combineAlignment(otherAlignment);
}

bool AggregateCompanionFunctionBase::accumulatorUsesExternalMemory() const {
  return fn_->accumulatorUsesExternalMemory();
}

bool AggregateCompanionFunctionBase::isFixedSize() const {
  return fn_->isFixedSize();
}

void AggregateCompanionFunctionBase::setAllocatorInternal(
    HashStringAllocator* allocator) {
  fn_->setAllocator(allocator);
}

void AggregateCompanionFunctionBase::destroy(folly::Range<char**> groups) {
  fn_->destroy(groups);
}

void AggregateCompanionFunctionBase::clearInternal() {
  fn_->clear();
}

void AggregateCompanionFunctionBase::initializeNewGroups(
    char** groups,
    folly::Range<const vector_size_t*> indices) {
  fn_->initializeNewGroups(groups, indices);
}

void AggregateCompanionFunctionBase::addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addRawInput(groups, rows, args, mayPushdown);
}

void AggregateCompanionFunctionBase::addSingleGroupRawInput(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addSingleGroupRawInput(group, rows, args, mayPushdown);
}

void AggregateCompanionFunctionBase::addIntermediateResults(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addIntermediateResults(groups, rows, args, mayPushdown);
}

void AggregateCompanionFunctionBase::addSingleGroupIntermediateResults(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addSingleGroupIntermediateResults(group, rows, args, mayPushdown);
}

void AggregateCompanionFunctionBase::extractAccumulators(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  fn_->extractAccumulators(groups, numGroups, result);
}

void AggregateCompanionAdapter::PartialFunction::extractValues(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  fn_->extractAccumulators(groups, numGroups, result);
}

void AggregateCompanionAdapter::MergeFunction::addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addIntermediateResults(groups, rows, args, mayPushdown);
}

void AggregateCompanionAdapter::MergeFunction::addSingleGroupRawInput(
    char* group,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool mayPushdown) {
  fn_->addSingleGroupIntermediateResults(group, rows, args, mayPushdown);
}

void AggregateCompanionAdapter::MergeFunction::extractValues(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  fn_->extractAccumulators(groups, numGroups, result);
}

void AggregateCompanionAdapter::MergeExtractFunction::extractValues(
    char** groups,
    int32_t numGroups,
    VectorPtr* result) {
  fn_->extractValues(groups, numGroups, result);
}

int32_t AggregateCompanionAdapter::ExtractFunction::setOffset() const {
  int32_t rowSizeOffset = bits::nbytes(1);
  int32_t offset = rowSizeOffset;
  offset = bits::roundUp(offset, fn_->accumulatorAlignmentSize());
  fn_->setOffsets(
      offset,
      RowContainer::nullByte(0),
      RowContainer::nullMask(0),
      rowSizeOffset);
  return offset;
}

char** AggregateCompanionAdapter::ExtractFunction::allocateGroups(
    AllocationPool& allocationPool,
    const SelectivityVector& rows,
    uint64_t offsetInGroup) const {
  auto* groups =
      (char**)allocationPool.allocateFixed(sizeof(char*) * rows.end());

  auto size = fn_->accumulatorFixedWidthSize();
  auto alignment = fn_->accumulatorAlignmentSize();
  rows.applyToSelected([&](auto row) {
    groups[row] = allocationPool.allocateFixed(size + offsetInGroup, alignment);
  });
  return groups;
}

std::tuple<vector_size_t, BufferPtr>
AggregateCompanionAdapter::ExtractFunction::compactGroups(
    memory::MemoryPool* pool,
    const SelectivityVector& rows,
    char** groups) const {
  BufferPtr indices = allocateIndices(rows.end(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t count = 0;
  rows.applyToSelected([&](auto row) {
    if (count < row) {
      groups[count] = groups[row];
    }
    rawIndices[row] = count;
    ++count;
  });
  return std::make_tuple(count, indices);
}

void AggregateCompanionAdapter::ExtractFunction::apply(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) const {
  // Set up data members of fn_.
  HashStringAllocator stringAllocator{context.pool()};
  AllocationPool allocationPool{context.pool()};
  fn_->setAllocator(&stringAllocator);

  auto offset = setOffset();
  char** groups = allocateGroups(allocationPool, rows, offset);

  // Perform per-row aggregation.
  std::vector<vector_size_t> allSelectedRange;
  rows.applyToSelected([&](auto row) { allSelectedRange.push_back(row); });
  fn_->initializeNewGroups(groups, allSelectedRange);
  fn_->addIntermediateResults(groups, rows, args, false);

  auto localResult = BaseVector::create(outputType, rows.end(), context.pool());
  const auto& [groupCount, rowsToGroupsIndices] =
      compactGroups(context.pool(), rows, groups);
  fn_->extractValues(groups, groupCount, &localResult);
  localResult = BaseVector::wrapInDictionary(
      nullptr, rowsToGroupsIndices, rows.end(), localResult);
  context.moveOrCopyResult(localResult, rows, result);
}

bool CompanionFunctionsRegistrar::registerPartialFunction(
    const std::string& name,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  auto partialSignatures =
      CompanionSignatures::partialFunctionSignatures(signatures);
  if (partialSignatures.empty()) {
    return false;
  }

  return exec::registerAggregateFunction(
      CompanionSignatures::partialFunctionName(name),
      std::move(partialSignatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
        if (auto func = getAggregateFunctionEntry(name)) {
          if (!exec::isRawInput(step)) {
            step = core::AggregationNode::Step::kIntermediate;
          }
          auto fn = func->factory(step, argTypes, resultType);
          VELOX_CHECK_NOT_NULL(fn);
          return std::make_unique<AggregateCompanionAdapter::PartialFunction>(
              std::move(fn), resultType);
        }
        VELOX_FAIL(
            "Original aggregation function {} not found: {}",
            name,
            CompanionSignatures::partialFunctionName(name));
      });
}

bool CompanionFunctionsRegistrar::registerMergeFunction(
    const std::string& name,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  auto mergeSignatures =
      CompanionSignatures::mergeFunctionSignatures(signatures);
  if (mergeSignatures.empty()) {
    return false;
  }

  return exec::registerAggregateFunction(
      CompanionSignatures::mergeFunctionName(name),
      std::move(mergeSignatures),
      [name](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
        if (auto func = getAggregateFunctionEntry(name)) {
          auto fn = func->factory(
              core::AggregationNode::Step::kIntermediate, argTypes, resultType);
          VELOX_CHECK_NOT_NULL(fn);
          return std::make_unique<AggregateCompanionAdapter::MergeFunction>(
              std::move(fn), resultType);
        }
        VELOX_FAIL(
            "Original aggregation function {} not found: {}",
            name,
            CompanionSignatures::mergeFunctionName(name));
      });
}

bool CompanionFunctionsRegistrar::registerMergeExtractFunctionWithSuffix(
    const std::string& name,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  auto groupedSignatures =
      CompanionSignatures::groupSignaturesByReturnType(signatures);
  bool registered = false;
  for (const auto& [type, signatureGroup] : groupedSignatures) {
    auto mergeExtractSignatures =
        CompanionSignatures::mergeExtractFunctionSignatures(signatureGroup);
    if (mergeExtractSignatures.empty()) {
      continue;
    }

    auto mergeExtractFunctionName =
        CompanionSignatures::mergeExtractFunctionNameWithSuffix(name, type);

    registered |= exec::registerAggregateFunction(
        mergeExtractFunctionName,
        std::move(mergeExtractSignatures),
        [name, mergeExtractFunctionName](
            core::AggregationNode::Step /*step*/,
            const std::vector<TypePtr>& argTypes,
            const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
          const auto& [originalResultType, _] =
              resolveAggregateFunction(mergeExtractFunctionName, argTypes);
          if (!originalResultType) {
            // TODO: limitation -- result type must be resolveable given
            // intermediate type of the original UDAF.
            VELOX_UNREACHABLE(
                "Signatures whose result types are not resolvable given intermediate types should have been excluded.");
          }

          if (auto func = getAggregateFunctionEntry(name)) {
            auto fn = func->factory(
                core::AggregationNode::Step::kFinal,
                argTypes,
                originalResultType);
            VELOX_CHECK_NOT_NULL(fn);
            return std::make_unique<
                AggregateCompanionAdapter::MergeExtractFunction>(
                std::move(fn), resultType);
          }
          VELOX_FAIL(
              "Original aggregation function {} not found: {}",
              name,
              mergeExtractFunctionName);
        });
  }
  return registered;
}

bool CompanionFunctionsRegistrar::registerMergeExtractFunction(
    const std::string& name,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  if (CompanionSignatures::hasSameIntermediateTypesAcrossSignatures(
          signatures)) {
    return registerMergeExtractFunctionWithSuffix(name, signatures);
  }

  auto mergeExtractSignatures =
      CompanionSignatures::mergeExtractFunctionSignatures(signatures);
  if (mergeExtractSignatures.empty()) {
    return false;
  }

  auto mergeExtractFunctionName =
      CompanionSignatures::mergeExtractFunctionName(name);
  return exec::registerAggregateFunction(
      mergeExtractFunctionName,
      std::move(mergeExtractSignatures),
      [name, mergeExtractFunctionName](
          core::AggregationNode::Step /*step*/,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
        const auto& [originalResultType, _] =
            resolveAggregateFunction(mergeExtractFunctionName, argTypes);
        if (!originalResultType) {
          // TODO: limitation -- result type must be resolveable given
          // intermediate type of the original UDAF.
          VELOX_UNREACHABLE(
              "Signatures whose result types are not resolvable given intermediate types should have been excluded.");
        }

        if (auto func = getAggregateFunctionEntry(name)) {
          auto fn = func->factory(
              core::AggregationNode::Step::kFinal,
              argTypes,
              originalResultType);
          VELOX_CHECK_NOT_NULL(fn);
          return std::make_unique<
              AggregateCompanionAdapter::MergeExtractFunction>(
              std::move(fn), resultType);
        }
        VELOX_FAIL(
            "Original aggregation function {} not found: {}",
            name,
            mergeExtractFunctionName);
      });
}

bool CompanionFunctionsRegistrar::registerExtractFunctionWithSuffix(
    const std::string& originalName,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  auto groupedSignatures =
      CompanionSignatures::groupSignaturesByReturnType(signatures);
  bool registered = false;
  for (const auto& [type, signatureGroup] : groupedSignatures) {
    auto extractSignatures =
        CompanionSignatures::extractFunctionSignatures(signatureGroup);
    if (extractSignatures.empty()) {
      continue;
    }

    auto factory = [originalName](
                       const std::string& name,
                       const std::vector<VectorFunctionArg>& inputArgs)
        -> std::shared_ptr<VectorFunction> {
      std::vector<TypePtr> argTypes{inputArgs.size()};
      std::transform(
          inputArgs.begin(),
          inputArgs.end(),
          argTypes.begin(),
          [](auto inputArg) { return inputArg.type; });

      auto resultType = resolveVectorFunction(name, argTypes);
      if (!resultType) {
        // TODO: limitation -- result type must be resolveable given
        // intermediate type of the original UDAF.
        VELOX_UNREACHABLE(
            "Signatures whose result types are not resolvable given intermediate types should have been excluded.");
      }

      if (auto func = getAggregateFunctionEntry(originalName)) {
        auto fn = func->factory(
            core::AggregationNode::Step::kFinal, argTypes, resultType);
        VELOX_CHECK_NOT_NULL(fn);
        return std::make_shared<AggregateCompanionAdapter::ExtractFunction>(
            std::move(fn));
      }
      VELOX_FAIL(
          "Original aggregation function {} not found: {}", originalName, name);
    };

    registered |= exec::registerStatefulVectorFunction(
        CompanionSignatures::extractFunctionNameWithSuffix(originalName, type),
        extractSignatures,
        factory);
  }
  return registered;
}

bool CompanionFunctionsRegistrar::registerExtractFunction(
    const std::string& originalName,
    const std::vector<AggregateFunctionSignaturePtr>& signatures) {
  if (CompanionSignatures::hasSameIntermediateTypesAcrossSignatures(
          signatures)) {
    return registerExtractFunctionWithSuffix(originalName, signatures);
  }

  auto extractSignatures =
      CompanionSignatures::extractFunctionSignatures(signatures);
  if (extractSignatures.empty()) {
    return false;
  }

  auto factory = [originalName](
                     const std::string& name,
                     const std::vector<VectorFunctionArg>& inputArgs)
      -> std::shared_ptr<VectorFunction> {
    std::vector<TypePtr> argTypes{inputArgs.size()};
    std::transform(
        inputArgs.begin(),
        inputArgs.end(),
        argTypes.begin(),
        [](auto inputArg) { return inputArg.type; });

    auto resultType = resolveVectorFunction(name, argTypes);
    if (!resultType) {
      // TODO: limitation -- result type must be resolveable given
      // intermediate type of the original UDAF.
      VELOX_UNREACHABLE(
          "Signatures whose result types are not resolvable given intermediate types should have been excluded.");
    }

    if (auto func = getAggregateFunctionEntry(originalName)) {
      auto fn = func->factory(
          core::AggregationNode::Step::kFinal, argTypes, resultType);
      VELOX_CHECK_NOT_NULL(fn);
      return std::make_shared<AggregateCompanionAdapter::ExtractFunction>(
          std::move(fn));
    }
    VELOX_FAIL(
        "Original aggregation function {} not found: {}", originalName, name);
  };
  return exec::registerStatefulVectorFunction(
      CompanionSignatures::extractFunctionName(originalName),
      extractSignatures,
      factory);
}

} // namespace facebook::velox::exec
