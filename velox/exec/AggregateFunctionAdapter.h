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
#pragma once

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/expression/SignatureBinder.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec {

using AggregateFunctionSignaturePtr =
    std::shared_ptr<AggregateFunctionSignature>;

struct AggregateFunctionAdapter {
  class PartialFunction : public Aggregate {
   public:
    explicit PartialFunction(
        std::unique_ptr<Aggregate> fn,
        const TypePtr& resultType)
        : Aggregate{resultType}, fn_{std::move(fn)} {}

    void setOffsets(
        int32_t offset,
        int32_t nullByte,
        uint8_t nullMask,
        int32_t rowSizeOffset) override {
      fn_->setOffsets(offset, nullByte, nullMask, rowSizeOffset);
    }

    int32_t accumulatorFixedWidthSize() const override {
      return fn_->accumulatorFixedWidthSize();
    }

    void initializeNewGroups(
        char** groups,
        folly::Range<const vector_size_t*> indices) override {
      fn_->initializeNewGroups(groups, indices);
    }

    void addRawInput(
        char** groups,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addRawInput(groups, rows, args, mayPushdown);
    }

    void addSingleGroupRawInput(
        char* group,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addSingleGroupRawInput(group, rows, args, mayPushdown);
    }

    void addIntermediateResults(
        char** groups,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addIntermediateResults(groups, rows, args, mayPushdown);
    }

    void addSingleGroupIntermediateResults(
        char* group,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addSingleGroupIntermediateResults(group, rows, args, mayPushdown);
    }

    void extractAccumulators(
        char** groups,
        int32_t numGroups,
        VectorPtr* result) override {
      fn_->extractAccumulators(groups, numGroups, result);
    }

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
        override {
      fn_->extractAccumulators(groups, numGroups, result);
    }

   private:
    std::unique_ptr<Aggregate> fn_;
  };

  class MergeFunction : public Aggregate {
   public:
    explicit MergeFunction(
        std::unique_ptr<Aggregate> fn,
        const TypePtr& resultType)
        : Aggregate{resultType}, fn_{std::move(fn)} {}

    void setOffsets(
        int32_t offset,
        int32_t nullByte,
        uint8_t nullMask,
        int32_t rowSizeOffset) override {
      fn_->setOffsets(offset, nullByte, nullMask, rowSizeOffset);
    }

    int32_t accumulatorFixedWidthSize() const override {
      return fn_->accumulatorFixedWidthSize();
    }

    int32_t accumulatorAlignmentSize() const override {
      return fn_->accumulatorAlignmentSize();
    }

    void initializeNewGroups(
        char** groups,
        folly::Range<const vector_size_t*> indices) override {
      fn_->initializeNewGroups(groups, indices);
    }

    void addRawInput(
        char** groups,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addIntermediateResults(groups, rows, args, mayPushdown);
    }

    void addSingleGroupRawInput(
        char* group,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addSingleGroupIntermediateResults(group, rows, args, mayPushdown);
    }

    void addIntermediateResults(
        char** groups,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addIntermediateResults(groups, rows, args, mayPushdown);
    }

    void addSingleGroupIntermediateResults(
        char* group,
        const SelectivityVector& rows,
        const std::vector<VectorPtr>& args,
        bool mayPushdown) override {
      fn_->addSingleGroupIntermediateResults(group, rows, args, mayPushdown);
    }

    void extractAccumulators(
        char** groups,
        int32_t numGroups,
        VectorPtr* result) override {
      fn_->extractAccumulators(groups, numGroups, result);
    }

    void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
        override {
      fn_->extractAccumulators(groups, numGroups, result);
    }

   private:
    std::unique_ptr<Aggregate> fn_;
  };

  class RetractFunction : public VectorFunction {
   public:
    explicit RetractFunction(std::unique_ptr<Aggregate> fn)
        : fn_{std::move(fn)} {}

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& outputType,
        exec::EvalCtx& context,
        VectorPtr& result) const override {
      // Set up data members of fn_.
      HashStringAllocator stringAllocator{context.pool()};
      fn_->setAllocator(&stringAllocator);

      // Null byte.
      int32_t rowSizeOffset = bits::nbytes(1);
      int32_t offset = rowSizeOffset;
      offset = bits::roundUp(offset, fn_->accumulatorAlignmentSize());
      fn_->setOffsets(
          offset,
          RowContainer::nullByte(0),
          RowContainer::nullMask(0),
          rowSizeOffset);

      // Allocate groups.
      auto accumulatorsHeader =
          stringAllocator.allocate(sizeof(char*) * rows.size());
      auto accumulators = (char**)accumulatorsHeader->begin();
      std::vector<HashStringAllocator::Header*> headers;
      auto size = fn_->accumulatorFixedWidthSize();
      for (auto i = 0; i < rows.size(); ++i) {
        headers.push_back(stringAllocator.allocate(size + offset));
        accumulators[i] = headers.back()->begin();
      }

      // Perform per-row aggregation.
      VELOX_CHECK_EQ(args.size(), 2, "Expect two arguments");
      std::vector<vector_size_t> range;
      rows.applyToSelected([&](auto row) { range.push_back(row); });

      fn_->initializeNewGroups(accumulators, range);
      fn_->addIntermediateResults(accumulators, rows, {args[0]}, false);
      fn_->retractIntermediateResults(accumulators, rows, {args[1]}, false);
      if (!result) {
        result = BaseVector::create(outputType, rows.end(), context.pool());
      }
      fn_->extractAccumulators(accumulators, rows.size(), &result);

      // Free allocated space.
      for (auto i = 0; i < rows.size(); ++i) {
        stringAllocator.free(headers[i]);
      }
      stringAllocator.free(accumulatorsHeader);
    }

   private:
    std::unique_ptr<Aggregate> fn_;
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
        VectorPtr& result) const override {
      // Set up data members of fn_.
      HashStringAllocator stringAllocator{context.pool()};
      fn_->setAllocator(&stringAllocator);

      // Null byte.
      int32_t rowSizeOffset = bits::nbytes(1);
      int32_t offset = rowSizeOffset;
      offset = bits::roundUp(offset, fn_->accumulatorAlignmentSize());
      fn_->setOffsets(
          offset,
          RowContainer::nullByte(0),
          RowContainer::nullMask(0),
          rowSizeOffset);

      // Allocate groups.
      auto accumulatorsHeader =
          stringAllocator.allocate(sizeof(char*) * rows.size());
      auto accumulators = (char**)accumulatorsHeader->begin();
      std::vector<HashStringAllocator::Header*> headers;
      auto size = fn_->accumulatorFixedWidthSize();
      for (auto i = 0; i < rows.size(); ++i) {
        headers.push_back(stringAllocator.allocate(size + offset));
        accumulators[i] = headers.back()->begin();
      }

      // Perform per-row aggregation.
      std::vector<vector_size_t> range;
      rows.applyToSelected([&](auto row) { range.push_back(row); });

      fn_->initializeNewGroups(accumulators, range);
      fn_->addIntermediateResults(accumulators, rows, args, false);
      if (!result) {
        result = BaseVector::create(outputType, rows.end(), context.pool());
      }
      fn_->extractValues(accumulators, rows.size(), &result);

      // Free allocated space.
      for (auto i = 0; i < rows.size(); ++i) {
        stringAllocator.free(headers[i]);
      }
      stringAllocator.free(accumulatorsHeader);
    }

   private:
    std::unique_ptr<Aggregate> fn_;
  };
};

void addVariablesInTypeToList(
    const TypeSignature& type,
    const std::unordered_map<std::string, SignatureVariable>& allVariables,
    std::unordered_map<std::string, SignatureVariable>& usedVariables);

std::unordered_map<std::string, SignatureVariable> getUsedTypeVariables(
    const std::vector<TypeSignature>& types,
    const std::unordered_map<std::string, SignatureVariable>& allVariables);

class RegisterAdapter {
 public:
  static std::vector<AggregateFunctionSignaturePtr> partialFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& aggregateSignatures) {
    std::vector<AggregateFunctionSignaturePtr> signatures;
    for (const auto& signature : aggregateSignatures) {
      std::vector<TypeSignature> usedTypes = signature->argumentTypes();
      usedTypes.push_back(signature->intermediateType());
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());

      signatures.push_back(std::make_shared<AggregateFunctionSignature>(
          variables,
          signature->intermediateType(),
          signature->intermediateType(),
          signature->argumentTypes(),
          signature->constantArguments(),
          signature->variableArity()));
    }
    return signatures;
  }

  static bool registerPartialFunction(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& originalSignatures) {
    auto signatures = partialFunctionSignatures(originalSignatures);
    exec::registerAggregateFunction(
        name + "_partial",
        std::move(signatures),
        [name](
            core::AggregationNode::Step step,
            const std::vector<TypePtr>& argTypes,
            const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
          if (auto func = getAggregateFunctionEntry(name)) {
            if (exec::isRawInput(step)) {
              auto fn = func.value()->factory(step, argTypes, resultType);
              return std::make_unique<
                  AggregateFunctionAdapter::PartialFunction>(
                  std::move(fn), resultType);
            } else {
              auto fn = func.value()->factory(
                  core::AggregationNode::Step::kIntermediate,
                  argTypes,
                  resultType);
              return std::make_unique<
                  AggregateFunctionAdapter::PartialFunction>(
                  std::move(fn), argTypes[0]);
            }
          }
          VELOX_FAIL(
              "Original aggregation function {} not found: {}",
              name,
              name + "_partial");
        });
    return true;
  }

  static std::vector<AggregateFunctionSignaturePtr> mergeFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& aggregateSignatures) {
    std::unordered_set<TypeSignature> distinctIntermediateTypes;
    std::vector<AggregateFunctionSignaturePtr> signatures;
    for (const auto& signature : aggregateSignatures) {
      if (distinctIntermediateTypes.count(signature->intermediateType()) > 0) {
        continue;
      }
      distinctIntermediateTypes.insert(signature->intermediateType());

      std::vector<TypeSignature> usedTypes = {signature->intermediateType()};
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());
      std::vector<bool> constantArguments = {false};

      signatures.push_back(std::make_shared<AggregateFunctionSignature>(
          variables,
          signature->intermediateType(),
          signature->intermediateType(),
          std::vector<TypeSignature>{signature->intermediateType()},
          std::move(constantArguments),
          signature->variableArity()));
    }
    return signatures;
  }

  static std::vector<AggregateFunctionSignaturePtr>
  countMergeFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& aggregateSignatures) {
    std::unordered_set<TypeSignature> distinctIntermediateTypes;
    std::vector<AggregateFunctionSignaturePtr> signatures;
    for (const auto& signature : aggregateSignatures) {
      if (signature->constantArguments().size() == 0) {
        // For count_merge, the input cannot be empty.
        continue;
      }
      if (distinctIntermediateTypes.count(signature->intermediateType()) > 0) {
        continue;
      }
      distinctIntermediateTypes.insert(signature->intermediateType());

      std::vector<TypeSignature> usedTypes = {signature->intermediateType()};
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());

      signatures.push_back(std::make_shared<AggregateFunctionSignature>(
          variables,
          signature->intermediateType(),
          signature->intermediateType(),
          std::vector<TypeSignature>{signature->intermediateType()},
          signature->constantArguments(),
          signature->variableArity()));
    }
    return signatures;
  }

  static bool registerMergeFunction(
      const std::string& name,
      const std::vector<AggregateFunctionSignaturePtr>& originalSignatures) {
    std::vector<AggregateFunctionSignaturePtr> signatures;
    if (name == "count") {
      signatures = countMergeFunctionSignatures(originalSignatures);
    } else {
      signatures = mergeFunctionSignatures(originalSignatures);
    }
    exec::registerAggregateFunction(
        name + "_merge",
        std::move(signatures),
        [name](
            core::AggregationNode::Step step,
            const std::vector<TypePtr>& argTypes,
            const TypePtr& resultType) -> std::unique_ptr<Aggregate> {
          if (auto func = getAggregateFunctionEntry(name)) {
            auto fn = func.value()->factory(
                core::AggregationNode::Step::kIntermediate,
                argTypes,
                resultType);
            return std::make_unique<AggregateFunctionAdapter::MergeFunction>(
                std::move(fn), argTypes[0]);
          }
          VELOX_FAIL(
              "Original aggregation function {} not found: {}",
              name,
              name + "_merge");
        });
    return true;
  }

  static bool hasSameIntermediateTypesAcrossSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::unordered_set<TypeSignature> seenTypes;
    for (const auto& signature : signatures) {
      if (seenTypes.count(signature->intermediateType()) > 0) {
        return true;
      }
      seenTypes.insert(signature->intermediateType());
    }
    return false;
  }

  static FunctionSignaturePtr extractFunctionSignature(
      const AggregateFunctionSignaturePtr& signature) {
    std::vector<TypeSignature> usedTypes = {
        signature->intermediateType(), signature->returnType()};
    auto variables = getUsedTypeVariables(usedTypes, signature->variables());
    return std::make_shared<FunctionSignature>(
        variables,
        signature->returnType(),
        std::vector<TypeSignature>{signature->intermediateType()},
        std::vector<bool>{false},
        false);
  }

  static bool registerExtractFunctionWithSuffix(
      const std::string& originalName,
      const std::vector<AggregateFunctionSignaturePtr>& originalSignatures) {
    for (const auto& signature : originalSignatures) {
      auto extractSignature = extractFunctionSignature(signature);
      auto factory = [extractSignature, originalName](
                         const std::string& name,
                         const std::vector<VectorFunctionArg>& inputArgs)
          -> std::shared_ptr<VectorFunction> {
        std::vector<TypePtr> argTypes{inputArgs.size()};
        std::transform(
            inputArgs.begin(),
            inputArgs.end(),
            argTypes.begin(),
            [](auto inputArg) { return inputArg.type; });

        SignatureBinder binder{*extractSignature, argTypes};
        binder.tryBind();
        auto resultType = binder.tryResolveReturnType();
        if (!resultType) {
          // TODO: limitation -- result type must be resolveable given
          // intermediate type of the original UDAF.
          VELOX_NYI();
        }

        if (auto func = getAggregateFunctionEntry(originalName)) {
          auto fn = func.value()->factory(
              core::AggregationNode::Step::kFinal, argTypes, resultType);
          return std::make_shared<AggregateFunctionAdapter::ExtractFunction>(
              std::move(fn));
        }
        return nullptr;
      };

      auto extractFunctionName = originalName + "_extract_" +
          extractSignature->returnType().toString();
      exec::registerStatefulVectorFunction(
          extractFunctionName, {extractSignature}, factory);
    }
    return true;
  }

  static std::vector<FunctionSignaturePtr> extractFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::vector<FunctionSignaturePtr> extractSignatures;
    for (const auto& signature : signatures) {
      extractSignatures.push_back(extractFunctionSignature(signature));
    }
    return extractSignatures;
  }

  static bool registerExtractFunction(
      const std::string& originalName,
      const std::vector<AggregateFunctionSignaturePtr>& originalSignatures) {
    if (hasSameIntermediateTypesAcrossSignatures(originalSignatures)) {
      return registerExtractFunctionWithSuffix(
          originalName, originalSignatures);
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
        VELOX_FAIL(
            "Result type should be resolveable given intermediate type of the original UDAF");
      }

      if (auto func = getAggregateFunctionEntry(originalName)) {
        auto fn = func.value()->factory(
            core::AggregationNode::Step::kFinal, argTypes, resultType);
        return std::make_shared<AggregateFunctionAdapter::ExtractFunction>(
            std::move(fn));
      }
      return nullptr;
    };
    exec::registerStatefulVectorFunction(
        originalName + "_extract",
        extractFunctionSignatures(originalSignatures),
        factory);

    return true;
  }

  static std::vector<FunctionSignaturePtr> retractFunctionSignatures(
      const std::vector<AggregateFunctionSignaturePtr>& signatures) {
    std::vector<FunctionSignaturePtr> retractSignatures;
    for (const auto& signature : signatures) {
      std::vector<TypeSignature> usedTypes = {signature->intermediateType()};
      auto variables = getUsedTypeVariables(usedTypes, signature->variables());
      retractSignatures.push_back(std::make_shared<FunctionSignature>(
          variables,
          signature->intermediateType(),
          std::vector<TypeSignature>{
              signature->intermediateType(), signature->intermediateType()},
          std::vector<bool>{false, false},
          false));
    }
    return retractSignatures;
  }

  static bool registerRetractFunction(
      const std::string& originalName,
      const std::vector<AggregateFunctionSignaturePtr>& originalSignatures) {
    auto factory = [originalName](
                       const std::string& name,
                       const std::vector<VectorFunctionArg>& inputArgs)
        -> std::shared_ptr<VectorFunction> {
      VELOX_CHECK_EQ(inputArgs.size(), 2);
      std::vector<TypePtr> argTypes{inputArgs.size()};
      std::transform(
          inputArgs.begin(),
          inputArgs.end(),
          argTypes.begin(),
          [](auto inputArg) { return inputArg.type; });
      VELOX_CHECK(argTypes[0]->equivalent(*argTypes[1]));

      if (auto func = getAggregateFunctionEntry(originalName)) {
        auto fn = func.value()->factory(
            core::AggregationNode::Step::kIntermediate,
            {argTypes[0]},
            argTypes[0]);
        return std::make_shared<AggregateFunctionAdapter::RetractFunction>(
            std::move(fn));
      }
      return nullptr;
    };
    exec::registerStatefulVectorFunction(
        originalName + "_retract",
        retractFunctionSignatures(originalSignatures),
        factory);

    return true;
  }
};

} // namespace facebook::velox::exec
