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
#include "velox/functions/sparksql/ArrayUnion.h"

#include <folly/container/F14Set.h>

#include "velox/expression/DecodedArgs.h"
#include "velox/functions/lib/NormalizeFloatingPoint.h"
#include "velox/type/FloatingPointUtil.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions::sparksql {
namespace {

// Set of distinct elements of primitive type T. Elements are given by their
// index in 'elements', a flat vector.
template <typename T>
class PrimitiveElementSet {
 public:
  explicit PrimitiveElementSet(const BaseVector& elements)
      : elements_(elements.asUnchecked<FlatVector<T>>()) {}

  // Returns true if the element at 'index' was not in the set.
  bool insert(vector_size_t index) {
    return values_.insert(elements_->valueAt(index)).second;
  }

  void clear() {
    values_.clear();
  }

 private:
  const FlatVector<T>* const elements_;
  util::floating_point::HashSetNaNAware<T> values_;
};

// Set of distinct elements of a complex type, or of a type with custom
// comparison. Elements are given by their index in 'elements'. The hash of each
// element is stored with its index, so it is computed once.
class ComplexElementSet {
 public:
  explicit ComplexElementSet(const BaseVector& elements)
      : elements_(&elements), keys_(0, Hash{}, EqualTo{&elements}) {}

  // Returns true if the element at 'index' was not in the set.
  bool insert(vector_size_t index) {
    return keys_.insert({elements_->hashValueAt(index), index}).second;
  }

  void clear() {
    keys_.clear();
  }

 private:
  struct Key {
    uint64_t hash;
    vector_size_t index;
  };

  struct Hash {
    size_t operator()(const Key& key) const {
      return key.hash;
    }
  };

  struct EqualTo {
    const BaseVector* elements;

    bool operator()(const Key& left, const Key& right) const {
      return elements
          ->equalValueAt(
              elements,
              left.index,
              right.index,
              CompareFlags::NullHandlingMode::kNullAsValue)
          .value();
    }
  };

  const BaseVector* const elements_;
  folly::F14FastSet<Key, Hash, EqualTo> keys_;
};

// Copies the elements of the left and then the right array of each row next
// to each other in a new vector, and keeps the first occurrence of each
// element and of null.
template <typename ElementSet>
class ArrayUnionFunction : public exec::VectorFunction {
 public:
  explicit ArrayUnionFunction(bool normalizeFloatingPoint)
      : normalizeFloatingPoint_(normalizeFloatingPoint) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    auto* pool = context.pool();
    exec::DecodedArgs decodedArgs(rows, args, context);
    const auto* left = decodedArgs.at(0);
    const auto* right = decodedArgs.at(1);
    const auto* leftArray = left->base()->asUnchecked<ArrayVector>();
    const auto* rightArray = right->base()->asUnchecked<ArrayVector>();

    std::vector<BaseVector::CopyRange> leftRanges;
    std::vector<BaseVector::CopyRange> rightRanges;
    vector_size_t numElements{0};
    rows.applyToSelected([&](vector_size_t row) {
      const auto leftIndex = left->index(row);
      leftRanges.push_back(
          {leftArray->offsetAt(leftIndex),
           numElements,
           leftArray->sizeAt(leftIndex)});
      numElements += leftArray->sizeAt(leftIndex);

      const auto rightIndex = right->index(row);
      rightRanges.push_back(
          {rightArray->offsetAt(rightIndex),
           numElements,
           rightArray->sizeAt(rightIndex)});
      numElements += rightArray->sizeAt(rightIndex);
    });

    auto elements =
        BaseVector::create(outputType->childAt(0), numElements, pool);
    elements->copyRanges(leftArray->elements().get(), leftRanges);
    elements->copyRanges(rightArray->elements().get(), rightRanges);

    const auto numRows = rows.end();
    auto offsets = allocateOffsets(numRows, pool);
    auto sizes = allocateSizes(numRows, pool);
    auto indices = allocateIndices(numElements, pool);
    auto* rawOffsets = offsets->asMutable<vector_size_t>();
    auto* rawSizes = sizes->asMutable<vector_size_t>();
    auto* rawIndices = indices->asMutable<vector_size_t>();

    ElementSet elementSet(*elements);
    vector_size_t numDistinct{0};
    vector_size_t rangeIndex{0};
    rows.applyToSelected([&](vector_size_t row) {
      rawOffsets[row] = numDistinct;
      const auto begin = leftRanges[rangeIndex].targetIndex;
      const auto end =
          rightRanges[rangeIndex].targetIndex + rightRanges[rangeIndex].count;
      ++rangeIndex;

      bool nullAdded{false};
      for (auto i = begin; i < end; ++i) {
        if (elements->isNullAt(i)) {
          if (!nullAdded) {
            nullAdded = true;
            rawIndices[numDistinct++] = i;
          }
        } else if (elementSet.insert(i)) {
          rawIndices[numDistinct++] = i;
        }
      }
      elementSet.clear();
      rawSizes[row] = numDistinct - rawOffsets[row];
    });
    indices->setSize(numDistinct * sizeof(vector_size_t));

    auto resultElements =
        BaseVector::transpose(std::move(indices), std::move(elements));
    if (normalizeFloatingPoint_) {
      resultElements = normalizeFloatingPoint(resultElements, pool);
    }
    auto localResult = std::make_shared<ArrayVector>(
        pool,
        outputType,
        nullptr,
        numRows,
        std::move(offsets),
        std::move(sizes),
        std::move(resultElements));
    context.moveOrCopyResult(localResult, rows, result);
  }

 private:
  const bool normalizeFloatingPoint_;
};

template <TypeKind kind>
std::shared_ptr<exec::VectorFunction> makePrimitiveArrayUnion(
    bool normalizeFloatingPoint) {
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_shared<ArrayUnionFunction<PrimitiveElementSet<T>>>(
      normalizeFloatingPoint);
}

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> arrayUnionSignatures() {
  return {
      exec::FunctionSignatureBuilder()
          .typeVariable("T")
          .returnType("array(T)")
          .argumentType("array(T)")
          .argumentType("array(T)")
          .build(),
  };
}

std::shared_ptr<exec::VectorFunction> makeArrayUnion(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  VELOX_USER_CHECK_EQ(
      inputArgs.size(), 2, "{} requires exactly two arguments", name);
  const auto& elementType = inputArgs[0].type->childAt(0);
  const bool normalize = containsFloatingPoint(*elementType);
  if (elementType->isPrimitiveType() && !elementType->isUnknown() &&
      !elementType->providesCustomComparison()) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        makePrimitiveArrayUnion, elementType->kind(), normalize);
  }
  return std::make_shared<ArrayUnionFunction<ComplexElementSet>>(normalize);
}

} // namespace facebook::velox::functions::sparksql
