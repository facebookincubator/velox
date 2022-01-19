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
#include <stdint.h>

#include <folly/container/F14Set.h>

#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions {
namespace {

FOLLY_ALWAYS_INLINE vector_size_t getTotalElementsCount(
    const SelectivityVector& rows,
    DecodedVector& leftArrayDecoded,
    DecodedVector& rightArrayDecoded) {
  auto leftElementsCount = countElements<ArrayVector>(rows, leftArrayDecoded);
  auto rightElementsCount = countElements<ArrayVector>(rows, rightArrayDecoded);

  if (UNLIKELY(
          leftElementsCount < 0 || rightElementsCount < 0 ||
          leftElementsCount > INT32_MAX - rightElementsCount)) {
    VELOX_FAIL("Vector size overflow in array_union.");
  }

  return leftElementsCount + rightElementsCount;
}

template <typename T>
void addScalarElementsToUnion(
    const DecodedVector* elementsDecoded,
    int size,
    int offset,
    bool& hasNull,
    int& newElementsCount,
    FlatVectorPtr<T>& newElements,
    folly::F14FastSet<T>& unionSet) {
  for (int i = 0; i < size; ++i) {
    if (elementsDecoded->isNullAt(offset + i)) {
      if (!hasNull) {
        hasNull = true;
        newElements->setNull(newElementsCount++, true);
      }
    } else {
      auto value = elementsDecoded->valueAt<T>(offset + i);
      if (unionSet.count(value) == 0) {
        unionSet.insert(value);
        newElements->set(newElementsCount++, value);
      }
    }
  }
}

struct ComplexElementLocator {
  ComplexElementLocator(const BaseVector* array, const vector_size_t index)
      : array(array), index(index) {}

  const BaseVector* array;
  const vector_size_t index;
};

struct ComplexElementEquality {
  bool operator()(
      const ComplexElementLocator& lhs,
      const ComplexElementLocator& rhs) const {
    return lhs.array->equalValueAt(rhs.array, lhs.index, rhs.index);
  }
};

struct ComplexElementHash {
  size_t operator()(const ComplexElementLocator& operand) const {
    return operand.array->hashValueAt(operand.index);
  }
};

void addComplexElementsToUnion(
    const BaseVector* elementsBase,
    int size,
    int offset,
    bool& hasNull,
    int& newElementsCount,
    VectorPtr& newElements,
    folly::F14FastSet<
        ComplexElementLocator,
        ComplexElementHash,
        ComplexElementEquality>& unionSet) {
  for (int i = 0; i < size; ++i) {
    if (elementsBase->isNullAt(offset + i)) {
      if (!hasNull) {
        hasNull = true;
        newElements->setNull(newElementsCount++, true);
      }
    } else {
      ComplexElementLocator locator(elementsBase, offset + i);
      if (unionSet.count(locator) == 0) {
        unionSet.insert(locator);
        newElements->copy(elementsBase, newElementsCount++, offset + i, 1);
      }
    }
  }
}

template <TypeKind kind>
void applyTyped(
    const SelectivityVector& rows,
    exec::EvalCtx* context,
    const TypePtr& elementType,
    DecodedVector& leftArrayDecoded,
    DecodedVector& rightArrayDecoded,
    VectorPtr* result) {
  auto leftBaseArray = leftArrayDecoded.base()->as<ArrayVector>();
  auto leftRawSizes = leftBaseArray->rawSizes();
  auto leftRawOffsets = leftBaseArray->rawOffsets();
  auto leftIndices = leftArrayDecoded.indices();

  auto rightBaseArray = rightArrayDecoded.base()->as<ArrayVector>();
  auto rightRawSizes = rightBaseArray->rawSizes();
  auto rightRawOffsets = rightBaseArray->rawOffsets();
  auto rightIndices = rightArrayDecoded.indices();

  auto leftElements = leftArrayDecoded.base()->as<ArrayVector>()->elements();
  exec::LocalSelectivityVector leftNestedRows(context, leftElements->size());
  leftNestedRows.get()->setAll();
  exec::LocalDecodedVector leftElementsHolder(
      context, *leftElements, *leftNestedRows.get());
  auto leftElementsDecoded = leftElementsHolder.get();

  auto rightElements = rightArrayDecoded.base()->as<ArrayVector>()->elements();
  exec::LocalSelectivityVector rightNestedRows(context, rightElements->size());
  rightNestedRows.get()->setAll();
  exec::LocalDecodedVector rightElementsHolder(
      context, *rightElements, *rightNestedRows.get());
  auto rightElementsDecoded = rightElementsHolder.get();

  auto totalElementsCount =
      getTotalElementsCount(rows, leftArrayDecoded, rightArrayDecoded);
  auto newElements =
      BaseVector::create(elementType, totalElementsCount, context->pool());

  BufferPtr newOffsets =
      AlignedBuffer::allocate<vector_size_t>(rows.end(), context->pool());
  BufferPtr newSizes =
      AlignedBuffer::allocate<vector_size_t>(rows.end(), context->pool());

  auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();
  auto rawNewSizes = newSizes->asMutable<vector_size_t>();

  bool hasNull;
  int newElementsCount = 0;

  auto createUnion = [&](auto leftElements,
                         auto rightElements,
                         auto& newElements,
                         auto& unionSet,
                         auto addElementsToUnion) {
    rows.applyToSelected([&](auto row) {
      unionSet.clear();
      hasNull = false;

      auto leftSize = leftRawSizes[leftIndices[row]];
      auto leftOffset = leftRawOffsets[leftIndices[row]];
      addElementsToUnion(
          leftElements,
          leftSize,
          leftOffset,
          hasNull,
          newElementsCount,
          newElements,
          unionSet);

      auto rightSize = rightRawSizes[rightIndices[row]];
      auto rightOffset = rightRawOffsets[rightIndices[row]];
      addElementsToUnion(
          rightElements,
          rightSize,
          rightOffset,
          hasNull,
          newElementsCount,
          newElements,
          unionSet);

      rawNewSizes[row] = unionSet.size();
      if (hasNull) {
        ++rawNewSizes[row];
      }

      rawNewOffsets[row] = newElementsCount - rawNewSizes[row];
    });
  };

  if constexpr (TypeTraits<kind>::isPrimitiveType) {
    using T = typename TypeTraits<kind>::NativeType;

    folly::F14FastSet<T> unionSet;
    auto flatNewElements =
        std::dynamic_pointer_cast<FlatVector<T>>(newElements);

    createUnion(
        leftElementsDecoded,
        rightElementsDecoded,
        flatNewElements,
        unionSet,
        addScalarElementsToUnion<T>);
  } else {
    auto leftElementsBase = leftElementsDecoded->base();
    auto rightElementsBase = rightElementsDecoded->base();

    folly::F14FastSet<
        ComplexElementLocator,
        ComplexElementHash,
        ComplexElementEquality>
        unionSet;

    createUnion(
        leftElementsBase,
        rightElementsBase,
        newElements,
        unionSet,
        addComplexElementsToUnion);
  }

  newElements->resize(newElementsCount);

  auto resultArray = std::make_shared<ArrayVector>(
      context->pool(),
      ARRAY(elementType),
      BufferPtr(nullptr),
      rows.end(),
      newOffsets,
      newSizes,
      newElements);
  context->moveOrCopyResult(resultArray, rows, result);
}

class ArrayUnionFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    const auto& leftVector = args[0];
    const auto& rightVector = args[1];
    VELOX_CHECK(leftVector->type()->asArray().elementType()->kindEquals(
        rightVector->type()->asArray().elementType()));

    exec::DecodedArgs decodedArgs(rows, args, context);

    VELOX_DYNAMIC_TYPE_DISPATCH(
        applyTyped,
        leftVector->type()->asArray().elementType()->kind(),
        rows,
        context,
        leftVector->type()->asArray().elementType(),
        *decodedArgs.at(0),
        *decodedArgs.at(1),
        result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // array(T), array(T) -> array(T)
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("array(T)")
                .argumentType("array(T)")
                .argumentType("array(T)")
                .build()};
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_union,
    ArrayUnionFunction::signatures(),
    std::make_unique<ArrayUnionFunction>());

} // namespace facebook::velox::functions
