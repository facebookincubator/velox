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
#include <iostream>
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
struct SetWithNull {
  bool hasNull;
  std::unordered_set<T> set;

  void reset() {
    hasNull = false;
    set.clear();
  }
};

DecodedVector* decodeArrayElements(
    exec::LocalDecodedVector& arrayDecoder,
    exec::LocalDecodedVector& elementsDecoder,
    const SelectivityVector& rows) {
  auto decodedVector = arrayDecoder.get();
  auto baseArrayVector = arrayDecoder->base()->as<ArrayVector>();

  // Decode and acquire array elements vector.
  auto elementsVector = baseArrayVector->elements();
  auto elementsSelectivityRows = toElementRows(
      elementsVector->size(), rows, baseArrayVector, decodedVector->indices());
  elementsDecoder.get()->decode(*elementsVector, elementsSelectivityRows);
  auto decodedElementsVector = elementsDecoder.get();
  return decodedElementsVector;
}

// See documentation at https://prestodb.io/docs/current/functions/array.html
template <typename T>
class ArrayUnionFunction : public exec::VectorFunction {
 public:
  ArrayUnionFunction() = default;

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    memory::MemoryPool* pool = context->pool();
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftDecoder(context, *left, rows);
    auto decodedLeftArray = leftDecoder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>();
    exec::LocalDecodedVector leftElementsDecoder(context);
    DecodedVector* decodedLeftElements =
        decodeArrayElements(leftDecoder, leftElementsDecoder, rows);

    exec::LocalDecodedVector rightDecoder(context, *right, rows);
    auto decodedRightArray = rightDecoder.get();
    auto baseRightArray = decodedRightArray->base()->as<ArrayVector>();
    exec::LocalDecodedVector rightElementsDecoder(context);
    DecodedVector* decodedRightElements =
        decodeArrayElements(rightDecoder, rightElementsDecoder, rows);

    vector_size_t rowCount = rows.size();

    // Allocate new vectors for indices, nulls, length and offsets.
    BufferPtr newSizes = allocateSizes(rowCount, pool);
    auto rawNewSizes = newSizes->asMutable<vector_size_t>();

    BufferPtr newOffsets = allocateOffsets(rowCount, pool);
    auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

    // Allocate the new elements array
    auto leftElementsCount =
        countElements<ArrayVector>(rows, *decodedLeftArray);
    auto rightElementsCount =
        countElements<ArrayVector>(rows, *decodedRightArray);
    auto newElementsCount = leftElementsCount + rightElementsCount;
    VectorPtr newElementsPtr =
        BaseVector::create(CppToType<T>::create(), newElementsCount, pool);

    vector_size_t offsetCursor = 0;
    // copy the data
    rows.template applyToSelected([&](vector_size_t row) {
      rawNewOffsets[row] = offsetCursor;

      auto leftIndex = leftDecoder->index(row);
      vector_size_t leftSize = baseLeftArray->sizeAt(leftIndex);
      vector_size_t leftOffset = baseLeftArray->offsetAt(leftIndex);
      newElementsPtr->copy(
          decodedLeftElements->base(), offsetCursor, leftOffset, leftSize);
      offsetCursor += leftSize;

      auto rightIndex = rightDecoder->index(row);
      vector_size_t rightOffset = baseRightArray->offsetAt(rightIndex);
      vector_size_t rightSize = baseRightArray->sizeAt(rightIndex);
      newElementsPtr->copy(
          decodedRightElements->base(), offsetCursor, rightOffset, rightSize);
      offsetCursor += rightSize;

      rawNewSizes[row] = leftSize + rightSize;
    });

    // handle duplicates
    SelectivityVector elementsSelectivityVector(newElementsCount);
    vector_size_t duplicateCount = 0;
    auto processRow = [&](SetWithNull<T>& uniqValueRecorder,
                          vector_size_t row) {
      uniqValueRecorder.reset();
      auto offset = rawNewOffsets[row];
      auto size = rawNewSizes[row];

      // there are duplicates, update the offsets
      if (duplicateCount) {
        rawNewOffsets[row] -= duplicateCount;
      }

      FlatVector<T>* flatVector = newElementsPtr->template asFlatVector<T>();
      for (auto i = offset; i < offset + size; i++) {
        if (newElementsPtr->isNullAt(i)) {
          if (uniqValueRecorder.hasNull) {
            duplicateCount++;
            // "delete" this duplicates
            elementsSelectivityVector.setValid(i, false);
            rawNewSizes[row]--;
          } else {
            uniqValueRecorder.hasNull = true;
          }
        } else {
          T val = flatVector->valueAt(i);
          if (uniqValueRecorder.set.count(val) > 0) {
            duplicateCount++;
            // "delete" this duplicates
            elementsSelectivityVector.setValid(i, false);
            rawNewSizes[row]--;
          } else {
            uniqValueRecorder.set.insert(val);
          }
        }
      }
    };

    SetWithNull<T> uniqMarker;
    rows.template applyToSelected(
        [&](vector_size_t row) { processRow(uniqMarker, row); });

    VectorPtr finalElementsPtr = newElementsPtr;
    // has duplicates, we wrap the elements array with the selected indices
    if (duplicateCount > 0) {
      elementsSelectivityVector.updateBounds();
      BufferPtr newIndices =
          allocateIndices(elementsSelectivityVector.countSelected(), pool);
      auto rawNewIndices = newIndices->template asMutable<vector_size_t>();
      vector_size_t index = 0;
      elementsSelectivityVector.applyToSelected(
          [&](vector_size_t row) { rawNewIndices[index++] = row; });

      auto uniqElements =
          BaseVector::transpose(newIndices, std::move(newElementsPtr));
      finalElementsPtr = uniqElements;
    }

    VectorPtr resultArray = std::make_shared<ArrayVector>(
        pool,
        ARRAY(CppToType<T>::create()),
        BufferPtr(nullptr),
        rowCount,
        newOffsets,
        newSizes,
        finalElementsPtr);

    context->moveOrCopyResult(resultArray, rows, result);
  }

}; // class ArrayUnion

template <TypeKind kind>
std::shared_ptr<exec::VectorFunction> createTypedArrayUnion(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  using T = typename TypeTraits<kind>::NativeType;
  return std::make_shared<ArrayUnionFunction<T>>();
}

std::shared_ptr<exec::VectorFunction> createArrayUnion(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  auto elementType = inputArgs.front().type->childAt(0);

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      createTypedArrayUnion, elementType->kind(), inputArgs);
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures(
    const std::string& returnType) {
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType(returnType)
              .argumentType("array(T)")
              .argumentType("array(T)")
              .build()};
}

} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_array_union,
    signatures("array(T)"),
    createArrayUnion);
} // namespace facebook::velox::functions
