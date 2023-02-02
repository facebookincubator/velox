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

#include "velox/expression/DecodedArgs.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/RowsTranslationUtil.h"

namespace facebook::velox::functions {
namespace {

template <TypeKind kind>
void applyTyped(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    DecodedVector& searchDecoded,
    vector_size_t& indicesCursor,
    BufferPtr& newIndices,
    BufferPtr& newElementNulls,
    BufferPtr& newLengths,
    BufferPtr& newOffsets) {
  using T = typename TypeTraits<kind>::NativeType;

  auto base = arrayDecoded.base()->as<ArrayVector>();
  auto indices = arrayDecoded.indices();

  // Pointers and cursors to the raw data.
  auto rawNewIndices = newIndices->asMutable<vector_size_t>();
  auto rawNewElementNulls = newElementNulls->asMutable<uint64_t>();
  auto rawNewLengths = newLengths->asMutable<vector_size_t>();
  auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

  rows.applyToSelected([&](auto row) {
    rawNewOffsets[row] = indicesCursor;

    if (searchDecoded.isNullAt(row)) {
      bits::setNull(rawNewElementNulls, indicesCursor++, true);
    } else {
      auto elementToRemove = searchDecoded.valueAt<T>(row);

      auto size = base->sizeAt(indices[row]);
      auto offset = base->offsetAt(indices[row]);

      for (auto i = offset; i < offset + size; i++) {
        if (elementsDecoded.isNullAt(i)) {
          // We always keep null values from the input array,
          // they cannot be filtered out because search base is non-null.
          bits::setNull(rawNewElementNulls, indicesCursor++, true);
        } else if (elementsDecoded.valueAt<T>(i) != elementToRemove) {
          rawNewIndices[indicesCursor++] = i;
        }
      }
    }

    rawNewLengths[row] = indicesCursor - rawNewOffsets[row];
  });
}

void applyComplexType(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    DecodedVector& searchDecoded,
    vector_size_t& indicesCursor,
    BufferPtr& newIndices,
    BufferPtr& newElementNulls,
    BufferPtr& newLengths,
    BufferPtr& newOffsets) {
  auto base = arrayDecoded.base()->as<ArrayVector>();
  auto indices = arrayDecoded.indices();

  // Pointers and cursors to the raw data.
  auto rawNewIndices = newIndices->asMutable<vector_size_t>();
  auto rawNewElementNulls = newElementNulls->asMutable<uint64_t>();
  auto rawNewLengths = newLengths->asMutable<vector_size_t>();
  auto rawNewOffsets = newOffsets->asMutable<vector_size_t>();

  // We need to use base elements instead of elementsDecoded as the indices
  // could be remapped.
  auto elementsBase = base->elements();
  auto searchBase = searchDecoded.base();
  auto searchIndices = searchDecoded.indices();

  rows.applyToSelected([&](auto row) {
    rawNewOffsets[row] = indicesCursor;

    auto searchIndex = searchIndices[row];

    if (searchBase->isNullAt(searchIndex)) {
      bits::setNull(rawNewElementNulls, indicesCursor++, true);
    } else {
      auto size = base->sizeAt(indices[row]);
      auto offset = base->offsetAt(indices[row]);

      for (auto i = offset; i < offset + size; i++) {
        if (elementsBase->isNullAt(i)) {
          // We always keep null values from the input array,
          // they cannot be filtered out because search base is non-null.
          bits::setNull(rawNewElementNulls, indicesCursor++, true);
        } else if (!elementsBase->equalValueAt(searchBase, i, searchIndex)) {
          rawNewIndices[indicesCursor++] = i;
        }
      }
    }

    rawNewLengths[row] = indicesCursor - rawNewOffsets[row];
  });
}

template <>
void applyTyped<TypeKind::ARRAY>(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    DecodedVector& searchDecoded,
    vector_size_t& indicesCursor,
    BufferPtr& newIndices,
    BufferPtr& newElementNulls,
    BufferPtr& newLengths,
    BufferPtr& newOffsets) {
  applyComplexType(
      rows,
      arrayDecoded,
      elementsDecoded,
      searchDecoded,
      indicesCursor,
      newIndices,
      newElementNulls,
      newLengths,
      newOffsets);
}

template <>
void applyTyped<TypeKind::MAP>(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    DecodedVector& searchDecoded,
    vector_size_t& indicesCursor,
    BufferPtr& newIndices,
    BufferPtr& newElementNulls,
    BufferPtr& newLengths,
    BufferPtr& newOffsets) {
  applyComplexType(
      rows,
      arrayDecoded,
      elementsDecoded,
      searchDecoded,
      indicesCursor,
      newIndices,
      newElementNulls,
      newLengths,
      newOffsets);
}

template <>
void applyTyped<TypeKind::ROW>(
    const SelectivityVector& rows,
    DecodedVector& arrayDecoded,
    DecodedVector& elementsDecoded,
    DecodedVector& searchDecoded,
    vector_size_t& indicesCursor,
    BufferPtr& newIndices,
    BufferPtr& newElementNulls,
    BufferPtr& newLengths,
    BufferPtr& newOffsets) {
  applyComplexType(
      rows,
      arrayDecoded,
      elementsDecoded,
      searchDecoded,
      indicesCursor,
      newIndices,
      newElementNulls,
      newLengths,
      newOffsets);
}

class ArrayRemoveFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_USER_CHECK_EQ(
        args.size(), 2, "array_remove requires exactly 2 parameters");

    const auto& arrayVector = args[0];
    const auto& searchVector = args[1];

    // Both vectors must be array vectors of the same type.
    VELOX_USER_CHECK(
        arrayVector->type()->isArray(),
        "array_remove requires arguments of type ARRAY");
    VELOX_CHECK(
        arrayVector->type()->asArray().elementType()->kindEquals(
            searchVector->type()),
        "array_remove requires all arguments of the same type: {} vs. {}",
        searchVector->type(),
        arrayVector->type()->asArray().elementType());

    exec::DecodedArgs decodedArgs(rows, args, context);

    DecodedVector* arrayDecoded = decodedArgs.at(0);
    auto base = arrayDecoded->base()->as<ArrayVector>();

    exec::LocalSelectivityVector nestedRows(context, base->elements()->size());
    nestedRows->setAll();
    exec::LocalDecodedVector elementsDecoded(
        context, *base->elements(), *nestedRows);
    DecodedVector* searchDecoded = decodedArgs.at(1);

    vector_size_t elementsCount =
        countElements<ArrayVector>(rows, *arrayDecoded);
    vector_size_t rowCount = arrayDecoded->size();

    // Allocate new vectors for indices, length, and offsets.
    memory::MemoryPool* pool = context.pool();
    BufferPtr newIndices = allocateIndices(elementsCount, pool);
    BufferPtr newElementNulls =
        AlignedBuffer::allocate<bool>(elementsCount, pool, bits::kNotNull);
    BufferPtr newLengths = allocateSizes(rowCount, pool);
    BufferPtr newOffsets = allocateOffsets(rowCount, pool);

    vector_size_t indicesCursor = 0;

    VELOX_DYNAMIC_TYPE_DISPATCH(
        applyTyped,
        searchVector->typeKind(),
        rows,
        *arrayDecoded,
        *elementsDecoded,
        *searchDecoded,
        indicesCursor,
        newIndices,
        newElementNulls,
        newLengths,
        newOffsets);

    auto newElements = BaseVector::wrapInDictionary(
        newElementNulls, newIndices, indicesCursor, base->elements());

    auto localResult = std::make_shared<ArrayVector>(
        pool,
        ARRAY(base->elements()->type()),
        nullptr,
        rowCount,
        newOffsets,
        newLengths,
        newElements,
        0);

    context.moveOrCopyResult(localResult, rows, result);
  }
};

static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  // array(T), T -> array(T)
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .argumentType("array(T)")
              .argumentType("T")
              .returnType("array(T)")
              .build()};
}

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_array_remove,
    signatures(),
    std::make_unique<ArrayRemoveFunction>());

} // namespace facebook::velox::functions
