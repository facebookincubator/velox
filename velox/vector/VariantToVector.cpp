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

#include "velox/vector/VariantToVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::core {
namespace {

template <TypeKind KIND>
ArrayVectorPtr variantArrayToVectorImpl(
    const TypePtr& arrayType,
    const std::vector<variant>& variantArray,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<KIND>::NativeType;

  // First generate internal arrayVector elements.
  const size_t variantArraySize = variantArray.size();

  // Create array elements flat vector.
  auto arrayElements = BaseVector::create<FlatVector<T>>(
      arrayType->childAt(0), variantArraySize, pool);

  // Populate internal array elements (flat vector).
  for (vector_size_t i = 0; i < variantArraySize; i++) {
    const auto& value = variantArray[i];
    if (!value.isNull()) {
      // `getOwnedValue` copies the content to its internal buffers (in case of
      // string/StringView); no-op for other primitive types.
      arrayElements->set(i, T(value.value<KIND>()));
    } else {
      arrayElements->setNull(i, true);
    }
  }

  // Create ArrayVector around the FlatVector containing array elements.
  BufferPtr offsets = allocateOffsets(1, pool);
  BufferPtr sizes = allocateSizes(1, pool);

  auto rawSizes = sizes->asMutable<vector_size_t>();
  rawSizes[0] = variantArraySize;

  return std::make_shared<ArrayVector>(
      pool, arrayType, nullptr, 1, offsets, sizes, arrayElements);
}
} // namespace

ArrayVectorPtr variantArrayToVector(
    const TypePtr& arrayType,
    const std::vector<variant>& variantArray,
    velox::memory::MemoryPool* pool) {
  VELOX_CHECK_EQ(TypeKind::ARRAY, arrayType->kind());

  if (arrayType->childAt(0)->isUnKnown()) {
    return variantArrayToVectorImpl<TypeKind::UNKNOWN>(
        arrayType, variantArray, pool);
  }

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      variantArrayToVectorImpl,
      arrayType->childAt(0)->kind(),
      arrayType,
      variantArray,
      pool);
}

struct NestedTypeCounter {
  TypeKind kind = TypeKind::UNKNOWN;
  vector_size_t memberCount = 0;
  vector_size_t membersInserted = 0;
  std::vector<NestedTypeCounter> children;
};

static void validateOrSetType(NestedTypeCounter& counter, const variant& v) {
  if (v.isNull()) // If a variant is null, it can be UNKNOWN, INVALID, or
                  // matching our expected kind
  {
    if (v.kind() != TypeKind::UNKNOWN && v.kind() != TypeKind::INVALID &&
        v.kind() != counter.kind) {
      throw std::invalid_argument("Variant was of an unexpected kind");
    }
    return;
  } else {
    if (v.kind() == TypeKind::UNKNOWN || v.kind() == TypeKind::INVALID) {
      throw std::invalid_argument(
          "Non-null variant has unknown or invalid kind");
    }
    if (v.kind() != counter.kind) {
      if (counter.kind == TypeKind::UNKNOWN) {
        counter.kind = v.kind();
      } else {
        throw std::invalid_argument("Variant kind doesn't match predecessors");
      }
    }
  }
}

static void computeRowCountsAndType(
    NestedTypeCounter& counter,
    const variant& v) {
  validateOrSetType(counter, v);
  counter.memberCount += 1;
  if (v.isNull()) {
    return;
  }
  // At this point, we know that v.kind() equals counter.kind
  switch (counter.kind) {
    case TypeKind::ARRAY: {
      const std::vector<variant>& elements = v.array();
      counter.children.resize(1);
      for (const variant& elt : elements) {
        computeRowCountsAndType(counter.children[0], elt);
      }
    } break;
    default:
      // we have a simple type with no children
      break;
  }
}

static VectorPtr allocateVector(
    const NestedTypeCounter& counter,
    velox::memory::MemoryPool* pool) {
  switch (counter.kind) {
    case TypeKind::ARRAY: {
      BufferPtr sizes =
          AlignedBuffer::allocate<vector_size_t>(counter.memberCount, pool, 0);
      BufferPtr offsets =
          AlignedBuffer::allocate<vector_size_t>(counter.memberCount, pool, 0);
      BufferPtr nulls =
          AlignedBuffer::allocate<vector_size_t>(counter.memberCount, pool, 0);
      VectorPtr elements = allocateVector(counter.children.at(0), pool);
      return std::make_shared<ArrayVector>(
          pool,
          ARRAY(elements->type()),
          nulls,
          counter.memberCount,
          std::move(offsets),
          std::move(sizes),
          std::move(elements));
    }
    default: {
      TypePtr type;
      type = createScalarType(counter.kind);
      return BaseVector::create(std::move(type), counter.memberCount, pool);
    }
  }
}

template <TypeKind Kind>
void setElementInFlatVector(
    vector_size_t idx,
    const variant& v,
    VectorPtr& vector) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto asFlat = vector->asFlatVector<NativeType>();
  asFlat->set(idx, NativeType{v.value<NativeType>()});
}

static void incrementChildRowsInserted(NestedTypeCounter& counter) {
  for (NestedTypeCounter& child : counter.children) {
    incrementChildRowsInserted(child);
    child.membersInserted += 1;
  }
}

static void insertVariantIntoVector(
    NestedTypeCounter& counter,
    const variant& v,
    VectorPtr& vector) {
  if (v.isNull()) {
    vector->setNull(counter.membersInserted, true);
    if (counter.kind == TypeKind::ROW) {
      incrementChildRowsInserted(counter);
    }
  } else
    switch (counter.kind) {
      case TypeKind::ARRAY: {
        auto asArray = vector->as<ArrayVector>();
        const std::vector<variant>& elements = v.array();
        vector_size_t offset = 0;
        if (counter.membersInserted != 0) {
          offset = asArray->offsetAt(counter.membersInserted - 1) +
              asArray->sizeAt(counter.membersInserted - 1);
        }
        asArray->setOffsetAndSize(
            counter.membersInserted, offset, elements.size());
        for (const variant& elt : elements) {
          insertVariantIntoVector(
              counter.children.at(0), elt, asArray->elements());
        }
        break;
      }
      default: {
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            setElementInFlatVector,
            counter.kind,
            counter.membersInserted,
            v,
            vector);
        break;
      }
    }
  counter.membersInserted += 1;
}

VectorPtr variantsToVector(
    const std::vector<variant>& variants,
    velox::memory::MemoryPool* pool) {
  NestedTypeCounter counter;
  for (const variant& v : variants) {
    computeRowCountsAndType(counter, v);
  }
  VectorPtr resultVector = allocateVector(counter, pool);
  for (const variant& v : variants) {
    insertVariantIntoVector(counter, v, resultVector);
  }
  return resultVector;
}

} // namespace facebook::velox::core
