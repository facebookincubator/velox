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
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <map>

namespace facebook::velox {

VectorPtr variantToVector(
    const TypePtr& type,
    const Variant& value,
    memory::MemoryPool* pool) {
  return BaseVector::createConstant(type, value, 1, pool);
}

namespace {

Variant nullVariant(const TypePtr& type) {
  return Variant(type->kind());
}

template <TypeKind kind>
Variant variantAt(const VectorPtr& vector, int32_t row) {
  using T = typename TypeTraits<kind>::NativeType;

  const T value = vector->as<SimpleVector<T>>()->valueAt(row);

  if (vector->type()->providesCustomComparison()) {
    return Variant::typeWithCustomComparison<kind>(value, vector->type());
  }

  return Variant(value);
}

template <>
Variant variantAt<TypeKind::VARBINARY>(const VectorPtr& vector, int32_t row) {
  return Variant::binary(
      std::string(vector->as<SimpleVector<StringView>>()->valueAt(row)));
}

Variant variantAt(const VectorPtr& vector, vector_size_t row);

Variant arrayVariantAt(const VectorPtr& vector, vector_size_t row) {
  auto arrayVector = vector->wrappedVector()->as<ArrayVector>();
  auto& elements = arrayVector->elements();

  auto wrappedRow = vector->wrappedIndex(row);
  auto offset = arrayVector->offsetAt(wrappedRow);
  auto size = arrayVector->sizeAt(wrappedRow);

  std::vector<Variant> array;
  array.reserve(size);
  for (vector_size_t i = 0; i < size; ++i) {
    array.push_back(variantAt(elements, offset + i));
  }

  return Variant::array(array);
}

Variant mapVariantAt(const VectorPtr& vector, vector_size_t row) {
  auto mapVector = vector->wrappedVector()->as<MapVector>();
  auto& keys = mapVector->mapKeys();
  auto& values = mapVector->mapValues();

  auto wrappedRow = vector->wrappedIndex(row);
  auto offset = mapVector->offsetAt(wrappedRow);
  auto size = mapVector->sizeAt(wrappedRow);

  std::map<Variant, Variant> map;
  for (vector_size_t i = 0; i < size; ++i) {
    map.emplace(variantAt(keys, offset + i), variantAt(values, offset + i));
  }

  return Variant::map(std::move(map));
}

Variant rowVariantAt(const VectorPtr& vector, vector_size_t row) {
  auto rowVector = vector->wrappedVector()->as<RowVector>();
  auto wrappedRow = vector->wrappedIndex(row);

  std::vector<Variant> rowValues;
  rowValues.reserve(rowVector->childrenSize());
  for (vector_size_t i = 0; i < rowVector->childrenSize(); ++i) {
    rowValues.push_back(variantAt(rowVector->childAt(i), wrappedRow));
  }

  return Variant::row(rowValues);
}

Variant variantAt(const VectorPtr& vector, vector_size_t row) {
  if (vector->isNullAt(row)) {
    return nullVariant(vector->type());
  }

  auto typeKind = vector->typeKind();

  if (vector->isConstantEncoding()) {
    if (vector->valueVector()) {
      return variantAt(vector->valueVector(), vector->wrappedIndex(row));
    }
    return variantAt(BaseVector::wrappedVectorShared(vector), 0);
  }

  switch (typeKind) {
    case TypeKind::ARRAY:
      return arrayVariantAt(vector, row);
    case TypeKind::MAP:
      return mapVariantAt(vector, row);
    case TypeKind::ROW:
      return rowVariantAt(vector, row);
    case TypeKind::OPAQUE:
      return Variant::opaque(
          vector->as<SimpleVector<std::shared_ptr<void>>>()->valueAt(row),
          std::dynamic_pointer_cast<const OpaqueType>(vector->type()));
    default:
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          variantAt, typeKind, vector, row);
  }
}

} // namespace

Variant vectorToVariant(const VectorPtr& vector, vector_size_t index) {
  return variantAt(vector, index);
}

} // namespace facebook::velox
