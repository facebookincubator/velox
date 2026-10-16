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
#include "velox/functions/lib/NormalizeFloatingPoint.h"

#include <cstring>

#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
bool needsNormalization(T value) {
  const T normalized = normalizeFloatingPoint(value);
  return std::memcmp(&normalized, &value, sizeof(T)) != 0;
}

template <typename T>
VectorPtr normalizeFlat(const VectorPtr& vector, memory::MemoryPool* pool) {
  const auto* flat = vector->asUnchecked<FlatVector<T>>();
  const auto size = vector->size();
  vector_size_t firstToNormalize = 0;
  while (firstToNormalize < size &&
         (flat->isNullAt(firstToNormalize) ||
          !needsNormalization(flat->valueAtFast(firstToNormalize)))) {
    ++firstToNormalize;
  }
  if (firstToNormalize == size) {
    return vector;
  }

  auto result = BaseVector::create<FlatVector<T>>(vector->type(), size, pool);
  for (vector_size_t i = 0; i < size; ++i) {
    if (flat->isNullAt(i)) {
      result->setNull(i, true);
    } else {
      result->set(i, normalizeFloatingPoint(flat->valueAtFast(i)));
    }
  }
  return result;
}

template <typename T>
VectorPtr normalizeConstant(const VectorPtr& vector, memory::MemoryPool* pool) {
  const auto* constant = vector->asUnchecked<ConstantVector<T>>();
  if (constant->isNullAt(0) || !needsNormalization(constant->valueAt(0))) {
    return vector;
  }
  return std::make_shared<ConstantVector<T>>(
      pool,
      vector->size(),
      false,
      vector->type(),
      normalizeFloatingPoint(constant->valueAt(0)));
}

VectorPtr normalizeArray(const VectorPtr& vector, memory::MemoryPool* pool) {
  const auto* array = vector->asUnchecked<ArrayVector>();
  auto elements = normalizeFloatingPoint(array->elements(), pool);
  if (elements == array->elements()) {
    return vector;
  }
  return std::make_shared<ArrayVector>(
      pool,
      vector->type(),
      vector->nulls(),
      vector->size(),
      array->offsets(),
      array->sizes(),
      std::move(elements));
}

VectorPtr normalizeMap(const VectorPtr& vector, memory::MemoryPool* pool) {
  const auto* map = vector->asUnchecked<MapVector>();
  auto keys = normalizeFloatingPoint(map->mapKeys(), pool);
  auto values = normalizeFloatingPoint(map->mapValues(), pool);
  if (keys == map->mapKeys() && values == map->mapValues()) {
    return vector;
  }
  return std::make_shared<MapVector>(
      pool,
      vector->type(),
      vector->nulls(),
      vector->size(),
      map->offsets(),
      map->sizes(),
      std::move(keys),
      std::move(values));
}

VectorPtr normalizeRow(const VectorPtr& vector, memory::MemoryPool* pool) {
  const auto* row = vector->asUnchecked<RowVector>();
  std::vector<VectorPtr> children;
  children.reserve(row->childrenSize());
  bool changed = false;
  for (const auto& child : row->children()) {
    children.push_back(normalizeFloatingPoint(child, pool));
    changed |= children.back() != child;
  }
  if (!changed) {
    return vector;
  }
  return std::make_shared<RowVector>(
      pool, vector->type(), vector->nulls(), vector->size(), children);
}

} // namespace

bool containsFloatingPoint(const Type& type) {
  if (type.kind() == TypeKind::REAL || type.kind() == TypeKind::DOUBLE) {
    return true;
  }
  for (uint32_t i = 0; i < type.size(); ++i) {
    if (containsFloatingPoint(*type.childAt(i))) {
      return true;
    }
  }
  return false;
}

VectorPtr normalizeFloatingPoint(
    const VectorPtr& vector,
    memory::MemoryPool* pool) {
  if (vector == nullptr || !containsFloatingPoint(*vector->type())) {
    return vector;
  }

  switch (vector->encoding()) {
    case VectorEncoding::Simple::DICTIONARY: {
      auto base = normalizeFloatingPoint(vector->valueVector(), pool);
      if (base == vector->valueVector()) {
        return vector;
      }
      return BaseVector::wrapInDictionary(
          vector->nulls(), vector->wrapInfo(), vector->size(), std::move(base));
    }
    case VectorEncoding::Simple::CONSTANT: {
      if (vector->type()->kind() == TypeKind::REAL) {
        return normalizeConstant<float>(vector, pool);
      }
      if (vector->type()->kind() == TypeKind::DOUBLE) {
        return normalizeConstant<double>(vector, pool);
      }
      if (vector->isNullAt(0)) {
        return vector;
      }
      auto base = normalizeFloatingPoint(vector->valueVector(), pool);
      if (base == vector->valueVector()) {
        return vector;
      }
      return BaseVector::wrapInConstant(
          vector->size(),
          vector->asUnchecked<ConstantVector<ComplexType>>()->index(),
          std::move(base));
    }
    case VectorEncoding::Simple::FLAT:
      if (vector->type()->kind() == TypeKind::REAL) {
        return normalizeFlat<float>(vector, pool);
      }
      return normalizeFlat<double>(vector, pool);
    case VectorEncoding::Simple::ARRAY:
      return normalizeArray(vector, pool);
    case VectorEncoding::Simple::MAP:
      return normalizeMap(vector, pool);
    case VectorEncoding::Simple::ROW:
      return normalizeRow(vector, pool);
    case VectorEncoding::Simple::LAZY:
      return normalizeFloatingPoint(
          BaseVector::loadedVectorShared(vector), pool);
    default:
      VELOX_UNSUPPORTED(
          "Unsupported encoding for normalizeFloatingPoint: {}",
          vector->encoding());
  }
}

} // namespace facebook::velox::functions
