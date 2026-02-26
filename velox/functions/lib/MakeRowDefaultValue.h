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

#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/VectorPool.h"

namespace facebook::velox::functions {

// Used to create a vector of a given type with default values. For primitive
// types, it will be set to the c++ default, like 0 for INTEGER and empty string
// for VARCHAR. For Array/Map they will be set to an empty array or map. Not
// supported for ROW or UNKNOWN types.
class MakeRowDefaultValue {
 public:
  // Returns a vector with flat encoding of the given type and size with default
  // values. Optionally accepts a vector pool to enable recycling of vectors.
  static VectorPtr createFlat(
      const TypePtr& type,
      const vector_size_t size,
      memory::MemoryPool& pool,
      VectorPool* vectorPool) {
    VectorPtr result = vectorPool ? vectorPool->get(type, size)
                                  : BaseVector::create(type, size, &pool);
    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(fillWithDefaultValue, type->kind(), result);
    return result;
  }

  // Returns a vector with constant encoding of the given type and size with
  // default values.
  static VectorPtr createConstant(
      const TypePtr& type,
      const vector_size_t size,
      memory::MemoryPool& pool) {
    VectorPtr result = velox::BaseVector::create(type, 1, &pool);
    VELOX_DYNAMIC_TYPE_DISPATCH_ALL(fillWithDefaultValue, type->kind(), result);
    result = velox::BaseVector::wrapInConstant(size, 0, result);
    return result;
  }

 private:
  template <TypeKind ValueKind>
  static void fillWithDefaultValue(VectorPtr& vector) {
    VELOX_CHECK_NOT_NULL(vector);
    if constexpr (
        (TypeTraits<ValueKind>::isPrimitiveType ||
         ValueKind == TypeKind::OPAQUE) &&
        ValueKind != TypeKind::UNKNOWN) {
      using T = typename TypeTraits<ValueKind>::NativeType;
      VELOX_CHECK(vector->isFlatEncoding());
      auto* rawValues = vector->asFlatVector<T>()->mutableRawValues();
      std::fill(rawValues, rawValues + vector->size(), T());
    } else if constexpr (
        ValueKind == TypeKind::ARRAY || ValueKind == TypeKind::MAP) {
      auto vectorSize = vector->size();
      auto arrayBaseVector = vector->asChecked<ArrayVectorBase>();
      VELOX_CHECK_NOT_NULL(arrayBaseVector);
      auto* rawOffsets = arrayBaseVector->mutableOffsets(vectorSize)
                             ->asMutable<vector_size_t>();
      auto* rawSizes =
          arrayBaseVector->mutableSizes(vectorSize)->asMutable<vector_size_t>();
      std::fill(rawOffsets, rawOffsets + vectorSize, 0);
      std::fill(rawSizes, rawSizes + vectorSize, 0);
    } else {
      VELOX_USER_FAIL(
          "Unsupported type for replacing nulls: {}",
          vector->type()->toString());
    }
  }
};

} // namespace facebook::velox::functions
