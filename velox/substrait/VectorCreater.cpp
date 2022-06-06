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

#include "velox/substrait/VectorCreater.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::substrait {

template <TypeKind KIND>
VectorPtr setVectorFromVariantsByKind(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<KIND>::NativeType;

  auto flatVector = std::dynamic_pointer_cast<FlatVector<T>>(
      BaseVector::create(CppToType<T>::create(), value.size(), pool));

  for (vector_size_t i = 0; i < value.size(); i++) {
    if (value[i].isNull()) {
      flatVector->setNull(i, true);
    } else {
      flatVector->set(i, value[i].value<T>());
    }
  }
  return flatVector;
}

template <>
VectorPtr setVectorFromVariantsByKind<TypeKind::VARBINARY>(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  throw std::invalid_argument("Return of VARBINARY data is not supported");
}

template <>
VectorPtr setVectorFromVariantsByKind<TypeKind::VARCHAR>(
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  auto flatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
      BaseVector::create(VARCHAR(), value.size(), pool));

  for (vector_size_t i = 0; i < value.size(); i++) {
    if (value[i].isNull()) {
      flatVector->setNull(i, true);
    } else {
      flatVector->set(i, StringView(value[i].value<Varchar>()));
    }
  }
  return flatVector;
}

VectorPtr setVectorFromVariants(
    const TypePtr& type,
    const std::vector<velox::variant>& value,
    memory::MemoryPool* pool) {
  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      setVectorFromVariantsByKind, type->kind(), value, pool);
}

ArrayVectorPtr
toArrayVector(TypePtr type, VectorPtr vector, memory::MemoryPool* pool) {
  vector_size_t size = 1;
  BufferPtr offsets = AlignedBuffer::allocate<vector_size_t>(size, pool);
  BufferPtr sizes = AlignedBuffer::allocate<vector_size_t>(size, pool);
  BufferPtr nulls = AlignedBuffer::allocate<uint64_t>(size, pool);

  auto rawOffsets = offsets->asMutable<vector_size_t>();
  auto rawSizes = sizes->asMutable<vector_size_t>();
  auto rawNulls = nulls->asMutable<uint64_t>();

  bits::fillBits(rawNulls, 0, size, pool);
  vector_size_t nullCount = 0;

  *rawSizes++ = vector->size();
  *rawOffsets++ = 0;

  return std::make_shared<ArrayVector>(
      pool, ARRAY(type), nulls, size, offsets, sizes, vector, nullCount);
}

} // namespace facebook::velox::substrait
