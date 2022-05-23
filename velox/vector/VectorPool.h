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

#include "velox/vector/FlatVector.h"

namespace facebook::velox {

// A thread-level cache of preallocated flat vectors of different types.
class VectorPool {
 public:
  VectorPtr
  get(const TypePtr& type, vector_size_t size, memory::MemoryPool& pool) {
    auto kind = static_cast<int32_t>(type->kind());
    if (kind < kNumCachedVectorTypes && size <= kMaxRecycleSize) {
      return vectors_[kind].pop(type, size, pool);
    }
    return BaseVector::create(type, size, &pool);
  }

  // Moves vector into 'this' if it is flat, recursively singly referenced and
  // there is space.
  void release(VectorPtr& vector) {
    if (!vector.unique() || vector->size() > kMaxRecycleSize) {
      return;
    }
    auto kind = static_cast<int32_t>(vector->typeKind());
    if (kind < kNumCachedVectorTypes) {
      vectors_[kind].maybe_push_back(vector);
    }
  }

  void release(std::vector<VectorPtr>& vectors) {
    for (auto& vector : vectors) {
      release(vector);
    }
  }

 private:
  static constexpr int32_t kNumCachedVectorTypes =
      static_cast<int32_t>(TypeKind::ARRAY);
  // Max number of elements for a vector to be recyclable. The larger
  // the batch the less the win from recycling.
  static constexpr vector_size_t kMaxRecycleSize = 64 * 1024;
  static constexpr int32_t kNumPerType = 10;
  struct TypePool {
    int32_t size{0};
    std::array<VectorPtr, kNumPerType> vectors;

    void maybe_push_back(VectorPtr& vector) {
      if (!vector->isRecyclable()) {
        return;
      }
      if (size < kNumPerType) {
        vector->prepareForReuse();
        vectors[size++] = std::move(vector);
      }
    }

    VectorPtr pop(
        const TypePtr& type,
        vector_size_t vectorSize,
        memory::MemoryPool& pool) {
      if (size) {
        auto result = std::move(vectors[--size]);
        if (UNLIKELY(result->rawNulls() != nullptr)) {
          // This is a recyclable vector, no need to check uniqueness.
          simd::memset(
              const_cast<uint64_t*>(result->rawNulls()),
              bits::kNotNullByte,
              bits::roundUp(std::min<int32_t>(vectorSize, result->size()), 64) /
                  8);
        }
        if (UNLIKELY(
                result->typeKind() == TypeKind::VARCHAR ||
                result->typeKind() == TypeKind::VARBINARY)) {
          simd::memset(
              const_cast<void*>(result->valuesAsVoid()),
              0,
              std::min<int32_t>(vectorSize, result->size()) *
                  sizeof(StringView));
        }
        if (result->size() != vectorSize) {
          result->resize(vectorSize);
        }
        return result;
      }
      return BaseVector::create(type, vectorSize, &pool);
    }
  };

  // Caches of preallocated vectors indexed by typeKind.
  std::array<TypePool, kNumCachedVectorTypes> vectors_;
};

} // namespace facebook::velox
