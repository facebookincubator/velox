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

#include "velox/common/base/BitUtil.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql {

/// Creates an index vector where each element contains its 0-based position
/// within its respective array. For example, if we have arrays [a, b] and
/// [c, d, e], the index vector will be [0, 1, 0, 1, 2].
/// Spark uses IntegerType (32-bit) for the index parameter.
inline VectorPtr createIndexVector(
    const ArrayVectorPtr& flatArray,
    vector_size_t numElements,
    memory::MemoryPool* pool) {
  auto indexVector =
      BaseVector::create<FlatVector<int32_t>>(INTEGER(), numElements, pool);

  auto* rawOffsets = flatArray->rawOffsets();
  auto* rawSizes = flatArray->rawSizes();
  auto* rawNulls = flatArray->rawNulls();
  auto* rawIndices = indexVector->mutableRawValues();

  for (vector_size_t row = 0; row < flatArray->size(); ++row) {
    if (rawNulls && bits::isBitNull(rawNulls, row)) {
      continue;
    }
    auto offset = rawOffsets[row];
    auto size = rawSizes[row];
    for (vector_size_t i = 0; i < size; ++i) {
      rawIndices[offset + i] = i;
    }
  }

  return indexVector;
}

} // namespace facebook::velox::functions::sparksql
