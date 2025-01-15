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

#include "velox/serializers/PrestoSerializerSerializationUtils.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/VectorStream.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::serializer::presto::detail {
template <TypeKind Kind>
void estimateFlatSerializedSize(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes) {
  const auto valueSize = vector->type()->cppSizeInBytes();
  if (vector->mayHaveNulls()) {
    const auto* rawNulls = vector->rawNulls();
    for (int32_t i = 0; i < ranges.size(); ++i) {
      const auto end = ranges[i].begin + ranges[i].size;
      const auto numValues = bits::countBits(rawNulls, ranges[i].begin, end);
      // Add the size of the values.
      *(sizes[i]) += numValues * valueSize;
      // Add the size of the null bit mask if there are nulls in the range.
      if (numValues != ranges[i].size) {
        *(sizes[i]) += bits::nbytes(ranges[i].size);
      }
    }
  } else {
    for (int32_t i = 0; i < ranges.size(); ++i) {
      // Add the size of the values (there's not bit mask since there are no
      // nulls).
      *(sizes[i]) += ranges[i].size * valueSize;
    }
  }
}

void estimateFlatSerializedSizeVarcharOrVarbinary(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes);

template <>
inline void estimateFlatSerializedSize<TypeKind::VARCHAR>(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, ranges, sizes);
}

template <>
inline void estimateFlatSerializedSize<TypeKind::VARBINARY>(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes) {
  estimateFlatSerializedSizeVarcharOrVarbinary(vector, ranges, sizes);
}

template <>
inline void estimateFlatSerializedSize<TypeKind::OPAQUE>(
    const BaseVector*,
    const folly::Range<const IndexRange*>&,
    vector_size_t**) {
  VELOX_FAIL("Opaque type support is not implemented.");
}

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    Scratch& scratch);

void estimateSerializedSizeInt(
    const BaseVector* vector,
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes,
    Scratch& scratch);

void estimateWrapperSerializedSize(
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    const BaseVector* wrapper,
    Scratch& scratch);

void expandRepeatedRanges(
    const BaseVector* vector,
    const vector_size_t* rawOffsets,
    const vector_size_t* rawSizes,
    const folly::Range<const IndexRange*>& ranges,
    vector_size_t** sizes,
    std::vector<IndexRange>* childRanges,
    std::vector<vector_size_t*>* childSizes);
} // namespace facebook::velox::serializer::presto::detail
