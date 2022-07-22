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

#include "velox/common/memory/ByteStream.h"
#include "velox/functions/prestosql/aggregates/MaxSizeForStatsEstimator.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

namespace {

void estimateSizeOfArrayElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) {
  VELOX_CHECK(offset >= 0 && offset + length <= source.size());
  auto array = source.wrappedVector()->asUnchecked<ArrayVector>();
  for (auto i = 0; i < length; ++i) {
    auto index = i + offset;
    auto wrappedIndex = source.wrappedIndex(index);
    if (!array->isNullAt(wrappedIndex)) {
      MaxSizeForStatsEstimator::instance().estimateSizeOfVectorElements(
          *array->elements(),
          array->offsetAt(wrappedIndex),
          array->sizeAt(wrappedIndex),
          size_out);
    }
  }
}

void estimateSizeOfMapElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) {
  VELOX_CHECK(offset >= 0 && offset + length <= source.size());
  auto map = source.wrappedVector()->asUnchecked<MapVector>();
  for (auto i = 0; i < length; ++i) {
    auto index = i + offset;
    auto wrappedIndex = source.wrappedIndex(index);

    if (!map->isNullAt(wrappedIndex)) {
      auto kv_array_offset = map->offsetAt(wrappedIndex);
      auto kv_array_size = map->sizeAt(wrappedIndex);

      MaxSizeForStatsEstimator::instance().estimateSizeOfVectorElements(
          *map->mapKeys(), kv_array_offset, kv_array_size, size_out);
      MaxSizeForStatsEstimator::instance().estimateSizeOfVectorElements(
          *map->mapValues(), kv_array_offset, kv_array_size, size_out);
    }
  }
}

void estimateSizeOfRowElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) {
  VELOX_CHECK(offset >= 0 && offset + length <= source.size());
  auto row = source.wrappedVector()->asUnchecked<RowVector>();
  const auto& type = row->type()->as<TypeKind::ROW>();
  auto childrenSize = type.size();
  auto children = row->children();

  for (auto child_index = 0; child_index < childrenSize; ++child_index) {
    if (!children[child_index] || child_index >= children.size()) {
      continue;
    }
    MaxSizeForStatsEstimator::instance().estimateSizeOfVectorElements(
        *children[child_index], offset, length, size_out);
  }
}

void estimateSizeOfVarcharOrVarbinaryElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) {
  VELOX_CHECK(offset >= 0 && offset + length <= source.size());
  auto array = source.asUnchecked<SimpleVector<StringView>>();
  for (auto i = 0; i < length; i++) {
    if (!array->isNullAt(i + offset)) {
      size_out += array->valueAt(i + offset).size();
    }
  }
}

template <TypeKind Kind>
void estimateSizeOfScalarElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) {
  VELOX_CHECK(offset >= 0 && offset + length <= source.size());
  using T = typename TypeTraits<Kind>::NativeType;
  const size_t native_size = sizeof(typename TypeTraits<Kind>::NativeType);
  size_t value_size = 0;

  // timestamp is special in that it's native size does not match the size that
  // presto should report, thus, special case it.
  switch (TypeTraits<Kind>::typeKind) {
    case TypeKind::TIMESTAMP:
      value_size = 8;
      break;
    default:
      value_size = native_size;
  }

  for (auto i = 0; i < length; ++i) {
    if (!source.isNullAt(i + offset)) {
      size_out += value_size;
    }
  }
}
} // namespace

const MaxSizeForStatsEstimator& MaxSizeForStatsEstimator::instance() {
  static auto instance = std::make_unique<MaxSizeForStatsEstimator>();
  return *instance;
}

void MaxSizeForStatsEstimator::estimateSizeOfVectorElements(
    const BaseVector& source,
    vector_size_t offset,
    vector_size_t length,
    size_t& size_out) const {
  switch (source.type()->kind()) {
    case TypeKind::TINYINT:
      return estimateSizeOfScalarElements<TypeKind::TINYINT>(
          source, offset, length, size_out);
    case TypeKind::SMALLINT:
      return estimateSizeOfScalarElements<TypeKind::SMALLINT>(
          source, offset, length, size_out);
    case TypeKind::INTEGER:
      return estimateSizeOfScalarElements<TypeKind::INTEGER>(
          source, offset, length, size_out);
    case TypeKind::BIGINT:
      return estimateSizeOfScalarElements<TypeKind::BIGINT>(
          source, offset, length, size_out);
    case TypeKind::DOUBLE:
      return estimateSizeOfScalarElements<TypeKind::DOUBLE>(
          source, offset, length, size_out);
    case TypeKind::REAL:
      return estimateSizeOfScalarElements<TypeKind::REAL>(
          source, offset, length, size_out);
    case TypeKind::BOOLEAN:
      return estimateSizeOfScalarElements<TypeKind::BOOLEAN>(
          source, offset, length, size_out);
    case TypeKind::DATE:
      return estimateSizeOfScalarElements<TypeKind::DATE>(
          source, offset, length, size_out);
    case TypeKind::TIMESTAMP:
      return estimateSizeOfScalarElements<TypeKind::TIMESTAMP>(
          source, offset, length, size_out);
    case TypeKind::ROW:
      return estimateSizeOfRowElements(source, offset, length, size_out);
    case TypeKind::MAP:
      return estimateSizeOfMapElements(source, offset, length, size_out);
    case TypeKind::ARRAY:
      return estimateSizeOfArrayElements(source, offset, length, size_out);
    case TypeKind::VARBINARY:
    case TypeKind::VARCHAR:
      return estimateSizeOfVarcharOrVarbinaryElements(
          source, offset, length, size_out);
    default:
      VELOX_FAIL(
          "Unknown input type for {} estimateSizeOfVectorElements",
          source.type()->kindName());
  }
}
} // namespace facebook::velox::exec