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

#include "velox/common/base/CompareFlags.h"
#include "velox/common/memory/ByteStream.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/RowContainer.h"
#include "velox/type/Type.h"
#include "velox/vector/SimpleVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {

/// Compares ARRAY/MAP/ROW columns between two deserialized spill rows.
/// Both values must be non-null; null handling is done by compareColumn.
inline int compareComplexType(
    const char* left,
    const char* right,
    const Type* type,
    RowColumn column,
    CompareFlags flags) {
  // out-of-line bytes are flat and contiguous after deserializeRow.
  const auto& leftView =
      *reinterpret_cast<const std::string_view*>(left + column.offset());
  const auto& rightView =
      *reinterpret_cast<const std::string_view*>(right + column.offset());
  auto* leftBytes =
      reinterpret_cast<uint8_t*>(const_cast<char*>(leftView.data()));
  auto* rightBytes =
      reinterpret_cast<uint8_t*>(const_cast<char*>(rightView.data()));
  BufferInputStream leftStream(
      {ByteRange{leftBytes, static_cast<int32_t>(leftView.size()), 0}});
  BufferInputStream rightStream(
      {ByteRange{rightBytes, static_cast<int32_t>(rightView.size()), 0}});
  return ContainerRowSerde::compare(leftStream, rightStream, type, flags);
}

template <TypeKind Kind>
FOLLY_ALWAYS_INLINE int compareColumn(
    const char* left,
    const char* right,
    RowColumn column,
    const Type* type,
    CompareFlags flags) {
  using T = typename KindToFlatVector<Kind>::HashRowType;

  const bool leftIsNull = RowContainer::isNullAt(left, column);
  const bool rightIsNull = RowContainer::isNullAt(right, column);
  if (leftIsNull) {
    return rightIsNull ? 0 : flags.nullsFirst ? -1 : 1;
  }
  if (rightIsNull) {
    return flags.nullsFirst ? 1 : -1;
  }

  if constexpr (
      Kind == TypeKind::ARRAY || Kind == TypeKind::MAP ||
      Kind == TypeKind::ROW) {
    return compareComplexType(left, right, type, column, flags);
  } else if constexpr (
      Kind == TypeKind::VARCHAR || Kind == TypeKind::VARBINARY) {
    const auto& leftStr =
        *reinterpret_cast<const StringView*>(left + column.offset());
    const auto& rightStr =
        *reinterpret_cast<const StringView*>(right + column.offset());
    const int result = leftStr.compare(rightStr);
    return flags.ascending ? result : -result;
  } else if constexpr (Kind == TypeKind::OPAQUE) {
    VELOX_NYI("OPAQUE type is not supported for row-based spill");
  } else {
    // comparePrimitiveAsc aligns with RowContainer::compare.
    const T leftValue = *reinterpret_cast<const T*>(left + column.offset());
    const T rightValue = *reinterpret_cast<const T*>(right + column.offset());
    const int result =
        SimpleVector<T>::comparePrimitiveAsc(leftValue, rightValue);
    return flags.ascending ? result : -result;
  }
}

/// Compares two deserialized spill rows on one sort column.
FOLLY_ALWAYS_INLINE int32_t compareRow(
    const char* left,
    const char* right,
    RowColumn column,
    const TypePtr& type,
    CompareFlags flags) {
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      compareColumn, type->kind(), left, right, column, type.get(), flags);
}

} // namespace facebook::velox::exec
