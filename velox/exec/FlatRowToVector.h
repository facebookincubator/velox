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

#include <cstring>

#include "velox/common/memory/ByteStream.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/RowContainer.h"
#include "velox/type/Type.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {

/// Extracts one column from deserialized flat rows into 'result'. Returns
/// the total out-of-line byte size of the column's variable-length payload
/// across the extracted rows (0 for fixed-width columns).
template <TypeKind Kind>
FOLLY_ALWAYS_INLINE uint64_t extractFlatColumn(
    const char* const* flatRows,
    int32_t numRows,
    RowColumn column,
    const TypePtr& type,
    const VectorPtr& result,
    vector_size_t resultOffset) {
  if constexpr (
      Kind == TypeKind::ARRAY || Kind == TypeKind::MAP ||
      Kind == TypeKind::ROW) {
    // Out-of-line bytes are contiguous after deserializeRow.
    const auto nullByte = column.nullByte();
    const auto nullMask = column.nullMask();
    const auto offset = column.offset();
    VELOX_DCHECK_LE(numRows + resultOffset, result->size());
    uint64_t totalOutOfLineBytes{0};
    for (int32_t i = 0; i < numRows; ++i) {
      const vector_size_t outIdx = resultOffset + i;
      const char* flatRow = flatRows[i];
      if (flatRow == nullptr ||
          RowContainer::isNullAt(flatRow, nullByte, nullMask)) {
        result->setNull(outIdx, true);
      } else {
        const auto& sv =
            *reinterpret_cast<const std::string_view*>(flatRow + offset);
        totalOutOfLineBytes += sv.size();
        auto* bytes = reinterpret_cast<uint8_t*>(const_cast<char*>(sv.data()));
        BufferInputStream stream(
            {ByteRange{bytes, static_cast<int32_t>(sv.size()), 0}});
        ContainerRowSerde::deserialize(stream, outIdx, result.get());
      }
    }
    return totalOutOfLineBytes;
  } else if constexpr (Kind == TypeKind::OPAQUE) {
    VELOX_NYI(
        "Type is not supported for row-based spill: {}",
        TypeKindName::toName(Kind));
  } else if constexpr (
      Kind == TypeKind::VARCHAR || Kind == TypeKind::VARBINARY) {
    const auto nullByte = column.nullByte();
    const auto nullMask = column.nullMask();
    const auto offset = column.offset();
    auto* flatResult = result->as<FlatVector<StringView>>();
    // Pre-allocate the string buffer in one pass to avoid reallocations.
    size_t totalOutOfLineBytes = 0;
    for (int32_t i = 0; i < numRows; ++i) {
      const char* flatRow = flatRows[i];
      if (flatRow == nullptr ||
          RowContainer::isNullAt(flatRow, nullByte, nullMask)) {
        continue;
      }
      const auto& sv = *reinterpret_cast<const StringView*>(flatRow + offset);
      if (!sv.isInline()) {
        totalOutOfLineBytes += sv.size();
      }
    }
    char* bufPos = totalOutOfLineBytes > 0
        ? flatResult->getRawStringBufferWithSpace(totalOutOfLineBytes, true)
        : nullptr;
    // Second pass: fill values from the contiguous out-of-line region.
    for (int32_t i = 0; i < numRows; ++i) {
      const vector_size_t outIdx = resultOffset + i;
      const char* flatRow = flatRows[i];
      if (flatRow == nullptr ||
          RowContainer::isNullAt(flatRow, nullByte, nullMask)) {
        flatResult->setNull(outIdx, true);
        continue;
      }
      const auto& sv = *reinterpret_cast<const StringView*>(flatRow + offset);
      if (sv.isInline()) {
        flatResult->set(outIdx, sv);
      } else {
        std::memcpy(bufPos, sv.data(), sv.size());
        flatResult->setNoCopy(outIdx, StringView(bufPos, sv.size()));
        bufPos += sv.size();
      }
    }
    return totalOutOfLineBytes;
  } else {
    // Fixed-width: same layout in live rows and deserialized flat rows;
    // delegate to RowContainer.
    RowContainer::extractColumn(
        flatRows,
        numRows,
        column,
        /*columnHasNulls=*/true,
        resultOffset,
        result);
    return 0;
  }
}

/// Extracts 'col' from numRows deserialized flat rows into
/// result[0..numRows). Returns the total out-of-line byte size of the
/// column's variable-length payload (0 for fixed-width columns).
FOLLY_ALWAYS_INLINE uint64_t extractColumnFromFlatRow(
    const char* const* flatRows,
    int32_t numRows,
    RowColumn col,
    const TypePtr& type,
    const VectorPtr& result) {
  result->resize(numRows);
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractFlatColumn,
      type->kind(),
      flatRows,
      numRows,
      col,
      type,
      result,
      0);
}

/// Same as above but writes into 'result' starting at 'resultOffset'.
FOLLY_ALWAYS_INLINE uint64_t extractColumnFromFlatRow(
    const char* const* flatRows,
    int32_t numRows,
    RowColumn col,
    const TypePtr& type,
    const VectorPtr& result,
    vector_size_t resultOffset) {
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractFlatColumn,
      type->kind(),
      flatRows,
      numRows,
      col,
      type,
      result,
      resultOffset);
}

} // namespace facebook::velox::exec
