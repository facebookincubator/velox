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

#include "folly/CPortability.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

/// Flat-row format:
/// [fixedArea][out-of-line col0][out-of-line col1]...[padding]
/// On write (serializeRow): the fixed area is copied verbatim and each
/// variable-width column's out-of-line bytes are appended in column order.
/// VARCHAR/VARBINARY data pointers are zeroed in the serialized copy to
/// improve compression; for ARRAY/MAP/ROW the whole string_view field is
/// zeroed and the out-of-line byte count is written at offset 0.
/// On read (deserializeRow): pointers are patched back in place to reference
/// the out-of-line region, so the row can be used directly without
/// rebuilding a RowVector.

/// Patches StringView/string_view pointers in-place to point into the
/// out-of-line region following the fixed area. Returns the aligned row size
/// and advances 'current' past the row. Pointers are only valid while the
/// owning buffer is alive.
FOLLY_ALWAYS_INLINE uint32_t
deserializeRow(char*& current, const RowContainerLayout& layout) {
  VELOX_DCHECK_EQ(
      reinterpret_cast<std::uintptr_t>(current) % layout.alignment, 0);

  char* varBufferStart = current + layout.serializedFixRowSize;

  for (const auto& vc : layout.variableColumns) {
    if (RowContainer::isNullAt(current, vc.column)) {
      continue;
    }
    if (vc.kind == TypeKind::VARCHAR || vc.kind == TypeKind::VARBINARY) {
      StringView& sv =
          *reinterpret_cast<StringView*>(current + vc.column.offset());
      if (!sv.isInline()) {
        sv = StringView(varBufferStart, sv.size());
        varBufferStart += sv.size();
      }
    } else {
      // The slot holds the out-of-line byte count written by serializeRow;
      // the field is not a valid string_view until patched so read size_t
      // directly.
      const size_t outOfLineSize =
          *reinterpret_cast<const size_t*>(current + vc.column.offset());
      *reinterpret_cast<std::string_view*>(current + vc.column.offset()) =
          std::string_view(varBufferStart, outOfLineSize);
      varBufferStart += outOfLineSize;
    }
  }

  // Derive the row size from the varBufferStart distance;
  // layout.serializedSize() is unreliable after StringView pointers are
  // patched.
  const auto rawSize = static_cast<uint32_t>(varBufferStart - current);
  const uint32_t rowSize = bits::roundUp(rawSize, layout.alignment);
  current += rowSize;
  return rowSize;
}

/// Copies the fixed area and appends the out-of-line bytes of each variable
/// column into [current, bufferEnd). Zeros pointers in the serialized copy to
/// improve compression. Advances 'current' past the serialized row and
/// returns the aligned serialized row size.
FOLLY_ALWAYS_INLINE uint32_t serializeRow(
    char* row,
    char*& current,
    const char* bufferEnd,
    const RowContainerLayout& layout) {
  char* rowStart = current;
  simd::memcpy(current, row, layout.serializedFixRowSize);
  current += layout.serializedFixRowSize;

  for (const auto& vc : layout.variableColumns) {
    if (RowContainer::isNullAt(row, vc.column)) {
      continue;
    }
    if (vc.kind == TypeKind::VARCHAR || vc.kind == TypeKind::VARBINARY) {
      const StringView& sv =
          *reinterpret_cast<const StringView*>(row + vc.column.offset());
      if (!sv.isInline()) {
        const auto* header = HashStringAllocator::headerOf(sv.data());
        if (header->size() >= sv.size()) {
          simd::memcpy(current, sv.data(), sv.size());
        } else {
          auto stream = HashStringAllocator::InputStream(header);
          stream.readBytes(reinterpret_cast<uint8_t*>(current), sv.size());
        }
        // Zero the data pointer to improve compression.
        *reinterpret_cast<char**>(
            rowStart + vc.column.offset() + sizeof(uint32_t) * 2) = nullptr;
        current += sv.size();
      }
    } else {
      const auto& value =
          *reinterpret_cast<const std::string_view*>(row + vc.column.offset());
      const size_t outOfLineSize = value.size();
      auto stream = HashStringAllocator::InputStream(
          HashStringAllocator::headerOf(value.data()));
      stream.readBytes(reinterpret_cast<uint8_t*>(current), outOfLineSize);
      current += outOfLineSize;
      // Zero the full string_view field, then store the out-of-line byte
      // count at offset 0, so this is independent of std::string_view's
      // layout.
      std::memset(rowStart + vc.column.offset(), 0, sizeof(std::string_view));
      *reinterpret_cast<size_t*>(rowStart + vc.column.offset()) = outOfLineSize;
    }
  }

  const auto rawSize = current - rowStart;
  current += bits::roundUp(rawSize, layout.alignment) - rawSize;
  VELOX_DCHECK_LE(
      static_cast<const void*>(current),
      static_cast<const void*>(bufferEnd),
      "Row serialization overran write buffer");
  return static_cast<uint32_t>(current - rowStart);
}

} // namespace facebook::velox::exec
