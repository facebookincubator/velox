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

#include <memory>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

namespace facebook::velox::connector::hive::iceberg {

/// Reads an Iceberg V3 deletion vector and applies it to the delete bitmap.
///
/// Supports both 64-bit Roaring64Bitmap format (used by Java's Roaring64Bitmap
/// at Meta's 500 PB scale) and legacy 32-bit RoaringBitmap format.
///
/// 64-bit format: [numGroups: uint64] then for each group:
///   [highBits: uint32] [32-bit RoaringBitmap in portable format]
///
/// 32-bit format: [cookie: uint32] [containerCount: uint32] ...
///   Detected by checking if the first 8 bytes match a 32-bit cookie (12346
///   or 12347).
class DeletionVectorReader {
 public:
  /// @param dvFile Iceberg delete file metadata containing the Puffin file
  /// path,
  ///        blob offset, and blob length.
  /// @param splitOffset File position of the first row in the split.
  /// @param pool Memory pool for internal allocations.
  DeletionVectorReader(
      const IcebergDeleteFile& dvFile,
      uint64_t splitOffset,
      memory::MemoryPool* pool);

  /// Reads deleted positions from the DV and sets corresponding bits in the
  /// deleteBitmap for the current batch range.
  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      BufferPtr deleteBitmap);

  /// Returns true when there is no more data.
  bool noMoreData() const;

  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  void loadBitmap();

  // Deserializes a 64-bit roaring bitmap (Roaring64Bitmap format).
  void deserializeRoaring64Bitmap(const std::string& data);

  // Deserializes a 32-bit roaring bitmap from portable binary format.
  // Positions are offset by highBitsOffset (upper 32 bits for 64-bit mode,
  // 0 for legacy 32-bit mode).
  void deserialize32BitRoaringBitmap(
      const uint8_t* ptr,
      const uint8_t* end,
      int64_t highBitsOffset);

  // The deletion vector file metadata from the Iceberg manifest.
  const IcebergDeleteFile dvFile_;

  // Base offset of the split within the data file (for position mapping).
  const uint64_t splitOffset_;

  // Sorted list of deleted row positions (absolute, file-level positions).
  std::vector<int64_t> deletedPositions_;

  // Current scan position within deletedPositions_.
  size_t positionIndex_{0};

  // Whether the bitmap has been loaded from the DV file.
  bool loaded_{false};
};

} // namespace facebook::velox::connector::hive::iceberg
