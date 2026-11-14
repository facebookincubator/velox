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

#include <cstdint>
#include <string>
#include <vector>

#include "velox/connectors/paimon/PaimonDeletionFile.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::connector::paimon {

/// Reads a Paimon deletion vector and applies it to a DWIO delete bitmap.
///
/// Paimon stores deletion vectors as framed Roaring bitmaps in index files,
/// and the data-file metadata gives the byte range of each vector. Both frame
/// versions Paimon Java writes are supported; PaimonDeletionFile describes
/// them. The positions are 0-based row offsets within the data file,
/// independent of the physical file format used for the data rows.
class PaimonDeletionVectorReader {
 public:
  /// @param deletionFile Paimon deletion-file metadata containing the file
  ///        path, byte offset, byte length, and cardinality if recorded.
  /// @param splitOffset File position of the first row in the split.
  explicit PaimonDeletionVectorReader(
      const PaimonDeletionFile& deletionFile,
      uint64_t splitOffset = 0);

  /// Reads deleted positions for the current batch range and sets matching
  /// bits in deleteBitmap.
  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      BufferPtr deleteBitmap);

  /// Returns true when all deleted positions have been consumed.
  bool noMoreData() const;

 private:
  // Reads the frame of the deletion vector, checks its length against the
  // deletion file and expands the bitmap into sorted positions.
  void loadBitmap();

  // Appends the positions of a Roaring64 bitmap in [ptr, end).
  void deserializeRoaring64Bitmap(const uint8_t* ptr, const uint8_t* end);

  // Appends the positions of the 32-bit RoaringBitmap at ptr, each combined
  // with highBitsOffset, and returns the end of that bitmap.
  const uint8_t* deserialize32BitRoaringBitmap(
      const uint8_t* ptr,
      const uint8_t* end,
      int64_t highBitsOffset);

  const PaimonDeletionFile deletionFile_;
  const uint64_t splitOffset_;
  std::vector<int64_t> deletedPositions_;
  size_t positionIndex_{0};
  bool loaded_{false};
};

} // namespace facebook::velox::connector::paimon
