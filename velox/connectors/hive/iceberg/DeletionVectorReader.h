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
/// Iceberg V3 replaces positional delete files with deletion vectors — compact
/// roaring bitmaps stored as blobs inside Puffin files. Each DV is associated
/// with a single base data file and contains 0-based row positions of deleted
/// rows.
///
/// Compared to the V2 PositionalDeleteFileReader:
///   - No columnar file parsing (Puffin blobs are raw binary, not Parquet)
///   - No file_path filtering (each DV is pre-associated with its data file)
///   - No sorted merge of multiple delete files (one DV per data file)
///   - Direct bitmap-to-bitmap conversion instead of row-by-row position
///   reading
///
/// The coordinator extracts the blob offset and length from the Puffin footer
/// and provides them via the IcebergDeleteFile metadata. The reader opens the
/// file, reads the raw bytes at the given offset, deserializes the roaring
/// bitmap, and sets bits in the delete bitmap for the current batch range.
class DeletionVectorReader {
 public:
  /// Constructs a reader for a single deletion vector.
  ///
  /// @param dvFile Metadata about the deletion vector file. Must have
  ///   content == FileContent::kDeletionVector. The filePath points to the
  ///   Puffin file, and the DV blob offset/length are encoded in the
  ///   lowerBounds/upperBounds fields (key = kDvOffsetFieldId for offset,
  ///   kDvLengthFieldId for length).
  /// @param splitOffset File position of the first row in the split.
  /// @param pool Memory pool for bitmap allocations.
  /// @param fileSystem Filesystem to read the Puffin file from.
  DeletionVectorReader(
      const IcebergDeleteFile& dvFile,
      uint64_t splitOffset,
      memory::MemoryPool* pool);

  /// Reads deleted positions from the DV and sets corresponding bits in the
  /// deleteBitmap for the current batch range.
  ///
  /// @param baseReadOffset Read offset from the beginning of the split in
  ///   number of rows for the current batch.
  /// @param size Number of rows in the current batch.
  /// @param deleteBitmap Output bitmap. Bit i is set if the row at file
  ///   position (splitOffset + baseReadOffset + i) is deleted.
  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      BufferPtr deleteBitmap);

  /// Returns true when there is no more data. For DVs this is always true
  /// after the first readDeletePositions() call since the entire bitmap is
  /// loaded eagerly.
  bool noMoreData() const;

  /// Field IDs used to encode DV blob offset and length in the
  /// IcebergDeleteFile bounds maps. The coordinator encodes these when
  /// building splits from Puffin file metadata.
  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  /// Loads the deletion vector bitmap from the Puffin file. Called lazily
  /// on the first readDeletePositions() call.
  void loadBitmap();

  /// Parses a roaring bitmap from its portable binary serialization format
  /// and populates deletedPositions_ with all set positions.
  void deserializeRoaringBitmap(const std::string& data);

  const IcebergDeleteFile dvFile_;
  const uint64_t splitOffset_;
  memory::MemoryPool* const pool_;

  /// Sorted vector of deleted row positions (0-based, relative to the
  /// start of the base data file). Populated by loadBitmap().
  std::vector<int64_t> deletedPositions_;

  /// Index into deletedPositions_ tracking how far we've consumed.
  size_t positionIndex_{0};

  /// Whether the bitmap has been loaded from the file.
  bool loaded_{false};
};

} // namespace facebook::velox::connector::hive::iceberg
