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
#include <map>
#include <string>
#include <vector>

namespace facebook::velox::connector::hive::iceberg {

/// Writes Iceberg V3 deletion vectors as serialized 64-bit roaring bitmaps.
///
/// Deletion vectors are an Iceberg V3 feature (format version 3). The roaring
/// bitmap encoding itself is version-independent, but the Puffin blob type
/// "deletion-vector-v1" is defined by the Iceberg V3 spec.
///
/// DVs are compact roaring bitmaps stored as blobs inside Puffin files to mark
/// deleted rows in data files. This writer collects deleted row positions and
/// serializes them in the Roaring64Bitmap portable format, compatible with
/// Java's Roaring64Bitmap used by the Presto coordinator.
///
/// The 64-bit format partitions positions by their upper 32 bits into groups,
/// each group containing a standard 32-bit RoaringBitmap for the lower 32
/// bits. This supports files with more than 4 billion rows.
///
/// Usage:
///   DeletionVectorWriter writer;
///   writer.addDeletedPosition(5);
///   writer.addDeletedPosition(5'000'000'000LL);
///   std::string blob = writer.serialize();
class DeletionVectorWriter {
 public:
  DeletionVectorWriter() = default;

  /// Adds a deleted row position (0-based file row offset).
  void addDeletedPosition(int64_t position);

  /// Adds multiple deleted row positions.
  void addDeletedPositions(const std::vector<int64_t>& positions);

  /// Returns the number of deleted positions collected so far.
  size_t numPositions() const {
    return positions_.size();
  }

  /// Serializes collected positions into a 64-bit roaring bitmap.
  ///
  /// Format: Roaring64Bitmap portable serialization —
  ///   [numGroups: uint64]
  ///   For each group (sorted by highBits):
  ///     [highBits: uint32]
  ///     [32-bit RoaringBitmap in portable format]
  ///
  /// Each 32-bit RoaringBitmap uses SERIAL_COOKIE_NO_RUNCONTAINER (12346)
  /// with array containers (cardinality <= 4096) or bitmap containers.
  ///
  /// @return Binary string containing the serialized 64-bit roaring bitmap.
  std::string serialize() const;

  /// Clears all collected positions.
  void clear();

 private:
  /// Serializes a single 32-bit roaring bitmap from sorted, deduplicated
  /// values in the range [0, 2^32).
  std::string serialize32(const std::vector<uint32_t>& sorted) const;

  std::vector<int64_t> positions_;
};

/// Writes a Puffin file containing a single deletion vector blob.
///
/// The Puffin file format consists of:
///   - 4-byte magic: "PUF1"
///   - Blob data (the serialized roaring bitmap)
///   - Footer: blob metadata + footer payload size + magic
///
/// @param filePath Path to write the Puffin file.
/// @param blobData Serialized roaring bitmap bytes.
/// @param referencedDataFile Path of the data file this DV applies to.
/// @return Pair of (blobOffset, blobLength) within the written file.
std::pair<uint64_t, uint64_t> writePuffinFile(
    const std::string& filePath,
    const std::string& blobData,
    const std::string& referencedDataFile);

} // namespace facebook::velox::connector::hive::iceberg
