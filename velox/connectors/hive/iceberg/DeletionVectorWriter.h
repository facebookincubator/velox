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

/// Writes Iceberg V3 deletion vectors as serialized roaring bitmaps.
///
/// Iceberg V3 uses deletion vectors (DVs) — compact roaring bitmaps stored
/// as blobs inside Puffin files — to mark deleted rows in data files. This
/// writer collects deleted row positions and serializes them in the standard
/// roaring bitmap portable format for consumption by DeletionVectorReader.
///
/// Usage:
///   DeletionVectorWriter writer;
///   writer.addDeletedPosition(5);
///   writer.addDeletedPosition(10);
///   std::string blob = writer.serialize();
///   // Write blob to a Puffin file at the appropriate offset.
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

  /// Serializes the collected positions into a roaring bitmap in the
  /// portable binary format (SERIAL_COOKIE_NO_RUNCONTAINER, cookie = 12346).
  ///
  /// Uses array containers for cardinality <= 4096 and bitmap containers
  /// for cardinality > 4096, matching the standard roaring bitmap spec.
  ///
  /// @return Binary string containing the serialized roaring bitmap.
  std::string serialize() const;

  /// Clears all collected positions.
  void clear();

 private:
  std::vector<int64_t> positions_;
};

/// Writes a Puffin file containing a single deletion vector blob.
///
/// The Puffin file format consists of:
///   - 4-byte magic: "PUF1"
///   - Blob data (the serialized roaring bitmap)
///   - Footer: blob metadata + footer payload size + magic
///
/// This is a simplified writer that produces files compatible with the
/// Iceberg Puffin spec for single-blob deletion vectors.
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
