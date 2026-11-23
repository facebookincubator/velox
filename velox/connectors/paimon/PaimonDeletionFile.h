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

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <cstdint>
#include <optional>
#include <string>

namespace facebook::velox::connector::paimon {

/// Paimon deletion file — a serialized Roaring Bitmap tracking deleted row
/// positions within an existing data file. Primary-key and append-only tables
/// both attach them to data files when deletion-vectors.enabled=true.
///
/// # How Deletion Vectors Work
///
/// When a primary-key table has deletion vectors enabled, an UPDATE or DELETE
/// does not rewrite the original data file. Instead:
///   1. The new/updated record is written to a new L0 data file (as usual).
///   2. A deletion bitmap is written marking the old row's position in the
///      original data file (0-based row index).
///   3. The reader loads the bitmap and skips those positions during scanning.
///
/// This improves read performance: without deletion vectors, the reader must
/// do merge-on-read to discover that a key in an older file has been superseded
/// by a newer file. With deletion vectors, the reader skips deleted rows
/// immediately — no merge needed. A split with deletion vectors can be
/// rawConvertible=true because the bitmaps already tell the reader which rows
/// to skip (old and new values don't overlap from the reader's perspective).
///
/// Similar to Iceberg's positional delete files.
///
/// # Manifest Integration
///
/// When deletion vectors are written, Paimon creates a new snapshot with
/// updated manifest entries for the affected data file:
///   REMOVE: old manifest entry (without or with old deletion file)
///   ADD:    same data file, now with deletion file reference attached
/// The data file itself is NOT rewritten — only the manifest entry changes.
///
/// Each data file has at most ONE deletion file at any point in time. If more
/// rows are deleted later, the bitmap is replaced with a new one containing
/// all deleted positions (old + new, merged). The old bitmap becomes obsolete.
///
/// Deletion files can be attached to data files at ANY LSM level (L0, L1,
/// L2, etc.), not just L0.
///
/// # File Format
///
/// Deletion files are NOT stored in a columnar format (not Parquet/ORC/Nimble)
/// and not in a row-based format (not compact row). A deletion file is a
/// Paimon index file: a version byte followed by the deletion vectors of
/// several data files. The manifest, not the index file, maps each data file
/// to the byte range of its vector. Each vector is framed as
///
///   [length][magic][bitmap][CRC]
///
/// where the length and the CRC are big-endian int32s and the length counts
/// the magic and the bitmap. Paimon writes two frame versions:
///   - V1: big-endian magic 1581511376 and a 32-bit RoaringBitmap in the
///     portable RoaringFormatSpec serialization. 'length' equals the frame
///     length.
///   - V2: little-endian magic 1681511377 and a Roaring64 bitmap, the Iceberg
///     deletion vector layout. 'length' also counts the frame length field and
///     the CRC, so it is the frame length plus 8.
///
/// Each bitmap contains 0-based row positions of deleted rows within its
/// associated data file. 'offset' is the position of the frame's length field
/// within the index file.
///
/// # Relationship to RowKind-Based Deletes
///
/// Deletion files are orthogonal to RowKind-based deletes:
///
///   RowKind=-D: A full record written to a NEW data file at level 0,
///     containing the key + all column values tagged with -D. Used for logical
///     deletes (CDC source, SQL DELETE). Compaction resolves it by removing the
///     key entirely.
///
///   Deletion file (this struct): A bitmap referencing row positions in an
///     EXISTING data file. No new data record written. Used as a physical
///     optimization to avoid rewriting files during compaction or partial
///     deletes.
///
/// For batch reads, the reader loads deletion bitmaps and filters out deleted
/// row positions during scanning.
struct PaimonDeletionFile {
  /// @param path Path to the deletion file.
  /// @param offset Byte offset of the deletion vector frame in the file.
  /// @param length Deletion vector length as Paimon records it, which depends
  ///        on the frame version (must be > 0).
  /// @param cardinality Number of deleted rows (must be > 0), or std::nullopt
  ///        if Paimon did not record it.
  PaimonDeletionFile(
      std::string path,
      uint64_t offset,
      uint64_t length,
      std::optional<uint64_t> cardinality);

  std::string path;

  // Byte offset of the deletion vector frame within the deletion file.
  uint64_t offset;

  // Deletion vector length as Paimon records it: the frame length for V1
  // frames, and the frame length plus 8 for V2 frames. Must be > 0.
  uint64_t length;

  // Number of deleted rows (pre-computed for stats without reading bitmap).
  // Must be > 0. Paimon 0.9 does not record it.
  std::optional<uint64_t> cardinality;

  std::string toString() const;
  folly::dynamic serialize() const;
  static PaimonDeletionFile create(const folly::dynamic& obj);
};

} // namespace facebook::velox::connector::paimon
