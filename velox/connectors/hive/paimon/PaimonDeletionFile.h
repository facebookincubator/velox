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
#include <string>

namespace facebook::velox::connector::hive::paimon {

/// Paimon deletion file — a serialized Roaring Bitmap tracking deleted row
/// positions within an existing data file. Only used for primary-key tables
/// with deletion-vectors.enabled=true. Append-only tables have no concept of
/// row-level deletes (no primary key to identify specific rows).
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
/// and not in a row-based format (not compact row). They use a custom binary
/// format:
///   - Core payload: standard RoaringBitmap portable binary serialization
///     (cross-language spec from RoaringFormatSpec). C++ can read this with
///     the CRoaring library; Java uses org.roaringbitmap.RoaringBitmap.
///   - Container packing: when multiple bitmaps are packed into a single file,
///     Paimon uses a simple custom binary layout with length-prefixed entries
///     mapping data file names to their bitmap bytes.
///
/// Each bitmap contains 0-based row positions of deleted rows within its
/// associated data file.
///
/// Multiple deletion bitmaps can be packed into a single container file.
/// 'offset' and 'length' specify the byte range within the container where
/// this bitmap's data lives. For standalone deletion files, offset is 0 and
/// length equals the file size.
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
  /// @param offset Byte offset within the container file.
  /// @param length Number of bytes of bitmap data (must be > 0).
  /// @param cardinality Number of deleted rows (must be > 0).
  PaimonDeletionFile(
      std::string path,
      uint64_t offset,
      uint64_t length,
      uint64_t cardinality);

  std::string path;

  // Byte offset within the container file where this bitmap starts.
  uint64_t offset;

  // Number of bytes of bitmap data. Must be > 0.
  uint64_t length;

  // Number of deleted rows (pre-computed for stats without reading bitmap).
  // Must be > 0.
  uint64_t cardinality;

  std::string toString() const;
  folly::dynamic serialize() const;
  static PaimonDeletionFile create(const folly::dynamic& obj);
};

} // namespace facebook::velox::connector::hive::paimon
