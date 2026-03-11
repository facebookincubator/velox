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
#include <ostream>
#include <string>

#include <folly/CPortability.h>

#include "velox/connectors/hive/paimon/PaimonDeletionFile.h"

namespace facebook::velox::connector::hive::paimon {

/// Represents a single Paimon file (data or changelog) within a split.
/// Mirrors Apache Paimon's DataFileMeta structure.
///
/// Both data files and changelog files use the same physical file format
/// (Parquet, ORC, Nimble) and carry the same metadata. The distinction is
/// captured by `type`:
///   - kData: regular data file (current state of records).
///   - kChangelog: changelog file with RowKind (+I/-U/+U/-D) for streaming.
struct PaimonDataFile {
  /// Whether this file is a data file or a changelog file. Both use the
  /// same physical format — this enum distinguishes their semantic role.
  ///
  /// Data files contain the current state of records. Changelog files
  /// contain the change history with RowKind tags. The coordinator sends
  /// data files for batch reads and changelog files for streaming reads
  /// (when changelog-producer=lookup is configured).
  enum class Type {
    /// Regular data file containing current record state.
    kData,

    /// Changelog file containing records tagged with RowKind (+I/-U/+U/-D).
    /// Only produced when changelog-producer=lookup is configured on the table.
    /// The coordinator sends these for streaming reads instead of data files.
    kChangelog,
  };

  /// Origin of a Paimon file — how it was produced, not what it contains.
  /// Mirrors Paimon's DataFileMeta.FileSource enum.
  ///
  /// This does NOT indicate whether the file contains `_rowkind`. The presence
  /// of `_rowkind` depends on the changelog mode (upsert vs input changelog),
  /// not the source type.
  enum class Source {
    /// File produced by a normal write (flush/append).
    kAppend,

    /// File produced by compaction. This could be:
    ///   - Without changelog-producer=lookup: a regular data file (same as
    ///     kAppend but produced by compaction instead of a write).
    ///   - With changelog-producer=lookup: compaction produces TWO files —
    ///     (1) a data file (no `_rowkind` in upsert mode) and (2) a separate
    ///     changelog file (with `_rowkind` containing correct +I/-U/+U/-D).
    ///     Both are marked kCompact. The coordinator sends the right one based
    ///     on the read mode (data file for batch, changelog file for
    ///     streaming).
    kCompact,
  };

  /// Returns the string name of the file type (e.g., "DATA").
  static std::string typeString(Type type);

  /// Parses a file type from its string name.
  static Type typeFromString(const std::string& str);

  /// Returns the string name of the source type (e.g., "APPEND").
  static std::string sourceString(Source source);

  /// Parses a source type from its string name.
  static Source sourceFromString(const std::string& str);

  /// Path to the file (ORC, Parquet, etc.).
  std::string path;

  /// Size of the file in bytes.
  uint64_t size{0};

  /// Number of rows in this file.
  uint64_t rowCount{0};

  /// LSM-tree level of this file. Level 0 contains the newest (unflushed)
  /// data; higher levels contain progressively more compacted data. Within
  /// level 0, files CAN have overlapping keys. Within level 1+, compaction
  /// guarantees non-overlapping key ranges across files.
  /// Always 0 for append-only tables (no compaction).
  int32_t level{0};

  /// Sequence number range of records in this file. Auto-generated per-commit,
  /// monotonically increasing. For level 0 files, min == max (single commit).
  /// For compacted files (level 1+), min < max (merged from multiple commits).
  /// Used during merge-on-read to resolve duplicate keys — higher sequence
  /// number wins. Per-file metadata only (not per-row); after compaction,
  /// per-row sequence attribution is lost but LSM level order makes it
  /// unnecessary.
  int64_t minSequenceNumber{0};
  int64_t maxSequenceNumber{0};

  /// Number of rows in this file with RowKind = DELETE or UPDATE_BEFORE.
  /// row_count = addRowCount + deleteRowCount, where addRowCount is the
  /// number of INSERT or UPDATE_AFTER rows.
  ///
  /// Only applicable to primary-key tables. Append-only tables have no
  /// _rowkind column and every row is implicitly +I, so deleteRowCount is
  /// always 0.
  ///
  /// This is independent of deletionFile — deleteRowCount counts changelog
  /// records stored inside the file, while deletionFile is an external bitmap
  /// of positionally deleted rows.
  ///
  /// Used to determine rawConvertible: if deleteRowCount > 0, the file
  /// contains changelog records and cannot be read raw (needs RowKind
  /// filtering during merge-on-read).
  int64_t deleteRowCount{0};

  /// Timestamp (epoch millis) when this file was created.
  int64_t creationTimeMs{0};

  /// Whether this is a data file or changelog file.
  Type type{Type::kData};

  /// How this file was produced (write vs compaction).
  Source source{Source::kAppend};

  /// Deletion file for this data file. Contains a roaring bitmap of deleted
  /// row positions. Nullopt if no rows have been deleted from this file.
  /// Applies to both primary-key and append-only tables (when
  /// deletion-vectors.enabled is set). Orthogonal to deleteRowCount —
  /// deletionFile marks positional deletes, while deleteRowCount counts
  /// RowKind-based changelog records.
  /// See PaimonDeletionFile for details.
  std::optional<PaimonDeletionFile> deletionFile;

  std::string toString() const;
  folly::dynamic serialize() const;
  static PaimonDataFile create(const folly::dynamic& obj);
};

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonDataFile::Type type) {
  os << PaimonDataFile::typeString(type);
  return os;
}

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonDataFile::Source source) {
  os << PaimonDataFile::sourceString(source);
  return os;
}

} // namespace facebook::velox::connector::hive::paimon

template <>
struct fmt::formatter<
    facebook::velox::connector::hive::paimon::PaimonDataFile::Type>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::paimon::PaimonDataFile::Type type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::paimon::PaimonDataFile::typeString(
            type),
        ctx);
  }
};

template <>
struct fmt::formatter<
    facebook::velox::connector::hive::paimon::PaimonDataFile::Source>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::paimon::PaimonDataFile::Source source,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::paimon::PaimonDataFile::sourceString(
            source),
        ctx);
  }
};
