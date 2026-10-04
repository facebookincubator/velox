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

#include "velox/connectors/hive/FileProperties.h"
#include "velox/connectors/hive/paimon/PaimonDeletionFile.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive::paimon {

/// A data or changelog file in a planner-supplied Paimon split. File role and
/// origin are independent of the physical DWIO format and per-row RowKind.
struct PaimonDataFile {
  enum class Type {
    kData,
    kChangelog,
  };

  /// Normalized origin from Paimon's DataFileMeta.FileSource. The caller is
  /// responsible for interpreting historical metadata with a missing source.
  enum class Source {
    kAppend,
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

  /// Historical table schema identity, unrelated to the wire or format version.
  std::optional<int64_t> schemaId;

  /// Per-file physical format. When absent, the explicitly supplied split
  /// format applies; the executor never infers it from the filename.
  std::optional<dwio::common::FileFormat> fileFormat;

  /// Optional content-equivalent read location and filesystem access context.
  std::string physicalFilePath;
  std::optional<FileProperties> properties;

  /// LSM level from Paimon metadata. This alone does not establish sorted-run
  /// boundaries or prove that a primary-key file can be read raw.
  int32_t level{0};

  /// Sequence bounds summarize the per-row versions; they do not replace the
  /// _SEQUENCE_NUMBER column in a primary-key file, including after compaction.
  int64_t minSequenceNumber{0};
  int64_t maxSequenceNumber{0};

  /// Number of DELETE / UPDATE_BEFORE records, independent of positional DV
  /// deletions. Missing means unknown, not zero. Ordinary append semantics can
  /// establish that rows are inserts; PK raw reads require a known zero count.
  std::optional<int64_t> deleteRowCount;

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
  void validate() const;
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
