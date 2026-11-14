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
#include <cstdint>
#include <ostream>
#include <string>

#include <folly/CPortability.h>

namespace facebook::velox::connector::paimon {

/// Paimon primary-key data and changelog files lay out each row as
///
///   [_KEY_<key>..., _SEQUENCE_NUMBER, _VALUE_KIND, <table columns>...]
///
/// where the key columns are the primary key without the partition columns.
/// Files written with data-file.thin-mode=true omit the _KEY_ columns, since
/// the table columns already hold the keys. Append-only files hold only the
/// table columns. At the file format level (Parquet, ORC, Nimble) these are
/// regular columns — the file format has no special knowledge of them.

/// Name prefix of the key columns.
static constexpr std::string_view kKeyColumnPrefix = "_KEY_";

/// BIGINT column holding the sequence number that orders the versions of a
/// key.
static constexpr std::string_view kSequenceNumberColumn = "_SEQUENCE_NUMBER";

/// TINYINT column holding the PaimonRowKind of each row.
static constexpr std::string_view kValueKindColumn = "_VALUE_KIND";

/// Row-level change type stored in the `_VALUE_KIND` column.
/// Values match the Paimon Java RowKind enum (0-3).
///
/// Only meaningful for primary-key tables. Append-only tables have no
/// concept of updates or deletes — every row is implicitly +I.
enum class PaimonRowKind : int8_t {
  /// +I: New row inserted.
  kInsert = 0,
  /// -U: Old value being replaced (retraction). Always followed by +U.
  kUpdateBefore = 1,
  /// +U: New value replacing the old one. Always preceded by -U.
  kUpdateAfter = 2,
  /// -D: Row deleted. Carries the full before-image (all column values).
  kDelete = 3,
};

std::string paimonRowKindString(PaimonRowKind kind);
PaimonRowKind paimonRowKindFromValue(int8_t value);

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonRowKind kind) {
  os << paimonRowKindString(kind);
  return os;
}

/// Changelog mode determines how a table produces changelog files. This is a
/// table-level property set via the `changelog-producer` option.
/// Only meaningful for primary-key tables — append-only tables have no
/// updates or deletes, so every row is implicitly +I regardless of this
/// setting.
///
/// Every primary-key data file stores `_VALUE_KIND` whatever the mode, and
/// changelog files store it too. Batch reads use it to drop retracted keys
/// during merge-on-read and read no changelog files. This enum becomes
/// relevant for streaming.
enum class PaimonChangelogMode {
  /// Default for primary-key tables. No changelog producer configured.
  /// Compaction does NOT generate changelog files. Streaming reads emit all
  /// delta rows as +I (cannot distinguish INSERT from UPDATE).
  kNone,
  /// Source provides RowKind explicitly (CDC/input changelog), and writes
  /// copy the input records into changelog files.
  kInput,
  /// Compactor generates changelog files by looking up old values during
  /// every compaction (both partial and full). Provides low-latency changelog
  /// for streaming consumers at the cost of higher write amplification.
  kLookup,
  /// Same mechanism as kLookup (looking up old values to produce changelog),
  /// but changelog files are only generated during full compaction — partial
  /// compactions do NOT produce changelog. Between full compactions,
  /// streaming consumers get degraded output (all rows emitted as +I).
  /// Lower write cost than kLookup, but higher changelog latency.
  kFullCompaction,
};

std::string paimonChangelogModeString(PaimonChangelogMode mode);
PaimonChangelogMode paimonChangelogModeFromString(const std::string& str);

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonChangelogMode mode) {
  os << paimonChangelogModeString(mode);
  return os;
}

} // namespace facebook::velox::connector::paimon

template <>
struct fmt::formatter<facebook::velox::connector::paimon::PaimonRowKind>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::paimon::PaimonRowKind kind,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::paimon::paimonRowKindString(kind), ctx);
  }
};

template <>
struct fmt::formatter<facebook::velox::connector::paimon::PaimonChangelogMode>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::paimon::PaimonChangelogMode mode,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::paimon::paimonChangelogModeString(mode),
        ctx);
  }
};
