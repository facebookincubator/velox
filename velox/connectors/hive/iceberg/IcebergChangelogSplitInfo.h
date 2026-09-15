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
#include <string_view>

namespace facebook::velox::connector::hive::iceberg {

/// Type of operation recorded in a changelog entry.
enum class ChangelogOperation { INSERT, DELETE, UPDATE_BEFORE, UPDATE_AFTER };

/// String names for each ChangelogOperation value, used as the literal value
/// stored in the "operation" output column.
inline constexpr std::string_view kChangelogOpInsert = "INSERT";
inline constexpr std::string_view kChangelogOpDelete = "DELETE";
inline constexpr std::string_view kChangelogOpUpdateBefore = "UPDATE_BEFORE";
inline constexpr std::string_view kChangelogOpUpdateAfter = "UPDATE_AFTER";

/// Column names of the changelog output schema
/// (operation VARCHAR, ordinal BIGINT, snapshotid BIGINT, rowdata ROW<…>).
inline constexpr std::string_view kChangelogColOperation = "operation";
inline constexpr std::string_view kChangelogColOrdinal = "ordinal";
inline constexpr std::string_view kChangelogColSnapshotId = "snapshotid";
inline constexpr std::string_view kChangelogColRowdata = "rowdata";

/// Metadata for a changelog split describing the operation type, sequence,
/// and snapshot ID for a batch of changelog records.
struct ChangelogSplitInfo {
  /// Type of change: INSERT, DELETE, UPDATE_BEFORE, or UPDATE_AFTER.
  ChangelogOperation operation;
  /// Sequence number for ordering changes within a snapshot.
  int64_t ordinal;
  /// Snapshot this row-level change was made in.
  int64_t snapshotId;
};

} // namespace facebook::velox::connector::hive::iceberg
