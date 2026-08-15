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

namespace facebook::velox::connector::hive::iceberg {

/// Type of operation recorded in a changelog entry.
enum class ChangelogOperation { INSERT, DELETE, UPDATE_BEFORE, UPDATE_AFTER };

/// Metadata for a changelog split describing the operation type, sequence,
/// and snapshot ID for a batch of changelog records.
struct ChangelogSplitInfo {
  /// Type of change: INSERT, DELETE, UPDATE_BEFORE, or UPDATE_AFTER.
  ChangelogOperation operation;
  /// Sequence number for ordering changes within a snapshot.
  int64_t ordinal;
  /// Snapshot this row-level change was made in.
  int64_t snapshotId;

  ChangelogSplitInfo(ChangelogOperation op, int64_t ord, int64_t snapId)
      : operation(op), ordinal(ord), snapshotId(snapId) {}
};

} // namespace facebook::velox::connector::hive::iceberg
