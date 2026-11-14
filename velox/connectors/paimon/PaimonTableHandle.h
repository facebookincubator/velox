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

#include "velox/connectors/common/FileTableHandle.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"

namespace facebook::velox::connector::paimon {

/// Paimon merge engine of a primary-key table, which decides how merge-on-read
/// combines the rows of one key. Set by the 'merge-engine' table option.
enum class PaimonMergeEngine {
  /// Keeps the latest row of each key.
  kDeduplicate,
  /// Keeps the latest non-null value of each column.
  kPartialUpdate,
  /// Aggregates each column with its configured function.
  kAggregation,
  /// Keeps the first row of each key.
  kFirstRow,
};

/// Returns the 'merge-engine' option value of the merge engine, e.g.
/// "first-row".
std::string paimonMergeEngineString(PaimonMergeEngine mergeEngine);

/// Parses a 'merge-engine' option value, ignoring capitalization as Paimon
/// Java does.
PaimonMergeEngine paimonMergeEngineFromString(const std::string& str);

/// Table handle for Paimon scans. Stores the fixed table definition and scan
/// conditions, including a schema version ID and stable field IDs aligned with
/// the top-level columns of its Velox RowType.
class PaimonTableHandle final : public connector::FileTableHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  /// Merge-on-read settings of a primary-key table, parsed from the table
  /// options as Paimon Java's CoreOptions does.
  struct MergeOptions {
    /// The 'merge-engine' option.
    PaimonMergeEngine mergeEngine{PaimonMergeEngine::kDeduplicate};

    /// Whether merge-on-read skips DELETE and UPDATE_BEFORE rows. Set by
    /// 'ignore-delete', or else by the first of 'first-row.ignore-delete',
    /// 'deduplicate.ignore-delete' and 'partial-update.ignore-delete'.
    bool ignoreDelete{false};

    /// Columns from the 'sequence.field' option, which order the rows of a key
    /// before their sequence numbers do.
    std::vector<std::string> sequenceFields;

    /// Whether the 'sequence.field.sort-order' option is ascending.
    bool sequenceFieldsAscending{true};
  };

  /// rowType includes partition columns; fieldIds follow its column order.
  /// dataColumns() is the complete rowType because Paimon data files store
  /// partition columns too. options are resolved by the caller. primaryKeys
  /// are the primary key columns of a primary-key table, which merge-on-read
  /// needs.
  PaimonTableHandle(
      std::string connectorId,
      std::string tableName,
      int64_t schemaId,
      RowTypePtr rowType,
      std::vector<int32_t> fieldIds,
      PaimonTableType tableType,
      std::vector<std::string> partitionKeys = {},
      std::unordered_map<std::string, std::string> options = {},
      common::SubfieldFilters subfieldFilters = {},
      core::TypedExprPtr remainingFilter = nullptr,
      std::vector<connector::FileColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "",
      std::vector<std::string> primaryKeys = {});

  /// Identifier of the fixed table schema, distinct from a snapshot ID.
  int64_t schemaId() const {
    return schemaId_;
  }

  /// Complete logical table schema, including partition columns.
  const RowTypePtr& rowType() const {
    return rowType_;
  }

  /// Stable Paimon field IDs aligned with rowType()'s top-level columns.
  /// IDs identify fields independently of their names and positions.
  const std::vector<int32_t>& fieldIds() const {
    return fieldIds_;
  }

  PaimonTableType tableType() const {
    return tableType_;
  }

  /// Ordered partition column names for the complete table.
  const std::vector<std::string>& partitionKeys() const {
    return partitionKeys_;
  }

  /// Native table options resolved by the caller, not file-reader parameters.
  const std::unordered_map<std::string, std::string>& options() const {
    return options_;
  }

  /// Primary key columns of a primary-key table.
  const std::vector<std::string>& primaryKeys() const {
    return primaryKeys_;
  }

  /// Primary key columns that are not partition columns. Rows of one split
  /// share their partition, so merge-on-read compares only these, and Paimon
  /// copies them into data files as the _KEY_ columns.
  const std::vector<std::string>& trimmedPrimaryKeys() const {
    return trimmedPrimaryKeys_;
  }

  /// Merge-on-read settings, which keep their defaults for append-only tables.
  const MergeOptions& mergeOptions() const {
    return mergeOptions_;
  }

  std::vector<connector::FileColumnHandlePtr> filterColumnHandles()
      const override {
    return filterColumnHandles_;
  }

  std::string toString() const override;
  folly::dynamic serialize() const override;
  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);
  static void registerSerDe();

 private:
  const int64_t schemaId_;
  const RowTypePtr rowType_;
  const std::vector<int32_t> fieldIds_;
  const PaimonTableType tableType_;
  const std::vector<std::string> partitionKeys_;
  const std::unordered_map<std::string, std::string> options_;
  const std::vector<connector::FileColumnHandlePtr> filterColumnHandles_;
  const std::vector<std::string> primaryKeys_;
  const std::vector<std::string> trimmedPrimaryKeys_;
  const MergeOptions mergeOptions_;
};

} // namespace facebook::velox::connector::paimon

template <>
struct fmt::formatter<facebook::velox::connector::paimon::PaimonMergeEngine>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::paimon::PaimonMergeEngine mergeEngine,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::paimon::paimonMergeEngineString(
            mergeEngine),
        ctx);
  }
};
