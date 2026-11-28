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
#include "velox/connectors/paimon/PaimonMergeEngine.h"

namespace facebook::velox::connector::paimon {

/// Names of the Paimon table options that PaimonTableHandle parses into
/// PaimonTableHandle::MergeOptions, as Paimon Java's CoreOptions names them.
class PaimonTableOptions {
 public:
  /// Merge engine of a primary-key table, one of the PaimonMergeEngine values.
  static constexpr const char* kMergeEngine = "merge-engine";

  /// Whether merge-on-read skips DELETE and UPDATE_BEFORE rows. When it is not
  /// set, the first of the three fallback names below that is set applies,
  /// whatever the merge engine.
  static constexpr const char* kIgnoreDelete = "ignore-delete";

  /// First fallback name of kIgnoreDelete.
  static constexpr const char* kFirstRowIgnoreDelete =
      "first-row.ignore-delete";

  /// Second fallback name of kIgnoreDelete.
  static constexpr const char* kDeduplicateIgnoreDelete =
      "deduplicate.ignore-delete";

  /// Third fallback name of kIgnoreDelete.
  static constexpr const char* kPartialUpdateIgnoreDelete =
      "partial-update.ignore-delete";

  /// Comma-separated columns that order the rows of a key before their
  /// sequence numbers do.
  static constexpr const char* kSequenceField = "sequence.field";

  /// Order of the sequence fields, 'ascending' or 'descending'.
  static constexpr const char* kSequenceFieldSortOrder =
      "sequence.field.sort-order";

  /// Whether the rows of a key are ordered by the snapshot that committed them.
  static constexpr const char* kSequenceSnapshotOrdering =
      "sequence.snapshot-ordering";

  /// How a primary-key table writes changelog files, one of the
  /// PaimonChangelogProducer values.
  static constexpr const char* kChangelogProducer = "changelog-producer";

  /// Whether writes mark replaced rows in deletion vectors.
  static constexpr const char* kDeletionVectorsEnabled =
      "deletion-vectors.enabled";

  /// Whether compaction looks up keys in higher levels even if no other option
  /// asks for it.
  static constexpr const char* kForceLookup = "force-lookup";

  /// Aggregate function of the columns of an aggregation table that do not set
  /// one with fieldAggregateFunction(), one of the PaimonAggregateFunction
  /// values.
  static constexpr const char* kFieldsDefaultAggregateFunction =
      "fields.default-aggregate-function";

  /// Whether a DELETE row of an aggregation table replaces the merged row of
  /// its key.
  static constexpr const char* kAggregationRemoveRecordOnDelete =
      "aggregation.remove-record-on-delete";

  /// Returns the name of the option that sets the aggregate function of
  /// 'column' in an aggregation table, 'fields.<column>.aggregate-function'.
  /// Its value is one of the PaimonAggregateFunction values.
  static std::string fieldAggregateFunction(std::string_view column);

  /// Returns the name of the option that makes the aggregate function of
  /// 'column' in an aggregation table keep its value on DELETE and
  /// UPDATE_BEFORE rows, 'fields.<column>.ignore-retract'.
  static std::string fieldIgnoreRetract(std::string_view column);
};

/// Values of the PaimonTableOptions::kChangelogProducer table option, which
/// Paimon Java accepts in any capitalization.
class PaimonChangelogProducer {
 public:
  /// Writes no changelog files.
  static constexpr const char* kNone = "none";

  /// Writes the input rows as changelog files when the writer flushes them.
  static constexpr const char* kInput = "input";

  /// Writes changelog files with each full compaction.
  static constexpr const char* kFullCompaction = "full-compaction";

  /// Writes changelog files with lookup compaction, which looks up keys in
  /// higher levels.
  static constexpr const char* kLookup = "lookup";
};

/// Names of the aggregate functions of the columns of an aggregation table, the
/// values of the PaimonTableOptions::fieldAggregateFunction() and
/// PaimonTableOptions::kFieldsDefaultAggregateFunction options, as Paimon
/// Java's FieldAggregatorFactory implementations name them.
class PaimonAggregateFunction {
 public:
  /// Takes the value of the last row. Paimon gives it to primary key columns,
  /// whose rows of a key share one value.
  static constexpr const char* kPrimaryKey = "primary-key";

  /// Takes the value of the last row, even if it is null.
  static constexpr const char* kLastValue = "last_value";

  /// Takes the last non-null value. Columns take it when neither
  /// PaimonTableOptions::fieldAggregateFunction() nor
  /// PaimonTableOptions::kFieldsDefaultAggregateFunction sets their function.
  static constexpr const char* kLastNonNullValue = "last_non_null_value";

  /// Takes the value of the first row, even if it is null.
  static constexpr const char* kFirstValue = "first_value";

  /// Takes the first non-null value.
  static constexpr const char* kFirstNonNullValue = "first_non_null_value";

  /// Legacy name of kFirstNonNullValue.
  static constexpr const char* kFirstNotNullValue = "first_not_null_value";

  /// Adds the values.
  static constexpr const char* kSum = "sum";

  /// Multiplies the values.
  static constexpr const char* kProduct = "product";

  /// Takes the smallest value.
  static constexpr const char* kMin = "min";

  /// Takes the largest value.
  static constexpr const char* kMax = "max";

  /// Takes the logical AND of the values.
  static constexpr const char* kBoolAnd = "bool_and";

  /// Takes the logical OR of the values.
  static constexpr const char* kBoolOr = "bool_or";

  /// Concatenates strings with a delimiter.
  static constexpr const char* kListAgg = "listagg";

  /// Collects the elements of arrays into one array.
  static constexpr const char* kCollect = "collect";

  /// Merges maps.
  static constexpr const char* kMergeMap = "merge_map";

  /// Merges maps of rows, keeping for each key the row with the latest time.
  static constexpr const char* kMergeMapWithKeyTime = "merge_map_with_keytime";

  /// Updates a nested table, an array of rows.
  static constexpr const char* kNestedUpdate = "nested_update";

  /// Updates some columns of the rows of a nested table, an array of rows.
  static constexpr const char* kNestedPartialUpdate = "nested_partial_update";

  /// Unions 32-bit Roaring bitmaps.
  static constexpr const char* kRoaringBitmap32 = "rbm32";

  /// Unions 64-bit Roaring bitmaps.
  static constexpr const char* kRoaringBitmap64 = "rbm64";

  /// Unions HyperLogLog sketches.
  static constexpr const char* kHllSketch = "hll_sketch";

  /// Unions Theta sketches.
  static constexpr const char* kThetaSketch = "theta_sketch";
};

/// Table handle for Paimon scans. Stores the fixed table definition and scan
/// conditions, including a schema version ID and stable field IDs aligned with
/// the top-level columns of its Velox RowType.
class PaimonTableHandle final : public connector::FileTableHandle {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  /// Merge-on-read settings of a primary-key table, parsed from the table
  /// options as Paimon Java's CoreOptions does. Merge-on-read sorts the rows
  /// of a key by the sequence fields and then by _SEQUENCE_NUMBER, and picks
  /// the last row for deduplicate and the first for first-row.
  struct MergeOptions {
    /// The 'merge-engine' option.
    PaimonMergeEngine mergeEngine{PaimonMergeEngine::kDeduplicate};

    /// Whether merge-on-read skips DELETE and UPDATE_BEFORE rows, as if the
    /// files did not have them. With deduplicate, a key whose latest row is a
    /// DELETE then returns its latest INSERT or UPDATE_AFTER row instead of
    /// nothing. With first-row, such a row in a key with other rows no longer
    /// fails the query. A key with only such rows returns nothing either way.
    /// For example, with deduplicate and these (id, value, row kind) rows:
    ///
    ///   older file: (1, 10, INSERT), (2, 20, INSERT)
    ///   newer file: (2, 21, DELETE), (3, 30, DELETE)
    ///
    /// the merge returns (1, 10), and with the option set (1, 10) and (2, 20).
    ///
    /// Paimon Java's writer drops such rows when the option is set, so they
    /// come from Paimon 0.7 and earlier, which wrote them anyway, or from
    /// writes before the option was set. Set by 'ignore-delete', or else by the
    /// first of 'first-row.ignore-delete', 'deduplicate.ignore-delete' and
    /// 'partial-update.ignore-delete' that is set, whatever the merge engine.
    bool ignoreDelete{false};

    /// Columns from the 'sequence.field' option, which order the rows of a key
    /// before their sequence numbers do. The row with the largest values is
    /// then the latest whatever the write order, which suits an event time
    /// column: a late write of an older event does not replace a newer one.
    /// Several comma-separated columns compare in order. Not supported with
    /// first-row, as in Paimon Java. For example, with 'sequence.field' = 'ts'
    /// and these (id, ts, value) rows:
    ///
    ///   older file: (1, 5, 10)
    ///   newer file: (1, 3, 11)
    ///
    /// deduplicate returns (1, 5, 10), while without the option it returns the
    /// later write, (1, 3, 11). Rows with equal values fall back to sequence
    /// numbers.
    std::vector<std::string> sequenceFields;

    /// Whether the 'sequence.field.sort-order' option is ascending, so the
    /// largest sequence field values are the latest. When descending, the
    /// smallest are, so the example above returns (1, 3, 11). Nulls sort first
    /// in either order, so a null value is older than any other.
    bool sequenceFieldsAscending{true};

    /// Whether the 'sequence.snapshot-ordering' option, which Paimon Java
    /// added after 1.4.1, is true. It orders the rows of a key by the snapshot
    /// that committed them rather than by writer sequence numbers, which can
    /// overlap: Paimon Java gives every row of a file that a commit wrote the
    /// minimum sequence number of the file, which is the snapshot ID, while
    /// rows of compacted files keep their own. For example, a row with
    /// sequence number 100 from snapshot 10 is then older than one with
    /// sequence number 2 from snapshot 11. Merge-on-read does not support it.
    bool sequenceSnapshotOrdering{false};

    /// Whether compaction looks up keys in higher levels, so that a row of a
    /// level above 0 already merges the rows of its key in higher levels. Then
    /// merge-on-read merges the rows of a key in level 0 with only the first
    /// row of the lowest level above 0, as Paimon Java's LookupMergeFunction
    /// does for every merge engine but first-row. Set when 'changelog-producer'
    /// is 'lookup', 'deletion-vectors.enabled' or 'force-lookup' is true, or
    /// the merge engine is first-row.
    bool needLookup{false};

    /// Aggregate function of a column, from the 'fields.<column>.*' options.
    struct FieldAggregation {
      /// Name of the function, e.g. 'sum'.
      std::string function;

      /// The 'fields.<column>.ignore-retract' option: whether the function
      /// keeps its value on DELETE and UPDATE_BEFORE rows.
      bool ignoreRetract{false};
    };

    /// Aggregate functions of the columns of an aggregation table by column
    /// name, for every table column. As in Paimon Java, sequence fields take
    /// 'last_value' and primary keys 'primary-key'. Other columns take
    /// 'fields.<column>.aggregate-function', or else
    /// 'fields.default-aggregate-function', or else 'last_non_null_value'.
    /// Empty for other merge engines.
    std::unordered_map<std::string, FieldAggregation> fieldAggregations;

    /// The 'aggregation.remove-record-on-delete' option: whether a DELETE row
    /// of an aggregation table replaces the merged row of its key, which the
    /// key returns as a DELETE when that row comes last.
    bool aggregationRemoveRecordOnDelete{false};
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

  /// Primary key columns that are not partition columns, which Paimon Java
  /// calls the trimmed primary keys. Rows of one split share their partition,
  /// so merge-on-read compares only these, and Paimon copies them into data
  /// files as the _KEY_ columns.
  const std::vector<std::string>& nonPartitionPrimaryKeys() const {
    return nonPartitionPrimaryKeys_;
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
  const std::vector<std::string> nonPartitionPrimaryKeys_;
  const MergeOptions mergeOptions_;
};

} // namespace facebook::velox::connector::paimon
