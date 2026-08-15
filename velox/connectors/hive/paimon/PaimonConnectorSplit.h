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
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <folly/CPortability.h>

#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/paimon/PaimonDataFileMeta.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive::paimon {

/// Paimon table type determines read and write semantics.
///
/// # Table Types
///
/// Append-only: No primary key. Each write appends new independent files. All
/// files are at LSM level 0 and rawConvertible is always true. Files can be
/// read in any order for batch queries. No merge-on-read needed. Compaction
/// only reduces the number of small files (concatenation, no deduplication).
///
/// Primary-key: Uses an LSM tree to organize data files per bucket. New writes
/// go to level 0; background compaction merges files into higher levels,
/// deduplicating by primary key. The merge engine determines how duplicate
/// keys are resolved:
///
///   Deduplicate (upsert): Latest write wins — entire row replaced.
///     Write {id=1, name="Alice"} then {id=1, name="Bob"}
///     → result: {id=1, name="Bob"}
///
///   Partial-update: Only non-null columns in new record overwrite old values.
///     Write {id=1, name="Alice", age=25} then {id=1, name=NULL, age=26}
///     → result: {id=1, name="Alice", age=26}
///
/// # Partitioning and Bucketing
///
/// Paimon uses a fixed physical layout: partition-dirs / bucket-dir / files.
/// The hierarchy is strictly partition → bucket → files with no nesting or
/// reordering (no bucket-before-partition or bucket-after-bucket).
///
/// Partitioning: Hive-style directory partitioning by column values. Zero or
/// more partition columns, ordered. Each unique combination of partition values
/// creates a physical directory. Partition columns are NOT stored in the data
/// files (values are in the directory path). Enables partition pruning.
///
/// Bucketing: Hash-distributes rows within a partition into N buckets. Each
/// bucket is a physical directory (bucket-0/, bucket-1/, ...) and acts as an
/// independent LSM tree instance. For primary-key tables, the bucket key
/// defaults to the primary key, ensuring each key maps to exactly one bucket
/// (merge-on-read only needs to look within a single bucket). For append-only
/// tables, bucket key can be any column(s) for write parallelism.
///
/// Unbucketed tables use a single implicit bucket-0/ directory. All four
/// combinations of partitioned/unpartitioned × bucketed/unbucketed are valid:
///
///   Partitioned + bucketed:   dt=2024-01-01/bucket-0/data.orc
///   Partitioned + unbucketed: dt=2024-01-01/bucket-0/data.orc  (bucket-0 only)
///   Unpartitioned + bucketed: bucket-0/data.orc, bucket-1/data.orc, ...
///   Unpartitioned + unbucketed: bucket-0/data.orc  (bucket-0 only)
///
/// Each PaimonConnectorSplit represents one partition + one bucket. The
/// partitionKeys field carries partition column values, and tableBucketNumber
/// identifies the bucket (nullopt for unbucketed tables).
///
/// # LSM Tree Structure (Primary-Key Tables)
///
/// Level 0: Newest data. Each flush creates a new file. Files within level 0
///   CAN have overlapping keys (multiple files may contain the same key).
///   minSequenceNumber == maxSequenceNumber (single commit per file).
///
/// Level 1+: Compacted data. Compaction merges lower-level files, deduplicating
///   keys and producing non-overlapping key ranges. No two files at the same
///   level share a key. minSequenceNumber < maxSequenceNumber (merged from
///   multiple commits).
///
/// # Compaction Strategy (Universal Compaction)
///
/// Primary-key tables use Universal Compaction (similar to RocksDB):
///
/// Sorted runs: Each L0 file is its own sorted run. All files at the same
///   higher level (L1, L2, ...) together form one sorted run with
///   non-overlapping keys. Compaction merges sorted runs.
///
/// Trigger: When the number of sorted runs exceeds
///   num-sorted-run.compaction-trigger (default: 5), compaction is triggered.
///
/// Picking strategy: Compaction picks the OLDEST sorted runs first and merges
///   them until the count drops below the trigger threshold. The output level
///   depends on which sorted runs are merged. Multiple non-L0 levels can
///   exist (L1, L2, ...). Within each level, files have non-overlapping key
///   ranges. L0 files can overlap with any level.
///
/// Who runs compaction:
///   - Default: the writer process (Flink task) runs compaction asynchronously
///     after each commit.
///   - Dedicated mode: a separate compaction job can be configured to offload
///     compaction from writers.
///   - Either way, compaction creates a new snapshot with updated manifest
///     entries (REMOVE old files, ADD compacted file).
///
/// Compaction output:
///   - Data file: deduplicated keys, latest value wins (by sequence number).
///   - With changelog-producer=lookup: also generates a changelog file.
///   - With deletion-vectors.enabled: may generate deletion bitmaps instead of
///     rewriting data files.
///
/// Append-only compaction: Only concatenates small files into larger ones (no
///   key deduplication). All files stay at level 0. Triggered when the number
///   of small files exceeds a threshold.
///
/// rawConvertible: Per-split flag set by the Paimon planner.
///   false = keys overlap across levels, merge-on-read required.
///   true  = fully compacted (single level, no overlapping keys), read
///   directly. Always true for append-only tables.
///
/// # Snapshots and Manifests
///
/// Every Paimon read (batch or streaming) goes through snapshots — immutable
/// point-in-time views of the table. A snapshot points to a manifest-list,
/// which references manifest files that record which data files are active.
///
/// Physical layout on storage:
///
///   table-path/
///     snapshot/
///       snapshot-1      ← JSON: points to manifest-list-1
///       snapshot-2      ← JSON: points to manifest-list-2
///     manifest/
///       manifest-list-1 ← lists which manifest files to read
///       manifest-1      ← data file entries (ADD/REMOVE)
///       manifest-2
///     dt=2024-01-01/
///       bucket-0/
///         data-001.orc  ← actual data files
///         data-002.orc
///     schema/
///       schema-0        ← table schema
///
/// Each manifest entry contains both structured metadata and the file path:
///
///   {action: ADD,
///    partition: {dt: "2024-01-01"},       ← structured, for fast pruning
///    bucket: 0,                           ← structured, for fast pruning
///    filePath: "dt=2024-01-01/bucket-0/data-001.orc",  ← for I/O
///    level: 0, minSeq: 10, maxSeq: 10, ...}
///
/// The partition/bucket values are redundant with the file path (which
/// encodes them in the directory structure) — the structured fields enable
/// fast partition/bucket pruning without path parsing.
///
/// The Paimon planner reads the manifest, applies partition pruning, and
/// generates one PaimonConnectorSplit per partition × bucket combination:
///
///   Manifest entries after pruning for dt='2024-01-01':
///     partition={dt: "2024-01-01"}, bucket=0: [data-001.orc, data-002.orc]
///     partition={dt: "2024-01-01"}, bucket=1: [data-001.orc]
///   → Split 1: partitionKeys={dt: "2024-01-01"}, bucket=0, 2 files
///   → Split 2: partitionKeys={dt: "2024-01-01"}, bucket=1, 1 file
///
/// Snapshot isolation: Readers see only files referenced by their snapshot.
/// Concurrent writers creating new snapshots don't affect in-flight reads.
/// Old data files are only deleted after snapshot expiration (no snapshot
/// references them). The manifest is authoritative — partition directories
/// may contain old compacted-away files that no snapshot references.
///
/// # Read Modes
///
/// All reads are snapshot-based. The mode determines how many snapshots are
/// read and what output is produced:
///
/// Batch (snapshot read): Reads the full file set at ONE snapshot. For
///   primary-key tables, merge-on-read ALWAYS required to deduplicate by key
///   and return the current state — regardless of changelog mode. This is the
///   Velox/Presto read path.
///
/// Time travel: Batch read at a user-specified older snapshot (instead of
///   the latest). Same mechanism, different snapshot ID. Only works if the
///   snapshot hasn't been expired.
///
/// Streaming: Reads DELTAS between consecutive snapshots (N→N+1→N+2→...).
///   Only processes newly added files (the manifest diff), emitting changelog
///   records. Runs continuously, polling for new snapshots. Specifies only a
///   start snapshot — the end is unbounded. Merge-on-read is NOT needed for
///   streaming — the reader only sees delta files, not the full file set, so
///   there is nothing to merge against. For input changelog mode, RowKind is
///   stored in the files and emitted directly. For upsert mode, the reader
///   infers changelog from the delta. Compaction deltas are a special case:
///   the changelog is collapsed, so changelog-producer=lookup is needed to
///   regenerate meaningful changelog records during compaction.
///
/// Incremental: Bounded streaming — reads deltas from snapshot N to M, then
///   stops. Same delta mechanism as streaming but with an end bound.
///
/// Merge-on-read summary for primary-key tables:
///   Batch/time-travel: Always required (full file set, must deduplicate).
///   Streaming/incremental: Never required (delta files only, no merge).
///
/// # Coordinator/Reader Responsibility Split
///
/// Split generation follows a coordinator/reader architecture (mirrors
/// Flink's enumerator/reader pattern):
///
/// Coordinator (Java planner):
///   - Batch: reads manifest at one snapshot, groups files by partition ×
///     bucket, generates one PaimonConnectorSplit per group.
///   - Streaming: starts at a given snapshot, continuously polls for new
///     snapshots. For each new snapshot (N→N+1), computes the manifest diff
///     (newly added files), groups delta files by partition × bucket, and
///     generates one split per group. Repeats indefinitely.
///   - Incremental: same as streaming but stops at the end snapshot M.
///   - With changelog-producer=lookup: sends changelog files (kCompact) for
///     streaming, data files (kAppend) for batch.
///   - Without changelog-producer: sends data files (kAppend) for all modes.
///
/// Reader (Velox worker):
///   - Stateless — receives splits and reads files. Does not track snapshots,
///     does not compute deltas, does not decide which file type to read.
///   - For batch: performs merge-on-read if rawConvertible is false.
///   - For streaming: emits rows directly (as +I without changelog-producer,
///     or with stored RowKind for changelog files / input changelog mode).
///
/// # Changelog Semantics
///
/// Primary-key tables can produce a changelog stream with four row kinds:
///   +I (INSERT):        New key added.
///   -U (UPDATE_BEFORE): Old value being replaced (for retraction).
///   +U (UPDATE_AFTER):  New value replacing it.
///   -D (DELETE):        Key removed.
///
/// -U and +U are always emitted as a consecutive pair for the same key.
/// Intermediate updates are collapsed — the consumer sees only the net
/// change (oldest value → newest value), not every step.
///
/// # RowKind and the _rowkind Column
///
/// Every primary-key table supports a hidden system column `_rowkind` that
/// encodes the change type per row. At the file format level (Parquet, ORC,
/// Nimble), `_rowkind` is a regular TINYINT column — the file format has no
/// special knowledge of it. It is "hidden" only at the SQL layer (not shown
/// in SELECT *). Values: 0=+I, 1=-U, 2=+U, 3=-D.
///
/// Whether `_rowkind` is actually written to data files depends on the mode:
///
/// Upsert mode (normal writes): `_rowkind` is omitted from data files since
///   all records are implicitly +I/+U (insert or update, latest wins). The
///   streaming reader INFERS -U/+U pairs during merge-on-read by comparing
///   old vs new values for the same key across LSM levels. When a SQL DELETE
///   is issued, Paimon writes a record with `_rowkind=-D` to level 0, along
///   with the full before-image (all column values populated). Compaction
///   removes the key entirely.
///
///   Limitation: Streaming reads in upsert mode only process delta files
///   (newly added files between snapshots) — the reader does NOT look back
///   at previous snapshots. A key appearing once in the delta is emitted as
///   +I, even if that key already existed in a prior snapshot (should be +U).
///   For correct +I vs +U distinction, use changelog-producer=lookup which
///   looks up old values from existing files to determine the actual change
///   type. This limitation does NOT apply to input changelog mode — the
///   source provides the correct RowKind (+I/+U/-U/-D) explicitly.
///
/// Input changelog mode (CDC source writes): `_rowkind` IS written for every
///   row. The external source (Flink CDC, Kafka) provides the change type.
///   The streaming reader emits rows with their stored RowKind directly — no
///   merge inference needed for uncompacted data. Files must be read in
///   sequence-number order since -U/+U pairs may span files. Compaction
///   collapses the stored changelog; use changelog-producer=lookup to
///   regenerate it during compaction.
///
/// rowkind.field mode: Instead of the hidden `_rowkind` column, the table
///   designates a user-visible column (e.g., "op") to carry the change type.
///   Useful when the CDC source provides change type as a regular field. At
///   the file level, it's just a regular column — the Paimon engine
///   interprets its values as RowKind.
///
/// # Deletes
///
/// A delete record is NOT a row with null values. It's a complete row (full
/// before-image) tagged with RowKind=-D. The mechanism depends on the source:
///
///   SQL DELETE:       Paimon engine writes {_rowkind=-D, id=1, name="Alice"}
///   CDC source:       External source provides -D record with full values
///   rowkind.field:    User column carries the delete marker
///
/// The full old column values (before-image) are carried for streaming/
/// changelog consumers that need retraction semantics. Without old values,
/// the consumer would need its own state lookup to know what was deleted:
///
///   Example — aggregation counting users by country:
///     State: {id=1, name="Alice", country="US"}  → US count = 1
///
///     -D {id=1} (key only):
///       Consumer doesn't know which country to decrement — must look up.
///     -D {id=1, name="Alice", country="US"} (full before-image):
///       Consumer decrements US count directly. No lookup needed.
///
/// Same reason -U (UPDATE_BEFORE) carries the full old row — the consumer
/// retracts old values before applying +U (UPDATE_AFTER) with new values.
/// For batch reads, the old values don't matter (compaction removes the key).
///
/// Separately, deletion files (PaimonDeletionFile) are a bitmap-based
/// mechanism that marks row positions as deleted within an existing data
/// file without rewriting it. This is orthogonal to RowKind-based deletes.
///
/// For batch/snapshot reads (Presto), changelog semantics don't apply — the
/// reader returns the current state of each row, not the change history.
///
/// Append-only tables only produce +I (INSERT) records — no updates or
/// deletes since there's no primary key to identify rows. Streaming reads
/// from append-only tables emit all new rows as +I; file read order doesn't
/// matter since there are no retraction pairs.
///
/// # Sequence Numbers
///
/// Auto-generated per-commit, monotonically increasing. Used within a
/// snapshot for merge-on-read ordering (higher sequence number wins for
/// duplicate keys). NOT used for snapshot isolation (that's handled by the
/// manifest chain). Sequence numbers are per-file metadata, not per-row —
/// after compaction, per-row attribution is lost but the LSM level order
/// makes it unnecessary.
///
/// # changelog-producer=lookup
///
/// When configured on a primary-key table, the compactor generates TWO
/// outputs per compaction: (1) a data file (no `_rowkind` in upsert mode)
/// and (2) a separate changelog file containing the pre-computed changelog
/// with correct RowKind (+I/+U/-U/-D, includes `_rowkind`). Both files are
/// marked as PaimonFileSource::kCompact (produced by compaction). Without
/// this flag, compaction produces only a data file.
///
/// The coordinator (Java planner) sends the right files based on read mode:
///
///   Batch read:      sends data files only → merge-on-read as usual
///   Streaming read:  sends changelog files only → RowKind already correct,
///                    no inference or merge needed
///
/// Without changelog-producer=lookup, streaming reads work as follows:
///
///   Streaming read (no changelog-producer):
///     Coordinator sends the delta data files (newly added between snapshots).
///     The reader does NOT do merge-on-read — it only processes delta files,
///     not the full file set. Each row in the delta is emitted as +I. The
///     limitation is that the reader cannot distinguish +I (new key) from +U
///     (existing key updated) because it doesn't look back at previous
///     snapshots. For many use cases (e.g., downstream upsert sinks) this is
///     acceptable since the consumer treats +I and +U identically.
///
///   Compaction snapshots are problematic without changelog-producer: the
///   compacted file contains ALL keys (not just changed ones), so the reader
///   would emit +I for every key — including unchanged ones. With
///   changelog-producer=lookup, the compactor compares old vs new values and
///   generates a changelog file with only actual changes.
///
///   If the same key is updated across multiple consecutive snapshots (N→N+1,
///   N+1→N+2, ...), each delta is processed independently. The reader emits
///   one +I per delta that touches the key. The downstream consumer must be
///   idempotent (e.g., upsert by key) to handle this correctly. Without
///   merge-on-read, the reader has no version information — it does not know
///   whether a key is new or updated, nor can it deduplicate across deltas.
///   This is by design: streaming reads trade correctness of change types for
///   simplicity and performance.
///
///   Example — key updated across two snapshots without changelog-producer:
///
///     Delta N→N+1:   {id=1, name="Alice"}  → emitted as +I
///     Delta N+1→N+2: {id=1, name="Bob"}    → emitted as +I (not +U)
///
///   This works for upsert sinks (consumer does INSERT-or-UPDATE by key —
///   final state is correct regardless of +I vs +U). It breaks for retraction
///   sinks (e.g., aggregation counting by name):
///
///     Correct:    +I "Alice" (count=1), -U "Alice" + +U "Bob" (count=1)
///     Without changelog-producer: +I "Alice" (count=1), +I "Bob" (count=2) ✗
///
///   For exact changelog semantics, use changelog-producer=lookup.
///
/// The file origin (write vs compaction) is indicated by PaimonFileSource in
/// PaimonDataFile::fileSource. Both data files and changelog files use
/// the same physical file format. Note that PaimonFileSource indicates HOW
/// the file was produced, not WHETHER it is a data or changelog file — both
/// are marked kCompact when produced by compaction. The coordinator tracks
/// which is which via separate manifest entries.
///
/// Schema difference: In upsert mode (no input changelog), data files do NOT
/// contain the `_rowkind` column (all records are implicitly +I/+U). Changelog
/// files generated by changelog-producer=lookup DO contain `_rowkind` since
/// they carry explicit change types. The coordinator knows which file type it
/// is sending, so the reader schema is adjusted accordingly. In input changelog
/// mode, both data and changelog files contain `_rowkind`.
///
/// # Velox Connector Scope
///
/// Currently only batch reads (Presto) are supported. Only data files are
/// needed — changelog files and streaming semantics are not required.
///
/// Future streaming support: We will require changelog-producer=lookup (or
/// full-compaction) to be enabled on the table. This means the compactor
/// generates pre-computed changelog files, and the streaming reader consumes
/// them directly. We do NOT plan to support beforeFiles-based diffing (where
/// the reader merges old and new file sets to infer changelog at read time).
/// This avoids expensive read-time merge and simplifies the reader — the
/// coordinator sends changelog files for compaction snapshots and delta data
/// files for non-compaction snapshots.
enum class PaimonTableType {
  /// No primary key. Data files are independent — no merge-on-read needed.
  /// All files at level 0, rawConvertible always true.
  kAppendOnly,
  /// Has a primary key. Uses LSM tree with merge-on-read to deduplicate
  /// records by key when data spans multiple compaction levels.
  kPrimaryKey,
};

/// Returns the string name of the table type (e.g., "APPEND_ONLY").
std::string paimonTableTypeString(PaimonTableType type);

/// Parses a table type from its string name.
PaimonTableType paimonTableTypeFromString(const std::string& str);

FOLLY_ALWAYS_INLINE std::ostream& operator<<(
    std::ostream& os,
    PaimonTableType type) {
  os << paimonTableTypeString(type);
  return os;
}

} // namespace facebook::velox::connector::hive::paimon

template <>
struct fmt::formatter<facebook::velox::connector::hive::paimon::PaimonTableType>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::paimon::PaimonTableType type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::paimon::paimonTableTypeString(type),
        ctx);
  }
};

namespace facebook::velox::connector::hive::paimon {

/// Represents a Paimon DataSplit — a collection of data files in one
/// partition/bucket.
///
/// Unlike HiveConnectorSplit (one file per split), a Paimon split maps to a
/// logical bucket which may contain multiple physical files across LSM-tree
/// levels.
///
/// NOTE: Table-wide metadata (primary key columns, merge engine, table schema)
/// is NOT carried in the split. A future PaimonTableHandle (extending
/// HiveTableHandle) will carry this information, needed for merge-on-read
/// (key deduplication) and schema evolution. For batch reads of
/// rawConvertible=true splits, the existing HiveTableHandle suffices.
class PaimonConnectorSplit : public connector::ConnectorSplit {
 public:
  /// @param connectorId Connector identifier.
  /// @param snapshotId Paimon table snapshot version this split was generated
  ///        from.
  /// @param tableType Whether this is an append-only or primary-key table.
  /// @param fileFormat File format of the data files (e.g., ORC, Parquet).
  /// @param dataFiles Data files in this split, each representing a physical
  ///        file in the LSM-tree.
  /// @param partitionKeys Partition key-value pairs. Keys map to partition
  ///        column names; values are nullopt for null partitions.
  /// @param tableBucketNumber Bucket number within the Paimon table's bucket
  ///        distribution. Nullopt for unbucketed tables.
  /// @param rawConvertible Per-split flag indicating whether all files in this
  ///        split (partition × bucket) can be read without merge-on-read (no
  ///        key deduplication needed across LSM levels). Set by the Paimon
  ///        planner based on the compaction state of the entire bucket. Only
  ///        meaningful for primary-key tables.
  PaimonConnectorSplit(
      const std::string& connectorId,
      int64_t snapshotId,
      PaimonTableType tableType,
      dwio::common::FileFormat fileFormat,
      const std::vector<PaimonDataFile>& dataFiles,
      std::unordered_map<std::string, std::optional<std::string>> partitionKeys,
      std::optional<int32_t> tableBucketNumber,
      bool rawConvertible = true);

  int64_t snapshotId() const {
    return snapshotId_;
  }

  PaimonTableType tableType() const {
    return tableType_;
  }

  const std::vector<PaimonDataFile>& dataFiles() const {
    return dataFiles_;
  }

  const std::unordered_map<std::string, std::optional<std::string>>&
  partitionKeys() const {
    return partitionKeys_;
  }

  std::optional<int32_t> tableBucketNumber() const {
    return tableBucketNumber_;
  }

  bool rawConvertible() const {
    return rawConvertible_;
  }

  dwio::common::FileFormat fileFormat() const {
    return fileFormat_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static std::shared_ptr<PaimonConnectorSplit> create(
      const folly::dynamic& obj);

  static void registerSerDe();

 private:
  const int64_t snapshotId_;
  const PaimonTableType tableType_;
  const dwio::common::FileFormat fileFormat_;
  const std::vector<PaimonDataFile> dataFiles_;
  const std::unordered_map<std::string, std::optional<std::string>>
      partitionKeys_;
  const std::optional<int32_t> tableBucketNumber_;
  const bool rawConvertible_;
};

/// Builder for PaimonConnectorSplit construction.
class PaimonConnectorSplitBuilder {
 public:
  PaimonConnectorSplitBuilder(
      std::string connectorId,
      int64_t snapshotId,
      PaimonTableType tableType,
      dwio::common::FileFormat fileFormat)
      : connectorId_(std::move(connectorId)),
        snapshotId_(snapshotId),
        tableType_(tableType),
        fileFormat_(fileFormat) {}

  PaimonConnectorSplitBuilder&
  addFile(std::string filePath, uint64_t fileSize, int32_t level = 0);

  PaimonConnectorSplitBuilder& partitionKey(
      std::string name,
      std::optional<std::string> value);

  PaimonConnectorSplitBuilder& tableBucketNumber(int32_t bucketId);

  PaimonConnectorSplitBuilder& rawConvertible(bool value);

  std::shared_ptr<PaimonConnectorSplit> build();

 private:
  const std::string connectorId_;
  const int64_t snapshotId_;
  const PaimonTableType tableType_;
  const dwio::common::FileFormat fileFormat_;
  std::vector<PaimonDataFile> dataFiles_;
  std::unordered_map<std::string, std::optional<std::string>> partitionKeys_;
  std::optional<int32_t> tableBucketNumber_;
  bool rawConvertible_{true};
};

} // namespace facebook::velox::connector::hive::paimon
