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

#include "velox/connectors/hive/FileConfig.h"

namespace facebook::velox::connector::hive {

/// Hive connector configs. Extends FileConfig with Hive-specific settings
/// for partitioning, bucketing, write-path options, GCS storage, and other
/// Hive table format concerns.
class HiveConfig : public FileConfig {
 public:
  enum class InsertExistingPartitionsBehavior {
    kError,
    kOverwrite,
  };

  static std::string insertExistingPartitionsBehaviorString(
      InsertExistingPartitionsBehavior behavior);

  /// Behavior on insert into existing partitions.
  static constexpr const char* kInsertExistingPartitionsBehaviorSession =
      "insert_existing_partitions_behavior";
  static constexpr const char* kInsertExistingPartitionsBehavior =
      "insert-existing-partitions-behavior";

  /// Maximum number of (bucketed) partitions per a single table writer
  /// instance.
  static constexpr const char* kMaxPartitionsPerWriters =
      "max-partitions-per-writers";
  static constexpr const char* kMaxPartitionsPerWritersSession =
      "max_partitions_per_writers";

  /// Maximum number of buckets allowed to output by the table writers.
  static constexpr const char* kMaxBucketCount = "max-bucket-count";
  static constexpr const char* kMaxBucketCountSession = "max_bucket_count";

  /// Whether new data can be inserted into an unpartition table.
  /// Velox currently does not support appending data to existing partitions.
  static constexpr const char* kImmutablePartitions = "immutable-partitions";

  /// The GCS storage endpoint server.
  static constexpr const char* kGcsEndpoint = "gcs.endpoint";

  /// The GCS service account configuration JSON key file.
  static constexpr const char* kGcsCredentialsPath = "gcs.json-key-file-path";

  /// The GCS maximum retry counter of transient errors.
  static constexpr const char* kGcsMaxRetryCount = "gcs.max-retry-count";

  /// The GCS maximum time allowed to retry transient errors.
  static constexpr const char* kGcsMaxRetryTime = "gcs.max-retry-time";

  static constexpr const char* kGcsAuthAccessTokenProvider =
      "gcs.auth.access-token-provider";

  static constexpr const char* kPartitionPathAsLowerCaseSession =
      "partition_path_as_lower_case";

  static constexpr const char* kAllowNullPartitionKeys =
      "allow-null-partition-keys";
  static constexpr const char* kAllowNullPartitionKeysSession =
      "allow_null_partition_keys";

  /// Maximum number of entries in the file handle cache.
  static constexpr const char* kNumCacheFileHandles = "num_cached_file_handles";

  /// Expiration time in ms for a file handle in the cache. A value of 0
  /// means cache will not evict the handle after kFileHandleExprationDurationMs
  /// has passed.
  static constexpr const char* kFileHandleExpirationDurationMs =
      "file-handle-expiration-duration-ms";

  /// Enable file handle cache.
  static constexpr const char* kEnableFileHandleCache =
      "file-handle-cache-enabled";

  /// Config used to create write files. This config is provided to underlying
  /// file system through hive connector and data sink. The config is free form.
  /// The form should be defined by the underlying file system.
  static constexpr const char* kWriteFileCreateConfig =
      "write-file-create-config";

  /// Maximum number of rows for sort writer in one batch of output.
  static constexpr const char* kSortWriterMaxOutputRows =
      "sort-writer-max-output-rows";
  static constexpr const char* kSortWriterMaxOutputRowsSession =
      "sort_writer_max_output_rows";

  /// Maximum bytes for sort writer in one batch of output.
  static constexpr const char* kSortWriterMaxOutputBytes =
      "sort-writer-max-output-bytes";
  static constexpr const char* kSortWriterMaxOutputBytesSession =
      "sort_writer_max_output_bytes";

  /// Sort Writer will exit finish() method after this many milliseconds even if
  /// it has not completed its work yet. Zero means no time limit.
  static constexpr const char* kSortWriterFinishTimeSliceLimitMs =
      "sort-writer-finish-time-slice-limit-ms";
  static constexpr const char* kSortWriterFinishTimeSliceLimitMsSession =
      "sort_writer_finish_time_slice_limit_ms";

  /// Maximum target file size. When a file exceeds this size during writing,
  /// the writer will close the current file and start writing to a new file.
  /// Accepts human-readable values like "1GB". Zero means no limit (default).
  static constexpr const char* kMaxTargetFileSize = "max-target-file-size";
  static constexpr const char* kMaxTargetFileSizeSession =
      "max_target_file_size";

  /// Maximum number of output rows to return per index lookup request.
  /// The limit is applied to the actual output rows after filtering.
  /// 0 means no limit (default).
  static constexpr const char* kMaxRowsPerIndexRequest =
      "max-rows-per-index-request";
  static constexpr const char* kMaxRowsPerIndexRequestSession =
      "max_rows_per_index_request";

  static constexpr const char* kUser = "user";
  static constexpr const char* kSource = "source";
  static constexpr const char* kSchema = "schema";

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior(
      const config::ConfigBase* session) const;

  uint32_t maxPartitionsPerWriters(const config::ConfigBase* session) const;

  uint32_t maxBucketCount(const config::ConfigBase* session) const;

  bool immutablePartitions() const;

  std::string gcsEndpoint() const;

  std::string gcsCredentialsPath() const;

  std::optional<int> gcsMaxRetryCount() const;

  std::optional<std::string> gcsMaxRetryTime() const;

  std::optional<std::string> gcsAuthAccessTokenProvider() const;

  bool isPartitionPathAsLowerCase(const config::ConfigBase* session) const;

  bool allowNullPartitionKeys(const config::ConfigBase* session) const;

  int32_t numCacheFileHandles() const;

  uint64_t fileHandleExpirationDurationMs() const;

  bool isFileHandleCacheEnabled() const;

  uint64_t fileWriterFlushThresholdBytes() const;

  std::string writeFileCreateConfig() const;

  uint32_t sortWriterMaxOutputRows(const config::ConfigBase* session) const;

  uint64_t sortWriterMaxOutputBytes(const config::ConfigBase* session) const;

  uint64_t sortWriterFinishTimeSliceLimitMs(
      const config::ConfigBase* session) const;

  uint64_t maxTargetFileSizeBytes(const config::ConfigBase* session) const;

  /// Returns the maximum number of rows to read per index lookup request.
  /// 0 means no limit (default).
  uint32_t maxRowsPerIndexRequest(const config::ConfigBase* session) const;

  /// User of the query. Used for storage logging.
  std::string user(const config::ConfigBase* session) const;

  /// Source of the query. Used for storage access and logging.
  std::string source(const config::ConfigBase* session) const;

  /// Schema of the query. Used for storage logging.
  std::string schema(const config::ConfigBase* session) const;

  HiveConfig(std::shared_ptr<const config::ConfigBase> config)
      : FileConfig(std::move(config)) {}
};

} // namespace facebook::velox::connector::hive
