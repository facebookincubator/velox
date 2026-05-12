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
#include "velox/connectors/hive/HiveConfigMacrosDefine.h"

namespace facebook::velox::connector::hive {

/// Hive connector configs. Extends FileConfig with Hive-specific settings
/// for partitioning, bucketing, write-path options, GCS storage, and other
/// Hive table format concerns.
class HiveConfig : public FileConfig {
 public:
  /// Returns all registered session-overridable properties from both
  /// FileConfig and HiveConfig.
  static const std::vector<config::ConfigProperty>& registeredProperties();

  enum class InsertExistingPartitionsBehavior {
    kError,
    kOverwrite,
  };

  static std::string insertExistingPartitionsBehaviorString(
      InsertExistingPartitionsBehavior behavior);

  // --- VELOX_HIVE_CONFIG_PROPERTY properties ---

  VELOX_HIVE_CONFIG_PROPERTY(
      kInsertExistingPartitionsBehaviorSession,
      "insert_existing_partitions_behavior",
      std::string,
      "ERROR",
      "Behavior when inserting into existing partitions: ERROR, OVERWRITE.")
  static constexpr const char* kInsertExistingPartitionsBehavior =
      "insert-existing-partitions-behavior";

  VELOX_HIVE_CONFIG_PROPERTY(
      kSortWriterMaxOutputBytesSession,
      "sort_writer_max_output_bytes",
      std::string,
      "10MB",
      "Maximum output bytes for sort writer.")
  static constexpr const char* kSortWriterMaxOutputBytes =
      "sort-writer-max-output-bytes";

  VELOX_HIVE_CONFIG_PROPERTY(
      kMaxTargetFileSizeSession,
      "max_target_file_size",
      std::string,
      "0B",
      "Maximum target file size. 0 means no limit.")
  static constexpr const char* kMaxTargetFileSize = "max-target-file-size";

  // --- VELOX_HIVE_CONFIG properties ---

  VELOX_HIVE_CONFIG(
      kMaxPartitionsPerWritersSession,
      maxPartitionsPerWriters,
      "max_partitions_per_writers",
      uint32_t,
      128,
      "Maximum partitions per writer.")
  static constexpr const char* kMaxPartitionsPerWriters =
      "max-partitions-per-writers";

  VELOX_HIVE_CONFIG(
      kAllowNullPartitionKeysSession,
      allowNullPartitionKeys,
      "allow_null_partition_keys",
      bool,
      true,
      "Allow null partition keys.")
  static constexpr const char* kAllowNullPartitionKeys =
      "allow-null-partition-keys";

  VELOX_HIVE_CONFIG_PROPERTY(
      kPartitionPathAsLowerCaseSession,
      "partition_path_as_lower_case",
      bool,
      true,
      "Write partition path segments as lower case.")
  bool isPartitionPathAsLowerCase(const config::ConfigBase* session) const {
    return session->get<bool>(
        kPartitionPathAsLowerCaseSession,
        kPartitionPathAsLowerCaseSessionProperty::defaultValue);
  }

  VELOX_HIVE_CONFIG(
      kSortWriterMaxOutputRowsSession,
      sortWriterMaxOutputRows,
      "sort_writer_max_output_rows",
      uint32_t,
      1024,
      "Maximum output rows for sort writer.")
  static constexpr const char* kSortWriterMaxOutputRows =
      "sort-writer-max-output-rows";

  VELOX_HIVE_CONFIG(
      kSortWriterFinishTimeSliceLimitMsSession,
      sortWriterFinishTimeSliceLimitMs,
      "sort_writer_finish_time_slice_limit_ms",
      uint64_t,
      5000,
      "Time slice limit in ms for sort writer finish. 0 means no limit.")
  static constexpr const char* kSortWriterFinishTimeSliceLimitMs =
      "sort-writer-finish-time-slice-limit-ms";

  VELOX_HIVE_CONFIG(kUser, user, "user", std::string, "", "User of the query.")

  VELOX_HIVE_CONFIG(
      kSource,
      source,
      "source",
      std::string,
      "",
      "Source of the query.")

  VELOX_HIVE_CONFIG(
      kSchema,
      schema,
      "schema",
      std::string,
      "",
      "Schema of the query.")

  // --- VELOX_HIVE_CONFIG_LEGACY properties ---

  VELOX_HIVE_CONFIG_LEGACY(
      kMaxBucketCountSession,
      kMaxBucketCount,
      maxBucketCount,
      "max_bucket_count",
      "max-bucket-count",
      uint32_t,
      100'000,
      "Maximum bucket count.")

  VELOX_HIVE_CONFIG_LEGACY(
      kMaxRowsPerIndexRequestSession,
      kMaxRowsPerIndexRequest,
      maxRowsPerIndexRequest,
      "max_rows_per_index_request",
      "max-rows-per-index-request",
      uint32_t,
      0,
      "Maximum rows per index lookup request. 0 means no limit.")
  // --- Server-only properties (no macro) ---

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

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior(
      const config::ConfigBase* session) const;

  bool immutablePartitions() const;

  std::string gcsEndpoint() const;

  std::string gcsCredentialsPath() const;

  std::optional<int> gcsMaxRetryCount() const;

  std::optional<std::string> gcsMaxRetryTime() const;

  std::optional<std::string> gcsAuthAccessTokenProvider() const;

  int32_t numCacheFileHandles() const;

  uint64_t fileHandleExpirationDurationMs() const;

  bool isFileHandleCacheEnabled() const;

  uint64_t fileWriterFlushThresholdBytes() const;

  std::string writeFileCreateConfig() const;

  uint64_t sortWriterMaxOutputBytes(const config::ConfigBase* session) const;

  uint64_t maxTargetFileSizeBytes(const config::ConfigBase* session) const;

  explicit HiveConfig(std::shared_ptr<const config::ConfigBase> config)
      : FileConfig(std::move(config)) {}
};

} // namespace facebook::velox::connector::hive

#include "velox/connectors/hive/HiveConfigMacrosUndef.h"
