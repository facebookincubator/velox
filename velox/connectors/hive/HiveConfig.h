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

#include <optional>
#include <string>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::connector::hive {

/// Hive connector configs.
class HiveConfig {
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
  ///
  /// TODO: remove hive_orc_use_column_names since it doesn't exist in presto,
  /// right now this is only used for testing.
  static constexpr const char* kMaxPartitionsPerWriters =
      "max-partitions-per-writers";
  static constexpr const char* kMaxPartitionsPerWritersSession =
      "max_partitions_per_writers";

  /// Whether new data can be inserted into an unpartition table.
  /// Velox currently does not support appending data to existing partitions.
  static constexpr const char* kImmutablePartitions =
      "hive.immutable-partitions";

  /// The GCS storage endpoint server.
  static constexpr const char* kGCSEndpoint = "hive.gcs.endpoint";

  /// The GCS service account configuration JSON key file.
  static constexpr const char* kGCSCredentialsPath =
      "hive.gcs.json-key-file-path";

  /// The GCS maximum retry counter of transient errors.
  static constexpr const char* kGCSMaxRetryCount = "hive.gcs.max-retry-count";

  /// The GCS maximum time allowed to retry transient errors.
  static constexpr const char* kGCSMaxRetryTime = "hive.gcs.max-retry-time";

  /// Maps table field names to file field names using names, not indices.
  // TODO: remove hive_orc_use_column_names since it doesn't exist in presto,
  // right now this is only used for testing.
  static constexpr const char* kOrcUseColumnNames = "hive.orc.use-column-names";
  static constexpr const char* kOrcUseColumnNamesSession =
      "hive_orc_use_column_names";

  /// Maps table field names to file field names using names, not indices.
  static constexpr const char* kParquetUseColumnNames =
      "hive.parquet.use-column-names";
  static constexpr const char* kParquetUseColumnNamesSession =
      "parquet_use_column_names";

  /// Reads the source file column name as lower case.
  static constexpr const char* kFileColumnNamesReadAsLowerCase =
      "file-column-names-read-as-lower-case";
  static constexpr const char* kFileColumnNamesReadAsLowerCaseSession =
      "file_column_names_read_as_lower_case";

  static constexpr const char* kPartitionPathAsLowerCaseSession =
      "partition_path_as_lower_case";

  static constexpr const char* kAllowNullPartitionKeys =
      "allow-null-partition-keys";
  static constexpr const char* kAllowNullPartitionKeysSession =
      "allow_null_partition_keys";

  static constexpr const char* kIgnoreMissingFilesSession =
      "ignore_missing_files";

  /// The max coalesce bytes for a request.
  static constexpr const char* kMaxCoalescedBytes = "max-coalesced-bytes";

  /// The max coalesce distance bytes for combining requests.
  static constexpr const char* kMaxCoalescedDistanceBytes =
      "max-coalesced-distance-bytes";

  /// The number of prefetch rowgroups
  static constexpr const char* kPrefetchRowGroups = "prefetch-rowgroups";

  /// The total size in bytes for a direct coalesce request. Up to 8MB load
  /// quantum size is supported when SSD cache is enabled.
  static constexpr const char* kLoadQuantum = "load-quantum";

  /// Maximum number of entries in the file handle cache.
  static constexpr const char* kNumCacheFileHandles = "num_cached_file_handles";

  /// Enable file handle cache.
  static constexpr const char* kEnableFileHandleCache =
      "file-handle-cache-enabled";

  /// The size in bytes to be fetched with Meta data together, used when the
  /// data after meta data will be used later. Optimization to decrease small IO
  /// request
  static constexpr const char* kFooterEstimatedSize = "footer-estimated-size";

  /// The threshold of file size in bytes when the whole file is fetched with
  /// meta data together. Optimization to decrease the small IO requests
  static constexpr const char* kFilePreloadThreshold = "file-preload-threshold";

  /// Maximum stripe size in orc writer.
  static constexpr const char* kOrcWriterMaxStripeSize =
      "hive.orc.writer.stripe-max-size";
  static constexpr const char* kOrcWriterMaxStripeSizeSession =
      "orc_optimized_writer_max_stripe_size";

  /// Maximum dictionary memory that can be used in orc writer.
  static constexpr const char* kOrcWriterMaxDictionaryMemory =
      "hive.orc.writer.dictionary-max-memory";
  static constexpr const char* kOrcWriterMaxDictionaryMemorySession =
      "orc_optimized_writer_max_dictionary_memory";

  /// Configs to control dictionary encoding.
  static constexpr const char* kOrcWriterIntegerDictionaryEncodingEnabled =
      "hive.orc.writer.integer-dictionary-encoding-enabled";
  static constexpr const char*
      kOrcWriterIntegerDictionaryEncodingEnabledSession =
          "orc_optimized_writer_integer_dictionary_encoding_enabled";
  static constexpr const char* kOrcWriterStringDictionaryEncodingEnabled =
      "hive.orc.writer.string-dictionary-encoding-enabled";
  static constexpr const char*
      kOrcWriterStringDictionaryEncodingEnabledSession =
          "orc_optimized_writer_string_dictionary_encoding_enabled";

  /// Enables historical based stripe size estimation after compression.
  static constexpr const char* kOrcWriterLinearStripeSizeHeuristics =
      "hive.orc.writer.linear-stripe-size-heuristics";
  static constexpr const char* kOrcWriterLinearStripeSizeHeuristicsSession =
      "orc_writer_linear_stripe_size_heuristics";

  /// Minimal number of items in an encoded stream.
  static constexpr const char* kOrcWriterMinCompressionSize =
      "hive.orc.writer.min-compression-size";
  static constexpr const char* kOrcWriterMinCompressionSizeSession =
      "orc_writer_min_compression_size";

  /// The compression level to use with ZLIB and ZSTD.
  static constexpr const char* kOrcWriterCompressionLevel =
      "hive.orc.writer.compression-level";
  static constexpr const char* kOrcWriterCompressionLevelSession =
      "orc_optimized_writer_compression_level";

  /// Config used to create write files. This config is provided to underlying
  /// file system through hive connector and data sink. The config is free form.
  /// The form should be defined by the underlying file system.
  static constexpr const char* kWriteFileCreateConfig =
      "hive.write_file_create_config";

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
      "sort-writer_finish_time_slice_limit_ms";
  static constexpr const char* kSortWriterFinishTimeSliceLimitMsSession =
      "sort_writer_finish_time_slice_limit_ms";

  // The unit for reading timestamps from files.
  static constexpr const char* kReadTimestampUnit =
      "hive.reader.timestamp-unit";
  static constexpr const char* kReadTimestampUnitSession =
      "hive.reader.timestamp_unit";

  static constexpr const char* kCacheNoRetention = "cache.no_retention";
  static constexpr const char* kCacheNoRetentionSession = "cache.no_retention";

  InsertExistingPartitionsBehavior insertExistingPartitionsBehavior(
      const config::ConfigBase* session) const;

  uint32_t maxPartitionsPerWriters(const config::ConfigBase* session) const;

  bool immutablePartitions() const;

  std::string gcsEndpoint() const;

  std::string gcsCredentialsPath() const;

  std::optional<int> gcsMaxRetryCount() const;

  std::optional<std::string> gcsMaxRetryTime() const;

  bool isOrcUseColumnNames(const config::ConfigBase* session) const;

  bool isParquetUseColumnNames(const config::ConfigBase* session) const;

  bool isFileColumnNamesReadAsLowerCase(
      const config::ConfigBase* session) const;

  bool isPartitionPathAsLowerCase(const config::ConfigBase* session) const;

  bool allowNullPartitionKeys(const config::ConfigBase* session) const;

  bool ignoreMissingFiles(const config::ConfigBase* session) const;

  int64_t maxCoalescedBytes() const;

  int32_t maxCoalescedDistanceBytes() const;

  int32_t prefetchRowGroups() const;

  int32_t loadQuantum() const;

  int32_t numCacheFileHandles() const;

  bool isFileHandleCacheEnabled() const;

  uint64_t fileWriterFlushThresholdBytes() const;

  uint64_t orcWriterMaxStripeSize(const config::ConfigBase* session) const;

  uint64_t orcWriterMaxDictionaryMemory(
      const config::ConfigBase* session) const;

  bool isOrcWriterIntegerDictionaryEncodingEnabled(
      const config::ConfigBase* session) const;

  bool isOrcWriterStringDictionaryEncodingEnabled(
      const config::ConfigBase* session) const;

  bool orcWriterLinearStripeSizeHeuristics(
      const config::ConfigBase* session) const;

  uint64_t orcWriterMinCompressionSize(const config::ConfigBase* session) const;

  std::optional<uint8_t> orcWriterCompressionLevel(
      const config::ConfigBase* session) const;

  uint8_t orcWriterZLIBCompressionLevel(
      const config::ConfigBase* session) const;

  uint8_t orcWriterZSTDCompressionLevel(
      const config::ConfigBase* session) const;

  std::string writeFileCreateConfig() const;

  uint32_t sortWriterMaxOutputRows(const config::ConfigBase* session) const;

  uint64_t sortWriterMaxOutputBytes(const config::ConfigBase* session) const;

  uint64_t sortWriterFinishTimeSliceLimitMs(
      const config::ConfigBase* session) const;

  uint64_t footerEstimatedSize() const;

  uint64_t filePreloadThreshold() const;

  // Returns the timestamp unit used when reading timestamps from files.
  uint8_t readTimestampUnit(const config::ConfigBase* session) const;

  /// Returns true to evict out a query scanned data out of in-memory cache
  /// right after the access, and also skip staging to the ssd cache. This helps
  /// to prevent the cache space pollution from the one-time table scan by large
  /// batch query when mixed running with interactive query which has high data
  /// locality.
  bool cacheNoRetention(const config::ConfigBase* session) const;

  HiveConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for HiveConfig initialization");
    config_ = std::move(config);
    // TODO: add sanity check
  }

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

 private:
  std::shared_ptr<const config::ConfigBase> config_;
};

/// Build config required to initialize an S3FileSystem instance.
/// All hive.s3 options can be set on a per-bucket basis.
/// The bucket-specific option is set by replacing the hive.s3. prefix on an
/// option with hive.s3.bucket.BUCKETNAME., where BUCKETNAME is the name of the
/// bucket.
/// When connecting to a bucket, all options explicitly set will override the
/// base hive.s3. values.
/// These semantics are similar to the Apache Hadoop-Aws module.
/// https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/index.html
class S3Config {
 public:
  S3Config() = delete;

  /// S3 config prefix.
  static constexpr const char* kS3Prefix = "hive.s3.";

  /// S3 bucket config prefix
  static constexpr const char* kS3BucketPrefix = "hive.s3.bucket.";

  /// Log granularity of AWS C++ SDK.
  static constexpr const char* kS3LogLevel = "hive.s3.log-level";

  /// S3FileSystem default identity.
  static constexpr const char* kDefaultS3Identity = "s3-default-identity";

  /// Keys to identify the config
  enum class Keys {
    kBegin,
    kEndpoint = kBegin,
    kAccessKey,
    kSecretKey,
    kPathStyleAccess,
    kSSLEnabled,
    kUseInstanceCredentials,
    kIamRole,
    kIamRoleSessionName,
    kConnectTimeout,
    kSocketTimeout,
    kMaxConnections,
    kMaxAttempts,
    kRetryMode,
    kUseProxyFromEnv,
    kEnd
  };

  /// Map of keys -> <suffixString, optional defaultValue>.
  /// New config must be added here along with a getter function below.
  static const std::unordered_map<
      Keys,
      std::pair<std::string_view, std::optional<std::string_view>>>&
  configTraits() {
    static const std::unordered_map<
        Keys,
        std::pair<std::string_view, std::optional<std::string_view>>>
        config = {
            {Keys::kEndpoint, std::make_pair("endpoint", "")},
            {Keys::kAccessKey, std::make_pair("aws-access-key", std::nullopt)},
            {Keys::kSecretKey, std::make_pair("aws-secret-key", std::nullopt)},
            {Keys::kPathStyleAccess,
             std::make_pair("path-style-access", "false")},
            {Keys::kSSLEnabled, std::make_pair("ssl.enabled", "true")},
            {Keys::kUseInstanceCredentials,
             std::make_pair("use-instance-credentials", "false")},
            {Keys::kIamRole, std::make_pair("iam-role", std::nullopt)},
            {Keys::kIamRoleSessionName,
             std::make_pair("iam-role-session-name", "velox-session")},
            {Keys::kConnectTimeout,
             std::make_pair("connect-timeout", std::nullopt)},
            {Keys::kSocketTimeout,
             std::make_pair("socket-timeout", std::nullopt)},
            {Keys::kMaxConnections,
             std::make_pair("max-connections", std::nullopt)},
            {Keys::kMaxAttempts, std::make_pair("max-attempts", std::nullopt)},
            {Keys::kRetryMode, std::make_pair("retry-mode", std::nullopt)},
            {Keys::kUseProxyFromEnv,
             std::make_pair("use-proxy-from-env", "false")}};
    return config;
  }

  S3Config(
      std::string_view bucket,
      std::shared_ptr<const config::ConfigBase> config);

  /// Identity is used as a key for the S3FileSystem instance map.
  /// This will be the bucket endpoint or the base endpoint or the
  /// default identity in that order.
  static std::string identity(
      std::string_view bucket,
      std::shared_ptr<const config::ConfigBase> config);

  /// Return the base config for the input Key.
  static std::string baseConfigKey(Keys key) {
    std::stringstream buffer;
    buffer << kS3Prefix << configTraits().find(key)->second.first;
    return buffer.str();
  }

  /// Return the bucket config for the input key.
  static std::string bucketConfigKey(Keys key, std::string_view bucket) {
    std::stringstream buffer;
    buffer << kS3BucketPrefix << bucket << "."
           << configTraits().find(key)->second.first;
    return buffer.str();
  }

  /// The S3 storage endpoint server. This can be used to connect to an
  /// S3-compatible storage system instead of AWS.
  std::string endpoint() const {
    return config_.find(Keys::kEndpoint)->second.value();
  }

  /// Access key to use.
  std::optional<std::string> accessKey() const {
    return config_.find(Keys::kAccessKey)->second;
  }

  /// Secret key to use
  std::optional<std::string> secretKey() const {
    return config_.find(Keys::kSecretKey)->second;
  }

  /// Virtual addressing is used for AWS S3 and is the default
  /// (path-style-access is false). Path access style is used for some on-prem
  /// systems like Minio.
  bool useVirtualAddressing() const {
    auto value = config_.find(Keys::kPathStyleAccess)->second.value();
    return !folly::to<bool>(value);
  }

  /// Use HTTPS to communicate with the S3 API.
  bool useSSL() const {
    auto value = config_.find(Keys::kSSLEnabled)->second.value();
    return folly::to<bool>(value);
  }

  /// Use the EC2 metadata service to retrieve API credentials.
  bool useInstanceCredentials() const {
    auto value = config_.find(Keys::kUseInstanceCredentials)->second.value();
    return folly::to<bool>(value);
  }

  /// IAM role to assume.
  std::optional<std::string> iamRole() const {
    return config_.find(Keys::kIamRole)->second;
  }

  /// Session name associated with the IAM role.
  std::string iamRoleSessionName() const {
    return config_.find(Keys::kIamRoleSessionName)->second.value();
  }

  /// Socket connect timeout.
  std::optional<std::string> connectTimeout() const {
    return config_.find(Keys::kConnectTimeout)->second;
  }

  /// Socket read timeout.
  std::optional<std::string> socketTimeout() const {
    return config_.find(Keys::kSocketTimeout)->second;
  }

  /// Maximum concurrent TCP connections for a single http client.
  std::optional<uint32_t> maxConnections() const {
    auto val = config_.find(Keys::kMaxConnections)->second;
    if (val.has_value()) {
      return folly::to<uint32_t>(val.value());
    }
    return std::optional<uint32_t>();
  }

  /// Maximum retry attempts for a single http client.
  std::optional<int32_t> maxAttempts() const {
    auto val = config_.find(Keys::kMaxAttempts)->second;
    if (val.has_value()) {
      return folly::to<int32_t>(val.value());
    }
    return std::optional<int32_t>();
  }

  /// Retry mode for a single http client.
  std::optional<std::string> retryMode() const {
    return config_.find(Keys::kRetryMode)->second;
  }

  bool useProxyFromEnv() const {
    auto value = config_.find(Keys::kUseProxyFromEnv)->second.value();
    return folly::to<bool>(value);
  }

 private:
  std::unordered_map<Keys, std::optional<std::string>> config_;
};

} // namespace facebook::velox::connector::hive
