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
#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive {

/// Configuration for file-based data sources. Contains settings for file
/// reading, I/O coalescing, format-specific options, and metadata caching.
/// HiveConfig extends this with Hive-specific settings like partitioning,
/// bucketing, and write-path options.
class FileConfig {
 public:
  /// Maps table field names to file field names using names, not indices.
  static constexpr const char* kOrcUseColumnNames = "orc.use-column-names";
  static constexpr const char* kOrcUseColumnNamesSession =
      "orc_use_column_names";

  /// Maps table field names to file field names using names, not indices.
  static constexpr const char* kParquetUseColumnNames =
      "parquet.use-column-names";
  static constexpr const char* kParquetUseColumnNamesSession =
      "parquet_use_column_names";

  /// Allows reading INT32 physical type columns as a narrower integer type.
  static constexpr const char* kAllowInt32Narrowing =
      "parquet.allow-int32-narrowing";
  static constexpr const char* kAllowInt32NarrowingSession =
      "allow_int32_narrowing";

  /// Reads the source file column name as lower case.
  static constexpr const char* kFileColumnNamesReadAsLowerCase =
      "file-column-names-read-as-lower-case";
  static constexpr const char* kFileColumnNamesReadAsLowerCaseSession =
      "file_column_names_read_as_lower_case";

  static constexpr const char* kIgnoreMissingFilesSession =
      "ignore_missing_files";

  /// The max coalesce bytes for a request.
  static constexpr const char* kMaxCoalescedBytes = "max-coalesced-bytes";
  static constexpr const char* kMaxCoalescedBytesSession =
      "max-coalesced-bytes";

  /// The max merge distance to combine read requests.
  /// Note: The session property name differs from the constant name for
  /// backward compatibility with Presto.
  static constexpr const char* kMaxCoalescedDistance = "max-coalesced-distance";
  static constexpr const char* kMaxCoalescedDistanceSession =
      "orc_max_merge_distance";

  /// The number of prefetch rowgroups.
  static constexpr const char* kPrefetchRowGroups = "prefetch-rowgroups";

  /// The total size in bytes for a direct coalesce request. Up to 8MB load
  /// quantum size is supported when SSD cache is enabled.
  static constexpr const char* kLoadQuantum = "load-quantum";
  static constexpr const char* kLoadQuantumSession = "load-quantum";

  /// The threshold of file size in bytes when the whole file is fetched with
  /// meta data together. Optimization to decrease the small IO requests.
  static constexpr const char* kFilePreloadThreshold = "file-preload-threshold";

  /// When set to be larger than 0, parallel unit loader feature is enabled and
  /// it configures how many units (e.g., stripes) we load in parallel.
  /// When set to 0, parallel unit loader feature is disabled and on demand unit
  /// loader would be used.
  static constexpr const char* kParallelUnitLoadCount =
      "parallel-unit-load-count";
  static constexpr const char* kParallelUnitLoadCountSession =
      "parallel_unit_load_count";

  // The unit for reading timestamps from files.
  static constexpr const char* kReadTimestampUnit = "reader.timestamp-unit";
  static constexpr const char* kReadTimestampUnitSession =
      "reader.timestamp_unit";

  static constexpr const char* kReadTimestampPartitionValueAsLocalTime =
      "reader.timestamp-partition-value-as-local-time";
  static constexpr const char* kReadTimestampPartitionValueAsLocalTimeSession =
      "reader.timestamp_partition_value_as_local_time";

  static constexpr const char* kReadStatsBasedFilterReorderDisabled =
      "stats-based-filter-reorder-disabled";
  static constexpr const char* kReadStatsBasedFilterReorderDisabledSession =
      "stats_based_filter_reorder_disabled";

  /// Whether to preserve flat maps in memory as FlatMapVectors instead of
  /// converting them to MapVectors.
  static constexpr const char* kPreserveFlatMapsInMemory =
      "preserve-flat-maps-in-memory";
  static constexpr const char* kPreserveFlatMapsInMemorySession =
      "preserve_flat_maps_in_memory";

  /// Whether to use the cluster index for filter-based row pruning.
  /// When enabled, filters from ScanSpec are converted to index bounds for
  /// efficient row skipping based on the file's cluster index.
  static constexpr const char* kIndexEnabled = "index-enabled";
  static constexpr const char* kIndexEnabledSession = "index_enabled";

  /// Whether to collect per-column timing stats (decode/decompress CPU time).
  /// Disabled by default to avoid overhead (~100ns per operation).
  static constexpr const char* kReaderCollectColumnCpuMetrics =
      "reader.collect-column-cpu-metrics";
  static constexpr const char* kReaderCollectColumnCpuMetricsSession =
      "reader.collect_column_cpu_metrics";

  /// Whether to cache file metadata (footer, stripes, index) in the
  /// process-wide AsyncDataCache. When enabled, the first reader performs a
  /// speculative tail read and populates the cache; subsequent readers on the
  /// same file hit the cache for zero-IO metadata init.
  static constexpr const char* kFileMetadataCacheEnabled =
      "file-metadata-cache-enabled";
  static constexpr const char* kFileMetadataCacheEnabledSession =
      "file_metadata_cache_enabled";

  /// Whether to pin parsed metadata objects (e.g., StripeGroup, IndexGroup)
  /// in the reader's metadata cache with strong references so they are never
  /// evicted. This avoids re-reading and re-parsing metadata on every stripe
  /// access when weak-pointer cache entries would otherwise expire.
  /// Currently only supported by Nimble format.
  static constexpr const char* kPinFileMetadata = "pin-file-metadata";
  static constexpr const char* kPinFileMetadataSession = "pin_file_metadata";

  /// Speculative tail-read size in bytes when opening files. Controls how many
  /// bytes are read from the end of the file to load the footer and nearby
  /// metadata in a single IO operation. Format-specific configs with different
  /// defaults: ORC/Parquet default to 256KB, Nimble defaults to 8MB.
  static constexpr const char* kOrcFooterSpeculativeIoSize =
      "orc.footer-speculative-io-size";
  static constexpr const char* kOrcFooterSpeculativeIoSizeSession =
      "orc_footer_speculative_io_size";
  static constexpr const char* kParquetFooterSpeculativeIoSize =
      "parquet.footer-speculative-io-size";
  static constexpr const char* kParquetFooterSpeculativeIoSizeSession =
      "parquet_footer_speculative_io_size";
  static constexpr const char* kNimbleFooterSpeculativeIoSize =
      "nimble.footer-speculative-io-size";
  static constexpr const char* kNimbleFooterSpeculativeIoSizeSession =
      "nimble_footer_speculative_io_size";

  explicit FileConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for FileConfig initialization");
    config_ = std::move(config);
  }

  virtual ~FileConfig() = default;

  bool isOrcUseColumnNames(const config::ConfigBase* session) const;

  bool isParquetUseColumnNames(const config::ConfigBase* session) const;

  bool allowInt32Narrowing(const config::ConfigBase* session) const;

  bool isFileColumnNamesReadAsLowerCase(
      const config::ConfigBase* session) const;

  bool ignoreMissingFiles(const config::ConfigBase* session) const;

  int64_t maxCoalescedBytes(const config::ConfigBase* session) const;

  int32_t maxCoalescedDistanceBytes(const config::ConfigBase* session) const;

  int32_t prefetchRowGroups() const;

  size_t parallelUnitLoadCount(const config::ConfigBase* session) const;

  int32_t loadQuantum(const config::ConfigBase* session) const;

  uint64_t filePreloadThreshold() const;

  // Returns the timestamp unit used when reading timestamps from files.
  uint8_t readTimestampUnit(const config::ConfigBase* session) const;

  // Whether to read timestamp partition value as local time. If false, read as
  // UTC.
  bool readTimestampPartitionValueAsLocalTime(
      const config::ConfigBase* session) const;

  /// Returns true if the stats based filter reorder for read is disabled.
  bool readStatsBasedFilterReorderDisabled(
      const config::ConfigBase* session) const;

  /// Whether to preserve flat maps in memory as FlatMapVectors instead of
  /// converting them to MapVectors.
  bool preserveFlatMapsInMemory(const config::ConfigBase* session) const;

  /// Whether to use the cluster index for filter-based row pruning.
  bool indexEnabled(const config::ConfigBase* session) const;

  /// Whether to collect per-column timing stats (decode/decompress CPU time).
  /// Disabled by default to avoid overhead (~100ns per operation).
  bool readerCollectColumnCpuMetrics(const config::ConfigBase* session) const;

  /// Whether to cache file metadata in the process-wide AsyncDataCache.
  bool fileMetadataCacheEnabled(const config::ConfigBase* session) const;

  /// Whether to pin parsed metadata objects in the reader's metadata cache.
  bool pinFileMetadata(const config::ConfigBase* session) const;

  /// Returns the speculative tail read size in bytes for footer.
  /// ORC/Parquet default to 256KB, Nimble defaults to 8MB. 0 means adaptive.
  uint64_t orcFooterSpeculativeIoSize(const config::ConfigBase* session) const;
  uint64_t parquetFooterSpeculativeIoSize(
      const config::ConfigBase* session) const;
  uint64_t nimbleFooterSpeculativeIoSize(
      const config::ConfigBase* session) const;

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

 protected:
  static constexpr const char* kLegacyPrefix = "hive.";

  // Looks up a config value by 'key', falling back to 'hive.' + key if not
  // found. Used during migration away from redundant 'hive.' prefix in
  // config keys.
  template <typename T>
  T configValue(const std::string& key, const T& defaultValue) const {
    if (auto val = config_->get<T>(key)) {
      return val.value();
    }
    return config_->get<T>(std::string(kLegacyPrefix) + key, defaultValue);
  }

  // Looks up a session property by 'sessionKey' (then 'hive.' + sessionKey),
  // falling back to connector config by 'configKey' (then 'hive.' + configKey).
  // Used during migration away from redundant 'hive.' prefix in property
  // names.
  template <typename T>
  T sessionValue(
      const config::ConfigBase* session,
      const std::string& sessionKey,
      const std::string& configKey,
      const T& defaultValue) const {
    if (auto val = session->get<T>(sessionKey)) {
      return val.value();
    }
    if (auto val = session->get<T>(std::string(kLegacyPrefix) + sessionKey)) {
      return val.value();
    }
    return configValue<T>(configKey, defaultValue);
  }

  std::shared_ptr<const config::ConfigBase> config_;
};

} // namespace facebook::velox::connector::hive
