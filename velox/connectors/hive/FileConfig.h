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

#include <string>
#include "velox/common/base/Exceptions.h"
#include "velox/common/config/Config.h"
#include "velox/common/config/ConfigProperty.h"
#include "velox/connectors/hive/HiveConfigMacrosDefine.h"

namespace facebook::velox::connector::hive {

/// Configuration for file-based data sources. Contains settings for file
/// reading, I/O coalescing, format-specific options, and metadata caching.
/// HiveConfig extends this with Hive-specific settings like partitioning,
/// bucketing, and write-path options.
class FileConfig {
 public:
  static const std::vector<config::ConfigProperty>& registeredProperties();

  // --- VELOX_HIVE_CONFIG_LEGACY properties ---

  VELOX_HIVE_CONFIG_LEGACY(
      kOrcUseColumnNamesSession,
      kOrcUseColumnNames,
      isOrcUseColumnNames,
      "orc_use_column_names",
      "orc.use-column-names",
      bool,
      false,
      "Map ORC table field names to file field names using names, not indices.")

  VELOX_HIVE_CONFIG_LEGACY(
      kParquetUseColumnNamesSession,
      kParquetUseColumnNames,
      isParquetUseColumnNames,
      "parquet_use_column_names",
      "parquet.use-column-names",
      bool,
      false,
      "Map Parquet table field names to file field names using names, not indices.")

  VELOX_HIVE_CONFIG_LEGACY(
      kAllowInt32NarrowingSession,
      kAllowInt32Narrowing,
      allowInt32Narrowing,
      "allow_int32_narrowing",
      "parquet.allow-int32-narrowing",
      bool,
      false,
      "Allow reading INT32 Parquet columns as a narrower integer type.")

  VELOX_HIVE_CONFIG_LEGACY(
      kReadTimestampPartitionValueAsLocalTimeSession,
      kReadTimestampPartitionValueAsLocalTime,
      readTimestampPartitionValueAsLocalTime,
      "reader.timestamp_partition_value_as_local_time",
      "reader.timestamp-partition-value-as-local-time",
      bool,
      true,
      "Read timestamp partition values as local time.")

  VELOX_HIVE_CONFIG_LEGACY(
      kPreserveFlatMapsInMemorySession,
      kPreserveFlatMapsInMemory,
      preserveFlatMapsInMemory,
      "preserve_flat_maps_in_memory",
      "preserve-flat-maps-in-memory",
      bool,
      false,
      "Preserve flat maps in memory as FlatMapVectors.")

  VELOX_HIVE_CONFIG_LEGACY(
      kReaderCollectColumnCpuMetricsSession,
      kReaderCollectColumnCpuMetrics,
      readerCollectColumnCpuMetrics,
      "reader.collect_column_cpu_metrics",
      "reader.collect-column-cpu-metrics",
      bool,
      false,
      "Collect per-column CPU timing stats.")

  VELOX_HIVE_CONFIG_LEGACY(
      kOrcFooterSpeculativeIoSizeSession,
      kOrcFooterSpeculativeIoSize,
      orcFooterSpeculativeIoSize,
      "orc_footer_speculative_io_size",
      "orc.footer-speculative-io-size",
      uint64_t,
      256UL << 10,
      "Speculative tail-read size in bytes for ORC files.")

  VELOX_HIVE_CONFIG_LEGACY(
      kParquetFooterSpeculativeIoSizeSession,
      kParquetFooterSpeculativeIoSize,
      parquetFooterSpeculativeIoSize,
      "parquet_footer_speculative_io_size",
      "parquet.footer-speculative-io-size",
      uint64_t,
      256UL << 10,
      "Speculative tail-read size in bytes for Parquet files.")

  VELOX_HIVE_CONFIG_LEGACY(
      kNimbleFooterSpeculativeIoSizeSession,
      kNimbleFooterSpeculativeIoSize,
      nimbleFooterSpeculativeIoSize,
      "nimble_footer_speculative_io_size",
      "nimble.footer-speculative-io-size",
      uint64_t,
      8UL << 20,
      "Speculative tail-read size in bytes for Nimble files.")

  // --- VELOX_HIVE_CONFIG properties ---

  VELOX_HIVE_CONFIG(
      kFileColumnNamesReadAsLowerCaseSession,
      isFileColumnNamesReadAsLowerCase,
      "file_column_names_read_as_lower_case",
      bool,
      false,
      "Read source file column names as lower case.")
  static constexpr const char* kFileColumnNamesReadAsLowerCase =
      "file-column-names-read-as-lower-case";

  VELOX_HIVE_CONFIG_PROPERTY(
      kIgnoreMissingFilesSession,
      "ignore_missing_files",
      bool,
      false,
      "Ignore missing files during table scan.")
  bool ignoreMissingFiles(const config::ConfigBase* session) const {
    return session->get<bool>(
        kIgnoreMissingFilesSession,
        kIgnoreMissingFilesSessionProperty::defaultValue);
  }

  VELOX_HIVE_CONFIG(
      kMaxCoalescedBytesSession,
      maxCoalescedBytes,
      "max-coalesced-bytes",
      int64_t,
      128 << 20,
      "Maximum coalesced bytes for a read request.")
  static constexpr const char* kMaxCoalescedBytes = "max-coalesced-bytes";

  VELOX_HIVE_CONFIG(
      kLoadQuantumSession,
      loadQuantum,
      "load-quantum",
      int32_t,
      8 << 20,
      "Total size in bytes for a direct coalesce request.")
  static constexpr const char* kLoadQuantum = "load-quantum";

  VELOX_HIVE_CONFIG(
      kReadStatsBasedFilterReorderDisabledSession,
      readStatsBasedFilterReorderDisabled,
      "stats_based_filter_reorder_disabled",
      bool,
      false,
      "Disable stats-based filter reordering.")
  static constexpr const char* kReadStatsBasedFilterReorderDisabled =
      "stats-based-filter-reorder-disabled";

  VELOX_HIVE_CONFIG(
      kIndexEnabledSession,
      indexEnabled,
      "index_enabled",
      bool,
      false,
      "Use cluster index for filter-based row pruning.")
  static constexpr const char* kIndexEnabled = "index-enabled";

  VELOX_HIVE_CONFIG(
      kFileMetadataCacheEnabledSession,
      fileMetadataCacheEnabled,
      "file_metadata_cache_enabled",
      bool,
      false,
      "Cache file metadata in AsyncDataCache.")
  static constexpr const char* kFileMetadataCacheEnabled =
      "file-metadata-cache-enabled";

  VELOX_HIVE_CONFIG(
      kPinFileMetadataSession,
      pinFileMetadata,
      "pin_file_metadata",
      bool,
      false,
      "Pin parsed metadata objects in reader cache.")
  static constexpr const char* kPinFileMetadata = "pin-file-metadata";

  // --- VELOX_HIVE_CONFIG_PROPERTY properties ---

  VELOX_HIVE_CONFIG_PROPERTY(
      kMaxCoalescedDistanceSession,
      "orc_max_merge_distance",
      std::string,
      "512kB",
      "Maximum merge distance to combine read requests.")
  static constexpr const char* kMaxCoalescedDistance = "max-coalesced-distance";

  VELOX_HIVE_CONFIG_PROPERTY(
      kParallelUnitLoadCountSession,
      "parallel_unit_load_count",
      size_t,
      0,
      "Number of units to load in parallel. 0 disables.")
  static constexpr const char* kParallelUnitLoadCount =
      "parallel-unit-load-count";

  VELOX_HIVE_CONFIG_PROPERTY(
      kReadTimestampUnitSession,
      "reader.timestamp_unit",
      uint8_t,
      3,
      "Unit for reading timestamps (0=second, 3=millisecond, 6=microsecond, 9=nanosecond).")
  static constexpr const char* kReadTimestampUnit = "reader.timestamp-unit";

  // --- Server-only properties (no macro) ---

  /// The number of prefetch rowgroups.
  static constexpr const char* kPrefetchRowGroups = "prefetch-rowgroups";

  /// The threshold of file size in bytes when the whole file is fetched with
  /// meta data together. Optimization to decrease the small IO requests.
  static constexpr const char* kFilePreloadThreshold = "file-preload-threshold";

  explicit FileConfig(std::shared_ptr<const config::ConfigBase> config) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for FileConfig initialization");
    config_ = std::move(config);
  }

  virtual ~FileConfig() = default;

  int32_t maxCoalescedDistanceBytes(const config::ConfigBase* session) const;

  int32_t prefetchRowGroups() const;

  size_t parallelUnitLoadCount(const config::ConfigBase* session) const;

  uint64_t filePreloadThreshold() const;

  // Returns the timestamp unit used when reading timestamps from files.
  uint8_t readTimestampUnit(const config::ConfigBase* session) const;

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

#include "velox/connectors/hive/HiveConfigMacrosUndef.h"
