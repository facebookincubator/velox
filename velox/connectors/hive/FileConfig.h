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

#include <limits>
#include <string>
#include <string_view>
#include <utility>
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
      kNimbleFooterSpeculativeIoSizeSession,
      kNimbleFooterSpeculativeIoSize,
      nimbleFooterSpeculativeIoSize,
      "nimble_footer_speculative_io_size",
      "nimble.footer-speculative-io-size",
      uint64_t,
      8UL << 20,
      "Speculative tail-read size in bytes for Nimble files.")

  VELOX_HIVE_CONFIG_LEGACY(
      kNimbleStringDecoderZeroCopySession,
      kNimbleStringDecoderZeroCopy,
      nimbleStringDecoderZeroCopy,
      "nimble_string_decoder_zero_copy",
      "nimble.string-decoder-zero-copy",
      bool,
      false,
      "Enable zero-copy string decoding in Nimble selective reader.")

  VELOX_HIVE_CONFIG_LEGACY(
      kNimblePreserveDictionaryEncodingSession,
      kNimblePreserveDictionaryEncoding,
      nimblePreserveDictionaryEncoding,
      "nimble_preserve_dictionary_encoding",
      "nimble.preserve-dictionary-encoding",
      bool,
      false,
      "Preserve dictionary encoding for Nimble string column reads.")

  VELOX_HIVE_CONFIG_LEGACY(
      kNimbleLazyColumnIoSession,
      kNimbleLazyColumnIo,
      nimbleLazyColumnIo,
      "nimble_lazy_column_io",
      "nimble.lazy-column-io",
      bool,
      false,
      "Defer I/O for projected columns without pushdown filters, remaining filters, or transforms.")

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
      kCacheMetadataSession,
      cacheMetadata,
      "cache_metadata",
      bool,
      false,
      "Cache file metadata in AsyncDataCache.")
  static constexpr const char* kCacheMetadata = "cache-metadata";

  VELOX_HIVE_CONFIG(
      kPinMetadataSession,
      pinMetadata,
      "pin_metadata",
      bool,
      false,
      "Pin parsed metadata objects in reader cache.")
  static constexpr const char* kPinMetadata = "pin-metadata";

  VELOX_HIVE_CONFIG(
      kCacheIndexSession,
      cacheIndex,
      "cache_index",
      bool,
      false,
      "Cache index data in AsyncDataCache.")
  static constexpr const char* kCacheIndex = "cache-index";

  VELOX_HIVE_CONFIG(
      kPinIndexSession,
      pinIndex,
      "pin_index",
      bool,
      false,
      "Pin parsed index objects in reader cache.")
  static constexpr const char* kPinIndex = "pin-index";

  VELOX_HIVE_CONFIG(
      kSelectiveNimbleReaderEnabledSession,
      selectiveNimbleReaderEnabled,
      "selective_nimble_reader_enabled",
      bool,
      true,
      "Enable selective Nimble reader.")

  VELOX_HIVE_CONFIG(
      kNimbleDirectBufferedInputEnabledSession,
      nimbleDirectBufferedInputEnabled,
      "nimble_direct_buffered_input_enabled",
      bool,
      false,
      "Use DirectBufferedInput for Nimble reads. Loads streams in quanta "
      "(loadQuantum-sized chunks) instead of full-stream reads. The first "
      "quantum per stream is always issued; subsequent quanta are loaded on "
      "demand. Small streams that fit in one quantum see no reduction. "
      "Streams coalesced with eager columns may also be loaded early.")
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
      "Unit for reading timestamps (3=millisecond, 6=microsecond, 9=nanosecond).")
  static constexpr const char* kReadTimestampUnit = "reader.timestamp-unit";

  // --- Server-only properties (no macro) ---

  /// The number of prefetch rowgroups.
  static constexpr const char* kPrefetchRowGroups = "prefetch-rowgroups";

  /// The threshold of file size in bytes when the whole file is fetched with
  /// meta data together. Optimization to decrease the small IO requests.
  static constexpr const char* kFilePreloadThreshold = "file-preload-threshold";

  explicit FileConfig(
      std::shared_ptr<const config::ConfigBase> config,
      std::string connectorConfigPrefix)
      : connectorConfigPrefix_(std::move(connectorConfigPrefix)) {
    VELOX_CHECK_NOT_NULL(
        config, "Config is null for FileConfig initialization");
    config_ = std::move(config);
  }

  virtual ~FileConfig() = default;

  /// Returns the connector-owned config prefix. For example, "hive" returns
  /// "hive.".
  static std::string makeConnectorConfigPrefix(std::string_view connectorName) {
    return std::string(connectorName) + ".";
  }

  int32_t maxCoalescedDistanceBytes(const config::ConfigBase* session) const;

  int32_t prefetchRowGroups() const;

  size_t parallelUnitLoadCount(const config::ConfigBase* session) const;

  uint64_t filePreloadThreshold() const;

  // Returns the timestamp unit used when reading timestamps from files.
  uint8_t readTimestampUnit(const config::ConfigBase* session) const;

  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

  std::string_view connectorConfigPrefix() const {
    return connectorConfigPrefix_;
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
  // Prefix used to extract connector-scoped format configs, e.g. "hive.".
  const std::string connectorConfigPrefix_;
};

} // namespace facebook::velox::connector::hive

#include "velox/connectors/hive/HiveConfigMacrosUndef.h"
