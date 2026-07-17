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
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/config/Config.h"
#include "velox/common/config/ConfigProperty.h"

namespace facebook::velox::parquet {

#define VELOX_PARQUET_CONFIG_PROPERTY(               \
    constName, keyStr, CppType, defaultVal, desc)    \
  struct constName##Property {                       \
    using type = CppType;                            \
    static constexpr std::string_view key = keyStr;  \
    static constexpr auto defaultValue = defaultVal; \
    static constexpr const char* description = desc; \
  };                                                 \
  static constexpr std::string_view constName = keyStr;

#define VELOX_PARQUET_CONFIG(                                              \
    constName,                                                             \
    configConstName,                                                       \
    accessorName,                                                          \
    key,                                                                   \
    configKey,                                                             \
    CppType,                                                               \
    defaultVal,                                                            \
    desc)                                                                  \
  VELOX_PARQUET_CONFIG_PROPERTY(constName, key, CppType, defaultVal, desc) \
  static constexpr std::string_view configConstName = configKey;           \
  static CppType accessorName(                                             \
      const config::ConfigBase& connectorConfig,                           \
      const config::ConfigBase& session) {                                 \
    return session.getWithFallback<CppType>(constName, connectorConfig)    \
        .value_or(constName##Property::defaultValue);                      \
  }

/// Defines Parquet reader config and session properties.
class ParquetConfig {
 public:
  VELOX_PARQUET_CONFIG(
      kUseColumnNamesSession,
      kUseColumnNames,
      useColumnNames,
      "use_column_names",
      "use-column-names",
      bool,
      false,
      "Map Parquet table field names to file field names using names, not indices.")

  VELOX_PARQUET_CONFIG(
      kFooterSpeculativeIoSizeSession,
      kFooterSpeculativeIoSize,
      footerSpeculativeIoSize,
      "footer_speculative_io_size",
      "footer-speculative-io-size",
      uint64_t,
      256UL << 10,
      "Speculative tail-read size in bytes for Parquet files.")

  VELOX_PARQUET_CONFIG(
      kAllowInt32NarrowingSession,
      kAllowInt32Narrowing,
      allowInt32Narrowing,
      "allow_int32_narrowing",
      "allow-int32-narrowing",
      bool,
      false,
      "Allow reading INT32 Parquet columns as a narrower integer type.")

  VELOX_PARQUET_CONFIG(
      kDictionaryRowGroupSkippingEnabledSession,
      kDictionaryRowGroupSkippingEnabled,
      dictionaryRowGroupSkippingEnabled,
      "dictionary_row_group_skipping_enabled",
      "dictionary-row-group-skipping-enabled",
      bool,
      false,
      "Skip Parquet row groups whose column-chunk dictionary holds no value "
      "passing the filter. Reads the dictionary page during filtering, which "
      "is a blocking operation that can slow scans of wide columns on remote "
      "storage.")

  static constexpr uint64_t kDefaultFooterMemoryTrackingThreshold =
      std::numeric_limits<uint64_t>::max();
  VELOX_PARQUET_CONFIG(
      kFooterMemoryTrackingThresholdSession,
      kFooterMemoryTrackingThreshold,
      footerMemoryTrackingThreshold,
      "footer_memory_tracking_threshold",
      "footer-memory-tracking-threshold",
      uint64_t,
      kDefaultFooterMemoryTrackingThreshold,
      "Serialized footer size in bytes beyond which the Parquet reader "
      "estimates and reports the deserialized footer's heap footprint to "
      "the memory pool. Defaults to disabled (max uint64).")

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterTimestampUnitSession,
      "writer_timestamp_unit",
      uint8_t,
      9,
      "Timestamp unit used by the Parquet writer. Supported values are 3 "
      "(milliseconds), 6 (microseconds), and 9 (nanoseconds).")
  static constexpr std::string_view kWriterTimestampUnit =
      "writer.timestamp-unit";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterEnableDictionarySession,
      "writer_enable_dictionary",
      bool,
      true,
      "Enable dictionary encoding in the Parquet writer.")
  static constexpr std::string_view kWriterEnableDictionary =
      "writer.enable-dictionary";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterEnableStoreDecimalAsIntegerSession,
      "writer_enable_store_decimal_as_integer",
      bool,
      true,
      "Store short DECIMAL values as integers in the Parquet writer.")
  static constexpr std::string_view kWriterEnableStoreDecimalAsInteger =
      "writer.enable-store-decimal-as-integer";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterDictionaryPageSizeLimitSession,
      "writer_dictionary_page_size_limit",
      std::string_view,
      "1MB",
      "Dictionary page size limit for the Parquet writer.")
  static constexpr std::string_view kWriterDictionaryPageSizeLimit =
      "writer.dictionary-page-size-limit";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterDataPageVersionSession,
      "writer_datapage_version",
      std::string_view,
      "V1",
      "Data page version used by the Parquet writer. Supported values are V1 "
      "and V2.")
  static constexpr std::string_view kWriterDataPageVersion =
      "writer.datapage-version";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterPageSizeSession,
      "writer_page_size",
      std::string_view,
      "1MB",
      "Data page size for the Parquet writer.")
  static constexpr std::string_view kWriterPageSize = "writer.page-size";

  VELOX_PARQUET_CONFIG_PROPERTY(
      kWriterBatchSizeSession,
      "writer_batch_size",
      std::string_view,
      "1024",
      "Write batch size for the Parquet writer.")
  static constexpr std::string_view kWriterBatchSize = "writer.batch-size";

  static constexpr std::string_view kWriterCreatedBy = "writer.created-by";

  // Writer config accessors expect format-scoped configs. Connector prefixes
  // such as "hive.parquet." and "iceberg.parquet." are stripped by
  // ConfigBase::rawConfigsWithPrefix before Parquet sees these keys.
  static std::optional<uint8_t> writerTimestampUnit(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<uint8_t>(
        kWriterTimestampUnitSession, connectorConfig, kWriterTimestampUnit);
  }

  static std::optional<std::string> writerEnableDictionary(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterEnableDictionarySession,
        connectorConfig,
        kWriterEnableDictionary);
  }

  static std::optional<std::string> writerEnableStoreDecimalAsInteger(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterEnableStoreDecimalAsIntegerSession,
        connectorConfig,
        kWriterEnableStoreDecimalAsInteger);
  }

  static std::optional<std::string> writerDictionaryPageSizeLimit(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterDictionaryPageSizeLimitSession,
        connectorConfig,
        kWriterDictionaryPageSizeLimit);
  }

  static std::optional<std::string> writerDataPageVersion(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterDataPageVersionSession, connectorConfig, kWriterDataPageVersion);
  }

  static std::optional<std::string> writerPageSize(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterPageSizeSession, connectorConfig, kWriterPageSize);
  }

  static std::optional<std::string> writerBatchSize(
      const config::ConfigBase& connectorConfig,
      const config::ConfigBase& session) {
    return session.getLegacyWithFallback<std::string>(
        kWriterBatchSizeSession, connectorConfig, kWriterBatchSize);
  }

  static std::optional<std::string> writerCreatedBy(
      const config::ConfigBase& connectorConfig) {
    return connectorConfig.get<std::string>(std::string(kWriterCreatedBy));
  }

  /// Serde parameter key for overriding the Parquet writer timestamp unit.
  /// Accepts numeric values 3 (milliseconds), 6 (microseconds), and 9
  /// (nanoseconds).
  static constexpr std::string_view kWriterSerdeTimestampUnit =
      "parquet.writer.timestamp.unit";

  /// Serde parameter key for overriding the Parquet writer timestamp timezone.
  /// Empty string disables timezone conversion.
  static constexpr std::string_view kWriterSerdeTimestampTimezone =
      "parquet.writer.timestamp.timezone";

  /// Registers Parquet session properties with a config provider.
  static void registerProperties(
      std::vector<config::ConfigProperty>& properties,
      std::string_view sessionPrefix) {
    registerProperty<kUseColumnNamesSessionProperty>(properties, sessionPrefix);
    registerProperty<kFooterSpeculativeIoSizeSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kAllowInt32NarrowingSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kDictionaryRowGroupSkippingEnabledSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kFooterMemoryTrackingThresholdSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterTimestampUnitSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterEnableDictionarySessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterEnableStoreDecimalAsIntegerSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterDictionaryPageSizeLimitSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterDataPageVersionSessionProperty>(
        properties, sessionPrefix);
    registerProperty<kWriterPageSizeSessionProperty>(properties, sessionPrefix);
    registerProperty<kWriterBatchSizeSessionProperty>(
        properties, sessionPrefix);
  }

 private:
  template <typename Property>
  static void registerProperty(
      std::vector<config::ConfigProperty>& properties,
      std::string_view sessionPrefix) {
    properties.push_back({
        std::string(sessionPrefix) + std::string(Property::key),
        config::detail::configPropertyTypeOf<typename Property::type>(),
        config::detail::configPropertyDefaultToString<typename Property::type>(
            Property::defaultValue),
        Property::description,
    });
  }
};

#undef VELOX_PARQUET_CONFIG
#undef VELOX_PARQUET_CONFIG_PROPERTY

} // namespace facebook::velox::parquet
