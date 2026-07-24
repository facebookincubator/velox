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
#include "velox/dwio/common/FormatConfig.h"

namespace facebook::velox::parquet {

/// Defines Parquet reader config and session properties.
class ParquetConfig {
 public:
  VELOX_FORMAT_CONFIG(
      kAllowInt32NarrowingSession,
      kAllowInt32Narrowing,
      allowInt32Narrowing,
      "allow_int32_narrowing",
      "allow-int32-narrowing",
      bool,
      false,
      "Allow reading INT32 Parquet columns as a narrower integer type.")

  static constexpr uint64_t kDefaultFooterMemoryTrackingThreshold =
      std::numeric_limits<uint64_t>::max();
  VELOX_FORMAT_CONFIG(
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

  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterTimestampUnitSession,
      kWriterTimestampUnit,
      "writer_timestamp_unit",
      "writer.timestamp-unit",
      uint8_t,
      9,
      "Timestamp unit used by the Parquet writer. Supported values are 3 "
      "(milliseconds), 6 (microseconds), and 9 (nanoseconds).")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterEnableDictionarySession,
      kWriterEnableDictionary,
      "writer_enable_dictionary",
      "writer.enable-dictionary",
      bool,
      true,
      "Enable dictionary encoding in the Parquet writer.")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterEnableStoreDecimalAsIntegerSession,
      kWriterEnableStoreDecimalAsInteger,
      "writer_enable_store_decimal_as_integer",
      "writer.enable-store-decimal-as-integer",
      bool,
      true,
      "Store short DECIMAL values as integers in the Parquet writer.")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterDictionaryPageSizeLimitSession,
      kWriterDictionaryPageSizeLimit,
      "writer_dictionary_page_size_limit",
      "writer.dictionary-page-size-limit",
      std::string_view,
      "1MB",
      "Dictionary page size limit for the Parquet writer.")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterDataPageVersionSession,
      kWriterDataPageVersion,
      "writer_datapage_version",
      "writer.datapage-version",
      std::string_view,
      "V1",
      "Data page version used by the Parquet writer. Supported values are V1 "
      "and V2.")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterPageSizeSession,
      kWriterPageSize,
      "writer_page_size",
      "writer.page-size",
      std::string_view,
      "1MB",
      "Data page size for the Parquet writer.")
  VELOX_FORMAT_CONFIG_PROPERTY(
      kWriterBatchSizeSession,
      kWriterBatchSize,
      "writer_batch_size",
      "writer.batch-size",
      std::string_view,
      "1024",
      "Write batch size for the Parquet writer.")
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
    dwio::common::registerFormatConfigProperty<
        kAllowInt32NarrowingSessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kFooterMemoryTrackingThresholdSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kWriterTimestampUnitSessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kWriterEnableDictionarySessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kWriterEnableStoreDecimalAsIntegerSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kWriterDictionaryPageSizeLimitSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<
        kWriterDataPageVersionSessionProperty>(properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<kWriterPageSizeSessionProperty>(
        properties, sessionPrefix);
    dwio::common::registerFormatConfigProperty<kWriterBatchSizeSessionProperty>(
        properties, sessionPrefix);
  }
};

} // namespace facebook::velox::parquet
