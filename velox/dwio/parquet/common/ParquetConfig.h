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
  static constexpr std::string_view kPrefix = "parquet.";
  static constexpr std::string_view kSessionPrefix = "parquet_";

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

  /// Registers Parquet reader session properties with a config provider.
  static void registerProperties(
      std::vector<config::ConfigProperty>& properties) {
    registerProperty<kUseColumnNamesSessionProperty>(properties);
    registerProperty<kFooterSpeculativeIoSizeSessionProperty>(properties);
    registerProperty<kAllowInt32NarrowingSessionProperty>(properties);
    registerProperty<kFooterMemoryTrackingThresholdSessionProperty>(properties);
  }

 private:
  template <typename Property>
  static void registerProperty(
      std::vector<config::ConfigProperty>& properties) {
    properties.push_back({
        std::string(kSessionPrefix) + std::string(Property::key),
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
