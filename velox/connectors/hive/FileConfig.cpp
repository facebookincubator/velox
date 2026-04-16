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

#include "velox/connectors/hive/FileConfig.h"

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive {

const std::vector<config::ConfigProperty>& FileConfig::registeredProperties() {
  static const std::vector<config::ConfigProperty> kProperties = [] {
    std::vector<config::ConfigProperty> properties;
#define VELOX_HIVE_CONFIG_REGISTER(constName) \
  config::registerConfigProperty<FileConfig::constName##Property>(properties)

    VELOX_HIVE_CONFIG_REGISTER(kOrcUseColumnNamesSession);
    VELOX_HIVE_CONFIG_REGISTER(kParquetUseColumnNamesSession);
    VELOX_HIVE_CONFIG_REGISTER(kAllowInt32NarrowingSession);
    VELOX_HIVE_CONFIG_REGISTER(kReadTimestampPartitionValueAsLocalTimeSession);
    VELOX_HIVE_CONFIG_REGISTER(kPreserveFlatMapsInMemorySession);
    VELOX_HIVE_CONFIG_REGISTER(kReaderCollectColumnCpuMetricsSession);
    VELOX_HIVE_CONFIG_REGISTER(kOrcFooterSpeculativeIoSizeSession);
    VELOX_HIVE_CONFIG_REGISTER(kParquetFooterSpeculativeIoSizeSession);
    VELOX_HIVE_CONFIG_REGISTER(kNimbleFooterSpeculativeIoSizeSession);
    VELOX_HIVE_CONFIG_REGISTER(kFileColumnNamesReadAsLowerCaseSession);
    VELOX_HIVE_CONFIG_REGISTER(kIgnoreMissingFilesSession);
    VELOX_HIVE_CONFIG_REGISTER(kMaxCoalescedBytesSession);
    VELOX_HIVE_CONFIG_REGISTER(kLoadQuantumSession);
    VELOX_HIVE_CONFIG_REGISTER(kReadStatsBasedFilterReorderDisabledSession);
    VELOX_HIVE_CONFIG_REGISTER(kIndexEnabledSession);
    VELOX_HIVE_CONFIG_REGISTER(kFileMetadataCacheEnabledSession);
    VELOX_HIVE_CONFIG_REGISTER(kPinFileMetadataSession);
    VELOX_HIVE_CONFIG_REGISTER(kMaxCoalescedDistanceSession);
    VELOX_HIVE_CONFIG_REGISTER(kParallelUnitLoadCountSession);
    VELOX_HIVE_CONFIG_REGISTER(kReadTimestampUnitSession);

#undef VELOX_HIVE_CONFIG_REGISTER

    return properties;
  }();
  return kProperties;
}

int32_t FileConfig::maxCoalescedDistanceBytes(
    const config::ConfigBase* session) const {
  const auto distance = config::toCapacity(
      session->get<std::string>(
          kMaxCoalescedDistanceSession,
          config_->get<std::string>(kMaxCoalescedDistance, "512kB")),
      config::CapacityUnit::BYTE);
  VELOX_USER_CHECK_LE(
      distance,
      std::numeric_limits<int32_t>::max(),
      "The max merge distance to combine read requests must be less than 2GB."
      " Got {} bytes.",
      distance);
  return int32_t(distance);
}

int32_t FileConfig::prefetchRowGroups() const {
  return config_->get<int32_t>(kPrefetchRowGroups, 1);
}

size_t FileConfig::parallelUnitLoadCount(
    const config::ConfigBase* session) const {
  auto count = session->get<size_t>(
      kParallelUnitLoadCountSession,
      config_->get<size_t>(kParallelUnitLoadCount, 0));
  VELOX_CHECK_LE(count, 100, "parallelUnitLoadCount too large: {}", count);
  return count;
}

uint64_t FileConfig::filePreloadThreshold() const {
  return config_->get<uint64_t>(kFilePreloadThreshold, 8UL << 20);
}

uint8_t FileConfig::readTimestampUnit(const config::ConfigBase* session) const {
  const auto unit = sessionValue<uint8_t>(
      session, kReadTimestampUnitSession, kReadTimestampUnit, 3 /*milli*/);
  VELOX_CHECK(
      unit == 3 || unit == 6 /*micro*/ || unit == 9 /*nano*/,
      "Invalid timestamp unit.");
  return unit;
}

} // namespace facebook::velox::connector::hive
