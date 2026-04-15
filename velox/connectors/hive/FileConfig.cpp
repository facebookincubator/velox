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

bool FileConfig::isOrcUseColumnNames(const config::ConfigBase* session) const {
  return session->get<bool>(
      kOrcUseColumnNamesSession, configValue<bool>(kOrcUseColumnNames, false));
}

bool FileConfig::isParquetUseColumnNames(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kParquetUseColumnNamesSession,
      configValue<bool>(kParquetUseColumnNames, false));
}

bool FileConfig::allowInt32Narrowing(const config::ConfigBase* session) const {
  return session->get<bool>(
      kAllowInt32NarrowingSession,
      configValue<bool>(kAllowInt32Narrowing, false));
}

bool FileConfig::isFileColumnNamesReadAsLowerCase(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kFileColumnNamesReadAsLowerCaseSession,
      config_->get<bool>(kFileColumnNamesReadAsLowerCase, false));
}

bool FileConfig::ignoreMissingFiles(const config::ConfigBase* session) const {
  return session->get<bool>(kIgnoreMissingFilesSession, false);
}

int64_t FileConfig::maxCoalescedBytes(const config::ConfigBase* session) const {
  return session->get<int64_t>(
      kMaxCoalescedBytesSession,
      config_->get<int64_t>(kMaxCoalescedBytes, 128 << 20)); // 128MB
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

int32_t FileConfig::loadQuantum(const config::ConfigBase* session) const {
  return session->get<int32_t>(
      kLoadQuantumSession, config_->get<int32_t>(kLoadQuantum, 8 << 20));
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

bool FileConfig::readTimestampPartitionValueAsLocalTime(
    const config::ConfigBase* session) const {
  return sessionValue<bool>(
      session,
      kReadTimestampPartitionValueAsLocalTimeSession,
      kReadTimestampPartitionValueAsLocalTime,
      true);
}

bool FileConfig::readStatsBasedFilterReorderDisabled(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kReadStatsBasedFilterReorderDisabledSession,
      config_->get<bool>(kReadStatsBasedFilterReorderDisabled, false));
}

bool FileConfig::preserveFlatMapsInMemory(
    const config::ConfigBase* session) const {
  return sessionValue<bool>(
      session,
      kPreserveFlatMapsInMemorySession,
      kPreserveFlatMapsInMemory,
      false);
}

bool FileConfig::indexEnabled(const config::ConfigBase* session) const {
  return session->get<bool>(
      kIndexEnabledSession, config_->get<bool>(kIndexEnabled, false));
}

bool FileConfig::readerCollectColumnCpuMetrics(
    const config::ConfigBase* session) const {
  return sessionValue<bool>(
      session,
      kReaderCollectColumnCpuMetricsSession,
      kReaderCollectColumnCpuMetrics,
      false);
}

bool FileConfig::fileMetadataCacheEnabled(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kFileMetadataCacheEnabledSession,
      config_->get<bool>(kFileMetadataCacheEnabled, false));
}

bool FileConfig::pinFileMetadata(const config::ConfigBase* session) const {
  return session->get<bool>(
      kPinFileMetadataSession, config_->get<bool>(kPinFileMetadata, false));
}

uint64_t FileConfig::orcFooterSpeculativeIoSize(
    const config::ConfigBase* session) const {
  return session->get<uint64_t>(
      kOrcFooterSpeculativeIoSizeSession,
      configValue<uint64_t>(kOrcFooterSpeculativeIoSize, 256UL << 10));
}

uint64_t FileConfig::parquetFooterSpeculativeIoSize(
    const config::ConfigBase* session) const {
  return session->get<uint64_t>(
      kParquetFooterSpeculativeIoSizeSession,
      configValue<uint64_t>(kParquetFooterSpeculativeIoSize, 256UL << 10));
}

uint64_t FileConfig::nimbleFooterSpeculativeIoSize(
    const config::ConfigBase* session) const {
  return session->get<uint64_t>(
      kNimbleFooterSpeculativeIoSizeSession,
      configValue<uint64_t>(kNimbleFooterSpeculativeIoSize, 8UL << 20));
}

} // namespace facebook::velox::connector::hive
