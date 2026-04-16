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

#include "velox/connectors/hive/HiveConfig.h"
#include "velox/common/config/Config.h"
#include "velox/core/QueryConfig.h"

#include <boost/algorithm/string.hpp>

namespace facebook::velox::connector::hive {

namespace {

HiveConfig::InsertExistingPartitionsBehavior
stringToInsertExistingPartitionsBehavior(const std::string& strValue) {
  auto upperValue = boost::algorithm::to_upper_copy(strValue);
  if (upperValue == "ERROR") {
    return HiveConfig::InsertExistingPartitionsBehavior::kError;
  }
  if (upperValue == "OVERWRITE") {
    return HiveConfig::InsertExistingPartitionsBehavior::kOverwrite;
  }
  VELOX_UNSUPPORTED(
      "Unsupported insert existing partitions behavior: {}.", strValue);
}

} // namespace

// static
std::string HiveConfig::insertExistingPartitionsBehaviorString(
    InsertExistingPartitionsBehavior behavior) {
  switch (behavior) {
    case InsertExistingPartitionsBehavior::kError:
      return "ERROR";
    case InsertExistingPartitionsBehavior::kOverwrite:
      return "OVERWRITE";
    default:
      return fmt::format("UNKNOWN BEHAVIOR {}", static_cast<int>(behavior));
  }
}

HiveConfig::InsertExistingPartitionsBehavior
HiveConfig::insertExistingPartitionsBehavior(
    const config::ConfigBase* session) const {
  return stringToInsertExistingPartitionsBehavior(session->get<std::string>(
      kInsertExistingPartitionsBehaviorSession,
      config_->get<std::string>(kInsertExistingPartitionsBehavior, "ERROR")));
}

uint32_t HiveConfig::maxPartitionsPerWriters(
    const config::ConfigBase* session) const {
  return session->get<uint32_t>(
      kMaxPartitionsPerWritersSession,
      config_->get<uint32_t>(kMaxPartitionsPerWriters, 128));
}

uint32_t HiveConfig::maxBucketCount(const config::ConfigBase* session) const {
  return sessionValue<uint32_t>(
      session, kMaxBucketCountSession, kMaxBucketCount, 100'000);
}

bool HiveConfig::immutablePartitions() const {
  return configValue<bool>(kImmutablePartitions, false);
}

std::string HiveConfig::gcsEndpoint() const {
  return configValue<std::string>(kGcsEndpoint, std::string(""));
}

std::string HiveConfig::gcsCredentialsPath() const {
  return configValue<std::string>(kGcsCredentialsPath, std::string(""));
}

std::optional<int> HiveConfig::gcsMaxRetryCount() const {
  if (auto val = config_->get<int>(kGcsMaxRetryCount)) {
    return val;
  }
  return config_->get<int>(std::string(kLegacyPrefix) + kGcsMaxRetryCount);
}

std::optional<std::string> HiveConfig::gcsMaxRetryTime() const {
  if (auto val = config_->get<std::string>(kGcsMaxRetryTime)) {
    return val;
  }
  return config_->get<std::string>(
      std::string(kLegacyPrefix) + kGcsMaxRetryTime);
}

std::optional<std::string> HiveConfig::gcsAuthAccessTokenProvider() const {
  if (auto val = config_->get<std::string>(kGcsAuthAccessTokenProvider)) {
    return val;
  }
  return config_->get<std::string>(
      std::string(kLegacyPrefix) + kGcsAuthAccessTokenProvider);
}

bool HiveConfig::isPartitionPathAsLowerCase(
    const config::ConfigBase* session) const {
  return session->get<bool>(kPartitionPathAsLowerCaseSession, true);
}

bool HiveConfig::allowNullPartitionKeys(
    const config::ConfigBase* session) const {
  return session->get<bool>(
      kAllowNullPartitionKeysSession,
      config_->get<bool>(kAllowNullPartitionKeys, true));
}

int32_t HiveConfig::numCacheFileHandles() const {
  return config_->get<int32_t>(kNumCacheFileHandles, 20'000);
}

uint64_t HiveConfig::fileHandleExpirationDurationMs() const {
  return config_->get<uint64_t>(kFileHandleExpirationDurationMs, 0);
}

bool HiveConfig::isFileHandleCacheEnabled() const {
  return config_->get<bool>(kEnableFileHandleCache, true);
}

std::string HiveConfig::writeFileCreateConfig() const {
  // Legacy key used snake_case: "hive.write_file_create_config".
  if (auto val = config_->get<std::string>(kWriteFileCreateConfig)) {
    return val.value();
  }
  return config_->get<std::string>("hive.write_file_create_config", "");
}

uint32_t HiveConfig::sortWriterMaxOutputRows(
    const config::ConfigBase* session) const {
  return session->get<uint32_t>(
      kSortWriterMaxOutputRowsSession,
      config_->get<uint32_t>(kSortWriterMaxOutputRows, 1024));
}

uint64_t HiveConfig::sortWriterMaxOutputBytes(
    const config::ConfigBase* session) const {
  return config::toCapacity(
      session->get<std::string>(
          kSortWriterMaxOutputBytesSession,
          config_->get<std::string>(kSortWriterMaxOutputBytes, "10MB")),
      config::CapacityUnit::BYTE);
}

uint64_t HiveConfig::sortWriterFinishTimeSliceLimitMs(
    const config::ConfigBase* session) const {
  return session->get<uint64_t>(
      kSortWriterFinishTimeSliceLimitMsSession,
      config_->get<uint64_t>(kSortWriterFinishTimeSliceLimitMs, 5'000));
}
uint64_t HiveConfig::maxTargetFileSizeBytes(
    const config::ConfigBase* session) const {
  return config::toCapacity(
      session->get<std::string>(
          kMaxTargetFileSizeSession,
          config_->get<std::string>(kMaxTargetFileSize, "0B")),
      config::CapacityUnit::BYTE);
}

uint32_t HiveConfig::maxRowsPerIndexRequest(
    const config::ConfigBase* session) const {
  return sessionValue<uint32_t>(
      session, kMaxRowsPerIndexRequestSession, kMaxRowsPerIndexRequest, 0);
}

std::string HiveConfig::user(const config::ConfigBase* session) const {
  return session->get<std::string>(kUser, config_->get<std::string>(kUser, ""));
}

std::string HiveConfig::source(const config::ConfigBase* session) const {
  return session->get<std::string>(
      kSource, config_->get<std::string>(kSource, ""));
}

std::string HiveConfig::schema(const config::ConfigBase* session) const {
  return session->get<std::string>(
      kSchema, config_->get<std::string>(kSchema, ""));
}

} // namespace facebook::velox::connector::hive
