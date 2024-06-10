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
#include "velox/core/Config.h"
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
HiveConfig::insertExistingPartitionsBehavior(const Config* session) const {
  return stringToInsertExistingPartitionsBehavior(session->get<std::string>(
      kInsertExistingPartitionsBehaviorSession,
      config_->get<std::string>(kInsertExistingPartitionsBehavior, "ERROR")));
}

uint32_t HiveConfig::maxPartitionsPerWriters(const Config* session) const {
  return session->get<uint32_t>(
      kMaxPartitionsPerWritersSession,
      config_->get<uint32_t>(kMaxPartitionsPerWriters, 100));
}

bool HiveConfig::immutablePartitions() const {
  return config_->get<bool>(kImmutablePartitions, false);
}

bool HiveConfig::s3UseVirtualAddressing() const {
  return !config_->get(kS3PathStyleAccess, false);
}

std::string HiveConfig::s3GetLogLevel() const {
  return config_->get(kS3LogLevel, std::string("FATAL"));
}

bool HiveConfig::s3UseSSL() const {
  return config_->get(kS3SSLEnabled, true);
}

bool HiveConfig::s3UseInstanceCredentials() const {
  return config_->get(kS3UseInstanceCredentials, false);
}

std::string HiveConfig::s3Endpoint() const {
  return config_->get(kS3Endpoint, std::string(""));
}

std::optional<std::string> HiveConfig::s3AccessKey() const {
  return static_cast<std::optional<std::string>>(config_->get(kS3AwsAccessKey));
}

std::optional<std::string> HiveConfig::s3SecretKey() const {
  return static_cast<std::optional<std::string>>(config_->get(kS3AwsSecretKey));
}

std::optional<std::string> HiveConfig::s3IAMRole() const {
  return static_cast<std::optional<std::string>>(config_->get(kS3IamRole));
}

std::string HiveConfig::s3IAMRoleSessionName() const {
  return config_->get(kS3IamRoleSessionName, std::string("velox-session"));
}

std::optional<std::string> HiveConfig::s3ConnectTimeout() const {
  return static_cast<std::optional<std::string>>(
      config_->get<std::string>(kS3ConnectTimeout));
}

std::optional<std::string> HiveConfig::s3SocketTimeout() const {
  return static_cast<std::optional<std::string>>(
      config_->get<std::string>(kS3SocketTimeout));
}

std::optional<uint32_t> HiveConfig::s3MaxConnections() const {
  return static_cast<std::optional<std::uint32_t>>(
      config_->get<uint32_t>(kS3MaxConnections));
}

std::optional<uint32_t> HiveConfig::s3MaxAttempts() const {
  return static_cast<std::optional<std::uint32_t>>(
      config_->get<uint32_t>(kS3MaxAttempts));
}

std::optional<std::string> HiveConfig::s3RetryMode() const {
  return static_cast<std::optional<std::string>>(
      config_->get<std::string>(kS3RetryMode));
}

std::string HiveConfig::gcsEndpoint() const {
  return config_->get<std::string>(kGCSEndpoint, std::string(""));
}

std::string HiveConfig::gcsScheme() const {
  return config_->get<std::string>(kGCSScheme, std::string("https"));
}

std::string HiveConfig::gcsCredentials() const {
  return config_->get<std::string>(kGCSCredentials, std::string(""));
}

std::optional<int> HiveConfig::gcsMaxRetryCount() const {
  return static_cast<std::optional<int>>(config_->get<int>(kGCSMaxRetryCount));
}

std::optional<std::string> HiveConfig::gcsMaxRetryTime() const {
  return static_cast<std::optional<std::string>>(
      config_->get<std::string>(kGCSMaxRetryTime));
}

bool HiveConfig::isOrcUseColumnNames(const Config* session) const {
  return session->get<bool>(
      kOrcUseColumnNamesSession, config_->get<bool>(kOrcUseColumnNames, false));
}

bool HiveConfig::isFileColumnNamesReadAsLowerCase(const Config* session) const {
  return session->get<bool>(
      kFileColumnNamesReadAsLowerCaseSession,
      config_->get<bool>(kFileColumnNamesReadAsLowerCase, false));
}

bool HiveConfig::isPartitionPathAsLowerCase(const Config* session) const {
  return session->get<bool>(kPartitionPathAsLowerCaseSession, true);
}

bool HiveConfig::ignoreMissingFiles(const Config* session) const {
  return session->get<bool>(kIgnoreMissingFilesSession, false);
}

int64_t HiveConfig::maxCoalescedBytes() const {
  return config_->get<int64_t>(kMaxCoalescedBytes, 128 << 20);
}

int32_t HiveConfig::maxCoalescedDistanceBytes() const {
  return config_->get<int32_t>(kMaxCoalescedDistanceBytes, 512 << 10);
}

int32_t HiveConfig::prefetchRowGroups() const {
  return config_->get<int32_t>(kPrefetchRowGroups, 1);
}

int32_t HiveConfig::loadQuantum() const {
  return config_->get<int32_t>(kLoadQuantum, 8 << 20);
}

int32_t HiveConfig::numCacheFileHandles() const {
  return config_->get<int32_t>(kNumCacheFileHandles, 20'000);
}

bool HiveConfig::isFileHandleCacheEnabled() const {
  return config_->get<bool>(kEnableFileHandleCache, true);
}

uint64_t HiveConfig::orcWriterMaxStripeSize(const Config* session) const {
  return toCapacity(
      session->get<std::string>(
          kOrcWriterMaxStripeSizeSession,
          config_->get<std::string>(kOrcWriterMaxStripeSize, "64MB")),
      core::CapacityUnit::BYTE);
}

uint64_t HiveConfig::orcWriterMaxDictionaryMemory(const Config* session) const {
  return toCapacity(
      session->get<std::string>(
          kOrcWriterMaxDictionaryMemorySession,
          config_->get<std::string>(kOrcWriterMaxDictionaryMemory, "16MB")),
      core::CapacityUnit::BYTE);
}

bool HiveConfig::orcWriterLinearStripeSizeHeuristics(
    const Config* session) const {
  return session->get<bool>(
      kOrcWriterLinearStripeSizeHeuristicsSession,
      config_->get<bool>(kOrcWriterLinearStripeSizeHeuristics, true));
}

uint64_t HiveConfig::orcWriterMinCompressionSize(const Config* session) const {
  return session->get<uint64_t>(
      kOrcWriterMinCompressionSizeSession,
      config_->get<uint64_t>(kOrcWriterMinCompressionSize, 1024));
}

std::string HiveConfig::writeFileCreateConfig() const {
  return config_->get<std::string>(kWriteFileCreateConfig, "");
}

uint32_t HiveConfig::sortWriterMaxOutputRows(const Config* session) const {
  return session->get<uint32_t>(
      kSortWriterMaxOutputRowsSession,
      config_->get<uint32_t>(kSortWriterMaxOutputRows, 1024));
}

uint64_t HiveConfig::sortWriterMaxOutputBytes(const Config* session) const {
  return toCapacity(
      session->get<std::string>(
          kSortWriterMaxOutputBytesSession,
          config_->get<std::string>(kSortWriterMaxOutputBytes, "10MB")),
      core::CapacityUnit::BYTE);
}

uint64_t HiveConfig::footerEstimatedSize() const {
  return config_->get<uint64_t>(kFooterEstimatedSize, 1UL << 20);
}

uint64_t HiveConfig::filePreloadThreshold() const {
  return config_->get<uint64_t>(kFilePreloadThreshold, 8UL << 20);
}

bool HiveConfig::s3UseProxyFromEnv() const {
  return config_->get<bool>(kS3UseProxyFromEnv, false);
}

uint8_t HiveConfig::parquetWriteTimestampUnit(const Config* session) const {
  const auto unit = session->get<uint8_t>(
      kParquetWriteTimestampUnitSession,
      config_->get<uint8_t>(kParquetWriteTimestampUnit, 9 /*nano*/));
  VELOX_CHECK(
      unit == 0 /*second*/ || unit == 3 /*milli*/ || unit == 6 /*micro*/ ||
          unit == 9,
      "Invalid timestamp unit.");
  return unit;
}

bool HiveConfig::cacheNoRetention(const Config* session) const {
  return session->get<bool>(
      kCacheNoRetentionSession,
      config_->get<bool>(kCacheNoRetention, /*defaultValue=*/false));
}

} // namespace facebook::velox::connector::hive
