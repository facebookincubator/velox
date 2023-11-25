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
#include <core/QueryConfig.h>
#include "velox/core/Config.h"
#include "velox/core/QueryConfig.h"

#include <boost/algorithm/string.hpp>

namespace facebook::velox::connector::hive {

InsertExistingPartitionsBehavior stringToInsertExistingPartitionsBehavior(
    const std::string& strValue) {
  auto upperValue = boost::algorithm::to_upper_copy(strValue);
  if (upperValue == "ERROR") {
    return InsertExistingPartitionsBehavior::kError;
  }
  if (upperValue == "OVERWRITE") {
    return InsertExistingPartitionsBehavior::kOverwrite;
  }
  VELOX_UNSUPPORTED(
      "Unsupported insert existing partitions behavior: {}.", strValue);
}

std::string insertExistingPartitionsBehaviorString(
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

InsertExistingPartitionsBehavior HiveConfig::insertExistingPartitionsBehavior()
    const {
  return insertExistingPartitionsBehavior_;
}

uint32_t HiveConfig::maxPartitionsPerWriters() const {
  return maxPartitionsPerWriters_;
}

bool HiveConfig::immutablePartitions() const {
  return immutablePartitions_;
}

bool HiveConfig::s3UseVirtualAddressing() const {
  return s3PathStyleAccess_;
}

std::string HiveConfig::s3GetLogLevel() const {
  return s3LogLevel_;
}

bool HiveConfig::s3UseSSL() const {
  return s3SSLEnabled_;
}

bool HiveConfig::s3UseInstanceCredentials() const {
  return s3UseInstanceCredentials_;
}

std::string HiveConfig::s3Endpoint() const {
  return s3Endpoint_;
}

std::optional<std::string> HiveConfig::s3AccessKey() const {
  return s3AwsAccessKey_;
}

std::optional<std::string> HiveConfig::s3SecretKey() const {
  return s3AwsSecretKey_;
}

std::optional<std::string> HiveConfig::s3IAMRole() const {
  return s3IamRole_;
}

std::string HiveConfig::s3IAMRoleSessionName() const {
  return s3IamRoleSessionName_;
}

std::string HiveConfig::gcsEndpoint() const {
  return GCSEndpoint_;
}

std::string HiveConfig::gcsScheme() const {
  return GCSScheme_;
}

std::string HiveConfig::gcsCredentials() const {
  return GCSCredentials_;
}

bool HiveConfig::isOrcUseColumnNames() const {
  return orcUseColumnNames_;
}

bool HiveConfig::isFileColumnNamesReadAsLowerCase() const {
  return fileColumnNamesReadAsLowerCase_;
}

int64_t HiveConfig::maxCoalescedBytes() const {
  return maxCoalescedBytes_;
}

int32_t HiveConfig::maxCoalescedDistanceBytes() const {
  return maxCoalescedDistanceBytes_;
}

int32_t HiveConfig::numCacheFileHandles() const {
  return numCacheFileHandles_;
}

bool HiveConfig::isFileHandleCacheEnabled() const {
  return enableFileHandleCache_;
}

uint32_t HiveConfig::sortWriterMaxOutputRows() const {
  return sortWriterMaxOutputRows_;
}

uint64_t HiveConfig::sortWriterMaxOutputBytes() const {
  return sortWriterMaxOutputBytes_;
}

uint64_t HiveConfig::getOrcWriterMaxStripeSize() const {
  return orcWriterMaxStripeSize_;
}

uint64_t HiveConfig::getOrcWriterMaxDictionaryMemory() const {
  return orcWriterMaxDictionaryMemory_;
}

// static
void HiveConfig::setInsertExistingPartitionsBehavior(
    HiveConfig* hiveConfig,
    std::string insertExistingPartitionsBehavior) {
  hiveConfig->insertExistingPartitionsBehavior_ =
      stringToInsertExistingPartitionsBehavior(
          insertExistingPartitionsBehavior);
}

// static
void HiveConfig::setMaxPartitionsPerWriters(
    HiveConfig* hiveConfig,
    std::string maxPartitionsPerWriters) {
  hiveConfig->maxPartitionsPerWriters_ =
      folly::to<uint32_t>(maxPartitionsPerWriters);
}

// static
void HiveConfig::setImmutablePartitions(
    HiveConfig* hiveConfig,
    std::string immutablePartitions) {
  hiveConfig->immutablePartitions_ = folly::to<bool>(immutablePartitions);
}

// static
void HiveConfig::setS3PathStyleAccess(
    HiveConfig* hiveConfig,
    std::string s3PathStyleAccess) {
  hiveConfig->s3PathStyleAccess_ = folly::to<bool>(s3PathStyleAccess);
}

// static
void HiveConfig::setS3LogLevel(HiveConfig* hiveConfig, std::string s3LogLevel) {
  hiveConfig->s3LogLevel_ = folly::to<std::string>(s3LogLevel);
}

// static
void HiveConfig::setS3SSLEnabled(
    HiveConfig* hiveConfig,
    std::string s3SSLEnabled) {
  hiveConfig->s3SSLEnabled_ = folly::to<bool>(s3SSLEnabled);
}

// static
void HiveConfig::setS3UseInstanceCredentials(
    HiveConfig* hiveConfig,
    std::string s3UseInstanceCredentials) {
  hiveConfig->s3UseInstanceCredentials_ =
      folly::to<bool>(s3UseInstanceCredentials);
}

// static
void HiveConfig::setS3Endpoint(HiveConfig* hiveConfig, std::string s3Endpoint) {
  hiveConfig->s3Endpoint_ = folly::to<std::string>(s3Endpoint);
}

// static
void HiveConfig::setS3AwsAccessKey(
    HiveConfig* hiveConfig,
    std::string s3AwsAccessKey) {
  hiveConfig->s3AwsAccessKey_ =
      std::optional(folly::to<std::string>(s3AwsAccessKey));
}

// static
void HiveConfig::setS3AwsSecretKey(
    HiveConfig* hiveConfig,
    std::string s3AwsSecretKey) {
  hiveConfig->s3AwsSecretKey_ =
      std::optional(folly::to<std::string>(s3AwsSecretKey));
}

// static
void HiveConfig::setS3IamRole(HiveConfig* hiveConfig, std::string s3IamRole) {
  hiveConfig->s3IamRole_ = std::optional(folly::to<std::string>(s3IamRole));
}

// static
void HiveConfig::setS3IamRoleSessionName(
    HiveConfig* hiveConfig,
    std::string s3IamRoleSessionName) {
  hiveConfig->s3IamRoleSessionName_ =
      folly::to<std::string>(s3IamRoleSessionName);
}

// static
void HiveConfig::setGCSEndpoint(
    HiveConfig* hiveConfig,
    std::string GCSEndpoint) {
  hiveConfig->GCSEndpoint_ = folly::to<std::string>(GCSEndpoint);
}

// static
void HiveConfig::setGCSScheme(HiveConfig* hiveConfig, std::string GCSScheme) {
  hiveConfig->GCSScheme_ = folly::to<std::string>(GCSScheme);
}

// static
void HiveConfig::setGCSCredentials(
    HiveConfig* hiveConfig,
    std::string GCSCredentials) {
  hiveConfig->GCSCredentials_ = folly::to<std::string>(GCSCredentials);
}

// static
void HiveConfig::setOrcUseColumnNames(
    HiveConfig* hiveConfig,
    std::string orcUseColumnNames) {
  hiveConfig->orcUseColumnNames_ = folly::to<bool>(orcUseColumnNames);
}

// static
void HiveConfig::setFileColumnNamesReadAsLowerCase(
    HiveConfig* hiveConfig,
    std::string fileColumnNamesReadAsLowerCase) {
  hiveConfig->fileColumnNamesReadAsLowerCase_ =
      folly::to<bool>(fileColumnNamesReadAsLowerCase);
}

// static
void HiveConfig::setMaxCoalescedBytes(
    HiveConfig* hiveConfig,
    std::string maxCoalescedBytes) {
  hiveConfig->maxCoalescedBytes_ = folly::to<int64_t>(maxCoalescedBytes);
}

// static
void HiveConfig::setMaxCoalescedDistanceBytes(
    HiveConfig* hiveConfig,
    std::string maxCoalescedDistanceBytes) {
  hiveConfig->maxCoalescedDistanceBytes_ =
      folly::to<int32_t>(maxCoalescedDistanceBytes);
}

// static
void HiveConfig::setNumCacheFileHandles(
    HiveConfig* hiveConfig,
    std::string numCacheFileHandles) {
  hiveConfig->numCacheFileHandles_ = folly::to<int32_t>(numCacheFileHandles);
}

// static
void HiveConfig::setEnableFileHandleCache(
    HiveConfig* hiveConfig,
    std::string enableFileHandleCache) {
  hiveConfig->enableFileHandleCache_ = folly::to<bool>(enableFileHandleCache);
}

// static
void HiveConfig::setSortWriterMaxOutputRows(
    HiveConfig* hiveConfig,
    std::string sortWriterMaxOutputRows) {
  hiveConfig->sortWriterMaxOutputRows_ =
      folly::to<int32_t>(sortWriterMaxOutputRows);
}

// static
void HiveConfig::setSortWriterMaxOutputBytes(
    HiveConfig* hiveConfig,
    std::string sortWriterMaxOutputBytes) {
  hiveConfig->sortWriterMaxOutputBytes_ =
      folly::to<uint64_t>(sortWriterMaxOutputBytes);
}

// static
void HiveConfig::setOrcWriterMaxStripeSize(
    HiveConfig* hiveConfig,
    std::string orcWriterMaxStripeSize) {
  hiveConfig->orcWriterMaxStripeSize_ =
      folly::to<uint64_t>(orcWriterMaxStripeSize);
}

// static
void HiveConfig::setOrcWriterMaxDictionaryMemory(
    HiveConfig* hiveConfig,
    std::string orcWriterMaxDictionaryMemory) {
  hiveConfig->orcWriterMaxDictionaryMemory_ =
      folly::to<uint64_t>(orcWriterMaxDictionaryMemory);
}

} // namespace facebook::velox::connector::hive
