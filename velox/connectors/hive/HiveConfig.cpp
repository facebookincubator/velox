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

#include <boost/algorithm/string.hpp>
#include "velox/common/EnumDefine.h"
#include "velox/common/config/Config.h"
#include "velox/dwio/common/Options.h"

namespace facebook::velox::connector::hive {

namespace {
const auto& insertExistingPartitionsBehaviorNames() {
  static const folly::
      F14FastMap<HiveConfig::InsertExistingPartitionsBehavior, std::string_view>
          kNames = {
              {HiveConfig::InsertExistingPartitionsBehavior::kError, "ERROR"},
              {HiveConfig::InsertExistingPartitionsBehavior::kOverwrite,
               "OVERWRITE"},
          };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    HiveConfig,
    InsertExistingPartitionsBehavior,
    insertExistingPartitionsBehaviorNames)

const std::vector<config::ConfigProperty>& HiveConfig::registeredProperties() {
  static const std::vector<config::ConfigProperty> kProperties = [] {
    // Start with FileConfig properties.
    auto properties = connector::FileConfig::registeredProperties();
#define VELOX_FILE_CONFIG_REGISTER(constName) \
  config::registerConfigProperty<HiveConfig::constName##Property>(properties)

    VELOX_FILE_CONFIG_REGISTER(kInsertExistingPartitionsBehaviorSession);
    VELOX_FILE_CONFIG_REGISTER(kSortWriterMaxOutputBytesSession);
    VELOX_FILE_CONFIG_REGISTER(kParquetMaxTargetFileSizeSession);
    VELOX_FILE_CONFIG_REGISTER(kOrcMaxTargetFileSizeSession);
    VELOX_FILE_CONFIG_REGISTER(kNimbleMaxTargetFileSizeSession);
    VELOX_FILE_CONFIG_REGISTER(kMaxPartitionsPerWritersSession);
    VELOX_FILE_CONFIG_REGISTER(kAllowNullPartitionKeysSession);
    VELOX_FILE_CONFIG_REGISTER(kPartitionPathAsLowerCaseSession);
    VELOX_FILE_CONFIG_REGISTER(kSortWriterMaxOutputRowsSession);
    VELOX_FILE_CONFIG_REGISTER(kSortWriterFinishTimeSliceLimitMsSession);
    VELOX_FILE_CONFIG_REGISTER(kUser);
    VELOX_FILE_CONFIG_REGISTER(kSource);
    VELOX_FILE_CONFIG_REGISTER(kSchema);
    VELOX_FILE_CONFIG_REGISTER(kMaxBucketCountSession);
    VELOX_FILE_CONFIG_REGISTER(kMaxRowsPerIndexRequestSession);

#undef VELOX_FILE_CONFIG_REGISTER

    registerFormatProperties(properties);
    return properties;
  }();
  return kProperties;
}

HiveConfig::InsertExistingPartitionsBehavior
HiveConfig::insertExistingPartitionsBehavior(
    const config::ConfigBase* session) const {
  const auto value = session->get<std::string>(
      kInsertExistingPartitionsBehaviorSession,
      config_->get<std::string>(kInsertExistingPartitionsBehavior, "ERROR"));
  const auto behavior = tryToInsertExistingPartitionsBehavior(
      boost::algorithm::to_upper_copy(value));
  VELOX_USER_CHECK(
      behavior.has_value(),
      "Unsupported insert existing partitions behavior: {}",
      value);
  return behavior.value();
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

std::string HiveConfig::writeFileCreateConfig() const {
  // Legacy key used snake_case: "hive.write_file_create_config".
  if (auto val = config_->get<std::string>(kWriteFileCreateConfig)) {
    return val.value();
  }
  return config_->get<std::string>("hive.write_file_create_config", "");
}

uint64_t HiveConfig::sortWriterMaxOutputBytes(
    const config::ConfigBase* session) const {
  return config::toCapacity(
      session->get<std::string>(
          kSortWriterMaxOutputBytesSession,
          config_->get<std::string>(kSortWriterMaxOutputBytes, "10MB")),
      config::CapacityUnit::BYTE);
}

uint64_t HiveConfig::maxTargetFileSizeBytes(
    dwio::common::FileFormat fileFormat,
    const config::ConfigBase* session) const {
  const char* sessionKey;
  const char* configKey;
  const char* defaultValue;
  switch (fileFormat) {
    case dwio::common::FileFormat::PARQUET:
      sessionKey = kParquetMaxTargetFileSizeSession;
      configKey = kParquetMaxTargetFileSize;
      defaultValue = kParquetMaxTargetFileSizeSessionProperty::defaultValue;
      break;
    case dwio::common::FileFormat::DWRF:
    case dwio::common::FileFormat::ORC:
      sessionKey = kOrcMaxTargetFileSizeSession;
      configKey = kOrcMaxTargetFileSize;
      defaultValue = kOrcMaxTargetFileSizeSessionProperty::defaultValue;
      break;
    case dwio::common::FileFormat::NIMBLE:
      sessionKey = kNimbleMaxTargetFileSizeSession;
      configKey = kNimbleMaxTargetFileSize;
      defaultValue = kNimbleMaxTargetFileSizeSessionProperty::defaultValue;
      break;
    default:
      return 0;
  }

  auto maxTargetFileSize = session->get<std::string>(sessionKey);
  if (!maxTargetFileSize.has_value()) {
    maxTargetFileSize = config_->get<std::string>(configKey);
  }
  return config::toCapacity(
      maxTargetFileSize.value_or(defaultValue), config::CapacityUnit::BYTE);
}

} // namespace facebook::velox::connector::hive
