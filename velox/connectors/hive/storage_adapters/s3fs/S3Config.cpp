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

#include "velox/connectors/hive/storage_adapters/s3fs/S3Config.h"

#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <map>

namespace facebook::velox::filesystems {

static constexpr size_t kMinimumMultipartMinPartSize = 5U << 20; // 5MB
static constexpr size_t kMaximumMultipartMinPartSize = 5U << 30; // 5GB

namespace {

std::optional<std::string> normalizedS3ConfigKey(std::string_view key) {
  if (key.rfind(S3Config::kS3Prefix, 0) == 0) {
    return std::nullopt;
  }

  const auto pos = key.find(S3Config::kS3Prefix);
  if (pos == std::string_view::npos || pos == 0 || key[pos - 1] != '.') {
    return std::nullopt;
  }

  return std::string(key.substr(pos));
}

std::optional<std::string> getConfigValue(
    const std::shared_ptr<const config::ConfigBase>& properties,
    std::string_view configKey) {
  return static_cast<std::optional<std::string>>(
      properties->get<std::string>(std::string(configKey)));
}

} // namespace

std::shared_ptr<const config::ConfigBase> S3Config::normalizeConfig(
    std::shared_ptr<const config::ConfigBase> config) {
  if (config == nullptr) {
    return std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>());
  }

  auto configs = config->rawConfigsCopy();
  std::map<std::string, std::string> connectorScopedS3Configs;
  for (const auto& [key, value] : config->rawConfigsCopy()) {
    // Connector configs may use keys scoped by connector name, e.g.
    // hive.s3.endpoint or iceberg.s3.bucket.my-bucket.endpoint. The S3 storage
    // adapter consumes connector-agnostic keys under s3.*, so map them here.
    auto normalizedKey = normalizedS3ConfigKey(key);
    if (!normalizedKey.has_value()) {
      continue;
    }
    auto [it, inserted] =
        connectorScopedS3Configs.emplace(normalizedKey.value(), value);
    VELOX_CHECK(
        inserted || it->second == value,
        "Multiple connector-scoped S3 configs map to '{}' with different "
        "values. Pass a connector-specific config object to disambiguate.",
        normalizedKey.value());
  }
  for (auto& [key, value] : connectorScopedS3Configs) {
    configs.insert_or_assign(std::move(key), std::move(value));
  }
  return std::make_shared<config::ConfigBase>(std::move(configs));
}

std::string S3Config::cacheKey(
    std::string_view bucket,
    std::shared_ptr<const config::ConfigBase> config) {
  auto normalizedConfig = normalizeConfig(std::move(config));
  auto bucketEndpoint = bucketConfigKey(Keys::kEndpoint, bucket);
  if (normalizedConfig->valueExists(bucketEndpoint)) {
    return fmt::format(
        "{}-{}",
        normalizedConfig->get<std::string>(bucketEndpoint).value(),
        bucket);
  }

  auto baseEndpoint = baseConfigKey(Keys::kEndpoint);
  if (normalizedConfig->valueExists(baseEndpoint)) {
    return fmt::format(
        "{}-{}",
        normalizedConfig->get<std::string>(baseEndpoint).value(),
        bucket);
  }
  return std::string(bucket);
}

S3Config::S3Config(
    std::string_view bucket,
    const std::shared_ptr<const config::ConfigBase> properties)
    : bucket_(bucket) {
  const auto normalizedProperties = normalizeConfig(properties);
  for (int key = static_cast<int>(Keys::kBegin);
       key < static_cast<int>(Keys::kEnd);
       key++) {
    auto s3Key = static_cast<Keys>(key);
    auto value = S3Config::configTraits().find(s3Key)->second;
    auto configDefault = value.second;

    // Set bucket S3 config "s3.bucket.*" if present.
    auto bucketConfig = bucketConfigKey(s3Key, bucket);
    auto configVal = getConfigValue(normalizedProperties, bucketConfig);
    if (configVal.has_value()) {
      config_[s3Key] = configVal.value();
    } else {
      // Set base config "s3.*" if present.
      auto baseConfig = baseConfigKey(s3Key);
      configVal = getConfigValue(normalizedProperties, baseConfig);
      if (configVal.has_value()) {
        config_[s3Key] = configVal.value();
      } else {
        config_[s3Key] = configDefault;
      }
    }
  }
  payloadSigningPolicy_ =
      normalizedProperties->get<std::string>(kS3PayloadSigningPolicy, "Never");

  VELOX_CHECK_GE(
      minPartSize(),
      kMinimumMultipartMinPartSize,
      "The min-part-size S3 configuration must exceed 5MB.");
  VELOX_CHECK_LE(
      minPartSize(),
      kMaximumMultipartMinPartSize,
      "The min-part-size S3 configuration must not exceed 5GB.");
}

std::optional<std::string> S3Config::endpointRegion() const {
  auto region = config_.find(Keys::kEndpointRegion)->second;
  if (!region.has_value()) {
    // If region is not set, try inferring from the endpoint value for AWS
    // endpoints.
    auto endpointValue = endpoint();
    if (endpointValue.has_value()) {
      region = parseAWSStandardRegionName(endpointValue.value());
    }
  }
  return region;
}

size_t S3Config::minPartSize() const {
  return config::toCapacity(
      config_.find(Keys::kMultipartMinPartSize)->second.value(),
      config::CapacityUnit::BYTE);
}

} // namespace facebook::velox::filesystems
