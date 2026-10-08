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

namespace facebook::velox::filesystems {

static constexpr size_t kMinimumMultipartMinPartSize = 5U << 20; // 5MB
static constexpr size_t kMaximumMultipartMinPartSize = 5U << 30; // 5GB

std::optional<std::string> S3Config::configValue(
    const config::ConfigBase& config,
    std::string_view configKey) {
  if (auto value = config.get<std::string>(std::string(configKey))) {
    return value;
  }
  // Fall back to the deprecated "hive.s3." prefix.
  VELOX_CHECK(
      configKey.substr(0, std::string_view(kS3Prefix).size()) == kS3Prefix,
      "S3 config key must be prefixed with '{}': {}",
      kS3Prefix,
      configKey);
  const auto suffix = configKey.substr(std::string_view(kS3Prefix).size());
  return config.get<std::string>(
      fmt::format("{}{}", kS3DeprecatedPrefix, suffix));
}

std::string S3Config::cacheKey(
    std::string_view bucket,
    std::shared_ptr<const config::ConfigBase> config) {
  if (auto bucketEndpoint =
          configValue(*config, bucketConfigKey(Keys::kEndpoint, bucket))) {
    return fmt::format("{}-{}", bucketEndpoint.value(), bucket);
  }
  if (auto baseEndpoint =
          configValue(*config, baseConfigKey(Keys::kEndpoint))) {
    return fmt::format("{}-{}", baseEndpoint.value(), bucket);
  }
  return std::string(bucket);
}

S3Config::S3Config(
    std::string_view bucket,
    const std::shared_ptr<const config::ConfigBase> properties)
    : bucket_(bucket) {
  for (int key = static_cast<int>(Keys::kBegin);
       key < static_cast<int>(Keys::kEnd);
       key++) {
    auto s3Key = static_cast<Keys>(key);
    auto value = S3Config::configTraits().find(s3Key)->second;
    auto configDefault = value.second;

    // Prefer the bucket-specific "s3.bucket.*" config, then the base "s3.*"
    // config, then the default. Each lookup falls back to the deprecated
    // "hive.s3." prefix when the canonical key is absent.
    if (auto configVal =
            configValue(*properties, bucketConfigKey(s3Key, bucket))) {
      config_[s3Key] = configVal.value();
    } else if (auto baseVal = configValue(*properties, baseConfigKey(s3Key))) {
      config_[s3Key] = baseVal.value();
    } else {
      config_[s3Key] = configDefault;
    }
  }
  payloadSigningPolicy_ =
      configValue(*properties, kS3PayloadSigningPolicy).value_or("Never");

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
