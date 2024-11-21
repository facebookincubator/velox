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

namespace facebook::velox::filesystems {

std::string S3Config::identity(
    std::string_view bucket,
    std::shared_ptr<const config::ConfigBase> config) {
  auto bucketEndpoint = bucketConfigKey(Keys::kEndpoint, bucket);
  if (config->valueExists(bucketEndpoint)) {
    auto value = config->get<std::string>(bucketEndpoint);
    if (value.has_value()) {
      return value.value();
    }
  }
  auto baseEndpoint = baseConfigKey(Keys::kEndpoint);
  if (config->valueExists(baseEndpoint)) {
    auto value = config->get<std::string>(baseEndpoint);
    if (value.has_value()) {
      return value.value();
    }
  }
  return kDefaultS3Identity;
}

S3Config::S3Config(
    std::string_view bucket,
    const std::shared_ptr<const config::ConfigBase> properties) {
  for (int key = static_cast<int>(Keys::kBegin);
       key < static_cast<int>(Keys::kEnd);
       key++) {
    auto s3Key = static_cast<Keys>(key);
    auto value = S3Config::configTraits().find(s3Key)->second;
    auto configSuffix = value.first;
    auto configDefault = value.second;

    // Set bucket S3 config "hive.s3.bucket.*" if present.
    std::stringstream bucketConfig;
    bucketConfig << kS3BucketPrefix << bucket << "." << configSuffix;
    auto configVal = static_cast<std::optional<std::string>>(
        properties->get<std::string>(bucketConfig.str()));
    if (configVal.has_value()) {
      config_[s3Key] = configVal.value();
    } else {
      // Set base config "hive.s3.*" if present.
      std::stringstream baseConfig;
      baseConfig << kS3Prefix << configSuffix;
      configVal = static_cast<std::optional<std::string>>(
          properties->get<std::string>(baseConfig.str()));
      if (configVal.has_value()) {
        config_[s3Key] = configVal.value();
      } else {
        // Set the default value.
        config_[s3Key] = configDefault;
      }
    }
  }
}

} // namespace facebook::velox::filesystems
