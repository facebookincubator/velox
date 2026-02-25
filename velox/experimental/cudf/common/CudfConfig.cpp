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

#include "velox/experimental/cudf/common/CudfConfig.h"

#include "velox/common/base/Exceptions.h"

#include <folly/Conv.h>

#include <algorithm>

namespace facebook::velox::cudf_velox {

CudfQueryConfig::CudfQueryConfig()
    : config_(
          std::make_shared<velox::config::ConfigBase>(
              std::unordered_map<std::string, std::string>{},
              true)) {}

CudfQueryConfig::CudfQueryConfig(
    std::unordered_map<std::string, std::string>&& values)
    : config_(
          std::make_shared<velox::config::ConfigBase>(
              std::move(values),
              true)) {
  // Initialize config entries metadata
  configEntries_ = {
      kCudfEnabledEntry,
      kCudfDebugEnabledEntry,
      kCudfLogFallbackEntry,
      kCudfTopNBatchSizeEntry};

  validateConfigs();
}

CudfQueryConfig::CudfQueryConfig(
    CudfQueryConfig::ConfigTag /*tag*/,
    std::shared_ptr<const velox::config::IConfig> config)
    : config_(config) {}

CudfQueryConfig& CudfQueryConfig::getInstance() {
  static CudfQueryConfig instance;
  return instance;
}

void CudfQueryConfig::validateConfigs() {
  if (auto cudfTopNBatchSize = get<int32_t>(kCudfTopNBatchSizeEntry)) {
    auto batchSize = folly::to<int32_t>(cudfTopNBatchSize);
    VELOX_USER_CHECK_GT(
        batchSize, 0, "cudf.topn_batch_size must be greater than 0");
  }
}

const std::vector<CudfQueryConfig::CudfQueryConfigEntry>&
CudfQueryConfig::getConfigs() const {
  return configEntries_;
}

CudfSystemConfig::CudfSystemConfig()
    : velox::config::ConfigBase(
          std::unordered_map<std::string, std::string>{},
          true) {}

CudfSystemConfig::CudfSystemConfig(
    std::unordered_map<std::string, std::string>&& values)
    : velox::config::ConfigBase(std::move(values), true) {
  validateConfigs();
}

CudfSystemConfig& CudfSystemConfig::getInstance() {
  static CudfSystemConfig instance;
  return instance;
}

void CudfSystemConfig::validateConfigs() {
  // Validate memoryPercent if provided
  if (auto cudfMemoryPercent = get<int32_t>(kCudfMemoryPercent)) {
    VELOX_USER_CHECK_GT(
        cudfMemoryPercent,
        0,
        "cudf.memory_percent must be greater than 0 to initialize cuDF memory resource");
  }
}

void CudfSystemConfig::updateConfigs(
    std::unordered_map<std::string, std::string>&& config) {
  for (auto& [key, value] : config) {
    set(key, value);

    // TODO(ps): Deprecate fallback to underscore-delimited query level config.
    CudfQueryConfig::getInstance().set(key, value);
  }
}

} // namespace facebook::velox::cudf_velox
