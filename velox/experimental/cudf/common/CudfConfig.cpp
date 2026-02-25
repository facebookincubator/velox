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

namespace facebook::velox::cudf_velox {

CudfConfig::CudfConfig() : velox::config::ConfigBase({}, true) {
  // Initialize the set of all configuration entries with their metadata
  configEntries_ = {
      kCudfEnabledEntry,
      kCudfDebugEnabledEntry,
      kCudfMemoryResourceEntry,
      kCudfMemoryPercentEntry,
      kCudfFunctionNamePrefixEntry,
      kCudfAstExpressionEnabledEntry,
      kCudfAstExpressionPriorityEntry,
      kCudfJitExpressionEnabledEntry,
      kCudfJitExpressionPriorityEntry,
      kCudfAllowCpuFallbackEntry,
      kCudfLogFallbackEntry};

  // Register all config entries with their default values and build valid keys
  // set
  for (const auto& entry : configEntries_) {
    set(entry.name, entry.defaultValue);
    configKeys_.insert(entry.name);
  }
}

CudfConfig& CudfConfig::getInstance() {
  static CudfConfig instance;
  return instance;
}

void CudfConfig::updateConfigs(
    std::unordered_map<std::string, std::string>&& config) {
  // Update configs with optional validation.
  for (auto& [key, value] : config) {
    VELOX_CHECK(
        configKeys_.find(key) != configKeys_.end(),
        "{} is not a valid CudfConfig.",
        key);
    // Validate memoryPercent if provided
    if (key == kCudfMemoryPercentEntry.name) {
      auto percent = folly::to<int32_t>(value);
      VELOX_USER_CHECK_GT(
          percent,
          0,
          "cudf.memory_percent must be greater than 0 to initialize cuDF memory resource");
    }
    set(key, value);
  }
}

const std::vector<CudfConfig::CudfConfigEntry>& CudfConfig::getConfigs() const {
  return configEntries_;
}

} // namespace facebook::velox::cudf_velox
