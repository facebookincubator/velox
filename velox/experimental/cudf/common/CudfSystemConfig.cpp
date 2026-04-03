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

#include "velox/experimental/cudf/common/CudfSystemConfig.h"

#include <algorithm>

namespace facebook::velox::cudf_velox {

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
    // TODO(ps): Revert support for legacy config names.
    auto canonicalKey = key;
    std::replace(canonicalKey.begin(), canonicalKey.end(), '_', '-');
    set(canonicalKey, value);
  }
}

} // namespace facebook::velox::cudf_velox
