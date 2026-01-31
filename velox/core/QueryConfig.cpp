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

#include <re2/re2.h>

#include "velox/common/config/Config.h"
#include "velox/core/QueryConfig.h"
#include "velox/type/tz/TimeZoneMap.h"

namespace facebook::velox::core {

QueryConfig::QueryConfig(std::unordered_map<std::string, std::string> values)
    : QueryConfig{
          ConfigTag{},
          std::make_shared<config::ConfigBase>(std::move(values))} {}

QueryConfig::QueryConfig(ConfigTag /*tag*/, config::ConfigPtr config)
    : config_{std::move(config)} {
  validateConfig();
}

void QueryConfig::validateConfig() {
  // Validate if timezone name can be recognized.
  if (auto tz = config_->get<std::string>(QueryConfig::kSessionTimezone)) {
    VELOX_USER_CHECK(
        tz::getTimeZoneID(*tz, false) != -1,
        "session '{}' set with invalid value '{}'",
        QueryConfig::kSessionTimezone,
        *tz);
  }
}

void QueryConfig::testingOverrideConfigUnsafe(
    std::unordered_map<std::string, std::string>&& values) {
  config_ = std::make_unique<config::ConfigBase>(std::move(values));
}

std::unordered_map<std::string, std::string> QueryConfig::rawConfigsCopy()
    const {
  return config_->rawConfigsCopy();
}

} // namespace facebook::velox::core
