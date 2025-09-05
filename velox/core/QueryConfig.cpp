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

QueryConfig::QueryConfig() : config_(std::make_shared<config::ConfigBase>(std::unordered_map<std::string, std::string>{})) {
}

QueryConfig::QueryConfig(
    const std::unordered_map<std::string, std::string>& values)
    : config_{std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>(values))} {
  validateConfig();
}

QueryConfig::QueryConfig(std::unordered_map<std::string, std::string>&& values)
    : config_{std::make_shared<config::ConfigBase>(std::move(values))} {
  validateConfig();
}

QueryConfig::QueryConfig(const std::shared_ptr<config::IConfig>& config) : config_(config) {}

void QueryConfig::validateConfig() {
  // Validate if timezone name can be recognized.
  if (config_->Has(QueryConfig::kSessionTimezone)) {
    VELOX_USER_CHECK(
        tz::getTimeZoneID(
            std::any_cast<std::string>(config_->Get(QueryConfig::kSessionTimezone).value()),
            false) != -1,
        fmt::format(
            "session '{}' set with invalid value '{}'",
            QueryConfig::kSessionTimezone,
            std::any_cast<std::string>(config_->Get(QueryConfig::kSessionTimezone).value())));
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
