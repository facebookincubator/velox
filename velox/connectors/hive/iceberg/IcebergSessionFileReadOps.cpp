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

#include "velox/connectors/hive/iceberg/IcebergSessionFileReadOps.h"

#include <vector>

#include <folly/String.h>

#include "velox/common/config/Config.h"

namespace facebook::velox::connector::hive::iceberg {

folly::F14FastMap<std::string, std::string> sessionFileReadOps(
    const config::ConfigBase* connectorConfig,
    const config::ConfigBase* sessionProperties) {
  folly::F14FastMap<std::string, std::string> fileReadOps;
  if (connectorConfig == nullptr || sessionProperties == nullptr) {
    return fileReadOps;
  }
  const auto keysCsv = connectorConfig->get<std::string>(
      std::string(kSessionFileReadOpKeysConfig));
  if (!keysCsv.has_value() || keysCsv.value().empty()) {
    return fileReadOps;
  }

  std::vector<std::string_view> keys;
  folly::split(',', keysCsv.value(), keys);
  for (const auto rawKey : keys) {
    const auto key = folly::trimWhitespace(rawKey);
    if (key.empty()) {
      continue;
    }
    const auto value = sessionProperties->get<std::string>(std::string(key));
    if (value.has_value() && !value.value().empty()) {
      fileReadOps[std::string(key)] = value.value();
    }
  }
  return fileReadOps;
}

} // namespace facebook::velox::connector::hive::iceberg
