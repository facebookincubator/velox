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
#include "velox/connectors/hive/HiveConfigProvider.h"

#include <glog/logging.h>
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/HiveConfig.h"

namespace facebook::velox::connector::hive {

std::vector<config::ConfigProperty> HiveConfigProvider::properties() const {
  auto props = HiveConfig::registeredProperties();
  if (catalogConfig_ == nullptr) {
    return props;
  }

  // Build a lookup map from session property name to index in props.
  std::unordered_map<std::string, size_t> propIndex;
  for (size_t i = 0; i < props.size(); ++i) {
    propIndex[props[i].name] = i;
  }

  // Iterate config-file entries, convert each key to session property name,
  // and override the code default. Log unrecognized keys.
  for (const auto& [configKey, value] : catalogConfig_->rawConfigs()) {
    auto sessionKey = config::ConfigBase::toSessionKey(configKey);
    auto it = propIndex.find(sessionKey);
    if (it != propIndex.end()) {
      props[it->second].defaultValue = normalize(sessionKey, value);
    } else {
      LOG(WARNING) << "Unrecognized config property in catalog '"
                   << catalogName_ << "': " << configKey;
    }
  }

  return props;
}

std::string HiveConfigProvider::normalize(
    std::string_view /*name*/,
    std::string_view value) const {
  return std::string(value);
}

} // namespace facebook::velox::connector::hive
