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
#include "velox/core/QueryConfigProvider.h"

#include <glog/logging.h>
#include "velox/core/QueryConfig.h"

namespace facebook::velox::core {

std::vector<config::ConfigProperty> QueryConfigProvider::properties() const {
  auto props = QueryConfig::registeredProperties();
  if (configOverrides_.empty()) {
    return props;
  }

  // Build a lookup map from property name to index.
  std::unordered_map<std::string, size_t> propIndex;
  for (size_t i = 0; i < props.size(); ++i) {
    propIndex[props[i].name] = i;
  }

  // Override code defaults with config-file values. Warn on unknown keys.
  for (const auto& [key, value] : configOverrides_) {
    auto it = propIndex.find(key);
    if (it != propIndex.end()) {
      props[it->second].defaultValue = normalize(key, value);
    } else {
      LOG(WARNING) << "Unregistered session property in execution config: "
                   << key;
    }
  }

  return props;
}

std::string QueryConfigProvider::normalize(
    std::string_view /*name*/,
    std::string_view value) const {
  return std::string(value);
}

} // namespace facebook::velox::core
