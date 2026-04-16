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

#include "velox/connectors/hive/HiveConfig.h"

namespace facebook::velox::connector::hive {

std::vector<config::ConfigProperty> HiveConfigProvider::properties() const {
  return HiveConfig::registeredProperties();
}

std::string HiveConfigProvider::normalize(
    std::string_view /*name*/,
    std::string_view value) const {
  return std::string(value);
}

} // namespace facebook::velox::connector::hive
