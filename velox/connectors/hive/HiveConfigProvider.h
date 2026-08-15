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
#pragma once

#include "velox/common/config/ConfigProvider.h"

namespace facebook::velox::config {
class ConfigBase;
} // namespace facebook::velox::config

namespace facebook::velox::connector::hive {

/// Exposes Hive connector session-overridable properties.
/// Optionally initialized with config-file values from the catalog's
/// .properties file. When provided, config-file values override code
/// defaults in ConfigProperty::defaultValue, enabling a three-layer
/// cascade: code default → config file → session override.
class HiveConfigProvider : public config::ConfigProvider {
 public:
  /// Creates a provider with code defaults only.
  HiveConfigProvider() = default;

  /// Creates a provider with config-file overrides. For each session
  /// property, the corresponding config-file key is looked up via
  /// ConfigBase::toConfigKey() (underscore → dash conversion).
  HiveConfigProvider(
      const std::string& catalogName,
      const config::ConfigBase* catalogConfig)
      : catalogName_(catalogName), catalogConfig_(catalogConfig) {}

  /// Returns all session-overridable Hive connector properties.
  std::vector<config::ConfigProperty> properties() const override;

  /// Validates and normalizes a property value.
  std::string normalize(std::string_view name, std::string_view value)
      const override;

 private:
  std::string catalogName_;
  const config::ConfigBase* catalogConfig_{nullptr};
};

} // namespace facebook::velox::connector::hive
