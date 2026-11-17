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

#include <unordered_map>
#include "velox/common/config/ConfigProvider.h"

namespace facebook::velox::core {

/// Exposes QueryConfig properties and the deployment values that change their
/// defaults.
///
/// Config overrides replace code defaults. Fixed values also replace code
/// defaults, and the provider marks their properties as not overridable by a
/// session. Use fixed values for values chosen once per process, such as one
/// that must match how functions were registered. Both maps use QueryConfig's
/// snake_case property names.
///
/// Example:
///
///   auto provider = QueryConfigProvider::withFixedValues(
///       /*configOverrides=*/{{"spill_enabled", "true"}},
///       /*fixedValues=*/{{"ignore_time_zone_label", "true"}});
///
/// properties() throws if a fixed value:
/// - names a property QueryConfig does not register, or
/// - differs from a config override for the same property.
class QueryConfigProvider : public config::ConfigProvider {
 public:
  QueryConfigProvider() = default;

  explicit QueryConfigProvider(
      std::unordered_map<std::string, std::string> configOverrides)
      : configOverrides_(std::move(configOverrides)) {}

  /// Creates a provider with config overrides and fixed values.
  static QueryConfigProvider withFixedValues(
      std::unordered_map<std::string, std::string> configOverrides,
      std::unordered_map<std::string, std::string> fixedValues);

  std::vector<config::ConfigProperty> properties() const override;

  std::string normalize(std::string_view name, std::string_view value)
      const override;

 private:
  // A public two-map constructor would make brace-initialized calls of the
  // one-map constructor ambiguous, so the factory uses this tag instead.
  struct FixedValuesTag {};

  QueryConfigProvider(
      std::unordered_map<std::string, std::string> configOverrides,
      std::unordered_map<std::string, std::string> fixedValues,
      FixedValuesTag)
      : configOverrides_(std::move(configOverrides)),
        fixedValues_(std::move(fixedValues)) {}

  // Deployment values that replace QueryConfig code defaults.
  std::unordered_map<std::string, std::string> configOverrides_;
  // Deployment values that replace code defaults and that sessions cannot
  // override.
  std::unordered_map<std::string, std::string> fixedValues_;
};

} // namespace facebook::velox::core
