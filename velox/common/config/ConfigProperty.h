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

#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include "velox/common/Enums.h"

namespace facebook::velox::config {

enum class ConfigPropertyType {
  kBoolean,
  kInteger,
  kDouble,
  kString,
};

VELOX_DECLARE_ENUM_NAME(ConfigPropertyType);

/// Describes a single configuration property: name, type, default, and
/// human-readable description.
struct ConfigProperty {
  /// Property name, e.g. "session_timezone".
  std::string name;

  ConfigPropertyType type;

  /// Default value as a string, or nullopt if the property has no default.
  std::optional<std::string> defaultValue;

  /// Human-readable description of the property.
  std::string description;
};

namespace detail {

template <typename T>
constexpr ConfigPropertyType configPropertyTypeOf() {
  if constexpr (std::is_same_v<T, bool>) {
    return ConfigPropertyType::kBoolean;
  } else if constexpr (std::is_integral_v<T>) {
    return ConfigPropertyType::kInteger;
  } else if constexpr (std::is_floating_point_v<T>) {
    return ConfigPropertyType::kDouble;
  } else {
    return ConfigPropertyType::kString;
  }
}

template <typename T>
std::string configPropertyDefaultToString(const T& value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_arithmetic_v<T>) {
    return fmt::format("{}", value);
  } else {
    return std::string(value);
  }
}

} // namespace detail

/// Registers a property from a traits struct into a vector.
/// The traits struct must have: key, type, defaultValue, description.
template <typename PropertyTraits>
void registerConfigProperty(std::vector<ConfigProperty>& registry) {
  registry.push_back({
      PropertyTraits::key,
      detail::configPropertyTypeOf<typename PropertyTraits::type>(),
      detail::configPropertyDefaultToString<typename PropertyTraits::type>(
          PropertyTraits::defaultValue),
      PropertyTraits::description,
  });
}

} // namespace facebook::velox::config
