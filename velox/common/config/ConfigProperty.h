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
#include <type_traits>
#include <vector>

#include "velox/common/EnumDeclare.h"

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

std::string toStringValue(double value);

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
    if constexpr (std::is_floating_point_v<T>) {
      return toStringValue(static_cast<double>(value));
    } else {
      static_assert(!std::is_same_v<T, char>, "Use int, not char.");
      return std::to_string(value);
    }
  } else {
    return std::string(value);
  }
}

template <typename T>
struct IsOptional : std::false_type {};
template <typename T>
struct IsOptional<std::optional<T>> : std::true_type {};

template <typename T>
struct OptionalValueType {
  using type = T;
};
template <typename T>
struct OptionalValueType<std::optional<T>> {
  using type = T;
};

} // namespace detail

/// Registers a property from a traits struct into a vector.
/// The traits struct must have: key, type, defaultValue, description.
/// A std::optional trait type registers the inner type; an empty default
/// registers no default, in which case the property is omitted unless set.
template <typename PropertyTraits>
void registerConfigProperty(std::vector<ConfigProperty>& registry) {
  using TraitsType = typename PropertyTraits::type;
  using ValueType = typename detail::OptionalValueType<TraitsType>::type;
  std::optional<std::string> defaultValue;
  if constexpr (detail::IsOptional<TraitsType>::value) {
    if (PropertyTraits::defaultValue.has_value()) {
      defaultValue = detail::configPropertyDefaultToString<ValueType>(
          PropertyTraits::defaultValue.value());
    }
  } else {
    defaultValue = detail::configPropertyDefaultToString<TraitsType>(
        PropertyTraits::defaultValue);
  }
  registry.push_back({
      PropertyTraits::key,
      detail::configPropertyTypeOf<ValueType>(),
      std::move(defaultValue),
      PropertyTraits::description,
  });
}

} // namespace facebook::velox::config
