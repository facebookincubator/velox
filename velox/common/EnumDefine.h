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

#include "folly/container/F14Map.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox {

struct Enums {
  /// Helper function to invert a mapping from enum type to name.
  template <typename EnumType, typename StringType>
  static auto invertMap(
      const folly::F14FastMap<EnumType, StringType>& mapping) {
    folly::F14FastMap<StringType, EnumType> inverted;
    for (const auto& [key, value] : mapping) {
      const bool emplaced = inverted.emplace(value, key).second;
      VELOX_USER_CHECK(
          emplaced, "Cannot invert a map with duplicate values: {}", value);
    }
    return inverted;
  }
};

} // namespace facebook::velox

/// DEFINE macros for enum-to-name mappings. Include only in .cpp files.
///
/// See EnumDeclare.h for full usage documentation and the DECLARE macros.

#define VELOX_DEFINE_ENUM_NAME(EnumType, Names)                             \
  std::string_view EnumType##Name::toName(EnumType value) {                 \
    const auto& names = Names();                                            \
    auto it = names.find(value);                                            \
    VELOX_CHECK(                                                            \
        it != names.end(),                                                  \
        "Invalid enum value: {}",                                           \
        static_cast<std::underlying_type_t<EnumType>>(value));              \
    return it->second;                                                      \
  }                                                                         \
                                                                            \
  std::optional<EnumType> EnumType##Name::tryTo##EnumType(                  \
      std::string_view name) {                                              \
    static const auto kValues = facebook::velox::Enums::invertMap(Names()); \
                                                                            \
    auto it = kValues.find(name);                                           \
    if (it == kValues.end()) {                                              \
      return std::nullopt;                                                  \
    }                                                                       \
    return it->second;                                                      \
  }                                                                         \
  std::ostream& operator<<(std::ostream& os, const EnumType& value) {       \
    os << EnumType##Name::toName(value);                                    \
    return os;                                                              \
  }                                                                         \
                                                                            \
  EnumType EnumType##Name::to##EnumType(std::string_view name) {            \
    const auto maybeType = EnumType##Name::tryTo##EnumType(name);           \
    VELOX_CHECK(maybeType, "Invalid enum name: {}", name);                  \
    return *maybeType;                                                      \
  }

#define VELOX_DEFINE_EMBEDDED_ENUM_NAME(Class, EnumType, Names)             \
  std::string_view Class::toName(Class::EnumType value) {                   \
    const auto& names = Names();                                            \
    auto it = names.find(value);                                            \
    VELOX_CHECK(                                                            \
        it != names.end(),                                                  \
        "Invalid enum value: {}",                                           \
        static_cast<std::underlying_type_t<Class::EnumType>>(value));       \
    return it->second;                                                      \
  }                                                                         \
                                                                            \
  std::optional<Class::EnumType> Class::tryTo##EnumType(                    \
      std::string_view name) {                                              \
    static const auto kValues = facebook::velox::Enums::invertMap(Names()); \
                                                                            \
    auto it = kValues.find(name);                                           \
    if (it == kValues.end()) {                                              \
      return std::nullopt;                                                  \
    }                                                                       \
    return it->second;                                                      \
  }                                                                         \
                                                                            \
  Class::EnumType Class::to##EnumType(std::string_view name) {              \
    const auto maybeType = Class::tryTo##EnumType(name);                    \
    VELOX_CHECK(maybeType, "Invalid enum name: {}", name);                  \
    return *maybeType;                                                      \
  }
