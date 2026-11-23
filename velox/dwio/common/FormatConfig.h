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

#include <string>
#include <string_view>
#include <vector>

#include "velox/common/config/Config.h"
#include "velox/common/config/ConfigProperty.h"

#define VELOX_FORMAT_CONFIG_PROPERTY(                         \
    sessionConstName,                                         \
    configConstName,                                          \
    sessionKey,                                               \
    configKey,                                                \
    CppType,                                                  \
    defaultVal,                                               \
    desc)                                                     \
  struct sessionConstName##Property {                         \
    using type = CppType;                                     \
    static constexpr const char* key = sessionKey;            \
    static constexpr auto defaultValue = defaultVal;          \
    static constexpr const char* description = desc;          \
  };                                                          \
  static constexpr const char* sessionConstName = sessionKey; \
  static constexpr const char* configConstName = configKey;

#define VELOX_FORMAT_CONFIG(                                                   \
    sessionConstName,                                                          \
    configConstName,                                                           \
    accessorName,                                                              \
    sessionKey,                                                                \
    configKey,                                                                 \
    CppType,                                                                   \
    defaultVal,                                                                \
    desc)                                                                      \
  VELOX_FORMAT_CONFIG_PROPERTY(                                                \
      sessionConstName,                                                        \
      configConstName,                                                         \
      sessionKey,                                                              \
      configKey,                                                               \
      CppType,                                                                 \
      defaultVal,                                                              \
      desc)                                                                    \
  static CppType accessorName(                                                 \
      const config::ConfigBase& connectorConfig,                               \
      const config::ConfigBase& session) {                                     \
    return session.getWithFallback<CppType>(sessionConstName, connectorConfig) \
        .value_or(sessionConstName##Property::defaultValue);                   \
  }

namespace facebook::velox::dwio::common {

/// Registers a format-specific session property after prepending the format's
/// session prefix, e.g. "orc_" or "parquet_".
template <typename PropertyTraits>
void registerFormatConfigProperty(
    std::vector<config::ConfigProperty>& registry,
    std::string_view prefix) {
  registry.push_back({
      std::string(prefix) + PropertyTraits::key,
      config::detail::configPropertyTypeOf<typename PropertyTraits::type>(),
      config::detail::configPropertyDefaultToString<
          typename PropertyTraits::type>(PropertyTraits::defaultValue),
      PropertyTraits::description,
  });
}

} // namespace facebook::velox::dwio::common
