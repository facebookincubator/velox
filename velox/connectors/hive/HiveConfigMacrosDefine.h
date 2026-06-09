// NOLINT(facebook-hte-MultipleIncludeGuardMissing)
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
// Macros for defining Hive connector config properties. Same traits-based
// pattern as VELOX_QUERY_CONFIG in QueryConfig.h.
//
// No #pragma once — this file is intentionally included multiple times
// (once per header that defines config properties). Each inclusion must
// be paired with HiveConfigMacrosUndef.h to scope the macros.
//
// To add a new property, add two lines (order doesn't matter, but
// grouping related properties together helps readability):
//
// 1. In FileConfig.h or HiveConfig.h (inside the class body):
//   VELOX_HIVE_CONFIG(kMaxPartitionsPerWritersSession,
//       maxPartitionsPerWriters, "max_partitions_per_writers",
//       uint32_t, 128, "Maximum partitions per writer.")
//
// 2. In the matching .cpp file (inside registeredProperties()):
//   VELOX_HIVE_CONFIG_REGISTER(kMaxPartitionsPerWritersSession);
//
// VELOX_HIVE_CONFIG generates:
//   - struct kMaxPartitionsPerWritersSessionProperty {
//         using type = uint32_t;
//         static constexpr const char* key = "max_partitions_per_writers";
//         static constexpr auto defaultValue = 128;
//         static constexpr const char* description = "Maximum partitions...";
//     };
//   - static constexpr const char* kMaxPartitionsPerWritersSession =
//         "max_partitions_per_writers"
//   - uint32_t maxPartitionsPerWriters(const ConfigBase* session) const {
//         return session->get<uint32_t>(kMaxPartitionsPerWritersSession,
//             config_->get<uint32_t>(toConfigKey(...), 128));
//     }
//
// VELOX_HIVE_CONFIG_PROPERTY generates the same but without the accessor.
// Used for properties with custom accessor logic: capacity parsing,
// validation, or session-only properties that should not fall back to
// connector config (e.g., ignoreMissingFiles, isPartitionPathAsLowerCase).
// New properties should use VELOX_HIVE_CONFIG and put validation in
// HiveConfigProvider::normalize() and HiveConnector constructor.
// TODO: Unify validation into a single path.
//
// VELOX_HIVE_CONFIG_LEGACY is the same as VELOX_HIVE_CONFIG but takes an
// explicit config key and uses sessionValue() for legacy 'hive.' prefix
// fallback.

#ifdef VELOX_HIVE_CONFIG_MACROS_DEFINED
#error "HiveConfigMacrosDefine.h included twice without HiveConfigMacrosUndef.h"
#endif
// NOLINTBEGIN(facebook-modularize-issue-check)
#define VELOX_HIVE_CONFIG_MACROS_DEFINED

#define VELOX_HIVE_CONFIG_PROPERTY(                  \
    constName, keyStr, CppType, defaultVal, desc)    \
  struct constName##Property {                       \
    using type = CppType;                            \
    static constexpr const char* key = keyStr;       \
    static constexpr auto defaultValue = defaultVal; \
    static constexpr const char* description = desc; \
  };                                                 \
  static constexpr const char* constName = keyStr;

#define VELOX_HIVE_CONFIG(                                              \
    constName, accessorName, key, CppType, defaultVal, desc)            \
  VELOX_HIVE_CONFIG_PROPERTY(constName, key, CppType, defaultVal, desc) \
  CppType accessorName(const config::ConfigBase* session) const {       \
    return session->get<CppType>(                                       \
        constName,                                                      \
        config_->get<CppType>(                                          \
            config::ConfigBase::toConfigKey(constName), defaultVal));   \
  }

#define VELOX_HIVE_CONFIG_LEGACY(                                       \
    constName,                                                          \
    configConstName,                                                    \
    accessorName,                                                       \
    key,                                                                \
    configKey,                                                          \
    CppType,                                                            \
    defaultVal,                                                         \
    desc)                                                               \
  VELOX_HIVE_CONFIG_PROPERTY(constName, key, CppType, defaultVal, desc) \
  static constexpr const char* configConstName = configKey;             \
  CppType accessorName(const config::ConfigBase* session) const {       \
    return sessionValue<CppType>(                                       \
        session, constName, configConstName, defaultVal);               \
  }
// NOLINTEND(facebook-modularize-issue-check)
