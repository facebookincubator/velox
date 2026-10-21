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

#include <folly/container/F14Map.h>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::connector::hive::iceberg {

/// Name of the connector (build-time) config property listing, comma-separated,
/// the session-property keys whose per-query values are copied verbatim into
/// the fileReadOps of Iceberg data-file reads. A FileSystem can use these
/// values to tag its requests, for example for usage attribution. Empty by
/// default, so a stock deployment forwards nothing.
///
/// The forwarded values are plain request metadata:
/// - Task traces record them unredacted.
/// - Writes do not receive them.
///
/// List credentials under 'kSessionCredentialKeysConfig' instead. Leave out
/// 'dbName' and 'tableName': the reader sets those itself and overwrites a
/// forwarded value.
inline constexpr std::string_view kSessionFileReadOpKeysConfig =
    "hive.session-file-read-op-keys";

/// Reads the key list named by 'kSessionFileReadOpKeysConfig' from
/// 'connectorConfig' and returns the key -> value pairs whose keys have a
/// non-empty value in 'sessionProperties'. Returns an empty map when either
/// argument is null, the config property is unset or empty, or no listed key
/// has a value.
folly::F14FastMap<std::string, std::string> sessionFileReadOps(
    const config::ConfigBase* connectorConfig,
    const config::ConfigBase* sessionProperties);

} // namespace facebook::velox::connector::hive::iceberg
