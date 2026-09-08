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

#include <string_view>

#include <folly/container/F14Map.h>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::connector::hive::iceberg {

/// Name of the connector (build-time) config property listing, comma-separated,
/// the session-property keys whose per-query values should be forwarded to the
/// FileSystem. A delegated-credential FileSystem uses these values to authorize
/// I/O as the caller rather than the service identity. Empty by default, so a
/// stock (OSS) deployment forwards nothing and stays deployment-agnostic; the
/// deployment-specific credential name lives in the config layer, not in code.
inline constexpr std::string_view kSessionCredentialKeysConfig =
    "hive.session-credential-keys";

/// Reads the credential key list named by 'kSessionCredentialKeysConfig' from
/// 'connectorConfig' and returns the key -> value pairs whose keys have a
/// non-empty value in 'sessionProperties'. Returns an empty map when either
/// argument is null, the config property is unset/empty, or no listed key has a
/// value. Shared by the Iceberg read (IcebergSplitReader::prepareSplit) and
/// write (IcebergConnector::createDataSink) paths so they cannot diverge.
folly::F14FastMap<std::string, std::string> sessionCredentials(
    const config::ConfigBase* connectorConfig,
    const config::ConfigBase* sessionProperties);

} // namespace facebook::velox::connector::hive::iceberg
