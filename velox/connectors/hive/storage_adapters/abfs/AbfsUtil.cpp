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

#include "velox/connectors/hive/storage_adapters/abfs/AbfsUtil.h"
#include "velox/common/config/Config.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"

namespace facebook::velox::filesystems {

std::vector<CacheKey> extractCacheKeyFromConfig(
    const config::ConfigBase& config) {
  std::vector<CacheKey> cacheKeys;
  constexpr std::string_view authTypePrefix{kAzureAccountAuthType};
  for (const auto& [key, value] : config.rawConfigs()) {
    if (key.find(authTypePrefix) == 0) {
      // Extract the full accountNameWithSuffix after
      // "fs.azure.account.auth.type." This includes the suffix (e.g.,
      // "account1.dfs.core.windows.net")
      auto accountNameWithSuffix =
          std::string_view(key).substr(authTypePrefix.size() + 1);
      VELOX_USER_CHECK(
          !accountNameWithSuffix.empty() &&
              accountNameWithSuffix.find(".") != std::string_view::npos,
          "Invalid Azure account auth type key: {}",
          key);
      cacheKeys.emplace_back(CacheKey{accountNameWithSuffix, value});
    }
  }
  return cacheKeys;
}

} // namespace facebook::velox::filesystems
