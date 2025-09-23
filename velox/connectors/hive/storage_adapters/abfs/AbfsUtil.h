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

#include <azure/storage/common/storage_exception.hpp>
#include <fmt/format.h>
#include "velox/common/config/Config.h"
#include "velox/common/file/File.h"
#include "velox/connectors/hive/storage_adapters/abfs/AbfsPath.h"

namespace facebook::velox::filesystems {
namespace {
constexpr std::string_view kAbfsScheme{"abfs://"};
constexpr std::string_view kAbfssScheme{"abfss://"};
} // namespace

inline bool isAbfsFile(const std::string_view filename) {
  return filename.find(kAbfsScheme) == 0 || filename.find(kAbfssScheme) == 0;
}

inline std::string throwStorageExceptionWithOperationDetails(
    std::string operation,
    std::string path,
    Azure::Storage::StorageException& error) {
  const auto errMsg = fmt::format(
      "Operation '{}' to path '{}' encountered azure storage exception, Details: '{}'.",
      operation,
      path,
      error.what());
  if (error.StatusCode == Azure::Core::Http::HttpStatusCode::NotFound) {
    VELOX_FILE_NOT_FOUND_ERROR(errMsg);
  }
  VELOX_FAIL(errMsg);
}

inline void extractCacheKeyFromConfig(
    const config::ConfigBase& config,
    std::string& accountName,
    std::string& authenticationType) {
  for (const auto& [key, value] : config.rawConfigs()) {
    constexpr std::string_view authTypePrefix{kAzureAccountAuthType};
    if (key.find(authTypePrefix) == 0) {
      std::string_view skey = key;
      // Extract the accountName after "fs.azure.account.auth.type.".
      auto remaining = skey.substr(authTypePrefix.size() + 1);
      auto dot = remaining.find(".");
      VELOX_USER_CHECK_NE(
          dot,
          std::string_view::npos,
          "Invalid Azure account auth type key: {}",
          key);
      accountName = std::string(remaining.substr(0, dot));
      authenticationType = value;
    }
  }
}

} // namespace facebook::velox::filesystems
