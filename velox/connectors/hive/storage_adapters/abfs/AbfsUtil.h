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
#include <folly/hash/Hash.h>
#include <vector>
#include "velox/common/file/File.h"

namespace facebook::velox::filesystems {
namespace {
constexpr std::string_view kAbfsScheme{"abfs://"};
constexpr std::string_view kAbfssScheme{"abfss://"};
} // namespace

class ConfigBase;

struct CacheKey {
  const std::string accountNameWithSuffix;
  const std::string authType;

  CacheKey(std::string_view accountNameWithSuffix, std::string_view authType)
      : accountNameWithSuffix(accountNameWithSuffix), authType(authType) {}

  bool operator==(const CacheKey& other) const {
    return accountNameWithSuffix == other.accountNameWithSuffix &&
        authType == other.authType;
  }
};

} // namespace facebook::velox::filesystems

// Specialize std::hash for CacheKey and vector<CacheKey> to enable their use
// in unordered_map
namespace std {
template <>
struct hash<facebook::velox::filesystems::CacheKey> {
  size_t operator()(const facebook::velox::filesystems::CacheKey& key) const {
    return folly::hash::hash_combine(
        std::hash<std::string>{}(key.accountNameWithSuffix),
        std::hash<std::string>{}(key.authType));
  }
};

template <>
struct hash<std::vector<facebook::velox::filesystems::CacheKey>> {
  size_t operator()(
      const std::vector<facebook::velox::filesystems::CacheKey>& keys) const {
    size_t seed = 0;
    for (const auto& key : keys) {
      seed = folly::hash::hash_combine(
          seed, std::hash<facebook::velox::filesystems::CacheKey>{}(key));
    }
    return seed;
  }
};
} // namespace std

namespace facebook::velox::filesystems {

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

std::vector<CacheKey> extractCacheKeyFromConfig(
    const config::ConfigBase& config);

} // namespace facebook::velox::filesystems
