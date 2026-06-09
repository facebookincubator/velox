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

#include "velox/common/base/BitUtil.h"
#include "velox/common/caching/CachedFactory.h"
#include "velox/common/caching/FileIds.h"
#include "velox/common/caching/FileProperties.h"
#include "velox/common/config/Config.h"
#include "velox/common/file/File.h"
#include "velox/common/file/TokenProvider.h"

namespace facebook::velox {

/// fileReadOps keys for passing table identity (db and table name)
/// through the file handle layer. Written by the connector (FileSplitReader)
/// and read by storage implementations to build per-table access token keys.
constexpr std::string_view kDbNameKey = "dbName";
constexpr std::string_view kTableNameKey = "tableName";

/// File pointer plus cache-friendly identifiers for downstream caching.
struct FileHandle {
  std::shared_ptr<ReadFile> file;

  // Each time we make a new FileHandle we assign it a uuid and use that id as
  // the identifier in downstream data caching structures. This saves a lot of
  // memory compared to using the filename as the identifier.
  StringIdLease uuid;

  // Id for the group of files this belongs to, e.g. its
  // directory. Used for coarse granularity access tracking, for
  // example to decide placing on SSD.
  StringIdLease groupId;
};

/// Estimates the memory usage of a FileHandle object.
struct FileHandleSizer {
  uint64_t operator()(const FileHandle& a);
};

struct FileHandleKey {
  std::string filename;
  std::shared_ptr<filesystems::TokenProvider> tokenProvider{nullptr};

  bool operator==(const FileHandleKey& other) const {
    if (filename != other.filename) {
      return false;
    }

    if (tokenProvider == other.tokenProvider) {
      return true;
    }

    if (!tokenProvider || !other.tokenProvider) {
      return false;
    }

    return tokenProvider->equals(*other.tokenProvider);
  }
};

} // namespace facebook::velox

namespace std {
template <>
struct hash<facebook::velox::FileHandleKey> {
  size_t operator()(const facebook::velox::FileHandleKey& key) const noexcept {
    size_t filenameHash = std::hash<std::string>()(key.filename);
    return key.tokenProvider ? facebook::velox::bits::hashMix(
                                   filenameHash, key.tokenProvider->hash())
                             : filenameHash;
  }
};
} // namespace std

namespace facebook::velox {
using FileHandleCache =
    SimpleLRUCache<facebook::velox::FileHandleKey, FileHandle>;

/// Creates FileHandles via the Generator interface the CachedFactory requires.
class FileHandleGenerator {
 public:
  FileHandleGenerator() {}
  FileHandleGenerator(std::shared_ptr<const config::ConfigBase> properties)
      : properties_(std::move(properties)) {}
  std::unique_ptr<FileHandle> operator()(
      const FileHandleKey& filename,
      const FileProperties* properties,
      IoStats* stats);

 private:
  const std::shared_ptr<const config::ConfigBase> properties_;
};

using FileHandleFactory = CachedFactory<
    FileHandleKey,
    FileHandle,
    FileHandleGenerator,
    FileProperties,
    IoStats,
    FileHandleSizer>;

using FileHandleCachedPtr = CachedPtr<FileHandleKey, FileHandle>;

using FileHandleCacheStats = SimpleLRUCacheStats;

} // namespace facebook::velox
