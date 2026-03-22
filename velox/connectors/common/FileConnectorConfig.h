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

#include <memory>
#include <optional>
#include <string>

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::connector {

/// Base configuration for file-based connectors. Stores the connector-level
/// ConfigBase and provides accessors needed by FileConnector (file handle
/// cache settings). Subclasses (HiveConfig, PaimonConfig) add format-specific
/// settings.
class FileConnectorConfig {
 public:
  /// Maximum number of entries in the file handle cache.
  static constexpr const char* kNumCacheFileHandles = "num_cached_file_handles";

  /// Enable file handle cache.
  static constexpr const char* kEnableFileHandleCache =
      "file-handle-cache-enabled";

  /// Expiration time in ms for a file handle in the cache. A value of 0
  /// means cache will not evict the handle after the expiration duration
  /// has passed.
  static constexpr const char* kFileHandleExpirationDurationMs =
      "file-handle-expiration-duration-ms";

  explicit FileConnectorConfig(
      std::shared_ptr<const config::ConfigBase> config);

  virtual ~FileConnectorConfig() = default;

  /// Number of file handles to cache.
  int32_t numCacheFileHandles() const;

  /// Whether file handle caching is enabled.
  bool isFileHandleCacheEnabled() const;

  /// Expiration duration in ms for cached file handles. 0 means no expiration.
  uint64_t fileHandleExpirationDurationMs() const;

  /// Access the underlying config.
  const std::shared_ptr<const config::ConfigBase>& config() const {
    return config_;
  }

 protected:
  const std::shared_ptr<const config::ConfigBase> config_;
};

} // namespace facebook::velox::connector
