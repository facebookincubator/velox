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

#include "velox/experimental/cudf/connectors/hive/io_sources/CudfIoSourceRegistry.h"
#include "velox/experimental/cudf/connectors/hive/io_sources/kvikio/RegisterCudfKvikIoSource.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/types.hpp>

#include <array>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

namespace {

constexpr std::string_view kFilePrefix = "file:";

// S3-wire-compatible prefixes KvikIO can serve
constexpr std::array<std::string_view, 6> kS3CompatiblePrefixes{
    "s3://",
    "s3a://",
    "s3n://",
    "oss://",
    "cos://",
    "cosn://"};

constexpr std::string_view kS3CanonicalPrefix = "s3://";

bool startsWithAnyS3Prefix(std::string_view path) {
  for (const auto& prefix : kS3CompatiblePrefixes) {
    if (path.starts_with(prefix)) {
      return true;
    }
  }
  return false;
}

// Matches the paths served by KvikIO datasource: local files (with or without a
// `file:` prefix) and S3 URIs.
bool kvikIoMatcher(std::string_view path) {
  return path.starts_with('/') || path.starts_with(kFilePrefix) ||
      startsWithAnyS3Prefix(path);
}

// Strip the `file:` prefix and rewrite any S3-compatible scheme so the path is
// in a form that KvikIO expects.
std::string normalizeKvikIoPath(std::string_view path) {
  if (path.starts_with(kFilePrefix)) {
    return std::string(path.substr(kFilePrefix.size()));
  }
  for (const auto& prefix : kS3CompatiblePrefixes) {
    if (path.starts_with(prefix)) {
      if (prefix == kS3CanonicalPrefix) {
        return std::string(path);
      }
      std::string normalized(kS3CanonicalPrefix);
      normalized.append(path.substr(prefix.size()));
      return normalized;
    }
  }
  return std::string(path);
}

} // namespace

void registerCudfKvikIoSource() {
  constexpr std::string_view kKvikIoSourceName = "kvikio";
  registerCudfIoSource(
      std::string(kKvikIoSourceName),
      kvikIoMatcher,
      [](std::string_view path,
         [[maybe_unused]] const std::shared_ptr<const config::ConfigBase>&
             properties) -> std::shared_ptr<cudf::io::datasource> {
        auto sources = cudf::io::make_datasources(
            cudf::io::source_info{normalizeKvikIoPath(path)});
        return std::shared_ptr<cudf::io::datasource>(
            std::move(sources.front()));
      });
}

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
