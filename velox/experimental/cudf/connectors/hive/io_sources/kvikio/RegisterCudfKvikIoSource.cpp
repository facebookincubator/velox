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

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

namespace {

constexpr std::string_view kFilePrefix = "file:";
constexpr std::string_view kS3Prefix = "s3://";
constexpr std::string_view kS3aPrefix = "s3a://";

// Matches the paths served by KvikIO datasource: local files (with or without a
// `file:` prefix) and S3 URIs (`s3://`, `s3a://`)
bool kvikIoMatcher(std::string_view path) {
  return path.starts_with('/') || path.starts_with(kFilePrefix) ||
      path.starts_with(kS3Prefix) || path.starts_with(kS3aPrefix);
}

// Strip the `file:` prefix and rewrite `s3a://` to `s3://` so the path
// is in the form KvikIO datasource expects.
std::string normalizeKvikIoPath(std::string_view path) {
  if (path.starts_with(kFilePrefix)) {
    return std::string(path.substr(kFilePrefix.size()));
  }
  if (path.starts_with(kS3aPrefix)) {
    std::string normalized(path);
    // "s3a://..." -> "s3://..." by dropping the 'a' at index 2.
    normalized.erase(2, 1);
    return normalized;
  }
  return std::string(path);
}

} // namespace

void registerCudfKvikIoSource() {
  registerCudfIoSource(
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
