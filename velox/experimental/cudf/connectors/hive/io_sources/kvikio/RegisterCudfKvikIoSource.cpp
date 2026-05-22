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

// Remove "file:" prefix from the file path if present. Normalize "s3a:" to
// "s3:" prefix.
std::string normalizeKvikIoPath(std::string_view path) {
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3aPrefix = "s3a:";
  if (path.size() >= kFilePrefix.size() &&
      path.substr(0, kFilePrefix.size()) == kFilePrefix) {
    return std::string(path.substr(kFilePrefix.size()));
  }
  if (path.size() >= kS3aPrefix.size() &&
      path.substr(0, kS3aPrefix.size()) == kS3aPrefix) {
    // "s3a:" -> "s3:" (drop the 'a' before the colon).
    std::string normalized(path);
    normalized.erase(kS3aPrefix.size() - 2, 1);
    return normalized;
  }
  return std::string(path);
}

} // namespace

void registerCudfKvikIoSource() {
  registerCudfDefaultIoSource(
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
