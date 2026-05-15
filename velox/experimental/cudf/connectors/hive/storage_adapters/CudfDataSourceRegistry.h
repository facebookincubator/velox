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

#include "velox/common/config/Config.h"

#include <cudf/io/datasource.hpp>

#include <functional>
#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox::filesystems {

/// Predicate that decides whether a registered generator should handle
/// `path`. Mirrors the role of the matcher in
/// `facebook::velox::filesystems::registerFileSystem`.
using CudfDataSourceMatcher = std::function<bool(std::string_view path)>;

/// Factory that constructs a `cudf::io::datasource` for `path` using
/// `properties` for credentials / endpoint configuration.
using CudfDataSourceGenerator =
    std::function<std::shared_ptr<cudf::io::datasource>(
        std::string_view path,
        const std::shared_ptr<const config::ConfigBase>& properties)>;

/// Registers a (matcher, generator) pair. The first registered matcher
/// whose predicate accepts a given path is the one that handles it.
/// Mirrors `facebook::velox::filesystems::registerFileSystem` but kept on
/// the cudf side so cudf::io types do not leak into the upstream
/// filesystem registry.
void registerCudfDataSource(
    CudfDataSourceMatcher matcher,
    CudfDataSourceGenerator generator);

/// Returns a datasource for `path` if any registered matcher accepts it,
/// or nullptr if none does.
std::shared_ptr<cudf::io::datasource> getCudfDataSource(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties);

/// Removes every previously registered (matcher, generator) entry.
/// Intended for test teardown so successive test binaries do not stack
/// duplicate registrations.
void unregisterCudfDataSources();

} // namespace facebook::velox::cudf_velox::filesystems
