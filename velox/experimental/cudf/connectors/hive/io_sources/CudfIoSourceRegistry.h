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

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

/// Predicate that decides whether a registered factory should handle
/// `path`. Mirrors the role of the matcher in
/// `facebook::velox::filesystems::registerFileSystem`, but operates on
/// cuDF IO sources rather than Velox filesystems.
using CudfIoSourceMatcher = std::function<bool(std::string_view path)>;

/// Factory that constructs a `cudf::io::datasource` for `path` using
/// `properties` for credentials / endpoint configuration.
using CudfIoSourceFactory =
    std::function<std::shared_ptr<cudf::io::datasource>(
        std::string_view path,
        const std::shared_ptr<const config::ConfigBase>& properties)>;

/// Registers a (matcher, factory) pair. Matchers are tried in
/// registration order on lookup; the first matcher whose predicate
/// accepts a given path produces the IO source for it. Use this for
/// per-backend native IO sources (e.g. ABFS) whose matcher recognizes a
/// specific URI scheme.
void registerCudfIoSource(
    CudfIoSourceMatcher matcher,
    CudfIoSourceFactory factory);

/// Installs a catch-all factory used when no matcher-based registration
/// claims the path. Only one default slot exists; calling this again
/// replaces the previously installed factory. KvikIO registers itself
/// here via `registerCudfKvikIoSource` so any path not claimed by a
/// backend-specific source is served by cuDF's bundled datasources.
void registerCudfDefaultIoSource(CudfIoSourceFactory factory);

/// Returns an IO source for `path`. Walks matcher-based registrations
/// first (FIFO), then falls through to the default factory if one is
/// installed. Returns nullptr if nothing matches and no default is set.
std::shared_ptr<cudf::io::datasource> getCudfIoSource(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties);

/// Removes every previously registered (matcher, factory) entry and
/// clears the default factory slot. Intended for test teardown so
/// successive test binaries do not stack duplicate registrations.
void unregisterCudfIoSources();

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
