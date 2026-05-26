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
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive::io_sources {

/// Predicate that decides whether a registered generator should handle
/// `path`. Mirrors the role of the matcher in
/// `facebook::velox::filesystems::registerFileSystem`, but operates on
/// cuDF IO sources rather than Velox filesystems.
using SourceMatcher = std::function<bool(std::string_view path)>;

/// Generator that constructs a `cudf::io::datasource` for `path` using
/// `properties` for credentials / endpoint configuration.
using SourceGenerator = std::function<std::shared_ptr<cudf::io::datasource>(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties)>;

/// Registers a (matcher, generator) pair under `name`. Matchers are tried in
/// registration order on lookup and the first matcher whose predicate accepts a
/// given path produces the IO source for it. Each backend registers exactly one
/// such entry.
void registerCudfIoSource(
    std::string name,
    SourceMatcher matcher,
    SourceGenerator generator);

/// Returns an IO source for `path` by walking matcher-based
/// registrations in FIFO order and returning the first hit. Returns
/// nullptr if no registered backend claims the path; callers are
/// expected to surface this as an error (see `CudfSplitReader`).
std::shared_ptr<cudf::io::datasource> getCudfIoSource(
    std::string_view path,
    const std::shared_ptr<const config::ConfigBase>& properties);

/// Removes every previously registered (matcher, factory) entry.
/// Intended for test teardown so successive test binaries do not stack
/// duplicate registrations.
void unregisterCudfIoSources();

} // namespace facebook::velox::cudf_velox::connector::hive::io_sources
