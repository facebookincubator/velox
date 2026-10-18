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

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/file/File.h"

#include <cudf/io/datasource.hpp>

#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive {

/// Returns 'kvikioSource' wrapped so that its reads go through the async data
/// cache, or 'kvikioSource' itself when the async data cache is null or the
/// split is not cacheable. Presto marks splits cacheable only under soft or
/// hard affinity scheduling (Hive config hive.node-selection-strategy, session
/// property node_selection_strategy). With Presto's default NO_PREFERENCE, the
/// async data cache is therefore never used.
std::unique_ptr<cudf::io::datasource> maybeCacheKvikioDataSource(
    std::unique_ptr<cudf::io::datasource> kvikioSource,
    std::string_view path,
    cache::AsyncDataCache* cache,
    bool cacheable,
    std::shared_ptr<IoStats> ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
