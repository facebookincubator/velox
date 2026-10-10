/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/type/Filter.h"

#include <memory>

namespace facebook::velox::cudf_velox::connector::hive {

/// Marks bounds for row-group pruning only. The producing join must remain.
/// Shared copies preserve this mode; clones and merges require full filtering.
common::FilterPtr makeRowGroupOnlyFilter(
    std::unique_ptr<common::Filter> filter);

/// Whether the filter can be used for statistics pruning without a row mask.
bool isRowGroupOnlyFilter(const common::FilterPtr& filter);

} // namespace facebook::velox::cudf_velox::connector::hive
