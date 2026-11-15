/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/dwio/common/ColumnSelector.h"
#include "velox/dwio/nimble/reader/BatchReader.h"

namespace facebook::nimble {

/// Fills the flat-map feature (key) selection of `params` from a Velox column
/// selection. `selector` carries per-column key projections as JSON key lists,
/// e.g. `col#["k1","k2"]` to read only those keys, or `col#["!k1"]` to read all
/// keys but k1. `asStructMap` maps column node ids to the keys of flat maps
/// that should be read as structs, as in
/// RowReaderOptions::mapColumnIdAsStruct().
void populateFeatureSelector(
    const velox::dwio::common::ColumnSelector& selector,
    const std::unordered_map<uint32_t, std::vector<std::string>>& asStructMap,
    BatchReadParams& params);

} // namespace facebook::nimble
