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

#include <string>
#include <vector>

namespace facebook::velox::functions::aggregate::sparksql {

/// Register collect_list aggregate function.
/// @param names Names to use for the aggregate function.
/// @param withCompanionFunctions Also register companion functions.
/// @param overwrite Whether to overwrite existing entry in the function
/// registry.
/// @param pinIgnoreNulls When true, null inputs are always ignored and the
/// collect_list.ignore_nulls session config is not consulted. Use for a name
/// whose contract fixes null handling and must not follow session config.
void registerCollectListAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite,
    bool pinIgnoreNulls);

} // namespace facebook::velox::functions::aggregate::sparksql
