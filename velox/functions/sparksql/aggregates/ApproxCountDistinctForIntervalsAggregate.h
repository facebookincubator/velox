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

namespace facebook::velox::functions::aggregate::sparksql {

/// Registers the Spark approx_count_distinct_for_intervals aggregate, which
/// estimates the number of distinct values in each interval defined by a
/// constant array of endpoints, under the given prefix.
/// @param prefix Prefix added to the function name.
/// @param withCompanionFunctions Whether to also register the companion
/// (_partial, _merge, _merge_extract, _extract) functions.
/// @param overwrite Whether to overwrite an existing registration.
void registerApproxCountDistinctForIntervalsAggregate(
    const std::string& prefix,
    bool withCompanionFunctions,
    bool overwrite);

} // namespace facebook::velox::functions::aggregate::sparksql
