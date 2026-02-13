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

#include <cstddef>
namespace facebook::velox::exec {

/// Runs the fuzzer for SpatialJoin operator. Generates random geometry data
/// and spatial join plans with various predicates (ST_Intersects, ST_Contains,
/// ST_Within, ST_Distance), comparing SpatialJoin results against
/// NestedLoopJoin as the reference implementation.
///
/// The fuzzer tests:
/// - Different spatial predicates
/// - INNER and LEFT join types (the only types supported by SpatialJoin)
/// - Different geometry types (POINT, POLYGON, LINESTRING)
/// - Various data distributions (uniform, clustered, sparse)
/// - Different sizes of probe and build sides
/// - Plans with and without filters
/// - Different output column projections
void spatialJoinFuzzer(size_t seed);

} // namespace facebook::velox::exec
