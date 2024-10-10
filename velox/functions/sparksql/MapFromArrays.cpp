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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/Map.h"

// Returns a map created using the given key/value arrays.
// See documentation at https://spark.apache.org/docs/latest/api/sql/#map_from_arrays
//
// Example:
// Select map_from_arrays(array(1,2,3), array('a','b','c'));
//
// Result:
// {1:"a",2:"b",3:"c"}

namespace facebook::velox::functions {
namespace {

std::unique_ptr<exec::VectorFunction> createMapFromArrayFunction(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  const auto mapKeyDedupPolicy = config.sparkMapKeyDedupPolicy();

  if (mapKeyDedupPolicy == core::SparkMapKeyDedupPolicy::LAST_WIN) {
    return std::make_unique<MapFunction</*AllowDuplicateKeys=*/true>>();
  } else if (mapKeyDedupPolicy == core::SparkMapKeyDedupPolicy::EXCEPTION) {
    return std::make_unique<MapFunction</*AllowDuplicateKeys=*/false>>();
  } else {
    VELOX_UNREACHABLE();
  }
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_map_from_arrays,
    MapFunction<false>::signatures(),
    createMapFromArrayFunction);
} // namespace facebook::velox::functions
