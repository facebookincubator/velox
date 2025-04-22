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
#include "velox/functions/lib/Map.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_map,
    MapFunction<false>::signatures(),
    std::make_unique<MapFunction</*AllowDuplicateKeys=*/false>>());

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_map_allow_duplicates,
    MapFunction</*AllowDuplicateKeys=*/true>::signatures(),
    std::make_unique<MapFunction</*AllowDuplicateKeys=*/true>>());

void registerMapFunction(const std::string& name, bool allowDuplicateKeys) {
  if (allowDuplicateKeys) {
    VELOX_REGISTER_VECTOR_FUNCTION(udf_map_allow_duplicates, name);
  } else {
    VELOX_REGISTER_VECTOR_FUNCTION(udf_map, name);
  }
}
} // namespace facebook::velox::functions
