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

namespace facebook::velox::functions {
namespace {
std::vector<std::shared_ptr<exec::FunctionSignature>> mapFromArraysSignatures() {
    // array(K), array(V) -> map(K,V)
    return {exec::FunctionSignatureBuilder()
            .typeVariable("K")
            .typeVariable("V")
            .returnType("map(K,V)")
            .argumentType("array(K)")
            .argumentType("array(V)")
            .build()};
}

std::unique_ptr<exec::VectorFunction> createMapFromArrayFunction(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  const std::string mapKeyDedupPolicy = config.sparkMapKeyDedupPolicy();

  if (mapKeyDedupPolicy == "LAST_WIN") {
    return std::make_unique<MapFunction</*AllowDuplicateKeys=*/true>>();
  } else if (mapKeyDedupPolicy == "EXCEPTION") {
    return std::make_unique<MapFunction</*AllowDuplicateKeys=*/false>>();
  } else {
    VELOX_FAIL("Unknown mapKeyDedupPolicy: {}", mapKeyDedupPolicy);
  }
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(
    udf_map_from_arrays,
    mapFromArraysSignatures(),
    createMapFromArrayFunction);
} // namespace facebook::velox::functions
