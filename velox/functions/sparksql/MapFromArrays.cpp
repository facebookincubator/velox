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
#include "velox/functions/sparksql/MapFromArrays.h"

namespace facebook::velox::functions::sparksql {

std::shared_ptr<exec::VectorFunction> makeMapFromArrays(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  if (config.throwExceptionOnDuplicateMapKeys()) {
    return std::make_unique<
        MapFunction</*AllowDuplicateKeys=*/false, /*DeduplicateKeys=*/false>>();
  } else {
    return std::make_unique<
        MapFunction</*AllowDuplicateKeys=*/true, /*DeduplicateKeys=*/true>>();
  }
}

std::vector<std::shared_ptr<exec::FunctionSignature>>
getMapFromArraysSignature() {
  return MapFunction<false, false>::signatures();
}

void registerMapFromArrays(const std::string& name) {
  exec::registerStatefulVectorFunction(
      name, getMapFromArraysSignature(), makeMapFromArrays);
}

} // namespace facebook::velox::functions::sparksql
