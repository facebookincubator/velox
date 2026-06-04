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

#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/PrestoFunctions.h"

namespace facebook::velox::cudf_velox {
namespace {

void registerPrestoArrayAccessFunctions(const std::string& prefix) {
  // Presto element_at is 1-based, allows negative indices from the end, and
  // returns NULL for out-of-bounds indices.
  registerArrayAccessFunction(
      prefix + "element_at",
      ArrayAccessPolicy{
          .allowNegativeIndices = true,
          .nullOnNegativeIndices = false,
          .allowOutOfBound = true,
          .indexStartsAtOne = true,
      },
      arrayAccessSignatures({"integer", "bigint"}));

  // Presto subscript is 1-based and raises on negative or out-of-bounds
  // indices.
  registerArrayAccessFunction(
      prefix + "subscript",
      ArrayAccessPolicy{
          .allowNegativeIndices = false,
          .nullOnNegativeIndices = false,
          .allowOutOfBound = false,
          .indexStartsAtOne = true,
      },
      arrayAccessSignatures({"integer", "bigint"}));
}

} // namespace

void registerPrestoFunctions(const std::string& prefix) {
  registerPrestoArrayAccessFunctions(prefix);
}

} // namespace facebook::velox::cudf_velox
