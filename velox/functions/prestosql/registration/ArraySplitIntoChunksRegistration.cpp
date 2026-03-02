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

#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/ArrayFunctions.h"

namespace facebook::velox::functions {
namespace {
template <typename T>
inline void registerArraySplitIntoChunksFunctions(const std::string& prefix) {
  registerFunction<
      ArraySplitIntoChunksFunction,
      Array<Array<T>>,
      Array<T>,
      int32_t>({prefix + "array_split_into_chunks"});
}

} // namespace
void registerArraySplitIntoChunksFunctions(const std::string& prefix) {
  registerArraySplitIntoChunksFunctions<int8_t>(prefix);
  registerArraySplitIntoChunksFunctions<int16_t>(prefix);
  registerArraySplitIntoChunksFunctions<int32_t>(prefix);
  registerArraySplitIntoChunksFunctions<int64_t>(prefix);
  registerArraySplitIntoChunksFunctions<int128_t>(prefix);
  registerArraySplitIntoChunksFunctions<float>(prefix);
  registerArraySplitIntoChunksFunctions<double>(prefix);
  registerArraySplitIntoChunksFunctions<bool>(prefix);
  registerArraySplitIntoChunksFunctions<Timestamp>(prefix);
  registerArraySplitIntoChunksFunctions<Date>(prefix);
  registerArraySplitIntoChunksFunctions<Varbinary>(prefix);
  registerArraySplitIntoChunksFunctions<Generic<T1>>(prefix);
  registerFunction<
      ArraySplitIntoChunksFunctionString,
      Array<Array<Varchar>>,
      Array<Varchar>,
      int32_t>({prefix + "array_split_into_chunks"});
}
} // namespace facebook::velox::functions
