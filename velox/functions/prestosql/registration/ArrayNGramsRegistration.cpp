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
inline void registerArrayNGramsFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<ArrayNGramsFunction, Array<Array<T>>, Array<T>, int32_t>(
      {prefix + "ngrams"}, {}, true, defaultOwner);
}

} // namespace
void registerArrayNGramsFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerArrayNGramsFunctions<int8_t>(prefix, defaultOwner);
  registerArrayNGramsFunctions<int16_t>(prefix, defaultOwner);
  registerArrayNGramsFunctions<int32_t>(prefix, defaultOwner);
  registerArrayNGramsFunctions<int64_t>(prefix, defaultOwner);
  registerArrayNGramsFunctions<int128_t>(prefix, defaultOwner);
  registerArrayNGramsFunctions<float>(prefix, defaultOwner);
  registerArrayNGramsFunctions<double>(prefix, defaultOwner);
  registerArrayNGramsFunctions<bool>(prefix, defaultOwner);
  registerArrayNGramsFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayNGramsFunctions<Date>(prefix, defaultOwner);
  registerArrayNGramsFunctions<Varbinary>(prefix, defaultOwner);
  registerArrayNGramsFunctions<Generic<T1>>(prefix, defaultOwner);
  registerFunction<
      ArrayNGramsFunctionString,
      Array<Array<Varchar>>,
      Array<Varchar>,
      int32_t>({prefix + "ngrams"}, {}, true, defaultOwner);
}
} // namespace facebook::velox::functions
