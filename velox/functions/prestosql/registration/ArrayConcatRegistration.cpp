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
inline void registerArrayConcatFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<ArrayConcatFunction, T>,
      Array<T>,
      Array<T>,
      T>({prefix + "concat"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<ArrayConcatFunction, T>,
      Array<T>,
      T,
      Array<T>>({prefix + "concat"}, true, defaultOwner);
  registerFunction<
      ParameterBinder<ArrayConcatFunction, T>,
      Array<T>,
      Array<T>,
      Array<T>,
      Variadic<Array<T>>>({prefix + "concat"}, true, defaultOwner);
}
} // namespace

void registerArrayConcatFunctions(
    const std::string& prefix,
    std::string_view defaultOwner) {
  registerArrayConcatFunctions<Generic<T1>>(prefix, defaultOwner);
  // Fast paths for primitives types.
  registerArrayConcatFunctions<int8_t>(prefix, defaultOwner);
  registerArrayConcatFunctions<int16_t>(prefix, defaultOwner);
  registerArrayConcatFunctions<int32_t>(prefix, defaultOwner);
  registerArrayConcatFunctions<int64_t>(prefix, defaultOwner);
  registerArrayConcatFunctions<int128_t>(prefix, defaultOwner);
  registerArrayConcatFunctions<float>(prefix, defaultOwner);
  registerArrayConcatFunctions<double>(prefix, defaultOwner);
  registerArrayConcatFunctions<bool>(prefix, defaultOwner);
  registerArrayConcatFunctions<Varchar>(prefix, defaultOwner);
  registerArrayConcatFunctions<Timestamp>(prefix, defaultOwner);
  registerArrayConcatFunctions<Date>(prefix, defaultOwner);
}
} // namespace facebook::velox::functions
