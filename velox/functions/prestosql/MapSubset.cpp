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

#include "velox/functions/prestosql/MapSubset.h"
#include "velox/functions/Registerer.h"

namespace facebook::velox::functions {
template <typename T>
void registerMapSubsetPrimitive(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapSubsetPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({name}, true, defaultOwner);
}

void registerMapSubset(const std::string& name, std::string_view defaultOwner) {
  registerMapSubsetPrimitive<bool>(name, defaultOwner);
  registerMapSubsetPrimitive<int8_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int16_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int32_t>(name, defaultOwner);
  registerMapSubsetPrimitive<int64_t>(name, defaultOwner);
  registerMapSubsetPrimitive<float>(name, defaultOwner);
  registerMapSubsetPrimitive<double>(name, defaultOwner);
  registerMapSubsetPrimitive<Timestamp>(name, defaultOwner);
  registerMapSubsetPrimitive<Date>(name, defaultOwner);

  registerFunction<
      MapSubsetVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({name}, {}, true, defaultOwner);

  registerFunction<
      MapSubsetFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({name}, {}, true, defaultOwner);
}
} // namespace facebook::velox::functions
