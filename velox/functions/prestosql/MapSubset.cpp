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

#include "velox/functions/prestosql/MapSubset.h"
#include "velox/functions/Registerer.h"

namespace facebook::velox::functions {
template <typename T>
void registerMapSubsetPrimitive(const std::string& name) {
  registerFunction<
      ParameterBinder<MapSubsetPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({name});
}

void registerMapSubset(const std::string& name) {
  registerMapSubsetPrimitive<bool>(name);
  registerMapSubsetPrimitive<int8_t>(name);
  registerMapSubsetPrimitive<int16_t>(name);
  registerMapSubsetPrimitive<int32_t>(name);
  registerMapSubsetPrimitive<int64_t>(name);
  registerMapSubsetPrimitive<float>(name);
  registerMapSubsetPrimitive<double>(name);
  registerMapSubsetPrimitive<Timestamp>(name);
  registerMapSubsetPrimitive<Date>(name);

  registerFunction<
      MapSubsetVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({name});

  registerFunction<
      MapSubsetFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      Array<Generic<T1>>>({name});
}
} // namespace facebook::velox::functions
