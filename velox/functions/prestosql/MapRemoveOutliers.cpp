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

#include "velox/functions/prestosql/MapRemoveOutliers.h"
#include "velox/functions/Registerer.h"

namespace facebook::velox::functions {

template <typename Key, typename Value>
void registerMapRemoveOutliersPrimitive(const std::string& name) {
  registerFunction<
      ParameterBinder<MapRemoveOutliersFunction, Key, Value>,
      Map<Key, Value>,
      Map<Key, Value>,
      double>({name});
}

void registerMapRemoveOutliers(const std::string& name) {
  registerMapRemoveOutliersPrimitive<bool, int8_t>(name);
  registerMapRemoveOutliersPrimitive<bool, int16_t>(name);
  registerMapRemoveOutliersPrimitive<bool, int32_t>(name);
  registerMapRemoveOutliersPrimitive<bool, int64_t>(name);
  registerMapRemoveOutliersPrimitive<bool, float>(name);
  registerMapRemoveOutliersPrimitive<bool, double>(name);

  registerMapRemoveOutliersPrimitive<int8_t, int8_t>(name);
  registerMapRemoveOutliersPrimitive<int8_t, int16_t>(name);
  registerMapRemoveOutliersPrimitive<int8_t, int32_t>(name);
  registerMapRemoveOutliersPrimitive<int8_t, int64_t>(name);
  registerMapRemoveOutliersPrimitive<int8_t, float>(name);
  registerMapRemoveOutliersPrimitive<int8_t, double>(name);

  registerMapRemoveOutliersPrimitive<int16_t, int8_t>(name);
  registerMapRemoveOutliersPrimitive<int16_t, int16_t>(name);
  registerMapRemoveOutliersPrimitive<int16_t, int32_t>(name);
  registerMapRemoveOutliersPrimitive<int16_t, int64_t>(name);
  registerMapRemoveOutliersPrimitive<int16_t, float>(name);
  registerMapRemoveOutliersPrimitive<int16_t, double>(name);

  registerMapRemoveOutliersPrimitive<int32_t, int8_t>(name);
  registerMapRemoveOutliersPrimitive<int32_t, int16_t>(name);
  registerMapRemoveOutliersPrimitive<int32_t, int32_t>(name);
  registerMapRemoveOutliersPrimitive<int32_t, int64_t>(name);
  registerMapRemoveOutliersPrimitive<int32_t, float>(name);
  registerMapRemoveOutliersPrimitive<int32_t, double>(name);

  registerMapRemoveOutliersPrimitive<int64_t, int8_t>(name);
  registerMapRemoveOutliersPrimitive<int64_t, int16_t>(name);
  registerMapRemoveOutliersPrimitive<int64_t, int32_t>(name);
  registerMapRemoveOutliersPrimitive<int64_t, int64_t>(name);
  registerMapRemoveOutliersPrimitive<int64_t, float>(name);
  registerMapRemoveOutliersPrimitive<int64_t, double>(name);

  registerMapRemoveOutliersPrimitive<float, int8_t>(name);
  registerMapRemoveOutliersPrimitive<float, int16_t>(name);
  registerMapRemoveOutliersPrimitive<float, int32_t>(name);
  registerMapRemoveOutliersPrimitive<float, int64_t>(name);
  registerMapRemoveOutliersPrimitive<float, float>(name);
  registerMapRemoveOutliersPrimitive<float, double>(name);

  registerMapRemoveOutliersPrimitive<double, int8_t>(name);
  registerMapRemoveOutliersPrimitive<double, int16_t>(name);
  registerMapRemoveOutliersPrimitive<double, int32_t>(name);
  registerMapRemoveOutliersPrimitive<double, int64_t>(name);
  registerMapRemoveOutliersPrimitive<double, float>(name);
  registerMapRemoveOutliersPrimitive<double, double>(name);

  registerMapRemoveOutliersPrimitive<Timestamp, int8_t>(name);
  registerMapRemoveOutliersPrimitive<Timestamp, int16_t>(name);
  registerMapRemoveOutliersPrimitive<Timestamp, int32_t>(name);
  registerMapRemoveOutliersPrimitive<Timestamp, int64_t>(name);
  registerMapRemoveOutliersPrimitive<Timestamp, float>(name);
  registerMapRemoveOutliersPrimitive<Timestamp, double>(name);

  registerMapRemoveOutliersPrimitive<Date, int8_t>(name);
  registerMapRemoveOutliersPrimitive<Date, int16_t>(name);
  registerMapRemoveOutliersPrimitive<Date, int32_t>(name);
  registerMapRemoveOutliersPrimitive<Date, int64_t>(name);
  registerMapRemoveOutliersPrimitive<Date, float>(name);
  registerMapRemoveOutliersPrimitive<Date, double>(name);

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, bool>,
      Map<bool, Varchar>,
      Map<bool, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, int8_t>,
      Map<int8_t, Varchar>,
      Map<int8_t, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, int16_t>,
      Map<int16_t, Varchar>,
      Map<int16_t, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, int32_t>,
      Map<int32_t, Varchar>,
      Map<int32_t, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, int64_t>,
      Map<int64_t, Varchar>,
      Map<int64_t, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, float>,
      Map<float, Varchar>,
      Map<float, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, double>,
      Map<double, Varchar>,
      Map<double, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, Timestamp>,
      Map<Timestamp, Varchar>,
      Map<Timestamp, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, Date>,
      Map<Date, Varchar>,
      Map<Date, Varchar>,
      double>({name});

  registerFunction<
      ParameterBinder<MapRemoveOutliersVarcharFunction, Varchar>,
      Map<Varchar, Varchar>,
      Map<Varchar, Varchar>,
      double>({name});

  registerFunction<
      MapRemoveOutliersGenericFunction,
      Map<Generic<T1>, Generic<T2>>,
      Map<Generic<T1>, Generic<T2>>,
      double>({name});
}

} // namespace facebook::velox::functions
