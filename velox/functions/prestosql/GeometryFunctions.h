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

#pragma once

#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

enum class GeometryType : int {
  POINT = 0,
  MULTI_POINT = 1,
  LINE_STRING = 2,
  MULTI_LINE_STRING = 3,
  POLYGON = 4,
  MULTI_POLYGON = 5,
  GEOMETRY_COLLECTION = 6,
  ENVELOPE = 7
};

template <typename Geometry>
Geometry deserializeGeometry() {}

template <typename T>
struct StContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      bool& result,
      const arg_type<Geometry>& left,
      const arg_type<Geometry>& right) {
    auto leftType =
        static_cast<GeometryType>(*reinterpret_cast<const int*>(left.data()));
    auto rightType =
        static_cast<GeometryType>(*reinterpret_cast<const int*>(right.data()));

    // Deserialize based on types
  }
};
} // namespace facebook::velox::functions
