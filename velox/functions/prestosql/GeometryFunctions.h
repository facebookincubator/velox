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

#include <geos/io/WKBReader.h>
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>

#include "velox/functions/Macros.h"
#include "velox/functions/UDFOutputString.h"
#include "velox/functions/prestosql/GeometryUtils.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

template <typename T>
struct StGeometryFromTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Varchar>& wkt) {
    thread_local geos::io::WKTReader reader;
    auto geosGeometry = reader.read(wkt);
    GeometryUtils::serialize(geosGeometry, result);
    return true;
  }
};

template <typename T>
struct StGeomFromBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Varbinary>& wkb) {
    thread_local geos::io::WKBReader reader;
    auto geosGeometry = reader.read(
        reinterpret_cast<const unsigned char*>(wkb.data()), wkb.size());
    GeometryUtils::serialize(geosGeometry, result);
    return true;
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Geometry>& geometryBinary) {
    auto geometry = GeometryUtils::deserialize(geometryBinary);
    thread_local geos::io::WKTWriter writer;
    result = writer.write(geometry.get());
    return true;
  }
};

template <typename T>
struct StAsBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varbinary>& result,
      const arg_type<Geometry>& geometryBinary) {
    auto geometry = GeometryUtils::deserialize(geometryBinary);
    thread_local geos::io::WKBWriter writer;
    std::ostringstream os;
    writer.write(*geometry, os);
    const auto str = os.str();
    result.resize(str.size());
    std::memcpy(result.data(), str.data(), str.size());
    return true;
  }
};

} // namespace facebook::velox::functions
