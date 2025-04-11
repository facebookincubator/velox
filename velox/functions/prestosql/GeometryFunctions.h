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

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Varchar>& wkt) {
    try {
      geos::io::WKTReader reader;
      auto geosGeometry = reader.read(wkt);
      GeometryUtils::serialize(geosGeometry, result);
    } catch (const geos::util::GEOSException& e) {
      return Status::UserError(
          fmt::format("Failed to parse WKT: {}", e.what()));
    } catch (const std::exception& e) {
      return Status::UserError(
          fmt::format("Error converting WKT to Geometry: {}", e.what()));
    }

    return Status::OK();
  }
};

template <typename T>
struct StGeomFromBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Varbinary>& wkb) {
    try {
      geos::io::WKBReader reader;
      auto geosGeometry =
          reader.read(reinterpret_cast<const uint8_t*>(wkb.data()), wkb.size());
      GeometryUtils::serialize(geosGeometry, result);
    } catch (const geos::util::GEOSException& e) {
      return Status::UserError(
          fmt::format("Failed to parse WKB: {}", e.what()));
    } catch (const std::exception& e) {
      return Status::UserError(
          fmt::format("Error converting WKB to Geometry: {}", e.what()));
    }

    return Status::OK();
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& geometry) {
    try {
      geos::io::WKTWriter writer;
      auto geosGeometry = GeometryUtils::deserialize(geometry);
      result = writer.write(geosGeometry.get());
    } catch (const std::exception& e) {
      return Status::UserError(fmt::format(
          "Error deserializing Geometry for WKT conversion: {}", e.what()));
    }

    return Status::OK();
  }
};

template <typename T>
struct StAsBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varbinary>& result, const arg_type<Geometry>& geometry) {
    try {
      geos::io::WKBWriter writer;
      std::ostringstream os;
      auto geosGeometry = GeometryUtils::deserialize(geometry);
      writer.write(*geosGeometry, os);
      const auto str = os.str();
      result.resize(str.size());
      std::memcpy(result.data(), str.data(), str.size());
    } catch (const std::exception& e) {
      return Status::UserError(fmt::format(
          "Error deserializing Geometry for WKB conversion: {}", e.what()));
    }

    return Status::OK();
  }
};

} // namespace facebook::velox::functions
