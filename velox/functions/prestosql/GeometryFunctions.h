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

#include <velox/type/StringView.h>
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/geospatial/GeometrySerde.h"
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

// Constructors and Serde

template <typename T>
struct StGeometryFromTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Varchar>& wkt) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry;
    GEOS_TRY(
        {
          geos::io::WKTReader reader;
          geosGeometry = reader.read(wkt);
        },
        "Failed to parse WKT");
    result = geospatial::serializeGeometry(*geosGeometry);
    return Status::OK();
  }
};

template <typename T>
struct StGeomFromBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Varbinary>& wkb) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry;

    GEOS_TRY(
        {
          geos::io::WKBReader reader;
          geosGeometry = reader.read(
              reinterpret_cast<const uint8_t*>(wkb.data()), wkb.size());
        },
        "Failed to parse WKB");
    result = geospatial::serializeGeometry(*geosGeometry);
    return Status::OK();
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::deserializeGeometry(geometry);

    GEOS_TRY(
        {
          geos::io::WKTWriter writer;
          writer.setTrim(true);
          result = writer.write(geosGeometry.get());
        },
        "Failed to write WKT");
    return Status::OK();
  }
};

template <typename T>
struct StAsBinaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varbinary>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::deserializeGeometry(geometry);
    GEOS_TRY(
        {
          geos::io::WKBWriter writer;
          std::ostringstream outputStream;
          writer.write(*geosGeometry, outputStream);
          result = outputStream.str();
        },
        "Failed to write WKB");
    return Status::OK();
  }
};

// Predicates

template <typename T>
struct StRelateFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry,
      const arg_type<Varchar>& relation) {
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->relate(*rightGeosGeometry, relation);
             , "Failed to check geometry relation");

    return Status::OK();
  }
};

template <typename T>
struct StContainsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->contains(&*rightGeosGeometry);
             , "Failed to check geometry contains");

    return Status::OK();
  }
};

template <typename T>
struct StCrossesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->crosses(&*rightGeosGeometry);
             , "Failed to check geometry crosses");

    return Status::OK();
  }
};

template <typename T>
struct StDisjointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->disjoint(&*rightGeosGeometry);
             , "Failed to check geometry disjoint");

    return Status::OK();
  }
};

template <typename T>
struct StEqualsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->equals(&*rightGeosGeometry);
             , "Failed to check geometry equals");

    return Status::OK();
  }
};

template <typename T>
struct StIntersectsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->intersects(&*rightGeosGeometry);
             , "Failed to check geometry intersects");

    return Status::OK();
  }
};

template <typename T>
struct StOverlapsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->overlaps(&*rightGeosGeometry);
             , "Failed to check geometry overlaps");

    return Status::OK();
  }
};

template <typename T>
struct StTouchesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->touches(&*rightGeosGeometry);
             , "Failed to check geometry touches");

    return Status::OK();
  }
};

template <typename T>
struct StWithinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<bool>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);
    GEOS_TRY(result = leftGeosGeometry->within(&*rightGeosGeometry);
             , "Failed to check geometry within");

    return Status::OK();
  }
};

// Overlay operations

template <typename T>
struct StDifferenceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    // if envelopes are disjoint
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(outputGeometry = leftGeosGeometry->difference(&*rightGeosGeometry);
             , "Failed to compute geometry difference");

    result = geospatial::serializeGeometry(*outputGeometry);
    return Status::OK();
  }
};

template <typename T>
struct StIntersectionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    // if envelopes are disjoint
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->intersection(&*rightGeosGeometry);
        , "Failed to compute geometry intersection");

    result = geospatial::serializeGeometry(*outputGeometry);
    return Status::OK();
  }
};

template <typename T>
struct StSymDifferenceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit
    // if envelopes are disjoint
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->symDifference(&*rightGeosGeometry);
        , "Failed to compute geometry symdifference");

    result = geospatial::serializeGeometry(*outputGeometry);
    return Status::OK();
  }
};

template <typename T>
struct StUnionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    // TODO: When #12771 is merged, check envelopes and short-circuit if
    // one/both are empty
    std::unique_ptr<geos::geom::Geometry> leftGeosGeometry =
        geospatial::deserializeGeometry(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::deserializeGeometry(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(outputGeometry = leftGeosGeometry->Union(&*rightGeosGeometry);
             , "Failed to compute geometry union");

    result = geospatial::serializeGeometry(*outputGeometry);
    return Status::OK();
  }
};

} // namespace facebook::velox::functions
