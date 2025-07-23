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

#include <geos/geom/Coordinate.h>
#include <geos/geom/Envelope.h>
#include <geos/io/WKBReader.h>
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>
#include <geos/simplify/TopologyPreservingSimplifier.h>
#include <geos/util/AssertionFailedException.h>
#include <geos/util/UnsupportedOperationException.h>
#include <cmath>

#include <velox/type/StringView.h>
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/geospatial/GeometrySerde.h"
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

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
    geospatial::GeometrySerializer::serialize(*geosGeometry, result);
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
    geospatial::GeometrySerializer::serialize(*geosGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

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
        geospatial::GeometryDeserializer::deserialize(geometry);
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

template <typename T>
struct StPointFunction {
  StPointFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<double>& x,
      const arg_type<double>& y) {
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return Status::UserError(fmt::format(
          "ST_Point requires finite coordinates, got x={} y={}", x, y));
    }
    GEOS_TRY(
        {
          auto point = std::unique_ptr<geos::geom::Point>(
              factory_->createPoint(geos::geom::Coordinate(x, y)));
          geospatial::GeometrySerializer::serialize(*point, result);
        },
        "Failed to create point geometry");
    return Status::OK();
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StPolygonFunction {
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
    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_Polygon");

    geospatial::GeometrySerializer::serialize(*geosGeometry, result);

    return validate;
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(outputGeometry = leftGeosGeometry->difference(&*rightGeosGeometry);
             , "Failed to compute geometry difference");

    geospatial::GeometrySerializer::serialize(*outputGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StBoundaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;

    GEOS_TRY(geospatial::GeometrySerializer::serialize(
                 *geosGeometry->getBoundary(), result);
             , "Failed to compute geometry boundary");

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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->intersection(&*rightGeosGeometry);
        , "Failed to compute geometry intersection");

    geospatial::GeometrySerializer::serialize(*outputGeometry, result);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->symDifference(&*rightGeosGeometry);
        , "Failed to compute geometry symdifference");

    geospatial::GeometrySerializer::serialize(*outputGeometry, result);
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
        geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(outputGeometry = leftGeosGeometry->Union(&*rightGeosGeometry);
             , "Failed to compute geometry union");

    geospatial::GeometrySerializer::serialize(*outputGeometry, result);
    return Status::OK();
  }
};

// Accessors

template <typename T>
struct StIsValidFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    GEOS_TRY(result = geosGeometry->isValid();
             , "Failed to check geometry isValid");

    return Status::OK();
  }
};

template <typename T>
struct StIsSimpleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    GEOS_TRY(result = geosGeometry->isSimple();
             , "Failed to check geometry isSimple");

    return Status::OK();
  }
};

template <typename T>
struct GeometryInvalidReasonFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    std::optional<std::string> messageOpt =
        geospatial::geometryInvalidReason(geosGeometry.get());

    if (messageOpt.has_value()) {
      result = messageOpt.value();
    }
    return messageOpt.has_value();
  }
};

template <typename T>
struct StAreaFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<double>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;

    GEOS_TRY(result = geosGeometry->getArea();
             , "Failed to compute geometry area");

    return Status::OK();
  }
};

template <typename T>
struct StCentroidFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    auto validate = facebook::velox::functions::geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POINT,
         geos::geom::GeometryTypeId::GEOS_MULTIPOINT,
         geos::geom::GeometryTypeId::GEOS_LINESTRING,
         geos::geom::GeometryTypeId::GEOS_MULTILINESTRING,
         geos::geom::GeometryTypeId::GEOS_POLYGON,
         geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON},
        "ST_Centroid");

    if (!validate.ok()) {
      return validate;
    }

    geos::geom::GeometryTypeId type = geosGeometry->getGeometryTypeId();
    if (type == geos::geom::GeometryTypeId::GEOS_POINT) {
      result = input;
      return Status::OK();
    }

    if (geosGeometry->getNumPoints() == 0) {
      GEOS_TRY(
          {
            geos::geom::GeometryFactory::Ptr factory =
                geos::geom::GeometryFactory::create();
            std::unique_ptr<geos::geom::Point> point = factory->createPoint();
            geospatial::GeometrySerializer::serialize(*point, result);
            factory->destroyGeometry(point.release());
          },
          "Failed to create point geometry");
      return Status::OK();
    }

    geospatial::GeometrySerializer::serialize(
        *(geosGeometry->getCentroid()), result);
    return Status::OK();
  }
};

template <typename T>
struct StXFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);
    if (geosGeometry->getGeometryTypeId() !=
        geos::geom::GeometryTypeId::GEOS_POINT) {
      VELOX_USER_FAIL(fmt::format(
          "ST_X requires a Point geometry, found {}",
          geosGeometry->getGeometryType()));
    }
    if (geosGeometry->isEmpty()) {
      return false;
    }
    auto coordinate = geosGeometry->getCoordinate();
    result = coordinate->x;
    return true;
  }
};

template <typename T>
struct StYFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);
    if (geosGeometry->getGeometryTypeId() !=
        geos::geom::GeometryTypeId::GEOS_POINT) {
      VELOX_USER_FAIL(fmt::format(
          "ST_Y requires a Point geometry, found {}",
          geosGeometry->getGeometryType()));
    }
    if (geosGeometry->isEmpty()) {
      return false;
    }
    auto coordinate = geosGeometry->getCoordinate();
    result = coordinate->y;
    return true;
  }
};

template <typename T>
struct StXMinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(out_type<double>& result, const arg_type<Geometry>& geometry) {
    const std::unique_ptr<geos::geom::Envelope> env =
        geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (env->isNull()) {
      return false;
    }
    result = env->getMinX();
    return true;
  }
};

template <typename T>
struct StYMinFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(out_type<double>& result, const arg_type<Geometry>& geometry) {
    const std::unique_ptr<geos::geom::Envelope> env =
        geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (env->isNull()) {
      return false;
    }
    result = env->getMinY();
    return true;
  }
};

template <typename T>
struct StXMaxFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(out_type<double>& result, const arg_type<Geometry>& geometry) {
    const std::unique_ptr<geos::geom::Envelope> env =
        geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (env->isNull()) {
      return false;
    }
    result = env->getMaxX();
    return true;
  }
};

template <typename T>
struct StYMaxFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  bool call(out_type<double>& result, const arg_type<Geometry>& geometry) {
    const std::unique_ptr<geos::geom::Envelope> env =
        geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (env->isNull()) {
      return false;
    }
    result = env->getMaxY();
    return true;
  }
};

template <typename T>
struct SimplifyGeometryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<double>& tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0) {
      return Status::UserError(
          "simplification tolerance must be a non-negative finite number");
    }
    if (tolerance == 0) {
      result = geometry;
      return Status::OK();
    }

    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      result = geometry;
      return Status::OK();
    }

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        {
          outputGeometry =
              geos::simplify::TopologyPreservingSimplifier::simplify(
                  geosGeometry.get(), tolerance);
        },
        "Failed to compute simplified geometry");

    geospatial::GeometrySerializer::serialize(*outputGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StGeometryTypeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(input);

    result = geosGeometry->getGeometryType();

    return Status::OK();
  }
};

template <typename T>
struct StDistanceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Geometry>& geometry1,
      const arg_type<Geometry>& geometry2) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry1 =
        geospatial::GeometryDeserializer::deserialize(geometry1);
    std::unique_ptr<geos::geom::Geometry> geosGeometry2 =
        geospatial::GeometryDeserializer::deserialize(geometry2);

    if (geosGeometry1->getSRID() != geosGeometry2->getSRID()) {
      VELOX_USER_FAIL(fmt::format(
          "Input geometries must have the same spatial reference, found {} and {}",
          geosGeometry1->getSRID(),
          geosGeometry2->getSRID()));
    }

    if (geosGeometry1->isEmpty() || geosGeometry2->isEmpty()) {
      return false;
    }

    GEOS_RETHROW(result = geosGeometry1->distance(geosGeometry2.get());
                 , "Failed to calculate geometry distance");

    return true;
  }
};

template <typename T>
struct StIsClosedFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING,
         geos::geom::GeometryTypeId::GEOS_MULTILINESTRING},
        "ST_IsClosed");

    if (!validate.ok()) {
      return validate;
    };

    if (geos::geom::LineString* lineString =
            dynamic_cast<geos::geom::LineString*>(geosGeometry.get())) {
      result = lineString->isClosed();
      return validate;
    }
    if (geos::geom::MultiLineString* multiLineString =
            dynamic_cast<geos::geom::MultiLineString*>(geosGeometry.get())) {
      result = multiLineString->isClosed();
      return validate;
    }

    VELOX_FAIL(
        "Validation passed but type not recognized as LineString or MultiLineString");
  }
};

template <typename T>
struct StIsEmptyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& geometry) {
    GEOS_TRY(
        {
          const std::unique_ptr<geos::geom::Envelope> env =
              geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
          result = env->isNull();
        },
        "Failed to get envelope from geometry");
    return Status::OK();
  }
};

template <typename T>
struct StIsRingFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "ST_IsRing");

    if (!validate.ok()) {
      return validate;
    };

    if (geos::geom::LineString* lineString =
            dynamic_cast<geos::geom::LineString*>(geosGeometry.get())) {
      result = lineString->isRing();
      return validate;
    }

    VELOX_FAIL("Validation passed but type not recognized as LineString");
  }
};

template <typename T>
struct StLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<double>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING,
         geos::geom::GeometryTypeId::GEOS_MULTILINESTRING},
        "ST_Length");

    if (!validate.ok()) {
      return validate;
    };

    if (geos::geom::LineString* lineString =
            dynamic_cast<geos::geom::LineString*>(geosGeometry.get())) {
      result = lineString->getLength();
      return validate;
    }
    if (geos::geom::MultiLineString* multiLineString =
            dynamic_cast<geos::geom::MultiLineString*>(geosGeometry.get())) {
      result = multiLineString->getLength();
      return validate;
    }

    VELOX_FAIL(
        "Validation passed but type not recognized as LineString or MultiLineString");
  }
};

template <typename T>
struct StPointNFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& index) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "ST_PointN");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    };

    if (geos::geom::LineString* lineString =
            dynamic_cast<geos::geom::LineString*>(geosGeometry.get())) {
      if (index < 1 || index > lineString->getNumPoints()) {
        return false;
      }
      geospatial::GeometrySerializer::serialize(
          *lineString->getPointN(index - 1), result);
      return true;
    }

    VELOX_FAIL(
        "Validation passed but type not recognized as LineString or MultiLineString");
  }
};

template <typename T>
struct StStartPointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "ST_StartPoint");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }
    if (geosGeometry->isEmpty()) {
      return false;
    }
    geos::geom::LineString* lineString =
        static_cast<geos::geom::LineString*>(geosGeometry.get());
    geospatial::GeometrySerializer::serialize(
        *(lineString->getStartPoint()), result);

    return true;
  }
};

template <typename T>
struct StEndPointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "ST_EndPoint");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }
    if (geosGeometry->isEmpty()) {
      return false;
    }
    geos::geom::LineString* lineString =
        static_cast<geos::geom::LineString*>(geosGeometry.get());
    geospatial::GeometrySerializer::serialize(
        *lineString->getEndPoint(), result);

    return true;
  }
};

template <typename T>
struct StGeometryNFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& index) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      return false;
    }

    if (!geospatial::isMultiType(*geosGeometry)) {
      if (index == 1) {
        geospatial::GeometrySerializer::serialize(*geosGeometry, result);
        return true;
      }
      return false;
    }

    if (geos::geom::GeometryCollection* geomCollection =
            dynamic_cast<geos::geom::GeometryCollection*>(geosGeometry.get())) {
      if (index < 1 || index > geomCollection->getNumGeometries()) {
        return false;
      }
      geospatial::GeometrySerializer::serialize(
          *(geosGeometry->getGeometryN(index - 1)), result);
      return true;
    }

    return false;
  }
};

template <typename T>
struct StInteriorRingNFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& index) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_InteriorRingN");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    geos::geom::Polygon* polygon =
        static_cast<geos::geom::Polygon*>(geosGeometry.get());
    if (index < 1 || index > polygon->getNumInteriorRing()) {
      return false;
    }
    geospatial::GeometrySerializer::serialize(
        *(polygon->getInteriorRingN(index - 1)), result);
    return true;
  }
};

template <typename T>
struct StNumGeometriesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<int32_t>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      result = 0;
    } else {
      uint64_t numGeometries = geosGeometry->getNumGeometries();
      if (numGeometries > std::numeric_limits<int32_t>::max()) {
        return Status::UserError(
            "Number of geometries exceeds the maximum value of int32");
      }
      result = static_cast<int32_t>(numGeometries);
    }
    return Status::OK();
  }
};

template <typename T>
struct StNumInteriorRingFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<int32_t>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_NumInteriorRing");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    if (geosGeometry->isEmpty()) {
      return false;
    }

    geos::geom::Polygon* polygon =
        static_cast<geos::geom::Polygon*>(geosGeometry.get());
    uint64_t numInteriorRings = polygon->getNumInteriorRing();
    if (numInteriorRings > std::numeric_limits<int32_t>::max()) {
      VELOX_USER_FAIL(
          "Number of interior rings exceeds the maximum value of int32");
    }
    result = static_cast<int32_t>(numInteriorRings);
    return true;
  }
};

template <typename T>
struct StConvexHullFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty() ||
        geosGeometry->getGeometryTypeId() ==
            geos::geom::GeometryTypeId::GEOS_POINT) {
      result = geometry;
    } else {
      geospatial::GeometrySerializer::serialize(
          *(geosGeometry->convexHull()), result);
    }
    return Status::OK();
  }
};

template <typename T>
struct StDimensionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<int8_t>& result, const arg_type<Geometry>& geometry) {
    result =
        geospatial::GeometryDeserializer::deserialize(geometry)->getDimension();

    return Status::OK();
  }
};

template <typename T>
struct StExteriorRingFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_ExteriorRing");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    if (geosGeometry->isEmpty()) {
      return false;
    }

    geos::geom::Polygon* polygon =
        dynamic_cast<geos::geom::Polygon*>(geosGeometry.get());

    VELOX_CHECK_NOT_NULL(
        polygon, "Validation passed but type not recognized as Polygon");

    geospatial::GeometrySerializer::serialize(
        *(polygon->getExteriorRing()), result);

    return true;
  }
};

template <typename T>
struct StEnvelopeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  StEnvelopeFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<const geos::geom::Geometry> geosGeometry =
        geospatial::GeometryDeserializer::deserialize(geometry);

    auto env = geosGeometry->getEnvelope();

    if (env->isEmpty()) {
      GEOS_TRY(
          {
            auto polygon =
                std::unique_ptr<geos::geom::Polygon>(factory_->createPolygon());
            geospatial::GeometrySerializer::serialize(*polygon, result);
          },
          "Failed to create empty polygon in ST_Envelope");
    }

    geospatial::GeometrySerializer::serialize(*env, result);

    return Status::OK();
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

} // namespace facebook::velox::functions
