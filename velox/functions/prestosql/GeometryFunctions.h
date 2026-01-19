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
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/Envelope.h>
#include <geos/io/GeoJSON.h>
#include <geos/io/GeoJSONReader.h>
#include <geos/io/GeoJSONWriter.h>
#include <geos/io/WKBReader.h>
#include <geos/io/WKBWriter.h>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTWriter.h>
#include <geos/linearref/LengthIndexedLine.h>
#include <geos/operation/distance/DistanceOp.h>
#include <geos/simplify/TopologyPreservingSimplifier.h>
#include <geos/util/AssertionFailedException.h>
#include <geos/util/UnsupportedOperationException.h>
#include <cmath>

#include <velox/type/StringView.h>
#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include "velox/functions/prestosql/types/BingTileType.h"
#include "velox/functions/prestosql/types/GeometryType.h"
#include "velox/functions/prestosql/types/SphericalGeographyType.h"
#include "velox/type/Variant.h"

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
          geosGeometry = reader.read(std::string(wkt));
        },
        "Failed to parse WKT");
    common::geospatial::GeometrySerializer::serialize(*geosGeometry, result);
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
    common::geospatial::GeometrySerializer::serialize(*geosGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
struct SphericalAsTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      const arg_type<SphericalGeography>& geography) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geography);

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
        common::geospatial::GeometryDeserializer::deserialize(geometry);
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
      return Status::UserError(
          fmt::format(
              "ST_Point requires finite coordinates, got x={} y={}", x, y));
    }
    GEOS_TRY(
        {
          auto point = std::unique_ptr<geos::geom::Point>(
              factory_->createPoint(geos::geom::Coordinate(x, y)));
          common::geospatial::GeometrySerializer::serialize(*point, result);
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
          geosGeometry = reader.read(std::string(wkt));
        },
        "Failed to parse WKT");
    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_Polygon");

    common::geospatial::GeometrySerializer::serialize(*geosGeometry, result);

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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result =
            leftGeosGeometry->relate(*rightGeosGeometry, std::string(relation));
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->contains(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->crosses(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->disjoint(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->equals(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->intersects(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->overlaps(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->touches(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);
    GEOS_TRY(
        result = leftGeosGeometry->within(&*rightGeosGeometry);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->difference(&*rightGeosGeometry);
        , "Failed to compute geometry difference");

    common::geospatial::GeometrySerializer::serialize(*outputGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StBoundaryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(input);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;

    GEOS_TRY(
        common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->intersection(&*rightGeosGeometry);
        , "Failed to compute geometry intersection");

    common::geospatial::GeometrySerializer::serialize(*outputGeometry, result);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->symDifference(&*rightGeosGeometry);
        , "Failed to compute geometry symdifference");

    common::geospatial::GeometrySerializer::serialize(*outputGeometry, result);
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
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    std::unique_ptr<geos::geom::Geometry> rightGeosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);

    std::unique_ptr<geos::geom::Geometry> outputGeometry;
    GEOS_TRY(
        outputGeometry = leftGeosGeometry->Union(&*rightGeosGeometry);
        , "Failed to compute geometry union");

    common::geospatial::GeometrySerializer::serialize(*outputGeometry, result);
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
        common::geospatial::GeometryDeserializer::deserialize(input);

    GEOS_TRY(
        result = geosGeometry->isValid();, "Failed to check geometry isValid");

    return Status::OK();
  }
};

template <typename T>
struct StIsSimpleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<bool>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(input);

    GEOS_TRY(
        result = geosGeometry->isSimple();
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
        common::geospatial::GeometryDeserializer::deserialize(input);

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
        common::geospatial::GeometryDeserializer::deserialize(input);

    GEOS_TRY(
        result = geosGeometry->getArea();, "Failed to compute geometry area");

    return Status::OK();
  }
};

template <typename T>
struct StCentroidFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(input);

    if (geosGeometry->isEmpty()) {
      return false;
    }

    auto validate = facebook::velox::functions::geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POINT,
         geos::geom::GeometryTypeId::GEOS_MULTIPOINT,
         geos::geom::GeometryTypeId::GEOS_LINESTRING,
         geos::geom::GeometryTypeId::GEOS_MULTILINESTRING,
         geos::geom::GeometryTypeId::GEOS_POLYGON,
         geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON},
        "ST_Centroid");
    VELOX_USER_CHECK(validate.ok(), validate.message());

    geos::geom::GeometryTypeId type = geosGeometry->getGeometryTypeId();
    if (type == geos::geom::GeometryTypeId::GEOS_POINT) {
      result = input;
      return true;
    }

    if (geosGeometry->getNumPoints() == 0) {
      GEOS_RETHROW(
          {
            geos::geom::GeometryFactory::Ptr factory =
                geos::geom::GeometryFactory::create();
            std::unique_ptr<geos::geom::Point> point = factory->createPoint();
            common::geospatial::GeometrySerializer::serialize(*point, result);
            factory->destroyGeometry(point.release());
          },
          "Failed to create point geometry");
      return true;
    }

    common::geospatial::GeometrySerializer::serialize(
        *(geosGeometry->getCentroid()), result);
    return true;
  }
};

template <typename T>
struct StXFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);
    if (geosGeometry->getGeometryTypeId() !=
        geos::geom::GeometryTypeId::GEOS_POINT) {
      VELOX_USER_FAIL(
          fmt::format(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);
    if (geosGeometry->getGeometryTypeId() !=
        geos::geom::GeometryTypeId::GEOS_POINT) {
      VELOX_USER_FAIL(
          fmt::format(
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
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
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
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
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
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
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
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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

    common::geospatial::GeometrySerializer::serialize(*outputGeometry, result);
    return Status::OK();
  }
};

template <typename T>
struct StGeometryTypeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(input);

    if (geosGeometry->getGeometryTypeId() ==
        geos::geom::GeometryTypeId::GEOS_GEOMETRYCOLLECTION) {
      result = "ST_GeomCollection";
    } else {
      result = "ST_" + geosGeometry->getGeometryType();
    }

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
        common::geospatial::GeometryDeserializer::deserialize(geometry1);
    std::unique_ptr<geos::geom::Geometry> geosGeometry2 =
        common::geospatial::GeometryDeserializer::deserialize(geometry2);

    if (geosGeometry1->getSRID() != geosGeometry2->getSRID()) {
      VELOX_USER_FAIL(
          fmt::format(
              "Input geometries must have the same spatial reference, found {} and {}",
              geosGeometry1->getSRID(),
              geosGeometry2->getSRID()));
    }

    if (geosGeometry1->isEmpty() || geosGeometry2->isEmpty()) {
      return false;
    }

    GEOS_RETHROW(
        result = geosGeometry1->distance(geosGeometry2.get());
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
              common::geospatial::GeometryDeserializer::deserializeEnvelope(
                  geometry);
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
      common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
    common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
    common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      return false;
    }

    if (!geospatial::isMultiType(*geosGeometry)) {
      if (index == 1) {
        common::geospatial::GeometrySerializer::serialize(
            *geosGeometry, result);
        return true;
      }
      return false;
    }

    if (geos::geom::GeometryCollection* geomCollection =
            dynamic_cast<geos::geom::GeometryCollection*>(geosGeometry.get())) {
      if (index < 1 || index > geomCollection->getNumGeometries()) {
        return false;
      }
      common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
    common::geospatial::GeometrySerializer::serialize(
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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
      out_type<int64_t>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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
    result = static_cast<int64_t>(numInteriorRings);
    return true;
  }
};

template <typename T>
struct StConvexHullFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty() ||
        geosGeometry->getGeometryTypeId() ==
            geos::geom::GeometryTypeId::GEOS_POINT) {
      result = geometry;
    } else {
      common::geospatial::GeometrySerializer::serialize(
          *(geosGeometry->convexHull()), result);
    }
    return Status::OK();
  }
};
class StCoordDimFunction : public facebook::velox::exec::VectorFunction {
 public:
  void apply(
      const facebook::velox::SelectivityVector& rows,
      std::vector<facebook::velox::VectorPtr>& /*args*/,
      const std::shared_ptr<const facebook::velox::Type>& outputType,
      facebook::velox::exec::EvalCtx& context,
      facebook::velox::VectorPtr& result) const override {
    // Create a constant vector of value 2, with size equal to the number of
    // rows
    result = facebook::velox::BaseVector::createConstant(
        outputType, static_cast<int8_t>(2), rows.size(), context.pool());
  }
  static std::vector<std::shared_ptr<facebook::velox::exec::FunctionSignature>>
  signatures() {
    return {facebook::velox::exec::FunctionSignatureBuilder()
                .returnType("tinyint")
                .argumentType("geometry")
                .build()};
  }
};

template <typename T>
struct StDimensionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<int8_t>& result, const arg_type<Geometry>& geometry) {
    result = common::geospatial::GeometryDeserializer::deserialize(geometry)
                 ->getDimension();

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
        common::geospatial::GeometryDeserializer::deserialize(geometry);

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

    common::geospatial::GeometrySerializer::serialize(
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
    std::unique_ptr<geos::geom::Envelope> env =
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);

    common::geospatial::GeometrySerializer::serializeEnvelope(*env, result);

    return Status::OK();
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StBufferFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<double>& distance) {
    if (distance < 0) {
      VELOX_USER_FAIL(
          fmt::format(
              "Provided distance must not be negative. Provided distance: {}",
              distance));
    }
    if (distance == 0) {
      result = geometry;
      return true;
    }

    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);
    if (geosGeometry->isEmpty()) {
      return false;
    }
    common::geospatial::GeometrySerializer::serialize(
        *(geosGeometry->buffer(distance)), result);
    return true;
  }
};

template <typename T>
struct StPointsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Geometry>>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      return false;
    }

    auto numPoints = geosGeometry->getNumPoints();
    result.reserve(static_cast<vector_size_t>(numPoints));

    writePoints(geosGeometry.get(), result);
    return true;
  }

 private:
  void writePoints(
      const geos::geom::Geometry* geosGeometry,
      out_type<Array<Geometry>>& result) {
    auto geometryType = geosGeometry->getGeometryTypeId();

    if (geometryType == geos::geom::GeometryTypeId::GEOS_POINT) {
      common::geospatial::GeometrySerializer::serialize(
          *geosGeometry, result.add_item());
    } else if (
        geometryType == geos::geom::GeometryTypeId::GEOS_GEOMETRYCOLLECTION) {
      auto geometryCollection =
          static_cast<const geos::geom::GeometryCollection*>(geosGeometry);
      for (int i = 0; i < geometryCollection->getNumGeometries(); i++) {
        auto entry = geometryCollection->getGeometryN(i);
        writePoints(entry, result);
      }
    } else {
      auto geometryFactory = geosGeometry->getFactory();
      auto vertices = geosGeometry->getCoordinates();
      auto vertexCount = vertices->getSize();
      for (auto i = 0; i < vertexCount; i++) {
        const geos::geom::Coordinate& coordinate = vertices->getAt(i);
        auto point = std::unique_ptr<geos::geom::Point>(
            geometryFactory->createPoint(coordinate));
        common::geospatial::GeometrySerializer::serialize(
            *point, result.add_item());
      }
    }
  }
};

template <typename T>
struct StEnvelopeAsPtsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  StEnvelopeAsPtsFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Geometry>>& result,
      const arg_type<Geometry>& geometry) {
    auto envelope =
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);

    if (envelope->isNull()) {
      return false;
    }

    auto lowerLeft = std::unique_ptr<geos::geom::Point>(factory_->createPoint(
        geos::geom::Coordinate(envelope->getMinX(), envelope->getMinY())));

    auto upperRight = std::unique_ptr<geos::geom::Point>(factory_->createPoint(
        geos::geom::Coordinate(envelope->getMaxX(), envelope->getMaxY())));

    common::geospatial::GeometrySerializer::serialize(
        *lowerLeft, result.add_item());
    common::geospatial::GeometrySerializer::serialize(
        *upperRight, result.add_item());

    return true;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StNumPointsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<int64_t>& result,
      const arg_type<Geometry>& geometry) {
    auto geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    result = pointCount(*geosGeometry);
  }

 private:
  int32_t pointCount(const geos::geom::Geometry& geometry) {
    if (geometry.isEmpty()) {
      return 0;
    }
    if (geometry.getGeometryTypeId() ==
        geos::geom::GeometryTypeId::GEOS_POINT) {
      return 1;
    }
    if (geometry.getGeometryTypeId() ==
        geos::geom::GeometryTypeId::GEOS_LINESTRING) {
      return static_cast<int32_t>(geometry.getNumPoints());
    }
    if (geospatial::isMultiType(geometry)) {
      auto numGeometries = geometry.getNumGeometries();
      auto multiTypePointCount = 0;
      for (int i = 0; i < numGeometries; i++) {
        auto entry = geometry.getGeometryN(i);
        multiTypePointCount += pointCount(*entry);
      }
      return multiTypePointCount;
    }
    if (geometry.getGeometryTypeId() ==
        geos::geom::GeometryTypeId::GEOS_POLYGON) {
      auto polygon = static_cast<const geos::geom::Polygon*>(&geometry);
      // Subtract 1 to remove closing point, since we don't count the closing
      // point in Java.
      auto polygonPointCount = polygon->getExteriorRing()->getNumPoints() - 1;
      for (int i = 0; i < polygon->getNumInteriorRing(); i++) {
        auto interiorRing = polygon->getInteriorRingN(i);
        polygonPointCount += interiorRing->getNumPoints() - 1;
      }
      return static_cast<int32_t>(polygonPointCount);
    }
    VELOX_FAIL(
        fmt::format(
            "Unexpected failure in ST_NumPoints: geometry type {}",
            geometry.getGeometryType()));
  }
};

template <typename T>
struct GeometryNearestPointsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  GeometryNearestPointsFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Geometry>>& result,
      const arg_type<Geometry>& leftGeometry,
      const arg_type<Geometry>& rightGeometry) {
    auto left =
        common::geospatial::GeometryDeserializer::deserialize(leftGeometry);
    auto right =
        common::geospatial::GeometryDeserializer::deserialize(rightGeometry);

    if (left->isEmpty() || right->isEmpty()) {
      return false;
    }

    GEOS_RETHROW(
        {
          std::unique_ptr<geos::geom::CoordinateSequence> nearestCoordinates =
              geos::operation::distance::DistanceOp::nearestPoints(
                  left.get(), right.get());

          auto pointA =
              std::unique_ptr<geos::geom::Point>(factory_->createPoint(
                  geos::geom::Coordinate(nearestCoordinates->getAt(0))));
          auto pointB =
              std::unique_ptr<geos::geom::Point>(factory_->createPoint(
                  geos::geom::Coordinate(nearestCoordinates->getAt(1))));

          result.reserve(2);
          common::geospatial::GeometrySerializer::serialize(
              *pointA, result.add_item());
          common::geospatial::GeometrySerializer::serialize(
              *pointB, result.add_item());
        },
        "Failed to compute nearest points between geometries");

    return true;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct LineLocatePointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<Geometry>& inputLine,
      const arg_type<Geometry>& inputPoint) {
    auto line =
        common::geospatial::GeometryDeserializer::deserialize(inputLine);
    auto point =
        common::geospatial::GeometryDeserializer::deserialize(inputPoint);

    if (line->isEmpty() || point->isEmpty()) {
      return false;
    }

    auto lineType = line->getGeometryTypeId();
    if (lineType != geos::geom::GeometryTypeId::GEOS_LINESTRING &&
        lineType != geos::geom::GeometryTypeId::GEOS_MULTILINESTRING) {
      VELOX_USER_FAIL(
          fmt::format(
              "First argument to line_locate_point must be a LineString or a MultiLineString. Got: {}",
              line->getGeometryType()));
    }

    auto pointType = point->getGeometryTypeId();
    if (pointType != geos::geom::GeometryTypeId::GEOS_POINT) {
      VELOX_USER_FAIL(
          fmt::format(
              "Second argument to line_locate_point must be a Point. Got: {}",
              point->getGeometryType()));
    }

    result = geos::linearref::LengthIndexedLine(line.get())
                 .indexOf(*(point->getCoordinate())) /
        line->getLength();

    return true;
  }
};

template <typename T>
struct LineInterpolatePointFunction {
  LineInterpolatePointFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& inputLine,
      const arg_type<double>& fraction) {
    if (!(0.0 <= fraction && fraction <= 1.0)) {
      return Status::UserError(
          fmt::format(
              "line_interpolate_point: Fraction must be between 0 and 1, but is {}",
              fraction));
    }
    auto line =
        common::geospatial::GeometryDeserializer::deserialize(inputLine);
    Status validate = Status::OK();
    validate = geospatial::validateType(
        *line,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "line_interpolate_point");

    if (!validate.ok()) {
      return validate;
    }

    if (line->isEmpty()) {
      common::geospatial::GeometrySerializer::serialize(
          *(factory_->createPoint()), result);
    }

    geos::geom::Coordinate coordinate =
        geos::linearref::LengthIndexedLine(line.get())
            .extractPoint(fraction * line->getLength());

    auto resultPoint =
        std::unique_ptr<geos::geom::Point>(factory_->createPoint(coordinate));
    common::geospatial::GeometrySerializer::serialize(*resultPoint, result);

    return validate;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StInteriorRingsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Geometry>>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_POLYGON},
        "ST_InteriorRings");

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

    auto numInteriorRings = polygon->getNumInteriorRing();
    result.reserve(static_cast<int32_t>(numInteriorRings));

    for (int i = 0; i < numInteriorRings; i++) {
      common::geospatial::GeometrySerializer::serialize(
          *(polygon->getInteriorRingN(i)), result.add_item());
    }

    return true;
  }
};

template <typename T>
struct StGeometriesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Array<Geometry>>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    if (geosGeometry->isEmpty()) {
      return false;
    }

    if (!geospatial::isMultiType(*geosGeometry)) {
      result.reserve(1);
      common::geospatial::GeometrySerializer::serialize(
          *(geosGeometry), result.add_item());
      return true;
    }

    geos::geom::GeometryCollection* geomCollection =
        dynamic_cast<geos::geom::GeometryCollection*>(geosGeometry.get());

    VELOX_CHECK_NOT_NULL(
        geomCollection,
        "Failure in ST_Geometries: geometry should be multi type but cast to GeometryCollection failed");

    int32_t numGeometries =
        static_cast<int32_t>(geomCollection->getNumGeometries());
    result.reserve(numGeometries);

    for (int i = 0; i < numGeometries; i++) {
      common::geospatial::GeometrySerializer::serialize(
          *(geomCollection->getGeometryN(i)), result.add_item());
    }

    return true;
  }
};

template <typename T>
struct FlattenGeometryCollectionsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Array<Geometry>>& result, const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);

    geospatial::GeometryCollectionIterator it(geosGeometry.get());
    while (it.hasNext()) {
      common::geospatial::GeometrySerializer::serialize(
          *(it.next()), result.add_item());
    }

    return Status::OK();
  }
};

template <typename T>
struct ExpandEnvelopeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  ExpandEnvelopeFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Geometry>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<double>& distance) {
    if (std::isnan(distance)) {
      return Status::UserError("Distance must be a non-NaN number");
    }
    if (distance < 0) {
      return Status::UserError("Distance must be a non-negative number");
    }
    if (distance == std::numeric_limits<double>::infinity()) {
      common::geospatial::GeometrySerializer::serialize(
          *factory_->createPolygon(), result);
      return Status::OK();
    }

    const std::unique_ptr<geos::geom::Envelope> envelope =
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (envelope->isNull()) {
      common::geospatial::GeometrySerializer::serializeEnvelope(
          *envelope, result);
      return Status::OK();
    }

    envelope->expandBy(distance);
    common::geospatial::GeometrySerializer::serializeEnvelope(
        *envelope, result);

    return Status::OK();
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct BingTilePolygonFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Geometry>& result,
      const arg_type<BingTile>& tile) {
    auto x = BingTileType::bingTileX(tile);
    auto y = BingTileType::bingTileY(tile);
    auto zoom = BingTileType::bingTileZoom(tile);

    double minX = BingTileType::tileXToLongitude(x, zoom);
    double maxX = BingTileType::tileXToLongitude(x + 1, zoom);
    double minY = BingTileType::tileYToLatitude(y, zoom);
    double maxY = BingTileType::tileYToLatitude(y + 1, zoom);

    common::geospatial::GeometrySerializer::serializeEnvelope(
        minX, minY, maxX, maxY, result);
  }
};

template <typename T>
struct GeometryAsGeoJsonFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Varchar>& result,
      const arg_type<Geometry>& geometry) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry =
        common::geospatial::GeometryDeserializer::deserialize(geometry);
    if (geospatial::isAtomicType(*geosGeometry) && geosGeometry->isEmpty()) {
      return false;
    }

    auto writer = geos::io::GeoJSONWriter();
    result = writer.write(geosGeometry.get());
    return true;
  }
};

template <typename T>
struct GeometryFromGeoJsonFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Geometry>& result,
      const arg_type<Varchar>& geometry) {
    auto reader = geos::io::GeoJSONReader();
    auto geosGeometry = reader.read(std::string(geometry));
    common::geospatial::GeometrySerializer::serialize(*geosGeometry, result);
  }
};

template <typename T>
struct GreatCircleDistanceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<double>& result,
      const arg_type<double>& lat1,
      const arg_type<double>& long1,
      const arg_type<double>& lat2,
      const arg_type<double>& long2) {
    Status status = geospatial::validateLatitudeLongitude(lat1, long1);
    if (FOLLY_UNLIKELY(!status.ok())) {
      return status;
    }

    status = geospatial::validateLatitudeLongitude(lat2, long2);
    if (FOLLY_UNLIKELY(!status.ok())) {
      return status;
    }

    result = BingTileType::greatCircleDistance(lat1, long1, lat2, long2);
    return status;
  }
};

template <typename T>
struct GeometryToBingTilesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  uint8_t maxZoomShift = 5;

  FOLLY_ALWAYS_INLINE void initialize(
      const std::vector<TypePtr>& /* inputTypes */,
      const core::QueryConfig& config,
      const arg_type<Geometry>* /* geometry */,
      const arg_type<int32_t>* /* zoom */) {
    maxZoomShift =
        std::max<uint8_t>(config.debugBingTileChildrenMaxZoomShift(), 1);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<BingTile>>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& zoom) {
    if (zoom < 0 || zoom > 23) {
      return Status::UserError("Zoom level must be between 0 and 23");
    }
    auto envelope =
        common::geospatial::GeometryDeserializer::deserializeEnvelope(geometry);
    if (envelope->isNull()) {
      return Status::OK();
    }

    std::vector<int64_t> covering;
    auto geom = common::geospatial::GeometryDeserializer::deserialize(geometry);

    if (geom->getGeometryTypeId() == geos::geom::GeometryTypeId::GEOS_POINT) {
      covering = geospatial::getMinimalTilesCoveringGeometry(*envelope, zoom);
    } else {
      if (geospatial::isPointOrRectangle(*geom)) {
        covering = geospatial::getMinimalTilesCoveringGeometry(*envelope, zoom);
      } else {
        covering = geospatial::getMinimalTilesCoveringGeometry(
            *geom, zoom, maxZoomShift);
      }
    }

    result.add_items(covering);
    return Status::OK();
  }
};

template <typename T>
struct GeometryToDissolvedBingTilesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<BingTile>>& result,
      const arg_type<Geometry>& geometry,
      const arg_type<int32_t>& zoom) {
    if (zoom < 0 || zoom > 23) {
      return Status::UserError("Zoom level must be between 0 and 23");
    }
    auto geom = common::geospatial::GeometryDeserializer::deserialize(geometry);
    if (geom->isEmpty()) {
      return Status::OK();
    }

    std::vector<int64_t> covering =
        geospatial::getDissolvedTilesCoveringGeometry(*geom, zoom);

    result.add_items(covering);
    return Status::OK();
  }
};

template <typename T>
struct GeometryUnionFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  GeometryUnionFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Array<Geometry>>& geometries) {
    std::vector<std::unique_ptr<geos::geom::Geometry>> geometryVector;
    if (geometries.size() == 0) {
      return false;
    }
    // There seems to be a bug in the current version of geos where union with
    // all empty geometries is returning incorrect results or even crashing.
    // The isCompletelyEmpty check is a hack to avoid this case.
    bool isCompletelyEmpty = true;

    for (int i = 0; i < geometries.size(); i++) {
      // Ignore null inputs
      if (!geometries[i].has_value()) {
        continue;
      }
      std::unique_ptr<geos::geom::Geometry> geosGeometry =
          common::geospatial::GeometryDeserializer::deserialize(*geometries[i]);
      if (!geosGeometry->isEmpty()) {
        isCompletelyEmpty = false;
      }
      geometryVector.push_back(std::move(geosGeometry));
    }

    if (FOLLY_UNLIKELY(isCompletelyEmpty)) {
      auto emptyCollection = factory_->createPolygon();
      common::geospatial::GeometrySerializer::serialize(
          *emptyCollection, result);
      return true;
    }

    std::unique_ptr<geos::geom::Geometry> collection(
        factory_->createGeometryCollection(std::move(geometryVector)));

    // If geometry is not completely empty, we can proceed with union.
    std::unique_ptr<geos::geom::Geometry> geomUnion = collection->Union();
    common::geospatial::GeometrySerializer::serialize(*geomUnion, result);
    return true;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StLineFromTextFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Geometry>& result, const arg_type<Varchar>& wkt) {
    std::unique_ptr<geos::geom::Geometry> geosGeometry;
    GEOS_TRY(
        {
          geos::io::WKTReader reader;
          geosGeometry = reader.read(std::string(wkt));
        },
        "Failed to parse WKT");
    auto validate = geospatial::validateType(
        *geosGeometry,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING},
        "ST_LineFromText");

    if (validate.ok()) {
      common::geospatial::GeometrySerializer::serialize(*geosGeometry, result);
    }

    return validate;
  }
};

template <typename T>
struct StLineStringFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  StLineStringFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE void call(
      out_type<Geometry>& result,
      const arg_type<Array<Geometry>>& input) {
    std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
        common::geospatial::GeometryDeserializer::deserializePointsToCoordinate<
            Geometry>(input, "ST_LineString", true);

    std::unique_ptr<geos::geom::LineString> lineString;
    if (input.size() < 2) {
      lineString = factory_->createLineString();
    } else {
      lineString = factory_->createLineString(std::move(coords));
    }
    common::geospatial::GeometrySerializer::serialize(*lineString, result);
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StMultiPointFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  StMultiPointFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<Geometry>& result,
      const arg_type<Array<Geometry>>& input) {
    std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
        common::geospatial::GeometryDeserializer::deserializePointsToCoordinate<
            Geometry>(input, "ST_MultiPoint", false);

    if (coords->size() == 0) {
      return false;
    }

    auto multiPoint = std::unique_ptr<geos::geom::MultiPoint>(
        factory_->createMultiPoint(*coords));
    common::geospatial::GeometrySerializer::serialize(*multiPoint, result);
    return true;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct ToGeometryFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void call(
      out_type<Geometry>& result,
      const arg_type<SphericalGeography>& input) {
    // Every SphericalGeography object is a valid geometry object
    result = input;
  }
};

template <typename T>
struct ToSphericalGeographyFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<SphericalGeography>& result, const arg_type<Geometry>& input) {
    std::unique_ptr<geos::geom::Envelope> env =
        common::geospatial::GeometryDeserializer::deserializeEnvelope(input);
    Status status = Status::OK();
    if (!env->isNull()) {
      status =
          geospatial::validateLatitudeLongitude(env->getMinY(), env->getMinX());
      if (FOLLY_UNLIKELY(!status.ok())) {
        return status;
      }
      status =
          geospatial::validateLatitudeLongitude(env->getMaxY(), env->getMaxX());
      if (FOLLY_UNLIKELY(!status.ok())) {
        return status;
      }
    }

    result = input;
    return status;
  }
};

template <typename T>
struct StSphericalCentroidFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  StSphericalCentroidFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE bool call(
      out_type<SphericalGeography>& result,
      const arg_type<SphericalGeography>& input) {
    std::unique_ptr<geos::geom::Geometry> geometry =
        common::geospatial::GeometryDeserializer::deserialize(input);

    if (geometry->isEmpty()) {
      return false;
    }

    // Validate type is POINT or MULTI_POINT
    auto validate = geospatial::validateType(
        *geometry,
        {geos::geom::GeometryTypeId::GEOS_POINT,
         geos::geom::GeometryTypeId::GEOS_MULTIPOINT},
        "ST_Centroid[SphericalGeography]");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    auto geometryType = geometry->getGeometryTypeId();
    // Handle single point case
    if (geometryType == geos::geom::GeometryTypeId::GEOS_POINT) {
      common::geospatial::GeometrySerializer::serialize(*geometry, result);
      return true;
    }

    const geos::geom::MultiPoint& multiPoint =
        static_cast<const geos::geom::MultiPoint&>(*geometry);

    // Compute the spherical centroid
    auto centroidCoords = geospatial::computeSphericalCentroid(multiPoint);

    // Create a Point geometry from the spherical coordinates
    auto centroidPoint =
        std::unique_ptr<geos::geom::Point>(factory_->createPoint(
            geos::geom::Coordinate(
                centroidCoords.first, centroidCoords.second)));

    common::geospatial::GeometrySerializer::serialize(*centroidPoint, result);
    return true;
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

template <typename T>
struct StSphericalDistanceFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<SphericalGeography>& left,
      const arg_type<SphericalGeography>& right) {
    std::unique_ptr<geos::geom::Geometry> leftGeom =
        common::geospatial::GeometryDeserializer::deserialize(left);
    std::unique_ptr<geos::geom::Geometry> rightGeom =
        common::geospatial::GeometryDeserializer::deserialize(right);

    auto validateLeft = geospatial::validateType(
        *leftGeom,
        {geos::geom::GeometryTypeId::GEOS_POINT},
        "ST_Distance[SphericalGeography]");
    if (!validateLeft.ok()) {
      VELOX_USER_FAIL(validateLeft.message());
    };

    auto validateRight = geospatial::validateType(
        *rightGeom,
        {geos::geom::GeometryTypeId::GEOS_POINT},
        "ST_Distance[SphericalGeography]");
    if (!validateRight.ok()) {
      VELOX_USER_FAIL(validateRight.message());
    }

    if (leftGeom->isEmpty() || rightGeom->isEmpty()) {
      return false;
    }

    geos::geom::Point* leftPoint =
        static_cast<geos::geom::Point*>(leftGeom.get());
    geos::geom::Point* rightPoint =
        static_cast<geos::geom::Point*>(rightGeom.get());

    result = BingTileType::greatCircleDistance(
                 leftPoint->getY(),
                 leftPoint->getX(),
                 rightPoint->getY(),
                 rightPoint->getX()) *
        1000;

    return true;
  }
};

template <typename T>
struct StSphericalLengthFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<SphericalGeography>& input) {
    std::unique_ptr<geos::geom::Geometry> geom =
        common::geospatial::GeometryDeserializer::deserialize(input);
    if (geom->isEmpty()) {
      return false;
    }

    auto validate = geospatial::validateType(
        *geom,
        {geos::geom::GeometryTypeId::GEOS_LINESTRING,
         geos::geom::GeometryTypeId::GEOS_MULTILINESTRING},
        "ST_Length[SphericalGeography]");

    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    double sum = 0.0;
    if (geom->getGeometryTypeId() ==
        geos::geom::GeometryTypeId::GEOS_LINESTRING) {
      const geos::geom::LineString* lineString =
          static_cast<const geos::geom::LineString*>(geom.get());
      sum += geospatial::getSphericalLength(*lineString);

    } else { // Multipoint
      geos::geom::MultiLineString* multiLineString =
          static_cast<geos::geom::MultiLineString*>(geom.get());
      auto numLineStrings = multiLineString->getNumGeometries();
      for (int i = 0; i < numLineStrings; i++) {
        const geos::geom::LineString* lineString =
            multiLineString->getGeometryN(i);
        sum += geospatial::getSphericalLength(*lineString);
      }
    }

    result = sum * 1000;
    return true;
  }
};

template <typename T>
struct StSphericalAreaFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      out_type<double>& result,
      const arg_type<SphericalGeography>& input) {
    std::unique_ptr<geos::geom::Geometry> geom =
        common::geospatial::GeometryDeserializer::deserialize(input);
    if (geom->isEmpty()) {
      return false;
    }

    auto validate = geospatial::validateType(
        *geom,
        {geos::geom::GeometryTypeId::GEOS_POLYGON,
         geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON},
        "ST_Area[SphericalGeography]");
    if (!validate.ok()) {
      VELOX_USER_FAIL(validate.message());
    }

    double sphericalExcess = 0.0;
    if (geom->getGeometryTypeId() == geos::geom::GEOS_POLYGON) {
      geos::geom::Polygon* polygon =
          static_cast<geos::geom::Polygon*>(geom.get());
      // Single polygon: calculate excess for that polygon only
      sphericalExcess = geospatial::computeSphericalExcess(*polygon);
    } else if (geom->getGeometryTypeId() == geos::geom::GEOS_MULTIPOLYGON) {
      geos::geom::MultiPolygon* multiPolygon =
          static_cast<geos::geom::MultiPolygon*>(geom.get());
      // MultiPolygon: calculate sum excess of all Polygons
      auto polygonCount = multiPolygon->getNumGeometries();
      for (int i = 0; i < polygonCount; i++) {
        sphericalExcess +=
            geospatial::computeSphericalExcess(*multiPolygon->getGeometryN(i));
      }
    } else {
      VELOX_FAIL(
          fmt::format(
              "Unexpected failure in ST_Area initial type validation. Type: {}",
              geom->getGeometryType()));
    }

    // abs is required here because for Polygons with a 2D area of 0
    // isExteriorRing returns false for the exterior ring
    result = std::abs(
        (sphericalExcess * (BingTileType::kEarthRadiusKm * 1000.0) *
         (BingTileType::kEarthRadiusKm * 1000.0)));

    return true;
  }
};

} // namespace facebook::velox::functions
