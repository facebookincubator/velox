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

#include <fmt/ranges.h>
#include <geos/geom/Geometry.h>
#include <geos/io/WKTReader.h>

#include <geos/util/AssertionFailedException.h>
#include <geos/util/UnsupportedOperationException.h>
#include <optional>

#include "velox/common/base/Status.h"

namespace facebook::velox::functions::geospatial {

/// Utility macro used to wrap GEOS library calls in a try-catch block,
/// returning a velox::Status with error message if an exception is caught.
#define GEOS_TRY(func, user_error_message)                       \
  try {                                                          \
    func                                                         \
  } catch (const geos::util::UnsupportedOperationException& e) { \
    return Status::UnknownError(                                 \
        fmt::format("Internal geometry error: {}", e.what()));   \
  } catch (const geos::util::AssertionFailedException& e) {      \
    return Status::UnknownError(                                 \
        fmt::format("Internal geometry error: {}", e.what()));   \
  } catch (const geos::util::GEOSException& e) {                 \
    return Status::UserError(                                    \
        fmt::format("{}: {}", user_error_message, e.what()));    \
  }

/// Utility macro used to wrap GEOS library calls in a try-catch block,
/// throwing a velox::Status with error message if an exception is caught.
#define GEOS_RETHROW(func, user_error_message)                             \
  try {                                                                    \
    func                                                                   \
  } catch (const geos::util::UnsupportedOperationException& e) {           \
    VELOX_USER_FAIL(fmt::format("Internal geometry error: {}", e.what())); \
  } catch (const geos::util::AssertionFailedException& e) {                \
    VELOX_FAIL(fmt::format("Internal geometry error: {}", e.what()));      \
  } catch (const geos::util::GEOSException& e) {                           \
    VELOX_FAIL(fmt::format("{}: {}", user_error_message, e.what()));       \
  }

class GeometryCollectionIterator {
 public:
  explicit GeometryCollectionIterator(const geos::geom::Geometry* geometry);
  bool hasNext();
  const geos::geom::Geometry* next();

 private:
  std::deque<const geos::geom::Geometry*> geometriesDeque;
};

geos::geom::GeometryFactory* getGeometryFactory();

FOLLY_ALWAYS_INLINE const
    std::unordered_map<geos::geom::GeometryTypeId, std::string>&
    getGeosTypeToStringIdentifier() {
  static const geos::geom::GeometryFactory::Ptr factory =
      geos::geom::GeometryFactory::create();

  static const std::unordered_map<geos::geom::GeometryTypeId, std::string>
      geosTypeToStringIdentifier{
          {geos::geom::GeometryTypeId::GEOS_POINT,
           factory->createPoint()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_LINESTRING,
           factory->createLineString()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_LINEARRING,
           factory->createLinearRing()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_POLYGON,
           factory->createPolygon()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_MULTIPOINT,
           factory->createMultiPoint()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_MULTILINESTRING,
           factory->createMultiLineString()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON,
           factory->createMultiPolygon()->getGeometryType()},
          {geos::geom::GeometryTypeId::GEOS_GEOMETRYCOLLECTION,
           factory->createGeometryCollection()->getGeometryType()}};
  return geosTypeToStringIdentifier;
};

FOLLY_ALWAYS_INLINE std::vector<std::string> getGeosTypeNames(
    const std::vector<geos::geom::GeometryTypeId>& geometryTypeIds) {
  std::vector<std::string> geometryTypeNames;
  geometryTypeNames.reserve(geometryTypeIds.size());
  for (auto geometryTypeId : geometryTypeIds) {
    geometryTypeNames.push_back(
        getGeosTypeToStringIdentifier().at(geometryTypeId));
  }
  return geometryTypeNames;
}

FOLLY_ALWAYS_INLINE Status validateType(
    const geos::geom::Geometry& geometry,
    const std::vector<geos::geom::GeometryTypeId>& validTypes,
    std::string callerFunctionName) {
  geos::geom::GeometryTypeId type = geometry.getGeometryTypeId();
  if (!std::count(validTypes.begin(), validTypes.end(), type)) {
    return Status::UserError(
        fmt::format(
            "{} only applies to {}. Input type is: {}",
            callerFunctionName,
            fmt::join(getGeosTypeNames(validTypes), " or "),
            getGeosTypeToStringIdentifier().at(type)));
  }
  return Status::OK();
}

FOLLY_ALWAYS_INLINE bool isMultiType(const geos::geom::Geometry& geometry) {
  geos::geom::GeometryTypeId type = geometry.getGeometryTypeId();

  static const std::vector<geos::geom::GeometryTypeId> multiTypes{
      geos::geom::GeometryTypeId::GEOS_MULTILINESTRING,
      geos::geom::GeometryTypeId::GEOS_MULTIPOINT,
      geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON,
      geos::geom::GeometryTypeId::GEOS_GEOMETRYCOLLECTION};

  return std::count(multiTypes.begin(), multiTypes.end(), type);
}

FOLLY_ALWAYS_INLINE bool isAtomicType(const geos::geom::Geometry& geometry) {
  geos::geom::GeometryTypeId type = geometry.getGeometryTypeId();

  static const std::vector<geos::geom::GeometryTypeId> atomicTypes{
      geos::geom::GeometryTypeId::GEOS_LINESTRING,
      geos::geom::GeometryTypeId::GEOS_POLYGON,
      geos::geom::GeometryTypeId::GEOS_POINT};

  return std::count(atomicTypes.begin(), atomicTypes.end(), type);
}

std::optional<std::string> geometryInvalidReason(
    const geos::geom::Geometry* geometry);

enum ClockwiseResult { CW, CCW, ZERO_AREA };

/// Determines if a ring of coordinates (from `start` to `end`) is oriented
/// clockwise. A return value of 1 indicates clockwise orientation, 0 indicates
/// counterclockwise, and -1 represents a polygon which has no orientation due
/// to having an area of 0.
FOLLY_ALWAYS_INLINE ClockwiseResult isClockwise(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end) {
  double sum = 0.0;
  for (size_t i = start; i < end - 1; i++) {
    const auto& p1 = coordinates->getAt(i);
    const auto& p2 = coordinates->getAt(i + 1);
    sum += (p2.x - p1.x) * (p2.y + p1.y);
  }
  if (FOLLY_UNLIKELY(sum == 0.0)) {
    return ClockwiseResult::ZERO_AREA;
  }
  return sum > 0.0 ? ClockwiseResult::CW : ClockwiseResult::CCW;
}

/// Reverses the order of coordinates in the sequence between `start` and `end`
FOLLY_ALWAYS_INLINE void reverse(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end) {
  for (size_t i = 0; i < (end - start) / 2; ++i) {
    auto temp = coordinates->getAt(start + i);
    coordinates->setAt(coordinates->getAt(end - 1 - i), start + i);
    coordinates->setAt(temp, end - 1 - i);
  }
}

/// Ensures that a polygon ring has the canonical orientation:
/// - Exterior rings (shells) must be clockwise.
/// - Interior rings (holes) must be counter-clockwise.
/// A return value of true indicates a zero-area ring was encountered
FOLLY_ALWAYS_INLINE bool canonicalizePolygonCoordinates(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end,
    bool isShell) {
  ClockwiseResult isClockwiseFlag = isClockwise(coordinates, start, end);
  if (isClockwiseFlag == ClockwiseResult::ZERO_AREA) {
    return true;
  }

  if ((isShell && isClockwiseFlag == ClockwiseResult::CCW) ||
      (!isShell && isClockwiseFlag == ClockwiseResult::CW)) {
    reverse(coordinates, start, end);
  }
  return false;
}

/// Applies `canonicalizePolygonCoordinates` to all rings in a polygon.
/// A return value of true indicates at least one zero-area ring was
/// encountered.
FOLLY_ALWAYS_INLINE bool canonicalizePolygonCoordinates(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    const std::vector<size_t>& partIndexes,
    const std::vector<bool>& shellPart) {
  bool zeroAreaRingEncountered = false;
  for (size_t part = 0; part < partIndexes.size() - 1; part++) {
    zeroAreaRingEncountered |= canonicalizePolygonCoordinates(
        coordinates, partIndexes[part], partIndexes[part + 1], shellPart[part]);
  }
  if (!partIndexes.empty()) {
    zeroAreaRingEncountered |= canonicalizePolygonCoordinates(
        coordinates, partIndexes.back(), coordinates->size(), shellPart.back());
  }
  return zeroAreaRingEncountered;
}

Status validateLatitudeLongitude(double latitude, double longitude);

std::vector<const geos::geom::Geometry*> flattenCollection(
    const geos::geom::Geometry* geometry);

std::vector<int64_t> getMinimalTilesCoveringGeometry(
    const geos::geom::Envelope& envelope,
    int32_t zoom);

std::vector<int64_t> getMinimalTilesCoveringGeometry(
    const geos::geom::Geometry& geometry,
    int32_t zoom,
    uint8_t maxZoomShift);

std::vector<int64_t> getDissolvedTilesCoveringGeometry(
    const geos::geom::Geometry& geometry,
    int32_t zoom);

bool isPointOrRectangle(const geos::geom::Geometry& geometry);

/// Computes the centroid of a non-empty MultiPoint geometry on a sphere.
/// Uses 3D Cartesian coordinates to properly average points on a spherical
/// surface.
/// @param geometry A MultiPoint geometry (must not be empty)
/// @return A pair of (longitude, latitude) in degrees representing the centroid
std::pair<double, double> computeSphericalCentroid(
    const geos::geom::MultiPoint& multiPoint);

/// Represents a point in 3D Cartesian coordinates, useful for spherical
/// geometry calculations. Provides conversions to/from spherical coordinates
/// (longitude, latitude).
class CartesianPoint {
 public:
  /// Constructs a CartesianPoint from a spherical point (longitude, latitude)
  /// in degrees. Assumes the point is on Earth's surface.
  /// @param longitude Longitude in degrees (x-coordinate in spherical system)
  /// @param latitude Latitude in degrees (y-coordinate in spherical system)
  CartesianPoint(double longitude, double latitude);

  /// Constructs a CartesianPoint from Cartesian coordinates.
  /// @param x X-coordinate in Cartesian system
  /// @param y Y-coordinate in Cartesian system
  /// @param z Z-coordinate in Cartesian system
  CartesianPoint(double x, double y, double z);

  double getX() const {
    return x_;
  }
  double getY() const {
    return y_;
  }
  double getZ() const {
    return z_;
  }

  /// Converts this Cartesian point back to spherical coordinates.
  /// @return A pair of (longitude, latitude) in degrees
  std::pair<double, double> toSphericalPoint() const;

 private:
  double x_;
  double y_;
  double z_;
};

double getSphericalLength(const geos::geom::LineString& lineString);

} // namespace facebook::velox::functions::geospatial
