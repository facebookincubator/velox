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
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include <geos/geom/prep/PreparedGeometryFactory.h>
#include <geos/operation/valid/IsSimpleOp.h>
#include <geos/operation/valid/IsValidOp.h>
#include <queue>
#include "velox/common/base/Exceptions.h"
#include "velox/common/geospatial/GeometryConstants.h"
#include "velox/functions/prestosql/types/BingTileType.h"

using geos::operation::valid::IsSimpleOp;
using geos::operation::valid::IsValidOp;

namespace {

class SphericalExcessCalculator {
  static constexpr double TWO_PI = 2 * M_PI;
  static constexpr double THREE_PI = 3 * M_PI;

  double sphericalExcess = 0.0;
  double courseDelta = 0.0;

  bool firstPoint = true;
  double firstInitialBearing = 0.0;
  double previousFinalBearing = 0.0;

  double previousPhi = 0.0;
  double previousCos = 0.0;
  double previousSin = 0.0;
  double previousTan = 0.0;
  double previousLongitude = 0.0;

  bool done = false;

  static double toRadians(double deg) {
    return deg * M_PI / 180.0;
  }

 public:
  explicit SphericalExcessCalculator(const geos::geom::Coordinate& endPoint) {
    previousPhi = toRadians(endPoint.y);
    previousSin = std::sin(previousPhi);
    previousCos = std::cos(previousPhi);
    previousTan = std::tan(previousPhi / 2);
    previousLongitude = toRadians(endPoint.x);
  }

  void add(const geos::geom::Coordinate& point) {
    VELOX_CHECK(!done, "Computation of spherical excess is complete");

    double phi = toRadians(point.y);
    double tan = std::tan(phi / 2);
    double longitude = toRadians(point.x);

    VELOX_USER_CHECK(
        (longitude != previousLongitude || phi != previousPhi),
        "Polygon is not valid: it has two identical consecutive vertices");

    double deltaLongitude = longitude - previousLongitude;
    sphericalExcess += 2 *
        std::atan2(std::tan(deltaLongitude / 2) * (previousTan + tan),
                   1 + previousTan * tan);

    double cos = std::cos(phi);
    double sin = std::sin(phi);
    double sinOfDeltaLongitude = std::sin(deltaLongitude);
    double cosOfDeltaLongitude = std::cos(deltaLongitude);

    // Initial bearing from previous to current
    double y = sinOfDeltaLongitude * cos;
    double x = previousCos * sin - previousSin * cos * cosOfDeltaLongitude;
    double initialBearing = std::fmod(std::atan2(y, x) + TWO_PI, TWO_PI);

    // Final bearing from previous to current = opposite of bearing from current
    // to previous
    double finalY = -sinOfDeltaLongitude * previousCos;
    double finalX = previousSin * cos - previousCos * sin * cosOfDeltaLongitude;
    double finalBearing = std::fmod(std::atan2(finalY, finalX) + M_PI, TWO_PI);

    if (firstPoint) {
      // Keep our initial bearing around, and we'll use it at the end
      // with the last final bearing
      firstInitialBearing = initialBearing;
      firstPoint = false;
    } else {
      courseDelta +=
          std::fmod(initialBearing - previousFinalBearing + THREE_PI, TWO_PI) -
          M_PI;
    }

    courseDelta +=
        std::fmod(finalBearing - initialBearing + THREE_PI, TWO_PI) - M_PI;

    previousFinalBearing = finalBearing;
    previousCos = cos;
    previousSin = sin;
    previousPhi = phi;
    previousTan = tan;
    previousLongitude = longitude;
  }

  double computeSphericalExcess() {
    if (!done) {
      courseDelta +=
          std::fmod(
              firstInitialBearing - previousFinalBearing + THREE_PI, TWO_PI) -
          M_PI;

      // The courseDelta should be 2Pi or - 2Pi, unless a pole is enclosed (and
      // then it should be ~ 0) In which case we need to correct the spherical
      // excess by 2Pi
      if (std::abs(courseDelta) < M_PI / 4) {
        sphericalExcess = std::abs(sphericalExcess) - TWO_PI;
      }
      done = true;
    }
    return sphericalExcess;
  }

  static double excessFromCoordinates(
      const geos::geom::CoordinateSequence& coords) {
    int start = 0;
    size_t end = coords.size();
    // Our calculations rely on not processing the same point twice
    if (coords.getAt(end - 1).equals(coords.getAt(start))) {
      end = end - 1;
    }

    // A path with less than 3 distinct points is not valid for calculating an
    // area
    VELOX_USER_CHECK(
        end - start > 2,
        "Polygon is not valid: a loop contains less then 3 vertices.");

    // Initialize the calculator with the last point
    SphericalExcessCalculator calculator(coords.getAt(end - 1));

    for (int i = start; i < end; i++) {
      calculator.add(coords.getAt(i));
    }

    return calculator.computeSphericalExcess();
  }
};
} // namespace

namespace facebook::velox::functions::geospatial {

static constexpr int32_t kMaxCoveringCount = 1'000'000;

GeometryCollectionIterator::GeometryCollectionIterator(
    const geos::geom::Geometry* geometry) {
  if (!geometry) {
    VELOX_USER_FAIL("geometry is null");
  }
  geometriesDeque.push_back(geometry);
}
// Returns true if there is a next geometry to iterate over
bool GeometryCollectionIterator::hasNext() {
  while (!geometriesDeque.empty()) {
    const geos::geom::Geometry* top = geometriesDeque.back();
    // Check if top is a GeometryCollection
    if (top->getGeometryTypeId() == geos::geom::GEOS_GEOMETRYCOLLECTION) {
      geometriesDeque.pop_back();
      const geos::geom::GeometryCollection* collection =
          dynamic_cast<const geos::geom::GeometryCollection*>(top);
      if (collection) {
        // Push children in reverse order so that the first child is on top of
        // the stack
        for (int i = static_cast<int>(collection->getNumGeometries()) - 1;
             i >= 0;
             --i) {
          geometriesDeque.push_back(collection->getGeometryN(i));
        }
        continue; // Check again with new top
      } else {
        VELOX_FAIL("Failed to cast to GeometryCollection");
      }
    } else {
      // Top is not a collection, so we have a next geometry
      return true;
    }
  }
  return false;
}

const geos::geom::Geometry* GeometryCollectionIterator::next() {
  if (!hasNext()) {
    throw std::out_of_range("No more geometries");
  }
  const geos::geom::Geometry* nextGeometry = geometriesDeque.back(); // NOLINT
  geometriesDeque.pop_back();
  return nextGeometry;
}

std::vector<const geos::geom::Geometry*> flattenCollection(
    const geos::geom::Geometry* geometry) {
  std::vector<const geos::geom::Geometry*> result;
  GeometryCollectionIterator it(geometry);
  while (it.hasNext()) {
    result.push_back(it.next());
  }
  return result;
}

std::optional<std::string> geometryInvalidReason(
    const geos::geom::Geometry* geometry) {
  if (geometry == nullptr) {
    // Null geometries are a problem, but they not invalid or non-simple.
    return std::nullopt;
  }

  IsValidOp isValidOp(geometry);
  const geos::operation::valid::TopologyValidationError*
      topologyValidationError = isValidOp.getValidationError();
  if (topologyValidationError != nullptr) {
    return fmt::format(
        "Invalid {}: {}",
        geometry->getGeometryType(),
        topologyValidationError->getMessage());
  }

  IsSimpleOp isSimpleOp(geometry);
  if (isSimpleOp.isSimple()) {
    return std::nullopt;
  }

  std::string_view description;
  switch (geometry->getGeometryTypeId()) {
    case geos::geom::GeometryTypeId::GEOS_POINT:
      description = "Invalid point";
      break;
    case geos::geom::GeometryTypeId::GEOS_MULTIPOINT:
      description = "Repeated point";
      break;
    case geos::geom::GeometryTypeId::GEOS_LINESTRING:
    case geos::geom::GeometryTypeId::GEOS_LINEARRING:
    case geos::geom::GeometryTypeId::GEOS_MULTILINESTRING:
      description = "Self-intersection at or near";
      break;
    case geos::geom::GeometryTypeId::GEOS_POLYGON:
    case geos::geom::GeometryTypeId::GEOS_MULTIPOLYGON:
    case geos::geom::GeometryTypeId::GEOS_GEOMETRYCOLLECTION:
      // In OGC (which GEOS follows): Polygons, MultiPolygons, Geometry
      // Collections are simple.
      // This shouldn't happen, but in case it does, return a reasonable
      // generic message.
      description = "Topology exception at or near";
      break;
    default:
      return fmt::format(
          "Unknown Geometry type: {}", geometry->getGeometryType());
  }
  geos::geom::Coordinate nonSimpleLocation = isSimpleOp.getNonSimpleLocation();
  return fmt::format(
      "Non-simple {}: {} ({} {})",
      geometry->getGeometryType(),
      description,
      nonSimpleLocation.x,
      nonSimpleLocation.y);
}

Status validateLatitudeLongitude(double latitude, double longitude) {
  if (FOLLY_UNLIKELY(
          latitude < common::geospatial::kMinLatitude ||
          latitude > common::geospatial::kMaxLatitude ||
          longitude < common::geospatial::kMinLongitude ||
          longitude > common::geospatial::kMaxLongitude ||
          std::isnan(latitude) || std::isnan(longitude))) {
    return Status::UserError(
        fmt::format(
            "Latitude must be in range [{}, {}] and longitude must be in range [{}, {}]. Got latitude: {} and longitude: {}",
            common::geospatial::kMinLatitude,
            common::geospatial::kMaxLatitude,
            common::geospatial::kMinLongitude,
            common::geospatial::kMaxLongitude,
            latitude,
            longitude));
  }
  return Status::OK();
}

namespace {

FOLLY_ALWAYS_INLINE void checkLatitudeLongitudeBounds(
    double latitude,
    double longitude) {
  if (FOLLY_UNLIKELY(
          latitude > common::geospatial::kMaxBingTileLatitude ||
          latitude < common::geospatial::kMinBingTileLatitude)) {
    VELOX_USER_FAIL(
        fmt::format(
            "Latitude span for the geometry must be in [{:.2f}, {:.2f}] range",
            common::geospatial::kMinBingTileLatitude,
            common::geospatial::kMaxBingTileLatitude));
  }
  if (FOLLY_UNLIKELY(
          longitude > common::geospatial::kMaxLongitude ||
          longitude < common::geospatial::kMinLongitude)) {
    VELOX_USER_FAIL(
        fmt::format(
            "Longitude span for the geometry must be in [{:.2f}, {:.2f}] range",
            common::geospatial::kMinLongitude,
            common::geospatial::kMaxLongitude));
  }
}

std::optional<std::vector<int64_t>> handleTrivialMinimalTileCoveringCases(
    const geos::geom::Envelope& envelope,
    int32_t zoom) {
  if (envelope.isNull()) {
    return std::vector<int64_t>{};
  }

  checkLatitudeLongitudeBounds(envelope.getMinY(), envelope.getMinX());
  checkLatitudeLongitudeBounds(envelope.getMaxY(), envelope.getMaxX());

  if (FOLLY_UNLIKELY(zoom == 0)) {
    return std::vector<int64_t>{
        static_cast<int64_t>(BingTileType::bingTileCoordsToInt(0, 0, 0))};
  }

  if (FOLLY_UNLIKELY(
          envelope.getMaxX() == envelope.getMinX() &&
          envelope.getMaxY() == envelope.getMinY())) {
    auto res = BingTileType::latitudeLongitudeToTile(
        envelope.getMaxY(), envelope.getMaxX(), zoom);
    if (res.hasError()) {
      VELOX_FAIL(res.error());
    }
    return std::vector<int64_t>{static_cast<int64_t>(res.value())};
  }
  return std::nullopt;
}

std::unique_ptr<geos::geom::Envelope> tileToEnvelope(int64_t tile) {
  uint8_t zoom = BingTileType::bingTileZoom(tile);
  uint32_t x = BingTileType::bingTileX(tile);
  uint32_t y = BingTileType::bingTileY(tile);

  double minX = BingTileType::tileXToLongitude(x, zoom);
  double maxX = BingTileType::tileXToLongitude(x + 1, zoom);
  double minY = BingTileType::tileYToLatitude(y, zoom);
  double maxY = BingTileType::tileYToLatitude(y + 1, zoom);

  return std::make_unique<geos::geom::Envelope>(minX, maxX, minY, maxY);
}

struct TilingEntry {
  TilingEntry(int64_t tile, geos::geom::GeometryFactory* factory)
      : tile{tile},
        envelope{tileToEnvelope(tile)},
        geometry{factory->toGeometry(envelope.get())} {}

  int64_t tile;
  std::unique_ptr<geos::geom::Envelope> envelope;
  std::unique_ptr<geos::geom::Geometry> geometry;
};

bool satisfiesTileEdgeCondition(
    const geos::geom::Envelope& query,
    const TilingEntry& tilingEntry) {
  int64_t tile = tilingEntry.tile;

  int64_t maxXy = (1 << BingTileType::bingTileZoom(tile)) - 1;
  if (BingTileType::bingTileY(tile) < maxXy &&
      query.getMaxY() == tilingEntry.envelope->getMinY()) {
    return false;
  }
  if (BingTileType::bingTileX(tile) < maxXy &&
      query.getMinX() == tilingEntry.envelope->getMaxX()) {
    return false;
  }
  return true;
}

std::vector<int64_t> getRawTilesCoveringGeometry(
    const geos::geom::Geometry& geometry,
    int32_t maxZoom) {
  const geos::geom::Envelope* envelope = geometry.getEnvelopeInternal();
  geos::geom::GeometryFactory::Ptr factory =
      geos::geom::GeometryFactory::create();

  std::optional<std::vector<int64_t>> trivialCases =
      handleTrivialMinimalTileCoveringCases(*envelope, maxZoom);
  if (trivialCases.has_value()) {
    return trivialCases.value();
  }

  auto preparedGeometry =
      geos::geom::prep::PreparedGeometryFactory::prepare(&geometry);
  std::stack<TilingEntry> stack;

  auto addIntersecting = [&](std::int64_t tile) {
    auto tilingEntry = TilingEntry(tile, factory.get());
    if (satisfiesTileEdgeCondition(*envelope, tilingEntry) &&
        preparedGeometry->intersects(tilingEntry.geometry.get())) {
      stack.push(std::move(tilingEntry));
    }
  };

  std::vector<uint64_t> baseTiles = {
      BingTileType::bingTileCoordsToInt(0, 0, 1),
      BingTileType::bingTileCoordsToInt(0, 1, 1),
      BingTileType::bingTileCoordsToInt(1, 0, 1),
      BingTileType::bingTileCoordsToInt(1, 1, 1)};
  std::for_each(baseTiles.begin(), baseTiles.end(), addIntersecting);

  std::vector<int64_t> outputTiles;

  while (!stack.empty()) {
    TilingEntry entry = std::move(stack.top());
    stack.pop();
    if (BingTileType::bingTileZoom(entry.tile) == maxZoom ||
        preparedGeometry->contains(entry.geometry.get())) {
      outputTiles.push_back(entry.tile);
    } else {
      auto children = BingTileType::bingTileChildren(
          entry.tile, BingTileType::bingTileZoom(entry.tile) + 1, 1);
      if (FOLLY_UNLIKELY(children.hasError())) {
        VELOX_FAIL(children.error());
      }
      std::for_each(
          children.value().begin(), children.value().end(), addIntersecting);
      VELOX_CHECK(
          outputTiles.size() + stack.size() <= kMaxCoveringCount,
          "The zoom level is too high or the geometry is too large to compute a set of covering Bing tiles. Please use a lower zoom level, or tile only a section of the geometry.");
    }
  }
  return outputTiles;
}
} // namespace

std::vector<int64_t> getMinimalTilesCoveringGeometry(
    const geos::geom::Geometry& geometry,
    int32_t zoom,
    uint8_t maxZoomShift) {
  std::vector<int64_t> outputTiles;

  std::stack<int64_t, std::vector<int64_t>> stack(
      getRawTilesCoveringGeometry(geometry, zoom));

  while (!stack.empty()) {
    int64_t thisTile = stack.top();
    stack.pop();
    auto expectedChildren =
        BingTileType::bingTileChildren(thisTile, zoom, maxZoomShift);
    if (FOLLY_UNLIKELY(expectedChildren.hasError())) {
      VELOX_FAIL(expectedChildren.error());
    }
    outputTiles.insert(
        outputTiles.end(),
        expectedChildren.value().begin(),
        expectedChildren.value().end());
    if (FOLLY_UNLIKELY(outputTiles.size() + stack.size() > kMaxCoveringCount)) {
      VELOX_USER_FAIL(
          "The zoom level is too high or the geometry is too large to compute a set of covering Bing tiles. Please use a lower zoom level, or tile only a section of the geometry.");
    }
  }

  return outputTiles;
}

std::vector<int64_t> getMinimalTilesCoveringGeometry(
    const geos::geom::Envelope& envelope,
    int32_t zoom) {
  auto trivialCases = handleTrivialMinimalTileCoveringCases(envelope, zoom);
  if (trivialCases.has_value()) {
    return trivialCases.value();
  }

  // envelope x,y (longitude,latitude) goes NE as they increase.
  // tile x,y goes SE as they increase
  auto seRes = BingTileType::latitudeLongitudeToTile(
      envelope.getMinY(), envelope.getMaxX(), zoom);
  if (FOLLY_UNLIKELY(seRes.hasError())) {
    VELOX_FAIL(seRes.error());
  }

  auto nwRes = BingTileType::latitudeLongitudeToTile(
      envelope.getMaxY(), envelope.getMinX(), zoom);
  if (FOLLY_UNLIKELY(nwRes.hasError())) {
    VELOX_FAIL(nwRes.error());
  }

  uint64_t seTile = seRes.value();
  uint64_t nwTile = nwRes.value();

  uint32_t minY = BingTileType::bingTileY(nwTile);
  uint32_t minX = BingTileType::bingTileX(nwTile);
  uint32_t maxY = BingTileType::bingTileY(seTile);
  uint32_t maxX = BingTileType::bingTileX(seTile);

  uint32_t numTiles = (maxX - minX + 1) * (maxY - minY + 1);
  if (numTiles > kMaxCoveringCount) {
    VELOX_USER_FAIL(
        "The zoom level is too high or the geometry is too large to compute a set of covering Bing tiles. Please use a lower zoom level, or tile only a section of the geometry.");
  }

  std::vector<int64_t> results;
  results.reserve((maxX - minX + 1) * (maxY - minY + 1));

  for (uint32_t y = minY; y <= maxY; ++y) {
    for (uint32_t x = minX; x <= maxX; ++x) {
      results.push_back(
          static_cast<int64_t>(BingTileType::bingTileCoordsToInt(x, y, zoom)));
    }
  }
  return results;
}
std::vector<int64_t> getDissolvedTilesCoveringGeometry(
    const geos::geom::Geometry& geometry,
    int32_t zoom) {
  std::vector<int64_t> rawTiles = getRawTilesCoveringGeometry(geometry, zoom);

  const geos::geom::Envelope* envelope = geometry.getEnvelopeInternal();
  checkLatitudeLongitudeBounds(envelope->getMinY(), envelope->getMinX());
  checkLatitudeLongitudeBounds(envelope->getMaxY(), envelope->getMaxX());

  std::vector<int64_t> results;
  if (rawTiles.empty()) {
    return results;
  }

  results.reserve(rawTiles.size());
  std::set<int64_t> candidates;

  auto tileComparator = [](int64_t a, int64_t b) {
    uint8_t za = BingTileType::bingTileZoom(a),
            zb = BingTileType::bingTileZoom(b);
    if (za != zb) {
      return za < zb;
    }
    return BingTileType::bingTileToQuadKey(a) <
        BingTileType::bingTileToQuadKey(b);
  };

  std::priority_queue<int64_t, std::vector<int64_t>, decltype(tileComparator)>
      queue(tileComparator);

  for (auto t : rawTiles) {
    queue.push(t);
  }

  while (!queue.empty()) {
    int64_t candidate = queue.top();
    queue.pop();

    if (BingTileType::bingTileZoom(candidate) == 0) {
      results.push_back(candidate);
      continue;
    }

    auto parentZoom = BingTileType::bingTileZoom(candidate) - 1;
    auto parentResult = BingTileType::bingTileParent(candidate, parentZoom);
    VELOX_CHECK(parentResult.hasValue(), parentResult.error());
    uint64_t parent = parentResult.value();
    candidates.insert(candidate);

    while (!queue.empty() &&
           (BingTileType::bingTileParent(queue.top(), parentZoom).value() ==
            parent)) {
      candidates.insert(queue.top());
      queue.pop();
    }

    if (candidates.size() == 4) {
      // All siblings present, coalesce to parent
      queue.push(static_cast<int64_t>(parent));
    } else {
      results.insert(results.end(), candidates.begin(), candidates.end());
    }
    candidates.clear();
  }

  return results;
}

bool isPointOrRectangle(const geos::geom::Geometry& geometry) {
  if (geometry.getGeometryTypeId() == geos::geom::GeometryTypeId::GEOS_POINT) {
    return true;
  }
  if (geometry.getGeometryTypeId() !=
      geos::geom::GeometryTypeId::GEOS_POLYGON) {
    return false;
  }
  const geos::geom::Polygon& polygon =
      static_cast<const geos::geom::Polygon&>(geometry);

  if (polygon.getNumPoints() != 5) {
    return false;
  }

  auto envelope = geometry.getEnvelopeInternal();

  std::vector<std::tuple<int32_t, int32_t>> coords;
  coords.emplace_back(envelope->getMinX(), envelope->getMinY());
  coords.emplace_back(envelope->getMinX(), envelope->getMaxY());
  coords.emplace_back(envelope->getMaxX(), envelope->getMinY());
  coords.emplace_back(envelope->getMaxX(), envelope->getMaxY());

  for (int i = 0; i < 4; i++) {
    const geos::geom::Point* point =
        static_cast<const geos::geom::Point*>(polygon.getGeometryN(i));
    auto query = std::tuple<int32_t, int32_t>(point->getX(), point->getY());
    if (std::find(coords.begin(), coords.end(), query) == coords.end()) {
      return false;
    }
  }
  return true;
}

std::pair<double, double> computeSphericalCentroid(
    const geos::geom::MultiPoint& multiPoint) {
  VELOX_CHECK(
      !multiPoint.isEmpty(),
      "computeSphericalCentroid does not handle empty geometries");
  auto numPoints = multiPoint.getNumGeometries();

  // If only one point in the multipoint, return it
  if (numPoints == 1) {
    const geos::geom::Point* point =
        static_cast<const geos::geom::Point*>(multiPoint.getGeometryN(0));
    double longitude = point->getX();
    double latitude = point->getY();

    return {longitude, latitude};
  }

  // Convert all points to Cartesian coordinates and sum
  double x3DTotal = 0.0;
  double y3DTotal = 0.0;
  double z3DTotal = 0.0;

  for (int i = 0; i < numPoints; i++) {
    const geos::geom::Point* point = multiPoint.getGeometryN(i);
    double longitude = point->getX();
    double latitude = point->getY();

    // Convert to Cartesian coordinates
    CartesianPoint cp(longitude, latitude);
    x3DTotal += cp.getX();
    y3DTotal += cp.getY();
    z3DTotal += cp.getZ();
  }

  // Calculate the length of the centroid vector
  double centroidVectorLength = std::sqrt(
      x3DTotal * x3DTotal + y3DTotal * y3DTotal + z3DTotal * z3DTotal);

  VELOX_CHECK(
      centroidVectorLength != 0.0,
      fmt::format(
          "Unexpected error. Average vector length adds to zero ({}, {}, {})",
          x3DTotal,
          y3DTotal,
          z3DTotal));

  // Normalize and convert back to spherical coordinates
  CartesianPoint centroid(
      x3DTotal / centroidVectorLength,
      y3DTotal / centroidVectorLength,
      z3DTotal / centroidVectorLength);

  return centroid.toSphericalPoint();
}

CartesianPoint::CartesianPoint(double longitude, double latitude) {
  // Angle from North Pole down to Latitude, in Radians
  double phi = (90.0 - latitude) * M_PI / 180.0;
  double sinPhi = std::sin(phi);
  // Angle from Greenwich to Longitude, in Radians
  double theta = longitude * M_PI / 180.0;

  x_ = BingTileType::kEarthRadiusKm * sinPhi * std::cos(theta);
  y_ = BingTileType::kEarthRadiusKm * sinPhi * std::sin(theta);
  z_ = BingTileType::kEarthRadiusKm * std::cos(phi);
}

CartesianPoint::CartesianPoint(double x, double y, double z)
    : x_(x), y_(y), z_(z) {}

std::pair<double, double> CartesianPoint::toSphericalPoint() const {
  // Angle from North Pole down to Latitude, in Radians
  double phi = std::atan2(std::sqrt(x_ * x_ + y_ * y_), z_);
  // Angle from Greenwich to Longitude, in Radians
  double theta = std::atan2(y_, x_);
  double latitude = 90.0 - phi * 180.0 / M_PI;
  double longitude = theta * 180.0 / M_PI;
  return {longitude, latitude};
}

double getSphericalLength(const geos::geom::LineString& lineString) {
  double sum = 0.0;
  auto numPoints = lineString.getNumPoints();
  auto lastPoint = lineString.getCoordinateN(0);

  for (int i = 1; i < numPoints; i++) {
    auto thisPoint = lineString.getCoordinateN(i);
    sum += BingTileType::greatCircleDistance(
        lastPoint.y, lastPoint.x, thisPoint.y, thisPoint.x);
    lastPoint = thisPoint;
  }

  return sum;
}

double computeSphericalExcess(const geos::geom::Polygon& polygon) {
  double sphericalExcess = std::abs(
      SphericalExcessCalculator::excessFromCoordinates(
          *polygon.getExteriorRing()->getCoordinates()));
  auto interiorRingCount = polygon.getNumInteriorRing();
  for (int i = 0; i < interiorRingCount; i++) {
    sphericalExcess -= std::abs(
        SphericalExcessCalculator::excessFromCoordinates(
            *polygon.getInteriorRingN(i)->getCoordinates()));
  }
  return sphericalExcess;
}

} // namespace facebook::velox::functions::geospatial
