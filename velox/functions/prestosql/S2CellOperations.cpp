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

#include "velox/functions/prestosql/S2CellOperations.h"

#include <memory>

#include <vector>

#include <geos/geom/Geometry.h>
#include <geos/geom/LineString.h>
#include <geos/geom/LinearRing.h>
#include <geos/geom/Polygon.h>
#include <s2/s2cell.h>
#include <s2/s2cell_id.h>
#include <s2/s2earth.h>
#include <s2/s2latlng.h>
#include <s2/s2loop.h>
#include <s2/s2point.h>
#include <s2/s2polygon.h>
#include <s2/s2polyline.h>
#include <s2/s2region_coverer.h>

namespace facebook::velox::functions {

namespace {

// Maps a GEOS coordinate (x=longitude, y=latitude) to a point on the
// unit sphere.
S2Point toS2Point(const geos::geom::Coordinate& coord) {
  return S2LatLng::FromDegrees(coord.y, coord.x).ToPoint();
}

// Converts a GEOS linear ring to an S2Loop. Skips the closing vertex
// (GEOS rings repeat the first vertex; S2 loops are implicitly closed).
// Normalizes the loop so it encloses the smaller of the two spherical
// regions it defines.
std::unique_ptr<S2Loop> toS2Loop(const geos::geom::LinearRing& ring) {
  auto coords = ring.getCoordinates();
  auto numVertices = coords->getSize() - 1;
  std::vector<S2Point> points;
  points.reserve(numVertices);
  for (size_t i = 0; i < numVertices; ++i) {
    points.push_back(toS2Point(coords->getAt(i)));
  }
  auto loop = std::make_unique<S2Loop>(points);
  loop->Normalize();
  return loop;
}

// Converts a GEOS geometry to S2 objects and appends the cell covering.
// A polygon valid in GEOS can be invalid on the sphere (e.g.,
// self-intersecting edges after geodesic projection), which is why we
// check S2Polygon::IsValid().
std::string appendCovering(
    const geos::geom::Geometry& geom,
    int32_t level,
    S2RegionCoverer& coverer,
    std::vector<uint64_t>& result) {
  if (geom.isEmpty()) {
    return {};
  }

  auto typeId = geom.getGeometryTypeId();

  if (typeId == geos::geom::GeometryTypeId::GEOS_POINT) {
    result.push_back(
        S2CellId(toS2Point(*geom.getCoordinate())).parent(level).id());

  } else if (typeId == geos::geom::GeometryTypeId::GEOS_LINESTRING) {
    auto coords = geom.getCoordinates();
    if (coords->getSize() < 2) {
      return "LineString must have at least 2 points";
    }
    std::vector<S2Point> points;
    points.reserve(coords->getSize());
    for (size_t i = 0; i < coords->getSize(); ++i) {
      points.push_back(toS2Point(coords->getAt(i)));
    }
    S2Polyline polyline(points);
    std::vector<S2CellId> covering;
    coverer.GetCovering(polyline, &covering);
    for (const auto& cellId : covering) {
      result.push_back(cellId.id());
    }

  } else if (typeId == geos::geom::GeometryTypeId::GEOS_POLYGON) {
    const auto& polygon = static_cast<const geos::geom::Polygon&>(geom);

    auto* exteriorRing = polygon.getExteriorRing();
    if (exteriorRing->getNumPoints() < 4) {
      return "Polygon ring must have at least 4 points";
    }

    // Build S2Polygon from exterior ring and holes. The S2Polygon
    // constructor determines hole nesting automatically.
    std::vector<std::unique_ptr<S2Loop>> loops;
    loops.reserve(1 + polygon.getNumInteriorRing());
    loops.push_back(toS2Loop(*exteriorRing));
    for (size_t i = 0; i < polygon.getNumInteriorRing(); ++i) {
      loops.push_back(toS2Loop(*polygon.getInteriorRingN(i)));
    }

    S2Polygon s2polygon(std::move(loops));
    if (!s2polygon.IsValid()) {
      return "Polygon is not valid on the sphere";
    }
    std::vector<S2CellId> covering;
    coverer.GetCovering(s2polygon, &covering);
    for (const auto& cellId : covering) {
      result.push_back(cellId.id());
    }

  } else {
    for (size_t i = 0; i < geom.getNumGeometries(); ++i) {
      auto error =
          appendCovering(*geom.getGeometryN(i), level, coverer, result);
      if (!error.empty()) {
        return error;
      }
    }
  }

  return {};
}

} // namespace

namespace {

S2CellId s2CellId(int64_t cellId) {
  return S2CellId(static_cast<uint64_t>(cellId));
}

int64_t fromS2CellId(S2CellId cellId) {
  return static_cast<int64_t>(cellId.id());
}

} // namespace

bool S2CellOp::isValid(int64_t cellId) {
  return s2CellId(cellId).is_valid();
}

double S2CellOp::areaSqKm(int64_t cellId) {
  return S2Earth::SteradiansToSquareKm(::S2Cell{s2CellId(cellId)}.ExactArea());
}

int S2CellOp::level(int64_t cellId) {
  return s2CellId(cellId).level();
}

int64_t S2CellOp::parent(int64_t cellId, int level) {
  return fromS2CellId(s2CellId(cellId).parent(level));
}

bool S2CellOp::contains(int64_t parent, int64_t child) {
  return s2CellId(parent).contains(s2CellId(child));
}

int64_t S2CellOp::fromToken(std::string_view token) {
  return fromS2CellId(S2CellId::FromToken(token));
}

std::string S2CellOp::toToken(int64_t cellId) {
  return s2CellId(cellId).ToToken();
}

int64_t
S2CellOp::cellIdFromFacePositionLevel(int face, uint64_t position, int level) {
  // Start with a valid face cell and descend to the target level by picking
  // a child (0-3) at each step, determined by successive 2-bit groups of
  // 'position'.
  auto cellId = S2CellId::FromFace(face);
  for (int i = 0; i < level; ++i) {
    cellId = cellId.child(static_cast<int>(position & 3));
    position >>= 2;
  }
  return fromS2CellId(cellId);
}

S2Covering S2CellOp::tryCovering(
    const geos::geom::Geometry& geom,
    int32_t level) {
  S2RegionCoverer::Options options;
  options.set_min_level(level);
  options.set_max_level(level);
  S2RegionCoverer coverer(options);

  std::vector<uint64_t> cellIds;
  auto error = appendCovering(geom, level, coverer, cellIds);
  if (!error.empty()) {
    return {{}, std::move(error)};
  }
  return {std::vector<int64_t>(cellIds.begin(), cellIds.end()), {}};
}

S2Covering S2CellOp::tryDissolvedCovering(
    const geos::geom::Geometry& geom,
    int32_t minLevel,
    int32_t maxLevel,
    int32_t maxCells) {
  S2RegionCoverer::Options options;
  options.set_min_level(minLevel);
  options.set_max_level(maxLevel);
  options.set_max_cells(maxCells);
  S2RegionCoverer coverer(options);

  std::vector<uint64_t> cellIds;
  auto error = appendCovering(geom, maxLevel, coverer, cellIds);
  if (!error.empty()) {
    return {{}, std::move(error)};
  }
  return {std::vector<int64_t>(cellIds.begin(), cellIds.end()), {}};
}

} // namespace facebook::velox::functions
