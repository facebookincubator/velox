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

#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/LineString.h>
#include <geos/geom/Point.h>
#include <geos/geom/Polygon.h>

#include "velox/common/base/IOUtils.h"
#include "velox/common/geospatial/GeometrySerde.h"

using facebook::velox::common::InputByteStream;
using facebook::velox::common::geospatial::EsriShapeType;
using facebook::velox::common::geospatial::GeometrySerializationType;

namespace facebook::velox::common::geospatial {

void GeometrySerializer::serializeEnvelope(
    double xMin,
    double yMin,
    double xMax,
    double yMax,
    exec::StringWriter& writer) {
  appendBytes(
      writer, static_cast<uint8_t>(GeometrySerializationType::ENVELOPE));
  appendBytes(writer, xMin);
  appendBytes(writer, yMin);
  appendBytes(writer, xMax);
  appendBytes(writer, yMax);
}

void GeometrySerializer::serializeEnvelope(
    geos::geom::Envelope& envelope,
    exec::StringWriter& writer) {
  if (FOLLY_UNLIKELY(envelope.isNull())) {
    serializeEnvelope(
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        writer);
  } else {
    serializeEnvelope(
        envelope.getMinX(),
        envelope.getMinY(),
        envelope.getMaxX(),
        envelope.getMaxY(),
        writer);
  }
}

bool GeometrySerializer::isClockwise(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end) {
  double sum = 0.0;
  for (size_t i = start; i < end - 1; i++) {
    const auto& p1 = coordinates->getAt(i);
    const auto& p2 = coordinates->getAt(i + 1);
    sum += (p2.x - p1.x) * (p2.y + p1.y);
  }
  return sum > 0.0;
}

void GeometrySerializer::reverse(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end) {
  for (size_t i = 0; i < (end - start) / 2; ++i) {
    auto temp = coordinates->getAt(start + i);
    coordinates->setAt(coordinates->getAt(end - 1 - i), start + i);
    coordinates->setAt(temp, end - 1 - i);
  }
}

void GeometrySerializer::canonicalizePolygonCoordinates(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    size_t start,
    size_t end,
    bool isShell) {
  bool isClockwiseFlag = isClockwise(coordinates, start, end);

  if ((isShell && !isClockwiseFlag) || (!isShell && isClockwiseFlag)) {
    reverse(coordinates, start, end);
  }
}

void GeometrySerializer::canonicalizePolygonCoordinates(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
    const std::vector<size_t>& partIndexes,
    const std::vector<bool>& shellPart) {
  for (size_t part = 0; part < partIndexes.size() - 1; part++) {
    canonicalizePolygonCoordinates(
        coordinates, partIndexes[part], partIndexes[part + 1], shellPart[part]);
  }
  if (!partIndexes.empty()) {
    canonicalizePolygonCoordinates(
        coordinates, partIndexes.back(), coordinates->size(), shellPart.back());
  }
}

void GeometrySerializer::writeGeometry(
    const geos::geom::Geometry& geometry,
    exec::StringWriter& writer) {
  auto geometryType = geometry.getGeometryTypeId();
  switch (geometryType) {
    case geos::geom::GEOS_POINT:
      writePoint(geometry, writer);
      break;
    case geos::geom::GEOS_MULTIPOINT:
      writeMultiPoint(geometry, writer);
      break;
    case geos::geom::GEOS_LINESTRING:
    case geos::geom::GEOS_LINEARRING:
      writePolyline(geometry, writer, false);
      break;
    case geos::geom::GEOS_MULTILINESTRING:
      writePolyline(geometry, writer, true);
      break;
    case geos::geom::GEOS_POLYGON:
      writePolygon(geometry, writer, false);
      break;
    case geos::geom::GEOS_MULTIPOLYGON:
      writePolygon(geometry, writer, true);
      break;
    case geos::geom::GEOS_GEOMETRYCOLLECTION:
      writeGeometryCollection(geometry, writer);
      break;
    default:
      VELOX_FAIL(
          "Unrecognized geometry type: {}",
          static_cast<uint32_t>(geometryType));
      break;
  }
}

void GeometrySerializer::writeEnvelope(
    const geos::geom::Geometry& geometry,
    exec::StringWriter& writer) {
  if (geometry.isEmpty()) {
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
    return;
  }

  auto envelope = geometry.getEnvelopeInternal();
  appendBytes(writer, envelope->getMinX());
  appendBytes(writer, envelope->getMinY());
  appendBytes(writer, envelope->getMaxX());
  appendBytes(writer, envelope->getMaxY());
}

void GeometrySerializer::writeCoordinates(
    const std::unique_ptr<geos::geom::CoordinateSequence>& coords,
    exec::StringWriter& writer) {
  for (size_t i = 0; i < coords->size(); ++i) {
    appendBytes(writer, coords->getX(i));
    appendBytes(writer, coords->getY(i));
  }
}

void GeometrySerializer::writePoint(
    const geos::geom::Geometry& point,
    exec::StringWriter& writer) {
  appendBytes(writer, static_cast<uint8_t>(GeometrySerializationType::POINT));
  if (!point.isEmpty()) {
    writeCoordinates(point.getCoordinates(), writer);
  } else {
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
    appendBytes(writer, std::numeric_limits<double>::quiet_NaN());
  }
}

void GeometrySerializer::writeMultiPoint(
    const geos::geom::Geometry& geometry,
    exec::StringWriter& writer) {
  appendBytes(
      writer, static_cast<uint8_t>(GeometrySerializationType::MULTI_POINT));
  appendBytes(writer, static_cast<int32_t>(EsriShapeType::MULTI_POINT));
  writeEnvelope(geometry, writer);
  appendBytes(writer, static_cast<int32_t>(geometry.getNumPoints()));
  writeCoordinates(geometry.getCoordinates(), writer);
}

void GeometrySerializer::writePolyline(
    const geos::geom::Geometry& geometry,
    exec::StringWriter& writer,
    bool multiType) {
  size_t numParts;
  size_t numPoints = geometry.getNumPoints();

  if (multiType) {
    numParts = geometry.getNumGeometries();
    appendBytes(
        writer,
        static_cast<uint8_t>(GeometrySerializationType::MULTI_LINE_STRING));
  } else {
    numParts = (numPoints > 0) ? 1 : 0;
    appendBytes(
        writer, static_cast<uint8_t>(GeometrySerializationType::LINE_STRING));
  }

  appendBytes(writer, static_cast<int32_t>(EsriShapeType::POLYLINE));

  writeEnvelope(geometry, writer);

  appendBytes(writer, static_cast<int32_t>(numParts));
  appendBytes(writer, static_cast<int32_t>(numPoints));

  size_t partIndex = 0;
  for (size_t geomIdx = 0; geomIdx < numParts; ++geomIdx) {
    appendBytes(writer, static_cast<int32_t>(partIndex));
    partIndex += geometry.getGeometryN(geomIdx)->getNumPoints();
  }

  if (multiType) {
    for (size_t partIdx = 0; partIdx < numParts; ++partIdx) {
      const auto* part = geometry.getGeometryN(partIdx);
      writeCoordinates(part->getCoordinates(), writer);
    }
  } else {
    writeCoordinates(geometry.getCoordinates(), writer);
  }
}

void GeometrySerializer::writePolygon(
    const geos::geom::Geometry& geometry,
    exec::StringWriter& writer,
    bool multiType) {
  size_t numGeometries = geometry.getNumGeometries();
  size_t numParts = 0;
  size_t numPoints = geometry.getNumPoints();

  for (size_t geomIdx = 0; geomIdx < numGeometries; geomIdx++) {
    auto polygon = dynamic_cast<const geos::geom::Polygon*>(
        geometry.getGeometryN(geomIdx));
    if (polygon && polygon->getNumPoints() > 0) {
      numParts += polygon->getNumInteriorRing() + 1;
    }
  }

  if (multiType) {
    appendBytes(
        writer, static_cast<uint8_t>(GeometrySerializationType::MULTI_POLYGON));
  } else {
    appendBytes(
        writer, static_cast<uint8_t>(GeometrySerializationType::POLYGON));
  }

  appendBytes(writer, static_cast<int32_t>(EsriShapeType::POLYGON));
  writeEnvelope(geometry, writer);

  appendBytes(writer, static_cast<int32_t>(numParts));
  appendBytes(writer, static_cast<int32_t>(numPoints));

  if (numParts == 0) {
    return;
  }

  std::vector<size_t> partIndexes(numParts);
  std::vector<bool> shellPart(numParts);

  size_t currentPart = 0;
  size_t currentPoint = 0;
  for (size_t geomIdx = 0; geomIdx < numGeometries; geomIdx++) {
    const geos::geom::Polygon* polygon =
        dynamic_cast<const geos::geom::Polygon*>(
            geometry.getGeometryN(geomIdx));

    partIndexes[currentPart] = currentPoint;
    shellPart[currentPart] = true;
    currentPart++;
    currentPoint += polygon->getExteriorRing()->getNumPoints();

    size_t holesCount = polygon->getNumInteriorRing();
    for (size_t holeIndex = 0; holeIndex < holesCount; holeIndex++) {
      partIndexes[currentPart] = currentPoint;
      shellPart[currentPart] = false;
      currentPart++;
      currentPoint += polygon->getInteriorRingN(holeIndex)->getNumPoints();
    }
  }

  for (size_t partIndex : partIndexes) {
    appendBytes(writer, static_cast<int32_t>(partIndex));
  }

  auto coordinates = geometry.getCoordinates();
  canonicalizePolygonCoordinates(coordinates, partIndexes, shellPart);
  writeCoordinates(coordinates, writer);
}

void GeometrySerializer::writeGeometryCollection(
    const geos::geom::Geometry& collection,
    exec::StringWriter& writer) {
  appendBytes(
      writer,
      static_cast<uint8_t>(GeometrySerializationType::GEOMETRY_COLLECTION));
  for (size_t geometryIndex = 0; geometryIndex < collection.getNumGeometries();
       ++geometryIndex) {
    const auto* geometry = collection.getGeometryN(geometryIndex);
    size_t lengthOffset = writer.size();
    // Placeholder for the length of the collection
    appendBytes(writer, int32_t{0});
    writeGeometry(*geometry, writer);
    int32_t length =
        static_cast<int32_t>(writer.size() - lengthOffset - sizeof(int32_t));
    writer.writeAt(length, lengthOffset);
  }
}

// GeometryDeserializer implementations

geos::geom::GeometryFactory* GeometryDeserializer::getGeometryFactory() {
  thread_local static geos::geom::GeometryFactory::Ptr geometryFactory =
      geos::geom::GeometryFactory::create();
  return geometryFactory.get();
}

std::unique_ptr<geos::geom::Geometry> GeometryDeserializer::readGeometry(
    velox::common::InputByteStream& stream,
    size_t size) {
  auto geometryType = static_cast<GeometrySerializationType>(
      stream.read<GeometrySerializationType>());
  switch (geometryType) {
    case GeometrySerializationType::POINT:
      return readPoint(stream);
    case GeometrySerializationType::MULTI_POINT:
      return readMultiPoint(stream);
    case GeometrySerializationType::LINE_STRING:
      return readPolyline(stream, false);
    case GeometrySerializationType::MULTI_LINE_STRING:
      return readPolyline(stream, true);
    case GeometrySerializationType::POLYGON:
      return readPolygon(stream, false);
    case GeometrySerializationType::MULTI_POLYGON:
      return readPolygon(stream, true);
    case GeometrySerializationType::ENVELOPE:
      return readEnvelope(stream);
    case GeometrySerializationType::GEOMETRY_COLLECTION:
      return readGeometryCollection(stream, size);
    default:
      VELOX_FAIL(
          "Unrecognized geometry type: {}", static_cast<uint8_t>(geometryType));
  }
}

const std::unique_ptr<geos::geom::Envelope>
GeometryDeserializer::deserializeEnvelope(const StringView& geometry) {
  velox::common::InputByteStream inputStream(geometry.data());
  auto geometryType = inputStream.read<GeometrySerializationType>();

  switch (geometryType) {
    case GeometrySerializationType::POINT:
      return std::make_unique<geos::geom::Envelope>(
          *readPoint(inputStream)->getEnvelopeInternal());
    case GeometrySerializationType::MULTI_POINT:
    case GeometrySerializationType::LINE_STRING:
    case GeometrySerializationType::MULTI_LINE_STRING:
    case GeometrySerializationType::POLYGON:
    case GeometrySerializationType::MULTI_POLYGON:
      skipEsriType(inputStream);
      return deserializeEnvelope(inputStream);
    case GeometrySerializationType::ENVELOPE:
      return deserializeEnvelope(inputStream);
    case GeometrySerializationType::GEOMETRY_COLLECTION:
      return std::make_unique<geos::geom::Envelope>(
          *readGeometryCollection(inputStream, geometry.size())
               ->getEnvelopeInternal());
    default:
      VELOX_FAIL(
          "Unrecognized geometry type: {}", static_cast<uint8_t>(geometryType));
  }
}

std::unique_ptr<geos::geom::Envelope> GeometryDeserializer::deserializeEnvelope(
    velox::common::InputByteStream& input) {
  auto xMin = input.read<double>();
  auto yMin = input.read<double>();
  auto xMax = input.read<double>();
  auto yMax = input.read<double>();

  if (FOLLY_UNLIKELY(
          isEsriNaN(xMin) || isEsriNaN(yMin) || isEsriNaN(xMax) ||
          isEsriNaN(yMax))) {
    return std::make_unique<geos::geom::Envelope>();
  }

  return std::make_unique<geos::geom::Envelope>(xMin, xMax, yMin, yMax);
}

geos::geom::Coordinate GeometryDeserializer::readCoordinate(
    velox::common::InputByteStream& input) {
  auto x = input.read<double>();
  auto y = input.read<double>();
  return {x, y};
}

std::unique_ptr<geos::geom::CoordinateSequence>
GeometryDeserializer::readCoordinates(
    velox::common::InputByteStream& input,
    size_t count) {
  auto coords = std::make_unique<geos::geom::CoordinateArraySequence>(count, 2);
  for (size_t i = 0; i < count; ++i) {
    // TODO: Consider using setOrdinate if there's a performance issue.
    coords->setAt(readCoordinate(input), i);
  }
  return coords;
}

std::unique_ptr<geos::geom::Point> GeometryDeserializer::readPoint(
    velox::common::InputByteStream& input) {
  geos::geom::Coordinate coordinate = readCoordinate(input);
  if (std::isnan(coordinate.x) || std::isnan(coordinate.y)) {
    return getGeometryFactory()->createPoint();
  }
  return std::unique_ptr<geos::geom::Point>(
      getGeometryFactory()->createPoint(coordinate));
}

std::unique_ptr<geos::geom::Geometry> GeometryDeserializer::readMultiPoint(
    velox::common::InputByteStream& input) {
  skipEsriType(input);
  skipEnvelope(input);
  size_t pointCount = input.read<int32_t>();
  auto coords = readCoordinates(input, pointCount);
  std::vector<std::unique_ptr<geos::geom::Point>> points;
  points.reserve(coords->size());
  for (size_t i = 0; i < coords->size(); ++i) {
    points.push_back(
        std::unique_ptr<geos::geom::Point>(getGeometryFactory()->createPoint(
            geos::geom::Coordinate(coords->getX(i), coords->getY(i)))));
  }
  return getGeometryFactory()->createMultiPoint(std::move(points));
}

std::unique_ptr<geos::geom::Geometry> GeometryDeserializer::readPolyline(
    velox::common::InputByteStream& input,
    bool multiType) {
  skipEsriType(input);
  skipEnvelope(input);
  size_t partCount = input.read<int32_t>();
  size_t pointCount = input.read<int32_t>();

  if (partCount == 0) {
    if (multiType) {
      return getGeometryFactory()->createMultiLineString();
    }
    return getGeometryFactory()->createLineString();
  }

  std::vector<size_t> startIndexes(partCount);
  for (size_t i = 0; i < partCount; ++i) {
    startIndexes[i] = input.read<int32_t>();
  }

  std::vector<size_t> partLengths(partCount);
  if (partCount > 1) {
    partLengths[0] = startIndexes[1];
    for (size_t i = 1; i < partCount - 1; ++i) {
      partLengths[i] = startIndexes[i + 1] - startIndexes[i];
    }
  }
  partLengths[partCount - 1] = pointCount - startIndexes[partCount - 1];

  std::vector<std::unique_ptr<geos::geom::LineString>> lineStrings;
  lineStrings.reserve(partCount);
  for (size_t i = 0; i < partCount; ++i) {
    lineStrings.push_back(
        getGeometryFactory()->createLineString(
            readCoordinates(input, partLengths[i])));
  }

  if (multiType) {
    return getGeometryFactory()->createMultiLineString(std::move(lineStrings));
  }

  if (lineStrings.size() != 1) {
    VELOX_FAIL("Expected a single LineString for non-multiType polyline.");
  }

  return std::move(lineStrings[0]);
}

std::unique_ptr<geos::geom::Geometry> GeometryDeserializer::readPolygon(
    velox::common::InputByteStream& input,
    bool multiType) {
  skipEsriType(input);
  skipEnvelope(input);

  size_t partCount = input.read<int32_t>();
  size_t pointCount = input.read<int32_t>();
  if (partCount == 0) {
    if (multiType) {
      return getGeometryFactory()->createMultiPolygon();
    }
    return getGeometryFactory()->createPolygon();
  }

  std::vector<size_t> startIndexes(partCount);
  for (size_t i = 0; i < partCount; i++) {
    startIndexes[i] = input.read<int32_t>();
  }

  std::vector<size_t> partLengths(partCount);
  if (partCount > 1) {
    partLengths[0] = startIndexes[1];
    for (size_t i = 1; i < partCount - 1; i++) {
      partLengths[i] = startIndexes[i + 1] - startIndexes[i];
    }
  }
  partLengths[partCount - 1] = pointCount - startIndexes[partCount - 1];

  std::unique_ptr<geos::geom::LinearRing> shell = nullptr;
  std::vector<std::unique_ptr<geos::geom::LinearRing>> holes;
  std::vector<std::unique_ptr<geos::geom::Polygon>> polygons;

  // Shells _should_ be clockwise and holes _should_ be counter-clockwise,
  // but this doesn't always happen for single Polygons. For single Polygons,
  // we read the first ring as a shell and the rest as holes. For MultiPolygons,
  // we read the first ring as a shell, and any counter-clockwise rings as
  // holes, then push a polygon and reset if a clockwise ring is encountered.
  for (size_t i = 0; i < partCount; i++) {
    auto coordinates = readCoordinates(input, partLengths[i]);

    if (multiType) {
      bool clockwiseFlag =
          GeometrySerializer::isClockwise(coordinates, 0, coordinates->size());
      if (shell && clockwiseFlag) {
        // next polygon has started
        polygons.push_back(
            getGeometryFactory()->createPolygon(
                std::move(shell), std::move(holes)));
        holes.clear();
        shell = nullptr;
      }
    }

    auto ring = getGeometryFactory()->createLinearRing(std::move(coordinates));
    if (shell == nullptr) {
      shell = std::move(ring);
    } else {
      holes.push_back(std::move(ring));
    }
  }

  polygons.push_back(
      getGeometryFactory()->createPolygon(std::move(shell), std::move(holes)));

  if (multiType) {
    return getGeometryFactory()->createMultiPolygon(std::move(polygons));
  }

  if (polygons.size() != 1) {
    VELOX_FAIL("Expected exactly one polygon, but found multiple.");
  }
  return std::move(polygons[0]);
}

std::unique_ptr<geos::geom::Geometry> GeometryDeserializer::readEnvelope(
    velox::common::InputByteStream& input) {
  auto xMin = input.read<double>();
  auto yMin = input.read<double>();
  auto xMax = input.read<double>();
  auto yMax = input.read<double>();

  if (isEsriNaN(xMin) || isEsriNaN(yMin) || isEsriNaN(xMax) ||
      isEsriNaN(yMax)) {
    return getGeometryFactory()->createPolygon();
  }

  auto coordinates = std::make_unique<geos::geom::CoordinateArraySequence>();
  coordinates->add(geos::geom::Coordinate(xMin, yMin));
  coordinates->add(geos::geom::Coordinate(xMin, yMax));
  coordinates->add(geos::geom::Coordinate(xMax, yMax));
  coordinates->add(geos::geom::Coordinate(xMax, yMin));
  coordinates->add(geos::geom::Coordinate(xMin, yMin)); // Close the ring

  auto shell = getGeometryFactory()->createLinearRing(std::move(coordinates));
  return getGeometryFactory()->createPolygon(std::move(shell), {});
}

std::unique_ptr<geos::geom::Geometry>
GeometryDeserializer::readGeometryCollection(
    velox::common::InputByteStream& input,
    size_t size) {
  std::vector<std::unique_ptr<geos::geom::Geometry>> geometries;

  auto offset = input.offset();
  while (size - offset > 0) {
    // Skip the length field
    input.read<int32_t>();
    geometries.push_back(readGeometry(input, size));
    offset = input.offset();
  }
  std::vector<const geos::geom::Geometry*> rawGeometries;
  rawGeometries.reserve(geometries.size());
  for (const auto& geometry : geometries) {
    rawGeometries.push_back(geometry.get());
  }

  return std::unique_ptr<geos::geom::GeometryCollection>(
      getGeometryFactory()->createGeometryCollection(rawGeometries));
}

} // namespace facebook::velox::common::geospatial
