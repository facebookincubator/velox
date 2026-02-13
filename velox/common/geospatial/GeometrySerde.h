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

#pragma once

#include <folly/Likely.h>
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>

#include "velox/common/base/IOUtils.h"
#include "velox/common/geospatial/GeometryConstants.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/type/StringView.h"

namespace facebook::velox::common::geospatial {

/**
 * VarbinaryWriter is a utility for serializing raw binary data to a
 * generic writer interface. It supports writing either raw byte arrays
 * or trivially copyable types.
 *
 * @tparam StringWriter A type that provides an `append(std::string_view)`
 * method, used to consume the binary output. Examples include `std::string` or
 *         `core::StringWriter`.
 */
template <typename StringWriter>
class VarbinaryWriter {
 public:
  /* implicit */ VarbinaryWriter(StringWriter& stringWriter)
      : stringWriter_(stringWriter) {}
  VarbinaryWriter() = delete;

  void write(const char* data, size_t size) {
    stringWriter_.append(std::string_view(data, size));
  }

  template <typename T>
  void write(const T& value) {
    static_assert(
        std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    stringWriter_.append(
        std::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
  }

 private:
  StringWriter& stringWriter_;
};

class GeometrySerializer {
 public:
  /// Serialize geometry into Velox's internal format.  Do not call this within
  /// GEOS_TRY macro: it will catch the exceptions that need to bubble up.
  template <typename StringWriter>
  static void serialize(
      const geos::geom::Geometry& geometry,
      StringWriter& stringWriter) {
    VarbinaryWriter writer(stringWriter);
    writeGeometry(geometry, writer);
  }

  template <typename StringWriter>
  static void serializeEnvelope(
      double xMin,
      double yMin,
      double xMax,
      double yMax,
      StringWriter& stringWriter) {
    VarbinaryWriter writer(stringWriter);
    writer.write(static_cast<uint8_t>(GeometrySerializationType::ENVELOPE));
    writer.write(xMin);
    writer.write(yMin);
    writer.write(xMax);
    writer.write(yMax);
  }

  template <typename StringWriter>
  static void serializeEnvelope(
      geos::geom::Envelope& envelope,
      StringWriter& stringWriter) {
    if (FOLLY_UNLIKELY(envelope.isNull())) {
      serializeEnvelope(
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          stringWriter);
    } else {
      serializeEnvelope(
          envelope.getMinX(),
          envelope.getMinY(),
          envelope.getMaxX(),
          envelope.getMaxY(),
          stringWriter);
    }
  }

  /// Determines if a ring of coordinates (from `start` to `end`) is oriented
  /// clockwise.
  FOLLY_ALWAYS_INLINE static bool isClockwise(
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

  /// Reverses the order of coordinates in the sequence between `start` and
  /// `end`
  FOLLY_ALWAYS_INLINE static void reverse(
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
  FOLLY_ALWAYS_INLINE static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      size_t start,
      size_t end,
      bool isShell) {
    bool isClockwiseFlag = isClockwise(coordinates, start, end);

    if ((isShell && !isClockwiseFlag) || (!isShell && isClockwiseFlag)) {
      reverse(coordinates, start, end);
    }
  }

  /// Applies `canonicalizePolygonCoordinates` to all rings in a polygon.
  FOLLY_ALWAYS_INLINE static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      const std::vector<size_t>& partIndexes,
      const std::vector<bool>& shellPart) {
    for (size_t part = 0; part < partIndexes.size() - 1; part++) {
      canonicalizePolygonCoordinates(
          coordinates,
          partIndexes[part],
          partIndexes[part + 1],
          shellPart[part]);
    }
    if (!partIndexes.empty()) {
      canonicalizePolygonCoordinates(
          coordinates,
          partIndexes.back(),
          coordinates->size(),
          shellPart.back());
    }
  }

 private:
  template <typename T>
  static void writeGeometry(
      const geos::geom::Geometry& geometry,
      VarbinaryWriter<T>& writer) {
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

  template <typename T>
  static void writeEnvelope(
      const geos::geom::Geometry& geometry,
      VarbinaryWriter<T>& writer) {
    if (geometry.isEmpty()) {
      writer.write(std::numeric_limits<double>::quiet_NaN());
      writer.write(std::numeric_limits<double>::quiet_NaN());
      writer.write(std::numeric_limits<double>::quiet_NaN());
      writer.write(std::numeric_limits<double>::quiet_NaN());
      return;
    }

    auto envelope = geometry.getEnvelopeInternal();
    writer.write(envelope->getMinX());
    writer.write(envelope->getMinY());
    writer.write(envelope->getMaxX());
    writer.write(envelope->getMaxY());
  }

  template <typename T>
  static void writeCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coords,
      VarbinaryWriter<T>& writer) {
    for (size_t i = 0; i < coords->size(); ++i) {
      writer.write(coords->getX(i));
      writer.write(coords->getY(i));
    }
  }

  template <typename T>
  static void writePoint(
      const geos::geom::Geometry& point,
      VarbinaryWriter<T>& writer) {
    writer.write(static_cast<uint8_t>(GeometrySerializationType::POINT));
    if (!point.isEmpty()) {
      writeCoordinates(point.getCoordinates(), writer);
    } else {
      writer.write(std::numeric_limits<double>::quiet_NaN());
      writer.write(std::numeric_limits<double>::quiet_NaN());
    }
  }

  template <typename T>
  static void writeMultiPoint(
      const geos::geom::Geometry& geometry,
      VarbinaryWriter<T>& writer) {
    writer.write(static_cast<uint8_t>(GeometrySerializationType::MULTI_POINT));
    writer.write(static_cast<int32_t>(EsriShapeType::MULTI_POINT));
    writeEnvelope(geometry, writer);
    writer.write(static_cast<int32_t>(geometry.getNumPoints()));
    writeCoordinates(geometry.getCoordinates(), writer);
  }

  template <typename T>
  static void writePolyline(
      const geos::geom::Geometry& geometry,
      VarbinaryWriter<T>& writer,
      bool multiType) {
    size_t numParts;
    size_t numPoints = geometry.getNumPoints();

    if (multiType) {
      numParts = geometry.getNumGeometries();
      writer.write(
          static_cast<uint8_t>(GeometrySerializationType::MULTI_LINE_STRING));
    } else {
      numParts = (numPoints > 0) ? 1 : 0;
      writer.write(
          static_cast<uint8_t>(GeometrySerializationType::LINE_STRING));
    }

    writer.write(static_cast<int32_t>(EsriShapeType::POLYLINE));

    writeEnvelope(geometry, writer);

    writer.write(static_cast<int32_t>(numParts));
    writer.write(static_cast<int32_t>(numPoints));

    size_t partIndex = 0;
    for (size_t geomIdx = 0; geomIdx < numParts; ++geomIdx) {
      writer.write(static_cast<int32_t>(partIndex));
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

  template <typename T>
  static void writePolygon(
      const geos::geom::Geometry& geometry,
      VarbinaryWriter<T>& writer,
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
      writer.write(
          static_cast<uint8_t>(GeometrySerializationType::MULTI_POLYGON));
    } else {
      writer.write(static_cast<uint8_t>(GeometrySerializationType::POLYGON));
    }

    writer.write(static_cast<int32_t>(EsriShapeType::POLYGON));
    writeEnvelope(geometry, writer);

    writer.write(static_cast<int32_t>(numParts));
    writer.write(static_cast<int32_t>(numPoints));

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
      writer.write(static_cast<int32_t>(partIndex));
    }

    auto coordinates = geometry.getCoordinates();
    canonicalizePolygonCoordinates(coordinates, partIndexes, shellPart);
    writeCoordinates(coordinates, writer);
  }

  template <typename T>
  static void writeGeometryCollection(
      const geos::geom::Geometry& collection,
      VarbinaryWriter<T>& writer) {
    writer.write(
        static_cast<uint8_t>(GeometrySerializationType::GEOMETRY_COLLECTION));

    for (size_t geometryIndex = 0;
         geometryIndex < collection.getNumGeometries();
         ++geometryIndex) {
      auto* geometry = collection.getGeometryN(geometryIndex);
      // Use a temporary buffer to serialize the geometry and calculate its
      // length
      std::string tempBuffer;
      VarbinaryWriter tempOutput(tempBuffer);

      writeGeometry(*geometry, tempOutput);

      int32_t length = static_cast<int32_t>(tempBuffer.size());
      writer.write(length);
      writer.write(tempBuffer.data(), tempBuffer.size());
    }
  }
};

class GeometryDeserializer {
 public:
  /// Deserialize Velox's internal format to a geometry.  Do not call this
  /// within GEOS_TRY macro: it will catch the exceptions that need to bubble
  /// up.
  static std::unique_ptr<geos::geom::Geometry> deserialize(
      const StringView& geometryString) {
    velox::common::InputByteStream inputStream(geometryString.data());
    return readGeometry(inputStream, geometryString.size());
  }

  static const std::unique_ptr<geos::geom::Envelope> deserializeEnvelope(
      const StringView& geometry);

  template <typename T>
  static std::unique_ptr<geos::geom::CoordinateArraySequence>
  deserializePointsToCoordinate(
      const exec::ArrayView<true, T>& input,
      const std::string& functionName,
      bool forbidDuplicates) {
    std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
        std::make_unique<geos::geom::CoordinateArraySequence>(input.size(), 2);

    double lastX = std::numeric_limits<double>::signaling_NaN();
    double lastY = std::numeric_limits<double>::signaling_NaN();
    for (int i = 0; i < input.size(); i++) {
      if (!input[i].has_value()) {
        VELOX_USER_FAIL(
            fmt::format(
                "Invalid input to {}: input array contains null at index {}.",
                functionName,
                i));
      }

      StringView view = *input[i];

      velox::common::InputByteStream inputStream(view.data());
      auto geometryType = inputStream.read<GeometrySerializationType>();
      if (geometryType != GeometrySerializationType::POINT) {
        VELOX_USER_FAIL(
            fmt::format(
                "Non-point geometry in {} input at index {}.",
                functionName,
                i));
      }
      auto x = inputStream.read<double>();
      auto y = inputStream.read<double>();
      if (std::isnan(x) || std::isnan(y)) {
        VELOX_USER_FAIL(
            fmt::format(
                "Empty point in {} input at index {}.", functionName, i));
      }
      if (forbidDuplicates && x == lastX && y == lastY) {
        VELOX_USER_FAIL(
            fmt::format(
                "Repeated point sequence in {}: point {},{} at index {}.",
                functionName,
                x,
                y,
                i));
      }
      lastX = x;
      lastY = y;
      coords->setAt({x, y}, i);
    }
    return coords;
  }

  /// Returns the thread-local GEOS geometry factory.
  static geos::geom::GeometryFactory* getGeometryFactory();

 private:
  static std::unique_ptr<geos::geom::Geometry> readGeometry(
      velox::common::InputByteStream& stream,
      size_t size);

  static bool isEsriNaN(double d) {
    return std::isnan(d) || d < -1.0E38;
  }

  static void skipEsriType(velox::common::InputByteStream& input) {
    input.read<int32_t>(); // Esri type is an integer
  }

  static void skipEnvelope(velox::common::InputByteStream& input) {
    input.read<double>(4); // Envelopes are 4 doubles (minX, minY, maxX, maxY)
  }

  static std::unique_ptr<geos::geom::Envelope> deserializeEnvelope(
      velox::common::InputByteStream& input);

  static geos::geom::Coordinate readCoordinate(
      velox::common::InputByteStream& input);

  static std::unique_ptr<geos::geom::CoordinateSequence> readCoordinates(
      velox::common::InputByteStream& input,
      size_t count);

  static std::unique_ptr<geos::geom::Point> readPoint(
      velox::common::InputByteStream& input);

  static std::unique_ptr<geos::geom::Geometry> readMultiPoint(
      velox::common::InputByteStream& input);

  static std::unique_ptr<geos::geom::Geometry> readPolyline(
      velox::common::InputByteStream& input,
      bool multiType);

  static std::unique_ptr<geos::geom::Geometry> readPolygon(
      velox::common::InputByteStream& input,
      bool multiType);

  static std::unique_ptr<geos::geom::Geometry> readEnvelope(
      velox::common::InputByteStream& input);

  static std::unique_ptr<geos::geom::Geometry> readGeometryCollection(
      velox::common::InputByteStream& input,
      size_t size);
};

} // namespace facebook::velox::common::geospatial
