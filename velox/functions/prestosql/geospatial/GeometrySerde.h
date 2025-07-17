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

#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>

#include "velox/common/base/IOUtils.h"
#include "velox/functions/prestosql/geospatial/GeometryUtils.h"
#include "velox/type/StringView.h"

namespace facebook::velox::functions::geospatial {

enum class GeometrySerializationType : uint8_t {
  POINT = 0,
  MULTI_POINT = 1,
  LINE_STRING = 2,
  MULTI_LINE_STRING = 3,
  POLYGON = 4,
  MULTI_POLYGON = 5,
  GEOMETRY_COLLECTION = 6,
  ENVELOPE = 7
};

enum class EsriShapeType : int {
  POINT = 1,
  POLYLINE = 3,
  POLYGON = 5,
  MULTI_POINT = 8
};

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
  VarbinaryWriter(StringWriter& stringWriter) : stringWriter_(stringWriter) {}
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
            "Unrecognized geometry type: {}", static_cast<int>(geometryType));
        break;
    }
  }

 private:
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
    int numParts;
    int numPoints = geometry.getNumPoints();

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

    writer.write(numParts);
    writer.write(numPoints);

    int partIndex = 0;
    for (int i = 0; i < numParts; ++i) {
      writer.write(partIndex);
      partIndex += geometry.getGeometryN(i)->getNumPoints();
    }

    if (multiType) {
      for (int i = 0; i < numParts; ++i) {
        const auto* part = geometry.getGeometryN(i);
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
    int numGeometries = geometry.getNumGeometries();
    int numParts = 0;
    int numPoints = geometry.getNumPoints();

    for (int i = 0; i < numGeometries; i++) {
      auto polygon =
          dynamic_cast<const geos::geom::Polygon*>(geometry.getGeometryN(i));
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

    writer.write(numParts);
    writer.write(numPoints);

    if (numParts == 0) {
      return;
    }

    std::vector<int> partIndexes(numParts);
    std::vector<bool> shellPart(numParts);

    int currentPart = 0;
    int currentPoint = 0;
    for (int i = 0; i < numGeometries; i++) {
      const geos::geom::Polygon* polygon =
          dynamic_cast<const geos::geom::Polygon*>(geometry.getGeometryN(i));

      partIndexes[currentPart] = currentPoint;
      shellPart[currentPart] = true;
      currentPart++;
      currentPoint += polygon->getExteriorRing()->getNumPoints();

      int holesCount = polygon->getNumInteriorRing();
      for (int holeIndex = 0; holeIndex < holesCount; holeIndex++) {
        partIndexes[currentPart] = currentPoint;
        shellPart[currentPart] = false;
        currentPart++;
        currentPoint += polygon->getInteriorRingN(holeIndex)->getNumPoints();
      }
    }

    for (int partIndex : partIndexes) {
      writer.write(partIndex);
    }

    auto coordinates = geometry.getCoordinates();
    canonicalizePolygonCoordinates(coordinates, partIndexes, shellPart);
    writeCoordinates(coordinates, writer);
  }

  template <typename T>
  static void writeGeometryCollection(
      const geos::geom::Geometry& collection,
      VarbinaryWriter<T>& output) {
    output.write(
        static_cast<uint8_t>(GeometrySerializationType::GEOMETRY_COLLECTION));

    for (size_t geometryIndex = 0;
         geometryIndex < collection.getNumGeometries();
         ++geometryIndex) {
      auto* geometry = collection.getGeometryN(geometryIndex);
      // Use a temporary buffer to serialize the geometry and calculate its
      // length
      std::string tempBuffer;
      VarbinaryWriter tempOutput(tempBuffer);

      serialize(*geometry, tempOutput);

      int32_t length = static_cast<int32_t>(tempBuffer.size());
      output.write(length);
      output.write(tempBuffer.data(), tempBuffer.size());
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
    return deserialize(inputStream, geometryString.size());
  }

  static const std::unique_ptr<geos::geom::Envelope> deserializeEnvelope(
      const StringView& geometry);

 private:
  static std::unique_ptr<geos::geom::Geometry> deserialize(
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
      int count);

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

} // namespace facebook::velox::functions::geospatial
