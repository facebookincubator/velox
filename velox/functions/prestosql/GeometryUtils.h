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

#include <geos/geom/CoordinateSequenceFactory.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/LineString.h>
#include <geos/geom/Point.h>

#include <velox/common/base/Exceptions.h>

namespace facebook::velox::functions {

class GeosGeometrySerde {
 public:
  static void serialize(
      const geos::geom::Geometry* geometry,
      exec::StringWriter<>& stringWriter) {
    SliceOutput sliceOutput(stringWriter);
    switch (geometry->getGeometryTypeId()) {
      case geos::geom::GEOS_POINT:
        writePoint(geometry, sliceOutput);
        break;
      case geos::geom::GEOS_MULTIPOINT:
        writeMultiPoint(geometry, sliceOutput);
        break;
      case geos::geom::GEOS_LINESTRING:
      case geos::geom::GEOS_LINEARRING:
        writePolyline(geometry, sliceOutput, false);
        break;
      case geos::geom::GEOS_MULTILINESTRING:
        writePolyline(geometry, sliceOutput, true);
        break;
      default:
        VELOX_FAIL("Cannot serialize geometry type");
        break;
    }
  }

  static std::unique_ptr<geos::geom::Geometry> deserialize(
      const StringView& geometryString) {
    SliceInput sliceInput(geometryString.data(), geometryString.size());
    auto geometryType =
        static_cast<GeometrySerializationType>(sliceInput.readByte());
    switch (geometryType) {
      case GeometrySerializationType::POINT:
        return readPoint(sliceInput);
      case GeometrySerializationType::MULTI_POINT:
        return readMultiPoint(sliceInput);
      case GeometrySerializationType::LINE_STRING:
        return readPolyline(sliceInput, false);
      case GeometrySerializationType::MULTI_LINE_STRING:
        return readPolyline(sliceInput, true);
      default:
        VELOX_FAIL("Cannot deserialize geometry type");
        break;
    }
    return nullptr;
  }

 private:
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

  class SliceInput {
   public:
    SliceInput(const void* rawData, size_t dataSize)
        : data_(reinterpret_cast<const uint8_t*>(rawData)),
          size_(dataSize),
          position_(0) {}

    uint8_t readByte() {
      return read<uint8_t>();
    }

    int16_t readShort() {
      return read<int16_t>();
    }

    int32_t readInt() {
      return read<int32_t>();
    }

    int32_t readLong() {
      return read<int64_t>();
    }

    float readFloat() {
      return read<float>();
    }

    double readDouble() {
      return read<double>();
    }

    size_t remaining() const {
      return size_ - position_;
    }

    void skip(size_t size) {
      if (position_ + size > size_) {
        VELOX_USER_FAIL("SliceInput read exceeds buffer size.");
      }
      position_ += size;
    }

   private:
    template <typename T>
    T read() {
      if (position_ + sizeof(T) > size_) {
        VELOX_USER_FAIL("SliceInput read exceeds buffer size.");
      }
      T value;
      std::memcpy(&value, &data_[position_], sizeof(T));
      position_ += sizeof(T);
      return value;
    }

    const uint8_t* data_;
    size_t size_;
    size_t position_;
  };

  class SliceOutput {
   public:
    SliceOutput(exec::StringWriter<>& stringWriter)
        : stringWriter_(stringWriter) {}

    SliceOutput() = delete;

    void writeByte(uint8_t value) {
      write<uint8_t>(value);
    }

    void writeShort(int16_t value) {
      write<int16_t>(value);
    }

    void writeInt(int32_t value) {
      write<int32_t>(value);
    }

    void writeLong(int64_t value) {
      write<int64_t>(value);
    }

    void writeFloat(float value) {
      write<float>(value);
    }

    void writeDouble(double value) {
      write<double>(value);
    }

   private:
    template <typename T>
    void write(const T& value) {
      stringWriter_.append(
          std::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
    }

    exec::StringWriter<>& stringWriter_;
  };

  static void writeEnvelope(
      const geos::geom::Geometry* geometry,
      SliceOutput& output) {
    auto envelope = geometry->getEnvelopeInternal();
    output.writeDouble(envelope->getMinX());
    output.writeDouble(envelope->getMinY());
    output.writeDouble(envelope->getMaxX());
    output.writeDouble(envelope->getMaxY());
  }

  static void writeCoordinates(
      const geos::geom::CoordinateSequence* coords,
      SliceOutput& output) {
    for (size_t i = 0; i < coords->size(); ++i) {
      output.writeDouble(coords->getX(i));
      output.writeDouble(coords->getY(i));
    }
  }

  static void writePoint(
      const geos::geom::Geometry* point,
      SliceOutput& output) {
    output.writeByte(static_cast<uint8_t>(GeometrySerializationType::POINT));
    if (!point->isEmpty()) {
      writeCoordinates(point->getCoordinates().get(), output);
    } else {
      output.writeDouble(std::nan(""));
      output.writeDouble(std::nan(""));
    }
  }

  static void writeMultiPoint(
      const geos::geom::Geometry* geometry,
      SliceOutput& output) {
    output.writeByte(
        static_cast<uint8_t>(GeometrySerializationType::MULTI_POINT));
    output.writeInt(static_cast<int32_t>(EsriShapeType::MULTI_POINT));
    writeEnvelope(geometry, output);
    output.writeInt(geometry->getNumPoints());
    writeCoordinates(geometry->getCoordinates().get(), output);
  }

  static void writePolyline(
      const geos::geom::Geometry* geometry,
      SliceOutput& output,
      bool multiType) {
    int numParts;
    int numPoints = geometry->getNumPoints();

    if (multiType) {
      numParts = geometry->getNumGeometries();
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::MULTI_LINE_STRING));
    } else {
      numParts = (numPoints > 0) ? 1 : 0;
      output.writeByte(
          static_cast<uint8_t>(GeometrySerializationType::LINE_STRING));
    }

    output.writeInt(static_cast<int32_t>(EsriShapeType::POLYLINE));

    writeEnvelope(geometry, output);

    output.writeInt(numParts);
    output.writeInt(numPoints);

    int partIndex = 0;
    for (int i = 0; i < numParts; ++i) {
      output.writeInt(partIndex);
      partIndex += geometry->getGeometryN(i)->getNumPoints();
    }

    if (multiType) {
      for (int i = 0; i < numParts; ++i) {
        const auto* part = geometry->getGeometryN(i);
        writeCoordinates(part->getCoordinates().get(), output);
      }
    } else {
      writeCoordinates(geometry->getCoordinates().get(), output);
    }
  }

  inline static auto GEOMETRY_FACTORY = geos::geom::GeometryFactory::create();

  static void skipEsriType(SliceInput& input) {
    input.skip(4); // Esri type is an integer
  }

  static void skipEnvelope(SliceInput& input) {
    input.skip(4 * 8); // Envelopes are 4 doubles (minX, minY, maxX, maxY)
  }

  static geos::geom::Coordinate readCoordinate(SliceInput& input) {
    double x = input.readDouble();
    double y = input.readDouble();
    return {x, y};
  }

  static std::unique_ptr<geos::geom::CoordinateSequence> readCoordinates(
      SliceInput& input,
      int count) {
    auto coords =
        GEOMETRY_FACTORY->getCoordinateSequenceFactory()->create(count, 2);
    for (int i = 0; i < count; ++i) {
      coords->setAt(readCoordinate(input), i);
    }
    return coords;
  }

  static std::unique_ptr<geos::geom::Point> readPoint(SliceInput& input) {
    geos::geom::Coordinate coordinate = readCoordinate(input);
    if (std::isnan(coordinate.x) || std::isnan(coordinate.y)) {
      return GEOMETRY_FACTORY->createPoint();
    }
    return std::unique_ptr<geos::geom::Point>(
        GEOMETRY_FACTORY->createPoint(coordinate));
  }

  static std::unique_ptr<geos::geom::Geometry> readMultiPoint(
      SliceInput& input) {
    skipEsriType(input);
    skipEnvelope(input);
    int pointCount = input.readInt();
    auto coords = readCoordinates(input, pointCount);
    std::vector<std::unique_ptr<geos::geom::Point>> points;
    for (size_t i = 0; i < coords->size(); ++i) {
      points.push_back(
          std::unique_ptr<geos::geom::Point>(GEOMETRY_FACTORY->createPoint(
              geos::geom::Coordinate(coords->getX(i), coords->getY(i)))));
    }
    return GEOMETRY_FACTORY->createMultiPoint(std::move(points));
  }

  static std::unique_ptr<geos::geom::Geometry> readPolyline(
      SliceInput& input,
      bool multiType) {
    skipEsriType(input);
    skipEnvelope(input);
    int partCount = input.readInt();

    if (partCount == 0) {
      if (multiType) {
        return GEOMETRY_FACTORY->createMultiLineString();
      }
      return GEOMETRY_FACTORY->createLineString();
    }

    int pointCount = input.readInt();
    std::vector<int> startIndexes(partCount);

    for (int i = 0; i < partCount; ++i) {
      startIndexes[i] = input.readInt();
    }

    std::vector<int> partLengths(partCount);
    if (partCount > 1) {
      partLengths[0] = startIndexes[1];
      for (int i = 1; i < partCount - 1; ++i) {
        partLengths[i] = startIndexes[i + 1] - startIndexes[i];
      }
    }
    partLengths[partCount - 1] = pointCount - startIndexes[partCount - 1];

    std::vector<std::unique_ptr<geos::geom::LineString>> lineStrings;
    for (int i = 0; i < partCount; ++i) {
      lineStrings.push_back(GEOMETRY_FACTORY->createLineString(
          readCoordinates(input, partLengths[i])));
    }

    if (multiType) {
      return GEOMETRY_FACTORY->createMultiLineString(std::move(lineStrings));
    }

    if (lineStrings.size() != 1) {
      VELOX_USER_FAIL(
          "Expected a single LineString for non-multitype polyline.");
    }

    return std::move(lineStrings[0]);
  }
};
} // namespace facebook::velox::functions
