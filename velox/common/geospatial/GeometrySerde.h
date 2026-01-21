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

#include <folly/Likely.h>
#include <geos/geom/CoordinateArraySequence.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>

#include "velox/common/base/IOUtils.h"
#include "velox/common/geospatial/GeometryConstants.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/expression/StringWriter.h"
#include "velox/type/StringView.h"

namespace facebook::velox::common::geospatial {

class GeometrySerializer {
 public:
  /// Serialize geometry into Velox's internal format.  Do not call this within
  /// GEOS_TRY macro: it will catch the exceptions that need to bubble up.
  static void serialize(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer) {
    writeGeometry(geometry, writer);
  }

  static void serializeEnvelope(
      double xMin,
      double yMin,
      double xMax,
      double yMax,
      exec::StringWriter& writer);

  static void serializeEnvelope(
      geos::geom::Envelope& envelope,
      exec::StringWriter& writer);

  /// Determines if a ring of coordinates (from `start` to `end`) is oriented
  /// clockwise.
  static bool isClockwise(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      size_t start,
      size_t end);

 private:
  /// Append a trivially copyable value to the writer as a binary string.
  template <typename T>
  static void appendBytes(exec::StringWriter& writer, const T& value) {
    static_assert(
        std::is_trivially_copyable_v<T>, "T must be trivially copyable");
    writer.append(
        std::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
  }

  /// Reverses the order of coordinates in the sequence between `start` and
  /// `end`
  static void reverse(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      size_t start,
      size_t end);

  /// Ensures that a polygon ring has the canonical orientation:
  /// - Exterior rings (shells) must be clockwise.
  /// - Interior rings (holes) must be counter-clockwise.
  static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      size_t start,
      size_t end,
      bool isShell);

  /// Applies `canonicalizePolygonCoordinates` to all rings in a polygon.
  static void canonicalizePolygonCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coordinates,
      const std::vector<size_t>& partIndexes,
      const std::vector<bool>& shellPart);
  static void writeGeometry(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer);

  static void writeEnvelope(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer);

  static void writeCoordinates(
      const std::unique_ptr<geos::geom::CoordinateSequence>& coords,
      exec::StringWriter& writer);

  static void writePoint(
      const geos::geom::Geometry& point,
      exec::StringWriter& writer);

  static void writeMultiPoint(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer);

  static void writePolyline(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer,
      bool multiType);

  static void writePolygon(
      const geos::geom::Geometry& geometry,
      exec::StringWriter& writer,
      bool multiType);

  static void writeGeometryCollection(
      const geos::geom::Geometry& collection,
      exec::StringWriter& writer);
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
