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

#include <cmath>
#include <string>

#include <geos/geom/Coordinate.h>
#include <geos/geom/Point.h>

#include "velox/common/geospatial/GeometrySerde.h"
#include "velox/functions/Macros.h"
#include "velox/functions/prestosql/types/GeometryType.h"

namespace facebook::velox::functions {

namespace {

/// Encode a single delta value using Google Polyline encoding
/// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
/// Algorithm:
/// 1. Convert signed to unsigned
/// 2. Encode in 5 bit lumps
/// 3. Add 63 to each lump to make it printable ASCII
inline void encodeNextDelta(int64_t delta, std::string& result) {
  int64_t unsignedDelta = delta << 1;
  if (delta < 0) {
    unsignedDelta = ~unsignedDelta;
  }

  while (unsignedDelta >= 0x20) {
    int64_t nextChunk = (0x20 | (unsignedDelta & 0x1f)) + 63;
    result.push_back(static_cast<char>(nextChunk));
    unsignedDelta >>= 5;
  }

  result.push_back(static_cast<char>(unsignedDelta + 63));
}

/// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
/// Algorithm:
/// 1. Read 5 bit lumps until continuation bit is not set
/// 2. Combine lumps into unsigned value
/// 3. Convert unsigned to signed
inline int64_t decodeNextDelta(const std::string& encoded, size_t& index) {
  int64_t result = 0;
  int shift = 0;
  int64_t b;

  do {
    VELOX_USER_CHECK_LT(
        index,
        encoded.length(),
        "Invalid polyline encoding: unexpected end of input");

    b = static_cast<int64_t>(static_cast<unsigned char>(encoded[index++])) - 63;
    result |= (b & 0x1f) << shift;
    shift += 5;
  } while (b >= 0x20);

  return ((result & 1) != 0) ? ~(result >> 1) : (result >> 1);
}

} // namespace

// Default and minimum precision values for Google Polyline encoding
constexpr int64_t kDefaultPrecisionExponent = 5;
constexpr int64_t kMinimumPrecisionExponent = 1;

/// Google Polyline Encode Function
/// Encodes an array of Point geometries into a Google Polyline encoded string.
/// Input: array(Geometry) :Array of Point geometries
/// Output: varchar :Google Polyline encoded string
/// Optional: precision exponent
template <typename T>
struct GooglePolylineEncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Encode array of Point geometries with default precision
  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points) {
    callImpl(result, points, kDefaultPrecisionExponent);
  }

  // Encode array of Point geometries with custom precision
  FOLLY_ALWAYS_INLINE void call(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
    callImpl(result, points, precisionExponent);
  }

  private:
  void callImpl(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
  VELOX_USER_CHECK_GE(
      precisionExponent,
      kMinimumPrecisionExponent,
      "Minimum Polyline precision exponent should be {}",
      kMinimumPrecisionExponent);

  double precision = std::pow(10.0, static_cast<double>(precisionExponent));

  std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
      common::geospatial::GeometryDeserializer::deserializePointsToCoordinate
                           <Geometry>(points, "google_polyline_encode", false);

  std::string encoded;
  int64_t prevX = 0;
  int64_t prevY = 0;

  for (size_t i = 0; i < coords->size(); ++i) {
    int64_t x = std::llround(coords->getX(i) * precision);
    int64_t y = std::llround(coords->getY(i) * precision);

    if (i == 0) {
      encodeNextDelta(x, encoded);
      encodeNextDelta(y, encoded);
    } else {
      encodeNextDelta(x - prevX, encoded);
      encodeNextDelta(y - prevY, encoded);
    }
    prevX = x;
    prevY = y;
  }

  result.resize(encoded.size());
  std::memcpy(result.data(), encoded.data(), encoded.size());
}
};

/// Google Polyline Decode Function
/// Decodes a Google Polyline encoded string into an array of Point geometries.
/// Input: varchar :Google Polyline encoded string
/// Output: array(Geometry) :Array of Point geometries
/// Optional: precision exponent
template <typename T>
struct GooglePolylineDecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  GooglePolylineDecodeFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }
  // Decode polyline to array of Point geometries with default precision
  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded) {
    callImpl(result, encoded, kDefaultPrecisionExponent);
  }

  // Decode polyline to array of Point geometries with custom precision
  FOLLY_ALWAYS_INLINE void call(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent) {
    callImpl(result, encoded, precisionExponent);
  }

  private:
  void callImpl(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent) {
    VELOX_USER_CHECK_GE(
        precisionExponent,
        kMinimumPrecisionExponent,
        "Polyline precision exponent must be at least {}",
        kMinimumPrecisionExponent);

    double precision = std::pow(10.0, static_cast<double>(precisionExponent));
    std::string encodedStr(encoded.data(), encoded.size());
    size_t index = 0;

    int64_t x = 0;
    int64_t y = 0;

    while (index < encodedStr.length()) {
      int64_t deltaX = decodeNextDelta(encodedStr, index);
      int64_t deltaY = decodeNextDelta(encodedStr, index);

      x += deltaX;
      y += deltaY;

      double coordX = static_cast<double>(x) / precision;
      double coordY = static_cast<double>(y) / precision;

      auto point = factory_->createPoint(geos::geom::Coordinate(coordX, coordY));

      auto& itemWriter = result.add_item();
      common::geospatial::GeometrySerializer::serialize(*point, itemWriter);
    }
  }
  geos::geom::GeometryFactory::Ptr factory_;
};

} // namespace facebook::velox::functions
