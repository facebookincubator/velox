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
// Google Polyline encoding constants
constexpr int32_t kAsciiOffset = 63; // 0x3f - ASCII offset for printable chars
constexpr int32_t kLumpMask = 0x20; // Continuation bit for multi-lump values
constexpr int32_t kDataMask = 0x1f; // 5-bit data mask per lump
constexpr int32_t kLumpSize = 5; // 5-bit lumps

/// Encode a single delta value using Google Polyline encoding.
/// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
/// Algorithm:
/// 1. Convert signed to unsigned using ZigZag encoding:
///    - Left shift by 1 bit
///    - If negative, invert all bits
///    This ensures efficient encoding of both positive and negative deltas.
/// 2. Break into 5-bit lumps
/// 3. Set continuation bit (0x20) for all lumps except the last
/// 4. Add ASCII offset (63) to make printable characters
inline void encodeNextDelta(int64_t delta, std::string& result) {
  int64_t unsignedDelta = delta << 1;
  if (delta < 0) {
    unsignedDelta = ~unsignedDelta;
  }

  while (unsignedDelta >= kLumpMask) {
    int64_t nextLump = (kLumpMask | (unsignedDelta & kDataMask)) + kAsciiOffset;
    result.push_back(static_cast<char>(nextLump));
    unsignedDelta >>= kLumpSize;
  }

  result.push_back(static_cast<char>(unsignedDelta + kAsciiOffset));
}

/// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
/// Algorithm:
/// 1. Read 5 bit lumps until continuation bit is not set.
/// 2. Combine lumps into unsigned value.
/// 3. Convert unsigned to signed.
inline Status
decodeNextDelta(int64_t& result, const StringView& encoded, size_t& index) {
  int64_t value = 0;
  int shift = 0;
  int64_t b;

  do {
    // Check for unexpected end of input
    if (index >= encoded.size()) {
      return Status::UserError(
          "Invalid polyline encoding: unexpected end of input");
    }

    b = static_cast<int64_t>(
            static_cast<unsigned char>(encoded.data()[index++])) -
        kAsciiOffset;
    value |= (b & kDataMask) << shift;
    shift += kLumpSize;
  } while (b >= kLumpMask);

  result = ((value & 1) != 0) ? ~(value >> 1) : (value >> 1);
  return Status::OK();
}
} // namespace

// Default and min/max precision values for Google Polyline encoding.
constexpr int64_t kDefaultPrecisionExponent = 5;
constexpr int64_t kMinimumPrecisionExponent = 1;
constexpr int64_t kMaximumPrecisionExponent = 16;
constexpr double kDefaultPrecision = 100000.0; // 10^5

/// Google Polyline Encode Function
/// Encodes an array of Point geometries into a Google Polyline encoded string.
/// Input: array(Geometry) :Array of Point geometries.
/// Output: varchar :Google Polyline encoded string.
/// Optional: precision exponent.
template <typename T>
struct GooglePolylineEncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  // Encode array of Point geometries with default precision.
  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Array<Geometry>>& points) {
    return callImpl(result, points, kDefaultPrecisionExponent);
  }

  // Encode array of Point geometries with custom precision.
  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
    return callImpl(result, points, precisionExponent);
  }

 private:
  Status callImpl(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
    if (precisionExponent < kMinimumPrecisionExponent) {
      return Status::UserError(
          fmt::format(
              "Polyline precision must be greater or equal to {}",
              kMinimumPrecisionExponent));
    }

    if (precisionExponent > kMaximumPrecisionExponent) {
      return Status::UserError(
          fmt::format(
              "Polyline precision exponent must not exceed {}",
              kMaximumPrecisionExponent));
    }

    double precision = (precisionExponent == kDefaultPrecisionExponent)
        ? kDefaultPrecision
        : std::pow(10.0, static_cast<double>(precisionExponent));

    std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
        common::geospatial::GeometryDeserializer::deserializePointsToCoordinate<
            Geometry>(points, "google_polyline_encode", false);

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
    return Status::OK();
  }
};

/// Google Polyline Decode Function.
/// Decodes a Google Polyline encoded string into an array of Point geometries.
/// Input: varchar :Google Polyline encoded string.
/// Output: array(Geometry) :Array of Point geometries.
/// Optional: precision exponent.
template <typename T>
struct GooglePolylineDecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  GooglePolylineDecodeFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }
  // Decode polyline to array of Point geometries with default precision.
  FOLLY_ALWAYS_INLINE Status
  call(out_type<Array<Geometry>>& result, const arg_type<Varchar>& encoded) {
    return callImpl(result, encoded, kDefaultPrecisionExponent);
  }

  // Decode polyline to array of Point geometries with custom precision.
  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent) {
    return callImpl(result, encoded, precisionExponent);
  }

 private:
  Status callImpl(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent) {
    if (precisionExponent < kMinimumPrecisionExponent) {
      return Status::UserError(
          fmt::format(
              "Polyline precision must be greater or equal to {}",
              kMinimumPrecisionExponent));
    }

    if (precisionExponent > kMaximumPrecisionExponent) {
      return Status::UserError(
          fmt::format(
              "Polyline precision exponent must not exceed {}",
              kMaximumPrecisionExponent));
    }

    double precision = (precisionExponent == kDefaultPrecisionExponent)
        ? kDefaultPrecision
        : std::pow(10.0, static_cast<double>(precisionExponent));

    size_t index = 0;
    int64_t x = 0;
    int64_t y = 0;

    while (index < encoded.size()) {
      int64_t deltaX, deltaY;

      // Check Status return from decodeNextDelta
      auto status = decodeNextDelta(deltaX, encoded, index);
      if (!status.ok()) {
        return status;
      }

      status = decodeNextDelta(deltaY, encoded, index);
      if (!status.ok()) {
        return status;
      }

      x += deltaX;
      y += deltaY;

      double coordX = static_cast<double>(x) / precision;
      double coordY = static_cast<double>(y) / precision;

      auto point =
          factory_->createPoint(geos::geom::Coordinate(coordX, coordY));

      auto& itemWriter = result.add_item();
      common::geospatial::GeometrySerializer::serialize(*point, itemWriter);
    }
    return Status::OK();
  }
  geos::geom::GeometryFactory::Ptr factory_;
};

} // namespace facebook::velox::functions
