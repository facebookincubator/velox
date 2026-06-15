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

namespace detail {

constexpr int32_t kAsciiOffset = 63;
constexpr int32_t kLumpMask = 0x20;
constexpr int32_t kDataMask = 0x1f;
constexpr int32_t kLumpSize = 5;

void encodeNextDelta(int64_t delta, std::string& result);

Status
decodeNextDelta(int64_t& result, const StringView& encoded, size_t& index);

Status validateAndComputePrecision(
    int64_t precisionExponent,
    double& precision);

} // namespace detail

constexpr int64_t kDefaultPrecisionExponent = 5;
constexpr int64_t kMinimumPrecisionExponent = 1;
constexpr int64_t kMaximumPrecisionExponent = 16;
constexpr double kDefaultPrecision = 100000.0;

template <typename TExec>
struct GooglePolylineEncodeFunctionImpl {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE static Status callImpl(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
    double precision;
    auto status =
        detail::validateAndComputePrecision(precisionExponent, precision);
    if (!status.ok()) {
      return status;
    }

    std::unique_ptr<geos::geom::CoordinateArraySequence> coords =
        common::geospatial::GeometryDeserializer::deserializePointsToCoordinate<
            Geometry>(points, "google_polyline_encode", false);

    std::string encoded;
    int64_t prevX = 0;
    int64_t prevY = 0;

    for (size_t i = 0; i < coords->size(); ++i) {
      int64_t x = std::llround(coords->getX(i) * precision);
      int64_t y = std::llround(coords->getY(i) * precision);

      detail::encodeNextDelta(x - prevX, encoded);
      detail::encodeNextDelta(y - prevY, encoded);

      prevX = x;
      prevY = y;
    }

    result.append(encoded);
    return Status::OK();
  }
};

template <typename TExec>
struct GooglePolylineDecodeFunctionImpl {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  FOLLY_ALWAYS_INLINE static Status callImpl(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent,
      geos::geom::GeometryFactory::Ptr& factory) {
    double precision;
    auto status =
        detail::validateAndComputePrecision(precisionExponent, precision);
    if (!status.ok()) {
      return status;
    }

    size_t index = 0;
    int64_t x = 0;
    int64_t y = 0;

    while (index < encoded.size()) {
      int64_t deltaX, deltaY;

      status = detail::decodeNextDelta(deltaX, encoded, index);
      if (!status.ok()) {
        return status;
      }

      status = detail::decodeNextDelta(deltaY, encoded, index);
      if (!status.ok()) {
        return status;
      }

      x += deltaX;
      y += deltaY;

      double coordX = static_cast<double>(x) / precision;
      double coordY = static_cast<double>(y) / precision;

      std::unique_ptr<geos::geom::Point> point(
          factory->createPoint(geos::geom::Coordinate(coordX, coordY)));

      auto& itemWriter = result.add_item();
      common::geospatial::GeometrySerializer::serialize(*point, itemWriter);
    }
    return Status::OK();
  }
};

template <typename T>
struct GooglePolylineEncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Varchar>& result, const arg_type<Array<Geometry>>& points) {
    return GooglePolylineEncodeFunctionImpl<T>::callImpl(
        result, points, kDefaultPrecisionExponent);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Varchar>& result,
      const arg_type<Array<Geometry>>& points,
      int64_t precisionExponent) {
    return GooglePolylineEncodeFunctionImpl<T>::callImpl(
        result, points, precisionExponent);
  }
};

template <typename T>
struct GooglePolylineDecodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  GooglePolylineDecodeFunction() {
    factory_ = geos::geom::GeometryFactory::create();
  }

  FOLLY_ALWAYS_INLINE Status
  call(out_type<Array<Geometry>>& result, const arg_type<Varchar>& encoded) {
    return GooglePolylineDecodeFunctionImpl<T>::callImpl(
        result, encoded, kDefaultPrecisionExponent, factory_);
  }

  FOLLY_ALWAYS_INLINE Status call(
      out_type<Array<Geometry>>& result,
      const arg_type<Varchar>& encoded,
      int64_t precisionExponent) {
    return GooglePolylineDecodeFunctionImpl<T>::callImpl(
        result, encoded, precisionExponent, factory_);
  }

 private:
  geos::geom::GeometryFactory::Ptr factory_;
};

} // namespace facebook::velox::functions
