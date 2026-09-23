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

#ifdef VELOX_ENABLE_GEO
#include "velox/functions/prestosql/GooglePolylineFunctions.h"

namespace facebook::velox::functions {

namespace detail {

// Encodes a delta value using variable-length encoding per the Google Polyline
// algorithm specification.
// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
void encodeNextDelta(int64_t delta, std::string& result) {
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

// Decodes the next delta value from an encoded polyline string per the Google
// Polyline algorithm specification.
// https://developers.google.com/maps/documentation/utilities/polylinealgorithm
Status
decodeNextDelta(int64_t& result, const StringView& encoded, size_t& index) {
  int64_t value = 0;
  int shift = 0;
  int64_t b;

  do {
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

Status validateAndComputePrecision(
    int64_t precisionExponent,
    double& precision) {
  if (precisionExponent < kMinimumPrecisionExponent) {
    return Status::UserError(
        "Polyline precision must be greater or equal to {}",
        kMinimumPrecisionExponent);
  }

  if (precisionExponent > kMaximumPrecisionExponent) {
    return Status::UserError(
        "Polyline precision exponent must not exceed {}",
        kMaximumPrecisionExponent);
  }

  precision = (precisionExponent == kDefaultPrecisionExponent)
      ? kDefaultPrecision
      : std::pow(10.0, static_cast<double>(precisionExponent));

  return Status::OK();
}

} // namespace detail

} // namespace facebook::velox::functions

#endif // VELOX_ENABLE_GEO
