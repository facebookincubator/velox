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
// Based off of https://threadlocalmutex.com/?p=126

#pragma once

#include <cstdint>
#include <limits>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

class HilbertIndex {
 public:
  /// Construct a Hilber index. If a min value is greater than the max value,
  /// this will panic.
  HilbertIndex(float minX, float minY, float maxX, float maxY)
      : minX_(minX), minY_(minY), maxX_(maxX), maxY_(maxY) {
    VELOX_CHECK(minX_ <= maxX_);
    VELOX_CHECK(minY_ <= maxY_);

    float deltaX = maxX_ - minX_;
    // Subnormals cause numerical instability.
    // NOLINTNEXTLINE(facebook-hte-FloatingPointMin)
    if (deltaX < std::numeric_limits<float>::min()) {
      xScale_ = 0;
    } else {
      xScale_ = kHilbertMax / deltaX;
    }

    float deltaY = maxY_ - minY_;
    // Subnormals cause numerical instability.
    // NOLINTNEXTLINE(facebook-hte-FloatingPointMin)
    if (deltaY < std::numeric_limits<float>::min()) {
      yScale_ = 0;
    } else {
      yScale_ = kHilbertMax / deltaY;
    }
  }

  uint32_t inline indexOf(float x, float y) const {
    if (!(x >= minX_ && x <= maxX_ && y >= minY_ && y <= maxY_)) {
      // Put things outside the bounds at the end of the Hilbert curve.
      // Negation handles NaNs
      return std::numeric_limits<uint32_t>::max();
    }

    float maxFloat = static_cast<float>(std::numeric_limits<uint32_t>::max());

    uint32_t xInt = static_cast<uint32_t>(
        std::clamp(xScale_ * (x - minX_), 0.0f, maxFloat));
    uint32_t yInt = static_cast<uint32_t>(
        std::clamp(yScale_ * (y - minY_), 0.0f, maxFloat));
    return discreteIndexOf(xInt, yInt);
  }

 private:
  static inline uint32_t interleave(uint32_t x) {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
  }

  static inline uint32_t discreteIndexOf(uint32_t x, uint32_t y) {
    uint32_t A, B, C, D;

    // Initial prefix scan round, prime with x and y
    {
      uint32_t a = x ^ y;
      uint32_t b = 0xFFFF ^ a;
      uint32_t c = 0xFFFF ^ (x | y);
      uint32_t d = x & (y ^ 0xFFFF);

      A = a | (b >> 1);
      B = (a >> 1) ^ a;

      C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
      D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;
    }

    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      A = ((a & (a >> 2)) ^ (b & (b >> 2)));
      B = ((a & (b >> 2)) ^ (b & ((a ^ b) >> 2)));

      C ^= ((a & (c >> 2)) ^ (b & (d >> 2)));
      D ^= ((b & (c >> 2)) ^ ((a ^ b) & (d >> 2)));
    }

    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      A = ((a & (a >> 4)) ^ (b & (b >> 4)));
      B = ((a & (b >> 4)) ^ (b & ((a ^ b) >> 4)));

      C ^= ((a & (c >> 4)) ^ (b & (d >> 4)));
      D ^= ((b & (c >> 4)) ^ ((a ^ b) & (d >> 4)));
    }

    // Final round and projection
    {
      uint32_t a = A;
      uint32_t b = B;
      uint32_t c = C;
      uint32_t d = D;

      C ^= ((a & (c >> 8)) ^ (b & (d >> 8)));
      D ^= ((b & (c >> 8)) ^ ((a ^ b) & (d >> 8)));
    }

    // Undo transformation prefix scan
    uint32_t a = C ^ (C >> 1);
    uint32_t b = D ^ (D >> 1);

    // Recover index bits
    uint32_t i0 = x ^ y;
    uint32_t i1 = b | (0xFFFF ^ (i0 | a));

    return (interleave(i1) << 1) | interleave(i0);
  }

  static const int8_t kHilbertBits = 16;
  static constexpr float kHilbertMax = (1 << kHilbertBits) - 1;

  const float minX_;
  const float minY_;
  const float maxX_;
  const float maxY_;
  float xScale_;
  float yScale_;
};

} // namespace facebook::velox::exec
