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

#include <cstdint>

// ZigZag encoding maps signed integers to unsigned integers so that values
// with small absolute value have small encoded values, making them efficient
// for varint encoding.
//
//   zigzagEncode32(0)  == 0
//   zigzagEncode32(-1) == 1
//   zigzagEncode32(1)  == 2
//   zigzagEncode32(-2) == 3
//   ...
//
// In general:
//   if x >= 0, zigzagEncode32(x) == 2*x
//   if x <  0, zigzagEncode32(x) == -2*x - 1

namespace facebook::nimble::zigzag {

inline constexpr uint32_t zigzagEncode32(int32_t val) noexcept {
  return static_cast<uint32_t>((val << 1) ^ (val >> 31));
}

inline constexpr int32_t zigzagDecode32(uint32_t val) noexcept {
  return static_cast<int32_t>((val >> 1) ^ -(val & 1));
}

inline constexpr uint64_t zigzagEncode64(int64_t val) noexcept {
  return static_cast<uint64_t>((val << 1) ^ (val >> 63));
}

inline constexpr int64_t zigzagDecode64(uint64_t val) noexcept {
  return static_cast<int64_t>((val >> 1) ^ -(val & 1));
}

} // namespace facebook::nimble::zigzag
