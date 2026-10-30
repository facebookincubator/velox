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
#include <type_traits>

#include "velox/dwio/nimble/common/Types.h"

// Maps an integer of logical type T, held in its unsigned physical type, to a
// uint64_t whose unsigned order is the logical order of T. Encodings that
// subtract or compare values, such as Elias-Fano, work on this form.
//
// A signed value has its sign bit flipped. For int8_t:
//
//   toOrdered<int8_t>(-128) == 0x00
//   toOrdered<int8_t>(-1)   == 0x7F
//   toOrdered<int8_t>(0)    == 0x80
//   toOrdered<int8_t>(127)  == 0xFF
//
// An unsigned value is only widened.

namespace facebook::nimble {

/// Returns 'value', of logical type T, in its order-preserving form.
template <typename T>
inline constexpr uint64_t toOrdered(
    typename TypeTraits<T>::physicalType value) noexcept {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  static_assert(std::is_unsigned_v<PhysicalType>);
  if constexpr (std::is_signed_v<T>) {
    return static_cast<uint64_t>(value) ^
        (uint64_t{1} << (sizeof(PhysicalType) * 8 - 1));
  } else {
    return static_cast<uint64_t>(value);
  }
}

/// Restores the physical value of logical type T from its toOrdered() form.
template <typename T>
inline constexpr typename TypeTraits<T>::physicalType fromOrdered(
    uint64_t value) noexcept {
  using PhysicalType = typename TypeTraits<T>::physicalType;
  static_assert(std::is_unsigned_v<PhysicalType>);
  if constexpr (std::is_signed_v<T>) {
    return static_cast<PhysicalType>(
        value ^ (uint64_t{1} << (sizeof(PhysicalType) * 8 - 1)));
  } else {
    return static_cast<PhysicalType>(value);
  }
}

} // namespace facebook::nimble
