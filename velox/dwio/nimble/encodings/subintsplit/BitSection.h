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

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble::subintsplit {

/// Half-open-free, inclusive range of bit positions within a value, shared by
/// the planner (which chooses ranges) and the codec (which stores them).
struct BitSection {
  /// Lowest bit position covered, counting from the LSB.
  int bitStart{0};

  /// Highest bit position covered, inclusive.
  int bitEnd{0};

  int width() const noexcept {
    return bitEnd - bitStart + 1;
  }

  /// Mask selecting `width()` low bits, applied after shifting right by
  /// `bitStart`.
  uint64_t mask() const noexcept {
    return width() >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width()) - 1);
  }
};

/// Narrowest unsigned integer type a section of `bitWidth` bits is stored as.
/// Narrow sections would otherwise pay an 8-byte-per-value penalty when they
/// land in e.g. Dictionary or Trivial encoding.
constexpr uint8_t sectionStorageBytes(int bitWidth) noexcept {
  if (bitWidth <= 8) {
    return 1;
  }
  if (bitWidth <= 16) {
    return 2;
  }
  if (bitWidth <= 32) {
    return 4;
  }
  return 8;
}

constexpr uint8_t sectionStorageBits(int bitWidth) noexcept {
  return static_cast<uint8_t>(sectionStorageBytes(bitWidth) * 8);
}

/// Calls `fn.template operator()<StorageType>()` with the unsigned integer type
/// a section of `storageBytes` bytes is stored as.
///
/// Every path that touches section storage -- extraction, nested encode,
/// constant folding, chunk decode -- is otherwise the same four-armed switch.
template <typename Fn>
decltype(auto) dispatchStorageType(uint8_t storageBytes, Fn&& fn) {
  switch (storageBytes) {
    case 1:
      return fn.template operator()<uint8_t>();
    case 2:
      return fn.template operator()<uint16_t>();
    case 4:
      return fn.template operator()<uint32_t>();
    default:
      NIMBLE_DCHECK_EQ(
          storageBytes, 8, "Invalid SubIntSplit section storage width.");
      return fn.template operator()<uint64_t>();
  }
}

/// One section of a chosen split: which bits it covers, and what the planner
/// estimated for it.
struct SectionPlan {
  /// Bit range this section covers.
  int bitStart{0};
  int bitEnd{0};

  /// Encoding the cost model expects to win for this range. Advisory: the
  /// nested encoding selection makes the final choice at encode time.
  EncodingType encoding{EncodingType::Trivial};

  /// Estimated total bits to store this section across the full stream.
  double cost{0.0};

  int width() const noexcept {
    return bitEnd - bitStart + 1;
  }

  BitSection range() const noexcept {
    return {.bitStart = bitStart, .bitEnd = bitEnd};
  }
};

} // namespace facebook::nimble::subintsplit
