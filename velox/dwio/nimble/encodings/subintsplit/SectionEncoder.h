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
#include <span>
#include <string_view>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"

namespace facebook::nimble::subintsplit {

/// Copies bits [range] out of each value into `out`, right-aligned.
template <typename PhysicalType, typename StorageType>
void extractSectionValues(
    std::span<const PhysicalType> values,
    BitSection range,
    StorageType* out) noexcept {
  const int shift = range.bitStart;
  const uint64_t mask = range.mask();
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t bits = 0;
    __builtin_memcpy(&bits, &values[i], sizeof(PhysicalType));
    out[i] = static_cast<StorageType>((bits >> shift) & mask);
  }
}

/// Extracts one bit range and runs nested encoding selection over it.
///
/// The section is handed to the selector as the narrowest unsigned type that
/// fits its width, so a narrow section does not pay an 8-byte-per-value penalty
/// when it lands in e.g. Dictionary or Trivial encoding.
template <typename PhysicalType>
std::string_view encodeSection(
    EncodingSelection<PhysicalType>& selection,
    std::span<const PhysicalType> values,
    BitSection range,
    NestedEncodingIdentifier identifier,
    Buffer& sectionBuffer,
    const Encoding::Options& options) {
  return dispatchStorageType(
      sectionStorageBytes(range.width()),
      [&]<typename StorageType>() -> std::string_view {
        Vector<StorageType> sectionValues{
            &sectionBuffer.getMemoryPool(), values.size()};
        extractSectionValues(values, range, sectionValues.data());
        return selection.template encodeNested<StorageType>(
            identifier,
            std::span<const StorageType>(
                sectionValues.data(), sectionValues.size()),
            sectionBuffer,
            options);
      });
}

} // namespace facebook::nimble::subintsplit
