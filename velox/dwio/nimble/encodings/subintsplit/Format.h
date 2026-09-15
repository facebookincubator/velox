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

#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"

// On-disk layout of a SubIntSplit encoding, after the standard Encoding prefix:
//
//   [1 byte]  numSections (1..64)
//   [1 byte]  flags
//   [numSections × 6 bytes]  {bitStart(1B), bitEnd(1B), encodedSize(4B)}
//   [section_0_bytes][section_1_bytes]...[section_{N-1}_bytes]
//
// Sections are stored in LSB-first order (section 0 covers the lowest bits).
// Section identifiers equal the section index (0, 1, …, numSections-1).

namespace facebook::nimble::subintsplit {

/// Flag bits in the header's second byte, which older writers always wrote as
/// zero.
inline constexpr uint8_t kFlagDelta = 1u << 0;

/// Bytes per section header entry: bitStart + bitEnd + encodedSize.
inline constexpr uint32_t kSectionHeaderSize = 6;

/// Bytes the SubIntSplit-specific header occupies, excluding the section
/// payloads and the standard Encoding prefix.
constexpr uint32_t specificHeaderSize(uint8_t numSections) noexcept {
  return 2u + static_cast<uint32_t>(numSections) * kSectionHeaderSize;
}

/// The two bytes preceding the section headers.
struct StreamHeader {
  uint8_t numSections{0};
  uint8_t flags{0};
};

/// Reads the stream header, leaving `pos` on the first section header entry.
inline StreamHeader readStreamHeader(const char*& pos) {
  StreamHeader header;
  header.numSections = encoding::read<uint8_t>(pos);
  header.flags = encoding::read<uint8_t>(pos);
  return header;
}

/// One section header entry as stored on disk.
struct SectionHeader {
  BitSection range;
  uint32_t encodedSize{0};
};

/// Reads one section header entry, advancing `pos` past it.
inline SectionHeader readSectionHeader(const char*& pos) {
  SectionHeader header;
  header.range.bitStart = encoding::read<uint8_t>(pos);
  header.range.bitEnd = encoding::read<uint8_t>(pos);
  header.encodedSize = encoding::readUint32(pos);
  return header;
}

/// Writes one section header entry, advancing `pos` past it.
inline void
writeSectionHeader(BitSection range, uint32_t encodedSize, char*& pos) {
  encoding::write<uint8_t>(static_cast<uint8_t>(range.bitStart), pos);
  encoding::write<uint8_t>(static_cast<uint8_t>(range.bitEnd), pos);
  encoding::writeUint32(encodedSize, pos);
}

} // namespace facebook::nimble::subintsplit
