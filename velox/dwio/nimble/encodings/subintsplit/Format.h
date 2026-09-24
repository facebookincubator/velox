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
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/subintsplit/BitSection.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"

// On-disk layout of a SubIntSplit encoding, after the standard Encoding prefix:
//
//   [1 byte]  numSections (1..64)
//   [1 byte]  flags: kFlagDelta, kFlagRowFrame, kFlagTransforms
//   [17 bytes, only with kFlagRowFrame]  {guard(1B), slope(8B), base(8B)}
//   [transform block, only with kFlagTransforms]
//     {keySection(1B), transformId(1B) per section}
//   [numSections × 6 bytes]  {bitStart(1B), bitEnd(1B), encodedSize(4B)}
//   [section_0_bytes][section_1_bytes]...[section_{N-1}_bytes]
//
// Sections are stored in LSB-first order (section 0 covers the lowest bits).
// Section identifiers equal the section index (0, 1, …, numSections-1).
//
// A stream with flags zero is laid out exactly as streams written before any
// flag existed.

namespace facebook::nimble::subintsplit {

/// Flag bits in the header's second byte, which older writers always wrote as
/// zero.
///
/// The sections hold zigzag deltas of the values rather than the values.
inline constexpr uint8_t kFlagDelta = 1u << 0;
/// A row frame block follows the flag byte; see RowFrame.
inline constexpr uint8_t kFlagRowFrame = 1u << 1;
/// A section transform block follows the row frame block; see TransformInfo.
inline constexpr uint8_t kFlagTransforms = 1u << 2;
/// Every flag this reader understands. A stream carrying any other bit was
/// written by a newer writer, and reading it as if the bit were absent would
/// return wrong values with no error.
inline constexpr uint8_t kKnownFlags =
    kFlagDelta | kFlagRowFrame | kFlagTransforms;

/// Bytes preceding the section header entries.
inline constexpr uint32_t kStreamHeaderSize = 2;

/// Bytes per section header entry: bitStart + bitEnd + encodedSize.
inline constexpr uint32_t kSectionHeaderSize = 6;

/// Bytes the SubIntSplit-specific header occupies, excluding the optional row
/// frame and transform blocks, the section payloads and the standard Encoding
/// prefix.
constexpr uint32_t specificHeaderSize(uint8_t numSections) noexcept {
  return kStreamHeaderSize +
      static_cast<uint32_t>(numSections) * kSectionHeaderSize;
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

/// Bytes the row frame block occupies, zero for a stream without a frame so
/// that such a stream is byte-identical to one written before frames existed.
inline uint32_t rowFrameHeaderSize(const RowFrame& frame) noexcept {
  return frame.active() ? kRowFrameHeaderSize : 0;
}

/// Whether the stream whose SubIntSplit header starts at `dataOffset` stores
/// zigzag deltas. Such a stream can only be decoded from row zero.
inline bool isDeltaStream(std::string_view data, uint32_t dataOffset) {
  NIMBLE_CHECK_LE(
      dataOffset + 2, data.size(), "SubIntSplit stream is truncated.");
  return (static_cast<uint8_t>(data[dataOffset + 1]) & kFlagDelta) != 0;
}

/// Bits in one value of the stream, read from the data type in its Encoding
/// prefix. The sections of a stream tile exactly this many bits.
inline int streamValueBits(std::string_view data) {
  NIMBLE_CHECK_FILE(
      data.size() > static_cast<size_t>(EncodingPrefix::kDataTypeOffset),
      "SubIntSplit stream is truncated.");
  switch (static_cast<DataType>(data[EncodingPrefix::kDataTypeOffset])) {
    case DataType::Int8:
    case DataType::Uint8:
      return 8;
    case DataType::Int16:
    case DataType::Uint16:
      return 16;
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float:
      return 32;
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Double:
      return 64;
    default:
      NIMBLE_CHECK_FILE(false, "SubIntSplit stream has an unsupported type.");
      return 0;
  }
}

/// Per-section transform metadata, as the header carries it.
///
/// A section is transformed only where it pays for itself, so the ids are per
/// section rather than one for the stream, letting each section decline.
struct TransformInfo {
  /// Index of the section the key-derived permutation sorts by. That section
  /// is stored unpermuted, since it is what rebuilds the order. kNoKeySection
  /// when no section is used as a key.
  static constexpr uint8_t kNoKeySection = 0xFF;
  uint8_t keySection{kNoKeySection};
  /// Transform id per section, 0 where the section was left alone.
  std::vector<uint8_t> transformIds;

  bool anyTransform() const {
    for (uint8_t id : transformIds) {
      if (id != 0) {
        return true;
      }
    }
    return false;
  }
};

/// Bytes the transform block occupies for `info`, zero when nothing is
/// transformed so that an untransformed stream is byte-identical to one written
/// before transforms existed.
inline uint32_t transformHeaderSize(const TransformInfo& info) {
  if (!info.anyTransform()) {
    return 0;
  }
  // The key section, then one id per section.
  return 1 + static_cast<uint32_t>(info.transformIds.size());
}

/// One section of a SubIntSplit stream, as the header describes it: which bits
/// it covers, how to mask and shift its values back into place, the width it
/// was stored at, and the bytes of its sub-stream.
struct StoredSection {
  int bitStart{0};
  int bitEnd{0};
  /// (1 << width) - 1, or ~0 for a full 64-bit section.
  uint64_t mask{0};
  /// 1, 2, 4, or 8 -- matches the section's DataType.
  uint8_t storageBytes{8};
  std::string_view stream;
};

/// Walks the SubIntSplit header: numSections, the flag byte, the row frame
/// block when the flag byte announces one, the transform block when it
/// announces that, one {bitStart, bitEnd, encodedSize} triple per section, then
/// the section payloads back to back in LSB-first order.
///
/// Shared by the encoding and the view so a wire format change cannot reach
/// only one of them. `data` is the whole stream, `dataOffset` its prefix size.
/// Rejects a flag it does not know, since skipping it would read the stream
/// wrongly with no error. `flags`, when not null, receives the flag byte.
inline std::vector<StoredSection> parseSections(
    std::string_view data,
    uint32_t dataOffset,
    TransformInfo* transformInfo = nullptr,
    RowFrame* rowFrame = nullptr,
    uint8_t* flags = nullptr) {
  const char* pos = data.data() + dataOffset;
  // Every field below comes off the wire as arbitrary bytes; without these
  // checks a bad length walks pos past the buffer, a bad entry count resizes
  // a vector by billions of elements, or a bad bit range shifts negatively.
  const char* const streamEnd = data.data() + data.size();
  // A short stream is a corrupt file, not a caller error, so each check
  // names the part that is missing.
  auto requireBytes = [&](size_t bytes, const char* truncationMessage) {
    NIMBLE_CHECK_FILE(
        pos <= streamEnd && static_cast<size_t>(streamEnd - pos) >= bytes,
        truncationMessage);
  };

  const int valueBits = streamValueBits(data);
  requireBytes(kStreamHeaderSize, "SubIntSplit stream header is truncated.");
  const auto header = readStreamHeader(pos);
  const uint8_t numSections = header.numSections;
  NIMBLE_CHECK_FILE(
      numSections > 0,
      "SubIntSplit stream must contain at least one section.");
  // A section covers at least one bit of a value, so there can be no more
  // sections than bits.
  NIMBLE_CHECK_FILE(
      numSections <= valueBits,
      "SubIntSplit stream has too many sections.");
  NIMBLE_CHECK_FILE(
      (header.flags & ~kKnownFlags) == 0,
      "SubIntSplit stream has unsupported flags.");
  // Delta residuals are undone by the running sum over every earlier row, so
  // neither a frame nor a transform, both of which undo per row or per block,
  // can be layered under them.
  NIMBLE_CHECK_FILE(
      (header.flags & kFlagDelta) == 0 ||
          (header.flags & (kFlagRowFrame | kFlagTransforms)) == 0,
      "SubIntSplit delta streams carry no row frame or transforms.");
  if (flags != nullptr) {
    *flags = header.flags;
  }

  RowFrame parsedFrame;
  if ((header.flags & kFlagRowFrame) != 0) {
    requireBytes(
        kRowFrameHeaderSize, "SubIntSplit row frame block is truncated.");
    NIMBLE_CHECK_FILE(
        encoding::read<uint8_t>(pos) == kRowFrameGuard,
        "SubIntSplit row frame block is corrupt.");
    parsedFrame.slope = encoding::read<uint64_t>(pos);
    parsedFrame.base = encoding::read<uint64_t>(pos);
  }
  if (rowFrame != nullptr) {
    *rowFrame = parsedFrame;
  }

  if ((header.flags & kFlagTransforms) != 0) {
    TransformInfo parsed;
    requireBytes(1 + numSections, "SubIntSplit transform block is truncated.");
    parsed.keySection = encoding::read<uint8_t>(pos);
    // The key section is indexed directly when a transform inverts, so a bad
    // value here would read outside the section vector.
    NIMBLE_CHECK_FILE(
        parsed.keySection == TransformInfo::kNoKeySection ||
            parsed.keySection < numSections,
        "SubIntSplit stream names a key section that does not exist.");
    parsed.transformIds.resize(numSections);
    for (uint8_t s = 0; s < numSections; ++s) {
      parsed.transformIds[s] = encoding::read<uint8_t>(pos);
    }
    if (transformInfo != nullptr) {
      *transformInfo = std::move(parsed);
    }
  } else if (transformInfo != nullptr) {
    transformInfo->transformIds.assign(numSections, 0);
  }

  std::vector<StoredSection> sections(numSections);
  // The triples are contiguous, so read them all before walking the payloads.
  std::vector<uint32_t> encodedSizes(numSections);
  requireBytes(
      static_cast<size_t>(numSections) * kSectionHeaderSize,
      "SubIntSplit section headers are truncated.");
  int nextBitStart = 0;
  uint64_t payloadBytes = 0;
  for (uint8_t s = 0; s < numSections; ++s) {
    const auto entry = readSectionHeader(pos);
    sections[s].bitStart = entry.range.bitStart;
    sections[s].bitEnd = entry.range.bitEnd;
    encodedSizes[s] = entry.encodedSize;
    payloadBytes += entry.encodedSize;
    // An inverted or out-of-range pair would give a negative or over-wide
    // shift when the mask is built below, and a gap or an overlap would
    // reassemble values from the wrong bits.
    NIMBLE_CHECK_FILE(
        sections[s].bitStart == nextBitStart &&
            sections[s].bitEnd >= sections[s].bitStart &&
            sections[s].bitEnd < valueBits,
        "SubIntSplit sections must cover the value bits once in order.");
    nextBitStart = sections[s].bitEnd + 1;
  }
  NIMBLE_CHECK_FILE(
      nextBitStart == valueBits,
      "SubIntSplit sections must cover the value bits once in order.");
  NIMBLE_CHECK_FILE(
      payloadBytes == static_cast<uint64_t>(streamEnd - pos),
      "SubIntSplit section payload sizes do not match the stream.");

  for (uint8_t s = 0; s < numSections; ++s) {
    auto& section = sections[s];
    const BitSection range{
        .bitStart = section.bitStart, .bitEnd = section.bitEnd};
    section.mask = range.mask();
    section.storageBytes = sectionStorageBytes(range.width());
    requireBytes(encodedSizes[s], "SubIntSplit section payload is truncated.");
    section.stream = std::string_view{pos, encodedSizes[s]};
    pos += encodedSizes[s];
  }
  return sections;
}

} // namespace facebook::nimble::subintsplit
