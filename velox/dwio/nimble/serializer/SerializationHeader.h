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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "folly/io/Cursor.h"
#include "folly/io/IOBuf.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/RowRange.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/serializer/Options.h"

namespace facebook::nimble::serde {

// ---- Generic serialization header (all versions) ----

/// Parsed flags shared across serialization header layouts.
struct HeaderFlags {
  /// Row/FlatMap null streams contain real nulls and block dense concatenation.
  bool requiresNullBarrier{false};
  /// Encoding stream prefixes store row counts as varints instead of fixed u32.
  bool streamEncodingUsesVarintRowCount{true};
  /// Each kTablet stream retains its encoding chunk header.
  bool streamHasChunkHeader{false};
};

/// Parsed serialization header common to all versions.
struct SerializationHeader {
  /// Header flags byte layout. bit0 is set when the serialized batch contains
  /// Row/FlatMap null streams with real nulls. Readers treat this as a null
  /// barrier and decode the batch through the per-batch path instead of the
  /// dense concat fast path. bit1 is set when stream encoding row counts use
  /// varint instead of fixed u32. For kTablet, bit2 is set when stream payloads
  /// include encoding chunk headers. The remaining bits are reserved for
  /// future use.
  static constexpr uint8_t kNullBarrierRequiredFlag{0x01};
  static constexpr uint8_t kStreamVarintRowCountFlag{0x02};
  static constexpr uint8_t kStreamChunkHeaderFlag{0x04};

  SerializationVersion version{SerializationVersion::kLegacy};
  uint32_t rowCount{0};
  /// Parsed flag values; defaults represent headers without a flags byte.
  HeaderFlags flags;
  /// Row range embedded in kTablet headers. nullopt for other versions.
  std::optional<RowRange> rowRange;
};

namespace detail {

template <typename T>
void reserveForAppend(T& /* buffer */, uint64_t /* requiredSize */) {}

/// Reserves geometrically for Vector<char> append paths (starting at 4 KB and
/// doubling) so repeated appends stay amortized O(1); the exact final payload
/// size is unknown without buffering or a dry run. No-op for other buffer
/// types.
inline void reserveForAppend(Vector<char>& buffer, uint64_t requiredSize) {
  constexpr uint64_t kInitialAppendCapacity = 4096;
  constexpr uint64_t kMaxDoublingCapacity =
      std::numeric_limits<uint64_t>::max() / 2;

  if (requiredSize <= buffer.capacity()) {
    return;
  }

  uint64_t newCapacity = buffer.capacity() == 0
      ? std::max(kInitialAppendCapacity, requiredSize)
      : buffer.capacity();
  while (newCapacity < requiredSize) {
    if (newCapacity > kMaxDoublingCapacity) {
      newCapacity = requiredSize;
      break;
    }
    newCapacity *= 2;
  }
  buffer.reserve(newCapacity);
}

template <typename T>
char* extend(T& buffer, uint32_t size) {
  const auto oldSize = buffer.size();
  reserveForAppend(buffer, oldSize + size);
  buffer.resize(oldSize + size);
  return buffer.data() + oldSize;
}

inline uint8_t makeFlagsByte(
    bool requiresNullBarrier,
    bool streamEncodingUsesVarintRowCount,
    bool streamHasChunkHeader) {
  return (requiresNullBarrier ? SerializationHeader::kNullBarrierRequiredFlag
                              : 0) |
      (streamEncodingUsesVarintRowCount
           ? SerializationHeader::kStreamVarintRowCountFlag
           : 0) |
      (streamHasChunkHeader ? SerializationHeader::kStreamChunkHeaderFlag : 0);
}

inline bool nullBarrierRequired(uint8_t flagsByte) {
  return (flagsByte & SerializationHeader::kNullBarrierRequiredFlag) != 0;
}

inline bool streamEncodingUsesVarintRowCount(uint8_t flagsByte) {
  return (flagsByte & SerializationHeader::kStreamVarintRowCountFlag) != 0;
}

inline bool streamHasChunkHeader(uint8_t flagsByte) {
  return (flagsByte & SerializationHeader::kStreamChunkHeaderFlag) != 0;
}

inline HeaderFlags parseHeaderFlags(uint8_t flagsByte) {
  return {
      .requiresNullBarrier = nullBarrierRequired(flagsByte),
      .streamEncodingUsesVarintRowCount =
          streamEncodingUsesVarintRowCount(flagsByte),
      .streamHasChunkHeader = streamHasChunkHeader(flagsByte),
  };
}

inline HeaderFlags parseVersionedHeaderFlags(
    uint8_t flagsByte,
    SerializationVersion version) {
  auto flags = parseHeaderFlags(flagsByte);
  if (!isTabletVersion(version)) {
    // Only kTablet uses fixed u32 stream row counts or per-stream chunk
    // headers. Producers predating kStreamVarintRowCountFlag leave that bit
    // clear, so it carries no usable signal for the other formats either.
    // TODO: Drop the row-count override once no producer predating
    // kStreamVarintRowCountFlag remains deployed.
    flags.streamEncodingUsesVarintRowCount = true;
    flags.streamHasChunkHeader = false;
  }
  return flags;
}

} // namespace detail

inline HeaderFlags readHeaderFlags(
    const char*& pos,
    SerializationVersion version) {
  if (!hasSerializationHeaderFlags(version)) {
    return {};
  }
  return detail::parseVersionedHeaderFlags(
      static_cast<uint8_t>(*pos++), version);
}

inline HeaderFlags readHeaderFlags(
    const char*& pos,
    std::optional<SerializationVersion> version) {
  if (!version.has_value()) {
    return {};
  }
  return readHeaderFlags(pos, version.value());
}

inline HeaderFlags readHeaderFlags(
    folly::io::Cursor& cursor,
    SerializationVersion version) {
  if (!hasSerializationHeaderFlags(version)) {
    return {};
  }
  return detail::parseVersionedHeaderFlags(cursor.read<uint8_t>(), version);
}

inline HeaderFlags readHeaderFlags(
    folly::io::Cursor& cursor,
    std::optional<SerializationVersion> version) {
  if (!version.has_value()) {
    return {};
  }
  return readHeaderFlags(cursor, version.value());
}

inline bool readRequiresNullBarrierFlag(
    const char*& pos,
    std::optional<SerializationVersion> version) {
  return readHeaderFlags(pos, version).requiresNullBarrier;
}

inline bool readRequiresNullBarrierFlag(
    const char*& pos,
    SerializationVersion version) {
  return readRequiresNullBarrierFlag(
      pos, std::optional<SerializationVersion>{version});
}

inline bool readRequiresNullBarrierFlag(
    folly::io::Cursor& cursor,
    std::optional<SerializationVersion> version) {
  return readHeaderFlags(cursor, version).requiresNullBarrier;
}

inline bool readRequiresNullBarrierFlag(
    folly::io::Cursor& cursor,
    SerializationVersion version) {
  return readRequiresNullBarrierFlag(
      cursor, std::optional<SerializationVersion>{version});
}

/// Reads the serialization header from `*pos`, detecting the version from the
/// first byte when `hasHeader` is true (otherwise defaults to kLegacy).
/// Advances `*pos` past all header fields. For kTablet, also reads the
/// row range and resume key length fields.
SerializationHeader
readSerializationHeader(const char*& pos, const char* end, bool hasHeader);

/// Writes a legacy serialization header that has no flags byte.
/// Writes [optional_version:1B][rowCount]. For kLegacy / nullopt, rowCount is
/// u32. For legacy encoded formats without a flags byte, rowCount is varint.
/// kSerialization / kProjection must use writeSerializationHeader() returning
/// the flags byte offset.
/// kTablet headers must use createTabletChunkHeader() instead.
///
/// This is only used for legacy versions (kLegacy without header, or kLegacy
/// with header but no flags). Non-legacy versions with headers are normalized
/// to kSerialization and use writeSerializationHeader() with flags.
template <typename T>
void writeLegacySerializationHeader(
    T& buffer,
    std::optional<SerializationVersion> version,
    uint32_t rowCount) {
  NIMBLE_CHECK(
      !hasSerializationHeaderFlags(version),
      "Serialization headers without flags cannot write versions with header flags. Got: {}",
      version.has_value() ? toString(version.value()) : "nullopt");

  if (version.has_value()) {
    auto* versionPos = detail::extend(buffer, 1);
    *versionPos = static_cast<char>(version.value());
  }

  if (usesVarintRowCount(version)) {
    auto* rowCountPos = detail::extend(buffer, varint::varintSize(rowCount));
    varint::writeVarint(rowCount, &rowCountPos);
  } else {
    auto* rowCountPos = detail::extend(buffer, sizeof(uint32_t));
    encoding::writeUint32(rowCount, rowCountPos);
  }
}

/// Writes a nullable-format serialization header to buffer.
/// Returns the byte offset of the flags byte within `buffer`. The flags byte
/// is initialized with the varint row-count flag and may be patched before the
/// buffer is consumed.
/// Writes [version:1B][rowCount:varint][flags:1B].
/// Legacy formats are read-only; new serializer/projector writes use
/// kSerialization or kProjection.
/// kTablet headers must use createTabletChunkHeader() instead.
template <typename T>
size_t writeSerializationHeader(
    T& buffer,
    SerializationVersion version,
    uint32_t rowCount) {
  NIMBLE_CHECK(
      usesCompactHeaderFlags(version),
      "Serialization header writes require kSerialization or kProjection. Got: {}",
      version);

  auto* versionPos = detail::extend(buffer, 1);
  *versionPos = static_cast<char>(version);

  auto* rowCountPos = detail::extend(buffer, varint::varintSize(rowCount));
  varint::writeVarint(rowCount, &rowCountPos);
  const size_t flagsOffset = buffer.size();
  auto* flagsPos = detail::extend(buffer, 1);
  *flagsPos = static_cast<char>(detail::makeFlagsByte(
      /*requiresNullBarrier=*/false,
      /*streamEncodingUsesVarintRowCount=*/true,
      /*streamHasChunkHeader=*/false));
  return flagsOffset;
}

/// Returns the exact byte size of a writable compact serialization header.
/// Valid only for kSerialization and kProjection.
inline size_t estimateSerializationHeaderSize(
    SerializationVersion version,
    uint32_t rowCount) {
  NIMBLE_CHECK(
      usesCompactHeaderFlags(version),
      "Serialization header writes require kSerialization or kProjection. Got: {}",
      version);
  return sizeof(uint8_t) + sizeof(uint8_t) + varint::varintSize(rowCount);
}

// ---- Tablet chunk header (kTablet only) ----

/// Parsed fields of a kTablet chunk slice header.
struct TabletChunkHeader {
  uint32_t rowCount{0};
  /// True when this chunk slice carries a Row/FlatMap null stream with real
  /// nulls; routes the slice to the per-batch barrier path on read.
  bool requiresNullBarrier{false};
  /// True when encoding stream prefixes store row counts as varints.
  bool streamEncodingUsesVarintRowCount{false};
  /// True when each encoding stream retains its tablet chunk header.
  bool streamHasChunkHeader{false};
  RowRange rowRange;
  std::optional<std::string> resumeKey{};
};

/// Builds the kTablet chunk slice header as a freshly-allocated IOBuf.
folly::IOBuf createTabletChunkHeader(const TabletChunkHeader& header);

/// Reads a kTablet chunk slice header including the version byte.
TabletChunkHeader readTabletChunkHeader(const char*& pos, const char* end);

} // namespace facebook::nimble::serde
