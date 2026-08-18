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

#include "velox/dwio/nimble/serializer/SerializationHeader.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::serde {

namespace {

inline SerializationVersion validateVersion(uint8_t versionByte) {
  const auto version = static_cast<SerializationVersion>(versionByte);
  NIMBLE_CHECK(
      version == SerializationVersion::kLegacy ||
          version == SerializationVersion::kLegacyCompact ||
          version == SerializationVersion::kLegacySerialization ||
          version == SerializationVersion::kTablet ||
          version == SerializationVersion::kSerialization ||
          version == SerializationVersion::kProjection,
      "Unsupported version {}",
      versionByte);
  return version;
}

TabletChunkHeader extractTabletChunkHeader(const char*& pos, const char* end) {
  TabletChunkHeader header;

  header.rowCount = varint::readVarint32(&pos);

  NIMBLE_CHECK_GT(end, pos, "Truncated chunk header (flags)");
  const auto flags = readHeaderFlags(pos, SerializationVersion::kTablet);
  header.requiresNullBarrier = flags.requiresNullBarrier;
  header.streamEncodingUsesVarintRowCount =
      flags.streamEncodingUsesVarintRowCount;

  const uint32_t startRow = varint::readVarint32(&pos);
  const uint32_t endRow = varint::readVarint32(&pos);
  NIMBLE_CHECK_LE(
      startRow, endRow, "Row range startRow must not exceed endRow");
  NIMBLE_CHECK_LE(
      endRow,
      header.rowCount,
      "Row range endRow must not exceed stripe rowCount");
  header.rowRange = RowRange{startRow, endRow};

  const uint32_t resumeKeyLength = varint::readVarint32(&pos);
  if (resumeKeyLength > 0) {
    const uint32_t resumeKeyBytes = resumeKeyLength - 1;
    NIMBLE_CHECK_GE(
        end - pos,
        static_cast<ptrdiff_t>(resumeKeyBytes),
        "Truncated resume key payload");
    header.resumeKey = std::string(pos, resumeKeyBytes);
    pos += resumeKeyBytes;
  }
  return header;
}

} // namespace

// ---- Generic serialization header ----

SerializationHeader
readSerializationHeader(const char*& pos, const char* end, bool hasHeader) {
  SerializationHeader header;

  if (hasHeader) {
    NIMBLE_CHECK_GE(end - pos, 1, "Truncated header (version)");
    header.version = validateVersion(static_cast<uint8_t>(*pos++));
  }

  if (isTabletVersion(header.version)) {
    auto tablet = extractTabletChunkHeader(pos, end);
    header.rowCount = tablet.rowCount;
    header.flags = {
        .requiresNullBarrier = tablet.requiresNullBarrier,
        .streamEncodingUsesVarintRowCount =
            tablet.streamEncodingUsesVarintRowCount,
    };
    // TODO: consider setting rowRange to nullopt when it covers the full
    // stripe (startRow==0 && endRow==rowCount) to let consumers skip the
    // range check.
    header.rowRange = tablet.rowRange;
    return header;
  }

  if (usesVarintRowCount(header.version)) {
    header.rowCount = varint::readVarint32(&pos);
  } else {
    header.rowCount = encoding::readUint32(pos);
  }

  header.flags = readHeaderFlags(pos, header.version);

  return header;
}

// ---- Tablet chunk header ----

folly::IOBuf createTabletChunkHeader(const TabletChunkHeader& header) {
  if (header.resumeKey.has_value()) {
    NIMBLE_CHECK_LT(
        header.resumeKey->size(),
        std::numeric_limits<uint32_t>::max(),
        "Resume key too large to fit in uint32 length sentinel");
  }
  const uint32_t resumeKeyLength = header.resumeKey.has_value()
      ? static_cast<uint32_t>(header.resumeKey->size() + 1)
      : 0;
  size_t headerBytes = /*version=*/sizeof(uint8_t) +
      varint::varintSize(header.rowCount) + /*flags=*/sizeof(uint8_t) +
      varint::varintSize(header.rowRange.startRow) +
      varint::varintSize(header.rowRange.endRow) +
      varint::varintSize(resumeKeyLength);
  if (resumeKeyLength > 0) {
    headerBytes += header.resumeKey->size();
  }

  auto buf = folly::IOBuf::create(headerBytes);
  auto* pos = reinterpret_cast<char*>(buf->writableData());
  *pos++ = static_cast<char>(SerializationVersion::kTablet);
  varint::writeVarint(/*val=*/header.rowCount, &pos);
  *pos++ = static_cast<char>(detail::makeFlagsByte(
      header.requiresNullBarrier, header.streamEncodingUsesVarintRowCount));
  varint::writeVarint(/*val=*/header.rowRange.startRow, &pos);
  varint::writeVarint(/*val=*/header.rowRange.endRow, &pos);
  varint::writeVarint(/*val=*/resumeKeyLength, &pos);
  // resumeKeyLength is encoded as (key.size() + 1) so that 0 means "absent"
  // and 1 means "present but empty". When resumeKeyLength == 1 the key is
  // empty and there are zero payload bytes to copy — only the length sentinel
  // was written above.
  if (resumeKeyLength > 0) {
    NIMBLE_CHECK(
        header.resumeKey.has_value(),
        "resumeKeyLength > 0 but resumeKey is absent");
    if (!header.resumeKey->empty()) {
      // NOLINTNEXTLINE(facebook-security-vulnerable-memcpy)
      std::memcpy(pos, header.resumeKey->data(), header.resumeKey->size());
    }
  }
  buf->append(headerBytes);
  return std::move(*buf);
}

TabletChunkHeader readTabletChunkHeader(const char*& pos, const char* end) {
  NIMBLE_CHECK_GE(end - pos, 1, "Truncated chunk header (version)");
  const auto version = static_cast<SerializationVersion>(*pos++);
  NIMBLE_CHECK(
      isTabletVersion(version), "Expected kTablet version, got: {}", version);
  return extractTabletChunkHeader(pos, end);
}

} // namespace facebook::nimble::serde
