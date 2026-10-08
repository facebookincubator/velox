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
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {
namespace {

// Bounds the input before handing it to the unchecked varint decoder.
uint32_t checkedVarint32Size(std::string_view data) {
  constexpr uint32_t kMaxBytes = varint::maxVarintSizeForBitWidth(32);
  for (uint32_t i = 0; i < kMaxBytes; ++i) {
    NIMBLE_CHECK_LT(i, data.size(), "Truncated uint32 varint.");
    const auto byte = static_cast<uint8_t>(data[i]);
    if (i == kMaxBytes - 1) {
      NIMBLE_CHECK_LT(byte, 16, "Invalid uint32 varint.");
    }
    if ((byte & 0x80) == 0) {
      return i + 1;
    }
  }
  NIMBLE_UNREACHABLE("Invalid uint32 varint.");
}

} // namespace

namespace encoding {

uint8_t readByte(std::string_view& data) {
  NIMBLE_CHECK(!data.empty(), "Truncated encoding byte.");
  const auto value = static_cast<uint8_t>(data.front());
  data.remove_prefix(1);
  return value;
}

uint16_t readUint16(std::string_view& data) {
  NIMBLE_CHECK_GE(data.size(), sizeof(uint16_t), "Truncated encoding uint16.");
  const uint16_t value =
      static_cast<uint8_t>(data[0]) | (static_cast<uint8_t>(data[1]) << 8);
  data.remove_prefix(sizeof(uint16_t));
  return value;
}

uint32_t readVarint32(std::string_view& data) {
  const auto size = checkedVarint32Size(data);
  const auto* cursor = data.data();
  const auto value = varint::readVarint32(&cursor);
  data.remove_prefix(size);
  return value;
}

std::string_view readLengthPrefixedBytes(std::string_view& data) {
  auto cursor = data;
  const auto size = readVarint32(cursor);
  NIMBLE_CHECK_LE(size, cursor.size(), "Truncated length-prefixed payload.");
  const auto payload = cursor.substr(0, size);
  cursor.remove_prefix(size);
  data = cursor;
  return payload;
}

} // namespace encoding

std::string_view EncodingPrefix::consume(
    std::string_view& data,
    bool useVarint) {
  uint32_t size = kFixedPrefixSize;
  if (useVarint) {
    NIMBLE_CHECK_GE(data.size(), kRowCountOffset, "Truncated encoding prefix.");
    size = kRowCountOffset + checkedVarint32Size(data.substr(kRowCountOffset));
  } else {
    NIMBLE_CHECK_GE(data.size(), size, "Truncated encoding prefix.");
  }
  const auto prefix = data.substr(0, size);
  data.remove_prefix(size);
  return prefix;
}

} // namespace facebook::nimble
