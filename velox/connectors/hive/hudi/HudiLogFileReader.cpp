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

#include "velox/connectors/hive/hudi/HudiLogFileReader.h"

#include <cstring>
#include <type_traits>

#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::hudi {

std::vector<HudiLogBlock> HudiLogFileReader::readAllBlocks() {
  std::vector<HudiLogBlock> blocks;
  while (true) {
    HudiLogBlock block;
    if (!readNextBlock(block)) {
      break;
    }
    blocks.push_back(std::move(block));
  }
  return blocks;
}

bool HudiLogFileReader::readNextBlock(HudiLogBlock& block) {
  // A clean end of file: not enough bytes remain for another magic marker.
  if (pos_ + kHudiLogMagic.size() > data_.size()) {
    return false;
  }
  // Enough bytes remain but they are not the magic marker: this is not a
  // clean end of file, but corruption partway through it.
  VELOX_USER_CHECK(
      data_.substr(pos_, kHudiLogMagic.size()) == kHudiLogMagic,
      "Hudi log file is corrupted: expected magic marker not found");
  pos_ += kHudiLogMagic.size();

  // blockLength spans every field after it, up to and including the trailing
  // total-block-length field. It is used to validate the framing.
  const auto blockLength = readBigEndian<uint64_t>();
  const auto blockStart = pos_;

  const auto formatVersion =
      static_cast<LogFormatVersion>(readBigEndian<uint32_t>());
  VELOX_USER_CHECK(
      formatVersion == LogFormatVersion::kV1,
      "Unsupported Hudi log format version: {}",
      static_cast<uint32_t>(formatVersion));
  block.formatVersion = formatVersion;

  block.blockType = static_cast<HudiLogBlockType>(readBigEndian<uint32_t>());
  block.header = readMetadataMap();

  const auto contentLength = readBigEndian<uint64_t>();
  block.content = std::string{readBytes(contentLength)};

  block.footer = readMetadataMap();

  // Trailing total-block-length; read and discarded.
  readBigEndian<uint64_t>();

  VELOX_CHECK_EQ(
      pos_ - blockStart, blockLength, "Hudi log block length mismatch");
  return true;
}

HudiLogBlockMetadata HudiLogFileReader::readMetadataMap() {
  HudiLogBlockMetadata metadata;
  const auto numEntries = readBigEndian<uint32_t>();
  for (uint32_t i = 0; i < numEntries; ++i) {
    const auto keyOrdinal = readBigEndian<uint32_t>();
    const auto valueLength = readBigEndian<uint32_t>();
    const auto value = readBytes(valueLength);
    metadata.emplace(
        static_cast<HudiLogBlockMetadataKey>(keyOrdinal), std::string{value});
  }
  return metadata;
}

std::string_view HudiLogFileReader::readBytes(size_t size) {
  VELOX_CHECK_LE(
      pos_ + size, data_.size(), "Truncated Hudi log file while reading block");
  const auto view = data_.substr(pos_, size);
  pos_ += size;
  return view;
}

template <typename T>
T HudiLogFileReader::readBigEndian() {
  static_assert(std::is_integral_v<T>, "readBigEndian requires an integer");
  const auto view = readBytes(sizeof(T));
  T value;
  std::memcpy(&value, view.data(), sizeof(T));
  return folly::Endian::big(value);
}

} // namespace facebook::velox::connector::hive::hudi
