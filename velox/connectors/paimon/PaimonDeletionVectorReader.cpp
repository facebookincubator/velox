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
#include "velox/connectors/paimon/PaimonDeletionVectorReader.h"

#include <algorithm>
#include <cstring>

#include <folly/lang/Bits.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector::paimon {
namespace {

constexpr uint32_t kSerialCookieNoRun = 12'346;
constexpr uint32_t kSerialCookie = 12'347;
constexpr uint32_t kRunContainersNoOffsetThreshold = 4;

// Paimon frames each deletion vector as [length][magic][bitmap][CRC]. The
// length and the CRC are big-endian int32s, and the length counts the magic
// and the bitmap. Paimon Java skips the CRC on read, and so does this reader.
constexpr uint64_t kFrameLengthBytes = 4;
constexpr uint64_t kMagicBytes = 4;
constexpr uint64_t kCrcBytes = 4;

// Version 1 frames (Java BitmapDeletionVector) hold a 32-bit RoaringBitmap
// after a big-endian magic number. Their deletion file length is the frame
// length.
constexpr uint32_t kBitmap32Magic = 1'581'511'376;

// Version 2 frames (Java Bitmap64DeletionVector, the Iceberg deletion vector
// layout) hold a Roaring64 bitmap after a little-endian magic number. Their
// deletion file length also counts the frame length and the CRC.
constexpr uint32_t kBitmap64Magic = 1'681'511'377;

} // namespace

PaimonDeletionVectorReader::PaimonDeletionVectorReader(
    const PaimonDeletionFile& deletionFile,
    uint64_t splitOffset)
    : deletionFile_(deletionFile), splitOffset_(splitOffset) {
  static constexpr uint64_t kMaxDeletionVectorCardinality = 10'000'000'000ULL;
  if (deletionFile_.cardinality.has_value()) {
    VELOX_CHECK_LE(
        *deletionFile_.cardinality,
        kMaxDeletionVectorCardinality,
        "Paimon deletion vector cardinality exceeds sanity limit: {}",
        *deletionFile_.cardinality);
  }
}

void PaimonDeletionVectorReader::loadBitmap() {
  if (loaded_) {
    return;
  }
  loaded_ = true;

  auto fs = filesystems::getFileSystem(deletionFile_.path, nullptr);
  auto readFile = fs->openFileForRead(deletionFile_.path);

  const auto fileSize = readFile->size();
  const auto offset = deletionFile_.offset;
  VELOX_CHECK_LE(
      offset,
      fileSize,
      "Paimon deletion-vector offset {} exceeds file size {}.",
      offset,
      fileSize);
  VELOX_CHECK_GE(
      fileSize - offset,
      kFrameLengthBytes + kMagicBytes,
      "Paimon deletion-vector frame header is truncated: {}",
      deletionFile_.path);

  uint8_t header[kFrameLengthBytes + kMagicBytes];
  readFile->pread(offset, sizeof(header), header);
  uint32_t frameLength;
  std::memcpy(&frameLength, header, sizeof(uint32_t));
  frameLength = folly::Endian::big(frameLength);
  uint32_t magic;
  std::memcpy(&magic, header + kFrameLengthBytes, sizeof(uint32_t));

  bool roaring64{false};
  if (folly::Endian::big(magic) == kBitmap32Magic) {
    VELOX_CHECK_EQ(
        frameLength,
        deletionFile_.length,
        "Paimon deletion-vector frame length does not match the deletion "
        "file length: {}",
        deletionFile_.path);
  } else if (folly::Endian::little(magic) == kBitmap64Magic) {
    VELOX_CHECK_EQ(
        frameLength + kFrameLengthBytes + kCrcBytes,
        deletionFile_.length,
        "Paimon deletion-vector frame length does not match the deletion "
        "file length: {}",
        deletionFile_.path);
    roaring64 = true;
  } else {
    VELOX_FAIL(
        "Paimon deletion vector has an unknown magic number: {}",
        folly::Endian::big(magic));
  }
  VELOX_CHECK_GE(
      frameLength,
      kMagicBytes,
      "Paimon deletion-vector frame is shorter than its magic number: {}",
      deletionFile_.path);

  const uint64_t bitmapOffset = offset + kFrameLengthBytes + kMagicBytes;
  const uint64_t bitmapLength = frameLength - kMagicBytes;
  VELOX_CHECK_LE(
      bitmapLength,
      fileSize - bitmapOffset,
      "Paimon deletion-vector range [{}, {}) exceeds file size {}.",
      bitmapOffset,
      bitmapOffset + bitmapLength,
      fileSize);

  std::string bitmap(bitmapLength, '\0');
  readFile->pread(bitmapOffset, bitmapLength, bitmap.data());
  const auto* begin = reinterpret_cast<const uint8_t*>(bitmap.data());
  const auto* end = begin + bitmap.size();
  if (roaring64) {
    deserializeRoaring64Bitmap(begin, end);
  } else {
    deserialize32BitRoaringBitmap(begin, end, /*highBitsOffset=*/0);
  }

  std::sort(deletedPositions_.begin(), deletedPositions_.end());
  deletedPositions_.erase(
      std::unique(deletedPositions_.begin(), deletedPositions_.end()),
      deletedPositions_.end());
  if (deletionFile_.cardinality.has_value()) {
    VELOX_CHECK_EQ(
        deletedPositions_.size(),
        *deletionFile_.cardinality,
        "Paimon deletion-vector cardinality mismatch for '{}'. Expected {}, "
        "read {}.",
        deletionFile_.path,
        *deletionFile_.cardinality,
        deletedPositions_.size());
  }
}

void PaimonDeletionVectorReader::deserializeRoaring64Bitmap(
    const uint8_t* ptr,
    const uint8_t* end) {
  VELOX_CHECK_GE(
      static_cast<size_t>(end - ptr),
      sizeof(uint64_t),
      "Truncated Paimon Roaring64Bitmap group count.");

  uint64_t numGroups;
  std::memcpy(&numGroups, ptr, sizeof(uint64_t));
  numGroups = folly::Endian::little(numGroups);
  ptr += sizeof(uint64_t);

  static constexpr uint64_t kMaxGroups = 1'000'000;
  VELOX_CHECK_LE(
      numGroups,
      kMaxGroups,
      "Paimon Roaring64Bitmap group count exceeds sanity limit: {}",
      numGroups);

  for (uint64_t groupIndex = 0; groupIndex < numGroups; ++groupIndex) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        sizeof(uint32_t),
        "Truncated Paimon Roaring64Bitmap group header.");

    uint32_t highBits;
    std::memcpy(&highBits, ptr, sizeof(uint32_t));
    highBits = folly::Endian::little(highBits);
    ptr += sizeof(uint32_t);

    ptr = deserialize32BitRoaringBitmap(
        ptr, end, static_cast<int64_t>(highBits) << 32);
  }
}

const uint8_t* PaimonDeletionVectorReader::deserialize32BitRoaringBitmap(
    const uint8_t* ptr,
    const uint8_t* end,
    int64_t highBitsOffset) {
  VELOX_CHECK_GE(
      static_cast<size_t>(end - ptr),
      8,
      "Paimon 32-bit roaring bitmap too small.");

  uint32_t cookie;
  std::memcpy(&cookie, ptr, sizeof(uint32_t));
  cookie = folly::Endian::little(cookie);
  ptr += sizeof(uint32_t);

  bool hasRunContainers = false;
  uint32_t numContainers = 0;

  if ((cookie & 0xFFFF) == kSerialCookie) {
    hasRunContainers = true;
    numContainers = (cookie >> 16) + 1;
  } else if (cookie == kSerialCookieNoRun) {
    std::memcpy(&numContainers, ptr, sizeof(uint32_t));
    numContainers = folly::Endian::little(numContainers);
    ptr += sizeof(uint32_t);
  } else {
    VELOX_FAIL(
        "Unknown Paimon roaring bitmap cookie: {}. Expected {} or {}.",
        cookie,
        kSerialCookieNoRun,
        kSerialCookie);
  }

  if (numContainers == 0) {
    return ptr;
  }

  std::vector<bool> isRunContainer(numContainers, false);
  if (hasRunContainers) {
    const auto runBitmapBytes = (numContainers + 7) / 8;
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        runBitmapBytes,
        "Truncated Paimon roaring run bitmap.");
    for (uint32_t i = 0; i < numContainers; ++i) {
      isRunContainer[i] = (ptr[i / 8] >> (i % 8)) & 1;
    }
    ptr += runBitmapBytes;
  }

  struct ContainerMeta {
    uint16_t key;
    uint32_t cardinality;
  };
  std::vector<ContainerMeta> containers;
  containers.reserve(numContainers);

  VELOX_CHECK_GE(
      static_cast<size_t>(end - ptr),
      numContainers * 4,
      "Truncated Paimon roaring container metadata.");
  for (uint32_t i = 0; i < numContainers; ++i) {
    uint16_t key;
    uint16_t cardinalityMinusOne;
    std::memcpy(&key, ptr, sizeof(uint16_t));
    key = folly::Endian::little(key);
    ptr += sizeof(uint16_t);
    std::memcpy(&cardinalityMinusOne, ptr, sizeof(uint16_t));
    cardinalityMinusOne = folly::Endian::little(cardinalityMinusOne);
    ptr += sizeof(uint16_t);
    containers.push_back({key, static_cast<uint32_t>(cardinalityMinusOne) + 1});
  }

  if (!hasRunContainers || numContainers >= kRunContainersNoOffsetThreshold) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        numContainers * 4,
        "Truncated Paimon roaring offset section.");
    ptr += numContainers * sizeof(uint32_t);
  }

  if (deletionFile_.cardinality.has_value()) {
    deletedPositions_.reserve(
        deletedPositions_.size() + *deletionFile_.cardinality);
  }

  for (uint32_t i = 0; i < numContainers; ++i) {
    const int64_t containerBase =
        highBitsOffset | (static_cast<int64_t>(containers[i].key) << 16);
    const auto cardinality = containers[i].cardinality;

    if (isRunContainer[i]) {
      uint16_t numRuns;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          sizeof(uint16_t),
          "Truncated Paimon roaring run container.");
      std::memcpy(&numRuns, ptr, sizeof(uint16_t));
      numRuns = folly::Endian::little(numRuns);
      ptr += sizeof(uint16_t);

      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          static_cast<size_t>(numRuns) * 4,
          "Truncated Paimon roaring run container data.");
      for (uint16_t runIndex = 0; runIndex < numRuns; ++runIndex) {
        uint16_t start;
        uint16_t lengthMinusOne;
        std::memcpy(&start, ptr, sizeof(uint16_t));
        start = folly::Endian::little(start);
        ptr += sizeof(uint16_t);
        std::memcpy(&lengthMinusOne, ptr, sizeof(uint16_t));
        lengthMinusOne = folly::Endian::little(lengthMinusOne);
        ptr += sizeof(uint16_t);
        for (uint32_t value = start;
             value <= static_cast<uint32_t>(start) + lengthMinusOne;
             ++value) {
          deletedPositions_.push_back(containerBase | value);
        }
      }
    } else if (cardinality <= 4'096) {
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          cardinality * 2,
          "Truncated Paimon roaring array container.");
      for (uint32_t j = 0; j < cardinality; ++j) {
        uint16_t value;
        std::memcpy(&value, ptr, sizeof(uint16_t));
        value = folly::Endian::little(value);
        ptr += sizeof(uint16_t);
        deletedPositions_.push_back(containerBase | value);
      }
    } else {
      static constexpr size_t kBitsetBytes = 8'192;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          kBitsetBytes,
          "Truncated Paimon roaring bitset container.");
      for (uint32_t wordIndex = 0; wordIndex < 1'024; ++wordIndex) {
        uint64_t bits;
        std::memcpy(&bits, ptr + wordIndex * 8, sizeof(uint64_t));
        bits = folly::Endian::little(bits);
        while (bits != 0) {
          const auto bit = __builtin_ctzll(bits);
          deletedPositions_.push_back(
              containerBase | static_cast<int64_t>(wordIndex * 64 + bit));
          bits &= bits - 1;
        }
      }
      ptr += kBitsetBytes;
    }
  }
  return ptr;
}

void PaimonDeletionVectorReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    BufferPtr deleteBitmap) {
  loadBitmap();

  if (deletedPositions_.empty()) {
    return;
  }

  auto* bitmap = deleteBitmap->asMutable<uint8_t>();
  const int64_t rowNumberLowerBound =
      static_cast<int64_t>(splitOffset_ + baseReadOffset);
  const int64_t rowNumberUpperBound =
      rowNumberLowerBound + static_cast<int64_t>(size);

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberLowerBound) {
    ++positionIndex_;
  }

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberUpperBound) {
    const auto bitIndex = static_cast<uint64_t>(
        deletedPositions_[positionIndex_] - rowNumberLowerBound);
    bits::setBit(bitmap, bitIndex);
    ++positionIndex_;
  }
}

bool PaimonDeletionVectorReader::noMoreData() const {
  return loaded_ && positionIndex_ >= deletedPositions_.size();
}

} // namespace facebook::velox::connector::paimon
