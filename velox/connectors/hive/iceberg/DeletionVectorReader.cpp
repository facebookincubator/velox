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

#include "velox/connectors/hive/iceberg/DeletionVectorReader.h"

#include <folly/lang/Bits.h>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {
static constexpr uint32_t kSerialCookieNoRun = 12'346;
static constexpr uint32_t kSerialCookie = 12'347;
} // namespace

DeletionVectorReader::DeletionVectorReader(
    const IcebergDeleteFile& dvFile,
    uint64_t splitOffset,
    memory::MemoryPool* /*pool*/)
    : dvFile_(dvFile), splitOffset_(splitOffset) {
  VELOX_CHECK(
      dvFile_.content == FileContent::kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile_.content));
  VELOX_CHECK_GT(dvFile_.recordCount, 0, "Empty deletion vector.");

  static constexpr int64_t kMaxDeletionVectorRecordCount = 10'000'000'000LL;
  VELOX_CHECK_LE(
      dvFile_.recordCount,
      kMaxDeletionVectorRecordCount,
      "Deletion vector record count exceeds sanity limit: {}",
      dvFile_.recordCount);
}

void DeletionVectorReader::loadBitmap() {
  if (loaded_) {
    return;
  }
  loaded_ = true;

  uint64_t blobOffset = 0;
  uint64_t blobLength = dvFile_.fileSizeInBytes;

  if (auto it = dvFile_.lowerBounds.find(kDvOffsetFieldId);
      it != dvFile_.lowerBounds.end()) {
    try {
      blobOffset = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob offset from bounds map: {}", e.what());
    }
  }
  if (auto it = dvFile_.upperBounds.find(kDvLengthFieldId);
      it != dvFile_.upperBounds.end()) {
    try {
      blobLength = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  auto fs = filesystems::getFileSystem(dvFile_.filePath, nullptr);
  auto readFile = fs->openFileForRead(dvFile_.filePath);

  auto fileSize = readFile->size();
  VELOX_CHECK_LE(
      blobOffset,
      fileSize,
      "DV blob offset {} exceeds file size {}.",
      blobOffset,
      fileSize);
  VELOX_CHECK_LE(
      blobLength,
      fileSize - blobOffset,
      "DV blob range [{}, {}) exceeds file size {}.",
      blobOffset,
      blobOffset + blobLength,
      fileSize);

  std::string blobData(blobLength, '\0');
  readFile->pread(blobOffset, blobLength, blobData.data());

  // Detect format: 64-bit Roaring64Bitmap vs 32-bit RoaringBitmap.
  // 64-bit format starts with [numGroups: uint64]. If the first 4 bytes
  // match a 32-bit cookie (12346 or 12347), it's a legacy 32-bit bitmap.
  // Otherwise, interpret as 64-bit format.
  deserializeRoaring64Bitmap(blobData);

  std::sort(deletedPositions_.begin(), deletedPositions_.end());
}

void DeletionVectorReader::deserializeRoaring64Bitmap(const std::string& data) {
  if (data.size() < 8) {
    VELOX_FAIL(
        "Deletion vector blob too small: {} bytes, expected at least 8.",
        data.size());
  }

  const auto* ptr = reinterpret_cast<const uint8_t*>(data.data());
  const auto* end = ptr + data.size();

  // Peek at first 4 bytes to detect 32-bit vs 64-bit format.
  uint32_t firstWord;
  std::memcpy(&firstWord, ptr, sizeof(uint32_t));
  firstWord = folly::Endian::little(firstWord);

  bool is32BitFormat = (firstWord == kSerialCookieNoRun) ||
      ((firstWord & 0xFFFF) == kSerialCookie);

  if (is32BitFormat) {
    // Legacy 32-bit RoaringBitmap — all positions in [0, 2^32).
    deserialize32BitRoaringBitmap(ptr, end, 0);
    return;
  }

  // 64-bit Roaring64Bitmap format:
  //   [numGroups: uint64]
  //   For each group (sorted by highBits):
  //     [highBits: uint32]
  //     [32-bit RoaringBitmap in portable format]
  uint64_t numGroups;
  std::memcpy(&numGroups, ptr, sizeof(uint64_t));
  numGroups = folly::Endian::little(numGroups);
  ptr += sizeof(uint64_t);

  static constexpr uint64_t kMaxGroups = 1'000'000;
  VELOX_CHECK_LE(
      numGroups,
      kMaxGroups,
      "Roaring64Bitmap group count exceeds sanity limit: {}",
      numGroups);

  for (uint64_t g = 0; g < numGroups; ++g) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        sizeof(uint32_t),
        "Truncated Roaring64Bitmap group header.");

    uint32_t highBits;
    std::memcpy(&highBits, ptr, sizeof(uint32_t));
    highBits = folly::Endian::little(highBits);
    ptr += sizeof(uint32_t);

    int64_t highBitsOffset = static_cast<int64_t>(highBits) << 32;

    // Deserialize the 32-bit bitmap for this group.
    // We need to find its size first by parsing the header.
    deserialize32BitRoaringBitmap(ptr, end, highBitsOffset);

    // Advance ptr past the 32-bit bitmap we just parsed.
    // Re-parse the header to compute the size.
    const auto* bitmapStart = ptr;

    uint32_t cookie;
    std::memcpy(&cookie, bitmapStart, sizeof(uint32_t));
    cookie = folly::Endian::little(cookie);

    bool hasRunContainers = false;
    uint32_t numContainers = 0;

    if ((cookie & 0xFFFF) == kSerialCookie) {
      hasRunContainers = true;
      numContainers = (cookie >> 16) + 1;
      ptr += sizeof(uint32_t);
    } else if (cookie == kSerialCookieNoRun) {
      ptr += sizeof(uint32_t);
      uint32_t containerCount;
      std::memcpy(&containerCount, ptr, sizeof(uint32_t));
      numContainers = folly::Endian::little(containerCount);
      ptr += sizeof(uint32_t);
    } else {
      VELOX_FAIL("Unknown roaring bitmap cookie in 64-bit group: {}", cookie);
    }

    if (numContainers == 0) {
      continue;
    }

    // Skip run bitmap if present.
    if (hasRunContainers) {
      uint32_t runBitmapBytes = (numContainers + 7) / 8;
      ptr += runBitmapBytes;
    }

    // Read key-cardinality pairs to compute container data sizes.
    struct ContainerMeta {
      uint16_t key;
      uint32_t cardinality;
      bool isRun;
    };
    std::vector<ContainerMeta> containers(numContainers);

    // Re-read run bitmap for container type detection.
    const auto* runBitmapPtr =
        hasRunContainers ? bitmapStart + sizeof(uint32_t) : nullptr;

    for (uint32_t i = 0; i < numContainers; ++i) {
      uint16_t key, cardMinus1;
      std::memcpy(&key, ptr, sizeof(uint16_t));
      key = folly::Endian::little(key);
      ptr += sizeof(uint16_t);
      std::memcpy(&cardMinus1, ptr, sizeof(uint16_t));
      cardMinus1 = folly::Endian::little(cardMinus1);
      ptr += sizeof(uint16_t);
      bool isRun = hasRunContainers && runBitmapPtr
          ? ((runBitmapPtr[i / 8] >> (i % 8)) & 1)
          : false;
      containers[i] = {key, static_cast<uint32_t>(cardMinus1) + 1, isRun};
    }

    // Skip offset section.
    if (numContainers >= 4) {
      ptr += numContainers * sizeof(uint32_t);
    }

    // Skip container data.
    for (uint32_t i = 0; i < numContainers; ++i) {
      if (containers[i].isRun) {
        uint16_t numRuns;
        std::memcpy(&numRuns, ptr, sizeof(uint16_t));
        numRuns = folly::Endian::little(numRuns);
        ptr += sizeof(uint16_t) + static_cast<size_t>(numRuns) * 4;
      } else if (containers[i].cardinality <= 4'096) {
        ptr += static_cast<size_t>(containers[i].cardinality) * 2;
      } else {
        ptr += 8'192;
      }
    }
  }
}

void DeletionVectorReader::deserialize32BitRoaringBitmap(
    const uint8_t* ptr,
    const uint8_t* end,
    int64_t highBitsOffset) {
  VELOX_CHECK_GE(static_cast<size_t>(end - ptr), 8, "32-bit bitmap too small.");

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
        "Unknown roaring bitmap cookie: {}. Expected {} or {}.",
        cookie,
        kSerialCookieNoRun,
        kSerialCookie);
  }

  if (numContainers == 0) {
    return;
  }

  // Read run bitmap if present.
  std::vector<bool> isRunContainer(numContainers, false);
  if (hasRunContainers) {
    uint32_t runBitmapBytes = (numContainers + 7) / 8;
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        runBitmapBytes,
        "Truncated run bitmap.");
    for (uint32_t i = 0; i < numContainers; ++i) {
      isRunContainer[i] = (ptr[i / 8] >> (i % 8)) & 1;
    }
    ptr += runBitmapBytes;
  }

  // Read key-cardinality pairs.
  struct ContainerMeta {
    uint16_t key;
    uint32_t cardinality;
  };
  std::vector<ContainerMeta> containers(numContainers);

  VELOX_CHECK_GE(
      static_cast<size_t>(end - ptr),
      numContainers * 4,
      "Truncated container metadata.");
  for (uint32_t i = 0; i < numContainers; ++i) {
    uint16_t key, cardMinus1;
    std::memcpy(&key, ptr, sizeof(uint16_t));
    key = folly::Endian::little(key);
    ptr += sizeof(uint16_t);
    std::memcpy(&cardMinus1, ptr, sizeof(uint16_t));
    cardMinus1 = folly::Endian::little(cardMinus1);
    ptr += sizeof(uint16_t);
    containers[i] = {key, static_cast<uint32_t>(cardMinus1) + 1};
  }

  // Skip offset section.
  if (numContainers >= 4) {
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        numContainers * 4,
        "Truncated offset section.");
    ptr += numContainers * sizeof(uint32_t);
  }

  // dvFile_.recordCount was already validated against
  // kMaxDeletionVectorRecordCount in the constructor.
  deletedPositions_.reserve(deletedPositions_.size() + dvFile_.recordCount);

  // Read container data.
  for (uint32_t i = 0; i < numContainers; ++i) {
    int64_t containerBase =
        highBitsOffset | (static_cast<int64_t>(containers[i].key) << 16);
    uint32_t cardinality = containers[i].cardinality;

    if (isRunContainer[i]) {
      uint16_t numRuns;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr), 2u, "Truncated run container.");
      std::memcpy(&numRuns, ptr, sizeof(uint16_t));
      numRuns = folly::Endian::little(numRuns);
      ptr += sizeof(uint16_t);

      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          static_cast<size_t>(numRuns) * 4,
          "Truncated run container data.");
      for (uint16_t r = 0; r < numRuns; ++r) {
        uint16_t start, lengthMinus1;
        std::memcpy(&start, ptr, sizeof(uint16_t));
        start = folly::Endian::little(start);
        ptr += sizeof(uint16_t);
        std::memcpy(&lengthMinus1, ptr, sizeof(uint16_t));
        lengthMinus1 = folly::Endian::little(lengthMinus1);
        ptr += sizeof(uint16_t);
        for (uint32_t v = start;
             v <= static_cast<uint32_t>(start) + lengthMinus1;
             ++v) {
          deletedPositions_.push_back(containerBase | v);
        }
      }
    } else if (cardinality <= 4'096) {
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          cardinality * 2,
          "Truncated array container.");
      for (uint32_t j = 0; j < cardinality; ++j) {
        uint16_t val;
        std::memcpy(&val, ptr, sizeof(uint16_t));
        val = folly::Endian::little(val);
        ptr += sizeof(uint16_t);
        deletedPositions_.push_back(containerBase | val);
      }
    } else {
      static constexpr size_t kBitsetBytes = 8'192;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          kBitsetBytes,
          "Truncated bitset container.");
      for (uint32_t word = 0; word < 1'024; ++word) {
        uint64_t bits;
        std::memcpy(&bits, ptr + word * 8, sizeof(uint64_t));
        bits = folly::Endian::little(bits);
        while (bits != 0) {
          uint32_t bit = __builtin_ctzll(bits);
          deletedPositions_.push_back(
              containerBase | static_cast<int64_t>(word * 64 + bit));
          bits &= bits - 1;
        }
      }
      ptr += kBitsetBytes;
    }
  }
}

void DeletionVectorReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    BufferPtr deleteBitmap) {
  loadBitmap();

  if (deletedPositions_.empty()) {
    return;
  }

  auto* bitmap = deleteBitmap->asMutable<uint8_t>();
  int64_t rowNumberLowerBound =
      static_cast<int64_t>(splitOffset_ + baseReadOffset);
  int64_t rowNumberUpperBound =
      rowNumberLowerBound + static_cast<int64_t>(size);

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberLowerBound) {
    ++positionIndex_;
  }

  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberUpperBound) {
    auto bitIndex = static_cast<uint64_t>(
        deletedPositions_[positionIndex_] - rowNumberLowerBound);
    bits::setBit(bitmap, bitIndex);
    ++positionIndex_;
  }
}

bool DeletionVectorReader::noMoreData() const {
  return loaded_ && positionIndex_ >= deletedPositions_.size();
}

} // namespace facebook::velox::connector::hive::iceberg
