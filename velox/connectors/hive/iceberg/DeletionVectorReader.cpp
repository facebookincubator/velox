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

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector::hive::iceberg {

DeletionVectorReader::DeletionVectorReader(
    const IcebergDeleteFile& dvFile,
    uint64_t splitOffset,
    memory::MemoryPool* pool)
    : dvFile_(dvFile), splitOffset_(splitOffset), pool_(pool) {
  VELOX_CHECK(
      dvFile_.content == FileContent::kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile_.content));
  VELOX_CHECK_GT(dvFile_.recordCount, 0, "Empty deletion vector.");
}

void DeletionVectorReader::loadBitmap() {
  if (loaded_) {
    return;
  }
  loaded_ = true;

  // Read the raw DV blob from the file. The blob offset and length are
  // encoded in the IcebergDeleteFile bounds maps by the coordinator.
  uint64_t blobOffset = 0;
  uint64_t blobLength = dvFile_.fileSizeInBytes;

  if (auto it = dvFile_.lowerBounds.find(kDvOffsetFieldId);
      it != dvFile_.lowerBounds.end()) {
    blobOffset = std::stoull(it->second);
  }
  if (auto it = dvFile_.upperBounds.find(kDvLengthFieldId);
      it != dvFile_.upperBounds.end()) {
    blobLength = std::stoull(it->second);
  }

  auto fs = filesystems::getFileSystem(dvFile_.filePath, nullptr);
  auto readFile = fs->openFileForRead(dvFile_.filePath);

  VELOX_CHECK_LE(
      blobOffset + blobLength,
      readFile->size(),
      "DV blob range [{}, {}) exceeds file size {}.",
      blobOffset,
      blobOffset + blobLength,
      readFile->size());

  // Read the blob bytes.
  std::string blobData(blobLength, '\0');
  readFile->pread(blobOffset, blobLength, blobData.data());

  // Deserialize the roaring bitmap from the portable binary format.
  // The Iceberg V3 spec uses the standard RoaringBitmap portable
  // serialization (https://roaringbitmap.org/). We parse the bitmap
  // directly without depending on CRoaring — the format is well-defined:
  //
  // For small deletion vectors, the coordinator may provide the deleted
  // positions directly as a sorted list encoded in the blob. For the
  // general case, we parse the roaring bitmap portable format.
  //
  // Portable format layout:
  //   - cookie: uint32 (identifies format version)
  //   - container count: uint32
  //   - per container: key (uint16) + cardinality-1 (uint16)
  //   - per container: offset (uint32) [if >4 containers]
  //   - container data: array or bitset containers
  //
  // We use a simplified parser that extracts all set positions.
  deserializeRoaringBitmap(blobData);

  // Sort positions for efficient batch-range scanning.
  std::sort(deletedPositions_.begin(), deletedPositions_.end());
}

void DeletionVectorReader::deserializeRoaringBitmap(const std::string& data) {
  if (data.size() < 8) {
    VELOX_FAIL(
        "Deletion vector blob too small: {} bytes, expected at least 8.",
        data.size());
  }

  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.data());
  const uint8_t* end = ptr + data.size();

  // Read cookie (first 4 bytes). The portable format has two variants:
  //   - SERIAL_COOKIE_NO_RUNCONTAINER (12346): standard format
  //   - SERIAL_COOKIE (12347): format with run containers
  uint32_t cookie;
  std::memcpy(&cookie, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  static constexpr uint32_t kSerialCookieNoRun = 12346;
  static constexpr uint32_t kSerialCookie = 12347;

  bool hasRunContainers = false;
  uint32_t numContainers = 0;

  if ((cookie & 0xFFFF) == kSerialCookie) {
    hasRunContainers = true;
    numContainers = (cookie >> 16) + 1;
  } else if (cookie == kSerialCookieNoRun) {
    std::memcpy(&numContainers, ptr, sizeof(uint32_t));
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

  // Read run bitmap if present (ceil(numContainers / 8) bytes).
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

  // Read key-cardinality pairs: (uint16 key, uint16 cardinality-1) per
  // container.
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
    ptr += sizeof(uint16_t);
    std::memcpy(&cardMinus1, ptr, sizeof(uint16_t));
    ptr += sizeof(uint16_t);
    containers[i] = {key, static_cast<uint32_t>(cardMinus1) + 1};
  }

  // Skip offset section for no-run format with > 4 containers.
  if (!hasRunContainers && numContainers >= 4) {
    // Offsets: uint32 per container.
    VELOX_CHECK_GE(
        static_cast<size_t>(end - ptr),
        numContainers * 4,
        "Truncated offset section.");
    ptr += numContainers * sizeof(uint32_t);
  }

  // Read container data.
  // Guard against unreasonable recordCount that could cause excessive
  // allocation.
  static constexpr int64_t kMaxDeletionVectorPositions = 1LL
      << 30; // ~1 billion
  VELOX_CHECK_LE(
      dvFile_.recordCount,
      kMaxDeletionVectorPositions,
      "Deletion vector recordCount exceeds maximum: {}",
      dvFile_.recordCount);
  deletedPositions_.reserve(dvFile_.recordCount);

  for (uint32_t i = 0; i < numContainers; ++i) {
    uint32_t highBits = static_cast<uint32_t>(containers[i].key) << 16;
    uint32_t cardinality = containers[i].cardinality;

    if (isRunContainer[i]) {
      // Run container: pairs of (start, length-1).
      uint16_t numRuns;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          2u,
          "Truncated run container header.");
      std::memcpy(&numRuns, ptr, sizeof(uint16_t));
      ptr += sizeof(uint16_t);

      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          static_cast<size_t>(numRuns) * 4,
          "Truncated run container data.");
      for (uint16_t r = 0; r < numRuns; ++r) {
        uint16_t start, lengthMinus1;
        std::memcpy(&start, ptr, sizeof(uint16_t));
        ptr += sizeof(uint16_t);
        std::memcpy(&lengthMinus1, ptr, sizeof(uint16_t));
        ptr += sizeof(uint16_t);
        for (uint32_t v = start;
             v <= static_cast<uint32_t>(start) + lengthMinus1;
             ++v) {
          deletedPositions_.push_back(static_cast<int64_t>(highBits | v));
        }
      }
    } else if (cardinality <= 4096) {
      // Array container: sorted uint16 values.
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          cardinality * 2,
          "Truncated array container.");
      for (uint32_t j = 0; j < cardinality; ++j) {
        uint16_t val;
        std::memcpy(&val, ptr, sizeof(uint16_t));
        ptr += sizeof(uint16_t);
        deletedPositions_.push_back(static_cast<int64_t>(highBits | val));
      }
    } else {
      // Bitset container: 2^16 bits = 8192 bytes.
      static constexpr size_t kBitsetBytes = 8192;
      VELOX_CHECK_GE(
          static_cast<size_t>(end - ptr),
          kBitsetBytes,
          "Truncated bitset container.");
      for (uint32_t word = 0; word < 1024; ++word) {
        uint64_t bits;
        std::memcpy(&bits, ptr + word * 8, sizeof(uint64_t));
        while (bits != 0) {
          uint32_t bit = __builtin_ctzll(bits);
          deletedPositions_.push_back(
              static_cast<int64_t>(highBits | (word * 64 + bit)));
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

  // Advance positionIndex_ past positions before the current batch.
  while (positionIndex_ < deletedPositions_.size() &&
         deletedPositions_[positionIndex_] < rowNumberLowerBound) {
    ++positionIndex_;
  }

  // Set bits for positions within the current batch range.
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
