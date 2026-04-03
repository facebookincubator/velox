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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"

#include <cstring>

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace {

static constexpr uint8_t kDvMagic[] = {0xD1, 0xD3, 0x39, 0x64};

uint32_t readU32BE(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
      (static_cast<uint32_t>(p[1]) << 16) |
      (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

} // namespace

std::string CudfDeletionVectorReader::loadBlob() {
  uint64_t blobOffset = 0;
  uint64_t blobLength = fileSizeInBytes_;

  if (auto it = lowerBounds_.find(kDvOffsetFieldId);
      it != lowerBounds_.end()) {
    try {
      blobOffset = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob offset from bounds map: {}", e.what());
    }
  }
  if (auto it = upperBounds_.find(kDvLengthFieldId);
      it != upperBounds_.end()) {
    try {
      blobLength = std::stoull(it->second);
    } catch (const std::exception& e) {
      VELOX_FAIL(
          "Failed to parse DV blob length from bounds map: {}", e.what());
    }
  }

  auto fs = filesystems::getFileSystem(filePath_, nullptr);
  auto readFile = fs->openFileForRead(filePath_);

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

  return blobData;
}

void CudfDeletionVectorReader::parseDvBlobEnvelope() {
  // DV-v1 blob format:
  //   [4B BE combined_length] [4B magic] [vector payload ...] [4B BE CRC]
  if (dvBlobBytes_.size() >= 12) {
    const auto* raw =
        reinterpret_cast<const uint8_t*>(dvBlobBytes_.data());
    if (std::memcmp(raw + 4, kDvMagic, 4) == 0) {
      uint32_t combinedLength = readU32BE(raw);
      if (combinedLength >= 4 &&
          dvBlobBytes_.size() >=
              static_cast<std::size_t>(4 + combinedLength + 4)) {
        dvPayloadOffset_ = 8;
        dvPayloadSize_ = combinedLength - 4;
        return;
      }
    }
  }

  dvPayloadOffset_ = 0;
  dvPayloadSize_ = dvBlobBytes_.size();
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
