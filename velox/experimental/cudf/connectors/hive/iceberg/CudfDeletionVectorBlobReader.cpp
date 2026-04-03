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

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorBlobReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

CudfDeletionVectorBlobReader::CudfDeletionVectorBlobReader(
    const IcebergDeleteFile& dvFile)
    : dvFile_(dvFile) {
  VELOX_CHECK(
      dvFile_.content ==
          ::facebook::velox::connector::hive::iceberg::FileContent::
              kDeletionVector,
      "Expected deletion vector file but got content type: {}",
      static_cast<int>(dvFile_.content));
}

std::string CudfDeletionVectorBlobReader::loadBlob() {
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

  return blobData;
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
