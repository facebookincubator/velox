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

#pragma once

#include "velox/experimental/cudf/connectors/hive/iceberg/CudfDeletionVectorReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox::iceberg::test {

namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

/// Serializes a roaring bitmap in the portable format (no-run variant,
/// cookie = 12346). Supports only array containers (cardinality <= 4096).
/// This is the simplest format the DeletionVectorReader needs to parse.
template <typename IndexType>
inline std::string serializeRoaringBitmapNoRun(
    const std::vector<IndexType>& positions) {
  if (positions.empty()) {
    // Empty bitmap: cookie + 0 containers.
    std::string data(8, '\0');
    constexpr uint32_t kCookie = 12346;
    constexpr uint32_t kNumContainers = 0;
    std::memcpy(data.data(), &kCookie, 4);
    std::memcpy(data.data() + 4, &kNumContainers, 4);
    return data;
  }

  // Group positions by high 16 bits.
  std::map<uint16_t, std::vector<uint16_t>> containers;
  for (auto pos : positions) {
    auto key = static_cast<uint16_t>(pos >> 16);
    auto low = static_cast<uint16_t>(pos & 0xFFFF);
    containers[key].push_back(low);
  }

  for (auto& [key, vals] : containers) {
    std::sort(vals.begin(), vals.end());
  }

  uint32_t numContainers = static_cast<uint32_t>(containers.size());

  std::string data;
  // Cookie and container count
  constexpr uint32_t kCookie = 12346;
  data.append(reinterpret_cast<const char*>(&kCookie), 4);
  data.append(reinterpret_cast<const char*>(&numContainers), 4);

  // Key-cardinality pairs.
  for (auto& [key, vals] : containers) {
    uint16_t cardMinus1 = static_cast<uint16_t>(vals.size() - 1);
    data.append(reinterpret_cast<const char*>(&key), 2);
    data.append(reinterpret_cast<const char*>(&cardMinus1), 2);
  }

  // Offset section (required for >= 4 containers).
  if (numContainers >= 4) {
    uint32_t offset = 4 + 4 + numContainers * 4 + numContainers * 4;
    for (auto& [key, vals] : containers) {
      data.append(reinterpret_cast<const char*>(&offset), 4);
      offset += static_cast<uint32_t>(vals.size()) * 2;
    }
  }

  // Container data (array containers: sorted uint16 values).
  for (auto& [key, vals] : containers) {
    for (auto v : vals) {
      data.append(reinterpret_cast<const char*>(&v), 2);
    }
  }

  return data;
}

/// Writes binary data to a temp file and returns the path.
inline std::shared_ptr<common::testutil::TempFilePath> writeDvFile(
    const std::string& bitmapData) {
  auto tempFile = common::testutil::TempFilePath::create();
  // Write directly via C++ streams since TempFilePath already creates the
  // file and the local filesystem openFileForWrite may not overwrite.
  std::ofstream out(tempFile->getPath(), std::ios::binary | std::ios::trunc);
  VELOX_CHECK(out.good(), "Failed to open temp file for writing");
  out.write(bitmapData.data(), static_cast<std::streamsize>(bitmapData.size()));
  out.close();
  return tempFile;
}

/// Creates an IcebergDeleteFile for a deletion vector.
inline velox_iceberg::IcebergDeleteFile makeDvDeleteFile(
    const std::string& filePath,
    uint64_t fileSize,
    int64_t recordCount = 1,
    uint64_t blobOffset = 0,
    std::optional<uint64_t> blobLength = std::nullopt,
    int64_t dataSequenceNumber = 0) {
  using connector::hive::iceberg::CudfDeletionVectorReader;

  std::unordered_map<int32_t, std::string> lowerBounds;
  std::unordered_map<int32_t, std::string> upperBounds;
  lowerBounds[CudfDeletionVectorReader::kDvOffsetFieldId] =
      std::to_string(blobOffset);
  upperBounds[CudfDeletionVectorReader::kDvLengthFieldId] =
      std::to_string(blobLength.value_or(fileSize));
  return velox_iceberg::IcebergDeleteFile(
      velox_iceberg::FileContent::kDeletionVector,
      filePath,
      dwio::common::FileFormat::DWRF,
      recordCount,
      fileSize,
      {},
      std::move(lowerBounds),
      std::move(upperBounds),
      dataSequenceNumber);
}

} // namespace facebook::velox::cudf_velox::iceberg::test
