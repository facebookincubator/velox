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

#include <folly/container/F14Map.h>
#include <memory>
#include <string>
#include <vector>

#include "velox/connectors/common/DeletionVectorReader.h"
#include "velox/connectors/hive/iceberg/IcebergDeleteFile.h"

namespace facebook::velox::connector::hive::iceberg {

/// Reads an Iceberg V3 deletion vector and applies it to the delete bitmap.
///
/// Supports both 64-bit Roaring64Bitmap format and legacy 32-bit RoaringBitmap
/// format.
///
/// 64-bit format: [numGroups: uint64] then for each group:
///   [highBits: uint32] [32-bit RoaringBitmap in portable format]
///
/// 32-bit format: [cookie: uint32] [containerCount: uint32] ...
///   Detected by checking if the first 8 bytes match a 32-bit cookie (12346
///   or 12347).
///
/// Loading unwraps the blob and parses the bitmap with RoaringBitmapView, which
/// keeps each container in its serialized form instead of expanding every
/// deleted row. Each batch's delete bitmap is then filled straight from the
/// containers it overlaps.
class DeletionVectorReader : public connector::DeletionVectorReader {
 public:
  /// @param dvFile Iceberg delete file metadata containing the Puffin file
  /// path,
  ///        blob offset, and blob length.
  /// @param splitOffset File position of the first row in the split.
  /// @param pool Memory pool for the bytes of the deletion vector.
  /// @param connectorConfig Connector configuration used to resolve the
  ///        filesystem for the Puffin file (e.g. Manifold client credentials).
  ///        Passing nullptr leaves config-requiring filesystems unconfigured.
  /// @param fileReadOps Options forwarded to the FileSystem when the Puffin
  ///        file is opened and on every read from it.
  /// @param tokenProvider Query token provider for the Puffin file open.
  DeletionVectorReader(
      const IcebergDeleteFile& dvFile,
      uint64_t splitOffset,
      memory::MemoryPool* pool,
      std::shared_ptr<const config::ConfigBase> connectorConfig,
      folly::F14FastMap<std::string, std::string> fileReadOps = {},
      std::shared_ptr<filesystems::TokenProvider> tokenProvider = nullptr);

  /// Loads the deletion vector (if not already loaded) and returns the sorted,
  /// absolute file-level deleted positions. Used to seed a DeletionVectorWriter
  /// when replacing an existing DV with the union of old and new positions.
  /// Expands every position, so the scan path does not call it.
  const std::vector<int64_t>& deletedPositions();

  static constexpr int32_t kDvOffsetFieldId = 100;
  static constexpr int32_t kDvLengthFieldId = 101;

 private:
  // Locates the blob from the delete file metadata or the Puffin footer,
  // reads it, removes its deletion-vector-v1 frame and parses the bitmap
  // inside.
  Bitmap readBitmap(ReadFile& file, const FileIoContext& ioContext) override;

  // The deletion vector file metadata from the Iceberg manifest.
  const IcebergDeleteFile dvFile_;

  // Expanded positions, built only by deletedPositions().
  std::vector<int64_t> deletedPositions_;
  bool positionsExpanded_{false};
};

} // namespace facebook::velox::connector::hive::iceberg
