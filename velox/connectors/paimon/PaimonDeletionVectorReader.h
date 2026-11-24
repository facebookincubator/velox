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

#include <memory>

#include "velox/common/config/Config.h"
#include "velox/common/file/TokenProvider.h"
#include "velox/connectors/common/DeletionVectorReader.h"
#include "velox/connectors/paimon/PaimonDeletionFile.h"

namespace facebook::velox::connector::paimon {

/// Reads a Paimon deletion vector and applies it to a DWIO delete bitmap.
///
/// Paimon stores deletion vectors as framed Roaring bitmaps in index files,
/// and the data-file metadata gives the byte range of each vector. Both frame
/// versions Paimon Java writes are supported; PaimonDeletionFile describes
/// them. The positions are 0-based row offsets within the data file,
/// independent of the physical file format used for the data rows.
class PaimonDeletionVectorReader : public connector::DeletionVectorReader {
 public:
  /// @param deletionFile Paimon deletion-file metadata containing the file
  ///        path, byte offset, byte length, and cardinality if recorded.
  /// @param pool Memory pool for the bytes of the deletion vector.
  /// @param connectorConfig Connector configuration that resolves the file
  ///        system of the deletion file, as it does for the data files.
  /// @param tokenProvider Query token provider for opening the deletion file.
  PaimonDeletionVectorReader(
      const PaimonDeletionFile& deletionFile,
      memory::MemoryPool* pool,
      std::shared_ptr<const config::ConfigBase> connectorConfig,
      std::shared_ptr<filesystems::TokenProvider> tokenProvider);

 private:
  // Reads the frame of the deletion vector from the start of the frame
  // through at least the end of its bitmap, checks its length against the
  // deletion file and parses its bitmap.
  Bitmap readBitmap(ReadFile& file, const FileIoContext& ioContext) override;

  // Location of the deletion vector frame in its index file.
  const PaimonDeletionFile deletionFile_;
};

} // namespace facebook::velox::connector::paimon
