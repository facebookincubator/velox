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

#include "velox/buffer/Buffer.h"
#include "velox/dwio/common/RoaringBitmapView.h"

namespace facebook::velox {
class ReadFile;
struct FileIoContext;
} // namespace facebook::velox

namespace facebook::velox::config {
class ConfigBase;
}

namespace facebook::velox::filesystems {
class TokenProvider;
}

namespace facebook::velox::connector {

/// Reads a deletion vector and applies it to the delete bitmaps of the batches
/// of a split. A deletion vector is a Roaring bitmap of the deleted row
/// positions of one data file, stored in a byte range of another file. Table
/// formats locate and frame the bitmap differently, so each subclass reads it
/// in readBitmap(). The file is opened and read on the first
/// readDeletePositions() call, and RoaringBitmapView applies the bitmap without
/// expanding its positions.
class DeletionVectorReader {
 public:
  virtual ~DeletionVectorReader() = default;

  /// Sets the bits of the deleted rows among the 'size' rows that start at
  /// row 'baseReadOffset' of the split. Batches must be read in increasing row
  /// order.
  void readDeletePositions(
      uint64_t baseReadOffset,
      uint64_t size,
      BufferPtr deleteBitmap);

  /// Returns true when all deleted positions have been consumed.
  bool noMoreData() const;

 protected:
  // The bytes of a deletion vector and its bitmap, which views them.
  struct Bitmap {
    BufferPtr bytes;
    dwio::common::RoaringBitmapView view;
  };

  // @param path File that holds the deletion vector.
  // @param splitOffset Row position in the data file of the first row of the
  //        split.
  // @param pool Memory pool for the bytes of the deletion vector.
  // @param connectorConfig Connector configuration that resolves the file
  //        system of 'path', as it does for the data files.
  // @param fileReadOps Options for opening and reading 'path'.
  // @param tokenProvider Query token provider for opening 'path'.
  DeletionVectorReader(
      std::string path,
      uint64_t splitOffset,
      memory::MemoryPool* pool,
      std::shared_ptr<const config::ConfigBase> connectorConfig,
      folly::F14FastMap<std::string, std::string> fileReadOps,
      std::shared_ptr<filesystems::TokenProvider> tokenProvider);

  // Locates the deletion vector in 'file', reads it with readBytes() and
  // returns its bitmap.
  virtual Bitmap readBitmap(ReadFile& file, const FileIoContext& ioContext) = 0;

  // Reads 'length' bytes at 'offset' of 'file' into a buffer from the pool.
  BufferPtr readBytes(
      ReadFile& file,
      uint64_t offset,
      uint64_t length,
      const FileIoContext& ioContext) const;

  // Returns the bitmap, reading the deletion vector on the first call.
  const dwio::common::RoaringBitmapView& bitmap();

 private:
  // Opens the file and reads the bitmap on the first call.
  void ensureLoaded();

  const std::string path_;

  const uint64_t splitOffset_;

  memory::MemoryPool* const pool_;

  // Connector config, file options and query token provider to open the file
  // with. Kept because the file is opened on first use, after the split
  // reader has returned.
  const std::shared_ptr<const config::ConfigBase> connectorConfig_;
  const folly::F14FastMap<std::string, std::string> fileReadOps_;
  const std::shared_ptr<filesystems::TokenProvider> tokenProvider_;

  // Bytes read from the file, which bitmap_ points into.
  BufferPtr bytes_;

  // The deleted row positions of the data file.
  dwio::common::RoaringBitmapView bitmap_;

  // Whether ensureLoaded() has read the deletion vector.
  bool loaded_{false};
};

} // namespace facebook::velox::connector
