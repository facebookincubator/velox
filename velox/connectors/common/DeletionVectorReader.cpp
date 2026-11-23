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

#include "velox/connectors/common/DeletionVectorReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"
#include "velox/common/file/FileSystems.h"

namespace facebook::velox::connector {

DeletionVectorReader::DeletionVectorReader(
    std::string path,
    uint64_t splitOffset,
    memory::MemoryPool* pool,
    std::shared_ptr<const config::ConfigBase> connectorConfig,
    folly::F14FastMap<std::string, std::string> fileReadOps,
    std::shared_ptr<filesystems::TokenProvider> tokenProvider)
    : path_(std::move(path)),
      splitOffset_(splitOffset),
      pool_(pool),
      connectorConfig_(std::move(connectorConfig)),
      fileReadOps_(std::move(fileReadOps)),
      tokenProvider_(std::move(tokenProvider)) {
  VELOX_CHECK_NOT_NULL(pool_);
}

void DeletionVectorReader::readDeletePositions(
    uint64_t baseReadOffset,
    uint64_t size,
    BufferPtr deleteBitmap) {
  ensureLoaded();
  const auto begin = static_cast<int64_t>(splitOffset_ + baseReadOffset);
  bitmap_.apply(
      begin,
      begin + static_cast<int64_t>(size),
      deleteBitmap->asMutable<uint8_t>());
}

bool DeletionVectorReader::noMoreData() const {
  return loaded_ && bitmap_.exhausted();
}

BufferPtr DeletionVectorReader::readBytes(
    ReadFile& file,
    uint64_t offset,
    uint64_t length,
    const FileIoContext& ioContext) const {
  // TODO: Read through a dwio input stream so that the read can use the cache
  // or direct IO.
  auto bytes = AlignedBuffer::allocate<char>(length, pool_);
  file.pread(offset, length, bytes->asMutable<char>(), ioContext);
  return bytes;
}

const dwio::common::RoaringBitmapView& DeletionVectorReader::bitmap() {
  ensureLoaded();
  return bitmap_;
}

void DeletionVectorReader::ensureLoaded() {
  if (loaded_) {
    return;
  }
  loaded_ = true;

  // The connector config carries the credentials of file systems that need
  // them, such as Manifold.
  auto fs = filesystems::getFileSystem(path_, connectorConfig_);
  VELOX_CHECK_NOT_NULL(
      fs, "No filesystem registered for deletion vector file: {}", path_);
  filesystems::FileOptions options;
  options.fileReadOps = fileReadOps_;
  options.tokenProvider = tokenProvider_;
  auto readFile = fs->openFileForRead(path_, options);
  VELOX_CHECK_NOT_NULL(
      readFile, "Failed to open deletion vector file: {}", path_);

  const FileIoContext ioContext(/*stats=*/nullptr, fileReadOps_);
  auto bitmap = readBitmap(*readFile, ioContext);
  bytes_ = std::move(bitmap.bytes);
  bitmap_ = std::move(bitmap.view);
}

} // namespace facebook::velox::connector
