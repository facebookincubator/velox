/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/index/IndexLookup.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/DataInput.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble::index {

const std::shared_ptr<const IndexLookup::LookupRequest::Options>&
IndexLookup::LookupRequest::defaultOptions() {
  static const auto kDefaultOptions = std::make_shared<const Options>();
  return kDefaultOptions;
}

void IndexLookup::Options::validate() const {
  NIMBLE_CHECK_NOT_NULL(file.get());
  NIMBLE_CHECK_NOT_NULL(ioOptions);
  NIMBLE_CHECK_NOT_NULL(ioOptions->indexIoStats(), "indexIoStats must be set");
  if (fileHandle != nullptr) {
    NIMBLE_CHECK_EQ(
        static_cast<const void*>(file.get()),
        static_cast<const void*>(fileHandle->file.get()),
        "file and fileHandle->file must point to the same file");
  }
  NIMBLE_USER_CHECK(
      !preloadIndex || pinIndex,
      "preloadIndex requires pinIndex=true to retain preloaded chunks");
}

std::shared_ptr<MetadataInput> createIndexMetadataInput(
    const IndexLookup::Options& options) {
  MetadataInput::Options metadataOptions{
      .pool = &options.ioOptions->memoryPool(),
      .ioStats = options.ioOptions->indexIoStats(),
      .maxCoalesceDistance = options.ioOptions->maxCoalesceDistance(),
      .maxCoalesceBytes = options.ioOptions->maxCoalesceBytes(),
      .executor = options.ioOptions->ioExecutor().get(),
      .fileHandle = options.fileHandle,
      .maxCacheEntrySize = options.maxCacheEntrySize,
      .fileIoContext = options.fileIoContext,
      .blockCache = options.blockCache,
  };
  return MetadataInput::create(options.file.get(), metadataOptions);
}

std::shared_ptr<DataInput> createIndexDataInput(
    const IndexLookup::Options& options) {
  auto* pool = &options.ioOptions->memoryPool();
  auto ioStats = options.ioOptions->indexIoStats();
  if (options.blockCache != nullptr) {
    NIMBLE_USER_CHECK_NOT_NULL(
        options.fileHandle, "blockCache requires fileHandle");
    NIMBLE_USER_CHECK(
        options.fileHandle->uuid.hasValue(),
        "blockCache requires fileHandle->uuid");
    return std::make_shared<CachedDataInput>(
        options.file,
        CachedDataInput::Options{
            .pool = pool,
            .cache = options.blockCache,
            .fileId = options.fileHandle->uuid,
            .ioStats = std::move(ioStats),
            .fileIoContext = options.fileIoContext,
        });
  }
  return std::make_shared<DirectDataInput>(
      options.file,
      DirectDataInput::Options{
          .pool = pool,
          .ioStats = std::move(ioStats),
          .maxCoalesceDistance = options.ioOptions->maxCoalesceDistance(),
          .maxCoalesceBytes = options.ioOptions->maxCoalesceBytes(),
          .fileIoContext = options.fileIoContext,
      });
}

std::string toString(IndexType indexType) {
  switch (indexType) {
    case IndexType::Cluster:
      return "Cluster";
    case IndexType::Hash:
      return "Hash";
    case IndexType::Sorted:
      return "Sorted";
    default:
      NIMBLE_UNREACHABLE("Unknown IndexType: {}", static_cast<int>(indexType));
  }
}

std::ostream& operator<<(std::ostream& out, IndexType indexType) {
  return out << toString(indexType);
}

} // namespace facebook::nimble::index
