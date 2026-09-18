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
#include "velox/dwio/common/CachedBufferedInput.h"
#include "velox/dwio/common/DirectBufferedInput.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/tablet/MetadataInput.h"

namespace facebook::nimble::index {

void IndexLookup::Options::validate() const {
  NIMBLE_CHECK_NOT_NULL(file.get());
  NIMBLE_CHECK_NOT_NULL(ioOptions);
  NIMBLE_CHECK_NOT_NULL(ioOptions->indexIoStats(), "indexIoStats must be set");
  NIMBLE_CHECK_EQ(
      fileHandle != nullptr,
      cache != nullptr,
      "fileHandle and cache must both be set or neither");
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
      .cache = options.cache};
  return MetadataInput::create(options.file.get(), metadataOptions);
}

std::shared_ptr<velox::dwio::common::BufferedInput> createIndexDataInput(
    const IndexLookup::Options& options) {
  auto readFile = options.file;
  auto indexIoStats = options.ioOptions->indexIoStats();
  if (options.cache != nullptr) {
    return std::make_shared<velox::dwio::common::CachedBufferedInput>(
        std::move(readFile),
        velox::dwio::common::MetricsLog::voidLog(),
        options.fileHandle->uuid,
        options.cache,
        nullptr,
        velox::StringIdLease(),
        std::move(indexIoStats),
        nullptr,
        nullptr,
        *options.ioOptions);
  }
  return std::make_shared<velox::dwio::common::DirectBufferedInput>(
      std::move(readFile),
      velox::dwio::common::MetricsLog::voidLog(),
      velox::StringIdLease(),
      nullptr,
      velox::StringIdLease(),
      std::move(indexIoStats),
      nullptr,
      nullptr,
      *options.ioOptions);
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
