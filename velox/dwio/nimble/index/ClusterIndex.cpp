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

#include "velox/dwio/nimble/index/ClusterIndex.h"

#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/KeyReader.h"

namespace facebook::nimble::index {

std::unique_ptr<ClusterIndex> ClusterIndex::create(
    Section rootSection,
    velox::memory::MemoryPool* pool,
    const Options& options) {
  const auto* root = flatbuffers::GetRoot<serialization::ClusterIndex>(
      rootSection.content().data());
  NIMBLE_CHECK_NOT_NULL(root);
  options.validate();
  NIMBLE_CHECK_EQ(
      static_cast<const void*>(&options.ioOptions->memoryPool()),
      static_cast<const void*>(pool),
      "ioOptions pool must match the provided pool");
  return std::unique_ptr<ClusterIndex>(new ClusterIndex(
      std::move(rootSection),
      createIndexMetadataInput(options),
      createIndexDataInput(options),
      options.pinIndex,
      options.preloadIndex,
      pool));
}

ClusterIndex::ClusterIndex(
    Section rootSection,
    std::shared_ptr<MetadataInput> metadataInput,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
    bool pinIndex,
    bool preloadIndex,
    velox::memory::MemoryPool* pool)
    : ClusterIndexBase(
          std::move(rootSection),
          std::move(metadataInput),
          std::move(dataInput),
          [](std::string_view encodedKeys,
             std::function<void*(uint32_t)> stringBufferFactory,
             velox::memory::MemoryPool* pool) {
            return createFlatKeyReader(
                encodedKeys, std::move(stringBufferFactory), pool);
          },
          pinIndex,
          preloadIndex,
          pool) {}

ClusterIndex::~ClusterIndex() = default;

} // namespace facebook::nimble::index
