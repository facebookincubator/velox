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

#include "velox/dwio/nimble/index/HierarchicalClusterIndex.h"

#include "velox/common/io/Options.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/index/HierarchicalKeyReader.h"

namespace facebook::nimble::index {

std::unique_ptr<HierarchicalClusterIndex> HierarchicalClusterIndex::create(
    Section rootSection,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
    velox::memory::MemoryPool* pool,
    const Options& options) {
  NIMBLE_CHECK_NOT_NULL(keyCodec);
  const auto* root = flatbuffers::GetRoot<serialization::ClusterIndex>(
      rootSection.content().data());
  NIMBLE_CHECK_NOT_NULL(root);
  options.validate();
  NIMBLE_CHECK_EQ(
      static_cast<const void*>(
          &velox::checkedNotNull(options.ioOptions)->memoryPool()),
      static_cast<const void*>(pool),
      "ioOptions pool must match the provided pool");

  // Hierarchical key chunks search per-column integer levels, whose order is
  // the order of the keys of the codec: bytewise by default.
  if (keyCodec->keyComparator() == nullptr) {
    NIMBLE_CHECK_NULL(
        options.keyComparator,
        "Hierarchical cluster index requires bytewise key order");
  } else {
    NIMBLE_CHECK_EQ(
        static_cast<const void*>(options.keyComparator),
        static_cast<const void*>(keyCodec->keyComparator()),
        "Hierarchical cluster index requires the key order of its codec");
  }
  return std::unique_ptr<HierarchicalClusterIndex>(new HierarchicalClusterIndex(
      std::move(rootSection),
      std::move(keyCodec),
      createIndexMetadataInput(options),
      createIndexDataInput(options),
      options.pinIndex,
      options.preloadIndex,
      options.cacheKeyEncodingViews,
      pool));
}

HierarchicalClusterIndex::HierarchicalClusterIndex(
    Section rootSection,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
    std::shared_ptr<MetadataInput> metadataInput,
    std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
    bool pinIndex,
    bool preloadIndex,
    bool cacheKeyEncodingViews,
    velox::memory::MemoryPool* pool)
    : ClusterIndexBase(
          std::move(rootSection),
          std::move(metadataInput),
          std::move(dataInput),
          [cacheKeyEncodingViews, keyCodec](
              std::string_view encodedKeys,
              std::function<void*(uint32_t)> stringBufferFactory,
              velox::memory::MemoryPool* keyPool) {
            return createHierarchicalKeyReader(
                encodedKeys,
                stringBufferFactory,
                cacheKeyEncodingViews,
                keyCodec,
                keyPool);
          },
          keyCodec->keyComparator(),
          pinIndex,
          preloadIndex,
          pool) {}

HierarchicalClusterIndex::~HierarchicalClusterIndex() = default;

} // namespace facebook::nimble::index
