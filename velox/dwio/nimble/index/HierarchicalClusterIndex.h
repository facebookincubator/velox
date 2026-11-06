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
#pragma once

#include "velox/dwio/nimble/index/ClusterIndexBase.h"
#include "velox/dwio/nimble/index/HierarchicalClusterIndexFactory.h"

namespace facebook::nimble::index {

/// Cluster index whose key chunks use HierarchicalKeyReader.
class HierarchicalClusterIndex final : public ClusterIndexBase {
 public:
  /// Opens the index. 'keyCodec' defines the keys that lookups take and return,
  /// and must be the codec of the factory that wrote the index. 'options' must
  /// carry the comparator of 'keyCodec', null for bytewise keys.
  static std::unique_ptr<HierarchicalClusterIndex> create(
      Section rootSection,
      std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
      velox::memory::MemoryPool* pool,
      const Options& options);

  ~HierarchicalClusterIndex() override;

 private:
  HierarchicalClusterIndex(
      Section rootSection,
      std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      bool pinIndex,
      bool preloadIndex,
      bool cacheKeyEncodingViews,
      velox::memory::MemoryPool* pool);
};

} // namespace facebook::nimble::index
