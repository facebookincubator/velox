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

namespace facebook::nimble::index {

/// Cluster index backed by flat Prefix or Trivial encoded key chunks.
class ClusterIndex final : public ClusterIndexBase {
 public:
  static std::unique_ptr<ClusterIndex> create(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const Options& options);

  ~ClusterIndex() override;

 private:
  ClusterIndex(
      Section rootSection,
      std::shared_ptr<MetadataInput> metadataInput,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      bool pinIndex,
      bool preloadIndex,
      velox::memory::MemoryPool* pool);

  friend class test::ClusterIndexTestHelper;
};

} // namespace facebook::nimble::index
