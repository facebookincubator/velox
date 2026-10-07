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

#include "velox/dwio/nimble/index/ClusterIndexWriterBase.h"

namespace facebook::nimble::index {

struct ClusterIndexConfig;

/// Writes a cluster index backed by hierarchical integral key chunks.
class HierarchicalClusterIndexWriter final : public ClusterIndexWriterBase {
 public:
  static std::unique_ptr<HierarchicalClusterIndexWriter> create(
      const IndexConfig& config,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool);

  ~HierarchicalClusterIndexWriter() override;

 private:
  HierarchicalClusterIndexWriter(
      const ClusterIndexConfig& config,
      const velox::RowTypePtr& inputType,
      std::vector<SortOrder> sortOrders,
      velox::memory::MemoryPool* pool);
};

} // namespace facebook::nimble::index
