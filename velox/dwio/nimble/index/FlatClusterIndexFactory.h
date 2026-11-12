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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/IndexConstants.h"

namespace facebook::nimble::index {

/// Creates the writers, readers and key encoders of the built-in flat cluster
/// index, kClusterIndexName, whose keys sort bytewise. An index that nests the
/// flat cluster index, such as a weighted cluster index, extends it.
class FlatClusterIndexFactory : public ClusterIndexFactory {
 public:
  std::string_view name() const override {
    return kClusterIndexName;
  }

  std::unique_ptr<IndexWriter> createWriter(
      const IndexConfig& config,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const override;

  std::unique_ptr<IndexKeyEncoder> createKeyEncoder(
      const std::vector<std::string>& columns,
      const velox::RowTypePtr& inputType,
      const std::vector<SortOrder>& sortOrders,
      velox::memory::MemoryPool* pool) const override;

  std::unique_ptr<ClusterIndexBase> createReader(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const IndexLookup::Options& options) const override;
};

} // namespace facebook::nimble::index
