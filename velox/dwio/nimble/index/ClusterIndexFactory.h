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
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexKeyEncoder.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/IndexWriter.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"
#include "velox/type/Type.h"

namespace facebook::nimble::index {

class ClusterIndex;

class ClusterIndexFactory {
 public:
  virtual ~ClusterIndexFactory() = default;

  virtual std::string_view name() const = 0;

  virtual std::unique_ptr<IndexWriter> createWriter(
      const IndexConfig& config,
      const velox::TypePtr& inputType,
      velox::memory::MemoryPool* pool) const = 0;

  virtual std::unique_ptr<IndexKeyEncoder> createKeyEncoder(
      const std::vector<std::string>& columns,
      const velox::RowTypePtr& inputType,
      const std::vector<SortOrder>& sortOrders,
      velox::memory::MemoryPool* pool) const = 0;

  virtual std::unique_ptr<ClusterIndex> createReader(
      Section rootSection,
      velox::memory::MemoryPool* pool,
      const IndexLookup::Options& options) const = 0;
};

/// Registers a process-wide factory. Call during process initialization,
/// before concurrent factory lookups begin.
void registerClusterIndexFactory(
    std::shared_ptr<const ClusterIndexFactory> factory);

/// Returns a process-lifetime reference. Concurrent access after registration
/// is supported.
const ClusterIndexFactory& clusterIndexFactory(std::string_view name);

} // namespace facebook::nimble::index
