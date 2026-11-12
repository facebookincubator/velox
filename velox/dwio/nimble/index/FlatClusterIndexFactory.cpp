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
#include "velox/dwio/nimble/index/FlatClusterIndexFactory.h"

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/index/ClusterIndexWriter.h"

namespace facebook::nimble::index {

std::unique_ptr<IndexWriter> FlatClusterIndexFactory::createWriter(
    const IndexConfig& config,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) const {
  NIMBLE_USER_CHECK_EQ(config.family, IndexFamily::Cluster);
  NIMBLE_USER_CHECK_EQ(config.name, name());
  return ClusterIndexWriter::create(config, inputType, pool);
}

std::unique_ptr<IndexKeyEncoder> FlatClusterIndexFactory::createKeyEncoder(
    const std::vector<std::string>& columns,
    const velox::RowTypePtr& inputType,
    const std::vector<SortOrder>& sortOrders,
    velox::memory::MemoryPool* pool) const {
  return createNimbleIndexKeyEncoder(columns, inputType, sortOrders, pool);
}

std::unique_ptr<ClusterIndexBase> FlatClusterIndexFactory::createReader(
    Section rootSection,
    velox::memory::MemoryPool* pool,
    const IndexLookup::Options& options) const {
  return ClusterIndex::create(std::move(rootSection), pool, options);
}

} // namespace facebook::nimble::index
