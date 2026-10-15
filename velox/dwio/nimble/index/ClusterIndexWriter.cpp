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

#include "velox/dwio/nimble/index/ClusterIndexWriter.h"

#include "velox/common/Casts.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/ClusterIndexConfig.h"
#include "velox/dwio/nimble/index/ClusterIndexFactory.h"
#include "velox/dwio/nimble/index/KeyChunkBuilder.h"

namespace facebook::nimble::index {
namespace {

std::vector<SortOrder> getSortOrders(const ClusterIndexConfig& config) {
  NIMBLE_USER_CHECK(
      config.sortOrders.empty() ||
          config.sortOrders.size() == config.columns.size(),
      "Cluster index columns and sort orders must have the same size");
  if (!config.sortOrders.empty()) {
    return config.sortOrders;
  }
  return std::vector<SortOrder>(
      config.columns.size(), SortOrder{.ascending = true});
}

} // namespace

std::unique_ptr<ClusterIndexWriter> ClusterIndexWriter::create(
    const IndexConfig& config,
    const velox::TypePtr& inputType,
    velox::memory::MemoryPool* pool) {
  NIMBLE_USER_CHECK_EQ(config.family, IndexFamily::Cluster);
  NIMBLE_CHECK_NOT_NULL(pool, "memory pool must not be null");
  const auto& clusterConfig = checkedIndexConfig<ClusterIndexConfig>(config);
  validateFlatKeyEncodingLayout(clusterConfig.encodingLayout);
  return std::unique_ptr<ClusterIndexWriter>(new ClusterIndexWriter(
      clusterConfig,
      velox::asRowType(inputType),
      getSortOrders(clusterConfig),
      pool));
}

ClusterIndexWriter::ClusterIndexWriter(
    const ClusterIndexConfig& config,
    const velox::RowTypePtr& inputType,
    const std::vector<SortOrder>& sortOrders,
    velox::memory::MemoryPool* pool)
    : ClusterIndexWriterBase{
          inputType,
          Options{
              .indexName = config.name,
              .columns = config.columns,
              .sortOrders = sortOrders,
              .maxRowsPerKeyChunk = config.maxRowsPerKeyChunk,
              .keyChunkCompressionType = config.keyChunkCompressionType,
          },
          createFlatKeyChunkBuilder(
              clusterIndexFactory(config.name)
                  .createKeyEncoder(
                      config.columns,
                      inputType,
                      sortOrders,
                      pool),
              config.encodingLayout,
              config.enforceKeyOrder,
              config.noDuplicateKey,
              pool),
          pool} {}

ClusterIndexWriter::~ClusterIndexWriter() = default;

} // namespace facebook::nimble::index
