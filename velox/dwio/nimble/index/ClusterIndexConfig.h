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
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"
#include "velox/dwio/nimble/index/SortOrder.h"

namespace facebook::nimble::index {

/// Configuration for generating a cluster index over sorted input.
///
/// EXPERIMENTAL: Cluster index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
struct ClusterIndexConfig final : IndexConfig {
  ClusterIndexConfig(
      std::string indexName,
      std::vector<std::string> columns,
      std::vector<SortOrder> sortOrders,
      bool enforceKeyOrder,
      bool noDuplicateKey,
      EncodingLayout encodingLayout,
      uint64_t maxRowsPerKeyChunk,
      CompressionType keyChunkCompressionType)
      : IndexConfig{IndexFamily::Cluster, std::move(indexName)},
        columns{std::move(columns)},
        sortOrders{std::move(sortOrders)},
        enforceKeyOrder{enforceKeyOrder},
        noDuplicateKey{noDuplicateKey},
        encodingLayout{std::move(encodingLayout)},
        maxRowsPerKeyChunk{maxRowsPerKeyChunk},
        keyChunkCompressionType{keyChunkCompressionType} {}

  /// Columns encoded into index keys for data pruning.
  std::vector<std::string> columns;
  /// Sort order for each key column. Empty means ascending for all columns;
  /// otherwise, the size must match columns.
  std::vector<SortOrder> sortOrders;
  /// Whether encoded keys must be in ascending order across the file.
  bool enforceKeyOrder;
  /// Whether duplicate encoded keys are rejected when key order is enforced.
  bool noDuplicateKey;
  /// Key-stream encoding. Only Prefix and Trivial encodings are supported.
  EncodingLayout encodingLayout;
  /// Maximum rows per key chunk. Smaller values improve lookup granularity at
  /// the cost of additional metadata.
  uint64_t maxRowsPerKeyChunk;
  /// Chunk compression for encoded keys. Only Uncompressed, Zstd, and Lz4 are
  /// supported.
  CompressionType keyChunkCompressionType;
};

/// Builds configuration for the built-in cluster index.
///
/// EXPERIMENTAL: Cluster index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
class ClusterIndexConfigBuilder {
 public:
  ClusterIndexConfigBuilder& withKeyColumns(std::vector<std::string> columns) {
    columns_ = std::move(columns);
    return *this;
  }

  ClusterIndexConfigBuilder& withSortOrders(std::vector<SortOrder> sortOrders) {
    sortOrders_ = std::move(sortOrders);
    return *this;
  }

  ClusterIndexConfigBuilder& withEnforceKeyOrder(bool enforceKeyOrder) {
    enforceKeyOrder_ = enforceKeyOrder;
    return *this;
  }

  ClusterIndexConfigBuilder& withNoDuplicateKey(bool noDuplicateKey) {
    noDuplicateKey_ = noDuplicateKey;
    return *this;
  }

  ClusterIndexConfigBuilder& withEncodingLayout(EncodingLayout encodingLayout) {
    encodingLayout_ = std::move(encodingLayout);
    return *this;
  }

  ClusterIndexConfigBuilder& withMaxRowsPerKeyChunk(
      uint64_t maxRowsPerKeyChunk) {
    maxRowsPerKeyChunk_ = maxRowsPerKeyChunk;
    return *this;
  }

  ClusterIndexConfigBuilder& withKeyChunkCompressionType(
      CompressionType compressionType) {
    keyChunkCompressionType_ = compressionType;
    return *this;
  }

  /// Builds an immutable configuration consumed by the index factory.
  std::shared_ptr<const IndexConfig> build() const {
    return std::make_shared<const ClusterIndexConfig>(
        std::string{kClusterIndexName},
        columns_,
        sortOrders_,
        enforceKeyOrder_,
        noDuplicateKey_,
        encodingLayout_,
        maxRowsPerKeyChunk_,
        keyChunkCompressionType_);
  }

 private:
  std::vector<std::string> columns_;
  std::vector<SortOrder> sortOrders_;
  bool enforceKeyOrder_{false};
  bool noDuplicateKey_{false};
  EncodingLayout encodingLayout_{
      EncodingType::Prefix,
      {},
      CompressionType::Uncompressed};
  uint64_t maxRowsPerKeyChunk_{10'000};
  CompressionType keyChunkCompressionType_{CompressionType::Uncompressed};
};

} // namespace facebook::nimble::index
