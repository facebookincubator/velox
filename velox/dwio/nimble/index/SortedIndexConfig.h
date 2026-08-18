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

namespace facebook::nimble::index {

/// Configuration for a secondary sorted index over unsorted input.
///
/// EXPERIMENTAL: Sorted index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
struct SortedIndexConfig final : IndexConfig {
  SortedIndexConfig(
      std::string indexName,
      std::vector<std::string> columns,
      EncodingLayout encodingLayout,
      uint64_t maxRowsPerKeyChunk)
      : IndexConfig{IndexFamily::Dense, std::move(indexName)},
        columns{std::move(columns)},
        encodingLayout{std::move(encodingLayout)},
        maxRowsPerKeyChunk{maxRowsPerKeyChunk} {}

  /// Columns forming the composite lookup key.
  std::vector<std::string> columns;
  /// Key-stream encoding. Only Prefix and Trivial encodings are supported.
  EncodingLayout encodingLayout;
  /// Maximum rows per key chunk. Zero produces one chunk per partition.
  uint64_t maxRowsPerKeyChunk;
};

/// Builds configuration for the built-in sorted index.
///
/// EXPERIMENTAL: Sorted index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
class SortedIndexConfigBuilder {
 public:
  SortedIndexConfigBuilder& withKeyColumns(std::vector<std::string> columns) {
    columns_ = std::move(columns);
    return *this;
  }

  SortedIndexConfigBuilder& withEncodingLayout(EncodingLayout encodingLayout) {
    encodingLayout_ = std::move(encodingLayout);
    return *this;
  }

  SortedIndexConfigBuilder& withMaxRowsPerKeyChunk(
      uint64_t maxRowsPerKeyChunk) {
    maxRowsPerKeyChunk_ = maxRowsPerKeyChunk;
    return *this;
  }

  /// Builds an immutable configuration consumed by the index factory.
  std::shared_ptr<const IndexConfig> build() const {
    return std::make_shared<const SortedIndexConfig>(
        std::string{kDenseSortedIndexName},
        columns_,
        encodingLayout_,
        maxRowsPerKeyChunk_);
  }

 private:
  std::vector<std::string> columns_;
  EncodingLayout encodingLayout_{
      EncodingType::Prefix,
      {},
      CompressionType::Uncompressed};
  uint64_t maxRowsPerKeyChunk_{0};
};

} // namespace facebook::nimble::index
