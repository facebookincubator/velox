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
#include <optional>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/index/IndexConfig.h"
#include "velox/dwio/nimble/index/IndexConstants.h"

namespace facebook::nimble::index {

/// Bloom filter configuration shared by dense index implementations.
struct BloomFilterConfig {
  /// Higher values reduce false positives at the cost of additional memory.
  /// 10 bits per key yields approximately a 1% false-positive rate.
  float bitsPerKey{10.0f};
};

/// Configuration for generating a hash index for point lookups.
///
/// EXPERIMENTAL: Hash index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
struct HashIndexConfig final : IndexConfig {
  HashIndexConfig(
      std::string indexName,
      std::vector<std::string> columns,
      float loadFactor,
      std::optional<BloomFilterConfig> bloomFilter,
      uint64_t maxPartitionSizeBytes)
      : IndexConfig{IndexFamily::Dense, std::move(indexName)},
        columns{std::move(columns)},
        loadFactor{loadFactor},
        bloomFilter{std::move(bloomFilter)},
        maxPartitionSizeBytes{maxPartitionSizeBytes} {}

  /// Columns forming the composite lookup key.
  std::vector<std::string> columns;
  /// Target hash-table load factor. Lower values reduce collisions at the
  /// cost of additional buckets.
  float loadFactor;
  /// Optional bloom filter for fast negative lookups.
  std::optional<BloomFilterConfig> bloomFilter;
  /// Maximum independently loadable partition size. Zero disables
  /// partitioning.
  uint64_t maxPartitionSizeBytes;
};

/// Builds configuration for the built-in hash index.
///
/// EXPERIMENTAL: Hash index is not production-ready. Do not enable for
/// production tables without consulting the Nimble team (oncall: dwios).
class HashIndexConfigBuilder {
 public:
  HashIndexConfigBuilder& withKeyColumns(std::vector<std::string> columns) {
    columns_ = std::move(columns);
    return *this;
  }

  HashIndexConfigBuilder& withLoadFactor(float loadFactor) {
    loadFactor_ = loadFactor;
    return *this;
  }

  HashIndexConfigBuilder& withBloomFilter(float bitsPerKey) {
    bloomFilterBitsPerKey_ = bitsPerKey;
    return *this;
  }

  HashIndexConfigBuilder& withMaxPartitionSizeBytes(
      uint64_t maxPartitionSizeBytes) {
    maxPartitionSizeBytes_ = maxPartitionSizeBytes;
    return *this;
  }

  /// Builds an immutable configuration consumed by the index factory.
  std::shared_ptr<const IndexConfig> build() const {
    std::optional<BloomFilterConfig> bloomFilter;
    if (bloomFilterBitsPerKey_.has_value()) {
      bloomFilter =
          BloomFilterConfig{.bitsPerKey = bloomFilterBitsPerKey_.value()};
    }
    return std::make_shared<const HashIndexConfig>(
        std::string{kDenseHashIndexName},
        columns_,
        loadFactor_,
        bloomFilter,
        maxPartitionSizeBytes_);
  }

 private:
  std::vector<std::string> columns_;
  float loadFactor_{0.7f};
  std::optional<float> bloomFilterBitsPerKey_;
  uint64_t maxPartitionSizeBytes_{0};
};

} // namespace facebook::nimble::index
