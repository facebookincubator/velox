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

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/index/BloomFilter.h"

namespace facebook::nimble::index {

/// Configuration for the split-block layout. It adds no knobs of its own; the
/// type exists so that the factory has something concrete to check, and so
/// that adding a knob later does not disturb the shared configuration.
class BlockedBloomFilterConfig final : public BloomFilterConfig {
 public:
  explicit BlockedBloomFilterConfig(float bitsPerKey = 10.0f)
      : BloomFilterConfig{BloomFilterType::kBlocked, bitsPerKey} {}
};

/// Builds a split-block bloom filter following the Parquet design. A key maps
/// to one 256-bit block and sets a single bit in each of that block's eight
/// 32-bit words, so both insert and lookup touch one cache line however large
/// the filter grows. The alternative — spreading probes over the whole bit
/// array — costs one cache miss per probe once the filter outgrows the cache.
class BlockedBloomFilterBuilder final : public BloomFilterBuilder {
 public:
  /// Sizes the filter for 'numKeys' keys at 'bitsPerKey', rounded up to whole
  /// blocks and never smaller than one block. Throws if 'bitsPerKey' is not
  /// finite and positive.
  BlockedBloomFilterBuilder(
      uint64_t numKeys,
      float bitsPerKey,
      velox::memory::MemoryPool* pool);

  void insert(std::string_view key) override;

  std::string_view finish() override;

 private:
  const uint32_t numBlocks_;
  // Holds the blocks followed by the trailer, which finish() fills in. Sized
  // for both up front so finish() does not have to copy the blocks.
  const velox::BufferPtr data_;
  bool finished_{false};
};

/// Reads a filter produced by BlockedBloomFilterBuilder.
class BlockedBloomFilterReader final : public BloomFilterReader {
 public:
  /// Copies 'payload' — the filter body, with the trailer already stripped —
  /// into 'pool'. Throws if its length is not a positive whole number of
  /// blocks.
  BlockedBloomFilterReader(
      std::string_view payload,
      velox::memory::MemoryPool* pool);

  bool maybeContains(std::string_view key) const override;

  void maybeContains(
      std::span<const std::string_view> keys,
      std::span<bool> out) const override;

 private:
  const uint32_t numBlocks_;
  const velox::BufferPtr data_;
};

/// Creates the split-block builder and reader. Registered for
/// BloomFilterType::kBlocked on first use of the factory registry.
class BlockedBloomFilterFactory final : public BloomFilterFactory {
 public:
  BloomFilterType type() const override {
    return BloomFilterType::kBlocked;
  }

  std::unique_ptr<BloomFilterBuilder> createBuilder(
      const BloomFilterConfig& config,
      uint64_t numKeys,
      velox::memory::MemoryPool* pool) const override;

  std::unique_ptr<BloomFilterReader> createReader(
      std::string_view payload,
      velox::memory::MemoryPool* pool) const override;
};

} // namespace facebook::nimble::index
