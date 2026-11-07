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
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/index/BloomFilter.h"

namespace facebook::nimble::index {

/// Configuration for the split-block layout. It adds no knobs of its own; the
/// type exists so that the factory has something concrete to check, and so
/// that adding a knob later does not disturb the shared configuration.
class BlockedBloomFilterConfig final : public BloomFilterConfig {
 public:
  explicit BlockedBloomFilterConfig(float bitsPerKey = 10.0f)
      : BloomFilterConfig{BloomFilterType::kBlocked, bitsPerKey} {}

  std::unique_ptr<BloomFilterConfig> clone() const override {
    return std::make_unique<BlockedBloomFilterConfig>(*this);
  }
};

/// Builds a split-block bloom filter following the Parquet design. A key maps
/// to one 256-bit block and sets a single bit in each of that block's eight
/// 32-bit words, so both insert and lookup touch one cache line however large
/// the filter grows. The alternative — spreading probes over the whole bit
/// array — costs one cache miss per probe once the filter outgrows the cache.
///
/// Given the key count up front, the builder sizes the filter when it is
/// created and each insert sets its bits. Without it, the builder keeps the
/// hash of every key and sizes the filter in finish().
class BlockedBloomFilterBuilder final : public BloomFilterBuilder {
 public:
  /// Validates 'config' and, when it carries a key count, sizes and allocates
  /// the filter. Throws if config.bitsPerKey is not finite and positive, or if
  /// the count asks for a filter too large to address. 'config' does not have
  /// to outlive the builder.
  BlockedBloomFilterBuilder(
      const BlockedBloomFilterConfig& config,
      velox::memory::MemoryPool* pool);

  void insert(std::string_view key) override;

  std::string_view finish() override;

 private:
  const float bitsPerKey_;
  velox::memory::MemoryPool* const pool_;
  // True when the config gave no key count, so the builder keeps the hash of
  // every key and sizes the filter in finish().
  const bool cacheHashes_;
  // Hash of every inserted key, kept only when cacheHashes_. A key repeated
  // back to back is kept once, so sorted input is sized for its distinct keys
  // rather than its rows. Released by finish().
  Vector<uint64_t> hashes_;
  // Zero until the filter is sized.
  uint32_t numBlocks_{0};
  // Holds the blocks followed by the trailer, which finish() fills in. Sized
  // for both at once so finish() does not have to copy the blocks. Null until
  // the filter is sized.
  velox::BufferPtr data_;
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
      velox::memory::MemoryPool* pool) const override;

  std::unique_ptr<BloomFilterReader> createReader(
      std::string_view payload,
      velox::memory::MemoryPool* pool) const override;
};

} // namespace facebook::nimble::index
