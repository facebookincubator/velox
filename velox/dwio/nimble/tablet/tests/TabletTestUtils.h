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

#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"

namespace facebook::nimble::test {

/// Creates a TabletReader::Options object for testing purposes.
/// Test only, production must plumb their own `IoStatistics`.
inline TabletReader::Options makeTestTabletOptions(
    velox::memory::MemoryPool* pool) {
  TabletReader::Options options;
  options.loadClusterIndex = true;
  options.loadDenseIndexes = true;
  options.ioOptions.emplace(pool)
      .setMetadataIoStats(std::make_shared<velox::io::IoStatistics>())
      .setIndexIoStats(std::make_shared<velox::io::IoStatistics>());
  return options;
}

/// Test helper class for TabletReader that provides access to private members
/// for testing purposes. This follows the same pattern as
/// AsyncDataCacheTestHelper in velox/common/caching/tests/CacheTestUtil.h.
class TabletReaderTestHelper {
 public:
  explicit TabletReaderTestHelper(const TabletReader* tabletReader)
      : tabletReader_(tabletReader) {}

  /// Returns the stripe group index for a given stripe index.
  uint32_t stripeGroupIndex(uint32_t stripeIndex) const {
    return tabletReader_->stripeGroupIndex(stripeIndex);
  }

  /// Returns the number of stripe groups from the tablet metadata.
  size_t numStripeGroups() const {
    return tabletReader_->stripeGroupsMetadata().size();
  }

  /// Returns the number of cached stripe groups.
  size_t cachedStripeGroupCount() const {
    return tabletReader_->stripeGroupCache_.testingCacheCount();
  }

  /// Returns true if the stripe group at the given index is cached.
  bool hasStripeGroupCached(uint32_t groupIndex) const {
    return tabletReader_->stripeGroupCache_.hasCacheEntry(groupIndex);
  }

  /// Returns true if the first stripe group is cached and it's the only one.
  /// This is useful for verifying that when the stripe group metadata is
  /// covered by footer IO, the first stripe group is pre-populated in the
  /// cache without additional reads.
  bool hasOnlyFirstStripeGroupCached() const {
    return cachedStripeGroupCount() == 1 && hasStripeGroupCached(0);
  }

  /// Returns the number of cached chunk stats groups.
  size_t cachedChunkStatsGroupCount() const {
    return tabletReader_->chunkStatsCache_.testingCacheCount();
  }

  /// Returns statistics for cluster-index reads.
  std::shared_ptr<velox::io::IoStatistics> indexIoStats() const {
    return tabletReader_->ioOptions_.indexIoStats();
  }

  /// Returns true if the chunk stats group at the given index is cached.
  bool hasChunkStatsGroupCached(uint32_t groupIndex) const {
    return tabletReader_->chunkStatsCache_.hasCacheEntry(groupIndex);
  }

  /// Returns true if the first chunk stats group is cached and it's the only
  /// one. This is useful for verifying that when the chunk stats is covered by
  /// footer IO, the first chunk stats group is pre-populated in the cache
  /// without additional reads.
  bool hasOnlyFirstChunkStatsGroupCached() const {
    return cachedChunkStatsGroupCount() == 1 && hasChunkStatsGroupCached(0);
  }

  /// Returns the stripe offsets array.
  std::vector<uint64_t> stripeOffsets() const {
    std::vector<uint64_t> offsets;
    offsets.reserve(tabletReader_->stripeCount());
    for (uint32_t i = 0; i < tabletReader_->stripeCount(); ++i) {
      offsets.push_back(tabletReader_->stripeOffsets_[i]);
    }
    return offsets;
  }

  /// Returns the stripe sizes array.
  /// Stripe size is computed as the difference between consecutive offsets,
  /// with the last stripe size computed using the file size.
  std::vector<uint32_t> stripeSizes() const {
    const uint32_t stripeCount = tabletReader_->stripeCount();
    std::vector<uint32_t> sizes;
    sizes.reserve(stripeCount);
    for (uint32_t i = 0; i < stripeCount; ++i) {
      const uint64_t start = tabletReader_->stripeOffsets_[i];
      const uint64_t end = (i + 1 < stripeCount)
          ? tabletReader_->stripeOffsets_[i + 1]
          : tabletReader_->fileSize();
      sizes.push_back(static_cast<uint32_t>(end - start));
    }
    return sizes;
  }

 private:
  const TabletReader* const tabletReader_;
};

} // namespace facebook::nimble::test
