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

#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

#include "velox/common/file/File.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/index/ChunkStatsGroup.h"
#include "velox/dwio/nimble/index/ClusterIndex.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble::index::test {

/// Base test class providing common test utilities for tablet index tests.
class ClusterIndexTestBase : public ::testing::Test {
 protected:
  static void SetUpTestCase();

  void SetUp() override {}

  struct Stream {
    uint32_t numChunks;
    std::vector<int32_t> chunkRows;
    std::vector<uint32_t> chunkOffsets;
  };

  struct KeyStream {
    uint32_t streamOffset;
    uint32_t streamSize;
    Stream stream;
    std::vector<std::string> chunkKeys;
  };

  struct Stripe {
    std::vector<Stream> streams;
    KeyStream keyStream;
  };

  /// Structure to hold the serialized index buffers.
  struct IndexBuffers {
    /// Serialized ClusterIndexGroup flatbuffers (cluster index) appended
    /// sequentially.
    std::string indexPartitions;
    /// Serialized ChunkStats flatbuffers appended sequentially.
    std::string chunkStatsGroups;
    /// Size of each ChunkStats flatbuffer in chunkStatsGroups.
    std::vector<size_t> chunkStatsGroupSizes;
    /// Serialized root Index flatbuffer.
    std::string rootIndex;
    /// Per-stripe row counts (from test data).
    std::vector<uint32_t> stripeRowCounts;
    /// Per-stripe group indices (which partition each stripe belongs to).
    std::vector<uint32_t> stripeGroupIndices;
    /// Each stripe with the location of its key stream, by stripe group, as
    /// the stripe group metadata gives them to an index in the Stripe layout.
    std::vector<std::vector<StripeMetadata>> stripeKeyStreams;
  };

  /// Id that test indexes in the Stripe layout give the key stream of every
  /// stripe.
  static constexpr uint32_t kKeyStreamId{3};

  /// Helper function to create a serialized Index flatbuffer for testing.
  /// Each stripe's keyStream locates that stripe's key chunks in the key
  /// stream file, with chunk offsets relative to keyStream.streamOffset. In the
  /// Partition layout, the key streams of a partition's stripes must be
  /// adjacent, since the partition reads them as one blob. In the Stripe
  /// layout, each stripe holds one key chunk, and its key stream can be
  /// anywhere in the file.
  IndexBuffers createTestClusterIndex(
      const std::vector<std::string>& indexColumns,
      const std::vector<std::string>& sortOrders,
      const std::string& minKey,
      const std::vector<Stripe>& stripes,
      const std::vector<int>& stripeGroups,
      KeyStreamLayout keyStreamLayout = KeyStreamLayout::kPartition);

  /// Convenience wrapper that uses default ascending sort orders.
  IndexBuffers createTestClusterIndex(
      const std::vector<std::string>& indexColumns,
      const std::string& minKey,
      const std::vector<Stripe>& stripes,
      const std::vector<int>& stripeGroups,
      KeyStreamLayout keyStreamLayout = KeyStreamLayout::kPartition);

  /// Fields of a one-group Stripe layout index. Unset fields are left out.
  struct StripeGroupIndex {
    /// stripe_keys of the group: the last key of each stripe.
    std::optional<std::vector<std::string>> stripeKeys{};
    /// Each stripe of the group with the location of its key stream, as its
    /// stripe group metadata gives them.
    std::vector<StripeMetadata> stripeKeyStreams{};
    /// Also locates the group in index_partitions.
    bool setIndexPartitions{false};
    /// Raw key_stream_layout of the root, which may be a value no layout has.
    uint8_t keyStreamLayout{static_cast<uint8_t>(KeyStreamLayout::kStripe)};
  };

  /// Creates an index of one stripe group of 4 rows, with partition keys "b"
  /// and "d", whose group holds the fields of 'index'.
  IndexBuffers createStripeGroupIndex(const StripeGroupIndex& index);

  /// Creates a chunk-index-only test tablet index (no value index, no stripe
  /// keys).
  struct ChunkOnlyStripe {
    std::vector<Stream> streams;
  };

  IndexBuffers createChunkOnlyTestClusterIndex(
      const std::vector<ChunkOnlyStripe>& stripes,
      const std::vector<int>& stripeGroups);

  /// Creates a ClusterIndex from the serialized IndexBuffers.
  /// @param keyStreamFile ReadFile for key stream data (or nullptr for
  ///        metadata-only tests that don't need key stream access).
  /// @param pinIndex Keeps every decoded key chunk.
  /// @param preloadIndex Decodes every key chunk at construction. Requires
  ///        pinIndex.
  /// @param withKeyStreamLocator Gives the index a key stream locator
  ///        that returns indexBuffers.stripeKeyStreams.
  std::unique_ptr<ClusterIndex> createClusterIndex(
      const IndexBuffers& indexBuffers,
      velox::ReadFile* keyStreamFile = nullptr,
      bool pinIndex = false,
      bool preloadIndex = false,
      bool withKeyStreamLocator = true);

  /// Creates a ChunkStatsGroup from the serialized IndexBuffers.
  std::shared_ptr<ChunkStatsGroup> createChunkStats(
      const IndexBuffers& indexBuffers,
      uint32_t stripeGroupIndex);

  std::shared_ptr<velox::memory::MemoryPool> rootPool_{
      velox::memory::memoryManager()->addRootPool("ClusterIndexTestBase")};
  std::shared_ptr<velox::memory::MemoryPool> pool_{
      rootPool_->addLeafChild("ClusterIndexTestBase")};
  std::shared_ptr<velox::ReadFile> metadataFile_;
  std::shared_ptr<velox::io::IoStatistics> ioStats_;
};

} // namespace facebook::nimble::index::test
