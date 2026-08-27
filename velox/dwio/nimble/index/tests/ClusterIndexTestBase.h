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
  };

  /// Helper function to create a serialized Index flatbuffer for testing.
  IndexBuffers createTestClusterIndex(
      const std::vector<std::string>& indexColumns,
      const std::vector<std::string>& sortOrders,
      const std::string& minKey,
      const std::vector<Stripe>& stripes,
      const std::vector<int>& stripeGroups);

  /// Convenience wrapper that uses default ascending sort orders.
  IndexBuffers createTestClusterIndex(
      const std::vector<std::string>& indexColumns,
      const std::string& minKey,
      const std::vector<Stripe>& stripes,
      const std::vector<int>& stripeGroups);

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
  std::unique_ptr<ClusterIndex> createClusterIndex(
      const IndexBuffers& indexBuffers,
      velox::ReadFile* keyStreamFile = nullptr);

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
