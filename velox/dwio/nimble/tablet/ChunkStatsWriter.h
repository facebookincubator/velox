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
#include <vector>

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

struct Chunk;

/// ChunkStatsWriter manages chunk-level position index data for streams.
///
/// This index enables O(1) chunk-level seeking within stripes via
/// ChunkedDecoder::skipWithIndex(). It can be used standalone (chunk-index-only
/// mode) or combined with a cluster index (ClusterIndexWriter).
///
/// Each stripe group produces a standalone ChunkStats flatbuffer stored as a
/// MetadataSection. The root ChunkStats table is written to the
/// "columnar.chunk.stats" optional section.
///
/// NOTE: This class is not thread-safe. All methods must be called from a
/// single thread.
class ChunkStatsWriter {
 public:
  /// @param pool Memory pool for allocations.
  /// @param minAvgChunksPerStream Skip writing chunk stats for a stripe group
  ///        if the average number of chunks per stream is below this threshold.
  ///        0 disables chunk stats skipping.
  explicit ChunkStatsWriter(
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream = 2);

  ChunkStatsWriter(const ChunkStatsWriter&) = delete;
  ChunkStatsWriter& operator=(const ChunkStatsWriter&) = delete;

  /// Initializes structures for writing a new stripe.
  void newStripe(size_t streamCount);

  /// Adds chunk-level index data for a stream.
  void addStream(uint32_t streamIndex, const std::vector<Chunk>& chunks);

  /// Writes a standalone ChunkStats flatbuffer for the current stripe group
  /// and stores the resulting MetadataSection.
  ///
  /// @param streamCount Total number of streams in the stripe group.
  /// @param stripeCount Number of stripes in the stripe group.
  /// @param createMetadataSection Callback to create a metadata section in the
  ///        file.
  void writeGroup(
      size_t streamCount,
      size_t stripeCount,
      const CreateMetadataSectionFn& createMetadataSection);

  /// Writes the root ChunkStats table to the "columnar.chunk.stats" optional
  /// section.
  ///
  /// @param writeOptionalSection Callback to persist the root chunk stats as a
  ///        named optional section in the file footer.
  void writeRoot(const WriteOptionalSectionFn& writeOptionalSection);

 private:
  // Holds chunk-level index data for a single stream within a stripe.
  struct StreamIndex {
    // Accumulated row counts per chunk.
    std::vector<uint32_t> chunkRows;
    // Byte offsets of each chunk within the stream.
    std::vector<uint32_t> chunkOffsets;
    // Per-chunk null-value count (statistic used for chunk skipping).
    std::vector<uint32_t> chunkNullCounts;
    // Number of chunks in this stripe for this stream.
    uint32_t chunkCount{0};
  };

  // Holds index data for all streams in a single stripe.
  struct StripeIndex {
    std::vector<StreamIndex> streams;
  };

  // Holds index data for stream chunks across all stripes in a stripe group.
  struct GroupIndex {
    std::vector<StripeIndex> stripes;

    bool empty() const {
      return stripes.empty();
    }
  };

  velox::memory::MemoryPool* const pool_;
  const float minAvgChunksPerStream_;
  std::unique_ptr<GroupIndex> groupIndex_;
  // Metadata sections for chunk stats flatbuffers (used by writeRoot).
  std::vector<MetadataSection> chunkStatsSections_;
  bool finalized_{false};
};

} // namespace facebook::nimble
