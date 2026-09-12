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

#include <cstdint>
#include <memory>
#include <vector>

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

struct Chunk;

/// Selects the on-disk representation for chunk statistics.
enum class ChunkStatsVersion : uint8_t {
  /// Stores statistics as raw FlatBuffer arrays.
  kV1 = 1,
  /// Stores statistics as Nimble-encoded arrays.
  kV2 = 2,
};

/// ChunkStatsWriter manages chunk-level position index data for streams.
///
/// This index enables O(1) chunk-level seeking within stripes via
/// ChunkedDecoder::skipWithIndex(). It can be used standalone (chunk-index-only
/// mode) or combined with a cluster index (ClusterIndexWriter).
///
/// Each stripe group produces a standalone chunk stats flatbuffer stored as a
/// MetadataSection. The root ChunkStats table is written to either the
/// "columnar.chunk.stats" or "columnar.chunk.stats.v2" optional section.
///
/// NOTE: This class is not thread-safe. All methods must be called from a
/// single thread.
class ChunkStatsWriter {
 public:
  /// Creates a writer for the requested on-disk version.
  /// @param version Chunk stats representation to write.
  /// @param pool Memory pool for allocations.
  /// @param minAvgChunksPerStream Skip writing chunk stats for a stripe group
  ///        if the average number of chunks per stream is below this threshold.
  ///        0 disables chunk stats skipping.
  static std::unique_ptr<ChunkStatsWriter> create(
      ChunkStatsVersion version,
      velox::memory::MemoryPool& pool,
      float minAvgChunksPerStream = 2);

  virtual ~ChunkStatsWriter() = default;

  ChunkStatsWriter(const ChunkStatsWriter&) = delete;
  ChunkStatsWriter& operator=(const ChunkStatsWriter&) = delete;

  /// Initializes structures for writing a new stripe.
  virtual void newStripe(size_t streamCount) = 0;

  /// Adds chunk-level index data for a stream.
  virtual void addStream(
      uint32_t streamIndex,
      const std::vector<Chunk>& chunks) = 0;

  /// Writes a standalone ChunkStats flatbuffer for the current stripe group
  /// and stores the resulting MetadataSection.
  ///
  /// @param streamCount Total number of streams in the stripe group.
  /// @param stripeCount Number of stripes in the stripe group.
  /// @param createMetadataSection Callback to create a metadata section in the
  ///        file.
  virtual void writeGroup(
      size_t streamCount,
      size_t stripeCount,
      const CreateMetadataSectionFn& createMetadataSection) = 0;

  /// Writes the root ChunkStats table to either the "columnar.chunk.stats" or
  /// "columnar.chunk.stats.v2" optional section.
  ///
  /// @param writeOptionalSection Callback to persist the root chunk stats as a
  ///        named optional section in the file footer.
  virtual void writeRoot(
      const WriteOptionalSectionFn& writeOptionalSection) = 0;

 protected:
  ChunkStatsWriter() = default;
};

} // namespace facebook::nimble
