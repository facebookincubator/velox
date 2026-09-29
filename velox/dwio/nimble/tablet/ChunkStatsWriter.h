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

#include "velox/dwio/nimble/tablet/Constants.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble {

struct Chunk;

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
  /// Configures the chunk statistics representation and collection thresholds.
  struct Options {
    /// Default maximum string or binary value length retained in bounds.
    static constexpr uint32_t kDefaultMaxChunkStringStatSize{64};

    /// Selects the on-disk chunk statistics representation.
    ChunkStatsVersion version{ChunkStatsVersion::kV2};

    /// Skips a stripe group below this average chunks-per-stream threshold.
    /// Zero disables this threshold.
    float minAvgChunksPerStream{2};

    /// Limits the string or binary value length retained in chunk bounds.
    uint32_t maxStringStatSize{kDefaultMaxChunkStringStatSize};
  };

  /// Creates a writer with the specified options.
  /// @param pool Memory pool for allocations.
  /// @param options Chunk statistics format and collection settings.
  static std::unique_ptr<ChunkStatsWriter> create(
      velox::memory::MemoryPool& pool,
      Options options);

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
