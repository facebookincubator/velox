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
#include <optional>

#include "velox/dwio/nimble/index/IndexTypes.h"

namespace facebook::nimble {
class MetadataBuffer;
} // namespace facebook::nimble

namespace facebook::nimble::index {

class StreamIndex;

/// Provides chunk-level seeking by row ID within a stripe group.
/// Implementations supply version-specific storage and stream indexes.
class ChunkStatsGroup : public std::enable_shared_from_this<ChunkStatsGroup> {
 public:
  /// Creates a V1 group and takes ownership of its decompressed metadata.
  /// @param firstStripe First stripe index in the group.
  /// @param stripeCount Number of stripes in the group.
  /// @param metadata Decompressed V1 chunk-statistics metadata.
  static std::shared_ptr<ChunkStatsGroup> create(
      uint32_t firstStripe,
      uint32_t stripeCount,
      std::unique_ptr<MetadataBuffer> metadata);

  virtual ~ChunkStatsGroup();

  /// Creates a StreamIndex for the specified stripe and stream ID.
  /// Returns nullptr if the stream is not indexed (has ≤1 chunk).
  /// @param streamSize Total byte size of the stream in this stripe.
  virtual std::shared_ptr<StreamIndex> createStreamIndex(
      uint32_t stripe,
      uint32_t streamId,
      uint32_t streamSize) const = 0;

  uint32_t firstStripe() const {
    return firstStripe_;
  }

  uint32_t numStripes() const {
    return stripeCount_;
  }

  uint32_t numStreams() const {
    return streamCount_;
  }

 protected:
  // Initializes state shared by all implementations.
  ChunkStatsGroup(
      uint32_t firstStripe,
      uint32_t stripeCount,
      uint32_t streamCount);

  uint32_t stripeOffset(uint32_t stripe) const;

 private:
  const uint32_t firstStripe_;
  const uint32_t stripeCount_;
  // Total number of streams indexed in this stripe group.
  const uint32_t streamCount_;
};

/// Provides chunk-level seeking for a specific stream within a stripe.
/// Implementations supply version-specific access to chunk statistics.
class StreamIndex {
 public:
  virtual ~StreamIndex();

  /// Lookup chunk by row ID within the stream's row range.
  virtual ChunkLocation lookupChunk(uint32_t rowId) const = 0;

  /// Returns the per-chunk null-value count for the chunk at the given absolute
  /// position (ChunkLocation::chunkIndex), or std::nullopt when per-chunk null
  /// statistics are absent (files written before chunk statistics were added).
  virtual std::optional<uint32_t> chunkNullCount(uint32_t chunkIndex) const = 0;

  /// Returns the total number of rows in this stream.
  virtual uint32_t rowCount() const = 0;

  /// Returns the stream ID this index is for.
  uint32_t streamId() const {
    return streamId_;
  }

 protected:
  // Initializes stream identity for an implementation.
  explicit StreamIndex(uint32_t streamId) : streamId_{streamId} {}

 private:
  const uint32_t streamId_;
};

} // namespace facebook::nimble::index
