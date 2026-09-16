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

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/nimble/index/IndexLookup.h"
#include "velox/dwio/nimble/index/KeyChunkDecoder.h"
#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble::serialization {
struct SortedIndex;
} // namespace facebook::nimble::serialization

namespace facebook::nimble::index {

/// A sorted index for point lookups and range scans on unsorted data.
///
/// Stores a sorted key stream where each entry is
/// encoded_key || fixed_width_row_id. Lookup uses binary search on chunk
/// boundary keys, then materializes and searches within the target chunk.
///
/// Lookup flow:
///   1. Binary search chunk keys to find the chunk(s)
///   2. Load and decode the chunk
///   3. Materialize entries, binary search by key prefix
///   4. Extract row IDs from entry suffix
class SortedIndex : public IndexLookup {
 public:
  static constexpr std::string_view kNumPointLookups{
      "sortedIndex.numPointLookups"};
  static constexpr std::string_view kNumRangeScans{"sortedIndex.numRangeScans"};
  static constexpr std::string_view kNumMinMaxSkips{
      "sortedIndex.numMinMaxSkips"};
  static constexpr std::string_view kNumMatchedRows{
      "sortedIndex.numMatchedRows"};

  const std::vector<std::string>& indexColumns() const override {
    return columns_;
  }

  LookupResult lookup(const LookupRequest& request) const override;

  std::string_view minKey() const override;

  std::string_view maxKey() const override;

  folly::F14FastMap<std::string, velox::RuntimeMetric> stats() const override;

  static std::unique_ptr<SortedIndex> create(
      std::vector<std::string> columns,
      std::unique_ptr<MetadataBuffer> indexMetadata,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      velox::memory::MemoryPool* pool);

 private:
  SortedIndex(
      std::vector<std::string> columns,
      std::unique_ptr<MetadataBuffer> indexMetadata,
      std::shared_ptr<velox::dwio::common::BufferedInput> dataInput,
      velox::memory::MemoryPool* pool);

  // Finds the chunk index for the given key via binary search on chunk keys.
  // Returns the index of the first chunk whose last key >= the given key,
  // or std::nullopt if no chunk contains the key.
  std::optional<uint32_t> findChunkIndex(std::string_view key) const;

  // Returns the last key of a chunk (used as the chunk boundary key).
  std::string_view chunkKey(uint32_t chunkIdx) const;

  // Returns the byte offset of a chunk in the key stream.
  uint32_t chunkOffset(uint32_t chunkIdx) const;

  // Returns the byte size of a chunk.
  uint32_t chunkSize(uint32_t chunkIdx) const;

  // Returns the row offset (number of entries before this chunk).
  uint32_t rowOffset(uint32_t chunkIdx) const;

  // Loads a chunk and creates the encoding without materializing entries.
  std::shared_ptr<DecodedKeyChunk> loadChunk(uint32_t chunkIdx) const;

  // Extracts the key prefix (without row ID suffix) from a composite entry.
  std::string_view extractKey(std::string_view compositeEntry) const;

  // Extracts the row ID from the last rowIdWidth_ bytes of a composite entry.
  uint32_t extractRowId(std::string_view compositeEntry) const;

  // Builds a composite seek key with minimum row ID suffix (all zeros).
  // Used as inclusive lower bound in binary search.
  std::string firstSeekKey(std::string_view key) const;

  // Builds a composite seek key with maximum row ID suffix (all 0xFF).
  // Used as exclusive upper bound in binary search.
  std::string lastSeekKey(std::string_view key) const;

  // Performs a point lookup for a single key and appends matching row ranges.
  void pointLookup(
      std::string_view key,
      const std::optional<RowRange>& filterRange,
      std::vector<RowRange>& rowRanges) const;

  // Performs a range scan and appends matching row ranges.
  void rangeScan(
      const velox::serializer::EncodedKeyBounds& bounds,
      const std::optional<RowRange>& filterRange,
      std::vector<RowRange>& rowRanges) const;

  // Scans entries across chunks and appends matching row ranges.
  // boundKey controls chunk iteration: for point lookup this is the lookup key,
  // for range scan this is the upper bound key.
  void scanEntries(
      uint32_t startChunk,
      std::string_view startSeekKey,
      std::string_view endSeekKey,
      std::string_view boundKey,
      bool isPointLookup,
      const std::optional<RowRange>& filterRange,
      std::vector<RowRange>& rowRanges) const;

  const std::vector<std::string> columns_;
  const std::unique_ptr<MetadataBuffer> indexMetadata_;
  velox::memory::MemoryPool* const pool_;
  const uint8_t rowIdWidth_;

  const std::shared_ptr<velox::dwio::common::BufferedInput> dataInput_;

  // Parsed from the FlatBuffer.
  const serialization::SortedIndex* const sortedIndex_;
  const uint32_t numChunks_;

  // Reusable buffer for multi-buffer chunk reads.
  mutable velox::BufferPtr chunkBuffer_;

  mutable std::atomic_uint64_t numPointLookups_{0};
  mutable std::atomic_uint64_t numRangeScans_{0};
  mutable std::atomic_uint64_t numMinMaxSkips_{0};
  mutable std::atomic_uint64_t numMatchedRows_{0};
};

} // namespace facebook::nimble::index
