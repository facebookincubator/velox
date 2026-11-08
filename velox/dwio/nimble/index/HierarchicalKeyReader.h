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
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/index/HierarchicalClusterIndexFactory.h"
#include "velox/dwio/nimble/index/KeyReader.h"

namespace facebook::nimble::index {

namespace detail {
class ChildOffsets;
class KeySequence;
} // namespace detail

/// Reads one hierarchical chunk for a sorted integral composite key.
///
/// Each column level holds one encoded sequence of the distinct values under
/// every parent value plus child offsets locating each parent's run of
/// children. A lookup descends one level at a time without rebuilding packed
/// keys for comparisons.
///
///   columns:  [7, 7, 7, 9]         [1, 1, 4, 2]
///   level 0:  [7, 9]               distinct values of column 0
///   level 1:  [1, 4] and [2]       children of 7, then of 9
///   offsets:  [0, 2, 3]            child runs of the level-0 values
///   runs:     [0, 2, 3, 4]         rows covered by each leaf value
///
/// Reads speak the keys of the reader's codec: by default the packed key
/// format that the rest of the index uses, one zero flag byte followed by the
/// native-width big-endian value bytes for each column. `get()` rebuilds a key
/// and `seek()` accepts one.
class HierarchicalKeyReader final : public KeyReader {
 public:
  /// Opens a serialized index without materializing its keys. With
  /// 'cacheEncodingViews', every sequence's view keeps the lookup structures
  /// of Encoding::Options::cacheEncodingView, which suits a chunk that stays
  /// resident and serves many lookups. 'keyCodec' defines the keys that reads
  /// take and return for the values of the levels.
  HierarchicalKeyReader(
      std::string_view data,
      bool cacheEncodingViews,
      std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
      velox::memory::MemoryPool* pool);

  ~HierarchicalKeyReader() override;

  /// Returns the first row at or after 'value' when inclusive, or the first
  /// row past it otherwise. Repeated keys form one run, so the two forms
  /// bracket that run.
  std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const override;

  std::string get(uint32_t row) const override;

  /// Appends the keys of the 'numRows' rows from 'startRow' to 'keys', and
  /// writes their bytes to one allocation of 'bufferAllocator'. Walks the
  /// levels of consecutive rows together rather than finding those of each
  /// row, so a scan pays for a value of a level once per change of the value.
  void readKeys(
      uint32_t startRow,
      uint32_t numRows,
      std::vector<std::string_view>& keys,
      const KeyCursor::BufferAllocator& bufferAllocator) const;

  std::vector<std::string> materialize(uint32_t startRow, uint32_t count)
      const override;

  std::unique_ptr<KeyCursor> cursor(uint32_t startRow) const override;

  uint32_t rowCount() const override;

  /// Returns the number of key columns.
  uint32_t numLevels() const;

  /// Returns the encoding of the values sequence at `level`. Test-only.
  EncodingType testingValuesEncodingType(uint32_t level) const;

 private:
  // Reads the byte width of each level.
  void readLevelWidths(const char*& position, const char* end);

  // Reads the sequences in serialized order: level-0 values, then per level
  // the child offsets above it and its values, then leaf row offsets when keys
  // repeat.
  void readSequences(
      const char*& position,
      const char* end,
      std::span<const uint32_t> levelCounts,
      velox::memory::MemoryPool* pool,
      bool cacheEncodingViews);

  // Returns the first source row covered by a component subtree.
  // 'componentIndex' may be one past the level's last value, which returns
  // rowCount_.
  uint32_t startRowOfSubtree(uint32_t level, uint32_t componentIndex) const;

  // Returns [startRow, endRow) for an exact key match. A missing key returns
  // [insertionRow, insertionRow).
  std::pair<uint32_t, uint32_t> seekBounds(
      std::span<const uint64_t> target) const;

  // Stores the normalized key components of a source row in 'levels', which
  // holds numLevels() values.
  void levelsAt(uint32_t row, std::span<uint64_t> levels) const;

  // Stores in 'levels' the values of the levels of the 'numRows' rows from
  // 'startRow', those of each row after those of the row before. Reads the
  // values that the rows span at each level, and their child offsets, together
  // rather than one by one where the level changes.
  void levelsOf(uint32_t startRow, uint32_t numRows, std::span<uint64_t> levels)
      const;

  // Defines the keys that reads take and return for the values of the levels.
  const std::shared_ptr<const HierarchicalKeyCodec> keyCodec_;

  // Total source rows represented by the leaf values.
  uint32_t rowCount_{0};
  // Number of key columns and hierarchy levels.
  uint32_t numLevels_{0};
  // The byte width of each level.
  std::vector<uint8_t> levelWidths_;
  // Maps each value to its child range. The leaf level maps complete keys to
  // their source-row ranges and is the identity when every key is distinct.
  std::vector<detail::ChildOffsets> childOffsets_;
  // Holds the values sequence of each level.
  std::vector<std::unique_ptr<detail::KeySequence>> levelSequences_;
};

/// Creates a HierarchicalKeyReader; 'cacheEncodingViews' and 'keyCodec' are
/// forwarded to it.
std::unique_ptr<KeyReader> createHierarchicalKeyReader(
    std::string_view encodedKeys,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    bool cacheEncodingViews,
    std::shared_ptr<const HierarchicalKeyCodec> keyCodec,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index
