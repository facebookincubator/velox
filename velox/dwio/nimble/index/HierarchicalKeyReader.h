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
#include "velox/dwio/nimble/index/KeyReader.h"

namespace facebook::nimble::index {

namespace detail {
class KeySequence;
}

/// Reads one hierarchical chunk for a sorted integral composite key.
///
/// Each column level holds one monotone integer sequence per parent value plus
/// a directory locating it. A lookup descends one level at a time without
/// rebuilding packed keys for comparisons.
///
///   columns:  [7, 7, 7, 9]         [1, 1, 4, 2]
///   level 0:  [7, 9]               distinct values of column 0
///   level 1:  [1, 4] and [2]       one sequence per level-0 value
///   runs:     [0, 2, 3, 4]         rows covered by each leaf value
///
/// Reads still speak the packed key format that the rest of the index uses —
/// one zero flag byte followed by the native-width big-endian value bytes for
/// each column. `get()` rebuilds a packed key and `seek()` accepts one.
class HierarchicalKeyReader final : public KeyReader {
 public:
  /// Opens a serialized index without materializing its keys.
  HierarchicalKeyReader(std::string_view data, velox::memory::MemoryPool* pool);

  ~HierarchicalKeyReader() override;

  /// Returns the first row at or after 'value' when inclusive, or the first
  /// row past it otherwise. Repeated keys form one run, so the two forms
  /// bracket that run.
  std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const override;

  std::string get(uint32_t row) const override;

  std::vector<std::string> materialize(uint32_t startRow, uint32_t count)
      const override;

  std::unique_ptr<KeyCursor> cursor(uint32_t startRow) const override;

  uint32_t rowCount() const override;

  /// Returns the number of key columns.
  uint32_t numLevels() const;

  /// Returns each sequence encoding at `level`. Test-only.
  std::vector<EncodingType> testingSequenceEncodingTypes(uint32_t level) const;

 private:
  // Reads each level's encoded value width and computes a complete key size.
  void readKeyValueWidths(const char*& position, const char* end);

  // Reads the per-level child directories, including leaf-to-row offsets.
  void readChildOffsets(
      const char*& position,
      const char* end,
      std::span<const uint32_t> levelCounts);

  // Reads encoded sequence sizes and creates each level's sequence views.
  void readLevelSequences(
      const char*& position,
      const char* end,
      std::span<const uint32_t> levelCounts,
      size_t numSequences,
      velox::memory::MemoryPool* pool);

  // Returns the first source row covered by a component subtree.
  // 'componentIndex' may be one past the level's last value, which returns
  // rowCount_.
  uint32_t startRowOfSubtree(uint32_t level, uint32_t componentIndex) const;

  // Returns [startRow, endRow) for an exact key match. A missing key returns
  // [insertionRow, insertionRow).
  std::pair<uint32_t, uint32_t> seekBounds(
      std::span<const uint64_t> target) const;

  // Parses one complete, non-null encoded key into normalized components.
  std::vector<uint64_t> parseEncodedKey(std::string_view key) const;

  // Encodes normalized key components into the byte-comparable key format.
  std::string encodeKey(std::span<const uint64_t> components) const;

  // Reconstructs the normalized key components for a source row.
  std::vector<uint64_t> keyAt(uint32_t row) const;

  // Total source rows represented by the leaf sequences.
  uint32_t rowCount_{0};
  // Number of key columns and hierarchy levels.
  uint32_t numLevels_{0};
  // Encoded byte width of each key value.
  std::vector<uint8_t> keyValueBytes_;
  // Total bytes in one complete encoded key, including non-null markers.
  uint32_t encodedKeyBytes_{0};
  // Maps each value to its child range. The leaf level maps complete keys to
  // their source-row ranges.
  std::vector<std::vector<uint32_t>> childOffsets_;
  // Holds one monotone value sequence per parent at every hierarchy level.
  std::vector<std::vector<std::unique_ptr<detail::KeySequence>>>
      levelSequences_;
};

std::unique_ptr<KeyReader> createHierarchicalKeyReader(
    std::string_view encodedKeys,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index
