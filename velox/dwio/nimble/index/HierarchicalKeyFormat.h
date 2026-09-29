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

namespace facebook::nimble::index {

/// Defines the serialized format shared by the hierarchical key writer and
/// reader.
///
/// The cluster-index descriptor selects the hierarchical reader, so the chunk
/// does not need a separate type marker.
///
/// ```
/// Fixed header (6 bytes)
/// +---------+-----------+----------------+
/// | version | numLevels | rowCount (u32) |
/// |   u8    |    u8     |                |
/// +---------+-----------+----------------+
///
/// Variable metadata
/// +------------------------+--------------------------------+
/// | physicalWidth[level]   | valueCount[level]              |
/// | u8 * numLevels         | u32 * numLevels                |
/// +------------------------+--------------------------------+
/// | childOffsets[level] for levels [0, numLevels - 1)       |
/// | u32 * (valueCount[level] + 1) per level                 |
/// +---------------------------------------------------------+
/// | leafRowOffsets (present when rowCount > leaf value count) |
/// | u32 * (valueCount[lastLevel] + 1)                       |
/// +---------------------------------------------------------+
/// | sequenceSize[sequence] | encodedSequence[sequence]      |
/// | u32 * numSequences     | bytes * numSequences           |
/// +------------------------+--------------------------------+
/// ```
///
/// Level 0 has one sequence. Each value at level N owns one sequence at level
/// N + 1, so `numSequences` is 1 plus the value counts of all non-leaf levels.
/// Child offsets delimit those owned sequences. Leaf row offsets map each
/// distinct complete key to the source rows it represents.
class HierarchicalKeyFormat final {
 public:
  HierarchicalKeyFormat() = delete;

  /// Identifies the serialized format version within a hierarchical chunk.
  static constexpr uint8_t kVersion{1};
  /// Locates rowCount after the version and level-count fields.
  static constexpr uint32_t kRowCountOffset{2 * sizeof(uint8_t)};
  /// Includes the two one-byte fields and the uint32 row count.
  static constexpr uint32_t kHeaderSize{kRowCountOffset + sizeof(uint32_t)};

  /// Returns whether a level can store values with the specified byte width.
  static constexpr bool isSupportedKeyValueBytes(uint8_t bytes) {
    return bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8;
  }
};

} // namespace facebook::nimble::index
