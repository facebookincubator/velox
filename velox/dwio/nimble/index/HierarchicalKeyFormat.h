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
///
/// Sequences, in order
/// +------------------------+--------------------------------+
/// | sequenceSize           | values[0]                      |
/// | u32                    | valueCount[0] values           |
/// +------------------------+--------------------------------+
/// | for each level N in [1, numLevels):                     |
/// +------------------------+--------------------------------+
/// | sequenceSize           | childOffsets[N - 1]            |
/// | u32                    | valueCount[N - 1] + 1 values   |
/// +------------------------+--------------------------------+
/// | sequenceSize           | values[N]                      |
/// | u32                    | valueCount[N] values           |
/// +------------------------+--------------------------------+
/// | sequenceSize           | leafRowOffsets (present when   |
/// | u32                    | rowCount > leaf value count)   |
/// |                        | valueCount[lastLevel] + 1      |
/// +------------------------+--------------------------------+
/// ```
///
/// Level 0 holds the distinct values of the first key column. Each value at
/// level N owns a run of children at level N + 1, and childOffsets[N] delimits
/// those runs. Leaf row offsets map each distinct complete key to the source
/// rows it represents.
///
/// values[N] concatenates every run of level N, so a level takes one encoding
/// header however many runs it has. Readers validate the child offsets but
/// trust each run of values to ascend, as the writer enforces key order.
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
