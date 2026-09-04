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
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "velox/dwio/nimble/tablet/MetadataBuffer.h"

namespace facebook::nimble::index {

enum class IndexFamily : uint8_t {
  Cluster,
  Dense,
};

std::string_view toString(IndexFamily family);

struct IndexDescriptor {
  IndexFamily family;
  std::string name;
  MetadataSection root;
};

/// Represents the location of a chunk within a stream.
/// Offsets are relative to the containing unit:
/// - For chunk index: relative to the stripe (stream start offset and stripe
///   start row).
/// - For cluster index: relative to the index partition (key data blob offset
///   and partition start row).
/// chunkIndex is the absolute position within the FlatBuffers array searched
/// during lookup:
/// - For chunk index: position in the stripe-scoped chunkRows array (offset
///   by the number of chunks belonging to earlier streams in the stripe).
/// - For cluster index: position in the partition-scoped chunk_keys array
///   (equivalently, partition-relative since the array is per-partition);
///   used to index into the partition's per-chunk cache.
struct ChunkLocation {
  uint32_t chunkIndex;
  uint32_t chunkOffset;
  uint32_t chunkSize;
  uint32_t rowOffset;

  ChunkLocation(
      uint32_t _chunkIndex,
      uint32_t _chunkOffset,
      uint32_t _chunkSize,
      uint32_t _rowOffset)
      : chunkIndex(_chunkIndex),
        chunkOffset(_chunkOffset),
        chunkSize(_chunkSize),
        rowOffset(_rowOffset) {}
};

} // namespace facebook::nimble::index

template <>
struct fmt::formatter<facebook::nimble::index::IndexFamily>
    : formatter<std::string_view> {
  auto format(
      facebook::nimble::index::IndexFamily family,
      format_context& context) const {
    return formatter<std::string_view>::format(
        facebook::nimble::index::toString(family), context);
  }
};
