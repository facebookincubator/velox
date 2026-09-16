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
#include <string_view>
#include <vector>

namespace facebook::nimble {

/// Represents a single chunk of encoded data within a stream.
/// A stream may be divided into multiple chunks to control memory usage during
/// writing. Each chunk contains the encoded data and row count.
struct Chunk {
  uint32_t rowCount{0};

  /// Number of null values in this chunk. Only populated when chunk statistics
  /// are enabled (WriterOptions::enableChunkIndex); left at 0 otherwise.
  uint32_t nullCount{0};

  /// The encoded and compressed data content of this chunk, stored as a vector
  /// of string views. Each string_view points to a buffer containing a portion
  /// of the chunk's data. Multiple buffers may be used for large chunks.
  std::vector<std::string_view> content;

  /// Returns the total byte size of all content in this chunk.
  uint32_t contentSize() const {
    uint32_t size = 0;
    for (const auto& c : content) {
      size += c.size();
    }
    return size;
  }
};

} // namespace facebook::nimble
