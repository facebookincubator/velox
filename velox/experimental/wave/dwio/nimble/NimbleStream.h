/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <string>
#include <string_view>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/experimental/wave/dwio/decode/DecodeOp.h"

namespace facebook::velox::wave {

/// Host side description of the chunks and encoding trees of one Nimble
/// stream in a stripe, made from the encoding headers without decoding values.
/// The chunk and node arrays are copied to device as is.
struct NimbleStream {
  /// Parses the chunks of 'data', the bytes of one stream. Compressed leaf
  /// payloads are decompressed with memory from 'pool' into 'decompressed'.
  /// Throws if an encoding has no GPU implementation. If 'isNulls' is true,
  /// the chunks are bool encodings of the nulls of a struct and have no
  /// accessors.
  static NimbleStream
  parse(std::string_view data, memory::MemoryPool* pool, bool isNulls = false);

  /// Offset of 'decompressed' from the start of the stream on device. Node
  /// offsets into decompressed payloads include this.
  int64_t decompressedOffset() const {
    return bits::roundUp(streamSize, sizeof(int64_t));
  }

  /// Returns the accessor types of the chunks as a comma separated list. The
  /// shape of a chunk is the position of its accessor in the list.
  std::string accessorList() const;

  std::vector<ChunkDesc> chunks;
  std::vector<NimbleNode> nodes;
  /// Distinct accessor types of the chunks' trees, e.g. "NFbw<uint64_t>".
  std::vector<std::string> accessors;
  /// Nodes that need a prepare op before values can be read: Nullable,
  /// MainlyConstant, RLE and Trivial strings. A node comes after the nodes
  /// below it.
  std::vector<int32_t> prepareNodes;
  /// Decompressed payloads of compressed leaves. Copied to device after the
  /// stream at decompressedOffset().
  std::string decompressed;
  int64_t streamSize{0};
  int32_t numRows{0};
  /// True if a chunk is Nullable.
  bool hasNulls{false};
};

} // namespace facebook::velox::wave
