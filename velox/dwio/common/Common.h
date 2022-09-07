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

#include <folly/Range.h>
#include <string>
//#include "DecoderUtil.h"

namespace facebook::velox::dwio::common {

using RowSet = folly::Range<const int32_t*>;

enum CompressionKind {
  CompressionKind_NONE = 0,
  CompressionKind_ZLIB = 1,
  CompressionKind_SNAPPY = 2,
  CompressionKind_LZO = 3,
  CompressionKind_ZSTD = 4,
  CompressionKind_LZ4 = 5,
  CompressionKind_MAX = INT64_MAX
};

/**
 * Get the name of the CompressionKind.
 */
std::string compressionKindToString(CompressionKind kind);

constexpr uint64_t DEFAULT_COMPRESSION_BLOCK_SIZE = 256 * 1024;

enum class RowSetDensity {
  kFull = 0,
  kDense = 1,
  kSparse = 2

};

enum class RowSetNullability { kNoNull = 0, kPartialNulls = 1, kAllNulls = 2 };

static FOLLY_ALWAYS_INLINE RowSetNullability rowSetNullability(
    const RowSet& rows,
    const uint64_t* FOLLY_NULLABLE nulls,
    uint64_t nullsOffset,
    uint64_t totalNumNulls) {
  uint32_t numRows = rows.size();
  uint64_t totalNumRows = rows[numRows - 1] + 1;
}

} // namespace facebook::velox::dwio::common
