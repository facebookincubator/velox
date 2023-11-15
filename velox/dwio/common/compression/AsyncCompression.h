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

#ifndef ASYNC_COMPRESSION_H_
#define ASYNC_COMPRESSION_H_

#include <folly/futures/Future.h>
#include "velox/common/compression/Compression.h"

namespace facebook::velox::dwio::common::compression {

using facebook::velox::common::CompressionKind;

class AsyncDecompressor {
 public:
  explicit AsyncDecompressor(){};

  virtual ~AsyncDecompressor() = default;

  virtual folly::SemiFuture<uint64_t> decompressAsync(
      const char* src,
      uint64_t srcLength,
      char* dest,
      uint64_t destLength) = 0;
};

std::unique_ptr<AsyncDecompressor> MakeIAAGzipCodec();

/**
 * Get the window size from zlib header(rfc1950).
 * 0   1
 * +---+---+
 * |CMF|FLG|   (more-->)
 * +---+---+
 * bits 0 to 3  CM     Compression method
 * bits 4 to 7  CINFO  Compression info
 * CM (Compression method) This identifies the compression method used in the
 * file. CM = 8 denotes the "deflate" compression method with a window size up
 * to 32K. CINFO (Compression info) For CM = 8, CINFO is the base-2 logarithm of
 * the LZ77 window size, minus eight (CINFO=7 indicates a 32K window size).
 * @param stream_ptr the compressed block length for raw decompression
 * @param stream_size compression options to use
 */
static int getZlibWindowBits(const uint8_t* stream_ptr, uint32_t stream_size) {
  static constexpr uint8_t CM_ZLIB_DEFAULT_VALUE = 8u;
  static constexpr uint32_t ZLIB_MIN_HEADER_SIZE = 2u;
  static constexpr uint32_t ZLIB_INFO_OFFSET = 4u;
  if (stream_size < ZLIB_MIN_HEADER_SIZE) {
    return -1;
  }
  const uint8_t compression_method_and_flag = *stream_ptr++;
  const uint8_t compression_method = compression_method_and_flag & 0xf;
  const uint8_t compression_info =
      compression_method_and_flag >> ZLIB_INFO_OFFSET;

  if (CM_ZLIB_DEFAULT_VALUE != compression_method) {
    return -1;
  }
  if (compression_info > 7) {
    return -1;
  }
  return CM_ZLIB_DEFAULT_VALUE + compression_info;
}

/**
 * Create a decompressor for the given compression kind in asynchronous mode.
 * @param kind the compression type to implement
 */
static std::unique_ptr<dwio::common::compression::AsyncDecompressor>
createAsyncDecompressor(facebook::velox::common::CompressionKind kind) {
  switch (static_cast<int64_t>(kind)) {
#ifdef VELOX_ENABLE_INTEL_IAA
    case CompressionKind::CompressionKind_GZIP:
      return MakeIAAGzipCodec();
#endif
    default:
      LOG(WARNING) << "Asynchronous mode not support for compression codec  "
                   << kind;
      return nullptr;
  }
  return nullptr;
}
} // namespace facebook::velox::dwio::common::compression

#endif