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

#include <zlib.h>
#include <cstdint>
#include <vector>

#include "velox/dwio/common/Reader.h"

namespace velox::dwio::text::compression {

using facebook::velox::common::CompressionKind;
using facebook::velox::memory::MemoryPool;

class ReaderDecompressor {
 private:
  const uint64_t iBufSize = (1ull << 20); // 1 MB
  const uint64_t bufSize = (3 * iBufSize); // 3 MB

  // PreloadableReader& iStream;
  uint64_t iStreamLength_;
  CompressionKind kind;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
  MemoryPool& pool;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  bool hiveDefaultMode;
  std::vector<uint8_t> iBuf;
  uint64_t iBufLen;
  uint64_t iBufPos; // offset of first byte in iBuf
  uint64_t iPos;
  z_stream strm;
  std::vector<uint8_t> buf;
  uint64_t bufLen;
  uint64_t bufPos; // offset of first byte in buf
  uint64_t pos; // offset of position in buf
  bool atEOF;
  uint64_t fileLength;

  void initZlib();
  void endZlib();
  bool iBufRefill();
  bool bufRefill();

 public:
  ReaderDecompressor(
      // PreloadableReader& s,
      CompressionKind k,
      MemoryPool& p,
      bool hiveDefaultMode = false);

  ~ReaderDecompressor();

  /**
   * Get the total length of the file in bytes.
   */
  uint64_t getLength() const;

  /**
   * Get the natural size for reads.
   * @return the number of bytes that should be read at once
   */
  uint64_t getNaturalReadSize() const;

  /**
   * Read length bytes from the file starting at offset into
   * the buffer starting at buf.
   * @param buf the starting position of a buffer.
   * @param length the number of bytes to read.
   * @param offset the position in the stream to read from.
   */
  void read(
      void* buf,
      uint64_t length,
      uint64_t offset,
      facebook::velox::dwio::common::LogType);
};

} // namespace velox::dwio::text::compression
