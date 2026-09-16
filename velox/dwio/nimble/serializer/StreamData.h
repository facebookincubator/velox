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
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/buffer/BufferPool.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/serializer/Options.h"
#include "velox/dwio/nimble/velox/SchemaTypes.h"

namespace facebook::nimble::serde {

class StreamData {
 public:
  /// Decode configuration for stream data.
  struct Options {
    /// Serialization version. Determines whether streams use legacy bytes or
    /// Nimble encodings.
    SerializationVersion version;
    /// True when encoding stream prefixes store row counts as varints.
    bool streamEncodingUsesVarintRowCount{true};
    /// Optional pool for encoding scratch buffers.
    velox::BufferPool* bufferPool{nullptr};
    /// Externally-owned decompression buffer. Required (must not be null).
    /// Owned by BatchedStreamDecoder to persist across segment transitions
    /// so the buffer capacity is reused.
    velox::BufferPtr* decompressionBuffer{nullptr};
  };

  StreamData(
      ScalarKind kind,
      std::vector<velox::BufferPtr>& stringBuffers,
      velox::memory::MemoryPool* pool,
      velox::BufferPtr* decompressionBuffer);

  /// @param kind Scalar kind for the stream data.
  /// @param data Stream data to initialize with.
  /// @param pool Memory pool for encoding buffer allocation.
  /// @param options Decode configuration (version, bufferPool).
  /// @param stringBuffers External vector where string buffers from encoding
  ///        are stored. The caller must keep the vector alive while
  ///        string_views from this StreamData are in use.
  StreamData(
      ScalarKind kind,
      std::string_view data,
      std::vector<velox::BufferPtr>& stringBuffers,
      velox::memory::MemoryPool* pool,
      const Options& options);

  struct DecodeResult {
    // Decoded output rows produced from the stream segment. For scattered
    // decode, this is the number of selected output rows, not the wider output
    // span including absent positions.
    uint32_t numOutputRows{0};
    // Non-null rows in the output range covered by this decode. For scattered
    // decode, this is in the output row domain, not the encoded row domain.
    uint32_t nonNullOutputRows{0};
    // True when this decode consumed the current physical stream segment.
    bool segmentExhausted{false};
  };

  uint32_t copyTo(char* output, uint32_t bufferSize);

  DecodeResult decodeStrings(uint32_t count, std::string_view* output);

  // Decode legacy raw fixed-width data. Legacy string streams use
  // decodeStrings().
  DecodeResult
  decodeLegacy(void* output, uint32_t offset, uint32_t count, uint32_t width);

  /// Decode nimble-encoded data to output. Dispatches to typed materialize
  /// based on width. Only valid when hasEncoding() is true.
  /// Returns decoded output rows and non-null rows in the output range.
  /// numOutputRows may be less than count if the encoding has fewer remaining
  /// rows.
  ///
  /// For nullable encodings or scattered output, getOutputNulls must return the
  /// mutable output null bitmap. If scatterOutputBitmap is provided, decoded
  /// rows and null bits are written using that scattered output layout;
  /// otherwise output is dense starting at offset.
  DecodeResult decode(
      void* output,
      uint32_t offset,
      uint32_t count,
      uint32_t width,
      const std::function<void*()>& getOutputNulls = nullptr,
      const velox::bits::Bitmap* scatterOutputBitmap = nullptr);

  /// Simplified decode for thrift decoder: decodes 'count' values to output.
  /// Uses sizeof(T) as width and offset 0.
  template <typename T>
  void decode(T* output, uint32_t count) {
    decode(output, /*offset=*/0, count, sizeof(T));
  }

  void reset(
      std::string_view data,
      SerializationVersion version,
      bool streamEncodingUsesVarintRowCount);

  /// Advance the encoded stream cursor by `count` rows without materializing
  /// output. Uses the encoding's native state-only skip primitive. Only valid
  /// when hasEncoding() is true (i.e. non-legacy streams).
  void skip(uint32_t count);

  /// Replace the external string-buffers vector this StreamData points to.
  /// Needed when the caller of a subsequent decode passes a different vector
  /// than was in effect at construction (e.g. after a skip() that used a
  /// scratch vector).
  void setStringBuffers(std::vector<velox::BufferPtr>* stringBuffers) {
    stringBuffers_ = stringBuffers;
  }

  bool hasEncoding() const {
    return encoding_ != nullptr;
  }

  /// Remaining encoded rows in the current stream segment.
  uint32_t remainingRows() const {
    NIMBLE_DCHECK_NOT_NULL(encoding_);
    return encoding_->rowCount() - readRows_;
  }

 private:
  // Initialize with data. For encoding path, creates Encoding object.
  // For legacy path, decompresses if not string/binary type.
  void init(std::string_view data);

  // Decompress legacy zstd-compressed data. Reads compression type prefix and
  // decompresses into decompressionBuffer_ if needed.
  void decompress();

  // Prepare nimble-encoded data for reading. Creates an Encoding object that
  // can materialize values on demand.
  void prepareForDecoding(std::string_view data);

  // Decode nullable encoded data, optionally using scatterOutputBitmap to map
  // encoded rows into sparse output positions.
  uint32_t decodeNullable(
      void* output,
      uint32_t offset,
      uint32_t readCount,
      uint32_t width,
      const std::function<void*()>& getOutputNulls,
      const velox::bits::Bitmap* scatterOutputBitmap);

  // Decode non-null encoded data into dense output positions.
  void decodeNonNull(
      void* output,
      uint32_t offset,
      uint32_t readCount,
      uint32_t width);

  // Materialize values from nimble encoding to typed output.
  template <typename T>
  void materialize(uint32_t count, T* output);

  velox::BufferPtr& decompressionBuf() {
    return *decompressionBuffer_;
  }

  void ensureDecompressionBuffer(size_t minBytes);

  const ScalarKind kind_{ScalarKind::Undefined};
  velox::memory::MemoryPool* const pool_{nullptr};
  // Whether nimble encoding is enabled. Non-const to allow reset() to change.
  bool encodingEnabled_{false};
  // Whether encoding headers use varint row counts (true for kLegacyCompact) or
  // fixed u32 (false for kTablet). Non-const to allow
  // reset() to change.
  bool useVarintRowCount_{true};
  // Optional pool for encoding scratch buffers. Owned externally
  // (typically by BatchedStreamDecoder) to persist across StreamData
  // lifetimes.
  velox::BufferPool* const bufferPool_{nullptr};
  // Externally-owned decompression buffer. Always non-null; owned by
  // BatchedStreamDecoder or thrift Decoder to persist across StreamData
  // lifetimes.
  velox::BufferPtr* const decompressionBuffer_{nullptr};

  const char* pos_{nullptr};
  const char* end_{nullptr};
  std::unique_ptr<Encoding> encoding_;
  // Track consumed rows for nimble encoding path.
  uint32_t readRows_{0};
  // External storage for string buffers from encoding. Owned by the caller
  // (BatchedStreamDecoder or thrift Decoder). Each buffer is allocated
  // separately to avoid pointer invalidation when the vector grows.
  std::vector<velox::BufferPtr>* stringBuffers_;
};

template <typename T>
void StreamData::materialize(uint32_t count, T* output) {
  if (count == 0) {
    return;
  }
  NIMBLE_CHECK_NOT_NULL(encoding_);
  encoding_->materialize(count, output);
  readRows_ += count;
}

} // namespace facebook::nimble::serde
