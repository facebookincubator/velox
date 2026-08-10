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

#include <span>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

/// PrefixEncoding stores sorted string data with prefix compression. Common
/// prefixes are shared across consecutive entries to reduce storage space.
///
/// Based on prefix encoding algorithms from RocksDB, Apache Kudu, and Doris:
/// - Consecutive sorted entries typically share common prefixes
/// - Store prefix length + suffix for each entry instead of full strings
/// - Periodically store full restart points for seek operations
///
/// Binary layout:
/// - EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
/// - 4 bytes: restart interval (number of entries between restart points)
/// - ZZ bytes: restart offsets array (uint32_t array of byte offsets,
///   size = ceil(rowCount / restartInterval))
/// - XX bytes: encoded entries (prefix_len | suffix_len | suffix_data)*
///
/// The restart offsets are placed at the head of the encoding block to
/// accelerate memory access patterns for seek operations, as prefix encoding
/// is primarily used for seeking in sorted data.
///
/// The number of restarts can be computed from rowCount and restartInterval:
///   numRestarts = (rowCount + restartInterval - 1) / restartInterval
///
/// Each entry stores:
/// - shared_prefix_len (uint32): bytes shared with previous entry
/// - suffix_len (uint32): length of unique suffix
/// - suffix_data: the suffix bytes
///
/// Restart points are full entries (shared_prefix_len = 0) stored at regular
/// intervals to enable efficient seek operations.

namespace facebook::nimble {

/// Encoding for sorted string data with prefix compression and seek support.
/// Only supports std::string_view data type.
class PrefixEncoding final
    : public TypedEncoding<std::string_view, std::string_view> {
 public:
  using cppDataType = std::string_view;
  using physicalType = std::string_view;

  static constexpr uint32_t kDefaultRestartInterval = 16;
  // Config key for restart interval in EncodingSelectionResult::encodingConfig
  static constexpr std::string_view kRestartIntervalConfigKey =
      "prefix-encoding.restart-interval";

  /// Constructs a PrefixEncoding with string buffer factory support.
  /// Pre-materializes all string data into a factory-allocated buffer,
  /// ensuring string data outlives the encoding instance.
  PrefixEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;

  /// Decodes and materializes string values into the output buffer.
  ///
  /// The output string_views remain valid until the next call to materialize(),
  /// reset(), or until the encoding is destroyed. The encoding maintains an
  /// internal buffer that holds the decoded string data.
  ///
  /// @param rowCount Number of rows to materialize.
  /// @param buffer Output buffer for std::string_view values.
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Encodes sorted string values with prefix compression.
  ///
  /// @param selection Encoding selection policy.
  /// @param values Values to encode. Must be sorted in ascending order.
  /// @param buffer Output buffer for encoded data.
  /// @return View of the encoded data.
  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  std::string debugString(int offset) const final;

 private:
  // Static helper methods for initializing const members.
  // startOffset is where encoding-specific data begins (after the base prefix).
  static uint32_t readRestartInterval(
      std::string_view data,
      uint32_t startOffset);
  static const char* restartOffsets(
      std::string_view data,
      uint32_t startOffset);
  static const char*
  dataStart(std::string_view data, uint32_t startOffset, uint32_t numRestarts);

  // Computes numRestarts from rowCount and restartInterval
  static uint32_t computeNumRestarts(
      uint32_t rowCount,
      uint32_t restartInterval);

  // While encoding, extracts the restart interval from the encoding selection
  // config. Returns the configured value or kDefaultRestartInterval if not set.
  //
  // @param selection Encoding selection with config.
  // @return The restart interval value.
  static uint32_t restartInterval(
      const EncodingSelection<physicalType>& selection);

  // Decodes entry at current position and returns the full string.
  // Stores result in decodedValue_ buffer.
  //
  // NOTE: The returned string_view is invalidated on the next decodeEntry call
  // as decodedValue_ buffer is overwritten.
  std::string_view decodeEntry();

  // Calls decodeEntry() then copies the decoded value into a string page
  // allocated via stringBufferFactory_. Returns a stable string_view.
  // Used by materialize() and readWithVisitor().
  std::string_view decodeToStringBuffer();

  // Seeks to the restart point at the given index.
  void seekToRestartPoint(uint32_t restartIndex);

  // Gets the byte offset for the restart point at the given index.
  uint32_t restartOffset(uint32_t restartIndex) const;

  // Returns the data pointer for the restart point at the given index.
  const char* restartPosition(uint32_t restartIndex) const {
    return dataStart_ + restartOffset(restartIndex);
  }

  // Allocates a new string page of at least minSize bytes via
  // stringBufferFactory_.
  void allocatePage(size_t minSize);

  // Factory for allocating string buffers tracked by ChunkedDecoder.
  // Used by decodeEntry() to allocate pages for stable string storage.
  const std::function<void*(uint32_t)> stringBufferFactory_;

  // Restart interval - number of entries between restart points
  const uint32_t restartInterval_;
  // Number of restart points (computed from rowCount and restartInterval)
  const uint32_t numRestarts_;
  // Offset to start of restart offsets array
  const char* const restartOffsets_;
  // Offset to start of data section
  const char* const dataStart_;

  // Current read position in the data
  const char* currentPos_{nullptr};
  // Current row index being read
  uint32_t currentRow_{0};
  // Working buffer for prefix reconstruction. The shared prefix from the
  // previous entry is already at the beginning of this buffer, so we only
  // need to append the suffix for each new entry.
  Vector<char> decodedValue_;

  // Current string page for stable storage of decoded values.
  char* currentPage_{nullptr};
  size_t pageCapacity_{0};
  size_t pageUsed_{0};

  static constexpr size_t kStringPageSize = 256 * 1024;
};

/// Template implementation (remain in header)
///
/// TODO: tune this later if needs for regular data column storage.
template <typename V>
void PrefixEncoding::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { this->skip(toSkip); },
      // We need to decode to the string buffers such that decoder
      // can directly forward the buffers to output vector.
      [&] { return decodeToStringBuffer(); });
}

} // namespace facebook::nimble
