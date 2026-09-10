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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble::index {

/// Thread-safe random-access key store for index lookups.
///
/// Provides seek(), get(), and materialize() for key-to-row resolution.
/// Fully decoupled from the sequential Encoding API — takes raw encoded
/// data directly and handles its own parsing internally.
/// Implementations are immutable after construction — no locks needed for
/// concurrent access.
///
/// Two implementations:
///   TrivialKeyEncoding: pre-materializes all values for O(log N) binary
///       search and O(1) get/materialize.
///   PrefixKeyEncoding: memory-compact, decodes on the fly from restart
///       points for O(log R + I) seek and O(I) get (R = restarts, I =
///       restart interval).
class KeyEncoding {
 public:
  virtual ~KeyEncoding() = default;

  /// Creates the appropriate KeyEncoding from raw encoded data.
  ///
  /// Internally creates a temporary Encoding to parse/decompress the data,
  /// extracts what it needs, then discards the temporary Encoding. The
  /// returned KeyEncoding does not depend on any Encoding object.
  ///
  /// String data buffers are allocated via stringBufferFactory. The caller
  /// must ensure these buffers outlive the returned KeyEncoding (typically
  /// owned by DecodedKeyChunk::stringBuffers).
  static std::unique_ptr<KeyEncoding> create(
      velox::memory::MemoryPool& pool,
      std::string_view encodedData,
      std::function<void*(uint32_t)> stringBufferFactory);

  /// Returns the row index of the first key matching the target.
  /// @param inclusive When true, returns first row >= value.
  ///        When false, returns first row > value.
  /// @return Row index if found, std::nullopt if no matching row exists.
  virtual std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const = 0;

  /// Returns the key at the given row index.
  virtual std::string get(uint32_t row) const = 0;

  /// Materializes keys in [startRow, startRow + count) as owned strings.
  /// Thread-safe — each call uses only local state.
  virtual std::vector<std::string> materialize(
      uint32_t startRow,
      uint32_t count) const = 0;

  virtual EncodingType encodingType() const = 0;

  virtual uint32_t rowCount() const = 0;
};

/// Pre-materialized key encoding for TrivialEncoding-backed keys.
/// All values are materialized at construction into an immutable sorted
/// vector for fast binary search and direct index access.
class TrivialKeyEncoding final : public KeyEncoding {
 public:
  explicit TrivialKeyEncoding(std::vector<std::string_view> values);

  std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const override;

  std::string get(uint32_t row) const override;

  std::vector<std::string> materialize(uint32_t startRow, uint32_t count)
      const override;

  EncodingType encodingType() const override {
    return EncodingType::Trivial;
  }

  uint32_t rowCount() const override {
    return static_cast<uint32_t>(values_.size());
  }

 private:
  const std::vector<std::string_view> values_;
};

/// Memory-compact key encoding for PrefixEncoding-backed keys.
/// No pre-materialization — decodes on the fly from const data using
/// restart-point binary search and linear scan with local variables.
class PrefixKeyEncoding final : public KeyEncoding {
 public:
  PrefixKeyEncoding(
      std::string_view encodedData,
      uint32_t rowCount,
      uint32_t dataOffset);

  std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const override;

  std::string get(uint32_t row) const override;

  std::vector<std::string> materialize(uint32_t startRow, uint32_t count)
      const override;

  EncodingType encodingType() const override {
    return EncodingType::Prefix;
  }

  uint32_t rowCount() const override {
    return rowCount_;
  }

 private:
  static std::string_view
  decodeEntryAt(const char*& pos, uint32_t& row, std::string& decoded);

  uint32_t restartOffset(uint32_t restartIndex) const;

  const char* restartPosition(uint32_t restartIndex) const {
    return dataStart_ + restartOffset(restartIndex);
  }

  const uint32_t rowCount_;
  const uint32_t restartInterval_;
  const uint32_t numRestarts_;
  const char* const restartOffsets_;
  const char* const dataStart_;
};

} // namespace facebook::nimble::index
