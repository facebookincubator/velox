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
#include "velox/dwio/nimble/index/KeyCursor.h"

namespace facebook::nimble::index {

class KeyEncoding;

/// Immutable random-access reader for one decoded key chunk.
class KeyReader {
 public:
  virtual ~KeyReader() = default;

  /// Returns the first row whose key is greater than or equal to `value` when
  /// `inclusive` is true, or strictly greater when false.
  virtual std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const = 0;

  /// Returns the encoded key at `row`.
  virtual std::string get(uint32_t row) const = 0;

  /// Returns owned encoded keys for `[startRow, startRow + count)`.
  virtual std::vector<std::string> materialize(
      uint32_t startRow,
      uint32_t count) const = 0;

  /// Returns a forward cursor positioned at `startRow`.
  virtual std::unique_ptr<KeyCursor> cursor(uint32_t startRow) const = 0;

  /// Returns the number of encoded keys.
  virtual uint32_t rowCount() const = 0;
};

/// Adapts a flat packed-key encoding to the key reader interface.
class FlatKeyReader final : public KeyReader {
 public:
  explicit FlatKeyReader(std::unique_ptr<KeyEncoding> encoding);
  ~FlatKeyReader() override;

  std::optional<uint32_t> seek(std::string_view value, bool inclusive)
      const override;

  std::string get(uint32_t row) const override;

  std::vector<std::string> materialize(uint32_t startRow, uint32_t count)
      const override;

  std::unique_ptr<KeyCursor> cursor(uint32_t startRow) const override;

  uint32_t rowCount() const override;

 private:
  const std::unique_ptr<KeyEncoding> encoding_;
};

/// Opens a flat packed-key reader.
std::unique_ptr<KeyReader> createFlatKeyReader(
    std::string_view encodedKeys,
    std::function<void*(uint32_t)> stringBufferFactory,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index
