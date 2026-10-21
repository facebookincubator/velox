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

#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/index/SortOrder.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::index {

class IndexKeyEncoder;

/// Buffers one partition's keys and encodes row ranges as key chunks.
class KeyChunkBuilder {
 public:
  virtual ~KeyChunkBuilder() = default;

  /// Buffers the keys of `input` and validates their order when configured.
  virtual void append(const velox::VectorPtr& input) = 0;

  /// Returns the number of buffered keys.
  virtual size_t size() const = 0;

  /// Returns an owned byte-comparable key for boundary metadata. Ownership is
  /// required because some implementations synthesize the key on demand.
  virtual std::string keyAt(size_t row) const = 0;

  /// Encodes buffered rows [offset, offset + count) into `buffer`.
  virtual std::string_view encode(size_t offset, uint32_t count, Buffer& buffer)
      const = 0;

  /// Drops buffered keys while retaining cross-clear ordering state.
  virtual void clear() = 0;
};

/// Encodes packed keys for both flat cluster and sorted index writers using a
/// Prefix or Trivial encoding layout.
std::string_view encodeFlatKeys(
    const EncodingLayout& layout,
    std::span<const std::string_view> keys,
    Buffer& buffer);

/// Creates a builder for flat Prefix or Trivial encoded keys.
std::unique_ptr<KeyChunkBuilder> createFlatKeyChunkBuilder(
    std::unique_ptr<IndexKeyEncoder> keyEncoder,
    EncodingLayout encodingLayout,
    bool enforceKeyOrder,
    bool noDuplicateKey,
    velox::memory::MemoryPool* pool);

} // namespace facebook::nimble::index
