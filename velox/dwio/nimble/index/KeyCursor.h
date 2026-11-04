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

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace facebook::nimble::index {

/// Forward cursor over encoded keys in row order.
///
/// The cursor is not thread-safe. Its source must outlive it.
class KeyCursor {
 public:
  /// Returns stable, non-null memory of the given size for a batch of keys.
  using BufferAllocator = std::function<void*(size_t)>;

  virtual ~KeyCursor() = default;

  /// Returns whether next() has another key to return.
  virtual bool hasNext() const = 0;

  /// Returns the current encoded key and advances by one row. The returned
  /// view remains valid until the next next() call or cursor destruction.
  virtual std::string_view next() = 0;

  /// Appends up to 'maxKeys' keys from the current row to 'keys' and advances
  /// past them, which costs less per key than next() for a scan. Writes the
  /// bytes of the keys to memory from 'bufferAllocator', which the caller
  /// owns, so the views remain valid as long as that memory does. A call may
  /// allocate more than once, and each allocation must stay valid. Appends
  /// fewer keys only at the end of the cursor, and none past it.
  virtual void next(
      size_t maxKeys,
      std::vector<std::string_view>& keys,
      const BufferAllocator& bufferAllocator) = 0;
};

} // namespace facebook::nimble::index
