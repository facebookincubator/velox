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

#include <string_view>

namespace facebook::nimble::index {

/// Forward cursor over encoded keys in row order.
///
/// The cursor is not thread-safe. Its source must outlive it.
class KeyCursor {
 public:
  virtual ~KeyCursor() = default;

  /// Returns whether next() has another key to return.
  virtual bool hasNext() const = 0;

  /// Returns the current encoded key and advances by one row. The returned
  /// view remains valid until the next next() call or cursor destruction.
  virtual std::string_view next() = 0;
};

} // namespace facebook::nimble::index
