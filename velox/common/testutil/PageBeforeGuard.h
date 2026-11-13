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

#include <cstddef>
#include <cstdint>

namespace facebook::velox::common::testutil {

/// Maps a readable and writable page followed by an inaccessible one, so that a
/// read past the end of the page faults instead of quietly succeeding. Tests
/// place a buffer so that its valid bytes, rounded up to its alignment, end at
/// end() to check that code stays within it, as memory from outside Velox may
/// have no padding. ASan doesn't check reads by x86 gather instructions, such
/// as the one in simd::gather8Bits, so it misses those over-reads. A guard page
/// catches them, and in builds without ASan too.
class PageBeforeGuard {
 public:
  PageBeforeGuard();

  ~PageBeforeGuard();

  PageBeforeGuard(const PageBeforeGuard&) = delete;
  PageBeforeGuard& operator=(const PageBeforeGuard&) = delete;

  /// Returns the end of the accessible page, which is the start of the guard.
  uint8_t* end() const {
    return begin_ + pageSize_;
  }

 private:
  const size_t pageSize_;
  // Start of the accessible page and of the mapping.
  uint8_t* begin_{nullptr};
};

} // namespace facebook::velox::common::testutil
