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
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Vector.h"

namespace facebook::nimble {

/// Keeps strings at fixed addresses in pages, so that the views a decoder
/// returns stay valid while it decodes more. Pages come from a memory pool,
/// which the arena owns, or from a factory such as an output vector's string
/// buffers, which owns them. Strings never move. They stay valid until
/// reset(), or until the owner of their page frees it.
///
///   StringPageArena arena{pool, /*minPageBytes=*/16 << 10, /*inlinePage=*/{}};
///   const auto value = arena.write(
///       /*maxBytes=*/compressedSize * 8 + 1,
///       [&](char* output) { return decode(compressed, output); });
///
/// Not thread-safe. A caller that shares an arena locks around it.
class StringPageArena {
 public:
  /// Returns a writable page of 'bytes' bytes. The owner of the factory keeps
  /// the page valid for as long as it uses the strings in it.
  using PageFactory = std::function<void*(uint32_t bytes)>;

  /// Allocates pages from 'pool' and frees them on destruction. A page has
  /// 'minPageBytes', or more for a string that needs more. Puts the first
  /// strings in 'inlinePage' unless it is empty, so that a decoder that keeps
  /// a few strings allocates nothing from the pool. The owner keeps
  /// 'inlinePage' alive and in place for the lifetime of the arena.
  StringPageArena(
      velox::memory::MemoryPool* pool,
      size_t minPageBytes,
      std::span<char> inlinePage);

  /// Takes pages from 'pageFactory', which owns them, so that the strings
  /// outlive the arena. Uses 'pool' only for scratch space. Fails the first
  /// write that needs a page if 'pageFactory' is empty, so that a decoder that
  /// writes no strings needs no factory.
  StringPageArena(
      velox::memory::MemoryPool* pool,
      PageFactory pageFactory,
      size_t minPageBytes);

  StringPageArena(const StringPageArena&) = delete;
  StringPageArena(StringPageArena&&) = delete;
  StringPageArena& operator=(const StringPageArena&) = delete;
  StringPageArena& operator=(StringPageArena&&) = delete;
  ~StringPageArena() = default;

  /// Copies 'value' into the current page, or into the next page if the
  /// current one has no room, and returns the copy. A next page has at least
  /// max(minPageBytes, value.size()) bytes. Returns an empty string without
  /// taking a page if 'value' is empty.
  std::string_view writeString(std::string_view value);

  /// Calls 'writer' with room for 'maxBytes' bytes and keeps the first n,
  /// where n is what 'writer' returns. Writes in place when the current page
  /// has room for 'maxBytes'. Before the first page, takes one for the write
  /// if 'maxBytes' fits in minPageBytes. Otherwise writes into scratch and
  /// keeps the n bytes with writeString(), so that a loose bound neither
  /// abandons the current page's tail nor allocates a page of 'maxBytes'.
  /// Returns an empty string without calling 'writer' if 'maxBytes' is 0.
  template <typename Writer>
  std::string_view write(size_t maxBytes, Writer&& writer);

  /// Starts again at the first page and keeps every page for reuse.
  /// Invalidates every string written before.
  void reset();

 private:
  // A page that the arena writes strings into.
  struct Page {
    // The first byte of the page.
    char* data;
    // The number of bytes of the page.
    size_t bytes;
    // Owns the page if it comes from the pool. Null for the inline page and
    // for pages from the factory.
    velox::BufferPtr buffer;
  };

  // Returns whether the current page has room for 'bytes' more bytes. Before
  // the first page, 'free_' and 'pageEnd_' are both null, so only 0 bytes fit.
  bool fits(size_t bytes) const {
    return bytes <= static_cast<size_t>(pageEnd_ - free_);
  }

  // Moves to the next page, which has at least max(minPageBytes, 'minBytes')
  // bytes: the page kept there from before the last reset() if it is large
  // enough, or a new page that replaces it. Returns the page's first byte.
  char* nextPage(size_t minBytes);

  // Returns a new page of 'bytes' bytes from 'pool_', or from 'pageFactory_' if
  // 'pool_' is null.
  Page newPage(size_t bytes);

  // Allocates the pages, which the arena owns. Null when the pages come from
  // 'pageFactory_', in which case the pool passed to the constructor backs
  // only 'scratch_'.
  velox::memory::MemoryPool* const pool_;
  // Allocates the pages, which its owner owns, if 'pool_' is null.
  const PageFactory pageFactory_;
  // The size of a new page, unless a string needs a larger one.
  const size_t minPageBytes_;
  // The pages in the order the arena fills them. The pages after the current
  // one come from before the last reset() and wait for reuse.
  std::vector<Page> pages_;
  // The index of the current page in 'pages_', if 'pages_' is not empty.
  size_t pageIndex_{0};
  // The first byte of the current page that no string holds, or null before
  // the first page. Kept with 'pageEnd_' outside 'pages_', so that a write
  // checks its room without reading the page.
  char* free_{nullptr};
  // The end of the current page, or null before the first page.
  char* pageEnd_{nullptr};
  // Holds the output of a write() whose bound does not fit the current page.
  Vector<char> scratch_;
};

// Decoders call writeString() and write() for every string, so both stay in
// the header.

inline std::string_view StringPageArena::writeString(std::string_view value) {
  if (value.empty()) {
    return {};
  }
  char* const output = fits(value.size()) ? free_ : nextPage(value.size());
  std::memcpy(output, value.data(), value.size());
  free_ = output + value.size();
  return {output, value.size()};
}

template <typename Writer>
std::string_view StringPageArena::write(size_t maxBytes, Writer&& writer) {
  if (maxBytes == 0) {
    return {};
  }
  char* output = free_;
  if (!fits(maxBytes)) {
    if (pages_.empty() && maxBytes <= minPageBytes_) {
      output = nextPage(maxBytes);
    } else {
      // Writes into scratch, since the output size is known only after the
      // write. writeString() then keeps the output in the current page's tail
      // if it fits, or else in a page sized for it, so that a loose bound
      // neither abandons the tail nor takes a page of 'maxBytes'.
      scratch_.resize(maxBytes);
      output = scratch_.data();
    }
  }
  const size_t bytes = writer(output);
  NIMBLE_CHECK_LE(bytes, maxBytes, "Write exceeds its bound.");

  // The write went into scratch.
  if (output != free_) {
    return writeString({output, bytes});
  }
  free_ = output + bytes;
  return {output, bytes};
}

} // namespace facebook::nimble
