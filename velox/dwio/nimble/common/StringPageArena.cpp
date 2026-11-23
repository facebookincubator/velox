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

#include "velox/dwio/nimble/common/StringPageArena.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace facebook::nimble {

StringPageArena::StringPageArena(
    velox::memory::MemoryPool* pool,
    size_t minPageBytes,
    std::span<char> inlinePage)
    : pool_{pool}, minPageBytes_{minPageBytes}, scratch_{pool} {
  NIMBLE_CHECK_NOT_NULL(pool, "Memory pool must not be null.");
  NIMBLE_CHECK_GT(minPageBytes_, 0, "Minimum page size must be positive.");
  if (!inlinePage.empty()) {
    pages_.emplace_back(
        Page{
            .data = inlinePage.data(),
            .bytes = inlinePage.size(),
            .buffer = nullptr,
        });
    free_ = inlinePage.data();
    pageEnd_ = free_ + inlinePage.size();
  }
}

StringPageArena::StringPageArena(
    velox::memory::MemoryPool* pool,
    PageFactory pageFactory,
    size_t minPageBytes)
    : pool_{nullptr},
      pageFactory_{std::move(pageFactory)},
      minPageBytes_{minPageBytes},
      scratch_{pool} {
  NIMBLE_CHECK_NOT_NULL(pool, "Memory pool must not be null.");
  NIMBLE_CHECK_GT(minPageBytes_, 0, "Minimum page size must be positive.");
}

void StringPageArena::reset() {
  pageIndex_ = 0;
  if (pages_.empty()) {
    return;
  }
  free_ = pages_.front().data;
  pageEnd_ = free_ + pages_.front().bytes;
}

char* StringPageArena::nextPage(size_t minBytes) {
  const auto pageBytes = std::max(minPageBytes_, minBytes);
  const size_t index = pages_.empty() ? 0 : pageIndex_ + 1;
  if (index == pages_.size()) {
    pages_.emplace_back(newPage(pageBytes));
  } else if (pages_[index].bytes < pageBytes) {
    // No string has used the kept page since the last reset(), so replacing
    // it frees a pool page safely.
    pages_[index] = newPage(pageBytes);
  }
  pageIndex_ = index;
  free_ = pages_[index].data;
  pageEnd_ = free_ + pages_[index].bytes;
  return free_;
}

StringPageArena::Page StringPageArena::newPage(size_t bytes) {
  if (pool_ != nullptr) {
    auto buffer = velox::AlignedBuffer::allocateExact<char>(bytes, pool_);
    auto* const data = buffer->asMutable<char>();
    return Page{.data = data, .bytes = bytes, .buffer = std::move(buffer)};
  }
  NIMBLE_CHECK_NOT_NULL(pageFactory_, "String page factory is empty.");

  // The factory takes a uint32 size. Strings come from decoding file bytes,
  // so a string that needs a larger page comes from a corrupt file.
  NIMBLE_CHECK_FILE_LE(
      bytes,
      std::numeric_limits<uint32_t>::max(),
      "String page exceeds the supported size.");
  auto* const data =
      static_cast<char*>(pageFactory_(static_cast<uint32_t>(bytes)));
  NIMBLE_CHECK_NOT_NULL(data, "String page factory returned null.");
  return Page{.data = data, .bytes = bytes, .buffer = nullptr};
}

} // namespace facebook::nimble
