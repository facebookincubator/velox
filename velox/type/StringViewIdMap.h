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

#include "velox/common/base/RawVector.h"
#include "velox/type/StringView.h"

namespace facebook::velox {

/// Map from StringView to consecutive int32_t ids. Used for mapping
/// strings to ids when making a dictionary or when assigning value
/// ids for use as part of normalized keys for hash tables.
class StringViewIdMap {
 public:
  static constexpr int64_t kNotFound = ~0L;
  // C function pointer for filling in out-of-line string pointer when
  // adding a new non-inlined StringView to the map. This is called on
  // the StringView being looked up, its new id, caller supplied
  // context data and the place where the address newly allocated copy
  // should be placed.
  using StringCopyHook = void (*)(
      const StringView* view,
      int32_t id,
      void* extraData,
      char** place);

  /// Initializes 'this' to have space for at least 'size' items before
  /// rehashing.
  StringViewIdMap(int32_t size);

  /// Returns an id for 'view'. If makeId is true and there is no id, a new one
  /// is assigned. If makeId is false, kNotFound is returned. If the 'view' is
  /// new and is stored in 'this' and does not fit inline, tailWriter is called
  /// to create a copy in memory owned by the caller. See StringCopyHook above
  /// for parameters.
  template <bool makeId>
  int32_t
  findId(const StringView& view, void* extraData, StringCopyHook tailWriter) {
    int64_t head;
    int32_t offset;
    int32_t id;
    if (makeId && numEntries_ > maxEntries_) {
      resize(capacity_ * 2);
    }
    auto table = table_.data();
    prepare1(table, &view, head, offset);
    probe1<makeId>(table, &view, head, offset, id, extraData, tailWriter);
    return id;
  }

#define PREPARE1(n)  \
  int32_t offset##n; \
  int64_t head##n;   \
  prepare1(table, &views[indices[n]], head##n, offset##n);

#define PROBE1(n, make)   \
  probe1<make>(           \
      table,              \
      &views[indices[n]], \
      head##n,            \
      offset##n,          \
      ids[n],             \
      extraData,          \
      tailWriter);

  /// Does a batch of 8 findIds on StringViews in an array. The
  /// indices are in 'indices' and the ids are returned in
  /// 'ids'. Other parameters are as in findId(). This is noticeably
  /// faster if data is not all in cache.
  template <bool makeId>
  void findIds8(
      const StringView* views,
      const int32_t* indices,
      int32_t* ids,
      void* extraData = nullptr,
      StringCopyHook tailWriter = nullptr) {
    if (makeId && numEntries_ + 8 > maxEntries_) {
      resize(capacity_ * 2);
    }
    auto* table = table_.data();
    PREPARE1(0);
    PREPARE1(1);
    PREPARE1(2);
    PREPARE1(3);
    PREPARE1(4);
    PREPARE1(5);
    PREPARE1(6);
    PREPARE1(7);
    PROBE1(0, makeId);
    PROBE1(1, makeId);
    PROBE1(2, makeId);
    PROBE1(3, makeId);
    PROBE1(4, makeId);
    PROBE1(5, makeId);
    PROBE1(6, makeId);
    PROBE1(7, makeId);
  }

#undef PREPARE
#undef PROBE

  void clear();

 private:
  static constexpr int32_t kNoEmpty = ~0;
  static constexpr int64_t kEmpty = 0;
  static constexpr int32_t kEntrySize = 20;

  int32_t startOffset(uint32_t seed, uint64_t data) {
    return (simd::crc32U64(seed, data) & sizeMask_) * kEntrySize;
  }

  template <typename T>
  T& itemAt(void* table, int32_t offset) {
    return *reinterpret_cast<T*>(reinterpret_cast<char*>(table) + offset);
  }

  template <typename T>
  T itemAt(const void* table, int32_t offset) {
    return *reinterpret_cast<const T*>(
        reinterpret_cast<const char*>(table) + offset);
  }

  FOLLY_ALWAYS_INLINE void prepare1(
      uint8_t* table,
      const StringView* view,
      int64_t& head,
      int32_t& offset) {
    head = view->sizeAndPrefixAsInt64();
    auto size = static_cast<uint32_t>(head);
    if (size <= 4) {
      offset = startOffset(1, head);
    } else if (size <= 12) {
      auto tail = itemAt<int64_t>(view, 8);
      offset = startOffset(head >> 32, tail);
    } else {
      auto string = itemAt<char*>(view, 8);
      auto last = itemAt<int64_t>(string, size - 8);
      offset = startOffset(head >> 32, last);
    }
    __builtin_prefetch(&itemAt<int64_t>(table, offset));
  }

  template <bool makeId = true>
  FOLLY_ALWAYS_INLINE void probe1(
      uint8_t* table,
      const StringView* view,
      int64_t head,
      int32_t offset,
      int32_t& id,
      void* extraData,
      StringCopyHook tailWriter) {
    auto size = static_cast<uint32_t>(head);
    if (size <= 4) {
      for (;;) {
        auto word = itemAt<int64_t>(table, offset);
        if (word == head) {
          id = itemAt<int32_t>(table, offset + sizeof(StringView));
          return;
        } else if (word == kEmpty) {
          if (!makeId) {
            id = kNotFound;
            return;
          }
          simd::memcpy(table + offset, view, sizeof(StringView));
          id = itemAt<int32_t>(table, offset + sizeof(StringView)) =
              numEntries_++;
          return;
        }
        offset = nextOffset(offset);
      }
    } else if (size <= 12) {
      for (;;) {
        auto word = itemAt<int64_t>(table, offset);
        if (head == word &&
            itemAt<int64_t>(table, offset + 8) == itemAt<int64_t>(view, 8)) {
          id = itemAt<int32_t>(table, offset + sizeof(StringView));
          return;
        } else if (word == kEmpty) {
          if (!makeId) {
            id = kNotFound;
            return;
          }
          simd::memcpy(table + offset, view, sizeof(StringView));
          id = itemAt<int32_t>(table, offset + sizeof(StringView)) =
              numEntries_++;
          return;
        }
        offset = nextOffset(offset);
      }
    } else {
      for (;;) {
        auto word = itemAt<int64_t>(table, offset);
        if (head == word) {
          auto tableString = itemAt<char*>(table, offset + 8);
          auto string = itemAt<char*>(view, 8);
          if (memcmp(string + 4, tableString + 4, size - 4) == 0) {
            id = itemAt<int32_t>(table, offset + sizeof(StringView));
            return;
          }
        } else if (word == kEmpty) {
          if (!makeId) {
            id = kNotFound;
            return;
          }
          itemAt<int64_t>(table, offset) = head;
          id = itemAt<int32_t>(table, offset + sizeof(StringView)) =
              numEntries_++;
          tailWriter(view, id, extraData, &itemAt<char*>(table, offset + 8));
          return;
        }
        offset = nextOffset(offset);
      }
    }
  }

  uint64_t hash1(const StringView& view) {
    uint64_t sizeAndPrefix = view.sizeAndPrefixAsInt64();
    int32_t size = static_cast<int32_t>(sizeAndPrefix);
    if (size > StringView::kInlineSize) {
      auto tail = itemAt<int64_t>(view.value_.data, size - 8);
      return simd::crc32U64(sizeAndPrefix >> 32, tail);
    } else if (size > 4) {
      auto tail = itemAt<int64_t>(&view, 8);
      return simd::crc32U64(sizeAndPrefix >> 32, tail);
    }
    return simd::crc32U64(1, sizeAndPrefix);
  }

  // Tests if entry matches view after the length and prefix have compared
  // equal.

  void resize(int32_t newSize);
  int32_t nextOffset(int32_t offset) {
    return offset == lastEntryOffset_ ? 0 : offset + kEntrySize;
  }

  int32_t emptyId_{kNoEmpty};
  // Count of 20 byte entries in 'table_'.
  uint64_t capacity_{0};

  // Mask corresponding to the table_.size().
  uint64_t sizeMask_{0};

  // Offset of last entry in 'table_'
  int32_t lastEntryOffset_{0};

  // Table with 20 bytes per entry, 16 for StringView and 4 for its id.
  raw_vector<uint8_t> table_;

  // Count of non-empty entries in 'table_'.
  int32_t numEntries_{0};

  // Count of entries after which a resize() should be done.
  int32_t maxEntries_;
  int32_t collisions_{0};
  int32_t collisions2_{0};
};

} // namespace facebook::velox
