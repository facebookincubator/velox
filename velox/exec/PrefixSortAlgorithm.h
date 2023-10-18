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

#include <cstdint>
#include <functional>
#include <memory>
#include "PrefixSortEncode.h"
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::exec {

template <typename T>
using unsafe_unique_array = std::unique_ptr<T[], std::default_delete<T>>;

template <class _Tp>
inline std::unique_ptr<_Tp[], std::default_delete<_Tp>>
make_unsafe_unique_array(size_t __n) {
  return std::unique_ptr<_Tp[], std::default_delete<_Tp>>(new _Tp[__n]());
}

struct PrefixSortContext {
  PrefixSortContext(idx_t entry_size, data_ptr_t end)
      : entry_size(entry_size),
        iter_swap_buf_ptr(make_unsafe_unique_array<data_t>(entry_size)),
        iter_swap_buf(iter_swap_buf_ptr.get()),
        end(end) {}
  const idx_t entry_size;
  unsafe_unique_array<data_t> iter_swap_buf_ptr;
  const data_ptr_t iter_swap_buf;
  const data_ptr_t end;
};

struct PrefixSortIterator {
  PrefixSortIterator(data_ptr_t ptr, const idx_t& entry_size)
      : ptr(ptr), entry_size(entry_size) {}

  PrefixSortIterator(const PrefixSortIterator& other)
      : ptr(other.ptr), entry_size(other.entry_size) {}

  inline const data_ptr_t& operator*() const {
    return ptr;
  }

  inline PrefixSortIterator& operator++() {
    ptr += entry_size;
    return *this;
  }

  inline PrefixSortIterator& operator--() {
    ptr -= entry_size;
    return *this;
  }

  inline PrefixSortIterator operator++(int) {
    auto tmp = *this;
    ptr += entry_size;
    return tmp;
  }

  inline PrefixSortIterator operator--(int) {
    auto tmp = *this;
    ptr -= entry_size;
    return tmp;
  }

  inline PrefixSortIterator operator+(const idx_t& i) const {
    auto result = *this;
    result.ptr += i * entry_size;
    return result;
  }

  inline PrefixSortIterator operator-(const idx_t& i) const {
    PrefixSortIterator result = *this;
    result.ptr -= i * entry_size;
    return result;
  }

  inline PrefixSortIterator& operator=(const PrefixSortIterator& other) {
    VELOX_CHECK(entry_size == other.entry_size);
    ptr = other.ptr;
    return *this;
  }

  inline friend idx_t operator-(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    VELOX_CHECK((*lhs - *rhs) % lhs.entry_size == 0);
    VELOX_CHECK(*lhs - *rhs >= 0);
    return (*lhs - *rhs) / lhs.entry_size;
  }

  inline friend bool operator<(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs < *rhs;
  }

  inline friend bool operator>(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs > *rhs;
  }

  inline friend bool operator>=(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs >= *rhs;
  }

  inline friend bool operator<=(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs <= *rhs;
  }

  inline friend bool operator==(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs == *rhs;
  }

  inline friend bool operator!=(
      const PrefixSortIterator& lhs,
      const PrefixSortIterator& rhs) {
    return *lhs != *rhs;
  }

 private:
  data_ptr_t ptr;
  const idx_t& entry_size;
};

static void iter_swap(
    const PrefixSortIterator& lhs,
    const PrefixSortIterator& rhs,
    const PrefixSortContext& context) {
  VELOX_CHECK(*lhs < context.end);
  VELOX_CHECK(*rhs < context.end);
  FastMemcpy(context.iter_swap_buf, *lhs, context.entry_size);
  FastMemcpy(*lhs, *rhs, context.entry_size);
  FastMemcpy(*rhs, context.iter_swap_buf, context.entry_size);
}

static void iter_range_swap(
    const PrefixSortIterator& start1,
    const PrefixSortIterator& start2,
    idx_t length,
    const PrefixSortContext& context) {
  VELOX_CHECK(*(start1 + length) <= context.end);
  VELOX_CHECK(*(start2 + length) <= context.end);
  for (idx_t i = 0; i < length; i++) {
    // TODO need a variable size buffer.
    FastMemcpy(context.iter_swap_buf, *(start1 + i), context.entry_size);
    FastMemcpy(*(start1 + i), *(start2 + i), context.entry_size);
    FastMemcpy(*(start2 + i), context.iter_swap_buf, context.entry_size);
  }
}

// Quick sort same as presto.
static int SMALL_SORT = 7;
static int MEDIUM_SORT = 40;

inline static PrefixSortIterator median3(
    const PrefixSortIterator& a,
    const PrefixSortIterator& b,
    const PrefixSortIterator& c,
    std::function<
        int(const PrefixSortIterator&, const PrefixSortIterator&)> const& cmp) {
  return cmp(a, b) < 0 ? (cmp(b, c) < 0       ? b
                              : cmp(a, c) < 0 ? c
                                              : a)
                       : (cmp(b, c) > 0       ? b
                              : cmp(a, c) > 0 ? c
                                              : a);
}

static void PrefixQuickSort(
    const PrefixSortContext& sortContext,
    const PrefixSortIterator& start,
    const PrefixSortIterator& end,
    std::function<
        int(const PrefixSortIterator&, const PrefixSortIterator&)> const&
        compare) {
  int len = end - start;
  // Insertion sort on smallest arrays
  if (len < SMALL_SORT) {
    for (PrefixSortIterator i = start; i < end; i++) {
      for (PrefixSortIterator j = i; j > start && (compare(j - 1, j) > 0);
           j--) {
        iter_swap(j, j - 1, sortContext);
      }
    }
    return;
  }
  // Choose a partition element, v
  PrefixSortIterator m = start + len / 2; // Small arrays, middle element
  if (len > SMALL_SORT) {
    PrefixSortIterator l = start;
    PrefixSortIterator n = end - 1;
    if (len > MEDIUM_SORT) { // Big arrays, pseudomedian of 9
      int s = len / 8;
      l = median3(l, l + s, l + 2 * s, compare);
      m = median3(m - s, m, m + s, compare);
      n = median3(n - 2 * s, n - s, n, compare);
    }
    m = median3(l, m, n, compare); // Mid-size, med of 3
  }
  PrefixSortIterator a = start;
  PrefixSortIterator b = a;
  PrefixSortIterator c = end - 1;
  // Establish Invariant(v means partition value): v* (<v)* (>v)* v*
  PrefixSortIterator d = c;
  while (true) {
    int comparison;
    while (b <= c) {
      comparison = compare(b, m);
      if (comparison > 0) {
        break;
      }
      if (comparison == 0) {
        if (a == m) {
          m = b;
        } else if (b == m) {
          m = a;
        }
        iter_swap(a++, b, sortContext);
      }
      b++;
    }
    while (c >= b) {
      comparison = compare(c, m);
      if (comparison < 0) {
        break;
      }
      if (comparison == 0) {
        if (c == m) {
          m = d;
        } else if (d == m) {
          m = c;
        }
        iter_swap(c, d--, sortContext);
      }
      c--;
    }
    if (b > c) {
      break;
    }
    if (b == m) {
      m = d;
    }
    iter_swap(b++, c--, sortContext);
  }
  // Swap partition elements back end middle
  int s;
  PrefixSortIterator n = end;
  s = std::min(a - start, b - a);
  iter_range_swap(start, b - s, s, sortContext);
  s = std::min(d - c, n - d - 1);
  iter_range_swap(b, n - s, s, sortContext);
  // Recursively sort non-partition-elements
  s = b - a;
  if (s > 1) {
    PrefixQuickSort(sortContext, start, start + s, compare);
  }
  s = d - c;
  if (s > 1) {
    PrefixQuickSort(sortContext, n - s, n, compare);
  }
}

} // namespace facebook::velox::exec