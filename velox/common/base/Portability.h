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

#include <fmt/core.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

inline size_t count_trailing_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_ctzll(x);
}

inline size_t count_trailing_zeros_32bits(uint32_t x) {
  return x == 0 ? 32 : __builtin_ctz(x);
}

inline size_t count_leading_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_clzll(x);
}

inline size_t count_leading_zeros_32bits(uint32_t x) {
  return x == 0 ? 32 : __builtin_clz(x);
}

namespace facebook::velox {

#if defined(__GNUC__) || defined(__clang__)
#define INLINE_LAMBDA __attribute__((__always_inline__))
#else
#define INLINE_LAMBDA
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TSAN_BUILD 1
#endif
#endif

/// tsan_atomic is relaxed atomic without use rmw operations.
/// For an example instead of fetch_add it does load, add, store.
/// This is not atomic operation, but it is ok for our use cases.
/// This allows declaring variables like statistics counters that do not have to
/// be exact nor have synchronized semantics.
/// TODO: Use different name.
template <typename T>
class tsan_atomic : protected std::atomic<T> {
 public:
  using std::atomic<T>::atomic;

  T operator=(T v) noexcept {
    this->store(v, std::memory_order_relaxed);
    return v;
  }

  operator T() const noexcept {
    return this->load(std::memory_order_relaxed);
  }

  T operator+=(T v) noexcept {
    const auto newValue = *this + v;
    return *this = newValue;
  }
  T operator-=(T v) noexcept {
    const auto newValue = *this - v;
    return *this = newValue;
  }
  T operator&=(T v) noexcept {
    const auto newValue = *this & v;
    return *this = newValue;
  }
  T operator|=(T v) noexcept {
    const auto newValue = *this | v;
    return *this = newValue;
  }
  T operator^=(T v) noexcept {
    const auto newValue = *this ^ v;
    return *this = newValue;
  }

  T operator++(int) noexcept {
    const T oldValue = *this;
    *this = oldValue + T(1);
    return oldValue;
  }
  T operator--(int) noexcept {
    const T oldValue = *this;
    *this = oldValue - T(1);
    return oldValue;
  }

  T operator++() noexcept {
    return *this += T(1);
  }
  T operator--() noexcept {
    return *this -= T(1);
  }
};

template <typename T>
inline T tsanAtomicValue(const tsan_atomic<T>& x) {
  return x;
}

template <typename T>
inline void resizeTsanAtomic(
    std::vector<tsan_atomic<T>>& vector,
    int32_t newSize) {
  std::vector<tsan_atomic<T>> newVector(newSize);
  auto numCopy = std::min<int32_t>(newSize, vector.size());
  for (auto i = 0; i < numCopy; ++i) {
    newVector[i] = tsanAtomicValue(vector[i]);
  }
  vector = std::move(newVector);
}
} // namespace facebook::velox

template <typename T>
struct fmt::formatter<facebook::velox::tsan_atomic<T>> : formatter<T> {
  template <typename FormatContext>
  auto format(const T& v, FormatContext& ctx) const {
    return formatter<T>::format(v, ctx);
  }
};
