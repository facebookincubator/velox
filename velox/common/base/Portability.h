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

namespace facebook::velox {

inline size_t count_trailing_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_ctzll(x);
}

inline size_t count_leading_zeros(uint64_t x) {
  return x == 0 ? 64 : __builtin_clzll(x);
}

#if defined(__GNUC__) || defined(__clang__)
#define INLINE_LAMBDA __attribute__((__always_inline__))
#else
#define INLINE_LAMBDA
#endif

/// Define asan_atomic<T> to be std::atomic<T> with asan and just T
/// otherwise. Applies to counters that do not need atomicity and are
/// not serialized on any mutex but generate warnings with asan.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
template <typename T>
using asan_atomic = std::atomic<T>;
#else
template <typename T>
using asan_atomic = T;
#endif
#else
template <typename T>
using asan_atomic = T;
#endif

} // namespace facebook::velox
