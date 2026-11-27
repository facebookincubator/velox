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
#include <type_traits>

namespace facebook::velox {

/// Points to data followed by at least simd::kPadding readable bytes, so that
/// a function taking it may read past the last element. If T is not const, the
/// function may also store to those bytes. An AlignedBuffer keeps a guard right
/// after capacity() in debug builds, so a function that stores new values past
/// the last element, rather than rewriting the bytes it read, needs that room
/// inside capacity(). raw_vector and ScratchPtr have no such guard.
/// Vector buffers in general have no padding (see Buffer), so a function that
/// needs it takes a Padded instead of a raw pointer. Its callers say where the
/// padding comes from: an AlignedBuffer (see paddedData() in Buffer.h), a
/// raw_vector or ScratchPtr (their paddedData()), or memory they know is padded
/// (assumePadded()).
template <typename T>
class Padded {
 public:
  /// Passes nullptr for an optional argument.
  Padded(std::nullptr_t) : data_(nullptr) {}

  /// Wraps 'data' that the caller knows is followed by simd::kPadding bytes.
  static Padded assumePadded(T* data) {
    return Padded(data);
  }

  /// Returns the pointer.
  T* get() const {
    return data_;
  }

  /// Returns the address 'offset' elements past get(). It is followed by the
  /// same padding, since the padding is after the end of the memory.
  Padded operator+(std::ptrdiff_t offset) const
    requires(!std::is_void_v<T>)
  {
    return Padded(data_ + offset);
  }

  /// Converts to Padded<const T>, or to Padded of void. A conversion to a base
  /// class is not allowed: operator+ would then step by the base class size.
  template <typename U>
  operator Padded<U>() const
    requires(
        !std::is_same_v<T, U> && std::is_convertible_v<T*, U*> &&
        (std::is_void_v<U> ||
         std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<U>>))
  {
    return Padded<U>::assumePadded(data_);
  }

 private:
  // Private, so that callers build a Padded only through assumePadded() or a
  // paddedData() that says where the padding comes from.
  explicit Padded(T* data) : data_(data) {}

  // Followed by at least simd::kPadding addressable bytes.
  T* data_;
};

} // namespace facebook::velox
