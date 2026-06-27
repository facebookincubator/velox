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

// Minimal std::span polyfill for MSVC C++17.
// Only supports dynamic extent and the subset used by Velox.

#include <cstddef>
#include <type_traits>

namespace std {

template <typename T>
class span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using iterator = pointer;

  constexpr span() noexcept : data_(nullptr), size_(0) {}
  constexpr span(pointer data, size_type size) noexcept
      : data_(data), size_(size) {}

  template <std::size_t N>
  constexpr span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}

  template <typename Container>
  constexpr span(Container& c) noexcept : data_(c.data()), size_(c.size()) {}

  template <typename Container>
  constexpr span(const Container& c) noexcept
      : data_(c.data()), size_(c.size()) {}

  constexpr pointer data() const noexcept {
    return data_;
  }
  constexpr size_type size() const noexcept {
    return size_;
  }
  constexpr bool empty() const noexcept {
    return size_ == 0;
  }
  constexpr reference operator[](size_type idx) const {
    return data_[idx];
  }
  constexpr iterator begin() const noexcept {
    return data_;
  }
  constexpr iterator end() const noexcept {
    return data_ + size_;
  }

 private:
  pointer data_;
  size_type size_;
};

} // namespace std
