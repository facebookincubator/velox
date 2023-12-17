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

#include <fmt/format.h>

#include <bitset>
#include <cerrno>
#include <type_traits>
#include <vector>

template <>
struct fmt::formatter<std::errc> : formatter<int> {
  auto format(const std::errc& s, format_context& ctx) {
    return formatter<int>::format(static_cast<int>(s), ctx);
  }
};

// -- Backport from fmt 10 see fmtlib/fmt#3570
#if FMT_VERSION < 100100
namespace fmt::detail {
template <typename T, typename Enable = void>
struct has_flip : std::false_type {};

template <typename T>
struct has_flip<T, std::void_t<decltype(std::declval<T>().flip())>>
    : std::true_type {};

template <typename T>
struct is_bit_reference_like {
  static constexpr const bool value = std::is_convertible<T, bool>::value &&
      std::is_nothrow_assignable<T, bool>::value && has_flip<T>::value;
};

#ifdef _LIBCPP_VERSION

// Workaround for libc++ incompatibility with C++ standard.
// According to the Standard, `bitset::operator[] const` returns bool.
template <typename C>
struct is_bit_reference_like<std::__bit_const_reference<C>> {
  static constexpr const bool value = true;
};

#endif
} // namespace fmt::detail

// We can't use std::vector<bool, Allocator>::reference and
// std::bitset<N>::reference because the compiler can't deduce Allocator and N
// in partial specialization.
template <typename BitRef, typename Char>
struct fmt::formatter<
    BitRef,
    Char,
    std::enable_if_t<fmt::detail::is_bit_reference_like<BitRef>::value>>
    : formatter<bool, Char> {
  template <typename FormatContext>
  FMT_CONSTEXPR auto format(const BitRef& v, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    return formatter<bool, Char>::format(v, ctx);
  }
};
#endif
// -- End of backport
