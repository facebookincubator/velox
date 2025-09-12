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

#include <atomic>
#include <bitset>
#include <format>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

template <typename EnumType, typename Char>
  requires std::is_enum_v<EnumType>
struct std::formatter<EnumType, Char>
    : formatter<std::underlying_type_t<EnumType>, Char> {
  template <typename FormatContext>
  auto format(const EnumType& enumValue, FormatContext& ctx) const {
    using T = std::underlying_type_t<EnumType>;
    return formatter<T>::format(static_cast<T>(enumValue), ctx);
  }
};

template <typename T, typename Char>
struct std::formatter<std::atomic<T>, Char> : formatter<T, Char> {
  template <typename FormatContext>
  auto format(const std::atomic<T>& atomicValue, FormatContext& ctx) const {
    return formatter<T>::format(
        atomicValue.load(std::memory_order_relaxed), ctx);
  }
};
