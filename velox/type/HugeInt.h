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
#include <folly/dynamic.h>
#include <sstream>
#include <string>
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/type/StringView.h"
#include "velox/type/custom_type/Int128.h"

#pragma once

namespace facebook::velox {

using int128_t = type::int128;
using uint128_t = type::uint128;
;

class HugeInt {
 public:
  static constexpr FOLLY_ALWAYS_INLINE int128_t
  build(uint64_t hi, uint64_t lo) {
    // GCC does not allow left shift negative value.
    return (static_cast<__uint128_t>(hi) << 64) | lo;
  }

  static constexpr FOLLY_ALWAYS_INLINE uint64_t lower(int128_t value) {
    return value.lo();
  }

  static constexpr FOLLY_ALWAYS_INLINE uint64_t upper(int128_t value) {
    return static_cast<uint64_t>(value.hi());
  }

  static FOLLY_ALWAYS_INLINE int128_t deserialize(const char* serializedData) {
    int128_t value;
    memcpy(&value, serializedData, sizeof(int128_t));
    return value;
  }

  static FOLLY_ALWAYS_INLINE void serialize(
      const int128_t& value,
      char* serializedData) {
    memcpy(serializedData, &value, sizeof(int128_t));
  }

  static int128_t parse(const std::string& str);
};

} // namespace facebook::velox

namespace std {
string to_string(facebook::velox::int128_t x);
} // namespace std
