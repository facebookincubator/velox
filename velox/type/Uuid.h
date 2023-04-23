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

#include "velox/type/StringView.h"

namespace facebook::velox {

using uint128_t = __uint128_t;

struct Uuid {
 public:
  constexpr Uuid() : id_(0) {}
  constexpr Uuid(uint128_t id) : id_(id) {}

  uint128_t id() const {
    return id_;
  }

  bool operator==(const Uuid& other) const {
    return id_ == other.id_;
  }

  bool operator!=(const Uuid& other) const {
    return id_ != other.id_;
  }

  bool operator<(const Uuid& other) const {
    return id_ < other.id_;
  }

  bool operator<=(const Uuid& other) const {
    return id_ <= other.id_;
  }

  bool operator>(const Uuid& other) const {
    return id_ > other.id_;
  }

  bool operator>=(const Uuid& other) const {
    return id_ >= other.id_;
  }

  // Needed for serialization of FlatVector<Uuid>
  operator StringView() const {VELOX_NYI()}

  std::string toString() const;

  operator std::string() const {
    return toString();
  }

 private:
  uint128_t id_;
};

template <typename T>
void toAppend(const ::facebook::velox::Uuid& value, T* result) {
  result->append(value.toString());
}

} // namespace facebook::velox

namespace std {
template <>
struct hash<facebook::velox::Uuid> {
  size_t operator()(const facebook::velox::Uuid& value) const {
    return std::hash<facebook::velox::uint128_t>{}(value.id());
  }
};

std::string to_string(const facebook::velox::Uuid& uuid);

} // namespace std

template <>
struct fmt::formatter<facebook::velox::Uuid> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const facebook::velox::Uuid& uuid, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "{}", std::to_string(uuid));
  }
};

namespace folly {
template <>
struct hasher<::facebook::velox::Uuid> {
  size_t operator()(const facebook::velox::Uuid& value) const {
    return std::hash<facebook::velox::uint128_t>{}(value.id());
  }
};

} // namespace folly