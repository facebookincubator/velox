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

#include <iomanip>
#include <sstream>
#include <string>

#include <folly/dynamic.h>

namespace facebook::velox {

struct Date {
 public:
  constexpr Date() : days_(0) {}
  constexpr Date(int32_t days) : days_(days) {}

  int32_t getDays() const {
    return days_;
  }

  bool operator==(const Date& b) const {
    return days_ == b.days_;
  }

  bool operator!=(const Date& b) const {
    return days_ != b.days_;
  }

  bool operator<(const Date& b) const {
    return days_ < b.days_;
  }

  bool operator<=(const Date& b) const {
    return days_ <= b.days_;
  }

  bool operator>(const Date& b) const {
    return days_ > b.days_;
  }

  bool operator>=(const Date& b) const {
    return days_ >= b.days_;
  }

  // Needed for serialization of FlatVector<Date>
  operator StringView() const {
    return StringView("TODO: Implement");
  };

  std::string toString() const {
    // Find the number of seconds for the days_;
    int64_t day_seconds = days_ * 86400;
    auto tmValue = gmtime((const time_t*)&day_seconds);
    if (!tmValue) {
      const auto& error_message = folly::to<std::string>(
          "Can't convert days to time: ", folly::to<std::string>(days_));
      throw std::runtime_error{error_message};
    }

    // return ISO 8601 time format.
    // %F - equivalent to "%Y-%m-%d" (the ISO 8601 date format)
    std::ostringstream oss;
    oss << std::put_time(tmValue, "%F");
    return oss.str();
  }

  operator std::string() const {
    return toString();
  }

  operator folly::dynamic() const {
    return folly::dynamic(days_);
  }

 private:
  // Number of days since the epoch ( 1970-01-01)
  int32_t days_;
};

void parseTo(folly::StringPiece in, ::facebook::velox::Date& out);

template <typename T>
void toAppend(const ::facebook::velox::Date& value, T* result) {
  // TODO Implement
}

} // namespace facebook::velox

namespace std {
template <>
struct hash<::facebook::velox::Date> {
  size_t operator()(const ::facebook::velox::Date value) const {
    return std::hash<int32_t>{}(value.getDays());
  }
};

std::string to_string(const ::facebook::velox::Date& ts);

} // namespace std

namespace folly {
template <>
struct hasher<::facebook::velox::Date> {
  size_t operator()(const ::facebook::velox::Date value) const {
    return std::hash<int32_t>{}(value.getDays());
  }
};

} // namespace folly
