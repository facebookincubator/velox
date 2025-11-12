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
#include "velox/expression/StringToFloatParser.h"

#include <cmath>

#include <fast_float/fast_float.h>
#include <folly/Likely.h>

#include "velox/functions/lib/string/StringImpl.h"

namespace facebook::velox {
inline bool isAllWhitespace(const char* first, const char* last) {
  return std::all_of(first, last, [](char c) {
    return functions::stringImpl::isAsciiWhiteSpace(c);
  });
}

template <typename T>
Status parseInfinityOrNaN(const std::string_view str, T& out) {
  const size_t length = str.length();
  if (length == 0) {
    return Status::Invalid("Empty input string");
  }

  size_t i = 0;
  bool negative = false;

  // Handle leading '+' or '-'
  if (str[i] == '-') {
    negative = true;
    i++;
  } else if (str[i] == '+') {
    i++;
  }

  // Find the last non-space character
  size_t j = length - 1;
  while (j >= i && functions::stringImpl::isAsciiWhiteSpace(str[j])) {
    j--;
  }

  // Extract the meaningful part (no leading/trailing spaces, no signs handled)
  auto data = str.substr(i, j - i + 1);
  if (data == "Infinity") {
    out = negative ? -INFINITY : INFINITY;
    return Status::OK();
  }

  if (data == "NaN") {
    out = negative ? -NAN : NAN;
    return Status::OK();
  }

  return Status::Invalid("Invalid input string: {}", str);
}

template <typename T>
Status StringToFloatParser::parse(const std::string_view str, T& out) {
  if (str.empty()) {
    return Status::Invalid("Empty input string");
  }

  // move through leading whitespace characters
  auto* end = str.end();
  auto* begin = std::find_if_not(str.begin(), end, [](char c) {
    return functions::stringImpl::isAsciiWhiteSpace(c);
  });

  if (begin == end) {
    return Status::Invalid("Empty input string");
  }

  fast_float::parse_options options{
      fast_float::chars_format::general |
      fast_float::chars_format::allow_leading_plus |
      fast_float::chars_format::no_infnan};
  auto [ptr, ec] = fast_float::from_chars_advanced(begin, end, out, options);
  auto isOutOfRange{ec == std::errc::result_out_of_range};
  auto isOk{ec == std::errc()};
  if ((!isOk && !isOutOfRange) || !isAllWhitespace(ptr, end)) {
    // handle "[+/-]Infinity" "[+/-]NaN", case sensitive
    return parseInfinityOrNaN(std::string_view(begin, end - begin), out);
  }

  return Status::OK();
}

template Status StringToFloatParser::parse<float>(
    const std::string_view str,
    float& out);
template Status StringToFloatParser::parse<double>(
    const std::string_view str,
    double& out);
} // namespace facebook::velox
