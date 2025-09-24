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

#include <fast_float/fast_float.h>
#include <folly/Likely.h>
#include <cmath>

#include "velox/type/StringToFloatParser.h"

namespace facebook::velox {

static inline bool characterIsSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
      c == '\r';
}

static inline bool isEither(const char currentChar, char lower, char upper) {
  return currentChar == lower || currentChar == upper;
}

static inline bool isInfinityString(const char* str, size_t length) {
  // length = 3: inf/INF
  // length = 8: infinity
  if (length != 3 && length != 8) {
    return false;
  }

  if (isEither(str[0], 'i', 'I') && isEither(str[1], 'n', 'N') &&
      isEither(str[2], 'f', 'F')) {
    if (length == 3) {
      return true;
    }

    return isEither(str[3], 'i', 'I') && isEither(str[4], 'n', 'N') &&
        isEither(str[5], 'i', 'I') && isEither(str[6], 't', 'T') &&
        isEither(str[7], 'y', 'Y');
  }

  return false;
}

static inline bool isNanString(const char* str, size_t length) {
  if (length != 3) {
    return false;
  }

  return isEither(str[0], 'n', 'N') && isEither(str[1], 'a', 'A') &&
      isEither(str[2], 'n', 'N');
}

template <typename T>
Status StringToFloatParser::parse(const std::string_view& str, T& out) {
  auto length = str.length();
  if (UNLIKELY(length <= 0)) {
    return Status::Invalid("empty string");
  }
  auto i = 0;
  // Skip leading spaces.
  for (; i < length; ++i) {
    if (!characterIsSpace(str[i])) {
      break;
    }
  }

  // Skip trailing spaces.
  auto j = length - 1;
  for (; j >= i; j--) {
    if (!characterIsSpace(str[j])) {
      break;
    }
  }

  auto negative = false;
  // Skip leading +/-.
  switch (str[i]) {
    case '-':
      negative = true;
      i++;
      break;
    case '+':
      i++;
  }

  // Check INF/INFINITY/NAN, ignoring case.
  if (isInfinityString(str.data() + i, j - i + 1)) {
    out = negative ? -INFINITY : INFINITY;
    return Status::OK();
  } else if (isNanString(str.data() + i, j - i + 1)) {
    out = negative ? -NAN : NAN;
    return Status::OK();
  }

  // Check invalid char.
  auto exponential = false;
  auto point = false;
  auto invalid = false;
  for (auto k = i; k <= j; ++k) {
    if (str[k] >= '0' && str[k] <= '9') {
      continue;
    } else if (str[k] == '.') {
      if (LIKELY(!point)) {
        point = true;
      } else {
        invalid = true;
        break;
      }
    } else if (str[k] == 'e' || str[k] == 'E') {
      if (LIKELY(!exponential)) {
        exponential = true;
      } else {
        invalid = true;
        break;
      }
    } else if (str[k] == '-' || str[k] == '+') {
      if (LIKELY(k > i && (str[k - 1] == 'e' || str[k - 1] == 'E'))) {
        continue;
      } else {
        invalid = true;
        break;
      }
    } else {
      invalid = true;
      break;
    }
  }

  if (invalid) {
    return Status::Invalid(
        "Non-whitespace character found after end of conversion");
  }

  double val;
  auto res = fast_float::from_chars(str.data() + i, str.data() + j + 1, val);

  if (LIKELY(res.ec == std::errc())) {
    if (UNLIKELY(val == std::numeric_limits<T>::infinity())) {
      return Status::UserError("Result overflows.");
    }

    out = negative ? (T)-val : (T)val;
    return Status::OK();
  }

  return Status::Invalid("Invalid input string: {}", str);
}

template Status StringToFloatParser::parse<float>(
    const std::string_view& str,
    float& out);
template Status StringToFloatParser::parse<double>(
    const std::string_view& str,
    double& out);
} // namespace facebook::velox