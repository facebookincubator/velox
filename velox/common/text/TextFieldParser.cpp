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

#include "velox/common/text/TextFieldParser.h"

#include <boost/algorithm/string/predicate.hpp>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace facebook::velox::text {

namespace {
constexpr size_t kStackBufferSize{64};

// Null-terminates `field` into either `stackBuffer` (for short fields) or
// `heapBuffer` (for longer fields), and returns a pointer to the
// null-terminated data suitable for sscanf.
const char* nullTerminate(
    std::string_view field,
    char (&stackBuffer)[kStackBufferSize],
    std::string& heapBuffer) {
  if (field.size() < kStackBufferSize) {
    std::memcpy(stackBuffer, field.data(), field.size());
    stackBuffer[field.size()] = '\0';
    return stackBuffer;
  }
  heapBuffer.assign(field.data(), field.size());
  return heapBuffer.c_str();
}
} // namespace

std::optional<int64_t> TextFieldParser::parseInt64(
    std::string_view field,
    bool allowTrailingDecimal) {
  if (field.empty()) {
    return std::nullopt;
  }
  const char first = field.front();
  if (first != '-' && !std::isdigit(static_cast<unsigned char>(first))) {
    return std::nullopt;
  }

  char stackBuffer[kStackBufferSize];
  std::string heapBuffer;
  const char* nullTerminatedField =
      nullTerminate(field, stackBuffer, heapBuffer);

  int64_t value{0};
  long long scanPosition{0};
  errno = 0;
  const int scanCount = std::sscanf(
      nullTerminatedField, "%" SCNd64 "%lln", &value, &scanPosition);
  if (scanCount != 1 || errno == ERANGE) {
    return std::nullopt;
  }

  if (static_cast<size_t>(scanPosition) < field.size()) {
    if (!allowTrailingDecimal) {
      return std::nullopt;
    }
    for (size_t i = static_cast<size_t>(scanPosition); i < field.size(); ++i) {
      const char character = nullTerminatedField[i];
      if (i == static_cast<size_t>(scanPosition) && character == '.') {
        continue;
      }
      if (character >= '0' && character <= '9') {
        continue;
      }
      return std::nullopt;
    }
  }
  return value;
}

std::optional<bool> TextFieldParser::parseBoolean(
    std::string_view field,
    bool allowOneZero) {
  if (field.empty()) {
    return std::nullopt;
  }
  if (allowOneZero && field.size() == 1) {
    if (field[0] == '1') {
      return true;
    }
    if (field[0] == '0') {
      return false;
    }
  }
  if (field.size() == 4 && field[0] == 'T' && field[1] == 'R' &&
      field[2] == 'U' && field[3] == 'E') {
    return true;
  }
  if (field.size() == 5 && field[0] == 'F' && field[1] == 'A' &&
      field[2] == 'L' && field[3] == 'S' && field[4] == 'E') {
    return false;
  }
  switch (field.size()) {
    case 4:
      if (boost::algorithm::iequals(field, std::string_view{"TRUE"})) {
        return true;
      }
      break;
    case 5:
      if (boost::algorithm::iequals(field, std::string_view{"FALSE"})) {
        return false;
      }
      break;
    default:
      break;
  }
  return std::nullopt;
}

} // namespace facebook::velox::text
