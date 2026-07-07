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

#include "velox/dwio/text/reader/TextFieldParser.h"

#include <boost/algorithm/string/predicate.hpp>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace facebook::velox::text {

namespace {
// Null-terminates `field` into either `stackBuf` (for short fields) or
// `heapBuf` (for longer fields), and returns a pointer to the
// null-terminated data suitable for sscanf.
const char* nullTerminate(
    std::string_view field,
    char (&stackBuf)[TextFieldParser::kStackBufSize],
    std::string& heapBuf) {
  if (field.size() < TextFieldParser::kStackBufSize) {
    std::memcpy(stackBuf, field.data(), field.size());
    stackBuf[field.size()] = '\0';
    return stackBuf;
  }
  heapBuf.assign(field.data(), field.size());
  return heapBuf.c_str();
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

  char stackBuf[kStackBufSize];
  std::string heapBuf;
  const char* cstr = nullTerminate(field, stackBuf, heapBuf);

  int64_t value{0};
  long long scanPos{0};
  errno = 0;
  const int scanCount = std::sscanf(cstr, "%" SCNd64 "%lln", &value, &scanPos);
  if (scanCount != 1 || errno == ERANGE) {
    return std::nullopt;
  }

  if (static_cast<size_t>(scanPos) < field.size()) {
    if (!allowTrailingDecimal) {
      return std::nullopt;
    }
    for (size_t i = static_cast<size_t>(scanPos); i < field.size(); ++i) {
      const char c = cstr[i];
      if (i == static_cast<size_t>(scanPos) && c == '.') {
        continue;
      }
      if (c >= '0' && c <= '9') {
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
