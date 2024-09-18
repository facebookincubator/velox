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

#include <exception>
#include <map>
#include <string>

#include <folly/Range.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::encoding {

/// Padding character used in encoding.
const static char kPadding = '=';

// Checks if there is padding in encoded input.
static inline bool isPadded(std::string_view input, size_t inputSize) {
  return (inputSize > 0 && input[inputSize - 1] == kPadding);
}

// Counts the number of padding characters in encoded input.
static inline size_t numPadding(std::string_view input, size_t inputSize) {
  size_t numPadding{0};
  while (inputSize > 0 && input[inputSize - 1] == kPadding) {
    numPadding++;
    inputSize--;
  }
  return numPadding;
}

// Validate the character in charset with ReverseIndex table
template <typename Charset, typename ReverseIndex>
constexpr bool checkForwardIndex(
    uint8_t idx,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  return (reverseIndex[static_cast<uint8_t>(charset[idx])] == idx) &&
      (idx > 0 ? checkForwardIndex(idx - 1, charset, reverseIndex) : true);
}

// Searches for a character within a charset up to a certain index.
template <typename Charset>
constexpr bool findCharacterInCharset(
    const Charset& charset,
    uint8_t index,
    const char character) {
  return index < charset.size() &&
      ((charset[index] == character) ||
       findCharacterInCharset(charset, index + 1, character));
}

// Checks the consistency of a reverse index mapping for a given character set.
template <typename Charset, typename ReverseIndex>
constexpr bool checkReverseIndex(
    uint8_t index,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  return (reverseIndex[index] == 255
              ? !findCharacterInCharset(charset, 0, static_cast<char>(index))
              : (charset[reverseIndex[index]] == index)) &&
      (index > 0 ? checkReverseIndex(index - 1, charset, reverseIndex) : true);
}

} // namespace facebook::velox::encoding
