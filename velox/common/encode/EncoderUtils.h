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

const size_t kCharsetSize = 64;
const size_t kReverseIndexSize = 256;

/// Character set used for encoding purposes.
/// Contains specific characters that form the encoding scheme.
using Charset = std::array<char, kCharsetSize>;

/// Reverse lookup table for decoding purposes.
/// Maps each possible encoded character to its corresponding numeric value
/// within the encoding base.
using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

/// Padding character used in encoding.
const static char kPadding = '=';
/// Checks if there is padding in encoded data.
static inline bool isPadded(const char* data, size_t len) {
  return (len > 0 && data[len - 1] == kPadding) ? true : false;
}

/// Counts the number of padding characters in encoded data.
static inline size_t numPadding(const char* src, size_t len) {
  size_t numPadding{0};
  while (len > 0 && src[len - 1] == kPadding) {
    numPadding++;
    len--;
  }
  return numPadding;
}
/// Performs a reverse lookup in the reverse index to retrieve the original
/// index of a character in the base.
inline uint8_t
baseReverseLookup(int base, char p, const ReverseIndex& reverseIndex) {
  auto curr = reverseIndex[(uint8_t)p];
  if (curr >= base) {
    VELOX_USER_FAIL("decode() - invalid input string: invalid characters");
  }
  return curr;
}

// Validate the character in charset with ReverseIndex table
static constexpr bool checkForwardIndex(
    uint8_t idx,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  for (uint8_t i = 0; i <= idx; ++i) {
    if (!(reverseIndex[static_cast<uint8_t>(charset[i])] == i)) {
      return false;
    }
  }
  return true;
}

/// Searches for a character within a charset up to a certain index.
static const bool findCharacterInCharSet(
    const Charset& charset,
    int base,
    uint8_t idx,
    const char c) {
  for (; idx < base; ++idx) {
    if (charset[idx] == c) {
      return true;
    }
  }
  return false;
}

/// Checks the consistency of a reverse index mapping for a given character
/// set.
static constexpr bool checkReverseIndex(
    uint8_t idx,
    int base,
    const Charset& charset,
    const ReverseIndex& reverseIndex) {
  for (uint8_t currentIdx = idx; currentIdx != static_cast<uint8_t>(-1);
       --currentIdx) {
    if (reverseIndex[currentIdx] == 255) {
      if (findCharacterInCharSet(
              charset, base, 0, static_cast<char>(currentIdx))) {
        return false;
      }
    } else {
      if (!(charset[reverseIndex[currentIdx]] == currentIdx)) {
        return false;
      }
    }
       }
  return true;
}

} // namespace facebook::velox::encoding
