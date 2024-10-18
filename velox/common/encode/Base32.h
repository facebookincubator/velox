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

#include <array>
#include <string>

#include "velox/common/base/GTestMacros.h"
#include "velox/common/base/Status.h"
#include "velox/common/encode/EncoderUtils.h"

namespace facebook::velox::encoding {

class Base32 {
 public:
  static const size_t kCharsetSize = 32;
  static const size_t kReverseIndexSize = 256;

  /// Character set used for encoding purposes.
  /// Contains specific characters that form the encoding scheme.
  using Charset = std::array<char, kCharsetSize>;

  /// Reverse lookup table for decoding purposes.
  /// Maps each possible encoded character to its corresponding numeric value
  /// within the encoding base.
  using ReverseIndex = std::array<uint8_t, kReverseIndexSize>;

  /// Returns encoded size for the input of the specified size.
  static size_t calculateEncodedSize(
      size_t inputSize,
      bool includePadding = true);

  /// Encodes the specified number of characters from the 'input' and writes the
  /// result to the 'output'. The output must have enough space, e.g., as
  /// returned by calculateEncodedSize().
  static Status encode(std::string_view input, char* output);

 private:
  // Encodes the specified input using the provided charset.
  template <class T>
  static Status encodeImpl(
      const T& input,
      const Charset& charset,
      bool includePadding,
      char* output);
};

} // namespace facebook::velox::encoding
