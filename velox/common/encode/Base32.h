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
#include "velox/common/encode/EncoderUtils.h"

namespace facebook::velox::encoding {

class Base32 {
 public:
  /// Returns encoded size for the input of the specified size.
  static size_t calculateEncodedSize(size_t size, bool withPadding = true);

  /// Encodes the specified number of characters from the 'data' and writes the
  /// result to the 'output'. The output must have enough space, e.g. as
  /// returned by the calculateEncodedSize().
  static void encode(const char* data, size_t size, char* output);

  /// Returns decoded size for the specified input. Adjusts the 'size' to
  /// subtract the length of the padding, if exists.
  static size_t calculateDecodedSize(const char* data, size_t& size);

  /// Decodes the specified number of characters from the 'src' and writes the
  /// result to the 'dst'. The destination must have enough space, e.g. as
  /// returned by the calculateDecodedSize().
  static size_t
  decode(const char* src, size_t src_len, char* dst, size_t dst_len);

 private:
  /// Decodes the specified number of base 32 encoded characters from the 'src'
  /// and writes to 'dst'
  static size_t decodeImpl(
      const char* src,
      size_t src_len,
      char* dst,
      size_t dst_len,
      const ReverseIndex& table);

  template <class T>
  static void encodeImpl(
      const T& data,
      const Charset& charset,
      bool include_pad,
      char* out);
};

} // namespace facebook::velox::encoding
