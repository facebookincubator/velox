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
#include <folly/io/IOBuf.h>
#include "velox/common/encode/EncoderUtils.h"

namespace facebook::velox::encoding {

class Base64 {
 public:
  static std::string encode(const char* data, size_t len);
  static std::string encode(folly::StringPiece text);
  static std::string encode(const folly::IOBuf* text);

  /// Returns encoded size for the input of the specified size.
  static size_t calculateEncodedSize(size_t size, bool withPadding = true);

  /// Encodes the specified number of characters from the 'data' and writes the
  /// result to the 'output'. The output must have enough space, e.g. as
  /// returned by the calculateEncodedSize().
  static void encode(const char* data, size_t size, char* output);

  // Appends the encoded text to out.
  static void encodeAppend(folly::StringPiece text, std::string& out);

  static std::string decode(folly::StringPiece encoded);

  /// Returns the actual size of the decoded data. Will also remove the padding
  /// length from the input data 'size'.
  static size_t calculateDecodedSize(const char* data, size_t& size);

  /// Decodes the specified number of characters from the 'data' and writes the
  /// result to the 'output'. The output must have enough space, e.g. as
  /// returned by the calculateDecodedSize().
  static void decode(const char* data, size_t size, char* output);

  static void decode(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);

  /// Encodes the specified number of characters from the 'data' and writes the
  /// result to the 'output'. The output must have enough space, e.g. as
  /// returned by the calculateEncodedSize().
  static void encodeUrl(const char* data, size_t size, char* output);

  // compatible with www's Base64URL::encode/decode
  // TODO rename encode_url/decode_url to encodeUrl/encodeUrl.
  static std::string encodeUrl(const char* data, size_t len);
  static std::string encodeUrl(const folly::IOBuf* data);
  static std::string encodeUrl(folly::StringPiece text);
  static void decodeUrl(
      const std::pair<const char*, int32_t>& payload,
      std::string& output);
  static std::string decodeUrl(folly::StringPiece text);

  static size_t
  decode(const char* src, size_t src_len, char* dst, size_t dst_len);

  static void
  decodeUrl(const char* src, size_t src_len, char* dst, size_t dst_len);

 private:
  template <class T>
  static std::string
  encodeImpl(const T& data, const Charset& charset, bool include_pad);

  template <class T>
  static void encodeImpl(
      const T& data,
      const Charset& charset,
      bool include_pad,
      char* out);

  static size_t decodeImpl(
      const char* src,
      size_t src_len,
      char* dst,
      size_t dst_len,
      const ReverseIndex& table);
};

} // namespace facebook::velox::encoding
