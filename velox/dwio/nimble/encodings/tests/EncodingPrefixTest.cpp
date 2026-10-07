/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"

#include <array>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"

using namespace facebook;

namespace {

std::string serializePrefix(
    nimble::EncodingType encodingType,
    nimble::DataType dataType,
    uint32_t rowCount,
    bool useVarint) {
  std::string data(
      nimble::EncodingPrefix::serializedSize(rowCount, useVarint), '\0');
  char* pos = data.data();
  nimble::EncodingPrefix::serialize(
      encodingType, dataType, rowCount, useVarint, pos);
  EXPECT_EQ(pos, data.data() + data.size());
  return data;
}

} // namespace

TEST(EncodingPrefixTest, readsEncodingTypeAndDataType) {
  struct Case {
    nimble::EncodingType encodingType;
    nimble::DataType dataType;
    uint32_t rowCount;
    bool useVarint;
    std::string_view name;
  };

  constexpr std::array<Case, 5> cases{{
      {nimble::EncodingType::Delta,
       nimble::DataType::Int64,
       123,
       false,
       "delta/int64/fixed"},
      {nimble::EncodingType::SharedDictionary,
       nimble::DataType::Int64,
       513,
       true,
       "shared_dictionary/int64/varint"},
      {nimble::EncodingType::Trivial,
       nimble::DataType::Uint32,
       1,
       false,
       "trivial/uint32/fixed"},
      {nimble::EncodingType::Nullable,
       nimble::DataType::Bool,
       64,
       true,
       "nullable/bool/varint"},
      {nimble::EncodingType::Fsst,
       nimble::DataType::String,
       4096,
       true,
       "fsst/string/varint"},
  }};

  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    const auto data = serializePrefix(
        testCase.encodingType,
        testCase.dataType,
        testCase.rowCount,
        testCase.useVarint);

    EXPECT_EQ(
        nimble::EncodingPrefix::encodingType(data), testCase.encodingType);
    EXPECT_EQ(nimble::EncodingPrefix::dataType(data), testCase.dataType);
    EXPECT_EQ(nimble::EncodingPrefix::readDataType(data), testCase.dataType);
  }
}

TEST(EncodingPrefixTest, readsFixedRowCountPrefix) {
  const auto data = serializePrefix(
      nimble::EncodingType::Trivial,
      nimble::DataType::Uint32,
      /*rowCount=*/123,
      /*useVarint=*/false);

  EXPECT_EQ(
      nimble::EncodingPrefix::readRowCount(data, /*useVarint=*/false), 123);
  EXPECT_EQ(
      nimble::EncodingPrefix::prefixSize(data, /*useVarint=*/false),
      nimble::EncodingPrefix::kFixedPrefixSize);
}

TEST(EncodingPrefixTest, readsVarintRowCountPrefix) {
  constexpr uint32_t rowCount = 1'000'000;
  const auto data = serializePrefix(
      nimble::EncodingType::RLE,
      nimble::DataType::String,
      rowCount,
      /*useVarint=*/true);

  EXPECT_EQ(
      nimble::EncodingPrefix::readRowCount(data, /*useVarint=*/true), rowCount);
  EXPECT_EQ(
      nimble::EncodingPrefix::prefixSize(data, /*useVarint=*/true),
      data.size());
}

TEST(EncodingPrefixTest, consumesCheckedPrefix) {
  for (const bool useVarint : {false, true}) {
    SCOPED_TRACE(useVarint);
    for (const uint32_t numRows :
         {0u,
          127u,
          128u,
          16'383u,
          16'384u,
          268'435'455u,
          268'435'456u,
          std::numeric_limits<uint32_t>::max()}) {
      SCOPED_TRACE(numRows);
      const auto prefix = serializePrefix(
          nimble::EncodingType::Trivial,
          nimble::DataType::Uint32,
          numRows,
          useVarint);
      const auto bytes = prefix + "payload";
      std::string_view cursor = bytes;
      const auto consumed = nimble::EncodingPrefix::consume(cursor, useVarint);
      EXPECT_EQ(consumed, prefix);
      EXPECT_EQ(consumed.data(), bytes.data());
      EXPECT_EQ(cursor, "payload");
      EXPECT_EQ(
          nimble::EncodingPrefix::readRowCount(consumed, useVarint), numRows);

      for (size_t size = 0; size < prefix.size(); ++size) {
        SCOPED_TRACE(size);
        cursor = std::string_view(prefix).substr(0, size);
        const auto truncated = cursor;
        EXPECT_THROW(
            nimble::EncodingPrefix::consume(cursor, useVarint),
            nimble::NimbleException);
        EXPECT_EQ(cursor, truncated);
      }
    }
  }
}

TEST(EncodingPrefixTest, readsCheckedFixedWidthMetadata) {
  const std::string bytes =
      "\x80\x34\x12"
      "tail";
  std::string_view cursor = bytes;
  EXPECT_EQ(nimble::encoding::readByte(cursor), 128);
  EXPECT_EQ(nimble::encoding::readUint16(cursor), 0x1234);
  EXPECT_EQ(cursor, "tail");

  cursor = {};
  EXPECT_THROW(nimble::encoding::readByte(cursor), nimble::NimbleException);
  EXPECT_TRUE(cursor.empty());
  for (size_t size = 0; size < sizeof(uint16_t); ++size) {
    cursor = std::string_view(bytes).substr(0, size);
    const auto truncated = cursor;
    EXPECT_THROW(nimble::encoding::readUint16(cursor), nimble::NimbleException);
    EXPECT_EQ(cursor, truncated);
  }
  cursor = "\xff\xff";
  EXPECT_EQ(nimble::encoding::readUint16(cursor), 0xffff);
  EXPECT_TRUE(cursor.empty());
}

TEST(EncodingPrefixTest, readsCheckedVarints) {
  const std::array<std::pair<std::string_view, uint32_t>, 9> cases{{
      {{"\x00", 1}, 0},
      {"\x7f", 127},
      {"\x80\x01", 128},
      {"\xff\x7f", 16'383},
      {"\x80\x80\x01", 16'384},
      {"\xff\xff\xff\x7f", 268'435'455},
      {"\x80\x80\x80\x80\x01", 268'435'456},
      {"\xff\xff\xff\xff\x0f", std::numeric_limits<uint32_t>::max()},
      // Non-minimal encodings remain valid when they fit in uint32.
      {{"\x80\x80\x80\x80\x00", 5}, 0},
  }};
  for (const auto& [encoded, expected] : cases) {
    SCOPED_TRACE(expected);
    const auto bytes = std::string(encoded) + "tail";
    std::string_view cursor = bytes;
    EXPECT_EQ(nimble::encoding::readVarint32(cursor), expected);
    EXPECT_EQ(cursor, "tail");
    for (size_t size = 0; size < encoded.size(); ++size) {
      SCOPED_TRACE(size);
      cursor = encoded.substr(0, size);
      const auto truncated = cursor;
      EXPECT_THROW(
          nimble::encoding::readVarint32(cursor), nimble::NimbleException);
      EXPECT_EQ(cursor, truncated);
    }
  }
}

TEST(EncodingPrefixTest, rejectsOverflowingVarints) {
  // The fifth byte may contain only four payload bits, without continuation.
  for (uint32_t lastByte = 16; lastByte <= 255; ++lastByte) {
    SCOPED_TRACE(lastByte);
    const auto bytes = std::string(4, '\x80') + static_cast<char>(lastByte);
    std::string_view cursor = bytes;
    EXPECT_THROW(
        nimble::encoding::readVarint32(cursor), nimble::NimbleException);
    EXPECT_EQ(cursor, bytes);

    const auto prefix = std::string("\x01\x02") + bytes;
    cursor = prefix;
    EXPECT_THROW(
        nimble::EncodingPrefix::consume(cursor, true), nimble::NimbleException);
    EXPECT_EQ(cursor, prefix);
  }
}

TEST(EncodingPrefixTest, readsCheckedLengthPrefixedBytes) {
  const std::string bytes(
      "\x03"
      "a\0b"
      "\x00"
      "tail",
      9);
  std::string_view cursor = bytes;
  const auto payload = nimble::encoding::readLengthPrefixedBytes(cursor);
  EXPECT_EQ(payload, std::string_view("a\0b", 3));
  EXPECT_EQ(payload.data(), bytes.data() + 1);
  EXPECT_TRUE(nimble::encoding::readLengthPrefixedBytes(cursor).empty());
  EXPECT_EQ(cursor, "tail");

  for (size_t size = 0; size < 4; ++size) {
    cursor = std::string_view(bytes).substr(0, size);
    const auto truncated = cursor;
    EXPECT_THROW(
        nimble::encoding::readLengthPrefixedBytes(cursor),
        nimble::NimbleException);
    EXPECT_EQ(cursor, truncated);
  }
  const std::string hugeLength = "\xff\xff\xff\xff\x0f";
  cursor = hugeLength;
  EXPECT_THROW(
      nimble::encoding::readLengthPrefixedBytes(cursor),
      nimble::NimbleException);
  EXPECT_EQ(cursor, hugeLength);
}

TEST(EncodingPrefixTest, factoryRejectsTruncatedTypePrefix) {
  auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  const std::string oneByte{static_cast<char>(nimble::EncodingType::Trivial)};

  for (const std::string_view data :
       {std::string_view{}, std::string_view{oneByte}}) {
    try {
      nimble::EncodingFactory{}.create(*pool, data, nullptr);
      FAIL() << "Expected a truncated prefix to fail";
    } catch (const nimble::NimbleUserError& error) {
      EXPECT_EQ(nimble::error_code::CorruptedFile, error.errorCode());
      EXPECT_EQ("Truncated encoding prefix.", error.errorMessage());
    }
  }
}
