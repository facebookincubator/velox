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
#include "velox/dwio/nimble/encodings/common/EncodingUtils.h"
#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

using namespace facebook::nimble;

TEST(EncodingUtilsTest, dataTypeSizeOneByte) {
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Int8));
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Uint8));
  EXPECT_EQ(1, detail::dataTypeSize(DataType::Bool));
}

TEST(EncodingUtilsTest, dataTypeSizeTwoBytes) {
  EXPECT_EQ(2, detail::dataTypeSize(DataType::Int16));
  EXPECT_EQ(2, detail::dataTypeSize(DataType::Uint16));
}

TEST(EncodingUtilsTest, dataTypeSizeFourBytes) {
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Int32));
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Uint32));
  EXPECT_EQ(4, detail::dataTypeSize(DataType::Float));
}

TEST(EncodingUtilsTest, dataTypeSizeEightBytes) {
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Int64));
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Uint64));
  EXPECT_EQ(8, detail::dataTypeSize(DataType::Double));
}

TEST(EncodingUtilsTest, dataTypeSizeUnsupported) {
  EXPECT_THROW(detail::dataTypeSize(DataType::String), NimbleUserError);
  EXPECT_THROW(detail::dataTypeSize(DataType::Undefined), NimbleUserError);
}

TEST(EncodingUtilsTest, writeVarintString) {
  const std::vector<std::string> values{
      "",
      "a",
      std::string(127, 'x'),
      std::string(128, 'y'),
      std::string(130, 'z')};
  size_t totalSize{0};
  for (const auto& value : values) {
    totalSize += varint::varintSize(value.size()) + value.size();
  }
  std::vector<char> buffer(totalSize);
  char* writePos = buffer.data();

  for (const auto& value : values) {
    encoding::writeVarintString(value, writePos);
  }

  const char* readPos = buffer.data();
  for (const auto& value : values) {
    SCOPED_TRACE(testing::Message() << "size=" << value.size());
    EXPECT_EQ(varint::readVarint32(&readPos), value.size());
    EXPECT_EQ(std::string_view(readPos, value.size()), value);
    readPos += value.size();
  }
  EXPECT_EQ(readPos, writePos);
}

TEST(EncodingUtilsTest, copyPackedBits) {
  struct TestCase {
    const char* name;
    uint8_t bitWidth;
    uint32_t sourceValueCount;
    uint32_t sourceValueBase;
    uint64_t sourceBitOffset;
    uint64_t bitCount;
    std::vector<uint64_t> expected;
  };

  for (const auto& testCase : {
           TestCase{
               .name = "byteAligned",
               .bitWidth = 4,
               .sourceValueCount = 16,
               .sourceValueBase = 0,
               .sourceBitOffset = 16,
               .bitCount = 32,
               .expected = {4, 5, 6, 7, 8, 9, 10, 11}},
           TestCase{
               .name = "misaligned",
               .bitWidth = 5,
               .sourceValueCount = 12,
               .sourceValueBase = 3,
               .sourceBitOffset = 5,
               .bitCount = 35,
               .expected = {4, 5, 6, 7, 8, 9, 10}},
       }) {
    SCOPED_TRACE(testCase.name);

    std::array<char, 8> source{};
    std::array<char, 8> output{};
    FixedBitArray sourceBits{source.data(), testCase.bitWidth};
    for (uint32_t i = 0; i < testCase.sourceValueCount; ++i) {
      sourceBits.set(i, i + testCase.sourceValueBase);
    }

    encoding::copyPackedBits(
        {source.data(), source.size()},
        testCase.sourceBitOffset,
        testCase.bitCount,
        output.data());

    FixedBitArray outputBits{output.data(), testCase.bitWidth};
    std::vector<uint64_t> actual;
    actual.reserve(testCase.expected.size());
    for (uint32_t i = 0; i < testCase.expected.size(); ++i) {
      actual.push_back(outputBits.get(i));
    }
    EXPECT_EQ(actual, testCase.expected);
  }
}

TEST(EncodingUtilsTest, copyPackedBitsRejectsEmptyRange) {
  std::array<char, 8> source{};
  std::array<char, 8> output{};
  VELOX_ASSERT_THROW(
      encoding::copyPackedBits(
          {source.data(), source.size()},
          /*sourceBitOffset=*/0,
          /*bitCount=*/0,
          output.data()),
      "Cannot copy zero bits.");
}
