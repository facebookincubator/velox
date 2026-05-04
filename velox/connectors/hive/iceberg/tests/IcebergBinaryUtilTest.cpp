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

#include "velox/connectors/hive/iceberg/IcebergBinaryUtil.h"

#include <gtest/gtest.h>
#include <optional>
#include <string>

using namespace facebook::velox::connector::hive::iceberg;

class IcebergBinaryUtilTest : public testing::Test {
 protected:
  void testRoundUpBinary(
      const std::string& input,
      int32_t truncateLength,
      const std::optional<std::string>& expected) {
    EXPECT_EQ(roundUpBinary(input, truncateLength), expected);
  }
};

TEST_F(IcebergBinaryUtilTest, roundUpBinary) {
  // Basic binary data with truncation.
  std::string binary = "Hello, world!";
  // Empty truncation returns nullopt.
  testRoundUpBinary(binary, 0, std::nullopt);
  // 'o' (0x6F) -> 'p' (0x70).
  testRoundUpBinary(binary, 5, "Hellp");
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(binary, binary.length(), binary);
  testRoundUpBinary(binary, binary.length() + 10, binary);

  // Test with numeric data.
  std::string numeric = "Customer#000001500";
  // '5' (0x35) -> '6' (0x36).
  testRoundUpBinary(numeric, 16, "Customer#0000016");

  // Test with binary data containing high bytes.
  std::string highBytes = "data\xFE\xFD";
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(highBytes, 6, highBytes);
  // Truncate to 5 bytes "data\xFE", 0xFE -> 0xFF.
  testRoundUpBinary(highBytes, 5, "data\xFF");

  // Test with all 0xFF bytes - should return nullopt.
  std::string allFF = "\xFF\xFF\xFF";
  testRoundUpBinary(allFF, 1, std::nullopt);
  testRoundUpBinary(allFF, 2, std::nullopt);
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(allFF, 3, allFF);

  // Test with trailing 0xFF bytes.
  std::string trailingFF = "abc\xFF\xFF";
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(trailingFF, 5, trailingFF);
  // Truncate to 4 bytes "abc\xFF", 0xFF overflows, 'c' (0x63) -> 'd' (0x64).
  testRoundUpBinary(trailingFF, 4, "abd");
  // Truncate to 3 bytes "abc", 'c' (0x63) -> 'd' (0x64).
  testRoundUpBinary(trailingFF, 3, "abd");

  // Test empty string.
  std::string empty = "";
  testRoundUpBinary(empty, 0, std::nullopt);
  testRoundUpBinary(empty, 5, "");

  // Test single byte.
  std::string single = "a";
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(single, 1, "a");
  testRoundUpBinary(single, 10, "a");

  // Test incrementing single byte with truncation.
  std::string singleZ = "zz";
  // Truncate to 1 byte "z", 'z' (0x7A) -> '{' (0x7B).
  testRoundUpBinary(singleZ, 1, "{");

  // Test with null bytes.
  std::string withNull = std::string("ab\0cd", 5);
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(withNull, 5, withNull);
  // Truncate to 4 bytes "ab\0c", 'c' (0x63) -> 'd' (0x64).
  testRoundUpBinary(withNull, 4, std::string("ab\0d", 4));

  // Test boundary case: 0xFE -> 0xFF.
  std::string boundaryFE = "test\xFE";
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(boundaryFE, 5, boundaryFE);
  // Truncate to 5 bytes and increment would give same result.
  std::string boundaryFE2 = std::string("test\xFE", 5) + "abc";
  testRoundUpBinary(boundaryFE2, 5, "test\xFF");

  // Test mixed case with overflow in middle.
  std::string mixedOverflow = "a\xFF\xFFz";
  // Truncate to 3 bytes "a\xFF\xFF", both 0xFF overflow, 'a' (0x61) -> 'b'
  // (0x62).
  testRoundUpBinary(mixedOverflow, 3, "b");

  // Test truncation removes trailing bytes after increment.
  std::string longString = "abcdefgh";
  // Truncate to 3 bytes "abc", 'c' (0x63) -> 'd' (0x64), result is "abd".
  testRoundUpBinary(longString, 3, "abd");
  // Truncate to 5 bytes "abcde", 'e' (0x65) -> 'f' (0x66), result is "abcdf".
  testRoundUpBinary(longString, 5, "abcdf");

  // Test with UTF-8 multi-byte sequences (treated as raw bytes).
  std::string utf8Bytes = "café";
  // Truncate to 3 bytes "caf", 'f' (0x66) -> 'g' (0x67).
  testRoundUpBinary(utf8Bytes, 3, "cag");
  // No truncation needed - returns input unchanged.
  testRoundUpBinary(utf8Bytes, 5, utf8Bytes);
  // Truncate to 5 bytes and increment last byte.
  std::string utf8Bytes2 = "café!";
  // Truncate to 5 bytes "café" (caf + 0xC3 0xA9), 0xA9 -> 0xAA.
  testRoundUpBinary(utf8Bytes2, 5, "caf\xC3\xAA");

  // Test with INVALID UTF-8 sequences - this is the key use case for
  // roundUpBinary. These sequences would cause roundUpUtf8 to fail, but
  // roundUpBinary treats them as raw bytes.

  // Invalid UTF-8: lone continuation byte 0x80.
  std::string invalidUtf8_1 = std::string("test\x80", 5);
  testRoundUpBinary(invalidUtf8_1, 5, invalidUtf8_1);
  testRoundUpBinary(invalidUtf8_1, 4, "tesu");

  // Invalid UTF-8: incomplete multi-byte sequence (0xC3 without continuation).
  std::string invalidUtf8_2 = std::string("data\xC3", 5);
  testRoundUpBinary(invalidUtf8_2, 5, invalidUtf8_2);
  testRoundUpBinary(invalidUtf8_2, 4, "datb");

  // Invalid UTF-8: overlong encoding (0xC0 0x80 for null byte).
  std::string invalidUtf8_3 = std::string("ab\xC0\x80", 4);
  testRoundUpBinary(invalidUtf8_3, 4, invalidUtf8_3);
  testRoundUpBinary(invalidUtf8_3, 3, std::string("ab\xC1", 3));

  // Invalid UTF-8: invalid start byte 0xFE.
  std::string invalidUtf8_4 = std::string("xyz\xFE", 4);
  testRoundUpBinary(invalidUtf8_4, 4, invalidUtf8_4);
  testRoundUpBinary(invalidUtf8_4, 3, "xy{");

  // Invalid UTF-8: truncated 3-byte sequence (0xE0 0x80 without third byte).
  std::string invalidUtf8_5 = std::string("foo\xE0\x80", 5);
  testRoundUpBinary(invalidUtf8_5, 5, invalidUtf8_5);
  testRoundUpBinary(invalidUtf8_5, 4, std::string("foo\xE1", 4));

  // Invalid UTF-8: sequence with 0xFF (which is never valid in UTF-8).
  std::string invalidUtf8_6 = std::string("bar\xFF", 4);
  testRoundUpBinary(invalidUtf8_6, 4, invalidUtf8_6);
  // Truncate to 3 bytes "bar", 'r' (0x72) -> 's' (0x73).
  testRoundUpBinary(invalidUtf8_6, 3, "bas");

  // Test with all 0xFF in invalid UTF-8 context.
  std::string invalidUtf8_7 = std::string("\xFF\xFF\xFF", 3);
  testRoundUpBinary(invalidUtf8_7, 2, std::nullopt);
  testRoundUpBinary(invalidUtf8_7, 3, invalidUtf8_7);
}
