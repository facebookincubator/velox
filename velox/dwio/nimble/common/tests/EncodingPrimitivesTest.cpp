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

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::encoding::test {

// --- write/read round-trips for typed helpers ---

TEST(EncodingPrimitivesTest, writeReadUint32) {
  char buf[16];
  char* wpos = buf;
  writeUint32(12345, wpos);
  EXPECT_EQ(wpos, buf + sizeof(uint32_t));

  const char* rpos = buf;
  EXPECT_EQ(readUint32(rpos), 12345u);
  EXPECT_EQ(rpos, buf + sizeof(uint32_t));
}

TEST(EncodingPrimitivesTest, writeReadUint64) {
  char buf[16];
  char* wpos = buf;
  writeUint64(0xDEADBEEFCAFEull, wpos);
  EXPECT_EQ(wpos, buf + sizeof(uint64_t));

  const char* rpos = buf;
  EXPECT_EQ(readUint64(rpos), 0xDEADBEEFCAFEull);
  EXPECT_EQ(rpos, buf + sizeof(uint64_t));
}

TEST(EncodingPrimitivesTest, writeReadChar) {
  char buf[4];
  char* wpos = buf;
  writeChar('Z', wpos);
  EXPECT_EQ(wpos, buf + sizeof(char));

  const char* rpos = buf;
  EXPECT_EQ(readChar(rpos), 'Z');
  EXPECT_EQ(rpos, buf + sizeof(char));
}

// --- write<T>/read<T> template round-trips ---

TEST(EncodingPrimitivesTest, writeReadTemplateInt32) {
  char buf[16];
  char* wpos = buf;
  write<int32_t>(-42, wpos);

  const char* rpos = buf;
  EXPECT_EQ((read<int32_t>(rpos)), -42);
}

TEST(EncodingPrimitivesTest, writeReadTemplateFloat) {
  char buf[16];
  char* wpos = buf;
  write<float>(3.14f, wpos);

  const char* rpos = buf;
  EXPECT_FLOAT_EQ((read<float>(rpos)), 3.14f);
}

TEST(EncodingPrimitivesTest, writeReadTemplateDouble) {
  char buf[16];
  char* wpos = buf;
  write<double>(2.718281828, wpos);

  const char* rpos = buf;
  EXPECT_DOUBLE_EQ((read<double>(rpos)), 2.718281828);
}

TEST(EncodingPrimitivesTest, writeReadTemplateUint64) {
  char buf[16];
  char* wpos = buf;
  write<uint64_t>(999999999999ull, wpos);

  const char* rpos = buf;
  EXPECT_EQ((read<uint64_t>(rpos)), 999999999999ull);
}

// --- writeString/readString with 4-byte length prefix ---

TEST(EncodingPrimitivesTest, writeReadString) {
  char buf[128];
  char* wpos = buf;
  std::string_view input = "hello";
  writeString(input, wpos);
  // Should advance by 4 (length prefix) + 5 (chars)
  EXPECT_EQ(wpos, buf + sizeof(uint32_t) + 5);

  const char* rpos = buf;
  auto result = readString(rpos);
  EXPECT_EQ(result, "hello");
  EXPECT_EQ(rpos, buf + sizeof(uint32_t) + 5);
}

TEST(EncodingPrimitivesTest, writeReadStringEmpty) {
  char buf[16];
  char* wpos = buf;
  writeString("", wpos);
  EXPECT_EQ(wpos, buf + sizeof(uint32_t));

  const char* rpos = buf;
  auto result = readString(rpos);
  EXPECT_EQ(result, "");
  EXPECT_EQ(result.size(), 0);
}

// --- readOwnedString ---

TEST(EncodingPrimitivesTest, readOwnedString) {
  char buf[128];
  char* wpos = buf;
  writeString("owned", wpos);

  const char* rpos = buf;
  std::string result = readOwnedString(rpos);
  EXPECT_EQ(result, "owned");
  // Result should be an independent std::string
  EXPECT_EQ(rpos, buf + sizeof(uint32_t) + 5);
}

// --- writeBytes ---

TEST(EncodingPrimitivesTest, writeBytes) {
  char buf[32];
  char* wpos = buf;
  std::string_view data = "raw bytes";
  writeBytes(data, wpos);
  EXPECT_EQ(wpos, buf + data.size());
  EXPECT_EQ(std::string_view(buf, data.size()), "raw bytes");
}

// --- peek ---

TEST(EncodingPrimitivesTest, peekDoesNotAdvance) {
  char buf[16];
  char* wpos = buf;
  write<uint32_t>(42, wpos);

  const char* rpos = buf;
  auto val = peek<uint32_t>(rpos);
  EXPECT_EQ(val, 42u);
  // peek should not advance the pointer
  EXPECT_EQ(rpos, buf);
}

TEST(EncodingPrimitivesTest, peekFloat) {
  char buf[16];
  char* wpos = buf;
  write<float>(1.5f, wpos);

  const char* rpos = buf;
  EXPECT_FLOAT_EQ((peek<float>(rpos)), 1.5f);
  EXPECT_EQ(rpos, buf);
}

// --- Sequential writes and pointer advancement ---

TEST(EncodingPrimitivesTest, sequentialWritesThenReads) {
  char buf[128];
  char* wpos = buf;
  write<uint32_t>(100, wpos);
  write<int64_t>(-200, wpos);
  write<float>(1.5f, wpos);
  writeString("test", wpos);

  const char* rpos = buf;
  EXPECT_EQ((read<uint32_t>(rpos)), 100u);
  EXPECT_EQ((read<int64_t>(rpos)), -200);
  EXPECT_FLOAT_EQ((read<float>(rpos)), 1.5f);
  EXPECT_EQ(readString(rpos), "test");

  // Both pointers should have advanced the same amount
  EXPECT_EQ(wpos - buf, rpos - buf);
}

// --- write/read via string_view template specialization ---

TEST(EncodingPrimitivesTest, writeReadTemplateStringView) {
  char buf[128];
  char* wpos = buf;
  write<std::string_view>(std::string_view("via template"), wpos);

  const char* rpos = buf;
  auto result = read<std::string_view, std::string_view>(rpos);
  EXPECT_EQ(result, "via template");
}

TEST(EncodingPrimitivesTest, writeReadTemplateStdString) {
  char buf[128];
  char* wpos = buf;
  std::string input = "std string";
  write<const std::string&>(input, wpos);

  const char* rpos = buf;
  auto result = read<std::string, std::string>(rpos);
  EXPECT_EQ(result, "std string");
}

} // namespace facebook::nimble::encoding::test
