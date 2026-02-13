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

// Adapted from Apache Arrow.

#include <gtest/gtest.h>

#include <string>

#include "arrow/util/endian.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"

namespace facebook::velox::parquet::arrow {

TEST(TestTypeToString, PhysicalTypes) {
  ASSERT_STREQ("BOOLEAN", typeToString(Type::kBoolean).c_str());
  ASSERT_STREQ("INT32", typeToString(Type::kInt32).c_str());
  ASSERT_STREQ("INT64", typeToString(Type::kInt64).c_str());
  ASSERT_STREQ("INT96", typeToString(Type::kInt96).c_str());
  ASSERT_STREQ("FLOAT", typeToString(Type::kFloat).c_str());
  ASSERT_STREQ("DOUBLE", typeToString(Type::kDouble).c_str());
  ASSERT_STREQ("BYTE_ARRAY", typeToString(Type::kByteArray).c_str());
  ASSERT_STREQ(
      "FIXED_LEN_BYTE_ARRAY", typeToString(Type::kFixedLenByteArray).c_str());
}

TEST(TestConvertedTypeToString, ConvertedTypes) {
  ASSERT_STREQ("NONE", convertedTypeToString(ConvertedType::kNone).c_str());
  ASSERT_STREQ("UTF8", convertedTypeToString(ConvertedType::kUtf8).c_str());
  ASSERT_STREQ("MAP", convertedTypeToString(ConvertedType::kMap).c_str());
  ASSERT_STREQ(
      "MAP_KEY_VALUE",
      convertedTypeToString(ConvertedType::kMapKeyValue).c_str());
  ASSERT_STREQ("LIST", convertedTypeToString(ConvertedType::kList).c_str());
  ASSERT_STREQ("ENUM", convertedTypeToString(ConvertedType::kEnum).c_str());
  ASSERT_STREQ(
      "DECIMAL", convertedTypeToString(ConvertedType::kDecimal).c_str());
  ASSERT_STREQ("DATE", convertedTypeToString(ConvertedType::kDate).c_str());
  ASSERT_STREQ(
      "TIME_MILLIS", convertedTypeToString(ConvertedType::kTimeMillis).c_str());
  ASSERT_STREQ(
      "TIME_MICROS", convertedTypeToString(ConvertedType::kTimeMicros).c_str());
  ASSERT_STREQ(
      "TIMESTAMP_MILLIS",
      convertedTypeToString(ConvertedType::kTimestampMillis).c_str());
  ASSERT_STREQ(
      "TIMESTAMP_MICROS",
      convertedTypeToString(ConvertedType::kTimestampMicros).c_str());
  ASSERT_STREQ("UINT_8", convertedTypeToString(ConvertedType::kUint8).c_str());
  ASSERT_STREQ(
      "UINT_16", convertedTypeToString(ConvertedType::kUint16).c_str());
  ASSERT_STREQ(
      "UINT_32", convertedTypeToString(ConvertedType::kUint32).c_str());
  ASSERT_STREQ(
      "UINT_64", convertedTypeToString(ConvertedType::kUint64).c_str());
  ASSERT_STREQ("INT_8", convertedTypeToString(ConvertedType::kInt8).c_str());
  ASSERT_STREQ("INT_16", convertedTypeToString(ConvertedType::kInt16).c_str());
  ASSERT_STREQ("INT_32", convertedTypeToString(ConvertedType::kInt32).c_str());
  ASSERT_STREQ("INT_64", convertedTypeToString(ConvertedType::kInt64).c_str());
  ASSERT_STREQ("JSON", convertedTypeToString(ConvertedType::kJson).c_str());
  ASSERT_STREQ("BSON", convertedTypeToString(ConvertedType::kBson).c_str());
  ASSERT_STREQ(
      "INTERVAL", convertedTypeToString(ConvertedType::kInterval).c_str());
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

TEST(TypePrinter, StatisticsTypes) {
  std::string smin;
  std::string smax;
  int32_t intMin = 1024;
  int32_t intMax = 2048;
  smin = std::string(reinterpret_cast<char*>(&intMin), sizeof(int32_t));
  smax = std::string(reinterpret_cast<char*>(&intMax), sizeof(int32_t));
  ASSERT_STREQ("1024", formatStatValue(Type::kInt32, smin).c_str());
  ASSERT_STREQ("2048", formatStatValue(Type::kInt32, smax).c_str());

  int64_t int64Min = 10240000000000;
  int64_t int64Max = 20480000000000;
  smin = std::string(reinterpret_cast<char*>(&int64Min), sizeof(int64_t));
  smax = std::string(reinterpret_cast<char*>(&int64Max), sizeof(int64_t));
  ASSERT_STREQ("10240000000000", formatStatValue(Type::kInt64, smin).c_str());
  ASSERT_STREQ("20480000000000", formatStatValue(Type::kInt64, smax).c_str());

  float floatMin = 1.024f;
  float floatMax = 2.048f;
  smin = std::string(reinterpret_cast<char*>(&floatMin), sizeof(float));
  smax = std::string(reinterpret_cast<char*>(&floatMax), sizeof(float));
  ASSERT_STREQ("1.024", formatStatValue(Type::kFloat, smin).c_str());
  ASSERT_STREQ("2.048", formatStatValue(Type::kFloat, smax).c_str());

  double doubleMin = 1.0245;
  double doubleMax = 2.0489;
  smin = std::string(reinterpret_cast<char*>(&doubleMin), sizeof(double));
  smax = std::string(reinterpret_cast<char*>(&doubleMax), sizeof(double));
  ASSERT_STREQ("1.0245", formatStatValue(Type::kDouble, smin).c_str());
  ASSERT_STREQ("2.0489", formatStatValue(Type::kDouble, smax).c_str());

#if ARROW_LITTLE_ENDIAN
  Int96 int96Min = {{1024, 2048, 4096}};
  Int96 int96Max = {{2048, 4096, 8192}};
#else
  Int96 int96Min = {{2048, 1024, 4096}};
  Int96 int96Max = {{4096, 2048, 8192}};
#endif
  smin = std::string(reinterpret_cast<char*>(&int96Min), sizeof(Int96));
  smax = std::string(reinterpret_cast<char*>(&int96Max), sizeof(Int96));
  ASSERT_STREQ("1024 2048 4096", formatStatValue(Type::kInt96, smin).c_str());
  ASSERT_STREQ("2048 4096 8192", formatStatValue(Type::kInt96, smax).c_str());

  smin = std::string("abcdef");
  smax = std::string("ijklmnop");
  ASSERT_STREQ("abcdef", formatStatValue(Type::kByteArray, smin).c_str());
  ASSERT_STREQ("ijklmnop", formatStatValue(Type::kByteArray, smax).c_str());

  // PARQUET-1357: FormatStatValue truncates binary statistics on zero
  // character.
  smax.push_back('\0');
  ASSERT_EQ(smax, formatStatValue(Type::kByteArray, smax));

  smin = std::string("abcdefgh");
  smax = std::string("ijklmnop");
  ASSERT_STREQ(
      "abcdefgh", formatStatValue(Type::kFixedLenByteArray, smin).c_str());
  ASSERT_STREQ(
      "ijklmnop", formatStatValue(Type::kFixedLenByteArray, smax).c_str());
}

TEST(TestInt96Timestamp, Decoding) {
  auto check = [](int32_t julianDay, uint64_t nanoseconds) {
#if ARROW_LITTLE_ENDIAN
    Int96 i96{
        static_cast<uint32_t>(nanoseconds),
        static_cast<uint32_t>(nanoseconds >> 32),
        static_cast<uint32_t>(julianDay)};
#else
    Int96 i96{
        static_cast<uint32_t>(nanoseconds >> 32),
        static_cast<uint32_t>(nanoseconds),
        static_cast<uint32_t>(julianDay)};
#endif
    // Official formula according to.
    // https://github.com/apache/parquet-format/pull/49
    int64_t expected =
        (julianDay - 2440588) * (86400LL * 1000 * 1000 * 1000) + nanoseconds;
    int64_t actual = int96GetNanoSeconds(i96);
    ASSERT_EQ(expected, actual);
  };

  // [2333837, 2547339] Is the range of Julian days that can be converted to.
  // 64-Bit Unix timestamps.
  check(2333837, 0);
  check(2333855, 0);
  check(2547330, 0);
  check(2547338, 0);
  check(2547339, 0);

  check(2547330, 13);
  check(2547330, 32769);
  check(2547330, 87654);
  check(2547330, 0x123456789abcdefULL);
  check(2547330, 0xfedcba9876543210ULL);
  check(2547339, 0xffffffffffffffffULL);
}

#if !(defined(_WIN32) || defined(__CYGWIN__))
#pragma GCC diagnostic pop
#elif _MSC_VER
#pragma warning(pop)
#endif

} // namespace facebook::velox::parquet::arrow
