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
#include "velox/type/DecimalUtils.h"
#include <gtest/gtest.h>
#include <sstream>
#include "velox/type/Type.h"

using namespace facebook;
using namespace facebook::velox;

#define MERGE_INT128(UPPER, LOWER) ((int128_t)UPPER << 64) | (LOWER)

TEST(DecimalCast, shortDecimalToString) {
  EXPECT_EQ("123.45", DecimalCasts::ShortDecimalToString(5, 2, 12345));
  EXPECT_EQ("-12.345", DecimalCasts::ShortDecimalToString(5, 3, -12345));
  EXPECT_EQ("0.12345", DecimalCasts::ShortDecimalToString(5, 5, 12345));
  EXPECT_EQ("-0.12345", DecimalCasts::ShortDecimalToString(5, 5, -12345));
  EXPECT_EQ("-0.00123", DecimalCasts::ShortDecimalToString(5, 5, -123));
  EXPECT_EQ("0.00123", DecimalCasts::ShortDecimalToString(5, 5, 123));
  EXPECT_EQ(
      "-1234567890123.45678",
      DecimalCasts::ShortDecimalToString(18, 5, -123456789012345678));
  EXPECT_EQ(
      "1234567890123.45678",
      DecimalCasts::ShortDecimalToString(18, 5, 123456789012345678));
  EXPECT_EQ(
      "123456789012345678",
      DecimalCasts::ShortDecimalToString(18, 0, 123456789012345678));
  EXPECT_EQ("0.00", DecimalCasts::ShortDecimalToString(10, 2, 0));
}

TEST(DecimalCasts, longDecimalToString) {
  EXPECT_EQ("0.12345", DecimalCasts::LongDecimalToString(5, 5, 12345));
  EXPECT_EQ("-0.12345", DecimalCasts::LongDecimalToString(5, 5, -12345));
  EXPECT_EQ("-0.00123", DecimalCasts::LongDecimalToString(5, 5, -123));
  EXPECT_EQ("0.00123", DecimalCasts::LongDecimalToString(5, 5, 123));
  EXPECT_EQ(
      "-1134274556403128210683733967.9994482962",
      DecimalCasts::LongDecimalToString(
          38, 10, MERGE_INT128(0xF777777777777777, 0xEEEEEEEEEEEEEEEE)));
  EXPECT_EQ(
      "-0.00000000000000000259484199970181026066",
      DecimalCasts::LongDecimalToString(
          38, 38, MERGE_INT128(0xFFFFFFFFFFFFFFF1, 0xEEEEEEEEEEEEEEEE)));
  EXPECT_EQ(
      "1234567890123.45678",
      DecimalCasts::LongDecimalToString(18, 5, 123456789012345678));
  EXPECT_EQ(
      "0.00000000000000000000000000000000000000",
      DecimalCasts::LongDecimalToString(38, 38, 0));
}
