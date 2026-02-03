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

#include "velox/functions/lib/XORShiftRandom.h"

#include <gtest/gtest.h>

namespace facebook::velox::functions::sparksql::test {
namespace {

using facebook::velox::functions::XORShiftRandom;

class XORShiftRandomTest : public testing::Test {};

// Character pool matching Spark's randstr implementation.
constexpr char kPool[] =
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
constexpr int kPoolSize = 62;

// Helper to generate randstr output using XORShiftRandom.
// Matches Spark's ExpressionImplUtils.randStr implementation.
std::string generateRandStr(int64_t seed, int32_t length) {
  XORShiftRandom rng;
  rng.setSeed(seed);
  std::string result(length, '\0');
  for (int32_t i = 0; i < length; ++i) {
    // Spark uses Math.abs(rng.nextInt() % 62) to select characters.
    int v = std::abs(rng.nextInt() % 62);
    if (v < 10) {
      result[i] = '0' + v;
    } else if (v < 36) {
      result[i] = 'a' + (v - 10);
    } else {
      result[i] = 'A' + (v - 36);
    }
  }
  return result;
}

// Test that XORShiftRandom produces the same results as Spark's XORShiftRandom.
// Expected values from Spark's RandomSuite.scala:
// - randstr(1, 0) = "c"
// - randstr(5, 0) = "ceV0P"
// - randstr(10, 0) = "ceV0PXaR2I"
TEST_F(XORShiftRandomTest, sparkCompatibility) {
  EXPECT_EQ(generateRandStr(0, 1), "c");
  EXPECT_EQ(generateRandStr(0, 5), "ceV0P");
  EXPECT_EQ(generateRandStr(0, 10), "ceV0PXaR2I");
}

// Test determinism: same seed produces same sequence.
TEST_F(XORShiftRandomTest, determinism) {
  XORShiftRandom rng1, rng2;
  rng1.setSeed(12345);
  rng2.setSeed(12345);

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(rng1.nextInt(1000), rng2.nextInt(1000));
  }
}

// Test that different seeds produce different sequences.
TEST_F(XORShiftRandomTest, differentSeeds) {
  XORShiftRandom rng1, rng2;
  rng1.setSeed(1);
  rng2.setSeed(2);

  // Collect first 10 values from each
  std::vector<int32_t> seq1, seq2;
  for (int i = 0; i < 10; ++i) {
    seq1.push_back(rng1.nextInt(1000));
    seq2.push_back(rng2.nextInt(1000));
  }

  // Sequences should differ
  EXPECT_NE(seq1, seq2);
}

// Test bound parameter works correctly.
TEST_F(XORShiftRandomTest, boundParameter) {
  XORShiftRandom rng;
  rng.setSeed(42);

  for (int i = 0; i < 1000; ++i) {
    int32_t val = rng.nextInt(10);
    EXPECT_GE(val, 0);
    EXPECT_LT(val, 10);
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
