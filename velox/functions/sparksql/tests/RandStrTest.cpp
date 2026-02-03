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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class RandStrTest : public SparkFunctionBaseTest {
 public:
  RandStrTest() {
    // Allow parsing literal integers as INTEGER, not BIGINT.
    options_.parseIntegerAsBigint = false;
  }

  std::optional<StringView>
  randstrSeed(int32_t len, int64_t seed, int32_t partitionIdx = 0) {
    setSparkPartitionId(partitionIdx);
    return evaluateOnce<StringView>(
        fmt::format("randstr({}, {})", len, seed), makeRowVector(ROW({}), 1));
  }
};

TEST_F(RandStrTest, lengthAndCharset) {
  auto s0 = randstrSeed(0, 42);
  ASSERT_TRUE(s0.has_value());
  EXPECT_EQ(s0->size(), 0);

  auto s5 = randstrSeed(5, 42);
  ASSERT_TRUE(s5.has_value());
  EXPECT_EQ(s5->size(), 5);

  // Check that all characters are from [0-9a-zA-Z].
  auto sv = s5.value();
  for (size_t i = 0; i < sv.size(); ++i) {
    const char c = sv.data()[i];
    const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z');
    EXPECT_TRUE(ok) << "Unexpected char: " << c;
  }
}

TEST_F(RandStrTest, seededDeterminismAndPartition) {
  auto a1 = randstrSeed(10, 1234, 0);
  auto a2 = randstrSeed(10, 1234, 0);
  ASSERT_TRUE(a1.has_value());
  ASSERT_TRUE(a2.has_value());
  // Same seed and partition -> same result.
  EXPECT_EQ(a1.value(), a2.value());

  // Different partition should change sequence.
  auto b = randstrSeed(10, 1234, 1);
  ASSERT_TRUE(b.has_value());
  EXPECT_NE(a1.value(), b.value());
}

TEST_F(RandStrTest, differentSeedsProduceDifferentResults) {
  auto r1 = randstrSeed(8, 111);
  auto r2 = randstrSeed(8, 222);
  auto r3 = randstrSeed(8, 333);
  ASSERT_TRUE(r1.has_value());
  ASSERT_TRUE(r2.has_value());
  ASSERT_TRUE(r3.has_value());
  // Different seeds should produce different results.
  EXPECT_NE(r1.value(), r2.value());
  EXPECT_NE(r1.value(), r3.value());
  EXPECT_NE(r2.value(), r3.value());
}

TEST_F(RandStrTest, negativeLengthError) {
  setSparkPartitionId(0);
  VELOX_ASSERT_THROW(
      evaluateOnce<StringView>("randstr(-1, 42)", makeRowVector(ROW({}), 1)),
      "length must be non-negative");
}

TEST_F(RandStrTest, nonConstantSeedError) {
  setSparkPartitionId(0);
  // Non-constant seed should throw an error.
  VELOX_ASSERT_THROW(
      evaluateOnce<StringView>("randstr(5, c0)", std::optional<int32_t>(42)),
      "seed must be a constant value");
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
