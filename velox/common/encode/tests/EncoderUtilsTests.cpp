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

#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/encode/EncoderUtils.h"

namespace facebook::velox::encoding {
class EncoderUtilsTest : public ::testing::Test {
 protected:
  const int binaryBlockByteSize = 3;
  const int encodedBlockByteSize = 4;
};

TEST_F(EncoderUtilsTest, isPadded) {
  EXPECT_TRUE(isPadded("ABC="));
  EXPECT_FALSE(isPadded("ABC"));
}

TEST_F(EncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, numPadding("ABC"));
  EXPECT_EQ(1, numPadding("ABC="));
  EXPECT_EQ(2, numPadding("AB=="));
}

TEST_F(EncoderUtilsTest, CalculateEncodedSizeTest) {
  EXPECT_EQ(
      calculateEncodedSize(3, true, binaryBlockByteSize, encodedBlockByteSize),
      4);
  EXPECT_EQ(
      calculateEncodedSize(3, false, binaryBlockByteSize, encodedBlockByteSize),
      4);
  EXPECT_EQ(
      calculateEncodedSize(6, true, binaryBlockByteSize, encodedBlockByteSize),
      8);
  EXPECT_EQ(
      calculateEncodedSize(0, true, binaryBlockByteSize, encodedBlockByteSize),
      0);
}

TEST_F(EncoderUtilsTest, BaseReverseLookupTest) {
  std::array<uint8_t, 256> reverseIndex{};
  reverseIndex.fill(255);
  reverseIndex['A'] = 0;
  reverseIndex['B'] = 1;
  uint8_t reverseLookupValue = 2;

  EXPECT_EQ(reverseLookup('A', reverseIndex, reverseLookupValue).value(), 0);

  EXPECT_EQ(reverseLookup('B', reverseIndex, reverseLookupValue).value(), 1);

  EXPECT_EQ(
      reverseLookup('Z', reverseIndex, reverseLookupValue).error(),
      Status::UserError(
          "decode() - invalid input string: invalid character 'Z'"));
}

} // namespace facebook::velox::encoding
