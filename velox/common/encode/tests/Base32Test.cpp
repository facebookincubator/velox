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

#include "velox/common/encode/Base32.h"
#include <gtest/gtest.h>
#include "velox/common/base/tests/GTestUtils.h"

namespace facebook::velox::encoding {

class Base32Test : public ::testing::Test {};

TEST_F(Base32Test, calculateEncodedSizeProperSize) {
  EXPECT_EQ(0, Base32::calculateEncodedSize(0, false));
  EXPECT_EQ(4, Base32::calculateEncodedSize(1, false));
  EXPECT_EQ(5, Base32::calculateEncodedSize(2, false));
  EXPECT_EQ(6, Base32::calculateEncodedSize(3, false));
  EXPECT_EQ(7, Base32::calculateEncodedSize(4, false));

  EXPECT_EQ(0, Base32::calculateEncodedSize(0, true));
  EXPECT_EQ(8, Base32::calculateEncodedSize(1, true));
  EXPECT_EQ(8, Base32::calculateEncodedSize(2, true));
  EXPECT_EQ(8, Base32::calculateEncodedSize(3, true));
  EXPECT_EQ(8, Base32::calculateEncodedSize(4, true));

  EXPECT_EQ(20, Base32::calculateEncodedSize(11, false));
  EXPECT_EQ(24, Base32::calculateEncodedSize(11, true));
}

TEST_F(Base32Test, calculateDecodedSizeProperSize) {
  size_t encoded_size{0};

  encoded_size = 8;
  EXPECT_EQ(1, Base32::calculateDecodedSize("ME======", encoded_size));
  EXPECT_EQ(2, encoded_size);

  encoded_size = 2;
  EXPECT_EQ(1, Base32::calculateDecodedSize("ME", encoded_size));
  EXPECT_EQ(2, encoded_size);

  encoded_size = 9;
  EXPECT_THROW(
      Base32::calculateDecodedSize("MFRA====", encoded_size),
      facebook::velox::encoding::EncoderException);

  encoded_size = 8;
  EXPECT_EQ(2, Base32::calculateDecodedSize("MFRA====", encoded_size));
  EXPECT_EQ(4, encoded_size);

  encoded_size = 8;
  EXPECT_EQ(3, Base32::calculateDecodedSize("MFRGG===", encoded_size));
  EXPECT_EQ(5, encoded_size);

  encoded_size = 24;
  EXPECT_EQ(
      11,
      Base32::calculateDecodedSize("NBSWY3DPEB3W64TMMQ======", encoded_size));
  EXPECT_EQ(18, encoded_size);

  encoded_size = 18;
  EXPECT_EQ(
      11, Base32::calculateDecodedSize("NBSWY3DPEB3W64TMMQ", encoded_size));
  EXPECT_EQ(18, encoded_size);
}

} // namespace facebook::velox::encoding