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

} // namespace facebook::velox::encoding