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
class EncoderUtilsTest : public ::testing::Test {};

TEST_F(EncoderUtilsTest, isPadded) {
  EXPECT_TRUE(isPadded("ABC=", 4));
  EXPECT_FALSE(isPadded("ABC", 3));
}

TEST_F(EncoderUtilsTest, numPadding) {
  EXPECT_EQ(0, numPadding("ABC", 3));
  EXPECT_EQ(1, numPadding("ABC=", 4));
  EXPECT_EQ(2, numPadding("AB==", 4));
}

} // namespace facebook::velox::encoding
