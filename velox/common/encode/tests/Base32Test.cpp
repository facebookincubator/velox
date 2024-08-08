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

TEST_F(Base32Test, calculateDecodedSizeProperSize) {
  struct TestCase {
    std::string encoded;
    size_t initial_size;
    int expected_decoded;
    size_t expected_size;
  };

  std::vector<TestCase> test_cases = {
      {"ME======", 8, 1, 2},
      {"ME", 2, 1, 2},
      {"MFRA====", 8, 2, 4},
      {"MFRGG===", 8, 3, 5},
      {"NBSWY3DPEB3W64TMMQ======", 24, 11, 18},
      {"NBSWY3DPEB3W64TMMQ", 18, 11, 18}};

  for (const auto& test : test_cases) {
    size_t encoded_size = test.initial_size;
    EXPECT_EQ(
        test.expected_decoded,
        Base32::calculateDecodedSize(test.encoded.c_str(), encoded_size));
    EXPECT_EQ(test.expected_size, encoded_size);
  }
}

TEST_F(Base32Test, errorWhenDecodedStringPartiallyPadded) {
  size_t encoded_size = 9;
  EXPECT_THROW(
      Base32::calculateDecodedSize("MFRA====", encoded_size), VeloxUserError);
}

} // namespace facebook::velox::encoding
