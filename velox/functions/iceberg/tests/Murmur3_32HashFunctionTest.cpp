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
#include "velox/functions/iceberg/Murmur3_32HashFunction.h"

#include <gtest/gtest.h>

using namespace facebook::velox::functions::iceberg;

namespace facebook::velox::functions::iceberg::test {

TEST(Murmur3_32HashFunctionTest, bigint) {
  Murmur3_32HashFunction func;
  EXPECT_EQ(func.hashBigint(10), -289985220);
  EXPECT_EQ(func.hashBigint(0), 1669671676);
  EXPECT_EQ(func.hashBigint(-5), 1222806974);
}

TEST(Murmur3_32HashFunctionTest, string) {
  Murmur3_32HashFunction func;

  const auto hash = [&](std::string input) {
    return func.hashString(input.c_str(), input.size());
  };

  EXPECT_EQ(hash("abcdefg"), -2009294074);
  EXPECT_EQ(hash("abc"), -1277324294);
  EXPECT_EQ(hash("abcd"), 1139631978);
  EXPECT_EQ(hash("abcde"), -392455434);
  EXPECT_EQ(hash("测试"), -25843656);
  EXPECT_EQ(hash("测试raul试测"), -912788207);
  EXPECT_EQ(hash(""), 0);
  EXPECT_EQ(hash("Товары"), 1817480714);
  EXPECT_EQ(hash("😀"), -1095487750);
}
}
