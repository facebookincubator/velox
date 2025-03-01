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
#include <optional>
#include <utility>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using facebook::velox::functions::test::FunctionBaseTest;

namespace {

class ParsePrestoDataSizeTest : public FunctionBaseTest {};

int128_t parseInt128FromString(const std::string& strValue) {
  return facebook::velox::HugeInt::parse(strValue);
}

TEST_F(ParsePrestoDataSizeTest, success) {
  const auto prestoDataSize = [&](std::optional<std::string> input) {
    return evaluateOnce<int128_t>(
        "parse_presto_data_size(c0)", std::move(input));
  };
  EXPECT_EQ(prestoDataSize("0B"), parseInt128FromString("0"));
  EXPECT_EQ(prestoDataSize("1B"), parseInt128FromString("1"));
  EXPECT_EQ(prestoDataSize("1.2B"), parseInt128FromString("1"));
  EXPECT_EQ(prestoDataSize("1.9B"), parseInt128FromString("1"));
  EXPECT_EQ(prestoDataSize("2.2kB"), parseInt128FromString("2252"));
  EXPECT_EQ(prestoDataSize("2.23kB"), parseInt128FromString("2283"));
  EXPECT_EQ(prestoDataSize("2.23kB"), parseInt128FromString("2283"));
  EXPECT_EQ(prestoDataSize("2.234kB"), parseInt128FromString("2287"));
  EXPECT_EQ(prestoDataSize("3MB"), parseInt128FromString("3145728"));
  EXPECT_EQ(prestoDataSize("4GB"), parseInt128FromString("4294967296"));
  EXPECT_EQ(prestoDataSize("4TB"), parseInt128FromString("4398046511104"));
  EXPECT_EQ(prestoDataSize("5PB"), parseInt128FromString("5629499534213120"));
  EXPECT_EQ(
      prestoDataSize("6EB"), parseInt128FromString("6917529027641081856"));
  EXPECT_EQ(
      prestoDataSize("8YB"),
      parseInt128FromString("9671406556917033397649408"));
  EXPECT_EQ(
      prestoDataSize("7ZB"), parseInt128FromString("8264141345021879123968"));
  EXPECT_EQ(
      prestoDataSize("8YB"),
      parseInt128FromString("9671406556917033397649408"));
  EXPECT_EQ(
      prestoDataSize("6917529027641081856EB"),
      parseInt128FromString("7975367974709495237422842361682067456"));
  EXPECT_EQ(
      prestoDataSize("69175290276410818560EB"),
      parseInt128FromString("79753679747094952374228423616820674560"));
}

TEST_F(ParsePrestoDataSizeTest, failures) {
  const auto prestoDataSize = [&](std::optional<std::string> input) {
    return evaluateOnce<int128_t>(
        "parse_presto_data_size(c0)", std::move(input));
  };
  VELOX_ASSERT_THROW(prestoDataSize(""), "Invalid data size: ''");
  VELOX_ASSERT_THROW(prestoDataSize("0"), "Invalid data size: '0'");
  VELOX_ASSERT_THROW(prestoDataSize("10KB"), "Invalid data size: '10KB'");
  VELOX_ASSERT_THROW(prestoDataSize("KB"), "Invalid data size: 'KB'");
  VELOX_ASSERT_THROW(prestoDataSize("-1B"), "Invalid data size: '-1B'");
  VELOX_ASSERT_THROW(prestoDataSize("12345K"), "Invalid data size: '12345K'");
  VELOX_ASSERT_THROW(prestoDataSize("A12345B"), "Invalid data size: 'A12345B'");
  VELOX_ASSERT_THROW(
      prestoDataSize("99999999999999YB"),
      "Value out of range: '99999999999999YB' ('120892581961461708544797985370825293824B')");
}

} // namespace
