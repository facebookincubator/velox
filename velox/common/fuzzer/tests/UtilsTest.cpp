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

#include "velox/common/fuzzer/Utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include "velox/type/Variant.h"

namespace facebook::velox::fuzzer::test {

class UtilsTest : public testing::Test {};

TEST_F(UtilsTest, testRuleList) {
  auto simple = RuleList(
      std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("Hello"),
          std::make_shared<ConstantRule>(","),
          std::make_shared<ConstantRule>(" "),
          std::make_shared<ConstantRule>("world"),
          std::make_shared<ConstantRule>("!"),
      });
  ASSERT_EQ(simple.generate(), "Hello, world!");

  FuzzerGenerator rng;
  auto fuzz = RuleList(
      std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("Hello"),
          std::make_shared<ConstantRule>(","),
          std::make_shared<ConstantRule>(" "),
          std::make_shared<WordRule>(rng),
          std::make_shared<ConstantRule>("!"),
      });
  ASSERT_TRUE(
      std::regex_match(fuzz.generate(), std::regex("Hello, \\w{1,20}!")));
}

TEST_F(UtilsTest, testOptionalRule) {
  FuzzerGenerator rng;
  auto simple = std::make_shared<OptionalRule>(
      rng,
      std::make_shared<RuleList>(std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("a"),
      }));
  ASSERT_TRUE(std::regex_match(simple->generate(), std::regex("^(|a)$")));
}

TEST_F(UtilsTest, testRepeatingRule) {
  FuzzerGenerator rng;
  auto simple = std::make_shared<RepeatingRule>(
      rng,
      std::make_shared<RuleList>(std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("a"),
      }),
      2,
      5);
  ASSERT_TRUE(std::regex_match(simple->generate(), std::regex("^a{2,5}$")));

  auto fuzz = std::make_shared<RepeatingRule>(
      rng,
      std::make_shared<RuleList>(std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("a"),
          std::make_shared<WordRule>(rng, 1, 1, false),
      }),
      2,
      5);
  ASSERT_TRUE(std::regex_match(fuzz->generate(), std::regex("^\\w{4,10}$")));
}

TEST_F(UtilsTest, testConstantRule) {
  auto rule = std::make_shared<ConstantRule>("a");
  ASSERT_EQ(rule->generate(), "a");

  auto rule_list = RuleList(
      std::vector<std::shared_ptr<Rule>>{
          std::make_shared<ConstantRule>("a"),
          std::make_shared<ConstantRule>("b"),
          std::make_shared<ConstantRule>("c")});
  ASSERT_EQ(rule_list.generate(), "abc");
}

TEST_F(UtilsTest, testStringRule) {
  FuzzerGenerator rng;
  auto simple = std::make_shared<StringRule>(rng);
  ASSERT_TRUE(
      std::regex_match(
          simple->generate(), std::regex("^[\x21-\x7F]+$"))); // printable ascii
  ASSERT_FALSE(std::regex_match(simple->generate(), std::regex("^\\w+$")));
  ASSERT_FALSE(std::regex_match(simple->generate(), std::regex("^\\d+$")));

  auto specified_flexible = std::make_shared<StringRule>(
      rng, std::vector<UTF8CharList>{UTF8CharList::ASCII}, 3, 7, true);
  ASSERT_TRUE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^[\\x21-\\x7F]{3,7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^[\\x21-\\x7F]{0,2}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^[\\x21-\\x7F]{8,}$")));

  auto specified_strict = std::make_shared<StringRule>(
      rng, std::vector<UTF8CharList>{UTF8CharList::ASCII}, 3, 7, false);
  ASSERT_TRUE(
      std::regex_match(
          specified_strict->generate(), std::regex("^[\x21-\x7F]{7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_strict->generate(), std::regex("^[\x21-\x7F]{8}$")));
}

TEST_F(UtilsTest, testWordRule) {
  FuzzerGenerator rng;
  auto simple = std::make_shared<WordRule>(rng);
  ASSERT_TRUE(std::regex_match(simple->generate(), std::regex("^[a-zA-Z]+$")));
  ASSERT_TRUE(std::regex_match(simple->generate(), std::regex("^\\w+$")));
  ASSERT_FALSE(std::regex_match(simple->generate(), std::regex("^\\d+$")));
  ASSERT_FALSE(std::regex_match(simple->generate(), std::regex("^\\W+$")));

  auto specified_flexible = std::make_shared<WordRule>(rng, 3, 7, true);
  ASSERT_TRUE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^[a-zA-Z]{3,7}$")));
  ASSERT_TRUE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\w{3,7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\d{3,7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\w{0,2}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\w{8,}$")));

  auto specified_strict = std::make_shared<WordRule>(rng, 3, 7, false);
  ASSERT_TRUE(
      std::regex_match(specified_strict->generate(), std::regex("^\\w{7}$")));
  ASSERT_FALSE(
      std::regex_match(specified_strict->generate(), std::regex("^\\w{8}$")));
}

TEST_F(UtilsTest, testNumRule) {
  FuzzerGenerator rng;
  auto simple = std::make_shared<NumRule>(rng);
  ASSERT_TRUE(std::regex_match(simple->generate(), std::regex("^\\d+$")));
  ASSERT_FALSE(std::regex_match(simple->generate(), std::regex("^\\D+$")));

  auto specified_flexible = std::make_shared<NumRule>(rng, 3, 7, true);
  ASSERT_TRUE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\d{3,7}$")));
  ASSERT_TRUE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\w{3,7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\D{3,7}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\d{0,2}$")));
  ASSERT_FALSE(
      std::regex_match(
          specified_flexible->generate(), std::regex("^\\d{8,}$")));

  auto specified_strict = std::make_shared<NumRule>(rng, 3, 7, false);
  ASSERT_TRUE(
      std::regex_match(specified_strict->generate(), std::regex("^\\d{7}$")));
  ASSERT_FALSE(
      std::regex_match(specified_strict->generate(), std::regex("^\\d{8}$")));
}

namespace {

// Classifies 10'000 values drawn with 'dataSpec' by sign and IEEE 754 class.
template <typename T>
std::set<std::string> drawKinds(DataSpec dataSpec) {
  using Bits =
      std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
  using Limits = std::numeric_limits<T>;
  constexpr auto kQuietBit = Bits{1} << (Limits::digits - 2);
  constexpr auto kPayloadMask = kQuietBit - 1;
  FuzzerGenerator rng;
  std::set<std::string> kinds;
  for (auto i = 0; i < 10'000; ++i) {
    const auto value = rand<T>(rng, dataSpec);
    const std::string sign = std::signbit(value) ? "-" : "";
    const auto bits = std::bit_cast<Bits>(value);
    if (std::isnan(value)) {
      if ((bits & kQuietBit) == 0) {
        kinds.insert(sign + "signalingNaN");
      } else if ((bits & kPayloadMask) != 0) {
        kinds.insert(sign + "payloadNaN");
      } else {
        kinds.insert(sign + "nan");
      }
    } else if (std::isinf(value)) {
      kinds.insert(sign + "inf");
    } else if (value == 0) {
      kinds.insert(sign + "0");
    } else if (std::abs(value) == Limits::max()) {
      kinds.insert(sign + "max");
    } else if (std::abs(value) == Limits::min()) {
      kinds.insert(sign + "min");
    } else if (std::abs(value) == Limits::denorm_min()) {
      kinds.insert(sign + "denormMin");
    } else if (std::abs(value) < Limits::min()) {
      kinds.insert(sign + "subnormal");
    } else if (std::abs(value) < 1) {
      kinds.insert(sign + "unit");
    } else {
      kinds.insert(sign + "other");
    }
  }
  return kinds;
}

} // namespace

TEST_F(UtilsTest, floatingPointDataSpec) {
  using testing::UnorderedElementsAre;
  auto test = [](auto tag) {
    using T = decltype(tag);
    EXPECT_THAT(
        drawKinds<T>(DataSpec{false, false}), UnorderedElementsAre("unit"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{true, true}),
        UnorderedElementsAre("unit", "nan", "inf"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{
            .includeNaN = false,
            .includeInfinity = false,
            .includeNegativeInfinity = true,
        }),
        UnorderedElementsAre("unit", "-inf"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{
            .includeNaN = false,
            .includeInfinity = false,
            .includeSignedZeros = true,
        }),
        UnorderedElementsAre("unit", "0", "-0"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{
            .includeNaN = false,
            .includeInfinity = false,
            .includeNonCanonicalNaN = true,
        }),
        UnorderedElementsAre(
            "unit",
            "-nan",
            "payloadNaN",
            "-payloadNaN",
            "signalingNaN",
            "-signalingNaN"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{
            .includeNaN = false,
            .includeInfinity = false,
            .includeExtremes = true,
        }),
        UnorderedElementsAre(
            "unit",
            "max",
            "-max",
            "min",
            "-min",
            "denormMin",
            "-denormMin",
            "subnormal",
            "-subnormal"));
    EXPECT_THAT(
        drawKinds<T>(DataSpec{
            .includeNaN = false,
            .includeInfinity = false,
            .includeNegative = true,
        }),
        UnorderedElementsAre("unit", "-unit"));
  };
  {
    SCOPED_TRACE("REAL");
    test(float{});
  }
  {
    SCOPED_TRACE("DOUBLE");
    test(double{});
  }
}

} // namespace facebook::velox::fuzzer::test
