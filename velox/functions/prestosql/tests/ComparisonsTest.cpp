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
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"

using namespace facebook::velox;

class ComparisonsTest : public functions::test::FunctionBaseTest {};

TEST_F(ComparisonsTest, between) {
  std::vector<std::tuple<int32_t, bool>> testData = {
      {0, false}, {1, true}, {4, true}, {5, true}, {10, false}, {-1, false}};

  auto result = evaluate<SimpleVector<bool>>(
      "c0 between 1 and 5",
      makeRowVector({makeFlatVector<int32_t, 0>(testData)}));

  for (int i = 0; i < testData.size(); ++i) {
    EXPECT_EQ(result->valueAt(i), std::get<1>(testData[i])) << "at " << i;
  }
}

TEST_F(ComparisonsTest, betweenVarchar) {
  using S = StringView;

  const auto between = [&](std::optional<std::string> s) {
    auto expr = "c0 between 'mango' and 'pear'";
    if (s.has_value()) {
      return evaluateOnce<bool>(expr, std::optional(S(s.value())));
    } else {
      return evaluateOnce<bool>(expr, std::optional<S>());
    }
  };

  EXPECT_EQ(std::nullopt, between(std::nullopt));
  EXPECT_EQ(false, between(""));
  EXPECT_EQ(false, between("apple"));
  EXPECT_EQ(false, between("pineapple"));
  EXPECT_EQ(true, between("mango"));
  EXPECT_EQ(true, between("orange"));
  EXPECT_EQ(true, between("pear"));
}

TEST_F(ComparisonsTest, betweenDate) {
  auto parseDate = [](const std::string& dateStr) {
    Date returnDate;
    parseTo(dateStr, returnDate);
    return returnDate;
  };
  std::vector<std::tuple<Date, bool>> testData = {
      {parseDate("2019-05-01"), false},
      {parseDate("2019-06-01"), true},
      {parseDate("2019-07-01"), true},
      {parseDate("2020-05-31"), true},
      {parseDate("2020-06-01"), true},
      {parseDate("2020-07-01"), false}};

  auto result = evaluate<SimpleVector<bool>>(
      "c0 between cast(\'2019-06-01\' as date) and cast(\'2020-06-01\' as date)",
      makeRowVector({makeFlatVector<Date, 0>(testData)}));

  for (int i = 0; i < testData.size(); ++i) {
    EXPECT_EQ(result->valueAt(i), std::get<1>(testData[i])) << "at " << i;
  }
}
