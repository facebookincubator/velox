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

#include "velox/functions/lib/TimeUtils.h"
#include <gtest/gtest.h>

namespace facebook::velox::functions::test {

class TimeUtilsTest : public testing::Test {};

TEST_F(TimeUtilsTest, getFirstDayOfWeek) {
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 1), 7);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 2), 1);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 3), 2);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 4), 3);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 5), 4);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 6), 5);
  EXPECT_EQ(getDayOfFirstDayOfWeek(2024, 7), 6);
}

TEST_F(TimeUtilsTest, getWeakYear) {
  EXPECT_EQ(getWeekYear(2017, 01, 01, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 01, 2, 4), 2016); // 2016W52
  EXPECT_EQ(getWeekYear(2017, 01, 02, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 02, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 03, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 03, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 04, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 04, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 05, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 05, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 06, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 06, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 07, 1, 1), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 01, 07, 2, 4), 2017); // 2017W1
  EXPECT_EQ(getWeekYear(2017, 12, 25, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 25, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 26, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 26, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 27, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 27, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 28, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 28, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 29, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 29, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 30, 1, 1), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 30, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2017, 12, 31, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2017, 12, 31, 2, 4), 2017); // 2017W52
  EXPECT_EQ(getWeekYear(2018, 01, 01, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 01, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 02, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 02, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 03, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 03, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 04, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 04, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 05, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 05, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 06, 1, 1), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 06, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 01, 07, 1, 1), 2018); // 2018W2
  EXPECT_EQ(getWeekYear(2018, 01, 07, 2, 4), 2018); // 2018W1
  EXPECT_EQ(getWeekYear(2018, 12, 25, 1, 1), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 25, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 26, 1, 1), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 26, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 27, 1, 1), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 27, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 28, 1, 1), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 28, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 29, 1, 1), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 29, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 30, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2018, 12, 30, 2, 4), 2018); // 2018W52
  EXPECT_EQ(getWeekYear(2018, 12, 31, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2018, 12, 31, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 01, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 01, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 02, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 02, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 03, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 03, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 04, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 04, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 05, 1, 1), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 05, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 06, 1, 1), 2019); // 2019W2
  EXPECT_EQ(getWeekYear(2019, 01, 06, 2, 4), 2019); // 2019W1
  EXPECT_EQ(getWeekYear(2019, 01, 07, 1, 1), 2019); // 2019W2
  EXPECT_EQ(getWeekYear(2019, 01, 07, 2, 4), 2019); // 2019W2
  EXPECT_EQ(getWeekYear(2019, 12, 25, 1, 1), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 25, 2, 4), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 26, 1, 1), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 26, 2, 4), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 27, 1, 1), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 27, 2, 4), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 28, 1, 1), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 28, 2, 4), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 29, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2019, 12, 29, 2, 4), 2019); // 2019W52
  EXPECT_EQ(getWeekYear(2019, 12, 30, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2019, 12, 30, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2019, 12, 31, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2019, 12, 31, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 01, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 01, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 02, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 02, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 03, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 03, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 04, 1, 1), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 04, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 05, 1, 1), 2020); // 2020W2
  EXPECT_EQ(getWeekYear(2020, 01, 05, 2, 4), 2020); // 2020W1
  EXPECT_EQ(getWeekYear(2020, 01, 06, 1, 1), 2020); // 2020W2
  EXPECT_EQ(getWeekYear(2020, 01, 06, 2, 4), 2020); // 2020W2
  EXPECT_EQ(getWeekYear(2020, 01, 07, 1, 1), 2020); // 2020W2
  EXPECT_EQ(getWeekYear(2020, 01, 07, 2, 4), 2020); // 2020W2
  EXPECT_EQ(getWeekYear(2020, 12, 25, 1, 1), 2020); // 2020W52
  EXPECT_EQ(getWeekYear(2020, 12, 25, 2, 4), 2020); // 2020W52
  EXPECT_EQ(getWeekYear(2020, 12, 26, 1, 1), 2020); // 2020W52
  EXPECT_EQ(getWeekYear(2020, 12, 26, 2, 4), 2020); // 2020W52
  EXPECT_EQ(getWeekYear(2020, 12, 27, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2020, 12, 27, 2, 4), 2020); // 2020W52
  EXPECT_EQ(getWeekYear(2020, 12, 28, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2020, 12, 28, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2020, 12, 29, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2020, 12, 29, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2020, 12, 30, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2020, 12, 30, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2020, 12, 31, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2020, 12, 31, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2021, 01, 01, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 01, 01, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2021, 01, 02, 1, 1), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 01, 02, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2021, 01, 03, 1, 1), 2021); // 2021W2
  EXPECT_EQ(getWeekYear(2021, 01, 03, 2, 4), 2020); // 2020W53
  EXPECT_EQ(getWeekYear(2021, 01, 04, 1, 1), 2021); // 2021W2
  EXPECT_EQ(getWeekYear(2021, 01, 04, 2, 4), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 01, 05, 1, 1), 2021); // 2021W2
  EXPECT_EQ(getWeekYear(2021, 01, 05, 2, 4), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 01, 06, 1, 1), 2021); // 2021W2
  EXPECT_EQ(getWeekYear(2021, 01, 06, 2, 4), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 01, 07, 1, 1), 2021); // 2021W2
  EXPECT_EQ(getWeekYear(2021, 01, 07, 2, 4), 2021); // 2021W1
  EXPECT_EQ(getWeekYear(2021, 12, 25, 1, 1), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2021, 12, 25, 2, 4), 2021); // 2021W51
  EXPECT_EQ(getWeekYear(2021, 12, 26, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 26, 2, 4), 2021); // 2021W51
  EXPECT_EQ(getWeekYear(2021, 12, 27, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 27, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2021, 12, 28, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 28, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2021, 12, 29, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 29, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2021, 12, 30, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 30, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2021, 12, 31, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2021, 12, 31, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2022, 01, 01, 1, 1), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 01, 01, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2022, 01, 02, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 02, 2, 4), 2021); // 2021W52
  EXPECT_EQ(getWeekYear(2022, 01, 03, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 03, 2, 4), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 01, 04, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 04, 2, 4), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 01, 05, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 05, 2, 4), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 01, 06, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 06, 2, 4), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 01, 07, 1, 1), 2022); // 2022W2
  EXPECT_EQ(getWeekYear(2022, 01, 07, 2, 4), 2022); // 2022W1
  EXPECT_EQ(getWeekYear(2022, 12, 25, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 25, 2, 4), 2022); // 2022W51
  EXPECT_EQ(getWeekYear(2022, 12, 26, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 26, 2, 4), 2022); // 2022W52
  EXPECT_EQ(getWeekYear(2022, 12, 27, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 27, 2, 4), 2022); // 2022W52
  EXPECT_EQ(getWeekYear(2022, 12, 28, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 28, 2, 4), 2022); // 2022W52
  EXPECT_EQ(getWeekYear(2022, 12, 29, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 29, 2, 4), 2022); // 2022W52
  EXPECT_EQ(getWeekYear(2022, 12, 30, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 30, 2, 4), 2022); // 2022W52
  EXPECT_EQ(getWeekYear(2022, 12, 31, 1, 1), 2022); // 2022W53
  EXPECT_EQ(getWeekYear(2022, 12, 31, 2, 4), 2022); // 2022W52
}

} // namespace facebook::velox::functions::test
