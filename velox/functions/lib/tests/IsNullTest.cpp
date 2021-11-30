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

class IsNullTest : public functions::test::FunctionBaseTest {};

TEST_F(IsNullTest, basic) {
  vector_size_t size = 20;

  // all nulls
  auto allNulls = makeFlatVector<int32_t>(
      size, [](vector_size_t /*row*/) { return 0; }, vectorMaker_.nullEvery(1));
  auto result =
      evaluate<SimpleVector<bool>>("is_null(c0)", makeRowVector({allNulls}));
  for (int i = 0; i < size; ++i) {
    EXPECT_TRUE(result->valueAt(i)) << "at " << i;
  }

  // nulls in odd positions: 0, null, 2, null,...
  auto oddNulls = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return row; },
      vectorMaker_.nullEvery(2, 1));
  result =
      evaluate<SimpleVector<bool>>("is_null(c0)", makeRowVector({oddNulls}));
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(result->valueAt(i), i % 2 == 1) << "at " << i;
  }

  // no nulls
  auto noNulls =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });
  result =
      evaluate<SimpleVector<bool>>("is_null(c0)", makeRowVector({noNulls}));
  for (int i = 0; i < size; ++i) {
    EXPECT_FALSE(result->valueAt(i)) << "at " << i;
  }
}

TEST_F(IsNullTest, somePositions) {
  vector_size_t size = 20;

  // nulls in odd positions: 0, null, 2, null,...
  auto oddNulls = makeFlatVector<int32_t>(
      size,
      [](vector_size_t row) { return row; },
      vectorMaker_.nullEvery(2, 1));

  auto isOdd = [](int i) { return i % 2; };

  auto result =
      evaluate<SimpleVector<bool>>("is_null(c0)", makeRowVector({oddNulls}));
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(result->valueAt(i), isOdd(i)) << "at " << i;
  }
}
