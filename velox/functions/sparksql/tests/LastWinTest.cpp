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
#include "velox/functions/sparksql/LastWin.h"

#include <bit>
#include <limits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

using Positions = std::vector<std::pair<vector_size_t, vector_size_t>>;

class LastWinTest : public SparkFunctionBaseTest {
 protected:
  Positions resolve(
      const BaseVector& keys,
      vector_size_t offset,
      vector_size_t numKeys,
      bool throwOnDuplicateKey) {
    const DecodedVector decoded{keys};
    Positions result;
    for (const auto& entry :
         lastWin_.resolve(decoded, offset, numKeys, throwOnDuplicateKey)) {
      result.emplace_back(entry.key, entry.value);
    }
    return result;
  }

  LastWin<SparkMapKeyEquality> lastWin_;
};

TEST_F(LastWinTest, lastWinKeepsFirstPositionAndLastValue) {
  const auto keys = makeFlatVector<int64_t>({1, 2, 1, 3, 1});
  EXPECT_EQ(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 4}, {1, 1}, {3, 3}}));
}

TEST_F(LastWinTest, exceptionThrowsOnFirstRepeat) {
  const auto keys = makeFlatVector<int64_t>({1, 2, 3, 2, 1});
  VELOX_ASSERT_THROW(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: 2");
}

TEST_F(LastWinTest, distinctKeys) {
  const auto keys = makeFlatVector<int64_t>({5, 6, 7});
  for (const bool throwOnDuplicateKey : {true, false}) {
    EXPECT_EQ(
        resolve(*keys, 0, 3, throwOnDuplicateKey),
        (Positions{{0, 0}, {1, 1}, {2, 2}}));
  }
}

TEST_F(LastWinTest, emptyRun) {
  const auto keys = makeFlatVector<int64_t>({1, 1});
  EXPECT_THAT(
      resolve(*keys, 1, 0, /*throwOnDuplicateKey=*/true), testing::IsEmpty());
}

TEST_F(LastWinTest, offset) {
  const auto keys = makeFlatVector<int64_t>({9, 9, 1, 2, 1});
  EXPECT_EQ(
      resolve(*keys, 2, 3, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 2}, {1, 1}}));
  VELOX_ASSERT_THROW(
      resolve(*keys, 2, 3, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: 1");
}

TEST_F(LastWinTest, nullKeyIsAnError) {
  const auto keys = makeNullableFlatVector<int64_t>({1, std::nullopt, 2});
  const auto dictionaryKeys = BaseVector::wrapInDictionary(
      makeNulls({false, true}),
      makeIndices({0, 0}),
      2,
      makeFlatVector<int64_t>({1}));
  for (const bool throwOnDuplicateKey : {true, false}) {
    VELOX_ASSERT_THROW(
        resolve(*keys, 0, 3, throwOnDuplicateKey),
        "Cannot use null as map key!");
    VELOX_ASSERT_THROW(
        resolve(*dictionaryKeys, 0, 2, throwOnDuplicateKey),
        "Cannot use null as map key!");
  }
}

TEST_F(LastWinTest, errorsFollowKeyOrder) {
  {
    const auto keys = makeNullableFlatVector<int64_t>({1, 1, std::nullopt});
    VELOX_ASSERT_THROW(
        resolve(*keys, 0, 3, /*throwOnDuplicateKey=*/true),
        "Duplicate map key was found: 1");
    VELOX_ASSERT_THROW(
        resolve(*keys, 0, 3, /*throwOnDuplicateKey=*/false),
        "Cannot use null as map key!");
  }
  {
    const auto keys = makeNullableFlatVector<int64_t>({1, std::nullopt, 1});
    for (const bool throwOnDuplicateKey : {true, false}) {
      VELOX_ASSERT_THROW(
          resolve(*keys, 0, 3, throwOnDuplicateKey),
          "Cannot use null as map key!");
    }
  }
}

TEST_F(LastWinTest, topLevelDoubleKeysUseJavaEquality) {
  const auto otherNaN = std::bit_cast<double>(uint64_t{0x7ff8'0000'0000'0001});
  const auto keys = makeFlatVector<double>(
      {-0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), otherNaN, -0.0});
  EXPECT_EQ(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 4}, {1, 1}, {2, 3}}));
  EXPECT_EQ(
      resolve(*keys, 0, 2, /*throwOnDuplicateKey=*/true),
      (Positions{{0, 0}, {1, 1}}));
  VELOX_ASSERT_THROW(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: NaN");
}

TEST_F(LastWinTest, topLevelRealKeysUseJavaEquality) {
  const auto otherNaN = std::bit_cast<float>(uint32_t{0x7fc0'0001});
  const auto keys = makeFlatVector<float>(
      {-0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN(), otherNaN, -0.0f});
  EXPECT_EQ(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 4}, {1, 1}, {2, 3}}));
  EXPECT_EQ(
      resolve(*keys, 0, 2, /*throwOnDuplicateKey=*/true),
      (Positions{{0, 0}, {1, 1}}));
  VELOX_ASSERT_THROW(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: NaN");
}

TEST_F(LastWinTest, nestedDoubleKeysUseSqlEquality) {
  const auto otherNaN = std::bit_cast<double>(uint64_t{0x7ff8'0000'0000'0001});
  const auto keys = makeArrayVector<double>({
      {-0.0},
      {0.0},
      {std::numeric_limits<double>::quiet_NaN()},
      {otherNaN},
  });
  EXPECT_EQ(
      resolve(*keys, 0, 4, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 1}, {2, 3}}));
  VELOX_ASSERT_THROW(
      resolve(*keys, 0, 4, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: {0}");
}

TEST_F(LastWinTest, rowKeys) {
  const auto keys = makeRowVector({
      makeFlatVector<int64_t>({1, 1, 1}),
      makeFlatVector<std::string>({"a", "b", "a"}),
  });
  EXPECT_EQ(
      resolve(*keys, 0, 3, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 2}, {1, 1}}));
}

TEST_F(LastWinTest, varcharKeys) {
  const auto keys = makeFlatVector<std::string>(
      {"apple", "banana", "apple", "a longer string than inline"});
  EXPECT_EQ(
      resolve(*keys, 0, 4, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 2}, {1, 1}, {3, 3}}));
}

TEST_F(LastWinTest, dictionaryEncodedKeys) {
  const auto keys = wrapInDictionary(
      makeIndices({0, 1, 0, 1, 0}), makeFlatVector<int64_t>({10, 20}));
  EXPECT_EQ(
      resolve(*keys, 0, 5, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 4}, {1, 3}}));
}

TEST_F(LastWinTest, dictionaryEncodedKeysWithOffset) {
  const auto keys = wrapInDictionary(
      makeIndices({1, 2, 0, 2, 1}), makeFlatVector<int64_t>({10, 20, 30}));
  EXPECT_EQ(
      resolve(*keys, 1, 4, /*throwOnDuplicateKey=*/false),
      (Positions{{0, 2}, {1, 1}, {3, 3}}));
  VELOX_ASSERT_THROW(
      resolve(*keys, 1, 4, /*throwOnDuplicateKey=*/true),
      "Duplicate map key was found: 30");
}

TEST_F(LastWinTest, reuseAcrossRuns) {
  {
    const auto keys = makeFlatVector<int64_t>({1, 2, 3, 4, 1, 2});
    EXPECT_EQ(
        resolve(*keys, 0, 6, /*throwOnDuplicateKey=*/false),
        (Positions{{0, 4}, {1, 5}, {2, 2}, {3, 3}}));
  }
  {
    const auto keys = makeFlatVector<int64_t>({4, 4});
    EXPECT_EQ(
        resolve(*keys, 0, 2, /*throwOnDuplicateKey=*/false),
        (Positions{{0, 1}}));
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
