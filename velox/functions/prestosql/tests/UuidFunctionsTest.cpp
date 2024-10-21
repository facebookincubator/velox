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

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions::prestosql {

namespace {

class UuidFunctionsTest : public functions::test::FunctionBaseTest {};

TEST_F(UuidFunctionsTest, uuid) {
  auto result =
      evaluate<FlatVector<int128_t>>("uuid()", makeRowVector(ROW({}), 10));

  // Sanity check results. All values are unique.
  std::unordered_set<int128_t> uuids;
  for (auto i = 0; i < 10; ++i) {
    const auto uuid = result->valueAt(i);
    ASSERT_TRUE(uuids.insert(uuid).second);
  }
  ASSERT_EQ(10, uuids.size());
}

TEST_F(UuidFunctionsTest, typeof) {
  auto result = evaluate("typeof(uuid())", makeRowVector(ROW({}), 10));

  auto expected = makeConstant("uuid", 10);
  velox::test::assertEqualVectors(expected, result);

  result = evaluate(
      "typeof(array_constructor(uuid(), uuid()))", makeRowVector(ROW({}), 10));

  expected = makeConstant("array(uuid)", 10);
  velox::test::assertEqualVectors(expected, result);
}

TEST_F(UuidFunctionsTest, castAsVarchar) {
  const vector_size_t size = 1'000;
  auto uuids =
      evaluate<FlatVector<int128_t>>("uuid()", makeRowVector(ROW({}), size));

  auto result = evaluate<FlatVector<StringView>>(
      "cast(c0 as varchar)", makeRowVector({uuids}));

  // Verify that CAST results as the same as boost::lexical_cast. We do not use
  // boost::lexical_cast to implement CAST because it is too slow.
  auto expected = makeFlatVector<std::string>(size, [&](auto row) {
    auto uuid = uuids->valueAt(row);
    auto charPtr = reinterpret_cast<const char*>(&uuid);

    boost::uuids::uuid u;
    for (size_t i = 0; i < 16; ++i) {
      u.data[15 - i] = charPtr[i];
    }

    return boost::lexical_cast<std::string>(u);
  });

  velox::test::assertEqualVectors(expected, result);

  // Sanity check results. All strings are unique. Each string is 36 bytes
  // long.
  std::unordered_set<std::string> uniqueUuids;
  for (auto i = 0; i < size; ++i) {
    const auto uuid = result->valueAt(i).str();
    ASSERT_EQ(36, uuid.size());
    ASSERT_TRUE(uniqueUuids.insert(uuid).second);
  }
  ASSERT_EQ(size, uniqueUuids.size());
}

TEST_F(UuidFunctionsTest, castRoundTrip) {
  auto strings = makeFlatVector<std::string>({
      "33355449-2c7d-43d7-967a-f53cd23215ad",
      "eed9f812-4b0c-472f-8a10-4ae7bff79a47",
      "f768f36d-4f09-4da7-a298-3564d8f3c986",
  });

  auto uuids = evaluate("cast(c0 as uuid)", makeRowVector({strings}));
  auto stringsCopy = evaluate("cast(c0 as varchar)", makeRowVector({uuids}));
  auto uuidsCopy = evaluate("cast(c0 as uuid)", makeRowVector({stringsCopy}));

  velox::test::assertEqualVectors(strings, stringsCopy);
  velox::test::assertEqualVectors(uuids, uuidsCopy);
}

TEST_F(UuidFunctionsTest, unsupportedCast) {
  auto input = makeRowVector(ROW({}), 10);
  VELOX_ASSERT_THROW(
      evaluate("cast(uuid() as integer)", input),
      "Cannot cast UUID to INTEGER");
  VELOX_ASSERT_THROW(
      evaluate("cast(123 as uuid())", input), "Cannot cast BIGINT to UUID.");
}

TEST_F(UuidFunctionsTest, comparisons) {
  const auto uuidEval = [&](const std::optional<std::string>& lhs, const std::string& operation, const std::optional<std::string>& rhs) {
    return evaluateOnce<bool>(fmt::format("cast(c0 as uuid) {} cast(c1 as uuid)", operation), lhs, rhs);
  };

  ASSERT_EQ(uuidEval("33355449-2c7d-43d7-967a-f53cd23215ad", "<", "ffffffff-ffff-ffff-ffff-ffffffffffff"), true);
  ASSERT_EQ(uuidEval("33355449-2c7d-43d7-967a-f53cd23215ad", "<", "00000000-0000-0000-0000-000000000000"), false);
  ASSERT_EQ(uuidEval("f768f36d-4f09-4da7-a298-3564d8f3c986", ">", "00000000-0000-0000-0000-000000000000"), true);
  ASSERT_EQ(uuidEval("f768f36d-4f09-4da7-a298-3564d8f3c986", ">", "ffffffff-ffff-ffff-ffff-ffffffffffff"), false);

  ASSERT_EQ(uuidEval("33355449-2c7d-43d7-967a-f53cd23215ad", "<=", "33355449-2c7d-43d7-967a-f53cd23215ad"), true);
  ASSERT_EQ(uuidEval("33355449-2c7d-43d7-967a-f53cd23215ad", "<=", "ffffffff-ffff-ffff-ffff-ffffffffffff"), true);
  ASSERT_EQ(uuidEval("33355449-2c7d-43d7-967a-f53cd23215ad", ">=", "33355449-2c7d-43d7-967a-f53cd23215ad"), true);
  ASSERT_EQ(uuidEval("ffffffff-ffff-ffff-ffff-ffffffffffff", ">=", "33355449-2c7d-43d7-967a-f53cd23215ad"), true);

  ASSERT_EQ(uuidEval("f768f36d-4f09-4da7-a298-3564d8f3c986", "==", "f768f36d-4f09-4da7-a298-3564d8f3c986"), true);
  ASSERT_EQ(uuidEval("eed9f812-4b0c-472f-8a10-4ae7bff79a47", "!=", "f768f36d-4f09-4da7-a298-3564d8f3c986"), true);

  ASSERT_EQ(uuidEval("11000000-0000-0022-0000-000000000000", "<", "22000000-0000-0011-0000-000000000000"), true);
  ASSERT_EQ(uuidEval("00000000-0000-0000-2200-000000000011", ">", "00000000-0000-0000-1100-000000000022"), true);
  ASSERT_EQ(uuidEval("00000000-0000-0000-0000-000000000011", ">", "22000000-0000-0000-0000-000000000000"), false);
  ASSERT_EQ(uuidEval("11000000-0000-0000-0000-000000000000", "<", "00000000-0000-0000-0000-000000000022"), false);
}

} // namespace
} // namespace facebook::velox::functions::prestosql
