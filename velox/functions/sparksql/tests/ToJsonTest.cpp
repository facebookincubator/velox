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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class ToJsonTest : public SparkFunctionBaseTest {
 protected:
  void testToJson(
      const std::vector<StringView>& expected,
      const std::string& expression,
      const VectorPtr& input) {
    testToJson(expected, expression, makeRowVector({input}));
  }

  void testToJson(
      const std::vector<StringView>& expected,
      const std::string& expression,
      const RowVectorPtr& input) {
    auto result = evaluate(expression, (input));
    velox::test::assertEqualVectors(
        makeFlatVector<StringView>(expected), result);
  }
};

TEST_F(ToJsonTest, basic) {
  auto arrayInput = makeNullableArrayVector<int64_t>({
      {1, 2, 3},
      {4, 4},
      {std::nullopt},
      {std::nullopt, 123},
      {std::numeric_limits<int64_t>::min(),
       std::numeric_limits<int64_t>::max()},
  });

  std::vector<StringView> expected{
      "[1,2,3]",
      "[4,4]",
      "[null]",
      "[null,123]",
      "[-9223372036854775808,9223372036854775807]",
  };

  testToJson(expected, "to_json(c0)", arrayInput);

  arrayInput = makeArrayVector<StringView>({
      {"hello", "velox!"},
      {"deadbeaf"},
      {},
  });

  expected = {
      R"(["hello","velox!"])",
      "[\"deadbeaf\"]",
      "[]",
  };

  testToJson(expected, "to_json(c0)", arrayInput);

  auto mapInput = makeMapVector<StringView, int64_t>({
      {{"k1", 1}, {"k2", 2}},
      {{"k3", 3}},
      {},
  });

  expected = {R"({"k1":1,"k2":2})", R"({"k3":3})", "{}"};
  testToJson(expected, "to_json(c0)", mapInput);

  auto bigintInput = makeFlatVector<int64_t>({0, 1024, 9527});
  auto stringInput = makeFlatVector<StringView>({"a", "b", "c"});

  auto rowInput = makeRowVector({"c0", "c1"}, {bigintInput, stringInput});
  expected = {
      R"([0,"a"])",
      R"([1024,"b"])",
      R"([9527,"c"])",
  };
  testToJson(expected, "to_json(c0)", makeRowVector({rowInput}));
}

TEST_F(ToJsonTest, complex) {
  auto arrayInput = makeArrayVector<StringView>({
      {"hello", "velox!"},
      {"deadbeaf"},
      {},
  });
  auto mapInput = makeMapVector<StringView, int64_t>({
      {{"k1", 1}, {"k2", 2}},
      {{"k3", 3}},
      {},
  });
  auto bigintInput = makeFlatVector<int64_t>({0, 1024, 9527});
  auto rowInput =
      makeRowVector({"c0", "c1", "c2"}, {bigintInput, arrayInput, mapInput});
  std::vector<StringView> expected = {
      R"([0,["hello","velox!"],{"k1":1,"k2":2}])",
      R"([1024,["deadbeaf"],{"k3":3}])",
      "[9527,[],{}]",
  };
  testToJson(expected, "to_json(c0)", makeRowVector({rowInput}));
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
