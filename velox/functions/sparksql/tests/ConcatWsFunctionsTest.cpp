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
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {

using namespace facebook::velox::test;
namespace {

std::string generateRandomString(size_t length) {
  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::string randomString;
  for (std::size_t i = 0; i < length; ++i) {
    randomString += chars[folly::Random::rand32() % chars.size()];
  }
  return randomString;
}

void testConcatWsFlatVector(
    const std::vector<std::vector<std::string>>& inputTable,
    const size_t argsCount,
    const std::string& separator) {
  std::vector<VectorPtr> inputVectors;

  for (int i = 0; i < argsCount; i++) {
    inputVectors.emplace_back(
        BaseVector::create(VARCHAR(), inputTable.size(), execCtx_.pool()));
  }

  for (int row = 0; row < inputTable.size(); row++) {
    auto isFirst = true;
    for (int col = 0; col < argsCount; col++) {
      std::static_pointer_cast<FlatVector<StringView>>(inputVectors[col])
          ->set(row, StringView(inputTable[row][col]));
    }
  }

  auto buildConcatQuery = [&]() {
    std::string output = "concat_ws('" + separator + "'";

    for (int i = 0; i < argsCount; i++) {
      output += ",c" + std::to_string(i);
    }
    output += ")";
    return output;
  };

  // Evaluate 'concat_ws' expression and verify no excessive
  // memory allocation. We expect 2 allocations: one for the values buffer and
  // another for the strings buffer. I.e. FlatVector<StringView>::values and
  // FlatVector<StringView>::stringBuffers.
  auto numAllocsBefore = pool()->stats().numAllocs;

  auto result = evaluate<FlatVector<StringView>>(
      buildConcatQuery(), makeRowVector(inputVectors));

  auto numAllocsAfter = pool()->stats().numAllocs;
  ASSERT_EQ(numAllocsAfter - numAllocsBefore, 2);

  auto concatStd = [&](const std::vector<std::string>& inputs) {
    auto isFirst = true;
    std::string output;
    for (int i = 0; i < inputs.size(); i++) {
      auto value = inputs[i];
      if (!value.empty()) {
        if (isFirst) {
          isFirst = false;
        } else {
          output += separator;
        }
        output += value;
      }
    }
    return output;
  };

  for (int i = 0; i < inputTable.size(); ++i) {
    EXPECT_EQ(result->valueAt(i), concatStd(inputTable[i])) << "at " << i;
  }
}

// Test concat_ws vector function
TEST_F(StringFunctionsTest, concat_ws) {
  size_t maxArgsCount = 10; // cols
  size_t rowCount = 100;
  size_t maxStringLength = 100;

  std::vector<std::vector<std::string>> inputTable;
  for (int argsCount = 1; argsCount <= maxArgsCount; argsCount++) {
    inputTable.clear();

    // Create table with argsCount columns
    inputTable.resize(rowCount, std::vector<std::string>(argsCount));

    // Fill the table
    for (int row = 0; row < rowCount; row++) {
      for (int col = 0; col < argsCount; col++) {
        inputTable[row][col] =
            generateRandomString(folly::Random::rand32() % maxStringLength);
      }
    }

    SCOPED_TRACE(fmt::format("Number of arguments: {}", argsCount));
    testConcatWsFlatVector(inputTable, argsCount, "testSep");
  }

  // Test constant input vector with 2 args
  {
    auto rows = makeRowVector(makeRowType({VARCHAR(), VARCHAR()}), 10);
    auto c0 = generateRandomString(20);
    auto c1 = generateRandomString(20);
    auto result = evaluate<SimpleVector<StringView>>(
        fmt::format("concat_ws('-', '{}', '{}')", c0, c1), rows);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(result->valueAt(i), c0 + "-" + c1);
    }
  }

  // Multiple consecutive constant inputs.
  {
    std::string value;
    auto data = makeRowVector({
        makeFlatVector<StringView>(
            1'000,
            [&](auto /* row */) {
              value = generateRandomString(
                  folly::Random::rand32() % maxStringLength);
              return StringView(value);
            }),
        makeFlatVector<StringView>(
            1'000,
            [&](auto /* row */) {
              value = generateRandomString(
                  folly::Random::rand32() % maxStringLength);
              return StringView(value);
            }),
    });

    auto c0 = data->childAt(0)->as<FlatVector<StringView>>()->rawValues();
    auto c1 = data->childAt(1)->as<FlatVector<StringView>>()->rawValues();

    auto result = evaluate<SimpleVector<StringView>>(
        "concat_ws('--', c0, c1, null, 'foo', 'bar', null)", data);

    auto expected = makeFlatVector<StringView>(1'000, [&](auto row) {
      value = "";
      const std::string& s0 = c0[row].str();
      const std::string& s1 = c1[row].str();

      if (s0.empty() && s1.empty()) {
        value = "foo--bar";
      } else if (!s0.empty() && !s1.empty()) {
        value = s0 + "--" + s1 + "--foo--bar";
      } else {
        value = s0 + s1 + "--foo--bar";
      }
      return StringView(value);
    });

    test::assertEqualVectors(expected, result);

    result = evaluate<SimpleVector<StringView>>(
        "concat_ws('$*@', 'aaa', 'bbb', c0, 'ccc', 'ddd', c1, 'eee', 'fff')",
        data);

    expected = makeFlatVector<StringView>(1'000, [&](auto row) {
      value = "";
      std::string delim = "$*@";
      const std::string& s0 =
          c0[row].str().empty() ? c0[row].str() : delim + c0[row].str();
      const std::string& s1 =
          c1[row].str().empty() ? c1[row].str() : delim + c1[row].str();

      value = "aaa" + delim + "bbb" + s0 + delim + "ccc" + delim + "ddd" + s1 +
          delim + "eee" + delim + "fff";
      return StringView(value);
    });
    test::assertEqualVectors(expected, result);
  }
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
