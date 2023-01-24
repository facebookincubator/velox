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
#include <gtest/gtest.h>
#include <array>
#include <cctype>
#include <random>
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/Expr.h"
#include "velox/functions/lib/StringEncodingUtils.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/parse/Expressions.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;
using namespace std::string_literals;

namespace {
/// Generate an ascii random string of size length
std::string generateRandomString(size_t length) {
  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::string randomString;
  for (std::size_t i = 0; i < length; ++i) {
    randomString += chars[folly::Random::rand32() % chars.size()];
  }
  return randomString;
}

/*
 * Some utility functions to setup the nullability and values of input vectors
 * The input vectors are string vector, start vector and length vector
 */
bool expectNullString(int i) {
  return i % 10 == 1;
}

bool expectNullStart(int i) {
  return i % 3 == 1;
}

bool expectNullLength(int i) {
  return i % 5 == 4;
}

int expectedStart(int i) {
  return i % 7;
}

int expectedLength(int i) {
  return i % 3;
}

std::string hexToDec(const std::string& str) {
  VELOX_CHECK_EQ(str.size() % 2, 0);
  std::string out;
  out.resize(str.size() / 2);
  for (int i = 0; i < out.size(); ++i) {
    int high = facebook::velox::functions::stringImpl::fromHex(str[2 * i]);
    int low = facebook::velox::functions::stringImpl::fromHex(str[2 * i + 1]);
    out[i] = (high << 4) | (low & 0xf);
  }
  return out;
}
} // namespace

class StringFunctionsTest : public FunctionBaseTest {
 protected:
  VectorPtr makeStrings(
      vector_size_t size,
      const std::vector<std::string>& inputStrings) {
    return makeFlatVector<StringView>(
        size,
        [&](auto row) { return StringView(inputStrings[row]); },
        expectNullString);
  }

  int bufferRefCounts(FlatVector<StringView>* vector) {
    int refCounts = 0;
    for (auto& buffer : vector->stringBuffers()) {
      refCounts += buffer->refCount();
    }
    return refCounts;
  }

  auto evaluateSubstr(
      std::string query,
      const std::vector<VectorPtr>& args,
      int stringVectorIndex = 0) {
    auto row = makeRowVector(args);
    auto stringVector = args[stringVectorIndex];
    int refCountBeforeEval =
        bufferRefCounts(stringVector->asFlatVector<StringView>());
    auto result = evaluate<FlatVector<StringView>>(query, row);

    int refCountAfterEval =
        bufferRefCounts(stringVector->asFlatVector<StringView>());
    EXPECT_EQ(refCountAfterEval, 2 * refCountBeforeEval) << "at " << query;

    return result;
  }

  void testUpperFlatVector(
      const std::vector<std::tuple<std::string, std::string>>& tests,
      std::optional<bool> ascii,
      bool multiReferenced,
      bool expectedAscii) {
    auto inputsFlatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), tests.size(), execCtx_.pool()));

    for (int i = 0; i < tests.size(); i++) {
      inputsFlatVector->set(i, StringView(std::get<0>(tests[i])));
    }

    if (ascii.has_value()) {
      inputsFlatVector->setAllIsAscii(ascii.value());
    }

    auto crossRefVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), 1, execCtx_.pool()));

    if (multiReferenced) {
      crossRefVector->acquireSharedStringBuffers(inputsFlatVector.get());
    }

    auto result = evaluate<FlatVector<StringView>>(
        "upper(c0)", makeRowVector({inputsFlatVector}));

    SelectivityVector all(tests.size());
    ASSERT_EQ(result->isAscii(all), expectedAscii);

    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), StringView(std::get<1>(tests[i])));
    }
  }

  void testLowerFlatVector(
      const std::vector<std::tuple<std::string, std::string>>& tests,
      std::optional<bool> ascii,
      bool multiReferenced,
      bool expectedAscii) {
    auto inputsFlatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), tests.size(), execCtx_.pool()));

    for (int i = 0; i < tests.size(); i++) {
      inputsFlatVector->set(i, StringView(std::get<0>(tests[i])));
    }

    if (ascii.has_value()) {
      inputsFlatVector->setAllIsAscii(ascii.value());
    }

    auto crossRefVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), 1, execCtx_.pool()));

    if (multiReferenced) {
      crossRefVector->acquireSharedStringBuffers(inputsFlatVector.get());
    }
    auto testQuery = [&](const std::string& query) {
      auto result = evaluate<FlatVector<StringView>>(
          query, makeRowVector({inputsFlatVector}));

      SelectivityVector all(tests.size());
      ASSERT_EQ(result->isAscii(all), expectedAscii);

      for (int32_t i = 0; i < tests.size(); ++i) {
        ASSERT_EQ(result->valueAt(i), StringView(std::get<1>(tests[i])));
      }
    };
    testQuery("lower(C0)");
    testQuery("lower(upper(C0))");
  }

  void testConcatFlatVector(
      const std::vector<std::vector<std::string>>& inputTable,
      const size_t argsCount) {
    std::vector<VectorPtr> inputVectors;

    for (int i = 0; i < argsCount; i++) {
      inputVectors.emplace_back(
          BaseVector::create(VARCHAR(), inputTable.size(), execCtx_.pool()));
    }

    for (int row = 0; row < inputTable.size(); row++) {
      for (int col = 0; col < argsCount; col++) {
        std::static_pointer_cast<FlatVector<StringView>>(inputVectors[col])
            ->set(row, StringView(inputTable[row][col]));
      }
    }

    auto buildConcatQuery = [&]() {
      std::string output = "concat(";
      for (int i = 0; i < argsCount; i++) {
        if (i != 0) {
          output += ",";
        }
        output += "c" + std::to_string(i);
      }
      output += ")";
      return output;
    };

    // Evaluate 'concat' expression and verify no excessive memory allocation.
    // We expect 2 allocations: one for the values buffer and another for the
    // strings buffer. I.e. FlatVector<StringView>::values and
    // FlatVector<StringView>::stringBuffers.
    auto numAllocsBefore = pool()->getMemoryUsageTracker()->numAllocs();

    auto result = evaluate<FlatVector<StringView>>(
        buildConcatQuery(), makeRowVector(inputVectors));

    auto numAllocsAfter = pool()->getMemoryUsageTracker()->numAllocs();
    ASSERT_EQ(numAllocsAfter - numAllocsBefore, 2);

    auto concatStd = [](const std::vector<std::string>& inputs) {
      std::string output;
      for (auto& input : inputs) {
        output += input;
      }
      return output;
    };

    for (int i = 0; i < inputTable.size(); ++i) {
      EXPECT_EQ(result->valueAt(i), StringView(concatStd(inputTable[i])))
          << "at " << i;
    }
  }

  void testLengthFlatVector(
      const std::vector<std::tuple<std::string, int64_t>>& tests,
      std::optional<bool> setAscii) {
    auto inputsFlatVector = makeFlatVector<StringView>(
        tests.size(),
        [&](auto row) { return StringView(std::get<0>(tests[row])); });
    if (setAscii.has_value()) {
      inputsFlatVector->setAllIsAscii(setAscii.value());
    }

    auto result = evaluate<FlatVector<int64_t>>(
        "length(c0)", makeRowVector({inputsFlatVector}));

    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), std::get<1>(tests[i]));
    }
  }

  void testAsciiPropagation(
      std::vector<std::string> firstColumn,
      std::vector<std::string> secondColumn,
      std::vector<std::string> thirdColumn,
      SelectivityVector rows,
      std::optional<bool> isAscii,
      std::string function = "multi_string_fn",
      std::set<size_t> computeAscinessFor = {0, 1}) {
    auto argFirst = makeFlatVector<std::string>(firstColumn);
    auto argSecond = makeFlatVector<std::string>(secondColumn);
    auto argThird = makeFlatVector<std::string>(thirdColumn);

    // Compute asciiness for required columns.
    if (computeAscinessFor.count(0)) {
      (argFirst->as<SimpleVector<StringView>>())->computeAndSetIsAscii(rows);
    }
    if (computeAscinessFor.count(1)) {
      (argSecond->as<SimpleVector<StringView>>())->computeAndSetIsAscii(rows);
    }
    if (computeAscinessFor.count(2)) {
      (argThird->as<SimpleVector<StringView>>())->computeAndSetIsAscii(rows);
    }

    auto result = evaluate<SimpleVector<StringView>>(
        fmt::format("{}(c0, c1, c2)", function),
        makeRowVector({argFirst, argSecond, argThird}));
    auto ascii = result->isAscii(rows);
    ASSERT_EQ(ascii, isAscii);
  }

  using strpos_input_test_t = std::vector<
      std::pair<std::tuple<std::string, std::string, int64_t>, int64_t>>;

  template <typename TInstance>
  void testStringPositionAllFlatVector(
      const strpos_input_test_t& tests,
      const std::vector<std::optional<bool>>& stringEncodings,
      bool withInstanceArgument);

  template <typename TInstance>
  void testStringPositionFromEndAllFlatVector(
      const strpos_input_test_t& tests,
      const std::vector<std::optional<bool>>& stringEncodings,
      bool withInstanceArgument);

  void testChrFlatVector(
      const std::vector<std::pair<int64_t, std::string>>& tests);

  void testCodePointFlatVector(
      const std::vector<std::pair<std::string, int32_t>>& tests);

  void testStringPositionFastPath(
      const std::vector<std::tuple<std::string, int64_t>>& tests,
      const std::string& subString,
      int64_t instance);

  using replace_input_test_t = std::vector<std::pair<
      std::tuple<std::string, std::string, std::string>,
      std::string>>;

  void testReplaceFlatVector(
      const replace_input_test_t& tests,
      bool withReplaceArgument);

  void testReplaceInPlace(
      const std::vector<std::pair<std::string, std::string>>& tests,
      const std::string& search,
      const std::string& replace,
      bool multiReferenced);

  void testXXHash64(
      const std::vector<std::tuple<std::string, int64_t, int64_t>>& tests);

  void testXXHash64(
      const std::vector<std::pair<std::string, int64_t>>& tests,
      bool stringVariant);

  static std::string generateInvalidUtf8() {
    std::string str;
    str.resize(2);
    // Create corrupt data below.
    char16_t c = u'\u04FF';
    str[0] = (char)c;
    str[1] = (char)c;
    return str;
  }

  // Generate complex encoding with the format:
  // whitespace|unicode line separator|ascii|two bytes encoding|three bytes
  // encoding|four bytes encoding|whitespace|unicode line separator
  static std::string generateComplexUtf8(bool invalid = false) {
    std::string str;
    // White spaces
    str.append(" \u2028"s);
    if (invalid) {
      str.append(generateInvalidUtf8());
    }
    // Ascii
    str.append("hello");
    // two bytes
    str.append(" \u017F");
    // three bytes
    str.append(" \u4FE1");
    // four bytes
    std::string tmp;
    tmp.resize(4);
    tmp[0] = 0xF0;
    tmp[1] = 0xAF;
    tmp[2] = 0xA8;
    tmp[3] = 0x9F;
    str.append(" ").append(tmp);
    if (invalid) {
      str.append(generateInvalidUtf8());
    }
    // white spaces
    str.append("\u2028 ");
    return str;
  }
};

/**
 * The test for vector of strings and constant values for start and length
 */
TEST_F(StringFunctionsTest, substrConstant) {
  vector_size_t size = 20;

  // Making input vector
  std::vector<std::string> strings(size);
  std::generate(strings.begin(), strings.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  // Creating vectors
  auto stringVector = makeStrings(size, strings);

  auto result = evaluateSubstr("substr(c0, 1, 2)", {stringVector});

  EXPECT_EQ(stringVector.use_count(), 1);
  // Destroying string vector
  stringVector = nullptr;

  for (int i = 0; i < size; ++i) {
    if (expectNullString(i)) {
      EXPECT_TRUE(result->isNullAt(i)) << "expected null at " << i;
    } else {
      EXPECT_EQ(result->valueAt(i).size(), 2) << "at " << i;
      EXPECT_EQ(result->valueAt(i).getString(), strings[i].substr(0, 2))
          << "at " << i;
    }
  }
}

/**
 * The test for vector of strings and vector of int for both start and length
 */
TEST_F(StringFunctionsTest, substrVariable) {
  std::shared_ptr<FlatVector<StringView>> result;
  vector_size_t size = 100;
  std::vector<std::string> ref_strings(size);

  std::vector<std::string> strings(size);
  std::generate(strings.begin(), strings.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  auto startVector =
      makeFlatVector<int32_t>(size, expectedStart, expectNullStart);

  auto lengthVector =
      makeFlatVector<int32_t>(size, expectedLength, expectNullLength);

  auto stringVector = makeStrings(size, strings);

  result = evaluateSubstr(
      "substr(c0, c1, c2)", {stringVector, startVector, lengthVector});
  EXPECT_EQ(stringVector.use_count(), 1);
  // Destroying string vector
  stringVector = nullptr;

  for (int i = 0; i < size; ++i) {
    // Checking the null results
    if (expectNullString(i) || expectNullStart(i) || expectNullLength(i)) {
      EXPECT_TRUE(result->isNullAt(i)) << "expected null at " << i;
    } else {
      if (expectedStart(i) != 0) {
        EXPECT_EQ(result->valueAt(i).size(), expectedLength(i)) << "at " << i;
        for (int l = 0; l < expectedLength(i); l++) {
          EXPECT_EQ(
              result->valueAt(i).data()[l],
              strings[i][expectedStart(i) - 1 + l])
              << "at " << i;
        }
      } else {
        // Special test for start = 0. The Presto semantic expect empty string
        EXPECT_EQ(result->valueAt(i).size(), 0);
      }
    }
  }
}

/**
 * The test for one of non-optimized cases (all constant values)
 */
TEST_F(StringFunctionsTest, substrSlowPath) {
  vector_size_t size = 100;

  auto dummyInput = makeRowVector(makeRowType({BIGINT()}), size);
  auto result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', 5, 2)", dummyInput);

  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(result->valueAt(i).size(), 2) << "at " << i;
  }
}

/**
 * The test for negative start indexes
 */
TEST_F(StringFunctionsTest, substrNegativeStarts) {
  vector_size_t size = 100;

  auto dummyInput = makeRowVector(makeRowType({BIGINT()}), size);

  auto result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -3, 3)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "ere");

  result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -1, 3)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "e");

  result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -2, 100)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "re");

  result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -2, -1)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "");

  result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -10)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "tring here");

  result = evaluate<SimpleVector<StringView>>(
      "substr('my string here', -100)", dummyInput);

  EXPECT_EQ(result->valueAt(0).getString(), "");
}

/**
 * The test for substr operating on single buffers with two string functions
 * using a conditional
 */
TEST_F(StringFunctionsTest, substrWithConditionalDoubleBuffer) {
  vector_size_t size = 20;

  auto indexVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });

  // Making input vector
  std::vector<std::string> strings(size);
  std::generate(strings.begin(), strings.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  // Creating vectors
  auto stringVector = makeStrings(size, strings);

  std::vector<std::string> strings2(size);
  std::generate(strings2.begin(), strings2.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_SECOND_STR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  auto result = evaluateSubstr(
      "if (c0 % 2 = 0, substr(c1, 1, length(c1)), substr(c1, -3))",
      {indexVector, stringVector},
      1 /* index of the string vector */);

  // Destroying original string vector to examine
  // the life time of the string buffer
  EXPECT_EQ(stringVector.use_count(), 1);
  stringVector = nullptr;

  for (int i = 0; i < size; ++i) {
    // Checking the null results
    if (expectNullString(i)) {
      EXPECT_TRUE(result->isNullAt(i)) << "expected null at " << i;
    } else {
      if (i % 2 == 0) {
        EXPECT_EQ(result->valueAt(i).size(), strings[i].size()) << "at " << i;
        EXPECT_EQ(result->valueAt(i).getString(), strings[i]) << "at " << i;
      } else {
        auto str = strings[i];
        EXPECT_EQ(result->valueAt(i).size(), 3) << "at " << i;
        EXPECT_EQ(result->valueAt(i).getString(), str.substr(str.size() - 3))
            << "at " << i;
      }
    }
  }
}

/**
 * The test for substr operating on two buffers of string using a conditional
 */
TEST_F(StringFunctionsTest, substrWithConditionalSingleBuffer) {
  vector_size_t size = 20;

  auto indexVector =
      makeFlatVector<int32_t>(size, [](vector_size_t row) { return row; });

  // Making input vector
  std::vector<std::string> strings(size);
  std::generate(strings.begin(), strings.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  // Creating vectors
  auto stringVector = makeStrings(size, strings);

  std::vector<std::string> strings2(size);
  std::generate(strings2.begin(), strings2.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_SECOND_STR_" + std::to_string(i * 100) +
        " - Making the string  large enough so they " +
        " are stored in block and not inlined";
  });

  // Creating vectors
  auto stringVector2 = makeStrings(size, strings);

  auto result = evaluateSubstr(
      "if (c0 % 2 = 0, substr(c1, 1, length(c1)), substr(c1, -3))",
      {indexVector, stringVector, stringVector2},
      1 /* index of the string vector */);

  // Destroying original string vector to examine
  // the life time of the string buffer
  EXPECT_EQ(stringVector.use_count(), 1);
  stringVector = nullptr;

  for (int i = 0; i < size; ++i) {
    // Checking the null results
    if (expectNullString(i)) {
      EXPECT_TRUE(result->isNullAt(i)) << "expected null at " << i;
    } else {
      if (i % 2 == 0) {
        EXPECT_EQ(result->valueAt(i).size(), strings[i].size()) << "at " << i;
        EXPECT_EQ(result->valueAt(i).getString(), strings[i]) << "at " << i;
      } else {
        auto str = strings[i];
        EXPECT_EQ(result->valueAt(i).size(), 3) << "at " << i;
        EXPECT_EQ(result->valueAt(i).getString(), str.substr(str.size() - 3))
            << "at " << i;
      }
    }
  }
}

namespace {
std::vector<std::tuple<std::string, std::string>> getUpperAsciiTestData() {
  return {
      {"abcdefg", "ABCDEFG"},
      {"ABCDEFG", "ABCDEFG"},
      {"a B c D e F g", "A B C D E F G"},
  };
}

std::vector<std::tuple<std::string, std::string>> getUpperUnicodeTestData() {
  return {
      {"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ", "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ"},
      {"αβγδεζηθικλμνξοπρςστυφχψ", "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΣΤΥΦΧΨ"},
      {"абвгдежзийклмнопрстуфхцчшщъыьэюя", "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"}};
}

std::vector<std::tuple<std::string, std::string>> getLowerAsciiTestData() {
  return {
      {"ABCDEFG", "abcdefg"},
      {"abcdefg", "abcdefg"},
      {"a B c D e F g", "a b c d e f g"},
  };
}

std::vector<std::tuple<std::string, std::string>> getLowerUnicodeTestData() {
  return {
      {"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ", "àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ"},
      {"ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΣΤΥΦΧΨ", "αβγδεζηθικλμνξοπρσστυφχψ"},
      {"АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ", "абвгдежзийклмнопрстуфхцчшщъыьэюя"}};
}
} // namespace

// Test upper vector function
TEST_F(StringFunctionsTest, upper) {
  auto upperStd = [](const std::string& input) {
    std::string output;
    for (auto c : input) {
      output += std::toupper(c);
    }
    return output;
  };

  // Making input vector
  std::vector<std::tuple<std::string, std::string>> allTests(10);
  std::generate(allTests.begin(), allTests.end(), [upperStd, i = -1]() mutable {
    i++;
    auto&& tmp = std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string large enough so they " +
        " are stored in block and not inlined";
    return std::make_tuple(tmp, upperStd(tmp));
  });

  auto asciiTests = getUpperAsciiTestData();
  allTests.insert(allTests.end(), asciiTests.begin(), asciiTests.end());

  // Test ascii fast paths
  testUpperFlatVector(
      allTests, true /*ascii*/, true /*multiRef*/, true /*expectedAscii*/);
  testUpperFlatVector(
      allTests, true /*ascii*/, false /*multiRef*/, true /*expectedAscii*/);

  auto&& unicodeTests = getUpperUnicodeTestData();
  allTests.insert(allTests.end(), unicodeTests.begin(), unicodeTests.end());

  // Test unicode
  testUpperFlatVector(
      allTests, false /*ascii*/, false, false /*expectedAscii*/);
  testUpperFlatVector(
      allTests, false /*ascii*/, false, false /*expectedAscii*/);
  testUpperFlatVector(allTests, std::nullopt, false, false /*expectedAscii*/);

  // Test constant vectors
  auto rows = makeRowVector(
      {makeFlatVector<int32_t>(10, [](vector_size_t row) { return row; })});
  auto result = evaluate<SimpleVector<StringView>>("upper('test upper')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView("TEST UPPER"));
  }
}

// Test lower vector function
TEST_F(StringFunctionsTest, lower) {
  auto lowerStd = [](const std::string& input) {
    std::string output;
    for (auto c : input) {
      output += std::tolower(c);
    }
    return output;
  };

  // Making input vector
  std::vector<std::tuple<std::string, std::string>> allTests(10);
  std::generate(allTests.begin(), allTests.end(), [lowerStd, i = -1]() mutable {
    i++;
    auto&& tmp = std::to_string(i) + "_MYSTR_" + std::to_string(i * 100) +
        " - Making the string large enough so they " +
        " are stored in block and not inlined";
    return std::make_tuple(tmp, lowerStd(tmp));
  });

  auto asciiTests = getLowerAsciiTestData();
  allTests.insert(allTests.end(), asciiTests.begin(), asciiTests.end());

  testLowerFlatVector(allTests, true /*ascii*/, true, true /*expectedAscii*/);
  testLowerFlatVector(allTests, true /*ascii*/, false, true /*expectedAscii*/);

  auto&& unicodeTests = getLowerUnicodeTestData();
  allTests.insert(allTests.end(), unicodeTests.begin(), unicodeTests.end());

  // Test unicode
  testLowerFlatVector(
      allTests, false /*ascii*/, false, false /*expectedAscii*/);
  testLowerFlatVector(allTests, std::nullopt, false, false /*expectedAscii*/);

  // Test constant vectors
  auto rows = makeRowVector({makeRowVector(
      {makeFlatVector<int32_t>(10, [](vector_size_t row) { return row; })})});
  auto result = evaluate<SimpleVector<StringView>>("lower('TEST LOWER')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView("test lower"));
  }
}

// Test concat vector function
TEST_F(StringFunctionsTest, concat) {
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
    testConcatFlatVector(inputTable, argsCount);
  }

  // Test constant input vector with 2 args
  {
    auto rows = makeRowVector(makeRowType({VARCHAR(), VARCHAR()}), 10);
    auto c0 = generateRandomString(20);
    auto c1 = generateRandomString(20);
    auto result = evaluate<SimpleVector<StringView>>(
        fmt::format("concat('{}', '{}')", c0, c1), rows);
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(result->valueAt(i), StringView(c0 + c1));
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
        "concat(c0, ',', c1, ',', 'foo', ',', 'bar')", data);

    auto expected = makeFlatVector<StringView>(1'000, [&](auto row) {
      value = c0[row].str() + "," + c1[row].str() + ",foo,bar";
      return StringView(value);
    });

    test::assertEqualVectors(expected, result);

    result = evaluate<SimpleVector<StringView>>(
        "concat('aaa', ',', 'bbb', ',', c0, ',', 'ccc', ',', 'ddd', ',', c1, ',', 'eee', ',', 'fff')",
        data);

    expected = makeFlatVector<StringView>(1'000, [&](auto row) {
      value =
          "aaa,bbb," + c0[row].str() + ",ccc,ddd," + c1[row].str() + ",eee,fff";
      return StringView(value);
    });
    test::assertEqualVectors(expected, result);

    result = evaluate<SimpleVector<StringView>>(
        "concat(c0, ',', c1, ',', 'A somewhat long string.', ',', 'bar')",
        data);

    expected = makeFlatVector<StringView>(1'000, [&](auto row) {
      value =
          c0[row].str() + "," + c1[row].str() + ",A somewhat long string.,bar";
      return StringView(value);
    });

    test::assertEqualVectors(expected, result);
  }
}

// Test length vector function
TEST_F(StringFunctionsTest, length) {
  auto lengthUtf8Ref = [](std::string string) {
    size_t size = 0;
    for (size_t i = 0; i < string.size(); i++) {
      if ((static_cast<const unsigned char>(string[i]) & 0xc0) != 0x80) {
        size++;
      }
    }
    return size;
  };

  // Test ascii
  std::vector<std::tuple<std::string, int64_t>> tests;
  for (auto& pair : getUpperAsciiTestData()) {
    auto& string = std::get<0>(pair);
    tests.push_back(std::make_tuple(string, lengthUtf8Ref(string)));
  }
  auto emptyString = "";
  tests.push_back(std::make_tuple(emptyString, 0));

  testLengthFlatVector(tests, true /*setAscii*/);
  testLengthFlatVector(tests, false /*setAscii*/);
  testLengthFlatVector(tests, std::nullopt);

  // Test unicode
  for (auto& pair : getUpperUnicodeTestData()) {
    auto& string = std::get<0>(pair);
    tests.push_back(std::make_tuple(string, lengthUtf8Ref(string)));
  };

  testLengthFlatVector(tests, false /*setAscii*/);
  testLengthFlatVector(tests, std::nullopt);

  // Test constant vectors
  auto rows = makeRowVector({makeRowVector(
      {makeFlatVector<int32_t>(10, [](vector_size_t row) { return row; })})});
  auto result = evaluate<SimpleVector<int64_t>>("length('test length')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), 11);
  }
}

// Test strpos function
template <typename TInstance>
void StringFunctionsTest::testStringPositionAllFlatVector(
    const strpos_input_test_t& tests,
    const std::vector<std::optional<bool>>& asciiEncodings,
    bool withInstanceArgument) {
  auto stringVector = makeFlatVector<StringView>(tests.size());
  auto subStringVector = makeFlatVector<StringView>(tests.size());
  auto instanceVector =
      withInstanceArgument ? makeFlatVector<TInstance>(tests.size()) : nullptr;

  for (int i = 0; i < tests.size(); i++) {
    stringVector->set(i, StringView(std::get<0>(tests[i].first)));
    subStringVector->set(i, StringView(std::get<1>(tests[i].first)));
    if (instanceVector) {
      instanceVector->set(i, std::get<2>(tests[i].first));
    }
  }

  if (asciiEncodings[0].has_value()) {
    stringVector->setAllIsAscii(asciiEncodings[0].value());
  }
  if (asciiEncodings[1].has_value()) {
    subStringVector->setAllIsAscii(asciiEncodings[1].value());
  }

  FlatVectorPtr<int64_t> result;
  if (withInstanceArgument) {
    result = evaluate<FlatVector<int64_t>>(
        "strpos(c0, c1,c2)",
        makeRowVector({stringVector, subStringVector, instanceVector}));
  } else {
    result = evaluate<FlatVector<int64_t>>(
        "strpos(c0, c1)", makeRowVector({stringVector, subStringVector}));
  }

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), tests[i].second);
  }
}

TEST_F(StringFunctionsTest, stringPosition) {
  strpos_input_test_t testsAscii = {
      {{"high", "ig", -1}, {2}},
      {{"high", "igx", -1}, {0}},
  };

  strpos_input_test_t testsAsciiWithPosition = {
      {{"high", "h", 2}, 4},
      {{"high", "h", 10}, 0},
  };

  strpos_input_test_t testsUnicodeWithPosition = {
      {{"\u4FE1\u5FF5,\u7231,\u5E0C\u671B", "\u7231", 1}, 4},
      {{"\u4FE1\u5FF5,\u7231,\u5E0C\u671B", "\u5E0C\u671B", 1}, 6},
  };

  // We dont have to try all encoding combinations here since there is a test
  // that test the encoding resolution but we want to to have a test for each
  // possible resolution
  testStringPositionAllFlatVector<int64_t>(testsAscii, {true, true}, false);

  // Try instance parameter using BIGINT and INTEGER.
  testStringPositionAllFlatVector<int32_t>(
      testsAsciiWithPosition, {false, false}, true);
  testStringPositionAllFlatVector<int64_t>(
      testsAsciiWithPosition, {false, false}, true);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result = evaluate<SimpleVector<int64_t>>("strpos('high', 'ig')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), 2);
  }
}

// Test strpos function
template <typename TInstance>
void StringFunctionsTest::testStringPositionFromEndAllFlatVector(
    const strpos_input_test_t& tests,
    const std::vector<std::optional<bool>>& asciiEncodings,
    bool withInstanceArgument) {
  auto stringVector = makeFlatVector<StringView>(tests.size());
  auto subStringVector = makeFlatVector<StringView>(tests.size());
  auto instanceVector =
      withInstanceArgument ? makeFlatVector<TInstance>(tests.size()) : nullptr;

  for (int i = 0; i < tests.size(); i++) {
    stringVector->set(i, StringView(std::get<0>(tests[i].first)));
    subStringVector->set(i, StringView(std::get<1>(tests[i].first)));
    if (instanceVector) {
      instanceVector->set(i, std::get<2>(tests[i].first));
    }
  }

  if (asciiEncodings[0].has_value()) {
    stringVector->setAllIsAscii(asciiEncodings[0].value());
  }
  if (asciiEncodings[1].has_value()) {
    subStringVector->setAllIsAscii(asciiEncodings[1].value());
  }

  FlatVectorPtr<int64_t> result;
  if (withInstanceArgument) {
    result = evaluate<FlatVector<int64_t>>(
        "strrpos(c0, c1,c2)",
        makeRowVector({stringVector, subStringVector, instanceVector}));
  } else {
    result = evaluate<FlatVector<int64_t>>(
        "strrpos(c0, c1)", makeRowVector({stringVector, subStringVector}));
  }

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), tests[i].second);
  }
}

TEST_F(StringFunctionsTest, stringPositionFromEnd) {
  strpos_input_test_t testsAscii = {
      {{"high", "ig", -1}, {2}},
      {{"high", "igx", -1}, {0}},
      {{"high", "h", -1}, {4}},
      {{"", "h", -1}, {0}},
      {{"high", "", -1}, {1}},
      {{"", "", -1}, {1}},
  };

  strpos_input_test_t testsAsciiWithPosition = {
      {{"high", "h", 2}, 1},
      {{"high", "h", 10}, 0},
      {{"high", "", 2}, {1}},
      {{"", "", 2}, {1}},
  };

  strpos_input_test_t testsUnicodeWithPosition = {
      {{"\u4FE1\u5FF5,\u7231,\u5E0C\u671B", "\u7231", 1}, 4},
      {{"\u4FE1\u5FF5,\u7231,\u5E0C\u671B", "\u5E0C\u671B", 1}, 6},
  };

  // We dont have to try all encoding combinations here since there is a test
  // that test the encoding resolution but we want to to have a test for each
  // possible resolution
  testStringPositionFromEndAllFlatVector<int64_t>(
      testsAscii, {true, true}, false);

  // Try instance parameter using BIGINT and INTEGER.
  testStringPositionFromEndAllFlatVector<int32_t>(
      testsAsciiWithPosition, {false, false}, true);
  testStringPositionFromEndAllFlatVector<int64_t>(
      testsAsciiWithPosition, {false, false}, true);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result = evaluate<SimpleVector<int64_t>>("strrpos('high', 'ig')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), 2);
  }
}

void StringFunctionsTest::testChrFlatVector(
    const std::vector<std::pair<int64_t, std::string>>& tests) {
  auto codePoints = makeFlatVector<int64_t>(tests.size());
  for (int i = 0; i < tests.size(); i++) {
    codePoints->set(i, tests[i].first);
  }

  auto result =
      evaluate<FlatVector<StringView>>("chr(c0)", makeRowVector({codePoints}));

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), StringView(tests[i].second));
  }
}

TEST_F(StringFunctionsTest, chr) {
  std::vector<std::pair<int64_t, std::string>> validInputTest = {
      {65, "A"},
      {9731, "\u2603"},
      {0, std::string("\0", 1)},
  };

  std::vector<std::pair<int64_t, std::string>> invalidInputTest{
      {65, "A"},
      {9731, "\u2603"},
      {0, std::string("\0", 1)},
      {8589934592, ""},
  };

  testChrFlatVector(validInputTest);

  EXPECT_THROW(
      testChrFlatVector(invalidInputTest), facebook::velox::VeloxUserError);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result = evaluate<SimpleVector<StringView>>("chr(65)", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView("A"));
  }
}

void StringFunctionsTest::testCodePointFlatVector(
    const std::vector<std::pair<std::string, int32_t>>& tests) {
  auto inputString = makeFlatVector<StringView>(tests.size());
  for (int i = 0; i < tests.size(); i++) {
    inputString->set(i, StringView(tests[i].first));
  }

  auto result = evaluate<FlatVector<int32_t>>(
      "codepoint(c0)", makeRowVector({inputString}));

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), tests[i].second);
  }
}

TEST_F(StringFunctionsTest, codePoint) {
  std::vector<std::pair<std::string, int32_t>> validInputTest = {
      {"x", 0x78},
      {"\u840C", 0x840C},
  };

  std::vector<std::pair<std::string, int32_t>> invalidInputTest{
      {"hello", 0},
      {"", 0},
  };

  testCodePointFlatVector(validInputTest);

  EXPECT_THROW(
      testCodePointFlatVector(invalidInputTest),
      facebook::velox::VeloxUserError);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result = evaluate<SimpleVector<int32_t>>("codepoint('x')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), 0x78);
  }
}

TEST_F(StringFunctionsTest, md5) {
  const auto md5 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "md5(c0)", {arg}, {VARBINARY()});
  };

  EXPECT_EQ(hexToDec("533f6357e0210e67d91f651bc49e1278"), md5("hashme"));
  EXPECT_EQ(hexToDec("eb2ac5b04180d8d6011a016aeb8f75b3"), md5("Infinity"));
  EXPECT_EQ(hexToDec("d41d8cd98f00b204e9800998ecf8427e"), md5(""));

  EXPECT_EQ(std::nullopt, md5(std::nullopt));
}

TEST_F(StringFunctionsTest, sha1) {
  const auto sha1 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "sha1(c0)", {arg}, {VARBINARY()});
  };

  // The result values were obtained from Presto Java sha1 function.

  EXPECT_EQ(hexToDec("DA39A3EE5E6B4B0D3255BFEF95601890AFD80709"), sha1(""));
  EXPECT_EQ(std::nullopt, sha1(std::nullopt));

  EXPECT_EQ(hexToDec("86F7E437FAA5A7FCE15D1DDCB9EAEAEA377667B8"), sha1("a"));
  EXPECT_EQ(hexToDec("382758154F5D9F9775B6A9F28B6EDD55773C87E3"), sha1("AB "));
  EXPECT_EQ(hexToDec("B858CB282617FB0956D960215C8E84D1CCF909C6"), sha1(" "));
  EXPECT_EQ(
      hexToDec("A47779C6198B85A1A2595C7C9AAAB26199EA8084"),
      sha1("               "));
  EXPECT_EQ(
      hexToDec("082DE68D348CBB63316DF2B7B74B0A2DBB716F4A"),
      sha1("SPECIAL_#@,$|%/^~?{}+-"));
  EXPECT_EQ(
      hexToDec("01B307ACBA4F54F55AAFC33BB06BBBF6CA803E9A"), sha1("1234567890"));
  EXPECT_EQ(
      hexToDec("E46990399602E8321A69285244B955816738981E"),
      sha1("12345.67890"));
  EXPECT_EQ(
      hexToDec("17BC9B38933EB1C0D5D1F8F6D9B6C375851B9685"),
      sha1("more_than_12_characters_string"));
}

TEST_F(StringFunctionsTest, sha256) {
  const auto sha256 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "sha256(c0)", {arg}, {VARBINARY()});
  };

  EXPECT_EQ(
      hexToDec(
          "02208b9403a87df9f4ed6b2ee2657efaa589026b4cce9accc8e8a5bf3d693c86"),
      sha256("hashme"));
  EXPECT_EQ(
      hexToDec(
          "d0067cad9a63e0813759a2bb841051ca73570c0da2e08e840a8eb45db6a7a010"),
      sha256("Infinity"));
  EXPECT_EQ(
      hexToDec(
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
      sha256(""));

  EXPECT_EQ(std::nullopt, sha256(std::nullopt));
}

TEST_F(StringFunctionsTest, sha512) {
  const auto sha512 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "sha512(c0)", {arg}, {VARBINARY()});
  };

  EXPECT_EQ(
      hexToDec(
          "1f6b05823a0453c1ec55009555087e8226d774c7c49d099784317b8460a0623ddaa083334f9218dda8075e0a0dc8319f89199f04e6b8f3980a73556866b388ae"),
      sha512("prestodb"));
  EXPECT_EQ(
      hexToDec(
          "7de872ed1c41ce3901bb7f12f20b0c0106331fe5b5ecc5fbbcf3ce6c79df4da595ebb7e221ab8b7fc5d918583eac6890ade1c26436335d3835828011204b7679"),
      sha512("Infinity"));
  EXPECT_EQ(
      hexToDec(
          "30163935c002fc4e1200906c3d30a9c4956b4af9f6dcaef1eb4b1fcb8fba69e7a7acdc491ea5b1f2864ea8c01b01580ef09defc3b11b3f183cb21d236f7f1a6b"),
      sha512("hash"));
  EXPECT_EQ(
      hexToDec(
          "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"),
      sha512(""));
  EXPECT_EQ(std::nullopt, sha512(std::nullopt));
}

TEST_F(StringFunctionsTest, spookyHashV232) {
  const auto spookyHashV232 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "spooky_hash_v2_32(c0)", {arg}, {VARBINARY()});
  };

  // The result values were obtained from Presto Java spooky_hash_v2_32
  // function.

  EXPECT_EQ(hexToDec("6BF50919"), spookyHashV232(""));
  EXPECT_EQ(std::nullopt, spookyHashV232(std::nullopt));

  EXPECT_EQ(hexToDec("D382E6CA"), spookyHashV232("hello"));
  EXPECT_EQ(hexToDec("4DB3FC9E"), spookyHashV232("       "));
  EXPECT_EQ(hexToDec("DC33E6F0"), spookyHashV232("special_#@,$|%/^~?{}+-"));
  EXPECT_EQ(hexToDec("C5CD219B"), spookyHashV232("1234567890"));
  EXPECT_EQ(
      hexToDec("B95F627C"), spookyHashV232("more_than_12_characters_string"));
}

TEST_F(StringFunctionsTest, spookyHashV264) {
  const auto spookyHashV264 = [&](std::optional<std::string> arg) {
    return evaluateOnce<std::string, std::string>(
        "spooky_hash_v2_64(c0)", {arg}, {VARBINARY()});
  };

  // The result values were obtained from Presto Java spooky_hash_v2_64
  // function.

  EXPECT_EQ(hexToDec("232706FC6BF50919"), spookyHashV264(""));
  EXPECT_EQ(std::nullopt, spookyHashV264(std::nullopt));

  EXPECT_EQ(hexToDec("3768826AD382E6CA"), spookyHashV264("hello"));
  EXPECT_EQ(hexToDec("8A63CCE34DB3FC9E"), spookyHashV264("       "));
  EXPECT_EQ(
      hexToDec("AAF4B42DDC33E6F0"), spookyHashV264("special_#@,$|%/^~?{}+-"));
  EXPECT_EQ(hexToDec("D9426F48C5CD219B"), spookyHashV264("1234567890"));
  EXPECT_EQ(
      hexToDec("3493AE21B95F627C"),
      spookyHashV264("more_than_12_characters_string"));
}

TEST_F(StringFunctionsTest, HmacSha1) {
  const auto hmacSha1 = [&](std::optional<std::string> arg,
                            std::optional<std::string> key) {
    return evaluateOnce<std::string, std::string>(
        "hmac_sha1(c0, c1)", {arg, key}, {VARBINARY(), VARBINARY()});
  };
  // Use python hmac lib results as the expected value.
  // >>> import hmac
  // >>> def sha1(data, key):
  //         print(hmac.new(key, data, digestmod='sha1').hexdigest())
  // >>> sha1(b"hashme", b"velox")
  // d49c944625bdde6c47ad93ea63952bfcf16a630a
  // >>> sha1(b"Infinity", b"velox")
  // c19b6b753fe4ac28579c7e84d18feb29760a0d1c
  // >>> sha1(b"", b"velox")
  // d0569c4a4f3df995b04ec497b12872c4a2f97517
  // >>> sha1(b"12345abcde54321", b"velox")
  // 183054bdaf8c83320fee4376e76ffd7e773a650f
  // sha1(b"velox", b"")
  // 3ec5ea98df0f5ddb139231ecee2c8a9810a82e08
  EXPECT_EQ(
      hexToDec("d49c944625bdde6c47ad93ea63952bfcf16a630a"),
      hmacSha1("hashme", "velox"));
  EXPECT_EQ(
      hexToDec("c19b6b753fe4ac28579c7e84d18feb29760a0d1c"),
      hmacSha1("Infinity", "velox"));
  EXPECT_EQ(
      hexToDec("d0569c4a4f3df995b04ec497b12872c4a2f97517"),
      hmacSha1("", "velox"));
  EXPECT_EQ(std::nullopt, hmacSha1(std::nullopt, "velox"));
  EXPECT_EQ(
      hexToDec("183054bdaf8c83320fee4376e76ffd7e773a650f"),
      hmacSha1("12345abcde54321", "velox"));
  EXPECT_EQ(
      hexToDec("3ec5ea98df0f5ddb139231ecee2c8a9810a82e08"),
      hmacSha1("velox", ""));
  EXPECT_EQ(std::nullopt, hmacSha1("velox", std::nullopt));
}

TEST_F(StringFunctionsTest, HmacSha256) {
  const auto hmacSha256 = [&](std::optional<std::string> arg,
                              std::optional<std::string> key) {
    return evaluateOnce<std::string, std::string>(
        "hmac_sha256(c0, c1)", {arg, key}, {VARBINARY(), VARBINARY()});
  };
  // Use python hmac lib results as the expected value.
  // >>> import hmac
  // >>> def sha256(data, key):
  //         print(hmac.new(key, data, digestmod='sha256').hexdigest())
  // >>> sha256(b"hashme", b"velox")
  // 24bb7fa25fd592ef6a4c939d4fb91b7f7f04f8813260961101117ec30f865794
  // >>> sha256(b"Infinity", b"velox")
  // f45718c9586ae7d761194485d15cbf6284b5b606ade4f9d5820fbdd1eaf52b75
  // >>> sha256(b"", b"velox")
  // fd8658b6a6b6601155fecf9a39b6f95cf030863e550073423a8e250a35c6f5a4
  EXPECT_EQ(
      hexToDec(
          "24bb7fa25fd592ef6a4c939d4fb91b7f7f04f8813260961101117ec30f865794"),
      hmacSha256("hashme", "velox"));
  EXPECT_EQ(
      hexToDec(
          "f45718c9586ae7d761194485d15cbf6284b5b606ade4f9d5820fbdd1eaf52b75"),
      hmacSha256("Infinity", "velox"));
  EXPECT_EQ(
      hexToDec(
          "fd8658b6a6b6601155fecf9a39b6f95cf030863e550073423a8e250a35c6f5a4"),
      hmacSha256("", "velox"));
  EXPECT_EQ(std::nullopt, hmacSha256(std::nullopt, "velox"));
}

TEST_F(StringFunctionsTest, HmacSha512) {
  const auto hmacSha512 = [&](std::optional<std::string> arg,
                              std::optional<std::string> key) {
    return evaluateOnce<std::string, std::string>(
        "hmac_sha512(c0, c1)", {arg, key}, {VARBINARY(), VARBINARY()});
  };
  // Use the same expected value from TestVarbinaryFunctions of presto java
  EXPECT_EQ(
      hexToDec(
          "84FA5AA0279BBC473267D05A53EA03310A987CECC4C1535FF29B6D76B8F1444A728DF3AADB89D4A9A6709E1998F373566E8F824A8CA93B1821F0B69BC2A2F65E"),
      hmacSha512("", "key"));
  EXPECT_EQ(
      hexToDec(
          "FEFA712B67DED871E1ED987F8B20D6A69EB9FCC87974218B9A1A6D5202B54C18ECDA4839A979DED22F07E0881CF40B762691992D120408F49D6212E112509D72"),
      hmacSha512("hashme", "key"));
  EXPECT_EQ(std::nullopt, hmacSha512(std::nullopt, "velox"));
}

void StringFunctionsTest::testReplaceInPlace(
    const std::vector<std::pair<std::string, std::string>>& tests,
    const std::string& search,
    const std::string& replace,
    bool multiReferenced) {
  auto makeInput = [&]() {
    auto stringVector = makeFlatVector<StringView>(tests.size());

    for (int i = 0; i < tests.size(); i++) {
      stringVector->set(i, StringView(tests[i].first));
    }
    auto crossRefVector = makeFlatVector<StringView>(1);

    if (multiReferenced) {
      crossRefVector->acquireSharedStringBuffers(stringVector.get());
    }
    return stringVector;
  };

  auto testResults = [&](const FlatVector<StringView>* results) {
    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(results->valueAt(i), StringView(tests[i].second));
    }
  };

  auto result = evaluate<FlatVector<StringView>>(
      fmt::format("replace(c0, '{}', '{}')", search, replace),
      makeRowVector({makeInput()}));
  testResults(result.get());

  // Test in place optimization. If in-place is expected, make sure it happened.
  // If its not expected make sure it did not happen.
  auto applyReplaceFunction = [&](std::vector<VectorPtr>& functionInputs,
                                  VectorPtr& resultPtr) {
    auto replaceFunction =
        exec::getVectorFunction("replace", {VARCHAR(), VARCHAR()}, {});
    SelectivityVector rows(tests.size());
    ExprSet exprSet({}, &execCtx_);
    RowVectorPtr inputRows = makeRowVector({});
    exec::EvalCtx evalCtx(&execCtx_, &exprSet, inputRows.get());
    replaceFunction->apply(rows, functionInputs, VARCHAR(), evalCtx, resultPtr);
  };

  std::vector<VectorPtr> functionInputs = {
      makeInput(),
      makeConstant(search.c_str(), tests.size()),
      makeConstant(replace.c_str(), tests.size())};

  VectorPtr resultPtr;
  applyReplaceFunction(functionInputs, resultPtr);
  testResults(resultPtr->asFlatVector<StringView>());

  if (!multiReferenced && search >= replace) {
    // Expected in-place.
    ASSERT_TRUE(resultPtr == functionInputs[0]);
  } else {
    ASSERT_FALSE(resultPtr == functionInputs[0]);
  }
}

void StringFunctionsTest::testReplaceFlatVector(
    const replace_input_test_t& tests,
    bool withReplaceArgument) {
  auto stringVector = makeFlatVector<StringView>(tests.size());
  auto searchVector = makeFlatVector<StringView>(tests.size());
  auto replaceVector =
      withReplaceArgument ? makeFlatVector<StringView>(tests.size()) : nullptr;

  for (int i = 0; i < tests.size(); i++) {
    stringVector->set(i, StringView(std::get<0>(tests[i].first)));
    searchVector->set(i, StringView(std::get<1>(tests[i].first)));
    if (withReplaceArgument) {
      replaceVector->set(i, StringView(std::get<2>(tests[i].first)));
    }
  }

  FlatVectorPtr<StringView> result;

  if (withReplaceArgument) {
    result = evaluate<FlatVector<StringView>>(
        "replace(c0, c1, c2)",
        makeRowVector({stringVector, searchVector, replaceVector}));
  } else {
    result = evaluate<FlatVector<StringView>>(
        "replace(c0, c1)", makeRowVector({stringVector, searchVector}));
  }

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), StringView(tests[i].second));
  }
}

TEST_F(StringFunctionsTest, replace) {
  replace_input_test_t testsThreeArgs = {
      {{"aaa", "a", "aa"}, {"aaaaaa"}},
      {{"123tech123", "123", "tech"}, {"techtechtech"}},
      {{"123tech123", "123", ""}, {"tech"}},
      {{"222tech", "2", "3"}, {"333tech"}},
  };

  replace_input_test_t testsTwoArgs = {
      {{"abcdefabcdef", "cd", ""}, {"abefabef"}},
      {{"123tech123", "123", ""}, {"tech"}},
      {{"", "", ""}, {""}},
  };

  testReplaceFlatVector(testsThreeArgs, true);

  testReplaceFlatVector(testsTwoArgs, false);

  // Test in place path
  std::vector<std::pair<std::string, std::string>> testsInplace = {
      {"aaa", "bbb"},
      {"aba", "bbb"},
      {"qwertyuiowertyuioqwertyuiopwertyuiopwertyuiopwertyuiopertyuioqwertyuiopwertyuiowertyuio",
       "qwertyuiowertyuioqwertyuiopwertyuiopwertyuiopwertyuiopertyuioqwertyuiopwertyuiowertyuio"},
      {"qwertyuiowertyuioqwertyuiopwertyuiopwertyuiopwertyuiopertyuioqwertyuiopwertyuiowertaaaa",
       "qwertyuiowertyuioqwertyuiopwertyuiopwertyuiopwertyuiopertyuioqwertyuiopwertyuiowertbbbb"},
  };

  testReplaceInPlace(testsInplace, "a", "b", true);
  testReplaceInPlace(testsInplace, "a", "b", false);
  testReplaceInPlace({{"a", "bb"}, {"aa", "bbbb"}}, "a", "bb", false);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result =
      evaluate<SimpleVector<StringView>>("replace('high', 'ig', 'f')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView("hfh"));
  }
}

TEST_F(StringFunctionsTest, replaceWithReusableInputButNoInplace) {
  auto c0 = ({
    auto values = makeFlatVector<std::string>({"foo"});
    auto indices = allocateIndices(100, execCtx_.pool());
    wrapInDictionary(indices, 100, values);
  });
  auto c1 =
      makeFlatVector<int64_t>(100, [](vector_size_t) { return 2033475965; });
  auto c2 = makeFlatVector<int64_t>(
      100,
      [](vector_size_t) { return 2851588633; },
      [](auto row) { return row >= 50; });
  auto result = evaluateSimplified<FlatVector<StringView>>(
      "substr(replace('bar', rtrim(c0)), c1, c2)", makeRowVector({c0, c1, c2}));
  ASSERT_EQ(result->size(), 100);
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(result->isNullAt(i));
    EXPECT_EQ(result->valueAt(i), "");
  }
  for (int i = 50; i < 100; ++i) {
    EXPECT_TRUE(result->isNullAt(i));
  }
}

TEST_F(StringFunctionsTest, controlExprEncodingPropagation) {
  std::vector<std::string> dataASCII({"ali", "ali", "ali"});
  std::vector<std::string> dataUTF8({"àáâãäåæçè", "àáâãäåæçè", "àáâãäå"});

  auto test = [&](std::string query, bool expectedEncoding) {
    auto conditionVector = makeFlatVector<bool>({false, true, false});

    auto result = evaluate<SimpleVector<StringView>>(
        query,
        makeRowVector({
            conditionVector,
            makeFlatVector(dataASCII),
            makeFlatVector(dataUTF8),
        }));
    SelectivityVector all(result->size());
    auto ascii = result->isAscii(all);
    ASSERT_EQ(ascii && ascii.value(), expectedEncoding);
  };

  // Test if expressions

  test("if(1=1, lower(C1), lower(C2))", true);

  test("if(1!=1, lower(C1), lower(C2))", false);
}

TEST_F(StringFunctionsTest, crc32) {
  const auto crc32 = [&](std::optional<std::string> value) {
    return evaluateOnce<int64_t, std::string>(
        "crc32(c0)", {value}, {VARBINARY()});
  };
  // use python3 zlib result as the expected values,
  // >>> import zlib
  // >>> print(zlib.crc32(b"DEAD_BEEF"))
  // 2634114297
  // >>> print(zlib.crc32(b"CRC32"))
  // 4128576900
  // >>> print(zlib.crc32(b"velox is an open source unified execution engine."))
  // 2173230066
  EXPECT_EQ(std::nullopt, crc32(std::nullopt));
  EXPECT_EQ(2634114297L, crc32("DEAD_BEEF"));
  EXPECT_EQ(4128576900L, crc32("CRC32"));
  EXPECT_EQ(
      2173230066L, crc32("velox is an open source unified execution engine."));
}

TEST_F(StringFunctionsTest, xxhash64) {
  const auto xxhash64 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string, std::string>(
        "xxhash64(c0)", {value}, {VARBINARY()});
  };

  const auto toVarbinary = [](const int64_t input) {
    std::string out;
    out.resize(sizeof(input));
    std::memcpy(out.data(), &input, sizeof(input));
    return out;
  };

  EXPECT_EQ(hexToDec("EF46DB3751D8E999"), xxhash64(""));
  EXPECT_EQ(std::nullopt, xxhash64(std::nullopt));

  EXPECT_EQ(hexToDec("F9D96E0E1165E892"), xxhash64("hashme"));
  EXPECT_EQ(hexToDec("26C7827D889F6DA3"), xxhash64("hello"));
  EXPECT_EQ(hexToDec("8B29AA4768367C53"), xxhash64("ABC "));
  EXPECT_EQ(hexToDec("2C32708C2F5068F9"), xxhash64("       "));
  EXPECT_EQ(hexToDec("C2B3E0336D3E0F35"), xxhash64("special_#@,$|%/^~?{}+-"));
  EXPECT_EQ(hexToDec("A9D4D4132EFF23B6"), xxhash64("1234567890"));
  EXPECT_EQ(
      hexToDec("D73C92CF24E6EC82"), xxhash64("more_than_12_characters_string"));
}

TEST_F(StringFunctionsTest, toHex) {
  const auto toHex = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("to_hex(cast(c0 as varbinary))", value);
  };

  EXPECT_EQ(std::nullopt, toHex(std::nullopt));
  EXPECT_EQ("", toHex(""));
  EXPECT_EQ("61", toHex("a"));
  EXPECT_EQ("616263", toHex("abc"));
  EXPECT_EQ("68656C6C6F20776F726C64", toHex("hello world"));
  EXPECT_EQ(
      "48656C6C6F20576F726C642066726F6D2056656C6F7821",
      toHex("Hello World from Velox!"));

  const auto toHexFromBase64 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("to_hex(from_base64(c0))", value);
  };

  EXPECT_EQ(
      "D763DAB175DA5814349354FCF23885",
      toHexFromBase64("12PasXXaWBQ0k1T88jiF"));
}

TEST_F(StringFunctionsTest, fromHex) {
  const auto fromHex = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("from_hex(c0)", value);
  };

  EXPECT_EQ(std::nullopt, fromHex(std::nullopt));
  EXPECT_EQ("", fromHex(""));
  EXPECT_EQ("a", fromHex("61"));
  EXPECT_EQ("abc", fromHex("616263"));
  EXPECT_EQ("azo", fromHex("617a6f"));
  EXPECT_EQ("azo", fromHex("617a6F"));
  EXPECT_EQ("azo", fromHex("617A6F"));
  EXPECT_EQ("hello world", fromHex("68656C6C6F20776F726C64"));
  EXPECT_EQ(
      "Hello World from Velox!",
      fromHex("48656C6C6F20576F726C642066726F6D2056656C6F7821"));

  EXPECT_THROW(fromHex("f/"), VeloxUserError);
  EXPECT_THROW(fromHex("f:"), VeloxUserError);
  EXPECT_THROW(fromHex("f@"), VeloxUserError);
  EXPECT_THROW(fromHex("f`"), VeloxUserError);
  EXPECT_THROW(fromHex("fg"), VeloxUserError);
  EXPECT_THROW(fromHex("fff"), VeloxUserError);

  const auto fromHexToBase64 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("to_base64(from_hex(c0))", value);
  };
  EXPECT_EQ(
      "12PasXXaWBQ0k1T88jiF",
      fromHexToBase64("D763DAB175DA5814349354FCF23885"));
}

TEST_F(StringFunctionsTest, toBase64) {
  const auto toBase64 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("to_base64(cast(c0 as varbinary))", value);
  };

  EXPECT_EQ(std::nullopt, toBase64(std::nullopt));
  EXPECT_EQ("", toBase64(""));
  EXPECT_EQ("YQ==", toBase64("a"));
  EXPECT_EQ("YWJj", toBase64("abc"));
  EXPECT_EQ("aGVsbG8gd29ybGQ=", toBase64("hello world"));
  EXPECT_EQ(
      "SGVsbG8gV29ybGQgZnJvbSBWZWxveCE=", toBase64("Hello World from Velox!"));
}

TEST_F(StringFunctionsTest, reverse) {
  const auto reverse = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("reverse(c0)", value);
  };

  std::string invalidStr = "Ψ\xFF\xFFΣΓΔA";
  std::string expectedInvalidStr = "AΔΓΣ\xFF\xFFΨ";

  EXPECT_EQ(std::nullopt, reverse(std::nullopt));
  EXPECT_EQ("", reverse(""));
  EXPECT_EQ("a", reverse("a"));
  EXPECT_EQ("cba", reverse("abc"));
  EXPECT_EQ("koobecaF", reverse("Facebook"));
  EXPECT_EQ("ΨΧΦΥΤΣΣΡΠΟΞΝΜΛΚΙΘΗΖΕΔΓΒΑ", reverse("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΣΤΥΦΧΨ"));
  EXPECT_EQ(
      " \u2028 \u671B\u5E0C \u7231 \u5FF5\u4FE1",
      reverse("\u4FE1\u5FF5 \u7231 \u5E0C\u671B \u2028 "));
  EXPECT_EQ(
      "\u671B\u5E0C\u2014\u7231\u2014\u5FF5\u4FE1",
      reverse("\u4FE1\u5FF5\u2014\u7231\u2014\u5E0C\u671B"));
  EXPECT_EQ(expectedInvalidStr, reverse(invalidStr));

  // Test unicode out of the valid range.
  std::string invalidUnicodeStr;
  invalidUnicodeStr.resize(3);
  // An invalid unicode within 0xD800--0xDFFF.
  int16_t invalidUnicode = 0xeda0;
  memcpy(invalidUnicodeStr.data(), &invalidUnicode, 2);
  invalidUnicodeStr[2] = '\0';
  EXPECT_THROW(reverse(invalidUnicodeStr), VeloxUserError);
}

TEST_F(StringFunctionsTest, fromBase64) {
  const auto fromBase64 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("from_base64(c0)", value);
  };

  EXPECT_EQ(std::nullopt, fromBase64(std::nullopt));
  EXPECT_EQ("", fromBase64(""));
  EXPECT_EQ("a", fromBase64("YQ=="));
  EXPECT_EQ("abc", fromBase64("YWJj"));
  EXPECT_EQ("hello world", fromBase64("aGVsbG8gd29ybGQ="));
  EXPECT_EQ(
      "Hello World from Velox!",
      fromBase64("SGVsbG8gV29ybGQgZnJvbSBWZWxveCE="));

  EXPECT_THROW(fromBase64("YQ="), VeloxUserError);
  EXPECT_THROW(fromBase64("YQ==="), VeloxUserError);
}

TEST_F(StringFunctionsTest, urlEncode) {
  const auto urlEncode = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("url_encode(c0)", value);
  };

  EXPECT_EQ(std::nullopt, urlEncode(std::nullopt));
  EXPECT_EQ("", urlEncode(""));
  EXPECT_EQ("http%3A%2F%2Ftest", urlEncode("http://test"));
  EXPECT_EQ(
      "http%3A%2F%2Ftest%3Fa%3Db%26c%3Dd", urlEncode("http://test?a=b&c=d"));
  EXPECT_EQ(
      "http%3A%2F%2F%E3%83%86%E3%82%B9%E3%83%88",
      urlEncode("http://\u30c6\u30b9\u30c8"));
  EXPECT_EQ("%7E%40%3A.-*_%2B+%E2%98%83", urlEncode("~@:.-*_+ \u2603"));
  EXPECT_EQ("test", urlEncode("test"));
}

TEST_F(StringFunctionsTest, urlDecode) {
  const auto urlDecode = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("url_decode(c0)", value);
  };

  EXPECT_EQ(std::nullopt, urlDecode(std::nullopt));
  EXPECT_EQ("", urlDecode(""));
  EXPECT_EQ("http://test", urlDecode("http%3A%2F%2Ftest"));
  EXPECT_EQ(
      "http://test?a=b&c=d", urlDecode("http%3A%2F%2Ftest%3Fa%3Db%26c%3Dd"));
  EXPECT_EQ(
      "http://\u30c6\u30b9\u30c8",
      urlDecode("http%3A%2F%2F%E3%83%86%E3%82%B9%E3%83%88"));
  EXPECT_EQ("~@:.-*_+ \u2603", urlDecode("%7E%40%3A.-*_%2B+%E2%98%83"));
  EXPECT_EQ("test", urlDecode("test"));

  EXPECT_THROW(urlDecode("http%3A%2F%2"), VeloxUserError);
  EXPECT_THROW(urlDecode("http%3A%2F%"), VeloxUserError);
  EXPECT_THROW(urlDecode("http%3A%2F%2H"), VeloxUserError);
}

TEST_F(StringFunctionsTest, toUtf8) {
  const auto toUtf8 = [&](std::optional<std::string> value) {
    return evaluateOnce<std::string>("to_utf8(c0)", value);
  };

  EXPECT_EQ(std::nullopt, toUtf8(std::nullopt));
  EXPECT_EQ("", toUtf8(""));
  EXPECT_EQ("test", toUtf8("test"));

  EXPECT_EQ(
      "abc",
      evaluateOnce<std::string>(
          "from_hex(to_hex(to_utf8(c0)))", std::optional<std::string>("abc")));

  // This case is a sanity check for the to_utf8 implementation to make sure the
  // intermediate flat vector created is of the right size. The following
  // expression reduces the selectivity vector passed to to_utf8('this') to
  // [0,1,0] (size=3, begin=1, end=2). Then the literal gets evaluated (due to
  // simplified evaluation the literal is not folded and instead evaluated
  // during execution) to a vector of size 2 and passed on to to_utf8(). Here,
  // if the intermediate flat vector is created for a size > 2 then the function
  // throws.
  EXPECT_NO_THROW(evaluateSimplified<FlatVector<bool>>(
      "to_utf8(c0) = to_utf8('this')",
      makeRowVector({makeNullableFlatVector<StringView>(
          {std::nullopt, "test"_sv, std::nullopt})})));
}

namespace {

class MultiStringFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& /*context*/,
      VectorPtr& result) const override {
    result = BaseVector::wrapInConstant(rows.size(), 0, args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        // varchar, varchar, varchar -> varchar
        exec::FunctionSignatureBuilder()
            .returnType("varchar")
            .argumentType("varchar")
            .argumentType("varchar")
            .argumentType("varchar")
            .build(),
    };
  }

  bool ensureStringEncodingSetAtAllInputs() const override {
    return true;
  }

  bool propagateStringEncodingFromAllInputs() const override {
    return true;
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_multi_string_function,
    MultiStringFunction::signatures(),
    std::make_unique<MultiStringFunction>());

TEST_F(StringFunctionsTest, ascinessOnDictionary) {
  using S = StringView;
  VELOX_REGISTER_VECTOR_FUNCTION(udf_multi_string_function, "multi_string_fn")
  vector_size_t size = 5;
  auto flatVector = makeFlatVector<StringView>({
      S("hello how do"),
      S("how are"),
      S("is this how"),
      S("abcd"),
      S("yes no"),
  });

  auto searchVector = makeFlatVector<StringView>(
      {S("hello"), S("how"), S("is"), S("abc"), S("yes")});
  auto replaceVector = makeFlatVector<StringView>(
      {S("hi"), S("hmm"), S("it"), S("mno"), S("xyz")});
  BufferPtr nulls = nullptr;

  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool());
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (int i = 0; i < size; i++) {
    rawIndices[i] = i;
  }

  auto dictionaryVector =
      BaseVector::wrapInDictionary(nulls, indices, size, flatVector);

  auto result = evaluate<SimpleVector<StringView>>(
      fmt::format("multi_string_fn(c0, c1, c2)"),
      makeRowVector({dictionaryVector, searchVector, replaceVector}));
  SelectivityVector all(size);
  auto ascii = result->isAscii(all);
  ASSERT_EQ(ascii && ascii.value(), true);
}

TEST_F(StringFunctionsTest, vectorAccessCheck) {
  using S = StringView;

  auto flatVectorWithNulls = makeNullableFlatVector<StringView>(
      std::vector<std::optional<StringView>>{
          S("hello"), std::nullopt, S("world")},
      VARCHAR());

  auto vectorWithNulls = flatVectorWithNulls->as<SimpleVector<StringView>>();
  SelectivityVector rows(vectorWithNulls->size());
  rows.setValid(1, false); // Dont access the middle element.
  vectorWithNulls->computeAndSetIsAscii(rows);
  auto ascii = vectorWithNulls->isAscii(rows);
  ASSERT_TRUE(ascii && ascii.value());
}

TEST_F(StringFunctionsTest, switchCaseCheck) {
  auto testConditionalPropagation = [&](std::vector<bool> conditionColumn,
                                        std::vector<std::string> firstColumn,
                                        std::vector<std::string> secondColumn,
                                        SelectivityVector rows,
                                        std::string query,
                                        bool isAscii) {
    auto conditionVector = makeFlatVector<bool>(conditionColumn);
    auto argASCII = makeFlatVector<std::string>(firstColumn);
    auto asciiVector = argASCII->as<SimpleVector<StringView>>();
    asciiVector->computeAndSetIsAscii(rows);

    auto argUTF8 = makeFlatVector<std::string>(secondColumn);
    auto utfVector = argUTF8->as<SimpleVector<StringView>>();
    utfVector->computeAndSetIsAscii(rows);

    auto result = evaluate<FlatVector<StringView>>(
        query, makeRowVector({conditionVector, argASCII, argUTF8}));
    auto ascii = result->isAscii(rows);
    ASSERT_EQ(ascii && ascii.value(), isAscii);
  };

  auto condition = std::vector<bool>{false, true, false};
  auto c1 = std::vector<std::string>{"ali", "ali", "ali"};
  auto c2 = std::vector<std::string>{"àáâãäåæçè", "àáâãäåæçè", "àáâãäå"};
  SelectivityVector rows(condition.size());
  testConditionalPropagation(
      condition, c1, c2, rows, "if(C0, upper(C1), lower(C1))", true);
  testConditionalPropagation(
      condition, c1, c2, rows, "lower(if(C0, C2, C2))", false);
  testConditionalPropagation(
      condition, c1, c2, rows, "lower(if(C0, C1, C1))", true);
}

TEST_F(StringFunctionsTest, asciiPropogation) {
  /// This test case catches case where we ensure that ascii propagation is
  /// the AND of all input vectors.

  VELOX_REGISTER_VECTOR_FUNCTION(udf_multi_string_function, "multi_string_fn")

  auto c1 = std::vector<std::string>{"a", "a", "a"};
  auto c2 = std::vector<std::string>{"à", "b", "å"};
  auto c3 = std::vector<std::string>{"a", "a", "a"};

  SelectivityVector rows(c1.size(), false);
  rows.setValid(2, true);
  rows.updateBounds();
  testAsciiPropagation(c1, c2, c3, rows, /*isAscii*/ false);

  // There is no row level asciiness, thus even the middle element will
  // be false.
  rows.clearAll();
  rows.setValid(1, true);
  rows.updateBounds();
  testAsciiPropagation(c1, c2, c3, rows, /*isAscii*/ false);

  // Only compute asciiness on first row.
  rows.setAll();
  testAsciiPropagation(
      c1, c2, c3, rows, /*isAscii*/ false, "multi_string_fn", {0});

  testAsciiPropagation(c1, c3, c3, rows, /*isAscii*/ true);
}

namespace {

class InputModifyingFunction : public MultiStringFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    MultiStringFunction::apply(rows, args, outputType, context, result);

    // Modify args and remove its asciness
    for (auto& arg : args) {
      auto input = arg->as<SimpleVector<StringView>>();
      input->invalidateIsAscii();
    }
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_input_modifying_string_function,
    InputModifyingFunction::signatures(),
    std::make_unique<InputModifyingFunction>());

TEST_F(StringFunctionsTest, asciiPropogationOnInputModification) {
  /// This test case catches case where we ensure that ascii propagation is
  /// still captured despite inputs being modified.

  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_input_modifying_string_function, "modifying_string_input")

  auto c1 = std::vector<std::string>{"a", "a", "a"};
  auto c2 = std::vector<std::string>{"à", "b", "å"};
  auto c3 = std::vector<std::string>{"a", "a", "a"};

  SelectivityVector rows(c1.size(), false);
  rows.setValid(2, true);
  rows.updateBounds();
  testAsciiPropagation(c1, c2, c3, rows, false, "modifying_string_input");
}

namespace {
class AsciiPropagationCheckFn : public MultiStringFunction {
 public:
  std::optional<std::vector<size_t>> propagateStringEncodingFrom()
      const override {
    return {{0, 1}};
  }

  bool propagateStringEncodingFromAllInputs() const override {
    return false;
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_ascii_propagation_check,
    AsciiPropagationCheckFn::signatures(),
    std::make_unique<AsciiPropagationCheckFn>());

TEST_F(StringFunctionsTest, asciiPropagationForSpecificInput) {
  /// This test case catches case where we ensure that ascii propagation is
  /// only propagated from the inputs specified.

  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_ascii_propagation_check, "index_ascii_propagation")

  auto c1 = std::vector<std::string>{"a", "a", "a"};
  auto c2 = std::vector<std::string>{"à", "à", "å"};
  auto c3 = std::vector<std::string>{"a", "a", "a"};

  SelectivityVector all(c1.size());
  testAsciiPropagation(c1, c2, c3, all, false, "index_ascii_propagation");

  testAsciiPropagation(c1, c3, c2, all, true, "index_ascii_propagation");
}

namespace {
class AsciiPropagationTestFn : public MultiStringFunction {
 public:
  std::optional<std::vector<size_t>> propagateStringEncodingFrom()
      const override {
    return {{0, 1}};
  }

  bool propagateStringEncodingFromAllInputs() const override {
    return false;
  }

  bool ensureStringEncodingSetAtAllInputs() const override {
    return false;
  }

  std::vector<size_t> ensureStringEncodingSetAt() const override {
    return {1};
  }
};
} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_ascii_propagation_input_check,
    InputModifyingFunction::signatures(),
    std::make_unique<AsciiPropagationTestFn>());

TEST_F(StringFunctionsTest, asciiPropagationWithDisparateInput) {
  /// This test case catches case where we ensure that ascii propagation
  /// should happen even without computing asciiness on the inputs.

  VELOX_REGISTER_VECTOR_FUNCTION(
      udf_ascii_propagation_check, "ascii_propagation_check")

  auto c1 = std::vector<std::string>{"a", "a", "a"};
  auto c2 = std::vector<std::string>{"à", "à", "å"};
  auto c3 = std::vector<std::string>{"a", "a", "a"};

  // Compute Asciness for inputs 2,3 explicitly.
  SelectivityVector all(c1.size());
  testAsciiPropagation(c1, c2, c3, all, false, "ascii_propagation_check");

  testAsciiPropagation(c1, c3, c2, all, true, "ascii_propagation_check");

  // Do not compute asciness explicitly.
  testAsciiPropagation(c1, c2, c3, all, false, "ascii_propagation_check", {});

  testAsciiPropagation(c1, c3, c2, all, true, "ascii_propagation_check", {});
}

TEST_F(StringFunctionsTest, trim) {
  // Making input vector
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(4, complexStr.size() - 8);

  const auto trim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("trim(c0)", input);
  };

  EXPECT_EQ("facebook", trim("  facebook  "));
  EXPECT_EQ("facebook", trim("  facebook"));
  EXPECT_EQ("facebook", trim("facebook  "));
  EXPECT_EQ("facebook", trim("\n\nfacebook \n "));
  EXPECT_EQ("", trim(" \n"));
  EXPECT_EQ("", trim(""));
  EXPECT_EQ("", trim("    "));
  EXPECT_EQ("a", trim("  a  "));

  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B \u2028 "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim("  \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      trim(" \u2028 \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));

  EXPECT_EQ(expectedComplexStr, trim(complexStr));
  EXPECT_EQ(
      "Ψ\xFF\xFFΣΓΔA", trim("\u2028 \r \t \nΨ\xFF\xFFΣΓΔA \u2028 \r \t \n"));
}

TEST_F(StringFunctionsTest, ltrim) {
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(4, complexStr.size() - 4);

  const auto ltrim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("ltrim(c0)", input);
  };

  EXPECT_EQ("facebook", ltrim("facebook"));
  EXPECT_EQ("facebook ", ltrim("  facebook "));
  EXPECT_EQ("facebook \n", ltrim("\n\nfacebook \n"));
  EXPECT_EQ("", ltrim("\n"));
  EXPECT_EQ("", ltrim(" "));
  EXPECT_EQ("", ltrim("     "));
  EXPECT_EQ("a  ", ltrim("  a  "));
  EXPECT_EQ("facebo ok", ltrim(" facebo ok"));
  EXPECT_EQ("move fast", ltrim("\tmove fast"));
  EXPECT_EQ("move fast", ltrim("\r\t move fast"));
  EXPECT_EQ("hello", ltrim("\n\t\r hello"));

  EXPECT_EQ("\u4F60\u597D", ltrim(" \u4F60\u597D"));
  EXPECT_EQ("\u4F60\u597D ", ltrim(" \u4F60\u597D "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      ltrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B ",
      ltrim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      ltrim("  \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      ltrim(" \u2028 \u4FE1\u5FF5 \u7231 \u5E0C\u671B"));

  EXPECT_EQ(expectedComplexStr, ltrim(complexStr));
  EXPECT_EQ("Ψ\xFF\xFFΣΓΔA", ltrim("  \u2028 \r \t \n   Ψ\xFF\xFFΣΓΔA"));
}

TEST_F(StringFunctionsTest, rtrim) {
  std::string complexStr = generateComplexUtf8();
  std::string expectedComplexStr = complexStr.substr(0, complexStr.size() - 4);

  const auto rtrim = [&](std::optional<std::string> input) {
    return evaluateOnce<std::string>("rtrim(c0)", input);
  };

  EXPECT_EQ("facebook", rtrim("facebook"));
  EXPECT_EQ(" facebook", rtrim(" facebook  "));
  EXPECT_EQ("\nfacebook", rtrim("\nfacebook \n\n"));
  EXPECT_EQ("", rtrim(" \n"));
  EXPECT_EQ("", rtrim(" "));
  EXPECT_EQ("", rtrim("     "));
  EXPECT_EQ("  a", rtrim("  a  "));
  EXPECT_EQ("facebo ok", rtrim("facebo ok "));
  EXPECT_EQ("move fast", rtrim("move fast\t"));
  EXPECT_EQ("move fast", rtrim("move fast\r\t "));
  EXPECT_EQ("hello", rtrim("hello\n\t\r "));

  EXPECT_EQ(" \u4F60\u597D", rtrim(" \u4F60\u597D"));
  EXPECT_EQ(" \u4F60\u597D", rtrim(" \u4F60\u597D "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      " \u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim(" \u4FE1\u5FF5 \u7231 \u5E0C\u671B "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  "));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B",
      rtrim("\u4FE1\u5FF5 \u7231 \u5E0C\u671B \u2028 "));

  EXPECT_EQ(expectedComplexStr, rtrim(complexStr));
  EXPECT_EQ("     Ψ\xFF\xFFΣΓΔA", rtrim("     Ψ\xFF\xFFΣΓΔA \u2028 \r \t \n"));
}

TEST_F(StringFunctionsTest, rpad) {
  const auto rpad = [&](std::optional<std::string> string,
                        std::optional<int64_t> size,
                        std::optional<std::string> padString) {
    return evaluateOnce<std::string>(
        "rpad(c0, c1, c2)", string, size, padString);
  };

  std::string invalidString = "Ψ\xFF\xFFΣΓΔA";
  std::string invalidPadString = "\xFFΨ\xFF";

  // Null arguments
  EXPECT_EQ(std::nullopt, rpad(std::nullopt, 16, "abc"));
  EXPECT_EQ(std::nullopt, rpad("xyz", std::nullopt, "abc"));
  EXPECT_EQ(std::nullopt, rpad("xyz", 16, std::nullopt));
  // ASCII strings with various values for size and padString
  EXPECT_EQ("textx", rpad("text", 5, "x"));
  EXPECT_EQ("text", rpad("text", 4, "x"));
  EXPECT_EQ("textxy", rpad("text", 6, "xy"));
  EXPECT_EQ("textxyx", rpad("text", 7, "xy"));
  EXPECT_EQ("textxyzxy", rpad("text", 9, "xyz"));
  // Non-ASCII strings with various values for size and padString
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u671B",
      rpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 10, "\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u671B\u671B",
      rpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 11, "\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u5E0C\u671B\u5E0C",
      rpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 12, "\u5E0C\u671B"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 \u5E0C\u671B  \u5E0C\u671B\u5E0C\u671B",
      rpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 13, "\u5E0C\u671B"));
  // Empty string
  EXPECT_EQ("aaa", rpad("", 3, "a"));
  // Truncating string
  EXPECT_EQ("", rpad("abc", 0, "e"));
  EXPECT_EQ("tex", rpad("text", 3, "xy"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 ",
      rpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 5, "\u671B"));
  // Invalid UTF-8 chars
  EXPECT_EQ(invalidString + "x", rpad(invalidString, 8, "x"));
  EXPECT_EQ("abc" + invalidPadString, rpad("abc", 6, invalidPadString));
}

TEST_F(StringFunctionsTest, lpad) {
  const auto lpad = [&](std::optional<std::string> string,
                        std::optional<int64_t> size,
                        std::optional<std::string> padString) {
    return evaluateOnce<std::string>(
        "lpad(c0, c1, c2)", string, size, padString);
  };

  std::string invalidString = "Ψ\xFF\xFFΣΓΔA";
  std::string invalidPadString = "\xFFΨ\xFF";

  // Null arguments
  EXPECT_EQ(std::nullopt, lpad(std::nullopt, 16, "abc"));
  EXPECT_EQ(std::nullopt, lpad("xyz", std::nullopt, "abc"));
  EXPECT_EQ(std::nullopt, lpad("xyz", 16, std::nullopt));
  // ASCII strings with various values for size and padString
  EXPECT_EQ("xtext", lpad("text", 5, "x"));
  EXPECT_EQ("text", lpad("text", 4, "x"));
  EXPECT_EQ("xytext", lpad("text", 6, "xy"));
  EXPECT_EQ("xyxtext", lpad("text", 7, "xy"));
  EXPECT_EQ("xyzxytext", lpad("text", 9, "xyz"));
  // Non-ASCII strings with various values for size and padString
  EXPECT_EQ(
      "\u671B\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 10, "\u671B"));
  EXPECT_EQ(
      "\u671B\u671B\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 11, "\u671B"));
  EXPECT_EQ(
      "\u5E0C\u671B\u5E0C\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 12, "\u5E0C\u671B"));
  EXPECT_EQ(
      "\u5E0C\u671B\u5E0C\u671B\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ",
      lpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 13, "\u5E0C\u671B"));
  // Empty string
  EXPECT_EQ("aaa", lpad("", 3, "a"));
  // Truncating string
  EXPECT_EQ("", lpad("abc", 0, "e"));
  EXPECT_EQ("tex", lpad("text", 3, "xy"));
  EXPECT_EQ(
      "\u4FE1\u5FF5 \u7231 ",
      lpad("\u4FE1\u5FF5 \u7231 \u5E0C\u671B  ", 5, "\u671B"));
  // Invalid UTF-8 chars
  EXPECT_EQ("x" + invalidString, lpad(invalidString, 8, "x"));
  EXPECT_EQ(invalidPadString + "abc", lpad("abc", 6, invalidPadString));
}

TEST_F(StringFunctionsTest, concatInSwitchExpr) {
  auto data = makeRowVector(
      {makeFlatVector<bool>({true, false}),
       makeFlatVector<StringView>(
           {"This is a long sentence"_sv, "This is some other sentence"_sv})});

  auto result =
      evaluate("if(c0, concat(c1, '-zzz'), concat('aaa-', c1))", data);
  auto expected = makeFlatVector<StringView>(
      {"This is a long sentence-zzz"_sv, "aaa-This is some other sentence"_sv});
  test::assertEqualVectors(expected, result);
}
