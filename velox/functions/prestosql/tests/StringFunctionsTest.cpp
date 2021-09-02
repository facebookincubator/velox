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
#include "velox/exec/tests/utils/FunctionUtils.h"
#include "velox/expression/Expr.h"
#include "velox/functions/Udf.h"
#include "velox/functions/lib/StringEncodingUtils.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/parse/Expressions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;
using facebook::velox::functions::stringCore::StringEncodingMode;

namespace {
/// Generate an ascii random string of size length
std::string generateRandomString(size_t length) {
  const std::string chars =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::string randomString;
  for (std::size_t i = 0; i < length; ++i) {
    randomString += chars[rand() % chars.size()];
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

int hexToDec(char c) {
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  VELOX_FAIL("Unsupported hex character: {}", c);
}

std::string hexToDec(const std::string& str) {
  char output[16];
  auto chars = str.data();
  for (int i = 0; i < 16; i++) {
    int high = hexToDec(chars[2 * i]);
    int low = hexToDec(chars[2 * i + 1]);
    output[i] = (high << 4) | (low & 0xf);
  }
  return std::string(output, 16);
}
} // namespace

class StringFunctionsTest : public FunctionBaseTest {
 protected:
  template <typename VC = FlatVector<StringView>>
  VectorPtr makeStrings(
      vector_size_t size,
      const std::vector<std::string>& inputStrings) {
    auto strings = std::dynamic_pointer_cast<VC>(BaseVector::create(
        CppToType<StringView>::create(), size, execCtx_.pool()));
    for (int i = 0; i < size; i++) {
      if (!expectNullString(i)) {
        strings->set(i, StringView(inputStrings[i].c_str()));
      } else {
        strings->setNull(i, true);
      }
    }
    return strings;
  }

  template <typename T>
  int bufferRefCounts(FlatVector<T>* vector) {
    int refCounts = 0;
    for (auto& buffer : vector->stringBuffers())
      refCounts += buffer->refCount();
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
      folly::Optional<StringEncodingMode> stringEncoding,
      bool multiReferenced,
      StringEncodingMode expectedResultEncoding) {
    auto inputsFlatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), tests.size(), execCtx_.pool()));

    for (int i = 0; i < tests.size(); i++) {
      inputsFlatVector->set(i, StringView(std::get<0>(tests[i])));
    }

    if (stringEncoding.has_value()) {
      inputsFlatVector->setStringEncoding(stringEncoding.value());
    }

    auto crossRefVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), 1, execCtx_.pool()));

    if (multiReferenced) {
      crossRefVector->acquireSharedStringBuffers(inputsFlatVector.get());
    }

    auto result = evaluate<FlatVector<StringView>>(
        "upper(c0)", makeRowVector({inputsFlatVector}));

    ASSERT_EQ(result->getStringEncoding().value(), expectedResultEncoding);

    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), StringView(std::get<1>(tests[i])));
    }
  }

  void testLowerFlatVector(
      const std::vector<std::tuple<std::string, std::string>>& tests,
      folly::Optional<StringEncodingMode> stringEncoding,
      bool multiReferenced,
      StringEncodingMode expectedResultEncoding) {
    auto inputsFlatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), tests.size(), execCtx_.pool()));

    for (int i = 0; i < tests.size(); i++) {
      inputsFlatVector->set(i, StringView(std::get<0>(tests[i])));
    }

    if (stringEncoding.has_value()) {
      inputsFlatVector->setStringEncoding(stringEncoding.value());
    }

    auto crossRefVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), 1, execCtx_.pool()));

    if (multiReferenced) {
      crossRefVector->acquireSharedStringBuffers(inputsFlatVector.get());
    }
    auto testQuery = [&](const std::string& query) {
      auto result = evaluate<FlatVector<StringView>>(
          query, makeRowVector({inputsFlatVector}));
      ASSERT_EQ(result->getStringEncoding().value(), expectedResultEncoding);

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
    auto result = evaluate<FlatVector<StringView>>(
        buildConcatQuery(), makeRowVector(inputVectors));

    auto concatStd = [](const std::vector<std::string>& inputs) {
      std::string output;
      for (auto& input : inputs) {
        output += input;
      }
      return output;
    };

    for (int i = 0; i < inputTable.size(); ++i) {
      EXPECT_EQ(result->valueAt(i), StringView(concatStd(inputTable[i])));
    }
  }

  void testLengthFlatVector(
      const std::vector<std::tuple<std::string, int64_t>>& tests,
      folly::Optional<StringEncodingMode> stringEncoding) {
    auto inputsFlatVector = std::dynamic_pointer_cast<FlatVector<StringView>>(
        BaseVector::create(VARCHAR(), tests.size(), execCtx_.pool()));

    for (int i = 0; i < tests.size(); i++) {
      inputsFlatVector->set(i, StringView(std::get<0>(tests[i])));
    }
    if (stringEncoding.has_value()) {
      inputsFlatVector->setStringEncoding(stringEncoding.value());
    }
    auto result = evaluate<FlatVector<int64_t>>(
        "length(c0)", makeRowVector({inputsFlatVector}));

    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), std::get<1>(tests[i]));
    }
  }

  using strpos_input_test_t = std::vector<
      std::pair<std::tuple<std::string, std::string, int64_t>, int64_t>>;

  void testStringPositionAllFlatVector(
      const strpos_input_test_t& tests,
      const std::vector<folly::Optional<StringEncodingMode>>& stringEncodings,
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

  void testStringEncodingResolution(
      const std::vector<std::vector<std::string>>& content,
      const std::vector<folly::Optional<StringEncodingMode>>& encodings,
      StringEncodingMode expectedResolvedEncoding);

  void testXXHash64(
      const std::vector<std::tuple<std::string, int64_t, int64_t>>& tests);

  void testXXHash64(
      const std::vector<std::pair<std::string, int64_t>>& tests,
      bool stringVariant);
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

/**
 * The test for user exception checking
 */
TEST_F(StringFunctionsTest, substrArgumentExceptionCheck) {
  vector_size_t size = 100;

  std::vector<std::string> strings(size);
  std::generate(strings.begin(), strings.end(), [i = -1]() mutable {
    i++;
    return std::to_string(i) + "_MYSTR_" + std::to_string(i * 100);
  });

  auto stringVector = makeStrings(size, strings);

  auto row = makeRowVector({stringVector});

  EXPECT_THROW(
      evaluate<FlatVector<StringView>>("substr('my string here', 'A')", row),
      std::invalid_argument);

  EXPECT_THROW(
      evaluate<FlatVector<StringView>>("substr('my string here', 1.0)", row),
      std::invalid_argument);

  EXPECT_THROW(
      evaluate<FlatVector<StringView>>("substr('my string here')", row),
      std::invalid_argument);
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
      allTests,
      StringEncodingMode::ASCII,
      true /*multiRef*/,
      StringEncodingMode::ASCII);
  testUpperFlatVector(
      allTests, StringEncodingMode::ASCII, false, StringEncodingMode::ASCII);

  auto&& unicodeTests = getUpperUnicodeTestData();
  allTests.insert(allTests.end(), unicodeTests.begin(), unicodeTests.end());

  // Test unicode
  testUpperFlatVector(
      allTests, StringEncodingMode::UTF8, false, StringEncodingMode::UTF8);
  testUpperFlatVector(
      allTests,
      StringEncodingMode::MOSTLY_ASCII,
      false,
      StringEncodingMode::MOSTLY_ASCII);
  testUpperFlatVector(
      allTests, folly::none, false, StringEncodingMode::MOSTLY_ASCII);

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

  testLowerFlatVector(
      allTests, StringEncodingMode::ASCII, true, StringEncodingMode::ASCII);
  testLowerFlatVector(
      allTests, StringEncodingMode::ASCII, false, StringEncodingMode::ASCII);

  auto&& unicodeTests = getLowerUnicodeTestData();
  allTests.insert(allTests.end(), unicodeTests.begin(), unicodeTests.end());

  // Test unicode
  testLowerFlatVector(
      allTests, StringEncodingMode::UTF8, false, StringEncodingMode::UTF8);
  testLowerFlatVector(
      allTests,
      StringEncodingMode::MOSTLY_ASCII,
      false,
      StringEncodingMode::MOSTLY_ASCII);
  testLowerFlatVector(
      allTests, folly::none, false, StringEncodingMode::MOSTLY_ASCII);

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
        inputTable[row][col] = generateRandomString(rand() % maxStringLength);
      }
    }
    testConcatFlatVector(inputTable, argsCount);
  }

  // Test constant input vector with 2 args
  auto rows = makeRowVector(makeRowType({VARCHAR(), VARCHAR()}), 10);
  auto c0 = generateRandomString(20);
  auto c1 = generateRandomString(20);
  auto result = evaluate<SimpleVector<StringView>>(
      fmt::format("concat('{}', '{}')", c0, c1), rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView(c0 + c1));
  }

  // Test string encoding propagation
  result =
      evaluate<SimpleVector<StringView>>("concat('ali', 'ali','ali')", rows);
  EXPECT_EQ(result->getStringEncoding().value(), StringEncodingMode::ASCII);

  result = evaluate<SimpleVector<StringView>>(
      "concat('ali', 'àáâãäåæçè','ali')", rows);
  EXPECT_EQ(result->getStringEncoding().value(), StringEncodingMode::UTF8);
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

  testLengthFlatVector(tests, StringEncodingMode::ASCII);
  testLengthFlatVector(tests, StringEncodingMode::MOSTLY_ASCII);
  testLengthFlatVector(tests, StringEncodingMode::UTF8);
  testLengthFlatVector(tests, folly::none);

  // Test unicode
  for (auto& pair : getUpperUnicodeTestData()) {
    auto& string = std::get<0>(pair);
    tests.push_back(std::make_tuple(string, lengthUtf8Ref(string)));
  };

  testLengthFlatVector(tests, StringEncodingMode::UTF8);
  testLengthFlatVector(tests, folly::none);

  // Test constant vectors
  auto rows = makeRowVector({makeRowVector(
      {makeFlatVector<int32_t>(10, [](vector_size_t row) { return row; })})});
  auto result = evaluate<SimpleVector<int64_t>>("length('test length')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), 11);
  }
}

// Test strpos function
void StringFunctionsTest::testStringPositionAllFlatVector(
    const strpos_input_test_t& tests,
    const std::vector<folly::Optional<StringEncodingMode>>& stringEncodings,
    bool withInstanceArgument) {
  auto stringVector = makeFlatVector<StringView>(tests.size());
  auto subStringVector = makeFlatVector<StringView>(tests.size());
  auto instanceVector =
      withInstanceArgument ? makeFlatVector<int64_t>(tests.size()) : nullptr;

  for (int i = 0; i < tests.size(); i++) {
    stringVector->set(i, StringView(std::get<0>(tests[i].first)));
    subStringVector->set(i, StringView(std::get<1>(tests[i].first)));
    if (instanceVector) {
      instanceVector->set(i, std::get<2>(tests[i].first));
    }
  }

  if (stringEncodings[0].has_value()) {
    stringVector->setStringEncoding(stringEncodings[0].value());
  }
  if (stringEncodings[1].has_value()) {
    subStringVector->setStringEncoding(stringEncodings[1].value());
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
  testStringPositionAllFlatVector(
      testsAscii,
      {StringEncodingMode::ASCII, StringEncodingMode::ASCII},
      false);

  testStringPositionAllFlatVector(
      testsAsciiWithPosition,
      {StringEncodingMode::MOSTLY_ASCII, StringEncodingMode::MOSTLY_ASCII},
      true);

  testStringPositionAllFlatVector(
      testsUnicodeWithPosition,
      {StringEncodingMode::UTF8, StringEncodingMode::UTF8},
      true);

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result = evaluate<SimpleVector<int64_t>>("strpos('high', 'ig')", rows);
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
  {
    const auto md5 = [&](std::optional<std::string> key,
                         std::optional<int32_t> radix) {
      return evaluateOnce<std::string>("md5(c0, c1)", key, radix);
    };
    EXPECT_EQ("533f6357e0210e67d91f651bc49e1278", md5("hashme", 16));
    EXPECT_EQ("110655053273001216628802061412889137784", md5("hashme", 10));
    EXPECT_EQ("d41d8cd98f00b204e9800998ecf8427e", md5("", 16));
    EXPECT_EQ("281949768489412648962353822266799178366", md5("", 10));

    EXPECT_THROW(
        try {
          md5("hashme", 2);
        } catch (const facebook::velox::VeloxUserError& err) {
          EXPECT_NE(
              err.message().find(
                  "Not a valid radix for md5: 2. Supported values are 10 or 16"),
              std::string::npos);
          throw;
        },
        facebook::velox::VeloxUserError);
  }

  {
    const auto md5 = [&](const std::string& key) {
      return evaluateOnce<std::string>(
          "md5(c0)",
          std::vector<std::optional<StringView>>{StringView(key)},
          {VARBINARY()});
    };

    EXPECT_EQ(hexToDec("533f6357e0210e67d91f651bc49e1278"), md5("hashme"));
    EXPECT_EQ(hexToDec("D41D8CD98F00B204E9800998ECF8427E"), md5(""));
  }

  // Test null input
  {
    auto result = evaluateOnce<std::string>(
        "md5(c0)", std::optional<std::string>(std::nullopt));
    ASSERT_EQ(result, std::nullopt);
  }
}

void StringFunctionsTest::testReplaceInPlace(
    const std::vector<std::pair<std::string, std::string>>& tests,
    const std::string& search,
    const std::string& replace,
    bool multiReferenced) {
  auto stringVector = makeFlatVector<StringView>(tests.size());

  for (int i = 0; i < tests.size(); i++) {
    stringVector->set(i, StringView(tests[i].first));
  }

  auto crossRefVector = makeFlatVector<StringView>(1);

  if (multiReferenced) {
    crossRefVector->acquireSharedStringBuffers(stringVector.get());
  }

  FlatVectorPtr<StringView> result = evaluate<FlatVector<StringView>>(
      fmt::format("replace(c0, '{}', '{}')", search, replace),
      makeRowVector({stringVector}));

  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), StringView(tests[i].second));
    if (!multiReferenced && !stringVector->valueAt(i).isInline() &&
        search.size() <= replace.size()) {
      ASSERT_EQ(result->valueAt(i), stringVector->valueAt(i));
    }
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

  // Test constant vectors
  auto rows = makeRowVector(makeRowType({BIGINT()}), 10);
  auto result =
      evaluate<SimpleVector<StringView>>("replace('high', 'ig', 'f')", rows);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->valueAt(i), StringView("hfh"));
  }
}

void StringFunctionsTest::testStringEncodingResolution(
    const std::vector<std::vector<std::string>>& content,
    const std::vector<folly::Optional<StringEncodingMode>>& encodings,
    StringEncodingMode expectedResolvedEncoding) {
  std::vector<FlatVectorPtr<StringView>> inputVectors;
  inputVectors.resize(content.size());

  for (int i = 0; i < content.size(); i++) {
    inputVectors[i] = makeFlatVector<StringView>(content[i].size());
    for (int j = 0; j < content[i].size(); j++) {
      inputVectors[i]->set(j, StringView(content[i][j]));
    }
  }

  for (int i = 0; i < encodings.size(); i++) {
    if (encodings[i].has_value()) {
      inputVectors[i]->setStringEncoding(encodings[i].value());
    }
  }

  SelectivityVector rows(content[0].size());
  rows.setAll();

  std::vector<BaseVector*> baseVectors;
  for (auto vector : inputVectors) {
    baseVectors.push_back(vector.get());
  }

  // A dummy context used to call determineStringEncoding
  exec::EvalCtx evalCtx(
      &execCtx_, nullptr, makeRowVector(ROW({}, {}), 0).get());

  auto resolvedEncoding = StringEncodingMode::ASCII;
  for (auto vector : baseVectors) {
    auto simpleVector = vector->as<SimpleVector<StringView>>();
    if (!simpleVector->getStringEncoding().has_value()) {
      SelectivityVector allRows(simpleVector->size());
      determineStringEncoding(&evalCtx, simpleVector, allRows);
    }

    resolvedEncoding = functions::maxEncoding(
        resolvedEncoding, simpleVector->getStringEncoding().value());
  }
  ASSERT_EQ(resolvedEncoding, expectedResolvedEncoding);
}

TEST_F(StringFunctionsTest, controlExprEncodingPropagation) {
  std::vector<std::string> dataASCII({"ali", "ali", "ali"});
  std::vector<std::string> dataUTF8({"àáâãäåæçè", "àáâãäåæçè", "àáâãäå"});

  auto test = [&](std::string query, StringEncodingMode expectedEncoding) {
    auto conditionVector = makeFlatVector<bool>({false, true, false});

    auto result = evaluate<SimpleVector<StringView>>(
        query,
        makeRowVector({
            conditionVector,
            makeFlatVector(dataASCII),
            makeFlatVector(dataUTF8),
        }));
    ASSERT_TRUE(result->getStringEncoding().has_value());
    ASSERT_EQ(result->getStringEncoding().value(), expectedEncoding);
  };

  auto testEncodingNotSet = [&](std::string query) {
    auto conditionVector = makeFlatVector<bool>({false, true, false});

    auto result = evaluate<SimpleVector<StringView>>(
        query,
        makeRowVector({
            conditionVector,
            makeFlatVector(dataASCII),
            makeFlatVector(dataUTF8),
        }));
    ASSERT_EQ(result->getStringEncoding(), std::nullopt);
  };

  // Test if expressions

  // Setting encoding on partially populated vector is not yet supported.
  testEncodingNotSet("if(C0, lower(C1), lower(C2))");

  // always then path
  test("if(1=1, lower(C1), lower(C2))", StringEncodingMode::ASCII);

  test("if(1!=1, lower(C1), lower(C2))", StringEncodingMode::UTF8);

  // if and shared expression

  // Setting encoding on partially populated vector is not yet supported.
  testEncodingNotSet("if(C0, lower(C1), lower(C1))");

  // Test const expressions ascii
  test("if(C0, 'ali', 'àáâãäåæçè')", StringEncodingMode::UTF8);

  test("if(C0, 'ali' ,'ali')", StringEncodingMode::ASCII);

  // Test some more complicated expression
  test(
      "if(C0, upper('ali'), if(C0,'ali','àáâãäåæçè'))",
      StringEncodingMode::UTF8);

  // Test field reference expression
  testEncodingNotSet("if(C0, C1, C2)");

  testEncodingNotSet("if(1=1, C1, C2)");
  testEncodingNotSet("if(1=1, C1, C1)");

  auto testWithEncodingPreset = [&](std::string query,
                                    StringEncodingMode expectedEncoding) {
    auto conditionVector = makeFlatVector<bool>({false, true, false});
    auto argASCII = makeFlatVector<std::string>({"ali", "ali", "ali"});
    argASCII->setStringEncoding(StringEncodingMode::ASCII);

    auto argUTF8 =
        makeFlatVector<std::string>({"àáâãäåæçè", "àáâãäåæçè", "àáâãäå"});
    argUTF8->setStringEncoding(StringEncodingMode::UTF8);

    auto result = evaluate<FlatVector<StringView>>(
        query, makeRowVector({conditionVector, argASCII, argUTF8}));
    ASSERT_EQ(result->getStringEncoding().value(), expectedEncoding);
  };

  testWithEncodingPreset("if(C0, C1, C1)", StringEncodingMode::ASCII);
  testWithEncodingPreset("if(C0, C2, C2)", StringEncodingMode::UTF8);
}

// Test the string encoding reselution
TEST_F(StringFunctionsTest, findCommonEncoding) {
  std::vector<std::string> asciiCol = {
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
  };

  std::vector<std::string> utf8Col = {
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
  };

  std::vector<std::string> mixUTF8Col = {
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "aa",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "aa",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "aa",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
  };

  std::vector<std::string> mostlyAsciiCol = {
      "aa",
      "\u4FE1\u5FF5,\u7231,\u5E0C\u671B",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
      "aa",
  };

  // Test identifying ascii
  testStringEncodingResolution(
      {asciiCol}, {folly::none}, StringEncodingMode::ASCII);

  testStringEncodingResolution(
      {asciiCol}, {StringEncodingMode::ASCII}, StringEncodingMode::ASCII);

  testStringEncodingResolution(
      {asciiCol, asciiCol},
      {folly::none, folly::none},
      StringEncodingMode::ASCII);

  testStringEncodingResolution(
      {asciiCol, asciiCol, asciiCol},
      {folly::none, folly::none, folly::none},
      StringEncodingMode::ASCII);

  // Test identifying likely ascii
  testStringEncodingResolution(
      {mostlyAsciiCol}, {folly::none}, StringEncodingMode::MOSTLY_ASCII);

  testStringEncodingResolution(
      {asciiCol, mostlyAsciiCol},
      {folly::none, folly::none},
      StringEncodingMode::MOSTLY_ASCII);

  testStringEncodingResolution(
      {asciiCol, mostlyAsciiCol},
      {StringEncodingMode::ASCII, StringEncodingMode::MOSTLY_ASCII},
      StringEncodingMode::MOSTLY_ASCII);

  testStringEncodingResolution(
      {asciiCol, asciiCol},
      {StringEncodingMode::MOSTLY_ASCII, StringEncodingMode::ASCII},
      StringEncodingMode::MOSTLY_ASCII);

  // Test identifying UTF8
  testStringEncodingResolution(
      {utf8Col}, {folly::none}, StringEncodingMode::UTF8);

  testStringEncodingResolution(
      {mixUTF8Col}, {folly::none}, StringEncodingMode::UTF8);

  testStringEncodingResolution(
      {asciiCol, mostlyAsciiCol, utf8Col},
      {folly::none, folly::none, folly::none},
      StringEncodingMode::UTF8);

  testStringEncodingResolution(
      {asciiCol, mostlyAsciiCol, utf8Col},
      {StringEncodingMode::ASCII,
       StringEncodingMode::MOSTLY_ASCII,
       folly::none},
      StringEncodingMode::UTF8);

  testStringEncodingResolution(
      {asciiCol, mostlyAsciiCol, utf8Col},
      {StringEncodingMode::ASCII,
       StringEncodingMode::MOSTLY_ASCII,
       StringEncodingMode::UTF8},
      StringEncodingMode::UTF8);
}

void StringFunctionsTest::testXXHash64(
    const std::vector<std::tuple<std::string, int64_t, int64_t>>& tests) {
  // Creating vectors for input strings and seed values
  auto inputString = makeFlatVector<StringView>(tests.size());
  auto inputSeed = makeFlatVector<int64_t>(tests.size());
  for (int i = 0; i < tests.size(); i++) {
    inputString->set(i, StringView(std::get<0>(tests[i])));
    inputSeed->set(i, std::get<1>(tests[i]));
  }
  auto rowVector = makeRowVector({inputString, inputSeed});

  // Evaluating the function for each input and seed
  auto result = evaluate<FlatVector<int64_t>>("xxhash64(c0, c1)", rowVector);

  // Checking the results
  for (int32_t i = 0; i < tests.size(); ++i) {
    ASSERT_EQ(result->valueAt(i), std::get<2>(tests[i]));
  }
}

void StringFunctionsTest::testXXHash64(
    const std::vector<std::pair<std::string, int64_t>>& tests,
    bool stringVariant) {
  auto type = stringVariant ? std::dynamic_pointer_cast<const Type>(VARCHAR())
                            : VARBINARY();
  // Creating vectors for input strings
  auto inputString = makeFlatVector<StringView>(tests.size(), type);
  for (int i = 0; i < tests.size(); i++) {
    inputString->set(i, StringView(tests[i].first));
  }
  auto rowVector = makeRowVector({inputString});

  // Evaluate and compare results
  if (stringVariant) {
    auto result = evaluate<FlatVector<int64_t>>("xxhash64(c0)", rowVector);
    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(result->valueAt(i), tests[i].second);
    }
  } else {
    auto result = evaluate<FlatVector<StringView>>("xxhash64(c0)", rowVector);
    for (int32_t i = 0; i < tests.size(); ++i) {
      ASSERT_EQ(
          std::memcmp(
              result->valueAt(i).data(),
              &tests[i].second,
              sizeof(tests[i].second)),
          0);
    }
  }
}

TEST_F(StringFunctionsTest, xxhash64) {
  // The first two cases are borrowed from the original HIVE implementation
  // unittests and the last two are corner cases
  // fbcode/fbjava/hive-udfs/core-udfs/src/main/java/com/facebook/hive/udf/UDFXxhash64.java
  {
    std::vector<std::tuple<std::string, int64_t, int64_t>> validInputTest = {
        {"hashmes", 0, 4920146668586838293},
        {"hashme", 1, 1571629256661355178},
        {"", 0, 0xEF46DB3751D8E999}};

    testXXHash64(validInputTest);
  }

  // Similar tests for the Presto variant
  {
    std::vector<std::pair<std::string, int64_t>> validInputTest = {
        {"hashmes", 4920146668586838293}, {"", 0xEF46DB3751D8E999}};

    testXXHash64(validInputTest, false);

    // Default value seed for string inputs
    testXXHash64(validInputTest, true);
  }
}
