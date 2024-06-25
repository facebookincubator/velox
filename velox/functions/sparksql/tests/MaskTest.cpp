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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/Expr.h"
#include "velox/functions/Udf.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"
#include "velox/parse/Expressions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions::test;
using namespace facebook::velox::test;

namespace facebook::velox::functions::sparksql::test {
namespace {

class MaskTest : public SparkFunctionBaseTest {
 protected:
  auto createFlatStringsFunctor(
      vector_size_t numRows,
      const std::optional<std::string>& arg) {
    return [&, numRows, arg](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows,
          [arg](vector_size_t row) {
            return StringView{arg.value_or("").c_str()};
          },
          [arg](vector_size_t row) { return !arg.has_value(); });
    };
  }

  auto createFlatStringsFunctor(
      const std::vector<std::optional<std::string>>& input) {
    return [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          input.size(),
          [&](vector_size_t row) {
            if (input[row].has_value()) {
              return StringView{input[row].value()};
            } else {
              return StringView{""};
            }
          },
          [&](vector_size_t row) { return !input[row].has_value(); });
    };
  }

  VectorPtr runWithFiveArgs(
      const std::vector<std::optional<std::string>>& input,
      const std::optional<std::string>& upperChar,
      const std::optional<std::string>& lowerChar,
      const std::optional<std::string>& digitChar,
      const std::optional<std::string>& otherChar,
      VectorEncoding::Simple encodingStrings = VectorEncoding::Simple::FLAT,
      VectorEncoding::Simple encodingUpperChar =
          VectorEncoding::Simple::CONSTANT,
      VectorEncoding::Simple encodingLowerChar =
          VectorEncoding::Simple::CONSTANT,
      VectorEncoding::Simple encodingDigitChar =
          VectorEncoding::Simple::CONSTANT,
      VectorEncoding::Simple encodingOtherChar =
          VectorEncoding::Simple::CONSTANT) {
    VectorPtr strings, upperChars, lowerChars, digitChars, otherChars;
    const vector_size_t numRows = input.size();

    // Functors to create flat vectors, used as is and for lazy vector.
    auto funcCreateFlatStrings = createFlatStringsFunctor(input);

    auto funcCreateFlatUpperChars =
        createFlatStringsFunctor(input.size(), upperChar);
    auto funcCreateFlatLowerChars =
        createFlatStringsFunctor(input.size(), lowerChar);
    auto funcCreateFlatDigitChars =
        createFlatStringsFunctor(input.size(), digitChar);
    auto funcCreateFlatOtherChars =
        createFlatStringsFunctor(input.size(), otherChar);

    auto funcReverseIndices = [&](vector_size_t row) {
      return numRows - 1 - row;
    };

    // Generate strings vector
    if (isFlat(encodingStrings)) {
      strings = funcCreateFlatStrings({});
    } else if (isConstant(encodingStrings)) {
      strings =
          BaseVector::wrapInConstant(numRows, 0, funcCreateFlatStrings({}));
    } else if (isLazy(encodingStrings)) {
      strings = std::make_shared<LazyVector>(
          execCtx_.pool(),
          CppToType<StringView>::create(),
          numRows,
          std::make_unique<SimpleVectorLoader>(funcCreateFlatStrings));
    } else if (isDictionary(encodingStrings)) {
      strings = wrapInDictionary(
          makeIndices(numRows, funcReverseIndices),
          numRows,
          funcCreateFlatStrings({}));
    }

    // Generate upperChars vector
    if (isFlat(encodingUpperChar)) {
      upperChars = funcCreateFlatUpperChars({});
    } else if (isConstant(encodingUpperChar)) {
      if (upperChar.has_value()) {
        upperChars = makeConstant(upperChar.value().c_str(), numRows);
      } else {
        upperChars = makeNullConstant(TypeKind::VARCHAR, numRows);
      }
    } else if (isLazy(encodingUpperChar)) {
      upperChars = std::make_shared<LazyVector>(
          execCtx_.pool(),
          CppToType<StringView>::create(),
          numRows,
          std::make_unique<SimpleVectorLoader>(funcCreateFlatUpperChars));
    } else if (isDictionary(encodingUpperChar)) {
      upperChars = wrapInDictionary(
          makeIndices(numRows, funcReverseIndices),
          numRows,
          funcCreateFlatUpperChars({}));
    }

    // Generate lowerChar vector
    if (isFlat(encodingLowerChar)) {
      lowerChars = funcCreateFlatLowerChars({});
    } else if (isConstant(encodingLowerChar)) {
      if (lowerChar.has_value()) {
        lowerChars = makeConstant(lowerChar.value().c_str(), numRows);
      } else {
        lowerChars = makeNullConstant(TypeKind::VARCHAR, numRows);
      }
    } else if (isLazy(encodingLowerChar)) {
      lowerChars = std::make_shared<LazyVector>(
          execCtx_.pool(),
          CppToType<StringView>::create(),
          numRows,
          std::make_unique<SimpleVectorLoader>(funcCreateFlatLowerChars));
    } else if (isDictionary(encodingLowerChar)) {
      lowerChars = wrapInDictionary(
          makeIndices(numRows, funcReverseIndices),
          numRows,
          funcCreateFlatLowerChars({}));
    }

    // Generate digitChar vector
    if (isFlat(encodingDigitChar)) {
      digitChars = funcCreateFlatDigitChars({});
    } else if (isConstant(encodingDigitChar)) {
      if (digitChar.has_value()) {
        digitChars = makeConstant(digitChar.value().c_str(), numRows);
      } else {
        digitChars = makeNullConstant(TypeKind::VARCHAR, numRows);
      }
    } else if (isLazy(encodingDigitChar)) {
      digitChars = std::make_shared<LazyVector>(
          execCtx_.pool(),
          CppToType<StringView>::create(),
          numRows,
          std::make_unique<SimpleVectorLoader>(funcCreateFlatDigitChars));
    } else if (isDictionary(encodingDigitChar)) {
      digitChars = wrapInDictionary(
          makeIndices(numRows, funcReverseIndices),
          numRows,
          funcCreateFlatDigitChars({}));
    }

    // Generate otherChar vector
    if (isFlat(encodingOtherChar)) {
      otherChars = funcCreateFlatOtherChars({});
    } else if (isConstant(encodingOtherChar)) {
      if (otherChar.has_value()) {
        otherChars = makeConstant(otherChar.value().c_str(), numRows);
      } else {
        otherChars = makeNullConstant(TypeKind::VARCHAR, numRows);
      }
    } else if (isLazy(encodingOtherChar)) {
      otherChars = std::make_shared<LazyVector>(
          execCtx_.pool(),
          CppToType<StringView>::create(),
          numRows,
          std::make_unique<SimpleVectorLoader>(funcCreateFlatOtherChars));
    } else if (isDictionary(encodingOtherChar)) {
      otherChars = wrapInDictionary(
          makeIndices(numRows, funcReverseIndices),
          numRows,
          funcCreateFlatOtherChars({}));
    }
    VectorPtr result = evaluate(
        "mask(C0, C1, C2, C3, C4)",
        makeRowVector(
            {strings, upperChars, lowerChars, digitChars, otherChars}));
    return VectorMaker::flatten(result);
  }

  VectorPtr runWithFourArgs(
      const std::vector<std::optional<std::string>>& input,
      const std::optional<std::string>& upperChar,
      const std::optional<std::string>& lowerChar,
      const std::optional<std::string>& digitChar,
      VectorEncoding::Simple encodingMaskedChar =
          VectorEncoding::Simple::CONSTANT) {
    VectorPtr strings, upperChars, lowerChars, digitChars;
    const vector_size_t numRows = input.size();

    auto funcCreateFlatStrings = createFlatStringsFunctor(input);
    strings = funcCreateFlatStrings({});
    if (isConstant(encodingMaskedChar)) {
      upperChars = makeConstant(
          upperChar.has_value() ? upperChar.value().c_str() : nullptr, numRows);
      lowerChars = makeConstant(
          lowerChar.has_value() ? lowerChar.value().c_str() : nullptr, numRows);
      digitChars = makeConstant(
          digitChar.has_value() ? digitChar.value().c_str() : nullptr, numRows);
    } else {
      auto funcCreateFlatUpperChars =
          createFlatStringsFunctor(input.size(), upperChar);
      auto funcCreateFlatLowerChars =
          createFlatStringsFunctor(input.size(), lowerChar);
      auto funcCreateFlatDigitChars =
          createFlatStringsFunctor(input.size(), digitChar);
      upperChars = funcCreateFlatUpperChars({});
      lowerChars = funcCreateFlatLowerChars({});
      digitChars = funcCreateFlatDigitChars({});
    }

    VectorPtr result = evaluate<BaseVector>(
        "mask(C0, C1, C2, C3)",
        makeRowVector({strings, upperChars, lowerChars, digitChars}));
    return VectorMaker::flatten(result);
  }

  VectorPtr runWithThreeArgs(
      const std::vector<std::optional<std::string>>& input,
      const std::optional<std::string>& upperChar,
      const std::optional<std::string>& lowerChar,
      VectorEncoding::Simple encodingMaskedChar =
          VectorEncoding::Simple::CONSTANT) {
    VectorPtr strings, upperChars, lowerChars;
    const vector_size_t numRows = input.size();

    auto funcCreateFlatStrings = createFlatStringsFunctor(input);
    strings = funcCreateFlatStrings({});
    if (isConstant(encodingMaskedChar)) {
      upperChars = makeConstant(
          upperChar.has_value() ? upperChar.value().c_str() : nullptr, numRows);
      lowerChars = makeConstant(
          lowerChar.has_value() ? lowerChar.value().c_str() : nullptr, numRows);
    } else {
      auto funcCreateFlatUpperChars =
          createFlatStringsFunctor(input.size(), upperChar);
      auto funcCreateFlatLowerChars =
          createFlatStringsFunctor(input.size(), lowerChar);
      upperChars = funcCreateFlatUpperChars({});
      lowerChars = funcCreateFlatLowerChars({});
    }

    VectorPtr result = evaluate<BaseVector>(
        "mask(C0, C1, C2)", makeRowVector({strings, upperChars, lowerChars}));
    return VectorMaker::flatten(result);
  }

  VectorPtr runWithTwoArgs(
      const std::vector<std::optional<std::string>>& input,
      const std::optional<std::string>& upperChar,
      VectorEncoding::Simple encodingMaskedChar =
          VectorEncoding::Simple::CONSTANT) {
    VectorPtr strings, upperChars;
    const vector_size_t numRows = input.size();

    auto funcCreateFlatStrings = createFlatStringsFunctor(input);
    strings = funcCreateFlatStrings({});
    if (isConstant(encodingMaskedChar)) {
      upperChars = makeConstant(
          upperChar.has_value() ? upperChar.value().c_str() : nullptr, numRows);
    } else {
      auto funcCreateFlatUpperChars =
          createFlatStringsFunctor(input.size(), upperChar);
      upperChars = funcCreateFlatUpperChars({});
    }

    VectorPtr result = evaluate<BaseVector>(
        "mask(C0, C1)", makeRowVector({strings, upperChars}));
    return VectorMaker::flatten(result);
  }

  VectorPtr runWithOneArgs(
      const std::vector<std::optional<std::string>>& input) {
    const vector_size_t numRows = input.size();

    auto funcCreateFlatStrings = createFlatStringsFunctor(input);
    VectorPtr strings = funcCreateFlatStrings({});

    VectorPtr result =
        evaluate<BaseVector>("mask(C0)", makeRowVector({strings}));
    return VectorMaker::flatten(result);
  }

  VectorPtr prepare(
      FlatVectorPtr<StringView>& expected,
      VectorEncoding::Simple stringEncoding = VectorEncoding::Simple::FLAT) {
    // Constant: we will have all rows as the 1st one.
    if (isConstant(stringEncoding)) {
      auto constVector =
          BaseVector::wrapInConstant(expected->size(), 0, expected);
      return VectorMaker::flatten(constVector);
    }

    // Dictionary: we will have reversed rows, because we use reverse index
    // functor to generate indices when wrapping in dictionary.
    if (isDictionary(stringEncoding)) {
      auto funcReverseIndices = [&](vector_size_t row) {
        return expected->size() - 1 - row;
      };

      auto dictVector = wrapInDictionary(
          makeIndices(expected->size(), funcReverseIndices),
          expected->size(),
          expected);
      return VectorMaker::flatten(dictVector);
    }

    // Non-const string. Unchanged.
    return expected;
  }
};

/**
 * Test mask vector function on vectors with different encodings.
 */
TEST_F(MaskTest, mask) {
  std::vector<std::optional<std::string>> inputStrings;
  std::string upperChar;
  std::string lowerChar;
  std::string digitChar;
  std::string otherChar;
  VectorPtr actual;

  // We want to check these encodings for the vectors.
  std::vector<VectorEncoding::Simple> encodings{
      VectorEncoding::Simple::CONSTANT,
      VectorEncoding::Simple::FLAT,
      VectorEncoding::Simple::LAZY,
      VectorEncoding::Simple::DICTIONARY,
  };

  upperChar = "Y";
  lowerChar = "y";
  digitChar = "d";
  otherChar = "*";

  inputStrings = std::vector<std::optional<std::string>>{
      {"AbCD123-@$#"}, {"abcd-EFGH-8765-4321"}, {std::nullopt}, {""}};

  auto expected = makeNullableFlatVector<StringView>({
      {"YyYYddd****"},
      {"yyyy*YYYY*dddd*dddd"},
      {std::nullopt},
      {""},
  });
  auto expected1 = makeNullableFlatVector<StringView>({
      {"AyCDddd****"},
      {"yyyy*EFGH*dddd*dddd"},
      {std::nullopt},
      {""},
  });
  auto expected2 = makeNullableFlatVector<StringView>({
      {"AbCDddd****"},
      {"abcd*EFGH*dddd*dddd"},
      {std::nullopt},
      {""},
  });
  auto expected3 = makeNullableFlatVector<StringView>({
      {"AbCD123****"},
      {"abcd*EFGH*8765*4321"},
      {std::nullopt},
      {""},
  });
  auto expected4 = makeNullableFlatVector<StringView>({
      {"YyYYddd-@$#"},
      {"yyyy-YYYY-dddd-dddd"},
      {std::nullopt},
      {""},
  });
  auto expected5 = makeNullableFlatVector<StringView>({
      {"AbCD123-@$#"},
      {"abcd-EFGH-8765-4321"},
      {std::nullopt},
      {""},
  });

  // Mix and match encodings.
  for (const auto& sEn : encodings) {
    for (const auto& uEn : encodings) {
      for (const auto& lEn : encodings) {
        for (const auto& dEn : encodings) {
          for (const auto& oEn : encodings) {
            auto actual = runWithFiveArgs(
                inputStrings,
                upperChar,
                lowerChar,
                digitChar,
                otherChar,
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected, sEn), actual);
            actual = runWithFiveArgs(
                inputStrings,
                std::nullopt, // upperChar is null.
                lowerChar,
                digitChar,
                otherChar,
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected1, sEn), actual);
            actual = runWithFiveArgs(
                inputStrings,
                std::nullopt, // upperChar is null.
                std::nullopt, // lowerChar is null.
                digitChar,
                otherChar,
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected2, sEn), actual);
            actual = runWithFiveArgs(
                inputStrings,
                std::nullopt, // upperChar is null.
                std::nullopt, // lowerChar is null.
                std::nullopt, // digitChar is null.
                otherChar,
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected3, sEn), actual);
            actual = runWithFiveArgs(
                inputStrings,
                upperChar,
                lowerChar,
                digitChar,
                std::nullopt, // otherChar is null.
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected4, sEn), actual);
            actual = runWithFiveArgs(
                inputStrings,
                std::nullopt, // upperChar is null.
                std::nullopt, // lowerChar is null.
                std::nullopt, // digitChar is null.
                std::nullopt, // otherChar is null.
                sEn,
                uEn,
                lEn,
                dEn,
                oEn);
            assertEqualVectors(prepare(expected5, sEn), actual);
          }
        }
      }
    }
  }

  // Test mask with 4 args provided.
  auto expected6 = makeNullableFlatVector<StringView>({
      {"YyYYddd-@$#"},
      {"yyyy-YYYY-dddd-dddd"},
      {std::nullopt},
      {""},
  });
  actual = runWithFourArgs(inputStrings, upperChar, lowerChar, digitChar);
  assertEqualVectors(prepare(expected6), actual);
  actual = runWithFourArgs(
      inputStrings,
      upperChar,
      lowerChar,
      digitChar,
      VectorEncoding::Simple::FLAT);
  assertEqualVectors(prepare(expected6), actual);

  // Test mask with 3 args provided.
  auto expected7 = makeNullableFlatVector<StringView>({
      {"YyYYnnn-@$#"},
      {"yyyy-YYYY-nnnn-nnnn"},
      {std::nullopt},
      {""},
  });
  actual = runWithThreeArgs(inputStrings, upperChar, lowerChar);
  assertEqualVectors(prepare(expected7), actual);
  actual = runWithThreeArgs(
      inputStrings, upperChar, lowerChar, VectorEncoding::Simple::FLAT);
  assertEqualVectors(prepare(expected7), actual);

  // Test mask with 2 args provided.
  auto expected8 = makeNullableFlatVector<StringView>({
      {"YxYYnnn-@$#"},
      {"xxxx-YYYY-nnnn-nnnn"},
      {std::nullopt},
      {""},
  });
  actual = runWithTwoArgs(inputStrings, upperChar);
  assertEqualVectors(prepare(expected8), actual);
  actual =
      runWithTwoArgs(inputStrings, upperChar, VectorEncoding::Simple::FLAT);
  assertEqualVectors(prepare(expected8), actual);

  // Test mask with 1 arg provided.
  auto expected9 = makeNullableFlatVector<StringView>({
      {"XxXXnnn-@$#"},
      {"xxxx-XXXX-nnnn-nnnn"},
      {std::nullopt},
      {""},
  });
  actual = runWithOneArgs(inputStrings);
  assertEqualVectors(prepare(expected9), actual);
}

TEST_F(MaskTest, maskWithError) {
  auto inputStrings = std::vector<std::optional<std::string>>{
      {"AbCD123-@$#"},
      {"abcd-EFGH-8765-4321"},
      {""},
  };

  std::string upperChar = "Y";
  std::string lowerChar = "y";
  std::string digitChar = "d";
  std::string otherChar = "*";
  VELOX_ASSERT_USER_THROW(
      runWithFiveArgs(inputStrings, "", lowerChar, digitChar, otherChar),
      "Length of upperChar should be 1");
  VELOX_ASSERT_USER_THROW(
      runWithFiveArgs(inputStrings, upperChar, "", digitChar, otherChar),
      "Length of lowerChar should be 1");
  VELOX_ASSERT_USER_THROW(
      runWithFiveArgs(inputStrings, upperChar, lowerChar, "", otherChar),
      "Length of digitChar should be 1");
  VELOX_ASSERT_USER_THROW(
      runWithFiveArgs(inputStrings, upperChar, lowerChar, digitChar, ""),
      "Length of otherChar should be 1");
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test
