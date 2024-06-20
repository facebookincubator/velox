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
  VectorPtr run(
      const std::vector<std::string>& input,
      const std::optional<std::string>& upperChar,
      const std::optional<std::string>& lowerChar,
      const std::optional<std::string>& digitChar,
      const std::optional<std::string>& otherChar,
      const char* query,
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
    auto funcCreateFlatStrings = [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows, [&](vector_size_t row) { return StringView{input[row]}; });
    };

    auto funcCreateFlatUpperChars = [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            return StringView{upperChar.value_or("").c_str()};
          },
          [&](vector_size_t row) { return !upperChar.has_value(); });
    };

    auto funcCreateFlatLowerChars = [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            return StringView{lowerChar.value_or("").c_str()};
          },
          [&](vector_size_t row) { return !lowerChar.has_value(); });
    };

    auto funcCreateFlatDigitChars = [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            return StringView{digitChar.value_or("").c_str()};
          },
          [&](vector_size_t row) { return !digitChar.has_value(); });
    };

    auto funcCreateFlatOtherChars = [&](RowSet /*rows*/) {
      return makeFlatVector<StringView>(
          numRows,
          [&](vector_size_t row) {
            return StringView{otherChar.value_or("").c_str()};
          },
          [&](vector_size_t row) { return !otherChar.has_value(); });
    };

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
      upperChars = makeConstant(
          upperChar.has_value() ? upperChar.value().c_str() : nullptr, numRows);
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
      lowerChars = makeConstant(
          lowerChar.has_value() ? lowerChar.value().c_str() : nullptr, numRows);
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
      digitChars = makeConstant(
          digitChar.has_value() ? digitChar.value().c_str() : nullptr, numRows);
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
      otherChars = makeConstant(
          otherChar.has_value() ? otherChar.value().c_str() : nullptr, numRows);
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

    VectorPtr result = evaluate<BaseVector>(
        query,
        makeRowVector(
            {strings, upperChars, lowerChars, digitChars, otherChars}));
    return VectorMaker::flatten(result);
  }

  VectorPtr prepare(
      FlatVectorPtr<StringView>& expected,
      VectorEncoding::Simple stringEncoding) {
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
  std::vector<std::string> inputStrings;
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

  upperChar = "Q";
  lowerChar = "q";
  digitChar = "d";
  otherChar = "o";

  inputStrings = std::vector<std::string>{
      {"abcd-EFGH-8765-4321"},
      {"AbCD123-@$#"},
      {""},
  };
  // Base expected data.
  auto expected = makeFlatVector<StringView>({
      {"qqqqoQQQQoddddodddd"},
      {"QqQQdddoooo"},
      {""},
  });

  // Mix and match encodings.
  for (const auto& sEn : encodings) {
    for (const auto& uEn : encodings) {
      for (const auto& lEn : encodings) {
        for (const auto& dEn : encodings) {
          for (const auto& oEn : encodings) {
            auto actual =
                run(inputStrings,
                    upperChar,
                    lowerChar,
                    digitChar,
                    otherChar,
                    "mask(C0, C1, C2, C3, C4)",
                    sEn,
                    uEn,
                    lEn,
                    dEn,
                    oEn);
            assertEqualVectors(prepare(expected, sEn), actual);
          }
        }
      }
    }
  }
}
} // namespace
} // namespace facebook::velox::functions::sparksql::test